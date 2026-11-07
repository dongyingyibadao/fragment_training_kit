"""Behavioral protocol cases; no tensors, jobs, device imports or experiments."""
from dataclasses import asdict, replace
import hashlib
import json

import pytest

from fragment_training_kit.vendor.jade_lotus.topology_switch import (
    CohortStatus, CommitReceipt, CoverageReceipt, FullStateReceipt, Member,
    RepairReceipt, Reservation, SessionKey, StateSummary, StaleSwitch,
    SwitchError, SwitchEstimate, SwitchPolicy, SwitchRecoveryRequired,
    TerminalReceipt, TopologySwitch, VersionFence,
)


RUN = "topology-test-run"
CONTRACT = hashlib.sha256(b"global256/atomic4/adamw/native/rng-v1/config").hexdigest()


def summary(step, *, optimizer_salt=""):
    return StateSummary(step, hashlib.sha256(f"model-{step}".encode()).hexdigest(),
                        hashlib.sha256(f"adam-{step}{optimizer_salt}".encode()).hexdigest())


def worker(name, resource):
    session = SessionKey(RUN, name, f"session-{name}", 100, 90, resource, 0)
    return Member(session, resource)


def state_receipts(members, fence, operation, state=None):
    return tuple(FullStateReceipt(m.session if isinstance(m, Member) else m,
        fence, operation, state or fence.state, f"verified-bytes/{operation}/{index}", True)
        for index, m in enumerate(members))


def estimate(**overrides):
    values = dict(current_step_seconds=10.0, candidate_step_seconds=8.0,
                  remaining_steps=100, load_seconds=1.0, sync_seconds=1.0,
                  rebuild_seconds=1.0, verify_seconds=1.0,
                  standby_sync_seconds=0.05, stable_observations=3)
    return SwitchEstimate(**(values | overrides))


def make_switch(*, active_cards=48, old_cards=16, budget=64, old_domain=None, policy=None):
    # The outgoing cohort consists of fragmented independent single-card jobs.
    old_resources = tuple(Reservation(f"old-job-{i}", 1, old_domain or f"old-domain-{i}", "running")
                          for i in range(old_cards))
    old = tuple(worker(f"old-{i}", r.resource_id) for i, r in enumerate(old_resources))
    steady_resources, steady = [], []
    for offset in range(0, active_cards - old_cards, 16):
        cards = min(16, active_cards - old_cards - offset)
        resource = Reservation(f"steady-pod-{offset}", cards, f"steady-domain-{offset}", "running")
        steady_resources.append(resource)
        steady.extend(worker(f"steady-{i}", resource.resource_id) for i in range(offset, offset + cards))
    fence = VersionFence(RUN, 0, summary(0), CONTRACT)
    cohorts = {"old": old}
    if steady:
        cohorts["steady"] = tuple(steady)
    switch = TopologySwitch(RUN, fence.state, contract_sha256=CONTRACT,
        reservations=old_resources + tuple(steady_resources), active_cohorts=cohorts,
        initial_receipts=state_receipts(old + tuple(steady), fence, "initial"),
        card_budget=budget, policy=policy)
    return switch, old


def status(switch, name):
    return next(c["status"] for c in switch.snapshot()["cohorts"] if c["cohort_id"] == name)


def queue(switch, *, name="candidate", replaces="old", domain="new-pod", pinned=False):
    switch.queue_candidate(name, replaces=replaces,
        reservations=(Reservation(f"{name}-job", 16, domain, pinned=pinned),),
        estimate=estimate(), fence=switch.fence)


def ready(switch, *, name="candidate", replaces="old", domain="new-pod"):
    queue(switch, name=name, replaces=replaces, domain=domain)
    members = tuple(worker(f"{name}-{i}", f"{name}-job") for i in range(16))
    switch.bind_candidate(name, members, running_receipt=f"official-running/{name}", fence=switch.fence)
    operation = f"sync-{name}"
    switch.begin_sync(name, operation, fence=switch.fence)
    switch.complete_sync(name, operation, state_receipts(members, switch.fence, operation), fence=switch.fence)
    return members


def coverage(plan, **overrides):
    start = plan.fence.state.committed_step * 256
    values = dict(step_token=plan.token, fence=plan.fence,
                  input_identity_sha256=plan.input_identity_sha256,
                  sample_positions=tuple(range(start, start + 256)),
                  verification_receipt="original-raw-contributions-verified",
                  verified_original_samples=True)
    return CoverageReceipt(**(values | overrides))


def commit(switch, token="step-0"):
    plan = switch.begin_step(token, fence=switch.fence)
    switch.authorize_apply(coverage(plan))
    receipt = CommitReceipt(token, plan.fence, summary(plan.fence.state.committed_step + 1),
        plan.input_identity_sha256, state_receipts(plan.members, plan.fence, token,
                                                 summary(plan.fence.state.committed_step + 1)))
    switch.commit_step(receipt)
    return plan, receipt


def sync_backup(switch, old, operation="sync-old"):
    switch.begin_sync("old", operation, fence=switch.fence)
    switch.complete_sync("old", operation, state_receipts(old, switch.fence, operation), fence=switch.fence)


def release(switch, name):
    switch.request_release(name, fence=switch.fence)
    resources = next(c["resource_ids"] for c in switch.snapshot()["cohorts"] if c["cohort_id"] == name)
    receipts = tuple(TerminalReceipt(resource, "job_stopped", f"official/{resource}") for resource in resources)
    switch.confirm_release(name, receipts, fence=switch.fence)
    return receipts


def repair_receipt(switch, **overrides):
    plan = switch.step
    lost = {SessionKey(**row) for row in switch.snapshot()["fenced_sessions"]}
    values = dict(step_token=plan.token, fence=plan.fence,
                  input_identity_sha256=plan.input_identity_sha256,
                  preserved_units=tuple(a.unit_id for a in plan.assignments if a.session not in lost),
                  verification_receipt="aborted-collectives-and-verified-original-healthy-gradients",
                  collectives_aborted=True, healthy_original_gradients_preserved=True)
    return RepairReceipt(**(values | overrides))


def test_switch_keeps_old_cards_until_success_and_requires_fresh_full_state_for_backup():
    switch, old = make_switch()
    ready(switch)
    switch.activate_candidate("candidate", estimate=estimate(), fence=switch.fence)
    assert status(switch, "candidate") == CohortStatus.TRIAL
    assert status(switch, "old") == CohortStatus.ROLLBACK_HOLD
    assert switch.reserved_cards == 64
    with pytest.raises(SwitchError):
        switch.request_release("old", fence=switch.fence)
    commit(switch)
    assert status(switch, "candidate") == CohortStatus.ACTIVE
    assert status(switch, "old") == CohortStatus.STANDBY_LOADING
    with pytest.raises(SwitchError):
        switch.activate_backup("old", fence=switch.fence)
    sync_backup(switch, old)
    assert status(switch, "old") == CohortStatus.HOT_STANDBY
    commit(switch, "step-1")
    assert status(switch, "old") == CohortStatus.STANDBY_LOADING
    with pytest.raises(SwitchError):
        switch.activate_backup("old", fence=switch.fence)
    sync_backup(switch, old, "sync-old-again")
    switch.activate_backup("old", fence=switch.fence)
    assert status(switch, "old") == CohortStatus.ACTIVE
    assert switch.committed_state == summary(2)


def test_release_waits_for_every_official_terminal_and_released_cards_cannot_be_backup():
    switch, old = make_switch()
    ready(switch)
    switch.activate_candidate("candidate", estimate=estimate(), fence=switch.fence)
    commit(switch)
    sync_backup(switch, old)
    switch.request_release("old", fence=switch.fence)
    assert switch.reserved_cards == 64
    terminals = tuple(TerminalReceipt(m.resource_id, "job_stopped", f"official/{m.resource_id}") for m in old)
    with pytest.raises(SwitchError):
        switch.confirm_release("old", terminals[:-1], fence=switch.fence)
    assert switch.reserved_cards == 64
    switch.confirm_release("old", terminals, fence=switch.fence)
    switch.confirm_release("old", terminals, fence=switch.fence)
    assert switch.reserved_cards == 48
    for action in (lambda: switch.activate_backup("old", fence=switch.fence),
                   lambda: switch.begin_sync("old", "resurrect", fence=switch.fence)):
        with pytest.raises(SwitchError):
            action()
    with pytest.raises(SwitchError):
        TerminalReceipt(old[0].resource_id, "deleting", "not-a-terminal-state")


def test_full_64_card_training_has_no_implicit_16_card_overlap_budget():
    switch, _ = make_switch(active_cards=64)
    before = switch.snapshot()
    with pytest.raises(SwitchError, match="budget"):
        queue(switch)
    assert switch.snapshot() == before


def test_queued_pinned_and_release_pending_candidates_still_reserve_budget():
    switch, _ = make_switch()
    queue(switch, pinned=True)
    assert switch.reserved_cards == 64
    claim = next(r for r in switch.snapshot()["resources"] if r["resource_id"] == "candidate-job")
    assert claim["pinned"] and claim["state"] == "queued"
    switch.request_release("candidate", fence=switch.fence)
    with pytest.raises(SwitchError, match="budget"):
        queue(switch, name="second")
    switch.confirm_release("candidate", (TerminalReceipt("candidate-job", "job_deleted", "official-deleted"),),
                           fence=switch.fence)
    queue(switch, name="second")
    assert switch.reserved_cards == 64


def test_base_weights_wrong_adam_wrong_step_and_pid_reuse_never_make_candidate_ready():
    switch, _ = make_switch()
    queue(switch)
    members = tuple(worker(f"candidate-{i}", "candidate-job") for i in range(16))
    switch.bind_candidate("candidate", members, running_receipt="official-running", fence=switch.fence)
    switch.begin_sync("candidate", "load-current", fence=switch.fence)
    rows = state_receipts(members, switch.fence, "load-current")
    bad_states = (summary(0, optimizer_salt="base-adam"),
                  StateSummary(1, summary(0).model_sha256, summary(0).optimizer_sha256))
    for bad in bad_states:
        with pytest.raises(StaleSwitch):
            switch.complete_sync("candidate", "load-current", tuple(replace(r, state=bad) for r in rows),
                                 fence=switch.fence)
    with pytest.raises(SwitchError):
        replace(rows[0], verified_full_state=False)
    with pytest.raises(SwitchError):
        switch.complete_sync("candidate", "load-current", rows[:-1], fence=switch.fence)
    with pytest.raises(SwitchError):
        switch.complete_sync("candidate", "load-current", rows[:-1] + (rows[0],), fence=switch.fence)
    with pytest.raises(SwitchError):
        switch.complete_sync("candidate", "load-current",
            (replace(rows[0], session=replace(rows[0].session, child_pid=101)),) + rows[1:], fence=switch.fence)
    assert status(switch, "candidate") == CohortStatus.CANDIDATE_LOADING
    switch.complete_sync("candidate", "load-current", tuple(reversed(rows)), fence=switch.fence)
    assert status(switch, "candidate") == CohortStatus.CANDIDATE_READY
    with pytest.raises(StaleSwitch):
        switch.complete_sync("candidate", "load-current", rows, fence=switch.fence)


def test_loading_does_not_block_commits_and_late_sync_cannot_admit_stale_state():
    switch, _ = make_switch()
    queue(switch)
    members = tuple(worker(f"candidate-{i}", "candidate-job") for i in range(16))
    switch.bind_candidate("candidate", members, running_receipt="official-running", fence=switch.fence)
    old_fence = switch.fence
    switch.begin_sync("candidate", "slow-load", fence=old_fence)
    old_rows = state_receipts(members, old_fence, "slow-load")
    commit(switch)
    assert status(switch, "candidate") == CohortStatus.CANDIDATE_LOADING
    with pytest.raises(StaleSwitch):
        switch.complete_sync("candidate", "slow-load", old_rows, fence=old_fence)
    with pytest.raises(StaleSwitch):
        switch.complete_sync("candidate", "slow-load", old_rows, fence=switch.fence)
    switch.begin_sync("candidate", "fresh-load", fence=switch.fence)
    switch.complete_sync("candidate", "fresh-load", state_receipts(members, switch.fence, "fresh-load"),
                         fence=switch.fence)


def test_candidate_pod_failure_does_not_interrupt_healthy_training_or_free_its_claim():
    switch, _ = make_switch()
    ready(switch)
    before = switch.active_sessions
    action = switch.report_loss("candidate-preempted", fence=switch.fence,
                               evidence_ref="official-pod-exit", failure_domain="new-pod")
    assert action == "discard_nonactive_cohort"
    assert status(switch, "candidate") == CohortStatus.LOST
    assert switch.active_sessions == before and switch.reserved_cards == 64
    commit(switch)
    release(switch, "candidate")
    assert switch.reserved_cards == 48


def test_trial_pod_loss_can_use_original_current_state_only_before_optimizer_writes():
    switch, old = make_switch()
    ready(switch)
    switch.activate_candidate("candidate", estimate=estimate(), fence=switch.fence)
    original = switch.begin_step("trial-step", fence=switch.fence)
    action = switch.report_loss("pod-lost", fence=switch.fence, evidence_ref="pod-heartbeat-timeout",
                               failure_domain="new-pod")
    assert action == "repair_pre_apply"
    external = repair_receipt(switch)
    repaired = switch.repair_pre_apply(external, "repaired-trial-step", backup_id="old")
    assert repaired.input_identity_sha256 == original.input_identity_sha256
    assert set(repaired.members) == set(m.session for m in old) | {
        a.session for a in original.assignments if a.unit_id in external.preserved_units}
    assert [a.sample_positions for a in repaired.assignments] == [a.sample_positions for a in original.assignments]
    for unit in external.preserved_units:
        assert repaired.assignments[unit] == original.assignments[unit]
    assert status(switch, "old") == CohortStatus.ACTIVE
    assert len(switch.snapshot()["fenced_sessions"]) == 16
    with pytest.raises(StaleSwitch):
        switch.authorize_apply(coverage(original))
    switch.authorize_apply(coverage(repaired))
    switch.commit_step(CommitReceipt(repaired.token, repaired.fence, summary(1), repaired.input_identity_sha256,
                                    state_receipts(repaired.members, repaired.fence, repaired.token, summary(1))))
    assert switch.committed_state == summary(1)
    assert switch.reserved_cards == 64
    release(switch, "candidate")


def test_single_trial_loss_preserves_other_frozen_workers_instead_of_replacing_healthy_peers():
    switch, old = make_switch()
    members = ready(switch)
    switch.activate_candidate("candidate", estimate=estimate(), fence=switch.fence)
    original = switch.begin_step("partial-trial", fence=switch.fence)
    switch.report_loss("one-new-worker", fence=switch.fence, evidence_ref="supervisor-exit",
                       sessions=(members[0].session,))
    external = repair_receipt(switch)
    before = switch.snapshot()
    with pytest.raises(SwitchError, match="healthy frozen"):
        switch.repair_pre_apply(external, "unsafe-repair", backup_id="old")
    assert switch.snapshot() == before
    repaired = switch.repair_pre_apply(external, "safe-repair")
    assert len(repaired.members) == 47
    assert all(repaired.assignments[unit] == original.assignments[unit] for unit in external.preserved_units)
    switch.authorize_apply(coverage(repaired))
    switch.commit_step(CommitReceipt(repaired.token, repaired.fence, summary(1), repaired.input_identity_sha256,
        state_receipts(repaired.members, repaired.fence, repaired.token, summary(1))))
    # A repaired shrink can run its next step; fenced sessions remain excluded.
    next_plan = switch.begin_step("next-shrunken-step", fence=switch.fence)
    assert len(next_plan.members) == 47 and members[0].session not in next_plan.members


@pytest.mark.parametrize("bad", [dict(collectives_aborted=False),
                                      dict(healthy_original_gradients_preserved=False),
                                      dict(preserved_units=()), dict(input_identity_sha256="0" * 64)])
def test_pre_apply_repair_requires_actual_abort_and_original_healthy_gradient_evidence(bad):
    switch, old = make_switch()
    switch.begin_step("collect", fence=switch.fence)
    switch.report_loss("loss", fence=switch.fence, evidence_ref="exit", sessions=(old[0].session,))
    with pytest.raises(SwitchError):
        switch.repair_pre_apply(repair_receipt(switch, **bad), "repair")
    assert switch.phase == "repair_required" and switch.committed_state == summary(0)


@pytest.mark.parametrize("changes", [dict(sample_positions=tuple(range(255)) + (254,)),
                                           dict(sample_positions=tuple(range(1, 257))),
                                           dict(sample_positions=tuple(float(p) for p in range(256))),
                                           dict(input_identity_sha256="0" * 64),
                                           dict(verified_original_samples=False)])
def test_apply_rejects_changed_duplicated_or_unverified_original_samples(changes):
    switch, _ = make_switch()
    plan = switch.begin_step("samples", fence=switch.fence)
    with pytest.raises(SwitchError):
        switch.authorize_apply(coverage(plan, **changes))
    assert switch.phase == "collecting"


def test_apply_loss_and_disagreeing_adam_receipts_require_explicit_recovery():
    switch, _ = make_switch()
    members = ready(switch)
    switch.activate_candidate("candidate", estimate=estimate(), fence=switch.fence)
    plan = switch.begin_step("apply-loss", fence=switch.fence)
    switch.authorize_apply(coverage(plan))
    assert switch.report_loss("lost-after-apply", fence=switch.fence,
        evidence_ref="supervisor-counter-timeout", failure_domain="new-pod") == "recovery_required"
    with pytest.raises(SwitchRecoveryRequired):
        switch.activate_backup("old", fence=switch.fence)
    with pytest.raises(SwitchRecoveryRequired):
        switch.commit_step(CommitReceipt(plan.token, plan.fence, summary(1), plan.input_identity_sha256,
            state_receipts(plan.members, plan.fence, plan.token, summary(1))))
    assert switch.committed_state == summary(0) and switch.reserved_cards == 64
    # Monitoring remains armed after entering recovery-required.
    switch.report_loss("another-loss", fence=switch.fence, evidence_ref="second-exit",
                       sessions=(next(s for s in plan.members if s not in {m.session for m in members}),))
    other, _ = make_switch()
    plan = other.begin_step("bad-adam", fence=other.fence)
    other.authorize_apply(coverage(plan))
    rows = state_receipts(plan.members, plan.fence, plan.token, summary(1))
    with pytest.raises(SwitchRecoveryRequired):
        other.commit_step(CommitReceipt(plan.token, plan.fence, summary(1), plan.input_identity_sha256,
            (replace(rows[0], state=summary(1, optimizer_salt="partial")),) + rows[1:]))
    assert other.phase == "recovery_required" and other.committed_state == summary(0)


def test_correlated_domain_loss_also_fences_backup_on_the_same_failure_domain():
    switch, _ = make_switch(old_domain="shared-domain")
    ready(switch, domain="shared-domain")
    switch.activate_candidate("candidate", estimate=estimate(), fence=switch.fence)
    switch.report_loss("shared-pod-or-host-loss", fence=switch.fence,
                       evidence_ref="host-unreachable", failure_domain="shared-domain")
    assert len(switch.snapshot()["fenced_sessions"]) == 32
    with pytest.raises(SwitchError):
        switch.activate_backup("old", fence=switch.fence)


def test_duplicate_commit_is_idempotent_without_rolling_back_later_state():
    switch, _ = make_switch()
    _plan, first = commit(switch)
    commit(switch, "step-1")
    before = switch.snapshot()
    assert switch.commit_step(first) == summary(1)
    assert switch.snapshot() == before and switch.committed_state == summary(2)
    with pytest.raises(StaleSwitch):
        switch.commit_step(replace(first, input_identity_sha256="0" * 64))


def test_loss_id_is_idempotent_but_cannot_change_session_or_evidence():
    switch, old = make_switch()
    kwargs = dict(fence=switch.fence, evidence_ref="official-exit", sessions=(old[0].session,))
    action = switch.report_loss("loss", **kwargs)
    before = switch.snapshot()
    assert switch.report_loss("loss", **kwargs) == action and switch.snapshot() == before
    with pytest.raises(StaleSwitch):
        switch.report_loss("loss", **(kwargs | dict(sessions=(old[1].session,))))


@pytest.mark.parametrize("changes", [dict(stable_observations=2), dict(candidate_step_seconds=9.8),
                                           dict(remaining_steps=1), dict(sync_seconds=100),
                                           dict(standby_sync_seconds=3)])
def test_estimated_gain_must_pay_transfer_rebuild_verification_and_recurring_standby_cost(changes):
    switch, _ = make_switch()
    before = switch.snapshot()
    with pytest.raises(SwitchError, match="threshold"):
        switch.queue_candidate("candidate", replaces="old", reservations=(Reservation("new", 16, "pod"),),
                               estimate=estimate(**changes), fence=switch.fence)
    assert switch.snapshot() == before


def test_cooldown_prevents_immediate_repeat_switch_after_a_success():
    switch, _ = make_switch()
    ready(switch)
    switch.activate_candidate("candidate", estimate=estimate(), fence=switch.fence)
    commit(switch)
    release(switch, "old")
    with pytest.raises(SwitchError, match="cooldown"):
        queue(switch, name="next-candidate", replaces="candidate")


def test_state_fence_rejects_foreign_run_contract_and_generation():
    switch, _ = make_switch()
    for fence in (replace(switch.fence, run_id="different-run"),
                  replace(switch.fence, contract_sha256="0" * 64),
                  replace(switch.fence, generation=1)):
        with pytest.raises(StaleSwitch):
            switch.begin_step("foreign", fence=fence)
    assert switch.phase == "idle"


def test_audit_chain_carries_evidence_and_cannot_be_mutated_through_a_reader():
    switch, _ = make_switch()
    commit(switch)
    previous = None
    for row in switch.audit_log:
        supplied = row.pop("sha256")
        assert row["previous_sha256"] == previous
        expected = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode()).hexdigest()
        assert supplied == expected
        previous = supplied
    changed = switch.audit_log[0]
    changed["detail"]["initial_receipts"].clear()
    assert switch.audit_log[0]["detail"]["initial_receipts"]
    json.dumps(switch.snapshot(), allow_nan=False)
    assert switch.snapshot()["integration_required"] is True
