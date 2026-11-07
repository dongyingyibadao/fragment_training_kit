"""Adversarial command-transport cases for explicit resident cohort switching."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import pytest

from fragment_training_kit.vendor.jade_lotus.topology_cohort_runtime import (
    CandidateUnavailable, CohortBoundaryRuntime, CohortRecoveryRequired, CommittedBoundaryLease,
    validate_topology_config,
)


def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


class Devices:
    """Independent fake device state; keeps parked Adam steps deliberately stale.

    This transport refuses a load into a live group and a join of an unparked
    device. It models command effects, rather than reproducing runtime statuses.
    It is not a tensor/HCCL correctness oracle.
    """
    def __init__(self):
        self.members = {f"w{i:06d}": {"worker_id": f"w{i:06d}", "session_id": f"s{i}",
            "pid": 100 + i, "replacement": i > 4, "address": "localhost", "port": 2000 + i}
            for i in range(1, 7)}
        self.device = {worker: {"step": 0, "group": 0 if i <= 4 else None, "parked": False}
                       for i, worker in enumerate(self.members, 1)}
        self.commands, self.reports, self.exports, self.events, self.waits = [], {}, {}, [], []
        self.fail_candidate = None
        self.corrupt_candidate = None
        self.fail_action = None
        self.unready = set()

    def state(self, worker):
        step = self.device[worker]["step"]
        return {"step": step, "model_sha256": sha(f"model-{step}"),
                "optimizer_sha256": sha(f"moments-step-options-{step}")}

    def advance(self, step, members):
        for member in members:
            assert self.device[member["worker_id"]]["group"] is not None
            self.device[member["worker_id"]]["step"] = step

    def ready(self, worker):
        if worker in self.unready:
            raise CandidateUnavailable("No independently loaded READY session")
        return deepcopy(self.members[worker])

    def send(self, members, action, **fields):
        token = f"command-{len(self.commands)}"
        ids = [member["worker_id"] for member in members]
        self.commands.append((action, ids, deepcopy(fields)))
        self.events.append(("command", action))
        if action == self.fail_action:
            raise RuntimeError(f"device failure while {action}")
        rows = {}
        for member in members:
            worker, device = member["worker_id"], self.device[member["worker_id"]]
            row = {**member, "completed_step": device["step"], **self.state(worker),
                   "optimizer_config_sha256": sha("optimizer-options")}
            if action == "export":
                assert device["step"] == fields["expected_step"]
                receipt = {**self.state(worker), "manifest_sha256": sha(fields["path"]),
                           "payload_bytes_verified": True}
                self.exports[fields["path"]] = receipt
                row.update(receipt=receipt, next_port=3000)
            elif action in {"load", "load_parked"}:
                assert device["group"] is None, "loading live active state is forbidden"
                if action == "load_parked":
                    assert device["parked"], "only explicit park authorizes parked load"
                else:
                    assert member["replacement"]
                if worker == self.fail_candidate:
                    raise CandidateUnavailable("candidate exited during loading")
                receipt = deepcopy(self.exports[fields["path"]])
                receipt.update(verified_loaded_state=True, parameter_pointers_preserved=True)
                device["step"] = receipt["step"]
                if worker == self.corrupt_candidate:
                    receipt["optimizer_sha256"] = sha("stale-Adam-but-current-model")
                row.update(completed_step=device["step"], receipt=receipt)
            elif action == "hold_committed_boundary":
                assert device["group"] == fields["expected_generation"]
                assert device["step"] == fields["expected_step"]
                row.update(generation=device["group"], hold_seconds=fields["hold_seconds"])
            elif action == "release_committed_boundary":
                assert device["group"] == fields["expected_generation"]
                assert device["step"] == fields["expected_step"]
                row.update(generation=device["group"])
            elif action == "park":
                assert device["group"] == fields["expected_generation"]
                assert device["step"] == fields["expected_step"]
                device["group"], device["parked"] = None, True
            elif action == "join":
                assert device["group"] is None, "every original group must first park"
                assert device["step"] == fields["expected_step"], "base weights cannot join a current optimizer"
                device["group"], device["parked"] = fields["group"]["generation"], False
            else:
                assert action == "inspect"
            rows[worker] = row
        self.reports[token] = rows
        return token

    def wait(self, members, token, phase, *, timeout_seconds=None):
        self.waits.append({"members": [m["worker_id"] for m in members], "token": token,
                           "phase": phase, "timeout_seconds": timeout_seconds})
        self.events.append(("wait", phase))
        return deepcopy(self.reports[token])

    def persist(self, record):
        self.events.append(("persist", record["events"][-1]["event"]))


def setup(plans=()):
    devices = Devices()
    spec = {"run_id": "run", "config_sha256": sha("config"), "run_dir": "/unused/pure-protocol-test",
            "cards": 4, "topology_slots": 6, "workers": list(devices.members), "phase_timeout": 20}
    runtime = CohortBoundaryRuntime(spec, list(plans), send=devices.send,
        wait_active=devices.wait, wait_candidate=devices.wait, ready=devices.ready,
        make_group=lambda members, generation, port=None: {"members": members, "generation": generation, "port": port},
        persist=devices.persist, metadata=lambda step: {"run_id": "run", "boundary_step": step,
                                                       "config_sha256": sha("config")})
    members = [devices.members[f"w{i:06d}"] for i in range(1, 5)]
    group = {"generation": 0, "members": members}
    runtime.start(members, group)
    return runtime, devices, members, group


def request(step=3, generation=0, request_id="to-new", *, back=False):
    return {"request_id": request_id, "run_id": "run", "config_sha256": sha("config"),
            "expected_step": step, "expected_generation": generation,
            "retire_workers": ["w000005", "w000006"] if back else ["w000003", "w000004"],
            "candidate_workers": ["w000003", "w000004"] if back else ["w000005", "w000006"]}


def test_four_active_plus_two_candidates_switch_at_3_then_realign_old_adam_and_switch_back_at_6():
    plans = [{"request_id": "to-new", "boundary_step": 3,
              "retire_workers": ["w000003", "w000004"], "candidate_workers": ["w000005", "w000006"]},
             {"request_id": "back", "boundary_step": 6,
              "retire_workers": ["w000005", "w000006"], "candidate_workers": ["w000003", "w000004"]}]
    runtime, devices, members, group = setup(plans)
    devices.advance(3, members)
    members, group = runtime.boundary(3, members, group)
    assert {m["worker_id"] for m in members} == {"w000001", "w000002", "w000005", "w000006"}
    assert group["generation"] == 1
    assert runtime.records["w000003"]["status"] == "rollback_hold"
    assert devices.device["w000003"] == {"step": 3, "group": None, "parked": True}
    assert not any("release" in action for action, _ids, _fields in devices.commands)
    devices.advance(4, members)
    runtime.on_commit(4, members)
    assert runtime.requests["to-new"]["first_new_commit"] == 4
    assert runtime.requests["to-new"]["status"] == "committed"
    assert runtime.records["w000003"]["status"] == "parked_stale"
    assert not runtime.requests["to-new"]["old_cohort_is_hot_standby"]
    devices.advance(6, members)
    assert devices.device["w000003"]["step"] == 3
    members, group = runtime.boundary(6, members, group)
    assert {m["worker_id"] for m in members} == {"w000001", "w000002", "w000003", "w000004"}
    assert group["generation"] == 2
    assert all(devices.device[m["worker_id"]]["step"] == 6 for m in members)
    parked_loads = [ids for action, ids, _fields in devices.commands if action == "load_parked"]
    assert parked_loads == [["w000003", "w000004"]]
    events = devices.events
    assert events.index(("persist", "original_group_park_started")) < events.index(("command", "park"))
    assert runtime.snapshot()["automatic_failover"] is False


@pytest.mark.parametrize("failure", ["not-ready", "load-exit", "wrong-adam"])
def test_candidate_failure_rejects_before_parking_and_original_group_remains_usable(failure):
    runtime, devices, members, group = setup()
    devices.advance(3, members)
    if failure == "not-ready":
        devices.unready.add("w000005")
    elif failure == "load-exit":
        devices.fail_candidate = "w000006"
    else:
        devices.corrupt_candidate = "w000005"
    new_members, new_group = runtime.boundary(3, members, group, request())
    assert new_members == members and new_group == group
    assert runtime.requests["to-new"]["status"] == "rejected"
    assert all(devices.device[m["worker_id"]]["group"] == 0 for m in members)
    assert not any(action in {"park", "join"} for action, _ids, _fields in devices.commands)
    devices.advance(4, members)
    assert all(devices.state(m["worker_id"])["step"] == 4 for m in members)


def test_failed_multi_worker_switch_back_can_retry_a_partially_loaded_parked_candidate():
    runtime, devices, members, group = setup()
    devices.advance(3, members)
    members, group = runtime.boundary(3, members, group, request())
    devices.advance(4, members)
    runtime.on_commit(4, members)
    devices.advance(6, members)
    devices.fail_candidate = "w000004"
    same, same_group = runtime.boundary(6, members, group, request(6, 1, "failed-back", back=True))
    assert same == members and same_group == group
    assert devices.device["w000003"]["step"] == 6 and devices.device["w000004"]["step"] == 3
    devices.fail_candidate = None
    devices.advance(7, members)
    members, group = runtime.boundary(7, members, group, request(7, 1, "retry-back", back=True))
    assert group["generation"] == 2
    assert {m["worker_id"] for m in members} == {"w000001", "w000002", "w000003", "w000004"}
    assert all(devices.device[m["worker_id"]]["step"] == 7 for m in members)


@pytest.mark.parametrize("action", ["park", "join"])
def test_group_parking_or_rebuild_failure_requires_explicit_recovery(action):
    runtime, devices, members, group = setup()
    devices.advance(3, members)
    devices.fail_action = action
    with pytest.raises(CohortRecoveryRequired):
        runtime.boundary(3, members, group, request())
    assert runtime.recovery_required
    assert runtime.requests["to-new"]["status"] == "recovery_required"


@pytest.mark.parametrize("changes", [dict(run_id="old-run"), dict(expected_generation=1),
                                           dict(expected_step=2), dict(expected_step=True),
                                           dict(config_sha256=sha("other-config")),
                                           dict(candidate_workers=["w000005", "w000005"]),
                                           dict(candidate_workers=["w000005", "w000007"]),
                                           dict(retire_workers=["w000005", "w000006"])])
def test_fenced_or_malformed_requests_cannot_change_active_members(changes):
    runtime, devices, members, group = setup()
    devices.advance(3, members)
    got, got_group = runtime.boundary(3, members, group, request() | changes)
    assert got == members and got_group == group
    assert runtime.requests["to-new"]["status"] == "rejected"
    assert devices.commands == []


def test_duplicate_request_cannot_repeat_transfer_and_conflicting_id_cannot_retarget():
    runtime, devices, members, group = setup()
    devices.advance(3, members)
    original = request()
    members, group = runtime.boundary(3, members, group, original)
    count = len(devices.commands)
    runtime.boundary(3, members, group, original)
    runtime.boundary(3, members, group, original | {"expected_generation": 1})
    assert len(devices.commands) == count
    assert runtime.events[-1]["event"] == "request_id_conflict"


def test_parked_worker_pid_reuse_is_rejected_before_any_full_state_transfer():
    runtime, devices, members, group = setup()
    devices.advance(3, members)
    members, group = runtime.boundary(3, members, group, request())
    devices.advance(4, members)
    runtime.on_commit(4, members)
    devices.advance(6, members)
    devices.members["w000003"]["pid"] += 10
    before = len(devices.commands)
    same, _ = runtime.boundary(6, members, group, request(6, 1, "back", back=True))
    assert same == members and len(devices.commands) == before
    assert runtime.requests["back"]["status"] == "rejected"


def test_future_request_is_not_consumed_before_its_commit_boundary():
    runtime, devices, members, group = setup()
    runtime.boundary(2, members, group, request())
    assert runtime.requests == {}
    devices.advance(3, members)
    _members, group = runtime.boundary(3, members, group, request())
    assert group["generation"] == 1


def mixed_candidates():
    runtime, devices, members, group = setup()
    devices.advance(3, members)
    # Both can be independently eligible, but a parked old worker must use
    # load_parked while a freshly initialized worker must use load.
    worker = "w000006"
    devices.device[worker]["parked"] = True
    runtime.records[worker] = {**devices.members[worker], "status": "parked_stale"}
    return runtime, devices, members, group


def test_mixed_candidates_start_all_loads_before_waiting_and_share_one_deadline():
    runtime, devices, members, group = mixed_candidates()
    now = [100.0]
    runtime.clock = lambda: now[0]
    def wait_candidate(cohort, token, phase, *, timeout_seconds):
        result = devices.wait(cohort, token, phase, timeout_seconds=timeout_seconds)
        now[0] += 5
        return result
    runtime.wait_candidate = wait_candidate
    members, group = runtime.boundary(3, members, group, request())
    assert group["generation"] == 1
    events = devices.events
    first_wait = events.index(("wait", "loaded"))
    assert events.index(("command", "load")) < first_wait
    assert events.index(("command", "load_parked")) < first_wait
    assert events.index(("wait", "boundary_held")) < events.index(("command", "load"))
    assert [r["timeout_seconds"] for r in devices.waits if r["phase"] == "loaded"] == [20, 15]
    hold = next(fields for action, _ids, fields in devices.commands if action == "hold_committed_boundary")
    assert hold == {"expected_step": 3, "expected_generation": 0, "hold_seconds": 60}


def test_shared_candidate_deadline_expires_without_parking_original_group():
    runtime, devices, members, group = mixed_candidates()
    now = [100.0]
    runtime.clock = lambda: now[0]
    def wait_candidate(cohort, token, phase, *, timeout_seconds):
        result = devices.wait(cohort, token, phase, timeout_seconds=timeout_seconds)
        now[0] += 21
        return result
    runtime.wait_candidate = wait_candidate
    same, same_group = runtime.boundary(3, members, group, request())
    assert same == members and same_group == group
    assert runtime.requests["to-new"]["status"] == "rejected"
    assert len([r for r in devices.waits if r["phase"] == "loaded"]) == 1
    assert devices.events[-2:] == [("wait", "boundary_released"), ("persist", "candidate_alignment_rejected")]
    assert not any(action in {"park", "join"} for action, _ids, _fields in devices.commands)
    assert all(devices.device[m["worker_id"]]["group"] == 0 for m in members)


def test_boundary_lease_is_finite_cannot_renew_and_releases_for_the_next_command():
    now = [100.0]
    lease = CommittedBoundaryLease(20, clock=lambda: now[0])
    command = {"expected_step": 3, "expected_generation": 0, "hold_seconds": 60}
    assert lease.remaining() == 20
    assert lease.arm(command, completed_step=3, generation=0) == 60
    now[0] += 21
    assert lease.remaining() == 39, "healthy active rank can wait beyond its ordinary idle timeout"
    with pytest.raises(ValueError, match="cannot renew"):
        lease.arm(command, completed_step=3, generation=0)
    now[0] = 160
    with pytest.raises(TimeoutError, match="lease expired"):
        lease.remaining()
    lease.release()
    assert lease.remaining() == 20
    lease.arm(command | {"expected_step": 6, "expected_generation": 1}, completed_step=6, generation=1)
    assert lease.remaining() == 60


@pytest.mark.parametrize("changes", [{"expected_step": 2}, {"expected_step": True},
                                    {"expected_generation": 1}, {"expected_generation": False},
                                    {"hold_seconds": 61}, {"hold_seconds": 0},
                                    {"hold_seconds": float("inf")}, {"hold_seconds": float("nan")}])
def test_boundary_lease_rejects_wrong_boundary_or_unbounded_wait(changes):
    lease = CommittedBoundaryLease(20)
    command = {"expected_step": 3, "expected_generation": 0, "hold_seconds": 60}
    with pytest.raises(ValueError):
        lease.arm(command | changes, completed_step=3, generation=0)
    assert lease.deadline is None


@pytest.mark.parametrize("slots", [True, "6", 3, 65])
def test_config_pool_slots_must_stay_within_active_and_physical_contract(slots):
    with pytest.raises(ValueError):
        validate_topology_config({"resident_topology_slots": slots}, 4, start_step=0, steps=8)


def test_config_rejects_out_of_pool_members_and_noninternal_boundaries():
    plan = {"boundary_step": 3, "retire_workers": ["w000003", "w000004"],
            "candidate_workers": ["w000005", "w000006"]}
    assert validate_topology_config({"resident_topology_slots": 6, "resident_topology_switches": [plan]},
                                    4, start_step=0, steps=8) == 6
    for changes in ({"boundary_step": 8}, {"boundary_step": 0}, {"boundary_step": 1.0},
                    {"candidate_workers": ["w000006", "w000007"]}):
        with pytest.raises(ValueError):
            validate_topology_config({"resident_topology_slots": 6, "resident_topology_switches": [plan | changes]},
                                     4, start_step=0, steps=8)


@pytest.fixture
def controller_liveness(tmp_path):
    """Exercise the real file adapter and real detector, without any devices."""
    path = Path(__file__).resolve().parents[1] / "vendor/scripts/probe_resident_openpi_fault.py"
    import_spec = importlib.util.spec_from_file_location("topology_active_liveness_adapter", path)
    module = importlib.util.module_from_spec(import_spec)
    import_spec.loader.exec_module(module)
    members = []
    for index in range(1, 7):
        member = {"worker_id": f"w{index:06d}", "session_id": f"session-{index}", "pid": 100 + index}
        supervisor = {**member, "run_id": "liveness-run", "supervisor_pid": 200 + index,
                      "heartbeat_counter": 1}
        module._write(tmp_path / "supervisors" / f"{member['worker_id']}.json", supervisor)
        members.append(member)
    watch = module.ControllerLiveness(tmp_path, {"run_id": "liveness-run", "heartbeat_timeout": 3600})
    watch.register_ready(members[:4], receipt="verified-initial-full-state")
    return module, watch, members


def finish_liveness_step(watch, token="old-step", *, generation=0, base_step=0):
    watch.begin_step(token, base_step=base_step, generation=generation, sessions=list(watch.keys.values()))
    for phase in ("reducing", "prepared", "applying", "committing"):
        watch.advance_phase(token, phase)
    watch.finish_step(token, committed_state_receipt=f"verified-commit/{token}")


def test_real_controller_liveness_excludes_parked_exit_from_the_new_active_step(controller_liveness):
    module, watch, members = controller_liveness
    finish_liveness_step(watch)
    active = members[:2] + members[4:]
    replacement = watch.rebuild_active(active, generation=1, committed_step=1, receipt="verified-boundary-1")
    assert set(replacement.keys) == {m["worker_id"] for m in active}
    # A previously active parked child exits after the group has switched.
    parked = members[2]
    supervisor = module._read(watch.run / "supervisors" / f"{parked['worker_id']}.json")
    module._write(watch.run / "exits" / f"{parked['worker_id']}-{parked['session_id']}.json",
                  {**supervisor, "exit_code": 137})
    assert replacement.poll() == ()
    finish_liveness_step(replacement, "new-step", generation=1, base_step=1)
    assert replacement.last_committed_step == 2
    journal = module._read(watch.run / "topology/liveness-generation-1.json")
    assert journal["retained_step_tokens"] == ["old-step"]
    assert {s["worker_id"] for s in journal["active_sessions"]} == set(replacement.keys)
    assert journal["parked_sessions_are_not_active_losses"] is True


def test_real_controller_liveness_keeps_fenced_sessions_tokens_epoch_and_generation(controller_liveness):
    from jade_lotus.resident_liveness import FencedSession
    module, watch, members = controller_liveness
    finish_liveness_step(watch)
    fenced = members[2]
    supervisor = module._read(watch.run / "supervisors" / f"{fenced['worker_id']}.json")
    module._write(watch.run / "exits" / f"{fenced['worker_id']}-{fenced['session_id']}.json",
                  {**supervisor, "exit_code": 137})
    assert watch.poll()
    fences = watch.detector.fenced_sessions
    replacement = watch.rebuild_active(members[:2] + members[4:], generation=1,
                                       committed_step=1, receipt="verified-boundary-1")
    assert replacement.detector.fenced_sessions == fences
    assert replacement.detector.epoch == watch.detector.epoch == 1
    with pytest.raises(FencedSession):
        replacement.register_ready([fenced], receipt="cannot-revive-with-another-receipt")
    with pytest.raises(FencedSession):
        replacement.begin_step("old-step", base_step=1, generation=1, sessions=list(replacement.keys.values()))
    with pytest.raises(FencedSession):
        replacement.begin_step("fresh-token-stale-group", base_step=1, generation=0,
                               sessions=list(replacement.keys.values()))
    finish_liveness_step(replacement, "fresh-step", generation=1, base_step=1)
    assert replacement.detector.epoch == 2
    assert list((watch.run / "losses").glob("*.json")), "old loss evidence must survive the active-view rebuild"


@pytest.mark.parametrize("phase", ["collecting", "prepared", "applying"])
def test_real_controller_liveness_never_rebuilds_a_pending_step(controller_liveness, phase):
    from jade_lotus.resident_liveness import LivenessError
    _module, watch, members = controller_liveness
    watch.begin_step("pending-step", base_step=0, generation=0, sessions=list(watch.keys.values()))
    if phase != "collecting":
        watch.advance_phase("pending-step", "reducing")
        watch.advance_phase("pending-step", "prepared")
    if phase == "applying":
        watch.advance_phase("pending-step", "applying")
    with pytest.raises(LivenessError):
        watch.rebuild_active(members[:2] + members[4:], generation=1, committed_step=1,
                             receipt="not-a-completed-commit")
    assert watch.detector.phase == phase and watch.detector.step_token == "pending-step"
    assert set(watch.keys) == {m["worker_id"] for m in members[:4]}


def test_real_controller_liveness_requires_the_actual_completed_commit_and_new_generation(controller_liveness):
    from jade_lotus.resident_liveness import LivenessError
    _module, watch, members = controller_liveness
    with pytest.raises(LivenessError):
        watch.rebuild_active(members[:2] + members[4:], generation=1, committed_step=1, receipt="no-commit-yet")
    finish_liveness_step(watch)
    for generation, step in ((0, 1), (1, 2)):
        with pytest.raises(LivenessError):
            watch.rebuild_active(members[:2] + members[4:], generation=generation, committed_step=step,
                                 receipt="wrong-generation-or-boundary")
