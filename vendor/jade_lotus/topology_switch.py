"""Auditable, IO-free protocol for whole-cohort resident topology changes.

This is a planning component, NOT an integrated elastic training controller.
It allocates no jobs, transfers no tensors and verifies no bytes. External
verifiers must check the full model, buffers, Adam moments/steps/options and
the config/RNG/precision contract before issuing a FullStateReceipt. Loading
base weights is never such a receipt. The single writer must persist each
transition before sending the associated command/acknowledgement.

Every queued/pinned/running/releasing reservation counts against the budget.
There is no implicit extra allocation: 64 active cards plus a 16-card candidate
needs another authorization, or a separately verified shrink/release first.
Old workers stay allocated across the first candidate COMMIT. They become hot
standby only after full-state synchronization to that COMMIT, and become stale
again at every subsequent COMMIT. Released/fenced sessions never revive.

Pre-APPLY repair requires an external collective-abort and healthy-gradient
preservation receipt. Loss after APPLY authorization requires explicit recovery;
this module deliberately has no automatic arbitrary-Adam-midpoint recovery.
CPU protocol tests establish none of the corresponding HCCL/NPU behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import hashlib
import json
import math
from typing import Mapping

from .resident_liveness import OFFICIAL_TERMINAL, SessionKey
from .resident_membership import StateSummary


class SwitchError(ValueError):
    """An operation violates the switching protocol."""


class StaleSwitch(SwitchError):
    """The run, state, generation, operation or session fence is obsolete."""


class SwitchRecoveryRequired(SwitchError):
    """Optimizer writes may have begun; outside full-state recovery is needed."""


class CohortStatus(str, Enum):
    ACTIVE = "active"
    CANDIDATE_LOADING = "candidate_loading"
    CANDIDATE_READY = "candidate_ready"
    TRIAL = "trial"
    ROLLBACK_HOLD = "rollback_hold"
    STANDBY_LOADING = "standby_loading"
    HOT_STANDBY = "hot_standby"
    LOST = "lost"
    RELEASE_PENDING = "release_pending"
    RELEASED = "released"


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise SwitchError(f"{name} must be nonempty text")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise SwitchError(f"{name} must be an integer >= {minimum}")


def _seconds(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise SwitchError(f"{name} must be finite and nonnegative")


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise SwitchError(f"{name} must be a lowercase SHA256")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _state(value):
    if not isinstance(value, StateSummary):
        raise SwitchError("A complete model/optimizer/step StateSummary is required")
    # Revalidate even an externally deserialized/bypassed dataclass instance.
    _integer(value.committed_step, "committed_step")
    _sha(value.model_sha256, "model_sha256")
    _sha(value.optimizer_sha256, "optimizer_sha256")


@dataclass(frozen=True)
class VersionFence:
    run_id: str
    generation: int
    state: StateSummary
    contract_sha256: str

    def __post_init__(self):
        _text(self.run_id, "run_id")
        _integer(self.generation, "generation")
        _state(self.state)
        _sha(self.contract_sha256, "config/RNG/precision contract")


@dataclass(frozen=True)
class Reservation:
    """A unique platform allocation incarnation, including queued/pinned jobs.

    resource_id must bind the official job ID/running round, never only a pod
    hostname. A pod with 16 devices is one 16-card claim/failure domain; 16
    independent jobs are 16 claims. IDs cannot be reused after release.
    """

    resource_id: str
    cards: int
    failure_domain: str
    state: str = "queued"
    pinned: bool = False

    def __post_init__(self):
        _text(self.resource_id, "resource_id")
        _integer(self.cards, "reserved cards", 1)
        _text(self.failure_domain, "failure_domain")
        if self.state not in {"queued", "running", "release_pending", "released"}:
            raise SwitchError("Unknown reservation state")
        if type(self.pinned) is not bool:
            raise SwitchError("pinned must be a bool")


@dataclass(frozen=True)
class Member:
    session: SessionKey
    resource_id: str

    def __post_init__(self):
        if not isinstance(self.session, SessionKey):
            raise SwitchError("An exact supervisor/child/job SessionKey is required")
        _text(self.resource_id, "member resource_id")


@dataclass(frozen=True)
class FullStateReceipt:
    session: SessionKey
    fence: VersionFence
    operation_id: str
    state: StateSummary
    verification_receipt: str
    verified_full_state: bool

    def __post_init__(self):
        if not isinstance(self.session, SessionKey) or not isinstance(self.fence, VersionFence):
            raise SwitchError("Full-state receipt needs exact session/version fences")
        _text(self.operation_id, "state operation_id")
        _state(self.state)
        _text(self.verification_receipt, "external byte-verification receipt")
        if self.verified_full_state is not True:
            raise SwitchError("Weights or digest announcements are not verified complete Adam state")


@dataclass(frozen=True)
class SwitchEstimate:
    current_step_seconds: float
    candidate_step_seconds: float
    remaining_steps: int
    load_seconds: float
    sync_seconds: float
    rebuild_seconds: float
    verify_seconds: float
    standby_sync_seconds: float
    stable_observations: int

    def __post_init__(self):
        for name in ("current_step_seconds", "candidate_step_seconds", "load_seconds", "sync_seconds",
                     "rebuild_seconds", "verify_seconds", "standby_sync_seconds"):
            _seconds(getattr(self, name), name)
        if self.current_step_seconds == 0 or self.candidate_step_seconds == 0:
            raise SwitchError("Step time estimates must be positive")
        _integer(self.remaining_steps, "remaining_steps", 1)
        _integer(self.stable_observations, "stable_observations")

    @property
    def boundary_pause_seconds(self):
        return self.sync_seconds + self.rebuild_seconds + self.verify_seconds + self.standby_sync_seconds

    @property
    def expected_savings_seconds(self):
        # Conservative: charge overlapping initial loading too. Repeated hot
        # standby synchronization is a per-step cost, not a one-off free backup.
        return ((self.current_step_seconds - self.candidate_step_seconds - self.standby_sync_seconds)
                * self.remaining_steps - self.load_seconds - self.sync_seconds
                - self.rebuild_seconds - self.verify_seconds)


@dataclass(frozen=True)
class SwitchPolicy:
    minimum_savings_seconds: float = 30.0
    minimum_speedup_fraction: float = 0.05
    minimum_stable_observations: int = 3
    cooldown_steps: int = 10
    maximum_boundary_pause_seconds: float = 60.0

    def __post_init__(self):
        for name in ("minimum_savings_seconds", "minimum_speedup_fraction", "maximum_boundary_pause_seconds"):
            _seconds(getattr(self, name), name)
        if self.minimum_speedup_fraction >= 1:
            raise SwitchError("minimum_speedup_fraction must be below 1")
        _integer(self.minimum_stable_observations, "minimum_stable_observations", 1)
        _integer(self.cooldown_steps, "cooldown_steps")

    def check(self, estimate, *, committed_step, last_switch_step):
        if not isinstance(estimate, SwitchEstimate):
            raise SwitchError("A measured, auditable SwitchEstimate is required")
        speedup = ((estimate.current_step_seconds - estimate.candidate_step_seconds
                    - estimate.standby_sync_seconds) / estimate.current_step_seconds)
        if (estimate.stable_observations < self.minimum_stable_observations
                or estimate.expected_savings_seconds < self.minimum_savings_seconds
                or speedup < self.minimum_speedup_fraction
                or estimate.boundary_pause_seconds > self.maximum_boundary_pause_seconds
                or (last_switch_step is not None and committed_step - last_switch_step < self.cooldown_steps)):
            raise SwitchError("Switch benefit/stability/boundary-pause/cooldown threshold is not met")


@dataclass(frozen=True)
class SampleAssignment:
    unit_id: int
    session: SessionKey
    sample_positions: tuple[int, ...]


@dataclass(frozen=True)
class StepPlan:
    token: str
    fence: VersionFence
    members: tuple[SessionKey, ...]
    assignments: tuple[SampleAssignment, ...]
    input_identity_sha256: str


@dataclass(frozen=True)
class CoverageReceipt:
    step_token: str
    fence: VersionFence
    input_identity_sha256: str
    sample_positions: tuple[int, ...]
    verification_receipt: str
    verified_original_samples: bool


@dataclass(frozen=True)
class RepairReceipt:
    step_token: str
    fence: VersionFence
    input_identity_sha256: str
    preserved_units: tuple[int, ...]
    verification_receipt: str
    collectives_aborted: bool
    healthy_original_gradients_preserved: bool


@dataclass(frozen=True)
class CommitReceipt:
    step_token: str
    fence: VersionFence
    target_state: StateSummary
    input_identity_sha256: str
    states: tuple[FullStateReceipt, ...]


@dataclass(frozen=True)
class TerminalReceipt:
    resource_id: str
    official_status: str
    evidence_ref: str

    def __post_init__(self):
        _text(self.resource_id, "terminal resource_id")
        _text(self.evidence_ref, "official terminal evidence_ref")
        if self.official_status not in OFFICIAL_TERMINAL:
            raise SwitchError("A queued/deleting/local-stop claim is not an official terminal state")


@dataclass
class _Cohort:
    cohort_id: str
    resource_ids: tuple[str, ...]
    members: tuple[Member, ...]
    status: CohortStatus
    state: StateSummary | None
    replaces: str | None = None
    backup_for: str | None = None
    releasable: bool = False
    sync_operation: tuple[str, VersionFence] | None = None
    receipts: tuple[FullStateReceipt, ...] = ()
    estimate: SwitchEstimate | None = None


class TopologySwitch:
    """Serialized controller-side protocol; all effects are caller-owned.

    Initial cohorts must each own entire reservations (no cross-cohort shared
    pod); this makes the card ledger and correlated failures unambiguous. Keep
    platform monitoring armed for every cohort, including queued/loading/hot
    standby. An unconfirmed timeout fences a session but never frees its claim.
    """

    def __init__(self, run_id, committed_state, *, contract_sha256, reservations,
                 active_cohorts: Mapping[str, tuple[Member, ...]], initial_receipts,
                 card_budget=64, policy=None):
        _integer(card_budget, "authorized card_budget", 1)
        self.card_budget = card_budget
        self.policy = policy if policy is not None else SwitchPolicy()
        if not isinstance(self.policy, SwitchPolicy):
            raise SwitchError("policy must be a SwitchPolicy")
        self._fence = VersionFence(run_id, 0, committed_state, contract_sha256)
        claims = tuple(reservations)
        if (any(not isinstance(r, Reservation) or r.state != "running" for r in claims)
                or len({r.resource_id for r in claims}) != len(claims)):
            raise SwitchError("Initial claims must be distinct running reservations")
        self._resources = {r.resource_id: r for r in claims}
        self._cohorts, self._lost, self._seen_sessions = {}, set(), set()
        self._unrepaired_losses = set()
        self.phase, self.step = "idle", None
        self._last_switch_step = None
        self._tokens, self._sync_tokens, self._losses, self._commits = set(), set(), {}, {}
        self._journal, self._terminal_receipts = [], {}
        if not active_cohorts:
            raise SwitchError("At least one active cohort is required")
        owned = set()
        for name, members in active_cohorts.items():
            _text(name, "cohort_id")
            members = tuple(members)
            resources = tuple(sorted({m.resource_id for m in members if isinstance(m, Member)}))
            self._check_members(members, resources)
            if owned.intersection(resources):
                raise SwitchError("A reservation cannot be shared by initial cohorts")
            owned.update(resources)
            self._cohorts[name] = _Cohort(name, resources, members, CohortStatus.ACTIVE, committed_state)
            self._seen_sessions.update(m.session for m in members)
        if owned != self._resources.keys() or self.reserved_cards > self.card_budget:
            raise SwitchError("Initial claims must all be owned and within the authorized budget")
        initial_receipts = tuple(initial_receipts)
        self._verify_states(initial_receipts, self.active_sessions, "initial", self.fence, committed_state)
        self._record("initialized", initial_receipts=[asdict(r) for r in initial_receipts])

    @property
    def fence(self):
        return self._fence

    @property
    def committed_state(self):
        return self.fence.state

    @property
    def reserved_cards(self):
        return sum(r.cards for r in self._resources.values() if r.state != "released")

    @property
    def active_sessions(self):
        return tuple(m.session for c in self._cohorts.values()
                     if c.status in {CohortStatus.ACTIVE, CohortStatus.TRIAL}
                     for m in c.members if m.session not in self._lost)

    @property
    def audit_log(self):
        # Roundtrip prevents callers from modifying nested evidence/history.
        return tuple(json.loads(json.dumps(record)) for record in self._journal)

    def snapshot(self):
        return {"schema": "jade-topology-switch-v1", "phase": self.phase,
                "fence": asdict(self.fence), "card_budget": self.card_budget,
                "reserved_cards": self.reserved_cards,
                "resources": [asdict(r) for r in self._resources.values()],
                "cohorts": [asdict(c) for c in self._cohorts.values()],
                "fenced_sessions": [asdict(s) for s in sorted(self._lost, key=repr)],
                "unrepaired_sessions": [asdict(s) for s in sorted(self._unrepaired_losses, key=repr)],
                "step": asdict(self.step) if self.step else None,
                "last_switch_step": self._last_switch_step,
                "audit_head": self._journal[-1]["sha256"] if self._journal else None,
                "integration_required": True}

    def _record(self, event, **detail):
        row = {"sequence": len(self._journal), "event": event, "phase": self.phase,
               "fence": asdict(self.fence), "reserved_cards": self.reserved_cards,
               "previous_sha256": self._journal[-1]["sha256"] if self._journal else None,
               "detail": detail}
        row["sha256"] = _hash(row)
        self._journal.append(row)

    def _check_fence(self, fence):
        if not isinstance(fence, VersionFence) or fence != self.fence:
            raise StaleSwitch("Run/state/contract/topology generation fence is not current")
        if self.phase == "recovery_required":
            raise SwitchRecoveryRequired("External full-state recovery is required before further switching")

    def _cohort(self, name):
        if name not in self._cohorts:
            raise StaleSwitch("Unknown cohort incarnation")
        return self._cohorts[name]

    def _check_members(self, members, resource_ids):
        if not members or any(not isinstance(m, Member) for m in members):
            raise SwitchError("A cohort needs exact Member records")
        sessions = tuple(m.session for m in members)
        if (len(set(sessions)) != len(sessions) or len({s.worker_id for s in sessions}) != len(sessions)
                or any(s.run_id != self.fence.run_id for s in sessions)
                or any(s in self._seen_sessions or s.worker_id in {old.worker_id for old in self._seen_sessions}
                       for s in sessions)
                or any(m.resource_id not in resource_ids for m in members)):
            raise StaleSwitch("Cohort sessions must be distinct, current-run and never previously admitted")
        for resource in resource_ids:
            if resource not in self._resources:
                raise SwitchError("Member references an unknown reservation")
            if sum(m.resource_id == resource for m in members) != self._resources[resource].cards:
                raise SwitchError("Exactly one worker per reserved card is required")

    def _verify_states(self, receipts, sessions, operation, fence, state):
        _state(state)
        receipts = tuple(receipts)
        if (any(not isinstance(r, FullStateReceipt) for r in receipts)
                or len({r.session for r in receipts}) != len(receipts)
                or {r.session for r in receipts} != set(sessions)):
            raise SwitchError("Exactly one full-state verification receipt per required session is needed")
        if any(r.fence != fence or r.operation_id != operation or r.state != state
               or r.verified_full_state is not True for r in receipts):
            raise StaleSwitch("Complete model/Adam/step/contract/operation verification does not match")

    def queue_candidate(self, candidate_id, *, replaces, reservations, estimate, fence):
        """Reserve budget BEFORE submitting any external candidate jobs."""
        self._check_fence(fence)
        _text(candidate_id, "candidate_id")
        target = self._cohort(replaces)
        if candidate_id in self._cohorts or target.status != CohortStatus.ACTIVE:
            raise SwitchError("A fresh candidate must replace an active cohort")
        if any(c.status in {CohortStatus.CANDIDATE_LOADING, CohortStatus.CANDIDATE_READY, CohortStatus.TRIAL}
               for c in self._cohorts.values()):
            raise SwitchError("Finish or cancel the previous candidate before reserving another")
        self.policy.check(estimate, committed_step=self.committed_state.committed_step,
                          last_switch_step=self._last_switch_step)
        claims = tuple(reservations)
        if (not claims or any(not isinstance(r, Reservation) or r.state != "queued" for r in claims)
                or len({r.resource_id for r in claims}) != len(claims)
                or any(r.resource_id in self._resources for r in claims)):
            raise SwitchError("Candidate needs fresh queued reservation incarnations")
        if sum(r.cards for r in claims) != len(target.members):
            raise SwitchError("A cohort replacement must reserve the entire target card count")
        if self.reserved_cards + sum(r.cards for r in claims) > self.card_budget:
            raise SwitchError("Candidate/queued/pinned/standby reservations exceed the authorized card budget")
        self._resources.update((r.resource_id, r) for r in claims)
        self._cohorts[candidate_id] = _Cohort(candidate_id, tuple(r.resource_id for r in claims), (),
            CohortStatus.CANDIDATE_LOADING, None, replaces=replaces, estimate=estimate)
        self._record("candidate_reserved", candidate_id=candidate_id, replaces=replaces,
                     reservations=[asdict(r) for r in claims], estimate=asdict(estimate))

    def bind_candidate(self, candidate_id, members, *, running_receipt, fence):
        self._check_fence(fence)
        cohort = self._cohort(candidate_id)
        members = tuple(members)
        _text(running_receipt, "official running/session binding receipt")
        if cohort.status != CohortStatus.CANDIDATE_LOADING or cohort.members:
            raise SwitchError("Only an unbound loading candidate can register workers")
        self._check_members(members, cohort.resource_ids)
        cohort.members = members
        self._seen_sessions.update(m.session for m in members)
        for resource in cohort.resource_ids:
            self._resources[resource] = replace(self._resources[resource], state="running")
        self._record("candidate_bound", cohort_id=candidate_id, receipt=running_receipt,
                     members=[asdict(m) for m in members])

    def begin_sync(self, cohort_id, operation_id, *, fence):
        self._check_fence(fence)
        _text(operation_id, "sync operation_id")
        cohort = self._cohort(cohort_id)
        if (cohort.status not in {CohortStatus.CANDIDATE_LOADING, CohortStatus.STANDBY_LOADING}
                or not cohort.members or any(m.session in self._lost for m in cohort.members)):
            raise SwitchError("Only a healthy allocated loading cohort may synchronize")
        if operation_id in self._sync_tokens:
            raise StaleSwitch("A full-state transfer must use a fresh operation ID")
        self._sync_tokens.add(operation_id)
        cohort.sync_operation = (operation_id, fence)
        # A failed/partial copy invalidates any earlier state verification.
        cohort.state, cohort.receipts = None, ()
        self._record("sync_started", cohort_id=cohort_id, operation_id=operation_id)

    def complete_sync(self, cohort_id, operation_id, receipts, *, fence):
        self._check_fence(fence)
        cohort = self._cohort(cohort_id)
        receipts = tuple(receipts)
        if (cohort.status not in {CohortStatus.CANDIDATE_LOADING, CohortStatus.STANDBY_LOADING}
                or cohort.sync_operation != (operation_id, fence)
                or any(m.session in self._lost for m in cohort.members)):
            raise StaleSwitch("Sync completion is outside its current healthy loading operation")
        self._verify_states(receipts, (m.session for m in cohort.members), operation_id, fence, self.committed_state)
        cohort.state, cohort.receipts, cohort.sync_operation = self.committed_state, receipts, None
        cohort.status = (CohortStatus.CANDIDATE_READY if cohort.replaces is not None
                         else CohortStatus.HOT_STANDBY)
        self._record("sync_verified", cohort_id=cohort_id, operation_id=operation_id,
                     receipts=[asdict(r) for r in receipts])

    def activate_candidate(self, candidate_id, *, estimate, fence):
        self._check_fence(fence)
        cohort = self._cohort(candidate_id)
        if self.phase != "idle" or cohort.status != CohortStatus.CANDIDATE_READY or cohort.state != self.committed_state:
            raise SwitchError("A current complete-state candidate activates only at an idle COMMIT boundary")
        target = self._cohort(cohort.replaces)
        if target.status != CohortStatus.ACTIVE:
            raise SwitchError("Candidate target is no longer active")
        self.policy.check(estimate, committed_step=self.committed_state.committed_step,
                          last_switch_step=self._last_switch_step)
        target.status, target.backup_for = CohortStatus.ROLLBACK_HOLD, candidate_id
        cohort.status, cohort.estimate = CohortStatus.TRIAL, estimate
        self._last_switch_step = self.committed_state.committed_step
        self._fence = replace(self.fence, generation=self.fence.generation + 1)
        self._record("candidate_trial", candidate_id=candidate_id, rollback_hold=target.cohort_id,
                     estimate=asdict(estimate))

    def _backup_pair(self, backup_id):
        backup = self._cohort(backup_id)
        if (backup.status not in {CohortStatus.HOT_STANDBY, CohortStatus.ROLLBACK_HOLD}
                or backup.state != self.committed_state or not backup.backup_for
                or any(m.session in self._lost for m in backup.members)):
            raise SwitchError("Backup must be allocated, healthy and verified at the exact current model/Adam/step")
        target = self._cohort(backup.backup_for)
        if target.status not in {CohortStatus.ACTIVE, CohortStatus.TRIAL}:
            raise SwitchError("Backup no longer corresponds to an active cohort")
        return backup, target

    def _promote_backup(self, backup, target):
        backup.status, backup.backup_for = CohortStatus.ACTIVE, None
        self._unrepaired_losses.difference_update(m.session for m in target.members)
        # Old trial/candidate sessions are not silently resurrected or left hot.
        target.status = (CohortStatus.LOST if any(m.session in self._lost for m in target.members)
                         else CohortStatus.STANDBY_LOADING)
        target.replaces, target.backup_for, target.releasable = None, backup.cohort_id, True
        target.sync_operation, target.state, target.receipts = None, None, ()
        self._last_switch_step = self.committed_state.committed_step

    def activate_backup(self, backup_id, *, fence):
        self._check_fence(fence)
        if self.phase != "idle":
            raise SwitchError("In-flight work requires verified pre-APPLY repair; a backup cannot bypass it")
        backup, target = self._backup_pair(backup_id)
        self._promote_backup(backup, target)
        self._fence = replace(self.fence, generation=self.fence.generation + 1)
        self._record("backup_activated", backup_id=backup_id, replaced=target.cohort_id)

    def begin_step(self, token, *, fence):
        self._check_fence(fence)
        _text(token, "step token")
        if self.phase != "idle" or token in self._tokens:
            raise StaleSwitch("A step needs an idle controller and a never-used token")
        active = tuple(c for c in self._cohorts.values() if c.status in {CohortStatus.ACTIVE, CohortStatus.TRIAL})
        if any(c.state != self.committed_state or any(m.session in self._unrepaired_losses for m in c.members) for c in active):
            raise SwitchError("Active loss/stale state must be repaired or replaced before starting a new step")
        members = self.active_sessions
        if not members or len(members) > 64:
            raise SwitchError("Global256 atomic4 needs 1..64 active sessions")
        start = self.committed_state.committed_step * 256
        rows = tuple(SampleAssignment(unit, members[unit % len(members)], tuple(range(start + 4 * unit, start + 4 * unit + 4)))
                     for unit in range(64))
        identity = _hash({"run_id": self.fence.run_id, "contract": self.fence.contract_sha256,
                          "base_state": asdict(self.committed_state), "sample_positions": list(range(start, start + 256))})
        self.step = StepPlan(token, fence, members, rows, identity)
        self._tokens.add(token)
        self.phase = "collecting"
        self._record("step_frozen", plan=asdict(self.step))
        return self.step

    def _step_fence(self, token, fence):
        self._check_fence(fence)
        if self.step is None or self.step.token != token or self.step.fence != fence:
            raise StaleSwitch("Step token or frozen generation is obsolete")

    def authorize_apply(self, coverage: CoverageReceipt):
        if not isinstance(coverage, CoverageReceipt):
            raise SwitchError("A caller-verified original-sample coverage receipt is required")
        self._step_fence(coverage.step_token, coverage.fence)
        start = self.committed_state.committed_step * 256
        if (self.phase != "collecting" or coverage.input_identity_sha256 != self.step.input_identity_sha256
                or type(coverage.sample_positions) is not tuple
                or any(type(p) is not int for p in coverage.sample_positions)
                or coverage.sample_positions != tuple(range(start, start + 256))
                or coverage.verified_original_samples is not True):
            raise SwitchError("APPLY needs exactly the original 256 samples once, with current verified contributions")
        _text(coverage.verification_receipt, "original contribution coverage receipt")
        # Persist this transition BEFORE any worker is authorized to write Adam.
        self.phase = "applying"
        self._record("apply_authorized", coverage=asdict(coverage))

    def commit_step(self, receipt: CommitReceipt):
        if not isinstance(receipt, CommitReceipt):
            raise SwitchError("A complete CommitReceipt is required")
        # Retain immutable evidence even when an integration supplied a list
        # despite the public tuple annotation.
        receipt = replace(receipt, states=tuple(receipt.states))
        if receipt.step_token in self._commits:
            old = self._commits[receipt.step_token]
            if old != receipt:
                raise StaleSwitch("A committed operation cannot change its evidence")
            return old.target_state  # Idempotent acknowledgement, never rollback.
        self._step_fence(receipt.step_token, receipt.fence)
        if self.phase != "applying":
            raise SwitchError("Only an authorized optimizer update can commit")
        try:
            _state(receipt.target_state)
            if (receipt.target_state.committed_step != self.committed_state.committed_step + 1
                    or receipt.input_identity_sha256 != self.step.input_identity_sha256):
                raise SwitchError("Commit target step or original input identity differs")
            self._verify_states(receipt.states, self.step.members, self.step.token, self.fence, receipt.target_state)
        except SwitchError:
            self.phase = "recovery_required"
            self._record("commit_verification_failed", step_token=receipt.step_token)
            raise SwitchRecoveryRequired("Complete applied states disagree; external full-state recovery is required")
        self._commits[receipt.step_token] = receipt
        trial_ids = {c.cohort_id for c in self._cohorts.values() if c.status == CohortStatus.TRIAL}
        for cohort in self._cohorts.values():
            cohort.sync_operation = None
            if cohort.backup_for in trial_ids:
                cohort.releasable = True
            if cohort.status in {CohortStatus.ACTIVE, CohortStatus.TRIAL}:
                cohort.status, cohort.state = CohortStatus.ACTIVE, receipt.target_state
                cohort.receipts = tuple(r for r in receipt.states if r.session in {m.session for m in cohort.members})
            elif cohort.status == CohortStatus.ROLLBACK_HOLD:
                # First successful new-group step permits release, but is NOT a
                # verification that the old group's Adam state was synchronized.
                cohort.status, cohort.releasable, cohort.replaces = CohortStatus.STANDBY_LOADING, True, None
                cohort.state, cohort.receipts = None, ()
            elif cohort.status in {CohortStatus.CANDIDATE_READY, CohortStatus.HOT_STANDBY}:
                cohort.status = (CohortStatus.CANDIDATE_LOADING if cohort.status == CohortStatus.CANDIDATE_READY
                                 else CohortStatus.STANDBY_LOADING)
                cohort.state, cohort.receipts = None, ()
        self._fence = replace(self.fence, state=receipt.target_state)
        self.phase, self.step = "idle", None
        self._record("step_committed", receipt=asdict(receipt))
        return receipt.target_state

    def report_loss(self, event_id, *, fence, evidence_ref, sessions=(), failure_domain=None):
        """Fence an exact session set OR all devices in a correlated pod/domain.

        No resource is freed here, including for unconfirmed heartbeat expiry.
        Duplicate events return the original action without extending leases.
        """
        _text(event_id, "loss event_id")
        _text(evidence_ref, "loss evidence_ref")
        sessions = tuple(sessions)
        payload = (fence, evidence_ref, sessions, failure_domain)
        if event_id in self._losses:
            if self._losses[event_id][0] != payload:
                raise StaleSwitch("A loss event ID cannot change its fenced identity")
            return self._losses[event_id][1]
        # Losses remain admissible while recovery_required so that monitoring
        # continues to fence further sessions; ordinary mutation stays blocked.
        if not isinstance(fence, VersionFence) or fence != self.fence:
            raise StaleSwitch("Loss belongs to an obsolete run/state/generation")
        if bool(sessions) == (failure_domain is not None):
            raise SwitchError("Supply exact sessions or one correlated failure_domain")
        if failure_domain is not None:
            _text(failure_domain, "loss failure_domain")
            resources = {r.resource_id for r in self._resources.values()
                         if r.failure_domain == failure_domain and r.state != "released"}
            if not resources:
                raise StaleSwitch("Unknown or released failure domain")
            sessions = tuple(m.session for c in self._cohorts.values() for m in c.members if m.resource_id in resources)
        if (len(set(sessions)) != len(sessions) or any(s not in self._seen_sessions for s in sessions)
                or any(c.status == CohortStatus.RELEASED and any(m.session in sessions for m in c.members)
                       for c in self._cohorts.values())):
            raise StaleSwitch("Loss must name unique admitted, unreleased exact sessions")
        active_lost = set(sessions).intersection(self.active_sessions)
        active_loss = bool(active_lost)
        self._unrepaired_losses.update(active_lost)
        self._lost.update(sessions)
        for cohort in self._cohorts.values():
            affected = any(m.session in sessions for m in cohort.members)
            # Queued pods have no worker sessions yet, but pod termination still
            # cancels the candidate without pretending its allocation is free.
            affected = affected or (failure_domain is not None and any(
                self._resources[r].failure_domain == failure_domain for r in cohort.resource_ids))
            if affected:
                cohort.sync_operation = None
                if cohort.status not in {CohortStatus.ACTIVE, CohortStatus.TRIAL, CohortStatus.RELEASE_PENDING}:
                    cohort.status, cohort.state, cohort.receipts = CohortStatus.LOST, None, ()
        if active_loss and self.phase in {"applying", "recovery_required"}:
            self.phase, action = "recovery_required", "recovery_required"
        elif active_loss and self.phase in {"collecting", "repair_required"}:
            self.phase, action = "repair_required", "repair_pre_apply"
        elif active_loss:
            action = "replace_at_boundary"
        else:
            action = "discard_nonactive_cohort"
        self._losses[event_id] = (payload, action)
        self._record("loss_fenced", event_id=event_id, evidence_ref=evidence_ref,
                     sessions=[asdict(s) for s in sessions], failure_domain=failure_domain, action=action)
        return action

    def repair_pre_apply(self, receipt: RepairReceipt, new_token, *, backup_id=None):
        """Preserve every healthy original unit; reassign only fenced units.

        A whole-cohort backup may enter this frozen step only if the entire
        outgoing cohort was fenced. Otherwise use the surviving frozen workers
        to finish the step and resynchronize a backup at its next boundary.
        Device abort/rebuild and preservation verification remain external.
        """
        if not isinstance(receipt, RepairReceipt):
            raise SwitchError("An external healthy-gradient/collective-abort RepairReceipt is required")
        self._step_fence(receipt.step_token, receipt.fence)
        _text(new_token, "fresh repaired step token")
        expected = tuple(a.unit_id for a in self.step.assignments if a.session not in self._lost)
        if (self.phase != "repair_required" or new_token in self._tokens
                or receipt.input_identity_sha256 != self.step.input_identity_sha256
                or type(receipt.preserved_units) is not tuple
                or any(type(unit) is not int for unit in receipt.preserved_units)
                or receipt.preserved_units != expected or receipt.collectives_aborted is not True
                or receipt.healthy_original_gradients_preserved is not True):
            raise SwitchError("Repair must abort old communication and preserve every healthy original gradient")
        _text(receipt.verification_receipt, "healthy original gradient preservation receipt")
        pair = self._backup_pair(backup_id) if backup_id is not None else None
        if pair is not None and any(m.session not in self._lost for m in pair[1].members):
            raise SwitchError("Do not discard healthy frozen workers to insert a whole-cohort backup mid-step")
        members = tuple(s for s in self.step.members if s not in self._lost)
        if pair is not None:
            members += tuple(m.session for m in pair[0].members)
        if not members:
            raise SwitchError("No verified current-state worker survives; explicit recovery is required")
        if pair is not None:
            self._promote_backup(*pair)
        load = {s: sum(a.session == s and a.unit_id in expected for a in self.step.assignments) for s in members}
        rows = []
        for assignment in self.step.assignments:
            if assignment.unit_id in expected:
                rows.append(assignment)
            else:
                chosen = min(members, key=lambda s: (load[s], repr(s)))
                rows.append(replace(assignment, session=chosen))
                load[chosen] += 1
        self._fence = replace(self.fence, generation=self.fence.generation + 1)
        self.step = StepPlan(new_token, self.fence, members, tuple(rows), self.step.input_identity_sha256)
        self._tokens.add(new_token)
        self._unrepaired_losses.difference_update(self._lost)
        self.phase = "collecting"
        self._record("pre_apply_repaired", receipt=asdict(receipt), plan=asdict(self.step), backup_id=backup_id)
        return self.step

    def request_release(self, cohort_id, *, fence):
        self._check_fence(fence)
        cohort = self._cohort(cohort_id)
        if cohort.status in {CohortStatus.RELEASE_PENDING, CohortStatus.RELEASED}:
            return
        if (self.phase != "idle" or cohort.status in {CohortStatus.ACTIVE, CohortStatus.TRIAL, CohortStatus.ROLLBACK_HOLD}
                or (cohort.backup_for is not None and not cohort.releasable)):
            raise SwitchError("Active/trial/rollback-hold resources cannot release before the first successful new COMMIT")
        cohort.status, cohort.state, cohort.sync_operation = CohortStatus.RELEASE_PENDING, None, None
        cohort.receipts = ()
        for resource in cohort.resource_ids:
            self._resources[resource] = replace(self._resources[resource], state="release_pending")
        self._record("release_requested", cohort_id=cohort_id)

    def confirm_release(self, cohort_id, receipts, *, fence):
        self._check_fence(fence)
        cohort = self._cohort(cohort_id)
        receipts = tuple(receipts)
        if cohort.status == CohortStatus.RELEASED:
            if self._terminal_receipts[cohort_id] != receipts:
                raise StaleSwitch("Released allocation evidence cannot change")
            return
        if (cohort.status != CohortStatus.RELEASE_PENDING
                or any(not isinstance(r, TerminalReceipt) for r in receipts)
                or len({r.resource_id for r in receipts}) != len(receipts)
                or {r.resource_id for r in receipts} != set(cohort.resource_ids)):
            raise SwitchError("Every exact reservation incarnation needs an official terminal receipt")
        for resource in cohort.resource_ids:
            self._resources[resource] = replace(self._resources[resource], state="released")
        self._lost.update(m.session for m in cohort.members)
        cohort.status = CohortStatus.RELEASED
        self._terminal_receipts[cohort_id] = receipts
        self._record("release_confirmed", cohort_id=cohort_id, receipts=[asdict(r) for r in receipts])
