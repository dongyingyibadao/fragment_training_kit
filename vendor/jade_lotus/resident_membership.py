"""Pure CPU protocol model for independently replaceable resident workers.

This module performs no IO, tensor operations or verification of storage bytes.
The caller must verify every supplied evidence record and persist transitions
atomically before sending the corresponding network acknowledgement. A state
digest covers the complete model and the complete optimizer (moments, step and
configuration), never just weights. A local device gradient is only ``ready``;
``accepted`` requires a verified durable copy outside its worker failure domain.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
import uuid


class MembershipError(ValueError):
    """A transition does not satisfy the resident protocol."""


class StaleMessage(MembershipError):
    """An epoch, session, assignment or update fence no longer matches."""


class RecoveryRequired(MembershipError):
    """An optimizer may have partially applied; explicit recovery is required."""


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise MembershipError(f"{name} must be a nonempty string")


def _digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise MembershipError(f"{name} must be a lowercase SHA256 digest")


@dataclass(frozen=True)
class StateSummary:
    committed_step: int
    model_sha256: str
    optimizer_sha256: str

    def __post_init__(self):
        if type(self.committed_step) is not int or self.committed_step < 0:
            raise MembershipError("committed_step must be a nonnegative integer")
        _digest(self.model_sha256, "model_sha256")
        _digest(self.optimizer_sha256, "optimizer_sha256")


@dataclass(frozen=True)
class StateEvidence:
    """A caller-verified durable checkpoint containing model AND optimizer."""

    state: StateSummary
    location: str
    verified: bool
    durable: bool


@dataclass(frozen=True)
class AppliedStateEvidence:
    """External byte-verification receipt for a live worker's complete p/m/v.

    A digest announcement is insufficient. The integration must verify actual
    model bytes, both optimizer moments, optimizer step/configuration and the
    exact prepared input identity before setting ``verified_full_state``. The
    receipt names that external verification; this CPU model cannot perform it.
    """

    update_id: str
    epoch: int
    input_identity_sha256: str
    worker_id: str
    session_id: str
    state: StateSummary
    verification_receipt: str
    verified_full_state: bool


@dataclass(frozen=True)
class Assignment:
    epoch: int
    base_step: int
    unit_id: int
    sample_positions: tuple[int, ...]
    worker_id: str
    session_id: str
    attempt_id: str


@dataclass(frozen=True)
class ReplicaEvidence:
    """Receipt from an external verifier, bound to exact contribution bytes.

    ``durable`` means the replica outlives loss of its source worker; another
    unreplicated device-local tensor does not satisfy this contract.
    """

    contribution_id: str
    gradient_sha256: str
    location: str
    failure_domain: str
    verified: bool
    durable: bool


@dataclass(frozen=True)
class Contribution:
    assignment: Assignment
    contribution_id: str
    gradient_sha256: str
    replica: ReplicaEvidence | None = None

    @property
    def accepted(self):
        return self.replica is not None


@dataclass(frozen=True)
class UpdatePlan:
    epoch: int
    update_id: str
    base_state: StateSummary
    target_step: int
    member_sessions: tuple[tuple[str, str], ...]
    contribution_ids: tuple[str, ...]
    input_identity_sha256: str


@dataclass
class _Member:
    session_id: str
    failure_domain: str
    status: str = "loading"
    state: StateSummary | None = None


class ResidentMembership:
    """Single-writer state machine; methods must be serialized by the caller.

    begin_step freezes READY sessions. Loading/replacement sessions cannot join
    that step; its surviving frozen workers can take missing four-sample units.
    During collection, accepted replicas and healthy local contributions survive
    a worker loss. Failure after optimizer application begins is a separate,
    explicit recovery state, since the remaining devices may hold mixed steps.
    """

    def __init__(self, committed_state: StateSummary, *, global_batch=256):
        if not isinstance(committed_state, StateSummary):
            raise MembershipError("A complete committed StateSummary is required")
        if type(global_batch) is not int or global_batch < 4 or global_batch % 4:
            raise MembershipError("global_batch must be a positive multiple of atomic size 4")
        self.committed_state = committed_state
        self.global_batch = global_batch
        self.unit_count = global_batch // 4
        self.phase = "idle"
        self.epoch = 0
        self._members = {}
        self._frozen = ()
        self._assignments = {}
        self._contributions = {}
        self._seen_contribution_ids = set()
        self._update = None
        self._applied = {}
        self._candidate_state = None
        self._reliable_applied = {}
        self.last_committed_contribution_ids = ()

    @property
    def frozen_members(self):
        return self._frozen

    @property
    def assignments(self):
        return tuple(self._assignments[k] for k in sorted(self._assignments))

    @property
    def contributions(self):
        return dict(self._contributions)

    @property
    def missing_units(self):
        return tuple(k for k in range(self.unit_count) if k not in self._contributions)

    def _member(self, worker_id, session_id, *, allow_lost=False):
        member = self._members.get(worker_id)
        if (member is None or member.session_id != session_id
                or (member.status == "lost" and not allow_lost)):
            raise StaleMessage("Worker session is no longer current")
        return member

    def member_status(self, worker_id, session_id):
        return self._member(worker_id, session_id, allow_lost=True).status

    def register(self, worker_id, session_id, *, failure_domain=None):
        _text(worker_id, "worker_id")
        _text(session_id, "session_id")
        domain = worker_id if failure_domain is None else failure_domain
        _text(domain, "failure_domain")
        previous = self._members.get(worker_id)
        if previous and previous.session_id == session_id:
            if previous.failure_domain != domain or previous.status == "lost":
                raise StaleMessage("A lost session cannot be revived or change failure domain")
            return
        if previous:
            self.lose(worker_id, previous.session_id)
        self._members[worker_id] = _Member(session_id, domain)

    def mark_ready(self, worker_id, session_id, state: StateSummary):
        if self.phase in ("recovery_required", "reconciling"):
            raise RecoveryRequired("Restore the committed full state before admitting workers")
        member = self._member(worker_id, session_id)
        if state != self.committed_state:
            raise MembershipError("READY requires matching model, optimizer and committed step")
        if member.status == "applied":
            raise MembershipError("An applied optimizer cannot be reset through mark_ready")
        member.state, member.status = state, "ready"

    def begin_step(self):
        if self.phase != "idle":
            raise MembershipError("A step is already active")
        frozen = tuple(sorted((wid, m.session_id) for wid, m in self._members.items()
                              if m.status == "ready" and m.state == self.committed_state))
        if not frozen:
            raise MembershipError("No fully synchronized READY worker is available")
        self.epoch += 1
        self._frozen = frozen
        self._assignments, self._contributions = {}, {}
        self.phase = "collecting"
        for unit in range(self.unit_count):
            self._assign(unit, frozen[unit % len(frozen)])
        return self.assignments

    def _assign(self, unit, member_session):
        wid, sid = member_session
        start = self.committed_state.committed_step * self.global_batch + unit * 4
        assignment = Assignment(self.epoch, self.committed_state.committed_step, unit,
                                tuple(range(start, start + 4)), wid, sid, uuid.uuid4().hex)
        self._assignments[unit] = assignment
        return assignment

    def _assignment(self, assignment):
        if (not isinstance(assignment, Assignment) or assignment.epoch != self.epoch
                or assignment.base_step != self.committed_state.committed_step
                or self._assignments.get(assignment.unit_id) != assignment):
            raise StaleMessage("Epoch or unit attempt is no longer current")
        member = self._member(assignment.worker_id, assignment.session_id)
        if member.status != "ready" or member.state != self.committed_state:
            raise StaleMessage("Assignment owner is not synchronized and READY")
        return member

    def submit_ready(self, assignment, contribution_id, gradient_sha256):
        if self.phase != "collecting":
            raise MembershipError("Local gradient readiness is only valid during collection")
        self._assignment(assignment)
        _text(contribution_id, "contribution_id")
        _digest(gradient_sha256, "gradient_sha256")
        old = self._contributions.get(assignment.unit_id)
        if old:
            if (old.assignment, old.contribution_id, old.gradient_sha256) == (
                    assignment, contribution_id, gradient_sha256):
                return old
            raise MembershipError("A unit already has a different contribution")
        if contribution_id in self._seen_contribution_ids:
            raise MembershipError("A contribution ID cannot represent different work")
        contribution = Contribution(assignment, contribution_id, gradient_sha256)
        self._contributions[assignment.unit_id] = contribution
        self._seen_contribution_ids.add(contribution_id)
        return contribution

    def accept_replicated(self, assignment, evidence: ReplicaEvidence):
        if self.phase != "collecting":
            raise MembershipError("Replica acceptance is only valid during collection")
        member = self._assignment(assignment)
        contribution = self._contributions.get(assignment.unit_id)
        if not isinstance(evidence, ReplicaEvidence) or contribution is None:
            raise MembershipError("A READY contribution and replica evidence are required")
        if (evidence.verified is not True or evidence.durable is not True
                or evidence.contribution_id != contribution.contribution_id
                or evidence.gradient_sha256 != contribution.gradient_sha256
                or evidence.failure_domain == member.failure_domain):
            raise MembershipError("Acceptance requires an exact verified durable independent replica")
        _text(evidence.location, "replica location")
        _text(evidence.failure_domain, "replica failure_domain")
        if contribution.replica is not None and contribution.replica != evidence:
            raise MembershipError("An accepted contribution cannot silently change replica evidence")
        accepted = replace(contribution, replica=evidence)
        self._contributions[assignment.unit_id] = accepted
        return accepted

    def _survivors(self):
        return tuple((wid, sid) for wid, sid in self._frozen
                     if (wid in self._members and self._members[wid].session_id == sid
                         and self._members[wid].status == "ready"
                         and self._members[wid].state == self.committed_state))

    def lose(self, worker_id, session_id):
        member = self._member(worker_id, session_id, allow_lost=True)
        if member.status == "lost":
            return ()
        member.status = "lost"
        update_member = (self._update is not None
                         and (worker_id, session_id) in self._update.member_sessions)
        if self.phase in ("applying", "reconciling") and update_member:
            self.phase = "recovery_required"
        elif self.phase == "prepared" and update_member:
            # No optimizer has been authorized yet. Re-prepare with survivors.
            self.phase, self._update = "collecting", None
        for unit, assignment in list(self._assignments.items()):
            if (assignment.worker_id, assignment.session_id) != (worker_id, session_id):
                continue
            contribution = self._contributions.get(unit)
            if contribution is not None and contribution.accepted:
                continue
            self._contributions.pop(unit, None)
            self._assignments.pop(unit)
        return self.reassign_missing() if self.phase == "collecting" else ()

    def reassign_missing(self):
        if self.phase != "collecting":
            raise MembershipError("Missing units can only be assigned during collection")
        survivors = self._survivors()
        if not survivors:
            return ()
        created = []
        load = {pair: sum((a.worker_id, a.session_id) == pair for a in self.assignments
                          if a.unit_id not in self._contributions) for pair in survivors}
        for unit in self.missing_units:
            if unit in self._assignments:
                continue
            pair = min(survivors, key=lambda p: (load[p], p))
            created.append(self._assign(unit, pair))
            load[pair] += 1
        return tuple(created)

    def refreeze_after_total_loss(self):
        """Admit restored replacements when no frozen READY session survives.

        The logical batch/base state stays unchanged. A new epoch fences every
        old local attempt, while already accepted independent replicas retain
        their original identities and need no recomputation.
        """
        if self.phase != "collecting" or self._survivors():
            raise MembershipError("Refreezing requires collection with no frozen READY survivor")
        frozen = tuple(sorted((wid, m.session_id) for wid, m in self._members.items()
                              if m.status == "ready" and m.state == self.committed_state))
        if not frozen:
            raise MembershipError("No restored READY replacement is available")
        self.epoch += 1
        self._frozen = frozen
        self._contributions = {u: c for u, c in self._contributions.items() if c.accepted}
        self._assignments = {u: c.assignment for u, c in self._contributions.items()}
        created = []
        for index, unit in enumerate(self.missing_units):
            created.append(self._assign(unit, frozen[index % len(frozen)]))
        return tuple(created)

    def prepare_update(self):
        if self.phase != "collecting":
            raise MembershipError("Update preparation requires collection state")
        if (len(self._contributions) != self.unit_count
                or not all(c.accepted for c in self._contributions.values())):
            raise MembershipError("Every atomic unit needs an accepted independent replica")
        survivors = self._survivors()
        if not survivors:
            raise MembershipError("No synchronized frozen worker can apply the update")
        identity = {"epoch": self.epoch, "base_step": self.committed_state.committed_step,
                    "model_sha256": self.committed_state.model_sha256,
                    "optimizer_sha256": self.committed_state.optimizer_sha256,
                    "global_batch": self.global_batch, "atomic_samples": 4,
                    "inputs": [{"unit_id": u, "contribution_id": c.contribution_id,
                                "gradient_sha256": c.gradient_sha256,
                                "assignment": {"epoch": c.assignment.epoch,
                                               "attempt_id": c.assignment.attempt_id,
                                               "sample_positions": c.assignment.sample_positions}}
                               for u, c in sorted(self._contributions.items())]}
        identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest()
        self._update = UpdatePlan(self.epoch, uuid.uuid4().hex, self.committed_state,
                                  self.committed_state.committed_step + 1, survivors,
                                  tuple(self._contributions[u].contribution_id
                                        for u in range(self.unit_count)), identity_sha)
        self._applied, self._candidate_state = {}, None
        self._reliable_applied = {}
        self.phase = "prepared"
        return self._update

    def _update_fence(self, update_id, phase):
        if self._update is None or self._update.update_id != update_id:
            raise StaleMessage("Update ID is no longer current")
        if self.phase == "recovery_required":
            raise RecoveryRequired("Partial optimizer application requires full-state recovery")
        allowed = phase if isinstance(phase, tuple) else (phase,)
        if self.phase not in allowed:
            raise MembershipError(f"Expected {phase} update state")

    def begin_apply(self, update_id):
        self._update_fence(update_id, "prepared")
        for wid, sid in self._update.member_sessions:
            member = self._member(wid, sid)
            if member.status != "ready" or member.state != self.committed_state:
                raise MembershipError("Prepared worker is no longer synchronized")
        self.phase = "applying"

    def acknowledge_applied(self, update_id, worker_id, session_id, state: StateSummary):
        self._update_fence(update_id, "applying")
        member = self._member(worker_id, session_id)
        pair = (worker_id, session_id)
        if pair not in self._update.member_sessions:
            raise StaleMessage("Worker session is outside this frozen update")
        if (not isinstance(state, StateSummary) or state.committed_step != self._update.target_step
                or (self._candidate_state is not None and state != self._candidate_state)):
            self.phase = "recovery_required"
            raise RecoveryRequired("Applied model/optimizer states disagree")
        if pair in self._applied and self._applied[pair] != state:
            self.phase = "recovery_required"
            raise RecoveryRequired("Worker changed its applied full-state acknowledgement")
        self._candidate_state = state
        self._applied[pair] = state
        member.state, member.status = state, "applied"

    def _verified_applied(self, evidence):
        if not isinstance(evidence, AppliedStateEvidence):
            raise MembershipError("A full-state byte-verification receipt is required, not a summary")
        if (self._update is None or evidence.update_id != self._update.update_id
                or evidence.epoch != self.epoch
                or evidence.input_identity_sha256 != self._update.input_identity_sha256):
            raise StaleMessage("Applied state belongs to a different prepared input or epoch")
        pair = (evidence.worker_id, evidence.session_id)
        member = self._member(*pair)
        if pair not in self._update.member_sessions:
            raise StaleMessage("Applied-state source is outside this frozen update")
        if (evidence.verified_full_state is not True or not isinstance(evidence.state, StateSummary)
                or evidence.state.committed_step != self._update.target_step):
            raise MembershipError("Actual complete target model/optimizer bytes must be verified")
        _text(evidence.verification_receipt, "full-state verification receipt")
        if self._candidate_state is not None and evidence.state != self._candidate_state:
            raise RecoveryRequired("Verified applied state disagrees with the update's full-state summary")
        return pair, member

    def recover_forward(self, evidence: AppliedStateEvidence):
        """Keep healthy applied workers; synchronize laggards to verified step+1.

        Only an alive frozen session with externally verified complete p/m/v
        and matching prepared inputs can become the recovery authority. Healthy
        acknowledged workers at that state keep their optimizer and gradients.
        No tensor copy is performed here: laggards stay LOADING until a separate
        complete-state verification receipt is acknowledged.
        """
        if self.phase != "recovery_required":
            raise MembershipError("Forward recovery requires a partial-update failure")
        pair, source = self._verified_applied(evidence)
        state = evidence.state
        targets = tuple(p for p in self._update.member_sessions
                        if (p[0] in self._members and self._members[p[0]].session_id == p[1]
                            and self._members[p[0]].status != "lost"))
        self._update = replace(self._update, member_sessions=targets)
        self._candidate_state = state
        self._applied = {p: s for p, s in self._applied.items() if p in targets and s == state}
        self._reliable_applied = {p: e for p, e in self._reliable_applied.items() if p in targets}
        self._applied[pair], self._reliable_applied[pair] = state, evidence
        source.state, source.status = state, "applied"
        laggards = []
        for target in targets:
            member = self._member(*target)
            if target in self._applied:
                member.state, member.status = state, "applied"
            else:
                member.state, member.status = None, "loading"
                laggards.append(target)
        self.phase = "reconciling"
        return tuple(laggards)

    def acknowledge_synchronized(self, evidence: AppliedStateEvidence):
        """A laggard loaded and byte-verified complete step+1 model AND optimizer."""
        if self.phase != "reconciling":
            raise MembershipError("Full-state synchronization requires forward recovery")
        pair, member = self._verified_applied(evidence)
        self._applied[pair] = evidence.state
        self._reliable_applied[pair] = evidence
        member.state, member.status = evidence.state, "applied"

    def commit_update(self, update_id):
        """Return commit state after every target acknowledges identical full state.

        This in-memory transition is not a durable checkpoint. The integration
        must journal it atomically before publishing a committed-step response.
        """
        self._update_fence(update_id, ("applying", "reconciling"))
        if set(self._applied) != set(self._update.member_sessions):
            raise MembershipError("Cannot commit a partially acknowledged optimizer update")
        self.committed_state = self._candidate_state
        self.last_committed_contribution_ids = self._update.contribution_ids
        for member in self._members.values():
            if member.status == "applied" and member.state == self.committed_state:
                member.status = "ready"
            elif member.status != "lost":
                member.status, member.state = "loading", None
        self.phase = "idle"
        self._frozen, self._assignments, self._contributions = (), {}, {}
        self._update, self._candidate_state, self._applied = None, None, {}
        self._reliable_applied = {}
        return self.committed_state

    def restore_committed(self, evidence: StateEvidence):
        """Fence partial application; workers must reload BOTH model and optimizer.

        Accepted independent replicas remain available for the same base step.
        This method does not load tensors. Each healthy session remains LOADING
        until mark_ready reports the exact restored full-state summary.
        """
        if self.phase != "recovery_required":
            raise MembershipError("Explicit recovery requires a partial-update failure")
        if any(wid in self._members and self._members[wid].session_id == sid
               and self._members[wid].status != "lost" for wid, sid in self._reliable_applied):
            raise RecoveryRequired("A verified applied full-state replica survives; recover forward")
        if (not isinstance(evidence, StateEvidence) or evidence.verified is not True
                or evidence.durable is not True or evidence.state != self.committed_state):
            raise MembershipError("Recovery requires the verified durable committed full state")
        _text(evidence.location, "checkpoint location")
        for member in self._members.values():
            if member.status != "lost":
                member.status, member.state = "loading", None
        for unit, contribution in list(self._contributions.items()):
            if not contribution.accepted:
                self._contributions.pop(unit)
                self._assignments.pop(unit, None)
        self._update, self._candidate_state, self._applied = None, None, {}
        self._reliable_applied = {}
        self.phase = "collecting"
