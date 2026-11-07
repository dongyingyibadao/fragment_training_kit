"""CPU liveness and irreversible session fences for resident NPU workers.

This single-writer component performs no IO, platform requests, process kills,
collective aborts, optimizer recovery or durable writes. The controller reads
CPU-supervisor counters before calling ``poll``; remote wall clocks and file
mtimes never renew a lease. Persist each loss fence before publishing recovery
commands or accepting another contribution. A timeout is an unconfirmed loss,
not proof that a process died, and that session can never be revived.

The caller must enter APPLYING *before* authorizing any device optimizer write.
A loss after that authorization requires explicit full-state recovery. Merely
testing a predeclared controlled exit does not establish arbitrary-preemption
detection or recovery; hardware integration must exercise this detector with
an unannounced child/container loss while monitors are always armed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
import uuid


SCHEMA = "jade-resident-liveness-v1"
PRE_APPLY_PHASES = frozenset({"collecting", "reducing", "prepared"})
APPLY_PHASES = frozenset({"applying", "committing", "recovery_required"})
OFFICIAL_TERMINAL = frozenset({"job_succeeded", "job_failed", "job_stopped", "job_completed",
                              "job_finished", "job_deleted", "job_cancelled", "job_canceled"})


class LivenessError(ValueError):
    """Invalid identity, evidence or controller phase transition."""


class FencedSession(LivenessError):
    """An old/lost worker session may not send new contributions."""


class RecoveryRequired(LivenessError):
    """Optimizer application was authorized; shrink/reapply is unsafe."""


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise LivenessError(f"{name} must be a nonempty string")


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise LivenessError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class SessionKey:
    run_id: str
    worker_id: str
    session_id: str
    child_pid: int
    supervisor_pid: int
    job_id: str | None = None
    running_round: int | None = None

    def __post_init__(self):
        for name in ("run_id", "worker_id", "session_id"):
            _text(getattr(self, name), name)
        _integer(self.child_pid, "child_pid", minimum=1)
        _integer(self.supervisor_pid, "supervisor_pid", minimum=1)
        if self.job_id is not None:
            _text(self.job_id, "job_id")
        if self.running_round is not None:
            _integer(self.running_round, "running_round")
            if self.job_id is None:
                raise LivenessError("running_round requires an official job_id")

    @classmethod
    def from_supervisor_record(cls, record, *, job_id=None, running_round=None):
        """Explicitly map the existing probe's ``pid`` to child_pid.

        Optional platform metadata must come from the controller's verified
        worker/job binding, never from a PID or pod hostname alone.
        """
        if job_id is not None and record.get("job_id", job_id) != job_id:
            raise LivenessError("Supervisor record conflicts with official job binding")
        if running_round is not None and record.get("running_round", running_round) != running_round:
            raise LivenessError("Supervisor record conflicts with official running round")
        return cls(record["run_id"], record["worker_id"], record["session_id"], record["pid"],
                   record["supervisor_pid"], record.get("job_id") if job_id is None else job_id,
                   record.get("running_round") if running_round is None else running_round)

    def matches_supervisor_record(self, record):
        return (isinstance(record, dict)
                and all(record.get(name) == getattr(self, name) for name in
                        ("run_id", "worker_id", "session_id", "supervisor_pid"))
                and record.get("pid") == self.child_pid
                and ("job_id" not in record or record["job_id"] == self.job_id)
                and ("running_round" not in record or record["running_round"] == self.running_round))


@dataclass(frozen=True)
class Heartbeat:
    session: SessionKey
    counter: int
    evidence_ref: str = "supervisor-heartbeat"

    def __post_init__(self):
        if not isinstance(self.session, SessionKey):
            raise LivenessError("Heartbeat needs an exact SessionKey")
        _integer(self.counter, "heartbeat counter")
        _text(self.evidence_ref, "heartbeat evidence_ref")

    @classmethod
    def from_supervisor_record(cls, session, record, *, evidence_ref="supervisor-heartbeat"):
        if not session.matches_supervisor_record(record):
            raise FencedSession("Heartbeat does not name this exact supervisor/child session")
        return cls(session, record["heartbeat_counter"], evidence_ref)


@dataclass(frozen=True)
class SupervisorExit:
    session: SessionKey
    exit_code: int
    evidence_ref: str

    def __post_init__(self):
        if not isinstance(self.session, SessionKey) or type(self.exit_code) is not int:
            raise LivenessError("Exit needs an exact SessionKey and integer child exit code")
        _text(self.evidence_ref, "exit evidence_ref")

    @classmethod
    def from_supervisor_record(cls, session, record, *, evidence_ref):
        if not session.matches_supervisor_record(record):
            raise FencedSession("Exit does not name this exact supervisor/child session")
        return cls(session, record["exit_code"], evidence_ref)


@dataclass(frozen=True)
class OfficialTerminal:
    """Caller-verified official status receipt for one job incarnation.

    The adapter must verify authenticated fleet/API evidence, its job ID and
    current running_round before constructing this receipt. This class cannot
    authenticate a file or turn a local terminal label into official evidence.
    """

    session: SessionKey
    job_id: str
    running_round: int
    status: str
    evidence_ref: str

    def __post_init__(self):
        if not isinstance(self.session, SessionKey):
            raise LivenessError("Official terminal needs an exact SessionKey")
        _text(self.job_id, "official job_id")
        _integer(self.running_round, "official running_round")
        _text(self.evidence_ref, "official evidence_ref")
        if self.status not in OFFICIAL_TERMINAL:
            raise LivenessError("Only explicit official job terminal states confirm a loss")
        if (self.session.job_id != self.job_id or self.session.running_round is None
                or self.session.running_round != self.running_round):
            raise FencedSession("Official terminal does not match this bound job incarnation")


@dataclass(frozen=True)
class LossNotice:
    event_id: str
    session: SessionKey
    source: str
    confirmed: bool
    evidence_ref: str
    detected_monotonic: float
    counter: int | None
    seconds_without_counter_progress: float
    step_token: str | None
    base_step: int | None
    generation: int | None
    epoch: int
    phase: str
    action: str
    newly_fenced: bool = True

    def as_record(self):
        return {"schema": SCHEMA, **asdict(self),
                "clock_scope": "Only the controller's own monotonic clock; never compare across processes/restarts.",
                "persistence_required_before_recovery_or_contribution_ack": True}


@dataclass
class _Tracked:
    session: SessionKey
    registered_monotonic: float
    last_progress_monotonic: float
    counter: int | None = None
    ready: bool = False
    full_state_receipt: str | None = None


class ResidentLiveness:
    """Single-controller detector; serialize calls with membership transitions.

    Read/observe all available heartbeat counters, then poll every 0.1--0.2s.
    A higher counter renews the lease at observation time. Repeated/backward
    counters do not. A stopped controller itself must not call poll before
    reading fresh counters on resumption. Old session fences are irreversible.

    This component only authorizes a *kind* of recovery action. The integration
    must abort communication, preserve raw healthy gradients, replay only lost
    work, rebuild groups and verify full state itself. It never labels device-
    local gradients accepted or claims checkpoint/durable-commit completion.
    """

    def __init__(self, run_id, *, heartbeat_timeout_seconds=15.0, clock=time.monotonic,
                 persisted_fences=()):
        _text(run_id, "run_id")
        if (isinstance(heartbeat_timeout_seconds, bool) or not isinstance(heartbeat_timeout_seconds, (int, float))
                or not math.isfinite(heartbeat_timeout_seconds) or heartbeat_timeout_seconds <= 0):
            raise LivenessError("heartbeat_timeout_seconds must be finite and positive")
        self.run_id, self.timeout, self.clock = run_id, float(heartbeat_timeout_seconds), clock
        self.phase, self.step_token, self.epoch = "idle", None, 0
        self.base_step, self.generation = None, None
        self._tracked, self._current, self._frozen = {}, {}, ()
        self._fenced, self._losses, self._pending, self._history = set(), {}, [], []
        self._used_tokens = set()
        self._last_clock = None
        for session in persisted_fences:
            self._check_key(session)
            self._fenced.add(session)

    def _now(self):
        value = self.clock()
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or (self._last_clock is not None and value < self._last_clock)):
            raise LivenessError("Controller monotonic clock must be finite and nondecreasing")
        self._last_clock = float(value)
        return self._last_clock

    def _check_key(self, session):
        if not isinstance(session, SessionKey) or session.run_id != self.run_id:
            raise FencedSession("Message belongs to a different run/session")

    def _current_tracker(self, session, *, allow_fenced=False):
        self._check_key(session)
        if self._current.get(session.worker_id) != session or session not in self._tracked:
            raise FencedSession("Worker session is not current")
        if session in self._fenced and not allow_fenced:
            raise FencedSession("A fenced session can never resume or contribute")
        return self._tracked[session]

    @property
    def fenced_sessions(self):
        return frozenset(self._fenced)

    @property
    def frozen_sessions(self):
        return self._frozen

    @property
    def loss_history(self):
        return tuple(self._history)

    def register(self, session):
        self._check_key(session)
        if session in self._fenced:
            raise FencedSession("A persisted/lost session cannot be registered again")
        if any(old.worker_id == session.worker_id and old.session_id == session.session_id and old != session
               for old in self._fenced | self._tracked.keys()):
            raise FencedSession("A replacement must use a fresh session ID, including after PID/job reuse")
        current = self._current.get(session.worker_id)
        if current == session:
            return  # Repeated registration never renews the heartbeat lease.
        if current is not None and current not in self._fenced:
            raise LivenessError("Fence the previous session before admitting its replacement")
        now = self._now()
        self._current[session.worker_id] = session
        self._tracked[session] = _Tracked(session, now, now)

    def observe_heartbeat(self, heartbeat):
        if not isinstance(heartbeat, Heartbeat):
            raise LivenessError("Need a validated Heartbeat record")
        tracked = self._current_tracker(heartbeat.session)
        now = self._now()
        if tracked.counter is not None and heartbeat.counter <= tracked.counter:
            return False
        tracked.counter, tracked.last_progress_monotonic = heartbeat.counter, now
        return True

    def mark_ready(self, session, *, full_state_receipt):
        tracked = self._current_tracker(session)
        _text(full_state_receipt, "external full model/optimizer readiness receipt")
        if self.phase == "recovery_required":
            raise RecoveryRequired("Resolve full optimizer-state recovery before admitting READY members")
        if tracked.counter is None:
            raise LivenessError("READY requires at least one matching CPU-supervisor heartbeat")
        tracked.ready, tracked.full_state_receipt = True, full_state_receipt

    def require_current(self, session):
        """Fence stale contributions; assignment/version checks are separate."""
        self._expire()
        self._current_tracker(session)

    def require_contribution(self, session, *, step_token):
        self._expire()
        self._current_tracker(session)
        if (step_token != self.step_token or session not in self._frozen
                or self.phase not in {"collecting", "reducing"}):
            raise FencedSession("Contribution is not from this step's current frozen session")

    def _fence(self, session, *, source, confirmed, evidence_ref, now):
        tracked = self._current_tracker(session, allow_fenced=True)
        if session in self._fenced:
            old = self._losses.get(session)
            if old is not None and confirmed and not old.confirmed:
                notice = replace(old, source=source, confirmed=True, evidence_ref=evidence_ref,
                                 action="confirmation_only", newly_fenced=False)
                self._losses[session] = notice
                self._pending.append(notice)
                self._history.append(notice)
                return notice
            return None
        phase = self.phase
        if session not in self._frozen:
            action = "remove_unfrozen"
        elif phase in PRE_APPLY_PHASES:
            action = "shrink_pre_apply"
        else:
            action = "recovery_required"
            self.phase = "recovery_required"
        notice = LossNotice(uuid.uuid4().hex, session, source, confirmed, evidence_ref, now, tracked.counter,
                            now - tracked.last_progress_monotonic, self.step_token, self.base_step,
                            self.generation, self.epoch, phase, action)
        self._fenced.add(session)
        tracked.ready = False
        self._losses[session] = notice
        self._pending.append(notice)
        self._history.append(notice)
        return notice

    def observe_exit(self, evidence):
        if not isinstance(evidence, SupervisorExit):
            raise LivenessError("Need an exact CPU-supervisor child-exit receipt")
        return self._fence(evidence.session, source="supervisor_child_exit", confirmed=True,
                           evidence_ref=evidence.evidence_ref, now=self._now())

    def observe_official_terminal(self, evidence):
        if not isinstance(evidence, OfficialTerminal):
            raise LivenessError("Need a caller-verified exact official terminal receipt")
        return self._fence(evidence.session, source="official_job_terminal", confirmed=True,
                           evidence_ref=evidence.evidence_ref, now=self._now())

    def _expire(self):
        now = self._now()
        for session in tuple(self._current.values()):
            tracked = self._tracked[session]
            if session not in self._fenced and now - tracked.last_progress_monotonic >= self.timeout:
                self._fence(session, source="supervisor_counter_timeout", confirmed=False,
                            evidence_ref="controller-monotonic-counter-observation", now=now)

    def poll(self):
        self._expire()
        notices, self._pending = tuple(self._pending), []
        return notices

    def begin_step(self, step_token, *, base_step, generation, sessions=None):
        _text(step_token, "step_token")
        if step_token in self._used_tokens:
            raise FencedSession("A step token cannot be reused")
        _integer(base_step, "base_step")
        _integer(generation, "communication generation")
        self._expire()
        if self.phase != "idle":
            raise LivenessError("Another step is active or full-state recovery is required")
        selected = tuple(self._current.values()) if sessions is None else tuple(sessions)
        if sessions is None:
            selected = tuple(s for s in selected if s not in self._fenced and self._tracked[s].ready)
        if not selected or len(set(selected)) != len(selected):
            raise LivenessError("Need at least one distinct READY session")
        for session in selected:
            if not self._current_tracker(session).ready:
                raise LivenessError("LOADING members cannot join a frozen step")
        self.step_token, self.epoch, self.phase = step_token, self.epoch + 1, "collecting"
        self.base_step, self.generation = base_step, generation
        self._used_tokens.add(step_token)
        self._frozen = tuple(sorted(selected, key=lambda s: s.worker_id))
        return self._frozen

    def advance_phase(self, step_token, phase):
        self._expire()
        if step_token != self.step_token:
            raise FencedSession("Phase authorization belongs to a stale step")
        if self.phase == "recovery_required":
            raise RecoveryRequired("Partial optimizer application requires explicit full-state recovery")
        if any(s in self._fenced for s in self._frozen):
            raise FencedSession("Resolve all lost frozen members before authorizing another phase")
        next_phase = {"collecting": "reducing", "reducing": "prepared", "prepared": "applying", "applying": "committing"}
        if next_phase.get(self.phase) != phase:
            raise LivenessError("Invalid phase order; APPLYING must precede all optimizer authorization")
        self.phase = phase

    def reconfigure_pre_apply(self, step_token, new_step_token, *, generation, survivors):
        """Fence old communication/token; retain every healthy frozen member.

        The caller has to abort/clean old communication and preserve its raw
        local gradients. No loading/replacement session may enter this step.
        This method does not perform or verify those device operations.
        """
        self._expire()
        if step_token != self.step_token:
            raise FencedSession("Reconfiguration belongs to a stale step")
        if self.phase in APPLY_PHASES:
            raise RecoveryRequired("Cannot shrink/reapply after optimizer authorization")
        if self.phase not in PRE_APPLY_PHASES:
            raise LivenessError("No pre-apply step is available to shrink")
        _text(new_step_token, "new_step_token")
        if new_step_token in self._used_tokens:
            raise LivenessError("Reconfigured communication needs a fresh step token")
        _integer(generation, "replacement communication generation")
        if generation <= self.generation:
            raise LivenessError("Replacement communication generation must increase")
        expected = set(self._frozen) - self._fenced
        selected = tuple(survivors)
        if not expected or len(selected) != len(set(selected)) or set(selected) != expected:
            raise LivenessError("Must retain exactly every healthy frozen session, excluding LOADING/replacements")
        if expected == set(self._frozen):
            raise LivenessError("No frozen loss requires reconfiguration")
        self._frozen = tuple(sorted(selected, key=lambda s: s.worker_id))
        self.step_token, self.epoch, self.phase = new_step_token, self.epoch + 1, "collecting"
        self.generation = generation
        self._used_tokens.add(new_step_token)
        return self._frozen

    def finish_step(self, step_token, *, committed_state_receipt):
        """Called after the independent membership/full-state commit succeeds."""
        self._expire()
        if self.phase == "recovery_required":
            raise RecoveryRequired("A loss during application cannot be called committed")
        if (step_token != self.step_token or self.phase != "committing"
                or any(s in self._fenced for s in self._frozen)):
            raise LivenessError("Only a complete, unfenced committing step can finish")
        _text(committed_state_receipt, "external committed model/optimizer receipt")
        self.step_token, self.phase, self._frozen = None, "idle", ()
        self.base_step, self.generation = None, None
