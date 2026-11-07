"""Explicit COMMIT-boundary cohort replacement for the resident controller.

The caller supplies the existing file transport and device commands. This
adapter never allocates resources: all allowed slots belong to the launcher's
already authorized physical pool. It implements export/load/park/join, with
candidate rejection before parking and explicit recovery-required afterwards.
The more general topology_switch protocol remains a separate planning model;
this bounded adapter does not implement its automatic placement or failover.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time

from .resident_membership import StateSummary


class CandidateUnavailable(ValueError):
    """A nonactive candidate is unavailable; retain the current active group."""


class CohortRecoveryRequired(RuntimeError):
    """A live group was parked/reconfigured or its complete state disagrees."""


class CommittedBoundaryLease:
    """One bounded idle wait for an exact committed group, never a keepalive.

    The worker's supervisor still has a fixed deadline if its controller dies.
    A second hold cannot renew an existing lease; the next non-hold command
    ends it. Three ordinary phase budgets cover hold acknowledgements, one
    shared candidate-load deadline, and dispatch of the next step/park command.
    """

    def __init__(self, phase_timeout, *, clock=time.monotonic):
        if (isinstance(phase_timeout, bool) or not isinstance(phase_timeout, (int, float))
                or not math.isfinite(phase_timeout) or phase_timeout <= 0):
            raise ValueError("Boundary lease requires a finite positive phase timeout")
        self.phase_timeout, self.clock, self.deadline = phase_timeout, clock, None

    def arm(self, command, *, completed_step, generation):
        duration = command.get("hold_seconds")
        if (self.deadline is not None or type(command.get("expected_step")) is not int
                or command["expected_step"] != completed_step
                or type(command.get("expected_generation")) is not int
                or command["expected_generation"] != generation
                or isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or not 0 < duration <= 3 * self.phase_timeout):
            raise ValueError("Boundary hold must match exact step/generation and cannot renew or exceed its finite lease")
        self.deadline = self.clock() + duration
        return duration

    def remaining(self):
        remaining = self.phase_timeout if self.deadline is None else self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError("Committed-boundary controller lease expired")
        return remaining

    def release(self):
        self.deadline = None


def identity(member):
    return {key: member[key] for key in ("worker_id", "session_id", "pid")}


def same_identity(record, member):
    return isinstance(record, dict) and all(record.get(k) == v for k, v in identity(member).items())


def _parked(record):
    return bool(record and (record.get("parked") or record["status"] in {"parked_stale", "rollback_hold"}))


def state_summary(receipt, *, boundary):
    if not isinstance(receipt, dict) or type(receipt.get("step")) is not int or receipt["step"] != boundary:
        raise ValueError("Complete state receipt is outside the exact committed boundary")
    return StateSummary(boundary, receipt.get("model_sha256"), receipt.get("optimizer_sha256"))


def validate_topology_config(config, active_cards, *, start_step, steps):
    slots = config.get("resident_topology_slots", active_cards)
    if type(slots) is not int or not active_cards <= slots <= 64:
        raise ValueError("resident_topology_slots must be within active cards..64 authorized physical slots")
    plans = config.get("resident_topology_switches", [])
    if not isinstance(plans, list):
        raise ValueError("resident_topology_switches must be a list")
    ids, boundaries = set(), set()
    for index, plan in enumerate(plans):
        if not isinstance(plan, dict):
            raise ValueError("Each topology plan must be an object")
        boundary = plan.get("boundary_step")
        request_id = plan.get("request_id", f"plan-{index}-{boundary}")
        if (type(boundary) is not int or not start_step < boundary < steps
                or boundary in boundaries or not isinstance(request_id, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", request_id) is None or request_id in ids):
            raise ValueError("Topology plans need unique safe IDs and distinct internal COMMIT boundaries")
        _worker_lists(plan, {f"w{i + 1:06d}" for i in range(slots)})
        boundaries.add(boundary)
        ids.add(request_id)
    return slots


def _worker_lists(request, allowed):
    outgoing, incoming = request.get("retire_workers"), request.get("candidate_workers")
    if (not isinstance(outgoing, list) or not isinstance(incoming, list) or not outgoing
            or len(outgoing) != len(incoming)
            or any(not isinstance(w, str) for w in outgoing + incoming)
            or len(set(outgoing)) != len(outgoing) or len(set(incoming)) != len(incoming)
            or set(outgoing).intersection(incoming) or not set(outgoing + incoming).issubset(allowed)):
        raise ValueError("Replacement needs distinct, disjoint, equal-sized outgoing/incoming allowed worker lists")
    return outgoing, incoming


class CohortBoundaryRuntime:
    """A small synchronous adapter, called only after the durable COMMIT record.

    send(members, action, **fields), wait_active(members, token, phase),
    wait_candidate(members, token, phase, timeout_seconds=...), ready(worker), make_group(members,
    generation, port=None), and persist(snapshot) are supplied by the controller.
    Candidate errors before park never affect the original process group.
    """

    def __init__(self, spec, plans, *, send, wait_active, wait_candidate, ready,
                 make_group, persist, metadata, clock=time.monotonic):
        self.spec, self.plans = spec, plans
        self.send, self.wait_active, self.wait_candidate = send, wait_active, wait_candidate
        self.ready, self.make_group, self.persist, self.metadata, self.clock = ready, make_group, persist, metadata, clock
        self.allowed = set(spec["workers"])
        self.records, self.events, self.requests, self.pending_trial = {}, [], {}, None
        self.recovery_required = False

    def snapshot(self):
        return {"schema": "jade-explicit-cohort-switch-v1", "run_id": self.spec["run_id"],
                "physical_pool_slots": self.spec.get("topology_slots", self.spec["cards"]),
                "authorized_pool_is_preallocated": True, "automatic_placement": False,
                "automatic_failover": False, "resident_records": self.records,
                "requests": self.requests, "pending_trial": self.pending_trial,
                "events": self.events, "recovery_required": self.recovery_required}

    def _record(self, event, **fields):
        self.events.append({"sequence": len(self.events), "event": event, **fields})
        self.persist(self.snapshot())

    def start(self, members, group):
        for member in members:
            self.records[member["worker_id"]] = {**identity(member), "status": "active"}
        self._record("initial_group", generation=group["generation"], members=[identity(m) for m in members])

    def on_commit(self, step, members):
        if self.pending_trial is not None and step > self.pending_trial["boundary_step"]:
            trial = self.pending_trial
            for worker in trial["retire_workers"]:
                self.records[worker]["status"] = "parked_stale"
                self.records[worker]["standby_matches_current_commit"] = False
            for member in members:
                self.records[member["worker_id"]] = {**identity(member), "status": "active"}
            request = self.requests[trial["request_id"]]
            request["status"] = "committed"
            request["first_new_commit"] = step
            request["old_resources_retained"] = True
            request["old_cohort_is_hot_standby"] = False
            self.pending_trial = None
            self._record("first_trial_commit", request_id=trial["request_id"], step=step,
                         old_cohort_status="parked_stale", automatic_failover=False)

    def boundary(self, boundary, members, group, external_request=None):
        requests = []
        for index, plan in enumerate(self.plans):
            if plan["boundary_step"] == boundary:
                requests.append({**plan, "request_id": plan.get("request_id", f"plan-{index}-{boundary}"),
                    "run_id": self.spec["run_id"], "expected_generation": group["generation"],
                    "expected_step": boundary, "config_sha256": self.spec["config_sha256"], "source": "frozen_config"})
        if external_request is not None:
            if (not isinstance(external_request, dict)
                    or type(external_request.get("expected_step")) is not int
                    or external_request["expected_step"] <= boundary):
                requests.append(external_request)
        for request in requests:
            members, group = self.switch(request, boundary, members, group)
        return members, group

    def switch(self, request, boundary, members, group):
        if self.recovery_required:
            raise CohortRecoveryRequired("A previous group/state failure requires explicit recovery before any new switch")
        request_id = request.get("request_id") if isinstance(request, dict) else None
        # An invalid ID never becomes a path component, but remains auditable.
        safe_id = isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", request_id)
        invalid_json = False
        try:
            canonical = json.dumps(request, sort_keys=True, allow_nan=False)
        except (ValueError, TypeError):
            canonical, invalid_json = repr(request), True
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        key = request_id if safe_id else f"invalid-{fingerprint[:16]}"
        if key in self.requests:
            if self.requests[key]["request_sha256"] != fingerprint:
                self._record("request_id_conflict", request_id=key, rejected_request_sha256=fingerprint)
            return members, group
        result = {"request_id": key, "request_sha256": fingerprint, "boundary_step": boundary,
                  "status": "validating", "automatic_failover": False}
        self.requests[key] = result
        try:
            if (invalid_json or not safe_id or request.get("run_id") != self.spec["run_id"]
                    or type(request.get("expected_generation")) is not int
                    or request["expected_generation"] != group["generation"]
                    or type(request.get("expected_step")) is not int or request["expected_step"] != boundary
                    or request.get("config_sha256") != self.spec["config_sha256"]):
                raise CandidateUnavailable("Request run/config/step/group-generation fence does not match this COMMIT")
            outgoing, incoming = _worker_lists(request, self.allowed)
            active_ids = {m["worker_id"] for m in members}
            if not set(outgoing).issubset(active_ids) or set(incoming).intersection(active_ids):
                raise CandidateUnavailable("Only active workers retire and only nonactive workers may be candidates")
            if self.pending_trial is not None:
                raise CandidateUnavailable("The previous trial has not completed its first new COMMIT")
            candidates = []
            for worker in incoming:
                candidate = self.ready(worker)
                if (not isinstance(candidate, dict) or candidate.get("worker_id") != worker
                        or not isinstance(candidate.get("session_id"), str) or not candidate["session_id"]
                        or type(candidate.get("pid")) is not int or candidate["pid"] < 1):
                    raise CandidateUnavailable("Candidate READY must bind its exact allowed worker/session/PID")
                previous = self.records.get(worker)
                parked = _parked(previous)
                if parked and not same_identity(candidate, previous):
                    raise CandidateUnavailable("A parked worker's exact session/PID changed")
                if not parked and candidate.get("replacement") is not True:
                    raise CandidateUnavailable("A fresh candidate must have loaded independently outside every active group")
                candidates.append(candidate)
        except (ValueError, KeyError, TypeError) as error:
            result.update(status="rejected", reason=str(error))
            self._record("request_rejected", **result)
            return members, group
        started = self.clock()
        state_metadata = {**self.metadata(boundary), "topology_request_id": key}
        state_path = str(Path(self.spec["run_dir"]) / "states" / f"topology-{key}-{boundary}")
        result.update(status="aligning_candidate", retire_workers=outgoing, candidate_workers=incoming,
                      old_generation=group["generation"])
        self._record("candidate_alignment_started", **result)
        # Failure of the source/active verifier is an active-state failure.
        try:
            export = self.send([members[0]], "export", path=state_path, expected_step=boundary, state_metadata=state_metadata)
            exported = self.wait_active([members[0]], export, "exported")[members[0]["worker_id"]]
            receipt = exported["receipt"]
            authority = state_summary(receipt, boundary=boundary)
            if (receipt.get("payload_bytes_verified") is not True
                    or not isinstance(receipt.get("manifest_sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", receipt["manifest_sha256"]) is None):
                raise ValueError("Export lacks complete verified payload bytes")
            inspect = self.send(members, "inspect", expected_step=boundary)
            inspected = self.wait_active(members, inspect, "inspected")
            optimizer_configs = {r["optimizer_config_sha256"] for r in inspected.values()}
            if (len(optimizer_configs) != 1 or any(not isinstance(c, str) or re.fullmatch(r"[0-9a-f]{64}", c) is None
                    for c in optimizer_configs) or any(r.get("completed_step") != boundary
                    or r.get("model_sha256") != authority.model_sha256
                    or r.get("optimizer_sha256") != authority.optimizer_sha256 for r in inspected.values())):
                raise ValueError("Original active group model/Adam/step/config states disagree")
            phase_budget = float(self.spec.get("phase_timeout", 600))
            # This does not park/destroy the old HCCL group. It gives active
            # workers one explicit, finite idle allowance while candidates
            # load; ordinary idle timeouts must not kill healthy active ranks.
            hold_seconds = 3 * phase_budget
            hold = self.send(members, "hold_committed_boundary", expected_step=boundary,
                             expected_generation=group["generation"], hold_seconds=hold_seconds)
            held = self.wait_active(members, hold, "boundary_held")
            if any(r.get("completed_step") != boundary or r.get("generation") != group["generation"]
                   or r.get("hold_seconds") != hold_seconds for r in held.values()):
                raise ValueError("Original members did not acknowledge the exact bounded boundary hold")
        except Exception as error:
            self.recovery_required = True
            result.update(status="recovery_required", reason=str(error), stage="active_state_verification")
            self._record("active_state_verification_failed", **result)
            raise CohortRecoveryRequired("Active complete state could not be verified at the COMMIT boundary") from error
        try:
            load_groups = {"load": [], "load_parked": []}
            for candidate in candidates:
                worker = candidate["worker_id"]
                previous = self.records.get(worker)
                was_parked = _parked(previous)
                load_groups["load_parked" if was_parked else "load"].append(candidate)
            # Issue every load before waiting for any candidate. Both kinds
            # share one deadline, so a mixed group cannot silently consume two
            # full phase budgets or multiply its pause by candidate count.
            alignment_deadline = self.clock() + phase_budget
            pending = []
            for action, cohort in load_groups.items():
                if cohort:
                    token = self.send(cohort, action, path=state_path, expected_step=boundary,
                                      state_metadata=state_metadata)
                    pending.append((action, cohort, token))
            for action, cohort, token in pending:
                remaining = alignment_deadline - self.clock()
                if remaining <= 0:
                    raise CandidateUnavailable("Candidate full-state alignment exceeded its shared deadline")
                loaded_rows = self.wait_candidate(cohort, token, "loaded", timeout_seconds=remaining)
                if self.clock() >= alignment_deadline:
                    raise CandidateUnavailable("Candidate full-state alignment exceeded its shared deadline")
                for candidate in cohort:
                    worker = candidate["worker_id"]
                    loaded = loaded_rows[worker]["receipt"]
                    if (state_summary(loaded, boundary=boundary) != authority
                            or any(loaded.get(field) is not True for field in
                                   ("verified_loaded_state", "payload_bytes_verified", "parameter_pointers_preserved"))
                            or loaded.get("manifest_sha256") != receipt.get("manifest_sha256")):
                        raise CandidateUnavailable("Candidate complete model/Adam/step/manifest failed byte verification")
                    self.records[worker] = {**identity(candidate), "status": "candidate_ready",
                        "parked": action == "load_parked",
                        "completed_step": boundary, "model_sha256": authority.model_sha256,
                        "optimizer_sha256": authority.optimizer_sha256}
            for candidate in candidates:
                if not same_identity(self.ready(candidate["worker_id"]), candidate):
                    raise CandidateUnavailable("Candidate session changed after full-state alignment")
        except (ValueError, KeyError, TypeError, AttributeError, TimeoutError) as error:
            # Release the bounded idle allowance explicitly before returning to
            # healthy updates (or handling another request at this boundary).
            try:
                release = self.send(members, "release_committed_boundary", expected_step=boundary,
                                    expected_generation=group["generation"])
                released = self.wait_active(members, release, "boundary_released")
                if any(r.get("completed_step") != boundary or r.get("generation") != group["generation"]
                       for r in released.values()):
                    raise ValueError("Original members did not release their exact boundary hold")
            except Exception as release_error:
                self.recovery_required = True
                result.update(status="recovery_required", reason=str(release_error), stage="active_hold_release",
                              candidate_alignment_error=str(error))
                self._record("active_hold_release_failed", **result)
                raise CohortRecoveryRequired("Active boundary hold could not be released after candidate rejection") from release_error
            result.update(status="rejected", reason=str(error), stage="candidate_alignment",
                          pause_seconds=self.clock() - started, original_group_preserved=True)
            self._record("candidate_alignment_rejected", **result)
            return members, group
        retiring = [m for m in members if m["worker_id"] in outgoing]
        new_members = [m for m in members if m["worker_id"] not in outgoing] + candidates
        result["status"] = "parking_original_group"
        self._record("original_group_park_started", **result)
        try:
            park = self.send(members, "park", expected_step=boundary, expected_generation=group["generation"])
            parked = self.wait_active(members, park, "parked")
            if any(r.get("completed_step") != boundary or r.get("model_sha256") != authority.model_sha256
                   or r.get("optimizer_sha256") != authority.optimizer_sha256 for r in parked.values()):
                raise ValueError("Parking changed the original committed model/Adam state")
            port = exported.get("next_port") if same_identity(new_members[0], members[0]) else None
            new_group = self.make_group(new_members, group["generation"] + 1, port=port)
            join = self.send(new_members, "join", group=new_group, expected_step=boundary)
            joined = self.wait_active(new_members, join, "joined")
            if any(r.get("completed_step") != boundary or r.get("model_sha256") != authority.model_sha256
                   or r.get("optimizer_sha256") != authority.optimizer_sha256
                   or r.get("optimizer_config_sha256") not in optimizer_configs for r in joined.values()):
                raise ValueError("New group did not join with the exact complete committed state")
        except Exception as error:
            self.recovery_required = True
            result.update(status="recovery_required", reason=str(error), stage="park_or_group_rebuild")
            self._record("active_group_rebuild_failed", **result)
            raise CohortRecoveryRequired("Group parking/rebuild failed; explicit recovery is required") from error
        for member in retiring:
            self.records[member["worker_id"]] = {**identity(member), "status": "rollback_hold", "completed_step": boundary}
        for member in new_members:
            self.records[member["worker_id"]] = {**identity(member), "status": "trial" if member in candidates else "active"}
        self.pending_trial = {"request_id": key, "boundary_step": boundary, "retire_workers": outgoing}
        result.update(status="trial", new_generation=new_group["generation"], full_state_alignment_verified=True,
                      model_sha256=authority.model_sha256, optimizer_sha256=authority.optimizer_sha256,
                      old_resources_retained=True, old_cohort_is_hot_standby=False,
                      export_load_park_join_pause_seconds=self.clock() - started)
        self._record("new_group_trial", **result)
        return new_members, new_group
