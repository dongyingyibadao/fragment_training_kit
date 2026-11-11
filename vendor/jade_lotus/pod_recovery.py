"""Pure ownership and survivor-evidence contracts for bounded pod recovery.

No device access, IO, liveness polling, authorization or membership mutation is
performed here. A caller must persist exact loss fences, quiesce the relevant
attempt, authenticate the supplied worker reports, and serialize recovery with
APPLY/COMMIT. Consistent hash announcements are not independent verification of
the actual model/Adam bytes, and this module never authorizes another optimizer
application or a COMMIT.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re


ATOMIC_UNITS = 64
SCHEMA = "jade-pod-forward-evidence-v1"


class PodRecoveryError(ValueError):
    """An ownership partition or recovery evidence violates its exact fence."""


def _text(value, name):
    if not isinstance(value, str) or not value or value != value.strip() or "\0" in value:
        raise PodRecoveryError(f"{name} must be a nonempty, unpadded string")
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise PodRecoveryError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PodRecoveryError(f"{name} must be a lowercase SHA256 digest")
    return value


def _sequence(value, name):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PodRecoveryError(f"{name} must be a sequence")
    return value


def repair_ownership(original, survivor_ids):
    """Return ``(mapping, replay)`` for exactly one global256 atomic-4 batch.

    ``original`` is the *current* complete ownership partition, including any
    units transferred by earlier successful recovery attempts. Every surviving
    owner's existing units retain their order. Lost units, in ascending order,
    are assigned to the currently least-loaded survivor, breaking ties by ID.
    Both returned dictionaries contain every survivor (including empty replay)
    in sorted ID order. Inputs are never mutated.

    A caller may append these replay gradients only after it proves the prior
    attempt completed all of the survivor's current units. Partial replay needs
    explicit unit accounting or whole-step recomputation, not this arithmetic.
    """
    if not isinstance(original, Mapping) or not 1 <= len(original) <= ATOMIC_UNITS:
        raise PodRecoveryError("original must contain 1..64 current worker ownerships")
    copied, all_units = {}, []
    for worker, units in original.items():
        _text(worker, "worker ID")
        units = list(_sequence(units, f"units for {worker}"))
        if any(type(unit) is not int or not 0 <= unit < ATOMIC_UNITS for unit in units):
            raise PodRecoveryError("Ownership units must be integers in 0..63")
        copied[worker] = units
        all_units.extend(units)
    if sorted(all_units) != list(range(ATOMIC_UNITS)):
        raise PodRecoveryError("Ownership must cover every atomic unit 0..63 exactly once")
    survivors = list(_sequence(survivor_ids, "survivor IDs"))
    for worker in survivors:
        _text(worker, "survivor ID")
    if not survivors or len(set(survivors)) != len(survivors):
        raise PodRecoveryError("Need distinct, nonempty survivor IDs")
    if not set(survivors).issubset(copied):
        raise PodRecoveryError("No new worker may enter an in-flight ownership repair")
    ordered = sorted(survivors)
    mapping = {worker: copied[worker][:] for worker in ordered}
    replay = {worker: [] for worker in ordered}
    missing = sorted(unit for worker, units in copied.items() if worker not in mapping for unit in units)
    for unit in missing:
        target = min(ordered, key=lambda worker: (len(mapping[worker]), worker))
        mapping[target].append(unit)
        replay[target].append(unit)
    return mapping, replay


def _member_identity(member):
    if not isinstance(member, Mapping):
        raise PodRecoveryError("A survivor must provide an exact worker/session/PID identity")
    return {"worker_id": _text(member.get("worker_id"), "worker ID"),
            "session_id": _text(member.get("session_id"), "session ID"),
            "pid": _integer(member.get("pid"), "child PID", 1)}


def _report(record, identity, *, run_id, token, phase):
    if not isinstance(record, Mapping):
        raise PodRecoveryError("Every survivor needs a complete report object")
    if (record.get("run_id") != run_id or record.get("token") != token
            or record.get("phase") != phase or _member_identity(record) != identity):
        raise PodRecoveryError("Report run/session/PID/token/phase fence mismatch")
    return record


def validate_forward_update(spec, members, update_token, base_step, generation,
                            updates, inspections, inspection_token):
    """Validate survivor reports for a single already-authorized update k→k+1.

    ``updates`` and ``inspections`` are mappings keyed by worker ID, containing
    exactly ``members`` and no lost/stale extras. Updates use phase ``updated``
    and the original update token/generation. Inspections use phase ``inspected``
    and a distinct inspection token; their generation, if present, must match.
    Both report sets must attest step k+1 and zero activation recomputation.

    Returns a deterministic consistency receipt with the full-state hash triple
    and a digest binding all checked evidence. The caller must independently
    establish current liveness, completed device synchronization, actual complete
    state verification and durable input/update identity. This gate alone does
    not authorize APPLY, state transfer, membership changes, or COMMIT.
    """
    if not isinstance(spec, Mapping):
        raise PodRecoveryError("spec must bind the exact run and training configuration")
    run_id = _text(spec.get("run_id"), "run ID")
    config_sha = _digest(spec.get("config_sha256"), "training configuration digest")
    _text(update_token, "update token")
    _text(inspection_token, "inspection token")
    if update_token == inspection_token:
        raise PodRecoveryError("Inspection requires a distinct token from the optimizer update")
    _integer(base_step, "base step")
    _integer(generation, "communication generation")
    identities = [_member_identity(member) for member in _sequence(members, "survivor members")]
    ids = [identity["worker_id"] for identity in identities]
    if not 1 <= len(ids) <= ATOMIC_UNITS or len(set(ids)) != len(ids):
        raise PodRecoveryError("Need 1..64 distinct exact survivor identities")
    if any(not isinstance(records, Mapping) or set(records) != set(ids) for records in (updates, inspections)):
        raise PodRecoveryError("Report mappings must contain exactly the frozen surviving workers")
    hashes, norm, checked = None, None, []
    for identity in sorted(identities, key=lambda item: item["worker_id"]):
        worker = identity["worker_id"]
        update = _report(updates[worker], identity, run_id=run_id, token=update_token, phase="updated")
        inspected = _report(inspections[worker], identity, run_id=run_id, token=inspection_token, phase="inspected")
        for record in (update, inspected):
            if (_integer(record.get("completed_step"), "reported completed step") != base_step + 1
                    or _integer(record.get("checkpoint_calls"), "activation recomputation count") != 0):
                raise PodRecoveryError("Survivors must report exactly step k+1 with no activation recomputation")
            if "base_step" in record and _integer(record["base_step"], "reported base step") != base_step:
                raise PodRecoveryError("Reported base step differs from this optimizer update")
            if "config_sha256" in record and record["config_sha256"] != config_sha:
                raise PodRecoveryError("Reported training configuration differs from this run")
        if _integer(update.get("generation"), "update generation") != generation:
            raise PodRecoveryError("Update generation does not match this optimizer authorization")
        if "generation" in inspected and _integer(inspected["generation"], "inspection generation") != generation:
            raise PodRecoveryError("Inspection generation differs from this optimizer update")
        value = update.get("grad_norm")
        try:
            finite = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except (OverflowError, ValueError):
            finite = False
        if not finite:
            raise PodRecoveryError("Gradient norms must be finite nonnegative numbers")
        if norm is not None and value != norm:
            raise PodRecoveryError("Surviving workers disagree on the reduced gradient norm")
        norm = value
        state = {key: _digest(inspected.get(key), key) for key in
                 ("model_sha256", "optimizer_sha256", "optimizer_config_sha256")}
        if hashes is not None and state != hashes:
            raise PodRecoveryError("Surviving complete model/Adam/configuration hashes disagree")
        hashes = state
        checked.append({**identity, "completed_step": base_step + 1, "gradient_norm": float(value),
                        "checkpoint_calls": 0, **state})
    receipt = {"schema": SCHEMA, "run_id": run_id, "config_sha256": config_sha,
               "update_token": update_token, "inspection_token": inspection_token,
               "base_step": base_step, "completed_step": base_step + 1, "generation": generation,
               "members": checked, "gradient_norm": float(norm), **hashes,
               "all_survivor_reports_consistent": True, "authorizes_optimizer_or_commit": False,
               "scope": "exact survivor reports and complete-state hash consistency; no independent byte verification"}
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    receipt["evidence_sha256"] = hashlib.sha256(canonical).hexdigest()
    return receipt
