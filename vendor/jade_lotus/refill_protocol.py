"""Standard-library contracts for immutable refill allocation and load gates.

These helpers neither import a device framework nor grant training admission.
A load permit authorizes cold loading for one exact wrapper incarnation only.
The controller separately aligns full state and admits members at COMMIT.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid


ALLOCATION_SCHEMA = "fragment-refill-allocation-v1"
RESERVATION_SCHEMA = "fragment-refill-reservation-v1"
PERMIT_SCHEMA = "fragment-refill-load-v1"
CANCEL_SCHEMA = "fragment-refill-cancel-v1"
SIZES = (1, 2, 4, 8, 16)
IDENTITY_FIELDS = ("run_dir", "config_sha256", "worker_prefix", "intent_id", "batch_id")


def _text(value):
    return isinstance(value, str) and bool(value) and value == value.strip() and "\0" not in value


def _worker(value):
    return isinstance(value, str) and re.fullmatch(r"w[0-9]{6}", value) is not None and value != "w000000"


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _mapping(mapping):
    if not isinstance(mapping, dict) or mapping.get("schema") != ALLOCATION_SCHEMA:
        raise ValueError("Invalid refill allocation schema")
    run = mapping.get("run_dir")
    if (not _text(run) or not Path(run).is_absolute() or str(Path(run).resolve()) != run
            or not _digest(mapping.get("config_sha256")) or not _worker(mapping.get("worker_prefix"))
            or not _text(mapping.get("intent_id"))):
        raise ValueError("Allocation needs an absolute canonical run, config digest and exact intent identity")
    role, batch = mapping.get("role"), mapping.get("batch_id")
    if (role not in ("initial", "candidate") or "batch_id" not in mapping
            or (role == "candidate" and not _text(batch))
            or (role == "initial" and batch is not None and not _text(batch))):
        raise ValueError("Allocation role or batch identity is invalid")
    members = mapping.get("members")
    if not isinstance(members, list) or len(members) not in SIZES:
        raise ValueError("Allocation must contain 1/2/4/8/16 device members")
    workers, indexes = set(), set()
    for member in members:
        if (not isinstance(member, dict) or not _worker(member.get("worker_id"))
                or type(member.get("device_index")) is not int
                or member["worker_id"] in workers or member["device_index"] in indexes):
            raise ValueError("Allocation requires unique worker IDs and integer device indexes")
        workers.add(member["worker_id"])
        indexes.add(member["device_index"])
    if indexes != set(range(len(members))):
        raise ValueError("Allocation logical device indexes must be contiguous from zero")
    return mapping


def mapping_digest(mapping):
    """Bind all allocation fields, including any additional immutable metadata."""
    _mapping(mapping)
    raw = json.dumps(mapping, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def allocation_slots(mapping):
    _mapping(mapping)
    return [(member["worker_id"], member["device_index"])
            for member in sorted(mapping["members"], key=lambda row: row["device_index"])]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in refill allocation")
        result[key] = value
    return result


def read_allocation(run, prefix, allocated):
    """Read one durable mapping; never reconstruct ownership from job history."""
    run = Path(run).resolve()
    if not _worker(prefix) or type(allocated) is not int or allocated not in SIZES:
        raise ValueError("Invalid job prefix or allocated device count")
    path = run / "refill" / "allocations" / f"{prefix}.json"
    if path.is_symlink() or not path.resolve().is_relative_to(run):
        raise ValueError("Refill allocation must be an ordinary file inside its run")
    mapping = _mapping(json.loads(path.read_text(), object_pairs_hook=_unique_object))
    digest = hashlib.sha256((run / "training.json").read_bytes()).hexdigest()
    if (mapping["run_dir"] != str(run) or mapping["worker_prefix"] != prefix
            or mapping["config_sha256"] != digest or len(mapping["members"]) != allocated):
        raise ValueError("Refill allocation differs from this run/config/job/device count")
    return mapping


def validate_permit(mapping, reservation, permit):
    """Return whether this exact live wrapper may cold-load its whole allocation.

    A stale or malformed permit never starts a child. Freshness and the official
    platform allocation are checked by the issuer; this receiver binds its own
    incarnation, immutable map and the exact 16-member batch.
    """
    try:
        _mapping(mapping)
        if (mapping["role"] != "candidate" or not isinstance(reservation, dict)
                or not isinstance(permit, dict) or reservation.get("schema") != RESERVATION_SCHEMA
                or permit.get("schema") != PERMIT_SCHEMA):
            return False
        digest = mapping_digest(mapping)
        if any(record.get(key) != mapping[key] for record in (reservation, permit) for key in IDENTITY_FIELDS):
            return False
        if (reservation.get("role") != "candidate" or reservation.get("members") != mapping["members"]
                or reservation.get("mapping_sha256") != digest or permit.get("mapping_sha256") != digest
                or not _text(reservation.get("wrapper_session"))
                or permit.get("wrapper_session") != reservation["wrapper_session"]
                or type(reservation.get("pid")) is not int or reservation["pid"] < 1
                or not _text(reservation.get("hostname"))
                or type(reservation.get("heartbeat_seq")) is not int or reservation["heartbeat_seq"] < 0
                or type(reservation.get("updated_at")) not in (int, float)
                or not math.isfinite(reservation["updated_at"])
                or reservation.get("state") not in ("reserved", "loading")):
            return False
        members = permit.get("batch_member_ids")
        if not isinstance(members, list) or len(members) != 16 or any(not _worker(worker) for worker in members):
            return False
        return len(set(members)) == 16 and {m["worker_id"] for m in mapping["members"]}.issubset(members)
    except (ValueError, TypeError, KeyError, OverflowError):
        return False


def validate_cancel(mapping, record):
    try:
        _mapping(mapping)
        return (isinstance(record, dict) and record.get("schema") == CANCEL_SCHEMA
                and all(record.get(key) == mapping[key] for key in IDENTITY_FIELDS))
    except (ValueError, TypeError, KeyError):
        return False


def atomic_write(path, value):
    """Atomically publish a complete private JSON record, retaining literal data."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
