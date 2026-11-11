"""Refill load authority binds immutable allocation, batch and live incarnation."""
import copy
import hashlib
import json
import os

import pytest

from fragment_training_kit import group_worker

protocol = group_worker._refill_protocol()


def allocation(tmp_path, *, role="candidate", count=2):
    run = tmp_path / "run"
    run.mkdir()
    (run / "training.json").write_text('{"training":"fixture"}')
    mapping = {"schema": protocol.ALLOCATION_SCHEMA, "run_dir": str(run.resolve()),
               "config_sha256": hashlib.sha256((run / "training.json").read_bytes()).hexdigest(),
               "worker_prefix": "w000071", "intent_id": "immutable-intent",
               "role": role, "batch_id": "batch-2" if role == "candidate" else None,
               "members": [{"worker_id": f"w{65+i:06d}", "device_index": i} for i in range(count)]}
    path = run / "refill/allocations/w000071.json"
    protocol.atomic_write(path, mapping)
    return run, mapping, path


def authority(mapping):
    reservation = {**copy.deepcopy(mapping), "schema": protocol.RESERVATION_SCHEMA,
                   "mapping_sha256": protocol.mapping_digest(mapping), "wrapper_session": "live-session",
                   "pid": 123, "hostname": "fixture-node", "heartbeat_seq": 7,
                   "updated_at": 1000.5, "state": "reserved"}
    permit = {"schema": protocol.PERMIT_SCHEMA,
              **{key: mapping[key] for key in protocol.IDENTITY_FIELDS},
              "mapping_sha256": protocol.mapping_digest(mapping), "wrapper_session": "live-session",
              "batch_member_ids": [f"w{i:06d}" for i in range(65, 81)]}
    return reservation, permit


def test_historical_member_ids_are_valid_and_devices_determine_slot_order(tmp_path):
    run, mapping, path = allocation(tmp_path)
    mapping["members"].reverse()
    protocol.atomic_write(path, mapping)
    restored = protocol.read_allocation(run, "w000071", 2)
    assert protocol.allocation_slots(restored) == [("w000065", 0), ("w000066", 1)]
    assert protocol.mapping_digest(dict(reversed(list(mapping.items())))) == protocol.mapping_digest(mapping)
    extra = {**mapping, "placement_hint": "node-a"}
    assert protocol.mapping_digest(extra) != protocol.mapping_digest(mapping)


@pytest.mark.parametrize("role,batch", [("initial", None), ("initial", "first"), ("candidate", "next")])
def test_supported_roles_bind_their_batch_value(tmp_path, role, batch):
    run, mapping, path = allocation(tmp_path, role=role)
    mapping["batch_id"] = batch
    protocol.atomic_write(path, mapping)
    assert protocol.read_allocation(run, "w000071", 2) == mapping


@pytest.mark.parametrize("field,value", [
    ("schema", "another-schema"), ("worker_prefix", "../w000071"), ("worker_prefix", "w000000"),
    ("worker_prefix", "w００００７１"), ("intent_id", ""), ("intent_id", "\0"),
    ("role", "standby"), ("batch_id", None), ("batch_id", ""), ("batch_id", True),
    ("config_sha256", "A" * 64), ("config_sha256", "f" * 63),
    ("run_dir", "relative/run"), ("members", []), ("members", None),
])
def test_malformed_mapping_fields_fail_closed(tmp_path, field, value):
    run, mapping, path = allocation(tmp_path)
    mapping[field] = value
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError):
        protocol.read_allocation(run, "w000071", 2)


@pytest.mark.parametrize("members", [
    [{"worker_id": "w000065", "device_index": False}],
    [{"worker_id": "w000065", "device_index": 0.0}],
    [{"worker_id": "w000065", "device_index": -1}],
    [{"worker_id": "w000065", "device_index": 1}],
    [{"worker_id": "w000065"}],
    [{"worker_id": "w000000", "device_index": 0}],
    [{"worker_id": "w000065", "device_index": 0}, {"worker_id": "w000065", "device_index": 1}],
    [{"worker_id": "w000065", "device_index": 0}, {"worker_id": "w000066", "device_index": 0}],
    [{"worker_id": "w000065", "device_index": 0}, {"worker_id": "w000066", "device_index": 2}],
    [{"worker_id": f"w{i:06d}", "device_index": i - 65} for i in range(65, 68)],
])
def test_invalid_members_never_resolve_to_a_device(tmp_path, members):
    _, mapping, _ = allocation(tmp_path)
    mapping["members"] = members
    with pytest.raises(ValueError):
        protocol.allocation_slots(mapping)


@pytest.mark.parametrize("mismatch", ["run", "config", "prefix", "size", "missing-batch"])
def test_mapping_cannot_cross_run_config_or_allocation(tmp_path, mismatch):
    run, mapping, path = allocation(tmp_path)
    count = 2
    if mismatch == "run":
        mapping["run_dir"] = str(tmp_path.resolve())
    elif mismatch == "config":
        (run / "training.json").write_text('{"training":"changed"}')
    elif mismatch == "prefix":
        mapping["worker_prefix"] = "w000072"
    elif mismatch == "size":
        count = 1
    else:
        mapping.pop("batch_id")
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError):
        protocol.read_allocation(run, "w000071", count)


def test_duplicate_json_keys_and_symlink_allocations_are_rejected(tmp_path):
    run, mapping, path = allocation(tmp_path)
    raw = json.dumps(mapping)
    path.write_text(raw[:-1] + ',"role":"candidate"}')
    with pytest.raises(ValueError, match="Duplicate"):
        protocol.read_allocation(run, "w000071", 2)
    outside = tmp_path / "outside.json"
    outside.write_text(raw)
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="ordinary file"):
        protocol.read_allocation(run, "w000071", 2)
    path.unlink()
    path.parent.rmdir()
    path.parent.symlink_to(tmp_path)
    (tmp_path / path.name).write_text(raw)
    with pytest.raises(ValueError, match="ordinary file"):
        protocol.read_allocation(run, "w000071", 2)


def test_only_exact_current_sixteen_member_authority_is_accepted(tmp_path):
    _, mapping, _ = allocation(tmp_path)
    reservation, permit = authority(mapping)
    assert protocol.validate_permit(mapping, reservation, permit)
    reservation["state"] = "loading"
    assert protocol.validate_permit(mapping, reservation, permit)
    reservation["wrapper_session"] = "next-incarnation"
    assert not protocol.validate_permit(mapping, reservation, permit)


@pytest.mark.parametrize("field,value", [
    ("schema", "wrong"), ("run_dir", "/somewhere-else"), ("config_sha256", "0" * 64),
    ("worker_prefix", "w000072"), ("intent_id", "another-intent"), ("batch_id", "batch-1"),
    ("mapping_sha256", "0" * 64), ("wrapper_session", "old-session"),
    ("batch_member_ids", None), ("batch_member_ids", [f"w{i:06d}" for i in range(65, 80)]),
    ("batch_member_ids", [f"w{i:06d}" for i in range(65, 82)]),
    ("batch_member_ids", ["w000065"] * 16),
    ("batch_member_ids", [f"w{i:06d}" for i in range(81, 97)]),
    ("batch_member_ids", ["w000000"] + [f"w{i:06d}" for i in range(65, 80)]),
])
def test_bad_permit_cannot_start_a_model(tmp_path, field, value):
    _, mapping, _ = allocation(tmp_path)
    reservation, permit = authority(mapping)
    permit[field] = value
    assert not protocol.validate_permit(mapping, reservation, permit)


@pytest.mark.parametrize("field,value", [
    ("schema", "wrong"), ("role", "initial"), ("members", []), ("mapping_sha256", "0" * 64),
    ("wrapper_session", ""), ("pid", True), ("pid", 0), ("hostname", ""),
    ("heartbeat_seq", False), ("heartbeat_seq", -1), ("updated_at", float("nan")),
    ("updated_at", float("inf")), ("updated_at", True), ("state", "cancelled"),
])
def test_bad_reservation_cannot_receive_load_authority(tmp_path, field, value):
    _, mapping, _ = allocation(tmp_path)
    reservation, permit = authority(mapping)
    reservation[field] = value
    assert not protocol.validate_permit(mapping, reservation, permit)


def test_initial_allocation_and_malformed_records_cannot_consume_candidate_permit(tmp_path):
    _, mapping, _ = allocation(tmp_path, role="initial")
    reservation, permit = authority(mapping)
    assert not protocol.validate_permit(mapping, reservation, permit)
    assert not protocol.validate_permit(None, reservation, permit)
    assert not protocol.validate_permit(mapping, None, permit)
    assert not protocol.validate_permit(mapping, reservation, [])


def test_cancellation_binds_exact_immutable_job_identity(tmp_path):
    _, mapping, _ = allocation(tmp_path)
    record = {"schema": protocol.CANCEL_SCHEMA,
              **{key: mapping[key] for key in protocol.IDENTITY_FIELDS}}
    assert protocol.validate_cancel(mapping, record)
    for field in ("schema", *protocol.IDENTITY_FIELDS):
        assert not protocol.validate_cancel(mapping, {**record, field: "stale"})
    assert not protocol.validate_cancel(mapping, None)


def test_atomic_record_serialization_failure_preserves_previous_record(tmp_path):
    path = tmp_path / "nested/record.json"
    protocol.atomic_write(path, {"literal": "$(echo data)\nsecond line"})
    original = path.read_bytes()
    assert os.stat(path).st_mode & 0o777 == 0o600
    with pytest.raises(ValueError):
        protocol.atomic_write(path, {"value": float("nan")})
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]
