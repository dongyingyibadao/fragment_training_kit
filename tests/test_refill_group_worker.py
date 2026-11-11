"""Candidate wrappers remain device-free until a complete batch can cold-load."""
import builtins
import copy
import hashlib
import json
import signal

import pytest

from fragment_training_kit import group_worker

protocol = group_worker._refill_protocol()


def prepare(tmp_path, role="candidate", count=2):
    run = tmp_path / "run"
    (run / "worker_logs").mkdir(parents=True)
    (run / "probe").mkdir()
    (run / "training.json").write_text('{"fixture":true}')
    (run / "site.json").write_text(json.dumps({"cards": 64, "active_cards": 64,
        "job_layout": [16] * 4, "training_config": {"resident_pod_refill": True}}))
    mapping = {"schema": protocol.ALLOCATION_SCHEMA, "run_dir": str(run.resolve()),
        "config_sha256": hashlib.sha256((run / "training.json").read_bytes()).hexdigest(),
        "worker_prefix": "w000080", "intent_id": "intent-80", "role": role,
        "batch_id": "batch-2" if role == "candidate" else None,
        "members": [{"worker_id": f"w{65+i:06d}", "device_index": i} for i in range(count)]}
    protocol.atomic_write(run / "refill/allocations/w000080.json", mapping)
    return run, mapping


def permit_for(mapping, session):
    return {"schema": protocol.PERMIT_SCHEMA, **{key: mapping[key] for key in protocol.IDENTITY_FIELDS},
        "mapping_sha256": protocol.mapping_digest(mapping), "wrapper_session": session,
        "batch_member_ids": [f"w{i:06d}" for i in range(65, 81)]}


def cancel_for(mapping):
    return {"schema": protocol.CANCEL_SCHEMA, **{key: mapping[key] for key in protocol.IDENTITY_FIELDS}}


def arguments(run, count=2):
    return ["--run-dir", str(run), "--worker-prefix", "w000080", "--allocated-devices", str(count)]


class FakeClock:
    def __init__(self, idle):
        self.now, self.idle = 0.0, idle

    def monotonic(self):
        return self.now

    def time(self):
        return 1000 + self.now

    def sleep(self, seconds):
        self.now = round(self.now + seconds, 8)
        assert self.now <= 5, "fake wrapper must terminate without blocking the test runner"
        self.idle(self.now)


class FakeChild:
    def __init__(self, pid, returncode=None):
        self.pid, self.returncode, self.waits = pid, returncode, []

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.returncode is None:
            self.returncode = -signal.SIGTERM
        return self.returncode


def test_candidate_ignores_stale_and_incomplete_permits_then_launches_once(tmp_path, monkeypatch):
    run, mapping = prepare(tmp_path)
    children, commands, records, launches = [], [], [], []
    reservation_path = run / "refill/reservations/w000080.json"
    permit_path = run / "refill/permits/w000080.json"
    original_write = protocol.atomic_write

    def write(path, value):
        if path == reservation_path:
            records.append(copy.deepcopy(value))
        original_write(path, value)

    monkeypatch.setattr(protocol, "atomic_write", write)
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", "4,7")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,7")
    original_import = builtins.__import__

    def import_without_device_runtime(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "torch_npu"}, "wrapper must never initialize devices"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_device_runtime)

    def idle(now):
        reservation = json.loads(reservation_path.read_text())
        permit = permit_for(mapping, reservation["wrapper_session"])
        if now < .5:
            permit["wrapper_session"] = "earlier-wrapper-incarnation"
        elif now < 1:
            permit["batch_member_ids"] = permit["batch_member_ids"][:15]
        original_write(permit_path, permit)
        if now < 1:
            assert not children
        if now >= 2.1:
            for child in children:
                child.returncode = 0

    clock = FakeClock(idle)
    monkeypatch.setattr(group_worker, "time", clock)

    def spawn(command, **kwargs):
        assert clock.now >= 1
        reservation = json.loads(reservation_path.read_text())
        assert reservation["state"] == "loading"
        assert protocol.validate_permit(mapping, reservation, json.loads(permit_path.read_text()))
        environment = kwargs["env"]
        assert environment["JADE_REFILL_WRAPPER_SESSION"] == reservation["wrapper_session"]
        assert environment["JADE_REFILL_BATCH_ID"] == "batch-2"
        assert environment["JADE_ALLOCATION_GROUP"] == "w000080"
        assert environment["ASCEND_VISIBLE_DEVICES"] == "4,7"
        assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "4,7"
        assert "--initial-standby" in command
        assert kwargs["start_new_session"] is True
        child = FakeChild(200000 + len(children))
        children.append(child)
        commands.append(command)
        launches.append(clock.now)
        return child

    monkeypatch.setattr(group_worker.subprocess, "Popen", spawn)
    monkeypatch.setattr(group_worker.os, "killpg", lambda *_: pytest.fail("completed children need no signal"))
    assert group_worker.main(arguments(run)) == 0
    assert len(children) == 2 and launches == [1, 1]
    assert [c[c.index("--worker-id") + 1] for c in commands] == ["w000065", "w000066"]
    assert [c[c.index("--device-index") + 1] for c in commands] == ["0", "1"]
    assert [r["heartbeat_seq"] for r in records] == list(range(len(records)))
    assert [r["updated_at"] for r in records if r["state"] == "reserved"] == [1000, 1000.5, 1001]
    assert [r["updated_at"] for r in records if r["state"] == "loading"] == [1001, 1001.5, 1002]
    assert all(child.waits == [5] for child in children)


def test_initial_dynamic_members_above_original_pool_start_without_standby(tmp_path, monkeypatch):
    run, _ = prepare(tmp_path, role="initial")
    commands = []
    monkeypatch.setenv("JADE_REFILL_BATCH_ID", "unrelated-parent-batch")

    def spawn(command, **kwargs):
        assert "--initial-standby" not in command
        assert kwargs["env"]["JADE_REFILL_WRAPPER_SESSION"]
        assert "JADE_REFILL_BATCH_ID" not in kwargs["env"]
        commands.append(command)
        return FakeChild(300000 + len(commands), returncode=0)

    monkeypatch.setattr(group_worker.subprocess, "Popen", spawn)
    assert group_worker.main(arguments(run)) == 0
    assert len(commands) == 2
    assert not (run / "refill/reservations").exists()


@pytest.mark.parametrize("after_load", [False, True])
def test_matching_cancel_stops_only_owned_children_and_wrong_batch_is_ignored(tmp_path, monkeypatch, after_load):
    run, mapping = prepare(tmp_path)
    children, signals = [], []
    reservation_path = run / "refill/reservations/w000080.json"
    cancel_path = run / "refill/cancelled/w000080.json"

    def idle(now):
        record = cancel_for(mapping)
        if now < .5:
            record["batch_id"] = "different-batch"
        protocol.atomic_write(cancel_path, record)
        if after_load:
            reservation = json.loads(reservation_path.read_text())
            protocol.atomic_write(run / "refill/permits/w000080.json",
                                  permit_for(mapping, reservation["wrapper_session"]))

    monkeypatch.setattr(group_worker, "time", FakeClock(idle))

    def spawn(*_, **kwargs):
        child = FakeChild(400000 + len(children))
        children.append(child)
        return child

    def send(pid, sig):
        assert pid in {child.pid for child in children}
        signals.append((pid, sig))

    monkeypatch.setattr(group_worker.subprocess, "Popen", spawn)
    monkeypatch.setattr(group_worker.os, "killpg", send)
    assert group_worker.main(arguments(run)) == 1
    assert group_worker.time.now == .5
    assert len(children) == (2 if after_load else 0)
    assert {pid for pid, _ in signals} == {child.pid for child in children}
    assert all(sig == signal.SIGTERM for _, sig in signals)
    assert all(child.waits == [5] for child in children)


@pytest.mark.parametrize("marker", ["STOP", "STOP.json", "COMPLETE", "COMPLETE.json", "FATAL", "FATAL.json"])
@pytest.mark.parametrize("location", ["", "probe"])
def test_markers_preempt_candidate_loading(tmp_path, monkeypatch, marker, location):
    run, _ = prepare(tmp_path)
    (run / location / marker).write_text("{}")
    monkeypatch.setattr(group_worker.subprocess, "Popen", lambda *_a, **_k: pytest.fail("stop marker must block loading"))
    assert group_worker.main(arguments(run)) == (0 if marker.startswith("COMPLETE") else 1)
    assert not (run / "refill/reservations").exists()


def test_cancel_has_priority_over_valid_permit(tmp_path, monkeypatch):
    run, mapping = prepare(tmp_path)

    def idle(_now):
        reservation = json.loads((run / "refill/reservations/w000080.json").read_text())
        protocol.atomic_write(run / "refill/permits/w000080.json", permit_for(mapping, reservation["wrapper_session"]))
        protocol.atomic_write(run / "refill/cancelled/w000080.json", cancel_for(mapping))

    monkeypatch.setattr(group_worker, "time", FakeClock(idle))
    monkeypatch.setattr(group_worker.subprocess, "Popen", lambda *_a, **_k: pytest.fail("cancelled job cannot load"))
    assert group_worker.main(arguments(run)) == 1


def test_changed_allocation_is_rejected_at_permit_consumption(tmp_path, monkeypatch):
    run, mapping = prepare(tmp_path)

    def idle(_now):
        reservation = json.loads((run / "refill/reservations/w000080.json").read_text())
        protocol.atomic_write(run / "refill/permits/w000080.json", permit_for(mapping, reservation["wrapper_session"]))
        protocol.atomic_write(run / "refill/allocations/w000080.json", {**mapping, "placement_reason": "changed"})

    monkeypatch.setattr(group_worker, "time", FakeClock(idle))
    monkeypatch.setattr(group_worker.subprocess, "Popen", lambda *_a, **_k: pytest.fail("changed allocation cannot load"))
    with pytest.raises(ValueError, match="Immutable allocation changed"):
        group_worker.main(arguments(run))
