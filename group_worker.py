"""Own several resident supervisors inside one allocated multi-NPU container.

The allocated visibility mask stays intact. Each child selects its own logical
device; this preserves the shared IPC domain needed by native local HCCL.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid


def _refill_protocol():
    # A source checkout contains vendor; a frozen run has jade_lotus beside
    # fragment_training_kit on PYTHONPATH. Neither path imports a device runtime.
    vendor = Path(__file__).resolve().parent / "vendor"
    if (vendor / "jade_lotus").is_dir() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    return importlib.import_module("jade_lotus.refill_protocol")


def _read_record(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError):
        return None  # A partial, absent or stale control record grants nothing.


def termination_reason(run, mapping=None):
    """Check both launcher and controller stop markers before any load/work."""
    for name in ("FATAL", "STOP", "COMPLETE"):
        if any((base / (name + suffix)).exists()
               for base in (run, run / "probe") for suffix in ("", ".json")):
            return name
    if mapping is not None:
        record = _read_record(run / "refill/cancelled" / f"{mapping['worker_prefix']}.json")
        if _refill_protocol().validate_cancel(mapping, record):
            return "CANCELLED"
    return None


def reservation_devices(allocated):
    """Read assigned device visibility without importing/initializing a model.

    Some QZ instance APIs omit device IDs. Retain that limitation in official
    evidence and separately record the container's assigned mask/device nodes.
    No masks are modified and no NPU memory is allocated by this probe.
    """
    for key in ('ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES', 'NPU_VISIBLE_DEVICES'):
        raw = os.environ.get(key, '')
        values = raw.split(',')
        if values and all(re.fullmatch(r'[0-9]+', v.strip()) for v in values):
            ids = [int(v.strip()) for v in values]
            if len(ids) == len(set(ids)) == allocated:
                return {'verified': True, 'kind': 'assigned_environment_mask', 'source': key,
                        'count': len(ids), 'devices': ids}
    devices = []
    for path in Path('/dev').glob('davinci[0-9]*'):
        if not re.fullmatch(r'davinci[0-9]+', path.name):
            continue
        try:
            if stat.S_ISCHR(path.stat().st_mode) and os.access(path, os.R_OK | os.W_OK):
                devices.append(path.name)
        except OSError:
            continue
    return {'verified': len(devices) == allocated, 'kind': 'accessible_character_devices',
            'count': len(devices), 'devices': sorted(devices)}


class RefillLoadGate:
    """Advertise an empty candidate allocation until its whole batch may load.

    Only the controller can grant a load permit; a permit is not admission to
    training. Supervisors still start in standby and await state alignment.
    """
    heartbeat_interval = .5

    def __init__(self, run, mapping, wrapper_session):
        self.run, self.mapping = run, mapping
        self.protocol = _refill_protocol()
        self.record = {**mapping, "schema": self.protocol.RESERVATION_SCHEMA,
                       "mapping_sha256": self.protocol.mapping_digest(mapping),
                       "wrapper_session": wrapper_session, "pid": os.getpid(),
                       "hostname": socket.gethostname(), "heartbeat_seq": -1,
                       "device_visibility": reservation_devices(len(mapping['members'])),
                       "state": "reserved"}
        try:
            self.record["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            pass
        self.last_heartbeat = None

    def heartbeat(self, *, state=None):
        now = time.monotonic()
        if (state is None and self.last_heartbeat is not None
                and now - self.last_heartbeat < self.heartbeat_interval - 1e-9):
            return
        if state is not None:
            self.record["state"] = state
        self.record["heartbeat_seq"] += 1
        self.record["updated_at"] = time.time()
        self.protocol.atomic_write(self.run / "refill/reservations" / f"{self.mapping['worker_prefix']}.json",
                                   self.record)
        self.last_heartbeat = now

    def wait(self, halted):
        permit_path = self.run / "refill/permits" / f"{self.mapping['worker_prefix']}.json"
        while not halted():
            self.heartbeat()
            permit = _read_record(permit_path)
            if self.protocol.validate_permit(self.mapping, self.record, permit):
                current = self.protocol.read_allocation(self.run, self.mapping["worker_prefix"],
                                                        len(self.mapping["members"]))
                if self.protocol.mapping_digest(current) != self.record["mapping_sha256"]:
                    raise ValueError("Immutable allocation changed while awaiting its load permit")
                if halted():
                    return None
                return permit
            time.sleep(.1)
        return None


def worker_slots(layout, prefix, allocated):
    if (not isinstance(layout, list) or not layout
            or any(type(n) is not int or n not in (1, 2, 4, 8, 16) for n in layout) or sum(layout) > 64):
        raise ValueError("Expected a bounded 1/2/4/8/16-card job layout")
    if not re.fullmatch(r"w\d{6}", prefix):
        raise ValueError("Unexpected fleet worker prefix")
    index = int(prefix[1:]) - 1
    if not 0 <= index < len(layout) or layout[index] != allocated:
        raise ValueError("Allocated job size differs from the frozen layout")
    start = sum(layout[:index])
    return [(f"w{start + local + 1:06d}", local) for local in range(allocated)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Prepared launcher run, not probe subdirectory")
    parser.add_argument("--worker-prefix", required=True)
    parser.add_argument("--allocated-devices", type=int, required=True)
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    site = json.loads((run / "site.json").read_text())
    mapping, gate, wrapper_session = None, None, None
    if site.get("training_config", {}).get("resident_pod_refill") is True:
        protocol = _refill_protocol()
        mapping = protocol.read_allocation(run, args.worker_prefix, args.allocated_devices)
        slots = protocol.allocation_slots(mapping)
        wrapper_session = uuid.uuid4().hex
        if mapping["role"] == "candidate":
            gate = RefillLoadGate(run, mapping, wrapper_session)
    else:
        layout = site.get("job_layout", [1] * site["cards"])
        slots = worker_slots(layout, args.worker_prefix, args.allocated_devices)
    children, logs = [], []
    stopped = False
    reason = None

    def stop(_signum=None, _frame=None):
        nonlocal stopped
        stopped = True
        for process in children:
            if process.poll() is None:
                # Each process group was created and is owned by this wrapper.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def halted():
        nonlocal reason
        if stopped:
            return True
        reason = termination_reason(run, mapping)
        if reason is not None:
            stop()
        return stopped

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        if gate is not None:
            if gate.wait(halted) is None:
                return 0 if reason == "COMPLETE" else 1
            if halted():
                return 0 if reason == "COMPLETE" else 1
            gate.heartbeat(state="loading")
        for worker, index in slots:
            if halted():
                break
            if gate is not None:
                gate.heartbeat()
            command = [sys.executable, "-u", str(run / "runtime/scripts/probe_resident_openpi_fault.py"),
                       "--worker", "--run-dir", str(run / "probe"), "--worker-id", worker,
                       "--device-index", str(index), "--allocated-devices", str(args.allocated_devices)]
            standby = (mapping["role"] == "candidate" if mapping is not None
                       else int(worker[1:]) > site.get("active_cards", site["cards"]))
            if standby:
                command.append("--initial-standby")
            environment = os.environ.copy()
            environment["JADE_ALLOCATION_GROUP"] = args.worker_prefix
            if wrapper_session is not None:
                environment["JADE_REFILL_WRAPPER_SESSION"] = wrapper_session
                if mapping["role"] == "candidate":
                    environment["JADE_REFILL_BATCH_ID"] = mapping["batch_id"]
                else:
                    environment.pop("JADE_REFILL_BATCH_ID", None)
            for key in ("ASCEND_WORK_PATH", "ASCEND_PROCESS_LOG_PATH", "ACL_OP_COMPILER_CACHE_DIR"):
                if environment.get(key):
                    environment[key] = str(Path(environment[key]) / worker)
                    Path(environment[key]).mkdir(parents=True, exist_ok=True)
            output = (run / "worker_logs" / f"{worker}-supervisor.log").open("x")
            logs.append(output)
            children.append(subprocess.Popen(command, env=environment, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True))
        # A stubborn child must not keep us out of the bounded shutdown below.
        while not halted() and any(p.poll() is None for p in children):
            if gate is not None:
                gate.heartbeat()
            time.sleep(.1)
        if reason == "COMPLETE":
            return 0
        return 1 if stopped or any(p.returncode != 0 for p in children) else 0
    finally:
        stop()
        for p in children:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # It exited between the timeout and the signal.
                p.wait(timeout=5)
        for output in logs:
            output.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
