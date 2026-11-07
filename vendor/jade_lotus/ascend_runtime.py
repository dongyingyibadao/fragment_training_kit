"""Admission of actual NPU processes before the first global-batch update."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

from jade_lotus.protocol import read_json, verify_sources
from jade_lotus.runtime import atomic_json, emit


def ready_workers(run_dir, config, now=None):
    """Only initialized, live, distinct assigned devices count as admitted."""
    run = Path(run_dir)
    now = time.time() if now is None else now
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    launches = {}
    for path in (run / "launches").glob("*.json"):
        record = read_json(path, {})
        if (record.get("backend") == "torch_npu" and record.get("config_digest") == digest
                and record.get("gradient_checkpointing") is False and record.get("job_id")
                and record.get("assigned_device_id") is not None):
            launches[(record.get("worker_id"), record.get("session_id"))] = record
    workers, devices, initial = [], set(), None
    for path in sorted((run / "workers").glob("*.json")):
        heartbeat = read_json(path, {})
        worker, session = heartbeat.get("worker_id"), heartbeat.get("session_id")
        launch = launches.get((worker, session))
        if (not launch or path.name != str(worker) + ".json" or heartbeat.get("config_digest") != digest
                or heartbeat.get("stage") not in ("initialized", "polling")
                or now - path.stat().st_mtime > float(config.get("lease_seconds", 120))):
            continue
        bootstrap = read_json(run / "bootstrap" / f"{worker}-{session}.json", {})
        identity = (bootstrap.get("arrays_sha256"), bootstrap.get("schema"))
        if not identity[0] or not identity[1]:
            continue
        if initial is None:
            initial = identity
        elif identity != initial:
            raise ValueError("NPU workers initialized different parameter identities before admission")
        physical = (launch["job_id"], launch.get("hostname"), str(launch["assigned_device_id"]))
        if physical in devices:
            continue
        devices.add(physical)
        workers.append({key: launch.get(key) for key in ("worker_id", "session_id", "job_id", "workspace_id",
                                                        "hostname", "assigned_device_id", "initialized_at")})
    return workers


def await_admission(run_dir, config, minimum, stopping, timeout_seconds=14400, poll_seconds=5):
    if type(minimum) is not int or minimum < 1 or minimum > 32:
        raise ValueError("NPU admission requires 1..32 actual initialized devices")
    began, last_count = time.monotonic(), None
    while not stopping():
        if (Path(run_dir) / "STOP").exists() or (Path(run_dir) / "FATAL.json").exists():
            return False
        workers = ready_workers(run_dir, config)
        if len(workers) != last_count:
            emit(run_dir, "npu_admission_wait", required_devices=minimum, ready_devices=len(workers))
            last_count = len(workers)
        if len(workers) >= minimum:
            record = {"backend": "torch_npu", "required_devices": minimum,
                      "admitted_devices": len(workers), "workers": workers, "time": time.time(),
                      "config_digest": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
            atomic_json(Path(run_dir) / "ADMISSION_READY.json", record)
            emit(run_dir, "npu_admission_ready", required_devices=minimum, admitted_devices=len(workers))
            return True
        if time.monotonic() - began >= timeout_seconds:
            raise TimeoutError(f"NPU admission deadline: {len(workers)}/{minimum} actual devices initialized")
        time.sleep(min(poll_seconds, 5))
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--minimum-workers", type=int)
    parser.add_argument("--admission-seconds", type=float, default=14400)
    args = parser.parse_args(argv)
    run = args.run_dir.expanduser().resolve()
    config = read_json(run / "run_config.json", {})
    if config.get("backend") != "torch_npu":
        raise ValueError("Ascend runtime requires a frozen torch_npu run")
    verify_sources(run, config)
    minimum = args.minimum_workers or (1 if config.get("validation_run") else config["acceptance_card_count"])
    if not config.get("validation_run") and minimum != 32:
        raise ValueError("The 500-update acceptance run must admit all 32 actual NPU devices")
    interrupted = False
    def stop(_number, _frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        # A restart with durable progress has already crossed the initial
        # barrier; surviving workers may continue while the fleet replenishes.
        restored = read_json(run / "checkpoint.json", {})
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        restart = (restored.get("config_digest") == digest and restored.get("step", 0) > 0
                   and read_json(run / "ADMISSION_READY.json", {}).get("config_digest") == digest)
        if not restart and not await_admission(run, config, minimum, lambda: interrupted,
                                               timeout_seconds=args.admission_seconds):
            return 0
        os.execv(sys.executable, [sys.executable, "-u", "-m", "jade_lotus.coordinator", "--run-dir", str(run)])
    except Exception as error:
        atomic_json(run / "FATAL.json", {"type": type(error).__name__, "error": str(error), "time": time.time()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
