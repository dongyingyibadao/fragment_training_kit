"""Own several resident supervisors inside one allocated multi-NPU container.

The allocated visibility mask stays intact. Each child selects its own logical
device; this preserves the shared IPC domain needed by native local HCCL.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


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
    layout = site.get("job_layout", [1] * site["cards"])
    slots = worker_slots(layout, args.worker_prefix, args.allocated_devices)
    children, logs = [], []
    stopped = False

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

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for worker, index in slots:
            if stopped:
                break
            command = [sys.executable, "-u", str(run / "runtime/scripts/probe_resident_openpi_fault.py"),
                       "--worker", "--run-dir", str(run / "probe"), "--worker-id", worker,
                       "--device-index", str(index), "--allocated-devices", str(args.allocated_devices)]
            if int(worker[1:]) > site.get("active_cards", site["cards"]):
                command.append("--initial-standby")
            environment = os.environ.copy()
            environment["JADE_ALLOCATION_GROUP"] = args.worker_prefix
            for key in ("ASCEND_WORK_PATH", "ASCEND_PROCESS_LOG_PATH", "ACL_OP_COMPILER_CACHE_DIR"):
                if environment.get(key):
                    environment[key] = str(Path(environment[key]) / worker)
                    Path(environment[key]).mkdir(parents=True, exist_ok=True)
            output = (run / "worker_logs" / f"{worker}-supervisor.log").open("x")
            logs.append(output)
            children.append(subprocess.Popen(command, env=environment, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True))
        # A stubborn child must not keep us out of the bounded shutdown below.
        while not stopped and any(p.poll() is None for p in children):
            time.sleep(.1)
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
