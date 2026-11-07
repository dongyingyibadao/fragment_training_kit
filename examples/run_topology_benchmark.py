"""Bounded QZ fleet for pure communication benchmarks, using frozen source.

prepare and verify do not contact QZ or use devices. run submits only this
run's frozen job_layout through launch.snapshot, with its durable shared
budget (at most 64 cards) and no POST retries. It never starts the resident
training controller. The site's toy training_config is only a prepare input.

    python examples/run_topology_benchmark.py prepare --config SITE.json \
      --benchmark-config BENCHMARK.json --run-dir /shared/new-benchmark-run
    python examples/run_topology_benchmark.py verify --run-dir /shared/new-benchmark-run
    python examples/run_topology_benchmark.py run --run-dir /shared/new-benchmark-run \
      --cookie-file /private/existing.cookies

BENCHMARK.json accepts schema topology-benchmark-fleet-v1, modes (an ordered
list such as ["flat", "hierarchy", "flat"]), payload (an ordered bucket list
with dtype/bytes), rdma_timeouts (such as [16,12,16]), warmup, iterations,
hierarchy_strategy, uneven_fallback, timeout_seconds, hard_timeout_seconds,
rendezvous_timeout_seconds, fleet_timeout_seconds, poll_seconds,
terminal_grace_seconds and cleanup_wait_seconds. Optional environment only
allows HCCL_ALGO, HCCL_ENTRY_LOG_ENABLE and explicit socket-interface names.
All parameters and payload files are resolved before prepare, then frozen in
the run and hashed with the executable snapshot. No external payload is read
by workers. Use a fresh directory for every run; failed evidence is retained.

Each allocated pod is an explicit launch domain. This does not prove that a
pod is a physical host or one HCCS domain. All its visible devices retain the
platform mask, and select separate logical local ranks. The first pod writes
one atomic rendezvous record; every rank uses that base port plus the same
timeout-point index. The full sweep stays on the same allocated fleet.

Inspect current device users/keepalive before run, preserving the monitor and
using the site's task-owned pause files when needed. The launcher itself does
not modify shared environments, keepalive state, or unrelated platform jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import uuid

KIT = Path(__file__).resolve().parents[1]
SCHEMA = "topology-benchmark-fleet-v1"
EXTRA_ENV = {"HCCL_ALGO", "HCCL_ENTRY_LOG_ENABLE", "GLOO_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME"}


def benchmark_module():
    sys.path.insert(0, str(KIT.parent))
    return importlib.import_module("fragment_training_kit.examples.benchmark_topology_collective")


def launch_api():
    # Workers deliberately never import launch: frozen launch.py's source-kit
    # bootstrap is a parent-side API, while workers use runtime/jade_lotus.
    sys.path.insert(0, str(KIT.parent))
    return importlib.import_module("fragment_training_kit.launch")


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict, *, once: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if once:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def first_error(run: Path, error: BaseException | str, *, source: str) -> dict:
    record = {"at": time.time(), "source": source,
              "error": f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else error}
    try:
        atomic_json(run / "first_benchmark_error.json", record, once=True)
    except FileExistsError:
        pass
    return read(run / "first_benchmark_error.json")


def layout_domains(layout: list[int]) -> list[list[int]]:
    if (not isinstance(layout, list) or not layout or any(type(n) is not int or n not in {1, 2, 4, 8, 16} for n in layout)
            or sum(layout) > 64):
        raise ValueError("Expected a bounded 1/2/4/8/16-card layout")
    start, domains = 0, []
    for size in layout:
        domains.append(list(range(start, start + size)))
        start += size
    return domains


def worker_ranks(layout: list[int], prefix: str, allocated: int) -> list[tuple[int, int]]:
    domains = layout_domains(layout)
    if not re.fullmatch(r"w\d{6}", prefix):
        raise ValueError("Unexpected fleet worker prefix")
    index = int(prefix[1:]) - 1
    if not 0 <= index < len(layout) or type(allocated) is not int or layout[index] != allocated:
        raise ValueError("Allocated pod size differs from its frozen job_layout slot")
    return [(rank, local) for local, rank in enumerate(domains[index])]


def normalize_plan(path: Path, site: dict) -> dict:
    raw = read(path)
    if not isinstance(raw, dict) or raw.get("schema", SCHEMA) != SCHEMA:
        raise ValueError(f"Expected {SCHEMA} benchmark configuration")
    allowed = {"schema", "modes", "payload", "payload_json", "dtype", "bytes", "rdma_timeouts",
               "warmup", "iterations", "hierarchy_strategy", "uneven_fallback", "timeout_seconds",
               "hard_timeout_seconds", "rendezvous_timeout_seconds", "fleet_timeout_seconds",
               "poll_seconds", "terminal_grace_seconds", "cleanup_wait_seconds", "environment"}
    if set(raw) - allowed:
        raise ValueError("Unsupported benchmark fields: " + ", ".join(sorted(set(raw) - allowed)))
    module = benchmark_module()
    if "payload" in raw and "payload_json" in raw:
        raise ValueError("Use payload or payload_json")
    if "payload" in raw:
        rows = raw["payload"]
        rows = rows["buckets"] if isinstance(rows, dict) else rows
        if not isinstance(rows, list) or not rows:
            raise ValueError("payload must be an ordered nonempty bucket list")
        payload = []
        for row in rows:
            dtype, nbytes = row["dtype"], module.byte_count(row["bytes"])
            if dtype not in module.WIDTHS or nbytes % module.WIDTHS[dtype]:
                raise ValueError("Payload bytes must be exactly aligned to their supported dtype")
            payload.append({"dtype": dtype, "bytes": nbytes, "numel": nbytes // module.WIDTHS[dtype]})
    else:
        payload_path = raw.get("payload_json")
        if payload_path is not None:
            payload_path = Path(payload_path).expanduser()
            if not payload_path.is_absolute():
                raise ValueError("payload_json must be an explicit absolute file")
        payload = module.read_payload(SimpleNamespace(payload_json=payload_path,
                          dtype=raw.get("dtype", "float32"), bytes=str(raw.get("bytes", "64MiB"))))
    modes = raw.get("modes", ["flat", "hierarchy"])
    modes = modes.split(",") if isinstance(modes, str) else modes
    if not isinstance(modes, list) or not modes or any(m not in {"flat", "hierarchy"} for m in modes):
        raise ValueError("modes must contain flat or hierarchy")
    rdma = raw.get("rdma_timeouts", [int(site["environment"]["HCCL_RDMA_TIMEOUT"])])
    if not isinstance(rdma, list) or not rdma or len(rdma) > 16 or any(type(n) is not int or not 0 <= n <= 31 for n in rdma):
        raise ValueError("rdma_timeouts must contain 1..16 explicit integer exponents in 0..31")
    plan = {"schema": SCHEMA, "purpose": "pure collective benchmark; no forward/backward/optimizer/controller",
            "plan_id": uuid.uuid4().hex, "input_sha256": sha(path), "cards": site["cards"],
            "job_layout": list(site["job_layout"]), "modes": modes, "payload": payload,
            "rdma_timeouts": rdma, "hierarchy_strategy": raw.get("hierarchy_strategy", "auto"),
            "uneven_fallback": raw.get("uneven_fallback", "leader")}
    defaults = {"warmup": 5, "iterations": 10, "timeout_seconds": 120, "hard_timeout_seconds": 600,
                "rendezvous_timeout_seconds": 600, "fleet_timeout_seconds": site["job_runtime_seconds"],
                "poll_seconds": 5, "terminal_grace_seconds": 30, "cleanup_wait_seconds": 120}
    for key, default in defaults.items():
        value = raw.get(key, default)
        if key in {"warmup", "iterations"}:
            if type(value) is not int or value < (0 if key == "warmup" else 1):
                raise ValueError(f"Invalid {key}")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        plan[key] = value
    if plan["fleet_timeout_seconds"] > site["job_runtime_seconds"]:
        raise ValueError("fleet_timeout_seconds must not exceed the frozen platform job runtime")
    if plan["poll_seconds"] > 60 or plan["terminal_grace_seconds"] > 300 or plan["cleanup_wait_seconds"] > 600:
        raise ValueError("Polling/grace/cleanup bounds are too large")
    environment = raw.get("environment", {})
    if (not isinstance(environment, dict) or set(environment) - EXTRA_ENV
            or any(not isinstance(value, str) or not value or "\n" in value for value in environment.values())):
        raise ValueError("Only explicit algorithm, entry-log and socket-interface overrides are allowed")
    plan["environment"] = environment
    topology = module.RankTopology.build(layout_domains(plan["job_layout"]), plan["cards"])
    if "hierarchy" in plan["modes"]:
        topology.select(plan["hierarchy_strategy"], plan["uneven_fallback"])
    return plan


def benchmark_shell(run: Path, site: dict, plan: dict) -> str:
    runtime = run / "runtime"
    paths = [str(runtime), *(str(path) for path in sorted((runtime / "project_sources").glob("source*")))]
    environment = dict(site["environment"], **plan["environment"])
    exports = "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in environment.items())
    return f'''#!/usr/bin/env bash
set -eo pipefail
benchmark_run="${{1:?run directory}}"
benchmark_prefix="${{2:?fleet worker prefix}}"
benchmark_allocated="${{3:?allocated devices}}"
source {shlex.quote(site['cann_env'])} >/dev/null 2>&1
export PATH={shlex.quote(str(Path(site['python']).parent))}:"$PATH"
export PYTHONPATH={shlex.quote(':'.join(paths))}:"${{PYTHONPATH:-}}"
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:"${{LD_LIBRARY_PATH:-}}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export ASCEND_WORK_PATH="$benchmark_run/device-runtime/$benchmark_prefix"
export ASCEND_PROCESS_LOG_PATH="$benchmark_run/device-logs/$benchmark_prefix"
export ACL_OP_COMPILER_CACHE_DIR="$benchmark_run/device-ops/$benchmark_prefix"
{exports}
mkdir -p "$ASCEND_WORK_PATH" "$ASCEND_PROCESS_LOG_PATH" "$ACL_OP_COMPILER_CACHE_DIR" "$benchmark_run/worker_logs"
ulimit -c 0
cd "$benchmark_run/runtime"
exec {shlex.quote(site['python'])} -u "$benchmark_run/runtime/fragment_training_kit/examples/run_topology_benchmark.py" worker --run-dir "$benchmark_run" --worker-prefix "$benchmark_prefix" --allocated-devices "$benchmark_allocated" > "$benchmark_run/worker_logs/$benchmark_prefix.log" 2>&1
'''


def prepare_benchmark(config: Path, benchmark_config: Path, run: Path, *, api=None) -> dict:
    api = api or launch_api()
    # Resolve all inputs before prepare copies any source. Workers only read
    # the canonical run-local files below, never these mutable input files.
    site = api.resolve_site(config)
    plan = normalize_plan(benchmark_config, site)
    result = api.prepare(config, run)
    frozen_site = read(run / "site.json")
    if frozen_site != site:
        raise ValueError("Site changed while preparing the benchmark")
    atomic_json(run / "benchmark_config.json", plan, once=True)
    atomic_json(run / "benchmark_payload.json", {"buckets": plan["payload"]}, once=True)
    atomic_json(run / "benchmark_topology.json", {
        "node_ranks": layout_domains(plan["job_layout"]),
        "evidence": {"scope": "allocated pod launch domains from frozen job_layout",
                     "job_layout": plan["job_layout"],
                     "limit": "Not a physical-host, module, or HCCS-domain discovery"}}, once=True)
    atomic_json(run / "controller_command.json", {"disabled": True, "reason": plan["purpose"]})
    (run / "launch.sh").write_text(benchmark_shell(run, site, plan))
    (run / "fleet_command.txt").write_text("bash {run_dir}/launch.sh {run_dir} {worker_id} {npu_count}\n")
    manifest = read(run / "source_manifest.json")
    for name in ("benchmark_config.json", "benchmark_payload.json", "benchmark_topology.json",
                 "controller_command.json", "launch.sh", "fleet_command.txt"):
        manifest["files"][name] = sha(run / name)
    atomic_json(run / "source_manifest.json", manifest)
    verify_benchmark(run, api=api)
    return dict(result, benchmark=True, training=False, steps=0, mode="pure-collective-benchmark",
                dummy_training_config_only=True, cases_per_rank=len(plan["rdma_timeouts"]) * len(plan["modes"]),
                report_count=plan["cards"] * len(plan["rdma_timeouts"]), results_dir=str(run / "benchmark_results"))


def verify_benchmark(run: Path, *, api=None) -> dict:
    api = api or launch_api()
    api.verify(run)
    plan, site = read(run / "benchmark_config.json"), read(run / "site.json")
    if (plan.get("schema") != SCHEMA or plan["cards"] != site["cards"]
            or plan["job_layout"] != site["job_layout"] or plan["cards"] > site["global_card_limit"]):
        raise ValueError("Frozen benchmark plan differs from its bounded site layout")
    if read(run / "controller_command.json").get("disabled") is not True:
        raise ValueError("Pure benchmark must have its resident controller disabled")
    if read(run / "benchmark_topology.json")["node_ranks"] != layout_domains(plan["job_layout"]):
        raise ValueError("Launch-domain topology differs from the frozen job layout")
    if read(run / "benchmark_payload.json")["buckets"] != plan["payload"]:
        raise ValueError("Payload differs from the frozen plan")
    return plan


def verify_worker_snapshot(run: Path) -> dict:
    expected = run / "runtime/fragment_training_kit/examples/run_topology_benchmark.py"
    if Path(__file__).resolve() != expected.resolve():
        raise ValueError("Worker must execute this run's frozen script, never the source project")
    manifest = read(run / "source_manifest.json")
    for name, digest in manifest["files"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or sha(path) != digest:
            raise ValueError(f"Frozen worker input changed: {name}")
    return read(run / "benchmark_config.json")


def publish_rendezvous(run: Path, plan: dict) -> dict:
    hostname = socket.gethostname()
    addresses = [row[4][0] for row in socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM)]
    addresses = sorted({address for address in addresses if not ipaddress.ip_address(address).is_loopback})
    if not addresses:
        raise ValueError("Rank-zero pod hostname did not resolve to a non-loopback IPv4 address")
    address, points = addresses[0], len(plan["rdma_timeouts"])
    for _attempt in range(32):
        sockets = []
        try:
            first = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(first)
            first.bind((address, 0))
            port = first.getsockname()[1]
            if port + points - 1 > 65535:
                continue
            for offset in range(1, points):
                reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockets.append(reservation)
                reservation.bind((address, port + offset))
            record = {"plan_id": plan["plan_id"], "hostname": hostname, "master_addr": address,
                      "master_port": port, "timeout_points": points, "publisher": "w000001", "at": time.time(),
                      "port_scope": "Consecutive ports probed free; all ranks use base plus timeout-point index"}
            atomic_json(run / "benchmark_rendezvous.json", record, once=True)
            return record
        except OSError:
            if (run / "benchmark_rendezvous.json").exists():
                raise
        finally:
            for reservation in sockets:
                reservation.close()
    raise RuntimeError("Could not find a consecutive free rendezvous port range")


def await_all_pods(run: Path, plan: dict) -> list[dict]:
    """Do not start a finite-timeout PG while the remaining pods are queued."""
    deadline = time.monotonic() + plan["rendezvous_timeout_seconds"]
    while True:
        if any((run / marker).exists() for marker in ("STOP", "FATAL.json")):
            raise InterruptedError("Owned run stopped while waiting for allocated pods")
        records, missing = [], False
        for index, ranks in enumerate(layout_domains(plan["job_layout"])):
            path = run / "benchmark_ready" / f"w{index + 1:06d}.json"
            if not path.exists():
                missing = True
                continue
            record = read(path)
            if (record.get("plan_id") != plan["plan_id"] or record.get("ranks") != ranks
                    or record.get("allocated_devices") != len(ranks)):
                raise ValueError("Pod readiness differs from its frozen launch-domain ownership")
            records.append(record)
        if not missing:
            return records
        if time.monotonic() >= deadline:
            raise TimeoutError("Waiting for all allocated pods exceeded the rendezvous deadline")
        time.sleep(min(.5, max(0, deadline - time.monotonic())))


def await_rendezvous(run: Path, plan: dict) -> dict:
    deadline = time.monotonic() + plan["rendezvous_timeout_seconds"]
    while True:
        if any((run / marker).exists() for marker in ("STOP", "FATAL.json")):
            raise InterruptedError("Owned run stopped before rendezvous")
        path = run / "benchmark_rendezvous.json"
        if path.exists():
            record = read(path)
            if (record["plan_id"] != plan["plan_id"] or record["timeout_points"] != len(plan["rdma_timeouts"])
                    or not 1 <= record["master_port"] <= 65535 - len(plan["rdma_timeouts"]) + 1):
                raise ValueError("Rendezvous record differs from this frozen plan")
            return record
        if time.monotonic() >= deadline:
            raise TimeoutError("Waiting for rank-zero pod rendezvous exceeded the deadline")
        time.sleep(min(.5, max(0, deadline - time.monotonic())))


def benchmark_command(run: Path, plan: dict, rendezvous: dict, rank: int, local_rank: int, allocated: int) -> list[str]:
    return [sys.executable, "-u", str(run / "runtime/fragment_training_kit/examples/benchmark_topology_collective.py"),
            "--device", "npu", "--backend", "hccl", "--rank", str(rank), "--world-size", str(plan["cards"]),
            "--local-rank", str(local_rank), "--device-index", str(local_rank),
            "--expected-visible-devices", str(allocated), "--master-addr", rendezvous["master_addr"],
            "--master-port", str(rendezvous["master_port"]), "--topology", str(run / "benchmark_topology.json"),
            "--payload-json", str(run / "benchmark_payload.json"), "--modes", ",".join(plan["modes"]),
            "--hierarchy-strategy", plan["hierarchy_strategy"], "--uneven-fallback", plan["uneven_fallback"],
            "--warmup", str(plan["warmup"]), "--iterations", str(plan["iterations"]),
            "--timeout-seconds", str(plan["timeout_seconds"]), "--hard-timeout-seconds", str(plan["hard_timeout_seconds"]),
            "--rdma-timeouts", ",".join(map(str, plan["rdma_timeouts"])),
            "--output-dir", str(run / "benchmark_results")]


def worker(run: Path, prefix: str, allocated: int) -> int:
    plan = verify_worker_snapshot(run)
    ranks = worker_ranks(plan["job_layout"], prefix, allocated)
    children, logs, records = [], [], []
    stopped = False
    def stop(_signum=None, _frame=None):
        nonlocal stopped
        stopped = True
        for process in children:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    report = {"schema": SCHEMA, "plan_id": plan["plan_id"], "worker_prefix": prefix,
              "ranks": [rank for rank, _ in ranks], "allocated_devices": allocated, "passed": False,
              "hostname_scope": "allocated pod launch domain; not verified physical host or HCCS domain",
              "hostname": socket.gethostname(), "pid": os.getpid(), "children": records}
    try:
        atomic_json(run / "benchmark_ready" / f"{prefix}.json", {
            "plan_id": plan["plan_id"], "worker_prefix": prefix, "ranks": report["ranks"],
            "allocated_devices": allocated, "hostname": report["hostname"], "at": time.time(),
            "scope": report["hostname_scope"], "visibility_masks": {key: os.environ.get(key) for key in
                ("ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")}}, once=True)
        if prefix == "w000001":
            report["all_pod_readiness"] = await_all_pods(run, plan)
            publish_rendezvous(run, plan)
        rendezvous = await_rendezvous(run, plan)
        report["rendezvous"] = rendezvous
        deadline = time.monotonic() + plan["fleet_timeout_seconds"]
        for rank, local in ranks:
            if stopped:
                raise InterruptedError("Benchmark pod interrupted during launch")
            command = benchmark_command(run, plan, rendezvous, rank, local, allocated)
            environment = dict(os.environ, RANK=str(rank), WORLD_SIZE=str(plan["cards"]), LOCAL_RANK=str(local),
                               LOCAL_WORLD_SIZE=str(allocated), GROUP_RANK=str(int(prefix[1:]) - 1),
                               MASTER_ADDR=rendezvous["master_addr"], MASTER_PORT=str(rendezvous["master_port"]),
                               JADE_ALLOCATION_GROUP=prefix)
            for key in ("ASCEND_WORK_PATH", "ASCEND_PROCESS_LOG_PATH", "ACL_OP_COMPILER_CACHE_DIR"):
                if environment.get(key):
                    environment[key] = str(Path(environment[key]) / f"rank-{rank:06d}")
                    Path(environment[key]).mkdir(parents=True, exist_ok=True)
            output = run / "worker_logs" / f"{prefix}-rank-{rank:06d}.log"
            output.parent.mkdir(parents=True, exist_ok=True)
            stream = output.open("x")
            logs.append(stream)
            process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            children.append(process)
            records.append({"rank": rank, "local_rank": local, "pid": process.pid, "command": command,
                            "visibility_masks": {key: environment.get(key) for key in
                                                 ("ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")}})
        while any(process.poll() is None for process in children):
            if stopped or any((run / marker).exists() for marker in ("STOP", "FATAL.json")):
                raise InterruptedError("Owned benchmark run stopped")
            failures = [process.returncode for process in children if process.poll() not in (None, 0)]
            if failures:
                raise RuntimeError(f"A benchmark rank supervisor failed: exit {failures[0]}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Allocated pod exceeded its hard fleet deadline")
            time.sleep(.2)
        if any(process.returncode for process in children):
            raise RuntimeError("One or more rank supervisors exited unsuccessfully")
        report["passed"] = True
        return 0
    except BaseException as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        first_error(run, error, source=prefix)
        return 1
    finally:
        stop()
        for process in children:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        for record, process in zip(records, children):
            record["exit_code"] = process.returncode
        report["finished_at"] = time.time()
        atomic_json(run / "benchmark_workers" / f"{prefix}.json", report, once=True)
        for stream in logs:
            stream.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def evidence_status(run: Path, plan: dict) -> dict:
    """Require every rank/timeout point and every owning pod to succeed."""
    module = benchmark_module()
    topology = module.RankTopology.build(layout_domains(plan["job_layout"]), plan["cards"])
    expected_sources = {name: sha(path) for name, path in {
        "benchmark_topology_collective.py": run / "runtime/fragment_training_kit/examples/benchmark_topology_collective.py",
        "topology_collective.py": run / "runtime/jade_lotus/topology_collective.py"}.items()}
    reports, missing, maxima = [], [], {}
    for point, exponent in enumerate(plan["rdma_timeouts"]):
        directory = run / "benchmark_results" / f"{point:03d}-rdma-{exponent}"
        watchdogs = sorted(directory.glob("watchdog-rank-*.json"))
        if watchdogs:
            raise RuntimeError(f"Benchmark hard deadline: {watchdogs[0]}")
        for rank in range(plan["cards"]):
            path = directory / f"rank-{rank:06d}.json"
            if not path.exists():
                missing.append(str(path.relative_to(run)))
                continue
            row = read(path)
            domain, local_rank = topology.locate(rank)
            if (row.get("passed") is not True or row.get("rank") != rank or row.get("world_size") != plan["cards"]
                    or row.get("payload") != plan["payload"] or row.get("source_sha256") != expected_sources
                    or row.get("environment", {}).get("HCCL_RDMA_TIMEOUT") != str(exponent)
                    or row.get("backend") != "hccl" or row.get("device") != f"npu:{local_rank}"
                    or row.get("visible_device_count") != plan["job_layout"][domain]
                    or row.get("node_ranks") != layout_domains(plan["job_layout"])
                    or len(row.get("cases", [])) != len(plan["modes"])):
                raise ValueError(f"Failed or inconsistent benchmark evidence: {path}; {row.get('error', '')}")
            for index, (case, mode) in enumerate(zip(row["cases"], plan["modes"], strict=True)):
                strategy = "flat" if mode == "flat" else topology.select(plan["hierarchy_strategy"], plan["uneven_fallback"])
                values = case.get("rank_max_seconds", [])
                if (case.get("mode") != mode or case.get("resolved_strategy") != strategy
                        or case.get("full_buffer_SUM_verified") is not True
                        or case.get("original_local_grad_preserved") is not True
                        or case.get("warmup_iterations") != plan["warmup"]
                        or case.get("measurement_iterations") != plan["iterations"]
                        or len(values) != plan["iterations"]
                        or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values)):
                    raise ValueError(f"Incomplete case validation: {path} case {index}")
                key = (point, index)
                if key in maxima and maxima[key] != values:
                    raise ValueError("Ranks disagree on complete-iteration maximum elapsed times")
                maxima[key] = values
            reports.append(str(path.relative_to(run)))
    for index, ranks in enumerate(layout_domains(plan["job_layout"])):
        path = run / "benchmark_workers" / f"w{index + 1:06d}.json"
        if not path.exists():
            missing.append(str(path.relative_to(run)))
            continue
        row = read(path)
        if (row.get("passed") is not True or row.get("plan_id") != plan["plan_id"]
                or row.get("ranks") != ranks or row.get("allocated_devices") != len(ranks)):
            raise ValueError(f"Allocated-pod supervisor failed: {path}; {row.get('error', '')}")
    return {"complete": not missing, "passed_rank_reports": len(reports), "missing": missing,
            "expected_rank_reports": plan["cards"] * len(plan["rdma_timeouts"]), "reports": reports}


def early_terminal_failure(run: Path, api) -> str | None:
    path = run / "ascend_fleet/state.json"
    if path.exists():
        for job in read(path)["jobs"]:
            if not api.active(job):
                worker_report = run / "benchmark_workers" / f"{job['worker_id']}.json"
                if not worker_report.exists() or read(worker_report).get("passed") is not True:
                    return f"Owned job {job['worker_id']} reached {job.get('status')} before successful benchmark handoff"
    return None


def run_benchmark(client, run: Path, *, api=None) -> int:
    api = api or launch_api()
    plan = verify_benchmark(run, api=api)
    with api.writer_lock(run / "benchmark-driver-lock"):
        if any((run / marker).exists() for marker in ("STOP", "FATAL.json", "COMPLETE.json")):
            raise ValueError("A terminal benchmark cannot be rerun; preserve it and prepare a fresh run")
        complete, receipt, original = False, None, None
        def interrupted(signum, _frame):
            raise InterruptedError(f"Benchmark fleet interrupted by signal {signum}")
        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            deadline, platform_failures = time.monotonic() + plan["fleet_timeout_seconds"], 0
            while True:
                if any((run / marker).exists() for marker in ("STOP", "FATAL.json")):
                    raise InterruptedError("Owned benchmark fleet was stopped")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Benchmark fleet reached its frozen hard deadline")
                proof = evidence_status(run, plan)
                if proof["complete"]:
                    atomic_json(run / "COMPLETE.json", {"at": time.time(), "benchmark": True, "training": False,
                        "plan_id": plan["plan_id"], "status": "passed", **proof}, once=True)
                    complete = True
                    break
                try:
                    status = api.snapshot(client, run, fill=True)
                    if status.get("errors"):
                        raise api.PlatformError(str(status["errors"]))
                    platform_failures = 0
                except api.WriterLockBusy as error:
                    atomic_json(run / "lock_retry.json", {"at": time.time(), "lock_path": str(error.path),
                        "error": str(error), "training_stop_requested": False})
                    # A registered fleet is doing a bounded budget operation;
                    # normal contention is not a failed platform observation.
                    time.sleep(min(plan["poll_seconds"], max(0, deadline - time.monotonic())))
                    continue
                except (api.PlatformError, OSError) as error:
                    platform_failures += 1
                    try:
                        atomic_json(run / "first_platform_error.json", {"at": time.time(), "error": str(error)}, once=True)
                    except FileExistsError:
                        pass
                    if platform_failures >= 3 or time.monotonic() >= deadline:
                        raise
                    time.sleep(min(plan["poll_seconds"], max(0, deadline - time.monotonic())))
                    continue
                if status.get("blocked_reason"):
                    raise RuntimeError(status["blocked_reason"])
                terminal = early_terminal_failure(run, api)
                if terminal:
                    raise RuntimeError(terminal)
                atomic_json(run / "benchmark_status.json", {"at": time.time(), "benchmark": True, "training": False,
                            "plan_id": plan["plan_id"], **status, **proof})
                print(json.dumps({"active_jobs": status.get("active_jobs"), "running_jobs": status.get("running_jobs"),
                                  "passed_rank_reports": proof["passed_rank_reports"],
                                  "expected_rank_reports": proof["expected_rank_reports"]}), flush=True)
                time.sleep(min(plan["poll_seconds"], max(0, deadline - time.monotonic())))
        except BaseException as error:
            original = first_error(run, error, source="fleet-parent")
            try:
                atomic_json(run / "FATAL.json", {"at": time.time(), "benchmark": True, "training": False,
                            "status": "failed", "first_error": original}, once=True)
            except FileExistsError:
                pass
        finally:
            if complete:
                grace = time.monotonic() + plan["terminal_grace_seconds"]
                while time.monotonic() < grace:
                    try:
                        status = api.snapshot(client, run, fill=False)
                        if status["active_jobs"] == 0 and status["local_requested_reserved_cards"] == 0:
                            break
                    except api.WriterLockBusy:
                        pass  # Retry within the same finite terminal grace.
                    except BaseException as error:
                        first_error(run, error, source="terminal-observation")
                        break
                    time.sleep(min(plan["poll_seconds"], max(0, grace - time.monotonic())))
            try:
                receipt = api.cleanup(client, run, wait_seconds=plan["cleanup_wait_seconds"], stop_controller=False)
            except BaseException as cleanup_error:
                if original is None:
                    original = first_error(run, cleanup_error, source="cleanup")
                atomic_json(run / "cleanup_failed.json", {"at": time.time(), "error": str(cleanup_error),
                            "resources_released": False, "first_error": original})
            released = bool(receipt and receipt.get("resources_released"))
            if not released:
                atomic_json(run / "NEEDS_CLEANUP.json", {"at": time.time(), "resources_released": False,
                    "first_error": original, "retry": "run_topology_benchmark.py cleanup --run-dir RUN --cookie-file PRIVATE"})
            atomic_json(run / "benchmark_summary.json", {"schema": SCHEMA, "plan_id": plan["plan_id"],
                        "benchmark": True, "training": False, "all_reports_passed": complete,
                        "resources_released": released, "passed": complete and released,
                        "first_error": original, "cleanup": receipt})
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return 0 if complete and released else (3 if not released else 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("prepare", "verify", "run", "observe", "cleanup", "worker"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--benchmark-config", type=Path)
    parser.add_argument("--worker-prefix")
    parser.add_argument("--allocated-devices", type=int)
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument("--cookie-file")
    auth.add_argument("--credentials")
    args = parser.parse_args(argv)
    run = args.run_dir.expanduser().resolve()
    try:
        if args.action == "worker":
            if args.worker_prefix is None or args.allocated_devices is None:
                parser.error("worker requires --worker-prefix and --allocated-devices")
            return worker(run, args.worker_prefix, args.allocated_devices)
        api = launch_api()
        if args.action == "prepare":
            if args.config is None or args.benchmark_config is None:
                parser.error("prepare requires --config and --benchmark-config")
            result = prepare_benchmark(args.config, args.benchmark_config, run, api=api)
        elif args.action == "verify":
            plan = verify_benchmark(run, api=api)
            result = {"verified": True, "benchmark": True, "training": False, "plan_id": plan["plan_id"]}
        else:
            if not args.cookie_file and not args.credentials:
                parser.error("Platform actions require private --cookie-file or --credentials")
            if args.action != "cleanup":
                verify_benchmark(run, api=api)
            client = api.make_client(credentials=args.credentials, cookie_file=args.cookie_file)
            if args.action == "run":
                return run_benchmark(client, run, api=api)
            if args.action == "cleanup":
                result = api.cleanup(client, run, stop_controller=False)
            else:
                result = api.snapshot(client, run, fill=False)
                result["benchmark_evidence"] = evidence_status(run, read(run / "benchmark_config.json"))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("resources_released", True) else 3
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({"action": args.action, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
