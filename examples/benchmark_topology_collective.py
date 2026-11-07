"""Compare API-level flat SUM and explicit hierarchy on already assigned devices.

No platform jobs, environment installs or device allocation are performed.
Before starting an experiment, inspect existing device users/keepalive and use
the site's task-owned pause-file workflow where needed. Preserve monitoring.

For a four-rank CPU smoke test, from the kit directory (not run automatically):

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 torchrun --standalone --nproc-per-node=4 \
      examples/benchmark_topology_collective.py --device cpu --ranks-per-node 2 \
      --bytes 1MiB --output-dir /tmp/unique-topology-benchmark

For existing independent NPU jobs, start one copy in each job with consistent
RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT and a verified --topology JSON. Use
LOCAL_RANK=0 for a job with one allocated visible NPU. A native multidevice job
can instead use torchrun. Existing visibility masks are preserved.

Topology JSON: {"node_ranks": [[0, 1], [2, 3]], "evidence": "explicit source"}.
Alternatively provide {"node_ids": ["host-a", "host-a", "host-b", "host-b"]}.
--ranks-per-node explicitly groups contiguous global ranks; it does not verify
hosts or HCCS domains. A pod hostname must not be treated as a physical node.

--rdma-timeouts 16,12,16 runs a new bounded child process for each value, with
the same visible devices/rank mapping/payload. Environment values are set
before torch/torch_npu import. Each point uses a fresh rendezvous port; reserve
MASTER_PORT through MASTER_PORT+points-1. --modes flat,hierarchy,flat gives an
in-process A/B/A with newly constructed subgroups. Add flat,flat for A/A.
Use external HCCL logs/device profiling to establish the actual algorithm and
transport; the Python API label "flat" does not establish an unlayered HCCL
implementation. RDMA timeout is an exponent, not a controller timeout in seconds.

--payload-json accepts {"buckets": [{"dtype": "bfloat16", "bytes": 536870912},
...]}; order and byte counts are preserved. This can reproduce a model's exact
mixed-dtype bucket sequence. The default is one 64MiB FP32 bucket. This tool's
phase-dependent hierarchy does not reproduce the historical flat benchmark's
13 concurrently outstanding AllReduce operations: compare both implementations
with this same scheduling contract, and report a separate historical replay.

Group creation, resets, source copies, preflight, verification and warmup are
outside the reported data time. Every measured iteration waits for all work
and device synchronization, then reports the maximum rank-local elapsed. No
uncalibrated cross-host timestamps are subtracted. Full-buffer min/max checks
verify SUM and source preservation; buffers contain deterministic rank-weighted
integer contributions. Floating reduction order can still differ in real training.

Failure invalidates the entire epoch. A parent process enforces a hard deadline
even if backend shutdown or device synchronization hangs. There is no dynamic
membership/recovery integration. No failed case is silently retried or replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import traceback
import uuid
from datetime import timedelta

KIT = Path(__file__).resolve().parents[1]


def runtime_root(kit: Path) -> Path:
    """Accept only the source kit layout or its explicit prepared snapshot."""
    if (kit / "vendor/jade_lotus/topology_collective.py").is_file():
        return kit / "vendor"
    if (kit.name == "fragment_training_kit" and kit.parent.name == "runtime"
            and (kit.parent / "jade_lotus/topology_collective.py").is_file()):
        return kit.parent
    raise RuntimeError("Missing explicit vendor/jade_lotus or prepared runtime/jade_lotus; no source fallback")


CORE = runtime_root(KIT)
sys.path.insert(0, str(CORE))
from jade_lotus.topology_collective import RankTopology, TopologyCollective, synchronize_device

WIDTHS = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
ENV_EVIDENCE = ("HCCL_RDMA_TIMEOUT", "HCCL_INTRA_ROCE_ENABLE", "HCCL_ALGO", "HCCL_BUFFSIZE",
                "HCCL_CONNECT_TIMEOUT", "HCCL_EXEC_TIMEOUT", "HCCL_EVENT_TIMEOUT",
                "HCCL_ENTRY_LOG_ENABLE", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES", "GLOO_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME")


def byte_count(value: str | int) -> int:
    if isinstance(value, bool):
        raise ValueError("Byte counts must be positive integers")
    match = re.fullmatch(r"([0-9]+)\s*(B|KiB|MiB|GiB|K|M|G)?", str(value).strip(), re.IGNORECASE)
    if not match:
        raise ValueError("Use integer bytes or KiB/MiB/GiB")
    unit = (match[2] or "B").lower()
    size = int(match[1]) * {"b": 1, "kib": 1024, "mib": 1024**2, "gib": 1024**3,
                            "k": 1024, "m": 1024**2, "g": 1024**3}[unit]
    if size < 1:
        raise ValueError("Byte counts must be positive integers")
    return size


def read_payload(args) -> list[dict]:
    if args.payload_json:
        payload = json.loads(Path(args.payload_json).read_text())
        buckets = payload["buckets"] if isinstance(payload, dict) else payload
    else:
        buckets = [{"dtype": args.dtype, "bytes": size} for size in args.bytes.split(",")]
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("Payload must contain an ordered nonempty bucket list")
    result = []
    for row in buckets:
        dtype, size = row["dtype"], byte_count(row["bytes"])
        if dtype not in WIDTHS or size % WIDTHS[dtype]:
            raise ValueError("Every bucket needs a supported dtype and an exact element-aligned byte count")
        result.append({"dtype": dtype, "bytes": size, "numel": size // WIDTHS[dtype]})
    return result


def read_topology(args, world_size: int) -> tuple[RankTopology, dict]:
    if args.topology:
        path = Path(args.topology)
        raw = path.read_bytes()
        content = json.loads(raw)
        if isinstance(content, list):
            topology = RankTopology.build(content, world_size)
        elif "node_ranks" in content:
            topology = RankTopology.build(content["node_ranks"], world_size)
        else:
            topology = RankTopology.from_node_ids(content["node_ids"])
            if topology.world_size != world_size:
                raise ValueError("Topology world size differs from the distributed world")
        return topology, {"source": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                          "evidence": content.get("evidence") if isinstance(content, dict) else None}
    if args.ranks_per_node:
        nodes = [list(range(start, min(start + args.ranks_per_node, world_size)))
                 for start in range(0, world_size, args.ranks_per_node)]
        return RankTopology.build(nodes, world_size), {
            "source": "explicit contiguous rank grouping",
            "limit": "Physical hosts, HCCS domains and transports were not discovered or verified"}
    if all(mode == "flat" for mode in args.modes.split(",")):
        return RankTopology.build([range(world_size)], world_size), {"source": "unused for flat SUM"}
    raise ValueError("Hierarchy requires --topology or an explicit --ranks-per-node")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    return ordered[lower] + (ordered[math.ceil(position)] - ordered[lower]) * (position - lower)


def write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard link publishes complete JSON atomically and
        # refuses to replace an existing evidence file.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=("cpu", "cuda", "npu"), default="npu")
    p.add_argument("--backend", choices=("gloo", "nccl", "hccl"))
    p.add_argument("--device-index", type=int)
    p.add_argument("--expected-visible-devices", type=int,
                   help="Require exactly this existing allocation size, preserving its visibility mask")
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    p.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    p.add_argument("--local-rank", "--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    p.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    p.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29500")))
    grouping = p.add_mutually_exclusive_group()
    grouping.add_argument("--topology")
    grouping.add_argument("--ranks-per-node", type=int)
    p.add_argument("--modes", default="flat,hierarchy", help="Ordered comma-separated flat/hierarchy modes; repeats allowed")
    p.add_argument("--hierarchy-strategy", choices=("auto", "sharded", "leader"), default="auto")
    p.add_argument("--uneven-fallback", choices=("leader", "flat", "error"), default="leader")
    p.add_argument("--dtype", choices=tuple(WIDTHS), default="float32")
    p.add_argument("--bytes", default="64MiB", help="Ordered comma-separated byte counts for --dtype")
    p.add_argument("--payload-json")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--timeout-seconds", type=float, default=120, help="Gloo/data PG timeout; independent of the RDMA exponent")
    p.add_argument("--hard-timeout-seconds", type=float, default=1200, help="Maximum child process lifetime, including initialization and cleanup")
    p.add_argument("--rdma-timeout", type=int)
    p.add_argument("--rdma-timeouts", help="Fresh-process sweep, e.g. 16,12,16; reserve consecutive rendezvous ports")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--plan-only", action="store_true", help="Validate and print the plan without importing torch or using devices")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return p


def validate_args(args) -> tuple[list[dict], RankTopology, dict, list[str], list[int | None]]:
    modes = args.modes.split(",")
    if not modes or any(mode not in {"flat", "hierarchy"} for mode in modes):
        raise ValueError("--modes accepts flat and hierarchy")
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("Invalid rank/world-size")
    if args.local_rank < 0 or (args.device_index is not None and args.device_index < 0):
        raise ValueError("Local device indices must be nonnegative")
    if args.expected_visible_devices is not None and args.expected_visible_devices < 1:
        raise ValueError("--expected-visible-devices must be positive")
    if args.ranks_per_node is not None and args.ranks_per_node < 1:
        raise ValueError("--ranks-per-node must be positive")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("Use nonnegative warmup and positive measurement iterations")
    for timeout in (args.timeout_seconds, args.hard_timeout_seconds):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeouts must be finite and positive")
    if args.rdma_timeouts and args.rdma_timeout is not None:
        raise ValueError("Use --rdma-timeout or --rdma-timeouts")
    timeouts = [int(value) for value in args.rdma_timeouts.split(",")] if args.rdma_timeouts else [args.rdma_timeout]
    if any(value is not None and not 0 <= value <= 31 for value in timeouts):
        raise ValueError("RDMA timeout exponents must be in 0..31; installed HCCL may impose a narrower range")
    if not 1 <= args.master_port <= 65535 - len(timeouts) + 1:
        raise ValueError("Rendezvous port range is invalid")
    topology, evidence = read_topology(args, args.world_size)
    if "hierarchy" in modes:
        topology.select(args.hierarchy_strategy, args.uneven_fallback)
    return read_payload(args), topology, evidence, modes, timeouts


def _child_args(argv: list[str]) -> list[str]:
    removed = {"--rdma-timeout", "--rdma-timeouts", "--output-dir", "--master-port"}
    result, index = [], 0
    while index < len(argv):
        key = argv[index].split("=", 1)[0]
        if key in removed:
            index += 1 if "=" in argv[index] else 2
        elif key == "--_worker":
            index += 1
        else:
            result.append(argv[index])
            index += 1
    return result


def _supervise(args, timeouts: list[int | None]) -> int:
    root = Path(args.output_dir).resolve()
    for index, exponent in enumerate(timeouts):
        label = f"{index:03d}-rdma-{exponent if exponent is not None else 'inherited'}"
        case = root / label
        output = case / f"rank-{args.rank:06d}.json"
        if output.exists() or (case / f"watchdog-rank-{args.rank:06d}.json").exists():
            raise FileExistsError(f"Preserving existing evidence: {output}")
        argv = [sys.executable, str(Path(__file__).resolve()), *_child_args(sys.argv[1:]),
                "--_worker", "--output-dir", str(case), "--master-port", str(args.master_port + index)]
        env = os.environ.copy()
        if exponent is not None:
            env["HCCL_RDMA_TIMEOUT"] = str(exponent)
            argv.extend(["--rdma-timeout", str(exponent)])
        child = subprocess.Popen(argv, env=env, start_new_session=True)
        try:
            status = child.wait(timeout=args.hard_timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=10)
            write_new_json(case / f"watchdog-rank-{args.rank:06d}.json", {
                "passed": False, "rank": args.rank, "reason": "hard process deadline exceeded",
                "hard_timeout_seconds": args.hard_timeout_seconds, "child_pid": child.pid,
                "rdma_timeout": exponent, "no_automatic_retry": True})
            return 124
        except BaseException:
            # Only this exact child process group is owned by this invocation.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
            raise
        if status:
            return status if status > 0 else 128 - status
    return 0


def supervise(args, timeouts: list[int | None]) -> int:
    def interrupted(signum, _frame):
        raise InterruptedError(f"Benchmark supervisor interrupted by signal {signum}")
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        return _supervise(args, timeouts)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def worker(args, payload: list[dict], topology: RankTopology, evidence: dict, modes: list[str]) -> int:
    # All HCCL configuration must precede importing a backend with cached env.
    if args.rdma_timeout is not None:
        os.environ["HCCL_RDMA_TIMEOUT"] = str(args.rdma_timeout)
    if args.device == "cpu":
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch
    import torch.distributed as dist

    output = Path(args.output_dir) / f"rank-{args.rank:06d}.json"
    report = {"schema": "topology-collective-benchmark-v1", "passed": False,
              "rank": args.rank, "world_size": args.world_size,
              "hostname_scope": "process/container hostname; not verified physical host",
              "hostname": socket.gethostname(), "pid": os.getpid(),
              "payload": payload, "payload_bytes": sum(row["bytes"] for row in payload),
              "node_ranks": topology.nodes, "topology_evidence": evidence,
              "environment": {name: os.environ.get(name) for name in ENV_EVIDENCE},
              "torch_version": str(torch.__version__), "cases": [],
              "process_group_timeout_seconds": args.timeout_seconds,
              "hard_process_timeout_seconds": args.hard_timeout_seconds,
              "physical_identity_limit": "Visibility masks are preserved; independently verify physical allocation identity",
              "scheduling": "Work waits between hierarchy phases; one final device sync per bucket sequence",
              "timing": "maximum of rank-local complete data times; copies/control/validation excluded",
              "dynamic_recovery_integrated": False, "platform_jobs_created": 0,
              "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__), CORE / "jade_lotus/topology_collective.py")}}
    reducer = None
    try:
        device_index = args.device_index if args.device_index is not None else args.local_rank
        backend = args.backend or {"cpu": "gloo", "cuda": "nccl", "npu": "hccl"}[args.device]
        if backend != {"cpu": "gloo", "cuda": "nccl", "npu": "hccl"}[args.device]:
            raise ValueError("Use the matching device backend for comparable data-path timing")
        if args.device == "npu":
            import torch_npu
            report["torch_npu_version"] = str(getattr(torch_npu, "__version__", "unknown"))
        if args.device == "cpu":
            device = torch.device("cpu")
        else:
            module = getattr(torch, args.device)
            if (args.expected_visible_devices is not None
                    and module.device_count() != args.expected_visible_devices):
                raise ValueError("Visible device count differs from the frozen allocated pod size")
            if not 0 <= device_index < module.device_count():
                raise ValueError("Local device index is outside the existing visible allocation")
            module.set_device(device_index)
            device = torch.device(args.device, device_index)
            report["visible_device_count"] = module.device_count()
            report["device_name"] = module.get_device_name(device_index)
        report.update(device=str(device), backend=backend)
        for name, value in {"MASTER_ADDR": args.master_addr, "MASTER_PORT": args.master_port,
                            "RANK": args.rank, "WORLD_SIZE": args.world_size}.items():
            os.environ[name] = str(value)
        dist.init_process_group(backend=backend, init_method="env://", rank=args.rank,
                                world_size=args.world_size, timeout=timedelta(seconds=args.timeout_seconds))
        for case_index, mode in enumerate(modes):
            strategy = "flat" if mode == "flat" else args.hierarchy_strategy
            reducer = TopologyCollective(topology, strategy=strategy, uneven_fallback=args.uneven_fallback,
                                         epoch=f"fixed-case-{case_index}", timeout_seconds=args.timeout_seconds)
            source = None
            allocation_error = None
            try:
                source = [torch.empty(row["numel"], dtype=getattr(torch, row["dtype"]), device=device)
                          for row in payload]
            except Exception as error:
                allocation_error = str(error)
            reducer._agree("source_allocation", None, allocation_error)
            case = {"mode": mode, "resolved_strategy": reducer.strategy,
                    "group_schedule": reducer.topology.group_schedule(reducer.strategy),
                    "warmup_iterations": args.warmup, "measurement_iterations": args.iterations,
                    "rank_max_seconds": [], "local_seconds": [],
                    "communication_model_per_bucket": []}
            for row in payload:
                local = len(reducer.node_ranks) if reducer.strategy == "sharded" else 1
                padded_bytes = ((row["numel"] + local - 1) // local) * local * WIDTHS[row["dtype"]]
                case["communication_model_per_bucket"].append(
                    topology.communication_model(padded_bytes, reducer.strategy))
            report["cases"].append(case)
            for iteration in range(args.warmup + args.iterations):
                reset_error = None
                try:
                    for bucket, tensor in enumerate(source):
                        tensor.fill_(1 + ((args.rank + bucket) % 3))
                    synchronize_device(source[0])
                except Exception as error:
                    reset_error = str(error)
                reducer._agree("source_reset", {"iteration": iteration}, reset_error)
                prepared = reducer.prepare(source)
                result = reducer.sum(prepared)
                check_error = None
                try:
                    for bucket, (original, actual) in enumerate(zip(source, result.tensors, strict=True)):
                        expected = sum(1 + ((rank + bucket) % 3) for rank in range(args.world_size))
                        # Integral values are exact up to 256 in BF16 and 2048
                        # in FP16. Larger worlds use explicit dtype tolerance.
                        tolerance = {"bfloat16": 0.03, "float16": 0.005,
                                     "float32": 0.00001, "float64": 1e-12}[payload[bucket]["dtype"]]
                        if expected <= {"bfloat16": 256, "float16": 2048,
                                        "float32": 2**24, "float64": 2**53}[payload[bucket]["dtype"]]:
                            tolerance = 0.0
                        observed = [float(actual.min().item()), float(actual.max().item())]
                        if any(not math.isfinite(value) or abs(value - expected) > tolerance * abs(expected)
                               for value in observed):
                            raise ValueError(f"Bucket {bucket}: full-buffer SUM extrema {observed}, expected {expected}")
                        own = 1 + ((args.rank + bucket) % 3)
                        if float(original.min().item()) != own or float(original.max().item()) != own:
                            raise ValueError("Original local_grad was modified")
                except Exception as error:
                    check_error = str(error)
                reducer._agree("full_buffer_SUM_and_source_check", {"iteration": iteration}, check_error)
                elapsed = [None] * args.world_size
                dist.all_gather_object(elapsed, result.communication_seconds, group=reducer.control_group)
                if iteration >= args.warmup:
                    case["local_seconds"].append(result.communication_seconds)
                    case["rank_max_seconds"].append(max(elapsed))
                del prepared, result
            values = case["rank_max_seconds"]
            case.update(mean_seconds=statistics.mean(values), p50_seconds=percentile(values, .5),
                        p95_seconds=percentile(values, .95), min_seconds=min(values), max_seconds=max(values),
                        full_buffer_SUM_verified=True, original_local_grad_preserved=True)
            reducer.close()
            reducer = None
            del source
        report["passed"] = True
        dist.destroy_process_group()
        write_new_json(output, report)
        if args.rank == 0:
            print(json.dumps({"passed": True, "output": str(output.resolve()),
                              "cases": [{"mode": case["mode"], "strategy": case["resolved_strategy"],
                                         "mean_seconds": case["mean_seconds"], "p95_seconds": case["p95_seconds"]}
                                        for case in report["cases"]]}, indent=2), flush=True)
        return 0
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc(),
                      failed_epoch_must_be_discarded=True)
        if reducer is not None:
            reducer.failed = True
        try:
            write_new_json(output, report)
        except FileExistsError:
            pass
        traceback.print_exc()
        # Avoid Python/torch finalizers blocking inside a failed backend. The
        # outer launcher observes this exit and peers have finite deadlines.
        sys.stderr.flush()
        os._exit(1)


def main() -> int:
    args = parser().parse_args()
    payload, topology, evidence, modes, timeouts = validate_args(args)
    if args.plan_only:
        print(json.dumps({"payload": payload, "payload_bytes": sum(row["bytes"] for row in payload),
                          "node_ranks": topology.nodes, "topology_evidence": evidence,
                          "modes": modes, "rdma_timeouts": timeouts,
                          "platform_jobs_created": 0, "devices_used": False}, indent=2))
        return 0
    if not args._worker:
        return supervise(args, timeouts)
    return worker(args, payload, topology, evidence, modes)


if __name__ == "__main__":
    raise SystemExit(main())
