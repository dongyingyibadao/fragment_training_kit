#!/usr/bin/env python3
"""Prepare, run and clean up isolated LOW/1 resident-training experiments.

No credentials are copied into prepared runs. Preparation is local; only run
and cleanup mutate the platform. This uses the same resident controller as the
OpenPI experiment, including its deliberately bounded fault-recovery contract.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

KIT = Path(__file__).resolve().parent
CORE = KIT / "vendor"
if not (CORE / "jade_lotus").is_dir():
    raise RuntimeError("Missing bundled runtime vendor/jade_lotus; restore this project's source snapshot")
sys.path.insert(0, str(CORE))

from jade_lotus.ascend_fleet import DEFAULT_REGISTRY, active, cards, plan_placements, reconcile
from jade_lotus.ascend_platform import RUNNING, WORKSPACES, make_client, public, validate_selection
from jade_lotus.platform import PlatformError
from jade_lotus.runtime import WriterLockBusy, writer_lock

SCHEMA = "fragment-training-kit-site-v1"
ENVIRONMENT = {
    "HCCL_RDMA_TIMEOUT": "12", "HCCL_BUFFSIZE": "512",
    "HCCL_INTRA_ROCE_ENABLE": "1", "HCCL_CONNECT_TIMEOUT": "120",
    "HCCL_EXEC_TIMEOUT": "120", "HCCL_EVENT_TIMEOUT": "180",
    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def absolute_file(value, name):
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{name} must name an existing absolute file: {path}")
    # Preserve a virtualenv Python's invocation path: resolving its interpreter
    # symlink can silently select the base environment instead of the venv.
    return os.path.abspath(path)


def positive(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return value


def resolve_site(source):
    site = dict(read(source))
    if site.get("schema") != SCHEMA:
        raise ValueError(f"Expected schema {SCHEMA}")
    for name in ("cookie_file", "credentials", "password", "token", "authorization"):
        if name in site:
            raise ValueError("Pass private authentication via CLI, never via a shareable site config")
    for name in ("python", "cann_env", "inventory"):
        site[name] = absolute_file(site[name], name)
    registry = Path(site.get("registry", DEFAULT_REGISTRY)).expanduser()
    if not registry.is_absolute():
        raise ValueError("registry must be absolute and shared by all fleets of the same account")
    site["registry"] = str(registry.resolve())
    site["cards"] = positive(site.get("cards", 2), "cards", 64)
    if site["cards"] < 2:
        raise ValueError("The resident controller requires at least two initial cards")
    site["global_card_limit"] = positive(site.get("global_card_limit", 64), "global_card_limit", 64)
    if site["cards"] > site["global_card_limit"]:
        raise ValueError("cards exceeds the shared global card budget")
    layout = site.get("job_layout", [1] * site["cards"])
    if (not isinstance(layout, list) or not layout
            or any(type(size) is not int or size not in (1, 2, 4, 8, 16) for size in layout)
            or sum(layout) != site["cards"]):
        raise ValueError("job_layout must contain 1/2/4/8/16-card jobs summing to cards")
    site["job_layout"] = layout
    selection_plan = site.get("job_selections")
    if selection_plan is not None:
        if not isinstance(selection_plan, list) or len(selection_plan) != len(layout):
            raise ValueError("job_selections must match every job_layout entry")
        for selection, size in zip(selection_plan, layout, strict=True):
            validate_selection(selection)
            if selection["resource_spec_price"]["gpu_count"] != size:
                raise ValueError("Planned job selection differs from its frozen size")
    site["active_cards"] = positive(site.get("active_cards", site["cards"]), "active_cards", site["cards"])
    if site["active_cards"] < 2:
        raise ValueError("active_cards must include at least two initial training members")
    site.setdefault("defer_standby_until_idle", False)
    if type(site["defer_standby_until_idle"]) is not bool:
        raise ValueError("defer_standby_until_idle must be boolean")
    if site["defer_standby_until_idle"]:
        boundaries = {sum(layout[:index + 1]) for index in range(len(layout))}
        if site["active_cards"] >= site["cards"] or site["active_cards"] not in boundaries:
            raise ValueError("Deferred candidates need an initial active pool ending at a whole job boundary")
    site.setdefault("isolate_model_cache", False)
    if type(site["isolate_model_cache"]) is not bool:
        raise ValueError("isolate_model_cache must be boolean")
    site.setdefault("clear_openpi_data_overrides", False)
    if type(site["clear_openpi_data_overrides"]) is not bool:
        raise ValueError("clear_openpi_data_overrides must be boolean")
    site["steps"] = positive(site.get("steps", 20), "steps", 20000)
    if site["steps"] < 2:
        raise ValueError("Use at least two updates")
    site.setdefault("mode", "external-loss")
    if site["mode"] not in {"external-loss", "controlled-fault"}:
        raise ValueError("mode must be external-loss or controlled-fault")
    site.setdefault("replacement", False)
    if type(site["replacement"]) is not bool:
        raise ValueError("replacement must be boolean")
    if site["mode"] == "external-loss" and site["replacement"]:
        raise ValueError("Arbitrary external-loss replacement is not implemented; use shrink or a fresh resume")
    site.setdefault("fault_step", min(3, site["steps"] - 1))
    if type(site["fault_step"]) is not int or not 1 <= site["fault_step"] < site["steps"]:
        raise ValueError("fault_step must be inside the run and before its final update")
    if site["mode"] == "controlled-fault" and site["replacement"] and site["steps"] < site["fault_step"] + 3:
        raise ValueError("Replacement needs two surviving-group updates and a subsequent joined update")
    site.setdefault("join_transport", "device")
    if site["join_transport"] not in {"file", "device"}:
        raise ValueError("join_transport must be file or device")
    for key, default, maximum in (("bucket_mib", 512, 4096), ("phase_timeout", 600, 86400),
                                  ("collective_timeout", 120, 3600), ("heartbeat_timeout", 15, 600),
                                  ("job_runtime_seconds", 1800, 86400)):
        site[key] = positive(site.get(key, default), key, maximum)
    site.setdefault("control_poll_seconds", .01)
    if site["control_poll_seconds"] not in (.01, .02, .05, .1):
        raise ValueError("control_poll_seconds must be .01, .02, .05 or .1")
    site.setdefault("workspace_ids", list(WORKSPACES))
    if (not isinstance(site["workspace_ids"], list) or not site["workspace_ids"]
            or not set(site["workspace_ids"]).issubset(WORKSPACES)):
        raise ValueError("This validated backend supports only its two explicit QZ WORKSPACES; port the backend for another site")
    environment = dict(ENVIRONMENT)
    for key, value in site.get("environment", {}).items():
        if key not in set(ENVIRONMENT) | {"ASCEND_GLOBAL_LOG_LEVEL", "HCCL_ENTRY_LOG_ENABLE"}:
            raise ValueError(f"Unsupported environment override {key}; device masks and CANN PYTHONPATH must remain assigned")
        if not isinstance(value, (int, str)) or not str(value).isdigit():
            raise ValueError(f"Environment {key} must be a nonnegative integer string")
        environment[key] = str(value)
    site["environment"] = environment
    if not isinstance(site.get("training_config"), dict):
        raise ValueError("training_config must be an inline model/data/optimizer configuration object")
    training = dict(site["training_config"])
    training.setdefault("batch_size", 256)
    training.setdefault("disable_activation_checkpointing", True)
    training.setdefault("resident_final_state", "checkpoint")
    training.setdefault("resident_data_workers", 0)
    if training["batch_size"] != 256 or training["disable_activation_checkpointing"] is not True:
        raise ValueError("This verified controller uses global batch 256 with activation recomputation disabled")
    if training.get("resident_adapter") and training["resident_data_workers"] != 0:
        raise ValueError("Generic adapters own data loading; use resident_data_workers=0")
    if training.get("resident_adapter"):
        for key in ("resident_rng_scheme", "resident_precision_contract"):
            if not isinstance(training.get(key), str) or not training[key].strip():
                raise ValueError(f"Generic adapters require a nonempty {key} before preparation")
    site["training_config"] = training
    if site["active_cards"] < site["cards"]:
        training["resident_topology_slots"] = site["cards"]
    sources = site.get("source_dirs", [])
    if not isinstance(sources, list):
        raise ValueError("source_dirs must list code directories to freeze and add to PYTHONPATH")
    site["source_dirs"] = []
    for value in sources:
        path = Path(value).expanduser()
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("Each source_dirs entry must be an existing absolute directory")
        site["source_dirs"].append(str(path.resolve()))
    return site


def copy_sources(run, site):
    runtime = run / "runtime"
    shutil.copytree(CORE / "jade_lotus", runtime / "jade_lotus",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (runtime / "scripts").mkdir()
    shutil.copy2(CORE / "scripts/probe_resident_openpi_fault.py", runtime / "scripts/probe_resident_openpi_fault.py")
    # Freeze executable kit Python, not its private site configs, vendor copies,
    # documentation, result files, credentials or other runs.
    for path in KIT.rglob("*.py"):
        relative = path.relative_to(KIT)
        if any(part.startswith(".") or part in {"vendor", "runs", "artifacts", "validation", "__pycache__", "tests"}
               for part in relative.parts):
            continue
        target = runtime / "fragment_training_kit" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    extras = []
    for index, source in enumerate(site["source_dirs"]):
        path = Path(source)
        if run.is_relative_to(path) or path.is_relative_to(run):
            raise ValueError("source_dirs and run must be disjoint to avoid recursive or mutable source copies")
        target = runtime / "project_sources" / f"source{index}"
        target.mkdir(parents=True)
        # Explicitly freeze Python source only. Models/data/weights and JSON
        # credentials remain external inputs, never an implicit project copy.
        for item in path.rglob("*.py"):
            relative = item.relative_to(path)
            if any(part.startswith(".") or part in {"__pycache__", "node_modules", "venv", "data", "datasets",
                    "weights", "checkpoints", "runs", "artifacts"} for part in relative.parts):
                continue
            if item.is_symlink() or not item.resolve().is_relative_to(path):
                raise ValueError("source_dirs must contain ordinary Python sources, not external symlinks")
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
        extras.append(str(target))
    return [str(runtime), *extras]


def controller_command(run, site):
    command = [site["python"], "-u", str(run / "runtime/scripts/probe_resident_openpi_fault.py"),
               "--controller", "--run-dir", str(run / "probe"), "--config", str(run / "training.json"),
               "--cards", str(site.get("active_cards", site["cards"])), "--steps", str(site["steps"]),
               "--fault-step", str(site["fault_step"]), "--phase-timeout", str(site["phase_timeout"]),
               "--collective-timeout", str(site["collective_timeout"]), "--bucket-mib", str(site["bucket_mib"]),
               "--heartbeat-timeout", str(site["heartbeat_timeout"]), "--control-poll-seconds", str(site["control_poll_seconds"]),
               "--join-transport", site["join_transport"]]
    if site["mode"] == "external-loss":
        command.append("--external-loss")
    if not site["replacement"]:
        command.append("--no-replacement")
    return command


def launcher_source(run, site, python_paths):
    environment = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in site["environment"].items())
    cache_environment = '''export HF_HOME="$kit_run/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_LEROBOT_HOME="$kit_run/cache/lerobot" OPENPI_DATA_HOME="$kit_run/cache/openpi"
export XDG_CACHE_HOME="$kit_run/cache/xdg"
# CANN creates multiprocessing Unix sockets here. A shared run path can exceed
# the AF_UNIX address limit even when the directory itself is valid.
export TMPDIR=/tmp
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$OPENPI_DATA_HOME" "$XDG_CACHE_HOME"
''' if site.get("isolate_model_cache") else ""
    data_environment = ("unset OPENPI_EXCLUDED_EPISODES_FILE OPENPI_EPISODE_PROMPTS_MANIFEST"
                        if site.get("clear_openpi_data_overrides") else "")
    grouped = any(size > 1 for size in site["job_layout"]) or site.get("active_cards", site["cards"]) < site["cards"]
    worker_command = (f'{shlex.quote(site["python"])} -u "$kit_run/runtime/fragment_training_kit/group_worker.py" --run-dir "$kit_run" --worker-prefix "$kit_worker" --allocated-devices "${{3:?allocated devices}}"'
                      if grouped else f'{shlex.quote(site["python"])} -u "$kit_run/runtime/scripts/probe_resident_openpi_fault.py" --worker --run-dir "$kit_run/probe" --worker-id "$kit_worker"')
    return f'''#!/usr/bin/env bash
set -eo pipefail
kit_run="${{1:?run directory}}"
kit_worker="${{2:?worker ID}}"
source {shlex.quote(site['cann_env'])} >/dev/null 2>&1
export PATH={shlex.quote(str(Path(site['python']).parent))}:"$PATH"
export PYTHONPATH={shlex.quote(':'.join(python_paths))}:"${{PYTHONPATH:-}}"
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:"${{LD_LIBRARY_PATH:-}}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export ASCEND_WORK_PATH="$kit_run/device-runtime/$kit_worker"
export ASCEND_PROCESS_LOG_PATH="$kit_run/device-logs/$kit_worker"
export ACL_OP_COMPILER_CACHE_DIR="$kit_run/device-ops/$kit_worker"
{cache_environment}
{data_environment}
{environment}
mkdir -p "$ASCEND_WORK_PATH" "$ASCEND_PROCESS_LOG_PATH" "$ACL_OP_COMPILER_CACHE_DIR" "$kit_run/worker_logs"
ulimit -c 0
cd "$kit_run/runtime"
exec {worker_command} > "$kit_run/worker_logs/$kit_worker.log" 2>&1
'''


def require_execution_budget(site):
    if site.get("budget_scope", "registered_fleets") != "registered_fleets":
        raise ValueError("Only registered_fleets budget execution is supported; account scope is read-only planning until atomic account accounting is implemented")


def prepare(config_path, run_dir):
    site = resolve_site(config_path)
    require_execution_budget(site)
    run = Path(run_dir).expanduser().resolve()
    if run.exists():
        raise ValueError("Refusing to overwrite an existing run; choose a fresh shared directory")
    inventory = public(read(site["inventory"]))
    inventory["selections"] = [selection for selection in inventory.get("selections", [])
        if selection.get("workspace_id") in site["workspace_ids"]
        and selection.get("resource_spec_price", {}).get("gpu_count") in set(site["job_layout"])]
    if not inventory["selections"]:
        raise ValueError("Inventory has no authorized LOW/1 selection matching job_layout")
    if not set(site["job_layout"]).issubset({s["resource_spec_price"]["gpu_count"] for s in inventory["selections"]}):
        raise ValueError("Inventory is missing a discovered specification for part of job_layout")
    if inventory.get("backend") != "ascend" or not inventory.get("user", {}).get("id"):
        raise ValueError("Use an authenticated Ascend discovery inventory")
    for selection in inventory["selections"]:
        validate_selection(selection)
    if site.get("job_selections") and any(selection not in inventory["selections"] for selection in site["job_selections"]):
        raise ValueError("Planned selections must match the supplied authorized discovery inventory")
    run.mkdir(parents=True, exist_ok=False)
    paths = copy_sources(run, site)
    write(run / "site.json", site)
    write(run / "training.json", site["training_config"])
    write(run / "inventory.json", inventory)
    write(run / "controller_command.json", controller_command(run, site))
    (run / "launch.sh").write_text(launcher_source(run, site, paths))
    grouped = any(n > 1 for n in site["job_layout"]) or site.get("active_cards", site["cards"]) < site["cards"]
    (run / "fleet_command.txt").write_text("bash {run_dir}/launch.sh {run_dir} {worker_id}" + (" {npu_count}" if grouped else "") + "\n")
    files = {str(path.relative_to(run)): digest(path) for path in sorted(run.rglob("*")) if path.is_file()}
    write(run / "source_manifest.json", {"schema": "fragment-training-kit-prepared-v1", "files": files})
    return {"prepared": True, "run_dir": str(run), "submitted": False, "cards": site["cards"],
            "steps": site["steps"], "frozen_files": len(files), "mode": site["mode"],
            "initial_intent_limit": len(site["job_layout"]), "job_layout": site["job_layout"],
            "arbitrary_fleet_refill_joins_supported": False}


def verify(run):
    manifest = read(run / "source_manifest.json")
    actual_runtime = {str(path.relative_to(run)) for path in (run / "runtime").rglob("*")
                      if path.is_file() and "__pycache__" not in path.parts}
    expected_runtime = {name for name in manifest["files"] if name.startswith("runtime/")}
    if actual_runtime != expected_runtime:
        raise ValueError("Prepared runtime file inventory changed")
    for name, expected in manifest["files"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or digest(path) != expected:
            raise ValueError(f"Prepared source/config changed: {name}")
    return {"verified": True, "frozen_files": len(manifest["files"])}


def mirror_markers(run):
    for name in ("COMPLETE.json", "FATAL.json"):
        source = run / "probe" / name
        if source.exists() and not (run / name).exists():
            write(run / name, read(source))
    path = run / "probe/summary.json"
    if path.exists():
        summary = read(path)
        if summary.get("status") in {"passed", "failed"}:
            passed = summary["status"] == "passed"
            site = read(run / "site.json")
            if passed and (summary.get("committed_steps") != site["steps"]
                           or not summary.get("final_states_identical")
                           or (site["training_config"].get("resident_final_state") == "checkpoint"
                               and not summary.get("final_checkpoint"))):
                raise ValueError("Passed controller summary lacks committed target/full-state evidence")
            marker = run / ("COMPLETE.json" if passed else "FATAL.json")
            if not marker.exists():
                write(marker, {"at": time.time(), "status": summary["status"],
                    "reason": "Exact resident controller summary reached a validated terminal state",
                    "error": summary.get("error")})


def process_identity(pid):
    try:
        proc = Path(f"/proc/{pid}")
        raw = (proc / "stat").read_text()
        # comm can contain spaces/parentheses; starttime is field 22.
        ticks = raw[raw.rfind(")") + 2:].split()[19]
        return {"start_ticks": ticks, "argv": (proc / "cmdline").read_bytes().split(b"\0")[:-1]}
    except (OSError, IndexError):
        return None


def start_controller(run):
    with writer_lock(run / "controller-launch-lock"):
        if (run / "controller_process.json").exists():
            return
        if any((run / marker).exists() for marker in ("STOP", "FATAL.json", "COMPLETE.json")):
            raise ValueError("A terminal run cannot start a controller")
        command = read(run / "controller_command.json")
        with (run / "controller.log").open("x") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                cwd=run / "runtime", env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1"),
                start_new_session=True)
        identity = process_identity(child.pid)
        write(run / "controller_process.json", {"pid": child.pid, "started_at": time.time(),
              "start_ticks": identity["start_ticks"] if identity else None, "command": command})


def stop_owned_controller(run):
    record_path = run / "controller_process.json"
    if not record_path.exists():
        return
    record = read(record_path)
    identity = process_identity(record["pid"])
    if (identity and identity["start_ticks"] == record.get("start_ticks")
            and identity["argv"] == [value.encode() for value in record["command"]]):
        os.kill(record["pid"], signal.SIGTERM)


def controller_disappeared(run):
    path = run / "controller_process.json"
    if not path.exists():
        return False
    record = read(path)
    identity = process_identity(record["pid"])
    return not (identity and identity["start_ticks"] == record.get("start_ticks")
                and identity["argv"] == [value.encode() for value in record["command"]])


def snapshot(client, run, *, fill=False):
    # Serialize the intent read together with planning and reconciliation.
    # The budget lock alone starts too late to protect a captured layout index.
    run = Path(run)
    with writer_lock(run / "fleet-snapshot-lock"):
        return _snapshot_locked(client, run, fill=fill)


def _snapshot_locked(client, run, *, fill=False):
    mirror_markers(run)
    site = read(run / "site.json")
    state_path = run / "ascend_fleet/state.json"
    jobs = read(state_path)["jobs"] if state_path.exists() else []
    intents = len(jobs)
    layout = site.get("job_layout", [1] * site["cards"])
    marker = any((run / name).exists() for name in ("STOP", "FATAL.json", "COMPLETE.json"))
    if fill and not marker:
        require_execution_budget(site)
    # The resident controller fences an initial worker-ID set. New fleet IDs
    # after loss cannot join merely by being allocated; do not waste cards.
    creating = fill and not marker and intents < len(layout)
    initial_room = sum(layout[intents:])
    # If an early initial intent already failed, bound total initial identities
    # as well as concurrent reservations: never create w(cards+1).
    occupied = sum(cards(job["payload"]) if job.get("payload") else 1 for job in jobs if active(job))
    limit = min(site["cards"], occupied + initial_room) if creating else site["cards"]
    kwargs = {}
    if site.get("defer_standby_until_idle"):
        progress_path = run / "probe/summary.json"
        started = progress_path.exists() and read(progress_path).get("committed_steps", 0) > 0
        kwargs["placement_planner"] = deferred_layout_planner(client, run, site, intents,
                                                               training_started=started)
    elif any(size > 1 for size in layout) or site.get("job_selections"):
        kwargs["placement_planner"] = layout_planner(layout, intents, site.get("job_selections"))
    result = reconcile(client, run, (run / "fleet_command.txt").read_text().strip(), run / "inventory.json",
        card_limit=limit, fill=creating or (fill and marker), registry_path=site["registry"],
        global_card_limit=site["global_card_limit"], job_runtime_seconds=site["job_runtime_seconds"], retries=0, **kwargs)
    keep = ("at", "marker", "active_jobs", "running_jobs", "pending_jobs", "local_requested_reserved_cards",
            "verified_allocated_cards", "created", "blocked_reason", "errors")
    status = {key: result[key] for key in keep}
    summary_path = run / "probe/summary.json"
    if summary_path.exists():
        summary = read(summary_path)
        status.update(committed_steps=summary.get("committed_steps"), stage=summary.get("stage"),
                      training_status=summary.get("status"))
    status["initial_intent_limit"] = len(layout)
    status["job_layout"] = layout
    status["arbitrary_fleet_refill_joins_supported"] = False
    write(run / "launcher_status.json", status)
    return status


def deferred_layout_planner(client, run, site, already_created, *, training_started):
    """Allocate frozen candidates only after a COMMIT and fresh idle-node proof.

    This callback runs inside reconcile's existing shared registry lock. Node
    query failures only defer candidates; they never stop an active trainer.
    It cannot replace failed intents, invent new slots, pin nodes, or release
    old groups. Candidate admission remains a separate controller decision.
    """
    layout, selections = site["job_layout"], site.get("job_selections")
    initial_jobs = next(index + 1 for index in range(len(layout))
                        if sum(layout[:index + 1]) == site["active_cards"])
    if already_created < initial_jobs:
        return layout_planner(layout[:initial_jobs], already_created,
                              selections[:initial_jobs] if selections else None)

    def plan(inventory, reserved, available, backoff, now):
        if not training_started or not available or already_created >= len(layout):
            return []
        from jade_lotus.ascend_nodes import discover_nodes
        from jade_lotus.topology_placement import plan_topology, reservation_ledger
        evidence = {"at": time.time(), "fixed_candidate_slots_only": True,
                    "created_intents": already_created, "allocation_verified": False}
        result = []
        try:
            state_path = str(Path(run) / "ascend_fleet/state.json")
            candidate_prefixes = {f"w{index + 1:06d}" for index in range(initial_jobs, already_created)}
            if any(state_path in row.get("fleet_paths", ())
                    and row.get("entry", {}).get("worker_id") in candidate_prefixes
                    and row.get("status") not in RUNNING for row in reserved):
                evidence["reason"] = "previous candidate has not reached an official running allocation"
                evidence["placements"] = []
                return []
            nodes = discover_nodes(client, inventory)
            evidence["node_snapshot"] = nodes
            if nodes.get("errors"):
                raise PlatformError("Incomplete node query; candidate allocation deferred")
            reservations = list(reserved)
            for index in range(already_created, len(layout)):
                size = layout[index]
                if size > available:
                    break
                subset = copy.deepcopy(inventory)
                subset["selections"] = [s for s in subset["selections"]
                    if s["resource_spec_price"]["gpu_count"] == size
                    and (selections is None or s == selections[index])]
                used = sum(row["cards"] for row in reservation_ledger(reservations))
                candidate = plan_topology(subset, nodes, reservations,
                    global_card_limit=used + size, local_card_limit=used + size,
                    group_backoff=backoff, queue_when_idle_exhausted=False)
                if not candidate["placements"]:
                    break
                placement = candidate["placements"][0]
                if placement["cards"] != size or placement["reason"] not in {"idle_whole_node", "idle_fragment"}:
                    raise PlatformError("Candidate plan has no exact fresh idle-node match")
                result.append(placement)
                chosen = placement["selection"]
                reservations.append({"selection": chosen, "cards": size, "status": "planned",
                    "workspace_id": chosen["workspace_id"], "group_id": chosen["logic_compute_group_id"],
                    "category": "candidate"})
                available -= size
                # A second call would reset per-node capacity to the original
                # snapshot. Wait for the next fresh observation instead.
                break
            evidence["placements"] = result
        except Exception as error:
            # Optional node discovery/admission must not turn malformed API
            # data into a fatal error for the already active training group.
            result = []
            evidence.update(error=f"{type(error).__name__}: {error}", placements=[], training_stop_requested=False)
        finally:
            try:
                write(Path(run) / "topology_allocation/latest.json", evidence)
            except (OSError, ValueError, TypeError):
                # A diagnostic write is optional; do not allocate without its
                # receipt, and do not stop healthy training to write one.
                result.clear()
        return result
    return plan


def layout_planner(layout, already_created, selections=None):
    """Keep job sizes and global rank ownership frozen across scheduler polls."""
    def plan(inventory, reserved, available, backoff, now):
        reservations = list(reserved)
        result = []
        for index in range(already_created, len(layout)):
            size = layout[index]
            if size > available:
                break
            subset = copy.deepcopy(inventory)
            subset["selections"] = [s for s in subset["selections"] if s["resource_spec_price"]["gpu_count"] == size]
            if selections is not None:
                subset["selections"] = [s for s in subset["selections"] if s == selections[index]]
            candidates = plan_placements(subset, reservations, size, backoff, now)
            if not candidates:
                break
            selected = candidates[0]
            result.append(selected)
            selection = selected["selection"]
            reservations.append({"cards": size, "status": "planned", "workspace_id": selection["workspace_id"],
                "group_id": selection["logic_compute_group_id"],
                "entry": {"payload": {"framework_config": [{"resource_spec_price": selection["resource_spec_price"]}]}}})
            available -= size
        return result
    return plan


def cleanup(client, run, *, wait_seconds=120, stop_controller=True):
    if not (run / "STOP").exists():
        (run / "STOP").write_text("Owned-run cleanup requested by fragment_training_kit.launch\n")
    deadline = time.monotonic() + wait_seconds
    status = {"active_jobs": None, "local_requested_reserved_cards": None}
    while True:
        try:
            status = snapshot(client, run, fill=True)
        except WriterLockBusy as error:
            status = {**status, "lock_busy": str(error), "lock_path": str(error.path)}
        released = status["active_jobs"] == 0 and status["local_requested_reserved_cards"] == 0
        if released or time.monotonic() >= deadline:
            # A failed attempt to acquire the budget lock must never kill the
            # controller while its still-running workers wait for APPLY.
            if released and stop_controller:
                stop_owned_controller(run)
            status["resources_released"] = released
            status["controller_stop_deferred"] = bool(stop_controller and not released)
            write(run / "cleanup_receipt.json", status)
            return status
        time.sleep(min(5, max(0, deadline - time.monotonic())))


def run_training(client, run, *, watch_seconds=1800, interval_seconds=10):
    verify(run)
    require_execution_budget(read(run / "site.json"))
    with writer_lock(run / "driver-lock"):
        start_controller(run)
        deadline = time.monotonic() + watch_seconds
        complete_seen = None
        platform_failures = 0
        def interrupted(signum, frame):
            raise InterruptedError(f"Launcher interrupted by signal {signum}")
        previous = {signum: signal.signal(signum, interrupted) for signum in (signal.SIGINT, signal.SIGTERM)}
        try:
            while True:
                mirror_markers(run)
                complete = (run / "COMPLETE.json").exists()
                fatal = (run / "FATAL.json").exists() or (run / "STOP").exists()
                if not complete and not fatal and controller_disappeared(run):
                    write(run / "FATAL.json", {"at": time.time(), "status": "failed",
                        "reason": "The exact owned controller exited without a terminal summary; inspect controller.log"})
                    fatal = True
                try:
                    status = snapshot(client, run, fill=not complete)
                    platform_failures = 0
                except WriterLockBusy as error:
                    write(run / "lock_retry.json", {"at": time.time(), "lock_path": str(error.path),
                        "error": str(error), "training_stop_requested": False})
                    if time.monotonic() >= deadline:
                        write(run / "WATCH_EXPIRED.json", {"watch_seconds": watch_seconds, "at": time.time()})
                        cleanup(client, run)
                        return 3
                    time.sleep(min(interval_seconds, max(0, deadline - time.monotonic())))
                    continue
                except (PlatformError, OSError) as error:
                    platform_failures += 1
                    diagnostic = {"at": time.time(), "attempt": platform_failures,
                                  "error": f"{type(error).__name__}: {error}"}
                    if not (run / "first_platform_error.json").exists():
                        write(run / "first_platform_error.json", diagnostic)
                    write(run / "platform_retry.json", diagnostic)
                    print(json.dumps({"platform_retry": diagnostic}, ensure_ascii=False), flush=True)
                    # A brief query failure must not kill a healthy trainer.
                    # Persistent loss of control is bounded and leaves an
                    # explicit retryable cleanup receipt if stop cannot work.
                    if platform_failures >= 3 or time.monotonic() >= deadline:
                        raise
                    time.sleep(min(5, max(0, deadline - time.monotonic())))
                    continue
                print(json.dumps(status, ensure_ascii=False), flush=True)
                if complete:
                    complete_seen = time.monotonic() if complete_seen is None else complete_seen
                    if status["active_jobs"] == 0:
                        status["resources_released"] = status["local_requested_reserved_cards"] == 0
                        write(run / "cleanup_receipt.json", status)
                        return 0
                    # Give naturally exiting workers time to become succeeded;
                    # after a bounded grace period stop only remaining owned jobs.
                    if time.monotonic() - complete_seen > 30:
                        final = cleanup(client, run, stop_controller=False)
                        return 0 if final["resources_released"] else 3
                elif fatal or status["blocked_reason"]:
                    cleanup(client, run)
                    return 1
                if time.monotonic() >= deadline:
                    write(run / "WATCH_EXPIRED.json", {"watch_seconds": watch_seconds, "at": time.time()})
                    cleanup(client, run)
                    return 3
                time.sleep(min(interval_seconds, max(0, deadline - time.monotonic())))
        except BaseException as error:
            # Persist the original failure before cleanup. A platform outage
            # can prevent confirmation; leave an explicit retryable receipt.
            write(run / "NEEDS_CLEANUP.json", {"at": time.time(), "original_error": f"{type(error).__name__}: {error}",
                  "resources_released": False, "retry": "launch.py cleanup --run-dir RUN --cookie-file PRIVATE"})
            try:
                result = cleanup(client, run, wait_seconds=30)
                if result["resources_released"]:
                    (run / "NEEDS_CLEANUP.json").unlink(missing_ok=True)
            except BaseException as cleanup_error:
                write(run / "cleanup_failed.json", {"at": time.time(), "error": f"{type(cleanup_error).__name__}: {cleanup_error}",
                      "resources_released": False})
            raise
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "run", "observe", "cleanup"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Site JSON, required only for prepare")
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument("--cookie-file", help="Existing private 0600 Netscape SSO jar; never copied")
    auth.add_argument("--credentials", help="Existing private official OAuth credentials; never copied")
    parser.add_argument("--watch-seconds", type=int, default=1800)
    parser.add_argument("--interval-seconds", type=int, default=10)
    args = parser.parse_args(argv)
    if not 1 <= args.watch_seconds <= 7 * 86400 or not 1 <= args.interval_seconds <= 60:
        parser.error("watch-seconds must be 1..604800 and interval-seconds 1..60")
    try:
        run = args.run_dir.expanduser().resolve()
        if args.action == "prepare":
            if args.config is None:
                parser.error("prepare requires --config")
            result = prepare(args.config, run)
        elif args.action == "verify":
            result = verify(run)
        else:
            if not args.cookie_file and not args.credentials:
                parser.error("Platform actions require a private --cookie-file or --credentials")
            # Prepared source/config integrity is checked before all platform
            # actions except cleanup: cleanup must remain possible after bugs.
            if args.action != "cleanup":
                verify(run)
            client = make_client(credentials=args.credentials, cookie_file=args.cookie_file)
            if args.action == "run":
                return run_training(client, run, watch_seconds=args.watch_seconds, interval_seconds=args.interval_seconds)
            result = cleanup(client, run) if args.action == "cleanup" else snapshot(client, run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("resources_released", True) else 3
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({"error": str(error), "action": args.action}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
