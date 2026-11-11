"""One durable, bounded physical-card budget across both QZ Ascend spaces."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys
import time
import uuid

from jade_lotus.ascend_platform import (API, RUNNING, TERMINAL, WORKSPACES, inspect_job,
    job_id, list_own_jobs, make_client, now_iso, positive_int, public,
    render_payload, resolve_submission, submit, validate_payload, validate_selection)
from jade_lotus.fleet import failure_kind as gpu_failure_kind
from jade_lotus.platform import PlatformError, resource_summary, write_json
from jade_lotus.runtime import writer_lock


DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "artifacts/ascend-fleet-registry.json"
MARKERS = ("STOP", "FATAL.json", "COMPLETE.json")


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (ValueError, TypeError):
        return default


def active(entry):
    # A local terminal label is insufficient; retain ambiguous remote intent.
    return not (entry.get("official_terminal") is True or
                entry.get("status") in {"submission_rejected", "cancelled_before_submission"})


def cards(payload):
    configs = payload.get("framework_config")
    if not isinstance(configs, list) or not configs:
        raise PlatformError("Cannot account a request with no framework_config")
    total = 0
    for component in configs:
        count = positive_int(component.get("instance_count"), "instance_count")
        device_count = positive_int(component.get("gpu_count"), "gpu_count", 32)
        spec = component.get("resource_spec_price") or component.get("instance_spec_price_info") or {}
        if spec.get("gpu_count") is not None and positive_int(spec["gpu_count"], "spec gpu_count", 32) != device_count:
            raise PlatformError("Conflicting device counts in a durable request")
        total += count * device_count
    return total


def failure_kind(status, evidence):
    result = gpu_failure_kind(status, evidence)
    if result == "resource_or_runtime":
        return result
    text = json.dumps(evidence, ensure_ascii=False).casefold()
    if any(fragment in text for fragment in ("npu out of memory", "npu oom", "acl error", "cann error",
            "hccl error", "err00100", "ez9999", "libhccl", "memory allocation failed", "acl_error")):
        return "software_or_capacity"
    return result


def failed_job_evidence(client, identifier, running_round=None):
    """Use current V2 job events and instances; API errors are diagnostics only."""
    evidence, diagnostics = {}, []
    for action, body in (("events", {"pageNum": -1, "pageSize": 100,
                                   "filter": {"object_ids": [identifier], "object_type": "job"},
                                   "sorter": [{"field": "last_timestamp", "sort": "ascend"}]}),
                         ("instances", {"jobId": identifier, "page_num": 1, "page_size": 100})):
        try:
            response = client.request("POST", API[action], body)
            if action == "instances" and running_round is not None:
                response = copy.deepcopy(response)
                data = response.get("data")
                if isinstance(data, dict):
                    for key in ("items", "instances", "list"):
                        if isinstance(data.get(key), list):
                            data[key] = [row for row in data[key] if row.get("running_round", running_round) == running_round]
            evidence[action] = response
        except (PlatformError, OSError, TimeoutError) as error:
            diagnostics.append({"action": action, "error": str(error)})
    kind = failure_kind("job_failed", evidence)
    if diagnostics:
        evidence["diagnostics"] = diagnostics
    return evidence, kind


def _time(value):
    if value in (None, ""):
        return None
    try:
        stamp = float(value)
        return stamp / 1000 if stamp > 1e12 else stamp
    except (ValueError, TypeError):
        try:
            return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None


def allocation(instances, running_round=None):
    """Count only explicit official allocated resources, never requested cards."""
    devices, count, starts, selected = set(), 0, [], []
    candidates = instances
    if running_round is not None:
        candidates = [row for row in candidates if row.get("running_round", running_round) == running_round]
    for row in candidates:
        if row.get("instance_status", row.get("status")) not in RUNNING | {"Running", "RUNNING", "instance_running"}:
            continue
        host = row.get("node_name") or row.get("node") or row.get("hostname") or row.get("host_ip")
        instance_id = row.get("instance_id") or row.get("name")
        device_ids = next((row[key] for key in ("allocated_device_ids", "allocated_npu_ids", "allocated_gpu_ids",
                                               "device_ids", "npu_ids", "gpu_ids") if row.get(key) is not None), None)
        if isinstance(device_ids, str):
            device_ids = [part.strip() for part in device_ids.split(",") if part.strip()]
        explicit_count = next((row[key] for key in ("allocated_npu_count", "allocated_gpu_count", "allocated_device_count")
                               if row.get(key) is not None), None)
        start = next((_time(row[key]) for key in ("allocation_started_at", "started_at", "start_time", "started")
                      if row.get(key) is not None), None)
        description = {"instance_id": instance_id, "host": host, "running_round": row.get("running_round"),
                       "allocation_started_at": start, "device_ids": None, "allocated_cards": None}
        if isinstance(device_ids, list) and device_ids and (host or instance_id):
            names = {str(value) for value in device_ids}
            identities = {(str(host or instance_id), value) for value in names}
            if len(names) != len(device_ids) or devices.intersection(identities):
                raise PlatformError("Duplicate official physical device identity in one job")
            devices.update(identities)
            description.update(device_ids=sorted(names), allocated_cards=len(names))
            count += len(names)
        elif explicit_count is not None and instance_id:
            description["allocated_cards"] = positive_int(explicit_count, "official allocated NPU count", 32)
            count += description["allocated_cards"]
        selected.append(description)
        if start is not None:
            starts.append(start)
    verified = bool(selected) and all(row["allocated_cards"] is not None for row in selected)
    return {"allocated_cards": count if verified else None, "allocation_verified": verified,
            "allocation_started_at": max(starts) if verified and len(starts) == len(selected) else None,
            "allocation_identities": selected}


def command_for(template, run_dir, worker_id, npu_count, selection, name):
    if "{run_dir}" not in template or "{worker_id}" not in template:
        raise PlatformError("Command template requires {run_dir} and {worker_id}")
    command = template.replace("{run_dir}", shlex.quote(str(run_dir))).replace("{worker_id}", shlex.quote(worker_id))
    command = command.replace("{npu_count}", str(npu_count)).replace("{gpu_count}", str(npu_count))
    command = command.replace("{workspace_id}", shlex.quote(selection["workspace_id"]))
    environment = {"JADE_QZ_WORKSPACE_ID": selection["workspace_id"], "JADE_QZ_GROUP_ID": selection["logic_compute_group_id"],
                   "JADE_QZ_JOB_NAME": name}
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in environment.items()) + " " + command


def _load(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def _registry_states(registry_path, current_path, current_state):
    registry = _load(registry_path, {"schema_version": 1, "fleets": []})
    if not isinstance(registry.get("fleets"), list):
        raise PlatformError("Ascend registry must contain a complete fleets list")
    paths = list(dict.fromkeys(str(Path(path).resolve()) for path in registry["fleets"]))
    if str(current_path) not in paths:
        paths.append(str(current_path))
    states = {}
    for path in paths:
        try:
            state = current_state if path == str(current_path) else json.loads(Path(path).read_text())
        except (OSError, ValueError):
            raise PlatformError(f"Registered Ascend fleet is unreadable: {path}") from None
        if not isinstance(state, dict) or not isinstance(state.get("jobs"), list):
            raise PlatformError("Registered Ascend fleet has incomplete reservations")
        by_record = {str(Path(entry["record_path"]).resolve()): entry for entry in state["jobs"]}
        for record_path in (Path(path).parent / "submissions").glob("*.json"):
            key = str(record_path.resolve())
            record = _load(record_path)
            if not isinstance(record, dict) or not record.get("request"):
                raise PlatformError("Orphan submission record is malformed")
            if key not in by_record:
                request = record["request"]
                selection = record.get("selection")
                if not isinstance(selection, dict):
                    raise PlatformError("Orphan submission has no verified immutable Ascend selection")
                validate_payload(request, selection)
                entry = {"record_path": key, "worker_id": record_path.stem, "payload": request,
                         "selection": selection,
                         "workspace_id": request["workspace_id"], "group_id": request["logic_compute_group_id"],
                         "status": record.get("status", "submission_unknown"), "job_id": job_id(record),
                         "recovered_orphan_record": True}
                state["jobs"].append(entry)
        states[path] = state
    write_json(registry_path, {"schema_version": 1, "fleets": paths})
    return states


def _write_job_mapping(fleet_dir, entry):
    """Publish the official job identity as soon as create/recovery resolves it."""
    if not entry.get("job_id"):
        return
    request = entry["payload"]
    mapping = {"schema_version": 1, "worker_prefix": entry["worker_id"],
        "workspace_id": request["workspace_id"], "project_id": request["project_id"],
        "job_id": entry["job_id"], "job_name": request["name"],
        "logic_compute_group_id": request["logic_compute_group_id"],
        "requested_cards": cards(request)}
    path = fleet_dir / "jobs" / (entry["worker_id"] + ".json")
    existing = _load(path)
    if existing is not None:
        if {key: value for key, value in existing.items() if key != "recorded_at"} != mapping:
            raise PlatformError("Worker prefix already maps to another official platform job")
        return
    write_json(path, dict(mapping, recorded_at=now_iso()))


def reservations(states, official_jobs):
    """Deduplicate by workspace/job, retaining all unknown POST reservations."""
    official = {(row["workspace_id"], row["job_id"]): row for row in official_jobs if row.get("job_id")}
    reserved = {}
    for path, state in states.items():
        for entry in state["jobs"]:
            request = entry.get("payload")
            record_path = Path(entry["record_path"])
            record = _load(record_path)
            if record is not None:
                if request is not None and record.get("request") != request:
                    raise PlatformError("State and durable submission request disagree")
                request = record.get("request")
                if entry.get("job_id") and job_id(record) and entry["job_id"] != job_id(record):
                    raise PlatformError("State and durable submission job IDs disagree")
            if not isinstance(request, dict):
                raise PlatformError("Reservation has no complete immutable request")
            jid = entry.get("job_id") or job_id(record or {})
            workspace = request["workspace_id"]
            current = official.get((workspace, jid))
            if current and current.get("status") in TERMINAL:
                continue
            if entry.get("official_terminal") is True:
                continue
            if (record or {}).get("status", entry.get("status")) == "submission_rejected":
                continue
            if entry.get("status") == "cancelled_before_submission" and record is None and not jid:
                continue
            count = cards(request)
            key = (workspace, jid) if jid else ("record", str(record_path.resolve()))
            value = {"workspace_id": workspace, "job_id": jid, "cards": count,
                     "group_id": request["logic_compute_group_id"], "fleet_paths": {path},
                     "status": (current or {}).get("status", entry.get("status")), "entry": entry}
            if key in reserved:
                previous = reserved[key]
                if (previous["cards"], previous["group_id"]) != (count, value["group_id"]):
                    raise PlatformError("Duplicate job identity has conflicting physical footprint")
                value["fleet_paths"].update(previous["fleet_paths"])
            reserved[key] = value
    return list(reserved.values())


def plan_placements(inventory, reserved, available_cards, group_backoff=None, now=None):
    """Prefer idle capacity and distribute tickets; shared group IDs count once."""
    if type(available_cards) is not int or not 0 <= available_cards <= 64:
        raise PlatformError("Available physical card budget must be in 0..64")
    now = time.time() if now is None else now
    group_backoff = group_backoff or {}
    groups = {}
    for group in inventory.get("groups", []):
        if group.get("resource_query", {}).get("status") != "fresh":
            continue
        current = {key: max(0.0, number(group.get("resources", {}).get(key))) for key in
                   ("gpu_available", "cpu_available", "memory_gi_available", "gpu_low_priority_used")}
        if group["id"] in groups:
            current = {key: min(value, groups[group["id"]][key]) for key, value in current.items()}
        groups[group["id"]] = current
    workspace_load, group_load = {}, {}
    for row in reserved:
        workspace_load[row["workspace_id"]] = workspace_load.get(row["workspace_id"], 0) + row["cards"]
        group_load[row["group_id"]] = group_load.get(row["group_id"], 0) + row["cards"]
        if row["status"] not in RUNNING and row["group_id"] in groups:
            spec = row["entry"]["payload"]["framework_config"][0]["resource_spec_price"]
            for resource, amount in (("gpu_available", row["cards"]), ("cpu_available", spec["cpu_count"]),
                                     ("memory_gi_available", spec["memory_size_gib"])):
                groups[row["group_id"]][resource] = max(0, groups[row["group_id"]][resource] - amount)
    selections = []
    for selection in inventory.get("selections", []):
        validate_selection(selection)
        if selection["logic_compute_group_id"] in groups:
            selections.append(selection)
    placements = []
    while available_cards:
        candidates = []
        for selection in selections:
            group_id, workspace = selection["logic_compute_group_id"], selection["workspace_id"]
            if number(group_backoff.get(group_id)) > now:
                continue
            spec, budget = selection["resource_spec_price"], groups[group_id]
            count = int(spec["gpu_count"])
            if count > available_cards:
                continue
            idle = (budget["gpu_available"] >= count and budget["cpu_available"] >= spec["cpu_count"]
                    and budget["memory_gi_available"] >= spec["memory_size_gib"])
            score = (int(idle), -workspace_load.get(workspace, 0),
                     budget["gpu_low_priority_used"] / (group_load.get(group_id, 0) + count),
                     -count, -spec["cpu_count"], workspace, group_id)
            candidates.append((score, selection, idle))
        if not candidates:
            break
        _, selection, idle = max(candidates, key=lambda row: row[0])
        spec, gid, workspace = selection["resource_spec_price"], selection["logic_compute_group_id"], selection["workspace_id"]
        count = int(spec["gpu_count"])
        placements.append({"selection": copy.deepcopy(selection), "cards": count,
                           "reason": "idle_gpu_cpu_memory" if idle else "queue_with_shared_budget"})
        for resource, amount in (("gpu_available", count), ("cpu_available", spec["cpu_count"]),
                                 ("memory_gi_available", spec["memory_size_gib"])):
            groups[gid][resource] = max(0, groups[gid][resource] - amount)
        available_cards -= count
        workspace_load[workspace] = workspace_load.get(workspace, 0) + count
        group_load[gid] = group_load.get(gid, 0) + count
    return placements


def refresh_inventory(client, inventory):
    result = copy.deepcopy(inventory)
    seen = {}
    errors = []
    for group in result.get("groups", []):
        gid = group["id"]
        workspace = group["workspace_id"]
        identity = (workspace, gid)
        if identity not in seen:
            try:
                resource = resource_summary(client.request("POST", API["resource"],
                    {"logic_compute_group_id": gid, "workspace_id": workspace}))
                if any(resource[key + "_available"] is None for key in ("gpu", "cpu", "memory_gi")):
                    raise PlatformError("Resource totals are incomplete")
                seen[identity] = (resource, {"status": "fresh", "observed_at": now_iso()})
            except (PlatformError, OSError, TimeoutError) as error:
                seen[identity] = (group.get("resources", {}), {"status": "error", "observed_at": now_iso(), "error": str(error)})
                errors.append({"workspace_id": workspace, "group_id": gid, "error": str(error)})
        group["resources"], group["resource_query"] = copy.deepcopy(seen[identity])
    result.update(at=now_iso(), resource_query_errors=errors)
    result.pop("evidence", None)
    return result


def _observe(client, entry, snapshot_id, observation_path):
    evidence = inspect_job(client, entry["job_id"])
    detail = evidence["detail"]
    request = entry["payload"]
    for key in ("workspace_id", "project_id", "logic_compute_group_id", "name", "command"):
        if detail.get(key) is not None and detail[key] != request[key]:
            raise PlatformError("Official job identity differs from its owned request")
    expected = entry["selection"]["priority"]
    priority_ok = detail.get("priority_level") == expected["level"] and str(detail.get("priority_name")) == str(expected["name"])
    record = {"schema_version": 1, "snapshot_id": snapshot_id, "at": now_iso(), "time": time.time(),
              "workspace_id": request["workspace_id"], "project_id": request["project_id"],
              "job_id": entry["job_id"], "worker_id": entry["worker_id"], "status": detail.get("status"),
              "priority_level": detail.get("priority_level"), "priority_name": detail.get("priority_name"),
              "priority_verified": priority_ok, "requested_cards": cards(request),
              "running_round": detail.get("current_running_round"),
              **allocation(evidence["instances"], detail.get("current_running_round")),
              "instances": evidence["instances"]}
    observation_path.parent.mkdir(parents=True, exist_ok=True)
    with observation_path.open("a") as stream:
        stream.write(json.dumps(public(record), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
    entry.update(status=detail.get("status", "unknown"), last_observation=record,
                 official_terminal=detail.get("status") in TERMINAL)
    return evidence, priority_ok


class IntentDeferred(RuntimeError):
    """An optional plan lost its authority before a durable intent/POST."""


def reconcile(client, run_dir, command_template, inventory_seed, card_limit=32, fill=False,
              registry_path=None, job_runtime_seconds=86400, retries=2, reset_failures=False,
              global_card_limit=32, placement_planner=None, state_planner=None,
              intent_hook=None, retire_predicate=None, isolate_candidate_failures=False, state_hook=None):
    # Larger experiments must explicitly opt in; existing callers retain the
    # historical 32-card ceiling across all registered fleets and spaces.
    positive_int(card_limit, "card_limit", 64)
    positive_int(global_card_limit, "global_card_limit", 64)
    run_dir = Path(run_dir).resolve()
    fleet_dir = run_dir / "ascend_fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    state_path = fleet_dir / "state.json"
    registry_path = Path(registry_path or DEFAULT_REGISTRY).expanduser().resolve()
    inventory = copy.deepcopy(inventory_seed) if isinstance(inventory_seed, dict) else _load(inventory_seed)
    if not isinstance(inventory, dict) or inventory.get("backend") != "ascend" or not inventory.get("user", {}).get("id"):
        raise PlatformError("A discovered authenticated Ascend inventory is required")
    snapshot_id = uuid.uuid4().hex
    with writer_lock(registry_path.parent / "ascend-fleet-global-lock"), writer_lock(fleet_dir / "reconcile-lock"):
        state = _load(state_path, {"schema_version": 1, "backend": "ascend", "run_dir": str(run_dir),
            "user_id": inventory["user"]["id"], "command_template": command_template, "next_worker": 1,
            "jobs": [], "failures": [], "group_backoff": {}})
        if state["command_template"] != command_template or state["user_id"] != inventory["user"]["id"]:
            raise PlatformError("Ascend fleet command and authenticated owner are immutable")
        if reset_failures:
            state.update(failures=[], group_backoff={}, blocked_reason=None)
        marker = next((name for name in MARKERS if (run_dir / name).exists()), None)
        write_json(state_path, state)
        states = _registry_states(registry_path, state_path, state)
        if state_hook is not None:
            state_hook(state)
            write_json(state_path, state)
        official, errors = [], []
        workspaces = tuple(dict.fromkeys([s["workspace_id"] for s in inventory.get("selections", [])]
            + [entry["payload"]["workspace_id"] for other in states.values() for entry in other["jobs"]]))
        for workspace_id in workspaces:
            try:
                official.extend(list_own_jobs(client, inventory["user"]["id"], (workspace_id,)))
            except (PlatformError, OSError, TimeoutError) as error:
                errors.append({"workspace_id": workspace_id, "error": str(error)})
        unreachable = {error["workspace_id"] for error in errors}
        now = time.time()
        for entry in state["jobs"]:
            if not active(entry):
                continue
            path = Path(entry["record_path"])
            record = _load(path)
            if record and not job_id(record) and record.get("status") != "submission_rejected":
                record = resolve_submission(record, official)
                write_json(path, record)
            if record:
                entry.update(job_id=job_id(record), status=record.get("status", entry["status"]))
                _write_job_mapping(fleet_dir, entry)
            if record and record.get("status") == "submission_rejected":
                continue
            if not entry.get("job_id") or entry["workspace_id"] in unreachable:
                continue
            try:
                evidence, priority_ok = _observe(client, entry, snapshot_id, fleet_dir / "observations.jsonl")
                write_json(fleet_dir / "status" / (entry["worker_id"] + ".json"), evidence)
            except (PlatformError, OSError, TimeoutError) as error:
                errors.append({"workspace_id": entry["workspace_id"], "job_id": entry["job_id"], "error": str(error)})
                entry["last_error"] = str(error)
                continue
            if not priority_ok:
                if isolate_candidate_failures and entry.get('refill_role') == 'candidate':
                    entry['refill_priority_error'] = True
                else:
                    state["blocked_reason"] = "An owned job is not at its verified LOW priority"
                if fill and not entry.get("stop_requested"):
                    client.request("POST", API["stop"], {"job_id": entry["job_id"]})
                    entry["stop_requested"] = now
            if entry.get("official_terminal") and not entry.get("terminal_at"):
                entry["terminal_at"] = now
                if entry["status"] != "job_succeeded" and not entry.get("stop_requested"):
                    events, _ = failed_job_evidence(client, entry["job_id"], entry.get("last_observation", {}).get("running_round"))
                    kind = failure_kind(entry["status"], events)
                    entry["failure_kind"] = kind
                    state["failures"].append({"at": now, "kind": kind, "group_id": entry["group_id"], "job_id": entry["job_id"],
                        "optional_candidate": bool(isolate_candidate_failures and entry.get("refill_role") == "candidate")})
                    write_json(fleet_dir / "events" / (entry["worker_id"] + ".json"), public(events))
                    previous = sum(row["group_id"] == entry["group_id"] and now - row["at"] < 3600 for row in state["failures"])
                    state["group_backoff"][entry["group_id"]] = now + min(1800, 30 * 2 ** min(previous - 1, 6))
        errors_recent = [row for row in state["failures"] if now - row["at"] < 900
                         and row["kind"] != "resource_or_runtime" and not row.get("optional_candidate")]
        if len(errors_recent) >= 3:
            state["blocked_reason"] = "Three software/capacity or unexplained failures in 15 minutes"
        if fill and not marker and retire_predicate is not None:
            for entry in state["jobs"]:
                if not active(entry) or not retire_predicate(entry):
                    continue
                if not entry.get("job_id") and not Path(entry["record_path"]).exists():
                    entry["status"] = "cancelled_before_submission"
                elif entry.get("job_id") and not entry.get("stop_requested") and entry["workspace_id"] not in unreachable:
                    client.request("POST", API["stop"], {"job_id": entry["job_id"]})
                    entry["stop_requested"] = now
                write_json(state_path, state)
        inventory = refresh_inventory(client, inventory) if not marker else inventory
        for group in inventory.get("groups", []):
            if group.get("workspace_id") in unreachable:
                group["resource_query"] = {"status": "error", "error": "workspace listing unavailable"}
        reserved = reservations(states, official)
        used = sum(row["cards"] for row in reserved)
        local = sum(row["cards"] for row in reserved if str(state_path) in row["fleet_paths"])
        available = min(max(0, global_card_limit - used), max(0, card_limit - local))
        planner = placement_planner or plan_placements
        if marker or state.get("blocked_reason"):
            placements = []
        elif state_planner is not None:
            placements = state_planner(state, inventory, reserved, available, state["group_backoff"], now)
        else:
            placements = planner(inventory, reserved, available, state["group_backoff"], now)
        if sum(p["cards"] for p in placements) > available:
            raise PlatformError("Placement plan exceeds the currently available physical-card budget")
        write_json(state_path, state)
        created = []
        if fill and not marker and not state.get("blocked_reason"):
            for placement in placements:
                selection = placement["selection"]
                index = state["next_worker"]
                worker_id = f"w{index:06d}"
                name = f"jade-npu-{hashlib.sha256(str(run_dir).encode()).hexdigest()[:8]}-{worker_id}"
                command = command_for(command_template, run_dir, worker_id, placement["cards"], selection, name)
                payload = render_payload(selection, command, name, job_runtime_seconds, retries)
                entry = {"worker_id": worker_id, "workspace_id": selection["workspace_id"],
                    "group_id": selection["logic_compute_group_id"], "selection": selection, "payload": payload,
                    "status": "planned", "created_at": now, "reason": placement["reason"],
                    "record_path": str(fleet_dir / "submissions" / (worker_id + ".json"))}
                if sum(row["cards"] for row in reservations(states, official)) + cards(payload) > global_card_limit:
                    raise PlatformError(f"New intent exceeds the global {global_card_limit} physical-card budget")
                if cards(payload) != placement["cards"]:
                    raise PlatformError("Placement size differs from its immutable payload")
                if intent_hook is not None:
                    # Publish immutable training identities before POST: a fast
                    # container must never derive them from a mutable layout.
                    try:
                        intent_hook(state, entry, placement)
                    except IntentDeferred:
                        break  # Optional candidate plan is safe to retry.
                state["next_worker"] += 1
                state["jobs"].append(entry)
                write_json(state_path, state)
            for entry in state["jobs"]:
                path = Path(entry["record_path"])
                if entry["status"] != "planned" or entry.get("job_id") or path.exists():
                    continue
                if retire_predicate is not None and retire_predicate(entry):
                    entry['status'] = 'cancelled_before_submission'
                    write_json(state_path, state)
                    continue
                if entry["workspace_id"] in unreachable or number(state["group_backoff"].get(entry["group_id"])) > now:
                    continue
                if (run_dir / "STOP").exists() or any((run_dir / name).exists() for name in ("FATAL.json", "COMPLETE.json")):
                    break
                try:
                    record = submit(client, entry["payload"], path, entry["selection"])
                    entry.update(job_id=job_id(record), status="submitted")
                    _write_job_mapping(fleet_dir, entry)
                    write_json(state_path, state)
                    evidence, priority_ok = _observe(client, entry, snapshot_id, fleet_dir / "observations.jsonl")
                    write_json(fleet_dir / "status" / (entry["worker_id"] + ".json"), evidence)
                    created.append({key: entry.get(key) for key in ("worker_id", "workspace_id", "group_id", "job_id", "status")})
                    if not priority_ok:
                        if isolate_candidate_failures and entry.get('refill_role') == 'candidate':
                            entry['refill_priority_error'] = True
                        else:
                            state["blocked_reason"] = "New owned job did not receive the verified LOW priority"
                        client.request("POST", API["stop"], {"job_id": entry["job_id"]})
                        entry["stop_requested"] = now
                        break
                except Exception as error:
                    record = _load(path, {})
                    entry.update(job_id=job_id(record) or entry.get("job_id"),
                                 status=record.get("status", entry["status"]), last_error=str(error))
                    errors.append({"workspace_id": entry["workspace_id"], "error": str(error)})
                    break
                finally:
                    write_json(state_path, state)
        if fill and marker:
            for entry in state["jobs"]:
                if not active(entry):
                    continue
                if not entry.get("job_id") and not Path(entry["record_path"]).exists():
                    entry["status"] = "cancelled_before_submission"
                elif entry.get("job_id") and not entry.get("stop_requested") and entry["workspace_id"] not in unreachable:
                    client.request("POST", API["stop"], {"job_id": entry["job_id"]})
                    entry["stop_requested"] = now
                write_json(state_path, state)
        write_json(state_path, state)
        reserved = reservations(states, official)
        observations = [entry["last_observation"] for entry in state["jobs"] if entry.get("last_observation", {}).get("snapshot_id") == snapshot_id]
        snapshot = {"schema_version": 1, "snapshot_id": snapshot_id, "at": now_iso(), "time": time.time(),
            "run_dir": str(run_dir), "card_limit": card_limit, "global_card_limit": global_card_limit, "fill": fill,
            "marker": marker, "blocked_reason": state.get("blocked_reason"),
            "requested_reserved_cards": sum(row["cards"] for row in reserved),
            "local_requested_reserved_cards": sum(row["cards"] for row in reserved if str(state_path) in row["fleet_paths"]),
            "verified_allocated_cards": sum(row["allocated_cards"] or 0 for row in observations if row["priority_verified"]),
            "allocation_observation_complete": bool(observations) and all(row["allocation_verified"] for row in observations),
            "unresolved_reservations": sum(row["job_id"] is None for row in reserved),
            "pending_jobs": sum(active(entry) and entry["status"] not in RUNNING for entry in state["jobs"]),
            "running_jobs": sum(active(entry) and entry["status"] in RUNNING for entry in state["jobs"]),
            "active_jobs": sum(active(entry) for entry in state["jobs"]),
            "errors": errors + inventory.get("resource_query_errors", []), "created": created,
            "planned": placements, "jobs": [{key: value for key, value in entry.items() if key not in ("payload", "selection")} for entry in state["jobs"]]}
        write_json(fleet_dir / "inventory.json", public(inventory))
        write_json(fleet_dir / "latest.json", public(snapshot))
        return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("reconcile", "watch"))
    parser.add_argument("--credentials")
    parser.add_argument("--cookie-file")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--command-template", required=True)
    parser.add_argument("--card-limit", type=int, default=32)
    parser.add_argument("--global-card-limit", type=int, default=32,
                        help="Explicit shared ceiling across registered fleets (maximum 64)")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--fill", action="store_true")
    parser.add_argument("--reset-failures", action="store_true")
    parser.add_argument("--job-runtime-seconds", type=int, default=86400)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--watch-seconds", type=int, default=86400,
                        help="Finite controller lifetime; resuming preserves all identities")
    args = parser.parse_args(argv)
    if not 10 <= args.interval_seconds <= 600 or not 1 <= args.watch_seconds <= 7 * 86400:
        parser.error("interval must be 10..600 seconds and watch lifetime 1 second..7 days")
    deadline = time.monotonic() + args.watch_seconds
    passes = 0
    try:
        client = make_client(args.credentials, args.cookie_file)
    except (PlatformError, OSError, ValueError) as error:
        print(f"ascend_fleet: {error}", file=sys.stderr)
        return 1
    while True:
        try:
            snapshot = reconcile(client, args.run_dir, args.command_template, args.inventory,
                args.card_limit, args.fill, args.registry, args.job_runtime_seconds, args.retries,
                args.reset_failures and passes == 0, global_card_limit=args.global_card_limit)
            print(json.dumps({key: snapshot[key] for key in ("at", "requested_reserved_cards", "verified_allocated_cards",
                "running_jobs", "pending_jobs", "marker", "blocked_reason", "created")}), flush=True)
            if args.action == "reconcile":
                return 2 if snapshot["blocked_reason"] else 0
            if snapshot["marker"] and snapshot["active_jobs"] == 0:
                return 0
        except (PlatformError, OSError, ValueError, RuntimeError) as error:
            diagnostic = {"at": now_iso(), "error": str(error)}
            write_json(args.run_dir / "ascend_fleet/monitor_error.json", diagnostic)
            print(json.dumps(diagnostic), file=sys.stderr, flush=True)
            if args.action == "reconcile":
                return 1
        passes += 1
        if time.monotonic() >= deadline:
            write_json(args.run_dir / "ascend_fleet/WATCH_EXPIRED.json", {"at": now_iso(), "passes": passes,
                "note": "Controller reached its finite deadline; owned task reservations remain in state.json"})
            return 3
        client.evidence.clear()
        time.sleep(min(args.interval_seconds, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    raise SystemExit(main())
