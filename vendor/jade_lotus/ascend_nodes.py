"""Read-only, auditable QZ node inventory for topology planning.

The official space overview and training node selector use ListNodeDimension,
not the administrator's cluster ListNodes response. The request scopes the
workspace and logic group; the returned rows do not repeat those scopes.
``specified_nodes`` is the official training form field, gated by
``train_enable_specified_nodes``. Discovery never probes CreateJobConsole.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import re
import time

from jade_lotus.ascend_platform import API, WORKSPACES, npu_type
from jade_lotus.platform import PlatformError


NODE_API = "/api/v2/workspace?Action=ListNodeDimension"
NODE_SCHEMA = "qz-node-dimension-v1"
DEFAULT_MAX_AGE_SECONDS = 120
SIZES = (16, 8, 4, 2, 1)
_NAME = re.compile(r"[A-Za-z0-9_.-]{1,253}")


def timestamp(value):
    """Require a finite epoch or a timezone-bearing ISO timestamp."""
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        try:
            parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed.timestamp() if parsed.tzinfo else None
        except (TypeError, ValueError, OverflowError):
            return None


def iso_time(value):
    return dt.datetime.fromtimestamp(value, dt.timezone.utc).isoformat()


def fresh(value, now, max_age_seconds=DEFAULT_MAX_AGE_SECONDS):
    stamp = timestamp(value)
    return stamp is not None and -5 <= now - stamp <= max_age_seconds


def _number(value, integer=False):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or (integer and int(number) != number):
        return None
    return int(number) if integer else number


def _duration(value):
    number = _number(value)
    if number is None or number <= 0:
        raise PlatformError("Node snapshot lifetime must be finite and positive")
    return number


def _resource(row, name, integer=False):
    raw = row.get(name)
    if not isinstance(raw, dict):
        return None, None
    available, total = (_number(raw.get(key), integer) for key in ("available", "total"))
    if available is None or total is None or available > total:
        return None, total
    return available, total


def _memory_gib(raw, available, total, confirmed_unit):
    if available is None or total is None:
        return None, "missing_or_invalid"
    unit = str((raw or {}).get("unit", "")).casefold()
    if confirmed_unit == "GiB_crosschecked_group_total" or unit in ("gib", "gi"):
        return available, confirmed_unit or "explicit_GiB"
    if unit in ("byte", "bytes", "b"):
        return available / 2 ** 30, "explicit_bytes"
    if unit in ("gb", "g", ""):
        # The official overview labels this column GB, but many responses leave
        # unit empty. Until a complete group total matches memory_gi_total, use
        # the smaller decimal-GB interpretation. This cannot overstate either
        # GB or GiB capacity and records the deliberate conservative conversion.
        return available * 10 ** 9 / 2 ** 30, "GB_lower_bound"
    return None, "unknown_unit"


def binding_capabilities(inventory):
    result = {}
    for workspace in inventory.get("workspaces", []):
        wid = workspace.get("workspace_id")
        if wid not in WORKSPACES:
            raise PlatformError("Node capabilities are outside authorized Ascend workspaces")
        flag = (workspace.get("schedule") or {}).get("train_enable_specified_nodes")
        result[wid] = {"field": "specified_nodes", "official_web_form_verified": True,
            "enabled": flag if type(flag) is bool else None,
            "create_acceptance_verified": False,
            "status": "enabled" if flag is True else "disabled" if flag is False else "unknown",
            "source": "GetTrainScheduleConfig.train_enable_specified_nodes"}
    return result


def normalize_node_observations(observations, *, now=None,
                                max_age_seconds=DEFAULT_MAX_AGE_SECONDS):
    """Deduplicate physical node IDs, retaining the least optimistic observation.

    Each observation contains ``node`` (an official row), request scope,
    observed_at, query_status and query_complete. A missing status, incomplete
    page sequence, old timestamp or unknown resource never becomes Ready.
    """
    now = time.time() if now is None else timestamp(now)
    if now is None:
        raise PlatformError("A finite current time is required")
    max_age_seconds = _duration(max_age_seconds)
    grouped, host_ids = {}, {}
    for index, observation in enumerate(observations):
        row = observation.get("node")
        if not isinstance(row, dict):
            raise PlatformError("Node observation must contain an official node object")
        node_id, name = row.get("id"), row.get("name")
        if not isinstance(node_id, str) or not node_id:
            key = "unknown:" + str(index)
        else:
            key = "node:" + node_id
        node = grouped.setdefault(key, {"node_id": node_id, "name": name,
            "cluster_name": row.get("cluster_name"), "memberships": [], "observations": [],
            "available_npus": None, "total_npus": None, "available_cpus": None,
            "available_memory_gib": None, "npu_type": None, "unavailable_reasons": []})
        reasons = []
        if key.startswith("unknown:"):
            reasons.append("missing_physical_node_id")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            reasons.append("missing_or_invalid_node_name")
        if (node["name"], node["cluster_name"]) != (name, row.get("cluster_name")):
            reasons.append("conflicting_node_identity")
        if node_id and name and row.get("cluster_name"):
            host_ids.setdefault((row["cluster_name"], name), set()).add(key)
        if row.get("status") != "Ready":
            reasons.append("not_explicitly_Ready")
        if row.get("resource_pool") != "online":
            reasons.append("not_explicitly_online")
        if observation.get("query_status") != "fresh" or observation.get("query_complete") is not True:
            reasons.append("incomplete_or_failed_query")
        if not fresh(observation.get("observed_at"), now, max_age_seconds):
            reasons.append("stale_or_unknown_observation_time")
        workspace, group = observation.get("workspace_id"), observation.get("logic_compute_group_id")
        if workspace not in WORKSPACES or not isinstance(group, str) or not group:
            reasons.append("unknown_or_unauthorized_membership")
        device = npu_type(row)
        if not device:
            reasons.append("not_explicit_Ascend_type")
        if node["npu_type"] not in (None, device):
            reasons.append("conflicting_device_type")
        node["npu_type"] = device
        npus, total_npus = _resource(row, "gpu", integer=True)
        cpus, _ = _resource(row, "cpu")
        memory, memory_total = _resource(row, "memory")
        memory, memory_policy = _memory_gib(row.get("memory"), memory, memory_total,
                                          observation.get("memory_unit"))
        values = {"available_npus": npus, "total_npus": total_npus,
                  "available_cpus": cpus, "available_memory_gib": memory}
        if any(value is None for value in values.values()):
            reasons.append("unknown_or_invalid_free_resources")
        if total_npus == 0:
            reasons.append("no_physical_NPU_capacity")
        previous_observations = bool(node["observations"])
        for field, value in values.items():
            previous = node[field]
            node[field] = (min(previous, value) if previous is not None and value is not None
                           else None if previous_observations else value)
        source = {key: copy.deepcopy(row.get(key)) for key in (
            "id", "name", "cluster_name", "status", "resource_pool", "cordon_type", "gpu_type",
            "gpu_info", "cpu", "memory", "gpu")}
        membership = {"workspace_id": workspace, "logic_compute_group_id": group,
            "observed_at": observation.get("observed_at"), "query_status": observation.get("query_status"),
            "query_complete": observation.get("query_complete") is True}
        node["memberships"].append(membership)
        node["observations"].append(dict(membership, source_fields=source, memory_policy=memory_policy,
            row_sha256=hashlib.sha256(json.dumps(row, sort_keys=True, allow_nan=False).encode()).hexdigest()))
        node["unavailable_reasons"].extend(reasons)
    for identities in host_ids.values():
        if len(identities) > 1:
            for key in identities:
                grouped[key]["unavailable_reasons"].append("conflicting_physical_ids_for_same_host")
    result = []
    for node in grouped.values():
        node["unavailable_reasons"] = sorted(set(node["unavailable_reasons"]))
        node["eligible"] = not node["unavailable_reasons"]
        node["ready"] = node["eligible"]
        node["observed_at"] = min((row.get("observed_at") for row in node["observations"]
            if timestamp(row.get("observed_at")) is not None), key=timestamp, default=None)
        result.append(node)
    return sorted(result, key=lambda node: (-(node["available_npus"] or 0), str(node["node_id"])))


def _memory_unit(rows, group):
    expected = _number((group.get("resources") or {}).get("memory_gi_total"))
    totals = [_number((row.get("memory") or {}).get("total")) for row in rows]
    if expected is not None and rows and all(value is not None for value in totals):
        # ListNodeDimension rounds to two decimals; the resource API retains
        # fractions. Compare complete totals with a per-node rounding bound.
        if abs(sum(totals) - expected) <= max(0.02, len(rows) * 0.011):
            return "GiB_crosschecked_group_total"
    return None


def discover_nodes(client, inventory, *, max_age_seconds=DEFAULT_MAX_AGE_SECONDS,
                   page_size=100, max_pages=100, now=None):
    """Read every authorized group with complete pagination; never create jobs.

    The caller supplies an already discovered project inventory. The client
    must allow NODE_API as a read-only action in ascend_platform.API.
    """
    max_age_seconds = _duration(max_age_seconds)
    if type(page_size) is not int or not 1 <= page_size <= 1000:
        raise PlatformError("Node page size must be in 1..1000")
    if type(max_pages) is not int or not 1 <= max_pages <= 1000:
        raise PlatformError("Node page limit must be in 1..1000")
    if inventory.get("backend") != "ascend":
        raise PlatformError("An authenticated Ascend inventory is required for node discovery")
    clock = time.time if now is None else lambda: timestamp(now)
    if clock() is None:
        raise PlatformError("A finite current time is required")
    scopes = {}
    for group in inventory.get("groups", []):
        workspace, gid = group.get("workspace_id"), group.get("id")
        if workspace not in WORKSPACES or not isinstance(gid, str) or not gid:
            raise PlatformError("Node queries require authorized workspace and group identities")
        scopes[(workspace, gid)] = group
    for selection in inventory.get("selections", []):
        workspace, gid = selection.get("workspace_id"), selection.get("logic_compute_group_id")
        if workspace not in WORKSPACES or not isinstance(gid, str) or not gid:
            raise PlatformError("Node queries require authorized workspace and group identities")
        scopes.setdefault((workspace, gid), {})
    observations, queries, errors = [], [], []
    for (workspace, gid), group in sorted(scopes.items()):
        query = {"workspace_id": workspace, "logic_compute_group_id": gid, "api": NODE_API,
                 "started_at": iso_time(clock()), "status": "error", "complete": False, "pages": []}
        rows, total, seen = [], None, set()
        try:
            for page in range(1, max_pages + 1):
                body = {"page_num": page, "page_size": page_size, "filter": {
                    "workspace_id": workspace, "logic_compute_group_id": gid, "resource_pool": "online"}}
                response = client.request("POST", NODE_API, body)
                data = response.get("data")
                if not isinstance(data, dict) or not isinstance(data.get("node_dimensions"), list):
                    raise PlatformError("Node API did not return node_dimensions")
                count = _number(data.get("total"), integer=True)
                if count is None or (total is not None and total != count):
                    raise PlatformError("Node API total is missing or changed during pagination")
                total = count
                batch = data["node_dimensions"]
                if len(batch) > page_size or any(not isinstance(row, dict) for row in batch):
                    raise PlatformError("Node API returned a malformed page")
                for row in batch:
                    identity = row.get("id")
                    if identity is not None and identity in seen:
                        raise PlatformError("Node API repeated a physical node across pages")
                    if identity is not None:
                        seen.add(identity)
                rows.extend(copy.deepcopy(batch))
                query["pages"].append({"request": body, "rows": len(batch), "total": total,
                    "observed_at": iso_time(clock()), "response_metadata": copy.deepcopy(response.get("metadata"))})
                if len(rows) == total:
                    query.update(status="fresh", complete=True)
                    break
                if len(rows) > total or len(batch) < page_size:
                    raise PlatformError("Node API page sequence is incomplete")
            else:
                raise PlatformError("Node pagination exceeded its explicit limit")
        except (PlatformError, OSError, TimeoutError) as error:
            query["error"] = str(error)
            errors.append({"workspace_id": workspace, "logic_compute_group_id": gid, "error": str(error)})
        query.update(observed_at=iso_time(clock()), rows=len(rows), total=total)
        query["memory_unit"] = _memory_unit(rows, group) if query["complete"] else None
        queries.append(query)
        for row in rows:
            observations.append({"node": row, "workspace_id": workspace, "logic_compute_group_id": gid,
                "observed_at": query["started_at"], "query_status": query["status"],
                "query_complete": query["complete"], "memory_unit": query["memory_unit"]})
    current = clock()
    return {"schema_version": 1, "schema": NODE_SCHEMA, "backend": "ascend", "source": NODE_API,
        "observed_at": iso_time(current), "max_age_seconds": max_age_seconds,
        "nodes": normalize_node_observations(observations, now=current, max_age_seconds=max_age_seconds),
        "queries": queries, "capabilities": binding_capabilities(inventory), "errors": errors,
        "submitted_jobs": 0, "allocation_verified": False}
