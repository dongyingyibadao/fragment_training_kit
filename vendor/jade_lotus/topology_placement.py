"""Budgeted 16/8/4/2/1 QZ placement intentions, never allocation claims."""
from __future__ import annotations

import copy
import math
import time

from jade_lotus.ascend_nodes import (DEFAULT_MAX_AGE_SECONDS, NODE_API, NODE_SCHEMA,
    SIZES, fresh, iso_time, timestamp)
from jade_lotus.ascend_platform import RUNNING, npu_type, positive_int, validate_selection
from jade_lotus.platform import PlatformError


def _finite(value, name, *, minimum=0):
    if isinstance(value, bool):
        raise PlatformError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise PlatformError(f"{name} must be a finite number") from None
    if not math.isfinite(result) or result < minimum:
        raise PlatformError(f"{name} must be a finite number >= {minimum}")
    return result


def _limit(value, name):
    if type(value) is not int or not 0 <= value <= 64:
        raise PlatformError(f"{name} must be an integer in 0..64")
    return value


def _footprint(row):
    entry = row.get("entry") or {}
    payload = row.get("payload") or entry.get("payload")
    explicit = row.get("cards", row.get("requested_cards", row.get("reserved_cards")))
    if explicit is not None:
        explicit = positive_int(explicit, "reservation cards", 64)
    if payload is not None:
        configs = payload.get("framework_config")
        if not isinstance(configs, list) or not configs:
            raise PlatformError("Reservation payload has no complete framework_config")
        count, cpu, memory, types = 0, 0, 0, set()
        for component in configs:
            instances = positive_int(component.get("instance_count"), "reservation instance_count")
            npus = positive_int(component.get("gpu_count"), "reservation gpu_count", 32)
            spec = component.get("resource_spec_price") or component.get("instance_spec_price_info") or {}
            if spec.get("gpu_count") is not None and positive_int(spec["gpu_count"], "spec gpu_count", 32) != npus:
                raise PlatformError("Reservation has conflicting physical card counts")
            count += instances * npus
            cpu += instances * positive_int(spec.get("cpu_count"), "reservation cpu_count")
            memory += instances * positive_int(spec.get("memory_size_gib"), "reservation memory_size_gib")
            types.add(npu_type(spec))
        if explicit is not None and count != explicit:
            raise PlatformError("Reservation cards disagree with its immutable payload")
        if count > 64 or len(types) != 1 or None in types:
            raise PlatformError("Reservation is not a bounded homogeneous Ascend request")
        return count, {"cpus": cpu, "memory_gib": memory, "npu_type": next(iter(types))}
    selection = row.get("selection") or entry.get("selection")
    if selection is not None:
        validate_selection(selection)
        spec = selection["resource_spec_price"]
        count = int(spec["gpu_count"])
        if explicit is not None and explicit != count:
            raise PlatformError("Reservation cards disagree with its immutable selection")
        return count, {"cpus": int(spec["cpu_count"]), "memory_gib": int(spec["memory_size_gib"]),
                       "npu_type": npu_type(spec)}
    if explicit is None:
        raise PlatformError("Every reservation must retain its physical card footprint")
    return explicit, None


def reservation_ledger(reservations):
    """Deduplicate durable jobs/intents; ambiguous or local terminal rows count.

    A reservation can have category active, candidate or hot_spare. An official
    terminal receipt releases it. A rejected or cancelled unsent intent can be
    released without a remote receipt only while it has no official job ID.
    """
    records, aliases = [], {}
    for index, row in enumerate(reservations):
        if not isinstance(row, dict):
            raise PlatformError("A reservation must be an object")
        entry = row.get("entry") or {}
        jid = row.get("job_id") or entry.get("job_id")
        status = row.get("status", entry.get("status", "unknown"))
        if row.get("official_terminal") is True or entry.get("official_terminal") is True:
            continue
        if not jid and status in {"submission_rejected", "cancelled_before_submission"}:
            continue
        payload = row.get("payload") or entry.get("payload") or {}
        selection = row.get("selection") or entry.get("selection") or {}
        workspace = row.get("workspace_id") or payload.get("workspace_id") or selection.get("workspace_id")
        group = row.get("group_id") or row.get("logic_compute_group_id") or payload.get("logic_compute_group_id") or selection.get("logic_compute_group_id")
        count, footprint = _footprint(row)
        category = row.get("category", row.get("role", entry.get("category", "active")))
        if category not in {"active", "candidate", "hot_spare"}:
            category = "active"
        identities = []
        if jid:
            if not workspace:
                raise PlatformError("An official job reservation must retain its workspace")
            identities.append(("job", workspace, str(jid)))
        record_path = row.get("record_path") or entry.get("record_path")
        if record_path:
            identities.append(("record", str(record_path)))
        intent = row.get("reservation_id") or row.get("intent_id")
        if intent:
            identities.append(("intent", str(intent)))
        if not identities:
            identities.append(("anonymous", index))
        matches = {aliases[key] for key in identities if key in aliases}
        current = {"cards": count, "workspace_id": workspace, "group_id": group, "job_id": jid,
            "category": category, "status": status, "footprint": footprint, "local": row.get("local", True) is True,
            "pending": status not in RUNNING, "identities": identities}
        for match in sorted(matches):
            previous = records[match]
            if previous is None:
                continue
            if (previous["cards"], previous["workspace_id"], previous["group_id"]) != (count, workspace, group):
                raise PlatformError("Duplicate reservation identity has conflicting physical footprint")
            if (previous["footprint"] is not None and current["footprint"] is not None
                    and previous["footprint"] != current["footprint"]):
                raise PlatformError("Duplicate reservation has conflicting CPU/memory/device resources")
            current["footprint"] = current["footprint"] or previous["footprint"]
            current["pending"] = current["pending"] or previous["pending"]
            current["local"] = current["local"] or previous["local"]
            current["job_id"] = current["job_id"] or previous["job_id"]
            current["identities"].extend(previous["identities"])
            if previous["category"] == "active":
                current["category"] = "active"
            records[match] = None
        position = len(records)
        records.append(current)
        for key in current["identities"]:
            aliases[key] = position
    return [row for row in records if row is not None]


def _node_capacity(snapshot, now, max_age_seconds, diagnostics):
    result = {}
    if not fresh(snapshot.get("observed_at"), now, max_age_seconds):
        diagnostics.append({"reason": "stale_or_unknown_node_snapshot", "idle_capacity_used": False})
        return result
    for node in snapshot.get("nodes", []):
        if node.get("eligible") is not True or not fresh(node.get("observed_at"), now, max_age_seconds):
            continue
        observations = node.get("observations")
        if (not isinstance(observations, list) or not observations or any(
                not fresh(row.get("observed_at"), now, max_age_seconds)
                or row.get("query_status") != "fresh" or row.get("query_complete") is not True
                or (row.get("source_fields") or {}).get("status") != "Ready"
                or (row.get("source_fields") or {}).get("resource_pool") != "online"
                for row in observations)):
            continue
        nid = node.get("node_id")
        if not nid or not node.get("name") or not node.get("npu_type"):
            continue
        if nid in result:
            raise PlatformError("Duplicate physical node in normalized snapshot")
        try:
            cards = _finite(node.get("available_npus"), "node available NPUs")
            cpus = _finite(node.get("available_cpus"), "node available CPUs")
            memory = _finite(node.get("available_memory_gib"), "node available memory")
            total = _finite(node.get("total_npus"), "node total NPUs")
        except PlatformError:
            continue
        if int(cards) != cards or int(total) != total or cards > total:
            continue
        result[nid] = {"node": node, "cards": int(cards), "cpus": cpus, "memory_gib": memory,
            "memberships": {(m.get("workspace_id"), m.get("logic_compute_group_id"))
                for m in node.get("memberships", []) if m.get("query_status") == "fresh"
                and m.get("query_complete") is True and fresh(m.get("observed_at"), now, max_age_seconds)}}
    return result


def plan_topology(inventory, node_snapshot, reservations=(), *, global_card_limit,
                  local_card_limit=None, now=None, max_age_seconds=DEFAULT_MAX_AGE_SECONDS,
                  group_backoff=None, queue_when_idle_exhausted=True):
    """Fill one shared card budget, choosing fullest feasible physical nodes.

    Every selection is an existing official single-job specification. An idle
    node can support one or more requests subject to CPU and memory deductions.
    Pending requests also consume group capacity; missing footprints block an
    idle-capacity assertion. Remaining budget may use LOW single-card queue
    intentions. No result asserts that a node or a card was allocated.
    """
    global_card_limit = _limit(global_card_limit, "global_card_limit")
    local_card_limit = global_card_limit if local_card_limit is None else _limit(local_card_limit, "local_card_limit")
    now = time.time() if now is None else timestamp(now)
    if now is None:
        raise PlatformError("A finite current time is required")
    max_age_seconds = _finite(max_age_seconds, "max_age_seconds", minimum=1e-9)
    if inventory.get("backend") != "ascend":
        raise PlatformError("Topology planning requires an Ascend inventory")
    if node_snapshot.get("schema") != NODE_SCHEMA or node_snapshot.get("source") != NODE_API:
        raise PlatformError("Topology planning requires a normalized official QZ node snapshot")
    if type(queue_when_idle_exhausted) is not bool:
        raise PlatformError("queue_when_idle_exhausted must be boolean")
    diagnostics, selections = [], []
    if node_snapshot.get("errors"):
        diagnostics.append({"reason": "node_discovery_incomplete",
            "errors": copy.deepcopy(node_snapshot["errors"]),
            "idle_capacity_requires_complete_fresh_queries": True})
    for selection in inventory.get("selections", []):
        validate_selection(selection)
        if selection["resource_spec_price"]["gpu_count"] in SIZES:
            selections.append(selection)
        else:
            diagnostics.append({"reason": "unsupported_job_size", "cards": selection["resource_spec_price"]["gpu_count"]})
    ledger = reservation_ledger(reservations)
    used = sum(row["cards"] for row in ledger)
    local_used = sum(row["cards"] for row in ledger if row["local"])
    available = min(max(0, global_card_limit - used), max(0, local_card_limit - local_used))
    original_available = available
    capacity = _node_capacity(node_snapshot, now, max_age_seconds, diagnostics)
    workspace_load, holds, blocked_groups = {}, {}, set()
    unknown_group_hold = False
    for row in ledger:
        workspace = row["workspace_id"]
        workspace_load[workspace] = workspace_load.get(workspace, 0) + row["cards"]
        if not row["pending"]:
            continue
        group, footprint = row["group_id"], row["footprint"]
        if not group:
            unknown_group_hold = True
        elif footprint is None:
            blocked_groups.add(group)
        else:
            key = (group, footprint["npu_type"])
            hold = holds.setdefault(key, {"cards": 0, "cpus": 0, "memory_gib": 0})
            hold["cards"] += row["cards"]
            hold["cpus"] += footprint["cpus"]
            hold["memory_gib"] += footprint["memory_gib"]
    if blocked_groups or unknown_group_hold:
        diagnostics.append({"reason": "pending_reservation_has_unknown_resource_footprint",
            "group_ids": sorted(blocked_groups), "unknown_group": unknown_group_hold,
            "idle_capacity_assertion_blocked": True})
    backoff = group_backoff or {}

    def allowed(selection):
        until = timestamp(backoff.get(selection["logic_compute_group_id"]))
        return until is None or until <= now

    def group_free(group, device):
        members = [value for value in capacity.values() if value["node"]["npu_type"] == device
            and any(gid == group for _, gid in value["memberships"])]
        hold = holds.get((group, device), {})
        return {resource: sum(row[resource] for row in members) - hold.get(resource, 0)
                for resource in ("cards", "cpus", "memory_gib")}

    placements = []
    while available:
        candidates = []
        for selection in selections:
            group, workspace = selection["logic_compute_group_id"], selection["workspace_id"]
            if not allowed(selection) or unknown_group_hold or group in blocked_groups:
                continue
            spec = selection["resource_spec_price"]
            count, cpu, memory = (int(spec[key]) for key in ("gpu_count", "cpu_count", "memory_size_gib"))
            device = npu_type(spec)
            if count > available:
                continue
            free = group_free(group, device)
            if free["cards"] < count or free["cpus"] < cpu or free["memory_gib"] < memory:
                continue
            excluded = set(selection.get("exclude_nodes", []))
            for nid, current in capacity.items():
                node = current["node"]
                if ((workspace, group) not in current["memberships"] or node["npu_type"] != device
                        or node["name"] in excluded or current["cards"] < count
                        or current["cpus"] < cpu or current["memory_gib"] < memory):
                    continue
                score = (-current["cards"], -count, cpu, memory,
                         workspace_load.get(workspace, 0), workspace, group, str(nid))
                candidates.append((score, selection, nid))
        if not candidates:
            break
        _, selection, nid = min(candidates, key=lambda value: value[0])
        current, spec = capacity[nid], selection["resource_spec_price"]
        node = current["node"]
        count, cpu, memory = (int(spec[key]) for key in ("gpu_count", "cpu_count", "memory_size_gib"))
        workspace = selection["workspace_id"]
        capability = (node_snapshot.get("capabilities") or {}).get(workspace, {})
        placements.append({"selection": copy.deepcopy(selection), "cards": count,
            "status": "planned_intent", "reason": "idle_whole_node" if count == node["total_npus"]
                and current["cards"] == node["total_npus"] else "idle_fragment",
            "expected_node_id": nid, "expected_node_name": node["name"],
            "node_observed_at": node["observed_at"],
            "available_before_intent": {key: current[key] for key in ("cards", "cpus", "memory_gib")},
            "binding": {"field": "specified_nodes", "enabled": capability.get("enabled"),
                "status": "unbound", "request_fields": {},
                "reason": "workspace_disabled" if capability.get("enabled") is False else
                    "requires_explicit_verified_renderer" if capability.get("enabled") is True else "capability_unknown"},
            "allocation_verified": False, "requires_postallocation_verification": True})
        current["cards"] -= count
        current["cpus"] -= cpu
        current["memory_gib"] -= memory
        available -= count
        workspace_load[workspace] = workspace_load.get(workspace, 0) + count
    queued = [selection for selection in selections if selection["resource_spec_price"]["gpu_count"] == 1
              and allowed(selection)]
    if queue_when_idle_exhausted:
        while available and queued:
            selection = min(queued, key=lambda s: (workspace_load.get(s["workspace_id"], 0),
                s["resource_spec_price"]["cpu_count"], s["resource_spec_price"]["memory_size_gib"],
                s["workspace_id"], s["logic_compute_group_id"]))
            placements.append({"selection": copy.deepcopy(selection), "cards": 1,
                "status": "planned_intent", "reason": "queue_low_priority_single_card",
                "expected_node_id": None, "expected_node_name": None,
                "binding": {"status": "unbound", "request_fields": {}},
                "allocation_verified": False, "requires_postallocation_verification": True})
            workspace = selection["workspace_id"]
            workspace_load[workspace] = workspace_load.get(workspace, 0) + 1
            available -= 1
        if available and not queued:
            diagnostics.append({"reason": "no_legal_single_card_queue_specification", "unplanned_cards": available})
    categories = {category: sum(row["cards"] for row in ledger if row["category"] == category)
                  for category in ("active", "candidate", "hot_spare")}
    return {"schema_version": 1, "backend": "ascend", "status": "planned_intentions",
        "observed_at": iso_time(now), "node_snapshot_at": node_snapshot.get("observed_at"),
        "placements": placements, "job_layout": [row["cards"] for row in placements],
        "budget": {"global_card_limit": global_card_limit, "local_card_limit": local_card_limit,
            "reserved_cards": used, "local_reserved_cards": local_used, "reserved_by_category": categories,
            "available_before_plan": original_available, "planned_cards": original_available - available,
            "remaining_cards": available, "total_reserved_and_planned": used + original_available - available,
            "already_over_limit": used > global_card_limit or local_used > local_card_limit},
        "diagnostics": diagnostics, "allocation_verified": False, "submitted_jobs": 0,
        "requires_postallocation_verification": True}
