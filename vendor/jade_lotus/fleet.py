"""Bounded LOW/1 H100/H200 worker fleet, with durable submission identities.

``reconcile`` plans by default; ``--fill`` authorizes the returned placements.
``watch --fill`` reconciles on a timer without spending model tokens polling.
Only jobs whose submission intents are in this run's fleet directory are owned.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import time

from jade_lotus.jobs import inspect_job, job_id, submit
from jade_lotus.platform import DEFAULT_WORKSPACE, InspireClient, OAuthCredentials, PlatformError, discover, items, render_payload, resource_summary, write_json
from jade_lotus.runtime import writer_lock


TERMINAL = {"job_succeeded", "job_failed", "job_stopped", "job_completed", "job_finished", "job_deleted", "job_cancelled", "job_canceled", "submission_rejected"}
RUNNING = {"job_running", "running"}
COMPLETION_MARKERS = ("COMPLETE.json", "FATAL.json", "STOP")
REGISTRY_PATH = Path(os.environ.get("JADE_FLEET_REGISTRY_PATH", str(Path(__file__).resolve().parents[1] / "artifacts" / "fleet-registry.json"))).expanduser().resolve()


def active(entry):
    # Unknown states and unresolved POSTs reserve capacity conservatively.
    return entry.get("status") not in TERMINAL


def pending(entry):
    return active(entry) and entry.get("status") not in RUNNING


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def costs(selection):
    spec = selection["resource_spec_price"]
    return {"gpu": int(spec["gpu_count"]), "cpu": number(spec["cpu_count"]), "memory_gi": number(spec["memory_size_gib"])}


def gpu_family(spec):
    """Read an explicit model token; opaque, missing or ambiguous types stay unknown."""
    # Discovery keeps gpu_type in gpu_info; render_payload uses the same
    # precedence to flatten it into the durable job's resource_spec_price.
    gpu_type = spec.get("gpu_type") or (spec.get("gpu_info") or {}).get("gpu_type")
    if not isinstance(gpu_type, str):
        return None
    models = set(re.findall(r"(?<![A-Z0-9])(H100|H200)(?![A-Z0-9])", gpu_type.upper()))
    return next(iter(models)) if len(models) == 1 else None



class AccountingError(PlatformError):
    """Incomplete or inconsistent durable evidence: refuse new planning."""


def positive_integer(value, label):
    # Accept integral JSON numbers, but never bools, fractions, NaN or infinity.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AccountingError(f"{label} must be an explicit positive integer")
    try:
        integer = int(value)
    except (ValueError, OverflowError):
        raise AccountingError(f"{label} must be an explicit positive integer") from None
    if integer != value or integer < 1:
        raise AccountingError(f"{label} must be an explicit positive integer")
    return integer


@dataclass(frozen=True)
class Limits:
    """All limits explicit; the old defaults stay 16, opt-in upper bound is 128.

    per_fleet_instance_limit corresponds to existing limit; global_instance_limit
    corresponds to existing global_limit/instance_ceiling. global_gpu_limit is
    a new independent physical-card budget, never an alias for instance count.
    """

    per_fleet_instance_limit: int = 16
    global_instance_limit: int = 16
    global_gpu_limit: int = 16

    def __post_init__(self):
        for field in ("per_fleet_instance_limit", "global_instance_limit", "global_gpu_limit"):
            value = getattr(self, field)
            if type(value) is not int or not 1 <= value <= 128:
                raise AccountingError(f"{field} must be an integer in 1..128")
        if self.per_fleet_instance_limit > self.global_instance_limit:
            raise AccountingError("per_fleet_instance_limit exceeds global_instance_limit")


@dataclass(frozen=True)
class Footprint:
    gpus: int
    instances: int
    two_gpu_instances: int


def payload_footprint(payload):
    if not isinstance(payload, Mapping):
        raise AccountingError("payload must be an object")
    configs = payload.get("framework_config")
    if not isinstance(configs, (list, tuple)) or not configs:
        raise AccountingError("framework_config is missing or empty")
    gpu_total = instance_total = doubles = 0
    for config in configs:
        if not isinstance(config, Mapping):
            raise AccountingError("framework_config component is not an object")
        count = positive_integer(config.get("instance_count"), "instance_count")
        spec = config.get("resource_spec_price", {})
        if not isinstance(spec, Mapping):
            raise AccountingError("resource_spec_price is not an object")
        declared = [positive_integer(value, "per-instance gpu_count")
                    for value in (config.get("gpu_count"), spec.get("gpu_count"))
                    if value is not None]
        if not declared or len(set(declared)) != 1:
            raise AccountingError("per-instance gpu_count missing or inconsistent")
        gpu = declared[0]
        instance_total += count
        gpu_total += count * gpu
        doubles += count if gpu == 2 else 0
    return Footprint(gpu_total, instance_total, doubles)


def footprint(entry, record=None):
    """Payload shapes win over cached costs; malformed explicit shapes never fall back.

    Missing dimensions reserve one GPU and one instance; explicit malformed
    dimensions fail closed. Legacy fallback is for fleet.py single-host entries, whose
    jobs.validate contract fixes instance_count=1. An explicit instance_count
    extends that fallback: gpu_count is per instance; costs.gpu may be a total.
    Taking the larger prevents stale explicit costs from reducing reservations.
    """
    payloads = []
    for payload in (entry.get("payload"), (record or {}).get("request")):
        if payload is None:
            continue
        if not isinstance(payload, Mapping):
            raise AccountingError("payload/request must be an object")
        if "framework_config" in payload:
            payloads.append(payload)
    if payloads:
        shapes = [payload_footprint(payload) for payload in payloads]
        if any(shape != shapes[0] for shape in shapes[1:]):
            raise AccountingError("state payload and durable request disagree")
        return shapes[0]
    count = positive_integer(entry.get("instance_count", 1), "legacy instance_count")
    gpu = entry.get("gpu_count")
    costs = entry.get("costs", {})
    if not isinstance(costs, Mapping):
        raise AccountingError("legacy costs is not an object")
    total = costs.get("gpu")
    per_instance = positive_integer(gpu, "legacy gpu_count") if gpu is not None else None
    explicit_total = positive_integer(total, "legacy costs.gpu") if total is not None else None
    if per_instance is None and explicit_total is None:
        # A lost/unknown request still consumes a conservative minimum slot.
        per_instance = 1
    if per_instance is None:
        if explicit_total % count:
            raise AccountingError("legacy total GPU count is not divisible by instance_count")
        per_instance = explicit_total // count
    physical = max(per_instance * count, explicit_total or 0)
    if physical != per_instance * count:
        # Preserve the larger GPU reservation without guessing its host shape.
        raise AccountingError("legacy GPU sources disagree on instance shape")
    return Footprint(physical, count, count if per_instance == 2 else 0)


def identifier(entry):
    if not isinstance(entry, Mapping):
        raise AccountingError("job/submission observation must be an object")
    direct = entry.get("job_id")
    response = entry.get("response") or {}
    data = (response.get("data") or {}) if isinstance(response, Mapping) else {}
    nested = data.get("job_id") if isinstance(data, Mapping) else None
    if direct and nested and direct != nested:
        raise AccountingError("direct and response job IDs disagree")
    value = direct or nested
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise AccountingError("job_id must be a nonempty string")
    return value


def group_id(entry, record=None):
    groups = {payload.get("logic_compute_group_id") for payload in
              (entry.get("payload"), (record or {}).get("request"))
              if isinstance(payload, Mapping) and payload.get("logic_compute_group_id")}
    if entry.get("group_id"):
        groups.add(entry["group_id"])
    if len(groups) != 1 or not all(isinstance(group, str) and group for group in groups):
        raise AccountingError("group ID missing or inconsistent")
    return next(iter(groups))


@dataclass(frozen=True)
class Reservation:
    identity: str
    job_id: str | None
    record_identities: frozenset[str]
    fleets: frozenset[str]
    group_id: str
    footprint: Footprint
    pending: bool


@dataclass(frozen=True)
class Accounting:
    reservations: tuple[Reservation, ...]
    errors: tuple[str, ...] = ()
    seen_record_identities: frozenset[str] = frozenset()

    @property
    def gpus(self):
        self.require_complete()
        return sum(item.footprint.gpus for item in self.reservations)

    @property
    def instances(self):
        self.require_complete()
        return sum(item.footprint.instances for item in self.reservations)

    @property
    def pending_two_by_group(self):
        self.require_complete()
        groups = defaultdict(int)
        for item in self.reservations:
            if item.pending:
                groups[item.group_id] += item.footprint.two_gpu_instances
        return {group: count for group, count in groups.items() if count}

    def fleet_instances(self, fleet):
        self.require_complete()
        return sum(item.footprint.instances for item in self.reservations if fleet in item.fleets)

    def require_complete(self):
        if self.errors:
            raise AccountingError("; ".join(self.errors))


def account(fleets: Mapping[str, Mapping], official_jobs: Sequence[Mapping] = (),
            submission_records: Mapping[str, Mapping] | None = None):
    """Deduplicate known IDs, never infer identity from payload/name/command.

    fleet keys and record_path values must be canonicalized by the caller.
    Unknown IDs retain one reservation per record_path; absent paths fall back
    to the durable state + worker ID (or the record's position in that state).
    Supplied record contents may recover IDs lost from state after a crash.
    Terminal status is trusted only from official_jobs for a known job ID.
    No clock/local success/deadline can release a resource reservation.
    """
    records = submission_records or {}
    errors, observations, grouped = [], {}, {}
    seen_records, known_records = set(), {}
    # Reconcile the same durable intent across states before grouping: one
    # stale copy may have lost the ID that another copy already persisted.
    for fleet, state in fleets.items():
        if not isinstance(state, Mapping) or not isinstance(state.get("jobs"), list):
            continue
        for index, entry in enumerate(state["jobs"]):
            if not isinstance(entry, Mapping):
                continue
            durable_id = entry.get("record_path") or f"{fleet}#{entry.get('worker_id', index)}"
            if not isinstance(durable_id, str):
                errors.append(f"{fleet}: record identity must be a string")
                continue
            seen_records.add(durable_id)
            try:
                ids = {jid for jid in (identifier(entry), identifier(records.get(durable_id, {}))) if jid}
                if durable_id in known_records:
                    ids.add(known_records[durable_id])
                if len(ids) > 1:
                    raise AccountingError("one durable record identifies multiple jobs")
                if ids:
                    known_records[durable_id] = next(iter(ids))
            except AccountingError as error:
                errors.append(f"{durable_id}: {error}")
    for observed in official_jobs:
        try:
            jid = identifier(observed)
            if not jid:
                raise AccountingError("official observation has no job_id")
            status = observed.get("status")
            if status is not None and not isinstance(status, str):
                raise AccountingError("official status must be a string or unknown")
            if jid in observations and observations[jid] != status:
                raise AccountingError(f"conflicting official observations for {jid}")
            observations[jid] = status
        except (AccountingError, AttributeError) as error:
            errors.append(str(error))
    for fleet, state in fleets.items():
        if not isinstance(state, Mapping):
            errors.append(f"{fleet}: state must be an object")
            continue
        jobs = state.get("jobs")
        if not isinstance(jobs, list):
            errors.append(f"{fleet}: jobs must be a complete list")
            continue
        for index, entry in enumerate(jobs):
            durable_id = f"{fleet}#jobs[{index}]"
            try:
                if not isinstance(entry, Mapping):
                    raise AccountingError("durable entry is not an object")
                durable_id = entry.get("record_path") or f"{fleet}#{entry.get('worker_id', index)}"
                record = records.get(durable_id)
                jid, recorded_jid = identifier(entry), identifier(record or {})
                if jid and recorded_jid and jid != recorded_jid:
                    raise AccountingError("state and submission record job IDs disagree")
                jid = jid or recorded_jid or known_records.get(durable_id)
                official_status = observations.get(jid)
                if jid and official_status in TERMINAL:
                    continue
                # Definitive create rejection has no remote reservation. A
                # local stopped/succeeded label alone cannot prove release.
                if not jid and ((record or {}).get("status", entry.get("status")) == "submission_rejected"):
                    continue
                if (not jid and record is None and entry.get("status") == "job_cancelled"
                        and entry.get("cancel_reason") == "completion_marker_before_submission"):
                    continue
                shape = footprint(entry, record)
                group = group_id(entry, record)
                key = f"job:{jid}" if jid else f"record:{durable_id}"
                pending = official_status not in RUNNING
                item = Reservation(key, jid, frozenset({durable_id}), frozenset({fleet}), group, shape, pending)
                if key in grouped:
                    previous = grouped[key]
                    if (previous.group_id, previous.footprint) != (group, shape):
                        raise AccountingError("duplicate job/record identity has inconsistent resource requests")
                    item = Reservation(key, jid, previous.record_identities | item.record_identities,
                                       previous.fleets | item.fleets, group, shape, previous.pending or pending)
                grouped[key] = item
            except (AccountingError, TypeError) as error:
                errors.append(f"{durable_id}: {error}")
    return Accounting(tuple(grouped.values()), tuple(errors), frozenset(seen_records | set(records)))


@dataclass(frozen=True)
class Admission:
    allowed: bool
    reason: str


def admit_new_intent(usage: Accounting, fleet: str, candidate: Mapping, limits: Limits):
    """Pure guard for a NEW intent; does not submit/retry/stop or mutate usage.

    After admission the controller must persist the intent while holding the
    global lock and re-account it before another admission. Existing planned
    intents are already reserved and need a separate resume path; unresolved
    submission records must still go through jobs.submit/resolve_submission.
    """
    usage.require_complete()
    if identifier(candidate):
        return Admission(False, "known job is not a new submission intent")
    if candidate.get("status") not in (None, "planned"):
        return Admission(False, "existing submission state is not a new intent")
    durable_id = candidate.get("record_path")
    if not isinstance(durable_id, str) or not durable_id:
        raise AccountingError("new intent requires an explicit durable record_path")
    if durable_id in usage.seen_record_identities:
        return Admission(False, "intent already reserved; never create a replacement")
    shape = footprint(candidate)
    group = group_id(candidate)
    if usage.instances + shape.instances > limits.global_instance_limit:
        return Admission(False, "global instance budget exhausted")
    if usage.gpus + shape.gpus > limits.global_gpu_limit:
        return Admission(False, "global physical GPU budget exhausted")
    if usage.fleet_instances(fleet) + shape.instances > limits.per_fleet_instance_limit:
        return Admission(False, "per-fleet instance budget exhausted")
    if shape.two_gpu_instances + usage.pending_two_by_group.get(group, 0) > 1:
        return Admission(False, "one pending two-GPU instance per group across all fleets")
    return Admission(True, "within explicit instance/GPU budgets")


def plan_placements(inventory, jobs, limit=16, now=None, group_backoff=None, allow_two=True,
                    diversify_groups=False, instance_ceiling=16, other_jobs=(), gpu_budget=None):
    """Choose real discovered specs; aggregate availability is only a hint.

    Idle budgets include GPU, CPU and memory and subtract our pending requests.
    When those budgets are exhausted, queue share follows observed LOW GPU use.
    At least three of every four active/planned instances are single GPU.
    Evaluations may prefer uncovered groups, then GPU models, among idle fits;
    coverage is only a tie-breaker when every candidate must queue.
    """
    if (type(instance_ceiling) is not int or not 1 <= instance_ceiling <= 128
            or type(limit) is not int or not 1 <= limit <= instance_ceiling):
        raise PlatformError("Fleet limit must be within the explicit global ceiling (1..128)")
    now = time.time() if now is None else now
    group_backoff = group_backoff or {}
    living = [j for j in jobs if active(j)]
    global_living = living + [j for j in other_jobs if active(j)]
    slots = max(0, limit - sum(int(j.get("instance_count", 1)) for j in living))
    if gpu_budget is None:
        gpu_budget = max(0, instance_ceiling - sum(int(j.get("gpu_count", 1)) * int(j.get("instance_count", 1)) for j in global_living))
    if type(gpu_budget) is not int or not 0 <= gpu_budget <= 128:
        raise PlatformError("Remaining physical GPU budget must be an integer in 0..128")
    groups = {g["id"]: g for g in inventory["groups"]}
    specs = {}
    for selection in inventory["selections"]:
        gid = selection["logic_compute_group_id"]
        if (gid not in groups or number(group_backoff.get(gid)) > now
                or groups[gid].get("resource_query", {}).get("status", "fresh") != "fresh"):
            continue
        n = costs(selection)["gpu"]
        if n not in (1, 2):
            continue
        # Usually one quota exists per size. If not, prefer the smallest real
        # CPU/memory request, without synthesizing unsupported resource shapes.
        old = specs.setdefault(gid, {}).get(n)
        if old is None or (costs(selection)["cpu"], costs(selection)["memory_gi"]) < (costs(old)["cpu"], costs(old)["memory_gi"]):
            specs[gid][n] = selection
    budgets = {gid: {k: max(0.0, number(g["resources"].get(k + "_available"))) for k in ("gpu", "cpu", "memory_gi")} for gid, g in groups.items()}
    queued_gpus = {gid: 0 for gid in groups}
    coverage = {gid: 0 for gid in groups}
    covered_families = set()
    pending_two = set()
    singles = sum(int(j.get("instance_count", 1)) for j in global_living if int(j.get("gpu_count", 1)) == 1)
    doubles = sum(int(j.get("instance_count", 1)) for j in global_living) - singles
    for entry in global_living:
        gid = entry["group_id"]
        if gid in coverage:
            coverage[gid] += 1
        if diversify_groups:
            for component in (entry.get("payload") or {}).get("framework_config") or ():
                family = gpu_family(component.get("resource_spec_price") or {})
                if family is not None:
                    covered_families.add(family)
        if pending(entry) and gid in budgets:
            request = entry.get("costs", {"gpu": entry["gpu_count"] * int(entry.get("instance_count", 1)), "cpu": 0, "memory_gi": 0})
            for resource in budgets[gid]:
                budgets[gid][resource] = max(0.0, budgets[gid][resource] - number(request.get(resource)))
            queued_gpus[gid] += int(entry["gpu_count"]) * int(entry.get("instance_count", 1))
            if int(entry["gpu_count"]) == 2:
                pending_two.add(gid)
    placements = []
    for _ in range(slots):
        if not gpu_budget:
            break
        want_two = allow_two and singles >= 3 * (doubles + 1)
        candidates = []
        for gid, available in specs.items():
            n = 2 if want_two and gpu_budget >= 2 and 2 in available and gid not in pending_two else 1
            if n not in available:
                continue
            selection = available[n]
            request = costs(selection)
            idle_fit = all(budgets[gid][k] >= request[k] for k in request)
            # If a double cannot fit but a single can, use that idle single.
            if n == 2 and not idle_fit and 1 in available:
                single_costs = costs(available[1])
                if all(budgets[gid][k] >= single_costs[k] for k in single_costs):
                    n, selection, request, idle_fit = 1, available[1], single_costs, True
            low = max(0.0, number(groups[gid]["resources"].get("gpu_low_priority_used")))
            # With no idle fit, distribute queues in proportion to the low-tier
            # GPU footprint instead of blindly stacking one resource group.
            queue_score = low / (queued_gpus[gid] + request["gpu"])
            family = gpu_family(selection["resource_spec_price"]) if diversify_groups else None
            uncovered_family = diversify_groups and family is not None and family not in covered_families
            score = (int(idle_fit), int(diversify_groups and idle_fit and not coverage[gid]),
                     int(idle_fit and uncovered_family),
                     budgets[gid]["gpu"] if idle_fit else queue_score,
                     low, -queued_gpus[gid], -coverage[gid] if diversify_groups else 0,
                     int(not idle_fit and uncovered_family), gid)
            candidates.append((score, selection, request, idle_fit))
        if not candidates:
            break
        _, selection, request, idle_fit = max(candidates, key=lambda x: x[0])
        gid = selection["logic_compute_group_id"]
        placements.append({"selection": copy.deepcopy(selection), "group_id": gid, "gpu_count": request["gpu"], "costs": request,
                           "reason": "idle_gpu_cpu_memory" if idle_fit else "queue_by_low_gpu_footprint"})
        for resource in budgets[gid]:
            budgets[gid][resource] = max(0.0, budgets[gid][resource] - request[resource])
        queued_gpus[gid] += request["gpu"]
        gpu_budget -= request["gpu"]
        coverage[gid] += 1
        if diversify_groups:
            family = gpu_family(selection["resource_spec_price"])
            if family is not None:
                covered_families.add(family)
        if request["gpu"] == 2:
            doubles += 1
            pending_two.add(gid)
        else:
            singles += 1
    return placements


def list_own_jobs(client, user_id):
    result = []
    for page in range(1, 51):
        response = client.request("POST", "/api/v1/train_job/list", {"workspace_id": DEFAULT_WORKSPACE, "created_by": user_id, "page_num": page, "page_size": 100})
        batch = items(response, "jobs")
        result.extend(batch)
        total = number(response.get("data", {}).get("total"), len(result))
        if len(result) >= total or len(batch) < 100:
            return result
    raise PlatformError("Job listing exceeded 5000 records; refusing an incomplete duplicate check")


def resolve_submission(record, platform_jobs):
    """Recover a lost create response only from an exact, unique owned match."""
    request = record["request"]
    matches = [j for j in platform_jobs if j.get("name") == request["name"]
               and j.get("command") == request["command"]
               and j.get("workspace_id") == request["workspace_id"]
               and j.get("project_id") == request["project_id"]
               and j.get("logic_compute_group_id") == request["logic_compute_group_id"]]
    if len(matches) > 1:
        raise PlatformError("Multiple platform jobs match one submission intent; no replacement will be posted")
    if len(matches) == 1:
        record = dict(record, job_id=matches[0]["job_id"], status="recovered_from_exact_list_match")
    return record


def failure_kind(status, events):
    def strings(value):
        # Field names such as {"evict": false} describe metadata, not a
        # termination event. Keep the actual reason/message values instead.
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from strings(child)

    text = "\n".join(strings(events)).casefold()
    # Some job-event messages embed the same negative flags in plain text.
    # Removing only the flag preserves a separate real preemption reason.
    text = re.sub(r"\b(?:preempt(?:ed|ion|ing)?|evict(?:ed|ion|ing)?)\b[\"']?\s*[:=]\s*[\"']?(?:false|0|null|none)\b", "", text)
    if (re.search(r"\b(?:preempt(?:ed|ion|ing)?|evict(?:ed|ion|ing)?)\b", text)
            or any(x in text for x in ("抢占", "node_not_ready", "node lost", "节点故障", "maxrunningtime", "deadlineexceeded"))):
        return "resource_or_runtime"
    if any(x in text for x in ("out of memory", "oomkilled", "cuda_error", "traceback", "modulenotfounderror", "filenotfounderror", "syntaxerror", "invalid argument")):
        return "software_or_capacity"
    return "stopped" if status in {"job_stopped", "job_cancelled", "job_canceled"} else "unknown"


def failed_job_evidence(client, identifier, running_round=None):
    """Supplement generic job failures with current Pod events, best effort.

    Keep API diagnostics separate from event evidence: an API error mentioning
    OOM or preemption is not proof of the workload's termination cause.
    """
    diagnostics = []

    def read(endpoint, body):
        try:
            return client.request("POST", "/api/v1/train_job/" + endpoint, body)
        except (PlatformError, OSError, TimeoutError) as error:
            diagnostics.append({"endpoint": endpoint, "request": body, "error": str(error)})
            return None

    job_events = read("job_event_list", {"jobId": identifier}) or {}
    evidence = copy.deepcopy(job_events)
    kind = failure_kind("job_failed", job_events)
    if kind == "unknown":
        instances = read("instance_list", {"jobId": identifier, "page_num": 1, "page_size": 20})
        if instances is not None:
            evidence["instance_list"] = instances
            candidates = items(instances)
            # Retry history may contain an earlier preemption followed by a
            # real software failure. Classify only the current/latest round.
            latest = max((number(i.get("running_round"), -1) for i in candidates), default=-1)
            selected_round = number(running_round, latest)
            if latest >= 0:
                candidates = [i for i in candidates if number(i.get("running_round"), -1) == selected_round]
                evidence["classification_running_round"] = selected_round
            instance_events = []
            evidence["instance_events"] = []
            for instance in candidates:
                name = instance.get("name")
                if not name:
                    continue
                response = read("instance_event_list", {"jobId": identifier, "instanceName": name})
                if response is not None:
                    evidence["instance_events"].append({"instance_name": name, "response": response})
                    instance_events.append(response)
            kind = failure_kind("job_failed", [job_events, instance_events])
    if diagnostics:
        evidence["diagnostics"] = diagnostics
    return evidence, kind


def refresh_inventory(client, fleet_dir, seed=None, resources=True):
    cache = fleet_dir / "inventory.json"
    if cache.exists():
        inventory = json.loads(cache.read_text())
    elif seed:
        inventory = json.loads(Path(seed).read_text())
    else:
        inventory = discover(client)
    # A prior research snapshot may contain only 1-GPU selections. Rebuild from
    # its actual priced specs, preserving project/image identity from each group.
    existing = {s["logic_compute_group_id"]: s for s in inventory["selections"]}
    selections = []
    errors = []
    for group in inventory["groups"]:
        gid = group["id"]
        previous_query = group.get("resource_query", {})
        observed_at = previous_query.get("observed_at", inventory.get("at"))
        checked_at = dt.datetime.now(dt.timezone.utc).isoformat()
        if resources:
            try:
                response = client.request("GET", "/api/v1/compute_resources/logic_compute_groups/" + gid)
                summary = resource_summary(response)
                if any(summary[k + "_available"] is None for k in ("gpu", "cpu", "memory_gi")):
                    raise PlatformError("Resource query returned incomplete GPU/CPU/memory totals")
                group["resources"] = summary
                group["resource_query"] = {"status": "fresh", "checked_at": checked_at, "observed_at": checked_at}
            except (PlatformError, OSError, TimeoutError) as error:
                # Keep priced specs and project/image identity so a later
                # successful query can reactivate this group. Cached resources
                # are diagnostic only and never justify a submission.
                group["resource_query"] = {"status": "error", "checked_at": checked_at,
                                           "observed_at": observed_at, "error": str(error)}
                errors.append({"group_id": gid, **group["resource_query"]})
        else:
            group["resource_query"] = {"status": "not_queried", "observed_at": observed_at,
                                       "reason": "completion_or_stop_marker"}
        if gid in existing:
            for spec in group["specs"]:
                if int(number(spec.get("gpu_count"))) in (1, 2):
                    selection = dict(existing[gid], resource_spec_price=copy.deepcopy(spec))
                    # Rendering validates the model and strict specification.
                    render_payload(selection, "true", "jade-spec-check")
                    selections.append(selection)
    inventory.update(at=dt.datetime.now(dt.timezone.utc).isoformat(), selections=selections,
                     resource_query_errors=errors)
    # HTTP evidence is preserved by initial discovery artifacts; do not grow a
    # monitor snapshot indefinitely with another full history on every pass.
    inventory.pop("evidence", None)
    write_json(cache, inventory)
    return inventory


def resource_report(inventory):
    return [{"group_id": g["id"], "name": g["name"], "gpu_idle_indicator": g["resources"].get("gpu_available"),
             "gpu_low_priority_used": g["resources"].get("gpu_low_priority_used"),
             "cpu_available": g["resources"].get("cpu_available"), "memory_gi_available": g["resources"].get("memory_gi_available"),
             "resource_query": g.get("resource_query", {})}
            for g in inventory["groups"]]


def command_for(template, run_dir, worker_id, gpu_count=1):
    if "{run_dir}" not in template or "{worker_id}" not in template:
        raise PlatformError("Command template must contain {run_dir} and {worker_id}; do not quote the placeholders")
    return template.replace("{run_dir}", shlex.quote(str(run_dir))).replace("{worker_id}", shlex.quote(worker_id)).replace("{gpu_count}", str(int(gpu_count)))


def registered_fleet_snapshot(state_path, state):
    """Load every durable reservation while the caller owns the global lock.

    An unreadable registered state is an accounting error, never spare capacity.
    Orphan submission records are attached to their owning fleet before counting.
    """
    registry = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {"fleets": []}
    if not isinstance(registry.get("fleets"), list):
        raise AccountingError("Fleet registry must contain a complete fleets list")
    current = str(Path(state_path).expanduser().resolve())
    paths = list(dict.fromkeys(str(Path(path).expanduser().resolve()) for path in registry["fleets"]))
    if current not in paths:
        paths.append(current)
    write_json(REGISTRY_PATH, {"fleets": paths})
    fleets, records = {}, {}
    for path in paths:
        try:
            snapshot = state if path == current else json.loads(Path(path).read_text())
        except (OSError, ValueError) as error:
            raise AccountingError(f"Registered fleet state is unreadable: {path}: {error}") from error
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("jobs"), list):
            raise AccountingError(f"Registered fleet state has no complete jobs list: {path}")
        fleets[path] = snapshot
        by_record = {}
        for entry in snapshot["jobs"]:
            if not isinstance(entry, dict):
                raise AccountingError(f"Registered fleet has a malformed reservation: {path}")
            if entry.get("record_path"):
                record_path = str(Path(entry["record_path"]).expanduser().resolve())
                entry["record_path"] = record_path
                by_record[record_path] = entry
        record_paths = set(by_record)
        if Path(path).name == "state.json":
            record_paths.update(str(record.resolve()) for record in (Path(path).parent / "submissions").glob("*.json"))
        for record_path in sorted(record_paths):
            if not Path(record_path).exists():
                continue
            try:
                record = json.loads(Path(record_path).read_text())
            except (OSError, ValueError) as error:
                raise AccountingError(f"Submission record is unreadable: {record_path}: {error}") from error
            if not isinstance(record, dict):
                raise AccountingError(f"Submission record is not an object: {record_path}")
            records[record_path] = record
            if record_path not in by_record:
                request = record.get("request") or {}
                entry = {"record_path": record_path, "payload": request,
                         "group_id": request.get("logic_compute_group_id"),
                         "status": record.get("status", "submission_unknown"),
                         "worker_id": "orphan-" + Path(record_path).stem,
                         "created_at": 0, "recovered_orphan_record": True}
                if job_id(record):
                    entry["job_id"] = job_id(record)
                snapshot["jobs"].append(entry)
    return fleets, records


def reservation_entries(usage, fleets, records, current):
    """Return one planner entry per deduplicated job/intent, including other runs."""
    usage.require_complete()
    sources = {}
    for fleet, state in fleets.items():
        for index, entry in enumerate(state["jobs"]):
            sources[entry.get("record_path") or f"{fleet}#{entry.get('worker_id', index)}"] = entry
    local, other = [], []
    for item in usage.reservations:
        record_identity = sorted(item.record_identities)[0]
        source = sources[record_identity]
        record = records.get(record_identity, {})
        payload = record.get("request") or source.get("payload") or {}
        request = {"gpu": item.footprint.gpus, "cpu": 0.0, "memory_gi": 0.0}
        if payload.get("framework_config"):
            for component in payload["framework_config"]:
                count = positive_integer(component.get("instance_count"), "instance_count")
                spec = component.get("resource_spec_price") or {}
                request["cpu"] += number(spec.get("cpu_count")) * count
                request["memory_gi"] += number(spec.get("memory_size_gib")) * count
        else:
            for resource in ("cpu", "memory_gi"):
                request[resource] = number((source.get("costs") or {}).get(resource))
        entry = {"group_id": item.group_id, "costs": request, "payload": payload,
                 "gpu_count": 2 if item.footprint.two_gpu_instances == item.footprint.instances else 1,
                 "instance_count": item.footprint.instances,
                 "status": "job_queuing" if item.pending else "job_running"}
        (local if current in item.fleets else other).append(entry)
    return local, other


def register_and_count_other_fleets(state_path, platform_jobs):
    """Compatibility wrapper; global reservation identity is counted only once."""
    state_path = Path(state_path).resolve()
    state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": []}
    fleets, records = registered_fleet_snapshot(state_path, state)
    usage = account(fleets, platform_jobs, records)
    usage.require_complete()
    return sum(item.footprint.instances for item in usage.reservations if str(state_path) not in item.fleets)


def refresh_submission_resources(client, group):
    """A fresh official GPU/CPU/memory query is required immediately before POST."""
    checked_at = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        response = client.request("GET", "/api/v1/compute_resources/logic_compute_groups/" + group["id"])
        summary = resource_summary(response)
        if any(summary[k + "_available"] is None for k in ("gpu", "cpu", "memory_gi")):
            raise PlatformError("Resource query returned incomplete GPU/CPU/memory totals")
        group["resources"] = summary
        group["resource_query"] = {"status": "fresh", "checked_at": checked_at, "observed_at": checked_at}
        return True
    except (PlatformError, OSError, TimeoutError) as error:
        group["resource_query"] = {"status": "error", "checked_at": checked_at,
                                   "error": str(error)}
        return False


def export_waiting(run_dir, marker, now):
    if marker != "COMPLETE.json":
        return False
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if config.get("skip_final_export") or (run_dir / "final_export" / "EXPORT_COMPLETE.json").exists():
        return False
    return now - (run_dir / marker).stat().st_mtime < 900


def reconcile(client, run_dir, command_template, limit=16, fill=False, inventory_seed=None,
              runtime=86400, retries=2, allow_two=True, reset_failures=False,
              diversify_groups=False, global_limit=16, global_gpu_limit=None):
    run_dir = Path(run_dir).resolve()
    fleet_dir = run_dir / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    state_path = fleet_dir / "state.json"
    with writer_lock(REGISTRY_PATH.parent / "fleet-global-lock"), writer_lock(fleet_dir / ".reconciler"):
        state = json.loads(state_path.read_text()) if state_path.exists() else {"schema_version": 1, "run_dir": str(run_dir), "command_template": command_template, "next_worker": 1, "jobs": [], "group_backoff": {}, "failures": []}
        if state["command_template"] != command_template:
            raise PlatformError("Fleet command template is immutable; use the same command for this run")
        command_for(command_template, run_dir, "validation")
        if global_gpu_limit is None:
            global_gpu_limit = global_limit
        limits = Limits(limit, global_limit, global_gpu_limit)
        if reset_failures:
            state.update(group_backoff={}, failures=[], blocked_reason=None)
        marker = next((name for name in COMPLETION_MARKERS if (run_dir / name).exists()), None)
        inventory = refresh_inventory(client, fleet_dir, inventory_seed, resources=not marker)
        platform_jobs = list_own_jobs(client, inventory["user"]["id"])
        by_id = {j["job_id"]: j for j in platform_jobs}
        fleets, records = registered_fleet_snapshot(state_path, state)
        current = str(state_path)
        now = time.time()
        for entry in state["jobs"]:
            if not active(entry):
                continue
            record_path = Path(entry["record_path"]) if entry.get("record_path") else None
            if record_path and record_path.exists():
                record = json.loads(record_path.read_text())
                if not job_id(record) and record.get("status") != "submission_rejected":
                    resolved = resolve_submission(record, platform_jobs)
                    if resolved != record:
                        write_json(record_path, resolved)
                    record = resolved
                records[str(record_path)] = record
                entry["job_id"] = job_id(record)
                if record.get("status") == "submission_rejected":
                    entry["status"] = "submission_rejected"
            identifier = entry.get("job_id")
            detail = by_id.get(identifier)
            if identifier and detail is None:
                detail = client.request("POST", "/api/v1/train_job/detail", {"job_id": identifier})["data"]
                by_id[identifier] = dict(detail, job_id=identifier)
            if detail:
                if detail.get("priority_level") != "LOW" or str(detail.get("priority_name")) != "1":
                    if fill:
                        client.request("POST", "/api/v1/train_job/stop", {"job_id": identifier})
                    state["blocked_reason"] = "Priority changed from LOW/1; inspect affected owned job"
                entry["status"] = detail.get("status")
                entry["running_round"] = detail.get("current_running_round")
            if not active(entry) and not entry.get("terminal_at"):
                entry["terminal_at"] = now
                events = {}
                kind = failure_kind(entry["status"], events)
                if identifier and entry["status"] == "job_failed":
                    events, kind = failed_job_evidence(client, identifier, entry.get("running_round"))
                    write_json(fleet_dir / "events" / (entry["worker_id"] + ".json"), events)
                entry["failure_kind"] = kind
                if entry["status"] != "job_succeeded" and not entry.get("stop_requested"):
                    state["failures"].append({"at": now, "group_id": entry["group_id"], "kind": kind, "worker_id": entry["worker_id"]})
                    recent_group = sum(f["group_id"] == entry["group_id"] and now - f["at"] < 3600 for f in state["failures"])
                    state["group_backoff"][entry["group_id"]] = now + min(1800, 30 * 2 ** min(recent_group, 6))
        recent_errors = [f for f in state["failures"] if now - f["at"] < 900 and f["kind"] in {"software_or_capacity", "unknown"}]
        if len(recent_errors) >= 3:
            state["blocked_reason"] = "Three unexplained/software failures in 15 minutes; inspect logs before --reset-failures"
        # Preserve existing FIFO tickets, including retries that made two
        # existing doubles pending. Only admission of another double is blocked.
        usage = account(fleets, list(by_id.values()), records)
        usage.require_complete()
        other_active = sum(item.footprint.instances for item in usage.reservations if current not in item.fleets)
        local_jobs, other_jobs = reservation_entries(usage, fleets, records, current)
        effective_limit = min(limit, max(0, global_limit - other_active))
        placements = [] if marker or state.get("blocked_reason") or not effective_limit else plan_placements(
            inventory, local_jobs, effective_limit, now, state["group_backoff"],
            allow_two, diversify_groups, instance_ceiling=global_limit,
            other_jobs=other_jobs, gpu_budget=max(0, global_gpu_limit - usage.gpus))
        write_json(state_path, state)
        created = []
        if fill and not marker and not state.get("blocked_reason"):
            # Include durable plans stranded by a crash before submit() wrote
            # its intent. No POST can precede that intent, so these are safe.
            stranded = [e for e in state["jobs"] if e["status"] == "planned"
                        and not e.get("job_id") and not Path(e["record_path"]).exists()]
            for placement in placements:
                index = state["next_worker"]
                state["next_worker"] += 1
                worker_id = f"w{index:06d}"
                name = f"jade-libero-{hashlib.sha256(str(run_dir).encode()).hexdigest()[:8]}-{worker_id}"
                payload = render_payload(placement["selection"], command_for(command_template, run_dir, worker_id, placement["gpu_count"]), name, runtime, retries)
                record_path = fleet_dir / "submissions" / (worker_id + ".json")
                entry = {k: placement[k] for k in ("group_id", "gpu_count", "costs", "reason")}
                entry.update(worker_id=worker_id, record_path=str(record_path), status="planned", created_at=now, payload=payload)
                usage = account(fleets, list(by_id.values()), records)
                admission = admit_new_intent(usage, current, entry, limits)
                if not admission.allowed:
                    state["last_admission_block"] = admission.reason
                    break
                state["jobs"].append(entry)
                write_json(state_path, state)
                stranded.append(entry)
            for entry in stranded:
                usage = account(fleets, list(by_id.values()), records)
                usage.require_complete()
                shape = footprint(entry)
                admission_block = None
                if usage.instances > global_limit or usage.fleet_instances(current) > limit:
                    admission_block = "existing reservations exceed instance ceiling"
                elif usage.gpus > global_gpu_limit:
                    admission_block = "existing reservations exceed physical GPU ceiling"
                elif shape.two_gpu_instances and usage.pending_two_by_group.get(entry["group_id"], 0) > 1:
                    admission_block = "another two-GPU intent is pending in this group"
                if admission_block:
                    entry["admission_block"] = admission_block
                    continue
                entry.pop("admission_block", None)
                if number(state["group_backoff"].get(entry["group_id"])) > now:
                    continue
                group = next((g for g in inventory["groups"] if g["id"] == entry["group_id"]), None)
                if group is None or group.get("resource_query", {}).get("status", "fresh") != "fresh":
                    continue
                if not refresh_submission_resources(client, group):
                    inventory.setdefault("resource_query_errors", []).append({"group_id": group["id"], **group["resource_query"]})
                    continue
                try:
                    record = submit(client, entry["payload"], entry["record_path"])
                    entry.update(job_id=job_id(record), status="submitted")
                    status = inspect_job(client, record)
                    entry["status"] = status["status"]
                    records[entry["record_path"]] = record
                    by_id[entry["job_id"]] = dict(status.get("detail", {}).get("data", {}),
                                                 job_id=entry["job_id"], status=entry["status"])
                    write_json(fleet_dir / "status" / (entry["worker_id"] + ".json"), status)
                    created.append({k: entry[k] for k in ("worker_id", "job_id", "group_id", "gpu_count", "status")})
                except Exception as error:
                    record_path = Path(entry["record_path"])
                    record = json.loads(record_path.read_text()) if record_path.exists() else {}
                    if record:
                        records[str(record_path)] = record
                    entry.update(job_id=job_id(record), status=record.get("status", "planned"), last_error=str(error))
                    # Do not drain the rest of a batch after a platform error.
                    write_json(state_path, state)
                    break
                write_json(state_path, state)
        waiting_for_export = export_waiting(run_dir, marker, now)
        if fill and marker:
            for entry in state["jobs"]:
                # submit() persists its intent before any create POST. Only a
                # planned entry without either identity can be cancelled
                # locally; unknown or recorded submissions keep their slots.
                if (entry["status"] == "planned" and not entry.get("job_id")
                        and not Path(entry["record_path"]).exists()):
                    entry.update(status="job_cancelled", terminal_at=now,
                                 cancel_reason="completion_marker_before_submission", cancel_marker=marker)
                    write_json(state_path, state)
                    continue
                if active(entry) and entry.get("job_id") and not entry.get("stop_requested"):
                    if waiting_for_export and entry["status"] in RUNNING:
                        continue
                    client.request("POST", "/api/v1/train_job/stop", {"job_id": entry["job_id"]})
                    entry["stop_requested"] = now
                    write_json(state_path, state)
        write_json(state_path, state)
        usage = account(fleets, list(by_id.values()), records)
        usage.require_complete()
        snapshot = {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "limit": limit,
                    "global_limit": global_limit,
                    "global_gpu_limit": global_gpu_limit,
                    "active_jobs": sum(active(e) for e in state["jobs"]),
                    "active_instances": usage.fleet_instances(current),
                    "running_jobs": sum(e["status"] in RUNNING for e in state["jobs"]),
                    "pending_jobs": sum(pending(e) for e in state["jobs"]),
                    "global_active_jobs": usage.instances,
                    "global_reserved_instances": usage.instances,
                    "active_gpus": sum(item.footprint.gpus for item in usage.reservations if current in item.fleets),
                    "global_reserved_gpus": usage.gpus,
                    "global_pending_two_by_group": usage.pending_two_by_group,
                    "unresolved_reservations": sum(item.job_id is None for item in usage.reservations),
                    "waiting_for_export": waiting_for_export,
                    "marker": marker, "blocked_reason": state.get("blocked_reason"), "fill": fill,
                    "diversify_groups": diversify_groups,
                    "resource_query_errors": inventory.get("resource_query_errors", []),
                    "fresh_resource_groups": sum(g.get("resource_query", {}).get("status", "fresh") == "fresh" for g in inventory["groups"]),
                    "resources": resource_report(inventory), "planned": placements, "created": created,
                    "jobs": [{k: v for k, v in e.items() if k != "payload"} for e in state["jobs"]]}
        write_json(fleet_dir / "latest.json", snapshot)
        return snapshot


def reconcile_with_global_lock_retry(*args, **kwargs):
    """Retry only short contention between our independently timed fleets."""
    expected = f"Another trainer owns {REGISTRY_PATH.parent / 'fleet-global-lock'}; refusing concurrent writes"
    for attempt in range(10):
        try:
            return reconcile(*args, **kwargs)
        except RuntimeError as error:
            if str(error) != expected or attempt == 9:
                raise
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("reconcile", "watch"))
    parser.add_argument("--credentials", default="~/.local/share/jade-lotus/credentials.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--command-template", required=True, help="Shell command with unquoted {run_dir} and {worker_id} placeholders")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--global-limit", type=int, default=16,
                        help="Explicit instance ceiling across all registered fleets, at most 128; also the default GPU ceiling")
    parser.add_argument("--global-gpu-limit", type=int,
                        help="Separate physical GPU ceiling across all fleets, at most 128; defaults to --global-limit")
    parser.add_argument("--fill", action="store_true", help="Actually submit/stop this run's managed jobs")
    parser.add_argument("--inventory", type=Path, help="Reuse discovered project/image/spec metadata; resources are always refreshed")
    parser.add_argument("--runtime-seconds", type=int, default=86400)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--single-only", action="store_true")
    parser.add_argument("--reset-failures", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.interval_seconds < 30:
        parser.error("Monitoring interval must be at least 30 seconds")
    client = InspireClient(OAuthCredentials(args.credentials))
    passes = 0
    while True:
        try:
            snapshot = reconcile_with_global_lock_retry(client, args.run_dir, args.command_template, args.limit, args.fill, args.inventory,
                                 args.runtime_seconds, args.retries, not args.single_only, args.reset_failures and passes == 0,
                                 global_limit=args.global_limit, global_gpu_limit=args.global_gpu_limit)
            print(json.dumps({k: snapshot[k] for k in ("at", "active_jobs", "running_jobs", "pending_jobs", "marker", "blocked_reason", "created")}), flush=True)
            if args.action == "reconcile" or (snapshot["marker"] and not snapshot["waiting_for_export"]) or snapshot["blocked_reason"]:
                return 0 if not snapshot["blocked_reason"] else 2
        except (PlatformError, OSError, ValueError, RuntimeError) as error:
            diagnostic = {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "error": str(error)}
            print(json.dumps(diagnostic), file=sys.stderr, flush=True)
            write_json(args.run_dir / "fleet" / "monitor_error.json", diagnostic)
            if args.action == "reconcile":
                return 1
        passes += 1
        client.evidence.clear()
        # A plain timer keeps long queue waits outside the model conversation.
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
