#!/usr/bin/env python3
"""Observe a frozen candidate pool; optionally request better packed membership.

This entrypoint never allocates/releases jobs or stops training. The default
prints read-only packing hypotheses. --apply may publish an exact future-step
request to the existing controller, which alone aligns complete model/Adam
state and decides whether the switch is admissible. All candidates must belong
to the already authorized, frozen physical pool of at most 64 cards.

A larger shared-visibility job is only a packing hypothesis, not a measured
speedup. Physical device identities absent from QZ are not invented. Retired
workers remain allocated and stale; this monitor provides no hot-spare or
multi-rank-loss recovery. Completed/in-progress requests are never replaced.
Only this monitor's acknowledged rejected request can be archived for a
bounded retry. Other writers must respect an occupied request path and the
candidate-watch writer lock; the controller is a read-only consumer.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

if __package__:
    from . import launch
else:
    import launch

SCHEMA = "fragment-candidate-watch-v1"
IDENTITY = ("worker_id", "session_id", "pid")


class Unavailable(ValueError):
    """No safe recommendation can be made from this observation."""


def read(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise Unavailable(f"Expected an object: {path}")
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_private(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def integer(value, name, minimum=0, maximum=64):
    if type(value) is not int or not minimum <= value <= maximum:
        raise Unavailable(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def identity(value):
    if (not isinstance(value, dict) or not isinstance(value.get("worker_id"), str)
            or re.fullmatch(r"w\d{6}", value["worker_id"]) is None
            or not isinstance(value.get("session_id"), str) or not value["session_id"]
            or type(value.get("pid")) is not int or value["pid"] < 1):
        raise Unavailable("Missing exact worker/session/process identity")
    return {key: value[key] for key in IDENTITY}


def load_frozen(run):
    site = read(run / "site.json")
    manifest = read(run / "source_manifest.json")
    if hashlib.sha256((run / "site.json").read_bytes()).hexdigest() != manifest.get("files", {}).get("site.json"):
        raise Unavailable("Frozen site identity differs")
    if site.get("schema") != launch.SCHEMA:
        raise Unavailable("Unsupported frozen site schema")
    cards = integer(site.get("cards"), "physical cards", 2)
    active = integer(site.get("active_cards", cards), "active cards", 2, cards)
    integer(site.get("global_card_limit"), "global card limit", cards)
    layout = site.get("job_layout")
    if (not isinstance(layout, list) or not layout
            or any(type(size) is not int or size not in (1, 2, 4, 8, 16) for size in layout)
            or sum(layout) != cards):
        raise Unavailable("Frozen layout does not cover the authorized physical pool")
    spec = read(run / "probe/spec.json")
    workers = [f"w{index + 1:06d}" for index in range(cards)]
    if (spec.get("cards") != active or spec.get("topology_slots") != cards
            or spec.get("workers") != workers or spec.get("initial_workers") != workers[:active]
            or not isinstance(spec.get("run_id"), str) or not spec["run_id"]
            or spec.get("run_dir") != str(run / "probe")):
        raise Unavailable("Controller membership does not match the frozen pool")
    config_path = Path(spec.get("config", "")).resolve()
    if (not config_path.is_relative_to(run)
            or hashlib.sha256(config_path.read_bytes()).hexdigest() != spec.get("config_sha256")):
        raise Unavailable("Controller training configuration identity differs")
    integer(spec.get("steps"), "target steps", 2, 20000)
    if spec["steps"] != site.get("steps"):
        raise Unavailable("Controller and frozen target steps differ")
    return site, spec


def domains(layout):
    result, start = [], 0
    for index, size in enumerate(layout):
        result.append({"prefix": f"w{index + 1:06d}", "size": size,
            "workers": [f"w{start + local + 1:06d}" for local in range(size)]})
        start += size
    return result


def validate_job(job, domain, *, now, max_age):
    prefix, size = domain["prefix"], domain["size"]
    observation = job.get("last_observation", {})
    if (job.get("worker_id") != prefix or not isinstance(job.get("job_id"), str) or not job["job_id"]
            or job.get("status") != "job_running" or job.get("official_terminal") is not False
            or observation.get("status") != "job_running" or observation.get("job_id") != job["job_id"]
            or observation.get("worker_id") != prefix or observation.get("priority_verified") is not True
            or observation.get("priority_level") != "LOW" or str(observation.get("priority_name")) != "1"
            or type(observation.get("requested_cards")) is not int or observation["requested_cards"] != size):
        raise Unavailable(f"{prefix}: job is not a confirmed running LOW/1 allocation of its frozen size")
    stamp = observation.get("time")
    if (isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
            or not -2 <= now - stamp <= max_age):
        raise Unavailable(f"{prefix}: official allocation observation is stale or unknown")
    configs = job.get("payload", {}).get("framework_config")
    if not isinstance(configs, list) or len(configs) != 1:
        raise Unavailable(f"{prefix}: unexpected allocation instance layout")
    config = configs[0]
    resources = config.get("resource_spec_price", {})
    # Discovery selections retain pricing and nested device metadata. The
    # create contract deliberately normalizes these fields; raw dict equality
    # would reject every real discovered specification despite matching cards.
    from jade_lotus.ascend_platform import validate_payload
    from jade_lotus.platform import PlatformError
    try:
        validate_payload(job.get("payload", {}), job.get("selection", {}))
    except (PlatformError, KeyError, TypeError, ValueError) as error:
        raise Unavailable(f"{prefix}: immutable allocation request differs from its selected contract") from error
    if (type(config.get("instance_count")) is not int or config["instance_count"] != 1
            or type(resources.get("gpu_count")) is not int or resources["gpu_count"] != size
            or type(resources.get("cpu_count")) is not int or resources["cpu_count"] < 1
            or type(resources.get("memory_size_gib")) is not int or resources["memory_size_gib"] < 1):
        raise Unavailable(f"{prefix}: requested resources differ from the selected specification")
    instances = observation.get("instances")
    if not isinstance(instances, list) or len(instances) != 1:
        raise Unavailable(f"{prefix}: exact single-instance placement is unavailable")
    instance = instances[0]
    if (not isinstance(instance.get("node"), str) or not instance["node"]
            or not isinstance(instance.get("name"), str) or not instance["name"]
            or instance.get("instance_status") != "instance_running"):
        raise Unavailable(f"{prefix}: actual running instance/node is unknown")
    return {"job_id": job["job_id"], "node": instance["node"], "hostname": instance["name"],
            "physical_device_ids_verified": observation.get("allocation_verified") is True}


def validate_worker(run, spec, worker, domain, local_rank, placement, *, now, max_age):
    ready = read(run / "probe/ready" / f"{worker}.json")
    supervisor_path = run / "probe/supervisors" / f"{worker}.json"
    supervisor = read(supervisor_path)
    same = identity(ready)
    age = now - supervisor_path.stat().st_mtime
    freshness = min(max_age, float(spec.get("heartbeat_timeout", 15)))
    if (ready.get("run_id") != spec["run_id"] or same["worker_id"] != worker
            or supervisor.get("run_id") != spec["run_id"] or identity(supervisor) != same
            or type(supervisor.get("heartbeat_counter")) is not int or supervisor["heartbeat_counter"] < 1
            or not -2 <= age <= freshness
            or ready.get("allocation_group") != domain["prefix"]
            or type(ready.get("allocated_devices")) is not int or ready["allocated_devices"] != domain["size"]
            or type(ready.get("device_index")) is not int or ready["device_index"] != local_rank
            or ready.get("hostname") != placement["hostname"]):
        raise Unavailable(f"{worker}: live worker identity or visible allocation differs")
    adapter = ready.get("adapter_device")
    if adapter is not None and (not isinstance(adapter, dict)
            or any(type(adapter.get(key)) is not int for key in ("visible_devices", "allocated_devices", "device_index"))
            or adapter.get("visible_devices") != domain["size"]
            or adapter.get("allocated_devices") != domain["size"] or adapter.get("device_index") != local_rank):
        raise Unavailable(f"{worker}: model adapter device proof differs")
    if (run / "probe/exits" / f"{worker}-{same['session_id']}.json").exists() or (run / "probe/errors" / f"{worker}-{same['session_id']}.json").exists():
        raise Unavailable(f"{worker}: this exact resident session has failed or exited")
    return ready


def best_retire(active_domains, target):
    """Subset of whole jobs, maximize domain reduction and keep early ranks."""
    choices = {0: []}
    for domain in reversed(active_domains):
        for count, chosen in list(choices.items()):
            total = count + domain["size"]
            if total <= target and (total not in choices or len(chosen) + 1 > len(choices[total])):
                choices[total] = chosen + [domain]
    return choices.get(target, [])


def advise(run, *, now=None, lead_steps=3, max_age=120):
    run = Path(run).resolve()
    now = time.time() if now is None else now
    base = {"schema": SCHEMA, "action": "wait", "allocation_requested": False,
            "resource_release_requested": False, "automatic_failover": False,
            "reason_scope": "packing hypothesis; no measured speedup claim"}
    if any((run / marker).exists() for marker in ("STOP", "FATAL.json", "COMPLETE.json")):
        return {**base, "action": "terminal", "reason": "run terminal marker"}
    site, spec = load_frozen(run)
    summary = read(run / "probe/summary.json")
    if summary.get("run_id") != spec["run_id"]:
        raise Unavailable("Summary belongs to a different controller run")
    if summary.get("status") in {"passed", "failed"}:
        return {**base, "action": "terminal", "reason": "controller terminal state"}
    if summary.get("status") != "running" or not summary.get("commits"):
        return {**base, "reason": "waiting for first committed active membership"}
    latest = summary["commits"][-1]
    step = integer(summary.get("committed_steps"), "committed step", 1, spec["steps"])
    if type(latest.get("step")) is not int or latest["step"] != step:
        raise Unavailable("Summary and latest COMMIT disagree")
    generation = integer(latest.get("generation"), "COMMIT generation", 0, 1000000)
    active = {identity(member)["worker_id"]: identity(member) for member in latest.get("members", [])}
    if len(active) != len(latest.get("members", [])) or len(active) != spec["cards"]:
        raise Unavailable("Latest active membership is duplicated or has shrunk; this packing monitor cannot repair losses")
    topology = read(run / "probe/topology/state.json")
    if (topology.get("run_id") != spec["run_id"] or topology.get("physical_pool_slots") != site["cards"]
            or topology.get("recovery_required") is not False):
        raise Unavailable("Cohort protocol identity or recovery state is not admissible")
    if topology.get("pending_trial") is not None or any(r.get("status") not in {"rejected", "committed"}
            for r in topology.get("requests", {}).values()):
        return {**base, "reason": "controller is processing an existing switch", "committed_step": step}
    future = step + integer(lead_steps, "lead steps", 1, 10000)
    planned = {p["boundary_step"] for p in site["training_config"].get("resident_topology_switches", [])}
    while future in planned:
        future += 1
    if future >= spec["steps"]:
        return {**base, "reason": "insufficient remaining internal COMMIT boundaries", "committed_step": step}
    fleet = read(run / "ascend_fleet/state.json")
    if fleet.get("run_dir") != str(run) or not isinstance(fleet.get("jobs"), list):
        raise Unavailable("Fleet does not belong to this prepared run")
    all_domains = domains(site["job_layout"])
    domain_by_prefix = {d["prefix"]: d for d in all_domains}
    jobs, job_ids = {}, set()
    for job in fleet["jobs"]:
        prefix, job_id = job.get("worker_id"), job.get("job_id")
        if prefix not in domain_by_prefix or prefix in jobs or (job_id is not None and job_id in job_ids):
            raise Unavailable("Fleet identities exceed or duplicate the frozen physical pool")
        jobs[prefix] = job
        if job_id is not None:
            job_ids.add(job_id)
    active_domains, candidate_domains, observations, unavailable = [], [], {}, []
    covered = set()
    for domain in all_domains:
        worker_set = set(domain["workers"])
        intersection = worker_set.intersection(active)
        if intersection and intersection != worker_set:
            raise Unavailable("Active membership splits an allocation group; whole-job packing cannot retire it")
        is_active = bool(intersection)
        if domain["prefix"] not in jobs:
            if is_active:
                raise Unavailable("An active job is absent from the official fleet ledger")
            continue
        try:
            placement = validate_job(jobs[domain["prefix"]], domain, now=now, max_age=max_age)
            residents = [validate_worker(run, spec, worker, domain, local, placement, now=now, max_age=max_age)
                         for local, worker in enumerate(domain["workers"])]
            for resident in residents:
                worker = resident["worker_id"]
                integer(resident.get("completed_step"), "READY completed step", 0, step)
                record = topology.get("resident_records", {}).get(worker)
                if is_active:
                    if identity(resident) != active[worker] or not record or record.get("status") != "active" or identity(record) != active[worker]:
                        raise Unavailable("Active commit/READY/cohort identities disagree")
                elif not (resident.get("replacement") is True or (record and record.get("status") == "parked_stale" and identity(record) == identity(resident))):
                    raise Unavailable("Candidate was not independently loaded or explicitly parked")
            observations[domain["prefix"]] = placement
            if is_active:
                active_domains.append(domain)
                covered.update(worker_set)
            else:
                candidate_domains.append(domain)
        except (Unavailable, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            if is_active:
                raise Unavailable(f"Active group cannot be independently verified: {error}") from error
            unavailable.append({"allocation_group": domain["prefix"], "reason": str(error)})
    if covered != set(active):
        raise Unavailable("Active worker coverage differs from complete allocated jobs")
    for candidate in sorted(candidate_domains, key=lambda d: (-d["size"], d["prefix"])):
        retiring = best_retire(active_domains, candidate["size"])
        if len(retiring) < 2:
            continue
        retire = sorted(worker for domain in retiring for worker in domain["workers"])
        return {**base, "action": "recommend", "run_id": spec["run_id"], "config_sha256": spec["config_sha256"],
            "committed_step": step, "expected_step": future, "expected_generation": generation,
            "retire_workers": retire, "candidate_workers": candidate["workers"],
            "retire_allocation_groups": sorted(d["prefix"] for d in retiring),
            "candidate_allocation_group": candidate["prefix"], "active_launch_domains_before": len(active_domains),
            "active_launch_domains_after": len(active_domains) - len(retiring) + 1,
            "reason": "packing hypothesis: fewer active allocation domains with unchanged active card count",
            "authorized_physical_pool_cards": site["cards"], "active_cards": len(active),
            "budget_scope": "frozen registered pool; no new platform request", "observed_allocations": observations,
            "unavailable_candidates": unavailable}
    return {**base, "reason": "no ready whole-job candidate reduces active allocation domains",
            "committed_step": step, "unavailable_candidates": unavailable}


def request_slot(run, state):
    path = run / "probe/topology/switch_request.json"
    if not path.exists():
        return "empty", None
    current = read(path)
    if not isinstance(current.get("request_id"), str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", current["request_id"]) is None:
        return "foreign", current
    owned = state.get("requests", {}).get(current.get("request_id"))
    if not owned or owned.get("request_sha256") != fingerprint(current) or current.get("watcher_id") != state["watcher_id"]:
        return "foreign", current
    topology = read(run / "probe/topology/state.json")
    receipt = topology.get("requests", {}).get(current["request_id"])
    if receipt is None:
        return "pending", current
    if receipt.get("request_sha256") != owned["request_sha256"]:
        return "foreign", current
    return receipt.get("status", "unknown"), current


def success_steps(run, state):
    topology = read(run / "probe/topology/state.json")
    result = []
    for request_id, owned in state.get("requests", {}).items():
        receipt = topology.get("requests", {}).get(request_id, {})
        if receipt.get("status") == "committed" and receipt.get("request_sha256") == owned["request_sha256"]:
            result.append(integer(receipt.get("first_new_commit"), "first successful switched COMMIT", 1, 20000))
    return sorted(result)


def publish(run, proposal, state, *, state_path, max_attempts=8):
    """Called under the monitor's exclusive writer lock; never clobber a slot."""
    if proposal.get("action") != "recommend":
        return {"published": False, "reason": "no admissible packing recommendation"}
    if len(state["requests"]) >= max_attempts:
        return {"published": False, "reason": "bounded request attempts exhausted"}
    slot, current = request_slot(run, state)
    if slot not in {"empty", "rejected"}:
        return {"published": False, "reason": f"request slot occupied: {slot}"}
    # Recheck immediately before publication. A stale recommendation never
    # weakens the controller's exact step/generation/session fences.
    summary = read(run / "probe/summary.json")
    latest = summary.get("commits", [])[-1] if summary.get("commits") else {}
    if (summary.get("status") != "running" or summary.get("run_id") != proposal["run_id"]
            or latest.get("generation") != proposal["expected_generation"]
            or summary.get("committed_steps", 0) >= proposal["expected_step"]
            or any((run / marker).exists() for marker in ("STOP", "FATAL.json", "COMPLETE.json"))):
        return {"published": False, "reason": "recommendation became late or the controller changed"}
    request_id = f"cw-{state['watcher_id']}-{len(state['requests']) + 1}"
    request = {key: proposal[key] for key in ("run_id", "config_sha256", "expected_generation", "expected_step",
                                               "retire_workers", "candidate_workers")}
    request.update(request_id=request_id, watcher_id=state["watcher_id"], source="candidate_watch",
                   reason="packing hypothesis; no measured speedup claim")
    state["requests"][request_id] = {"request_sha256": fingerprint(request), "request": request,
                                    "status": "publication_intent", "at": time.time()}
    atomic_private(state_path, state)  # Intent survives an interrupted publication.
    path = run / "probe/topology/switch_request.json"
    if current is not None:
        # Only an exact, acknowledged rejected request from this watcher is
        # consumed. Preserve bytes first; never remove another writer's file.
        before = path.stat()
        raw = path.read_bytes()
        if fingerprint(json.loads(raw)) != fingerprint(current):
            return {"published": False, "reason": "request owner changed before rejected-request archival"}
        archive = state_path.parent / f"rejected-{current['request_id']}.json"
        if not archive.exists():
            descriptor = os.open(archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        elif archive.read_bytes() != raw:
            raise Unavailable("Rejected-request archive differs; refusing to consume the occupied slot")
        after = path.stat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or path.read_bytes() != raw:
            return {"published": False, "reason": "request owner changed during archival"}
        path.unlink()
    temporary = state_path.parent / f"publish-{request_id}.json"
    atomic_private(temporary, request)
    try:
        try:
            os.link(temporary, path)  # O_EXCL publication: another writer wins safely.
        except FileExistsError:
            return {"published": False, "reason": "request slot became occupied"}
        state["requests"][request_id]["status"] = "published"
        atomic_private(state_path, state)
        return {"published": True, "request_id": request_id, "expected_step": request["expected_step"],
                "retire_workers": request["retire_workers"], "candidate_workers": request["candidate_workers"]}
    finally:
        temporary.unlink(missing_ok=True)


def watch(run, *, seconds=600, interval=1.0, apply=False, max_switches=1, cooldown_steps=10,
          max_attempts=8, lead_steps=3, max_age=120):
    run = Path(run).expanduser().resolve()
    integer(seconds, "watch seconds", 1, 7 * 86400)
    integer(max_switches, "max successful switches", 1, 64)
    integer(cooldown_steps, "cooldown steps", 0, 20000)
    integer(max_attempts, "max attempts", 1, 1000)
    if not math.isfinite(interval) or not .05 <= interval <= 60 or not math.isfinite(max_age) or not 1 <= max_age <= 600:
        raise ValueError("Use interval 0.05..60 seconds and observation age 1..600 seconds")
    launch.verify(run)
    directory = run / "probe/topology/candidate_watch"
    state_path = directory / "state.json"
    lock = launch.writer_lock(directory) if apply else nullcontext()
    deadline, rounds = time.monotonic() + seconds, 0
    with lock:
        state = read(state_path) if apply and state_path.exists() else {
            "schema": SCHEMA, "watcher_id": uuid.uuid4().hex, "run_dir": str(run), "requests": {}}
        if (state.get("schema") != SCHEMA or state.get("run_dir") != str(run) or not isinstance(state.get("requests"), dict)
                or not isinstance(state.get("watcher_id"), str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", state["watcher_id"]) is None):
            raise Unavailable("Existing candidate monitor journal does not belong to this run")
        while time.monotonic() < deadline:
            try:
                proposal = advise(run, lead_steps=lead_steps, max_age=max_age)
                result = {"apply": apply, "proposal": proposal}
                if proposal["action"] == "terminal":
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    return {"status": "controller_terminal", "rounds": rounds, "requests": len(state["requests"])}
                if apply:
                    successes = success_steps(run, state)
                    if len(successes) >= max_switches:
                        return {"status": "success_limit", "successful_switches": len(successes), "rounds": rounds}
                    if successes and proposal.get("committed_step", 0) - successes[-1] < cooldown_steps:
                        result["publication"] = {"published": False, "reason": "successful-switch cooldown"}
                    else:
                        result["publication"] = publish(run, proposal, state, state_path=state_path, max_attempts=max_attempts)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
                print(json.dumps({"apply": apply, "action": "blocked_observation", "reason": str(error),
                                  "training_stop_requested": False}, ensure_ascii=False), flush=True)
            rounds += 1
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(interval, remaining))
    return {"status": "bounded_monitor_complete", "rounds": rounds, "requests": len(state["requests"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--apply", action="store_true", help="Publish fenced requests; default is read-only advice")
    parser.add_argument("--max-switches", type=int, default=1)
    parser.add_argument("--cooldown-steps", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--lead-steps", type=int, default=3)
    parser.add_argument("--max-age", type=float, default=120)
    args = parser.parse_args(argv)
    result = watch(args.run_dir, seconds=args.seconds, interval=args.interval, apply=args.apply,
                   max_switches=args.max_switches, cooldown_steps=args.cooldown_steps,
                   max_attempts=args.max_attempts, lead_steps=args.lead_steps, max_age=args.max_age)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
