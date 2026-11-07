#!/usr/bin/env python3
"""Plan and prepare QZ topology intentions; watch fixed initial slots only.

Planning and preparation submit nothing. Watch is read-only unless --fill is
explicit. New arbitrary worker IDs cannot join the current training cohort;
this monitor never writes switch_request or stops a training controller.
Account-wide budgets support read-only planning only; execution is rejected
until account reservations can be enforced inside the allocation lock.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

if __package__:
    from . import launch
else:
    import launch

from jade_lotus.ascend_fleet import cards
from jade_lotus.ascend_nodes import discover_nodes, iso_time
from jade_lotus.ascend_platform import (API, TERMINAL, WORKSPACES, job_id, list_own_jobs,
    make_client, npu_type, public, resolve_submission, validate_payload)
from jade_lotus.platform import PlatformError, write_json
from jade_lotus.topology_placement import plan_topology, reservation_ledger


CAPABILITIES = {"fixed_initial_slots_only": True, "arbitrary_new_worker_ids_join": False,
    "automatic_switch_request": False, "monitor_errors_stop_training": False,
    "plans_are_allocations": False, "account_budget_execution": False}
ROUND_SECONDS = 60


def _read(path):
    try:
        raw = Path(path).read_bytes()
        return json.loads(raw), hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError):
        raise PlatformError(f"Required planning input is unreadable: {path}") from None


def _private_once(path, value):
    path = Path(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(public(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _output(path):
    result = Path(path).expanduser().resolve()
    if result.is_relative_to(launch.KIT.resolve()):
        raise ValueError("Topology evidence output must be outside the project directory")
    result.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not result.is_dir():
        raise ValueError("Topology output must be a directory")
    return result


def _batch(output, prefix="plan"):
    stamp = iso_time(time.time()).replace(":", "").replace("+", "_")
    result = output / f"{prefix}-{stamp}-{uuid.uuid4().hex[:10]}"
    result.mkdir(mode=0o700)
    return result


def _latest(output, batch, status):
    # Only the pointer changes. Every evidence file in a batch is exclusive.
    write_json(output / "latest.json", {"directory": batch.name, "status": status,
        "recorded_at": iso_time(time.time())}, private=True)


class DeadlineClient:
    """Bound a read round between requests; HTTP itself has a finite timeout."""
    def __init__(self, client, deadline):
        self.client, self.deadline = client, deadline

    def request(self, method, path, body=None):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Topology round reached its API deadline")
        result = self.client.request(method, path, body)
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Topology round reached its API deadline")
        return result


def filter_inventory(inventory, site):
    """Apply the same explicit placement scope to groups, choices and spaces."""
    if (not isinstance(inventory, dict) or inventory.get("backend") != "ascend"
            or not isinstance(inventory.get("user"), dict) or not inventory["user"].get("id")
            or not isinstance(inventory.get("selections"), list)):
        raise PlatformError("An authenticated Ascend discovery inventory is required")
    for key in ("groups", "selections", "workspaces"):
        rows = inventory.get(key, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise PlatformError(f"Ascend inventory has an invalid {key} collection")
    workspaces = set(site["workspace_ids"])
    groups = site.get("allowed_group_ids")
    if groups is not None and (not isinstance(groups, list) or not groups
            or any(not isinstance(gid, str) or not gid for gid in groups) or len(groups) != len(set(groups))):
        raise ValueError("allowed_group_ids must be a nonempty list of unique explicit group IDs")
    allowed = None if groups is None else set(groups)
    result = copy.deepcopy(inventory)
    result.pop("evidence", None)
    result["groups"] = [g for g in result.get("groups", []) if g.get("workspace_id") in workspaces
                        and (allowed is None or g.get("id") in allowed)]
    scopes = {(g["workspace_id"], g["id"]) for g in result["groups"]}
    # An older inventory may omit its group summaries. A selected immutable
    # specification still supplies an explicit scope for node discovery.
    result["selections"] = [s for s in result["selections"] if s.get("workspace_id") in workspaces
        and (allowed is None or s.get("logic_compute_group_id") in allowed)]
    for selection in result["selections"]:
        scope = (selection["workspace_id"], selection["logic_compute_group_id"])
        if scope not in scopes:
            result["groups"].append({"workspace_id": scope[0], "id": scope[1], "resources": {}})
            scopes.add(scope)
    if not result["selections"]:
        raise PlatformError("The configured workspace/group scope has no legal Ascend selections")
    result["workspaces"] = [w for w in result.get("workspaces", []) if w.get("workspace_id") in workspaces]
    return result


def _submission_files(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    if not directory.is_dir() or directory.is_symlink():
        raise PlatformError("Durable submissions must be an ordinary readable directory")
    try:
        entries = list(directory.iterdir())
        result = []
        for path in entries:
            if path.suffix != ".json":
                continue
            if path.is_symlink() or not path.is_file():
                raise PlatformError("Durable submission JSON must be an ordinary readable file")
            result.append(path)
        return sorted(result)
    except OSError:
        raise PlatformError("Durable submission directory is unreadable") from None


def _registered_states(registry_path, user_id, run_dir=None):
    """Read durable intent and orphan records without taking a writing lock."""
    registry_path = Path(registry_path).resolve()
    registry, digest = _read(registry_path)
    if not isinstance(registry, dict) or not isinstance(registry.get("fleets"), list):
        raise PlatformError("The shared registry must contain its complete fleets list")
    tracked = {str(registry_path): digest}
    paths = []
    for value in registry["fleets"]:
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise PlatformError("Every registered fleet state path must be absolute")
        path = Path(value).resolve()
        if path not in paths:
            paths.append(path)
    current = Path(run_dir).resolve() / "ascend_fleet/state.json" if run_dir else None
    absent_current = current is not None and not current.exists()
    if absent_current and _submission_files(current.parent / "submissions"):
        raise PlatformError("Current-run orphan submissions exist without their durable fleet state")
    if current is not None and not absent_current and current not in paths:
        paths.append(current)
    directories, rows = {}, []
    for state_path in paths:
        state, digest = _read(state_path)
        tracked[str(state_path)] = digest
        if (not isinstance(state, dict) or state.get("backend") != "ascend"
                or state.get("user_id") != user_id or not isinstance(state.get("jobs"), list)):
            raise PlatformError("Registered Ascend state has an unknown owner or incomplete reservations")
        entries = copy.deepcopy(state["jobs"])
        by_record = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("record_path"), str):
                raise PlatformError("Registered intent has no durable record path")
            record_path = Path(entry["record_path"])
            if not record_path.is_absolute():
                raise PlatformError("Durable submission record paths must be absolute")
            record_path = record_path.resolve()
            if record_path.parent != state_path.parent / "submissions":
                raise PlatformError("A registered submission record escaped its fleet directory")
            if str(record_path) in by_record:
                raise PlatformError("A fleet repeats the same durable submission record")
            by_record[str(record_path)] = entry
        submission_dir = state_path.parent / "submissions"
        records = _submission_files(submission_dir)
        directories[str(submission_dir)] = [str(p.resolve()) for p in records]
        for record_path in records:
            record, digest = _read(record_path)
            tracked[str(record_path.resolve())] = digest
            if not isinstance(record, dict) or not isinstance(record.get("request"), dict):
                raise PlatformError("Orphan or registered submission record is malformed")
            selection = record.get("selection")
            if not isinstance(selection, dict):
                raise PlatformError("A durable submission lacks its immutable selection")
            validate_payload(record["request"], selection)
            key = str(record_path.resolve())
            entry = by_record.get(key)
            if entry is None:
                entry = {"record_path": key, "payload": record["request"], "selection": selection,
                    "status": record.get("status", "submission_unknown"), "job_id": job_id(record),
                    "recovered_orphan_record": True}
                entries.append(entry)
            else:
                if entry.get("payload") != record["request"] or entry.get("selection") != selection:
                    raise PlatformError("Durable state and submission contract disagree")
                if entry.get("job_id") and job_id(record) and entry["job_id"] != job_id(record):
                    raise PlatformError("Durable state and submission job identities disagree")
                entry["job_id"] = entry.get("job_id") or job_id(record)
                entry["status"] = record.get("status", entry.get("status", "unknown"))
            entry["record"] = record
        for entry in entries:
            payload, selection = entry.get("payload"), entry.get("selection")
            if not isinstance(payload, dict) or not isinstance(selection, dict):
                raise PlatformError("Every durable intent must retain its full request and selection")
            validate_payload(payload, selection)
            if entry.get("workspace_id", payload["workspace_id"]) != payload["workspace_id"]:
                raise PlatformError("Durable intent workspace differs from its immutable request")
            if entry.get("group_id", payload["logic_compute_group_id"]) != payload["logic_compute_group_id"]:
                raise PlatformError("Durable intent group differs from its immutable request")
            entry["fleet_path"] = str(state_path)
            entry["local"] = current == state_path
            rows.append(entry)

    def unchanged():
        for path, previous in tracked.items():
            if _read(path)[1] != previous:
                raise PlatformError("The shared reservation ledger changed during this planning round; retry")
        for directory, previous in directories.items():
            current_files = [str(p.resolve()) for p in _submission_files(directory)]
            if current_files != previous:
                raise PlatformError("A durable submission appeared during planning; retry")
        if absent_current and current.exists():
            raise PlatformError("The current fleet state appeared during planning; retry")
        if absent_current and _submission_files(current.parent / "submissions"):
            raise PlatformError("A current-run orphan submission appeared during planning; retry")

    return rows, tracked, unchanged


def _official_footprint(client, row):
    """Use explicit official request resources; do not infer cards from names."""
    detail = row
    if not isinstance(detail.get("framework_config"), list):
        detail = client.request("POST", API["detail"], {"job_id": row["job_id"]}).get("data")
        if not isinstance(detail, dict):
            raise PlatformError("Official active job has no inspectable resource request")
        if detail.get("workspace_id", row["workspace_id"]) != row["workspace_id"]:
            raise PlatformError("Official job detail belongs to another workspace")
    configs = detail.get("framework_config")
    if not isinstance(configs, list) or not configs:
        raise PlatformError("Official active job has no complete resource footprint")
    types, counts = [], []
    for component in configs:
        if not isinstance(component, dict):
            raise PlatformError("Official active job has an invalid component")
        spec = component.get("resource_spec_price") or component.get("instance_spec_price_info") or {}
        if not isinstance(spec, dict):
            raise PlatformError("Official active job has an invalid resource specification")
        types.append(npu_type(spec))
        counts.append(component.get("gpu_count"))
    if all(value == 0 and type(value) is int for value in counts):
        return None, detail, 0
    if not all(types):
        raise PlatformError("Official active job Ascend resource identity is unknown")
    payload = {"workspace_id": row["workspace_id"],
        "logic_compute_group_id": detail.get("logic_compute_group_id") or row.get("logic_compute_group_id"),
        "framework_config": copy.deepcopy(configs)}
    count = cards(payload)
    # Also validate the CPU/memory footprint used to hold pending capacity.
    reservation_ledger([{"payload": payload, "cards": count}])
    return payload, detail, count


def shared_reservations(client, site, inventory, *, run_dir=None):
    """Reconcile registered fleets; report external jobs under an explicit scope."""
    scope = site.get("budget_scope", "registered_fleets")
    if scope not in {"registered_fleets", "account"}:
        raise ValueError("budget_scope must be registered_fleets or account")
    user = client.request("POST", API["user"], {}).get("data") or {}
    user_id = inventory["user"]["id"]
    if user.get("id") != user_id:
        raise PlatformError("Authenticated account differs from the inventory/shared registry owner")
    entries, tracked, unchanged = _registered_states(site["registry"], user_id, run_dir)
    official = list_own_jobs(client, user_id, WORKSPACES)
    by_job = {}
    for row in official:
        identity = (row["workspace_id"], row.get("job_id"))
        if not identity[1] or identity in by_job:
            raise PlatformError("Official job listing has missing or duplicate identities")
        by_job[identity] = row
    rows, registered, current_details = [], set(), {}
    for entry in entries:
        payload = entry["payload"]
        jid = entry.get("job_id")
        if not jid and entry.get("status") not in {"submission_rejected", "cancelled_before_submission"}:
            resolved = resolve_submission({"request": payload}, official)
            jid = job_id(resolved)
        identity = (payload["workspace_id"], jid)
        current = by_job.get(identity)
        if jid:
            registered.add(identity)
        terminal = entry.get("official_terminal") is True
        status = entry.get("status", "unknown")
        if current:
            status = current.get("status", "unknown")
            terminal = status in TERMINAL
            for key in ("workspace_id", "logic_compute_group_id", "project_id", "name", "command"):
                if current.get(key) is not None and current[key] != payload[key]:
                    raise PlatformError("Official job identity differs from its durable request")
            if not terminal:
                if identity not in current_details:
                    current_details[identity] = _official_footprint(client, current)
                _, _, official_count = current_details[identity]
                if official_count != cards(payload):
                    raise PlatformError("Official job card footprint differs from the registered request")
        rows.append({"workspace_id": payload["workspace_id"], "group_id": payload["logic_compute_group_id"],
            "job_id": jid, "record_path": entry["record_path"], "payload": payload,
            "selection": entry["selection"], "cards": cards(payload), "status": status,
            "official_terminal": terminal, "local": entry["local"],
            "category": entry.get("category", entry.get("role", "active"))})
    external = []
    for identity, row in by_job.items():
        if identity in registered or row.get("status") in TERMINAL:
            continue
        report = {"workspace_id": identity[0], "job_id": identity[1], "status": row.get("status", "unknown"),
            "registered": False, "counted_in_budget": scope == "account", "requested_cards": None}
        try:
            payload, detail, count = _official_footprint(client, row)
            report["requested_cards"] = count
            if scope == "account" and count:
                rows.append({"workspace_id": identity[0], "group_id": payload["logic_compute_group_id"],
                    "job_id": identity[1], "payload": payload, "cards": count,
                    "status": detail.get("status", row.get("status", "unknown")), "local": False})
        except (PlatformError, OSError, TimeoutError, ValueError) as error:
            report["footprint_error"] = str(error)
            if scope == "account":
                raise PlatformError("Account budget cannot resolve an unregistered active job footprint: "
                                    + identity[0] + "/" + identity[1]) from None
        external.append(report)
    unchanged()
    ledger = reservation_ledger(rows)
    evidence = {"budget_scope": scope, "registered_state_and_record_sha256": tracked,
        "official_workspaces_checked": list(WORKSPACES), "external_jobs": external,
        "reserved_cards": sum(row["cards"] for row in ledger),
        "local_reserved_cards": sum(row["cards"] for row in ledger if row["local"]),
        "reservations": [{key: value for key, value in row.items() if key != "identities"} for row in ledger]}
    return rows, evidence, unchanged


def plan_once(client, site, output, *, run_dir=None, deadline=None):
    output = _output(output)
    batch = _batch(output)
    bounded = DeadlineClient(client, deadline if deadline is not None else time.monotonic() + ROUND_SECONDS)
    try:
        inventory, _ = _read(site["inventory"])
        inventory = filter_inventory(inventory, site)
        _private_once(batch / "inventory.json", inventory)
        reserved, reservation_evidence, unchanged = shared_reservations(bounded, site, inventory, run_dir=run_dir)
        _private_once(batch / "reservations.json", reservation_evidence)
        nodes = discover_nodes(bounded, inventory)
        _private_once(batch / "nodes.json", nodes)
        if nodes.get("errors"):
            raise PlatformError("Node discovery was incomplete; this round cannot prepare or fill jobs")
        unchanged()
        plan = plan_topology(inventory, nodes, reserved, global_card_limit=site["global_card_limit"],
                             local_card_limit=site["cards"])
        plan["budget"]["budget_scope"] = reservation_evidence["budget_scope"]
        if reservation_evidence["budget_scope"] == "account":
            plan["budget"]["account_workspaces_checked"] = list(WORKSPACES)
        plan.update(budget_scope=reservation_evidence["budget_scope"],
            external_jobs=reservation_evidence["external_jobs"], capabilities=copy.deepcopy(CAPABILITIES),
            pinDisabled=sorted(wid for wid, value in nodes.get("capabilities", {}).items()
                               if value.get("enabled") is False),
            evidence_directory=str(batch), reservation_observation_complete=True)
        _private_once(batch / "plan.json", plan)
        _latest(output, batch, "planned_intentions")
        return plan, batch
    except Exception as error:
        _private_once(batch / "error.json", {"recorded_at": iso_time(time.time()),
            "error": f"{type(error).__name__}: {error}", "submitted_jobs": 0,
            "capabilities": CAPABILITIES})
        _latest(output, batch, "error")
        try:
            error.topology_evidence_directory = str(batch)
        except AttributeError:
            pass
        raise


def _require_registered_execution(site):
    if site.get("budget_scope", "registered_fleets") == "account":
        raise ValueError("budget_scope=account supports read-only planning only; preparation and --fill "
                         "are disabled until account reservations are enforced atomically at allocation")


def prepare_once(client, site, output, run_dir):
    _require_registered_execution(site)
    run = Path(run_dir).expanduser().resolve()
    if run.exists():
        raise ValueError("Refusing to overwrite an existing run; use a fresh shared directory")
    plan, batch = plan_once(client, site, output, run_dir=run)
    temporary = None
    try:
        if plan["budget"]["local_reserved_cards"] != 0 or sum(plan["job_layout"]) != site["cards"]:
            raise PlatformError("Preparation requires zero local reservations and a complete initial card layout")
        effective = copy.deepcopy(site)
        effective.update(inventory=str(batch / "inventory.json"), job_layout=plan["job_layout"],
            active_cards=site.get("active_cards", site["cards"]),
            job_selections=[placement["selection"] for placement in plan["placements"]])
        descriptor, temporary = tempfile.mkstemp(prefix=".effective-site-", suffix=".json", dir=batch)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(effective, stream, ensure_ascii=False, allow_nan=False)
        result = launch.prepare(temporary, run)
        _private_once(batch / "prepare.json", {"result": result, "capabilities": CAPABILITIES,
            "allocation_verified": False, "submitted_jobs": 0})
        return plan, result
    except Exception as error:
        _private_once(batch / "prepare-error.json", {"error": f"{type(error).__name__}: {error}",
            "submitted_jobs": 0, "capabilities": CAPABILITIES})
        raise
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def summary(plan):
    return {key: plan[key] for key in ("job_layout", "budget", "pinDisabled")}


def watch(client, run_dir, output, *, seconds, interval, fill=False):
    if type(seconds) is not int or not 1 <= seconds <= 7 * 86400:
        raise ValueError("seconds must be in 1..604800")
    if type(interval) is not int or not 1 <= interval <= 60:
        raise ValueError("interval must be in 1..60")
    run, output = Path(run_dir).expanduser().resolve(), _output(output)
    launch.verify(run)
    site = launch.resolve_site(run / "site.json")
    if fill:
        _require_registered_execution(site)
    session = _batch(output, "watch")
    deadline = time.monotonic() + seconds
    rounds, errors, filled = [], 0, 0
    try:
        while time.monotonic() < deadline:
            batch = None
            try:
                plan, batch = plan_once(client, site, output, run_dir=run,
                    deadline=min(deadline, time.monotonic() + ROUND_SECONDS))
                state = run / "ascend_fleet/state.json"
                jobs = _read(state)[0]["jobs"] if state.exists() else []
                if not isinstance(jobs, list) or any(not isinstance(job, dict) for job in jobs):
                    raise PlatformError("Current fixed-slot state changed into an invalid job inventory")
                layout = site["job_layout"]
                terminal = any((run / name).exists() for name in ("STOP", "FATAL.json", "COMPLETE.json"))
                status = {"fill_requested": fill, "filled": False,
                    "created_initial_intents": len(jobs), "initial_intent_limit": len(layout),
                    "capabilities": CAPABILITIES}
                if fill and not terminal and len(jobs) < len(layout) and plan["budget"]["available_before_plan"] > 0:
                    launch.verify(run)
                    status["snapshot"] = launch.snapshot(
                        DeadlineClient(client, min(deadline, time.monotonic() + ROUND_SECONDS)), run, fill=True)
                    status["filled"] = True
                    filled += 1
                _private_once(batch / "watch.json", status)
                print(json.dumps(summary(plan), ensure_ascii=False), flush=True)
                rounds.append({"directory": batch.name, "status": "planned_intentions", "filled": status["filled"]})
            except Exception as error:
                errors += 1
                if batch is None and getattr(error, "topology_evidence_directory", None):
                    batch = Path(error.topology_evidence_directory)
                if batch is not None:
                    _private_once(batch / "watch-error.json", {"error": f"{type(error).__name__}: {error}",
                        "capabilities": CAPABILITIES, "training_stop_requested": False})
                rounds.append({"directory": batch.name if batch else None, "status": "error", "error": str(error)})
                print(json.dumps({"job_layout": [], "budget": {"planning_blocked": True}, "pinDisabled": []}), flush=True)
            finally:
                evidence = getattr(client, "evidence", None)
                if isinstance(evidence, list):
                    evidence.clear()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(interval, remaining))
    finally:
        receipt = {"seconds": seconds, "interval": interval, "fill": fill, "rounds": rounds,
            "error_rounds": errors, "fill_calls": filled, "capabilities": CAPABILITIES,
            "training_stop_requested": False, "switch_request_written": False}
        _private_once(session / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("plan", "prepare", "watch"):
        action = actions.add_parser(name)
        action.add_argument("--output", required=True, type=Path, help="Private evidence directory outside this project")
        auth = action.add_mutually_exclusive_group(required=True)
        auth.add_argument("--cookie-file")
        auth.add_argument("--credentials")
        if name != "watch":
            action.add_argument("--config", required=True, type=Path)
        action.add_argument("--run-dir", required=name in {"prepare", "watch"}, type=Path)
        if name == "watch":
            action.add_argument("--seconds", type=int, default=3600)
            action.add_argument("--interval", type=int, default=30)
            action.add_argument("--fill", action="store_true", help="Create only missing fixed initial slots; no arbitrary new worker IDs")
    args = parser.parse_args(argv)
    try:
        client = make_client(credentials=args.credentials, cookie_file=args.cookie_file)
        if args.action == "watch":
            watch(client, args.run_dir, args.output, seconds=args.seconds, interval=args.interval, fill=args.fill)
        else:
            site = launch.resolve_site(args.config)
            if args.action == "prepare":
                plan, _ = prepare_once(client, site, args.output, args.run_dir)
            else:
                plan, _ = plan_once(client, site, args.output, run_dir=args.run_dir)
            print(json.dumps(summary(plan), ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"action": args.action, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
