"""Frozen-pool monitor admission; no platform requests, processes or devices."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import pytest

from fragment_training_kit import candidate_watch as watch


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def build_pool(tmp_path, layout=None):
    run = tmp_path / "run"
    run.mkdir()
    workers = [f"w{i:06d}" for i in range(1, 7)]
    config = {"batch_size": 256, "disable_activation_checkpointing": True}
    config_path = run / "training_config.json"
    write(config_path, config)
    site = {"schema": watch.launch.SCHEMA, "cards": 6, "active_cards": 4, "global_card_limit": 64,
            "job_layout": [1, 1, 1, 1, 2] if layout is None else layout, "steps": 400, "training_config": config}
    write(run / "site.json", site)
    write(run / "source_manifest.json", {"files": {
        "site.json": hashlib.sha256((run / "site.json").read_bytes()).hexdigest()}})
    spec = {"run_id": "run-id", "cards": 4, "topology_slots": 6, "workers": workers,
            "initial_workers": workers[:4], "run_dir": str(run / "probe"), "config": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(), "steps": 400,
            "heartbeat_timeout": 15}
    write(run / "probe/spec.json", spec)
    now = time.time()
    jobs, members, records = [], [], {}
    for domain in watch.domains(site["job_layout"]):
        prefix, size = domain["prefix"], domain["size"]
        from jade_lotus.ascend_platform import render_payload
        resources = {"gpu_count": size, "cpu_count": 10 * size, "memory_size_gib": 100 * size,
            "gpu_info": {"gpu_type": "ASCEND_910B_64G", "brand": "ASCEND"},
            "cpu_info": {"cpu_type": ""}, "quota_id": "fixture-quota", "total_price_per_hour": .35 * size}
        selection = {"workspace_id": watch.launch.WORKSPACES[1], "project_id": "fixture-project",
            "logic_compute_group_id": "fixture-group", "image": "registry.invalid/fixture:1",
            "image_type": "SOURCE_PUBLIC", "resource_spec_price": resources,
            "priority": {"verified": True, "level": "LOW", "name": "1", "task_priority": 1}}
        payload = render_payload(selection, "python fixture.py", f"fixture-{prefix}", 1800, 0)
        host = f"job-{prefix}-worker-0"
        observation = {"job_id": f"job-{prefix}", "worker_id": prefix, "status": "job_running",
            "priority_verified": True, "priority_level": "LOW", "priority_name": "1", "requested_cards": size,
            "time": now, "allocation_verified": False,
            "instances": [{"node": f"physical-{prefix}", "name": host, "instance_status": "instance_running"}]}
        jobs.append({"worker_id": prefix, "job_id": f"job-{prefix}", "status": "job_running",
            "official_terminal": False, "last_observation": observation,
            "payload": payload, "selection": selection})
        for local, worker in enumerate(domain["workers"]):
            member = {"worker_id": worker, "session_id": f"session-{worker}", "pid": 100 + int(worker[1:])}
            active = worker in workers[:4]
            ready = {**member, "run_id": spec["run_id"], "replacement": not active, "completed_step": 0,
                "allocation_group": prefix, "allocated_devices": size, "device_index": local, "hostname": host,
                "adapter_device": {"visible_devices": size, "allocated_devices": size, "device_index": local}}
            write(run / "probe/ready" / f"{worker}.json", ready)
            write(run / "probe/supervisors" / f"{worker}.json", {**member, "run_id": spec["run_id"], "heartbeat_counter": 2})
            if active:
                members.append(member)
                records[worker] = {**member, "status": "active"}
    write(run / "ascend_fleet/state.json", {"run_dir": str(run), "jobs": jobs})
    write(run / "probe/summary.json", {"run_id": spec["run_id"], "status": "running", "committed_steps": 10,
        "commits": [{"step": 10, "generation": 0, "members": members}]})
    write(run / "probe/topology/state.json", {"run_id": spec["run_id"], "physical_pool_slots": 6,
        "recovery_required": False, "pending_trial": None, "requests": {}, "resident_records": records})
    return run


@pytest.fixture
def pool(tmp_path):
    return build_pool(tmp_path)


def monitor_state(run):
    return {"schema": watch.SCHEMA, "watcher_id": "test-monitor", "run_dir": str(run), "requests": {}}


def test_read_only_proposal_preserves_rank_zero_and_replaces_whole_single_card_jobs(pool):
    before = sorted(str(p.relative_to(pool)) for p in pool.rglob("*"))
    proposal = watch.advise(pool)
    assert proposal["action"] == "recommend"
    assert proposal["retire_workers"] == ["w000003", "w000004"]
    assert proposal["candidate_workers"] == ["w000005", "w000006"]
    assert proposal["active_launch_domains_before"] == 4 and proposal["active_launch_domains_after"] == 3
    assert proposal["expected_step"] == 13 and proposal["expected_generation"] == 0
    assert proposal["reason_scope"].startswith("packing hypothesis")
    assert not proposal["allocation_requested"] and not proposal["resource_release_requested"]
    assert all(not row["physical_device_ids_verified"] for row in proposal["observed_allocations"].values())
    assert before == sorted(str(p.relative_to(pool)) for p in pool.rglob("*"))


@pytest.mark.parametrize("field,value", [("pid", True), ("device_index", False), ("allocated_devices", True),
                                        ("session_id", ""), ("hostname", "another-job")])
def test_candidate_numeric_or_process_identity_mismatch_cannot_switch(pool, field, value):
    path = pool / "probe/ready/w000005.json"
    ready = watch.read(path)
    ready[field] = value
    write(path, ready)
    result = watch.advise(pool)
    assert result["action"] == "wait" and result["unavailable_candidates"]


def test_active_session_mismatch_blocks_recommendation_without_training_side_effect(pool):
    path = pool / "probe/supervisors/w000003.json"
    value = watch.read(path)
    value["session_id"] = "new-session"
    write(path, value)
    with pytest.raises(watch.Unavailable, match="Active group"):
        watch.advise(pool)
    assert not (pool / "FATAL.json").exists() and not (pool / "STOP").exists()


def test_stale_official_candidate_placement_and_boolean_device_proof_are_rejected(pool):
    path = pool / "ascend_fleet/state.json"
    fleet = watch.read(path)
    fleet["jobs"][-1]["last_observation"]["time"] -= 1000
    write(path, fleet)
    assert watch.advise(pool)["action"] == "wait"
    fleet["jobs"][-1]["last_observation"]["time"] = time.time()
    write(path, fleet)
    ready_path = pool / "probe/ready/w000006.json"
    ready = watch.read(ready_path)
    ready["adapter_device"]["device_index"] = True  # bool must not masquerade as local rank 1.
    write(ready_path, ready)
    assert watch.advise(pool)["action"] == "wait"


def test_whole_job_subset_does_not_split_a_larger_active_domain():
    domains = watch.domains([4, 2, 1, 1])
    chosen = watch.best_retire(domains, 2)
    assert [d["size"] for d in chosen] == [1, 1]
    assert sum(d["size"] for d in chosen) == 2
    assert watch.best_retire(watch.domains([4]), 2) == []


def test_partial_active_job_is_rejected(pool):
    summary_path = pool / "probe/summary.json"
    summary = watch.read(summary_path)
    candidate = watch.identity(watch.read(pool / "probe/ready/w000005.json"))
    summary["commits"][-1]["members"][-1] = candidate
    write(summary_path, summary)
    topology_path = pool / "probe/topology/state.json"
    topology = watch.read(topology_path)
    topology["resident_records"].pop("w000004")
    topology["resident_records"]["w000005"] = {**candidate, "status": "active"}
    write(topology_path, topology)
    with pytest.raises(watch.Unavailable, match="splits an allocation group"):
        watch.advise(pool)


def test_no_candidate_or_pending_trial_cannot_produce_a_switch(pool):
    (pool / "probe/ready/w000006.json").unlink()
    assert watch.advise(pool)["action"] == "wait"
    path = pool / "probe/topology/state.json"
    value = watch.read(path)
    value["pending_trial"] = {"request_id": "existing"}
    write(path, value)
    assert "processing" in watch.advise(pool)["reason"]


def test_ready_candidate_without_domain_count_improvement_is_not_selected(tmp_path):
    run = build_pool(tmp_path, [2, 2, 2])
    result = watch.advise(run)
    assert result["action"] == "wait"
    assert "no ready whole-job candidate reduces" in result["reason"]


def test_unallocated_frozen_candidate_slot_is_an_ordinary_wait(pool):
    path = pool / "ascend_fleet/state.json"
    fleet = watch.read(path)
    fleet["jobs"].pop()
    write(path, fleet)
    assert watch.advise(pool)["action"] == "wait"


def test_frozen_fixed_plan_boundary_is_avoided(pool):
    path = pool / "site.json"
    site = watch.read(path)
    site["training_config"]["resident_topology_switches"] = [{"boundary_step": 13}]
    write(path, site)
    manifest = watch.read(pool / "source_manifest.json")
    manifest["files"]["site.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(pool / "source_manifest.json", manifest)
    assert watch.advise(pool)["expected_step"] == 14


def test_apply_publishes_exact_fenced_request_but_never_replaces_pending_or_completed(pool):
    state = monitor_state(pool)
    state_path = pool / "probe/topology/candidate_watch/state.json"
    proposal = watch.advise(pool)
    result = watch.publish(pool, proposal, state, state_path=state_path)
    assert result["published"]
    request_path = pool / "probe/topology/switch_request.json"
    original = request_path.read_bytes()
    request = watch.read(request_path)
    assert request["run_id"] == proposal["run_id"] and request["config_sha256"] == proposal["config_sha256"]
    assert request["expected_step"] == 13 and request["expected_generation"] == 0
    assert not watch.publish(pool, proposal, state, state_path=state_path)["published"]
    assert request_path.read_bytes() == original
    topology_path = pool / "probe/topology/state.json"
    topology = watch.read(topology_path)
    topology["requests"][request["request_id"]] = {"status": "committed", "first_new_commit": 14,
        "request_sha256": watch.fingerprint(request)}
    write(topology_path, topology)
    assert watch.success_steps(pool, state) == [14]
    assert not watch.publish(pool, proposal, state, state_path=state_path)["published"]
    assert request_path.read_bytes() == original


def test_rejected_own_request_is_archived_and_retry_has_new_id_and_bounded_attempts(pool):
    state = monitor_state(pool)
    state_path = pool / "probe/topology/candidate_watch/state.json"
    proposal = watch.advise(pool)
    first = watch.publish(pool, proposal, state, state_path=state_path, max_attempts=2)
    request_path = pool / "probe/topology/switch_request.json"
    request = watch.read(request_path)
    original = request_path.read_bytes()
    topology_path = pool / "probe/topology/state.json"
    topology = watch.read(topology_path)
    topology["requests"][request["request_id"]] = {"status": "rejected", "request_sha256": watch.fingerprint(request)}
    write(topology_path, topology)
    second = watch.publish(pool, proposal | {"expected_step": 16}, state, state_path=state_path, max_attempts=2)
    assert second["published"] and second["request_id"] != first["request_id"]
    assert (state_path.parent / f"rejected-{first['request_id']}.json").read_bytes() == original
    assert "exhausted" in watch.publish(pool, proposal, state, state_path=state_path, max_attempts=2)["reason"]


def test_foreign_writer_and_late_proposal_are_never_overwritten(pool):
    state, proposal = monitor_state(pool), watch.advise(pool)
    state_path = pool / "probe/topology/candidate_watch/state.json"
    request_path = pool / "probe/topology/switch_request.json"
    foreign = {"request_id": "human-request", "source": "someone-else"}
    write(request_path, foreign)
    original = request_path.read_bytes()
    assert "foreign" in watch.publish(pool, proposal, state, state_path=state_path)["reason"]
    assert request_path.read_bytes() == original and state["requests"] == {}
    request_path.unlink()
    summary_path = pool / "probe/summary.json"
    summary = watch.read(summary_path)
    summary["committed_steps"] = 13
    summary["commits"][-1]["step"] = 13
    write(summary_path, summary)
    assert "late" in watch.publish(pool, proposal, state, state_path=state_path)["reason"]
    assert not request_path.exists() and state["requests"] == {}


def test_terminal_controller_short_circuits_without_any_requests(pool):
    write(pool / "COMPLETE.json", {"passed": True})
    assert watch.advise(pool)["action"] == "terminal"
    assert not (pool / "probe/topology/candidate_watch").exists()


def test_default_watch_is_read_only_and_exits_for_controller_terminal(pool, monkeypatch):
    monkeypatch.setattr(watch.launch, "verify", lambda _run: {"verified": True})
    write(pool / "COMPLETE.json", {"passed": True})
    result = watch.watch(pool, seconds=1)
    assert result["status"] == "controller_terminal"
    assert not (pool / "probe/topology/candidate_watch").exists()


def test_watch_honors_success_limit_and_cooldown_without_another_publication(pool, monkeypatch):
    state = monitor_state(pool)
    state_path = pool / "probe/topology/candidate_watch/state.json"
    proposal = watch.advise(pool)
    first = watch.publish(pool, proposal, state, state_path=state_path)
    topology_path = pool / "probe/topology/state.json"
    topology = watch.read(topology_path)
    topology["requests"][first["request_id"]] = {"status": "committed", "first_new_commit": 14,
        "request_sha256": state["requests"][first["request_id"]]["request_sha256"]}
    write(topology_path, topology)
    monkeypatch.setattr(watch.launch, "verify", lambda _run: {"verified": True})
    monkeypatch.setattr(watch, "advise", lambda *_args, **_kw: proposal | {"committed_step": 15})
    monkeypatch.setattr(watch, "publish", lambda *_args, **_kw: pytest.fail("no second publication is allowed"))
    result = watch.watch(pool, seconds=1, apply=True, max_switches=1)
    assert result["status"] == "success_limit" and result["successful_switches"] == 1
    readings = iter([0, 0, 2, 2])
    monkeypatch.setattr(watch.time, "monotonic", lambda: next(readings))
    result = watch.watch(pool, seconds=1, apply=True, max_switches=2, cooldown_steps=10)
    assert result["rounds"] == 1


def test_modified_site_or_extra_fleet_identity_blocks_recommendation(pool):
    path = pool / "site.json"
    raw = path.read_bytes()
    path.write_bytes(raw + b"\n")
    with pytest.raises(watch.Unavailable, match="Frozen site"):
        watch.advise(pool)
    path.write_bytes(raw)
    fleet_path = pool / "ascend_fleet/state.json"
    fleet = watch.read(fleet_path)
    fleet["jobs"].append(deepcopy(fleet["jobs"][0]))
    write(fleet_path, fleet)
    with pytest.raises(watch.Unavailable, match="duplicate"):
        watch.advise(pool)
