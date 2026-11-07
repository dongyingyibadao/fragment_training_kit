"""Synthetic official shapes: no site inventory, credentials or device imports."""
import copy

import pytest

from fragment_training_kit import launch as _launcher  # Select this kit's vendor.
from jade_lotus.ascend_nodes import (NODE_API, NODE_SCHEMA, discover_nodes, iso_time,
    normalize_node_observations)
from jade_lotus.ascend_platform import WORKSPACES
from jade_lotus.platform import PlatformError
from jade_lotus.topology_placement import plan_topology, reservation_ledger


NOW = 1_800_000_000
SPACE = WORKSPACES[1]
GROUP = "lcg-fixture"


def selection(cards, workspace=SPACE, group=GROUP, **spec_changes):
    spec = {"gpu_count": cards, "gpu_type": "ASCEND_910B_64G", "cpu_count": cards * 2,
        "memory_size_gib": cards * 8, "quota_id": f"quota-fixture-{cards}"}
    spec.update(spec_changes)
    return {"workspace_id": workspace, "project_id": "project-fixture",
        "logic_compute_group_id": group, "image": "registry.invalid/fixture:v1",
        "image_type": "SOURCE_PRIVATE", "resource_spec_price": spec,
        "priority": {"verified": True, "level": "LOW", "name": "1", "task_priority": 1}}


def inventory(selections=None):
    selections = selections if selections is not None else [selection(n) for n in (16, 8, 4, 2, 1)]
    scopes = {(s["workspace_id"], s["logic_compute_group_id"]) for s in selections}
    return {"backend": "ascend", "user": {"id": "user-fixture"}, "selections": selections,
        "groups": [{"workspace_id": w, "id": g, "resources": {}} for w, g in sorted(scopes)],
        "workspaces": [{"workspace_id": w, "schedule": {"train_enable_specified_nodes": False}}
                       for w in sorted({w for w, _ in scopes})]}


def node(name, cards, *, total=16, cpus=128, memory=2048, status="Ready"):
    return {"id": "node-id-" + name, "name": name, "cluster_name": "cluster-fixture",
        "status": status, "resource_pool": "online", "gpu_type": "ASCEND_910B_64G",
        "gpu_info": {"gpu_type": "ASCEND_910B_64G", "brand": "ASCEND"},
        "gpu": {"available": cards, "total": total},
        "cpu": {"available": cpus, "total": 256},
        "memory": {"available": memory, "total": 4096, "unit": "GiB"}}


def observation(row, workspace=SPACE, group=GROUP, **changes):
    value = {"node": row, "workspace_id": workspace, "logic_compute_group_id": group,
        "observed_at": iso_time(NOW), "query_status": "fresh", "query_complete": True}
    value.update(changes)
    return value


def snapshot(observations=(), *, at=NOW):
    return {"schema": NODE_SCHEMA, "source": NODE_API, "observed_at": iso_time(at),
        "nodes": normalize_node_observations(observations, now=NOW),
        "capabilities": {w: {"enabled": False} for w in WORKSPACES}}


def plan(inv, snap, reservations=(), **kwargs):
    return plan_topology(inv, snap, reservations, global_card_limit=kwargs.pop("global_card_limit", 31),
                         now=NOW, **kwargs)


def test_fullest_nodes_and_largest_legal_jobs_make_16_8_4_2_1_plan():
    snap = snapshot([observation(node(f"host-{count}", count)) for count in (1, 2, 4, 8, 16)])
    result = plan(inventory(), snap)
    assert result["job_layout"] == [16, 8, 4, 2, 1]
    assert result["placements"][0]["reason"] == "idle_whole_node"
    assert all(p["reason"] == "idle_fragment" for p in result["placements"][1:])
    assert all(p["allocation_verified"] is False and p["binding"]["status"] == "unbound"
               and p["binding"]["request_fields"] == {} for p in result["placements"])
    assert result["budget"]["total_reserved_and_planned"] == 31
    assert result["submitted_jobs"] == 0


def test_cpu_and_memory_independently_limit_the_largest_fitting_specification():
    snap = snapshot([observation(node("cpu-limited", 16, cpus=8)),
                     observation(node("memory-limited", 8, memory=16))])
    result = plan(inventory(), snap, global_card_limit=16, queue_when_idle_exhausted=False)
    assert result["job_layout"] == [4, 2]
    assert result["budget"]["remaining_cards"] == 10


def test_deferred_candidate_waits_for_commit_and_fresh_idle_cpu_memory(monkeypatch, tmp_path):
    from jade_lotus import ascend_nodes, topology_placement
    monkeypatch.setattr(topology_placement.time, "time", lambda: NOW)
    site = {"job_layout": [1, 1, 1, 1, 2], "active_cards": 4}
    inv = inventory()
    rows = [observation(node("candidate-host", 2, cpus=3))]
    calls = []
    def discover(client, supplied):
        calls.append(client)
        return snapshot(rows)
    monkeypatch.setattr(ascend_nodes, "discover_nodes", discover)
    reserved = [{"selection": selection(1), "cards": 1, "status": "job_running"} for _ in range(4)]
    pre_commit = _launcher.deferred_layout_planner("client", tmp_path, site, 4, training_started=False)
    assert pre_commit(inv, reserved, 2, {}, NOW) == [] and calls == []
    live = _launcher.deferred_layout_planner("client", tmp_path, site, 4, training_started=True)
    assert live(inv, reserved, 2, {}, NOW) == []  # Two NPUs alone do not prove enough CPU.
    rows[:] = [observation(node("candidate-host", 2, cpus=4))]
    result = live(inv, reserved, 2, {}, NOW)
    assert [p["cards"] for p in result] == [2]
    assert result[0]["binding"]["status"] == "unbound"
    assert live(inv, reserved, 1, {}, NOW) == []  # Cannot downsize a frozen two-card identity.


def test_deferred_candidate_query_failure_does_not_stop_training(monkeypatch, tmp_path):
    import json
    from jade_lotus import ascend_nodes
    def unavailable(*args):
        raise PlatformError("node endpoint unavailable")
    monkeypatch.setattr(ascend_nodes, "discover_nodes", unavailable)
    site = {"job_layout": [2, 2], "active_cards": 2}
    callback = _launcher.deferred_layout_planner(None, tmp_path, site, 1, training_started=True)
    assert callback(inventory(), [], 2, {}, NOW) == []
    evidence = json.loads((tmp_path / "topology_allocation/latest.json").read_text())
    assert evidence["training_stop_requested"] is False and evidence["placements"] == []


@pytest.mark.parametrize("error_type", [TypeError, AttributeError, RuntimeError])
def test_malformed_optional_node_query_only_defers_candidates(monkeypatch, tmp_path, error_type):
    from jade_lotus import ascend_nodes
    def malformed(*args):
        raise error_type("malformed optional node observation")
    monkeypatch.setattr(ascend_nodes, "discover_nodes", malformed)
    site = {"job_layout": [2, 2], "active_cards": 2}
    callback = _launcher.deferred_layout_planner(None, tmp_path, site, 1, training_started=True)
    assert callback(inventory(), [], 2, {}, NOW) == []
    evidence = _launcher.read(tmp_path / "topology_allocation/latest.json")
    assert evidence["training_stop_requested"] is False
    assert evidence["error"].startswith(error_type.__name__)


def test_deferred_diagnostic_write_failure_neither_allocates_nor_stops(monkeypatch, tmp_path):
    from jade_lotus import ascend_nodes, topology_placement
    monkeypatch.setattr(topology_placement.time, "time", lambda: NOW)
    monkeypatch.setattr(ascend_nodes, "discover_nodes", lambda *args: snapshot([observation(node("free", 2))]))
    def cannot_write(*args):
        raise PermissionError("only optional evidence directory is unwritable")
    monkeypatch.setattr(_launcher, "write", cannot_write)
    callback = _launcher.deferred_layout_planner(None, tmp_path,
        {"job_layout": [2, 2], "active_cards": 2}, 1, training_started=True)
    assert callback(inventory(), [], 2, {}, NOW) == []


def test_deferred_candidates_do_not_reuse_one_full_node_for_two_jobs(monkeypatch, tmp_path):
    from jade_lotus import ascend_nodes, topology_placement
    monkeypatch.setattr(topology_placement.time, "time", lambda: NOW)
    nodes = snapshot([observation(node("full", 16)), observation(node("half-a", 8)),
                      observation(node("half-b", 8))])
    monkeypatch.setattr(ascend_nodes, "discover_nodes", lambda *args: nodes)
    site = {"job_layout": [2, 16, 16], "active_cards": 2}
    reserved = [{"selection": selection(2), "cards": 2, "status": "job_running"}]
    callback = _launcher.deferred_layout_planner(None, tmp_path, site, 1, training_started=True)
    result = callback(inventory(), reserved, 32, {}, NOW)
    assert [row["cards"] for row in result] == [16]
    assert result[0]["expected_node_name"] == "full"


def test_next_candidate_waits_until_previous_candidate_is_officially_running(monkeypatch, tmp_path):
    from jade_lotus import ascend_nodes
    def forbidden(*args):
        pytest.fail("Do not reuse free-node evidence while the earlier candidate is still pending")
    monkeypatch.setattr(ascend_nodes, "discover_nodes", forbidden)
    site = {"job_layout": [2, 16, 16], "active_cards": 2}
    reserved = [{"selection": selection(16), "cards": 16, "status": "job_pending",
        "fleet_paths": {str(tmp_path / "ascend_fleet/state.json")}, "entry": {"worker_id": "w000002"}}]
    callback = _launcher.deferred_layout_planner(None, tmp_path, site, 2, training_started=True)
    assert callback(inventory(), reserved, 16, {}, NOW) == []
    evidence = _launcher.read(tmp_path / "topology_allocation/latest.json")
    assert "previous candidate" in evidence["reason"]


def test_deferred_initial_allocation_excludes_candidate_slots(monkeypatch, tmp_path):
    from jade_lotus import ascend_nodes
    def forbidden(*args):
        raise AssertionError("Initial active allocation must not wait for candidate node discovery")
    monkeypatch.setattr(ascend_nodes, "discover_nodes", forbidden)
    inv = inventory()
    inv["groups"][0].update(resource_query={"status": "fresh"},
        resources={"gpu_available": 16, "cpu_available": 128, "memory_gi_available": 2048})
    site = {"job_layout": [2, 2, 8], "active_cards": 4}
    callback = _launcher.deferred_layout_planner(None, tmp_path, site, 0, training_started=False)
    assert [p["cards"] for p in callback(inv, [], 12, {}, NOW)] == [2, 2]


def test_shared_physical_node_is_counted_once_across_workspaces_and_uses_minimum_free():
    observations = [observation(node("shared", 16), workspace=WORKSPACES[0], group="group-a"),
                    observation(node("shared", 12), workspace=WORKSPACES[1], group="group-b")]
    snap = snapshot(observations)
    selections = [selection(16, WORKSPACES[0], "group-a")]
    selections += [selection(n, WORKSPACES[1], "group-b") for n in (16, 8, 4, 2, 1)]
    result = plan(inventory(selections), snap, global_card_limit=32)
    idle = [p for p in result["placements"] if p["expected_node_id"]]
    assert len(snap["nodes"]) == 1 and len(snap["nodes"][0]["memberships"]) == 2
    assert [p["cards"] for p in idle] == [8, 4]
    assert sum(p["cards"] for p in idle) == 12
    assert result["budget"]["planned_cards"] == 32


@pytest.mark.parametrize("condition", ["unknown_status", "not_ready", "backup", "unknown_npu",
    "unknown_cpu", "unknown_memory", "incomplete", "failed", "stale", "future"])
def test_unknown_nonready_incomplete_or_old_nodes_only_produce_queue_intentions(condition):
    raw = node("unusable", 16)
    obs = observation(raw)
    if condition == "unknown_status":
        raw.pop("status")
    elif condition == "not_ready":
        raw["status"] = "NotReady"
    elif condition == "backup":
        raw["resource_pool"] = "backup"
    elif condition.startswith("unknown_"):
        key = {"unknown_npu": "gpu", "unknown_cpu": "cpu", "unknown_memory": "memory"}[condition]
        raw[key].pop("available")
    elif condition == "incomplete":
        obs["query_complete"] = False
    elif condition == "failed":
        obs["query_status"] = "error"
    elif condition == "stale":
        obs["observed_at"] = iso_time(NOW - 121)
    else:
        obs["observed_at"] = iso_time(NOW + 10)
    snap = snapshot([obs])
    result = plan(inventory(), snap, global_card_limit=1)
    assert snap["nodes"][0]["eligible"] is False
    assert result["job_layout"] == [1]
    assert result["placements"][0]["reason"] == "queue_low_priority_single_card"
    assert result["placements"][0]["expected_node_id"] is None


def test_stale_snapshot_does_not_reuse_individually_fresh_node_rows():
    snap = snapshot([observation(node("once-free", 16))], at=NOW - 121)
    result = plan(inventory(), snap, global_card_limit=2)
    assert result["job_layout"] == [1, 1]
    assert result["diagnostics"][0]["reason"] == "stale_or_unknown_node_snapshot"


def test_unitless_memory_uses_a_conservative_auditable_lower_bound():
    raw = node("unitless", 1, memory=100)
    raw["memory"]["unit"] = ""
    normalized = snapshot([observation(raw)])["nodes"][0]
    assert normalized["available_memory_gib"] == pytest.approx(100 * 10 ** 9 / 2 ** 30)
    assert normalized["observations"][0]["memory_policy"] == "GB_lower_bound"
    assert normalized["observations"][0]["source_fields"]["memory"]["available"] == 100


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def request(self, method, path, body):
        self.calls.append((method, path, copy.deepcopy(body)))
        scope = body["filter"]
        result = self.responses[(scope["workspace_id"], scope["logic_compute_group_id"], body["page_num"])]
        if isinstance(result, Exception):
            raise result
        return {"data": copy.deepcopy(result), "metadata": {"Action": "ListNodeDimension", "Service": "workspace"}}


def test_discovery_scopes_every_page_and_deduplicates_shared_nodes_with_unit_crosscheck():
    first, second = node("shared", 16), node("other", 4)
    for raw in (first, second):
        raw["memory"]["unit"] = ""
    inv = inventory([selection(1, WORKSPACES[0], "group-a"), selection(1, WORKSPACES[1], "group-b")])
    for group in inv["groups"]:
        group["resources"]["memory_gi_total"] = 8192 if group["id"] == "group-a" else 4096
    client = FakeClient({(WORKSPACES[0], "group-a", 1): {"node_dimensions": [first], "total": 2},
                         (WORKSPACES[0], "group-a", 2): {"node_dimensions": [second], "total": 2},
                         (WORKSPACES[1], "group-b", 1): {"node_dimensions": [first], "total": 1}})
    result = discover_nodes(client, inv, page_size=1, now=NOW)
    assert len(result["nodes"]) == 2 and result["errors"] == []
    assert len(client.calls) == 3
    assert all(method == "POST" and path == NODE_API and body["filter"]["resource_pool"] == "online"
               for method, path, body in client.calls)
    assert all(query["complete"] for query in result["queries"])
    assert all(query["memory_unit"] == "GiB_crosschecked_group_total" for query in result["queries"])
    assert all(value["enabled"] is False for value in result["capabilities"].values())


def test_failed_second_page_preserves_first_page_evidence_without_counting_it_as_idle():
    client = FakeClient({(SPACE, GROUP, 1): {"node_dimensions": [node("partial", 16)], "total": 2},
                         (SPACE, GROUP, 2): PlatformError("read-only API unavailable")})
    result = discover_nodes(client, inventory(), page_size=1, now=NOW)
    assert result["queries"][0]["complete"] is False
    assert result["nodes"][0]["eligible"] is False
    assert result["nodes"][0]["observations"][0]["source_fields"]["gpu"]["available"] == 16
    assert plan(inventory(), result, global_card_limit=1)["placements"][0]["expected_node_id"] is None


def test_changed_pagination_total_cannot_produce_ready_capacity():
    client = FakeClient({(SPACE, GROUP, 1): {"node_dimensions": [node("first", 16)], "total": 2},
                         (SPACE, GROUP, 2): {"node_dimensions": [node("second", 16)], "total": 3}})
    result = discover_nodes(client, inventory(), page_size=1, now=NOW)
    assert result["errors"] and not any(n["eligible"] for n in result["nodes"])


def test_one_budget_counts_active_candidate_hot_spare_and_unknown_terminal_once():
    reserved = [
        {"workspace_id": SPACE, "group_id": GROUP, "job_id": "active-fixture", "cards": 2, "status": "job_running"},
        {"workspace_id": SPACE, "group_id": GROUP, "reservation_id": "candidate-fixture", "category": "candidate",
         "selection": selection(4), "status": "planned"},
        {"workspace_id": SPACE, "group_id": GROUP, "reservation_id": "spare-fixture", "category": "hot_spare",
         "selection": selection(1), "status": "planned"},
        {"workspace_id": SPACE, "group_id": GROUP, "job_id": "unknown-terminal", "cards": 1, "status": "job_failed"},
        {"workspace_id": SPACE, "group_id": GROUP, "job_id": "confirmed-terminal", "cards": 16,
         "status": "job_succeeded", "official_terminal": True}]
    reserved.append(copy.deepcopy(reserved[1]))
    result = plan(inventory(), snapshot([observation(node("free", 16))]), reserved, global_card_limit=10)
    assert result["budget"]["reserved_cards"] == 8
    assert result["budget"]["reserved_by_category"] == {"active": 3, "candidate": 4, "hot_spare": 1}
    assert result["budget"]["planned_cards"] == 2
    assert result["budget"]["total_reserved_and_planned"] == 10


def test_over_limit_reservations_block_all_new_intentions_without_silent_release():
    result = plan(inventory(), snapshot([observation(node("free", 16))]),
        [{"workspace_id": SPACE, "group_id": GROUP, "job_id": "job-fixture", "cards": 16, "status": "job_running"}],
        global_card_limit=8)
    assert result["placements"] == []
    assert result["budget"]["already_over_limit"] is True
    assert result["budget"]["reserved_cards"] == 16


def test_conflicting_duplicate_job_footprints_are_rejected():
    rows = [{"workspace_id": SPACE, "group_id": GROUP, "job_id": "same-job", "cards": n,
             "status": "job_running"} for n in (1, 2)]
    with pytest.raises(PlatformError, match="conflicting physical footprint"):
        reservation_ledger(rows)


def test_record_and_job_aliases_deduplicate_a_previously_unknown_submission():
    unresolved = {"workspace_id": SPACE, "group_id": GROUP, "record_path": "/fixture/submissions/w1.json",
                  "selection": selection(2), "status": "submission_unknown"}
    resolved = dict(unresolved, job_id="official-fixture", status="job_running")
    assert sum(row["cards"] for row in reservation_ledger([unresolved, resolved])) == 2


def test_without_legal_single_card_specification_the_planner_does_not_invent_one():
    result = plan(inventory([selection(16)]), snapshot(), global_card_limit=16)
    assert result["placements"] == []
    assert result["diagnostics"][0]["reason"] == "no_legal_single_card_queue_specification"


def test_pending_footprints_reduce_group_capacity_before_more_idle_intentions():
    inv = inventory()
    snap = snapshot([observation(node("small-host", 4, total=4, cpus=8, memory=32))])
    pending = {"reservation_id": "queued-fixture", "workspace_id": SPACE, "group_id": GROUP,
               "selection": selection(2), "category": "candidate", "status": "job_queuing"}
    result = plan(inv, snap, [pending], global_card_limit=6)
    assert result["job_layout"] == [2, 1, 1]
    assert result["placements"][0]["expected_node_id"]
    assert all(p["expected_node_id"] is None for p in result["placements"][1:])
    assert result["budget"]["total_reserved_and_planned"] == 6


def test_planning_preserves_inventory_snapshot_and_reservation_inputs():
    inv = inventory()
    snap = snapshot([observation(node("free-host", 16))])
    reserved = [{"workspace_id": SPACE, "group_id": GROUP, "cards": 1, "job_id": "active-fixture",
                 "status": "job_running"}]
    before = copy.deepcopy((inv, snap, reserved))
    result = plan(inv, snap, reserved, global_card_limit=8)
    assert (inv, snap, reserved) == before
    result["placements"][0]["selection"]["priority"]["name"] = "changed"
    assert inv["selections"][0]["priority"]["name"] == "1"


def test_same_host_with_conflicting_physical_ids_is_not_double_counted_as_capacity():
    a, b = node("same-host", 16), node("same-host", 16)
    b["id"] = "different-node-id"
    snap = snapshot([observation(a), observation(b, workspace=WORKSPACES[0])])
    assert not any(n["eligible"] for n in snap["nodes"])
    assert all("conflicting_physical_ids_for_same_host" in n["unavailable_reasons"] for n in snap["nodes"])
