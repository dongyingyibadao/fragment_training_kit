"""Finite official API doubles; topology entrypoints allocate no resources."""
import copy
import json
from pathlib import Path
import stat
import sys

import pytest

from fragment_training_kit import topology
from jade_lotus.ascend_nodes import NODE_API
from jade_lotus.ascend_platform import API, WORKSPACES, render_payload
from jade_lotus.platform import PlatformError


SPACE = WORKSPACES[1]
GROUP = "group-fixture"
USER = "user-fixture"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def selection(count, workspace=SPACE, group=GROUP):
    return {"workspace_id": workspace, "project_id": "project-fixture", "logic_compute_group_id": group,
        "image": "registry.invalid/fixture:v1", "image_type": "SOURCE_PRIVATE",
        "priority": {"verified": True, "level": "LOW", "name": "1", "task_priority": 1},
        "resource_spec_price": {"gpu_type": "ASCEND_910B_64G", "gpu_count": count,
            "cpu_count": count * 2, "memory_size_gib": count * 8, "quota_id": f"quota-{count}"}}


def payload(count, name="fixture-job"):
    return render_payload(selection(count), "python fixture.py", name, 1800, 0)


def official(request, identifier, status="job_running"):
    return dict(copy.deepcopy(request), job_id=identifier, status=status)


def raw_node(name="host-fixture"):
    return {"id": "node-fixture-" + name, "name": name, "cluster_name": "cluster-fixture",
        "status": "Ready", "resource_pool": "online", "gpu_type": "ASCEND_910B_64G",
        "gpu": {"available": 16, "total": 16}, "cpu": {"available": 185, "total": 185},
        "memory": {"available": 1987, "total": 1987, "unit": "GiB"}}


class Client:
    def __init__(self, jobs=(), *, fail_lists=0, details=None):
        self.jobs = list(jobs)
        self.fail_lists = fail_lists
        self.details = details or {}
        self.calls = []
        self.evidence = []
        self.after_nodes = None

    def request(self, method, path, body=None):
        self.calls.append((method, path, copy.deepcopy(body)))
        assert method == "POST"
        if path == API["user"]:
            return {"data": {"id": USER}}
        if path == API["list"]:
            if self.fail_lists:
                self.fail_lists -= 1
                raise PlatformError("official listing temporarily unavailable")
            rows = [j for j in self.jobs if j["workspace_id"] == body["workspace_id"]]
            return {"data": {"jobs": copy.deepcopy(rows), "total": len(rows)}}
        if path == API["detail"]:
            return {"data": copy.deepcopy(self.details.get(body["job_id"], {}))}
        if path == NODE_API:
            if self.after_nodes:
                self.after_nodes()
            return {"data": {"node_dimensions": [raw_node(body["filter"]["logic_compute_group_id"])], "total": 1}}
        raise AssertionError(f"Unexpected platform mutation or request: {path}")


@pytest.fixture
def site(tmp_path):
    cann, inv, registry = tmp_path / "cann.sh", tmp_path / "inventory.json", tmp_path / "registry.json"
    cann.write_text("true\n")
    selections = [selection(n) for n in (16, 8, 4, 2, 1)]
    write(inv, {"backend": "ascend", "user": {"id": USER}, "selections": selections,
        "groups": [{"workspace_id": SPACE, "id": GROUP, "resources": {"memory_gi_total": 1987}}],
        "workspaces": [{"workspace_id": SPACE, "schedule": {"train_enable_specified_nodes": False}}]})
    write(registry, {"schema_version": 1, "fleets": []})
    config = tmp_path / "site.json"
    write(config, {"schema": topology.launch.SCHEMA, "python": sys.executable, "cann_env": str(cann),
        "inventory": str(inv), "registry": str(registry), "workspace_ids": [SPACE], "cards": 4,
        "global_card_limit": 8, "job_layout": [2, 2], "active_cards": 2,
        "training_config": {"resident_adapter": "fixture:create_engine", "resident_rng_scheme": "fixture-rng",
            "resident_precision_contract": "fixture-FP32"}})
    result = topology.launch.resolve_site(config)
    result["config_path"] = str(config)
    return result


def add_fleet(site, tmp_path, counts=(2,), *, orphan=None):
    state = tmp_path / "registered/ascend_fleet/state.json"
    entries = []
    remote = []
    for index, count in enumerate(counts):
        request = payload(count, f"registered-{index}")
        record = state.parent / "submissions" / f"w{index + 1:06d}.json"
        entry = {"worker_id": record.stem, "record_path": str(record), "payload": request,
            "selection": selection(count), "workspace_id": SPACE, "group_id": GROUP,
            "job_id": f"job-registered-{index}", "status": "job_running"}
        entries.append(entry)
        remote.append(official(request, entry["job_id"]))
    write(state, {"schema_version": 1, "backend": "ascend", "user_id": USER, "jobs": entries})
    write(site["registry"], {"schema_version": 1, "fleets": [str(state)]})
    if orphan:
        request = payload(orphan, "orphan-fixture")
        write(state.parent / "submissions/orphan.json", {"request": request,
            "selection": selection(orphan), "status": "submission_unknown"})
    return state, remote


def test_real_shape_plan_accounts_registered_and_orphans_reports_external_without_debit(site, tmp_path):
    state, jobs = add_fleet(site, tmp_path, orphan=1)
    external = payload(16, "external-independent-project")
    external["framework_config"][0]["instance_count"] = 4
    client = Client(jobs + [official(external, "job-external-64")])
    before = {p: p.read_bytes() for p in (state, Path(site["registry"]), state.parent / "submissions/orphan.json")}
    plan, batch = topology.plan_once(client, site, tmp_path / "evidence")
    assert plan["budget_scope"] == "registered_fleets"
    assert plan["budget"]["reserved_cards"] == 3
    assert plan["budget"]["local_reserved_cards"] == 0
    assert plan["job_layout"] == [4]
    assert plan["external_jobs"] == [{"workspace_id": SPACE, "job_id": "job-external-64", "status": "job_running",
        "registered": False, "counted_in_budget": False, "requested_cards": 64}]
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in batch.iterdir())
    assert plan["capabilities"]["arbitrary_new_worker_ids_join"] is False
    assert plan["pinDisabled"] == [SPACE]
    assert not any(path in {API["create"], API["stop"]} for _, path, _ in client.calls)


@pytest.mark.parametrize("broken", ["missing_registry", "missing_state", "corrupt_orphan"])
def test_unreadable_durable_budget_inputs_refuse_planning_and_preserve_error(site, tmp_path, broken):
    state, jobs = add_fleet(site, tmp_path, orphan=1)
    if broken == "missing_registry":
        Path(site["registry"]).unlink()
    elif broken == "missing_state":
        state.unlink()
    else:
        (state.parent / "submissions/orphan.json").write_text("{")
    with pytest.raises(PlatformError):
        topology.plan_once(Client(jobs), site, tmp_path / "errors")
    batches = list((tmp_path / "errors").glob("plan-*"))
    assert len(batches) == 1 and (batches[0] / "error.json").exists()
    assert not (batches[0] / "plan.json").exists()


def test_ledger_change_during_nodes_query_invalidates_round(site, tmp_path):
    state, jobs = add_fleet(site, tmp_path)
    client = Client(jobs)
    def change():
        value = json.loads(state.read_text())
        value["new_reservation_pending"] = True
        write(state, value)
    client.after_nodes = change
    with pytest.raises(PlatformError, match="changed during"):
        topology.plan_once(client, site, tmp_path / "unstable")
    batch = next((tmp_path / "unstable").glob("plan-*"))
    assert (batch / "nodes.json").exists() and (batch / "error.json").exists()


def test_scope_filters_all_inventory_sections_but_checks_both_budget_workspaces(site, tmp_path):
    inv = json.loads(Path(site["inventory"]).read_text())
    inv["selections"] += [selection(1, WORKSPACES[0], "other-space"), selection(1, SPACE, "excluded-group")]
    inv["groups"] += [{"workspace_id": WORKSPACES[0], "id": "other-space"},
                      {"workspace_id": SPACE, "id": "excluded-group"}]
    inv["workspaces"].append({"workspace_id": WORKSPACES[0], "schedule": {"train_enable_specified_nodes": False}})
    write(site["inventory"], inv)
    site["allowed_group_ids"] = [GROUP]
    client = Client()
    _, batch = topology.plan_once(client, site, tmp_path / "scoped")
    filtered = json.loads((batch / "inventory.json").read_text())
    assert {g["id"] for g in filtered["groups"]} == {GROUP}
    assert {s["logic_compute_group_id"] for s in filtered["selections"]} == {GROUP}
    assert [w["workspace_id"] for w in filtered["workspaces"]] == [SPACE]
    assert {body["workspace_id"] for _, path, body in client.calls if path == API["list"]} == set(WORKSPACES)
    assert all(body["filter"]["logic_compute_group_id"] == GROUP for _, path, body in client.calls if path == NODE_API)


def test_prepare_freezes_exact_planned_choices_and_removes_only_temporary_site(site, tmp_path, monkeypatch):
    observed = {}
    def prepare(config_path, run):
        observed["config_path"] = Path(config_path)
        observed["site"] = json.loads(Path(config_path).read_text())
        assert not Path(run).exists()
        assert Path(observed["site"]["inventory"]).exists()
        return {"prepared": True, "submitted": False}
    monkeypatch.setattr(topology.launch, "prepare", prepare)
    plan, result = topology.prepare_once(Client(), site, tmp_path / "prepared-evidence", tmp_path / "new-run")
    assert result["submitted"] is False
    assert observed["site"]["job_layout"] == [4]
    assert observed["site"]["active_cards"] == 2
    assert observed["site"]["job_selections"] == [p["selection"] for p in plan["placements"]]
    assert not observed["config_path"].exists()
    assert Path(observed["site"]["inventory"]).exists()


def test_incomplete_layout_cannot_prepare_partial_training_run(site, tmp_path, monkeypatch):
    _, jobs = add_fleet(site, tmp_path, counts=(4, 2))
    monkeypatch.setattr(topology.launch, "prepare", lambda *args: pytest.fail("partial layout must not prepare"))
    with pytest.raises(PlatformError, match="complete initial card layout"):
        topology.prepare_once(Client(jobs), site, tmp_path / "partial", tmp_path / "new-run")
    assert not (tmp_path / "new-run").exists()
    batch = next((tmp_path / "partial").glob("plan-*"))
    assert json.loads((batch / "plan.json").read_text())["job_layout"] == [2]
    assert (batch / "prepare-error.json").exists()


class Clock:
    def __init__(self):
        self.current = 0

    def monotonic(self):
        return self.current

    def sleep(self, duration):
        assert 0 < duration <= 60
        self.current += duration


def prepare_watch(site, tmp_path, monkeypatch):
    run = tmp_path / "watch-run"
    run.mkdir()
    write(run / "site.json", site)
    monkeypatch.setattr(topology.launch, "verify", lambda path: {"verified": True})
    monkeypatch.setattr(topology.launch, "resolve_site", lambda path: copy.deepcopy(site))
    clock = Clock()
    monkeypatch.setattr(topology.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(topology.time, "sleep", clock.sleep)
    for name in ("start_controller", "stop_owned_controller", "cleanup"):
        monkeypatch.setattr(topology.launch, name, lambda *a, **k: pytest.fail("monitor must not control training lifecycle"))
    return run


def test_watch_defaults_readonly_keeps_evidence_and_continues_after_api_error(site, tmp_path, monkeypatch):
    run = prepare_watch(site, tmp_path, monkeypatch)
    monkeypatch.setattr(topology.launch, "snapshot", lambda *a, **k: pytest.fail("read-only watch cannot reconcile"))
    result = topology.watch(Client(fail_lists=1), run, tmp_path / "watch-evidence", seconds=3, interval=1)
    assert len(result["rounds"]) == 3 and result["error_rounds"] == 1
    assert result["fill_calls"] == 0 and result["training_stop_requested"] is False
    assert not (run / "STOP").exists() and not (run / "switch_request.json").exists()
    assert len(list((tmp_path / "watch-evidence").glob("plan-*/plan.json"))) == 2
    assert len(list((tmp_path / "watch-evidence").glob("plan-*/error.json"))) == 1


def test_fill_waits_for_successful_observation_and_only_fills_missing_fixed_slots(site, tmp_path, monkeypatch):
    run = prepare_watch(site, tmp_path, monkeypatch)
    calls = []
    client = Client(fail_lists=1)
    def snapshot(client_arg, target, *, fill):
        assert target == run and fill is True
        calls.append(fill)
        entries = []
        for index in range(2):
            request = payload(2, f"fixed-slot-{index}")
            entries.append({"record_path": str(run / f"ascend_fleet/submissions/w{index + 1:06d}.json"),
                "payload": request, "selection": selection(2), "status": "planned"})
        write(run / "ascend_fleet/state.json", {"backend": "ascend", "user_id": USER, "jobs": entries})
        return {"created": ["fixed-slot-1", "fixed-slot-2"]}
    monkeypatch.setattr(topology.launch, "snapshot", snapshot)
    result = topology.watch(client, run, tmp_path / "filled", seconds=3, interval=1, fill=True)
    assert calls == [True]
    assert result["error_rounds"] == 1 and result["fill_calls"] == 1
    assert result["capabilities"]["arbitrary_new_worker_ids_join"] is False
    assert result["switch_request_written"] is False


def test_account_scope_counts_external_jobs_and_default_scope_reports_unknown_footprints(site, tmp_path):
    external = official(payload(16, "external"), "external-fixture")
    external["framework_config"][0]["instance_count"] = 4
    account = dict(site, budget_scope="account")
    plan, _ = topology.plan_once(Client([external]), account, tmp_path / "account")
    assert plan["budget"]["reserved_cards"] == 64 and plan["job_layout"] == []
    assert plan["budget"]["already_over_limit"] is True
    unknown = {"workspace_id": SPACE, "job_id": "unknown-external", "status": "job_running"}
    default, _ = topology.plan_once(Client([unknown]), site, tmp_path / "default")
    assert default["budget"]["reserved_cards"] == 0
    assert default["external_jobs"][0]["requested_cards"] is None
    assert default["external_jobs"][0]["counted_in_budget"] is False
    with pytest.raises(PlatformError, match="Account budget cannot resolve"):
        topology.plan_once(Client([unknown]), account, tmp_path / "unknown-account")


def test_account_budget_cannot_prepare_an_executable_run(site, tmp_path, monkeypatch):
    account = dict(site, budget_scope="account")
    client = Client()
    monkeypatch.setattr(topology.launch, "prepare", lambda *a: pytest.fail("account execution must fail closed"))
    with pytest.raises(ValueError, match="account supports read-only planning only"):
        topology.prepare_once(client, account, tmp_path / "account-prepare", tmp_path / "new-run")
    assert client.calls == [] and not (tmp_path / "new-run").exists()


def test_account_watch_rejects_fill_before_observation_or_mutation(site, tmp_path, monkeypatch):
    account = dict(site, budget_scope="account")
    run = prepare_watch(account, tmp_path, monkeypatch)
    client = Client()
    monkeypatch.setattr(topology.launch, "snapshot", lambda *a, **kw: pytest.fail("account fill must fail closed"))
    with pytest.raises(ValueError, match="account supports read-only planning only"):
        topology.watch(client, run, tmp_path / "account-fill", seconds=1, interval=1, fill=True)
    assert client.calls == []
    assert not (run / "STOP").exists() and not (run / "ascend_fleet").exists()


def test_account_watch_retains_readonly_planning(site, tmp_path, monkeypatch):
    account = dict(site, budget_scope="account")
    run = prepare_watch(account, tmp_path, monkeypatch)
    monkeypatch.setattr(topology.launch, "snapshot", lambda *a, **kw: pytest.fail("read-only watch cannot submit"))
    result = topology.watch(Client(), run, tmp_path / "account-readonly", seconds=1, interval=1)
    assert result["error_rounds"] == 0 and result["fill_calls"] == 0
    assert result["capabilities"]["account_budget_execution"] is False


def test_evidence_batches_are_private_immutable_and_latest_is_only_a_pointer(site, tmp_path):
    output = tmp_path / "history"
    _, first = topology.plan_once(Client(), site, output)
    contents = {p.name: p.read_bytes() for p in first.iterdir()}
    _, second = topology.plan_once(Client(), site, output)
    assert first != second
    assert all((first / name).read_bytes() == value for name, value in contents.items())
    assert json.loads((output / "latest.json").read_text())["directory"] == second.name
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "latest.json").stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="outside the project"):
        topology._output(topology.launch.KIT / "must-not-create-evidence")


def test_cli_prints_only_layout_budget_and_pin_summary_without_copying_auth(site, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(topology, "make_client", lambda **kwargs: Client())
    result = topology.main(["plan", "--config", site["config_path"], "--output", str(tmp_path / "cli"),
                            "--cookie-file", "/private/not-opened.cookies"])
    printed = json.loads(capsys.readouterr().out)
    assert result == 0 and set(printed) == {"job_layout", "budget", "pinDisabled"}
    assert not any("cookie" in p.name for p in (tmp_path / "cli").rglob("*"))
