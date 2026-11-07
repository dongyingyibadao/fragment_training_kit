"""Pure local/mock tests for the bounded benchmark fleet, with no submissions."""
from contextlib import nullcontext
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from fragment_training_kit import launch
from fragment_training_kit.examples import benchmark_topology_collective as benchmark
from fragment_training_kit.examples import run_topology_benchmark as fleet


@pytest.fixture
def prepared_benchmark(tmp_path):
    cann = tmp_path / "cann.sh"
    cann.write_text("export PYTHONPATH=/cann/python:$PYTHONPATH\n")
    inventory = tmp_path / "inventory.json"
    selection = {"workspace_id": launch.WORKSPACES[0], "project_id": "benchmark-fixture",
        "logic_compute_group_id": "group-fixture", "image": "image-fixture", "image_type": "SOURCE_PRIVATE",
        "priority": {"verified": True, "level": "LOW", "name": "1", "task_priority": 1},
        "resource_spec_price": {"gpu_count": 1, "gpu_type": "ASCEND_910B", "quota_id": "quota-fixture",
                                "cpu_count": 4, "memory_size_gib": 16}}
    inventory.write_text(json.dumps({"backend": "ascend", "user": {"id": "fixture"}, "selections": [selection]}))
    site = {"schema": launch.SCHEMA, "python": sys.executable, "cann_env": str(cann),
            "inventory": str(inventory), "registry": str(tmp_path / "registry.json"),
            "cards": 2, "job_layout": [1, 1], "global_card_limit": 64, "steps": 20,
            "training_config": {"resident_adapter": "fixture:create_engine", "resident_rng_scheme": "fixture-rng",
                                "resident_precision_contract": "native FP32 model and moments"}}
    config = tmp_path / "site.json"
    config.write_text(json.dumps(site))
    plan = {"schema": fleet.SCHEMA, "modes": ["flat", "hierarchy", "flat"],
            "payload": [{"dtype": "bfloat16", "bytes": 34}, {"dtype": "float32", "bytes": 20}],
            "rdma_timeouts": [16, 12, 16], "warmup": 0, "iterations": 2,
            "fleet_timeout_seconds": 60, "poll_seconds": .01,
            "terminal_grace_seconds": .01, "cleanup_wait_seconds": .1}
    benchmark_config = tmp_path / "benchmark.json"
    benchmark_config.write_text(json.dumps(plan))
    run = tmp_path / "shared/benchmark-run"
    result = fleet.prepare_benchmark(config, benchmark_config, run, api=launch)
    return run, config, benchmark_config, result


def test_prepare_disables_training_and_freezes_all_benchmark_inputs(prepared_benchmark):
    run, config, benchmark_config, result = prepared_benchmark
    assert result["submitted"] is False and result["benchmark"] and not result["training"]
    assert result["report_count"] == 6 and result["cases_per_rank"] == 9
    assert not (run / "controller_process.json").exists() and not (run / "ascend_fleet").exists()
    assert launch.verify(run)["verified"]
    plan = fleet.verify_benchmark(run, api=launch)
    assert fleet.read(run / "controller_command.json")["disabled"]
    assert fleet.read(run / "benchmark_payload.json")["buckets"] == plan["payload"]
    shell = (run / "launch.sh").read_text()
    assert "run_topology_benchmark.py\" worker" in shell
    assert "probe_resident_openpi_fault.py" not in shell
    assert "ASCEND_VISIBLE_DEVICES=" not in shell and "ASCEND_RT_VISIBLE_DEVICES=" not in shell
    assert (run / "fleet_command.txt").read_text().strip().endswith("{npu_count}")
    benchmark_config.write_text('{"changed_after_prepare": true}')
    config.write_text("{}")
    assert fleet.verify_benchmark(run, api=launch)["payload"] == plan["payload"]


def test_runtime_bootstrap_accepts_only_explicit_source_or_frozen_layout(tmp_path):
    source = tmp_path / "kit"
    (source / "vendor/jade_lotus").mkdir(parents=True)
    (source / "vendor/jade_lotus/topology_collective.py").write_text("# fixture")
    assert benchmark.runtime_root(source) == source / "vendor"
    frozen = tmp_path / "run/runtime/fragment_training_kit"
    frozen.mkdir(parents=True)
    (frozen.parent / "jade_lotus").mkdir()
    (frozen.parent / "jade_lotus/topology_collective.py").write_text("# fixture")
    assert benchmark.runtime_root(frozen) == frozen.parent
    (frozen.parent / "jade_lotus/topology_collective.py").unlink()
    with pytest.raises(RuntimeError, match="no source fallback"):
        benchmark.runtime_root(frozen)


def test_frozen_plan_only_executes_without_mother_project_pythonpath(prepared_benchmark, tmp_path):
    run, _, _, _ = prepared_benchmark
    script = run / "runtime/fragment_training_kit/examples/benchmark_topology_collective.py"
    environment = dict(os.environ, PYTHONPATH=str(run / "runtime"), TORCH_DEVICE_BACKEND_AUTOLOAD="0",
                       PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, str(script), "--plan-only", "--device", "cpu", "--world-size", "2",
               "--topology", str(run / "benchmark_topology.json"),
               "--payload-json", str(run / "benchmark_payload.json"), "--output-dir", str(tmp_path / "unused")]
    result = subprocess.run(command, env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["devices_used"] is False
    assert not (tmp_path / "unused").exists()
    (run / "runtime/jade_lotus/topology_collective.py").unlink()
    missing = subprocess.run(command, env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert missing.returncode != 0 and "no source fallback" in missing.stderr


@pytest.mark.parametrize("layout,prefix,allocated,expected", [
    ([16, 16], "w000002", 16, [(rank, rank - 16) for rank in range(16, 32)]),
    ([1] * 32, "w000032", 1, [(31, 0)]),
    ([16, 8, 1], "w000003", 1, [(24, 0)]),
])
def test_pod_prefix_owns_exact_contiguous_global_ranks(layout, prefix, allocated, expected):
    assert fleet.worker_ranks(layout, prefix, allocated) == expected


@pytest.mark.parametrize("layout,prefix,allocated", [([16, 16], "w000003", 16),
                                                     ([16, 16], "w000001", 1),
                                                     ([1] * 65, "w000001", 1),
                                                     ([1], "w000000", 1),
                                                     ([True], "w000001", 1)])
def test_invalid_ownership_or_overbudget_layout_is_rejected(layout, prefix, allocated):
    with pytest.raises(ValueError):
        fleet.worker_ranks(layout, prefix, allocated)


def test_rank_commands_share_one_rendezvous_and_timeout_sequence(prepared_benchmark):
    run, _, _, _ = prepared_benchmark
    plan = fleet.read(run / "benchmark_config.json")
    rendezvous = {"master_addr": "10.0.0.7", "master_port": 31234}
    commands = [fleet.benchmark_command(run, plan, rendezvous, rank, 0, 1) for rank in range(2)]
    for rank, command in enumerate(commands):
        assert command[command.index("--master-port") + 1] == "31234"
        assert command[command.index("--rdma-timeouts") + 1] == "16,12,16"
        assert command[command.index("--rank") + 1] == str(rank)
        assert command[command.index("--expected-visible-devices") + 1] == "1"
        assert command[2].startswith(str(run / "runtime"))


def test_only_rank_zero_publishes_one_atomic_consecutive_port_range(tmp_path, monkeypatch):
    run = tmp_path / "run"
    binds, closed = [], []
    class Socket:
        def bind(self, address):
            binds.append(address)
        def getsockname(self):
            return ("10.0.0.7", 31234)
        def close(self):
            closed.append(True)
    monkeypatch.setattr(fleet.socket, "gethostname", lambda: "pod-zero")
    monkeypatch.setattr(fleet.socket, "getaddrinfo", lambda *args: [(0, 0, 0, "", ("10.0.0.7", 0))])
    monkeypatch.setattr(fleet.socket, "socket", lambda *args: Socket())
    plan = {"plan_id": "fixed-plan", "rdma_timeouts": [16, 12, 16], "rendezvous_timeout_seconds": .1}
    record = fleet.publish_rendezvous(run, plan)
    assert record["master_port"] == 31234
    assert binds == [("10.0.0.7", 0), ("10.0.0.7", 31235), ("10.0.0.7", 31236)]
    assert len(closed) == 3 and fleet.await_rendezvous(run, plan) == record
    with pytest.raises(FileExistsError):
        fleet.publish_rendezvous(run, plan)
    assert fleet.read(run / "benchmark_rendezvous.json") == record


def test_all_pod_readiness_is_checked_before_collective_rendezvous(tmp_path):
    run = tmp_path / "run"
    plan = {"plan_id": "fixed-plan", "job_layout": [16, 16], "rendezvous_timeout_seconds": .1}
    for index, ranks in enumerate(fleet.layout_domains(plan["job_layout"])):
        fleet.atomic_json(run / "benchmark_ready" / f"w{index + 1:06d}.json", {
            "plan_id": plan["plan_id"], "ranks": ranks, "allocated_devices": 16}, once=True)
    assert len(fleet.await_all_pods(run, plan)) == 2
    path = run / "benchmark_ready/w000002.json"
    row = fleet.read(path)
    row["allocated_devices"] = 1
    fleet.atomic_json(path, row)
    with pytest.raises(ValueError, match="launch-domain ownership"):
        fleet.await_all_pods(run, plan)


def test_supervisor_uses_identical_incremented_ports_for_all_timeout_points(tmp_path, monkeypatch):
    args = SimpleNamespace(output_dir=str(tmp_path / "results"), rank=3, master_port=31000,
                           hard_timeout_seconds=1)
    calls = []
    class Child:
        pid = 112233
        def wait(self, *, timeout):
            assert timeout == 1
            return 0
    def popen(command, *, env, start_new_session):
        assert start_new_session
        calls.append((command, env["HCCL_RDMA_TIMEOUT"]))
        return Child()
    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark", "--output-dir", str(tmp_path / "results"),
                        "--master-port", "31000", "--rdma-timeouts", "16,12,16"])
    assert benchmark._supervise(args, [16, 12, 16]) == 0
    assert [row[0][row[0].index("--master-port") + 1] for row in calls] == ["31000", "31001", "31002"]
    assert [row[1] for row in calls] == ["16", "12", "16"]


def populate_reports(run, plan):
    sources = {"benchmark_topology_collective.py": fleet.sha(run / "runtime/fragment_training_kit/examples/benchmark_topology_collective.py"),
               "topology_collective.py": fleet.sha(run / "runtime/jade_lotus/topology_collective.py")}
    domains = fleet.layout_domains(plan["job_layout"])
    topology = benchmark.RankTopology.build(domains, plan["cards"])
    paths = []
    for point, exponent in enumerate(plan["rdma_timeouts"]):
        for rank in range(plan["cards"]):
            domain, local = topology.locate(rank)
            cases = [{"mode": mode, "resolved_strategy": "flat" if mode == "flat" else topology.select(),
                      "warmup_iterations": plan["warmup"], "measurement_iterations": plan["iterations"],
                      "full_buffer_SUM_verified": True, "original_local_grad_preserved": True,
                      "rank_max_seconds": [1.0 + point] * plan["iterations"]} for mode in plan["modes"]]
            row = {"passed": True, "rank": rank, "world_size": plan["cards"], "payload": plan["payload"],
                   "source_sha256": sources, "environment": {"HCCL_RDMA_TIMEOUT": str(exponent)},
                   "backend": "hccl", "device": f"npu:{local}", "visible_device_count": len(domains[domain]),
                   "node_ranks": domains, "cases": cases}
            path = run / "benchmark_results" / f"{point:03d}-rdma-{exponent}" / f"rank-{rank:06d}.json"
            fleet.atomic_json(path, row, once=True)
            paths.append(path)
    for index, ranks in enumerate(domains):
        fleet.atomic_json(run / "benchmark_workers" / f"w{index + 1:06d}.json", {
            "passed": True, "plan_id": plan["plan_id"], "ranks": ranks, "allocated_devices": len(ranks)}, once=True)
    return paths


def test_completion_requires_all_points_ranks_pods_and_numerical_flags(prepared_benchmark):
    run, _, _, _ = prepared_benchmark
    plan = fleet.read(run / "benchmark_config.json")
    assert not fleet.evidence_status(run, plan)["complete"]
    paths = populate_reports(run, plan)
    assert fleet.evidence_status(run, plan)["complete"]
    saved = paths[-1].read_text()
    paths[-1].unlink()
    incomplete = fleet.evidence_status(run, plan)
    assert not incomplete["complete"] and incomplete["passed_rank_reports"] == 5
    paths[-1].write_text(saved)
    row = fleet.read(paths[-1])
    row["cases"][1]["full_buffer_SUM_verified"] = False
    fleet.atomic_json(paths[-1], row)
    with pytest.raises(ValueError, match="Incomplete case validation"):
        fleet.evidence_status(run, plan)


def test_rank_maxima_and_frozen_source_must_match(prepared_benchmark):
    run, _, _, _ = prepared_benchmark
    plan = fleet.read(run / "benchmark_config.json")
    paths = populate_reports(run, plan)
    row = fleet.read(paths[1])
    row["cases"][0]["rank_max_seconds"][0] += 1
    fleet.atomic_json(paths[1], row)
    with pytest.raises(ValueError, match="Ranks disagree"):
        fleet.evidence_status(run, plan)
    row["source_sha256"]["topology_collective.py"] = "0" * 64
    fleet.atomic_json(paths[1], row)
    with pytest.raises(ValueError, match="inconsistent"):
        fleet.evidence_status(run, plan)


def fake_api(run, calls, *, snapshot_error=None, cleanup_error=None):
    def snapshot(client, target, *, fill):
        assert target == run
        calls.append(("snapshot", fill))
        if snapshot_error:
            raise launch.PlatformError(f"{snapshot_error} {len(calls)}")
        return {"active_jobs": 0, "local_requested_reserved_cards": 0}
    def cleanup(client, target, *, wait_seconds, stop_controller):
        assert target == run and not stop_controller
        calls.append(("cleanup", stop_controller))
        if cleanup_error:
            raise launch.PlatformError(cleanup_error)
        return {"resources_released": True, "active_jobs": 0, "local_requested_reserved_cards": 0}
    return SimpleNamespace(writer_lock=lambda path: nullcontext(), snapshot=snapshot, cleanup=cleanup,
                           PlatformError=launch.PlatformError, WriterLockBusy=launch.WriterLockBusy,
                           active=launch.active)


def test_success_never_starts_training_and_confirms_owned_official_release(prepared_benchmark, monkeypatch):
    run, _, _, _ = prepared_benchmark
    plan = fleet.read(run / "benchmark_config.json")
    populate_reports(run, plan)
    calls = []
    api = fake_api(run, calls)
    monkeypatch.setattr(fleet, "verify_benchmark", lambda target, *, api: plan)
    assert fleet.run_benchmark(object(), run, api=api) == 0
    assert calls == [("snapshot", False), ("cleanup", False)]
    summary = fleet.read(run / "benchmark_summary.json")
    assert summary["passed"] and summary["resources_released"] and not summary["training"]
    assert (run / "COMPLETE.json").exists() and not (run / "controller_process.json").exists()


def test_failure_always_cleans_and_keeps_original_error_when_cleanup_also_fails(prepared_benchmark, monkeypatch):
    run, _, _, _ = prepared_benchmark
    plan = fleet.read(run / "benchmark_config.json")
    calls = []
    api = fake_api(run, calls, cleanup_error="cannot confirm remote terminal state")
    monkeypatch.setattr(fleet, "verify_benchmark", lambda target, *, api: plan)
    def fail_evidence(*args):
        raise ValueError("first original SUM failure")
    monkeypatch.setattr(fleet, "evidence_status", fail_evidence)
    assert fleet.run_benchmark(object(), run, api=api) == 3
    assert calls == [("cleanup", False)]
    assert "original SUM failure" in fleet.read(run / "first_benchmark_error.json")["error"]
    assert "cannot confirm" in fleet.read(run / "cleanup_failed.json")["error"]
    assert not fleet.read(run / "NEEDS_CLEANUP.json")["resources_released"]
    assert not (run / "COMPLETE.json").exists()


def test_persistent_query_failures_are_bounded_and_cleanup_runs(prepared_benchmark, monkeypatch):
    run, _, _, _ = prepared_benchmark
    plan = fleet.read(run / "benchmark_config.json")
    calls = []
    api = fake_api(run, calls, snapshot_error="platform query failure")
    monkeypatch.setattr(fleet, "verify_benchmark", lambda target, *, api: plan)
    monkeypatch.setattr(fleet.time, "sleep", lambda seconds: None)
    assert fleet.run_benchmark(object(), run, api=api) == 1
    assert calls == [("snapshot", True)] * 3 + [("cleanup", False)]
    assert fleet.read(run / "first_platform_error.json")["error"].endswith("failure 1")


def test_benchmark_budget_lock_contention_does_not_count_as_platform_failure(prepared_benchmark, monkeypatch):
    run, _, _, _ = prepared_benchmark
    plan = fleet.read(run / "benchmark_config.json")
    calls = []
    api = fake_api(run, calls)
    count = [0]
    def busy_then_finish(client, target, *, fill):
        assert not (run / "FATAL.json").exists() and not (run / "STOP").exists()
        count[0] += 1
        if count[0] <= 4:
            raise launch.WriterLockBusy(run / "global-budget-lock")
        if fill:
            populate_reports(run, plan)
        return {"active_jobs": 0, "local_requested_reserved_cards": 0, "blocked_reason": None}
    api.snapshot = busy_then_finish
    monkeypatch.setattr(fleet, "verify_benchmark", lambda target, *, api: plan)
    monkeypatch.setattr(fleet.time, "sleep", lambda seconds: None)
    assert fleet.run_benchmark(object(), run, api=api) == 0
    assert count[0] == 6
    assert calls == [("cleanup", False)]
    assert not (run / "first_platform_error.json").exists()
    assert not (run / "first_benchmark_error.json").exists()
    assert fleet.read(run / "lock_retry.json")["training_stop_requested"] is False


def test_benchmark_lock_retry_stops_at_frozen_deadline(prepared_benchmark, monkeypatch):
    run, _, _, _ = prepared_benchmark
    plan = fleet.read(run / "benchmark_config.json")
    plan["fleet_timeout_seconds"] = .03
    clock = [0.]
    calls = []
    api = fake_api(run, calls)
    def busy(*args, **kwargs):
        raise launch.WriterLockBusy(run / "global-budget-lock")
    api.snapshot = busy
    monkeypatch.setattr(fleet, "verify_benchmark", lambda target, *, api: plan)
    monkeypatch.setattr(fleet.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(fleet.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    assert fleet.run_benchmark(object(), run, api=api) == 1
    assert clock[0] == .03 and calls == [("cleanup", False)]
    assert "frozen hard deadline" in fleet.read(run / "first_benchmark_error.json")["error"]
    assert not (run / "first_platform_error.json").exists()


def test_completed_benchmark_retries_a_busy_terminal_observation(prepared_benchmark, monkeypatch):
    run, _, _, _ = prepared_benchmark
    plan = fleet.read(run / "benchmark_config.json")
    plan["terminal_grace_seconds"] = 1
    populate_reports(run, plan)
    calls = []
    api = fake_api(run, calls)
    count = [0]
    def busy_then_released(client, target, *, fill):
        assert not fill
        count[0] += 1
        if count[0] == 1:
            raise launch.WriterLockBusy(run / "global-budget-lock")
        return {"active_jobs": 0, "local_requested_reserved_cards": 0}
    api.snapshot = busy_then_released
    monkeypatch.setattr(fleet, "verify_benchmark", lambda target, *, api: plan)
    monkeypatch.setattr(fleet.time, "sleep", lambda seconds: None)
    assert fleet.run_benchmark(object(), run, api=api) == 0
    assert count[0] == 2
    assert not (run / "first_benchmark_error.json").exists()


def test_terminal_run_refuses_reexecution_without_rewriting_evidence(prepared_benchmark, monkeypatch):
    run, _, _, _ = prepared_benchmark
    plan = fleet.read(run / "benchmark_config.json")
    fleet.atomic_json(run / "FATAL.json", {"original": "preserve"}, once=True)
    monkeypatch.setattr(fleet, "verify_benchmark", lambda target, *, api: plan)
    calls = []
    with pytest.raises(ValueError, match="terminal benchmark"):
        fleet.run_benchmark(object(), run, api=fake_api(run, calls))
    assert fleet.read(run / "FATAL.json") == {"original": "preserve"} and not calls


def test_local_terminal_labels_do_not_release_an_ambiguous_owned_intent(tmp_path):
    run = tmp_path / "run"
    fleet.atomic_json(run / "ascend_fleet/state.json", {"jobs": [{"worker_id": "w000001", "status": "job_failed",
                                                                   "official_terminal": False}]})
    assert fleet.early_terminal_failure(run, launch) is None
    state = fleet.read(run / "ascend_fleet/state.json")
    state["jobs"][0]["official_terminal"] = True
    fleet.atomic_json(run / "ascend_fleet/state.json", state)
    assert "before successful benchmark" in fleet.early_terminal_failure(run, launch)
