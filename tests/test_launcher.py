"""Portable launch invariants: source freeze, lifecycle and fixed-ID budget."""
from pathlib import Path
import json
import sys

import pytest

from fragment_training_kit import launch


@pytest.fixture
def prepared(tmp_path):
    cann = tmp_path / "cann.sh"
    cann.write_text("export PYTHONPATH=/cann/python:$PYTHONPATH\n")
    inventory = tmp_path / "inventory.json"
    selection = {"workspace_id": launch.WORKSPACES[0], "project_id": "project-fixture",
        "logic_compute_group_id": "group-fixture", "image": "image-fixture", "image_type": "SOURCE_PRIVATE",
        "priority": {"verified": True, "level": "LOW", "name": "1", "task_priority": 1},
        "resource_spec_price": {"gpu_count": 1, "gpu_type": "ASCEND_910B", "quota_id": "quota-fixture",
                                "cpu_count": 4, "memory_size_gib": 16}}
    inventory.write_text(json.dumps({"backend": "ascend", "user": {"id": "user-fixture"}, "selections": [selection]}))
    site = {"schema": launch.SCHEMA, "python": sys.executable, "cann_env": str(cann),
            "inventory": str(inventory), "registry": str(tmp_path / "registry.json"),
            "cards": 2, "steps": 20, "training_config": {"resident_adapter": "fixture:create_engine",
                "resident_rng_scheme": "fixture-canonical-rng-v1",
                "resident_precision_contract": "native FP32 parameters and AdamW moments"}}
    config = tmp_path / "site.json"
    config.write_text(json.dumps(site))
    run = tmp_path / "shared/run"
    result = launch.prepare(config, run)
    return run, config, site, result


def test_prepare_freezes_without_starting_or_submitting(prepared):
    run, _, _, result = prepared
    assert result["submitted"] is False
    assert not (run / "controller_process.json").exists()
    assert not (run / "ascend_fleet").exists()
    assert launch.verify(run)["verified"]
    command = launch.read(run / "controller_command.json")
    assert "--external-loss" in command and "--no-replacement" in command
    assert command[command.index("--control-poll-seconds") + 1] == "0.01"
    shell = (run / "launch.sh").read_text()
    assert ':"${PYTHONPATH:-}"' in shell
    assert "ASCEND_VISIBLE_DEVICES=" not in shell
    assert "ASCEND_RT_VISIBLE_DEVICES=" not in shell
    assert "HCCL_RDMA_TIMEOUT=12" in shell


def test_prepare_rejects_reuse_and_verify_detects_mutation(prepared):
    run, config, _, _ = prepared
    with pytest.raises(ValueError, match="overwrite"):
        launch.prepare(config, run)
    (run / "training.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        launch.verify(run)


def test_config_refuses_secret_and_unsupported_elastic_replacement(prepared):
    _, config, site, _ = prepared
    site["cookie_file"] = "/never-copy/secret.cookies"
    launch.write(config, site)
    with pytest.raises(ValueError, match="private authentication"):
        launch.resolve_site(config)
    del site["cookie_file"]
    site["replacement"] = True
    launch.write(config, site)
    with pytest.raises(ValueError, match="not implemented"):
        launch.resolve_site(config)


def test_python_invocation_preserves_virtualenv_symlink(tmp_path):
    path = tmp_path / "venv-python"
    path.symlink_to(sys.executable)
    assert launch.absolute_file(str(path), "python") == str(path)


@pytest.mark.parametrize("key", ["resident_rng_scheme", "resident_precision_contract"])
@pytest.mark.parametrize("value", [None, "", "   ", 7])
def test_custom_adapter_identity_is_rejected_before_preparation(prepared, tmp_path, key, value):
    _, config, site, _ = prepared
    if value is None:
        del site["training_config"][key]
    else:
        site["training_config"][key] = value
    launch.write(config, site)
    run = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match=key):
        launch.prepare(config, run)
    assert not run.exists()


def test_passed_marker_requires_actual_target_and_native_state(prepared):
    run, _, _, _ = prepared
    launch.write(run / "probe/summary.json", {"status": "passed", "committed_steps": 19})
    with pytest.raises(ValueError, match="full-state evidence"):
        launch.mirror_markers(run)
    launch.write(run / "probe/summary.json", {"status": "passed", "committed_steps": 20,
        "final_states_identical": True, "final_checkpoint": {"path": "evidence"}})
    launch.mirror_markers(run)
    assert (run / "COMPLETE.json").exists()


def test_initial_intent_limit_prevents_refill_with_new_training_ids(prepared, monkeypatch):
    run, _, _, _ = prepared
    launch.write(run / "ascend_fleet/state.json", {"jobs": [
        {"status": "job_succeeded", "official_terminal": True},
        {"status": "job_running", "official_terminal": False}]})
    observed = []
    def reconcile(*args, **kwargs):
        observed.append(kwargs)
        return {key: None for key in ("at", "marker", "active_jobs", "running_jobs", "pending_jobs",
            "local_requested_reserved_cards", "verified_allocated_cards", "created", "blocked_reason", "errors")}
    monkeypatch.setattr(launch, "reconcile", reconcile)
    result = launch.snapshot(object(), run, fill=True)
    assert observed[0]["fill"] is False
    assert result["arbitrary_fleet_refill_joins_supported"] is False


def test_partial_initial_failure_still_cannot_create_more_than_cards_ids(prepared, monkeypatch):
    run, _, _, _ = prepared
    launch.write(run / "ascend_fleet/state.json", {"jobs": [
        {"status": "job_failed", "official_terminal": True}]})
    observed = []
    def reconcile(*args, **kwargs):
        observed.append(kwargs)
        return {key: None for key in ("at", "marker", "active_jobs", "running_jobs", "pending_jobs",
            "local_requested_reserved_cards", "verified_allocated_cards", "created", "blocked_reason", "errors")}
    monkeypatch.setattr(launch, "reconcile", reconcile)
    launch.snapshot(object(), run, fill=True)
    assert observed[0]["fill"] is True
    assert observed[0]["card_limit"] == 1


def test_cleanup_only_reconciles_the_owned_run_with_stop_marker(prepared, monkeypatch):
    run, _, _, _ = prepared
    calls = []
    monkeypatch.setattr(launch, "stop_owned_controller", lambda target: calls.append(("controller", target)))
    def snapshot(client, target, *, fill):
        calls.append(("snapshot", target, fill))
        assert (target / "STOP").exists()
        return {"active_jobs": 0, "local_requested_reserved_cards": 0}
    monkeypatch.setattr(launch, "snapshot", snapshot)
    result = launch.cleanup(object(), run)
    assert result["resources_released"]
    assert all(call[1] == run for call in calls)
    assert launch.read(run / "cleanup_receipt.json")["resources_released"]


def test_custom_source_copy_includes_only_ordinary_python(prepared, tmp_path):
    _, config, site, _ = prepared
    code = tmp_path / "my-code"
    (code / ".private").mkdir(parents=True)
    (code / "data").mkdir()
    (code / "model.py").write_text("def create_engine(): pass\n")
    (code / ".private/secrets.py").write_text("never copy this\n")
    (code / "credentials.json").write_text('{"token":"never copy this"}')
    (code / "data/huge.py").write_text("not source\n")
    site["source_dirs"] = [str(code)]
    launch.write(config, site)
    run = tmp_path / "pure-source-run"
    launch.prepare(config, run)
    copied = run / "runtime/project_sources/source0"
    assert {str(path.relative_to(copied)) for path in copied.rglob("*") if path.is_file()} == {"model.py"}


def test_persistent_platform_failure_records_unconfirmed_cleanup(prepared, monkeypatch):
    run, _, _, _ = prepared
    monkeypatch.setattr(launch, "start_controller", lambda target: None)
    monkeypatch.setattr(launch, "controller_disappeared", lambda target: False)
    monkeypatch.setattr(launch.time, "sleep", lambda seconds: None)
    calls = []
    def snapshot(*args, **kwargs):
        calls.append("query")
        raise launch.PlatformError("temporary platform failure " + str(len(calls)))
    def cleanup(*args, **kwargs):
        calls.append("cleanup")
        raise launch.PlatformError("cannot confirm remote stop")
    monkeypatch.setattr(launch, "snapshot", snapshot)
    monkeypatch.setattr(launch, "cleanup", cleanup)
    with pytest.raises(launch.PlatformError, match="failure 3"):
        launch.run_training(object(), run, watch_seconds=60)
    assert calls == ["query", "query", "query", "cleanup"]
    assert launch.read(run / "first_platform_error.json")["error"].endswith("failure 1")
    assert launch.read(run / "NEEDS_CLEANUP.json")["resources_released"] is False
    assert launch.read(run / "cleanup_failed.json")["resources_released"] is False
