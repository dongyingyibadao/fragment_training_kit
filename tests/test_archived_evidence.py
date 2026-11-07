"""Relocated real-run evidence remains verifiable without its original host."""
from pathlib import Path
import json
import shutil

import pytest

from fragment_training_kit.examples.audit_toy import audit_run, main


EVIDENCE = Path(__file__).resolve().parents[1] / "validation"


def copied_run(tmp_path, name="normal"):
    root = tmp_path / name
    shutil.copytree(EVIDENCE / name, root)
    return root / "probe"


def forbid_recorded_paths(monkeypatch, probe):
    """The historical files may still exist on a developer's machine."""
    spec = json.loads((probe / "spec.json").read_text())
    summary = json.loads((probe / "summary.json").read_text())
    recorded_config = Path(spec["config"])
    recorded_checkpoint = Path(summary["final_checkpoint"]["path"])
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == recorded_config or path.is_relative_to(recorded_checkpoint):
            raise AssertionError(f"Archived audit accessed a recorded execution path: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)


@pytest.mark.parametrize("name,had_fault", [("normal", False), ("fault", True)])
def test_relocated_evidence_audits_without_reading_recorded_paths(tmp_path, monkeypatch, capsys, name, had_fault):
    probe = copied_run(tmp_path, name)
    forbid_recorded_paths(monkeypatch, probe)
    output = tmp_path / "audit.json"
    assert main(["--run-dir", str(probe), "--archived-evidence", "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == result
    assert result["status"] == "passed"
    assert result["audit_mode"] == "archived_evidence"
    assert result["npu_training_rerun"] is False
    assert "no NPU training rerun" in result["scope"]
    assert result["run_dir"] == str(probe.resolve())
    assert result["steps"] == 20
    assert result["actual_fault_occurred"] is had_fault
    assert result["gradient_and_optimizer_reference_passed"]
    assert result["full_native_checkpoint_verified"]
    assert result["all_final_workers_state_hashes_identical"]


def test_archived_config_digest_is_required(tmp_path, monkeypatch):
    probe = copied_run(tmp_path)
    config = probe.parent / "training.json"
    config.write_bytes(config.read_bytes() + b"\n")
    forbid_recorded_paths(monkeypatch, probe)
    with pytest.raises(ValueError, match="Training configuration changed"):
        audit_run(probe, archived_evidence=True)


def test_archived_payload_bytes_are_verified(tmp_path, monkeypatch):
    probe = copied_run(tmp_path)
    payload = probe / "states/final-20/state.bin"
    raw = bytearray(payload.read_bytes())
    raw[0] ^= 1
    payload.write_bytes(raw)
    forbid_recorded_paths(monkeypatch, probe)
    with pytest.raises(ValueError, match="Native checkpoint chunk SHA256"):
        audit_run(probe, archived_evidence=True)


def test_archived_manifest_digest_is_required(tmp_path, monkeypatch):
    probe = copied_run(tmp_path)
    path = probe / "states/final-20/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["payload_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    forbid_recorded_paths(monkeypatch, probe)
    with pytest.raises(ValueError, match="Native checkpoint manifest SHA256"):
        audit_run(probe, archived_evidence=True)


def test_archived_final_worker_hash_is_required(tmp_path, monkeypatch):
    probe = copied_run(tmp_path)
    path = next((probe / "reports").glob("*/*/*/finished.json"))
    final = json.loads(path.read_text())
    final["optimizer_sha256"] = "0" * 64
    path.write_text(json.dumps(final))
    forbid_recorded_paths(monkeypatch, probe)
    with pytest.raises(ValueError, match="Checkpoint does not match an actual final worker state"):
        audit_run(probe, archived_evidence=True)


@pytest.mark.parametrize("missing", ["training.json", "probe/states/final-20"])
def test_missing_local_evidence_never_falls_back_to_recorded_paths(tmp_path, monkeypatch, missing):
    probe = copied_run(tmp_path)
    path = probe.parent / missing
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    forbid_recorded_paths(monkeypatch, probe)
    with pytest.raises(FileNotFoundError):
        audit_run(probe, archived_evidence=True)


@pytest.mark.parametrize("escaped", ["training.json", "probe/states/final-20"])
def test_archived_mapping_rejects_symlinks_outside_its_local_root(tmp_path, monkeypatch, escaped):
    probe = copied_run(tmp_path)
    path = probe.parent / escaped
    outside = tmp_path / "outside-evidence"
    if path.is_dir():
        shutil.move(str(path), str(outside))
        path.symlink_to(outside, target_is_directory=True)
    else:
        path.rename(outside)
        path.symlink_to(outside)
    forbid_recorded_paths(monkeypatch, probe)
    with pytest.raises(ValueError, match="Archived evidence path escapes"):
        audit_run(probe, archived_evidence=True)


def test_archived_report_path_cannot_escape_probe(tmp_path, monkeypatch):
    probe = copied_run(tmp_path)
    path = probe / "summary.json"
    summary = json.loads(path.read_text())
    summary["commits"][0]["members"][0]["session_id"] = "../../../../outside"
    path.write_text(json.dumps(summary))
    forbid_recorded_paths(monkeypatch, probe)
    with pytest.raises(ValueError, match="Archived evidence path escapes"):
        audit_run(probe, archived_evidence=True)


def test_default_mode_still_uses_recorded_live_paths(tmp_path):
    probe = copied_run(tmp_path)
    spec_path = probe / "spec.json"
    spec = json.loads(spec_path.read_text())
    # A live run may store configuration and checkpoints outside its run dir.
    config = tmp_path / "live-training.json"
    shutil.move(str(probe.parent / "training.json"), str(config))
    spec["config"] = str(config)
    spec_path.write_text(json.dumps(spec))
    summary_path = probe / "summary.json"
    summary = json.loads(summary_path.read_text())
    checkpoint = tmp_path / "live-checkpoint"
    shutil.move(str(probe / "states/final-20"), str(checkpoint))
    summary["final_checkpoint"]["path"] = str(checkpoint)
    summary_path.write_text(json.dumps(summary))
    result = audit_run(probe)
    assert result["status"] == "passed"
    assert result["audit_mode"] == "live_run"
    assert result["full_native_checkpoint_verified"]


def test_archived_flag_requires_run_directory(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--cpu-oracle", "--archived-evidence"])
    assert error.value.code == 2
    assert "--archived-evidence requires --run-dir" in capsys.readouterr().err
