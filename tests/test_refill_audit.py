"""Synthetic completed refill evidence; no Torch/model/device dependency."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from fragment_training_kit.examples import audit_refill as audit
from jade_lotus.refill_protocol import (ALLOCATION_SCHEMA, RESERVATION_SCHEMA, PERMIT_SCHEMA,
    CANCEL_SCHEMA, IDENTITY_FIELDS, mapping_digest)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def read(path):
    return json.loads(path.read_text())


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def member(index):
    return {"worker_id": f"w{index:06d}", "session_id": f"session-{index}", "pid": 1000 + index}


def checkpoint(run, spec, step):
    directory = run / "probe/states" / f"checkpoint-{step}"
    directory.mkdir(parents=True)
    defaults = {"lr": .1, "betas": {"__native_tuple__": [.9, .95]}, "eps": 1e-8}
    tensors, payload = {}, b""
    for name, value, role in [("model.parameter.weight", step / 10, "parameter_device"),
        ("optimizer.state.weight.exp_avg", .2, "parameter_device"),
        ("optimizer.state.weight.exp_avg_sq", .3, "parameter_device"),
        ("optimizer.state.weight.step", step, "cpu")]:
        raw = struct.pack("<f", value)
        offset = len(payload)
        leaf = hashlib.sha256(raw).hexdigest()
        tensors[name] = {"dtype": "float32", "shape": [], "nbytes": len(raw), "offset": offset,
            "device_role": role, "sha256": leaf, "chunks": [{"offset": offset, "nbytes": len(raw), "sha256": leaf}]}
        payload += raw
    manifest = {"format": "jade-native-resident-state-v1", "precision_contract": audit.PRECISION,
        "step": step, "chunk_bytes": 16, "metadata": {"run_id": spec["run_id"], "config_sha256": spec["config_sha256"],
            "boundary_step": step, "global_batch": 256, "activation_recomputation": False,
            "seed": spec["seed"], "rng_scheme": spec["rng_scheme"], "precision_contract": spec["precision_contract"]},
        "model_parameters": [{"name": "weight", "canonical": "weight", "requires_grad": True,
            "tensor": {"__native_tensor__": "model.parameter.weight"}}], "model_buffers": [],
        "optimizer_defaults": defaults, "optimizer_groups": [{"parameters": ["weight"], "options": defaults}],
        "optimizer_states": {"weight": {key: {"__native_tensor__": f"optimizer.state.weight.{key}"}
            for key in ("step", "exp_avg", "exp_avg_sq")}}, "tensors": tensors,
        "payload_bytes": len(payload), "payload_sha256": hashlib.sha256(payload).hexdigest()}

    def logical(value):
        if isinstance(value, dict):
            if set(value) == {"__native_tensor__"}:
                name = value["__native_tensor__"]
                return {"tensor": name, **{k: tensors[name][k] for k in ("dtype", "shape", "nbytes", "sha256")}}
            return {k: logical(v) for k, v in value.items()}
        return [logical(v) for v in value] if isinstance(value, list) else value

    manifest["model_sha256"] = sha({key: logical(manifest[key]) for key in ("model_parameters", "model_buffers")})
    manifest["optimizer_sha256"] = sha({**{key: logical(manifest[key]) for key in
        ("optimizer_defaults", "optimizer_groups", "optimizer_states")}, "step": step, "precision": audit.PRECISION})
    manifest["manifest_sha256"] = sha(manifest)
    write(directory / "manifest.json", manifest)
    (directory / "state.bin").write_bytes(payload)
    actual_options = {"lr": .1, "betas": [.9, .95], "eps": 1e-8}
    options_sha = hashlib.sha256(json.dumps({"groups": [actual_options], "defaults": actual_options}, sort_keys=True).encode()).hexdigest()
    receipt = {key: manifest[key] for key in ("step", "payload_bytes", "manifest_sha256", "model_sha256", "optimizer_sha256")}
    receipt["payload_bytes_verified"] = True
    return {"path": str(directory), "receipt": receipt, "step": step}, options_sha


def create_run(tmp_path, *, unused_failure=False, journal=False, config_overrides=None):
    run = tmp_path / "run"
    run.mkdir()
    config = {"resident_pod_refill": True, "resident_pod_recovery": True,
        "disable_activation_checkpointing": True, "batch_size": 256, "worker_microbatch_size": 4,
        "resident_adapter": "fixture.unimportable_model:create_engine"}
    config.update(config_overrides or {})
    site = {"cards": 64, "active_cards": 48, "steps": 8, "job_layout": [16, 16, 16]}
    write(run / "training.json", config)
    write(run / "site.json", site)
    source_path = run / "runtime/scripts/probe_resident_openpi_fault.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# Frozen synthetic controller; never imported by audit.\n")
    source_files = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (run / "training.json", run / "site.json", source_path)}
    write(run / "source_manifest.json", {"schema": "fragment-training-kit-prepared-v1", "files": source_files})
    spec = {"run_id": "fixture-run", "steps": 8, "start_step": 0, "cards": 48,
        "pod_refill": True, "pod_recovery": True, "config": str(run / "training.json"),
        "run_dir": str(run / "probe"), "config_sha256": source_files["training.json"],
        "source_sha256": source_files["runtime/scripts/probe_resident_openpi_fault.py"],
        "initial_workers": [member(i)["worker_id"] for i in range(1, 49)], "seed": 7,
        "rng_scheme": "fixture-counter-rng", "precision_contract": "native FP32 model and Adam"}
    write(run / "probe/spec.json", spec)
    allocations = []
    for index in range(4 + int(unused_failure)):
        prefix = f"w{index+1:06d}"
        candidate = index >= 3
        mapping = {"schema": ALLOCATION_SCHEMA, "run_dir": str(run), "config_sha256": spec["config_sha256"],
            "worker_prefix": prefix, "intent_id": f"intent-{prefix}", "role": "candidate" if candidate else "initial",
            "batch_id": ("b000001" if index == 3 else "b000002") if candidate else None,
            "members": [{"worker_id": member(16 * index + local + 1)["worker_id"], "device_index": local}
                        for local in range(1 if index == 4 else 16)]}
        write(run / "refill/allocations" / f"{prefix}.json", mapping)
        allocations.append(mapping)
        if index == 4:
            continue  # An unused failed loader is deliberately missing READY.
        for slot in mapping["members"]:
            worker = slot["worker_id"]
            write(run / "probe/ready" / f"{worker}.json", {**member(int(worker[1:])), "run_id": spec["run_id"],
                "allocated_devices": 16, "allocation_group": prefix, "device_index": slot["device_index"],
                "replacement": candidate, "refill_wrapper_session": "wrapper-4" if candidate else None,
                "refill_batch_id": mapping["batch_id"], "adapter_device": {"resident_adapter": config["resident_adapter"],
                    "device": f"npu:{slot['device_index']}", "visible_devices": 16, "device_name": "AscendFixture"}})
        if candidate:
            reservation = {**mapping, "schema": RESERVATION_SCHEMA, "mapping_sha256": mapping_digest(mapping),
                "wrapper_session": "wrapper-4", "pid": 100, "hostname": "fixture-host", "heartbeat_seq": 100,
                "updated_at": 1000., "state": "loading"}
            write(run / "refill/reservations" / f"{prefix}.json", reservation)
            write(run / "refill/permits" / f"{prefix}.json", {"schema": PERMIT_SCHEMA,
                **{key: mapping[key] for key in IDENTITY_FIELDS}, "mapping_sha256": mapping_digest(mapping),
                "wrapper_session": "wrapper-4", "batch_member_ids": [m["worker_id"] for m in mapping["members"]]})
    commits = []
    for step in range(1, 9):
        indices = list(range(1, 49)) if step <= 3 else list(range(1, 65)) if step <= 6 else list(range(17, 65))
        members = [member(i) for i in indices]
        generation = 0 if step <= 3 else 1 if step <= 6 else 2
        assignments = {m["worker_id"]: list(range(rank, 64, len(members))) for rank, m in enumerate(members)}
        token, reduction, compute, root_token = f"apply-{step}", f"reduce-{step}", f"compute-{step}", f"root-{step}"
        commit = {"step": step, "token": token, "reduction_token": reduction, "root_token": root_token,
            "members": members, "prepared_members": members, "assignments": assignments,
            "samples": 256, "atomic_units": 64, "pod_recovery": True, "generation": generation,
            "commit_seconds": 1. + len(members) / 64,
            "commit_interval_seconds": 10. if step == 4 else 20. if step == 7 else 1. + len(members) / 64}
        commits.append(commit)
        write(run / f"probe/pod/authorizations/{token}.json", {"run_id": spec["run_id"], "config_sha256": spec["config_sha256"],
            "base_step": step - 1, "update_token": token, "reduction_token": reduction, "root_token": root_token,
            "prepared_members": members, "assignments": assignments, "optimizer_replay_forbidden": True,
            "group": {"generation": generation, "members": members}})
        for m in members:
            local = {"units": assignments[m["worker_id"]], "samples": 4 * len(assignments[m["worker_id"]]),
                     "loss_contribution": 10. / (step + 1) * len(assignments[m["worker_id"]]) / 64}
            base = {**m, "run_id": spec["run_id"], "generation": generation, "local": local}
            reports = run / "probe/reports" / m["worker_id"] / m["session_id"]
            write(reports / compute / "pod_local.json", {**base, "token": compute, "phase": "pod_local", "completed_step": step - 1})
            write(reports / reduction / "pod_reduced.json", {**base, "token": reduction, "phase": "pod_reduced", "base_step": step - 1, "status": "reduced"})
            write(reports / token / "updated.json", {**base, "token": token, "phase": "updated", "completed_step": step,
                "checkpoint_calls": 0, "optimizer_applications": 1, "root_token": root_token, "reduction_token": reduction,
                "grad_norm": 1. / step, "timings": {"communication_seconds": .2, "install_clip_optimizer_seconds": .1}})
    boundary_checkpoint, options_sha = checkpoint(run, spec, 3)
    event = {"batch_id": "b000001", "status": "admitted", "step": 3, "candidate_count": 16,
        "candidates": [member(i) for i in range(49, 65)], "members": [member(i) for i in range(1, 65)],
        "full_native_state_alignment_verified": True, "boundary_pause_seconds": 8.,
        "optimizer_updates_during_admission": 0, "healthy_rollback_count": 0, "checkpoint": boundary_checkpoint,
        "state_hashes": [boundary_checkpoint["receipt"]["model_sha256"], boundary_checkpoint["receipt"]["optimizer_sha256"], options_sha]}
    write(run / "refill/admissions/b000001.json", event)
    for m in event["candidates"]:
        reports = run / "probe/reports" / m["worker_id"] / m["session_id"]
        write(reports / "admission-load/loaded.json", {**m, "run_id": spec["run_id"], "phase": "loaded",
            "token": "admission-load", "completed_step": 3, "receipt": {**boundary_checkpoint["receipt"],
                "checkpoint": boundary_checkpoint["path"], "verified_loaded_state": True,
                "parameter_pointers_preserved": True, "input_identity_verified": True}})
        write(reports / "admission-inspect/inspected.json", {**m, "run_id": spec["run_id"], "phase": "inspected",
            "token": "admission-inspect", "completed_step": 3, "checkpoint_calls": 0,
            "model_sha256": event["state_hashes"][0], "optimizer_sha256": event["state_hashes"][1],
            "optimizer_config_sha256": event["state_hashes"][2]})
    batches = [{"batch_id": "b000001", "phase": "admitted", "allocations": ["w000004"], "admission": event}]
    if unused_failure:
        mapping = allocations[-1]
        batches.append({"batch_id": "b000002", "phase": "cancelled", "allocations": ["w000005"]})
        write(run / "refill/cancelled/w000005.json", {"schema": CANCEL_SCHEMA, **{key: mapping[key] for key in IDENTITY_FIELDS}})
        write(run / "probe/errors/w000065-session-65.json", {**member(65), "run_id": spec["run_id"], "error": "Synthetic failed loader before READY"})
    write(run / "refill/state.json", {"run_dir": str(run), "config_sha256": spec["config_sha256"], "history": batches, "current": None})
    final_checkpoint, options_sha = checkpoint(run, spec, 8)
    final = [member(i) for i in range(17, 65)]
    for m in final:
        write(run / "probe/reports" / m["worker_id"] / m["session_id"] / "final-token/finished.json", {
            **m, "run_id": spec["run_id"], "token": "final-token", "phase": "finished", "completed_step": 8,
            "checkpoint_calls": 0, "model_sha256": final_checkpoint["receipt"]["model_sha256"],
            "optimizer_sha256": final_checkpoint["receipt"]["optimizer_sha256"], "optimizer_config_sha256": options_sha})
    summary = {"run_id": spec["run_id"], "status": "passed", "final_states_identical": True,
        "committed_steps": 8, "executed_steps": 8, "healthy_rollback_count": 0,
        "initial_members": [member(i) for i in range(1, 49)], "final_member_identities": final, "final_members": len(final),
        "final_checkpoint": final_checkpoint, "commits": commits, "refill_events": [event],
        "pod_recovery_events": [{"step": 6, "reason": "rebuild_group", "lost": [member(i) for i in range(1, 17)]}]}
    if journal:
        path = run / "probe/commits.jsonl"
        path.write_text("\n".join(json.dumps(c) for c in commits) + "\n")
        summary.update(commit_journal=str(path), commits=commits[-2:])
    write(run / "probe/summary.json", summary)
    write(run / "ascend_fleet/state.json", {"run_dir": str(run), "jobs": [{"worker_id": a["worker_prefix"],
        "member_ids": [m["worker_id"] for m in a["members"]], "intent_id": a["intent_id"],
        "job_id": f"job-{a['worker_prefix']}", "status": "job_succeeded", "official_terminal": True} for a in allocations]})
    write(run / "cleanup_receipt.json", {"resources_released": True, "active_jobs": 0, "local_requested_reserved_cards": 0})
    return run


@pytest.mark.parametrize("journal", [False, True])
def test_completed_generic_run_checks_bytes_membership_and_separates_stable_timing(tmp_path, journal):
    run = create_run(tmp_path, journal=journal)
    result = audit.audit_run(run)
    assert result["status"] == "passed" and result["updates_audited"] == 8
    assert result["initial_cards"] == result["final_cards"] == 48
    assert result["steady_by_cards"]["48"]["steps"] == [3, 8]
    assert result["steady_by_cards"]["64"]["steps"] == [5, 6]
    assert result["steady_excluded_steps"] == [1, 2, 4, 7]
    assert result["admission_pause_seconds_total"] == 8.
    assert result["checkpoint"]["payload_bytes_independently_verified"] is True
    assert result["checkpoint"]["payload_bytes"] == 16
    assert result["evidence_file_count"] > 1000
    assert not result["numerical_gradient_oracle_performed"] and not result["model_or_device_framework_imported"]
    assert result["evidence_index_sha256"] == sha(result["evidence_sha256"])


def test_cancelled_unused_loader_needs_no_ready_but_has_exact_cancellation(tmp_path):
    run = create_run(tmp_path, unused_failure=True)
    assert not (run / "probe/ready/w000065.json").exists()
    result = audit.audit_run(run)
    assert result["unused_cancelled_candidate_failures"] == ["probe/errors/w000065-session-65.json"]
    assert result["historical_allocated_member_count"] == 65


@pytest.mark.parametrize("overrides, accepted", [
    ({"resident_adapter": None, "worker_microbatch_size": None, "microbatch_size": 4}, True),
    ({"resident_adapter": None, "microbatch_size": 2}, False),
    ({"resident_adapter": None}, False),  # Native default is two physical samples.
    ({"resident_adapter": None, "microbatch_size": True}, False),
    ({"worker_microbatch_size": 4, "microbatch_size": 2}, True),
    ({"worker_microbatch_size": 2, "microbatch_size": 4}, False),
    ({"worker_microbatch_size": True, "microbatch_size": 4}, False),
    ({"worker_microbatch_size": None, "microbatch_size": 4}, False),
])
def test_native_and_custom_physical_batch_keys_follow_selected_engine(tmp_path, overrides, accepted):
    run = create_run(tmp_path, config_overrides=overrides)
    if accepted:
        assert audit.audit_run(run)["status"] == "passed"
    else:
        with pytest.raises(ValueError, match="global256/atomic4"):
            audit.audit_run(run)


@pytest.mark.parametrize("case", ["source", "configuration", "gap", "coverage", "identity", "reduction_token",
    "twice_adam", "recompute", "loss_nan", "norm_nan", "norm_disagreement", "admission_count", "alignment",
    "final_configuration", "platform_not_terminal", "cleanup_nonzero", "trained_candidate_error"])
def test_corrupt_or_incomplete_evidence_is_rejected(tmp_path, case):
    run = create_run(tmp_path)
    summary_path = run / "probe/summary.json"
    updated = run / "probe/reports/w000017/session-17/apply-5/updated.json"
    if case in ("source", "configuration"):
        path = run / ("runtime/scripts/probe_resident_openpi_fault.py" if case == "source" else "training.json")
        path.write_text(path.read_text() + " ")
    elif case in ("gap", "coverage", "admission_count", "alignment"):
        value = read(summary_path)
        if case == "gap": value["commits"][3]["step"] = 5
        elif case == "coverage": value["commits"][0]["assignments"]["w000001"] = [0, 0]
        elif case == "admission_count": value["refill_events"][0]["candidate_count"] = 15
        else: value["refill_events"][0]["full_native_state_alignment_verified"] = False
        write(summary_path, value)
    elif case in ("identity", "reduction_token", "twice_adam", "recompute", "norm_nan", "norm_disagreement"):
        value = read(updated)
        field, change = {"identity": ("pid", 999), "reduction_token": ("reduction_token", "stale"),
            "twice_adam": ("optimizer_applications", 2), "recompute": ("checkpoint_calls", 1),
            "norm_nan": ("grad_norm", float("nan")), "norm_disagreement": ("grad_norm", 100.)}[case]
        value[field] = change
        write(updated, value)
    elif case == "loss_nan":
        path = run / "probe/reports/w000017/session-17/reduce-5/pod_reduced.json"
        value = read(path); value["local"]["loss_contribution"] = float("nan"); write(path, value)
    elif case == "final_configuration":
        path = run / "probe/reports/w000017/session-17/final-token/finished.json"
        value = read(path); value["optimizer_config_sha256"] = "0" * 64; write(path, value)
    elif case == "platform_not_terminal":
        path = run / "ascend_fleet/state.json"
        value = read(path); value["jobs"][0]["official_terminal"] = False; write(path, value)
    elif case == "cleanup_nonzero":
        path = run / "cleanup_receipt.json"
        value = read(path); value["local_requested_reserved_cards"] = 1; write(path, value)
    else:
        write(run / "probe/errors/w000049-session-49.json", {**member(49), "run_id": "fixture-run", "error": "device failure"})
    with pytest.raises(ValueError):
        audit.audit_run(run)


@pytest.mark.parametrize("kind", ["bit_flip", "trailing_bytes", "manifest", "escaping_checkpoint"])
def test_final_checkpoint_is_independently_rehashed_and_confined_to_run(tmp_path, kind):
    run = create_run(tmp_path)
    path = run / "probe/states/checkpoint-8/state.bin"
    if kind == "bit_flip":
        value = bytearray(path.read_bytes()); value[3] ^= 1; path.write_bytes(value)
    elif kind == "trailing_bytes":
        path.write_bytes(path.read_bytes() + b"extra")
    elif kind == "manifest":
        path = path.with_name("manifest.json")
        value = read(path); value["model_sha256"] = "f" * 64; write(path, value)
    else:
        path = run / "probe/summary.json"
        value = read(path); value["final_checkpoint"]["path"] = str(tmp_path / "elsewhere"); write(path, value)
    with pytest.raises(ValueError):
        audit.audit_run(run)


def test_dynamic_member_may_train_only_starting_after_admission_boundary(tmp_path):
    run = create_run(tmp_path)
    write(run / "probe/reports/w000049/session-49/early-compute/pod_local.json", {
        **member(49), "run_id": "fixture-run", "token": "early-compute", "phase": "pod_local", "completed_step": 2})
    with pytest.raises(ValueError, match="before admission"):
        audit.audit_run(run)


@pytest.mark.parametrize("case", ["wrong_cancel", "hidden_compute"])
def test_cancelled_candidate_error_is_not_ignored_without_unused_exact_authority(tmp_path, case):
    run = create_run(tmp_path, unused_failure=True)
    if case == "wrong_cancel":
        path = run / "refill/cancelled/w000005.json"
        value = read(path); value["intent_id"] = "different-intent"; write(path, value)
    else:
        write(run / "probe/reports/w000065/session-65/hidden-compute/pod_local.json", {
            **member(65), "run_id": "fixture-run", "token": "hidden-compute", "phase": "pod_local", "completed_step": 2})
    with pytest.raises(ValueError):
        audit.audit_run(run)


def test_unlisted_second_optimizer_update_cannot_hide_outside_summary(tmp_path):
    run = create_run(tmp_path)
    source = run / "probe/reports/w000017/session-17/apply-5/updated.json"
    value = read(source); value["token"] = "unlisted-apply"
    write(source.parent.parent / "unlisted-apply/updated.json", value)
    with pytest.raises(ValueError, match="Unauthorized optimizer"):
        audit.audit_run(run)


@pytest.mark.parametrize("mode", ["absolute_script", "module"])
def test_cli_is_model_free_works_from_other_directory_and_never_overwrites(tmp_path, mode):
    run = create_run(tmp_path)
    output = tmp_path / "new-audit.json"
    script = Path(audit.__file__).resolve()
    code = """
import builtins, runpy, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torch_npu', 'openpi', 'numpy', 'fixture'}:
        raise AssertionError('Auditor imported a model/device framework: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
mode, script, parent, run, output = sys.argv[1:]
sys.argv = [script, '--run-dir', run, '--output', output]
if mode == 'module':
    sys.path.insert(0, parent)
    runpy.run_module('fragment_training_kit.examples.audit_refill', run_name='__main__')
else:
    runpy.run_path(script, run_name='__main__')
"""
    command = [sys.executable, "-I", "-c", code, mode, str(script), str(script.parents[2]), str(run), str(output)]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    original = output.read_bytes()
    assert json.loads(original)["status"] == "passed"
    repeated = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert repeated.returncode != 0 and "already exists" in repeated.stderr
    assert output.read_bytes() == original
