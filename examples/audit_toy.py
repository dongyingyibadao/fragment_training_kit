"""Independent full-global-batch CPU oracle and saved real-NPU run audit.

Run from the toolkit's parent directory:
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m fragment_training_kit.examples.audit_toy --cpu-oracle --steps 20
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m fragment_training_kit.examples.audit_toy --run-dir RUN/probe --output audit.json
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m fragment_training_kit.examples.audit_toy --run-dir validation/normal/probe --archived-evidence

The CPU oracle uses a single 256-row forward and ordinary AdamW, independent of
the adapter's atomic accumulation. Real-run validation compares every committed
loss/gradient norm and the final complete parameter/Adam state, then checks the
loss trend. Loss decrease alone is never treated as gradient correctness.
Archived evidence uses only the adjacent training.json and local states/final-N;
it audits saved evidence with a CPU oracle and does not rerun NPU training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

# This entry point only uses CPU reference math and saved bytes; do not load
# torch_npu/CANN through Torch backend auto-discovery in a login/CPU process.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from fragment_training_kit.examples.toy_regression import ToyEngine, batch_for_positions, build_model, config as toy_config


def optimizer_for(model, config):
    import torch
    from jade_lotus.coordinator import learning_rate
    return torch.optim.AdamW(model.parameters(), lr=learning_rate(config, 0),
                             betas=(.9, .95), eps=1e-8, weight_decay=1e-10)


def reference(config, steps):
    import torch
    from jade_lotus.coordinator import learning_rate
    model = build_model()
    optimizer = optimizer_for(model, config)
    history = []
    for step in range(steps):
        x, y = batch_for_positions(range(step * 256, (step + 1) * 256), seed=config.get("seed", 7))
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), config.get("clip_gradient_norm", 10.), foreach=False)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(config, step)
        optimizer.step()
        history.append({"step": step + 1, "loss_before_update": float(loss.detach()), "gradient_norm": float(norm)})
    with torch.no_grad():
        x, y = batch_for_positions(range(256), seed=config.get("seed", 7))
        final_loss = float(((model(x) - y) ** 2).mean())
    return model, optimizer, history, final_loss


def check_atomic_math(config):
    import torch
    from fragment_training_kit.adapters import accumulate_units
    initial = build_model()
    x, y = batch_for_positions(range(256), seed=config.get("seed", 7))
    ((initial(x) - y) ** 2).mean().backward()
    expected = {n: p.grad.clone() for n, p in initial.named_parameters()}
    layouts = []
    for world in (2, 3, 7):
        sums = {n: torch.zeros_like(v) for n, v in expected.items()}
        for rank in range(world):
            engine = ToyEngine(config)
            accumulate_units(engine, base_step=0, units=list(range(rank, 64, world)))
            for name, parameter in engine.parameters.items():
                sums[name] += parameter.grad
        error = max(float((sums[name] - expected[name]).abs().max()) for name in sums)
        if error > 2e-6:
            raise AssertionError(f"Atomic SUM vs full256 gradient differs: world={world}, error={error}")
        layouts.append({"world_size": world, "max_gradient_abs_error": error})
    engine = ToyEngine(config)
    accumulate_units(engine, base_step=0, units=list(range(0, 64, 2)))
    pointers = {n: p.grad.data_ptr() for n, p in engine.parameters.items()}
    accumulate_units(engine, base_step=0, units=list(range(1, 64, 2)), clear=False)
    accumulate_units(engine, base_step=0, units=[], clear=False)
    if any(engine.parameters[n].grad.data_ptr() != pointers[n] for n in pointers):
        raise AssertionError("Replay replaced healthy gradient storage")
    replay_error = max(float((engine.parameters[n].grad - expected[n]).abs().max()) for n in expected)
    if replay_error > 2e-6:
        raise AssertionError("Replay did not preserve/complete the global gradient")
    return {"passed": True, "layouts": layouts, "clear_false_replay_max_gradient_abs_error": replay_error,
            "healthy_gradient_pointers_preserved": True, "empty_replay_preserves_gradients": True,
            "scope": "CPU mathematical validation; not a real HCCL fault/recovery experiment"}


def _close(actual, expected, label, *, atol=2e-4, rtol=2e-4):
    if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol):
        raise AssertionError(f"{label}: actual={actual}, expected={expected}")


def _member_identities(members):
    if not isinstance(members, list) or not 1 <= len(members) <= 64:
        raise ValueError("Expected a nonempty bounded exact member list")
    identities = set()
    workers = set()
    for member in members:
        if (not isinstance(member, dict)
                or any(not isinstance(member.get(key), str) or not member[key] for key in ("worker_id", "session_id"))
                or type(member.get("pid")) is not int or member["pid"] < 1
                or member["worker_id"] in workers):
            raise ValueError("Expected unique exact member identities")
        workers.add(member["worker_id"])
        identities.add(tuple(member[key] for key in ("worker_id", "session_id", "pid")))
    return identities


def _archived_path(root, *parts):
    """Resolve an explicit local mapping, refusing traversal and escaping links."""
    for part in parts:
        path = Path(part)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Archived evidence path escapes its local root")
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Archived evidence path escapes its local root")
    return path


def audit_run(directory, *, archived_evidence=False):
    import torch
    from jade_lotus.resident_state import load_native_state
    directory = Path(directory).resolve()

    def evidence_path(*parts):
        if archived_evidence:
            return _archived_path(directory, *parts)
        return directory.joinpath(*parts)

    spec = json.loads(evidence_path("spec.json").read_text())
    summary = json.loads(evidence_path("summary.json").read_text())
    config_path = (_archived_path(directory.parent, "training.json")
                   if archived_evidence else Path(spec["config"]))
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != spec["config_sha256"]:
        raise ValueError("Training configuration changed after controller startup")
    config = json.loads(config_bytes)
    if config.get("resident_adapter") != "fragment_training_kit.examples.toy_regression:create_engine":
        raise ValueError("This oracle accepts only the deterministic toy adapter")
    if spec.get("start_step", 0) != 0 or summary.get("committed_steps") != spec["steps"]:
        raise ValueError("Expected a complete fresh toy run")
    if summary.get("status") != "passed" or not summary.get("final_states_identical"):
        raise ValueError("Training did not pass full-state agreement")
    if len(summary["commits"]) != spec["steps"]:
        raise ValueError("A full untrimmed commit ledger is required for toy validation")
    devices = []
    used_members = [m for commit in summary['commits'] for m in commit.get('prepared_members', commit['members'])]
    used_workers = sorted(set(spec['workers']) | {m['worker_id'] for m in used_members})
    if spec.get('pod_refill'):
        if archived_evidence:
            raise ValueError('Dynamic refill archives require their allocation/admission ledger; use the complete live run')
        from jade_lotus.refill_protocol import read_allocation
        admitted = {}
        for event in summary.get('refill_events', []):
            if event['status'] != 'admitted':
                continue
            if event['candidate_count'] != 16 or len(event['candidates']) != 16 or not event['full_native_state_alignment_verified']:
                raise ValueError('Refill admission lacks exact-16 complete-state evidence')
            for member in event['candidates']:
                admitted[member['worker_id']] = event['step']
        for commit in summary['commits']:
            for member in commit.get('prepared_members', commit['members']):
                worker = member['worker_id']
                if worker not in spec['initial_workers'] and (worker not in admitted or commit['step'] <= admitted[worker]):
                    raise ValueError('Candidate trained before its COMMIT admission')
        for worker in used_workers:
            ready = json.loads(evidence_path('ready', f'{worker}.json').read_text())
            mapping = read_allocation(directory.parent, ready['allocation_group'], ready['allocated_devices'])
            if not any(m['worker_id'] == worker and m['device_index'] == ready['device_index'] for m in mapping['members']):
                raise ValueError('Trained member does not match its immutable physical allocation')
    for worker in used_workers:
        ready = json.loads(evidence_path("ready", f"{worker}.json").read_text())
        proof = ready.get("adapter_device", {})
        index, allocated = proof.get("device_index", 0), proof.get("allocated_devices", 1)
        if (ready["run_id"] != spec["run_id"] or proof.get("resident_adapter") != config["resident_adapter"]
                or type(index) is not int or type(allocated) is not int or allocated not in (1, 2, 4, 8, 16)
                or not 0 <= index < allocated or proof.get("device") != f"npu:{index}"
                or proof.get("visible_devices") != allocated
                or "Ascend" not in proof.get("device_name", "")):
            raise ValueError("Expected actual assigned logical-NPU adapter readiness evidence")
        devices.append(proof)
    errors = list(evidence_path("errors").glob("*.json"))
    if errors:
        if not spec.get('pod_refill'):
            raise ValueError('Worker errors are present')
        for path in errors:
            record = json.loads(path.read_text())
            if record.get('worker_id') in used_workers:
                raise ValueError('A trained worker reported a device error')
            ready = json.loads(evidence_path('ready', f"{record['worker_id']}.json").read_text())
            from jade_lotus.refill_protocol import validate_cancel
            mapping = read_allocation(directory.parent, ready['allocation_group'], ready['allocated_devices'])
            cancellation = json.loads((directory.parent/'refill/cancelled'/f"{ready['allocation_group']}.json").read_text())
            if not validate_cancel(mapping, cancellation):
                raise ValueError('Untrained candidate error lacks a controller-owned cancellation')
    model, optimizer, oracle, final_loss = reference(config, spec["steps"])
    observed = []
    for commit, expected in zip(summary["commits"], oracle, strict=True):
        survivors = _member_identities(commit["members"])
        prepared_members = commit.get("prepared_members", commit["members"])
        prepared_identities = _member_identities(prepared_members)
        if not survivors.issubset(prepared_identities):
            raise ValueError("Commit survivors are not exact prepared members")
        assigned = [unit for units in commit["assignments"].values() for unit in units]
        if (commit["step"] != expected["step"] or commit["samples"] != 256
                or commit.get("atomic_units") != 64 or sorted(assigned) != list(range(64))
                or set(commit["assignments"]) != {m["worker_id"] for m in prepared_members}):
            raise ValueError("Commit sequence/global batch differs")
        total_loss, norms = 0., []
        if commit.get("pod_recovery"):
            # The frozen pre-APPLY group supplied every atomic unit. A member
            # lost after authorization still contributed to the global SUM;
            # inspect its immutable reduced report rather than recomputing it.
            for member in commit["prepared_members"]:
                prepared = json.loads(evidence_path("reports", member["worker_id"], member["session_id"],
                    commit["reduction_token"], "pod_reduced.json").read_text())
                if (prepared["run_id"] != spec["run_id"] or prepared["token"] != commit["reduction_token"]
                        or prepared["base_step"] != commit["step"]-1
                        or any(prepared[k] != member[k] for k in ("worker_id","session_id","pid"))
                        or sorted(prepared["local"]["units"]) != sorted(commit["assignments"][member["worker_id"]])):
                    raise ValueError("Pod prepared contribution has stale identity/coverage")
                total_loss += prepared["local"]["loss_contribution"]
        for member in commit["members"]:
            parts = ("reports", member["worker_id"], member["session_id"], commit["token"])
            record = json.loads(evidence_path(*parts, "updated.json").read_text())
            if (record["completed_step"] != commit["step"] or record["checkpoint_calls"]
                    or record["run_id"] != spec["run_id"] or record["token"] != commit["token"]
                    or any(record[key] != member[key] for key in ("worker_id", "session_id", "pid"))):
                raise ValueError("Incorrect completed step or activation recomputation")
            if not commit.get("pod_recovery"):
                total_loss += record["local"]["loss_contribution"]
            elif record.get("optimizer_applications") != 1 or record.get("reduction_token") != commit["reduction_token"]:
                raise ValueError("Pod optimizer applied a different reduction or more than once")
            computed_units = list(record["local"]["units"])
            repair = evidence_path(*parts, "repair_submitted.json")
            if repair.exists():
                replay = json.loads(repair.read_text())["replay"]
                total_loss += replay["loss_contribution"]
                computed_units.extend(replay["units"])
            if sorted(computed_units) != sorted(commit["assignments"][member["worker_id"]]):
                raise ValueError("Actual original/replayed units differ from committed coverage")
            norm = record["grad_norm"]
            _close(norm, expected["gradient_norm"], "NPU pre-clip gradient norm")
            norms.append(norm)
        _close(total_loss, expected["loss_before_update"], "NPU global loss")
        observed.append({"step": commit["step"], "loss": total_loss, "gradient_norms": norms,
                         "commit_seconds": commit["commit_seconds"],
                         "commit_interval_seconds": commit["commit_interval_seconds"],
                         "members": len(commit["members"])})
    checkpoint = summary.get("final_checkpoint", {}).get("path")
    if not checkpoint:
        raise ValueError("Set resident_final_state=checkpoint to audit actual model and Adam state")
    if archived_evidence:
        checkpoint = evidence_path("states", f"final-{spec['steps']}")
        # Validate both children before the loader opens them; retain its own
        # manifest, payload, tensor, and restored-state verification unchanged.
        _archived_path(checkpoint, "manifest.json")
        _archived_path(checkpoint, "state.bin")
    actual = build_model()
    actual_optimizer = optimizer_for(actual, config)
    expected_metadata = ({"run_id": spec["run_id"], "config_sha256": spec["config_sha256"],
                          "boundary_step": spec["steps"]} if archived_evidence else None)
    receipt = load_native_state(actual, actual_optimizer, checkpoint, expected_step=spec["steps"],
                                expected_metadata=expected_metadata)
    final_members = summary.get("final_member_identities", summary["commits"][-1]["members"])
    final_identities = _member_identities(final_members)
    final_count = summary.get("final_members", len(final_members))
    if (not final_identities.issubset(_member_identities(summary["commits"][-1]["members"]))
            or type(final_count) is not int or final_count != len(final_members)):
        raise ValueError("Final members differ from the last commit or final member count")
    final_hashes = set()
    for member in final_members:
        root = evidence_path("reports", member["worker_id"], member["session_id"])
        if archived_evidence:
            finals = []
            for child in root.iterdir():
                local_child = _archived_path(directory, child.relative_to(directory))
                if local_child.is_dir():
                    candidate = _archived_path(local_child, "finished.json")
                    if candidate.exists():
                        finals.append(candidate)
        else:
            finals = list(root.glob("*/finished.json"))
        if len(finals) != 1:
            raise ValueError("Expected one full final-state report per final member")
        final = json.loads(finals[0].read_text())
        if (final.get("run_id") != spec["run_id"] or final.get("phase") != "finished"
                or final.get("token") != finals[0].parent.name
                or any(final.get(key) != member[key] for key in ("worker_id", "session_id", "pid"))):
            raise ValueError("Final report has a stale run/session/token identity")
        if final["completed_step"] != spec["steps"] or any(final[key] != receipt[key] for key in ("model_sha256", "optimizer_sha256")):
            raise ValueError("Checkpoint does not match an actual final worker state")
        final_hashes.add(final["optimizer_config_sha256"])
    if len(final_hashes) != 1:
        raise ValueError("Final optimizer configuration hashes differ")
    parameter_errors, moment_errors = {}, {}
    for (name, parameter), (actual_name, value) in zip(model.named_parameters(), actual.named_parameters(), strict=True):
        assert name == actual_name
        parameter_errors[name] = float((parameter.detach() - value.detach()).abs().max())
        torch.testing.assert_close(value, parameter, atol=2e-4, rtol=2e-4)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            observed_state, expected_state = actual_optimizer.state[value][key], optimizer.state[parameter][key]
            moment_errors[name + "." + key] = float((observed_state - expected_state).abs().max())
            torch.testing.assert_close(observed_state, expected_state, atol=2e-4, rtol=2e-4)
    with torch.no_grad():
        x, y = batch_for_positions(range(256), seed=config.get("seed", 7))
        actual_final_loss = float(((actual(x) - y) ** 2).mean())
    _close(actual_final_loss, final_loss, "Final NPU checkpoint loss")
    if not actual_final_loss < observed[0]["loss"] * .5:
        raise AssertionError("Toy objective did not decrease by at least 50 percent")
    recovery_steps = ({summary["fault"]["base_step"] + 1} if "fault" in summary else set())
    recovery_steps.update(row["step"]+1 for row in summary.get("pod_recovery_events", [])
                          if 0 <= row["step"] < spec["steps"])
    recovery_steps.update(commit["step"] for commit in summary["commits"] if commit.get("forward_recovery"))
    switch_steps = {row["first_new_commit"] for row in summary.get("topology_switches", [])
                    if row.get("status") == "committed"}
    transition_steps = recovery_steps | switch_steps
    steady_rows = [r for r in observed[2:] if r["step"] not in transition_steps]
    steady = [r["commit_seconds"] for r in steady_rows]
    intervals = [r["commit_interval_seconds"] for r in steady_rows]
    scope = ("archived real NPU evidence vs independent full-global-batch CPU AdamW oracle; no NPU training rerun"
             if archived_evidence else "real NPU saved run vs independent full-global-batch CPU AdamW oracle")
    return {"status": "passed", "scope": scope,
            "audit_mode": "archived_evidence" if archived_evidence else "live_run",
            "npu_training_rerun": False,
            "run_dir": str(directory), "steps": spec["steps"], "cards_requested": spec["cards"],
            "initial_active_cards": spec["cards"],
            "physical_pool_slots": spec.get("topology_slots", spec["cards"]),
            "global_batch": 256, "npu_readiness_evidence": devices,
            "initial_loss": observed[0]["loss"], "final_checkpoint_loss": actual_final_loss,
            "relative_loss_drop": 1 - actual_final_loss / observed[0]["loss"],
            "gradient_and_optimizer_reference_passed": True, "full_native_checkpoint_verified": receipt["payload_bytes_verified"],
            "all_final_workers_state_hashes_identical": True, "parameter_max_abs_errors": parameter_errors,
            "adam_state_max_abs_errors": moment_errors,
            "steady_commit_seconds_mean": statistics.mean(steady) if steady else None,
            "steady_commit_seconds_median": statistics.median(steady) if steady else None,
            "steady_commit_interval_seconds_mean": statistics.mean(intervals) if intervals else None,
            "steady_commit_samples": len(steady), "steady_excluded_recovery_steps": sorted(recovery_steps),
            "steady_excluded_switch_steps": sorted(switch_steps),
            "topology_switches": summary.get("topology_switches", []),
            "recovery_updates": [r for r in observed if r["step"] in recovery_steps],
            "timing_scope": "Commit duration and actual commit interval reported separately; first two warmups, recovery update and first update after cohort switches excluded from steady values. Intervals may include earlier checkpoint/inspection boundaries.",
            "steps_observed": observed,
            "actual_fault_occurred": "fault" in summary or bool(summary.get("pod_recovery_events")),
            "pod_recovery_enabled": spec.get("pod_recovery", False),
            "pod_recovery_events": summary.get("pod_recovery_events", []),
            "limitations": ["Tiny FP32 regression is a correctness and onboarding smoke test, not large-model throughput.",
                            "Saved-evidence audit does not rerun HCCL; real failure injection is reported separately."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-oracle", action="store_true")
    mode.add_argument("--run-dir", type=Path)
    parser.add_argument("--archived-evidence", action="store_true",
                        help="Audit --run-dir using only adjacent training.json and local states/final-N; does not rerun NPU training")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.archived_evidence and args.run_dir is None:
        parser.error("--archived-evidence requires --run-dir")
    if args.cpu_oracle:
        config = toy_config(args.steps)
        mathematical = check_atomic_math(config)
        _model, _optimizer, history, final_loss = reference(config, args.steps)
        if not final_loss < history[0]["loss_before_update"] * .5:
            raise AssertionError("CPU oracle objective did not decrease by 50 percent")
        result = {"status": "passed", "scope": "CPU mathematical preflight only; no cards allocated",
                  "atomic_gradient_equivalence": mathematical, "steps": args.steps,
                  "initial_loss": history[0]["loss_before_update"], "final_loss": final_loss, "history": history}
    else:
        result = audit_run(args.run_dir, archived_evidence=args.archived_evidence)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
