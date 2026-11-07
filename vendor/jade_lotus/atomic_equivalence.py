"""One-off whole-bundle versus independent four-sample gradient validation.

This runs the normal engine.compute entry point without an optimizer or task
submission. A temporary observer records the concrete prepared inputs actually
used by each call; all instance state is restored before returning.
"""

from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np

from jade_lotus.atomic_batch import AtomicPreparedTask, RNG_SCHEME, describe_units, unit_rngs
from jade_lotus.numeric_validation import compare_gradient_arrays, validate_gradient_replay


_CHUNK_ELEMENTS = 1 << 20


def _input_arrays(engine, prepared):
    values = engine.jax.tree_util.tree_flatten_with_path((prepared.observation, prepared.actions))[0]
    return {engine.jax.tree_util.keystr(path): np.asarray(engine.jax.device_get(value)) for path, value in values}


def _input_summary(arrays):
    digest = hashlib.sha256()
    schema = {}
    for name, value in sorted(arrays.items()):
        if value.dtype.hasobject:
            raise ValueError("Prepared atomic inputs must contain concrete numeric arrays")
        description = [name, str(value.dtype), list(value.shape)]
        digest.update(json.dumps(description, separators=(",", ":")).encode())
        digest.update(memoryview(np.ascontiguousarray(value)))
        schema[name] = {"dtype": str(value.dtype), "shape": list(value.shape)}
    return {"sha256": digest.hexdigest(), "fields": schema}


def _gradient_schema(params, gradients):
    if not gradients or set(gradients) != set(params):
        raise ValueError("Atomic equivalence gradient must cover every authoritative parameter leaf")
    schema, elements, total_bytes = {}, 0, 0
    for name in sorted(params):
        value, expected = np.asarray(gradients[name]), np.asarray(params[name])
        if value.shape != expected.shape or value.dtype != np.float32 or expected.dtype != np.float32:
            raise ValueError(f"Atomic equivalence requires the exact full FP32 parameter/gradient schema: {name}")
        flat = value.reshape(-1)
        if any(not np.isfinite(flat[start:start + _CHUNK_ELEMENTS]).all()
               for start in range(0, flat.size, _CHUNK_ELEMENTS)):
            raise FloatingPointError(f"Nonfinite atomic equivalence gradient: {name}")
        schema[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elements += value.size
        total_bytes += value.nbytes
    return {"leaves": len(schema), "parameter_elements": elements, "bytes": total_bytes,
            "all_finite": True, "schema": schema}


def _add_weighted(accumulated, gradient, weight):
    """Use one FP32 mean buffer and bounded temporaries, even for 13 GB trees."""
    if accumulated is None:
        accumulated = {name: np.zeros_like(value, dtype=np.float32) for name, value in gradient.items()}
    for name, value in gradient.items():
        source, target = np.asarray(value).reshape(-1), accumulated[name].reshape(-1)
        for start in range(0, source.size, _CHUNK_ELEMENTS):
            stop = start + _CHUNK_ELEMENTS
            np.add(target[start:stop], np.multiply(source[start:stop], np.float32(weight)), out=target[start:stop])
    return accumulated


def verify_atomic_packing(engine, params, positions, seed):
    """Compare one 4/8/16 bundle with its independent four-sample means.

    Inputs, positions, version and typed RNG key bits require exact identity.
    Only the computed gradient arithmetic permits relative L2 <= 5e-3 across
    packing shapes. Replaying the identical bundle uses the separate strict
    full-mode arithmetic validator. This helper runs once per capacity entry;
    its wall time includes validation and must not enter throughput timers.
    """
    started = time.monotonic()
    report = {"status": "running", "comparison": "atomic_bundle_vs_independent_four_sample_mean",
              "relative_l2_packing_tolerance": 5e-3, "reference_accumulation_dtype": "float32",
              "bounded_temporary_elements": _CHUNK_ELEMENTS, "stages": [],
              "strict_identity_checks": {"positions": True, "parameter_version": True,
                                         "typed_rng_keys": True, "prepared_inputs": True},
              "notes": ["Input/sample/key identity is checked exactly, independently of floating-point tolerances.",
                        "Calls use engine.compute with one unchanged authoritative parameter version.",
                        "Validation/compilation/transfer time is separate from worker throughput measurements.",
                        "Only summaries and small RNG descriptors are retained; no full gradient arrays enter JSON."]}
    original_prepare = None
    had_instance_override = "prepare_task" in engine.__dict__
    instance_override = engine.__dict__.get("prepare_task")
    installed = False
    baseline_inputs = {}
    active_stage = None
    expected_positions = ()
    prepare_calls = 0
    try:
        if not getattr(engine, "atomic_batching", False) or getattr(engine, "training_mode", None) != "full":
            raise ValueError("Atomic packing verification requires full-mode atomic batching")
        version = engine._cached_version
        state = engine._cached_train_state
        if version is None or state is None:
            raise ValueError("Atomic packing verification requires a warm authoritative parameter version")
        units = describe_units(positions, run_seed=engine.seed, version=version,
                               batch_size=engine.config.get("batch_size", 256))
        positions = tuple(position for unit in units for position in unit.sample_positions)
        unit_by_start = {unit.sample_positions[0]: unit for unit in units}
        physical_capacity = getattr(engine, "physical_batch_capacity", len(positions))
        report.update(parameter_version=version, sample_positions=list(positions), atomic_size=4,
                      logical_task_batch_size=len(positions),
                      physical_batch_size=min(len(positions), physical_capacity),
                      physical_batch_capacity=physical_capacity, unit_count=len(units), run_seed=engine.seed,
                      legacy_task_seed=int(seed), legacy_task_seed_ignored_by_atomic_engine=True, rng_scheme=RNG_SCHEME,
                      units=[{"unit_id": unit.unit_id, "version": unit.version,
                              "sample_positions": list(unit.sample_positions), "seed": unit.seed} for unit in units])
        original_prepare = engine.prepare_task

        def observed_prepare(sample_positions, *, version=None):
            nonlocal prepare_calls
            prepare_calls += 1
            if tuple(sample_positions) != expected_positions or version != report["parameter_version"]:
                report["strict_identity_checks"]["positions"] = False
                report["strict_identity_checks"]["parameter_version"] = False
                raise ValueError("Atomic compute prepared a different sample assignment or parameter version")
            if prepare_calls != 1:
                raise ValueError("Atomic equivalence expects exactly one concrete preparation per compute call")
            prepared = original_prepare(sample_positions, version=version)
            if not isinstance(prepared, AtomicPreparedTask) or prepared.sample_positions != expected_positions:
                report["strict_identity_checks"]["positions"] = False
                raise ValueError("Prepared atomic sample identities differ from the requested assignment")
            if prepared.version != version:
                report["strict_identity_checks"]["parameter_version"] = False
                raise ValueError("Prepared atomic parameter version changed")
            descriptors = describe_units(expected_positions, run_seed=engine.seed, version=version,
                                         batch_size=engine.config.get("batch_size", 256))
            expected_keys = unit_rngs(descriptors)
            key_data = np.asarray(engine.jax.device_get(engine.jax.random.key_data(prepared.rngs)))
            canonical_bits = np.asarray(engine.jax.device_get(engine.jax.random.key_data(expected_keys)))
            if (prepared.rngs.shape != expected_keys.shape or prepared.rngs.dtype != expected_keys.dtype
                    or not np.array_equal(key_data, canonical_bits)):
                report["strict_identity_checks"]["typed_rng_keys"] = False
                raise ValueError("Prepared atomic RNG keys differ from the canonical 64-bit unit identities")
            arrays = _input_arrays(engine, prepared)
            summaries = []
            for index, descriptor in enumerate(descriptors):
                start = descriptor.sample_positions[0]
                if descriptor != unit_by_start.get(start):
                    raise ValueError("Repacking changed an atomic unit descriptor")
                if any(value.ndim < 2 or tuple(value.shape[:2]) != (len(descriptors), 4) for value in arrays.values()):
                    report["strict_identity_checks"]["prepared_inputs"] = False
                    raise ValueError("Prepared atomic inputs have inconsistent physical unit shapes")
                current = {name: value[index] for name, value in arrays.items()}
                if active_stage == "bundle":
                    baseline_inputs[start] = {name: value.copy() for name, value in current.items()}
                original = baseline_inputs.get(start)
                if (original is None or set(current) != set(original) or any(
                        current[name].shape != original[name].shape or current[name].dtype != original[name].dtype
                        or not np.array_equal(current[name], original[name]) for name in current)):
                    report["strict_identity_checks"]["prepared_inputs"] = False
                    raise ValueError("Independent atomic computation changed concrete model inputs")
                summaries.append({"unit_id": descriptor.unit_id, "seed": descriptor.seed,
                                  "sample_positions": list(descriptor.sample_positions),
                                  "typed_key_dtype": str(prepared.rngs.dtype), "key_data": key_data[index].tolist(),
                                  "inputs": _input_summary(current)})
            report["stages"][-1]["units"] = summaries
            return prepared

        engine.prepare_task = observed_prepare
        installed = True

        def compute(label, assignment):
            nonlocal active_stage, expected_positions, prepare_calls
            active_stage, expected_positions, prepare_calls = label, tuple(assignment), 0
            row = {"label": label, "sample_positions": list(assignment)}
            report["stages"].append(row)
            phase_started = time.monotonic()
            gradients, metrics = engine.compute(params, version, "atomic-equivalence-" + label, list(assignment), seed)
            if prepare_calls != 1:
                raise ValueError("Atomic compute did not expose exactly one concrete prepared input")
            if engine._cached_version != version or engine._cached_train_state is not state:
                report["strict_identity_checks"]["parameter_version"] = False
                raise ValueError("Atomic equivalence replaced the authoritative parameter state")
            if not np.isfinite(metrics["loss"]):
                raise FloatingPointError("Nonfinite atomic equivalence loss")
            row.update(loss=float(metrics["loss"]), wall_seconds=time.monotonic() - phase_started)
            return gradients, float(metrics["loss"])

        bundled, bundle_loss = compute("bundle", positions)
        report["gradient_schema"] = _gradient_schema(params, bundled)
        replayed, replay_loss = compute("bundle_replay", positions)
        report["bundle_replay"] = validate_gradient_replay(bundled, replayed, training_mode="full")
        report["bundle_replay_loss_absolute_error"] = abs(replay_loss - bundle_loss)
        del replayed
        independent, losses = None, []
        for unit in units:
            gradient, loss = compute("unit-" + str(unit.unit_id), unit.sample_positions)
            _gradient_schema(params, gradient)
            independent = _add_weighted(independent, gradient, 1 / len(units))
            losses.append(loss)
            del gradient
        report["packing_comparison"] = compare_gradient_arrays(independent, bundled)
        report["bundle_loss"] = bundle_loss
        report["independent_mean_loss"] = math.fsum(losses) / len(losses)
        report["loss_absolute_error"] = abs(report["independent_mean_loss"] - bundle_loss)
        relative = report["packing_comparison"]["relative_l2_error"]
        passed = (report["bundle_replay"]["passed"] and relative is not None
                  and relative <= report["relative_l2_packing_tolerance"]
                  and all(report["strict_identity_checks"].values()))
        report["status"] = "passed" if passed else "failed"
    except Exception as error:
        report.update(status="error", error={"type": type(error).__name__, "message": str(error)})
    finally:
        if installed:
            if had_instance_override:
                engine.prepare_task = instance_override
            else:
                del engine.prepare_task
        report["prepare_task_restored"] = not installed or engine.prepare_task == original_prepare
        report["elapsed_seconds"] = time.monotonic() - started
    return report
