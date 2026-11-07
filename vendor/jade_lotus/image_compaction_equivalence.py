"""Measure image-compaction rounding separately from throughput measurements."""

from __future__ import annotations

import time

from jade_lotus.gpu_equivalence import _summary
from jade_lotus.input_compaction import Pi05ImageCompaction
from jade_lotus.numeric_validation import compare_gradient_arrays, validate_gradient_replay


def _prepared_arrays(engine, prepared):
    """Concrete numeric inputs and key bits, independent of the loss policy."""
    import numpy as np

    atomic = getattr(engine, "atomic_batching", False)
    values = (prepared.observation, prepared.actions) if atomic else prepared
    arrays = {engine.jax.tree_util.keystr(path): np.asarray(engine.jax.device_get(value))
              for path, value in engine.jax.tree_util.tree_flatten_with_path(values)[0]}
    if atomic:
        arrays["atomic_rng_key_data"] = np.asarray(engine.jax.device_get(engine.jax.random.key_data(prepared.rngs)))
    return arrays


def verify_image_compaction(engine, params, positions, seed):
    """Compare compact -> original -> compact on the same loaded parameters.

    The reference uses a separately traced callable because the token shape
    changes. This check is intended for a small microbatch that fits both
    paths. Atomic mode requires relative L2 error <= 5e-3 and unchanged
    resident inputs/keys. Legacy mode preserves its diagnostic-only rounding
    comparison. Policy success equivalence requires a separate evaluation.
    The parameter cache must already be warm at the supplied version. The
    modified attributes and warmed compact callables are restored on
    failure; normal same-version compute preserves the existing parameter cache.
    """
    import numpy as np

    positions = list(positions)
    compaction = engine.input_compaction
    version = engine._cached_version
    if not compaction.enabled or version is None or engine._cached_train_state is None:
        return {"status": "skipped", "reason": "requires active compaction and a warmed parameter version"}
    atomic = getattr(engine, "atomic_batching", False)
    gradient, fused = engine._gradient, engine._fused_gradient
    atomic_gradient = getattr(engine, "_atomic_gradient", None)
    unwrapped = getattr(gradient, "__wrapped__", None)
    if not atomic and unwrapped is None:
        raise ValueError("Compaction verification requires the original traceable gradient function")
    if atomic and (atomic_gradient is None or not callable(getattr(engine, "_make_atomic_gradient", None))):
        raise ValueError("Atomic compaction verification requires the fresh atomic gradient factory")
    prepared = engine.prepare_task(positions, version=version) if atomic else None
    original_inputs = ({name: value.copy() for name, value in _prepared_arrays(engine, prepared).items()}
                       if atomic else {})
    original_input_summary = _summary(original_inputs) if atomic else None
    started = time.monotonic()
    report = {"status": "running", "same_engine": True, "parameter_version": version,
              "positions": list(positions), "task_seed": int(seed),
              "microbatch_size": engine.microbatch_size,
              "atomic_batching": atomic, "physical_batch_size": len(positions),
              "skip_image_keys": list(compaction.skip_image_keys), "stages": {},
              "prepared_inputs": original_input_summary,
              "input_and_rng_keys_bitwise_equal": True if atomic else None,
              "relative_l2_compaction_tolerance": 5e-3 if atomic else None,
              "notes": ["Original and compact token shapes require distinct compiled callables.",
                        "Timings include verification/recompilation and are not throughput results.",
                        "Numerical differences are reported; task success equivalence is not established."]}
    if atomic:
        report.update(atomic_size=4, atomic_units=len(positions) // 4,
                      rng_key_dtype=str(prepared.rngs.dtype),
                      rng_key_data=original_inputs["atomic_rng_key_data"].tolist())

    def validate_inputs():
        if not atomic:
            return
        current = _prepared_arrays(engine, prepared)
        matches = (set(current) == set(original_inputs) and all(
            current[name].shape == original_inputs[name].shape
            and current[name].dtype == original_inputs[name].dtype
            and np.array_equal(current[name], original_inputs[name]) for name in current))
        if atomic:
            matches &= (prepared.sample_positions == tuple(positions) and prepared.version == version
                        and str(prepared.rngs.dtype) == report["rng_key_dtype"]
                        and engine._cached_version == version)
        if not matches:
            report["input_and_rng_keys_bitwise_equal"] = False
            raise ValueError("Compaction verification changed prepared inputs, RNG keys, positions or parameter version")

    def measure(label):
        validate_inputs()
        if atomic:
            values, metrics = engine.compute_prepared(params, version, "image-compaction-" + label, prepared, seed)
        else:
            values, metrics = engine.compute(params, version, "image-compaction-" + label, positions, seed)
        arrays = {key: np.asarray(value) for key, value in values.items()}
        if not np.isfinite(metrics["loss"]) or any(not np.isfinite(a).all() for a in arrays.values()):
            raise FloatingPointError("Nonfinite image-compaction comparison")
        report["stages"][label] = {"gradients": _summary(arrays), "loss": metrics["loss"]}
        validate_inputs()
        return arrays

    try:
        before = measure("compact_before")
        engine.input_compaction = Pi05ImageCompaction()

        if atomic:
            engine._atomic_gradient = engine._make_atomic_gradient()
        else:
            def reference_gradient(*args):
                return unwrapped(*args)

            engine._gradient = engine.jax.jit(reference_gradient)
            engine._fused_gradient = None
        report["reference_callable_rebuilt"] = True
        report["reference_callable_kind"] = "atomic_factory" if atomic else "legacy_gradient"
        reference = measure("original")
        compaction_comparison = compare_gradient_arrays(reference, before)
        del reference
        engine.input_compaction = compaction
        engine._gradient, engine._fused_gradient = gradient, fused
        if atomic:
            engine._atomic_gradient = atomic_gradient
        after = measure("compact_after")
        comparisons = {"gradients_original_vs_compact": compaction_comparison,
                       "gradients_compact_replay": validate_gradient_replay(
                           before, after, training_mode=getattr(engine, "training_mode", "lora"))}
        report["comparisons"] = comparisons
        original_loss = report["stages"]["original"]["loss"]
        compact_loss = report["stages"]["compact_before"]["loss"]
        report["loss_absolute_error"] = abs(original_loss - compact_loss)
        report["loss_relative_error"] = abs(original_loss - compact_loss) / max(abs(original_loss), 1e-30)
        if not comparisons["gradients_compact_replay"]["passed"]:
            raise ValueError("Same compact callable replay exceeded the training-mode arithmetic tolerances")
        if not comparisons["gradients_original_vs_compact"]["schema_equal"]:
            raise ValueError("Compaction changed the trainable gradient schema")
        if atomic and (compaction_comparison["relative_l2_error"] is None
                      or compaction_comparison["relative_l2_error"] > report["relative_l2_compaction_tolerance"]):
            raise ValueError("Compaction gradient exceeded the relative L2 tolerance")
        report["status"] = "compared"
        report["bitwise_equal_to_original"] = comparisons["gradients_original_vs_compact"]["bitwise_equal"]
    except Exception as exc:
        report.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        engine.input_compaction = compaction
        engine._gradient, engine._fused_gradient = gradient, fused
        if atomic:
            engine._atomic_gradient = atomic_gradient
        report["restored"] = True
        report["elapsed_seconds"] = time.monotonic() - started
    return report
