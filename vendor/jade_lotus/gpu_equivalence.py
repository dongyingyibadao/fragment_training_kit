"""Compare original and optimized data paths inside one warmed worker engine.

This separates input changes from fresh-process GPU compilation differences.
It never constructs an engine, changes model settings, or selects a device.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping

from jade_lotus.fast_data import install_numeric_action_queries


def _flatten(tree, prefix=()):
    if isinstance(tree, Mapping):
        result = {}
        for key in sorted(tree):
            result.update(_flatten(tree[key], prefix + (str(key),)))
        return result
    return {"/".join(prefix): tree}


def _snapshot_inputs(engine, positions):
    """Stack the actual seeded sample fields exactly as the model input path."""
    import numpy as np

    rows = [_flatten(engine._sample(position)) for position in positions]
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("Sample fields differ within the task")
    return {key: np.stack([np.asarray(row[key]) for row in rows], axis=0).copy()
            for key in sorted(keys)}


def _array_digest(value):
    import numpy as np

    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    if value.dtype.hasobject:
        digest.update(json.dumps(value.tolist(), sort_keys=True).encode())
    else:
        digest.update(memoryview(np.ascontiguousarray(value)))
    return digest.hexdigest()


def _summary(arrays):
    import numpy as np

    digest = hashlib.sha256()
    fields = {}
    finite = True
    for key, value in sorted(arrays.items()):
        value = np.asarray(value)
        checksum = _array_digest(value)
        try:
            field_finite = bool(np.isfinite(value).all())
        except TypeError:
            field_finite = None
        finite &= field_finite is not False
        fields[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                       "sha256": checksum, "finite": field_finite}
        digest.update(key.encode())
        digest.update(checksum.encode())
    return {"sha256": digest.hexdigest(), "fields": fields, "all_numeric_fields_finite": finite}


def _compare(reference, actual):
    import numpy as np

    if set(reference) != set(actual):
        return {"schema_equal": False, "bitwise_equal": False,
                "missing": sorted(set(reference) - set(actual)),
                "extra": sorted(set(actual) - set(reference))}
    maximum = error_sq = reference_sq = 0.0
    mismatches = []
    schema_equal = True
    for key in sorted(reference):
        left, right = np.asarray(reference[key]), np.asarray(actual[key])
        if left.shape != right.shape or left.dtype != right.dtype:
            schema_equal = False
            mismatches.append({"field": key, "reason": "shape_or_dtype",
                               "reference_shape": list(left.shape), "actual_shape": list(right.shape),
                               "reference_dtype": str(left.dtype), "actual_dtype": str(right.dtype)})
            continue
        if _array_digest(left) == _array_digest(right):
            try:
                flattened = left.reshape(-1).astype(np.float64)
                if np.isfinite(flattened).all():
                    reference_sq += float(np.dot(flattened, flattened))
            except (TypeError, ValueError):
                pass
            continue
        entry = {"field": key, "reason": "values"}
        try:
            left_flat, right_flat = left.reshape(-1).astype(np.float64), right.reshape(-1).astype(np.float64)
            difference = left_flat - right_flat
            if not np.isfinite(difference).all():
                entry["nonfinite_difference"] = True
            else:
                field_max = float(np.max(np.abs(difference))) if difference.size else 0.0
                maximum = max(maximum, field_max)
                entry["max_abs_error"] = field_max
                error_sq += float(np.dot(difference, difference))
                reference_sq += float(np.dot(left_flat, left_flat))
        except (TypeError, ValueError):
            pass
        mismatches.append(entry)
    relative = (error_sq / reference_sq) ** 0.5 if reference_sq else 0.0 if not mismatches else None
    return {"schema_equal": schema_equal, "bitwise_equal": not mismatches,
            "max_abs_error": maximum, "relative_l2_error": relative, "mismatched_fields": mismatches}


def verify_data_path(engine, params, positions, seed):
    """Check fast -> original -> fast using one existing compiled engine.

    Returns passed/failed/skipped/error. Input or gradient differences fail
    exact equivalence; compute exceptions become an error report. Failure to
    restore the optimized data state raises, so the caller cannot continue
    unknowingly with the wrong dataset path. ``params`` must represent the
    already cached parameter version, as in the normal worker protocol.
    """
    handle = getattr(engine, "_fast_data_handle", None)
    if handle is None or not getattr(handle, "enabled", False):
        return {"status": "skipped", "reason": "fast_data_queries is not active on this engine"}
    version = getattr(engine, "_cached_version", None)
    if version is None or getattr(engine, "_cached_train_state", None) is None:
        return {"status": "skipped", "reason": "engine has no warmed cached parameter version"}
    positions = list(positions)
    if not positions:
        return {"status": "skipped", "reason": "empty sample positions"}
    original_config = engine.config
    missing = object()
    original_flag = original_config.get("fast_data_queries", missing)
    mode = handle.mode
    started = time.monotonic()
    report = {"status": "running", "parameter_version": version, "positions": positions,
              "task_seed": int(seed), "microbatch_size": engine.microbatch_size, "mode": mode,
              "fused_task_gradient": bool(getattr(engine, "fused_task_gradient", False)),
              "disable_activation_checkpointing": bool(getattr(engine, "disable_activation_checkpointing", False)),
              "same_engine": True, "stages": {},
              "notes": ["Reuses the existing engine, parameter version, task shape, seed and compiled callables.",
                        "Input fields are the seeded _sample outputs stacked exactly as in model batch preparation.",
                        "This verifies the data optimization; it does not establish cross-process GPU reproducibility."]}

    def measure(label):
        inputs = _snapshot_inputs(engine, positions)
        gradients, metrics = engine.compute(params, version, f"data-equivalence-{label}", positions, seed)
        # Keep snapshots independent even if a test engine or future adapter
        # reuses mutable host buffers on the next compute call.
        gradients = {key: value.copy() for key, value in gradients.items()}
        report["stages"][label] = {"inputs": _summary(inputs), "gradients": _summary(gradients),
                                    "metrics": metrics}
        return inputs, gradients

    def install():
        replacement = install_numeric_action_queries(engine.dataset, enabled=True, mode=mode)
        if not getattr(replacement, "enabled", False):
            raise RuntimeError(f"Could not restore fast data path: {getattr(replacement, 'reason', 'disabled')}")
        engine._fast_data_handle = replacement
        return replacement

    try:
        before_inputs, before_gradients = measure("fast_before")
        handle.restore()
        engine._fast_data_handle = None
        original_config["fast_data_queries"] = False
        original_inputs, original_gradients = measure("original")
        install()
        original_config["fast_data_queries"] = True
        after_inputs, after_gradients = measure("fast_after")
        full = engine.config.get("training_mode", "lora") == "full"
        if full:
            from jade_lotus.numeric_validation import validate_gradient_replay

            def compare_gradients(reference, actual):
                return validate_gradient_replay(reference, actual, training_mode="full")
        else:
            compare_gradients = _compare
        comparisons = {
            "inputs_fast_vs_original": _compare(before_inputs, original_inputs),
            "inputs_fast_before_vs_after": _compare(before_inputs, after_inputs),
            "gradients_fast_vs_original": compare_gradients(before_gradients, original_gradients),
            "gradients_fast_before_vs_after": compare_gradients(before_gradients, after_gradients),
        }
        report["comparisons"] = comparisons
        finite = all(stage[kind]["all_numeric_fields_finite"]
                     for stage in report["stages"].values() for kind in ("inputs", "gradients"))
        exact_inputs = all(comparisons[key]["bitwise_equal"] for key in
                           ("inputs_fast_vs_original", "inputs_fast_before_vs_after"))
        valid_gradients = all(comparisons[key]["passed" if full else "bitwise_equal"] for key in
                              ("gradients_fast_vs_original", "gradients_fast_before_vs_after"))
        report["computed_gradient_validation"] = "fp32_roundoff_tolerance" if full else "bitwise"
        report["status"] = "passed" if finite and exact_inputs and valid_gradients else "failed"
    except Exception as exc:
        report.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        try:
            active = getattr(engine, "_fast_data_handle", None)
            if active is None or not getattr(active, "enabled", False):
                install()
            report["restored"] = {"enabled": bool(engine._fast_data_handle.enabled),
                                  "mode": engine._fast_data_handle.mode}
        except Exception as exc:
            raise RuntimeError(f"Data-path verification could not restore fast_data_queries: {exc}") from exc
        finally:
            if original_flag is missing:
                original_config.pop("fast_data_queries", None)
            else:
                original_config["fast_data_queries"] = original_flag
            report["elapsed_seconds"] = time.monotonic() - started
    return report
