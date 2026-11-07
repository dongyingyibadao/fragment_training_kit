"""Bounded FP64 replay diagnostics for native snapshots and FP32 gradients.

Transport and input identity must still be checked exactly. These tolerances
match numeric_validation and apply only to replayed arithmetic, never to
sample/parameter identity. Legacy NVIDIA diagnostics keep their own imports.
"""

import math
from collections.abc import Mapping
import numpy as np


def _gradient_schema(gradients):
    from jade_lotus.native_snapshot import NativeSnapshot

    if isinstance(gradients, NativeSnapshot):
        if gradients.closed:
            raise ValueError("Gradient replay snapshot is closed")
        return gradients.logical_schema
    if not isinstance(gradients, Mapping) or not gradients:
        raise ValueError("Gradient replay requires a nonempty gradient schema")
    schema = {}
    for name, value in gradients.items():
        array = np.asarray(value)
        if not isinstance(name, str) or array.dtype != np.float32:
            raise ValueError(f"Gradient replay requires matching FP32 arrays: {name}")
        schema[name] = {"shape": list(array.shape), "dtype": "float32"}
    return schema


def _fp32_blocks(gradients, name, chunk_elements):
    from jade_lotus.native_snapshot import NativeSnapshot

    if isinstance(gradients, NativeSnapshot):
        yield from gradients.iter_fp32_blocks(name, max_elements=chunk_elements)
    else:
        flat = np.asarray(gradients[name]).reshape(-1)
        for start in range(0, flat.size, chunk_elements):
            yield start, flat[start:start + chunk_elements]


def iter_gradient_chunks(reference, actual, *, chunk_elements=1 << 20):
    """Compare owned native snapshots or FP32 maps without expanding a model."""
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError("chunk_elements must be a positive integer")
    schema = _gradient_schema(reference)
    if schema != _gradient_schema(actual):
        raise ValueError("Gradient replay schema differs")
    for name in sorted(schema):
        left = _fp32_blocks(reference, name, chunk_elements)
        right = _fp32_blocks(actual, name, chunk_elements)
        for (a_start, a), (b_start, b) in zip(left, right, strict=True):
            if a_start != b_start or a.shape != b.shape:
                raise ValueError(f"Gradient replay block coverage differs: {name}")
            yield name, a, b


def compare_gradient_arrays(reference, actual, *, chunk_elements=1 << 20):
    bitwise, maximum, reference_maximum, elements = True, 0.0, 0.0, 0
    errors, norms = [], []
    changed_leaves = set()
    for name, a, b in iter_gradient_chunks(reference, actual, chunk_elements=chunk_elements):
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise FloatingPointError(f"Nonfinite replay gradient: {name}")
        equal = np.array_equal(a, b)
        bitwise &= equal
        if not equal:
            changed_leaves.add(name)
        difference = np.subtract(a, b, dtype=np.float64)
        if a.size:
            maximum = max(maximum, float(np.max(np.abs(difference))))
            reference_maximum = max(reference_maximum, float(np.max(np.abs(a))))
        errors.append(float(np.einsum("i,i->", difference, difference, dtype=np.float64)))
        norms.append(float(np.einsum("i,i->", a, a, dtype=np.float64)))
        elements += a.size
    squared_error, squared_norm = math.fsum(errors), math.fsum(norms)
    relative = math.sqrt(squared_error / squared_norm) if squared_norm else (0.0 if not squared_error else None)
    return {"schema_equal": True, "bitwise_equal": bool(bitwise), "max_abs_error": maximum,
            "relative_l2_error": relative, "reference_l2_norm": math.sqrt(squared_norm),
            "reference_max_abs": reference_maximum, "parameter_elements": elements,
            "changed_leaves": sorted(changed_leaves), "all_finite": True}


def validate_gradient_replay(reference, actual, *, training_mode="lora"):
    if training_mode not in ("lora", "full"):
        raise ValueError("Unknown training mode for gradient replay")
    report = compare_gradient_arrays(reference, actual)
    relative_tolerance, absolute_tolerance = ((1e-6, 1e-7) if training_mode == "full" else (0.0, 0.0))
    passed = (report["bitwise_equal"] or (training_mode == "full"
        and report["relative_l2_error"] is not None
        and report["relative_l2_error"] <= relative_tolerance
        and report["max_abs_error"] <= absolute_tolerance))
    return {**report, "status": "passed" if passed else "failed", "passed": passed,
            "training_mode": training_mode, "relative_l2_tolerance": relative_tolerance,
            "max_abs_tolerance": absolute_tolerance,
            "comparison_scope": "computed_gradient_arithmetic_only"}
