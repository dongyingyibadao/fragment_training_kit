"""Bounded FP64 diagnostics for computed FP32 gradient roundoff.

Transport and input identity must still be checked exactly. These tolerances
apply only to replayed GPU arithmetic, never to sample/parameter identity.
"""

import math
import numpy as np


def compare_gradient_arrays(reference, actual, *, chunk_elements=1 << 20):
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError("chunk_elements must be a positive integer")
    if not reference or set(reference) != set(actual):
        raise ValueError("Gradient replay schema differs")
    bitwise, maximum, reference_maximum, elements = True, 0.0, 0.0, 0
    errors, norms = [], []
    changed_leaves = []
    for name in sorted(reference):
        left, right = np.asarray(reference[name]), np.asarray(actual[name])
        if left.shape != right.shape or left.dtype != np.float32 or right.dtype != np.float32:
            raise ValueError(f"Gradient replay requires matching FP32 arrays: {name}")
        left, right = left.reshape(-1), right.reshape(-1)
        leaf_equal = True
        for start in range(0, left.size, chunk_elements):
            a, b = left[start:start + chunk_elements], right[start:start + chunk_elements]
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise FloatingPointError(f"Nonfinite replay gradient: {name}")
            equal = np.array_equal(a, b)
            leaf_equal &= equal
            difference = np.subtract(a, b, dtype=np.float64)
            if a.size:
                maximum = max(maximum, float(np.max(np.abs(difference))))
                reference_maximum = max(reference_maximum, float(np.max(np.abs(a))))
            errors.append(float(np.einsum("i,i->", difference, difference, dtype=np.float64)))
            norms.append(float(np.einsum("i,i->", a, a, dtype=np.float64)))
        if not leaf_equal:
            changed_leaves.append(name)
        bitwise &= leaf_equal
        elements += left.size
    squared_error, squared_norm = math.fsum(errors), math.fsum(norms)
    relative = math.sqrt(squared_error / squared_norm) if squared_norm else (0.0 if not squared_error else None)
    return {"schema_equal": True, "bitwise_equal": bool(bitwise), "max_abs_error": maximum,
            "relative_l2_error": relative, "reference_l2_norm": math.sqrt(squared_norm),
            "reference_max_abs": reference_maximum, "parameter_elements": elements,
            "changed_leaves": changed_leaves, "all_finite": True}


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
