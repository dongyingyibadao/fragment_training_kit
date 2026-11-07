"""Owned, readonly native gradient snapshots with bounded FP32 replay views.

This module needs NumPy only. BF16 leaves contain their exact little-endian
uint16 words, while actual FP32 leaves retain every FP32 bit. Public creation
copies into independent native dtype flats; there is no implicit whole-model
FP32 conversion or Mapping interface. Closing releases this object's owners;
already obtained arrays/blocks retain their own references and remain stable.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from types import MappingProxyType

import numpy as np

from jade_lotus.array_blob import (
    BF16_WORD,
    FP32,
    decode_bfloat16_exact,
    make_native_wire_policy,
    native_wire_policy_sha256,
    validate_native_wire_policy,
)


_CHECK_ELEMENTS = 1 << 20
_DTYPES = {"bfloat16": BF16_WORD, "float32": FP32}


def _metadata(arrays, wire_policy, *, contiguous=False):
    if (not isinstance(arrays, Mapping) or not arrays
            or any(not isinstance(key, str) for key in arrays)):
        raise ValueError("Native snapshot requires a nonempty named ndarray mapping")
    policy = validate_native_wire_policy(wire_policy)
    rules = {leaf["key"]: leaf["dtype"] for leaf in policy["leaves"]}
    if set(arrays) != set(rules):
        raise ValueError("Native snapshot keys differ from the wire policy")
    values, schema = {}, {}
    for key in sorted(arrays):
        value = arrays[key]
        if (not isinstance(value, np.ndarray) or value.dtype != _DTYPES[rules[key]]
                or (contiguous and not value.flags.c_contiguous)):
            raise ValueError(f"Native snapshot dtype/endianness/layout differs: {key}")
        values[key] = value
        schema[key] = {"shape": list(value.shape), "dtype": "float32"}
    # This also validates nonempty arrays/keys, size limits, canonical schema
    # hashing and complete coverage using the existing wire-policy helpers.
    if make_native_wire_policy(schema, rules) != policy:
        raise ValueError("Native snapshot shape/logical schema differs from the wire policy")
    return values, schema, policy


def _check_finite(arrays):
    for key, value in arrays.items():
        flat = value.reshape(-1)
        for start in range(0, flat.size, _CHECK_ELEMENTS):
            block = flat[start:start + _CHECK_ELEMENTS]
            finite = (not np.any((block & np.uint16(0x7F80)) == np.uint16(0x7F80))
                      if value.dtype == BF16_WORD else bool(np.isfinite(block).all()))
            if not finite:
                raise FloatingPointError(f"Nonfinite native snapshot gradient: {key}")


class NativeSnapshot:
    """Own exact native leaves independently of inputs and reusable staging.

    ``from_arrays`` and the direct constructor copy, including strided inputs.
    Native arrays are sorted, C-contiguous and readonly. FP32 replay blocks are
    always newly allocated, readonly, and bounded by ``max_elements``. Metadata
    remains readable after close; data access/new iteration raises RuntimeError.
    """

    def __init__(self, arrays, wire_policy):
        values, _, policy = _metadata(arrays, wire_policy)
        sizes = {}
        for value in values.values():
            sizes[value.dtype] = sizes.get(value.dtype, 0) + value.size
        flats = {dtype: np.empty(size, dtype=dtype) for dtype, size in sizes.items()}
        offsets, owned = dict.fromkeys(flats, 0), {}
        for key, value in values.items():
            offset = offsets[value.dtype]
            destination = flats[value.dtype][offset:offset + value.size].reshape(value.shape)
            # Handles noncontiguous input without a full-leaf reshape copy.
            np.copyto(destination, value, casting="no")
            owned[key] = destination
            offsets[value.dtype] += value.size
        self._adopt(owned, policy, tuple(flats.values()))

    @classmethod
    def from_arrays(cls, arrays, wire_policy):
        return cls(arrays, wire_policy)

    @classmethod
    def _from_owned_arrays(cls, arrays, wire_policy, *, owners):
        """Private zero-copy adoption of already owned native clone flats.

        The caller transfers ownership of the supplied buffers. Never pass
        reusable pinned staging or any externally mutable input here. All leaf
        views must already be contiguous; validation and finite checks still run.
        """
        result = cls.__new__(cls)
        result._adopt(arrays, wire_policy, tuple(owners))
        return result

    def _adopt(self, arrays, wire_policy, owners):
        values, schema, policy = _metadata(arrays, wire_policy, contiguous=True)
        _check_finite(values)
        for owner in owners:
            if isinstance(owner, np.ndarray):
                owner.setflags(write=False)
        views = {}
        for key, value in values.items():
            value.setflags(write=False)
            view = value.view()
            view.setflags(write=False)
            views[key] = view
        self._arrays = MappingProxyType(views)
        self._owners = owners
        self._wire_policy = policy
        self._wire_policy_sha256 = native_wire_policy_sha256(policy)
        self._logical_schema = schema
        self._native_bytes = sum(value.nbytes for value in values.values())
        self._closed = False

    @property
    def wire_policy(self):
        return copy.deepcopy(self._wire_policy)

    @property
    def wire_policy_sha256(self):
        return self._wire_policy_sha256

    @property
    def logical_schema(self):
        return copy.deepcopy(self._logical_schema)

    @property
    def native_bytes(self):
        return self._native_bytes

    @property
    def closed(self):
        return self._closed

    def _require_open(self):
        if self.closed:
            raise RuntimeError("Native snapshot is closed")

    def native_arrays(self):
        self._require_open()
        return self._arrays

    def iter_fp32_blocks(self, key, max_elements=1 << 20):
        self._require_open()
        if type(max_elements) is not int or max_elements < 1:
            raise ValueError("max_elements must be a positive integer")
        value = self._arrays[key].reshape(-1)

        def blocks():
            for start in range(0, value.size, max_elements):
                self._require_open()
                source = value[start:start + max_elements]
                block = (decode_bfloat16_exact(source) if value.dtype == BF16_WORD
                         else source.copy())
                block.setflags(write=False)
                yield start, block

        return blocks()

    def close(self):
        if not self.closed:
            self._arrays = None
            self._owners = ()
            self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()
