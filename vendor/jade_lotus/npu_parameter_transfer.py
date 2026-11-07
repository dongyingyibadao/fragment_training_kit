"""Optional native-dtype parameter upload for standalone NPU gradient workers.

The CPU FP32 authority stays unchanged. Each native parameter dtype has one
reusable pinned host buffer and one device buffer. Parameter objects and tied
aliases survive; only their data storage is rebound, once, after a successful
initial upload. This must be installed before constructing an autograd graph,
and is not intended for a model already wrapped in DDP or a device optimizer.

Torch is imported lazily so CPU bootstrap/schema readers do not load a device
backend. CPU execution without pinning exercises packing and autograd in tests;
actual NPU execution always requires pinned memory and a synchronization fence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np


MODE = "batched_pinned_v1"
LEGACY_MODE = "leaf_copy_v1"
_CHECK_CHUNK = 1 << 20
_ALIGNMENT_BYTES = 512


def upload_metrics(*, batched=False, pinned=False):
    """Numeric-only metrics, including a zero-work parameter-cache hit."""
    return {
        "parameter_transfer_batched": int(batched),
        "parameter_transfer_pinned": int(pinned),
        "parameter_upload_cache_hit": 0,
        "parameter_validation_seconds": 0.0,
        "parameter_pack_seconds": 0.0,
        "parameter_upload_allocation_seconds": 0.0,
        "parameter_h2d_copy_seconds": 0.0,
        "parameter_h2d_sync_seconds": 0.0,
        "parameter_storage_bind_seconds": 0.0,
        "parameter_upload_total_seconds": 0.0,
        "parameter_upload_copy_calls": 0,
        "parameter_upload_source_bytes": 0,
        "parameter_upload_native_bytes": 0,
        "parameter_upload_packed_bytes": 0,
        "parameter_upload_dtype_groups": 0,
    }


def _finite(array):
    flat = array.reshape(-1)
    return all(np.isfinite(flat[start:start + _CHECK_CHUNK]).all()
               for start in range(0, flat.size, _CHECK_CHUNK))


def _storage_signature(parameter):
    return (parameter.untyped_storage().data_ptr(), parameter.storage_offset(), tuple(parameter.stride()))


@dataclass
class _Entry:
    name: str
    parameter: object
    aliases: list[str]
    shape: tuple
    dtype: object
    elements: int
    offset: int
    signature: tuple
    host_view: object = None
    device_view: object = None


@dataclass
class _Group:
    dtype: object
    elements: int = 0
    entries: list[_Entry] = field(default_factory=list)
    host: object = None
    device: object = None


class BatchedNpuParameterTransfer:
    """Preserve FP32-to-native rounding while replacing per-leaf device copies.

    ``version`` is only a local cache token. The caller must invalidate it when
    the authoritative generation or SHA changes, including equal step numbers.
    Validation/packing failure does not promote the new version. A device-copy,
    synchronization or binding failure poisons this uploader: a partially
    uploaded model must never be used through an old cached token.
    """

    def __init__(self, parameters, device, *, torch_module=None, pin_memory=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.device = torch_module.device(device)
        if self.device.type not in ("cpu", "npu"):
            raise ValueError("Batched parameter upload supports NPU and CPU validation only")
        self.pin_memory = self.device.type == "npu" if pin_memory is None else pin_memory
        if type(self.pin_memory) is not bool or (self.device.type == "npu" and not self.pin_memory):
            raise ValueError("Actual NPU batched parameter upload requires pinned host memory")
        self.parameters = dict(parameters)
        if not self.parameters:
            raise ValueError("Batched parameter upload requires a nonempty parameter schema")
        self.entries, self.groups = [], {}
        identities, storages = {}, {}
        supported = (torch_module.float32, torch_module.bfloat16, torch_module.float16)
        for name, parameter in sorted(self.parameters.items()):
            if (not isinstance(name, str) or not isinstance(parameter, torch_module.nn.Parameter)
                    or parameter.device != self.device or parameter.dtype not in supported
                    or parameter.layout != torch_module.strided or not parameter.is_contiguous()
                    or parameter.numel() < 1):
                raise ValueError(f"Unsupported parameter in batched upload: {name}")
            if id(parameter) in identities:
                identities[id(parameter)].aliases.append(name)
                continue
            signature = _storage_signature(parameter)
            if signature[0] in storages:
                raise ValueError("Distinct Parameter objects sharing storage require an explicit alias schema")
            storages[signature[0]] = name
            group = self.groups.setdefault(parameter.dtype, _Group(parameter.dtype))
            alignment = _ALIGNMENT_BYTES // parameter.element_size()
            offset = (group.elements + alignment - 1) // alignment * alignment
            entry = _Entry(name, parameter, [name], tuple(parameter.shape), parameter.dtype,
                           parameter.numel(), offset, signature)
            group.elements = offset + entry.elements
            group.entries.append(entry)
            self.entries.append(entry)
            identities[id(parameter)] = entry
        self.version = None
        self.bound = False
        self.poisoned = False

    def invalidate(self):
        self.version = None

    def _validate_layout(self):
        for entry in self.entries:
            parameter = entry.parameter
            if (any(self.parameters[name] is not parameter for name in entry.aliases)
                    or tuple(parameter.shape) != entry.shape or parameter.dtype != entry.dtype
                    or parameter.device != self.device or _storage_signature(parameter) != entry.signature):
                raise ValueError(f"Parameter identity/layout changed after upload setup: {entry.name}")

    def _validate_sources(self, params):
        if set(params) != set(self.parameters):
            raise ValueError("CPU authority parameter keys differ from batched upload schema")
        arrays = {}
        for entry in self.entries:
            canonical = None
            for name in entry.aliases:
                value = np.asarray(params[name])
                if value.dtype != np.float32 or tuple(value.shape) != entry.shape or not _finite(value):
                    raise ValueError(f"Invalid CPU authority parameter: {name}")
                if canonical is not None and not np.array_equal(value.view(np.uint32), canonical.view(np.uint32)):
                    raise ValueError(f"Conflicting CPU authority values for tied parameter alias: {name}")
                canonical = value if canonical is None else canonical
            arrays[entry.name] = canonical
        return arrays

    def _allocate(self):
        torch = self.torch
        for group in self.groups.values():
            if group.host is None:
                # Initialize alignment gaps once; subsequent versions overwrite
                # all parameter regions, preserving deterministic padding.
                group.host = torch.zeros(group.elements, dtype=group.dtype, device="cpu",
                                         pin_memory=self.pin_memory)
            if self.pin_memory and not group.host.is_pinned():
                raise RuntimeError("NPU parameter staging allocation is not pinned")
            if group.device is None:
                group.device = torch.empty(group.elements, dtype=group.dtype, device=self.device)
            for entry in group.entries:
                entry.host_view = group.host.narrow(0, entry.offset, entry.elements).view(entry.shape)
                entry.device_view = group.device.narrow(0, entry.offset, entry.elements).view(entry.shape)

    def _synchronize(self):
        if self.device.type == "npu":
            self.torch.npu.synchronize(self.device)

    def load(self, params, version):
        began = time.monotonic()
        metrics = upload_metrics(batched=True, pinned=self.pin_memory)
        if type(version) is not int or version < 0:
            raise ValueError("Parameter version must be a nonnegative integer")
        if self.poisoned:
            raise RuntimeError("Batched parameter uploader is unusable after a device transfer failure")
        self._validate_layout()
        if self.version == version:
            metrics["parameter_upload_cache_hit"] = 1
            return metrics
        if any(entry.parameter.grad is not None for entry in self.entries):
            raise ValueError("Parameter upload requires cleared gradients and no live autograd graph")
        arrays = self._validate_sources(params)
        metrics["parameter_validation_seconds"] = time.monotonic() - began
        phase = time.monotonic()
        if not self.bound:
            self._allocate()
        metrics["parameter_upload_allocation_seconds"] = time.monotonic() - phase
        phase = time.monotonic()
        with self.torch.no_grad():
            for entry in self.entries:
                source = arrays[entry.name]
                # Positive-stride noncontiguous arrays work with from_numpy;
                # normalize negative strides before constructing a Torch view.
                if any(stride < 0 for stride in source.strides):
                    source = np.ascontiguousarray(source)
                # Native Torch conversion uses round-to-nearest, ties-to-even
                # for FP32->BF16, as in the existing copy_ conversion path.
                entry.host_view.copy_(self.torch.from_numpy(source))
        metrics["parameter_pack_seconds"] = time.monotonic() - phase
        self.version = None
        try:
            phase = time.monotonic()
            with self.torch.no_grad():
                for group in self.groups.values():
                    group.device.copy_(group.host, non_blocking=self.pin_memory)
                    metrics["parameter_upload_copy_calls"] += 1
            metrics["parameter_h2d_copy_seconds"] = time.monotonic() - phase
            phase = time.monotonic()
            # Host staging is never reused while an asynchronous copy can
            # still reference it; every successful load ends at this fence.
            self._synchronize()
            metrics["parameter_h2d_sync_seconds"] = time.monotonic() - phase
            phase = time.monotonic()
            if not self.bound:
                with self.torch.no_grad():
                    for entry in self.entries:
                        entry.parameter.data = entry.device_view
                        entry.signature = _storage_signature(entry.parameter)
                self.bound = True
            metrics["parameter_storage_bind_seconds"] = time.monotonic() - phase
        except BaseException:
            self.poisoned = True
            # Best effort drain only. A failed fence never clears the poison,
            # even if a later synchronization happens to succeed.
            try:
                self._synchronize()
            except BaseException:
                pass
            raise
        self.version = version
        metrics["parameter_upload_source_bytes"] = sum(value.nbytes for value in arrays.values())
        metrics["parameter_upload_native_bytes"] = sum(entry.elements * entry.parameter.element_size()
                                                        for entry in self.entries)
        metrics["parameter_upload_packed_bytes"] = sum(group.device.numel() * group.device.element_size()
                                                        for group in self.groups.values())
        metrics["parameter_upload_dtype_groups"] = len(self.groups)
        metrics["parameter_upload_total_seconds"] = time.monotonic() - began
        return metrics
