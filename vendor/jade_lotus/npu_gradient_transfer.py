"""Optional pinned native-dtype gradient download for independent NPU workers.

Each actual native dtype has one reusable CPU staging flat. There is still one
device-to-host copy per unique gradient leaf; nonblocking native copies share
one final synchronization before any CPU read. No device flat is allocated and
neither Parameter nor gradient storage is rebound. ``download`` returns owned
FP32 snapshots; ``download_native`` returns an owned BF16-word/FP32 snapshot.
Replay checks and later tasks cannot overwrite either kind of output.

Torch is imported lazily. CPU tests can disable pinning; actual NPU execution
requires verified pinned staging. Calls must follow a completed backward and
must not overlap parameter/gradient mutation. There is no parameter-version
cache: different tasks at the same version always download fresh gradients.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import threading
import time

import numpy as np

from jade_lotus.array_blob import make_native_wire_policy
from jade_lotus.native_snapshot import NativeSnapshot


MODE = "pinned_native_v1"
LEGACY_MODE = "leaf_copy_v1"
_CHECK_CHUNK = 1 << 20


def download_metrics(*, pinned_native=False, pinned=False):
    """Numeric metrics shared by legacy and pinned native download modes.

    gradient_download_fp32_bytes remains the logical FP32-equivalent size.
    Actual native-export output bytes are gradient_native_snapshot_bytes;
    native exports never allocate that logical full FP32 output.
    """
    return {
        "gradient_transfer_pinned_native": int(pinned_native),
        "gradient_transfer_pinned": int(pinned),
        "gradient_validation_seconds": 0.0,
        "gradient_download_allocation_seconds": 0.0,
        "gradient_d2h_enqueue_seconds": 0.0,
        "gradient_d2h_sync_seconds": 0.0,
        "gradient_cpu_widen_snapshot_seconds": 0.0,
        "gradient_finite_seconds": 0.0,
        "gradient_download_total_seconds": 0.0,
        "gradient_download_copy_calls": 0,
        "gradient_download_native_bytes": 0,
        "gradient_download_fp32_bytes": 0,
        "gradient_download_staging_bytes": 0,
        "gradient_download_dtype_groups": 0,
        "gradient_cpu_widen_snapshot_calls": 0,
        "gradient_export_native_snapshot": 0,
        "gradient_native_snapshot_seconds": 0.0,
        "gradient_native_snapshot_bytes": 0,
        "gradient_native_snapshot_copy_calls": 0,
    }


def _finite(array):
    flat = array.reshape(-1)
    return all(np.isfinite(flat[start:start + _CHECK_CHUNK]).all()
               for start in range(0, flat.size, _CHECK_CHUNK))


@dataclass
class _Entry:
    name: str
    parameter: object
    aliases: list[str]
    shape: tuple
    dtype: object
    elements: int
    offset: int
    host_view: object = None


@dataclass
class _Group:
    dtype: object
    elements: int = 0
    entries: list[_Entry] = field(default_factory=list)
    host: object = None


class PinnedNpuGradientTransfer:
    """Download native BF16/FP32 leaves, then widen/snapshot only on CPU.

    Parameter storage may be rebound by the optional parameter uploader before
    backward. Only object identity, shape, dtype and device are frozen here;
    fresh gradient objects/storage after zero_grad(set_to_none=True) are valid.
    Copy/fence or post-copy processing failures poison this instance, so a
    partially filled staging buffer can never produce a later successful read.
    Preflight or allocation errors launch no copies and can be retried.
    """

    def __init__(self, parameters, device, *, torch_module=None, pin_memory=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.device = torch_module.device(device)
        if self.device.type not in ("cpu", "npu"):
            raise ValueError("Pinned gradient download supports NPU and CPU validation only")
        self.pin_memory = self.device.type == "npu" if pin_memory is None else pin_memory
        if type(self.pin_memory) is not bool or (self.device.type == "npu" and not self.pin_memory):
            raise ValueError("Actual NPU gradient download requires pinned host memory")
        if not isinstance(parameters, Mapping) or not parameters or any(not isinstance(name, str) for name in parameters):
            raise ValueError("Gradient download requires a nonempty named parameter mapping")
        self._owner, self.parameters = parameters, dict(parameters)
        self.entries, self.groups = [], {}
        identities = {}
        supported = (torch_module.float32, torch_module.bfloat16)
        for name, parameter in sorted(self.parameters.items()):
            if (not isinstance(parameter, torch_module.nn.Parameter) or parameter.device != self.device
                    or parameter.dtype not in supported or parameter.layout != torch_module.strided
                    or parameter.numel() < 1):
                raise ValueError(f"Unsupported parameter in pinned gradient download: {name}")
            if id(parameter) in identities:
                identities[id(parameter)].aliases.append(name)
                continue
            group = self.groups.setdefault(parameter.dtype, _Group(parameter.dtype))
            entry = _Entry(name, parameter, [name], tuple(parameter.shape), parameter.dtype,
                           parameter.numel(), group.elements)
            group.elements += entry.elements
            group.entries.append(entry)
            self.entries.append(entry)
            identities[id(parameter)] = entry
        self.poisoned = False
        self._lock = threading.Lock()

    def _validate(self):
        if set(self._owner) != set(self.parameters):
            raise ValueError("Gradient download parameter schema changed")
        gradients = []
        for entry in self.entries:
            parameter = entry.parameter
            if (any(self._owner[name] is not parameter for name in entry.aliases)
                    or tuple(parameter.shape) != entry.shape or parameter.dtype != entry.dtype
                    or parameter.device != self.device or parameter.layout != self.torch.strided):
                raise ValueError(f"Parameter identity/shape/dtype changed during gradient download: {entry.name}")
            gradient = parameter.grad
            if gradient is None:
                raise ValueError(f"Unexpected missing gradient in the active schema: {entry.name}")
            if (gradient.dtype != entry.dtype or gradient.device != self.device
                    or tuple(gradient.shape) != entry.shape or gradient.layout != self.torch.strided
                    or not gradient.is_contiguous()):
                raise ValueError(f"Unsupported native gradient layout/dtype/shape: {entry.name}")
            gradients.append(gradient.detach())
        return gradients

    def _allocate(self):
        for group in self.groups.values():
            if group.host is None:
                group.host = self.torch.empty(group.elements, dtype=group.dtype, device="cpu",
                                               pin_memory=self.pin_memory)
            if self.pin_memory and not group.host.is_pinned():
                raise RuntimeError("NPU gradient staging allocation is not pinned")
            for entry in group.entries:
                if entry.host_view is None:
                    entry.host_view = group.host.narrow(0, entry.offset, entry.elements).view(entry.shape)

    def _synchronize(self):
        if self.device.type == "npu":
            self.torch.npu.synchronize(self.device)

    def download(self):
        return self._locked_download(native=False)

    def download_native(self):
        """Return an independent NativeSnapshot without BF16 FP32 widening."""
        return self._locked_download(native=True)

    def _locked_download(self, *, native):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Pinned gradient download is already in progress")
        try:
            return self._download(native=native)
        finally:
            self._lock.release()

    def _download(self, *, native=False):
        began = time.monotonic()
        metrics = download_metrics(pinned_native=True, pinned=self.pin_memory)
        metrics["gradient_export_native_snapshot"] = int(native)
        if self.poisoned:
            raise RuntimeError("Pinned gradient downloader is unusable after a failed transfer")
        gradients = self._validate()
        policy = (make_native_wire_policy(
            {name: {"shape": list(parameter.shape), "dtype": "float32"}
             for name, parameter in self.parameters.items()},
            {name: str(parameter.dtype).removeprefix("torch.")
             for name, parameter in self.parameters.items()}) if native else None)
        metrics["gradient_validation_seconds"] = time.monotonic() - began
        phase = time.monotonic()
        self._allocate()
        metrics["gradient_download_allocation_seconds"] = time.monotonic() - phase
        synchronized = False
        try:
            phase = time.monotonic()
            with self.torch.no_grad():
                for entry, gradient in zip(self.entries, gradients, strict=True):
                    entry.host_view.copy_(gradient, non_blocking=self.pin_memory)
                    metrics["gradient_download_copy_calls"] += 1
            metrics["gradient_d2h_enqueue_seconds"] = time.monotonic() - phase
            phase = time.monotonic()
            # No CPU conversion, numpy view, finite check or staging reuse is
            # allowed before this fence completes all asynchronous copies.
            self._synchronize()
            synchronized = True
            metrics["gradient_d2h_sync_seconds"] = time.monotonic() - phase
            phase, snapshots = time.monotonic(), {}
            with self.torch.no_grad():
                for dtype, group in self.groups.items():
                    # .float() aliases an FP32 tensor, so FP32 needs clone().
                    # BF16 widening is exact and already allocates owned FP32.
                    snapshot = group.host.clone() if native or dtype == self.torch.float32 else group.host.float()
                    snapshots[dtype] = (snapshot.view(self.torch.uint16).numpy()
                                        if native and dtype == self.torch.bfloat16 else snapshot.numpy())
                    counter = "gradient_native_snapshot_copy_calls" if native else "gradient_cpu_widen_snapshot_calls"
                    metrics[counter] += 1
            result = {}
            for entry in self.entries:
                value = snapshots[entry.dtype][entry.offset:entry.offset + entry.elements].reshape(entry.shape)
                for name in entry.aliases:
                    result[name] = value
            timer = "gradient_native_snapshot_seconds" if native else "gradient_cpu_widen_snapshot_seconds"
            metrics[timer] = time.monotonic() - phase
            phase = time.monotonic()
            if native:
                # The same-dtype clones above already own their storage. The
                # private factory validates native bits without another copy
                # or a full BF16->FP32 expansion. This timer also includes the
                # small policy/schema binding work performed by the factory.
                result = NativeSnapshot._from_owned_arrays(result, policy, owners=snapshots.values())
                metrics["gradient_native_snapshot_bytes"] = result.native_bytes
            else:
                for dtype, snapshot in snapshots.items():
                    if not _finite(snapshot):
                        # Only the failure path rescans leaves to identify the
                        # offending key; successful scans cover each flat once.
                        bad = next(entry.name for entry in self.groups[dtype].entries if not _finite(result[entry.name]))
                        raise FloatingPointError(f"Nonfinite NPU gradient: {bad}")
            metrics["gradient_finite_seconds"] = time.monotonic() - phase
        except BaseException:
            self.poisoned = True
            if not synchronized:
                try:
                    self._synchronize()
                except BaseException:
                    pass
            raise
        metrics["gradient_download_native_bytes"] = sum(entry.elements * entry.parameter.element_size()
                                                       for entry in self.entries)
        metrics["gradient_download_fp32_bytes"] = sum(entry.elements * 4 for entry in self.entries)
        metrics["gradient_download_staging_bytes"] = sum(group.host.numel() * group.host.element_size()
                                                        for group in self.groups.values())
        metrics["gradient_download_dtype_groups"] = len(self.groups)
        metrics["gradient_download_total_seconds"] = time.monotonic() - began
        return result, metrics
