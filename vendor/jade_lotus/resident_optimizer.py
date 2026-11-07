"""Optional device-resident FP32 AdamW, independent of the fragment protocol.

The forward model keeps its explicit BF16/FP32 dtypes.  This adapter owns a
separate FP32 master and two FP32 moments; copying OpenPI's native AdamW would
instead give BF16 moments to BF16 parameters.  It consumes an already reduced
global gradient.  It neither creates a communication group nor commits a
distributed update or a durable checkpoint.

CPU execution validates the arithmetic and restore contract without torch-npu.
NPU construction requires an explicit memory budget and a measured training
peak before any master/moment allocation.  Device FP32 norms and Adam kernels
need numerical validation against the CPU FP64-norm reference, not a claim of
bitwise equality.  ``host_fp64`` is a reference/debug mode; on NPU it downloads
the gradient in bounded pieces.  An existing authoritative norm can instead
be supplied to ``update`` without downloading the gradient again.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import math
import time

import numpy as np


FORMAT = "jade-resident-fp32-adamw-v1"
DEFAULT_CHUNK_ELEMENTS = 16 * 1024 * 1024  # 64 MiB FP32 optimizer views.


def _integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def estimate_resident_optimizer_memory(parameter_elements, active_native_bytes, *,
                                       native_model_bytes=None, training_peak_bytes=None,
                                       chunk_elements=DEFAULT_CHUNK_ELEMENTS,
                                       input_gradient_device_bytes=0,
                                       headroom_bytes=2 * 1024**3):
    """Return a planning estimate, not a backend temporary-memory guarantee.

    The update estimate conservatively retains the forward model's native
    gradients alongside the incoming FP32 global gradient.  The FP32 gradient
    is released after each step, so it is absent from the next backward peak.
    Allocator reserve, communication libraries and other processes must fit in
    the caller's headroom/budget.  Real device peaks remain a required check.
    """
    count = _integer(parameter_elements, "parameter_elements")
    native = _integer(active_native_bytes, "active_native_bytes")
    full_model = native if native_model_bytes is None else _integer(native_model_bytes, "native_model_bytes")
    if full_model < native:
        raise ValueError("native_model_bytes cannot exclude active weights")
    chunk = min(_integer(chunk_elements, "chunk_elements"), count)
    input_gradient = _integer(input_gradient_device_bytes, "input_gradient_device_bytes", minimum=0)
    headroom = _integer(headroom_bytes, "headroom_bytes", minimum=0)
    if training_peak_bytes is not None:
        _integer(training_peak_bytes, "training_peak_bytes")
        if training_peak_bytes < full_model:
            raise ValueError("training_peak_bytes cannot be below the full model")
    state = count * 4 * 3
    gradient = count * 4
    temporary = chunk * 4 * 2
    update_static = full_model + native + state + gradient + input_gradient
    backward = None if training_peak_bytes is None else training_peak_bytes + state
    peak = max(update_static + temporary, backward or 0) + headroom
    return {"parameter_elements": count, "active_native_bytes": native,
            "native_model_bytes": full_model, "fp32_state_bytes": state,
            "fp32_gradient_bytes": gradient, "optimizer_temporary_estimate_bytes": temporary,
            "input_gradient_device_bytes": input_gradient,
            "update_static_bytes": update_static,
            "backward_estimated_bytes": backward, "headroom_bytes": headroom,
            "estimated_peak_bytes": peak}


class ResidentAdamW:
    """Keep the complete canonical FP32 master/m/v on one CPU or NPU.

    ``parameters`` contains canonical, unique active model leaves; seven
    structurally unused OpenPI leaves and tied aliases must be handled by the
    caller's existing schema.  ``initial_params`` optionally provides the
    precise FP32 authority instead of widening the model's native weights.

    Calls are serialized.  A device/update/copy failure poisons the instance:
    only loading a complete known state can make it usable again.  The caller
    must fence updates and keep a recoverable p/m/v copy outside a preemptible
    device; local success is not a durable distributed commit.
    """

    def __init__(self, parameters, config, *, device=None, initial_params=None):
        import torch
        from jade_lotus.coordinator import learning_rate

        self.torch, self._learning_rate = torch, learning_rate
        self.config = copy.deepcopy(dict(config))
        if self.config.get("training_mode") != "full":
            raise ValueError("Resident AdamW requires explicit full training")
        if self.config.get("resident_optimizer_precision", "fp32_master") != "fp32_master":
            raise ValueError("Only explicit fp32_master precision is implemented")
        _integer(self.config.get("target_steps"), "target_steps")
        if not isinstance(parameters, Mapping) or not parameters:
            raise ValueError("Canonical active parameters must be a nonempty mapping")
        if any(not isinstance(key, str) or not key for key in parameters):
            raise ValueError("Parameter names must be nonempty strings")
        self.parameters = {key: parameters[key] for key in sorted(parameters)}
        first = next(iter(self.parameters.values()))
        if not torch.is_tensor(first):
            raise ValueError("Model parameters must be Torch tensors")
        requested = first.device if device is None else torch.device(device)
        self.device = first.device if requested.type == first.device.type and requested.index is None else requested
        if self.device.type not in ("cpu", "npu"):
            raise ValueError("Resident AdamW supports CPU validation and NPU only")
        self.shapes, self.spans, schema, offset, storage_ranges = {}, {}, [], 0, {}
        for key, value in self.parameters.items():
            if (not torch.is_tensor(value) or value.device != self.device
                    or value.dtype not in (torch.float32, torch.bfloat16)
                    or value.layout != torch.strided or not value.is_contiguous() or not value.numel()):
                raise ValueError(f"Invalid native model parameter: {key}")
            storage = value.untyped_storage().data_ptr()
            lower = value.storage_offset() * value.element_size()
            upper = lower + value.numel() * value.element_size()
            occupied = storage_ranges.setdefault(storage, [])
            if any(lower < previous_upper and previous_lower < upper
                   for previous_lower, previous_upper in occupied):
                raise ValueError("Canonical parameters must not contain shared storage aliases")
            occupied.append((lower, upper))
            self.shapes[key] = tuple(value.shape)
            self.spans[key] = (offset, offset + value.numel())
            schema.append({"name": key, "shape": list(value.shape),
                           "native_dtype": str(value.dtype).removeprefix("torch.")})
            offset += value.numel()
        self.schema, self.elements = schema, offset
        self.chunk_elements = _integer(self.config.get("resident_optimizer_chunk_elements", DEFAULT_CHUNK_ELEMENTS),
                                       "resident_optimizer_chunk_elements")
        self.norm_chunk_elements = _integer(self.config.get("cpu_adam_norm_chunk_elements", 1 << 20),
                                            "cpu_adam_norm_chunk_elements")
        self.norm_mode = self.config.get("resident_optimizer_norm_mode", "device_fp32")
        if self.norm_mode not in ("device_fp32", "host_fp64"):
            raise ValueError("resident_optimizer_norm_mode must be device_fp32 or host_fp64")
        self.gradient_input_mode = self.config.get("resident_optimizer_gradient_input_mode", "host")
        if self.gradient_input_mode not in ("host", "device"):
            raise ValueError("resident_optimizer_gradient_input_mode must be host or device")
        self.contract = {"precision": "fp32_master_and_moments", "betas": [0.9, 0.95],
                         "eps": 1e-8, "weight_decay": 1e-10,
                         "clip_gradient_norm": float(self.config.get("clip_gradient_norm", 1.0)),
                         "peak_lr": float(self.config.get("peak_lr", 5e-5)),
                         "end_lr": float(self.config.get("end_lr", 5e-6)),
                         "warmup_steps": int(self.config.get("warmup_steps", 200)),
                         "decay_steps": int(self.config.get("decay_steps", self.config["target_steps"])),
                         "target_steps": self.config["target_steps"], "norm_mode": self.norm_mode}
        if (any(not math.isfinite(self.contract[key]) or self.contract[key] < 0
                for key in ("peak_lr", "end_lr", "clip_gradient_norm"))
                or self.contract["warmup_steps"] < 0 or self.contract["decay_steps"] < 0):
            raise ValueError("Invalid learning-rate or clipping configuration")
        native_bytes = sum(value.numel() * value.element_size() for value in self.parameters.values())
        full_bytes = self.config.get("resident_optimizer_native_model_bytes")
        training_peak = self.config.get("resident_optimizer_training_peak_bytes")
        budget = self.config.get("resident_optimizer_memory_budget_bytes")
        if self.device.type == "npu" and any(value is None for value in (full_bytes, training_peak, budget)):
            raise ValueError("NPU memory preflight requires native_model_bytes, training_peak_bytes and "
                             "memory_budget_bytes with the resident_optimizer_ prefix")
        self.memory_estimate = estimate_resident_optimizer_memory(
            self.elements, native_bytes, native_model_bytes=full_bytes, training_peak_bytes=training_peak,
            chunk_elements=self.chunk_elements,
            input_gradient_device_bytes=(4 * self.elements if self.gradient_input_mode == "device" else 0),
            headroom_bytes=self.config.get("resident_optimizer_headroom_bytes", 2 * 1024**3))
        if budget is not None and self.memory_estimate["estimated_peak_bytes"] > _integer(budget, "memory_budget_bytes"):
            raise ValueError(f"Resident optimizer memory estimate {self.memory_estimate['estimated_peak_bytes']} "
                             f"exceeds budget {budget}; no master/moment state was allocated")
        if initial_params is not None and (not isinstance(initial_params, Mapping)
                                           or set(initial_params) != set(self.parameters)):
            raise ValueError("Initial FP32 parameter leaves differ from the canonical schema")
        initial = ({key: self._fp32_leaf(key, initial_params[key]) for key in self.parameters}
                   if initial_params is not None else self.parameters)
        if (self.device.type == "npu" and initial_params is not None
                and any(value.device.type != "cpu" for value in initial.values())):
            raise ValueError("NPU initial FP32 state must come from CPU to avoid an unbudgeted full device copy")
        if any(not self._finite(value) for value in initial.values()):
            raise FloatingPointError("Initial parameters must be finite")
        self.master = torch.empty(self.elements, dtype=torch.float32, device=self.device)
        self.exp_avg = torch.zeros_like(self.master)
        self.exp_avg_sq = torch.zeros_like(self.master)
        with torch.no_grad():
            for key, value in initial.items():
                lower, upper = self.spans[key]
                self.master[lower:upper].view(self.shapes[key]).copy_(value)
        # Bounded parameter views prevent full-model Adam sqrt/denominator
        # temporaries.  All views share the three canonical flat allocations.
        self._chunks = [torch.nn.Parameter(self.master[start:start + self.chunk_elements])
                        for start in range(0, self.elements, self.chunk_elements)]
        self.optimizer = torch.optim.AdamW(self._chunks, lr=learning_rate(self.config, 0),
                                          betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10,
                                          foreach=False, fused=False)
        for index, parameter in enumerate(self._chunks):
            start, end = index * self.chunk_elements, index * self.chunk_elements + parameter.numel()
            self.optimizer.state[parameter] = {"step": torch.tensor(0.0, dtype=torch.float32),
                "exp_avg": self.exp_avg[start:end], "exp_avg_sq": self.exp_avg_sq[start:end]}
        self.step, self._poisoned = 0, False
        self.copy_to_model()

    def _finite(self, value):
        flat = value.detach().reshape(-1)
        checks = [self.torch.isfinite(flat[start:start + self.chunk_elements]).all()
                  for start in range(0, flat.numel(), self.chunk_elements)]
        return bool(self.torch.stack(checks).all().item())

    def _fp32_leaf(self, key, value):
        torch = self.torch
        if isinstance(value, np.ndarray):
            if value.dtype != np.float32:
                raise ValueError(f"FP32 leaf required: {key}")
            value = torch.from_numpy(value)
        if (not torch.is_tensor(value) or value.dtype != torch.float32
                or tuple(value.shape) != self.shapes[key] or value.layout != torch.strided):
            raise ValueError(f"FP32 leaf shape/dtype differs: {key}")
        if value.device.type != "cpu" and value.device != self.device:
            raise ValueError(f"FP32 leaf belongs to another device: {key}")
        return value.detach()

    def _synchronize(self):
        if self.device.type == "npu":
            self.torch.npu.synchronize(self.device)

    def _ready(self):
        if self._poisoned:
            raise RuntimeError("Resident optimizer is poisoned; restore a complete known state")

    def copy_to_model(self):
        """Copy FP32 masters into existing native Parameter objects."""
        self._ready()
        try:
            for entry in self.schema:
                value = self.parameters[entry["name"]]
                if (tuple(value.shape) != tuple(entry["shape"]) or value.device != self.device
                        or str(value.dtype).removeprefix("torch.") != entry["native_dtype"]):
                    raise ValueError("Native model schema changed after optimizer construction")
            with self.torch.no_grad():
                for key, parameter in self.parameters.items():
                    lower, upper = self.spans[key]
                    parameter.copy_(self.master[lower:upper].view(self.shapes[key]))
            self._synchronize()
        except BaseException:
            self._poisoned = True
            raise

    def _norm(self, gradient):
        if self.norm_mode == "device_fp32":
            return float(self.torch.linalg.vector_norm(gradient, dtype=self.torch.float32).item())
        squares = []
        for start in range(0, self.elements, self.norm_chunk_elements):
            chunk = gradient[start:start + self.norm_chunk_elements].detach().cpu().numpy()
            with np.errstate(over="ignore", invalid="ignore"):
                squares.append(float(np.einsum("i,i->", chunk, chunk, dtype=np.float64,
                                               casting="safe", optimize=False)))
        return math.sqrt(math.fsum(squares))

    def update(self, global_gradients, *, expected_step=None, global_norm=None):
        """Apply one already averaged global gradient exactly once locally.

        ``expected_step`` fences the current pre-update step.  ``global_norm``
        may be the authoritative CPU FP64 norm of this exact gradient; binding
        it to a version and gradient digest is the caller's responsibility.
        Input gradients are never clipped or changed in place.
        """
        self._ready()
        if expected_step is not None:
            _integer(expected_step, "expected_step", minimum=0)
            if expected_step != self.step:
                raise ValueError("Expected step differs from resident optimizer step")
        if self.step >= self.config["target_steps"]:
            raise ValueError("Resident optimizer already reached target_steps")
        if not isinstance(global_gradients, Mapping) or set(global_gradients) != set(self.parameters):
            raise ValueError("Global gradient leaves differ from the canonical schema")
        sources = {key: self._fp32_leaf(key, global_gradients[key]) for key in self.parameters}
        if (self.device.type == "npu" and self.gradient_input_mode == "host"
                and any(value.device.type != "cpu" for value in sources.values())):
            raise ValueError("Host-gradient memory contract rejects a device-resident gradient source")
        if global_norm is not None:
            if isinstance(global_norm, bool) or not isinstance(global_norm, (int, float)) or not math.isfinite(global_norm) or global_norm < 0:
                raise ValueError("Provided global norm must be finite and nonnegative")
        started, gradient = time.perf_counter(), self.torch.empty_like(self.master)
        try:
            for key, value in sources.items():
                lower, upper = self.spans[key]
                gradient[lower:upper].view(self.shapes[key]).copy_(value)
            if not self._finite(gradient):
                raise FloatingPointError("Global gradient must be finite")
            norm = self._norm(gradient) if global_norm is None else float(global_norm)
            if not math.isfinite(norm):
                raise FloatingPointError("Global gradient norm is not finite")
            clip = self.contract["clip_gradient_norm"]
            scale = clip / norm if norm > clip else 1.0  # CPU reference: no added epsilon.
            if scale != 1.0:
                gradient.mul_(scale)
            lr = self._learning_rate(self.config, self.step)
            self.optimizer.param_groups[0]["lr"] = lr
            for index, parameter in enumerate(self._chunks):
                start = index * self.chunk_elements
                parameter.grad = gradient[start:start + parameter.numel()]
            try:
                self.optimizer.step()
                self._synchronize()
                if not all(self._finite(value) for value in (self.master, self.exp_avg, self.exp_avg_sq)):
                    raise FloatingPointError("Updated FP32 parameters or moments are not finite")
                self.step += 1
                self.copy_to_model()
            except BaseException:
                self._poisoned = True
                raise
            return {"step": self.step, "lr": lr, "grad_norm": norm, "gradient_clip_scale": scale,
                    "resident_optimizer_norm_fp64": int(global_norm is None and self.norm_mode == "host_fp64"),
                    "resident_optimizer_norm_provided": int(global_norm is not None),
                    "resident_optimizer_chunks": len(self._chunks),
                    "resident_optimizer_seconds": time.perf_counter() - started}
        finally:
            # Do not carry an extra full FP32 global gradient into the next
            # backward peak.  The device allocator may reuse its freed blocks.
            self.optimizer.zero_grad(set_to_none=True)
            del gradient

    def state_dict(self, *, device="cpu"):
        """Return owned p/m/v snapshots; this is not a durable file commit."""
        self._ready()
        if self.torch.device(device).type != "cpu":
            raise ValueError("Complete p/m/v snapshots must use CPU to avoid doubling resident device state")
        if any(float(self.optimizer.state[p]["step"].item()) != self.step for p in self._chunks):
            raise RuntimeError("Optimizer chunk steps differ from the committed local step")
        return {"format": FORMAT, "step": self.step, "schema": copy.deepcopy(self.schema),
                "optimizer_contract": copy.deepcopy(self.contract),
                **{key: value.detach().to(device=device, copy=True)
                   for key, value in (("parameter", self.master), ("exp_avg", self.exp_avg),
                                      ("exp_avg_sq", self.exp_avg_sq))}}

    def load_state_dict(self, state, *, expected_step=None):
        """Restore the complete FP32 state, rejecting partial/mixed versions.

        Pass ``expected_step`` from the authoritative checkpoint/commit record;
        tensor values alone cannot establish which claimed version they belong to.
        The entire input is validated before the first resident state mutation.
        """
        required = {"format", "step", "schema", "optimizer_contract", "parameter", "exp_avg", "exp_avg_sq"}
        if not isinstance(state, Mapping) or set(state) != required or state.get("format") != FORMAT:
            raise ValueError("A complete resident FP32 checkpoint is required")
        step = _integer(state["step"], "checkpoint step", minimum=0)
        if step > self.config["target_steps"]:
            raise ValueError("Checkpoint step exceeds target_steps")
        if expected_step is not None and step != _integer(expected_step, "expected_step", minimum=0):
            raise ValueError("Checkpoint step differs from authoritative expected_step")
        if state["schema"] != self.schema or state["optimizer_contract"] != self.contract:
            raise ValueError("Checkpoint schema or optimizer precision/configuration differs")
        arrays = {}
        for key in ("parameter", "exp_avg", "exp_avg_sq"):
            value = state[key]
            if isinstance(value, np.ndarray) and value.dtype == np.float32:
                value = self.torch.from_numpy(value)
            if (not self.torch.is_tensor(value) or value.dtype != self.torch.float32
                    or tuple(value.shape) != (self.elements,) or value.layout != self.torch.strided
                    or not value.is_contiguous()):
                raise ValueError(f"Checkpoint FP32 state shape/dtype differs: {key}")
            if self.device.type == "npu" and value.device.type != "cpu":
                raise ValueError("Complete NPU restore state must come from CPU")
            if not self._finite(value):
                raise ValueError(f"Checkpoint state is not finite: {key}")
            arrays[key] = value.detach()
        variance = arrays["exp_avg_sq"]
        negative = [(variance[start:start + self.chunk_elements] < 0).any()
                    for start in range(0, self.elements, self.chunk_elements)]
        if bool(self.torch.stack(negative).any().item()):
            raise ValueError("Checkpoint second moments must be nonnegative")
        self._poisoned = True
        with self.torch.no_grad():
            for key, destination in (("parameter", self.master), ("exp_avg", self.exp_avg),
                                     ("exp_avg_sq", self.exp_avg_sq)):
                destination.copy_(arrays[key])
        for parameter in self._chunks:
            self.optimizer.state[parameter]["step"].fill_(step)
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.param_groups[0]["lr"] = self._learning_rate(self.config, step)
        self.step, self._poisoned = step, False
        self.copy_to_model()
