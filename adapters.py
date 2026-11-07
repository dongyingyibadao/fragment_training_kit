"""Explicit model hooks for the existing resident LOW/1 training controller.

This interface changes model/data computation only. Membership fencing, gradient
SUM, APPLY authorization, NPU AdamW, full-state transfer and recovery remain in
scripts/probe_resident_openpi_fault.py. The current contract is global256 /
64 atomic units of 4 samples, eager native AdamW and no activation recomputation.
"""
from __future__ import annotations

import importlib
import math
import re
import time


FACTORY = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*\Z")


class AtomicLossEngine:
    """Convenience base: implement prepare_samples(positions, base_step) and losses(batch).

    losses(batch) returns one scalar per example, shape [microbatch]. Do not
    call backward, zero_grad, all_reduce, optimizer.step, or divide by world size
    in either callback. Stochastic preprocessing/noise must derive from the
    provided canonical positions/base_step, never rank or a process-global RNG.
    For token losses, reduce tokens within each example before returning [N].
    """
    def __init__(self, config, *, model, device):
        import torch
        self.torch, self.config = torch, dict(config)
        self.torch_device = torch.device(device)
        self.model = model.to(self.torch_device).train()
        self.parameters = {name: parameter for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        self.microbatch_size = int(config.get("worker_microbatch_size", 4))
        if self.microbatch_size not in (1, 2, 4):
            raise ValueError("Atomic-4 microbatch must be 1, 2 or 4")
        self.checkpoint_call_count = 0

    def _sync(self):
        if self.torch_device.type == "npu":
            self.torch.npu.synchronize(self.torch_device)

    def prepare_samples(self, positions, *, base_step):
        raise NotImplementedError

    def losses(self, batch):
        raise NotImplementedError

    def _assert_no_checkpointing(self):
        for module in self.model.modules():
            if getattr(module, "gradient_checkpointing", False) or getattr(module, "is_gradient_checkpointing", False):
                raise ValueError("Disable model activation/gradient checkpointing before resident training")

    def backward_units(self, *, base_step, units, clear=True):
        self._assert_no_checkpointing()
        if clear:
            self.model.zero_grad(set_to_none=True)
        began = time.monotonic()
        total, calls, data_seconds, backward_seconds = 0., 0, 0., 0.
        for unit in units:
            for offset in range(0, 4, self.microbatch_size):
                position = base_step * 256 + unit * 4 + offset
                phase = time.monotonic()
                batch = self.prepare_samples(range(position, position + self.microbatch_size), base_step=base_step)
                self._sync()
                data_seconds += time.monotonic() - phase
                phase = time.monotonic()
                losses = self.losses(batch)
                if not self.torch.is_tensor(losses) or list(losses.shape) != [self.microbatch_size]:
                    raise ValueError("losses(batch) must return one scalar loss per example, shape [N]")
                loss = losses.sum() / 256
                if not bool(self.torch.isfinite(loss).item()):
                    raise FloatingPointError("Nonfinite model loss")
                loss.backward()
                self._sync()
                total += float(loss.detach().item())
                calls += 1
                backward_seconds += time.monotonic() - phase
        return {"units": list(units), "samples": 4 * len(units), "loss_contribution": total,
                "backward_calls": calls, "seconds": time.monotonic() - began,
                "data_seconds": data_seconds, "backward_seconds": backward_seconds,
                "timing_scope": "synchronized adapter data and forward/backward; excludes reduction/apply"}


def create_engine(config, *, device_index=0):
    """Import ``module:factory`` and validate the engine before READY/HCCL.

    The factory receives (config, device_index=...). It must build identical
    initial weights on all ranks and return an engine with model, parameters
    (ordered named trainable Parameters), torch, torch_device, config,
    checkpoint_call_count=0, _sync(), and backward_units(...).
    """
    name = config.get("resident_adapter")
    if not isinstance(name, str) or not FACTORY.fullmatch(name):
        raise ValueError("resident_adapter must be an importable module:factory")
    if config.get("batch_size") != 256 or config.get("disable_activation_checkpointing") is not True:
        raise ValueError("Adapter requires global256 and disabled activation recomputation")
    if config.get("resident_data_workers", 0):
        raise ValueError("Adapter owns its CPU data loading; resident_data_workers must be 0")
    module, factory = name.split(":")
    engine = getattr(importlib.import_module(module), factory)(config, device_index=device_index)
    for field in ("model", "parameters", "torch", "torch_device", "config", "checkpoint_call_count"):
        if not hasattr(engine, field):
            raise TypeError(f"Adapter engine is missing {field}")
    for method in ("_sync", "backward_units"):
        if not callable(getattr(engine, method, None)):
            raise TypeError(f"Adapter engine is missing callable {method}")
    if engine.checkpoint_call_count != 0 or engine.torch_device.type != "npu":
        raise ValueError("Resident training requires native NPU state and no recomputation")
    if engine.config.get("resident_adapter") != name:
        raise ValueError("Adapter discarded its training configuration")
    parameters = engine.parameters
    expected = {key: value for key, value in engine.model.named_parameters() if value.requires_grad}
    if not parameters or list(parameters) != list(expected) or any(parameters[key] is not value for key, value in expected.items()):
        raise ValueError("parameters must contain every unique trainable model Parameter in named order")
    for tensor in [*engine.model.parameters(), *engine.model.buffers()]:
        if tensor.device != engine.torch_device:
            raise ValueError("Every model parameter/buffer must reside on the assigned NPU")
    engine._sync()
    return engine


def accumulate_units(engine, *, base_step, units, clear=True):
    """Validate callback bookkeeping. The adapter implements the actual math.

    A unit is [base_step*256 + unit*4, ..., +3]. Each sample loss is weighted
    1/256, followed by HCCL SUM (never another world-size division). clear=False
    adds only lost units to healthy gradients; it must not zero those gradients.
    """
    units = list(units)
    if type(base_step) is not int or base_step < 0 or type(clear) is not bool:
        raise ValueError("Invalid committed step/clear flag")
    if (not units and clear) or len(units) != len(set(units)) or any(type(unit) is not int or not 0 <= unit < 64 for unit in units):
        raise ValueError("Expected distinct atomic-4 units in 0..63")
    result = engine.backward_units(base_step=base_step, units=units, clear=clear)
    if not isinstance(result, dict) or result.get("units") != units or result.get("samples") != 4 * len(units):
        raise ValueError("Adapter returned a different sample/unit assignment")
    for key in ("loss_contribution", "seconds", "data_seconds", "backward_seconds"):
        value = result.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or (key != "loss_contribution" and value < 0):
            raise ValueError(f"Invalid adapter measurement: {key}")
    if type(result.get("backward_calls")) is not int or result["backward_calls"] < (1 if units else 0):
        raise ValueError("Adapter must execute backward")
    if engine.checkpoint_call_count != 0:
        raise AssertionError("Activation recomputation occurred")
    return result
