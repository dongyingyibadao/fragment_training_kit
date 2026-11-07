"""Initialize the full CPU authority directly from the existing local base.

No gradients or GPU worker bootstrap payloads are needed. A private config
copy enables CPU construction and disables GPU atomic scheduling only during
initialization; the run's immutable config and parameter identity stay intact.
"""

from __future__ import annotations

import copy
import gc
from pathlib import Path
import time

import numpy as np

from jade_lotus.protocol import array_schema, arrays_digest


def _create_engine(config):
    import jax
    from jade_lotus.gradient_model import GradientEngine

    if jax.local_devices()[0].platform != "cpu":
        raise RuntimeError("Coordinator base initialization requires a CPU process; launch with JAX_PLATFORMS=cpu")
    return GradientEngine(config)


def load_full_initial_parameters(config):
    """Return the same FP32 initial arrays/digest a full GPU worker reports."""
    if config.get("training_mode") != "full" or config.get("coordinator_initializes_from_base") is not True:
        raise ValueError("CPU base bootstrap requires explicit full training and coordinator_initializes_from_base=true")
    from jade_lotus.engine_factory import backend_name

    if backend_name(config) == "torch_npu":
        from jade_lotus.npu_gradient_model import load_npu_initial_parameters
        from jade_lotus.protocol import file_sha256

        expected = config.get("model_weight_sha256")
        if expected is not None:
            path = Path(config["base_params"]).expanduser().resolve()
            if path.is_dir():
                path = path / "model.safetensors"
            if file_sha256(path) != expected:
                raise ValueError("Torch base checkpoint changed after run preparation")
        arrays, metadata = load_npu_initial_parameters(config)
        if (not arrays or any(value.dtype != np.float32 or value.size == 0 or not np.isfinite(value).all()
                              for value in arrays.values())):
            raise ValueError("CPU base bootstrap parameters must be finite, nonempty FP32 arrays")
        metadata = {**metadata, "schema": array_schema(arrays), "arrays_sha256": arrays_digest(arrays)}
        return arrays, metadata
    engine_config = copy.deepcopy(config)
    engine_config.update(allow_cpu=True, atomic_batching=False)
    # These only control a GPU worker's local accumulation. They must not
    # trigger atomic-worker guards in the CPU-only initialization constructor.
    engine_config.pop("worker_task_batch_size", None)
    engine_config["local_gradient_accumulation"] = "host"
    # GPU copies may be BF16; initialization of the CPU master is always FP32
    # and does not construct an atomic backward or upload a device version.
    engine_config["gpu_parameter_dtype"] = "float32"
    engine_config["device_gradient_dtype"] = "float32"
    for key in ("vision_parameter_dtype", "vision_compute_dtype", "vision_gradient_dtype"):
        engine_config.pop(key, None)
    import torch

    began, previous_threads = time.monotonic(), torch.get_num_threads()
    engine = None
    try:
        engine = _create_engine(engine_config)
        if engine.training_mode != "full" or engine.device.platform != "cpu":
            raise ValueError("CPU base bootstrap must construct the full model on CPU")
        arrays = engine.initial_params()
        if (not arrays or any(value.dtype != np.float32 or value.size == 0 or not np.isfinite(value).all()
                              for value in arrays.values())):
            raise ValueError("CPU base bootstrap parameters must be finite, nonempty FP32 arrays")
        metadata = {"source": "coordinator_local_base", "device": "cpu",
                    "schema": array_schema(arrays), "arrays_sha256": arrays_digest(arrays),
                    "parameter_count": sum(value.size for value in arrays.values()),
                    "parameter_leaves": len(arrays), "backward_executed": False,
                    "initialization_timings": copy.deepcopy(getattr(engine, "initialization_timings", {})),
                    "seconds": time.monotonic() - began}
        if config.get("base_params") is not None:
            metadata["base_params"] = str(Path(config["base_params"]).expanduser().resolve())
        return arrays, metadata
    finally:
        # GradientEngine initializes its data/torch helpers with one thread.
        # The authoritative CPU Adam has an independently benchmarked budget.
        torch.set_num_threads(previous_threads)
        del engine
        # Jitted callables close over their engine. Collect that cycle before
        # constructing CentralAdam so it cannot retain a second full base.
        gc.collect()
