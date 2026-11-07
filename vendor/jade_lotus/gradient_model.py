"""Pure gradient evaluation for the local, offline pi05 LIBERO experiment.

GPU processes never own an optimizer. The coordinator sends one authoritative
FP32 parameter version and a deterministic slice of the sample stream. The
default LoRA mode freezes the local backbone; optional full mode trains every
native pi05 parameter, including vision, without introducing adapter leaves.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import urllib.parse


DEFAULT_OPENPI = "/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/openpi"
DEFAULT_DATASET = "/inspire/ssd/project/robot-decision/public/datasets/HuggingFaceVLA/libero"
_REMAT_TRACE_LOCK = threading.RLock()


def _training_mode(config):
    mode = config.get("training_mode", "lora")
    if mode not in ("lora", "full"):
        raise ValueError("training_mode must be 'lora' or 'full'")
    return mode


def _gpu_precision_configuration(config):
    """Separate GPU storage/transfer precision from the FP32 CPU authority.

    This is an opt-in experiment. Initial/master parameters and returned host
    gradients keep their existing FP32 contract; BF16 device gradients are
    expanded only after D2H. No optimizer or on-disk codec changes here.
    """
    parameter_dtype = config.get("gpu_parameter_dtype", "float32")
    gradient_dtype = config.get("device_gradient_dtype", "float32")
    for name, value in (("gpu_parameter_dtype", parameter_dtype),
                        ("device_gradient_dtype", gradient_dtype)):
        if value not in ("float32", "bfloat16"):
            raise ValueError(f"{name} must be 'float32' or 'bfloat16'")
    if (parameter_dtype, gradient_dtype) != ("float32", "float32"):
        if _training_mode(config) != "full" or config.get("atomic_batching") is not True:
            raise ValueError("Reduced GPU precision requires explicit full atomic training")
    return parameter_dtype, gradient_dtype


def _vision_precision_configuration(config):
    result = {key: config.get(key) for key in
              ("vision_parameter_dtype", "vision_compute_dtype", "vision_gradient_dtype")}
    if any(value is not None for value in result.values()):
        if _training_mode(config) != "full" or config.get("atomic_batching") is not True:
            raise ValueError("Explicit vision precision requires full atomic training")
        if any(value not in (None, "float32", "bfloat16") for value in result.values()):
            raise ValueError("Vision precision must be 'float32' or 'bfloat16'")
    return result


def _is_vision_parameter(path):
    return path.startswith("PaliGemma/img/")


def _atomic_configuration(config, device):
    """Validate opt-in physical capacities without assuming a GPU batch fits."""
    enabled = config.get("atomic_batching", False)
    if type(enabled) is not bool:
        raise ValueError("atomic_batching must be a bool")
    if not enabled:
        return False, None
    if _training_mode(config) != "full":
        raise ValueError("atomic_batching requires training_mode='full'")
    if config.get("fused_task_gradient", False):
        raise ValueError("atomic_batching cannot use fused_task_gradient; it differentiates the mean unit loss once")
    capacities = config.get("worker_batch_capacities")
    if capacities is not None:
        if (not isinstance(capacities, dict) or set(capacities) != {"H100", "H200"}
                or type(capacities["H100"]) is not int or capacities["H100"] not in (4, 8)
                or type(capacities["H200"]) is not int or capacities["H200"] not in (4, 8, 16)):
            raise ValueError("worker_batch_capacities must explicitly contain H100=4/8 and H200=4/8/16")
    if device.platform == "cpu":
        physical = config.get("atomic_physical_batch_size")
        if not config.get("allow_cpu", False) or type(physical) is not int or physical not in (4, 8, 16):
            raise ValueError("CPU atomic tests require allow_cpu and explicit atomic_physical_batch_size=4/8/16")
        return True, physical
    if device.platform != "gpu" or capacities is None:
        raise ValueError("Atomic GPU workers require explicit worker_batch_capacities")
    if "atomic_physical_batch_size" in config:
        raise ValueError("atomic_physical_batch_size is a CPU test override; GPU capacity comes from device_kind")
    kind = str(device.device_kind).upper()
    matches = [name for name in ("H100", "H200") if name in kind]
    if len(matches) != 1:
        raise ValueError("Atomic GPU device_kind must identify exactly one H100 or H200")
    return True, capacities[matches[0]]


class _LinenWithoutRemat:
    """Delegate Linen unchanged except that remat is the identity transform."""

    def __init__(self, linen):
        self._linen = linen

    def __getattr__(self, name):
        return getattr(self._linen, name)

    @staticmethod
    def remat(target=None, **_kwargs):
        return target if target is not None else lambda function: function


@contextlib.contextmanager
def _model_rematerialization(*, disabled=False):
    """Override just OpenPI's Linen references while JAX traces this gradient.

    Gemma's checkpoint policy is hard-coded in setup(), and SigLIP builds its
    remat wrapper inside a compact method. A temporary module-local proxy
    removes both transforms without changing Flax itself, parameter names,
    layer scanning, model math or RNGs. The original references are restored
    even when tracing raises. Engine traces in either mode share this lock;
    compiled execution never depends on the temporary references.
    """
    with _REMAT_TRACE_LOCK:
        if not disabled:
            yield
            return
        from openpi.models import gemma, siglip

        originals = [(module, module.nn) for module in (gemma, siglip)]
        try:
            for module, linen in originals:
                module.nn = _LinenWithoutRemat(linen)
            yield
        finally:
            for module, linen in originals:
                module.nn = linen


def _imports(config: dict):
    root = Path(config.get("openpi_root", DEFAULT_OPENPI)).resolve()
    sys.path[:0] = [str(root / "src"), str(root / "packages/openpi-client/src")]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("OPENPI_DATA_HOME", str(root / "outputs/robotwin_clean50_abs_eef/cache"))
    import jax
    from openpi.shared import download

    # JAX 0.5.3's persistent cache was independently found to produce incorrect
    # cross-process outputs in this environment; retain per-process JIT only.
    jax.config.update("jax_enable_compilation_cache", False)

    def local_only_download(url, **_kwargs):
        parsed = urllib.parse.urlparse(str(url))
        if not parsed.scheme:
            target = Path(url).expanduser()
        else:
            target = Path(os.environ["OPENPI_DATA_HOME"]) / parsed.netloc / parsed.path.lstrip("/")
        if not target.exists():
            raise FileNotFoundError(f"Offline asset missing: {target}")
        return target.resolve()

    download.maybe_download = local_only_download
    return root


def _configure_rank(rank: int):
    """Change only local process LoRA configuration, never the user's checkout."""
    from openpi.models import gemma

    if rank < 1:
        raise ValueError("lora_rank must be positive")
    original = getattr(gemma.get_config, "_jade_original", gemma.get_config)

    def get_config(variant):
        result = original(variant)
        if "lora" in variant:
            result = dataclasses.replace(
                result,
                lora_configs={key: dataclasses.replace(value, rank=rank, alpha=float(rank))
                              for key, value in result.lora_configs.items()},
            )
        return result

    get_config._jade_original = original
    gemma.get_config = get_config


def make_configuration(config: dict):
    """Return the upstream model/data configs, using local metadata statistics."""
    training_mode = _training_mode(config)
    _imports(config)
    import numpy as np
    from flax import nnx
    from openpi import transforms
    from openpi.shared import nnx_utils, normalize
    from openpi.training import config as configs

    upstream = configs.get_config("pi05_libero")
    if training_mode == "full":
        model_config = upstream.model
        trainable_filter = nnx.Param
    else:
        _configure_rank(int(config.get("lora_rank", 8)))
        model_config = dataclasses.replace(
            upstream.model,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
        # Freeze every vision/base leaf. Heads outside PaliGemma and explicit
        # LoRA leaves remain trainable in the original default mode.
        trainable_filter = nnx.All(nnx.Param, nnx.Any(
            nnx_utils.PathRegex(".*lora.*"),
            nnx.Not(nnx_utils.PathRegex("PaliGemma/.*")),
        ))
    factory = dataclasses.replace(
        upstream.data,
        repo_id="local/libero",
        dataset_root=str(Path(config.get("dataset_root", DEFAULT_DATASET)).resolve()),
    )
    # create() supplies exactly the upstream LIBERO observation/policy/tokenizer
    # transforms; it may not find optional assets, which we set from meta/stats.
    data_config = factory.create(Path(config.get("assets_dir", "/nonexistent/jade-lotus-assets")), model_config)
    dataset_root = Path(factory.dataset_root)
    stats = json.loads((dataset_root / "meta/stats.json").read_text())
    norm_stats = {}
    for target, source in (("state", "observation.state"), ("actions", "action")):
        entry = stats[source]
        norm_stats[target] = normalize.NormStats(**{
            key: np.asarray(entry[key], dtype=np.float32) for key in ("mean", "std", "q01", "q99")
        })
    data_config = dataclasses.replace(
        data_config,
        action_sequence_keys=("action",),
        norm_stats=norm_stats,
        repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
            "observation/image": "observation.images.image",
            "observation/wrist_image": "observation.images.image2",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        })]),
    )
    return model_config, data_config, trainable_filter


class SampleStream:
    """Map global offsets to seeded epoch permutations, independent of workers."""

    def __init__(self, dataset_size: int, seed: int):
        self.dataset_size = dataset_size
        self.seed = seed
        self._permutations = {}

    def resolve(self, position: int):
        import numpy as np

        if position < 0:
            raise ValueError("Sample position must be nonnegative")
        epoch, offset = divmod(position, self.dataset_size)
        if epoch not in self._permutations:
            self._permutations[epoch] = np.random.default_rng(self.seed + epoch).permutation(self.dataset_size)
            for previous in list(self._permutations):
                if previous < epoch - 1:
                    self._permutations.pop(previous)
        return epoch, int(self._permutations[epoch][offset])


def _seed_for_sample(seed, epoch, index):
    digest = hashlib.blake2b(f"{seed}:{epoch}:{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32)


def _flat(tree):
    from flax import traverse_util
    return traverse_util.flatten_dict(tree, sep="/")


def _unflat(tree):
    from flax import traverse_util
    return traverse_util.unflatten_dict(tree, sep="/")


def _validate_native_checkpoint_schema(base_path, expected):
    """Inspect native metadata without loading full checkpoint values."""
    import numpy as np
    import orbax.checkpoint as ocp

    base_path = Path(base_path).expanduser().resolve()
    if not base_path.is_dir():
        raise FileNotFoundError(f"Offline native checkpoint missing: {base_path}")
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = _flat(checkpointer.metadata(base_path)["params"])
    if metadata and all(key.endswith("/value") for key in metadata):
        metadata = {key[:-len("/value")]: value for key, value in metadata.items()}
    missing, extra = set(expected) - set(metadata), set(metadata) - set(expected)
    mismatched = {key: {"expected": list(expected[key].shape), "checkpoint": list(metadata[key].shape)}
                  for key in set(expected) & set(metadata) if tuple(expected[key].shape) != tuple(metadata[key].shape)}
    if missing or extra or mismatched:
        raise ValueError(f"Native checkpoint schema differs from full pi05: "
                         f"missing={sorted(missing)}, extra={sorted(extra)}, shapes={mismatched}")
    return {"path": str(base_path), "schema_validated": True, "parameter_leaves": len(metadata),
            "parameter_count": sum(int(np.prod(value.shape)) for value in metadata.values()),
            "stored_parameter_bytes": sum(int(np.prod(value.shape)) * np.dtype(value.dtype).itemsize
                                           for value in metadata.values()),
            "stored_dtypes": sorted({str(value.dtype) for value in metadata.values()}),
            "metadata_only": True, "checkpoint_values_loaded": False}


def inspect_configuration(config: dict, *, sample_count=2):
    """CPU preflight checks real transformed samples and exact parameter shapes."""
    training_mode = _training_mode(config)
    _imports(config)
    import jax
    import numpy as np
    from flax import nnx
    from openpi.training import data_loader

    model_config, data_config, trainable = make_configuration(config)
    shape_model = nnx.eval_shape(lambda: model_config.create(jax.random.key(int(config.get("seed", 42)))))
    state = nnx.state(shape_model)
    trainable_shapes = _flat(state.filter(trainable).to_pure_dict())
    parameter_shapes = _flat(state.filter(nnx.Param).to_pure_dict())
    total_parameters = sum(int(np.prod(value.shape)) for value in parameter_shapes.values())
    checkpoint = None
    if training_mode == "full":
        base_path = config.get("base_params", Path(config.get("openpi_root", DEFAULT_OPENPI)) /
                               "outputs/robotwin_clean50_abs_eef/pretrained/pi05_base/params")
        checkpoint = _validate_native_checkpoint_schema(base_path, _flat(state.to_pure_dict()))
    dataset = data_loader.transform_dataset(
        data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config), data_config,
    )
    samples = []
    stream = SampleStream(len(dataset), int(config.get("seed", 42)))
    for position in range(sample_count):
        epoch, index = stream.resolve(position)
        sample = dataset[index]
        samples.append({"position": position, "epoch": epoch, "index": index,
                        "leaves": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                          "finite": bool(np.isfinite(value).all())}
                                   for key, value in _flat(sample).items()}})
    n_trainable = sum(int(np.prod(value.shape)) for value in trainable_shapes.values())
    report = {
        "training_mode": training_mode,
        "model": dataclasses.asdict(model_config),
        "dataset_size": len(dataset),
        "sampler": "numpy-default_rng(seed+epoch)-permutation-v1",
        "lora_rank": config.get("lora_rank", 8) if training_mode == "lora" else None,
        "trainable_parameters": n_trainable,
        "gradient_fp32_bytes": n_trainable * 4,
        "total_parameters": total_parameters,
        "trainable_shapes": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                             for key, value in trainable_shapes.items()},
        "samples": samples,
        "normalization": "local meta/stats.json q01/q99, upstream quantile transform",
    }
    if training_mode == "full":
        fp32_bytes = n_trainable * 4
        report.update(
            base_checkpoint=checkpoint,
            all_parameters_trainable=(set(parameter_shapes) == set(trainable_shapes)),
            component_parameters={name: sum(int(np.prod(value.shape)) for key, value in parameter_shapes.items()
                                             if predicate(key))
                                  for name, predicate in (
                                      ("vision", lambda key: key.startswith("PaliGemma/img/")),
                                      ("language_and_action_experts", lambda key: key.startswith("PaliGemma/llm/")),
                                      ("heads", lambda key: not key.startswith("PaliGemma/")))},
            memory_estimate={
                "fp32_parameter_bytes": fp32_bytes,
                "fp32_gradient_bytes_per_task": fp32_bytes,
                "cpu_adam_parameter_and_two_moments_bytes": fp32_bytes * 3,
                "cpu_adam_plus_one_gradient_and_accumulator_bytes": fp32_bytes * 5,
                "gpu_parameter_and_gradient_bytes_before_activations": fp32_bytes * 2,
                "gpu_parameter_bytes_during_version_replacement": fp32_bytes * 2,
                "not_a_peak_memory_measurement": True,
                "excludes": ["activations and attention/backward workspaces", "allocator/compiler overhead",
                             "host bootstrap/read buffers and prefetched gradients", "optimizer temporaries"],
            },
        )
    return report


class GradientEngine:
    """Evaluate LoRA or full-model mean gradients on one assigned device."""

    def __init__(self, config: dict, device_index: int | None = None):
        initialization_started = phase_started = time.perf_counter()
        self.initialization_timings = {}

        def initialization_phase(name):
            nonlocal phase_started
            now = time.perf_counter()
            self.initialization_timings[name] = now - phase_started
            phase_started = now

        self.training_mode = _training_mode(config)
        self.config = dict(config)
        from jade_lotus.host_precision import configured_host_widen_workers
        from jade_lotus.gpu_host_transfer import configured_gradient_host_transfer

        configured_host_widen_workers(config)  # Reject invalid CPU worker counts before model/device setup.
        configured_gradient_host_transfer(config)
        self.gpu_parameter_dtype, self.device_gradient_dtype = _gpu_precision_configuration(config)
        self.vision_precision = _vision_precision_configuration(config)
        self.root = _imports(config)
        import jax
        import jax.numpy as jnp
        import numpy as np
        import torch
        from flax import nnx
        from openpi.models import model as model_module
        from openpi.training import data_loader

        initialization_phase("imports_seconds")
        self.jax, self.jnp, self.np, self.nnx = jax, jnp, np, nnx
        devices = jax.local_devices()
        self.device = devices[0 if device_index is None else device_index]
        if self.device.platform != "gpu" and not config.get("allow_cpu", False):
            raise RuntimeError("Gradient workers require assigned GPUs; CPU preflight uses --preflight")
        self.atomic_batching, self.physical_batch_capacity = _atomic_configuration(config, self.device)
        self.task_batch_capacity = config.get("worker_task_batch_size", self.physical_batch_capacity)
        if "worker_task_batch_size" in config:
            if (not self.atomic_batching or type(self.task_batch_capacity) is not int
                    or self.task_batch_capacity not in (4, 8, 16)
                    or self.task_batch_capacity < self.physical_batch_capacity):
                raise ValueError("worker_task_batch_size requires atomic batching and must be 4/8/16, at least physical capacity")
        self.local_gradient_accumulation = config.get("local_gradient_accumulation", "host")
        if self.local_gradient_accumulation not in ("host", "gpu"):
            raise ValueError("local_gradient_accumulation must be 'host' or 'gpu'")
        if self.local_gradient_accumulation == "gpu" and not self.atomic_batching:
            raise ValueError("GPU local accumulation requires atomic batching")
        self._local_scan_gradient = self._local_scan_source = None
        torch.set_num_threads(1)
        self.model_config, self.data_config, self.trainable_filter = make_configuration(config)
        self.microbatch_size = int(config.get("microbatch_size", 4))
        self.fused_task_gradient = bool(config.get("fused_task_gradient", False))
        self.disable_activation_checkpointing = bool(config.get("disable_activation_checkpointing", False))
        from jade_lotus.input_compaction import Pi05ImageCompaction

        self.input_compaction = Pi05ImageCompaction(config.get("skip_masked_image_keys", ()))
        self._fused_gradient = None
        self.seed = int(config.get("seed", 42))
        if self.microbatch_size < 1:
            raise ValueError("microbatch_size must be positive")
        initialization_phase("device_and_configuration_seconds")
        with jax.default_device(self.device):
            shape_model = nnx.eval_shape(lambda: self.model_config.create(jax.random.key(self.seed)))
            vision_compute_dtype = self.vision_precision["vision_compute_dtype"]
            if vision_compute_dtype is not None:
                # SigLIP's dtype_mm is static Linen configuration carried by
                # the NNX bridge. Clone just this model's module; no upstream
                # mutation, parameter renaming, or global factory override.
                vision = shape_model.PaliGemma.img
                vision.module = vision.module.clone(dtype_mm=vision_compute_dtype)
            self.graphdef, self.template_state = nnx.split(shape_model)
            train_shapes = _flat(self.template_state.filter(self.trainable_filter).to_pure_dict())
            shape_flat = _flat(self.template_state.to_pure_dict())
            initialization_phase("model_shape_seconds")
            base_path = Path(config.get("base_params", self.root / "outputs/robotwin_clean50_abs_eef/pretrained/pi05_base/params"))
            print(json.dumps({"event": "loading_base_checkpoint" if self.training_mode == "full" else "loading_frozen_backbone",
                              "path": str(base_path), "device": str(self.device)}), flush=True)
            base = _flat(model_module.restore_params(base_path, restore_type=np.ndarray))
            initialization_phase("checkpoint_restore_seconds")
            pure = {}
            for path, spec in shape_flat.items():
                if path in base:
                    value = base.pop(path)
                    if tuple(value.shape) != tuple(spec.shape):
                        raise ValueError(f"Pretrained shape mismatch {path}: {value.shape} != {spec.shape}")
                    # The authoritative full-model initialization stays FP32.
                    # Optional BF16 GPU copies are made at version upload.
                    # LoRA retains its original frozen BF16 storage policy.
                    dtype = np.float32 if path in train_shapes else jnp.bfloat16
                    value = np.asarray(value).astype(dtype)
                elif self.training_mode == "lora" and "lora" in path:
                    # Initialize only missing adapter arrays. Avoid materializing
                    # random 3.4B base weights merely to overwrite them afterward.
                    path_seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{path}".encode()).digest()[:8], "little")
                    value = np.random.default_rng(path_seed).normal(0, 0.01, size=spec.shape).astype(np.float32)
                else:
                    raise KeyError(f"Native pi05 checkpoint is missing required base weight {path}")
                # Full workers stage their authoritative initialization on the
                # host. _set_parameter_version uploads weights on demand;
                # there is no need for a GPU->host initialization round trip.
                pure[path] = value if self.training_mode == "full" else jax.device_put(value, self.device)
            if base:
                raise ValueError(f"Unexpected pretrained weights: {sorted(base)[:5]}")
            jax.block_until_ready(pure)
            initialization_phase("parameter_cast_and_device_transfer_seconds")
            self.template_state.replace_by_pure_dict(_unflat(pure))
            train_state, self.frozen_state = self.template_state.split(self.trainable_filter, ...)
            self.trainable_paths = tuple(sorted(_flat(train_state.to_pure_dict())))
            self.train_template = train_state
            self._initial_params = {key: np.asarray(value, dtype=np.float32)
                                    for key, value in _flat(train_state.to_pure_dict()).items()}
            if self.training_mode == "full":
                # Templates provide structure only. Keeping their initial
                # arrays would retain a second complete model after the first
                # authoritative parameter version is uploaded.
                to_shape = lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype)
                self.train_template = jax.tree.map(to_shape, train_state)
                self.template_state = jax.tree.map(to_shape, self.template_state)
                del train_state, pure

            def loss_and_grad(train_state, frozen_state, rng, observation, actions):
                model = nnx.merge(self.graphdef, train_state, frozen_state)
                model.train()

                def loss_fn(model):
                    return jnp.mean(self.input_compaction.compute_loss(model, rng, observation, actions, train=True))

                with _model_rematerialization(disabled=self.disable_activation_checkpointing):
                    return nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, self.trainable_filter))(model)

            self._gradient = jax.jit(loss_and_grad)
            self._atomic_gradient = self._make_atomic_gradient() if self.atomic_batching else None
        initialization_phase("trainable_state_and_gradient_setup_seconds")
        self.dataset = data_loader.transform_dataset(
            data_loader.create_torch_dataset(self.data_config, self.model_config.action_horizon, self.model_config),
            self.data_config,
        )
        self._fast_data_handle = None
        if config.get("fast_data_queries", False):
            from jade_lotus.fast_data import install_numeric_action_queries
            self._fast_data_handle = install_numeric_action_queries(self.dataset)
        initialization_phase("dataset_seconds")
        self.stream = SampleStream(len(self.dataset), self.seed)
        self._cached_version = None
        self._cached_train_state = None
        self.initialization_timings["total_seconds"] = time.perf_counter() - initialization_started
        n = sum(value.size for value in self._initial_params.values())
        vision_n = sum(value.size for path, value in self._initial_params.items() if _is_vision_parameter(path))

        def precision_bytes(default, vision):
            return ((n - vision_n) * (2 if default == "bfloat16" else 4)
                    + vision_n * (2 if (vision or default) == "bfloat16" else 4))

        print(json.dumps({"event": "engine_ready", "device": str(self.device), "dataset_size": len(self.dataset),
                          "trainable_parameters": n, "gradient_fp32_bytes": n * 4,
                          "vision_trainable_parameters": vision_n,
                          "gpu_parameter_dtype": self.gpu_parameter_dtype,
                          "device_gradient_dtype": self.device_gradient_dtype,
                          "gpu_parameter_storage_bytes": precision_bytes(
                              self.gpu_parameter_dtype, self.vision_precision["vision_parameter_dtype"]),
                          "single_backward_gradient_payload_bytes": precision_bytes(
                              self.device_gradient_dtype, self.vision_precision["vision_gradient_dtype"]),
                          "device_gradient_payload_bytes": n * 4 if self.local_gradient_accumulation == "gpu" else precision_bytes(
                              self.device_gradient_dtype, self.vision_precision["vision_gradient_dtype"]),
                          "cpu_master_parameter_dtype": "float32",
                          "host_gradient_dtype": "float32",
                          "vision_precision": self.vision_precision,
                          "training_mode": self.training_mode,
                          "microbatch_size": self.microbatch_size,
                          "lora_rank": config.get("lora_rank", 8) if self.training_mode == "lora" else None,
                          "fused_task_gradient": self.fused_task_gradient,
                          "atomic_batching": self.atomic_batching,
                          "physical_batch_capacity": self.physical_batch_capacity,
                          "task_batch_capacity": self.task_batch_capacity,
                          "disable_activation_checkpointing": self.disable_activation_checkpointing,
                          "skip_masked_image_keys": list(self.input_compaction.skip_image_keys),
                          "initialization_timings": self.initialization_timings}), flush=True)

    def initial_params(self):
        return {key: value.copy() for key, value in self._initial_params.items()}

    def invalidate_parameter_cache(self):
        """Fence device arrays when the coordinator publishes a new generation."""
        self._cached_version = None
        self._cached_train_state = None

    def _sample(self, position):
        import torch

        epoch, index = self.stream.resolve(position)
        sample_seed = _seed_for_sample(self.seed, epoch, index)
        python_state = random.getstate()
        numpy_state = self.np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(sample_seed)
            self.np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            return self.dataset[index]
        finally:
            random.setstate(python_state)
            self.np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)

    def _set_parameter_version(self, params: dict, version: int):
        jax, jnp, np, nnx = self.jax, self.jnp, self.np, self.nnx
        if set(params) != set(self.trainable_paths):
            raise ValueError("Coordinator parameter schema differs from gradient engine")
        if self._cached_version != version:
            state = jax.tree.map(lambda leaf: leaf, self.train_template)
            for key, value in params.items():
                expected = self._initial_params[key]
                if value.shape != expected.shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid authoritative parameter {key}")
                if self.training_mode == "full" and value.dtype != np.float32:
                    raise ValueError(f"Full-model authoritative parameter must be FP32: {key}")
            # Cast on the host before upload: an FP32 staging copy on the GPU
            # would defeat the intended peak-memory reduction at each version.
            def device_parameter(key, value):
                dtype_name = getattr(self, "gpu_parameter_dtype", "float32")
                if _is_vision_parameter(key):
                    dtype_name = getattr(self, "vision_precision", {}).get("vision_parameter_dtype") or dtype_name
                dtype = jnp.bfloat16 if dtype_name == "bfloat16" else np.float32
                return jax.device_put(np.asarray(value, dtype=dtype), self.device)

            state.replace_by_pure_dict(_unflat({
                key: device_parameter(key, value)
                for key, value in params.items()}))
            self._cached_train_state = state
            self._cached_version = version
        return self._cached_train_state

    def _prepare_microbatch(self, positions):
        from openpi.models import model as model_module

        rows = [self._sample(position) for position in positions]
        batch = self.jax.tree.map(lambda *values: self.np.stack(values, axis=0), *rows)
        observation = model_module.Observation.from_dict(batch)
        self.input_compaction.validate_observation(observation)
        return self.jax.device_put((observation, batch["actions"]), self.device)

    def _atomic_units(self, sample_positions, version):
        from jade_lotus.atomic_batch import describe_units

        units = describe_units(sample_positions, run_seed=self.seed, version=version,
                               batch_size=self.config.get("batch_size", 256))
        if len(units) * 4 > getattr(self, "task_batch_capacity", self.physical_batch_capacity):
            raise ValueError("Atomic assignment exceeds this worker's explicit task batch capacity")
        return units

    def _make_atomic_gradient(self):
        """Build a fresh trace for the current compaction/rematerialization policy.

        Numerical reference checks must rebuild this callable when temporarily
        changing a Python policy; replacing only ``_gradient`` does not affect
        atomic execution. The scalar mean is differentiated once, producing
        one full gradient tree for all units in a physical batch.
        """
        from jade_lotus.atomic_batch import mean_unit_loss

        if not self.atomic_batching:
            raise ValueError("_make_atomic_gradient requires atomic_batching")
        jax, jnp, nnx = self.jax, self.jnp, self.nnx

        def atomic_loss_and_grad(train_state, frozen_state, rngs, observation, actions):
            model = nnx.merge(self.graphdef, train_state, frozen_state)
            model.train()

            def loss_fn(model):
                return mean_unit_loss(
                    lambda rng, obs, act: self.input_compaction.compute_loss(model, rng, obs, act, train=True),
                    rngs, observation, actions, return_unit_losses=True,
                )

            with _model_rematerialization(disabled=self.disable_activation_checkpointing):
                (loss, unit_losses), grads = nnx.value_and_grad(
                    loss_fn, argnums=nnx.DiffState(0, self.trainable_filter), has_aux=True,
                )(model)
            def device_gradient(path, value):
                dtype_name = getattr(self, "device_gradient_dtype", "float32")
                if _is_vision_parameter(path):
                    dtype_name = getattr(self, "vision_precision", {}).get("vision_gradient_dtype") or dtype_name
                return value.astype(jnp.bfloat16 if dtype_name == "bfloat16" else jnp.float32)

            pure = _unflat({path: device_gradient(path, value)
                            for path, value in _flat(grads.to_pure_dict()).items()})
            return loss.astype(jnp.float32), pure, jnp.all(jnp.isfinite(unit_losses))

        return jax.jit(atomic_loss_and_grad)

    def prepare_atomic_device(self, prepared, *, version, sample_positions=None):
        """Validate/pack resident inputs outside the timed single-backward call.

        Existing AtomicPreparedTask metadata is checked against the requested
        version and optional positions. A raw prepared list requires explicit
        sample_positions because its arrays cannot identify a stochastic unit.
        """
        from jade_lotus.atomic_batch import AtomicPreparedTask, pack_prepared_units, unit_rngs, unpack_prepared_units

        if not self.atomic_batching:
            raise ValueError("prepare_atomic_device requires atomic_batching")
        if isinstance(prepared, AtomicPreparedTask):
            if prepared.version != version:
                raise ValueError("Prepared atomic parameter version does not match the requested version")
            positions = prepared.sample_positions
            if sample_positions is not None and tuple(sample_positions) != positions:
                raise ValueError("Prepared atomic sample positions do not match the assignment")
            units = self._atomic_units(positions, version)
            raw_units = unpack_prepared_units(prepared.observation, prepared.actions)
            if len(raw_units) * 4 != len(positions):
                raise ValueError("Prepared atomic array/key counts differ from the sample assignment")
            expected_keys = unit_rngs(units)
            if (getattr(prepared.rngs, "shape", None) != expected_keys.shape
                    or getattr(prepared.rngs, "dtype", None) != expected_keys.dtype):
                raise ValueError("Prepared atomic keys must use the canonical typed Threefry key shape and dtype")
            if not self.np.array_equal(self.jax.device_get(self.jax.random.key_data(prepared.rngs)),
                                       self.jax.device_get(self.jax.random.key_data(expected_keys))):
                raise ValueError("Prepared atomic RNG keys differ from the sample assignment")
            for observation, _ in raw_units:
                self.input_compaction.validate_observation(observation)
            return prepared
        if sample_positions is None:
            raise ValueError("Atomic compute_prepared requires sample_positions for raw prepared inputs")
        positions = tuple(sample_positions)
        units = self._atomic_units(positions, version)
        with self.jax.default_device(self.device):
            observation, actions = pack_prepared_units(prepared)
            keys = unit_rngs(units)
        if actions.shape[0] != len(units):
            raise ValueError("Prepared atomic input count differs from the sample assignment")
        for unit_observation, _ in unpack_prepared_units(observation, actions):
            self.input_compaction.validate_observation(unit_observation)
        return AtomicPreparedTask(observation, actions, keys, positions, version)

    def atomic_device_gradient(self, prepared):
        """Return resident (loss, one mean gradient tree, finite-unit-loss flag)."""
        if not self.atomic_batching or self._atomic_gradient is None:
            raise ValueError("atomic_device_gradient requires atomic_batching")
        if self._cached_train_state is None or self._cached_version != prepared.version:
            raise ValueError("Initialize the matching parameter version before atomic_device_gradient")
        if len(prepared.sample_positions) > self.physical_batch_capacity:
            if getattr(self, "local_gradient_accumulation", "host") != "gpu":
                raise ValueError("Use atomic_physical_parts to honor the physical batch capacity")
            return self._atomic_accumulated_gradient(prepared)
        return self._atomic_gradient(self._cached_train_state, self.frozen_state, prepared.rngs,
                                     prepared.observation, prepared.actions)

    def _atomic_accumulated_gradient(self, prepared):
        """One compiled sequential loop, with one reusable FP32 mean buffer.

        A single XLA allocation plan avoids allocating a large new backward
        workspace around live buffers from a separate accumulation executable.
        Physical backward shapes and their unit keys stay unchanged.
        """
        jax, jnp = self.jax, self.jnp
        if self._local_scan_source is not self._atomic_gradient:
            physical_units = self.physical_batch_capacity // 4
            physical_gradient = self._atomic_gradient

            def accumulated_gradient(train_state, frozen_state, rngs, observation, actions):
                parts = rngs.shape[0] // physical_units
                if rngs.shape[0] % physical_units:
                    raise ValueError("Logical task must be divisible by its physical batch")
                inputs = jax.tree.map(
                    lambda value: value.reshape((parts, physical_units, *value.shape[1:])),
                    (rngs, observation, actions))
                zeros = jax.tree.map(lambda value: jnp.zeros_like(value, dtype=jnp.float32),
                                     train_state.to_pure_dict())
                weight = jnp.float32(1 / parts)

                def body(carry, part):
                    mean, loss_sum, finite_sum = carry
                    keys, obs, act = part
                    loss, gradient, finite = physical_gradient(train_state, frozen_state, keys, obs, act)
                    mean = jax.tree.map(lambda a, b: a + b * weight, mean, gradient)
                    return (mean, loss_sum + loss * weight, finite_sum & finite), None

                (mean, loss, finite), _ = jax.lax.scan(
                    body, (zeros, jnp.float32(0), jnp.bool_(True)), inputs)
                return loss, mean, finite

            self._local_scan_gradient = jax.jit(accumulated_gradient)
            self._local_scan_source = self._atomic_gradient
        return self._local_scan_gradient(self._cached_train_state, self.frozen_state, prepared.rngs,
                                         prepared.observation, prepared.actions)

    def atomic_physical_parts(self, prepared):
        """Split a logical bundle without changing any unit input, key or version.

        Host accumulation keeps only one physical backward on the GPU. A
        logical 16-sample task can therefore use four physical batches of four
        without needing a second full gradient buffer on an H100.
        """
        count = self.physical_batch_capacity // 4
        parts = []
        for start in range(0, len(prepared.sample_positions) // 4, count):
            stop = min(start + count, len(prepared.sample_positions) // 4)
            parts.append(prepared._replace(
                observation=self.jax.tree.map(lambda value: value[start:stop], prepared.observation),
                actions=prepared.actions[start:stop], rngs=prepared.rngs[start:stop],
                sample_positions=prepared.sample_positions[start * 4:stop * 4]))
        return tuple(parts)

    def _compute_prepared_atomic(self, version, task_id, prepared):
        jax, np = self.jax, self.np
        from jade_lotus.local_accumulation import HostGradientAccumulator
        from jade_lotus.host_precision import configured_host_widen_workers, widen_host_gradients
        from jade_lotus.gpu_host_transfer import configured_gradient_host_transfer, transfer_gradient_result

        host_widen_workers = configured_host_widen_workers(self.config)
        transfer_mode = configured_gradient_host_transfer(self.config)
        began = time.monotonic()
        physical_parts = self.atomic_physical_parts(prepared)
        device_accumulation = getattr(self, "local_gradient_accumulation", "host") == "gpu"
        parts = (prepared,) if device_accumulation else physical_parts
        samples = len(prepared.sample_positions)
        loss_sum = gpu_seconds = d2h_seconds = host_widen_seconds = 0.0
        stage_seconds = materialization_seconds = 0.0
        transferred_dtypes = set()
        workers = self.config.get("local_accumulation_workers", 8)
        with HostGradientAccumulator(samples, len(parts), workers=workers) as accumulator:
            for part in parts:
                with jax.default_device(self.device):
                    started = time.monotonic()
                    result = self.atomic_device_gradient(part)
                    jax.block_until_ready(result)
                    gpu_seconds += time.monotonic() - started
                    started = time.monotonic()
                    (loss, pure, finite), transfer_metrics = transfer_gradient_result(
                        jax, result, self.device, mode=transfer_mode)
                    d2h_seconds += time.monotonic() - started
                    stage_seconds += transfer_metrics["gradient_d2h_stage_seconds"]
                    materialization_seconds += transfer_metrics["gradient_numpy_materialization_seconds"]
                transferred_dtypes.update(str(value.dtype) for value in _flat(pure).values())
                if not bool(finite) or not np.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in {task_id} version {version}")
                started = time.monotonic()
                gradients = widen_host_gradients(_flat(pure), workers=host_widen_workers)
                host_widen_seconds += time.monotonic() - started
                part_samples = len(part.sample_positions)
                accumulator.add(gradients, part_samples)
                loss_sum += float(loss) * part_samples / samples
                del result, pure, gradients
            gradients = accumulator.finish()
            accumulation_seconds = accumulator.seconds
        elapsed = time.monotonic() - began
        return gradients, {"loss": loss_sum, "samples": samples, "microbatches": len(physical_parts),
                           "atomic_units": samples // 4, "atomic_size": 4,
                           "atomic_batching": True, "physical_batch_size": min(samples, self.physical_batch_capacity),
                           "physical_batch_capacity": self.physical_batch_capacity, "gradient_calls": len(physical_parts),
                           "gradient_d2h_calls": len(parts), "gpu_local_accumulation": device_accumulation,
                           "task_batch_capacity": getattr(self, "task_batch_capacity", self.physical_batch_capacity),
                           "gpu_parameter_dtype": getattr(self, "gpu_parameter_dtype", "float32"),
                           "configured_device_gradient_dtype": getattr(self, "device_gradient_dtype", "float32"),
                           "device_gradient_dtype": ",".join(sorted(transferred_dtypes)),
                           "host_gradient_dtype": "float32",
                           "vision_precision": getattr(self, "vision_precision", {}),
                           "gpu_seconds": gpu_seconds, "gradient_d2h_seconds": d2h_seconds,
                           "gradient_host_transfer": transfer_mode,
                           "gradient_d2h_stage_seconds": stage_seconds,
                           "gradient_numpy_materialization_seconds": materialization_seconds,
                           "host_widen_seconds": host_widen_seconds,
                           "local_accumulation_seconds": accumulation_seconds,
                           "compute_seconds": elapsed, "data_seconds": 0.0,
                           "samples_per_second": samples / elapsed, "prepared_inputs": True}

    def prepare_task(self, sample_positions: list[int], *, version=None):
        """Stage a fixed task on-device for compute-only and pipeline benchmarks."""
        if self.atomic_batching:
            positions = tuple(sample_positions)
            if not positions:
                raise ValueError("Atomic task must be nonempty")
            if version is None:
                version = int(positions[0]) // self.config.get("batch_size", 256)
            units = self._atomic_units(positions, version)
            prepared = [self._prepare_microbatch(unit.sample_positions) for unit in units]
            return self.prepare_atomic_device(prepared, version=version, sample_positions=positions)
        if not sample_positions or len(sample_positions) % self.microbatch_size:
            raise ValueError("Task size must be positive and divisible by microbatch_size")
        return [self._prepare_microbatch(sample_positions[start:start + self.microbatch_size])
                for start in range(0, len(sample_positions), self.microbatch_size)]

    def _make_fused_gradient(self):
        """Keep sequential FP32 accumulation and task RNGs inside one JIT.

        The scan carries just one trainable-gradient tree. It does not vmap
        four model backward passes, accept stale gradients, or reduce the
        number of samples. There is no host synchronization between passes.
        """
        jax, jnp = self.jax, self.jnp
        gradient = self._gradient

        def task_gradient(train_state, frozen_state, task_rng, observations, actions):
            count = actions.shape[0]

            def one(index):
                rng = jax.random.fold_in(task_rng, index)
                observation = jax.tree.map(lambda value: value[index], observations)
                loss, grads = gradient(train_state, frozen_state, rng, observation, actions[index])
                pure = jax.tree.map(lambda value: value.astype(jnp.float32), grads.to_pure_dict())
                return loss.astype(jnp.float32), pure

            first_loss, first_grads = one(jnp.int32(0))

            def accumulate(carry, index):
                loss_sum, grad_sum, finite_losses = carry
                loss, grads = one(index)
                return (loss_sum + loss, jax.tree.map(jnp.add, grad_sum, grads),
                        finite_losses & jnp.isfinite(loss)), None

            (loss_sum, grad_sum, finite_losses), _ = jax.lax.scan(
                accumulate, (first_loss, first_grads, jnp.isfinite(first_loss)),
                jnp.arange(1, count, dtype=jnp.int32),
            )
            return loss_sum / count, jax.tree.map(lambda value: value / count, grad_sum), finite_losses

        return jax.jit(task_gradient)

    def _compute_prepared_fused(self, train_state, version, task_id, prepared, seed):
        jax, jnp, np = self.jax, self.jnp, self.np
        began = time.monotonic()
        # Only the (small) task's input batches are stacked. Model state and
        # gradients retain their original shapes and are reused by the scan.
        observations, actions = jax.tree.map(lambda *values: jnp.stack(values, axis=0), *prepared)
        if self._fused_gradient is None:
            self._fused_gradient = self._make_fused_gradient()
        try:
            with jax.default_device(self.device):
                result = self._fused_gradient(
                    train_state, self.frozen_state, jax.random.key(int(seed) % (2**32)), observations, actions,
                )
                # One task boundary, rather than one synchronization and one
                # Python float(loss) conversion after every microbatch.
                loss, pure_grads, finite_losses = jax.device_get(result)
        except Exception as exc:
            raise RuntimeError(
                f"Fused task gradient failed for {task_id}; disable fused_task_gradient to use "
                f"the original microbatch path: {exc}"
            ) from exc
        loss_scalar = float(loss)
        if not bool(finite_losses) or not np.isfinite(loss_scalar):
            raise FloatingPointError(f"Nonfinite loss in {task_id} version {version}")
        gradients = {key: np.asarray(value, dtype=np.float32) for key, value in _flat(pure_grads).items()}
        if any(not np.isfinite(value).all() for value in gradients.values()):
            raise FloatingPointError(f"Nonfinite gradient in {task_id} version {version}")
        samples = len(prepared) * self.microbatch_size
        elapsed = time.monotonic() - began
        return gradients, {"loss": loss_scalar, "samples": samples, "microbatches": len(prepared),
                           "compute_seconds": elapsed, "data_seconds": 0.0,
                           "samples_per_second": samples / elapsed, "prepared_inputs": True,
                           "fused_task_gradient": True}

    def compute_prepared(self, params: dict, version: int, task_id: str, prepared, seed: int, *, sample_positions=None):
        """Compute identical microbatch RNGs from an already prepared task.

        The benchmark excludes input preparation, but includes gradient D2H and
        finite-value validation. Normal workers continue to use compute().
        """
        jax, jnp, np = self.jax, self.jnp, self.np
        if self.atomic_batching:
            prepared = self.prepare_atomic_device(prepared, version=version, sample_positions=sample_positions)
            self._set_parameter_version(params, version)
            return self._compute_prepared_atomic(version, task_id, prepared)
        if not prepared:
            raise ValueError("Prepared task must be nonempty")
        train_state = self._set_parameter_version(params, version)
        if self.fused_task_gradient:
            return self._compute_prepared_fused(train_state, version, task_id, prepared, seed)
        began = time.monotonic()
        accum, loss_sum = None, 0.0
        for microbatch, (observation, actions) in enumerate(prepared):
            rng = jax.random.fold_in(jax.random.key(int(seed) % (2**32)), microbatch)
            with jax.default_device(self.device):
                loss, grads = self._gradient(train_state, self.frozen_state, rng, observation, actions)
                loss, grads = jax.block_until_ready((loss, grads))
            loss_scalar = float(loss)
            if not np.isfinite(loss_scalar):
                raise FloatingPointError(f"Nonfinite loss in {task_id} version {version}")
            pure_grads = grads.to_pure_dict()
            accum = pure_grads if accum is None else jax.tree.map(jnp.add, accum, pure_grads)
            loss_sum += loss_scalar
        gradients = {key: np.asarray(value / len(prepared), dtype=np.float32)
                     for key, value in _flat(accum).items()}
        if any(not np.isfinite(value).all() for value in gradients.values()):
            raise FloatingPointError(f"Nonfinite gradient in {task_id} version {version}")
        samples = len(prepared) * self.microbatch_size
        elapsed = time.monotonic() - began
        return gradients, {"loss": loss_sum / len(prepared), "samples": samples,
                           "microbatches": len(prepared), "compute_seconds": elapsed,
                           "data_seconds": 0.0, "samples_per_second": samples / elapsed,
                           "prepared_inputs": True}

    def compute(self, params: dict, version: int, task_id: str, sample_positions: list[int], seed: int):
        jax, jnp, np = self.jax, self.jnp, self.np
        if self.atomic_batching:
            # The legacy task seed remains an API argument, but the atomic
            # stream is keyed by the run seed, version and absolute positions.
            self._atomic_units(sample_positions, version)
            began = time.monotonic()
            prepared = self.prepare_task(sample_positions, version=version)
            data_seconds = time.monotonic() - began
            gradients, metrics = self.compute_prepared(params, version, task_id, prepared, seed)
            elapsed = time.monotonic() - began
            first_epoch, first_index = self.stream.resolve(sample_positions[0])
            last_epoch, last_index = self.stream.resolve(sample_positions[-1])
            metrics.update(compute_seconds=elapsed, data_seconds=data_seconds, prepared_inputs=False,
                           samples_per_second=len(sample_positions) / elapsed,
                           first_epoch=first_epoch, first_index=first_index,
                           last_epoch=last_epoch, last_index=last_index)
            return gradients, metrics
        if not sample_positions or len(sample_positions) % self.microbatch_size:
            raise ValueError("Task size must be positive and divisible by microbatch_size")
        train_state = self._set_parameter_version(params, version)
        if self.fused_task_gradient:
            began = time.monotonic()
            prepared = self.prepare_task(sample_positions)
            data_seconds = time.monotonic() - began
            gradients, metrics = self.compute_prepared(params, version, task_id, prepared, seed)
            elapsed = time.monotonic() - began
            first_epoch, first_index = self.stream.resolve(sample_positions[0])
            last_epoch, last_index = self.stream.resolve(sample_positions[-1])
            metrics.update(compute_seconds=elapsed, data_seconds=data_seconds, prepared_inputs=False,
                           samples_per_second=len(sample_positions) / elapsed,
                           first_epoch=first_epoch, first_index=first_index,
                           last_epoch=last_epoch, last_index=last_index)
            return gradients, metrics
        accum = None
        loss_sum = 0.0
        began = time.monotonic()
        data_seconds = 0.0
        for start in range(0, len(sample_positions), self.microbatch_size):
            data_started = time.monotonic()
            chunk_positions = sample_positions[start:start + self.microbatch_size]
            observation, actions = self._prepare_microbatch(chunk_positions)
            data_seconds += time.monotonic() - data_started
            # Each task has a stable coordinator-assigned seed; retrying it on a
            # different worker uses identical flow noise/time/augmentation RNG.
            rng = jax.random.fold_in(jax.random.key(int(seed) % (2**32)), start // self.microbatch_size)
            with jax.default_device(self.device):
                loss, grads = self._gradient(train_state, self.frozen_state, rng, observation, actions)
                loss, grads = jax.block_until_ready((loss, grads))
            loss_scalar = float(loss)
            if not np.isfinite(loss_scalar):
                raise FloatingPointError(f"Nonfinite loss in {task_id} version {version}")
            # Accumulate on GPU; transfer one complete task gradient per leaf.
            pure_grads = grads.to_pure_dict()
            if accum is None:
                accum = pure_grads
            else:
                accum = jax.tree.map(jnp.add, accum, pure_grads)
            loss_sum += loss_scalar
        count = len(sample_positions) // self.microbatch_size
        gradients = {key: np.asarray(value / count, dtype=np.float32) for key, value in _flat(accum).items()}
        if any(not np.isfinite(value).all() for value in gradients.values()):
            raise FloatingPointError(f"Nonfinite gradient in {task_id} version {version}")
        elapsed = time.monotonic() - began
        first_epoch, first_index = self.stream.resolve(sample_positions[0])
        last_epoch, last_index = self.stream.resolve(sample_positions[-1])
        return gradients, {"loss": loss_sum / count, "samples": len(sample_positions),
                           "microbatches": count, "compute_seconds": elapsed,
                           "data_seconds": data_seconds, "samples_per_second": len(sample_positions) / elapsed,
                           "first_epoch": first_epoch, "first_index": first_index,
                           "last_epoch": last_epoch, "last_index": last_index}

    def export(self, params: dict, destination: Path):
        """Export full native pi05 weights, folding adapters only in LoRA mode."""
        import orbax.checkpoint as ocp
        from openpi.models import model as model_module
        from openpi.shared import normalize

        np = self.np
        destination = Path(destination).resolve()
        if destination.exists():
            raise FileExistsError(destination)
        if self.training_mode == "full":
            if set(params) != set(self.trainable_paths):
                raise ValueError("Full-model export requires every authoritative parameter leaf")
            if any(np.asarray(value).dtype != np.float32 for value in params.values()):
                raise ValueError("Full-model export requires FP32 authoritative parameters")
            merged = {key: np.asarray(value) for key, value in params.items()}
            inference_model = self.model_config
        else:
            full = {key: np.asarray(value)
                    for key, value in _flat(self.template_state.to_pure_dict()).items()}
            full.update(params)
            merged = merge_lora(full)
            del full
            inference_model = dataclasses.replace(self.model_config, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
        expected_model = self.nnx.eval_shape(lambda: inference_model.create(self.jax.random.key(0)))
        expected = _flat(self.nnx.state(expected_model).to_pure_dict())
        if set(expected) != set(merged) or any(expected[key].shape != merged[key].shape for key in expected):
            raise ValueError("Native export differs from the ordinary upstream pi05 parameter schema")
        assert_finite_arrays(merged)
        destination.mkdir(parents=True)
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(destination / "params", ocp.args.PyTreeSave({"params": _unflat(merged)}))
        # Verify the exact public restore API, all parameter shapes, and all
        # values after serialization before the export can become authoritative.
        # This reads host arrays, without allocating a second model on the GPU.
        restored = _flat(model_module.restore_params(destination / "params", restore_type=np.ndarray))
        if set(restored) != set(merged) or any(restored[key].shape != merged[key].shape for key in merged):
            raise ValueError("Native checkpoint restore differs from exported parameter schema")
        assert_finite_arrays(restored)
        if any(not np.array_equal(restored[key], merged[key]) for key in merged):
            raise ValueError("Native checkpoint restore changed exported parameter values")
        validation = {"native_restore_validated": True, "upstream_schema_validated": True,
                      "all_parameters_finite": True, "restore_bitwise_equal": True,
                      "training_mode": self.training_mode, "lora_merged": self.training_mode == "lora",
                      "parameter_leaves": len(restored),
                      "parameter_count": sum(value.size for value in restored.values()),
                      "parameter_bytes": sum(value.nbytes for value in restored.values())}
        del restored, merged
        normalize.save(destination / "assets/local/libero", self.data_config.norm_stats)
        (destination / "model_config.json").write_text(json.dumps({
            "model_class": "openpi.models.pi0_config.Pi0Config", "model": dataclasses.asdict(inference_model),
            "training_mode": self.training_mode,
            "source_lora_rank": self.config.get("lora_rank", 8) if self.training_mode == "lora" else None,
            "lora_merged": self.training_mode == "lora",
            "normalization_asset_id": "local/libero", "dataset_root": self.config.get("dataset_root", DEFAULT_DATASET),
        }, indent=2) + "\n")
        (destination / "load_model.py").write_text(
            '"""Load the standard native exported model; set PYTHONPATH to the local OpenPI src."""\n'
            'import json\nfrom pathlib import Path\nfrom openpi.models.pi0_config import Pi0Config\n'
            'from openpi.models.model import restore_params\n'
            'def load_model(path):\n    path = Path(path)\n'
            '    config = Pi0Config(**json.loads((path / "model_config.json").read_text())["model"])\n'
            '    return config.load(restore_params(path / "params"))\n'
        )
        from jade_lotus.runtime import atomic_json
        atomic_json(destination / "EXPORT_VALIDATION.json", validation)
        return destination


def assert_finite_arrays(arrays):
    """Check large host arrays with a bounded temporary allocation."""
    import numpy as np

    for key, value in arrays.items():
        flattened = np.asarray(value).reshape(-1)
        for start in range(0, flattened.size, 1024 * 1024):
            if not np.isfinite(flattened[start:start + 1024 * 1024]).all():
                raise FloatingPointError(f"Nonfinite parameter in export: {key}")


def merge_lora(flat_params: dict):
    """Fold OpenPI LoRA factors, including the leading scanned layer axis.

    LoRA attention uses alpha/rank=1 in this experiment; FFN's implementation
    directly adds A@B. Exporting folded kernels permits ordinary upstream
    gemma_2b/gemma_300m inference independent of the training rank override.
    """
    import numpy as np

    # Preserve frozen BF16 norm/bias dtypes. Casting them to FP32 can change
    # arithmetic such as BF16(1 + scale), even with identical stored values.
    result = {key: np.asarray(value).copy() for key, value in flat_params.items() if "lora" not in key}
    for key, a in flat_params.items():
        if not key.endswith("lora_a"):
            continue
        b_key = key[:-1] + "b"
        if b_key not in flat_params:
            raise ValueError(f"Missing LoRA pair {b_key}")
        if key.endswith("/lora_a"):
            kernel_key = key[:-len("lora_a")] + "w"
        else:
            kernel_key = key[:-len("_lora_a")]
        if kernel_key not in result:
            raise ValueError(f"Missing base kernel for {key}: {kernel_key}")
        b = np.asarray(flat_params[b_key], dtype=np.float32)
        if "/attn_vec_einsum" in key:
            # OpenPI's generic LoRA equation generator removes the head label N
            # from both intermediates for BTNH,NHD->BTD:
            #   BTNH,NHL->BTL ; BTL,NLD->BTD.
            # Thus B is summed across heads before folding into each A head.
            # A conventional per-head A@B would change this trained function.
            b = b.sum(axis=-3, keepdims=True)
        result[kernel_key] = np.asarray(result[kernel_key], dtype=np.float32) + np.matmul(
            np.asarray(a, dtype=np.float32), b,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    _imports(config)
    if not args.preflight:
        parser.error("Only --preflight CLI is supported; workers use GradientEngine")
    report = inspect_configuration(config)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
