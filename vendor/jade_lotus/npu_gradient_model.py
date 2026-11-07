"""OpenPI Torch/NPU mean gradients for the existing CPU-authoritative protocol.

No optimizer lives on the worker. Random augmentation, noise and flow time are
prepared for canonical four-sample units before physical microbatch slicing.
The seven structurally unused pi05 parameters are explicitly outside the CPU
optimizer schema, matching upstream AdamW's treatment of ``grad is None``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from types import SimpleNamespace
import types

import numpy as np

from jade_lotus.atomic_batch import ATOMIC_SIZE, describe_units
from jade_lotus.gradient_export import NATIVE_SNAPSHOT, gradient_export_mode
from jade_lotus.npu_gradient_transfer import (
    LEGACY_MODE as LEGACY_GRADIENT_TRANSFER_MODE,
    MODE as PINNED_GRADIENT_TRANSFER_MODE,
    PinnedNpuGradientTransfer,
    download_metrics,
)
from jade_lotus.npu_parameter_transfer import (
    LEGACY_MODE,
    MODE as BATCHED_TRANSFER_MODE,
    BatchedNpuParameterTransfer,
    upload_metrics,
)


PARAMETER_SCHEMA = "openpi-pi05-torch-unique-active-fp32-v1"
RNG_SCHEME = "torch-atomic4-preprocess-cpu-noise-inverse-beta-v1"
_EMBEDDING = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
_TIED_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"
_LAYER = re.compile(r"paligemma_with_expert\.paligemma\.model\.language_model\.layers\.(\d+)\.")
_CHECK_CHUNK = 1 << 20


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _checkpoint_file(config):
    value = config.get("base_params", config.get("pytorch_weight_path"))
    if not value:
        raise ValueError("NPU training requires an explicit local base_params checkpoint")
    path = Path(value).expanduser().resolve()
    path = path / "model.safetensors" if path.is_dir() else path
    if not path.is_file() or path.suffix != ".safetensors":
        raise FileNotFoundError(f"Missing local Torch checkpoint: {path}")
    return path


def _checkpoint_index(config):
    """Read the safetensors header without Torch imports or device allocation."""
    path = _checkpoint_file(config)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header")
        size = int.from_bytes(prefix, "little")
        if not 2 <= size <= 16 * 2**20:
            raise ValueError("Invalid safetensors header size")
        encoded = stream.read(size)
        if len(encoded) != size:
            raise ValueError("Truncated safetensors header")
        header = json.loads(encoded)
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not header:
        raise ValueError("Invalid safetensors parameter metadata")
    aliases = {}
    # safetensors stores this shared value as lm_head; named_parameters chooses
    # embed_tokens first. Keep the actual model's canonical name in our schema.
    if metadata.get(_EMBEDDING) == _TIED_HEAD:
        aliases[_TIED_HEAD] = _EMBEDDING
    index = {}
    previous_end = 0
    for stored, item in sorted(header.items(), key=lambda pair: pair[1]["data_offsets"][0]):
        name = aliases.get(stored, stored)
        shape, dtype, offsets = item["shape"], item["dtype"], item["data_offsets"]
        if (not isinstance(shape, list) or not shape or any(type(x) is not int or x < 1 for x in shape)
                or dtype not in {"F32", "BF16"} or len(offsets) != 2
                or any(type(x) is not int for x in offsets)):
            raise ValueError(f"Unsupported checkpoint tensor: {stored}")
        start, end = offsets
        expected = int(np.prod(shape, dtype=np.int64)) * (4 if dtype == "F32" else 2)
        if start != previous_end or end - start != expected or name in index:
            raise ValueError(f"Invalid checkpoint offsets or duplicate canonical name: {stored}")
        previous_end = end
        index[name] = {"stored_name": stored, "shape": tuple(shape), "dtype": dtype,
                       "offset": 8 + size + start, "bytes": expected}
    if 8 + size + previous_end != path.stat().st_size:
        raise ValueError("Checkpoint payload size differs from header")
    return path, index, metadata


def _unused_names(config, available):
    explicit = config.get("npu_unused_parameter_names")
    if explicit is not None:
        if (not isinstance(explicit, list) or any(not isinstance(x, str) for x in explicit)
                or len(set(explicit)) != len(explicit)):
            raise ValueError("npu_unused_parameter_names must be unique explicit names")
        missing = set(explicit) - set(available)
        if missing:
            raise ValueError(f"Declared unused parameters are absent: {sorted(missing)}")
        return tuple(sorted(explicit))
    if config.get("openpi_config", "pi05_libero_lerobot") != "pi05_libero_lerobot":
        raise ValueError("Other Torch recipes require an explicit unused-parameter schema")
    layers = [int(match.group(1)) for name in available if (match := _LAYER.match(name))]
    if not layers:
        raise ValueError("Expected pi05 prefix language layers are absent")
    prefix = f"paligemma_with_expert.paligemma.model.language_model.layers.{max(layers)}."
    names = [prefix + suffix for suffix in ("self_attn.o_proj.weight", "mlp.gate_proj.weight",
             "mlp.up_proj.weight", "mlp.down_proj.weight", "post_attention_layernorm.weight")]
    names += ["paligemma_with_expert.paligemma.model.language_model.norm.weight",
              "paligemma_with_expert.gemma_expert.lm_head.weight"]
    if not set(names).issubset(available):
        raise ValueError("pi05 structurally unused parameter schema changed")
    return tuple(sorted(names))


def npu_parameter_schema(config):
    """Cheap, versioned schema suitable for freezing before a model is loaded."""
    path, index, aliases = _checkpoint_index(config)
    unused = _unused_names(config, index)
    active = {name: {"shape": list(item["shape"]), "dtype": "float32"}
              for name, item in sorted(index.items()) if name not in unused}
    return {"name": PARAMETER_SCHEMA, "checkpoint": str(path), "schema": active,
            "excluded_parameter_names": list(unused), "shared_aliases": aliases,
            "parameter_count": sum(int(np.prod(v["shape"], dtype=np.int64)) for v in active.values()),
            "parameter_leaves": len(active)}


def _finite(array):
    flat = np.asarray(array).reshape(-1)
    return all(np.isfinite(flat[start:start + _CHECK_CHUNK]).all()
               for start in range(0, flat.size, _CHECK_CHUNK))


def load_npu_initial_parameters(config):
    """Return CPU FP32 authority arrays and provenance without a full CPU model.

    BF16 values are widened by their exact IEEE bit representation. This
    avoids importing torch_npu on a stable CPU coordinator or duplicating a
    3B-parameter model merely to discover names and shapes.
    """
    began = time.monotonic()
    path, index, aliases = _checkpoint_index(config)
    unused = _unused_names(config, index)
    arrays = {}
    for name, item in sorted(index.items()):
        if name in unused:
            continue
        raw = np.memmap(path, mode="r", offset=item["offset"],
                        dtype="<f4" if item["dtype"] == "F32" else "<u2", shape=item["shape"])
        if item["dtype"] == "BF16":
            value = (np.asarray(raw).astype(np.uint32) << 16).view(np.float32)
        else:
            value = np.array(raw, dtype=np.float32, copy=True, order="C")
        if not _finite(value):
            raise FloatingPointError(f"Nonfinite initial parameter: {name}")
        arrays[name] = value
        del raw
    if not arrays:
        raise ValueError("NPU parameter schema is empty")
    metadata = {"backend": "torch_npu", "parameter_schema": PARAMETER_SCHEMA,
                "source": "coordinator_local_torch_checkpoint", "device": "cpu",
                "base_params": str(path), "excluded_parameter_names": list(unused),
                "shared_aliases": aliases, "parameter_count": sum(a.size for a in arrays.values()),
                "parameter_leaves": len(arrays), "backward_executed": False,
                "initialization_timings": {"checkpoint_read_seconds": time.monotonic() - began},
                "seconds": time.monotonic() - began}
    return arrays, metadata


def _imports(config):
    root = Path(config.get("openpi_root", Path(__file__).resolve().parents[2] / "openpi")).resolve()
    if not (root / "src/openpi/models_pytorch/pi0_pytorch.py").is_file():
        raise FileNotFoundError(f"OpenPI Torch implementation is absent: {root}")
    for path in (root / "packages/openpi-client/src", root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("OPENPI_DATA_HOME", str(root / "data/ascend"))
    if config.get("tokenizer_path") is not None:
        tokenizer = Path(config["tokenizer_path"]).expanduser().resolve()
        if not tokenizer.is_file():
            raise FileNotFoundError(f"Frozen tokenizer is absent: {tokenizer}")
        os.environ["OPENPI_TOKENIZER_PATH"] = str(tokenizer)
    return root


def _tree_map(function, tree):
    if isinstance(tree, dict):
        return {key: _tree_map(function, value) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_tree_map(function, value) for value in tree)
    if isinstance(tree, list):
        return [_tree_map(function, value) for value in tree]
    return None if tree is None else function(tree)


def _stack_samples(samples):
    first = samples[0]
    if isinstance(first, dict):
        if any(set(sample) != set(first) for sample in samples):
            raise ValueError("Transformed sample fields differ")
        return {key: _stack_samples([sample[key] for sample in samples]) for key in first}
    return np.stack(samples)


def _observation(batch):
    import torch
    # Match Observation.from_dict's actual Torch reader boundary: transformed
    # LIBERO RGB is still uint8 NHWC and must become [-1, 1] NCHW here.
    images = {name: (image.to(dtype=torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
                     if image.dtype == torch.uint8 else image)
              for name, image in batch["image"].items()}
    return SimpleNamespace(images=images, image_masks=batch["image_mask"], state=batch["state"].float(),
                           tokenized_prompt=batch.get("tokenized_prompt"),
                           tokenized_prompt_mask=batch.get("tokenized_prompt_mask"),
                           token_ar_mask=batch.get("token_ar_mask"), token_loss_mask=batch.get("token_loss_mask"))


def _slice_observation(observation, start, end):
    fields = {key: _tree_map(lambda value: value[start:end], value)
              for key, value in vars(observation).items() if key != "_jade_preprocessed"}
    return SimpleNamespace(**fields, _jade_preprocessed=True)


@dataclass(frozen=True)
class NpuPreparedUnit:
    observation: object
    actions: object
    noise: object
    flow_time: object
    sample_positions: tuple[int, ...]
    seed: int


@dataclass(frozen=True)
class NpuPreparedTask:
    units: tuple[NpuPreparedUnit, ...]
    version: int
    sample_positions: tuple[int, ...]


class TorchNpuGradientEngine:
    """A single assigned NPU, with no optimizer and no activation checkpointing."""

    def __init__(self, config, device_index=None):
        began = time.monotonic()
        self.config = dict(config)
        if config.get("training_mode") != "full" or config.get("atomic_batching") is not True:
            raise ValueError("Torch NPU workers require explicit full atomic training")
        if config.get("disable_activation_checkpointing") is not True:
            raise ValueError("This NPU validation requires all gradient recomputation disabled")
        if config.get("pytorch_gradient_checkpointing", False) is not False:
            raise ValueError("pytorch_gradient_checkpointing must be false")
        gradient_export_mode(config)
        self.root = _imports(config)
        import torch
        import torch_npu  # noqa: F401
        import safetensors.torch
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
        from openpi.training import config as upstream_config
        from openpi.training import data_loader

        if not torch.npu.is_available():
            raise RuntimeError("No assigned NPU is available")
        index = 0 if device_index is None else device_index
        if type(index) is not int or not 0 <= index < torch.npu.device_count():
            raise ValueError("NPU device index is outside the assigned visible devices")
        torch_device = torch.device(f"npu:{index}")
        torch.npu.set_device(torch_device)
        torch.set_num_threads(int(config.get("npu_cpu_threads", 2)))
        self.initialization_timings = {"imports_seconds": time.monotonic() - began}
        phase = time.monotonic()
        recipe = upstream_config.get_config(config.get("openpi_config", "pi05_libero_lerobot"))
        checkpoint = _checkpoint_file(config)
        assets = dataclasses.replace(recipe.data.assets, assets_dir=str(checkpoint.parent / "assets"))
        model_config = dataclasses.replace(recipe.model, dtype=recipe.pytorch_training_precision)
        data_config = dataclasses.replace(recipe.data, assets=assets).create(recipe.assets_dirs, model_config)
        data_config = dataclasses.replace(data_config, dataset_root=str(Path(config["dataset_root"]).resolve()))
        if data_config.use_quantile_norm or data_config.norm_stats is None:
            raise ValueError("pi05_libero_lerobot must use its matching mean/std assets")
        dataset = data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)
        dataset = data_loader.transform_dataset(dataset, data_config)
        self.initialization_timings["dataset_seconds"] = time.monotonic() - phase
        phase = time.monotonic()
        model = PI0Pytorch(model_config).to(torch_device)
        safetensors.torch.load_model(model, str(checkpoint), strict=True)
        self.initialization_timings["model_load_seconds"] = time.monotonic() - phase
        schema = npu_parameter_schema(config)
        self._configure_components(torch, model, dataset, preprocess_observation_pytorch,
                                   torch_device, schema["excluded_parameter_names"])
        if set(self.parameters) != set(schema["schema"]):
            raise ValueError("Torch unique parameter names differ from frozen checkpoint schema")
        for name, parameter in self.parameters.items():
            if list(parameter.shape) != schema["schema"][name]["shape"]:
                raise ValueError(f"Torch parameter shape differs from checkpoint: {name}")
        self.model_config, self.data_config = model_config, data_config
        self.initialization_timings["total_seconds"] = time.monotonic() - began
        print(json.dumps({"time": time.time(), "event": "npu_engine_initialized", "backend": "torch_npu",
                          "device_kind": self.device.device_kind, "device_id": index,
                          "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                          "parameter_schema": PARAMETER_SCHEMA, "rng_scheme": RNG_SCHEME,
                          "physical_microbatch": self.microbatch_size,
                          "logical_task_capacity": self.task_batch_capacity,
                          "gradient_checkpointing_enabled": False, "checkpoint_guard_installed": True,
                          "parameter_count": sum(p.numel() for p in self.parameters.values()),
                          "parameter_leaves": len(self.parameters),
                          "npu_parameter_transfer": self.parameter_transfer_mode,
                          "npu_gradient_transfer": self.gradient_transfer_mode,
                          "npu_gradient_export": self.gradient_export_mode,
                          "excluded_parameter_names": list(self.unused_parameter_names),
                          "model_config": {"action_horizon": model_config.action_horizon,
                                           "action_dim": model_config.action_dim,
                                           "max_token_len": model_config.max_token_len},
                          "initialization_timings": self.initialization_timings}, sort_keys=True), flush=True)

    def _configure_components(self, torch, model, dataset, preprocess, torch_device, unused):
        """Shared runtime setup, also exercised with an actual tiny CPU model."""
        self.torch, self.model, self.dataset = torch, model, dataset
        self.torch_device, self.preprocess = torch_device, preprocess
        self.training_mode, self.atomic_batching = "full", True
        self.seed = int(self.config.get("seed", 42))
        self.microbatch_size = _positive_integer(self.config.get("microbatch_size", 2), "microbatch_size")
        if self.microbatch_size not in (1, 2, 4):
            raise ValueError("NPU physical microbatch must be 1, 2 or 4")
        self.physical_batch_capacity = self.microbatch_size
        self.task_batch_capacity = self.config.get("worker_task_batch_size", 8)
        if type(self.task_batch_capacity) is not int or self.task_batch_capacity not in (4, 8, 16):
            raise ValueError("Logical NPU task capacity must be 4, 8 or 16")
        self.device = SimpleNamespace(platform=torch_device.type,
            device_kind=torch.npu.get_device_name(torch_device) if torch_device.type == "npu" else "cpu",
            id=torch_device.index or 0)
        self.unused_parameter_names = tuple(sorted(unused))
        named = dict(model.named_parameters())
        if not set(unused).issubset(named):
            raise ValueError("Declared unused parameter absent from actual model")
        for name, parameter in named.items():
            parameter.requires_grad_(name not in unused)
        self.parameters = {name: parameter for name, parameter in sorted(named.items()) if name not in unused}
        if self.config.get("gradient_wire_dtype") == "native_mixed_exact":
            from jade_lotus.array_blob import make_native_wire_policy, native_wire_policy_sha256
            from jade_lotus.sharded_transport import require_sharded

            require_sharded(self.config)
            actual_schema = {name: {"shape": list(parameter.shape), "dtype": "float32"}
                             for name, parameter in self.parameters.items()}
            native_dtypes = {name: str(parameter.dtype).removeprefix("torch.")
                             for name, parameter in self.parameters.items()}
            actual_policy = make_native_wire_policy(actual_schema, native_dtypes)
            if (actual_policy != self.config["gradient_wire_policy"]
                    or native_wire_policy_sha256(actual_policy) != self.config["gradient_wire_policy_sha256"]):
                raise ValueError("Actual model native dtypes differ from the frozen gradient wire policy")
        self.model.train()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        for module in model.modules():
            for flag in ("gradient_checkpointing", "gradient_checkpointing_enabled"):
                if hasattr(module, flag):
                    setattr(module, flag, False)
        self.checkpoint_call_count = 0
        self._assert_no_checkpointing()
        self._parameter_version = None
        self.parameter_transfer_mode = self.config.get("npu_parameter_transfer", LEGACY_MODE)
        if self.parameter_transfer_mode not in (LEGACY_MODE, BATCHED_TRANSFER_MODE):
            raise ValueError("npu_parameter_transfer must be leaf_copy_v1 or batched_pinned_v1")
        self._parameter_transfer = (
            BatchedNpuParameterTransfer(self.parameters, torch_device, torch_module=torch)
            if self.parameter_transfer_mode == BATCHED_TRANSFER_MODE else None
        )
        self.gradient_transfer_mode = self.config.get("npu_gradient_transfer", LEGACY_GRADIENT_TRANSFER_MODE)
        if self.gradient_transfer_mode not in (LEGACY_GRADIENT_TRANSFER_MODE, PINNED_GRADIENT_TRANSFER_MODE):
            raise ValueError("npu_gradient_transfer must be leaf_copy_v1 or pinned_native_v1")
        self._gradient_transfer = (
            PinnedNpuGradientTransfer(self.parameters, torch_device, torch_module=torch)
            if self.gradient_transfer_mode == PINNED_GRADIENT_TRANSFER_MODE else None
        )
        self.gradient_export_mode = gradient_export_mode(self.config)
        self._permutations = {}
        original = getattr(model, "_preprocess_observation", None)
        if original is not None:
            def already_prepared(_model, observation, *, train=True):
                if not getattr(observation, "_jade_preprocessed", False):
                    return original(observation, train=train)
                return (list(observation.images.values()), list(observation.image_masks.values()),
                        observation.tokenized_prompt, observation.tokenized_prompt_mask, observation.state)
            model._preprocess_observation = types.MethodType(already_prepared, model)

    def _assert_no_checkpointing(self):
        enabled = [name for name, module in self.model.named_modules()
                   if any(bool(getattr(module, flag, False))
                          for flag in ("gradient_checkpointing", "gradient_checkpointing_enabled"))]
        if enabled:
            raise AssertionError(f"Activation checkpoint flags unexpectedly enabled: {enabled}")

    @contextmanager
    def _checkpoint_guard(self):
        import torch.utils.checkpoint
        checkpoint_module = self.torch.utils.checkpoint
        original = checkpoint_module.checkpoint
        bound = []
        def forbidden(*args, **kwargs):
            self.checkpoint_call_count += 1
            raise AssertionError("Activation checkpoint invoked although recomputation is disabled")
        checkpoint_module.checkpoint = forbidden
        for module in self.model.modules():
            if hasattr(module, "_gradient_checkpointing_func"):
                bound.append((module, module._gradient_checkpointing_func))
                module._gradient_checkpointing_func = forbidden
        try:
            yield
        finally:
            checkpoint_module.checkpoint = original
            for module, function in bound:
                module._gradient_checkpointing_func = function

    @contextmanager
    def _random_state(self, seed):
        torch = self.torch
        cpu_state = torch.random.get_rng_state()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        npu_state = torch.npu.get_rng_state(self.torch_device) if self.torch_device.type == "npu" else None
        try:
            seed = int(seed) & (2**63 - 1)
            torch.random.default_generator.manual_seed(seed)
            np.random.seed(seed % 2**32)
            random.seed(seed)
            if npu_state is not None:
                torch.npu.manual_seed(seed)
            yield
        finally:
            torch.random.set_rng_state(cpu_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            if npu_state is not None:
                torch.npu.set_rng_state(npu_state, self.torch_device)

    def _sample(self, position):
        epoch, offset = divmod(position, len(self.dataset))
        if epoch not in self._permutations:
            self._permutations[epoch] = np.random.default_rng(self.seed + epoch).permutation(len(self.dataset))
            for previous in list(self._permutations):
                if previous < epoch - 1:
                    self._permutations.pop(previous)
        index = int(self._permutations[epoch][offset])
        sample_seed = int.from_bytes(hashlib.blake2b(f"{self.seed}:{epoch}:{index}".encode(), digest_size=8).digest(), "big")
        with self._random_state(sample_seed):
            return self.dataset[index]

    def prepare_atomic_device(self, sample_positions, *, version):
        positions = tuple(sample_positions)
        units = describe_units(positions, run_seed=self.seed, version=version,
                               batch_size=self.config.get("batch_size", 256))
        if len(positions) > self.task_batch_capacity:
            raise ValueError("Task exceeds the worker's declared logical capacity")
        torch, prepared = self.torch, []
        for unit in units:
            batch = _stack_samples([self._sample(position) for position in unit.sample_positions])
            # Mean/std transforms retain NumPy FP64 intermediates. The official
            # Torch trainer uses FP32 actions; avoid creating any NPU FP64 copy.
            batch["actions"] = np.asarray(batch["actions"], dtype=np.float32)
            batch["state"] = np.asarray(batch["state"], dtype=np.float32)
            batch = _tree_map(lambda value: torch.as_tensor(np.asarray(value)).to(self.torch_device), batch)
            actions, observation = batch["actions"], _observation(batch)
            with self._random_state(unit.seed), torch.no_grad():
                observation = self.preprocess(observation, train=True)
            observation._jade_preprocessed = True
            # A separate CPU generator fixes tensor identity independently of
            # NPU RNG implementation, augmentation consumption and slicing.
            generator = torch.Generator(device="cpu").manual_seed(unit.seed & (2**63 - 1))
            noise = torch.randn(tuple(actions.shape), generator=generator, dtype=torch.float32)
            flow_time = torch.rand((ATOMIC_SIZE,), generator=generator, dtype=torch.float32).pow(2.0 / 3.0)
            flow_time = flow_time * 0.999 + 0.001
            prepared.append(NpuPreparedUnit(observation, actions.to(dtype=torch.float32),
                noise.to(self.torch_device), flow_time.to(self.torch_device), unit.sample_positions, unit.seed))
        return NpuPreparedTask(tuple(prepared), version, positions)

    def _sync(self):
        if self.torch_device.type == "npu":
            self.torch.npu.synchronize(self.torch_device)

    def initial_params(self):
        arrays, metadata = load_npu_initial_parameters(self.config)
        if set(arrays) != set(self.parameters):
            raise ValueError("Initial parameter schema differs from the worker model")
        self.initial_parameter_metadata = metadata
        return arrays

    def invalidate_parameter_cache(self):
        self._parameter_version = None
        if self._parameter_transfer is not None:
            self._parameter_transfer.invalidate()

    def _set_parameter_version(self, params, version):
        if type(version) is not int or version < 0:
            raise ValueError("Parameter version must be a nonnegative integer")
        if self._parameter_transfer is not None:
            # Clear before any new transfer. A partial copy must never leave
            # the previous version eligible for a stale cache hit.
            if self._parameter_version != version:
                self._parameter_version = None
            try:
                metrics = self._parameter_transfer.load(params, version)
            except BaseException:
                self.invalidate_parameter_cache()
                raise
            self._parameter_version = version
            return metrics
        metrics = upload_metrics()
        if self._parameter_version == version:
            metrics["parameter_upload_cache_hit"] = 1
            return metrics
        self._parameter_version = None
        began = time.monotonic()
        if set(params) != set(self.parameters):
            raise ValueError("NPU parameter keys differ from the frozen training schema")
        arrays = {}
        for name, parameter in self.parameters.items():
            value = np.asarray(params[name])
            if value.dtype != np.float32 or tuple(value.shape) != tuple(parameter.shape) or not _finite(value):
                raise ValueError(f"Invalid CPU authority parameter: {name}")
            arrays[name] = value
        metrics["parameter_validation_seconds"] = time.monotonic() - began
        phase = time.monotonic()
        with self.torch.no_grad():
            for name, parameter in self.parameters.items():
                parameter.copy_(self.torch.from_numpy(arrays[name]))
                metrics["parameter_upload_copy_calls"] += 1
        metrics["parameter_h2d_copy_seconds"] = time.monotonic() - phase
        phase = time.monotonic()
        self._sync()
        metrics["parameter_h2d_sync_seconds"] = time.monotonic() - phase
        self._parameter_version = version
        metrics["parameter_upload_source_bytes"] = sum(value.nbytes for value in arrays.values())
        metrics["parameter_upload_native_bytes"] = sum(p.numel() * p.element_size()
                                                       for p in self.parameters.values())
        metrics["parameter_upload_packed_bytes"] = metrics["parameter_upload_native_bytes"]
        metrics["parameter_upload_dtype_groups"] = len({p.dtype for p in self.parameters.values()})
        metrics["parameter_upload_total_seconds"] = time.monotonic() - began
        return metrics

    def _download_gradients(self):
        if self._gradient_transfer is not None:
            if self.gradient_export_mode == NATIVE_SNAPSHOT:
                snapshot, metrics = self._gradient_transfer.download_native()
                if (snapshot.wire_policy != self.config["gradient_wire_policy"]
                        or snapshot.wire_policy_sha256 != self.config["gradient_wire_policy_sha256"]):
                    snapshot.close()
                    raise ValueError("Downloaded native gradient dtypes differ from the frozen wire policy")
                return snapshot, metrics
            return self._gradient_transfer.download()
        began, gradients, dtypes = time.monotonic(), {}, set()
        metrics = download_metrics()
        for name, parameter in self.parameters.items():
            phase = time.monotonic()
            if parameter.grad is None:
                raise ValueError(f"Unexpected missing gradient in the active schema: {name}")
            metrics["gradient_validation_seconds"] += time.monotonic() - phase
            phase = time.monotonic()
            # Preserve the original native copy followed by CPU widening. Its
            # blocking copy timer includes implicit synchronization/allocation.
            host = parameter.grad.detach().to(device="cpu", copy=True)
            metrics["gradient_d2h_enqueue_seconds"] += time.monotonic() - phase
            metrics["gradient_download_copy_calls"] += 1
            phase = time.monotonic()
            value = host.float().numpy()
            metrics["gradient_cpu_widen_snapshot_seconds"] += time.monotonic() - phase
            metrics["gradient_cpu_widen_snapshot_calls"] += 1
            phase = time.monotonic()
            if not _finite(value):
                raise FloatingPointError(f"Nonfinite NPU gradient: {name}")
            metrics["gradient_finite_seconds"] += time.monotonic() - phase
            gradients[name] = value
            dtypes.add(parameter.grad.dtype)
            metrics["gradient_download_native_bytes"] += parameter.grad.numel() * parameter.grad.element_size()
            metrics["gradient_download_fp32_bytes"] += value.nbytes
        metrics["gradient_download_dtype_groups"] = len(dtypes)
        metrics["gradient_download_total_seconds"] = time.monotonic() - began
        return gradients, metrics

    def compute_prepared(self, params, version, task_id, prepared, seed):
        if seed != self.seed:
            raise ValueError("Atomic task seed differs from the frozen run seed")
        if not isinstance(prepared, NpuPreparedTask) or prepared.version != version:
            raise ValueError("Prepared NPU task has a different parameter version")
        expected = describe_units(prepared.sample_positions, run_seed=self.seed, version=version,
                                  batch_size=self.config.get("batch_size", 256))
        if len(expected) != len(prepared.units) or len(prepared.sample_positions) > self.task_batch_capacity:
            raise ValueError("Prepared NPU unit count or capacity differs")
        for reference, actual in zip(expected, prepared.units, strict=True):
            if reference.sample_positions != actual.sample_positions or reference.seed != actual.seed:
                raise ValueError("Prepared NPU unit random identity differs")
        self._assert_no_checkpointing()
        parameter_metrics = self._set_parameter_version(params, version)
        torch = self.torch
        self.model.zero_grad(set_to_none=True)
        total = len(prepared.sample_positions)
        loss_value, backward_count = 0.0, 0
        self._sync()
        started = time.monotonic()
        gradients = None
        try:
            with self._checkpoint_guard():
                for unit in prepared.units:
                    for start in range(0, ATOMIC_SIZE, self.microbatch_size):
                        end = min(start + self.microbatch_size, ATOMIC_SIZE)
                        observation = _slice_observation(unit.observation, start, end)
                        losses = self.model(observation, unit.actions[start:end],
                                            noise=unit.noise[start:end], time=unit.flow_time[start:end])
                        loss = losses.mean()
                        if not bool(torch.isfinite(loss).item()):
                            raise FloatingPointError("Nonfinite NPU task loss")
                        weight = (end - start) / total
                        (loss * weight).backward()
                        loss_value += float(loss.detach().item()) * weight
                        backward_count += 1
                        del observation, losses, loss
            self._sync()
            npu_seconds = time.monotonic() - started
            d2h_started = time.monotonic()
            gradients, gradient_metrics = self._download_gradients()
            d2h_seconds = time.monotonic() - d2h_started
            metrics = {"loss": loss_value, "npu_seconds": npu_seconds,
                "d2h_seconds": d2h_seconds, "compute_seconds": time.monotonic() - started,
                "physical_microbatch": self.microbatch_size, "physical_batch_size": self.microbatch_size,
                "logical_samples": total, "logical_batch_size": total, "backward_count": backward_count,
                "checkpoint_calls": self.checkpoint_call_count, "checkpoint_call_count": self.checkpoint_call_count,
                "activation_checkpointing": 0, "gradient_checkpointing_enabled": 0,
                "checkpoint_guard_installed": 1, "finite_gradients": 1,
                "parameter_leaves": len(self.parameters), **parameter_metrics, **gradient_metrics}
            if self.torch_device.type == "npu":
                metrics["peak_device_allocated_bytes"] = torch.npu.max_memory_allocated(self.torch_device)
            self._assert_no_checkpointing()
            return gradients, metrics
        except BaseException:
            if self.gradient_export_mode == NATIVE_SNAPSHOT and gradients is not None:
                gradients.close()
            raise
        finally:
            self.model.zero_grad(set_to_none=True)

    def compute(self, params, version, task_id, sample_positions, seed):
        began = time.monotonic()
        prepared = self.prepare_atomic_device(sample_positions, version=version)
        data_seconds = time.monotonic() - began
        gradients, metrics = self.compute_prepared(params, version, task_id, prepared, seed)
        metrics.update(data_seconds=data_seconds, compute_seconds=time.monotonic() - began,
                       samples_per_second=len(prepared.sample_positions) / max(time.monotonic() - began, 1e-9))
        return gradients, metrics


# A short alias keeps the backend factory readable and matches the legacy name.
GradientEngine = TorchNpuGradientEngine
