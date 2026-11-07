"""Load a resident OpenPI export without rounding its trained FP32 parameters.

``load_policy(...).infer(observation, noise=float32_array_50_by_32)`` is the
ordinary OpenPI Policy API. ``check_only=True`` verifies the export's files and
headers without importing Torch/OpenPI or allocating a model. Use ``device=
"cpu"`` to perform the real model-load checks on a CPU instead of an NPU.

The upstream inference factory repeats a blanket BF16 conversion after loading;
that conversion would round trained FP32 normalization/projection parameters.
Here precision is established by model construction, then checkpoint tensors are
loaded strictly and checked before and after the device transfer.
"""

from __future__ import annotations

from collections import Counter
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid
from typing import Any


_RECIPE = "pi05_libero_lerobot"
_F32_CHECK_MAX_BYTES = 1024 * 1024
_DTYPE_BYTES = {"F32": 4, "BF16": 2, "I64": 8}
_TORCH_DTYPES = {"F32": "float32", "BF16": "bfloat16", "I64": "int64"}
_EXPECTED_BUFFERS = {
    "paligemma_with_expert.gemma_expert.model.rotary_emb.inv_freq",
    "paligemma_with_expert.paligemma.model.language_model.rotary_emb.inv_freq",
    "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_ids",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "device": stat.st_dev, "inode": stat.st_ino}


def _read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors prefix: {path}")
        length = int.from_bytes(prefix, "little")
        if not 2 <= length <= 16 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header length: {path}")
        encoded = stream.read(length)
        if len(encoded) != length:
            raise ValueError(f"Truncated safetensors header: {path}")
    header = json.loads(encoded)
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not header:
        raise ValueError(f"Invalid safetensors metadata: {path}")
    previous_end = 0
    for name, item in sorted(header.items(), key=lambda pair: pair[1]["data_offsets"][0]):
        shape, dtype, offsets = item["shape"], item["dtype"], item["data_offsets"]
        if (dtype not in _DTYPE_BYTES or not isinstance(shape, list)
                or any(type(size) is not int or size < 0 for size in shape)
                or len(offsets) != 2 or any(type(value) is not int for value in offsets)):
            raise ValueError(f"Unsupported safetensors tensor: {path}:{name}")
        start, end = offsets
        if start != previous_end or end - start != math.prod(shape) * _DTYPE_BYTES[dtype]:
            raise ValueError(f"Invalid safetensors offsets: {path}:{name}")
        previous_end = end
    if 8 + length + previous_end != path.stat().st_size:
        raise ValueError(f"Safetensors file size does not match its header: {path}")
    return {"header_bytes": length, "metadata": metadata, "tensors": header,
            "tensor_count": len(header),
            "tensor_dtype_counts": dict(Counter(item["dtype"] for item in header.values()))}


def inspect_checkpoint(checkpoint: str | Path, *, expected_sha: str | None = None) -> dict[str, Any]:
    """Verify file identity, complete model SHA, assets and sidecar without ML imports.

    If ``expected_sha`` is omitted, the completed export receipt supplies the
    expected model hash. A caller-supplied hash must also agree with that receipt.
    The 7.5 GB model is hashed once per call, in bounded chunks.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    model_file = checkpoint / "model.safetensors"
    buffer_file = checkpoint / "resident_buffers.safetensors"
    norm_file = checkpoint / "assets/libero/norm_stats.json"
    export_file = checkpoint / "export_receipt.json"
    source_receipt = json.loads(export_file.read_text()) if export_file.is_file() else None
    if source_receipt is not None and source_receipt.get("status") != "passed":
        raise ValueError("Resident export receipt did not pass")
    receipt_sha = source_receipt["model"]["sha256"] if source_receipt else None
    if expected_sha is None:
        expected_sha = receipt_sha
    if expected_sha is None:
        raise ValueError("Supply expected_sha or a completed export_receipt.json")
    if (len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha)
            or (receipt_sha is not None and expected_sha != receipt_sha)):
        raise ValueError("Expected model SHA256 is invalid or differs from the export receipt")
    identity = _file_identity(model_file)
    model_header = _read_header(model_file)
    model_sha = _sha256(model_file)
    if model_sha != expected_sha:
        raise ValueError("Model SHA256 differs from the expected completed export")
    if _file_identity(model_file) != identity:
        raise RuntimeError("Model file changed during SHA256 verification")
    buffer_header = _read_header(buffer_file)
    if set(buffer_header["tensors"]) != _EXPECTED_BUFFERS:
        raise ValueError("Expected all three resident nonpersistent buffers")
    buffer_sha = _sha256(buffer_file)
    norm_sha = _sha256(norm_file)
    norm_json = json.loads(norm_file.read_text())
    if not {"state", "actions"}.issubset(norm_json.get("norm_stats", {})):
        raise ValueError("LIBERO normalization must contain state and actions")
    if source_receipt is not None:
        if model_header["tensor_count"] != source_receipt["model"]["tensor_count"]:
            raise ValueError("Export tensor count differs from its receipt")
        if buffer_sha != source_receipt["nonpersistent_buffers"]["sha256"]:
            raise ValueError("Nonpersistent buffers differ from the export receipt")
        if norm_sha != source_receipt["assets_sha256"]["assets/libero/norm_stats.json"]:
            raise ValueError("LIBERO normalization differs from the export receipt")
    return {
        "format": "jade-resident-libero-policy-load-v1", "status": "files_verified",
        "checkpoint": str(checkpoint), "model_sha256": model_sha,
        "expected_model_sha256": expected_sha, "model_file_identity": identity,
        "model_header": model_header, "buffer_header": buffer_header,
        "buffer_sha256": buffer_sha, "norm_stats_sha256": norm_sha,
        "export_receipt_sha256": _sha256(export_file) if source_receipt else None,
        "training_step": source_receipt.get("step") if source_receipt else None,
        "model_loaded": False,
    }


def _select_openpi(openpi_root: str | Path, tokenizer_path: str | Path | None) -> tuple[Path, Path]:
    root = Path(openpi_root).expanduser().resolve()
    src = root / "src"
    if not (src / "openpi/models_pytorch/pi0_pytorch.py").is_file():
        raise FileNotFoundError(f"Frozen OpenPI Torch source is absent: {root}")
    for name, module in tuple(sys.modules.items()):
        if name == "openpi" or name.startswith("openpi."):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(src):
                raise RuntimeError(f"OpenPI was already imported from another source: {name}: {origin}")
    for path in (root / "packages/openpi-client/src", src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    selected_tokenizer = tokenizer_path or os.environ.get("OPENPI_TOKENIZER_PATH")
    if not selected_tokenizer:
        raise ValueError("Supply a local tokenizer_path or set OPENPI_TOKENIZER_PATH")
    tokenizer = Path(selected_tokenizer).expanduser().resolve()
    if not tokenizer.is_file():
        raise FileNotFoundError(f"Local PaliGemma tokenizer is absent: {tokenizer}")
    os.environ["OPENPI_TOKENIZER_PATH"] = str(tokenizer)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    return root, tokenizer


def _check_tensor(tensor: Any, item: dict[str, Any], name: str) -> None:
    if list(tensor.shape) != item["shape"] or str(tensor.dtype) != "torch." + _TORCH_DTYPES[item["dtype"]]:
        raise ValueError(f"Loaded model tensor shape/dtype differs from checkpoint: {name}")


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _check_model_schema(model: Any, header: dict[str, Any]) -> dict[str, Any]:
    state = model.state_dict()
    stored = header["tensors"]
    aliases = {name: target for name, target in header["metadata"].items()
               if name not in stored and target in stored}
    if set(state) != set(stored) | set(aliases):
        raise ValueError(f"Model/checkpoint state keys differ: missing={sorted(set(state) - set(stored) - set(aliases))} "
                         f"unexpected={sorted(set(stored) - set(state))}")
    for name, tensor in state.items():
        _check_tensor(tensor, stored[aliases.get(name, name)], name)
    for name, target in aliases.items():
        left, right = state[name], state[target]
        if (left.data_ptr() != right.data_ptr() or left.storage_offset() != right.storage_offset()
                or left.stride() != right.stride()):
            raise ValueError(f"Checkpoint tied alias is no longer tied: {name} -> {target}")
    return {"state_dict_tensor_count_including_aliases": len(state),
            "stored_tensor_count": len(stored), "tied_aliases": aliases,
            "all_shapes_and_dtypes_match": True, "all_aliases_share_storage": True,
            "model_state_dtype_counts_including_aliases": dict(Counter(str(t.dtype) for t in state.values()))}


def _check_small_fp32_values(model: Any, model_file: Path, header: dict[str, Any]) -> dict[str, Any]:
    """Compare raw bytes of every FP32 tensor no larger than one MiB.

    Byte comparisons also preserve signed zero; no tolerance comparison or
    FP32->BF16->FP32 round trip can pass this check by accident.
    """
    state = model.state_dict()
    checks = []
    with model_file.open("rb") as stream:
        for name, item in header["tensors"].items():
            start, end = item["data_offsets"]
            if item["dtype"] != "F32" or end - start > _F32_CHECK_MAX_BYTES:
                continue
            stream.seek(8 + header["header_bytes"] + start)
            expected = stream.read(end - start)
            actual = state[name].detach().cpu().contiguous().numpy().tobytes()
            if actual != expected:
                raise ValueError(f"Loaded FP32 values differ from the exact export: {name}")
            checks.append({"name": name, "bytes": len(actual), "byte_equal": True,
                           "sha256": hashlib.sha256(actual).hexdigest()})
    if not checks:
        raise ValueError("Expected FP32 normalization/projection tensors for exact-load verification")
    return {"max_tensor_bytes": _F32_CHECK_MAX_BYTES, "tensor_count": len(checks),
            "bytes_compared": sum(item["bytes"] for item in checks), "all_byte_equal": True,
            "tensors": checks, "scope": "All FP32 stored tensors at or below max_tensor_bytes; not a full model value hash"}


def _write_receipt(path: str | Path | None, receipt: dict[str, Any]) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_policy(
    openpi_root: str | Path,
    checkpoint: str | Path,
    device: str = "npu:0",
    *,
    expected_sha: str | None = None,
    receipt_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    check_only: bool = False,
) -> Any:
    """Return an exact-weight OpenPI Policy with a JSON-safe ``load_receipt``.

    ``policy.infer(obs, noise=noise)`` accepts explicit NumPy FP32 noise with
    shape ``(50, 32)`` and returns unnormalized actions with shape ``(50, 7)``.
    ``check_only=True`` returns a file-verification receipt instead of a Policy.
    ``device='cpu'`` performs the complete construction/load/value verification.
    """
    began = time.monotonic()
    receipt = inspect_checkpoint(checkpoint, expected_sha=expected_sha)
    if check_only:
        receipt["seconds"] = time.monotonic() - began
        _write_receipt(receipt_path, receipt)
        return receipt
    root, tokenizer = _select_openpi(openpi_root, tokenizer_path)
    import torch
    import safetensors.torch
    from openpi.models_pytorch.device import resolve_device
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.policies.policy import Policy
    from openpi.training import config
    from openpi.training import checkpoints
    from openpi import transforms

    selected_device = resolve_device(device)
    if selected_device.type not in {"cpu", "npu"}:
        raise ValueError("This loader verifies CPU or assigned NPU inference only")
    if selected_device.type == "npu":
        torch.npu.set_device(selected_device)
    checkpoint_dir = Path(receipt["checkpoint"])
    recipe = config.get_config(_RECIPE)
    model_config = dataclasses.replace(recipe.model, dtype=recipe.pytorch_training_precision)
    if (model_config.action_horizon != 50 or model_config.action_dim != 32
            or not model_config.pi05 or not model_config.discrete_state_input):
        raise ValueError("Frozen pi05_libero_lerobot model semantics changed")
    assets = dataclasses.replace(recipe.data.assets, assets_dir=str(checkpoint_dir / "assets"))
    data_config = dataclasses.replace(recipe.data, assets=assets).create(recipe.assets_dirs, model_config)
    if (data_config.asset_id != "libero" or data_config.use_quantile_norm
            or data_config.norm_epsilon != 1e-8 or data_config.unnorm_epsilon != 0.0):
        raise ValueError("Frozen LIBERO mean/std normalization semantics changed")
    norm_stats = checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    model = PI0Pytorch(model_config)
    _check_model_schema(model, receipt["model_header"])
    safetensors.torch.load_model(model, str(checkpoint_dir / "model.safetensors"), strict=True)
    cpu_schema = _check_model_schema(model, receipt["model_header"])
    cpu_values = _check_small_fp32_values(model, checkpoint_dir / "model.safetensors", receipt["model_header"])

    saved_buffers = safetensors.torch.load_file(str(checkpoint_dir / "resident_buffers.safetensors"), device="cpu")
    restored = []
    with torch.no_grad():
        for name, tensor in saved_buffers.items():
            target = model.get_buffer(name)
            _check_tensor(target, receipt["buffer_header"]["tensors"][name], name)
            target.copy_(tensor)
            if _tensor_bytes(target) != _tensor_bytes(tensor):
                raise ValueError(f"Nonpersistent buffer restoration failed: {name}")
            restored.append({"name": name, "dtype": str(tensor.dtype),
                             "shape": list(tensor.shape), "byte_equal": True})
    policy = Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm,
                                 epsilon=data_config.norm_epsilon),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm,
                                   epsilon=data_config.unnorm_epsilon),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": 10}, metadata=recipe.policy_metadata,
        is_pytorch=True, pytorch_device=str(selected_device),
    )
    device_schema = _check_model_schema(model, receipt["model_header"])
    device_values = _check_small_fp32_values(model, checkpoint_dir / "model.safetensors", receipt["model_header"])
    for name, tensor in saved_buffers.items():
        target = model.get_buffer(name)
        _check_tensor(target, receipt["buffer_header"]["tensors"][name], name)
        if _tensor_bytes(target) != _tensor_bytes(tensor):
            raise ValueError(f"Buffer changed during device transfer: {name}")
    if _file_identity(checkpoint_dir / "model.safetensors") != receipt["model_file_identity"]:
        raise RuntimeError("Model file changed during model loading")
    source_files = ["models/model.py", "models/pi0_config.py", "models/tokenizer.py",
                    "models_pytorch/pi0_pytorch.py", "models_pytorch/gemma_pytorch.py",
                    "policies/policy.py", "policies/libero_policy.py", "training/config.py", "transforms.py"]
    receipt.update({
        "status": "passed", "model_loaded": True, "device": str(selected_device),
        "openpi_root": str(root), "recipe": _RECIPE,
        "cpu_load_schema": cpu_schema, "device_load_schema": device_schema,
        "cpu_fp32_value_checks": cpu_values, "device_fp32_value_checks": device_values,
        "buffer_restoration": {"count": len(restored), "all_restored_exactly": True,
                               "verified_after_device_transfer": True, "tensors": restored},
        "post_load_dtype_conversion": False,
        "strict_safetensors_load": True,
        "model_config": {"pi05": model_config.pi05, "dtype": model_config.dtype,
                         "action_horizon": model_config.action_horizon, "action_dim": model_config.action_dim,
                         "max_token_len": model_config.max_token_len,
                         "discrete_state_input": model_config.discrete_state_input},
        "data_semantics": {"asset_id": data_config.asset_id, "use_quantile_norm": data_config.use_quantile_norm,
                           "norm_epsilon": data_config.norm_epsilon, "unnorm_epsilon": data_config.unnorm_epsilon,
                           "camera_resize": [224, 224], "output_action_components": 7,
                           "denoise_steps": 10, "extra_delta_transform": bool(recipe.data.extra_delta_transform),
                           "input_transforms": [type(item).__name__ for item in data_config.data_transforms.inputs],
                           "model_input_transforms": [type(item).__name__ for item in data_config.model_transforms.inputs],
                           "output_transforms": [type(item).__name__ for item in data_config.data_transforms.outputs]},
        "tokenizer": {"path": str(tokenizer), "sha256": _sha256(tokenizer)},
        "openpi_source_sha256": {name: _sha256(root / "src/openpi" / name) for name in source_files},
        "torch_version": str(torch.__version__), "seconds": time.monotonic() - began,
        "verification_scope": "All model state shapes/dtypes and tied aliases; small FP32 values and three sidecar buffers "
                              "byte-exact before/after transfer. Full on-device state hash and policy quality are not verified here.",
    })
    policy.load_receipt = receipt
    _write_receipt(receipt_path, receipt)
    return policy
