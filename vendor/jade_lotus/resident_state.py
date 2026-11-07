"""Chunked byte-exact checkpoints for eager native torch.optim.AdamW.

The model and its moments retain their native dtypes, including BF16. There is
no FP32 master and no equivalence claim to the CPU FP32 optimizer. All named
model parameters (frozen and tied aliases included), buffers, optimizer states,
per-parameter steps, groups and defaults are covered by stable names.

No torch_npu import, network operation or training protocol lives here. Callers
must serialize save/load with optimizer application and synchronize any other
streams that mutate state. Loading never replaces Parameter objects. A failed
device copy may leave partial state: the target must remain LOADING until a
complete verified restore succeeds. Controller/RNG/scheduler state belongs in
the caller's metadata; this module does not infer it from weights.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid


FORMAT = "jade-native-resident-state-v1"
PRECISION = "native_model_and_native_adamw_moments_no_fp32_master"
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
WIDTHS = {"bool": 1, "uint8": 1, "int8": 1, "int16": 2, "int32": 4, "int64": 8,
          "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REF = "__native_tensor__"
_TUPLE = "__native_tuple__"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_json(path, value, *, durable=True):
    """Atomic exclusive publication; never overwrite an earlier report/manifest."""
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json(value) + b"\n")
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        os.link(temporary, path)
        if durable:
            _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_native_state_report(report, path):
    """Publish a new JSON report atomically, refusing any existing destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish_json(path, report)
    return str(path.resolve())


def _model_layout(model):
    parameters = dict(model.named_parameters(remove_duplicate=False))
    canonical = {id(value): name for name, value in model.named_parameters()}
    buffers = dict(model.named_buffers(remove_duplicate=False))
    canonical_buffers = {id(value): name for name, value in model.named_buffers()}
    if not parameters:
        raise ValueError("A model with named parameters is required")
    unique_parameters = {canonical[id(value)]: value for value in parameters.values()}
    unique_buffers = {canonical_buffers[id(value)]: value for value in buffers.values()}
    for value in list(unique_parameters.values()) + list(unique_buffers.values()):
        if str(value.dtype).removeprefix("torch.") not in WIDTHS or not value.is_contiguous():
            raise ValueError("Checkpoint tensors must have supported native dtypes and contiguous storage")
    occupied = {}
    for name, value in unique_parameters.items():
        if not value.numel():
            continue
        pointer = value.untyped_storage().data_ptr()
        lower = value.storage_offset() * value.element_size()
        upper = lower + value.numel() * value.element_size()
        if any(lower < end and start < upper for start, end in occupied.setdefault(pointer, [])):
            raise ValueError("Shared storage must use tied Parameter aliases, not distinct overlapping Parameters")
        occupied[pointer].append((lower, upper))
    parameter_table = [{"name": name, "canonical": canonical[id(value)],
                        "requires_grad": value.requires_grad,
                        "tensor": {_REF: "model.parameter." + canonical[id(value)]}}
                       for name, value in sorted(parameters.items())]
    buffer_table = [{"name": name, "canonical": canonical_buffers[id(value)],
                     "tensor": {_REF: "model.buffer." + canonical_buffers[id(value)]}}
                    for name, value in sorted(buffers.items())]
    return unique_parameters, unique_buffers, canonical, parameter_table, buffer_table


def _groups(optimizer, canonical):
    import torch

    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Only eager native torch.optim.AdamW is supported")
    groups, seen = [], set()
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            if id(parameter) not in canonical or id(parameter) in seen:
                raise ValueError("Optimizer parameters must be unique named model Parameters")
            names.append(canonical[id(parameter)])
            seen.add(id(parameter))
        groups.append(names)
    if not seen or any(id(parameter) not in seen for parameter in optimizer.state):
        raise ValueError("Optimizer state contains an unowned parameter or no parameter group")
    return groups


def _catalog(model, optimizer, step):
    import torch

    params, buffers, canonical, model_table, buffer_table = _model_layout(model)
    group_names = _groups(optimizer, canonical)
    tensors = {}

    def register(name, tensor, role):
        dtype = str(tensor.dtype).removeprefix("torch.")
        if dtype not in WIDTHS or not tensor.is_contiguous():
            raise ValueError(f"Unsupported native tensor: {name}")
        tensors[name] = (tensor.detach(), role)
        return {_REF: name}

    def encode(value, name, anchor_device):
        if torch.is_tensor(value):
            if value.device.type not in ("cpu", anchor_device.type):
                raise ValueError("Optimizer tensor belongs to an unrelated device")
            return register(name, value, "cpu" if value.device.type == "cpu" else "parameter_device")
        if isinstance(value, tuple):
            return {_TUPLE: [encode(child, f"{name}.{index}", anchor_device) for index, child in enumerate(value)]}
        if isinstance(value, list):
            return [encode(child, f"{name}.{index}", anchor_device) for index, child in enumerate(value)]
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("Optimizer metadata keys must be strings")
            return {key: encode(child, name + "." + key, anchor_device) for key, child in sorted(value.items())}
        if value is None or type(value) in (str, bool, int, float):
            _json(value)
            return value
        raise ValueError(f"Unsupported optimizer metadata: {name}")

    for name, value in sorted(params.items()):
        register("model.parameter." + name, value, "parameter_device")
    for name, value in sorted(buffers.items()):
        register("model.buffer." + name, value, "cpu" if value.device.type == "cpu" else "parameter_device")
    first = next(parameter for group in optimizer.param_groups for parameter in group["params"])
    defaults = encode(optimizer.defaults, "optimizer.defaults", first.device)
    groups, states = [], {}
    for index, (group, names) in enumerate(zip(optimizer.param_groups, group_names, strict=True)):
        anchor = group["params"][0].device if group["params"] else first.device
        groups.append({"parameters": names,
                       "options": encode({k: v for k, v in group.items() if k != "params"},
                                         f"optimizer.group.{index}", anchor)})
        for name, parameter in zip(names, group["params"], strict=True):
            state = optimizer.state.get(parameter, {})
            if not state:
                if step and parameter.requires_grad:
                    raise ValueError(f"Missing native optimizer state at committed step {step}: {name}")
                states[name] = {}
                continue
            expected = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if group.get("amsgrad") else set())
            if set(state) != expected:
                raise ValueError(f"Incomplete or unsupported native AdamW state: {name}")
            counter = state["step"]
            if (not torch.is_tensor(counter) or counter.numel() != 1
                    or not math.isfinite(float(counter.item())) or float(counter.item()) != int(counter.item())
                    or int(counter.item()) < 0 or int(counter.item()) > step
                    or (parameter.requires_grad and int(counter.item()) != step)):
                raise ValueError(f"Optimizer parameter step differs from committed step: {name}")
            for key in expected - {"step"}:
                value = state[key]
                if (not torch.is_tensor(value) or value.dtype != parameter.dtype
                        or value.shape != parameter.shape or value.device != parameter.device):
                    raise ValueError(f"Native optimizer moment shape/dtype/device differs: {name}.{key}")
            states[name] = encode(state, "optimizer.state." + name, parameter.device)
    structure = {"model_parameters": model_table, "model_buffers": buffer_table,
                 "optimizer_defaults": defaults, "optimizer_groups": groups, "optimizer_states": states}
    return structure, tensors


def _logical(value, tensors):
    if isinstance(value, dict):
        if set(value) == {_REF}:
            name = value[_REF]
            entry = tensors[name]
            return {"tensor": name, **{key: entry[key] for key in ("dtype", "shape", "nbytes", "sha256")}}
        return {key: _logical(child, tensors) for key, child in value.items()}
    if isinstance(value, list):
        return [_logical(child, tensors) for child in value]
    return value


def _state_digests(manifest):
    tensors = manifest["tensors"]
    model = {key: _logical(manifest[key], tensors) for key in ("model_parameters", "model_buffers")}
    optimizer = {key: _logical(manifest[key], tensors) for key in
                 ("optimizer_defaults", "optimizer_groups", "optimizer_states")}
    optimizer.update(step=manifest["step"], precision=PRECISION)
    return _digest(model), _digest(optimizer)


def _signature(tensors):
    return {name: (value.data_ptr(), value._version, str(value.device), str(value.dtype), tuple(value.shape))
            for name, (value, _role) in tensors.items()}


def _host_parts(tensor, chunk_bytes, *, nonnegative=False):
    import torch

    flat = tensor.detach().reshape(-1)
    count = max(1, chunk_bytes // tensor.element_size())
    for start in range(0, flat.numel(), count):
        host = flat[start:start + count].to(device="cpu", copy=True)
        if not bool(torch.isfinite(host).all().item()) or (nonnegative and bool((host < 0).any().item())):
            raise ValueError("Checkpoint model/optimizer values must be finite and second moments nonnegative")
        yield start, host, host.view(torch.uint8).numpy().tobytes()


def _report(directory, manifest, *, operation, seconds=None):
    return {"format": FORMAT, "precision_contract": PRECISION, "operation": operation,
            "checkpoint": str(Path(directory).resolve()), "step": manifest["step"],
            "manifest_sha256": manifest["manifest_sha256"], "model_sha256": manifest["model_sha256"],
            "optimizer_sha256": manifest["optimizer_sha256"], "payload_bytes": manifest["payload_bytes"],
            "tensor_count": len(manifest["tensors"]), "chunk_bytes": manifest["chunk_bytes"],
            "named_model_parameters": len(manifest["model_parameters"]),
            "unique_model_parameters": len({r["canonical"] for r in manifest["model_parameters"]}),
            "optimizer_parameter_states": len(manifest["optimizer_states"]),
            "payload_bytes_verified": operation in ("verify", "load"),
            "verified_loaded_state": operation == "load", "seconds": seconds}


def hash_native_state(model, optimizer, *, step, chunk_bytes=DEFAULT_CHUNK_BYTES):
    """Hash complete native in-memory state without writing any checkpoint.

    Logical model/optimizer hashes are identical to those produced by save,
    independent of chunk size. This reads every tensor byte, including frozen
    model parameters and native optimizer moments, with bounded host staging.
    Callers must serialize this operation with state mutation and synchronize
    any streams that modify the inspected state.
    """
    began = time.monotonic()
    step = _integer(step, "step")
    chunk_bytes = _integer(chunk_bytes, "chunk_bytes", 8)
    structure, tensors = _catalog(model, optimizer, step)
    before = _signature(tensors)
    entries, max_chunk = {}, 0
    for name, (tensor, _role) in tensors.items():
        digest = hashlib.sha256()
        for _start, _host, raw in _host_parts(tensor, chunk_bytes, nonnegative=name.endswith((".exp_avg_sq", ".max_exp_avg_sq"))):
            digest.update(raw)
            max_chunk = max(max_chunk, len(raw))
        entries[name] = {"dtype": str(tensor.dtype).removeprefix("torch."),
                         "shape": list(tensor.shape),
                         "nbytes": tensor.numel() * tensor.element_size(),
                         "sha256": digest.hexdigest()}
    current_structure, current_tensors = _catalog(model, optimizer, step)
    if structure != current_structure or before != _signature(tensors) or before != _signature(current_tensors):
        raise RuntimeError("Model/optimizer changed during native state hashing")
    model_sha, optimizer_sha = _state_digests({**structure, "step": step, "tensors": entries})
    return {"format": FORMAT, "precision_contract": PRECISION, "operation": "hash",
            "step": step, "model_sha256": model_sha, "optimizer_sha256": optimizer_sha,
            "state_sha256": _digest({"step": step, "model_sha256": model_sha,
                                      "optimizer_sha256": optimizer_sha, "precision_contract": PRECISION}),
            "payload_bytes": sum(entry["nbytes"] for entry in entries.values()),
            "tensor_count": len(entries), "chunk_bytes": chunk_bytes,
            "max_host_chunk_bytes": max_chunk,
            "named_model_parameters": len(structure["model_parameters"]),
            "unique_model_parameters": len({r["canonical"] for r in structure["model_parameters"]}),
            "optimizer_parameter_states": len(structure["optimizer_states"]),
            "source_values_finite_verified": True, "native_state_hashed": True,
            "tensors": entries, "seconds": time.monotonic() - began}


def save_native_state(model, optimizer, directory, *, step, chunk_bytes=DEFAULT_CHUNK_BYTES,
                      metadata=None, durable=True):
    """Write bounded host chunks, then atomically publish manifest.json last.

    A new directory is required. An interrupted save leaves no valid manifest;
    an existing checkpoint is never overwritten. Source mutation during the
    snapshot is rejected, and caller serialization remains mandatory.
    """
    began = time.monotonic()
    step = _integer(step, "step")
    chunk_bytes = _integer(chunk_bytes, "chunk_bytes", 8)
    metadata = json.loads(_json({} if metadata is None else metadata))
    if not isinstance(metadata, dict):
        raise ValueError("Native checkpoint metadata must be an object")
    structure, tensors = _catalog(model, optimizer, step)
    before = _signature(tensors)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    temporary = directory / ".state.bin.tmp"
    entries, payload_digest = {}, hashlib.sha256()
    try:
        with temporary.open("xb") as stream:
            for name, (tensor, role) in tensors.items():
                entry = {"dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape),
                         "device_role": role, "offset": stream.tell(), "nbytes": tensor.numel() * tensor.element_size(),
                         "chunks": []}
                leaf = hashlib.sha256()
                for _start, _host, raw in _host_parts(tensor, chunk_bytes, nonnegative=name.endswith((".exp_avg_sq", ".max_exp_avg_sq"))):
                    entry["chunks"].append({"offset": stream.tell(), "nbytes": len(raw),
                                            "sha256": hashlib.sha256(raw).hexdigest()})
                    stream.write(raw)
                    leaf.update(raw)
                    payload_digest.update(raw)
                entry["sha256"] = leaf.hexdigest()
                entries[name] = entry
            payload_bytes = stream.tell()
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        current_structure, current_tensors = _catalog(model, optimizer, step)
        if structure != current_structure or before != _signature(tensors) or before != _signature(current_tensors):
            raise RuntimeError("Model/optimizer changed during native checkpoint capture")
        os.replace(temporary, directory / "state.bin")
        manifest = {"format": FORMAT, "precision_contract": PRECISION, "step": step,
                    "chunk_bytes": chunk_bytes, "metadata": metadata, **structure, "tensors": entries,
                    "payload_bytes": payload_bytes, "payload_sha256": payload_digest.hexdigest()}
        manifest["model_sha256"], manifest["optimizer_sha256"] = _state_digests(manifest)
        manifest["manifest_sha256"] = _digest(manifest)
        _publish_json(directory / "manifest.json", manifest, durable=durable)
        if durable:
            _fsync_directory(directory.parent)
        result = _report(directory, manifest, operation="save", seconds=time.monotonic() - began)
        result.update(source_values_finite_verified=True, durable=durable,
                      max_host_chunk_bytes=max((c["nbytes"] for e in entries.values() for c in e["chunks"]), default=0))
        return result
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest(directory, expected_step, expected_metadata):
    directory = Path(directory).resolve()
    path = directory / "manifest.json"
    if path.is_symlink() or (directory / "state.bin").is_symlink():
        raise ValueError("Checkpoint manifest/payload must be owned files, not symbolic links")
    def invalid_constant(value):
        raise ValueError(f"Nonfinite checkpoint JSON constant: {value}")
    manifest = json.loads(path.read_text(), parse_constant=invalid_constant)
    if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT
            or manifest.get("precision_contract") != PRECISION):
        raise ValueError("Unsupported checkpoint format or native precision contract")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _digest(body):
        raise ValueError("Native checkpoint manifest SHA256 differs")
    _integer(manifest.get("step"), "checkpoint step")
    if expected_step is not None and manifest["step"] != _integer(expected_step, "expected_step"):
        raise ValueError("Native checkpoint step differs from expected committed step")
    if not isinstance(manifest.get("metadata"), dict):
        raise ValueError("Native checkpoint metadata must be an object")
    if expected_metadata is not None:
        if not isinstance(expected_metadata, dict) or any(manifest["metadata"].get(k) != v for k, v in expected_metadata.items()):
            raise ValueError("Native checkpoint metadata/input identity differs")
    _integer(manifest.get("chunk_bytes"), "checkpoint chunk_bytes", 8)
    _integer(manifest.get("payload_bytes"), "payload_bytes")
    if not isinstance(manifest.get("tensors"), dict) or not manifest["tensors"]:
        raise ValueError("Native checkpoint tensor catalog is missing")
    return directory, manifest


def verify_native_state(directory, *, expected_step=None, expected_metadata=None):
    """Verify every payload byte in one bounded pass; does not import Torch."""
    began = time.monotonic()
    directory, manifest = _read_manifest(directory, expected_step, expected_metadata)
    payload = directory / "state.bin"
    if not payload.is_file() or payload.stat().st_size != manifest["payload_bytes"]:
        raise ValueError("Native checkpoint payload is missing, truncated or has extra bytes")
    cursor, whole = 0, hashlib.sha256()
    try:
        ordered = sorted(manifest["tensors"].items(), key=lambda item: (item[1]["offset"], item[1]["nbytes"] != 0))
        with payload.open("rb") as stream:
            for name, entry in ordered:
                shape, dtype = entry["shape"], entry["dtype"]
                if (not isinstance(name, str) or dtype not in WIDTHS or not isinstance(shape, list)
                        or any(type(x) is not int or x < 0 for x in shape)
                        or entry["nbytes"] != math.prod(shape) * WIDTHS[dtype]
                        or type(entry["offset"]) is not int or entry["offset"] != cursor
                        or entry["device_role"] not in ("cpu", "parameter_device")
                        or not _SHA.fullmatch(entry["sha256"])):
                    raise ValueError("Native checkpoint tensor layout differs")
                leaf, consumed = hashlib.sha256(), 0
                for chunk in entry["chunks"]:
                    count = chunk["nbytes"]
                    if (type(count) is not int or not 0 < count <= manifest["chunk_bytes"]
                            or count % WIDTHS[dtype] or chunk["offset"] != cursor
                            or count > entry["nbytes"] - consumed):
                        raise ValueError("Native checkpoint chunk layout differs")
                    raw = stream.read(count)
                    if len(raw) != count or hashlib.sha256(raw).hexdigest() != chunk["sha256"]:
                        raise ValueError("Native checkpoint chunk SHA256 differs or payload is truncated")
                    leaf.update(raw)
                    whole.update(raw)
                    cursor += count
                    consumed += count
                if consumed != entry["nbytes"] or leaf.hexdigest() != entry["sha256"]:
                    raise ValueError("Native checkpoint tensor SHA256/length differs")
        if cursor != manifest["payload_bytes"] or whole.hexdigest() != manifest["payload_sha256"]:
            raise ValueError("Native checkpoint complete payload SHA256/length differs")
        model_sha, optimizer_sha = _state_digests(manifest)
        if (model_sha, optimizer_sha) != (manifest["model_sha256"], manifest["optimizer_sha256"]):
            raise ValueError("Native model/optimizer logical state SHA256 differs")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed native checkpoint layout or metadata") from error
    return _report(directory, manifest, operation="verify", seconds=time.monotonic() - began)


def _decode(value, targets):
    if isinstance(value, dict):
        if set(value) == {_REF}:
            return targets[value[_REF]]
        if set(value) == {_TUPLE}:
            return tuple(_decode(child, targets) for child in value[_TUPLE])
        return {key: _decode(child, targets) for key, child in value.items()}
    if isinstance(value, list):
        return [_decode(child, targets) for child in value]
    return value


def _references(value):
    if isinstance(value, dict):
        if set(value) == {_REF}:
            yield value[_REF]
        else:
            for child in value.values():
                yield from _references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _references(child)


def load_native_state(model, optimizer, directory, *, expected_step, expected_metadata=None):
    """Fully verify, restore by copy, then byte-verify the actual restored state.

    Existing matching optimizer tensors are reused. Every model Parameter object
    and data pointer stays unchanged. A failed copy poisons the validity marker;
    only a later complete successful load sets it true again.
    """
    import torch

    began = time.monotonic()
    verified = verify_native_state(directory, expected_step=expected_step, expected_metadata=expected_metadata)
    directory, manifest = _read_manifest(directory, expected_step, expected_metadata)
    if manifest["manifest_sha256"] != verified["manifest_sha256"]:
        raise ValueError("Checkpoint manifest changed between verification and restore")
    params, buffers, canonical, parameter_table, buffer_table = _model_layout(model)
    names = _groups(optimizer, canonical)
    if (parameter_table != manifest["model_parameters"] or buffer_table != manifest["model_buffers"]
            or names != [group["parameters"] for group in manifest["optimizer_groups"]]):
        raise ValueError("Target model aliases/frozen flags or optimizer parameter-name order differs")
    pointers = {name: (id(value), value.data_ptr()) for name, value in params.items()}
    entries, targets = manifest["tensors"], {}
    for name, value in params.items():
        targets["model.parameter." + name] = value
    for name, value in buffers.items():
        targets["model.buffer." + name] = value
    first = next(parameter for group in optimizer.param_groups for parameter in group["params"])

    def target(name, anchor, existing=None):
        entry = entries[name]
        dtype = getattr(torch, entry["dtype"])
        device = torch.device("cpu") if entry["device_role"] == "cpu" else anchor.device
        if existing is not None:
            if (not torch.is_tensor(existing) or existing.dtype != dtype
                    or list(existing.shape) != entry["shape"] or existing.device != device
                    or not existing.is_contiguous()):
                raise ValueError("Existing target optimizer tensor has incompatible native shape/dtype/device")
            return existing
        return torch.empty(entry["shape"], dtype=dtype, device=device)

    for parameter_name, fields in manifest["optimizer_states"].items():
        if parameter_name not in params:
            raise ValueError("Checkpoint optimizer parameter is absent from target model")
        parameter = params[parameter_name]
        for field, value in fields.items():
            if not isinstance(value, dict) or set(value) != {_REF}:
                raise ValueError("Native optimizer state must contain exact tensor references")
            name = value[_REF]
            targets[name] = target(name, parameter, optimizer.state.get(parameter, {}).get(field))
    for key in ("optimizer_defaults", "optimizer_groups"):
        for name in _references(manifest[key]):
            if name not in targets:
                targets[name] = target(name, first)
    if set(targets) != set(entries):
        raise ValueError("Checkpoint contains unbound/missing model or optimizer tensors")
    for name, value in targets.items():
        if list(value.shape) != entries[name]["shape"] or str(value.dtype).removeprefix("torch.") != entries[name]["dtype"]:
            raise ValueError("Target native model tensor dtype/shape differs")
    optimizer._jade_resident_state_valid = False
    with (directory / "state.bin").open("rb") as stream, torch.no_grad():
        for name, entry in sorted(entries.items(), key=lambda item: (item[1]["offset"], item[1]["nbytes"] != 0)):
            flat = targets[name].reshape(-1)
            offset = 0
            for chunk in entry["chunks"]:
                raw = bytearray(stream.read(chunk["nbytes"]))
                if len(raw) != chunk["nbytes"] or hashlib.sha256(raw).hexdigest() != chunk["sha256"]:
                    raise ValueError("Checkpoint changed between verification and restore")
                source = torch.frombuffer(raw, dtype=getattr(torch, entry["dtype"]))
                flat[offset:offset + source.numel()].copy_(source)
                offset += source.numel()
            digest = hashlib.sha256()
            for _start, _host, raw in _host_parts(targets[name], manifest["chunk_bytes"], nonnegative=name.endswith((".exp_avg_sq", ".max_exp_avg_sq"))):
                digest.update(raw)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError("Restored native tensor bytes differ")
    state = defaultdict(dict)
    for name, fields in manifest["optimizer_states"].items():
        state[params[name]] = _decode(fields, targets)
    optimizer.state = state
    optimizer.defaults = _decode(manifest["optimizer_defaults"], targets)
    for target_group, saved_group in zip(optimizer.param_groups, manifest["optimizer_groups"], strict=True):
        parameters = target_group["params"]
        target_group.clear()
        target_group.update(_decode(saved_group["options"], targets), params=parameters)
    restored, _tensors = _catalog(model, optimizer, manifest["step"])
    if any(restored[key] != manifest[key] for key in restored):
        raise ValueError("Restored native metadata differs from checkpoint")
    if pointers != {name: (id(value), value.data_ptr()) for name, value in params.items()}:
        raise RuntimeError("Native restore replaced an existing model Parameter")
    optimizer._jade_resident_state_valid = True
    result = _report(directory, manifest, operation="load", seconds=time.monotonic() - began)
    result.update(parameter_pointers_preserved=True, restored_values_finite_verified=True,
                  input_identity_verified=expected_metadata is not None,
                  max_host_chunk_bytes=max((c["nbytes"] for e in entries.values() for c in e["chunks"]), default=0))
    return result
