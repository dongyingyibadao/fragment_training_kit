"""Complete native AdamW state transfer through bounded device broadcasts.

This module intentionally couples to resident_state's private catalog/schema
helpers. Both modules must be frozen together. It never imports torch_npu or
serializes device state through pickle/disk. The caller distributes the small
JSON blueprint, creates the new communication group, and serializes all state
mutation with this operation. Source and healthy participants use scratch
buffers only; only replacement participants copy into resident state.

Successful CPU/Gloo tests do not establish HCCL/NPU recovery support. Device
collective/copy errors poison replacement validity until a complete transfer
succeeds. Callers must keep an unsuccessful replacement outside membership.
"""

from __future__ import annotations

import base64
from collections import defaultdict
import hashlib
import json
import math
import time

from . import resident_state as native


FORMAT = "jade-native-resident-broadcast-v1"
DEFAULT_INLINE_BYTES = 64 * 1024


def _metadata(value):
    value = json.loads(native._json(value))
    if not isinstance(value, dict) or not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise ValueError("Native state transfer metadata requires a nonempty run_id")
    return value


def _inline_allowed(name, tensor):
    return tensor.device.type == "cpu" and (
        name.endswith(".step") or name.startswith(("optimizer.defaults.", "optimizer.group.")))


def create_native_state_blueprint(model, optimizer, *, step, metadata, source_rank=0,
                                  chunk_bytes=native.DEFAULT_CHUNK_BYTES,
                                  max_inline_bytes=DEFAULT_INLINE_BYTES):
    """Describe complete state, inlining bounded CPU step/hyperparameter bytes.

    No model or moment bytes are copied to the host. Model/optimizer mutations
    must remain serialized from this boundary through the broadcast operation.
    All native tensor data travels in device chunks except the small explicitly
    inlined CPU optimizer tensors. The result is ordinary JSON-compatible data.
    """
    import torch

    step = native._integer(step, "step")
    source_rank = native._integer(source_rank, "source_rank")
    chunk_bytes = native._integer(chunk_bytes, "chunk_bytes", 8)
    max_inline_bytes = native._integer(max_inline_bytes, "max_inline_bytes")
    metadata = _metadata(metadata)
    structure, tensors = native._catalog(model, optimizer, step)
    before = native._signature(tensors)
    entries, inline_bytes = {}, 0
    for name, (tensor, role) in sorted(tensors.items()):
        entry = {"dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape),
                 "device_role": role, "nbytes": tensor.numel() * tensor.element_size()}
        if _inline_allowed(name, tensor):
            inline_bytes += entry["nbytes"]
            if inline_bytes > max_inline_bytes:
                raise ValueError("CPU step/hyperparameter bytes exceed the bounded blueprint budget")
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            entry.update(transfer="inline", bytes_base64=base64.b64encode(raw).decode(),
                         sha256=hashlib.sha256(raw).hexdigest())
        else:
            entry["transfer"] = "broadcast"
        entries[name] = entry
    current_structure, current_tensors = native._catalog(model, optimizer, step)
    if current_structure != structure or before != native._signature(tensors) or before != native._signature(current_tensors):
        raise RuntimeError("Native source changed during blueprint creation")
    blueprint = {"format": FORMAT, "native_state_format": native.FORMAT,
                 "precision_contract": native.PRECISION, "step": step, "source_rank": source_rank,
                 "metadata": metadata, "chunk_bytes": chunk_bytes,
                 "max_inline_bytes": max_inline_bytes, "inline_bytes": inline_bytes,
                 **structure, "tensors": entries}
    blueprint["blueprint_sha256"] = native._digest(blueprint)
    return blueprint


def _validate_blueprint(blueprint, *, expected_step, expected_metadata, source_rank):
    if (not isinstance(blueprint, dict) or blueprint.get("format") != FORMAT
            or blueprint.get("native_state_format") != native.FORMAT
            or blueprint.get("precision_contract") != native.PRECISION):
        raise ValueError("Unsupported native state transfer format/precision")
    if blueprint.get("blueprint_sha256") != native._digest({k: v for k, v in blueprint.items() if k != "blueprint_sha256"}):
        raise ValueError("Native transfer blueprint SHA256 differs")
    if native._integer(blueprint.get("step"), "blueprint step") != native._integer(expected_step, "expected_step"):
        raise ValueError("Native transfer boundary step differs")
    if native._integer(blueprint.get("source_rank"), "blueprint source_rank") != source_rank:
        raise ValueError("Native transfer source rank differs")
    metadata, expected = _metadata(blueprint.get("metadata")), _metadata(expected_metadata)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Native transfer run/input metadata differs")
    native._integer(blueprint.get("chunk_bytes"), "chunk_bytes", 8)
    limit = native._integer(blueprint.get("max_inline_bytes"), "max_inline_bytes")
    inline = 0
    if not isinstance(blueprint.get("tensors"), dict) or not blueprint["tensors"]:
        raise ValueError("Native transfer tensor catalog is missing")
    for name, entry in blueprint["tensors"].items():
        shape, dtype = entry["shape"], entry["dtype"]
        if (not isinstance(name, str) or dtype not in native.WIDTHS or not isinstance(shape, list)
                or any(type(size) is not int or size < 0 for size in shape)
                or type(entry["nbytes"]) is not int
                or entry["nbytes"] != math.prod(shape) * native.WIDTHS[dtype]
                or entry["device_role"] not in ("cpu", "parameter_device")):
            raise ValueError("Native transfer dtype/shape/device layout differs")
        if entry["transfer"] == "inline":
            if (entry["device_role"] != "cpu" or not (
                    name.endswith(".step") or name.startswith(("optimizer.defaults.", "optimizer.group.")))):
                raise ValueError("Only CPU optimizer steps/hyperparameters may be inlined")
            raw = base64.b64decode(entry["bytes_base64"], validate=True)
            inline += len(raw)
            if len(raw) != entry["nbytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("Native transfer inline bytes differ")
        elif entry["transfer"] != "broadcast":
            raise ValueError("Unknown native tensor transfer mode")
    if inline != blueprint.get("inline_bytes") or inline > limit:
        raise ValueError("Native transfer inline byte budget differs")


def _structure(blueprint):
    return {key: blueprint[key] for key in (
        "model_parameters", "model_buffers", "optimizer_defaults", "optimizer_groups", "optimizer_states")}


def _schema_matches(tensor, entry):
    return (str(tensor.dtype).removeprefix("torch.") == entry["dtype"]
            and list(tensor.shape) == entry["shape"] and tensor.is_contiguous())


def _bind_replacement(model, optimizer, blueprint, torch):
    params, buffers, canonical, parameters_table, buffers_table = native._model_layout(model)
    names = native._groups(optimizer, canonical)
    if (parameters_table != blueprint["model_parameters"] or buffers_table != blueprint["model_buffers"]
            or names != [group["parameters"] for group in blueprint["optimizer_groups"]]):
        raise ValueError("Replacement aliases/frozen flags/optimizer parameter-name order differs")
    entries = blueprint["tensors"]
    targets = {"model.parameter." + name: value for name, value in params.items()}
    targets.update({"model.buffer." + name: value for name, value in buffers.items()})
    for name, value in targets.items():
        if name not in entries or not _schema_matches(value, entries[name]):
            raise ValueError("Replacement native model dtype/shape differs")
    first = next(parameter for group in optimizer.param_groups for parameter in group["params"])

    def target(name, anchor, existing=None):
        entry = entries[name]
        device = torch.device("cpu") if entry["device_role"] == "cpu" else anchor.device
        if existing is not None:
            if not _schema_matches(existing, entry) or existing.device != device:
                raise ValueError("Replacement existing moment dtype/shape/device differs")
            return existing
        return torch.empty(entry["shape"], dtype=getattr(torch, entry["dtype"]), device=device)

    expected_names = {name for group in names for name in group}
    if set(blueprint["optimizer_states"]) != expected_names:
        raise ValueError("Replacement native optimizer parameter state coverage differs")
    for saved_group in blueprint["optimizer_groups"]:
        amsgrad = saved_group["options"].get("amsgrad", False)
        for parameter_name in saved_group["parameters"]:
            parameter = params[parameter_name]
            fields = blueprint["optimizer_states"][parameter_name]
            required = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if amsgrad else set())
            if fields and set(fields) != required:
                raise ValueError("Replacement native AdamW state is incomplete")
            if not fields and parameter.requires_grad and blueprint["step"]:
                raise ValueError("Replacement native AdamW active state is missing")
            for field, reference in fields.items():
                if not isinstance(reference, dict) or set(reference) != {native._REF}:
                    raise ValueError("Replacement native optimizer tensor reference differs")
                name = reference[native._REF]
                if name != f"optimizer.state.{parameter_name}.{field}":
                    raise ValueError("Replacement optimizer field is bound to an unrelated tensor")
                entry = entries[name]
                if field != "step" and (entry["dtype"] != str(parameter.dtype).removeprefix("torch.")
                                         or entry["shape"] != list(parameter.shape)
                                         or (entry["device_role"] == "cpu" and parameter.device.type != "cpu")):
                    raise ValueError("Replacement native moment differs from parameter")
                if field == "step" and math.prod(entry["shape"]) != 1:
                    raise ValueError("Replacement native optimizer step is not scalar")
                targets[name] = target(name, parameter, optimizer.state.get(parameter, {}).get(field))
    for key in ("optimizer_defaults", "optimizer_groups"):
        for name in native._references(blueprint[key]):
            if name not in targets:
                targets[name] = target(name, first)
    if set(targets) != set(entries):
        raise ValueError("Replacement native tensor coverage differs")
    return params, targets


def _bind_existing(model, optimizer, blueprint):
    import torch

    structure, tensors = native._catalog(model, optimizer, blueprint["step"])
    if structure != _structure(blueprint) or set(tensors) != set(blueprint["tensors"]):
        raise ValueError("Existing complete native state schema/metadata differs")
    for name, (value, role) in tensors.items():
        entry = blueprint["tensors"][name]
        if role != entry["device_role"] or not _schema_matches(value, entry):
            raise ValueError("Existing native tensor dtype/shape/device-role differs")
        if entry["transfer"] == "inline":
            raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("Existing CPU optimizer step/hyperparameter bytes differ")
    return tensors


def _chunks(blueprint):
    """Pack small leaves together; bound scratch by one native chunk per rank."""
    by_dtype = defaultdict(list)
    for name, entry in sorted(blueprint["tensors"].items()):
        if entry["transfer"] == "broadcast" and entry["nbytes"]:
            by_dtype[entry["dtype"]].append((name, entry))
    for dtype, entries in sorted(by_dtype.items()):
        capacity = blueprint["chunk_bytes"] // native.WIDTHS[dtype]
        segments, used = [], 0
        for name, entry in entries:
            offset, total = 0, math.prod(entry["shape"])
            while offset < total:
                count = min(total - offset, capacity - used)
                segments.append((name, offset, count, used))
                used += count
                offset += count
                if used == capacity:
                    yield dtype, used, segments
                    segments, used = [], 0
        if used:
            yield dtype, used, segments


def _sync(torch, device):
    if device.type != "cpu":
        getattr(torch, device.type).synchronize(device)


def _agree(dist, torch, device, group, success, message):
    ready = torch.tensor([int(success)], dtype=torch.int32, device=device)
    dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group)
    if not bool(ready.item()):
        raise ValueError(message)


def _source_digest(dist, torch, device, group, source_rank, digest):
    rank = dist.get_rank()
    values = [ord(character) for character in digest] if rank == source_rank else [0] * 64
    wire = torch.tensor(values, dtype=torch.int32, device=device)
    dist.broadcast(wire, src=source_rank, group=group)
    return "".join(chr(value) for value in wire.cpu().tolist())


def broadcast_native_state(model, optimizer, blueprint, *, source_rank, role,
                           expected_step, expected_metadata, transport_device,
                           group=None, verify=True):
    """Broadcast one complete boundary; every group member must call this API.

    Roles are source, healthy and replacement. The source rank is a global
    distributed rank, matching torch.distributed.broadcast's src convention.
    Healthy state is never overwritten, including optimizer metadata. The
    receiving model's Parameter objects and storage pointers are preserved.
    With verify=True every rank hashes its complete actual state and compares
    the result with the source; this expensive pass has separate timing.
    Without it the report states transport completion, not byte verification.
    """
    import torch
    import torch.distributed as dist

    began = time.monotonic()
    device = torch.device(transport_device)
    if not dist.is_initialized():
        raise ValueError("Native transfer requires an initialized distributed group")
    source_rank = native._integer(source_rank, "source_rank")
    replacement = role == "replacement"
    if replacement:
        optimizer._jade_resident_state_valid = False
    preflight_error, tensors, params, targets, signature = None, None, None, None, None
    try:
        if role not in ("source", "healthy", "replacement") or (dist.get_rank() == source_rank) != (role == "source"):
            raise ValueError("Native transfer source/healthy/replacement role differs from global rank")
        _validate_blueprint(blueprint, expected_step=expected_step,
                            expected_metadata=expected_metadata, source_rank=source_rank)
        _sync(torch, device)
        if replacement:
            params, targets = _bind_replacement(model, optimizer, blueprint, torch)
        else:
            tensors = _bind_existing(model, optimizer, blueprint)
            signature = native._signature(tensors)
        pointers = {name: (id(value), value.data_ptr()) for name, value in model.named_parameters()}
    except Exception as error:
        preflight_error = error
    # Every participant takes the same bounded control collectives even when
    # one local schema fails, before any resident state can be overwritten.
    local_digest = blueprint.get("blueprint_sha256", "0" * 64) if isinstance(blueprint, dict) else "0" * 64
    if not isinstance(local_digest, str) or not native._SHA.fullmatch(local_digest):
        local_digest = "0" * 64
    plan_digest = native._digest({"blueprint_sha256": local_digest, "verify": bool(verify)})
    source_digest = _source_digest(dist, torch, device, group, source_rank, plan_digest)
    _agree(dist, torch, device, group, preflight_error is None and plan_digest == source_digest,
           "Native transfer preflight failed on one or more ranks: " + str(preflight_error))
    preflight_seconds = time.monotonic() - began
    transfer_began, count, transported_bytes, max_scratch = time.monotonic(), 0, 0, 0
    with torch.no_grad():
        if replacement:
            for name, entry in blueprint["tensors"].items():
                if entry["transfer"] == "inline" and entry["nbytes"]:
                    raw = bytearray(base64.b64decode(entry["bytes_base64"], validate=True))
                    targets[name].reshape(-1).copy_(torch.frombuffer(raw, dtype=getattr(torch, entry["dtype"])))
        backing, backing_dtype = None, None
        for dtype, elements, segments in _chunks(blueprint):
            if dtype != backing_dtype:
                # Reuse one backing for every chunk of a dtype, rather than
                # enqueueing fresh allocations for the entire payload. Finish
                # the previous dtype before releasing its storage.
                if backing is not None:
                    _sync(torch, device)
                    del backing
                backing = torch.empty(elements, dtype=getattr(torch, dtype), device=device)
                backing_dtype = dtype
            scratch = backing[:elements]
            if role == "source":
                for name, offset, size, destination in segments:
                    scratch[destination:destination + size].copy_(tensors[name][0].reshape(-1)[offset:offset + size])
            dist.broadcast(scratch, src=source_rank, group=group, async_op=False)
            if replacement:
                for name, offset, size, origin in segments:
                    targets[name].reshape(-1)[offset:offset + size].copy_(scratch[origin:origin + size])
            nbytes = elements * native.WIDTHS[dtype]
            transported_bytes += nbytes
            max_scratch = max(max_scratch, nbytes)
            count += 1
            del scratch
        _sync(torch, device)
        if backing is not None:
            del backing
    broadcast_seconds = time.monotonic() - transfer_began
    finalize_began, finalize_error = time.monotonic(), None
    try:
        if replacement:
            optimizer.state = defaultdict(dict, {params[name]: native._decode(fields, targets)
                                                 for name, fields in blueprint["optimizer_states"].items()})
            optimizer.defaults = native._decode(blueprint["optimizer_defaults"], targets)
            for target_group, saved_group in zip(optimizer.param_groups, blueprint["optimizer_groups"], strict=True):
                parameters = target_group["params"]
                target_group.clear()
                target_group.update(native._decode(saved_group["options"], targets), params=parameters)
            _bind_existing(model, optimizer, blueprint)
        elif signature != native._signature(tensors):
            raise RuntimeError("Source/healthy native state mutated during broadcast")
        if pointers != {name: (id(value), value.data_ptr()) for name, value in model.named_parameters()}:
            raise RuntimeError("Native broadcast replaced a model Parameter or its storage")
    except Exception as error:
        finalize_error = error
    _agree(dist, torch, device, group, finalize_error is None,
           "Native transfer finalization failed on one or more ranks: " + str(finalize_error))
    finalize_seconds = time.monotonic() - finalize_began
    verify_began, state_report = time.monotonic(), None
    if verify:
        verify_error = None
        try:
            state_report = native.hash_native_state(model, optimizer, step=expected_step,
                                                    chunk_bytes=blueprint["chunk_bytes"])
        except Exception as error:
            verify_error = error
        local_state = state_report["state_sha256"] if state_report is not None else "0" * 64
        expected_state = _source_digest(dist, torch, device, group, source_rank, local_state)
        _agree(dist, torch, device, group, verify_error is None and local_state == expected_state,
               "Complete native state verification failed on one or more ranks: " + str(verify_error))
    verification_seconds = time.monotonic() - verify_began if verify else 0.0
    if replacement:
        optimizer._jade_resident_state_valid = True
    result = {"format": FORMAT, "precision_contract": native.PRECISION, "step": expected_step,
              "blueprint_sha256": blueprint["blueprint_sha256"], "source_rank": source_rank, "role": role,
              "transport_device": str(device), "distributed_backend": str(dist.get_backend(group)),
              "tensor_count": len(blueprint["tensors"]),
              "native_payload_bytes": sum(entry["nbytes"] for entry in blueprint["tensors"].values()),
              "inline_bytes": blueprint["inline_bytes"], "broadcast_bytes": transported_bytes,
              "broadcast_chunks": count, "max_payload_scratch_bytes": max_scratch,
              "max_device_control_tensor_bytes": 256,
              "max_device_scratch_bytes": max(max_scratch, 256),
              "scratch_scope": "Largest explicit payload/control tensor; excludes resident state, allocator cache and backend workspaces",
              "preflight_seconds": preflight_seconds, "broadcast_seconds": broadcast_seconds,
              "finalize_seconds": finalize_seconds, "verification_seconds": verification_seconds,
              "total_seconds": time.monotonic() - began, "transport_complete": True,
              "verified_loaded_state": bool(verify and replacement),
              "full_native_state_hash_verified": bool(verify), "parameter_pointers_preserved": True,
              "existing_state_not_overwritten": role != "replacement", "npu_hccl_validated": False}
    if state_report is not None:
        result["native_state"] = state_report
    return result
