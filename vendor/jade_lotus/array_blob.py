"""Immutable, checksummed array blobs and bounded FP32 shared-file reduction.

Each publication is a directory containing ``payload.bin`` and a canonical
``manifest.json``. Arrays follow sorted leaf names and C order in little-endian
FP32. The default v1 format stores FP32. Explicit gradient codecs store exact
BF16 values: v2 uses BF16 throughout, v3 retains vision leaves in FP32, and v4
uses a complete frozen dtype table. None rounds. Shards address fixed logical
FP32 ranges and may cross leaf boundaries. Mixed formats record the matching
variable-size wire ranges. Writers hash the actual stored bytes, then atomically
rename a complete directory.

This module does not decide whether a gradient belongs to a live model version,
deduplicate worker attempts, or delete inputs. The caller must fence those
decisions and retain source blobs until the complete update is acknowledged.
Published files must remain immutable while they are being read or mapped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

import numpy as np


FORMAT = "jade-array-blob-v1"
BF16_EXACT_FORMAT = "jade-array-blob-bfloat16-exact-v2"
BF16_EXACT_CODEC = "fp32-bfloat16-bit-exact-v1"
BF16_EXACT_WIRE = "bfloat16_exact"
MIXED_BF16_EXACT_WIRE = "bfloat16_exact_except_vision"
MIXED_BF16_EXACT_FORMAT = "jade-array-blob-bfloat16-exact-except-vision-v3"
MIXED_BF16_EXACT_CODEC = "fp32-bfloat16-exact-except-vision-v1"
NATIVE_MIXED_EXACT_WIRE = "native_mixed_exact"
NATIVE_MIXED_EXACT_FORMAT = "jade-array-blob-native-mixed-exact-v4"
NATIVE_MIXED_EXACT_CODEC = "fp32-native-mixed-bit-exact-v1"
NATIVE_WIRE_POLICY_FORMAT = "jade-native-gradient-wire-policy-v1"
VISION_PREFIX = "PaliGemma/img/"
VISION_WIRE_POLICY = {"vision_prefix": VISION_PREFIX, "vision_wire_dtype": "float32", "other_wire_dtype": "bfloat16"}
LOGICAL_CHUNK_PARTITION = "logical-fp32-ranges-v1"
MIXED_FORMATS = (MIXED_BF16_EXACT_FORMAT, NATIVE_MIXED_EXACT_FORMAT)
ENCODED_FORMATS = (BF16_EXACT_FORMAT, *MIXED_FORMATS)
WIRE_DTYPES = ("float32", BF16_EXACT_WIRE, MIXED_BF16_EXACT_WIRE, NATIVE_MIXED_EXACT_WIRE)
DEFAULT_SHARD_BYTES = 256 * 1024 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024 * 1024
MAX_CHUNK_BYTES = 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_LEAVES = MAX_CHUNKS = 65536
IO_BLOCK_BYTES = 8 * 1024 * 1024
FP32 = np.dtype("<f4")
BF16_WORD = np.dtype("<u2")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_REDUCTION = "ordered-normalized-weighted-mean-fp32-v1"
CHUNK_TREE_HASH = "sha256_chunk_tree_v1"


def _chunk_tree_digest(layout, chunks):
    return _digest({"algorithm": CHUNK_TREE_HASH, "payload_bytes": layout["payload_bytes"],
                    "chunk_bytes": layout["chunk_bytes"], "chunks": chunks})


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}: expected integer in {minimum}..{maximum}")
    return value


def _sha(value, name):
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError(f"Invalid {name}: expected lowercase SHA256")
    return value


def _shard_bytes(value):
    _integer(value, "chunk_bytes", 4, MAX_CHUNK_BYTES)
    if value % 4:
        raise ValueError("chunk_bytes must align to FP32 elements")
    return value


def _schema(leaves):
    return {leaf["key"]: {"shape": leaf["shape"], "dtype": leaf["dtype"]} for leaf in leaves}


def _validate_layout(layout):
    if not isinstance(layout, dict) or set(layout) != {"payload_bytes", "chunk_bytes", "schema_sha256", "leaves"}:
        raise ValueError("Invalid array layout fields")
    total = _integer(layout["payload_bytes"], "payload_bytes", 4, MAX_PAYLOAD_BYTES)
    shard = _shard_bytes(layout["chunk_bytes"])
    if total % 4 or math.ceil(total / shard) > MAX_CHUNKS:
        raise ValueError("Invalid payload alignment or excessive chunk count")
    leaves = layout["leaves"]
    if not isinstance(leaves, list) or not 1 <= len(leaves) <= MAX_LEAVES:
        raise ValueError("Invalid array leaf count")
    offset, previous = 0, None
    for leaf in leaves:
        if not isinstance(leaf, dict) or set(leaf) != {"key", "shape", "dtype", "offset", "nbytes"}:
            raise ValueError("Invalid array leaf fields")
        key = leaf["key"]
        if not isinstance(key, str) or not key or len(key.encode("utf-8")) > 4096:
            raise ValueError("Invalid array leaf key")
        if previous is not None and key <= previous:
            raise ValueError("Array leaf keys must be unique and sorted")
        previous = key
        shape = leaf["shape"]
        if not isinstance(shape, list) or len(shape) > 32:
            raise ValueError("Invalid array shape")
        elements = 1
        for dimension in shape:
            elements *= _integer(dimension, "shape dimension", 1, MAX_PAYLOAD_BYTES // 4)
            if elements > MAX_PAYLOAD_BYTES // 4:
                raise ValueError("Array shape exceeds payload limit")
        if leaf["dtype"] != "float32":
            raise ValueError("Array leaf dtype must be float32")
        if (_integer(leaf["offset"], "leaf offset", 0, MAX_PAYLOAD_BYTES) != offset
                or _integer(leaf["nbytes"], "leaf nbytes", 4, MAX_PAYLOAD_BYTES) != elements * 4):
            raise ValueError("Array leaf offset/shape coverage differs")
        offset += elements * 4
        if offset > total:
            raise ValueError("Array leaf coverage exceeds payload")
    if offset != total:
        raise ValueError("Array leaves do not exactly cover payload")
    if _sha(layout["schema_sha256"], "schema_sha256") != _digest(_schema(leaves)):
        raise ValueError("Array schema checksum mismatch")
    return layout


def _layout(manifest):
    # Reduction always addresses canonical FP32 element ranges, including
    # when the actual wire ranges contain two different element widths.
    if manifest.get("format") in ENCODED_FORMATS:
        return copy.deepcopy(manifest["logical_layout"])
    return {name: copy.deepcopy(manifest[name])
            for name in ("payload_bytes", "chunk_bytes", "schema_sha256", "leaves")}


def validate_native_wire_policy(policy, logical_layout=None):
    """Validate the complete, sorted native dtype table, without inferring types."""
    if (not isinstance(policy, dict)
            or set(policy) != {"format", "logical_schema_sha256", "leaves"}
            or policy.get("format") != NATIVE_WIRE_POLICY_FORMAT):
        raise ValueError("Invalid native wire policy fields or format")
    _sha(policy["logical_schema_sha256"], "native policy logical schema")
    leaves = policy["leaves"]
    if not isinstance(leaves, list) or not 1 <= len(leaves) <= MAX_LEAVES:
        raise ValueError("Invalid native wire policy leaf count")
    previous = None
    for leaf in leaves:
        if (not isinstance(leaf, dict) or set(leaf) != {"key", "dtype"}
                or not isinstance(leaf["key"], str) or not leaf["key"]
                or len(leaf["key"].encode("utf-8")) > 4096
                or leaf["dtype"] not in ("float32", "bfloat16")):
            raise ValueError("Invalid native wire policy leaf")
        if previous is not None and leaf["key"] <= previous:
            raise ValueError("Native wire policy keys must be unique and sorted")
        previous = leaf["key"]
    if logical_layout is not None:
        logical = _validate_layout(logical_layout)
        if (policy["logical_schema_sha256"] != logical["schema_sha256"]
                or [leaf["key"] for leaf in leaves] != [leaf["key"] for leaf in logical["leaves"]]):
            raise ValueError("Native wire policy must cover the exact logical schema")
    if len(_json_bytes(policy)) > MAX_MANIFEST_BYTES:
        raise ValueError("Native wire policy exceeds metadata limit")
    return copy.deepcopy(policy)


def native_wire_policy_sha256(policy):
    return _digest(validate_native_wire_policy(policy))


def make_native_wire_policy(schema, dtypes):
    """Build a frozen policy from canonical FP32 schema and explicit dtypes."""
    if (not isinstance(schema, Mapping) or not schema or len(schema) > MAX_LEAVES
            or any(not isinstance(key, str) for key in schema)
            or not isinstance(dtypes, Mapping) or set(dtypes) != set(schema)):
        raise ValueError("Native dtype table must cover the complete FP32 schema")
    leaves, offset = [], 0
    for key in sorted(schema):
        item = schema[key]
        if (not isinstance(item, Mapping) or set(item) != {"shape", "dtype"}
                or item["dtype"] != "float32" or not isinstance(item["shape"], list)
                or len(item["shape"]) > 32):
            raise ValueError("Native policy requires a canonical FP32 schema")
        size = 1
        for dimension in item["shape"]:
            size *= _integer(dimension, "shape dimension", 1, MAX_PAYLOAD_BYTES // 4)
            if size > MAX_PAYLOAD_BYTES // 4:
                raise ValueError("Native policy shape exceeds payload limit")
        leaves.append({"key": key, "shape": list(item["shape"]), "dtype": "float32",
                       "offset": offset, "nbytes": size * 4})
        offset += size * 4
    logical = {"payload_bytes": offset, "chunk_bytes": DEFAULT_SHARD_BYTES,
               "schema_sha256": _digest(_schema(leaves)), "leaves": leaves}
    return validate_native_wire_policy({"format": NATIVE_WIRE_POLICY_FORMAT,
        "logical_schema_sha256": logical["schema_sha256"],
        "leaves": [{"key": key, "dtype": dtypes[key]} for key in sorted(schema)]}, logical)


def _native_mixed_wire_layout(logical, policy):
    policy = validate_native_wire_policy(policy, logical)
    leaves, offset = [], 0
    for leaf, rule in zip(logical["leaves"], policy["leaves"]):
        nbytes = leaf["nbytes"] if rule["dtype"] == "float32" else leaf["nbytes"] // 2
        leaves.append({**leaf, "dtype": rule["dtype"], "offset": offset, "nbytes": nbytes})
        offset += nbytes
    return {"payload_bytes": offset, "chunk_bytes": logical["chunk_bytes"],
            "schema_sha256": logical["schema_sha256"], "wire_schema_sha256": _digest(_schema(leaves)),
            "leaves": leaves}


def _bf16_wire_layout(logical):
    _validate_layout(logical)
    leaves = [{**leaf, "dtype": "bfloat16", "offset": leaf["offset"] // 2, "nbytes": leaf["nbytes"] // 2}
              for leaf in logical["leaves"]]
    return {"payload_bytes": logical["payload_bytes"] // 2, "chunk_bytes": logical["chunk_bytes"] // 2,
            "schema_sha256": logical["schema_sha256"], "wire_schema_sha256": _digest(_schema(leaves)),
            "leaves": leaves}


def _mixed_wire_layout(logical):
    _validate_layout(logical)
    leaves, offset = [], 0
    for leaf in logical["leaves"]:
        vision = leaf["key"].startswith(VISION_PREFIX)
        nbytes = leaf["nbytes"] if vision else leaf["nbytes"] // 2
        leaves.append({**leaf, "dtype": "float32" if vision else "bfloat16", "offset": offset, "nbytes": nbytes})
        offset += nbytes
    return {"payload_bytes": offset, "chunk_bytes": logical["chunk_bytes"],
            "schema_sha256": logical["schema_sha256"], "wire_schema_sha256": _digest(_schema(leaves)), "leaves": leaves}


def _mixed_chunk_layout(logical, wire):
    """Map unchanged logical FP32 shard ranges to contiguous variable wire ranges."""
    def wire_offset(offset):
        if offset == logical["payload_bytes"]:
            return wire["payload_bytes"]
        for leaf, stored in zip(logical["leaves"], wire["leaves"]):
            if leaf["offset"] <= offset < leaf["offset"] + leaf["nbytes"]:
                width = 4 if stored["dtype"] == "float32" else 2
                return stored["offset"] + (offset - leaf["offset"]) // 4 * width
        raise ValueError("Logical shard boundary is outside its leaf layout")

    result = []
    for start in range(0, logical["payload_bytes"], logical["chunk_bytes"]):
        end = min(start + logical["chunk_bytes"], logical["payload_bytes"])
        left, right = wire_offset(start), wire_offset(end)
        result.append({"index": len(result), "offset": left, "nbytes": right - left,
                       "logical_offset": start, "logical_nbytes": end - start})
    return result


def encode_bfloat16_exact(values):
    """Encode finite FP32 values without rounding a single bit.

    BF16-expanded FP32 values have sixteen zero low bits, including signed
    zeros and BF16 subnormals. Inspecting those bits proves the round trip;
    arbitrary FP32 values are rejected rather than silently quantized.
    """
    values = np.asarray(values)
    if values.dtype != FP32:
        raise ValueError("BF16-exact encoding requires little-endian float32")
    flat = np.ascontiguousarray(values).reshape(-1)
    bits = flat.view("<u4")
    if not _all_finite(flat):
        raise ValueError("BF16-exact encoding rejects nonfinite values")
    if np.any(np.bitwise_and(bits, np.uint32(0xFFFF))):
        raise ValueError("FP32 gradient is not exactly representable as BF16; refusing to round")
    # Copy only the high halfword into a bounded, contiguous little-endian
    # buffer. No bfloat16 Python/JAX/ml_dtypes dependency enters this codec.
    return np.ascontiguousarray(flat.view(BF16_WORD)[1::2])


def decode_bfloat16_exact(words):
    """Expand little-endian BF16 words into owned FP32 storage exactly."""
    words = np.asarray(words)
    if words.dtype != BF16_WORD:
        raise ValueError("BF16-exact decoding requires little-endian uint16 words")
    bits = np.empty(words.shape, dtype="<u4")
    np.copyto(bits, words, casting="safe")
    np.left_shift(bits, np.uint32(16), out=bits)
    values = bits.view(FP32)
    if not _all_finite(values.reshape(-1)):
        raise ValueError("BF16-exact payload contains nonfinite values")
    return values


def _weights(weights, count):
    if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)) or len(weights) != count:
        raise ValueError("sample_weights must match the ordered blob list")
    result = []
    for value in weights:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("sample_weights must be positive finite numbers")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("sample_weights must be positive finite numbers")
        result.append(number)
    try:
        total = math.fsum(result)
    except OverflowError as error:
        raise ValueError("sample_weights total is not finite") from error
    if not math.isfinite(total) or total <= 0:
        raise ValueError("sample_weights total is not finite")
    normalized = [np.float32(value / total) for value in result]
    if any(not math.isfinite(float(value)) or value <= 0 for value in normalized):
        raise ValueError("Normalized sample weight is not representable in FP32")
    return result, normalized, total


def _validate_reduction(reduction, digest):
    if (not isinstance(reduction, dict)
            or set(reduction) not in ({"algorithm", "input_manifest_sha256", "sample_weights", "total_weight"},
                                     {"algorithm", "input_manifest_sha256", "sample_weights", "total_weight",
                                      "wire_input_format", "wire_codec"},
                                     {"algorithm", "input_manifest_sha256", "sample_weights", "total_weight",
                                      "wire_input_format", "wire_codec", "wire_policy_sha256"})
            or reduction["algorithm"] != _REDUCTION):
        raise ValueError("Invalid reduction metadata")
    if "wire_input_format" in reduction:
        if reduction["wire_input_format"] == BF16_EXACT_FORMAT:
            valid = reduction["wire_codec"] == BF16_EXACT_CODEC and "wire_policy_sha256" not in reduction
        elif reduction["wire_input_format"] == NATIVE_MIXED_EXACT_FORMAT:
            _sha(reduction.get("wire_policy_sha256"), "native reduction wire policy")
            valid = reduction["wire_codec"] == NATIVE_MIXED_EXACT_CODEC
        else:
            valid = (reduction["wire_input_format"] == MIXED_BF16_EXACT_FORMAT
                     and reduction["wire_codec"] == MIXED_BF16_EXACT_CODEC
                     and reduction.get("wire_policy_sha256") == _digest(VISION_WIRE_POLICY))
        if not valid:
            raise ValueError("Invalid reduction BF16-exact wire provenance")
    inputs = reduction["input_manifest_sha256"]
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= MAX_LEAVES:
        raise ValueError("Invalid ordered reduction inputs")
    for value in inputs:
        _sha(value, "input manifest checksum")
    _, _, total = _weights(reduction["sample_weights"], len(inputs))
    if type(reduction["total_weight"]) not in (int, float) or reduction["total_weight"] != total:
        raise ValueError("Reduction total weight differs")
    if _sha(digest, "reduction_sha256") != _digest(reduction):
        raise ValueError("Reduction metadata checksum mismatch")


def _validate_bf16_manifest(manifest, *, payload_size=None, expected_schema=None):
    required = {"format", "kind", "byte_order", "payload_file", "payload_bytes", "payload_sha256", "chunk_bytes",
                "schema_sha256", "leaves", "chunks", "logical_layout", "wire_schema_sha256", "wire_dtype", "codec"}
    if not required <= set(manifest) or set(manifest) - required - {"payload_hash_mode"}:
        raise ValueError("Invalid BF16-exact blob manifest fields")
    if (manifest["kind"] != "arrays" or manifest["byte_order"] != "little"
            or manifest["payload_file"] != "payload.bin" or manifest["wire_dtype"] != "bfloat16"
            or manifest["codec"] != BF16_EXACT_CODEC):
        raise ValueError("Unsupported BF16-exact kind, codec or payload filename")
    logical = _validate_layout(manifest["logical_layout"])
    wire = _bf16_wire_layout(logical)
    if _json_bytes({key: manifest.get(key) for key in wire}) != _json_bytes(wire):
        raise ValueError("BF16-exact wire/logical layout mapping differs")
    if expected_schema is not None and expected_schema != _schema(logical["leaves"]):
        raise ValueError("Blob schema differs from expected schema")
    total, shard = wire["payload_bytes"], wire["chunk_bytes"]
    if payload_size is not None and (type(payload_size) is not int or payload_size != total):
        raise ValueError("Blob payload length differs from manifest")
    chunks = manifest["chunks"]
    if not isinstance(chunks, list) or len(chunks) != (total + shard - 1) // shard:
        raise ValueError("BF16-exact chunks do not exactly cover wire payload")
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or set(chunk) != {"index", "offset", "nbytes", "sha256"}:
            raise ValueError("Invalid BF16-exact chunk fields")
        if (_integer(chunk["index"], "chunk index", 0, MAX_CHUNKS - 1) != index
                or _integer(chunk["offset"], "wire chunk offset", 0, MAX_PAYLOAD_BYTES // 2) != index * shard
                or _integer(chunk["nbytes"], "wire chunk nbytes", 2, MAX_CHUNK_BYTES // 2)
                    != min(shard, total - index * shard)):
            raise ValueError("BF16-exact chunk index/offset/length coverage differs")
        _sha(chunk["sha256"], "wire chunk sha256")
    _sha(manifest["payload_sha256"], "wire payload sha256")
    mode = manifest.get("payload_hash_mode", "sha256")
    if mode not in ("sha256", CHUNK_TREE_HASH):
        raise ValueError("Unknown blob payload hash mode")
    if mode == CHUNK_TREE_HASH and manifest["payload_sha256"] != _chunk_tree_digest(wire, chunks):
        raise ValueError("Blob chunk-tree checksum mismatch")
    if len(_json_bytes(manifest)) > MAX_MANIFEST_BYTES:
        raise ValueError("Blob manifest exceeds metadata limit")
    return copy.deepcopy(manifest)


def _validate_mixed_manifest(manifest, *, payload_size=None, expected_schema=None):
    required = {"format", "kind", "byte_order", "payload_file", "payload_bytes", "payload_sha256", "chunk_bytes",
                "schema_sha256", "leaves", "chunks", "logical_layout", "wire_schema_sha256", "wire_dtype", "codec",
                "wire_policy", "chunk_partition"}
    if not required <= set(manifest) or set(manifest) - required - {"payload_hash_mode"}:
        raise ValueError("Invalid mixed BF16-exact blob manifest fields")
    native = manifest["format"] == NATIVE_MIXED_EXACT_FORMAT
    if (manifest["kind"] != "arrays" or manifest["byte_order"] != "little"
            or manifest["payload_file"] != "payload.bin" or manifest["wire_dtype"] != "mixed_bfloat16_float32"
            or manifest["codec"] != (NATIVE_MIXED_EXACT_CODEC if native else MIXED_BF16_EXACT_CODEC)
            or (not native and manifest["wire_policy"] != VISION_WIRE_POLICY)
            or manifest["chunk_partition"] != LOGICAL_CHUNK_PARTITION):
        raise ValueError("Unsupported mixed BF16-exact kind, codec, vision policy or chunk partition")
    logical = _validate_layout(manifest["logical_layout"])
    wire = _native_mixed_wire_layout(logical, manifest["wire_policy"]) if native else _mixed_wire_layout(logical)
    if _json_bytes({key: manifest.get(key) for key in wire}) != _json_bytes(wire):
        raise ValueError("Mixed BF16-exact wire/logical leaf policy or layout differs")
    if expected_schema is not None and expected_schema != _schema(logical["leaves"]):
        raise ValueError("Blob schema differs from expected schema")
    if payload_size is not None and (type(payload_size) is not int or payload_size != wire["payload_bytes"]):
        raise ValueError("Blob payload length differs from manifest")
    expected = _mixed_chunk_layout(logical, wire)
    chunks = manifest["chunks"]
    if not isinstance(chunks, list) or len(chunks) != len(expected):
        raise ValueError("Mixed wire chunks do not cover the logical FP32 ranges")
    for chunk, spec in zip(chunks, expected):
        if (not isinstance(chunk, dict) or set(chunk) != set(spec) | {"sha256"}
                or _json_bytes({key: chunk.get(key) for key in spec}) != _json_bytes(spec)):
            raise ValueError("Mixed wire chunk logical mapping differs")
        _sha(chunk["sha256"], "wire chunk sha256")
    _sha(manifest["payload_sha256"], "wire payload sha256")
    mode = manifest.get("payload_hash_mode", "sha256")
    if mode not in ("sha256", CHUNK_TREE_HASH):
        raise ValueError("Unknown blob payload hash mode")
    if mode == CHUNK_TREE_HASH and manifest["payload_sha256"] != _chunk_tree_digest(wire, chunks):
        raise ValueError("Blob chunk-tree checksum mismatch")
    if len(_json_bytes(manifest)) > MAX_MANIFEST_BYTES:
        raise ValueError("Blob manifest exceeds metadata limit")
    return copy.deepcopy(manifest)


def validate_manifest(manifest, *, payload_size=None, expected_schema=None):
    """Validate bounds, canonical layout, exact coverage and metadata hashes.

    Data bytes are verified by ``ArrayBlob.read_chunk`` or ``verify_all_chunks``.
    The returned copy shares no mutable dictionaries with the supplied manifest.
    ``expected_schema`` uses the ordinary ``{key: {shape, dtype}}`` mapping.
    """
    if not isinstance(manifest, dict):
        raise ValueError("Blob manifest must be a JSON object")
    if manifest.get("format") == BF16_EXACT_FORMAT:
        return _validate_bf16_manifest(manifest, payload_size=payload_size, expected_schema=expected_schema)
    if manifest.get("format") in MIXED_FORMATS:
        return _validate_mixed_manifest(manifest, payload_size=payload_size, expected_schema=expected_schema)
    required = {"format", "kind", "byte_order", "payload_file", "payload_bytes", "payload_sha256",
                "chunk_bytes", "schema_sha256", "leaves", "chunks"}
    optional = {"reduction", "reduction_sha256", "source_layout", "source_chunk", "payload_hash_mode"}
    if not required <= set(manifest) or set(manifest) - required - optional:
        raise ValueError("Invalid blob manifest fields")
    if (manifest["format"] != FORMAT or manifest["kind"] not in ("arrays", "reduced_chunk")
            or manifest["byte_order"] != "little" or manifest["payload_file"] != "payload.bin"):
        raise ValueError("Unsupported blob format, kind or payload filename")
    layout = _validate_layout(_layout(manifest))
    total, shard = layout["payload_bytes"], layout["chunk_bytes"]
    if payload_size is not None and (type(payload_size) is not int or payload_size != total):
        raise ValueError("Blob payload length differs from manifest")
    if expected_schema is not None and expected_schema != _schema(layout["leaves"]):
        raise ValueError("Blob schema differs from expected schema")
    _sha(manifest["payload_sha256"], "payload_sha256")
    chunks = manifest["chunks"]
    count = (total + shard - 1) // shard
    if not isinstance(chunks, list) or len(chunks) != count:
        raise ValueError("Blob chunks do not exactly cover payload")
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or set(chunk) != {"index", "offset", "nbytes", "sha256"}:
            raise ValueError("Invalid chunk fields")
        if (_integer(chunk["index"], "chunk index", 0, MAX_CHUNKS - 1) != index
                or _integer(chunk["offset"], "chunk offset", 0, MAX_PAYLOAD_BYTES) != index * shard
                or _integer(chunk["nbytes"], "chunk nbytes", 4, MAX_CHUNK_BYTES) != min(shard, total - index * shard)):
            raise ValueError("Blob chunk index/offset/length coverage differs")
        _sha(chunk["sha256"], "chunk sha256")
    mode = manifest.get("payload_hash_mode", "sha256")
    if mode not in ("sha256", CHUNK_TREE_HASH):
        raise ValueError("Unknown blob payload hash mode")
    if mode == CHUNK_TREE_HASH and manifest["payload_sha256"] != _chunk_tree_digest(layout, chunks):
        raise ValueError("Blob chunk-tree checksum mismatch")
    reduction_fields = {"reduction", "reduction_sha256"} & set(manifest)
    if reduction_fields:
        if len(reduction_fields) != 2:
            raise ValueError("Incomplete reduction metadata")
        _validate_reduction(manifest["reduction"], manifest["reduction_sha256"])
    if manifest["kind"] == "arrays":
        if "source_layout" in manifest or "source_chunk" in manifest:
            raise ValueError("Ordinary array blob cannot contain a partial source chunk")
    else:
        if not {"reduction", "reduction_sha256", "source_layout", "source_chunk"} <= set(manifest):
            raise ValueError("Reduced chunk is missing source or reduction metadata")
        source = _validate_layout(manifest["source_layout"])
        source_chunk = manifest["source_chunk"]
        if not isinstance(source_chunk, dict) or set(source_chunk) != {"index", "offset", "nbytes"}:
            raise ValueError("Invalid source chunk fields")
        source_count = (source["payload_bytes"] + source["chunk_bytes"] - 1) // source["chunk_bytes"]
        index = _integer(source_chunk["index"], "source chunk index", 0, source_count - 1)
        offset = index * source["chunk_bytes"]
        nbytes = min(source["chunk_bytes"], source["payload_bytes"] - offset)
        if (type(source_chunk["offset"]) is not int or source_chunk["offset"] != offset
                or type(source_chunk["nbytes"]) is not int or source_chunk["nbytes"] != nbytes
                or total != nbytes or shard != source["chunk_bytes"]
                or layout["leaves"] != [{"key": "values", "shape": [nbytes // 4], "dtype": "float32",
                                          "offset": 0, "nbytes": nbytes}]):
            raise ValueError("Reduced chunk source coverage differs")
    if len(_json_bytes(manifest)) > MAX_MANIFEST_BYTES:
        raise ValueError("Blob manifest exceeds metadata limit")
    return copy.deepcopy(manifest)


def _all_finite(values):
    step = IO_BLOCK_BYTES // 4
    return all(bool(np.isfinite(values[start:start + step]).all()) for start in range(0, values.size, step))


def _array_blocks(arrays, keys):
    # nditer handles noncontiguous leaves without copying a full model leaf.
    for key in keys:
        iterator = np.nditer(arrays[key], flags=["external_loop", "buffered"],
                             op_flags=["readonly"], order="C", buffersize=IO_BLOCK_BYTES // 4)
        for piece in iterator:
            for start in range(0, piece.size, IO_BLOCK_BYTES // 4):
                block = np.ascontiguousarray(piece[start:start + IO_BLOCK_BYTES // 4], dtype=FP32)
                if not _all_finite(block):
                    raise ValueError(f"Blob array contains nonfinite values: {key}")
                yield memoryview(block).cast("B")


def _bf16_array_blocks(arrays, keys):
    for block in _array_blocks(arrays, keys):
        words = encode_bfloat16_exact(np.frombuffer(block, dtype=FP32))
        yield memoryview(words).cast("B")


def _mixed_array_blocks(arrays, keys):
    for key in keys:
        blocks = _array_blocks if key.startswith(VISION_PREFIX) else _bf16_array_blocks
        yield from blocks(arrays, [key])


def _native_mixed_array_blocks(arrays, keys, policy):
    dtypes = {leaf["key"]: leaf["dtype"] for leaf in policy["leaves"]}
    for key in keys:
        blocks = _array_blocks if dtypes[key] == "float32" else _bf16_array_blocks
        yield from blocks(arrays, [key])


def _native_snapshot_blocks(arrays):
    """Check and stream owned native words without widening or re-encoding."""
    for key in sorted(arrays):
        value = arrays[key].reshape(-1)
        step = IO_BLOCK_BYTES // value.dtype.itemsize
        for start in range(0, value.size, step):
            block = value[start:start + step]
            finite = (not bool(np.any((block & np.uint16(0x7F80)) == np.uint16(0x7F80)))
                      if value.dtype == BF16_WORD else _all_finite(block))
            if not finite:
                raise ValueError(f"Native blob array contains nonfinite values: {key}")
            yield memoryview(block).cast("B")


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _descriptor(path, manifest, manifest_sha256):
    result = {"format": manifest["format"], "path": str(path), "manifest_sha256": manifest_sha256,
            **{key: manifest[key] for key in ("kind", "schema_sha256", "payload_sha256", "payload_bytes", "chunk_bytes")},
            **({"payload_hash_mode": manifest["payload_hash_mode"]} if "payload_hash_mode" in manifest else {}),
            "chunk_count": len(manifest["chunks"])}
    if manifest["format"] in ENCODED_FORMATS:
        logical = manifest["logical_layout"]
        result.update(wire_dtype=manifest["wire_dtype"], codec=manifest["codec"],
                      wire_schema_sha256=manifest["wire_schema_sha256"],
                      logical_payload_bytes=logical["payload_bytes"], logical_chunk_bytes=logical["chunk_bytes"],
                      logical_layout_sha256=_digest(logical))
        if manifest["format"] in MIXED_FORMATS:
            result.update(wire_policy_sha256=_digest(manifest["wire_policy"]), chunk_partition=LOGICAL_CHUNK_PARTITION)
    return result


def _write_parallel_payload(path, layout, blocks, workers, durable, chunk_layout=None):
    """Bound CPU hashing and concurrent pwrite buffers to workers + one chunks."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    chunks, pending, written = [], deque(), 0
    shard = layout["chunk_bytes"]

    def write_chunk(index, values):
        view = memoryview(values)
        spec = (chunk_layout[index] if chunk_layout is not None else
                {"index": index, "offset": index * shard, "nbytes": len(view)})
        offset, position = spec["offset"], 0
        while position < len(view):
            count = os.pwrite(fd, view[position:position + IO_BLOCK_BYTES], offset + position)
            if count <= 0:
                raise OSError("Short parallel blob payload write")
            position += count
        return {**spec, "sha256": hashlib.sha256(view).hexdigest()}

    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="blob-write") as pool:
            current, filled, index = None, 0, 0
            for block in blocks:
                view, position = memoryview(block).cast("B"), 0
                while position < len(view):
                    if current is None:
                        remaining = layout["payload_bytes"] - written
                        if remaining <= 0:
                            raise ValueError("Blob writer exceeded declared payload size")
                        count = chunk_layout[index]["nbytes"] if chunk_layout is not None else min(shard, remaining)
                        current = bytearray(count)
                    count = min(len(view) - position, len(current) - filled)
                    memoryview(current)[filled:filled + count] = view[position:position + count]
                    position, filled, written = position + count, filled + count, written + count
                    if filled == len(current):
                        pending.append(pool.submit(write_chunk, index, current))
                        current, filled, index = None, 0, index + 1
                        if len(pending) >= workers:
                            chunks.append(pending.popleft().result())
            if written != layout["payload_bytes"] or filled:
                raise ValueError("Blob writer did not cover declared payload size")
            while pending:
                chunks.append(pending.popleft().result())
        if durable:
            os.fsync(fd)
    finally:
        os.close(fd)
    return chunks, written


def _publish(directory, layout, blocks, *, durable, extra=None, hash_mode="sha256", io_workers=1,
             wire_dtype="float32", wire_policy=None):
    if type(durable) is not bool:
        raise ValueError("durable must be a boolean")
    _validate_layout(layout)
    if wire_dtype not in WIRE_DTYPES:
        raise ValueError("Unknown blob wire dtype")
    if wire_dtype != NATIVE_MIXED_EXACT_WIRE and wire_policy is not None:
        raise ValueError("Explicit wire policy requires native_mixed_exact")
    logical_layout = copy.deepcopy(layout)
    chunk_layout = None
    if wire_dtype != "float32":
        if extra:
            raise ValueError("BF16-exact wire format is restricted to unreduced array blobs")
        if wire_dtype == BF16_EXACT_WIRE:
            layout = _bf16_wire_layout(logical_layout)
        else:
            if wire_dtype == NATIVE_MIXED_EXACT_WIRE:
                wire_policy = validate_native_wire_policy(wire_policy, logical_layout)
                layout = _native_mixed_wire_layout(logical_layout, wire_policy)
            else:
                layout = _mixed_wire_layout(logical_layout)
            chunk_layout = _mixed_chunk_layout(logical_layout, layout)
    _integer(io_workers, "io_workers", 1, 32)
    if hash_mode not in ("sha256", CHUNK_TREE_HASH):
        raise ValueError("Unknown blob payload hash mode")
    directory = Path(directory).absolute()
    directory.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(directory):
        raise FileExistsError(directory)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", suffix=".tmp", dir=directory.parent))
    try:
        payload_digest, chunk_digest = hashlib.sha256(), hashlib.sha256()
        chunks, written, in_chunk = [], 0, 0
        shard = layout["chunk_bytes"]
        if hash_mode == CHUNK_TREE_HASH:
            chunks, written = _write_parallel_payload(temporary / "payload.bin", layout, blocks, io_workers, durable, chunk_layout)
        else:
            stream = (temporary / "payload.bin").open("xb")
            try:
                for block in blocks:
                    view = memoryview(block).cast("B")
                    position = 0
                    while position < len(view):
                        chunk_limit = chunk_layout[len(chunks)]["nbytes"] if chunk_layout is not None else shard
                        count = min(len(view) - position, chunk_limit - in_chunk)
                        piece = view[position:position + count]
                        if written + count > layout["payload_bytes"]:
                            raise ValueError("Blob writer exceeded declared payload size")
                        if stream.write(piece) != count:
                            raise OSError("Short blob payload write")
                        payload_digest.update(piece)
                        chunk_digest.update(piece)
                        written, in_chunk, position = written + count, in_chunk + count, position + count
                        if in_chunk == chunk_limit:
                            spec = (chunk_layout[len(chunks)] if chunk_layout is not None else
                                    {"index": len(chunks), "offset": written - in_chunk, "nbytes": in_chunk})
                            chunks.append({**spec, "sha256": chunk_digest.hexdigest()})
                            chunk_digest, in_chunk = hashlib.sha256(), 0
                if written != layout["payload_bytes"]:
                    raise ValueError("Blob writer did not cover declared payload size")
                if in_chunk:
                    chunks.append({"index": len(chunks), "offset": written - in_chunk,
                                   "nbytes": in_chunk, "sha256": chunk_digest.hexdigest()})
                stream.flush()
                if durable:
                    os.fsync(stream.fileno())
            finally:
                stream.close()
        encoded_format = {"float32": FORMAT, BF16_EXACT_WIRE: BF16_EXACT_FORMAT,
                          MIXED_BF16_EXACT_WIRE: MIXED_BF16_EXACT_FORMAT,
                          NATIVE_MIXED_EXACT_WIRE: NATIVE_MIXED_EXACT_FORMAT}[wire_dtype]
        manifest = {"format": encoded_format,
                    "kind": "arrays", "byte_order": "little", "payload_file": "payload.bin",
                    **copy.deepcopy(layout), "payload_sha256": (_chunk_tree_digest(layout, chunks) if hash_mode == CHUNK_TREE_HASH else payload_digest.hexdigest()), "chunks": chunks,
                    **({"payload_hash_mode": hash_mode} if hash_mode != "sha256" else {}),
                    **({"logical_layout": logical_layout, "wire_dtype": "bfloat16", "codec": BF16_EXACT_CODEC}
                       if wire_dtype == BF16_EXACT_WIRE else
                       {"logical_layout": logical_layout, "wire_dtype": "mixed_bfloat16_float32",
                        "codec": NATIVE_MIXED_EXACT_CODEC if wire_dtype == NATIVE_MIXED_EXACT_WIRE else MIXED_BF16_EXACT_CODEC,
                        "wire_policy": copy.deepcopy(wire_policy if wire_dtype == NATIVE_MIXED_EXACT_WIRE else VISION_WIRE_POLICY),
                        "chunk_partition": LOGICAL_CHUNK_PARTITION}
                       if wire_dtype in (MIXED_BF16_EXACT_WIRE, NATIVE_MIXED_EXACT_WIRE) else {}),
                    **(extra or {})}
        validate_manifest(manifest, payload_size=written)
        encoded = _json_bytes(manifest)
        with (temporary / "manifest.json").open("xb") as stream:
            if stream.write(encoded) != len(encoded):
                raise OSError("Short blob manifest write")
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        if durable:
            _fsync_directory(temporary)
        if os.path.lexists(directory):
            raise FileExistsError(directory)
        try:
            # Every publication is a nonempty directory, so a racing published
            # blob cannot be replaced by rename. Existing empty paths are also
            # refused above. Writers never edit a published directory in place.
            os.rename(temporary, directory)
        except OSError as error:
            if error.errno in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(directory) from error
            raise
        if durable:
            _fsync_directory(directory.parent)
        return _descriptor(directory, manifest, hashlib.sha256(encoded).hexdigest())
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def save_blob(directory, arrays, shard_bytes=DEFAULT_SHARD_BYTES, durable=True, *, hash_mode="sha256", io_workers=1,
              wire_dtype="float32", wire_policy=None):
    """Atomically publish sorted, finite, nonempty FP32 leaves exactly once.

    Noncontiguous input is streamed in C order with at most one 8 MiB packing
    buffer. No payload reread is needed to produce either SHA256 checksum.
    The returned descriptor pins the manifest bytes and canonical schema.
    Exact BF16 wire modes reject any nonrepresentable input instead of rounding.
    The legacy mixed mode retains only ``PaliGemma/img/`` leaves in FP32;
    ``native_mixed_exact`` follows the complete, schema-bound ``wire_policy``.
    """
    _shard_bytes(shard_bytes)
    if wire_dtype not in WIRE_DTYPES:
        raise ValueError("Unknown blob wire dtype")
    if wire_dtype != NATIVE_MIXED_EXACT_WIRE and wire_policy is not None:
        raise ValueError("Explicit wire policy requires native_mixed_exact")
    if not isinstance(arrays, Mapping) or not 1 <= len(arrays) <= MAX_LEAVES:
        raise ValueError("Blob arrays must be a nonempty mapping")
    if any(not isinstance(key, str) for key in arrays):
        raise ValueError("Blob array keys must be strings")
    values, leaves, offset = {}, [], 0
    for key in sorted(arrays):
        value = np.asarray(arrays[key])
        if value.dtype != FP32:
            raise ValueError(f"Blob array dtype must be little-endian float32: {key}")
        if value.size == 0:
            raise ValueError(f"Blob array must be nonempty: {key}")
        values[key] = value
        leaves.append({"key": key, "shape": list(value.shape), "dtype": "float32", "offset": offset, "nbytes": value.nbytes})
        offset += value.nbytes
    layout = {"payload_bytes": offset, "chunk_bytes": shard_bytes, "leaves": leaves,
              "schema_sha256": _digest(_schema(leaves))}
    _validate_layout(layout)
    if wire_dtype == NATIVE_MIXED_EXACT_WIRE:
        wire_policy = validate_native_wire_policy(wire_policy, layout)
        blocks = _native_mixed_array_blocks(values, sorted(values), wire_policy)
    else:
        encoder = {"float32": _array_blocks, BF16_EXACT_WIRE: _bf16_array_blocks,
                   MIXED_BF16_EXACT_WIRE: _mixed_array_blocks}[wire_dtype]
        blocks = encoder(values, sorted(values))
    return _publish(directory, layout, blocks, durable=durable,
                    hash_mode=hash_mode, io_workers=io_workers, wire_dtype=wire_dtype, wire_policy=wire_policy)


def save_native_blob(directory, snapshot, shard_bytes=DEFAULT_SHARD_BYTES, durable=False, *,
                     hash_mode="sha256", io_workers=1, wire_policy=None):
    """Publish a NativeSnapshot as the existing exact mixed v4 wire format.

    BF16 leaves supply their original uint16 words; FP32 leaves retain every
    bit. Validation and streaming use bounded native blocks, never a complete
    FP32 expansion. Publication, chunk boundaries and checksums match
    ``save_blob`` on the same decoded values and native dtype policy.
    The caller retains ownership and must keep the snapshot open until return.
    """
    from jade_lotus.native_snapshot import NativeSnapshot

    if not isinstance(snapshot, NativeSnapshot):
        raise TypeError("save_native_blob requires a NativeSnapshot")
    if snapshot.closed:
        raise RuntimeError("NativeSnapshot is closed")
    if not np.little_endian:
        raise ValueError("Native blob writing requires a little-endian host")
    _shard_bytes(shard_bytes)
    policy = validate_native_wire_policy(snapshot.wire_policy)
    schema = snapshot.logical_schema
    expected_policy = make_native_wire_policy(schema, {leaf["key"]: leaf["dtype"] for leaf in policy["leaves"]})
    if policy != expected_policy or snapshot.wire_policy_sha256 != _digest(policy):
        raise ValueError("NativeSnapshot policy checksum or logical schema differs")
    leaves, offset = [], 0
    for key in sorted(schema):
        nbytes = math.prod(schema[key]["shape"]) * 4
        leaves.append({"key": key, "shape": list(schema[key]["shape"]), "dtype": "float32",
                       "offset": offset, "nbytes": nbytes})
        offset += nbytes
    layout = {"payload_bytes": offset, "chunk_bytes": shard_bytes, "leaves": leaves,
              "schema_sha256": _digest(_schema(leaves))}
    _validate_layout(layout)
    if wire_policy is not None and validate_native_wire_policy(wire_policy, layout) != policy:
        raise ValueError("Native blob wire policy differs from NativeSnapshot")
    wire_layout = _native_mixed_wire_layout(layout, policy)
    arrays = snapshot.native_arrays()
    if not isinstance(arrays, Mapping) or set(arrays) != set(schema):
        raise ValueError("NativeSnapshot arrays must cover the exact logical schema")
    if type(snapshot.native_bytes) is not int or snapshot.native_bytes != wire_layout["payload_bytes"]:
        raise ValueError("NativeSnapshot native byte count differs from its policy")
    # Retain the owned storage while serial writes or parallel chunk futures
    # consume it. Neither path borrows reusable NPU transfer staging.
    values = dict(arrays)
    for leaf in wire_layout["leaves"]:
        key, value = leaf["key"], values[leaf["key"]]
        expected_dtype = FP32 if leaf["dtype"] == "float32" else BF16_WORD
        if (not isinstance(value, np.ndarray) or value.dtype != expected_dtype
                or value.shape != tuple(leaf["shape"]) or value.nbytes != leaf["nbytes"]):
            raise ValueError(f"NativeSnapshot array shape or little-endian dtype differs: {key}")
        if not value.flags.c_contiguous or value.flags.writeable:
            raise ValueError(f"NativeSnapshot array must be read-only and C-contiguous: {key}")
    return _publish(directory, layout, _native_snapshot_blocks(values), durable=durable,
                    hash_mode=hash_mode, io_workers=io_workers,
                    wire_dtype=NATIVE_MIXED_EXACT_WIRE, wire_policy=policy)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in blob manifest: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"Nonfinite JSON constant in blob manifest: {value}")


class ArrayBlob:
    """Validated immutable metadata and an open payload descriptor.

    ``read_chunk`` owns just the requested FP32 chunk and verifies its checksum
    and finiteness. ``arrays`` returns read-only mmap views for FP32 storage,
    or owned decoded FP32 arrays for encoded storage. ``owned_arrays`` reads
    FP32 parameters once into verified RAM. Closing releases this object's
    references; returned views retain their storage independently.
    """

    def __init__(self, blob, *, expected_schema=None):
        self._fd = None
        self._mapping = None
        expected = blob if isinstance(blob, Mapping) else {}
        if isinstance(blob, Mapping):
            if not isinstance(expected.get("path"), str):
                raise ValueError("Blob descriptor requires a directory path")
            _sha(expected.get("manifest_sha256"), "descriptor manifest_sha256")
        path = expected.get("path") if expected else blob
        self.path = Path(path).absolute()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        manifest_fd = os.open(self.path / "manifest.json", flags)
        try:
            info = os.fstat(manifest_fd)
            if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= MAX_MANIFEST_BYTES:
                raise ValueError("Blob manifest is not a bounded regular file")
            with os.fdopen(manifest_fd, "rb", closefd=False) as stream:
                encoded = stream.read(MAX_MANIFEST_BYTES + 1)
            if len(encoded) != info.st_size:
                raise ValueError("Blob manifest length changed while reading")
        finally:
            os.close(manifest_fd)
        digest = hashlib.sha256(encoded).hexdigest()
        if expected and expected["manifest_sha256"] != digest:
            raise ValueError("Blob manifest checksum mismatch")
        try:
            manifest = json.loads(encoded, object_pairs_hook=_unique_object, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("Invalid blob manifest JSON") from error
        self._manifest = validate_manifest(manifest, expected_schema=expected_schema)
        self._descriptor = _descriptor(self.path, self._manifest, digest)
        for key, value in expected.items():
            if key != "path" and key in self._descriptor and value != self._descriptor[key]:
                raise ValueError(f"Blob descriptor differs from manifest: {key}")
        self._fd = os.open(self.path / "payload.bin", flags)
        try:
            self._check_payload_size()
        except BaseException:
            self.close()
            raise

    @property
    def manifest(self):
        return copy.deepcopy(self._manifest)

    @property
    def descriptor(self):
        return dict(self._descriptor)

    def _check_payload_size(self):
        if self._fd is None:
            raise ValueError("Blob is closed")
        info = os.fstat(self._fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != self._manifest["payload_bytes"]:
            raise ValueError("Blob payload length differs from manifest")

    def _read_storage_chunk(self, index):
        """Verify a range's hash on its actual on-disk bytes before decoding."""
        _integer(index, "chunk index", 0, len(self._manifest["chunks"]) - 1)
        self._check_payload_size()
        chunk = self._manifest["chunks"][index]
        dtype = (np.dtype("u1") if self._manifest["format"] in MIXED_FORMATS else
                 BF16_WORD if self._manifest["format"] == BF16_EXACT_FORMAT else FP32)
        values = np.empty(chunk["nbytes"] // dtype.itemsize, dtype=dtype)
        view, read = memoryview(values).cast("B"), 0
        while read < len(view):
            # preadv fills the owned array directly, without a second chunk-
            # sized bytes object. Short reads are retried until exact coverage.
            count = os.preadv(self._fd, [view[read:min(read + IO_BLOCK_BYTES, len(view))]], chunk["offset"] + read)
            if count == 0:
                raise ValueError("Blob chunk ended before its declared length")
            read += count
        if hashlib.sha256(view).hexdigest() != chunk["sha256"]:
            raise ValueError(f"Blob chunk checksum mismatch: {index}")
        return values

    def _decode_chunk(self, values, index):
        if self._manifest["format"] == BF16_EXACT_FORMAT:
            return decode_bfloat16_exact(values)
        if self._manifest["format"] in MIXED_FORMATS:
            chunk = self._manifest["chunks"][index]
            start, end = chunk["logical_offset"], chunk["logical_offset"] + chunk["logical_nbytes"]
            decoded = np.empty(chunk["logical_nbytes"] // 4, dtype=FP32)
            for logical, wire in zip(self._manifest["logical_layout"]["leaves"], self._manifest["leaves"]):
                left, right = max(start, logical["offset"]), min(end, logical["offset"] + logical["nbytes"])
                if left >= right:
                    continue
                dtype = FP32 if wire["dtype"] == "float32" else BF16_WORD
                count = (right - left) // 4
                offset = wire["offset"] + (left - logical["offset"]) // 4 * dtype.itemsize - chunk["offset"]
                source = np.frombuffer(values, dtype=dtype, count=count, offset=offset)
                destination = decoded[(left - start) // 4:(right - start) // 4]
                if dtype == FP32:
                    np.copyto(destination, source)
                else:
                    bits = destination.view("<u4")
                    np.copyto(bits, source, casting="safe")
                    np.left_shift(bits, np.uint32(16), out=bits)
            if not _all_finite(decoded):
                raise ValueError("Mixed BF16-exact payload contains nonfinite values")
            return decoded
        if not _all_finite(values):
            raise ValueError("Blob chunk contains nonfinite values")
        return values

    def read_chunk(self, index):
        """Return a canonical FP32 range after wire-hash and finite checks."""
        return self._decode_chunk(self._read_storage_chunk(index), index)

    def verify_all_chunks(self, workers=1):
        _integer(workers, "verify_workers", 1, 32)
        if self._manifest.get("payload_hash_mode") == CHUNK_TREE_HASH:
            def verify(index):
                self.read_chunk(index)
                return True
            indices = range(len(self._manifest["chunks"]))
            if workers == 1:
                for index in indices:
                    verify(index)
            else:
                # Futures return only a boolean, never a retained 256 MiB buffer.
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="blob-verify") as pool:
                    list(pool.map(verify, indices))
            return True
        digest = hashlib.sha256()
        for index in range(len(self._manifest["chunks"])):
            values = self._read_storage_chunk(index)
            self._decode_chunk(values, index)  # finite checks, preserving original hash bytes
            digest.update(memoryview(values).cast("B"))
            del values
        if digest.hexdigest() != self._manifest["payload_sha256"]:
            raise ValueError("Blob complete payload checksum mismatch")
        return True

    def owned_arrays(self, *, read_workers=4):
        """Read and verify an FP32 parameter snapshot into one owned RAM flat.

        Every chunk is read exactly once, hash-checked, and checked for finite
        values before copying. Ordinary SHA256 also checks the complete payload
        in byte order. Parallel reads retain at most ``read_workers`` chunks
        alongside the full output; all reads join before this method exits.
        Leaves are read-only views of the same flat and outlive this file FD.
        """
        _integer(read_workers, "parameter read_workers", 1, 16)
        self._check_payload_size()
        if self._manifest["format"] != FORMAT or self._manifest["kind"] != "arrays":
            raise ValueError("Owned parameter loading requires FP32 v1 array blobs")
        flat = np.empty(self._manifest["payload_bytes"] // 4, dtype=FP32)
        digest = (None if self._manifest.get("payload_hash_mode") == CHUNK_TREE_HASH
                  else hashlib.sha256())
        count = len(self._manifest["chunks"])

        def collect(index, values):
            if digest is not None:
                digest.update(memoryview(values).cast("B"))
            start = self._manifest["chunks"][index]["offset"] // 4
            np.copyto(flat[start:start + values.size], values)

        if read_workers == 1:
            for index in range(count):
                values = self.read_chunk(index)
                collect(index, values)
                del values
        else:
            # Futures retain only a bounded number of verified chunks. Consume
            # in storage order so the full-file SHA is over original bytes.
            with ThreadPoolExecutor(max_workers=read_workers, thread_name_prefix="parameter-read") as pool:
                pending = deque()
                next_chunk = 0
                for _ in range(min(read_workers, count)):
                    pending.append(pool.submit(self.read_chunk, next_chunk))
                    next_chunk += 1
                for index in range(count):
                    future = pending.popleft()
                    values = future.result()
                    collect(index, values)
                    del values, future
                    if next_chunk < count:
                        pending.append(pool.submit(self.read_chunk, next_chunk))
                        next_chunk += 1
        if digest is not None and digest.hexdigest() != self._manifest["payload_sha256"]:
            raise ValueError("Blob complete payload checksum mismatch")
        self._check_payload_size()
        flat.flags.writeable = False
        return {leaf["key"]: np.ndarray(tuple(leaf["shape"]), dtype=FP32, buffer=flat, offset=leaf["offset"])
                for leaf in self._manifest["leaves"]}

    def arrays(self):
        """Return read-only FP32 leaves, decoding exact wire formats if needed."""
        self._check_payload_size()
        if self._manifest["format"] in ENCODED_FORMATS:
            # Reducers use read_chunk and never allocate a whole decoded model.
            # Explicit whole-blob callers receive FP32 arrays, never uint16
            # views presented as float32. Parameter blobs stay ordinary v1.
            logical = _layout(self._manifest)
            flat = np.empty(logical["payload_bytes"] // 4, dtype=FP32)
            for index in range(len(self._manifest["chunks"])):
                values = self.read_chunk(index)
                start = index * logical["chunk_bytes"] // 4
                np.copyto(flat[start:start + values.size], values)
            flat.flags.writeable = False
            return {leaf["key"]: np.ndarray(tuple(leaf["shape"]), dtype=FP32, buffer=flat, offset=leaf["offset"])
                    for leaf in logical["leaves"]}
        if self._mapping is None:
            with os.fdopen(os.dup(self._fd), "rb") as stream:
                self._mapping = np.memmap(stream, dtype=np.uint8, mode="r", shape=(self._manifest["payload_bytes"],))
        return {leaf["key"]: np.ndarray(tuple(leaf["shape"]), dtype=FP32, buffer=self._mapping, offset=leaf["offset"])
                for leaf in self._manifest["leaves"]}

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._mapping = None  # NumPy views keep their mapping alive as needed.

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open_blob(blob, *, verify_chunks=False, expected_schema=None, verify_workers=1):
    """Open a descriptor or directory; optionally verify every chunk once."""
    if type(verify_chunks) is not bool:
        raise ValueError("verify_chunks must be a boolean")
    opened = ArrayBlob(blob, expected_schema=expected_schema)
    try:
        if verify_chunks:
            opened.verify_all_chunks(verify_workers)
        return opened
    except BaseException:
        opened.close()
        raise


def load_blob(blob, *, verify_chunks=True, expected_schema=None, verify_workers=1):
    """Load read-only FP32 leaf arrays, with optional streaming validation.

    ``verify_chunks=False`` trusts existing data bytes and checks metadata and
    file length only. It is intended for inputs already verified by the caller.
    V1 returns mmap views; exact wire formats allocate a decoded FP32 payload.
    Bounded reducers use ``read_chunk`` rather than this whole-blob API.
    """
    with open_blob(blob, verify_chunks=verify_chunks, expected_schema=expected_schema, verify_workers=verify_workers) as opened:
        return opened.arrays()


def load_blob_owned(blob, *, expected_schema=None, read_workers=4):
    """Load a fully verified FP32 parameter snapshot into read-only owned RAM.

    This opt-in alternative to ``load_blob`` does not mmap or reread payloads.
    It always verifies chunk hashes, finite values and the full payload hash.
    """
    with open_blob(blob, expected_schema=expected_schema) as opened:
        return opened.owned_arrays(read_workers=read_workers)


def _open_inputs(blobs):
    if not isinstance(blobs, Sequence) or isinstance(blobs, (str, bytes)) or not 1 <= len(blobs) <= MAX_LEAVES:
        raise ValueError("blobs must be a nonempty ordered sequence")
    opened, owned = [], []
    try:
        for blob in blobs:
            if isinstance(blob, ArrayBlob):
                blob._check_payload_size()
                opened.append(blob)
            else:
                item = open_blob(blob)
                opened.append(item)
                owned.append(item)
        return opened, owned
    except BaseException:
        for item in owned:
            item.close()
        raise


def reduce_chunk(blobs, chunk_index, sample_weights, output_directory, *, durable=True, read_workers=1):
    """Publish one ordered FP32 weighted mean shard, reading no other shards.

    Input gradient means use weights ``FP32(samples_i / sum(samples))`` and
    accumulate in the supplied input order. Parallel reads never reorder the
    FP32 additions. At most an accumulator and ``read_workers`` decoded input
    chunks are live. Encoded reads also hold their wire bytes while decoding,
    plus bounded I/O/finite-check buffers. Inputs must have identical full
    schemas, chunk layouts and native dtype policies. The output records their
    ordered manifest hashes so reconstruction cannot silently mix updates.
    """
    _integer(read_workers, "reduction read_workers", 1, 16)
    opened, owned = _open_inputs(blobs)
    try:
        weights, normalized, total = _weights(sample_weights, len(opened))
        first = opened[0]._manifest
        source = _layout(first)
        for item in opened:
            if (item._manifest["kind"] != "arrays" or _layout(item._manifest) != source
                    or item._manifest["format"] != first["format"]):
                raise ValueError("Reduction requires identical full array schemas and chunk layouts")
            if (first["format"] == NATIVE_MIXED_EXACT_FORMAT
                    and item._descriptor["wire_policy_sha256"] != opened[0]._descriptor["wire_policy_sha256"]):
                raise ValueError("Reduction requires identical native wire policies")
        _integer(chunk_index, "chunk index", 0, len(first["chunks"]) - 1)
        chunk = {"index": chunk_index, "offset": chunk_index * source["chunk_bytes"],
                 "nbytes": min(source["chunk_bytes"], source["payload_bytes"] - chunk_index * source["chunk_bytes"])}
        result = np.zeros(chunk["nbytes"] // 4, dtype=FP32)
        def accumulate(values, weight):
            with np.errstate(over="raise", invalid="raise"):
                np.multiply(values, weight, out=values)
                np.add(result, values, out=result)

        if read_workers == 1:
            for item, weight in zip(opened, normalized):
                values = item.read_chunk(chunk_index)
                accumulate(values, weight)
                del values
        else:
            # Keep the queue bounded independently of input count. Hashing,
            # reading and decoding overlap; addition order remains authoritative.
            with ThreadPoolExecutor(max_workers=read_workers, thread_name_prefix="reduce-read") as pool:
                pending = deque()
                next_input = 0
                for _ in range(min(read_workers, len(opened))):
                    pending.append(pool.submit(opened[next_input].read_chunk, chunk_index))
                    next_input += 1
                for weight in normalized:
                    future = pending.popleft()
                    values = future.result()
                    accumulate(values, weight)
                    del values, future
                    if next_input < len(opened):
                        pending.append(pool.submit(opened[next_input].read_chunk, chunk_index))
                        next_input += 1
        reduction = {"algorithm": _REDUCTION,
                     "input_manifest_sha256": [item._descriptor["manifest_sha256"] for item in opened],
                     "sample_weights": weights, "total_weight": total}
        if first["format"] in ENCODED_FORMATS:
            reduction.update(wire_input_format=first["format"], wire_codec=first["codec"])
            if first["format"] in MIXED_FORMATS:
                reduction["wire_policy_sha256"] = _digest(first["wire_policy"])
        leaves = [{"key": "values", "shape": [result.size], "dtype": "float32", "offset": 0, "nbytes": result.nbytes}]
        layout = {"payload_bytes": result.nbytes, "chunk_bytes": source["chunk_bytes"], "leaves": leaves,
                  "schema_sha256": _digest(_schema(leaves))}
        return _publish(output_directory, layout, _array_blocks({"values": result}, ["values"]), durable=durable,
                        extra={"kind": "reduced_chunk", "source_layout": source,
                               "source_chunk": {key: chunk[key] for key in ("index", "offset", "nbytes")},
                               "reduction": reduction, "reduction_sha256": _digest(reduction)})
    finally:
        for item in owned:
            item.close()


def _index_reduced_chunks(opened):
    first = opened[0]._manifest
    if first["kind"] != "reduced_chunk":
        raise ValueError("Reconstruction requires reduced chunk blobs")
    source, reduction = first["source_layout"], first["reduction"]
    count = (source["payload_bytes"] + source["chunk_bytes"] - 1) // source["chunk_bytes"]
    indexed = {}
    for item in opened:
        manifest = item._manifest
        if (manifest["kind"] != "reduced_chunk" or manifest["source_layout"] != source
                or manifest["reduction"] != reduction):
            raise ValueError("Reconstruction source schema or ordered reduction differs")
        index = manifest["source_chunk"]["index"]
        if index in indexed:
            raise ValueError("Duplicate reduced chunk index")
        indexed[index] = item
    if len(indexed) != count or set(indexed) != set(range(count)):
        raise ValueError("Reduced chunks do not exactly cover source payload")
    return indexed, source, reduction


def assemble_reduced_arrays(chunk_blobs, *, workers=1):
    """Verify reduced shards in parallel and assemble one owned RAM gradient.

    No reconstructed full payload is written or reread. Workers own bounded
    chunk buffers and copy into disjoint ranges of a single final allocation.
    The caller must match returned reduction provenance to its live plan.
    """
    _integer(workers, "assemble workers", 1, 32)
    opened, owned = _open_inputs(chunk_blobs)
    try:
        indexed, source, reduction = _index_reduced_chunks(opened)
        flat = np.empty(source["payload_bytes"] // 4, dtype=FP32)

        def copy_chunk(index):
            values = indexed[index].read_chunk(0)
            offset = index * source["chunk_bytes"] // 4
            np.copyto(flat[offset:offset + values.size], values)
            return True

        if workers == 1:
            for index in range(len(indexed)):
                copy_chunk(index)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="blob-assemble") as pool:
                list(pool.map(copy_chunk, range(len(indexed))))
        arrays = {leaf["key"]: np.ndarray(tuple(leaf["shape"]), dtype=FP32, buffer=flat, offset=leaf["offset"])
                  for leaf in source["leaves"]}
        provenance = {"reduction": copy.deepcopy(reduction), "schema_sha256": source["schema_sha256"],
                      "reduction_sha256": opened[0]._manifest["reduction_sha256"],
                      "payload_bytes": source["payload_bytes"],
                      "chunk_manifest_sha256": [indexed[index]._descriptor["manifest_sha256"] for index in range(len(indexed))]}
        return arrays, provenance
    finally:
        for item in owned:
            item.close()


def reconstruct(chunk_blobs, output_directory, *, durable=True):
    """Stream all reduced shards into an ordinary canonical full array blob.

    Shards may arrive in any order. Missing/duplicate ranges, differing source
    layouts or differing ordered reduction inputs/weights are rejected before
    payload reading or publication. Reconstruction retains all source files.
    """
    opened, owned = _open_inputs(chunk_blobs)
    try:
        indexed, source, reduction = _index_reduced_chunks(opened)

        def blocks():
            for index in range(len(indexed)):
                values = indexed[index].read_chunk(0)
                yield memoryview(values).cast("B")

        return _publish(output_directory, source, blocks(), durable=durable,
                        extra={"reduction": reduction, "reduction_sha256": opened[0]._manifest["reduction_sha256"]})
    finally:
        for item in owned:
            item.close()
