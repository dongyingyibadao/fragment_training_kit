"""Opt-in full/atomic raw transport and CPU reducer service for GPU pods.

The GPU loop can use RawWorkerClient while a background ReducerService polls
independent leases. Only small descriptors travel through the CPU authority;
the reducers read the source gradient shards from the shared filesystem.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from jade_lotus.array_blob import BF16_EXACT_CODEC, BF16_EXACT_FORMAT, BF16_EXACT_WIRE, FORMAT
from jade_lotus.array_blob import MIXED_BF16_EXACT_CODEC, MIXED_BF16_EXACT_FORMAT, MIXED_BF16_EXACT_WIRE
from jade_lotus.array_blob import NATIVE_MIXED_EXACT_CODEC, NATIVE_MIXED_EXACT_FORMAT, NATIVE_MIXED_EXACT_WIRE
from jade_lotus.array_blob import LOGICAL_CHUNK_PARTITION, VISION_WIRE_POLICY, WIRE_DTYPES
from jade_lotus.array_blob import CHUNK_TREE_HASH, DEFAULT_SHARD_BYTES, _bf16_wire_layout, _mixed_wire_layout
from jade_lotus.array_blob import _native_mixed_wire_layout, native_wire_policy_sha256, validate_native_wire_policy
from jade_lotus.array_blob import _digest, _shard_bytes, _validate_layout
from jade_lotus.array_blob import open_blob, reduce_chunk, save_blob
from jade_lotus.atomic_batch import describe_units
from jade_lotus.gradient_export import NATIVE_SNAPSHOT, gradient_export_mode
from jade_lotus.protocol import WorkerClient, read_json, safe_id, verify_sources
from jade_lotus.runtime import atomic_json


TRANSPORT = "sharded_blob_v1"
_TASK_FIELDS = ("task_id", "attempt_id", "generation", "version", "worker_id", "session_id",
                "unit_ids", "unit_seeds", "sample_positions", "params_path", "params_sha256", "config_digest")
_JOB_FIELDS = ("generation", "version", "config_digest", "params_sha256", "reduction_id", "identity_sha256")


def _native_policy(config, logical_layout=None):
    policy = validate_native_wire_policy(config.get("gradient_wire_policy"), logical_layout)
    if config.get("gradient_wire_policy_sha256") != native_wire_policy_sha256(policy):
        raise ValueError("Native gradient wire policy checksum differs from configuration")
    return policy


def require_sharded(config):
    if (config.get("gradient_transport") != TRANSPORT or config.get("training_mode") != "full"
            or config.get("atomic_batching") is not True):
        raise ValueError("sharded_blob_v1 requires explicit full training and atomic_batching=true")
    gradient_export_mode(config)
    if config.get("gradient_wire_dtype", "float32") not in WIRE_DTYPES:
        raise ValueError("Unknown gradient_wire_dtype")
    if config.get("gradient_wire_dtype", "float32") == NATIVE_MIXED_EXACT_WIRE:
        _native_policy(config)
    elif any(key in config for key in ("gradient_wire_policy", "gradient_wire_policy_sha256")):
        raise ValueError("Explicit gradient wire policy requires native_mixed_exact")
    if config.get("parameter_load_mode", "mmap_v1") not in ("mmap_v1", "owned_buffer_v1"):
        raise ValueError("Unknown parameter_load_mode")
    parameter_workers = config.get("parameter_load_workers", 4)
    if type(parameter_workers) is not int or not 1 <= parameter_workers <= 16:
        raise ValueError("parameter_load_workers must be an integer in 1..16")
    shard = _shard_bytes(config.get("gradient_shard_bytes", DEFAULT_SHARD_BYTES))
    if config.get("blob_hash_mode", "sha256") not in ("sha256", CHUNK_TREE_HASH):
        raise ValueError("Unknown blob_hash_mode")
    for key in ("blob_io_workers", "blob_verify_workers", "reducer_input_read_workers"):
        value = config.get(key, 1)
        if type(value) is not int or not 1 <= value <= 32:
            raise ValueError(f"{key} must be an integer in 1..32")
    if config.get("reducer_input_read_workers", 1) > 16:
        raise ValueError("reducer_input_read_workers must be at most 16")
    if type(config.get("reducer_plan_cache", False)) is not bool:
        raise ValueError("reducer_plan_cache must be a boolean")
    for key, default in (("reducer_lease_seconds", 120), ("reducer_timeout_seconds", 3600)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    return shard


def validate_wire_descriptor(descriptor, config, logical_layout=None):
    """Pin the actual storage format separately from its logical FP32 schema."""
    wire = config.get("gradient_wire_dtype", "float32")
    expected = {"float32": FORMAT, BF16_EXACT_WIRE: BF16_EXACT_FORMAT,
                MIXED_BF16_EXACT_WIRE: MIXED_BF16_EXACT_FORMAT,
                NATIVE_MIXED_EXACT_WIRE: NATIVE_MIXED_EXACT_FORMAT}.get(wire)
    if wire not in WIRE_DTYPES or descriptor.get("format") != expected:
        raise ValueError("Raw gradient wire format differs from configuration")
    if wire != "float32":
        if (type(descriptor.get("logical_payload_bytes")) is not int
                or type(descriptor.get("payload_bytes")) is not int
                or type(descriptor.get("chunk_bytes")) is not int
                or type(descriptor.get("logical_chunk_bytes")) is not int):
            raise ValueError("Raw BF16-exact wire/logical descriptor differs")
        if wire == BF16_EXACT_WIRE:
            valid = (descriptor.get("wire_dtype") == "bfloat16" and descriptor.get("codec") == BF16_EXACT_CODEC
                     and descriptor["logical_payload_bytes"] == 2 * descriptor["payload_bytes"]
                     and descriptor["logical_chunk_bytes"] == 2 * descriptor["chunk_bytes"]
                     and "wire_policy_sha256" not in descriptor and "chunk_partition" not in descriptor)
        elif wire == MIXED_BF16_EXACT_WIRE:
            valid = (descriptor.get("wire_dtype") == "mixed_bfloat16_float32"
                     and descriptor.get("codec") == MIXED_BF16_EXACT_CODEC
                     and descriptor.get("wire_policy_sha256") == _digest(VISION_WIRE_POLICY)
                     and descriptor.get("chunk_partition") == LOGICAL_CHUNK_PARTITION
                     and descriptor["logical_payload_bytes"] // 2 <= descriptor["payload_bytes"] <= descriptor["logical_payload_bytes"]
                     and descriptor["logical_chunk_bytes"] == descriptor["chunk_bytes"])
        else:
            policy = _native_policy(config, logical_layout)
            valid = (descriptor.get("wire_dtype") == "mixed_bfloat16_float32"
                     and descriptor.get("codec") == NATIVE_MIXED_EXACT_CODEC
                     and descriptor.get("wire_policy_sha256") == native_wire_policy_sha256(policy)
                     and descriptor.get("schema_sha256") == policy["logical_schema_sha256"]
                     and descriptor.get("chunk_partition") == LOGICAL_CHUNK_PARTITION
                     and descriptor["logical_payload_bytes"] // 2 <= descriptor["payload_bytes"] <= descriptor["logical_payload_bytes"]
                     and descriptor["logical_chunk_bytes"] == descriptor["chunk_bytes"])
        if not valid:
            raise ValueError("Raw BF16-exact wire/logical descriptor or vision policy differs")
        if logical_layout is not None:
            wire_layout = (_native_mixed_wire_layout(logical_layout, policy) if wire == NATIVE_MIXED_EXACT_WIRE else
                           (_bf16_wire_layout if wire == BF16_EXACT_WIRE else _mixed_wire_layout)(logical_layout))
            if (descriptor["logical_payload_bytes"] != logical_layout["payload_bytes"]
                    or descriptor["logical_chunk_bytes"] != logical_layout["chunk_bytes"]
                    or descriptor["payload_bytes"] != wire_layout["payload_bytes"]
                    or descriptor["chunk_bytes"] != wire_layout["chunk_bytes"]
                    or descriptor.get("logical_layout_sha256") != _digest(logical_layout)
                    or descriptor.get("wire_schema_sha256") != wire_layout["wire_schema_sha256"]):
                raise ValueError("Raw BF16-exact descriptor logical layout differs")
    elif any(key in descriptor for key in ("codec", "wire_dtype", "logical_payload_bytes", "logical_chunk_bytes",
                                           "logical_layout_sha256", "wire_schema_sha256", "wire_policy_sha256", "chunk_partition")):
        raise ValueError("FP32 raw gradient descriptor cannot claim a BF16 wire codec")


def validate_bundle(record, config):
    """Validate exact atomic sample identities and all scalar commit fields."""
    if not isinstance(record, dict) or any(field not in record for field in _TASK_FIELDS):
        raise ValueError("Incomplete raw atomic bundle descriptor")
    for name in ("task_id", "attempt_id", "generation", "worker_id", "session_id"):
        safe_id(record[name])
    if type(record["version"]) is not int or record["version"] < 0:
        raise ValueError("Invalid raw bundle version")
    units = describe_units(record["sample_positions"], run_seed=config["seed"], version=record["version"],
                           batch_size=config["batch_size"])
    if (record["unit_ids"] != [unit.unit_id for unit in units]
            or record["unit_seeds"] != [unit.seed for unit in units]
            or type(record.get("samples")) is not int or record["samples"] != 4 * len(units)):
        raise ValueError("Raw bundle sample/unit identities differ")
    if (record.get("transport") != TRANSPORT or not isinstance(record.get("metrics"), dict)
            or "loss" not in record["metrics"]
            or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in record["metrics"].values())):
        raise ValueError("Raw bundle requires finite scalar metrics and loss")
    if not isinstance(record.get("blob"), dict) or record.get("sha256") != record["blob"].get("manifest_sha256"):
        raise ValueError("Raw bundle manifest checksum differs")
    if record.get("gradient_wire_dtype", "float32") != config.get("gradient_wire_dtype", "float32"):
        raise ValueError("Raw bundle wire dtype differs from configuration")
    validate_wire_descriptor(record["blob"], config)
    if record.get("path") != record["blob"].get("path"):
        raise ValueError("Raw bundle payload path differs")
    return units


def raw_result_directory(run_dir, record):
    return (Path(run_dir) / "results" / "raw" / safe_id(record["generation"])
            / f"{record['version']:08d}" / (safe_id(record["attempt_id"]) + ".blob"))


def reduction_directory(run_dir, plan):
    return (Path(run_dir) / "reductions" / safe_id(plan["generation"])
            / f"{plan['version']:08d}" / safe_id(plan["reduction_id"]))


def plan_identity(plan):
    return {key: plan[key] for key in (*_JOB_FIELDS[:-1], "body")}


def job_fence(plan, job):
    return {**{key: plan[key] for key in _JOB_FIELDS}, "chunk_index": job["index"], "attempt_id": job["attempt_id"]}


def expected_reduction(plan):
    inputs = plan["body"]["inputs"]
    result = {"algorithm": "ordered-normalized-weighted-mean-fp32-v1",
            "input_manifest_sha256": [item["blob"]["manifest_sha256"] for item in inputs],
            "sample_weights": [float(item["samples"]) for item in inputs],
            "total_weight": float(plan["body"]["batch_size"])}
    if plan["body"].get("gradient_wire_dtype", "float32") == BF16_EXACT_WIRE:
        result.update(wire_input_format=BF16_EXACT_FORMAT, wire_codec=BF16_EXACT_CODEC)
    elif plan["body"].get("gradient_wire_dtype", "float32") == MIXED_BF16_EXACT_WIRE:
        result.update(wire_input_format=MIXED_BF16_EXACT_FORMAT, wire_codec=MIXED_BF16_EXACT_CODEC,
                      wire_policy_sha256=_digest(VISION_WIRE_POLICY))
    elif plan["body"].get("gradient_wire_dtype", "float32") == NATIVE_MIXED_EXACT_WIRE:
        result.update(wire_input_format=NATIVE_MIXED_EXACT_FORMAT, wire_codec=NATIVE_MIXED_EXACT_CODEC,
                      wire_policy_sha256=plan["body"]["gradient_wire_policy_sha256"])
    return result


def validate_plan(plan, config):
    require_sharded(config)
    if not isinstance(plan, dict) or plan.get("transport") != TRANSPORT or plan.get("status") != "reducing":
        raise ValueError("Invalid raw reduction plan")
    if plan.get("config_digest") != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest():
        raise ValueError("Reduction plan configuration differs")
    for key in ("generation", "reduction_id"):
        safe_id(plan[key])
    if type(plan.get("version")) is not int or plan["version"] < 0:
        raise ValueError("Invalid reduction version")
    if plan.get("identity_sha256") != _digest(plan_identity(plan)):
        raise ValueError("Reduction plan identity checksum differs")
    body = plan["body"]
    if body.get("batch_size") != config["batch_size"] or body.get("seed") != config["seed"]:
        raise ValueError("Reduction batch or sample RNG differs")
    wire_dtype = config.get("gradient_wire_dtype", "float32")
    if body.get("gradient_wire_dtype", "float32") != wire_dtype:
        raise ValueError("Reduction wire dtype differs from configuration")
    source = _validate_layout(body["source_layout"])
    if wire_dtype == NATIVE_MIXED_EXACT_WIRE:
        _native_policy(config, source)
        if body.get("gradient_wire_policy_sha256") != config["gradient_wire_policy_sha256"]:
            raise ValueError("Reduction native wire policy differs from configuration")
    elif "gradient_wire_policy_sha256" in body:
        raise ValueError("Reduction wire policy requires native_mixed_exact")
    if source["chunk_bytes"] != require_sharded(config):
        raise ValueError("Reduction shard size differs from configuration")
    records = body["inputs"]
    if not isinstance(records, list) or not records or len(records) > config["batch_size"] // 4:
        raise ValueError("Invalid reduction input count")
    units, previous = [], -1
    for record in records:
        described = validate_bundle(record, config)
        validate_wire_descriptor(record["blob"], config, source)
        payload_bytes = record["blob"].get("logical_payload_bytes", record["blob"]["payload_bytes"])
        chunk_bytes = record["blob"].get("logical_chunk_bytes", record["blob"]["chunk_bytes"])
        if (record["generation"] != plan["generation"] or record["version"] != plan["version"]
                or record["config_digest"] != plan["config_digest"] or record["params_sha256"] != plan["params_sha256"]
                or record["unit_ids"][0] <= previous
                or record["blob"]["schema_sha256"] != source["schema_sha256"]
                or payload_bytes != source["payload_bytes"]
                or chunk_bytes != source["chunk_bytes"]):
            raise ValueError("Reduction input model identity, order or schema differs")
        previous = record["unit_ids"][0]
        units.extend(unit.unit_id for unit in described)
    if sorted(units) != list(range(config["batch_size"] // 4)):
        raise ValueError("Reduction plan has missing or repeated sample units")
    count = (source["payload_bytes"] + source["chunk_bytes"] - 1) // source["chunk_bytes"]
    jobs = plan["jobs"]
    if not isinstance(jobs, list) or len(jobs) != count:
        raise ValueError("Reduction jobs do not cover all chunks")
    for index, job in enumerate(jobs):
        if type(job.get("index")) is not int or job["index"] != index:
            raise ValueError("Reduction job indices differ")
        safe_id(job["attempt_id"])
    return plan


def current_plan(run_dir, plan, job=None):
    """Check model, generation, immutable input set and current chunk attempt."""
    active = read_json(Path(run_dir) / "active.json", {})
    published = read_json(Path(run_dir) / "reduction_active.json", {})
    if (active.get("status") != "training" or any(active.get(key) != plan.get(key)
            for key in ("generation", "version", "config_digest", "params_sha256"))
            or published.get("status") != "reducing"
            or any(published.get(key) != plan.get(key) for key in _JOB_FIELDS)):
        return False
    if job is not None:
        jobs = published.get("jobs", [])
        index = job["index"]
        if index >= len(jobs) or jobs[index].get("attempt_id") != job["attempt_id"]:
            return False
    return True


class RawWorkerClient(WorkerClient):
    """Keep the legacy bootstrap, then exchange parameters/gradients as blobs."""

    def __init__(self, run_dir, worker_id, **kwargs):
        super().__init__(run_dir, worker_id, **kwargs)
        self.config = read_json(self.run_dir / "run_config.json")
        self.shard_bytes = require_sharded(self.config)

    def load_params(self, task):
        token = (task["generation"], task["version"], task["params_sha256"], task["params_path"])
        if self._cached_path != token:
            with open_blob({"path": task["params_path"], "manifest_sha256": task["params_sha256"]}) as opened:
                if opened.manifest["format"] != FORMAT or opened.manifest["kind"] != "arrays":
                    raise ValueError("Authoritative parameter broadcasts must remain FP32 v1 blobs")
                if self.config.get("parameter_load_mode", "mmap_v1") == "owned_buffer_v1":
                    self._cached_params = opened.owned_arrays(
                        read_workers=self.config.get("parameter_load_workers", 4))
                else:
                    opened.verify_all_chunks(self.config.get("blob_verify_workers", 1))
                    self._cached_params = opened.arrays()
            self._cached_path = token
        return self._cached_params

    def submit(self, task, gradients, metrics):
        if not self.is_current(task):
            self.task = None
            return False
        scalar_metrics = {key: float(value) for key, value in metrics.items()}
        record = {key: copy.deepcopy(task[key]) for key in _TASK_FIELDS}
        record.update(transport=TRANSPORT, samples=len(task["sample_positions"]), metrics=scalar_metrics, time=time.time(),
                      gradient_wire_dtype=self.config.get("gradient_wire_dtype", "float32"))
        path = raw_result_directory(self.run_dir, record)
        if gradient_export_mode(self.config) == NATIVE_SNAPSHOT:
            from jade_lotus.array_blob import save_native_blob

            descriptor = save_native_blob(path, gradients, shard_bytes=self.shard_bytes, durable=True,
                                         hash_mode=self.config.get("blob_hash_mode", "sha256"),
                                         io_workers=self.config.get("blob_io_workers", 1),
                                         wire_policy=self.config["gradient_wire_policy"])
        else:
            from jade_lotus.native_snapshot import NativeSnapshot

            if isinstance(gradients, NativeSnapshot):
                raise ValueError("Native snapshots require explicit native_snapshot_v1 export")
            descriptor = save_blob(path, gradients, shard_bytes=self.shard_bytes,
                                   hash_mode=self.config.get("blob_hash_mode", "sha256"),
                                   io_workers=self.config.get("blob_io_workers", 1),
                                   wire_dtype=self.config.get("gradient_wire_dtype", "float32"),
                                   wire_policy=self.config.get("gradient_wire_policy"))
        record.update(blob=descriptor, path=descriptor["path"], sha256=descriptor["manifest_sha256"])
        validate_bundle(record, self.config)
        # A large write can finish after the claim is replaced. Only publish a
        # commit when it is still current; an orphan blob is never an input.
        if not self.is_current(task):
            self.task = None
            self.heartbeat("stale_submission")
            return False
        atomic_json(self.run_dir / "results" / (safe_id(task["attempt_id"]) + ".json"), record)
        self.task = None
        self.heartbeat("submitted")
        return True


class _PublishedJsonCache:
    """Cache atomic publications for idle polling, never for commit fencing."""

    def __init__(self):
        self.entries = {}

    def read(self, path):
        path = Path(path)
        for _ in range(2):
            try:
                stat = path.stat()
            except FileNotFoundError:
                self.entries.pop(path, None)
                return {}
            signature = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
            previous = self.entries.get(path)
            if previous is not None and previous[0] == signature:
                return previous[1]
            value = read_json(path, {})
            try:
                after = path.stat()
            except FileNotFoundError:
                continue
            if signature == (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns, after.st_size):
                self.entries[path] = (signature, value)
                return value
        # A concurrent publisher can make a read immediately stale. The fresh
        # current_plan checks still fence actual claims and output publication.
        self.entries.pop(path, None)
        return read_json(path, {})


class ReducerService:
    """CPU-only leased shard work; independent of the GPU worker heartbeat."""

    def __init__(self, run_dir, worker_id, session_id, *, lease_seconds=None):
        self.run_dir = Path(run_dir).resolve()
        self.worker_id, self.session_id = safe_id(worker_id), safe_id(session_id)
        self.config = read_json(self.run_dir / "run_config.json")
        require_sharded(self.config)
        verify_sources(self.run_dir, self.config)
        self.lease_seconds = float(lease_seconds or self.config.get("reducer_lease_seconds", 120))
        self.current = None
        self._publication_cache = _PublishedJsonCache() if self.config.get("reducer_plan_cache", False) else None
        self._validated_plan = None
        self._exhausted_plan = None
        self._heartbeat_lock = threading.Lock()
        self._stop_event = None
        self.heartbeat_path = self.run_dir / "reducers" / f"{self.worker_id}-{self.session_id}.json"
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

    def heartbeat(self):
        with self._heartbeat_lock:
            claimed = self.current
            if claimed is None:
                return
            plan, job, owner = claimed
            atomic_json(self.heartbeat_path, {**owner, "stage": "reducing", "time": time.time()})

    def claim(self):
        if self._publication_cache is None:
            plan = read_json(self.run_dir / "reduction_active.json", {})
        else:
            active = self._publication_cache.read(self.run_dir / "active.json")
            if active.get("status") != "training":
                return None
            plan = self._publication_cache.read(self.run_dir / "reduction_active.json")
            if any(active.get(key) != plan.get(key)
                   for key in ("generation", "version", "config_digest", "params_sha256")):
                return None
        if plan.get("status") != "reducing":
            return None
        if self._publication_cache is None or self._validated_plan is not plan:
            validate_plan(plan, self.config)
            self._validated_plan = plan
            self._exhausted_plan = None
        if self._publication_cache is not None and self._exhausted_plan is plan:
            return None
        if self._publication_cache is None and not current_plan(self.run_dir, plan):
            return None
        root = reduction_directory(self.run_dir, plan)
        for job in plan["jobs"]:
            if (root / "results" / (job["attempt_id"] + ".json")).exists():
                continue
            directory = root / "claims" / safe_id(job["attempt_id"])
            try:
                directory.mkdir()
            except FileExistsError:
                continue
            owner = {**job_fence(plan, job), "worker_id": self.worker_id, "session_id": self.session_id,
                     "claim_token": uuid.uuid4().hex, "claimed_at": time.time(),
                     "heartbeat_path": str(self.heartbeat_path)}
            atomic_json(directory / "owner.json", owner)
            self.current = (copy.deepcopy(plan), dict(job), owner)
            self.heartbeat()
            if not current_plan(self.run_dir, plan, job):
                self.current = None
                return None
            return copy.deepcopy(owner)
        if self._publication_cache is not None:
            # Claims/results are never reopened for the same attempt. Expiry
            # rotates attempt_id and atomically republishes the plan, creating
            # a fresh cached object even though identity_sha256 stays the same.
            # Only a complete, successful no-work scan may exhaust this object.
            self._exhausted_plan = plan
        return None

    def run_once(self):
        if self.current is None and self.claim() is None:
            return False
        plan, job, owner = self.current
        if not current_plan(self.run_dir, plan, job):
            self.current = None
            return False
        root = reduction_directory(self.run_dir, plan)
        record = {**owner, "time": time.time()}
        try:
            descriptor = reduce_chunk([item["blob"] for item in plan["body"]["inputs"]], job["index"],
                                      [item["samples"] for item in plan["body"]["inputs"]],
                                      root / "outputs" / (job["attempt_id"] + ".blob"),
                                      read_workers=self.config.get("reducer_input_read_workers", 1))
            record.update(blob=descriptor, success=True)
        except Exception as error:
            # Bad source/hash/schema/finite data is a visible fatal result. It
            # cannot be silently skipped or represented by zero gradients.
            record.update(success=False, fatal=True, error=f"{type(error).__name__}: {error}")
        if (current_plan(self.run_dir, plan, job)
                and not (self._stop_event is not None and self._stop_event.is_set())):
            atomic_json(root / "results" / (job["attempt_id"] + ".json"), record)
        self.current = None
        return True

    def run(self, stop_event, *, poll_seconds=1.0):
        self._stop_event = stop_event
        renewal_stopped = threading.Event()

        def renew():
            while not renewal_stopped.wait(min(10.0, self.lease_seconds / 3)):
                if stop_event.is_set():
                    return
                self.heartbeat()

        heartbeat = threading.Thread(target=renew, name="raw-reducer-lease", daemon=True)
        heartbeat.start()
        try:
            while not stop_event.is_set():
                if any((self.run_dir / name).exists() for name in ("STOP", "FATAL.json", "COMPLETE.json")):
                    return
                if not self.run_once():
                    stop_event.wait(poll_seconds)
        finally:
            renewal_stopped.set()
            heartbeat.join(timeout=1)
            self._stop_event = None
