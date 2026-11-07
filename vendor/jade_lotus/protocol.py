"""Version-synchronous gradient work over the platform's shared filesystem.

Only the coordinator writes active.json. Workers claim with atomic mkdir, never
with a distributed advisory lock. Each assignment is fenced by a generation and
an attempt ID; partial files, duplicate and late results cannot update a model.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import uuid

import numpy as np

from jade_lotus.runtime import atomic_json


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_schema(arrays):
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sorted(arrays.items())}


def arrays_digest(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def save_arrays(path, arrays, durable=True):
    """Publish a complete uncompressed npz, then fsync its parent when durable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        digest = file_sha256(temporary)
        os.replace(temporary, path)
        if durable:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return digest
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_arrays(path, expected_sha=None):
    if expected_sha and file_sha256(path) != expected_sha:
        raise ValueError(f"Array file checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", value):
        raise ValueError("Worker/task identifiers must be safe filename components")
    return value


def verify_sources(run_dir, config):
    """Check the frozen runtime and existing upstream checkout against the run."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "source_manifest.json"
    if not manifest_path.exists():
        return  # Tiny tests and the bounded pilot predate the production freeze.
    manifest = read_json(manifest_path)
    digest = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    if manifest["sha256"] != digest or config.get("source_digest") != digest:
        raise ValueError("Frozen source manifest does not match the run configuration")
    for name, expected in manifest["files"].items():
        if name.startswith("upstream/"):
            path = Path(config["openpi_root"]) / name.removeprefix("upstream/")
        elif name.startswith("dataset/"):
            path = Path(config["dataset_root"]) / name.removeprefix("dataset/")
        else:
            path = run_dir / "runtime" / name
        if not path.exists() or file_sha256(path) != expected:
            raise ValueError(f"Source or dataset metadata changed after run preparation: {name}")


class WorkerClient:
    def __init__(self, run_dir, worker_id, lease_seconds=600):
        self.run_dir = Path(run_dir).resolve()
        self.worker_id = safe_id(worker_id)
        self.lease_seconds = lease_seconds
        config = read_json(self.run_dir / "run_config.json", {})
        self.atomic_batching = config.get("atomic_batching", False)
        self.coordinator_initializes_from_base = config.get("coordinator_initializes_from_base", False)
        if type(self.coordinator_initializes_from_base) is not bool:
            raise ValueError("coordinator_initializes_from_base must be a boolean")
        if type(self.atomic_batching) is not bool:
            raise ValueError("atomic_batching must be a boolean")
        self.capacity_samples = None
        self.device_kind = None
        self.config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        verify_sources(self.run_dir, config)
        self.session_id = uuid.uuid4().hex
        self.task = None
        self._cached_params = None
        self._cached_path = None
        self._heartbeat_lock = threading.Lock()
        for name in ("workers", "bootstrap", "claims", "results"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)

    def initialize(self, params):
        arrays = {key: np.asarray(value, dtype=np.float32) for key, value in params.items()}
        if not arrays or any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Initial trainable parameters must be finite and nonempty")
        prefix = self.worker_id + "-" + self.session_id
        path = self.run_dir / "bootstrap" / (prefix + ".npz")
        active = read_json(self.run_dir / "active.json", {})
        # Bootstrap is only necessary before the first published model. Other
        # workers still report the array digest to verify identical initialization.
        digest = arrays_digest(arrays)
        metadata = {"worker_id": self.worker_id, "session_id": self.session_id,
                    "schema": array_schema(arrays), "arrays_sha256": digest,
                    "path": str(path), "time": time.time()}
        if not active.get("params_path") and not self.coordinator_initializes_from_base:
            metadata["sha256"] = save_arrays(path, arrays)
        atomic_json(self.run_dir / "bootstrap" / (prefix + ".json"), metadata)
        self.heartbeat("initialized", arrays_sha256=digest)
        return True

    def heartbeat(self, stage="alive", **fields):
        task = self.task
        record = {"worker_id": self.worker_id, "session_id": self.session_id,
                  "time": time.time(), "stage": stage, "pid": os.getpid(),
                  "attempt_id": task.get("attempt_id") if task else None,
                    "generation": task.get("generation") if task else None,
                  "config_digest": self.config_digest, **fields}
        if self.capacity_samples is not None:
            record.update(capacity_samples=self.capacity_samples, device_kind=self.device_kind)
        with self._heartbeat_lock:
            atomic_json(self.run_dir / "workers" / (self.worker_id + ".json"), record)
        return task is None or self.is_current(task)

    def set_capacity(self, samples, device_kind):
        if not self.atomic_batching or type(samples) is not int or samples not in (4, 8, 16):
            raise ValueError("Atomic worker capacity must be exactly 4, 8 or 16 samples")
        self.capacity_samples, self.device_kind = samples, str(device_kind)

    def is_current(self, task):
        active = read_json(self.run_dir / "active.json", {})
        return (active.get("status") == "training" and active.get("generation") == task["generation"]
                and active.get("version") == task["version"]
                and any(t["attempt_id"] == task["attempt_id"] for t in active.get("tasks", [])))

    def claim(self):
        active = read_json(self.run_dir / "active.json", {})
        if active.get("status") != "training":
            return None
        if active.get("config_digest") != self.config_digest:
            raise ValueError("Worker configuration differs from the active coordinator")
        # Rotate slots so workers do not all contest the first task on GPFS.
        tasks = active.get("tasks", [])
        if self.atomic_batching:
            # The CPU authority publishes complete bundles in one active.json
            # commit. Workers never partially claim several independent units.
            tasks = [task for task in tasks if task.get("assigned_worker_id") == self.worker_id
                     and task.get("assigned_session_id") == self.session_id]
        if tasks:
            offset = int(hashlib.sha256(self.worker_id.encode()).hexdigest()[:8], 16) % len(tasks)
            tasks = tasks[offset:] + tasks[:offset]
        for descriptor in tasks:
            directory = self.run_dir / "claims" / safe_id(descriptor["attempt_id"])
            try:
                directory.mkdir()
            except FileExistsError:
                continue
            task = {**descriptor, "generation": active["generation"], "version": active["version"],
                    "params_path": active["params_path"], "params_sha256": active["params_sha256"],
                    "config_digest": active["config_digest"],
                    "worker_id": self.worker_id, "session_id": self.session_id}
            atomic_json(directory / "owner.json", {**task, "claimed_at": time.time()})
            self.task = task
            if not self.is_current(task):
                self.task = None
                continue
            self.heartbeat("claimed")
            return task
        self.task = None
        return None

    def load_params(self, task):
        if self._cached_path != task["params_path"]:
            self._cached_params = load_arrays(task["params_path"], task.get("params_sha256"))
            self._cached_path = task["params_path"]
        return self._cached_params

    def submit(self, task, gradients, metrics):
        if not self.is_current(task):
            self.task = None
            return False
        if not isinstance(metrics.get("loss"), (int, float, np.number)) or not np.isfinite(metrics["loss"]):
            raise ValueError("A finite mean training loss is required")
        arrays = {key: np.asarray(value, dtype=np.float32) for key, value in gradients.items()}
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Nonfinite gradients cannot be submitted")
        attempt = safe_id(task["attempt_id"])
        path = self.run_dir / "results" / (attempt + ".npz")
        sha = save_arrays(path, arrays)
        record = {key: task[key] for key in ("task_id", "attempt_id", "generation", "version", "worker_id", "session_id")}
        record.update(samples=len(task["sample_positions"]), path=str(path), sha256=sha,
                      metrics={key: float(v) for key, v in metrics.items()}, time=time.time())
        if self.atomic_batching:
            record.update(unit_ids=task["unit_ids"], unit_seeds=task["unit_seeds"],
                          sample_positions=task["sample_positions"])
        atomic_json(self.run_dir / "results" / (attempt + ".json"), record)
        self.task = None
        self.heartbeat("submitted")
        return True

    def done(self):
        active = read_json(self.run_dir / "active.json", {})
        return ((self.run_dir / "STOP").exists() or (self.run_dir / "COMPLETE.json").exists()
                or (self.run_dir / "FATAL.json").exists() or active.get("status") == "complete")
