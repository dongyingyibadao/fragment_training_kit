"""CPU-authoritative assignment of variable bundles of fixed four-sample units."""

import time
import uuid

import numpy as np

from jade_lotus.atomic_batch import describe_units
from jade_lotus.coordinator import Coordinator, ingestion_timing_defaults
from jade_lotus.protocol import read_json, save_arrays
from jade_lotus.runtime import emit


class AtomicCoordinator(Coordinator):
    def __init__(self, run_dir):
        super().__init__(run_dir)
        if self.config.get("atomic_batching") is not True or self.config.get("training_mode") != "full":
            raise ValueError("AtomicCoordinator requires explicit full atomic batching")
        if self.config["task_samples"] != 4 or self.config["batch_size"] % 4:
            raise ValueError("Atomic scheduling uses fixed four-sample units")
        self.unit_count = self.config["batch_size"] // 4
        self.unassigned_units = set()
        self.next_bundle = 0

    def publish(self):
        self.active.update(atomic_batching=True,
                           unassigned_unit_ids=sorted(self.unassigned_units),
                           accepted_unit_ids=sorted(unit for value in self.accepted.values()
                                                    for unit in value.get("unit_ids", [])))
        super().publish()

    def publish_parameters(self, version):
        """Transport hook; the default remains the original NPZ publication."""
        path = self.run_dir / "params" / self.generation / f"{version:08d}.npz"
        sha = save_arrays(path, self.optimizer.arrays())
        return {"params_path": str(path), "params_sha256": sha}

    def initialize_accumulator(self):
        """Transport hook for backends that assemble a final gradient later."""
        return {key: np.zeros_like(value) for key, value in self.optimizer.arrays().items()}

    def begin_version(self):
        started = time.monotonic()
        version = self.optimizer.step
        published = self.publish_parameters(version)
        publication_seconds = time.monotonic() - started
        self.active = {"status": "training", "generation": self.generation, "version": version,
            "target_steps": self.config["target_steps"], "batch_size": self.config["batch_size"],
            **published, "config_digest": self.config_digest}
        self.tasks, self.accepted, self.next_bundle = [], {}, 0
        self.unassigned_units = set(range(self.unit_count))
        self.accumulated = self.initialize_accumulator()
        self.ingestion_timings = ingestion_timing_defaults()
        self.ingestion_timings["parameter_publish_seconds"] = publication_seconds
        self.version_started = started
        self.publish()
        self.dispatch_pending()
        for old in (self.run_dir / "params" / self.generation).glob("*.npz"):
            if int(old.stem) < version - 1:
                old.unlink(missing_ok=True)

    def dispatch_pending(self):
        now, changed = time.time(), False
        busy = {task["assigned_worker_id"] for task in self.tasks}
        for path in sorted((self.run_dir / "workers").glob("*.json")):
            if not self.unassigned_units:
                break
            heartbeat = read_json(path, {})
            worker, session = heartbeat.get("worker_id"), heartbeat.get("session_id")
            capacity = heartbeat.get("capacity_samples")
            if (not worker or not session or path.name != worker + ".json" or worker in busy
                    or type(capacity) is not int or capacity not in (4, 8, 16)
                    or heartbeat.get("config_digest") != self.config_digest
                    or heartbeat.get("stage") not in ("polling", "initialized", "submitted")
                    or heartbeat.get("attempt_id") is not None
                    or now - path.stat().st_mtime > float(self.config.get("lease_seconds", 120))):
                continue
            bootstrap = read_json(self.run_dir / "bootstrap" / f"{worker}-{session}.json", {})
            if (bootstrap.get("arrays_sha256") != self.initial_digest
                    or bootstrap.get("schema") != self.schema):
                continue
            count = min(capacity // 4, len(self.unassigned_units))
            # Only supported physical shapes are issued, including the tail.
            count = max(value for value in (1, 2, 4) if value <= count)
            ids = sorted(self.unassigned_units)[:count]
            positions = [self.optimizer.step * self.config["batch_size"] + unit * 4 + row
                         for unit in ids for row in range(4)]
            units = describe_units(positions, run_seed=self.config["seed"], version=self.optimizer.step,
                                   batch_size=self.config["batch_size"])
            task = {"task_id": f"v{self.optimizer.step:08d}-t{self.next_bundle:04d}",
                    "attempt_id": uuid.uuid4().hex, "attempt": 1, "sample_positions": positions,
                    "seed": self.config["seed"], "unit_ids": ids, "unit_seeds": [u.seed for u in units],
                    "assigned_worker_id": worker, "assigned_session_id": session, "issued_at": now}
            self.next_bundle += 1
            self.tasks.append(task)
            self.unassigned_units.difference_update(ids)
            busy.add(worker)
            changed = True
            emit(self.run_dir, "atomic_bundle_assigned", version=self.optimizer.step,
                 task_id=task["task_id"], attempt_id=task["attempt_id"], worker_id=worker,
                 session_id=session, unit_ids=ids, sample_positions=positions, unit_seeds=task["unit_seeds"])
        if changed:
            self.publish()
        return changed

    def validate_result_owner(self, task, result):
        super().validate_result_owner(task, result)
        if (result["worker_id"] != task["assigned_worker_id"]
                or result["session_id"] != task["assigned_session_id"]
                or any(result.get(field) != task[field]
                       for field in ("unit_ids", "unit_seeds", "sample_positions"))):
            raise ValueError("Result does not match its authoritative atomic bundle")
        already = {unit for value in self.accepted.values() for unit in value["unit_ids"]}
        if already.intersection(task["unit_ids"]):
            raise ValueError("Atomic sample unit was already accepted")

    def process_results(self):
        super().process_results()
        self.dispatch_pending()

    def expire(self):
        now, changed = time.time(), False
        for task in list(self.tasks):
            heartbeat_path = self.run_dir / "workers" / (task["assigned_worker_id"] + ".json")
            heartbeat = read_json(heartbeat_path, {})
            same_session = heartbeat.get("session_id") == task["assigned_session_id"]
            last_seen = task["issued_at"]
            if same_session and heartbeat.get("attempt_id") == task["attempt_id"]:
                last_seen = max(last_seen, heartbeat_path.stat().st_mtime)
            reason = None
            if not same_session or now - last_seen > float(self.config.get("lease_seconds", 120)):
                reason = "atomic_lease"
            if now - task["issued_at"] > float(self.config.get("task_timeout_seconds", 1800)):
                reason = "atomic_timeout"
            if not reason:
                continue
            # A fully published descriptor is processed before expiry; retain
            # one arriving between the result scan and this lease scan.
            result = read_json(self.run_dir / "results" / (task["attempt_id"] + ".json"), {})
            if (result.get("generation") == self.generation and result.get("version") == self.optimizer.step
                    and result.get("attempt_id") == task["attempt_id"]
                    and result.get("worker_id") == task["assigned_worker_id"]
                    and result.get("session_id") == task["assigned_session_id"]):
                continue
            self.tasks.remove(task)
            self.unassigned_units.update(task["unit_ids"])
            self.stats["expired_tasks"] += 1
            changed = True
            emit(self.run_dir, "task_expired", version=self.optimizer.step, task_id=task["task_id"],
                 attempt_id=task["attempt_id"], worker_id=task["assigned_worker_id"],
                 unit_ids=task["unit_ids"], reason=reason)
        if changed:
            self.publish()
        self.dispatch_pending()

    def update(self):
        if self.tasks or self.unassigned_units:
            return False
        units = [unit for value in self.accepted.values() for unit in value["unit_ids"]]
        if sorted(units) != list(range(self.unit_count)):
            raise AssertionError("Global update has missing or repeated atomic sample units")
        return super().update()
