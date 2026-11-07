"""Metadata-only CPU authority with leased, ordered GPU-pod CPU reduction.

Full gradient payloads stay on shared storage until all their shards have been
reduced and one complete global-batch Adam update succeeds. The authority reads
only the final aggregate; it never loads every worker's full gradient.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import uuid

from jade_lotus.array_blob import FORMAT, NATIVE_MIXED_EXACT_WIRE, _digest, _layout, assemble_reduced_arrays, open_blob, save_blob
from jade_lotus.atomic_coordinator import AtomicCoordinator
from jade_lotus.coordinator import CentralAdam
from jade_lotus.protocol import read_json, safe_id
from jade_lotus.runtime import atomic_json, emit
from jade_lotus.sharded_transport import TRANSPORT, current_plan, expected_reduction, job_fence
from jade_lotus.sharded_transport import plan_identity, raw_result_directory, reduction_directory
from jade_lotus.sharded_transport import require_sharded, validate_bundle, validate_plan


class ShardedAtomicCoordinator(AtomicCoordinator):
    """Preserve atomic sample/Adam semantics, replacing only full-gradient IO."""

    def __init__(self, run_dir):
        super().__init__(run_dir)
        self.shard_bytes = require_sharded(self.config)
        self.gradient_wire_dtype = self.config.get("gradient_wire_dtype", "float32")
        self.received = {}
        self.source_layout = None
        self.reduction_plan = None
        self.completed_chunks = {}
        self.initializes_from_base = self.config.get("coordinator_initializes_from_base", False)
        if type(self.initializes_from_base) is not bool:
            raise ValueError("coordinator_initializes_from_base must be a boolean")
        cleanup = read_json(self.run_dir / "cleanup_pending.json", {})
        self.pending_cleanup = cleanup.get("items", []) if isinstance(cleanup, dict) else None
        if (not isinstance(self.pending_cleanup, list) or any(not isinstance(item, dict)
                or set(item) != {"path", "kind", "reason"}
                or not isinstance(item["path"], str) or not isinstance(item["reason"], str)
                or item["kind"] not in ("directory", "file") for item in self.pending_cleanup)):
            raise ValueError("Invalid pending cleanup journal")
        self._last_cleanup_retry = self._last_janitor = float("-inf")
        self._janitor_cursor = ""
        for name in ("reducers", "reductions", "results/raw"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)

    def initialize(self):
        if not self.initializes_from_base:
            return super().initialize()
        from jade_lotus.full_bootstrap import load_full_initial_parameters

        params, metadata = load_full_initial_parameters(self.config)
        self.initial_digest, self.schema = metadata["arrays_sha256"], metadata["schema"]
        self.optimizer = CentralAdam(params, self.config)
        del params
        self.restore()
        emit(self.run_dir, "optimizer_initialized", step=self.optimizer.step,
             parameters=metadata["parameter_count"], initial_digest=self.initial_digest,
             bootstrap_source=metadata["source"], bootstrap=metadata)
        if self.optimizer.step >= self.config["target_steps"]:
            self.complete()
        else:
            self.begin_version()
        return True

    def publish_parameters(self, version):
        path = self.run_dir / "params" / self.generation / f"{version:08d}.blob"
        descriptor = save_blob(path, self.optimizer.arrays(), shard_bytes=self.shard_bytes,
                               hash_mode=self.config.get("blob_hash_mode", "sha256"),
                               io_workers=self.config.get("blob_io_workers", 1))
        return {"params_path": descriptor["path"], "params_sha256": descriptor["manifest_sha256"],
                "params_format": descriptor["format"]}

    def initialize_accumulator(self):
        return None

    def begin_version(self):
        self.received, self.completed_chunks = {}, {}
        self.source_layout, self.reduction_plan = None, None
        # Fence the previous reducer plan before publishing the new authority.
        atomic_json(self.run_dir / "reduction_active.json", {"status": "idle", "generation": self.generation,
                    "version": self.optimizer.step})
        super().begin_version()
        try:
            for old in (self.run_dir / "params" / self.generation).glob("*.blob"):
                if old.stem.isdigit() and int(old.stem) < self.optimizer.step - 1:
                    self._queue_cleanup(old, "directory", "old_parameter_version")
        except OSError as error:
            self._maintenance_emit("old_parameter_cleanup_pending", error=str(error))
        self.retry_cleanup(force=True)

    def _assert_authority(self):
        published = read_json(self.run_dir / "active.json", {})
        if (published.get("status") != "training" or any(published.get(key) != self.active.get(key)
                for key in ("generation", "version", "params_sha256", "config_digest"))):
            raise RuntimeError("Sharded coordinator no longer owns this model version")

    def process_results(self):
        if self.active.get("status") != "training":
            return
        self._assert_authority()
        self.retry_cleanup()
        self.collect_orphan_blobs()
        began, changed = time.monotonic(), False
        for path in sorted((self.run_dir / "results").glob("*.json")):
            result = read_json(path)
            if not isinstance(result, dict) or result.get("attempt_id") in self.received:
                continue
            task = self.current_result_task(result)
            if task is None:
                # Late generations/attempts are not inputs. Retaining them is
                # safe; an orphan janitor can collect them independently.
                continue
            self.validate_result_owner(task, result)
            validate_bundle(result, self.config)
            expected = raw_result_directory(self.run_dir, result).resolve()
            if (path.name != result["attempt_id"] + ".json"
                    or Path(result["path"]).resolve() != expected
                    or any(result.get(key) != self.active.get(key)
                           for key in ("config_digest", "params_path", "params_sha256"))):
                raise ValueError("Raw gradient commit path or model identity differs")
            # Metadata/hash/size only: reducers validate data bytes chunkwise.
            with open_blob(result["blob"], expected_schema=self.schema) as opened:
                manifest = opened.manifest
                layout = _layout(manifest)
                if (manifest["kind"] != "arrays" or layout["chunk_bytes"] != self.shard_bytes
                        or manifest.get("payload_hash_mode", "sha256") != self.config.get("blob_hash_mode", "sha256")):
                    raise ValueError("Raw gradient kind, hash mode or shard size differs")
            if self.source_layout is not None and layout != self.source_layout:
                raise ValueError("Raw gradient full-model layouts differ")
            if json.dumps(read_json(path), sort_keys=True) != json.dumps(result, sort_keys=True):
                continue
            self._assert_authority()
            # No payload is released at provisional acceptance. All paths and
            # hashes are pinned into the immutable ordered reduction identity.
            self.source_layout = layout
            self.received[result["attempt_id"]] = copy.deepcopy(result)
            self.accepted[task["task_id"]] = {
                "samples": result["samples"], "loss": float(result["metrics"]["loss"]),
                "worker_id": result["worker_id"], "gradient_sha256": result["sha256"],
                "unit_ids": list(task["unit_ids"])}
            self.tasks.remove(task)
            self.stats["accepted_tasks"] += 1
            changed = True
            emit(self.run_dir, "gradient_metadata_accepted", version=self.optimizer.step,
                 task_id=task["task_id"], worker_id=result["worker_id"], samples=result["samples"],
                 unit_ids=task["unit_ids"], manifest_sha256=result["sha256"])
        if changed:
            self.publish()
        self.dispatch_pending()
        self.ingestion_timings["ingestion_wall_seconds"] += time.monotonic() - began

    def _publish_plan(self):
        records = sorted(self.received.values(), key=lambda record: record["unit_ids"][0])
        count = (self.source_layout["payload_bytes"] + self.shard_bytes - 1) // self.shard_bytes
        plan = {"transport": TRANSPORT, "status": "reducing", "generation": self.generation,
                "version": self.optimizer.step, "config_digest": self.config_digest,
                "params_sha256": self.active["params_sha256"], "reduction_id": uuid.uuid4().hex,
                "body": {"batch_size": self.config["batch_size"], "seed": self.config["seed"],
                         "gradient_wire_dtype": self.gradient_wire_dtype,
                         "inputs": copy.deepcopy(records), "source_layout": copy.deepcopy(self.source_layout)},
                "jobs": [{"index": index, "attempt_id": uuid.uuid4().hex, "attempt": 1}
                         for index in range(count)]}
        if self.gradient_wire_dtype == NATIVE_MIXED_EXACT_WIRE:
            plan["body"]["gradient_wire_policy_sha256"] = self.config["gradient_wire_policy_sha256"]
        plan["identity_sha256"] = _digest(plan_identity(plan))
        validate_plan(plan, self.config)
        root = reduction_directory(self.run_dir, plan)
        for name in ("claims", "results", "outputs"):
            (root / name).mkdir(parents=True, exist_ok=True)
        atomic_json(root / "plan.json", plan)
        atomic_json(self.run_dir / "reduction_active.json", plan)
        self.reduction_plan = plan
        emit(self.run_dir, "sharded_reduction_published", version=self.optimizer.step,
             reduction_id=plan["reduction_id"], identity_sha256=plan["identity_sha256"],
             source_gradients=len(records), shards=count, root_source_payload_reads=0,
             gradient_wire_dtype=self.gradient_wire_dtype,
             source_wire_payload_bytes=sum(record["blob"]["payload_bytes"] for record in records),
             source_logical_payload_bytes=len(records) * self.source_layout["payload_bytes"])

    def _result_owner(self, plan, job, result):
        root = reduction_directory(self.run_dir, plan)
        owner = read_json(root / "claims" / job["attempt_id"] / "owner.json", {})
        for key, value in job_fence(plan, job).items():
            if result.get(key) != value or owner.get(key) != value:
                raise ValueError("Reducer result does not match its current fenced shard")
        for key in ("worker_id", "session_id", "claim_token", "claimed_at", "heartbeat_path"):
            if key not in owner or result.get(key) != owner[key]:
                raise ValueError("Reducer result does not match its lease owner")
        safe_id(owner["worker_id"])
        safe_id(owner["session_id"])
        safe_id(owner["claim_token"])
        expected = self.run_dir / "reducers" / f"{owner['worker_id']}-{owner['session_id']}.json"
        if Path(owner["heartbeat_path"]).resolve() != expected.resolve():
            raise ValueError("Reducer heartbeat is outside its assigned path")
        return owner

    def _collect_chunks(self):
        plan = self.reduction_plan
        root = reduction_directory(self.run_dir, plan)
        for job in plan["jobs"]:
            if job["index"] in self.completed_chunks:
                continue
            path = root / "results" / (job["attempt_id"] + ".json")
            result = read_json(path)
            if result is None:
                continue
            if not isinstance(result, dict):
                raise ValueError("Reducer commit must be an object")
            self._result_owner(plan, job, result)
            if result.get("success") is not True:
                raise RuntimeError(f"Shard reduction failed: {result.get('error', 'invalid result')}")
            expected = root / "outputs" / (job["attempt_id"] + ".blob")
            if Path(result["blob"]["path"]).resolve() != expected.resolve():
                raise ValueError("Reduced shard result points outside its assigned output")
            with open_blob(result["blob"]) as opened:
                manifest = opened.manifest
                if (manifest["format"] != FORMAT or manifest["kind"] != "reduced_chunk"
                        or manifest["source_layout"] != plan["body"]["source_layout"]
                        or manifest["source_chunk"]["index"] != job["index"]
                        or manifest["reduction"] != expected_reduction(plan)):
                    raise ValueError("Reduced shard input order, weights or source identity differs")
            if json.dumps(read_json(path), sort_keys=True) != json.dumps(result, sort_keys=True):
                continue
            if not current_plan(self.run_dir, plan, job):
                raise RuntimeError("Reducer result arrived for an obsolete shard authority")
            self.completed_chunks[job["index"]] = copy.deepcopy(result["blob"])

    def expire(self):
        super().expire()
        plan = self.reduction_plan
        if plan is None or not current_plan(self.run_dir, plan):
            return
        root = reduction_directory(self.run_dir, plan)
        now, changed = time.time(), False
        for job in plan["jobs"]:
            if job["index"] in self.completed_chunks:
                continue
            result_path = root / "results" / (job["attempt_id"] + ".json")
            if result_path.exists():
                # Let the collector validate the small commit, even after a
                # reducer dies immediately following publication.
                continue
            directory = root / "claims" / job["attempt_id"]
            try:
                claimed_at = directory.stat().st_mtime
            except FileNotFoundError:
                continue
            owner = read_json(directory / "owner.json", {})
            last_seen = claimed_at
            worker, session = owner.get("worker_id"), owner.get("session_id")
            if isinstance(worker, str) and isinstance(session, str):
                heartbeat_path = self.run_dir / "reducers" / f"{safe_id(worker)}-{safe_id(session)}.json"
                heartbeat = read_json(heartbeat_path, {})
                fields = {**job_fence(plan, job), "worker_id": worker, "session_id": session,
                          "claim_token": owner.get("claim_token")}
                if all(heartbeat.get(key) == value for key, value in fields.items()):
                    try:
                        last_seen = max(last_seen, heartbeat_path.stat().st_mtime)
                    except FileNotFoundError:
                        pass
            hard = now - claimed_at > float(self.config.get("reducer_timeout_seconds", 3600))
            if not hard and now - last_seen <= float(self.config.get("reducer_lease_seconds", 120)):
                continue
            previous = job["attempt_id"]
            job["attempt_id"], job["attempt"] = uuid.uuid4().hex, job["attempt"] + 1
            changed = True
            emit(self.run_dir, "reducer_shard_expired", version=plan["version"], chunk_index=job["index"],
                 previous_attempt=previous, replacement_attempt=job["attempt_id"],
                 reason="timeout" if hard else "lease")
        if changed:
            # Immutable input identity remains unchanged through shard retries.
            validate_plan(plan, self.config)
            atomic_json(root / "plan.json", plan)
            atomic_json(self.run_dir / "reduction_active.json", plan)

    def _maintenance_emit(self, event, **fields):
        try:
            emit(self.run_dir, event, **fields)
        except OSError as error:
            # Storage cleanup/logging cannot invalidate an acknowledged Adam
            # update. Keep the in-memory queue if its journal is unavailable.
            print(json.dumps({"event": event, **fields, "logging_error": str(error)}), flush=True)

    def _queue_cleanup(self, path, kind, reason):
        value = {"path": str(Path(path).absolute()), "kind": kind, "reason": reason}
        if not any(item.get("path") == value["path"] for item in self.pending_cleanup):
            self.pending_cleanup.append(value)

    def _store_cleanup(self):
        try:
            atomic_json(self.run_dir / "cleanup_pending.json", {"schema_version": 1, "items": self.pending_cleanup})
        except OSError as error:
            self._maintenance_emit("cleanup_journal_pending", error=str(error), count=len(self.pending_cleanup))

    def _maintenance_state(self):
        active = read_json(self.run_dir / "active.json", {})
        plan = read_json(self.run_dir / "reduction_active.json", {})
        if (not isinstance(active, dict) or not isinstance(plan, dict)
                or not isinstance(active.get("generation"), str) or type(active.get("version")) is not int):
            return None
        references = set()

        def paths(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if (key == "path" or key.endswith("_path")) and isinstance(item, str) and Path(item).is_absolute():
                        references.add(Path(item).resolve())
                    else:
                        paths(item)
            elif isinstance(value, list):
                for item in value:
                    paths(item)

        paths(active)
        if (plan.get("status") == "reducing" and active.get("status") == "training"
                and all(active.get(key) == plan.get(key) for key in
                        ("generation", "version", "config_digest", "params_sha256"))):
            paths(plan)
            references.add(reduction_directory(self.run_dir, plan).resolve())
        current = {(active["generation"], active["version"])}
        if self.optimizer is not None:
            current.add((self.generation, self.optimizer.step))
        return current, references

    def _cleanup_path(self, item):
        path = Path(item["path"])
        if (not path.is_absolute() or path.absolute() != path.resolve() or self.run_dir not in path.parents
                or item.get("kind") not in ("directory", "file")):
            raise ValueError("Cleanup target is not a canonical path inside this run")
        parts = path.relative_to(self.run_dir).parts
        pair = None
        if len(parts) == 3 and parts[0] == "params":
            match = re.fullmatch(r"\.?([0-9]{8})\.blob(?:\..+\.tmp)?", parts[2])
            if match:
                pair = (safe_id(parts[1]), int(match[1]))
        elif len(parts) == 5 and parts[:2] == ("results", "raw") and re.fullmatch(r"[0-9]{8}", parts[3]):
            pair = (safe_id(parts[2]), int(parts[3]))
        elif len(parts) in (5, 6) and parts[0] == "reductions" and re.fullmatch(r"[0-9]{8}", parts[2]):
            safe_id(parts[3])
            if (len(parts) == 5 and parts[4] in ("outputs", "final.blob")) or (len(parts) == 6 and parts[4] == "outputs"):
                pair = (safe_id(parts[1]), int(parts[2]))
        elif len(parts) == 2 and parts[0] == "results" and parts[1].endswith(".json"):
            safe_id(path.stem)
            return path, None
        if pair is None:
            raise ValueError("Cleanup target is outside the supported blob namespaces")
        return path, pair

    def retry_cleanup(self, *, force=False):
        """Retry durable, idempotent cleanup without failing completed updates."""
        now = time.monotonic()
        if not force and now - self._last_cleanup_retry < 10:
            return 0
        self._last_cleanup_retry = now
        try:
            state = self._maintenance_state()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._maintenance_emit("blob_cleanup_scan_pending", error=str(error))
            return 0
        if state is None or not self.pending_cleanup:
            return 0
        current, references = state
        self._store_cleanup()  # A crash during deletion can only leave retries.
        remaining, removed = [], 0
        for item in self.pending_cleanup:
            try:
                path, pair = self._cleanup_path(item)
                if pair in current or any(path == ref or path in ref.parents or ref in path.parents for ref in references):
                    remaining.append(item)
                    continue
                if item["kind"] == "directory":
                    try:
                        shutil.rmtree(path)
                    except FileNotFoundError:
                        pass
                else:
                    path.unlink(missing_ok=True)
                removed += 1
                self._maintenance_emit("blob_cleanup_completed", **item)
            except (OSError, ValueError) as error:
                remaining.append(item)
                self._maintenance_emit("blob_cleanup_pending", **item, error=str(error))
        self.pending_cleanup = remaining
        self._store_cleanup()
        return removed

    @staticmethod
    def _blob_candidate(path):
        return path.name.endswith(".blob") or bool(re.fullmatch(r"\..+\.blob\..+\.tmp", path.name))

    @staticmethod
    def _newest_blob_mtime(path):
        # A long write changes payload.bin's mtime without changing its parent
        # directory. Check every member, and never follow a symlink.
        newest = path.lstat().st_mtime
        for directory, directories, files in os.walk(path, followlinks=False):
            for name in directories + files:
                member = Path(directory) / name
                if member.is_symlink():
                    return None
                newest = max(newest, member.lstat().st_mtime)
        return newest

    def collect_orphan_blobs(self, *, now=None, force=False):
        """Bounded janitor for old, unreferenced blob directories.

        Current generation/current version is never touched. Old generation
        and old version payloads are fenced from further training; small
        commit, plan and ACK metadata remain available for auditing.
        """
        monotonic = time.monotonic()
        if not force and monotonic - self._last_janitor < 60:
            return 0
        self._last_janitor = monotonic
        try:
            state = self._maintenance_state()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._maintenance_emit("blob_janitor_scan_pending", error=str(error))
            return 0
        if state is None:
            return 0
        try:
            timeout, lease = (float(self.config.get(key, default)) for key, default in
                              (("task_timeout_seconds", 1800), ("lease_seconds", 120)))
            if not all(math.isfinite(value) and value > 0 for value in (timeout, lease)):
                raise ValueError("nonfinite or nonpositive timeout")
        except (ValueError, TypeError, OverflowError) as error:
            self._maintenance_emit("blob_janitor_skipped", reason=str(error))
            return 0
        ttl, now = max(timeout + lease, 3600), time.time() if now is None else now
        current, references = state
        pairs = []
        try:
            for namespace, root in (("raw", self.run_dir / "results" / "raw"),
                                    ("params", self.run_dir / "params"),
                                    ("reductions", self.run_dir / "reductions")):
                for generation in root.iterdir():
                    if generation.is_symlink() or not generation.is_dir():
                        continue
                    safe_id(generation.name)
                    for entry in generation.iterdir():
                        if namespace == "params":
                            match = re.fullmatch(r"\.?([0-9]{8})\.blob(?:\..+\.tmp)?", entry.name)
                            version = int(match[1]) if match else None
                        else:
                            version = int(entry.name) if re.fullmatch(r"[0-9]{8}", entry.name) else None
                        if (version is None or (generation.name, version) in current
                                or (generation.name == self.generation and self.optimizer is not None
                                    and version >= self.optimizer.step)):
                            continue
                        pairs.append((f"{namespace}/{generation.name}/{entry.name}", namespace, entry))
        except (OSError, ValueError) as error:
            self._maintenance_emit("blob_janitor_scan_pending", error=str(error))
            return 0
        pairs.sort()
        pairs = ([value for value in pairs if value[0] > self._janitor_cursor]
                 + [value for value in pairs if value[0] <= self._janitor_cursor])[:64]
        queued = 0
        for key, namespace, entry in pairs:
            self._janitor_cursor = key
            try:
                if namespace == "params":
                    candidates = [entry]
                elif namespace == "raw":
                    candidates = list(entry.iterdir())
                else:
                    candidates = []
                    for reduction in entry.iterdir():
                        if reduction.is_symlink() or not reduction.is_dir():
                            continue
                        output = reduction / "outputs"
                        if output.is_dir() and not output.is_symlink():
                            candidates.extend(output.iterdir())
                        final = reduction / "final.blob"
                        if final.exists():
                            candidates.append(final)
                for path in sorted(candidates):
                    if queued >= 128:
                        break
                    if not self._blob_candidate(path) or path.is_symlink() or not path.is_dir():
                        continue
                    _, pair = self._cleanup_path({"path": str(path), "kind": "directory"})
                    if pair in current or any(path == ref or path in ref.parents or ref in path.parents for ref in references):
                        continue
                    newest = self._newest_blob_mtime(path)
                    if newest is None or now - newest < ttl:
                        continue
                    self._queue_cleanup(path, "directory", "expired_old_blob")
                    self._maintenance_emit("blob_orphan_queued", path=str(path), age_seconds=now - newest, ttl_seconds=ttl)
                    queued += 1
            except (OSError, ValueError) as error:
                self._maintenance_emit("blob_janitor_candidate_pending", path=str(entry), error=str(error))
            if queued >= 128:
                break
        if queued:
            self.retry_cleanup(force=True)
        return queued

    def _acknowledge(self, plan):
        root = reduction_directory(self.run_dir, plan)
        ack = {**{key: plan[key] for key in ("generation", "version", "reduction_id", "identity_sha256")},
               "optimizer_step": self.optimizer.step, "batch_size": plan["body"]["batch_size"],
               "gradient_wire_dtype": plan["body"].get("gradient_wire_dtype", "float32"),
               "source_manifest_sha256": [record["sha256"] for record in plan["body"]["inputs"]],
               "time": time.time()}
        if self.gradient_wire_dtype == NATIVE_MIXED_EXACT_WIRE:
            ack["gradient_wire_policy_sha256"] = plan["body"]["gradient_wire_policy_sha256"]
        # ACK precedes cleanup, so a crash can only leak files, never make
        # incomplete work appear like an accepted global-batch optimizer step.
        atomic_json(root / "ACK.json", ack)
        for record in plan["body"]["inputs"]:
            source = Path(record["path"])
            if source.absolute() != raw_result_directory(self.run_dir, record).absolute():
                raise ValueError("Refusing cleanup outside the acknowledged raw result path")
            self._queue_cleanup(source, "directory", "acknowledged_gradient")
            self._queue_cleanup(self.run_dir / "results" / (record["attempt_id"] + ".json"), "file", "acknowledged_gradient_commit")
        for name in ("outputs", "final.blob"):
            directory = root / name
            self._queue_cleanup(directory, "directory", "acknowledged_reduction")
        self.retry_cleanup(force=True)

    def update(self):
        if self.tasks or self.unassigned_units or self.active.get("status") != "training":
            return False
        self._assert_authority()
        units = [unit for value in self.accepted.values() for unit in value["unit_ids"]]
        if sorted(units) != list(range(self.unit_count)):
            raise AssertionError("Sharded update has missing or repeated atomic sample units")
        if self.reduction_plan is None:
            self._publish_plan()
            return False
        plan = self.reduction_plan
        self._collect_chunks()
        if len(self.completed_chunks) != len(plan["jobs"]):
            return False
        self._assert_authority()
        if not current_plan(self.run_dir, plan):
            raise RuntimeError("Sharded reduction authority changed before optimizer update")
        began = time.monotonic()
        chunks = [self.completed_chunks[index] for index in range(len(plan["jobs"]))]
        arrays, provenance = assemble_reduced_arrays(chunks, workers=self.config.get("blob_verify_workers", 1))
        reduction = expected_reduction(plan)
        if (provenance["reduction"] != reduction or provenance["reduction_sha256"] != _digest(reduction)
                or provenance["schema_sha256"] != plan["body"]["source_layout"]["schema_sha256"]
                or provenance["payload_bytes"] != plan["body"]["source_layout"]["payload_bytes"]
                or provenance["chunk_manifest_sha256"] != [chunk["manifest_sha256"] for chunk in chunks]):
            raise ValueError("Assembled gradient provenance differs from its authoritative ordered reduction plan")
        # Assembly verified every aggregate shard into one RAM buffer. There
        # is no reconstructed full-file write/reread, and no source-gradient IO.
        self.accumulated = arrays
        self._assert_authority()
        if not current_plan(self.run_dir, plan):
            raise RuntimeError("Sharded reduction authority changed during assembly")
        self.ingestion_timings["ingestion_aggregation_seconds"] += time.monotonic() - began
        updated = super().update()
        if updated:
            self.accumulated = None
            self._acknowledge(plan)
        return updated
