"""The single CPU optimizer for preemptible, changing OpenPI GPU workers."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import uuid

import numpy as np
import torch

from jade_lotus.protocol import array_schema, arrays_digest, load_arrays, read_json, save_arrays, verify_sources
from jade_lotus.runtime import atomic_json, emit, writer_lock


def ingestion_timing_defaults():
    # max_buffered_results is the queued-window slot high-water mark, including
    # ineligible descriptors. It is an upper bound on payload count, not RSS.
    return {"ingestion_read_validate_worker_seconds": 0.0,
            "ingestion_aggregation_seconds": 0.0, "ingestion_wall_seconds": 0.0,
            "ingestion_payloads_read": 0, "ingestion_max_buffered_results": 0}


def learning_rate(config, step):
    peak = float(config.get("peak_lr", 5e-5))
    warmup = int(config.get("warmup_steps", 200))
    end = float(config.get("end_lr", 5e-6))
    decay = int(config.get("decay_steps", config["target_steps"]))
    if step < warmup:
        initial = peak / (warmup + 1)
        return initial + (peak - initial) * step / max(1, warmup)
    fraction = min(1.0, (step - warmup) / max(1, decay - warmup))
    return end + (peak - end) * (1 + math.cos(math.pi * fraction)) / 2


class CentralAdam:
    """One flat FP32 CPU parameter and Adam moments, independent of GPU count."""
    def __init__(self, params, config):
        self.full_training = config.get("training_mode", "lora") == "full"
        self.fused = config.get("cpu_adam_fused", False)
        self.reuse_gradient_buffer = config.get("cpu_adam_reuse_gradient_buffer", self.full_training)
        if type(self.fused) is not bool:
            raise ValueError("cpu_adam_fused must be a boolean")
        if type(self.reuse_gradient_buffer) is not bool:
            raise ValueError("cpu_adam_reuse_gradient_buffer must be a boolean")
        self.norm_chunk_elements = config.get("cpu_adam_norm_chunk_elements", 1 << 20)
        if type(self.norm_chunk_elements) is not int or self.norm_chunk_elements < 1:
            raise ValueError("cpu_adam_norm_chunk_elements must be a positive integer")
        self.norm_workers = config.get("cpu_adam_norm_workers", 1)
        if type(self.norm_workers) is not int or not 1 <= self.norm_workers <= 32:
            raise ValueError("cpu_adam_norm_workers must be an integer in 1..32")
        self.memory_workers = config.get("cpu_adam_memory_workers", 1)
        if type(self.memory_workers) is not int or not 1 <= self.memory_workers <= 32:
            raise ValueError("cpu_adam_memory_workers must be an integer in 1..32")
        self.keys = sorted(params)
        self.shapes = {k: params[k].shape for k in self.keys}
        self.sizes = {k: params[k].size for k in self.keys}
        flat = np.concatenate([np.asarray(params[k], np.float32).reshape(-1) for k in self.keys])
        self.parameter = torch.nn.Parameter(torch.from_numpy(flat))
        self.config = config
        self.optimizer = torch.optim.AdamW([self.parameter], lr=learning_rate(config, 0),
                                          betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10,
                                          fused=self.fused)
        # A full gradient is 13.4 GB. Reuse its owned staging allocation;
        # clipping must never mutate the caller's accumulated gradient arrays.
        self._gradient_buffer = None
        self.step = 0

    @staticmethod
    def _fp64_gradient_norm(gradient, chunk_elements=1 << 20, workers=1):
        """Accumulate squared FP32 values in FP64 without a full FP64 copy.

        A single FP32 reduction can lose most contributions for billions of
        elements. einsum converts within its bounded iterator buffer, and
        fsum combines chunk totals without a second long FP32 reduction.
        """
        def squared(start):
            chunk = gradient[start:start + chunk_elements]
            with np.errstate(over="ignore", invalid="ignore"):
                return float(np.einsum("i,i->", chunk, chunk, dtype=np.float64,
                                       casting="safe", optimize=False))

        starts = range(0, gradient.size, chunk_elements)
        if workers == 1:
            squares = list(map(squared, starts))
        else:
            # map preserves the original chunk order, so math.fsum receives
            # exactly the serial sequence regardless of thread completion.
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="adam-norm") as pool:
                squares = list(pool.map(squared, starts))
        return math.sqrt(math.fsum(squares))

    def _pack_gradient(self, gradients):
        if self.full_training:
            if set(gradients) != set(self.keys):
                raise ValueError("Full-model gradient schema differs from optimizer")
            for key in self.keys:
                value = np.asarray(gradients[key])
                if value.shape != self.shapes[key] or value.dtype != np.float32:
                    raise ValueError(f"Full-model gradient must match FP32 parameter shape: {key}")
        reused = self._gradient_buffer is not None
        if not self.reuse_gradient_buffer:
            return (np.concatenate([gradients[k].reshape(-1) for k in self.keys])
                    .astype(np.float32, copy=False)), False
        if self._gradient_buffer is None:
            self._gradient_buffer = np.empty(self.parameter.numel(), dtype=np.float32)
        offset, copies = 0, []
        for key in self.keys:
            size = self.sizes[key]
            destination = self._gradient_buffer[offset:offset + size].reshape(self.shapes[key])
            copies.append((destination, gradients[key]))
            offset += size
        def copy_leaf(pair):
            np.copyto(pair[0], pair[1], casting="same_kind")
        if self.full_training and self.memory_workers > 1:
            with ThreadPoolExecutor(max_workers=self.memory_workers, thread_name_prefix="adam-copy") as pool:
                list(pool.map(copy_leaf, copies))
        else:
            for pair in copies:
                copy_leaf(pair)
        return self._gradient_buffer, reused

    def _parameters_finite(self):
        if not self.full_training:
            return bool(torch.isfinite(self.parameter).all())
        flat = self.parameter.detach().numpy()
        def finite(start):
            return bool(np.isfinite(flat[start:start + self.norm_chunk_elements]).all())
        starts = range(0, flat.size, self.norm_chunk_elements)
        if self.memory_workers == 1:
            return all(map(finite, starts))
        with ThreadPoolExecutor(max_workers=self.memory_workers, thread_name_prefix="adam-finite") as pool:
            return all(pool.map(finite, starts))

    def arrays(self):
        flat = self.parameter.detach().numpy()
        result, offset = {}, 0
        for key in self.keys:
            size = self.sizes[key]
            result[key] = flat[offset:offset + size].reshape(self.shapes[key])
            offset += size
        return result

    def update(self, gradients):
        began = time.perf_counter()
        gradient, reused = self._pack_gradient(gradients)
        packed = time.perf_counter()
        gradient_tensor = torch.from_numpy(gradient)
        norm = (self._fp64_gradient_norm(gradient, self.norm_chunk_elements, self.norm_workers) if self.full_training
                else float(torch.linalg.vector_norm(gradient_tensor)))
        normed = time.perf_counter()
        if not math.isfinite(norm):
            raise FloatingPointError("Global gradient norm is not finite")
        clip = float(self.config.get("clip_gradient_norm", 1.0))
        clip_scale = 1.0
        self.parameter.grad = gradient_tensor
        # Optax clips by norm exactly; avoid torch's extra 1e-6 denominator.
        if norm > clip:
            clip_scale = clip / norm
            self.parameter.grad.mul_(clip_scale)
        clipped = time.perf_counter()
        lr = learning_rate(self.config, self.step)
        self.optimizer.param_groups[0]["lr"] = lr
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        optimized = time.perf_counter()
        self.step += 1
        if not self._parameters_finite():
            raise FloatingPointError("Updated parameters are not finite")
        finished = time.perf_counter()
        return {"grad_norm": norm, "lr": lr, "gradient_clip_scale": clip_scale,
                "cpu_adam_fused": bool(self.optimizer.param_groups[0]["fused"]),
                "cpu_adam_norm_accumulation_bits": 64 if self.full_training else 32,
                "cpu_adam_norm_workers": self.norm_workers,
                "cpu_adam_memory_workers": self.memory_workers,
                "cpu_adam_gradient_buffer_reused": reused,
                "cpu_adam_pack_seconds": packed - began,
                "cpu_adam_norm_seconds": normed - packed,
                "cpu_adam_clip_seconds": clipped - normed,
                "cpu_adam_step_seconds": optimized - clipped,
                "cpu_adam_parameter_finite_seconds": finished - optimized,
                "cpu_adam_total_seconds": finished - began}

    def checkpoint_arrays(self):
        state = self.optimizer.state.get(self.parameter, {})
        # dict.get evaluates its default eagerly. Do not allocate/zero another
        # complete pair of full-model moments when they already exist.
        exp_avg = state["exp_avg"] if "exp_avg" in state else torch.zeros_like(self.parameter)
        exp_avg_sq = state["exp_avg_sq"] if "exp_avg_sq" in state else torch.zeros_like(self.parameter)
        return {"parameter": self.parameter.detach().numpy(),
                "exp_avg": exp_avg.numpy(), "exp_avg_sq": exp_avg_sq.numpy()}

    def restore(self, arrays, step):
        for key in ("parameter", "exp_avg", "exp_avg_sq"):
            if (arrays[key].shape != self.parameter.shape or not np.isfinite(arrays[key]).all()
                    or (self.full_training and arrays[key].dtype != np.float32)):
                raise ValueError("Invalid optimizer checkpoint arrays")
        with torch.no_grad():
            self.parameter.copy_(torch.from_numpy(arrays["parameter"]))
        self.optimizer.state[self.parameter] = {
            "step": torch.tensor(float(step)),
            "exp_avg": torch.from_numpy(arrays["exp_avg"].copy()),
            "exp_avg_sq": torch.from_numpy(arrays["exp_avg_sq"].copy()),
        }
        self.step = step


class Coordinator:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        self.config = read_json(self.run_dir / "run_config.json")
        if not self.config:
            raise ValueError("An immutable run_config.json is required")
        self.gradient_prefetch_workers = self.config.get("gradient_prefetch_workers", 1)
        if (type(self.gradient_prefetch_workers) is not int
                or not 1 <= self.gradient_prefetch_workers <= 4):
            raise ValueError("gradient_prefetch_workers must be an integer in 1..4")
        verify_sources(self.run_dir, self.config)
        self.config_digest = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.generation = uuid.uuid4().hex
        self.optimizer = None
        self.tasks = []
        self.accumulated = None
        self.accepted = {}
        self.version_started = time.monotonic()
        self.last_checkpoint_time = time.monotonic()
        self.last_checkpoint_step = -1
        self.initial_digest = None
        self.stop_requested = False
        self.active = {"status": "initializing", "generation": self.generation, "version": 0, "tasks": []}
        self.stats = {"accepted_tasks": 0, "expired_tasks": 0, "rejected_results": 0, "updates": 0}
        self.checked_bootstraps = set()
        self.ingestion_timings = ingestion_timing_defaults()
        for name in ("params", "claims", "results", "workers", "bootstrap", "checkpoints"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)
        if self.config["batch_size"] < 1 or self.config["batch_size"] % self.config["task_samples"]:
            raise ValueError("Global batch must be divisible by task_samples")

    def publish(self):
        self.active["tasks"] = self.tasks
        self.active["updated_at"] = time.time()
        atomic_json(self.run_dir / "active.json", self.active)

    def initialize(self):
        candidates = sorted((self.run_dir / "bootstrap").glob("*.json"))
        for path in candidates:
            metadata = read_json(path)
            if not metadata or not metadata.get("sha256"):
                continue
            try:
                params = load_arrays(metadata["path"], metadata["sha256"])
                if array_schema(params) != metadata["schema"] or arrays_digest(params) != metadata["arrays_sha256"]:
                    raise ValueError("Bootstrap array/schema mismatch")
            except (OSError, ValueError) as error:
                emit(self.run_dir, "invalid_bootstrap", path=str(path), error=str(error))
                continue
            self.initial_digest = metadata["arrays_sha256"]
            self.optimizer = CentralAdam(params, self.config)
            self.schema = array_schema(params)
            self.restore()
            emit(self.run_dir, "optimizer_initialized", step=self.optimizer.step,
                 parameters=sum(v.size for v in params.values()), initial_digest=self.initial_digest)
            if self.optimizer.step >= self.config["target_steps"]:
                self.complete()
            else:
                self.begin_version()
            return True
        return False

    def check_bootstraps(self):
        for path in (self.run_dir / "bootstrap").glob("*.json"):
            if str(path) in self.checked_bootstraps:
                continue
            metadata = read_json(path)
            if not metadata:
                continue
            if metadata["arrays_sha256"] != self.initial_digest or metadata["schema"] != self.schema:
                raise ValueError(f"Worker initialized a different model: {metadata['worker_id']}")
            self.checked_bootstraps.add(str(path))

    def restore(self):
        # Scan durable manifests newest-first; incomplete or corrupt saves cannot
        # hide the preceding complete checkpoint.
        candidates = sorted((self.run_dir / "checkpoints").glob("step-*.json"), reverse=True)
        if candidates and not any(read_json(p, {}).get("config_digest") == self.config_digest for p in candidates):
            raise ValueError("Committed checkpoints belong to another configuration; refusing to restart at step zero")
        for path in candidates:
            metadata = read_json(path)
            if not metadata or metadata.get("config_digest") != self.config_digest:
                continue
            try:
                arrays = load_arrays(metadata["path"], metadata["sha256"])
                if metadata["initial_digest"] != self.initial_digest:
                    raise ValueError("Checkpoint belongs to another initial model")
                self.optimizer.restore(arrays, metadata["step"])
            except (OSError, ValueError) as error:
                emit(self.run_dir, "checkpoint_rejected", path=str(path), error=str(error))
                continue
            self.last_checkpoint_step = self.optimizer.step
            self.stats.update(metadata.get("stats", {}))
            emit(self.run_dir, "checkpoint_restored", step=self.optimizer.step, generation=self.generation)
            return
        if candidates:
            raise ValueError("No valid committed checkpoint remains; refusing to reset training")

    def begin_version(self):
        version = self.optimizer.step
        path = self.run_dir / "params" / self.generation / f"{version:08d}.npz"
        sha = save_arrays(path, self.optimizer.arrays())
        self.active = {"status": "training", "generation": self.generation, "version": version,
                       "target_steps": self.config["target_steps"], "batch_size": self.config["batch_size"],
                       "params_path": str(path), "params_sha256": sha, "config_digest": self.config_digest}
        count = self.config["task_samples"]
        self.tasks = []
        for slot in range(self.config["batch_size"] // count):
            start = version * self.config["batch_size"] + slot * count
            task_seed = int.from_bytes(hashlib.sha256(f"{self.config['seed']}:{version}:{slot}".encode()).digest()[:4], "little")
            self.tasks.append({"task_id": f"v{version:08d}-t{slot:04d}", "attempt_id": uuid.uuid4().hex,
                               "sample_positions": list(range(start, start + count)), "seed": task_seed,
                               "attempt": 1})
        self.accumulated = {key: np.zeros_like(value) for key, value in self.optimizer.arrays().items()}
        self.accepted = {}
        self.ingestion_timings = ingestion_timing_defaults()
        self.version_started = time.monotonic()
        self.publish()
        # Only the preceding and current published versions need be retained.
        for old in path.parent.glob("*.npz"):
            if int(old.stem) < version - 1:
                old.unlink(missing_ok=True)

    def current_result_task(self, result):
        """Match a descriptor against main-thread authoritative state."""
        task = next((t for t in self.tasks if t["attempt_id"] == result.get("attempt_id")), None)
        if (not task or result.get("generation") != self.generation
                or result.get("version") != self.optimizer.step
                or result.get("task_id") != task["task_id"] or task["task_id"] in self.accepted):
            return None
        return task

    def validate_result_owner(self, task, result):
        owner = read_json(self.run_dir / "claims" / task["attempt_id"] / "owner.json", {})
        fields = ("task_id", "attempt_id", "generation", "version", "worker_id", "session_id")
        if (not result.get("worker_id") or not result.get("session_id")
                or any(owner.get(key) != result.get(key) for key in fields)
                or owner.get("config_digest") != self.config_digest
                or owner.get("sample_positions") != task["sample_positions"]
                or owner.get("seed") != task["seed"]
                or result.get("samples") != len(task["sample_positions"])):
            raise ValueError("Result does not match the issued sample/worker assignment")

    def read_result_payload(self, result):
        """Only immutable payload reads/validation run in background threads."""
        began = time.monotonic()
        try:
            arrays = load_arrays(result["path"], result["sha256"])
            if array_schema(arrays) != self.schema or any(not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError("Invalid gradient schema or values")
            loss = float(result["metrics"]["loss"])
            if not math.isfinite(loss):
                raise ValueError("Nonfinite loss")
        except (OSError, KeyError, ValueError, TypeError, OverflowError) as error:
            return [None, None, str(error), time.monotonic() - began]
        return [arrays, loss, None, time.monotonic() - began]

    def prepare_result_read(self, path, executor):
        result = read_json(path)
        job = {"path": path, "result": result, "future": None, "error": None, "ready": False}
        if not isinstance(result, dict) or not result:
            return job
        task = self.current_result_task(result)
        if task is None:
            return job
        try:
            self.validate_result_owner(task, result)
        except (OSError, KeyError, ValueError, TypeError) as error:
            job["error"] = str(error)
            return job
        job["ready"] = True
        if executor is not None:
            job["future"] = executor.submit(self.read_result_payload, result)
        return job

    def accept_result_read(self, job, loaded):
        """Recheck fences after IO, then mutate only on the main thread."""
        path, result = job["path"], job["result"]
        if not isinstance(result, dict) or not result:
            return False
        # A replaced commit descriptor must be read afresh on a later pass;
        # never delete its new payload using the older prefetched descriptor.
        # JSON's NaN spelling also compares consistently here, allowing the
        # payload validator to reject a nonfinite loss instead of retrying its
        # descriptor forever because float('nan') != float('nan').
        if json.dumps(read_json(path), sort_keys=True) != json.dumps(result, sort_keys=True):
            return False
        task = self.current_result_task(result)
        if task is None:
            self.stats["rejected_results"] += 1
            self.remove_result(path, result)
            return False
        arrays, loss, error, _ = loaded
        try:
            self.validate_result_owner(task, result)
        except (OSError, KeyError, ValueError, TypeError) as mismatch:
            error = str(mismatch)
        if error is not None:
            emit(self.run_dir, "result_rejected", task_id=task["task_id"], reason=error)
            self.stats["rejected_results"] += 1
            task["attempt_id"] = uuid.uuid4().hex
            task["attempt"] += 1
            self.remove_result(path, result)
            return True
        if not job["ready"]:
            return False
        began = time.monotonic()
        weight = result["samples"] / self.config["batch_size"]
        for key in self.accumulated:
            self.accumulated[key] += arrays[key] * weight
        self.ingestion_timings["ingestion_aggregation_seconds"] += time.monotonic() - began
        self.accepted[task["task_id"]] = {"samples": result["samples"], "loss": loss,
                                           "worker_id": result["worker_id"], "gradient_sha256": result["sha256"]}
        if "unit_ids" in task:
            self.accepted[task["task_id"]]["unit_ids"] = task["unit_ids"]
        self.tasks.remove(task)
        self.stats["accepted_tasks"] += 1
        emit(self.run_dir, "gradient_accepted", version=self.optimizer.step, task_id=task["task_id"],
             worker_id=result["worker_id"], samples=result["samples"], loss=loss)
        self.remove_result(path, result)
        return True

    def process_results(self):
        paths = sorted((self.run_dir / "results").glob("*.json"))
        if not paths:
            return
        began = time.monotonic()
        changed = False
        window = min(self.gradient_prefetch_workers, len(paths))
        pool = (ThreadPoolExecutor(max_workers=window, thread_name_prefix="gradient-prefetch")
                if window > 1 else nullcontext(None))
        with pool as executor:
            remaining = iter(paths)
            pending = deque(self.prepare_result_read(next(remaining), executor) for _ in range(window))
            self.ingestion_timings["ingestion_max_buffered_results"] = max(
                self.ingestion_timings["ingestion_max_buffered_results"], len(pending))
            while pending:
                job = pending.popleft()
                loaded = [None, None, job["error"], 0.0]
                if job["ready"]:
                    loaded = (job["future"].result() if executor is not None
                              else self.read_result_payload(job["result"]))
                    self.ingestion_timings["ingestion_read_validate_worker_seconds"] += loaded[3]
                    self.ingestion_timings["ingestion_payloads_read"] += 1
                changed = self.accept_result_read(job, loaded) or changed
                # Futures retain their returned arrays. Release both references
                # before refilling. Clearing the shared result container also
                # handles a worker still finishing Future.set_result(), so it
                # cannot transiently retain an (N+1)th complete payload.
                loaded[0] = None
                del job, loaded
                path = next(remaining, None)
                if path is not None:
                    pending.append(self.prepare_result_read(path, executor))
        if changed:
            self.publish()
        # Worker seconds overlap under prefetch; they must not be added to the
        # main-thread aggregation seconds and called a wall-time partition.
        self.ingestion_timings["ingestion_wall_seconds"] += time.monotonic() - began

    def remove_result(self, path, result):
        # Result array paths must remain within our result directory.
        data = Path(result.get("path", "")).resolve()
        if data.parent == (self.run_dir / "results").resolve():
            data.unlink(missing_ok=True)
        path.unlink(missing_ok=True)

    def expire(self):
        now, changed = time.time(), False
        lease = float(self.config.get("lease_seconds", 120))
        hard_limit = float(self.config.get("task_timeout_seconds", 1800))
        # A worker that completed a task in this version has already compiled
        # this graph. Use its genuinely idle slot to replace a cold/slow claim,
        # while preserving the version's exact samples and deterministic RNG.
        claimed_workers, pending = set(), 0
        for task in self.tasks:
            directory = self.run_dir / "claims" / task["attempt_id"]
            owner = read_json(directory / "owner.json", {})
            if owner.get("worker_id"):
                claimed_workers.add(owner["worker_id"])
            else:
                pending += 1
        warm_idle = 0
        for worker_id in {value["worker_id"] for value in self.accepted.values()} - claimed_workers:
            heartbeat_path = self.run_dir / "workers" / (worker_id + ".json")
            heartbeat = read_json(heartbeat_path, {})
            if (heartbeat.get("worker_id") == worker_id and heartbeat.get("session_id")
                    and heartbeat.get("stage") == "polling" and heartbeat.get("attempt_id") is None):
                try:
                    fresh = now - heartbeat_path.stat().st_mtime <= min(lease, 40.0)
                except FileNotFoundError:
                    fresh = False
                warm_idle += int(fresh)
        # Existing unclaimed work already reserves idle capacity. This also
        # prevents repeated expiry passes from reusing one idle heartbeat to
        # fence more slow attempts before its first replacement is claimed.
        warm_capacity = max(0, warm_idle - pending)
        for task in self.tasks:
            directory = self.run_dir / "claims" / task["attempt_id"]
            if not directory.exists():
                continue
            owner = read_json(directory / "owner.json", {})
            claimed_at = directory.stat().st_mtime
            last_seen = claimed_at
            heartbeat, matching_heartbeat = {}, False
            if owner:
                heartbeat_path = self.run_dir / "workers" / (owner["worker_id"] + ".json")
                heartbeat = read_json(heartbeat_path, {})
                if (heartbeat.get("attempt_id") == task["attempt_id"]
                        and heartbeat.get("session_id") == owner["session_id"]):
                    last_seen = max(last_seen, heartbeat_path.stat().st_mtime)
                    matching_heartbeat = True
            reason = None
            if now - claimed_at > hard_limit:
                reason = "timeout"
            elif now - last_seen > lease:
                reason = "lease"
            elif (warm_capacity > 0 and now - claimed_at > 60.0 and matching_heartbeat
                  and heartbeat.get("stage") in ("computing", "verifying_replay")):
                reason = "slow_task_with_warm_capacity"
            if reason:
                # submit() commits this JSON before clearing the worker's
                # heartbeat attempt. It may arrive just after process_results()
                # snapshots its glob. Keep the completed attempt until the next
                # result pass instead of falling back to its old claim time.
                # Check only the small commit descriptor here; payload checksum,
                # schema, sample count and finiteness remain process_results()'s
                # responsibility, without a second gradient-array read.
                result = read_json(self.run_dir / "results" / (task["attempt_id"] + ".json"), {})
                if (isinstance(result, dict) and owner.get("worker_id") and owner.get("session_id")
                        and result.get("generation") == self.generation
                        and result.get("version") == self.optimizer.step
                        and result.get("task_id") == task["task_id"]
                        and result.get("attempt_id") == task["attempt_id"]
                        and result.get("worker_id") == owner["worker_id"]
                        and result.get("session_id") == owner["session_id"]):
                    continue
                emit(self.run_dir, "task_expired", version=self.optimizer.step, task_id=task["task_id"],
                     attempt_id=task["attempt_id"], worker_id=owner.get("worker_id"), reason=reason)
                task["attempt_id"] = uuid.uuid4().hex
                task["attempt"] += 1
                self.stats["expired_tasks"] += 1
                warm_capacity = max(0, warm_capacity - 1)
                changed = True
        if changed:
            self.publish()

    def update(self):
        if self.tasks:
            return False
        milestone_steps = self.evaluation_checkpoint_steps()
        count = sum(v["samples"] for v in self.accepted.values())
        if count != self.config["batch_size"]:
            raise AssertionError(f"Update represents {count} samples instead of global batch")
        loss = sum(v["loss"] * v["samples"] for v in self.accepted.values()) / count
        info = self.optimizer.update(self.accumulated)
        self.stats["updates"] += 1
        emit(self.run_dir, "update", step=self.optimizer.step, loss=loss, batch_size=count,
             samples_consumed=self.optimizer.step * count, workers=len({v["worker_id"] for v in self.accepted.values()}),
             seconds=time.monotonic() - self.version_started, tasks=self.accepted,
             gradient_prefetch_workers=self.gradient_prefetch_workers, **self.ingestion_timings, **info)
        periodic = self.optimizer.step % int(self.config.get("checkpoint_steps", 20)) == 0
        if (periodic or self.optimizer.step in milestone_steps
                or time.monotonic() - self.last_checkpoint_time > float(self.config.get("checkpoint_seconds", 300))):
            self.save_checkpoint()
        if self.optimizer.step >= self.config["target_steps"]:
            self.complete()
        else:
            self.begin_version()
        return True

    def evaluation_checkpoint_steps(self):
        """The frozen evaluation_steps force saves and survive rolling pruning.

        Missing evaluation_steps preserves the historical checkpoint policy.
        Future milestones beyond a short validation target are harmless.
        """
        steps = self.config.get("evaluation_steps", [])
        if (not isinstance(steps, list) or any(type(step) is not int or step <= 0 for step in steps)
                or len(set(steps)) != len(steps)):
            raise ValueError("evaluation_steps must be a list of distinct positive integer updates")
        return frozenset(steps)

    def prune_checkpoints(self):
        """Retain the latest three committed snapshots and every evaluation one."""
        milestone_steps = self.evaluation_checkpoint_steps()
        for old in sorted((self.run_dir / "checkpoints").glob("step-*.json"))[:-3]:
            step = int(old.stem.removeprefix("step-"))
            if step in milestone_steps:
                continue
            old.with_suffix(".npz").unlink(missing_ok=True)
            old.unlink(missing_ok=True)

    def save_checkpoint(self):
        if self.optimizer.step == self.last_checkpoint_step:
            return
        milestone_steps = self.evaluation_checkpoint_steps()
        step = self.optimizer.step
        path = self.run_dir / "checkpoints" / f"step-{step:08d}.npz"
        sha = save_arrays(path, self.optimizer.checkpoint_arrays())
        metadata = {"step": step, "path": str(path), "sha256": sha,
                    "config_digest": self.config_digest, "initial_digest": self.initial_digest,
                    "schema": self.schema, "stats": self.stats, "samples_consumed": step * self.config["batch_size"]}
        atomic_json(path.with_suffix(".json"), metadata)
        atomic_json(self.run_dir / "checkpoint.json", metadata)
        self.last_checkpoint_step, self.last_checkpoint_time = step, time.monotonic()
        emit(self.run_dir, "checkpoint_committed", step=step, sha256=sha,
             evaluation_milestone=step in milestone_steps)
        self.prune_checkpoints()

    def complete(self):
        self.save_checkpoint()
        path = self.run_dir / "final_trainable.npz"
        sha = save_arrays(path, self.optimizer.arrays())
        self.active.update(status="complete", version=self.optimizer.step, tasks=[])
        self.tasks = []
        self.publish()
        summary = {"step": self.optimizer.step, "checkpoint_step": self.last_checkpoint_step,
                   "batch_size": self.config["batch_size"], "samples_consumed": self.optimizer.step * self.config["batch_size"],
                   "final_params": str(path), "sha256": sha, "config_digest": self.config_digest,
                   "initial_digest": self.initial_digest, "stats": self.stats, "completed_at": time.time()}
        atomic_json(self.run_dir / "COMPLETE.json", summary)
        emit(self.run_dir, "complete", **summary)

    def run(self):
        self.publish()
        emit(self.run_dir, "coordinator_started", generation=self.generation, config_digest=self.config_digest)
        while not self.stop_requested and not (self.run_dir / "STOP").exists():
            if self.optimizer is None:
                self.initialize()
            else:
                self.check_bootstraps()
                self.process_results()
                self.expire()
                self.update()
            if (self.run_dir / "COMPLETE.json").exists():
                return
            time.sleep(float(self.config.get("poll_seconds", 0.5)))
        if self.optimizer:
            self.save_checkpoint()
        self.active["status"] = "stopped"
        self.publish()
        emit(self.run_dir, "coordinator_stopped", step=self.optimizer.step if self.optimizer else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    configuration = read_json(args.run_dir / "run_config.json", {})
    threads = (args.threads if args.threads is not None else
               configuration.get("cpu_optimizer_threads", 16 if configuration.get("training_mode") == "full" else 8))
    if type(threads) is not int or not 1 <= threads <= 64:
        parser.error("CPU optimizer threads must be an integer in 1..64")
    torch.set_num_threads(threads)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.run_dir / "coordinator-lock"):
        coordinator_class = Coordinator
        if configuration.get("gradient_transport", "npz") == "sharded_blob_v1":
            from jade_lotus.sharded_coordinator import ShardedAtomicCoordinator

            coordinator_class = ShardedAtomicCoordinator
        elif configuration.get("gradient_transport", "npz") != "npz":
            raise ValueError("Unknown gradient transport")
        elif configuration.get("atomic_batching", False):
            from jade_lotus.atomic_coordinator import AtomicCoordinator

            coordinator_class = AtomicCoordinator
        coordinator = coordinator_class(args.run_dir)
        def stop(_signum, _frame):
            coordinator.stop_requested = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            coordinator.run()
        except BaseException as error:
            if not isinstance(error, (KeyboardInterrupt, SystemExit)):
                atomic_json(args.run_dir / "FATAL.json", {"type": type(error).__name__, "error": str(error), "time": time.time()})
            raise


if __name__ == "__main__":
    main()
