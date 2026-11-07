"""Interruptible GPU gradient worker for the coordinator's shared filesystem queue."""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import signal
import socket
import shutil
import threading
import time
import traceback
import uuid


def emit(event, **fields):
    print(json.dumps({"time": time.time(), "event": event, **fields}, sort_keys=True), flush=True)


def _publish_npu_launch(engine, client, config, run_dir, worker_id, started_at):
    """Record one actual device process, linked to its own submission identity."""
    from jade_lotus.protocol import read_json, safe_id
    from jade_lotus.runtime import atomic_json

    if config.get("backend") != "torch_npu":
        return
    prefix = os.environ.get("JADE_WORKER_PREFIX", "")
    submission = (read_json(Path(run_dir) / "ascend_fleet/jobs" / (safe_id(prefix) + ".json"), {})
                  if prefix else {})
    workspace = os.environ.get("JADE_WORKSPACE_ID") or submission.get("workspace_id")
    job = os.environ.get("JADE_JOB_ID") or submission.get("job_id")
    if submission and workspace and submission.get("workspace_id") != workspace:
        raise ValueError("Worker workspace differs from its recorded submission")
    local_trial = config.get("validation_run") is True and os.environ.get("JADE_LOCAL_PREFLIGHT") == "1"
    if not local_trial and (not workspace or not job or not submission):
        raise ValueError("NPU worker requires its own recorded workspace and platform job identity")
    device = engine.device
    if getattr(device, "platform", None) != "npu":
        raise ValueError("NPU worker launch did not initialize an NPU device")
    record = {"schema_version": 1, "backend": "torch_npu", "worker_id": worker_id,
              "session_id": client.session_id, "worker_prefix": prefix,
              "workspace_id": workspace or "local-preflight", "job_id": job or "local-preflight",
              "assigned_device_id": os.environ.get("JADE_NPU_ASSIGNED_DEVICE_ID", str(device.id)),
              "assigned_device_id_source": os.environ.get("JADE_NPU_ASSIGNED_DEVICE_ID_SOURCE", "logical_slot"),
              "visible_device_count": int(os.environ.get("JADE_NPU_VISIBLE_DEVICE_COUNT", "1")),
              "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "ascend_visible_devices": os.environ.get("ASCEND_VISIBLE_DEVICES"),
              "logical_device_index": device.id, "device_kind": device.device_kind,
              "hostname": socket.gethostname(), "pid": os.getpid(), "started_at": started_at,
              "initialized_at": time.time(), "config_digest": client.config_digest,
              "source_digest": config.get("source_digest"), "gradient_checkpointing": False,
              "local_preflight": local_trial}
    directory = Path(run_dir) / "launches"
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / (safe_id(client.session_id) + ".json"), record)
    emit("npu_worker_launch_attested", **record)


class ExportLease:
    """A fenced export claim that survives the owner's abrupt destruction.

    Export output stays beneath the random token's directory. Reclaiming that
    directory removes the old owner's publish source atomically, rather than
    relying on a token check followed by an unfenced rename from elsewhere.
    """

    def __init__(self, run_dir, worker_id, session_id, stale_seconds=120):
        self.run_dir = Path(run_dir)
        self.worker_id = worker_id
        self.session_id = session_id
        self.stale_seconds = stale_seconds
        self.root = self.run_dir / "export.claim"
        self.token = uuid.uuid4().hex
        self.directory = self.root / self.token

    def acquire(self):
        from jade_lotus.protocol import atomic_json

        # Prepare the entire generation before it becomes the public claim.
        # POSIX rename cannot replace a nonempty directory, so competing fully
        # formed claims have one winner. There is no half-created owner window.
        proposal = self.run_dir / f".export-proposal-{self.token}"
        proposal_token = proposal / self.token
        proposal_token.mkdir(parents=True)
        try:
            atomic_json(proposal_token / "owner.json", {
                "worker_id": self.worker_id, "session_id": self.session_id,
                "token": self.token, "time": time.time(), "pid": os.getpid(),
            })
            try:
                proposal.rename(self.root)
            except OSError as exc:
                if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                    return False
                raise
        finally:
            shutil.rmtree(proposal, ignore_errors=True)
        return self.owns()

    def owns(self):
        from jade_lotus.protocol import read_json

        owner = read_json(self.directory / "owner.json", {})
        return (owner.get("token") == self.token and owner.get("worker_id") == self.worker_id
                and owner.get("session_id") == self.session_id)

    def reclaim_stale(self, now=None):
        from jade_lotus.protocol import read_json

        now = time.time() if now is None else now
        try:
            entries = [path for path in self.root.iterdir() if path.is_dir()]
            root_age = now - self.root.stat().st_mtime
        except FileNotFoundError:
            return False
        if not entries:
            # The previous process may have died between the two mkdir calls.
            if root_age > self.stale_seconds:
                try:
                    self.root.rmdir()
                except (FileNotFoundError, OSError):
                    return False
                return True
            return False
        reclaimed = False
        for directory in entries:
            owner = read_json(directory / "owner.json", {})
            try:
                created = float(owner.get("time", directory.stat().st_mtime))
            except FileNotFoundError:
                continue
            heartbeat = read_json(self.run_dir / "workers" / f"{owner.get('worker_id', '')}.json", {})
            heartbeat_matches = (heartbeat.get("session_id") == owner.get("session_id")
                                 and heartbeat.get("export_token") == directory.name
                                 and heartbeat.get("stage") == "exporting")
            last_alive = max(created, float(heartbeat.get("time", 0)) if heartbeat_matches else 0)
            if now - last_alive <= self.stale_seconds:
                continue
            abandoned = self.run_dir / f".export-abandoned-{directory.name}-{uuid.uuid4().hex}"
            try:
                # The random source component fences concurrent reclaimers:
                # only one can move this exact owner's lease. A replacement
                # owner has a different token and cannot be moved by this call.
                directory.rename(abandoned)
            except FileNotFoundError:
                continue
            reclaimed = True
            emit("export_claim_reclaimed", old_worker_id=owner.get("worker_id"),
                 old_token=directory.name, abandoned=str(abandoned))
        if reclaimed:
            try:
                self.root.rmdir()  # Only succeeds if no other lease remains.
            except (FileNotFoundError, OSError):
                pass
        return reclaimed

    def abandon_own(self):
        if not self.owns():
            return
        abandoned = self.run_dir / f".export-failed-{self.token}-{uuid.uuid4().hex}"
        try:
            self.directory.rename(abandoned)
        except FileNotFoundError:
            return
        try:
            self.root.rmdir()
        except (FileNotFoundError, OSError):
            pass
        shutil.rmtree(abandoned, ignore_errors=True)

    def publish(self, destination):
        if not self.owns():
            return False
        try:
            (self.directory / "export.tmp").rename(destination)
        except FileNotFoundError:
            # The claimant may have expired between owns() and rename(); its
            # unique source path disappears as part of atomic reclamation.
            return False
        return True


def finish_export(engine, client, config, run_dir, worker_id, active, stopping, poll_seconds):
    """Wait for a live exporter or reclaim its stale lease and finish export."""
    from jade_lotus.protocol import atomic_json, load_arrays, read_json

    if config.get("skip_final_export", False):
        return
    complete = run_dir / "COMPLETE.json"
    final_export = run_dir / "final_export"
    exported = final_export / "EXPORT_COMPLETE.json"
    if not complete.is_file():
        return
    while not stopping.is_set():
        if (run_dir / "STOP").exists() or (run_dir / "FATAL.json").exists():
            return
        if exported.is_file():
            emit("export_available", worker_id=worker_id, path=str(final_export))
            return
        active.clear()
        active.update(stage="waiting_for_export")
        lease = ExportLease(run_dir, worker_id, client.session_id,
                            stale_seconds=float(config.get("export_lease_seconds", 120)))
        if not lease.acquire():
            lease.reclaim_stale()
            stopping.wait(poll_seconds)
            continue
        active.clear()
        active.update(stage="exporting", export_token=lease.token)
        client.heartbeat("exporting", export_token=lease.token)
        try:
            completion = read_json(complete)
            final_params = load_arrays(completion["final_params"], completion.get("sha256"))
            temporary_export = lease.directory / "export.tmp"
            engine.export(final_params, temporary_export)
            validation = read_json(temporary_export / "EXPORT_VALIDATION.json", {})
            atomic_json(temporary_export / "EXPORT_COMPLETE.json", {
                "worker_id": worker_id, "step": completion["step"],
                "batch_size": completion["batch_size"],
                "training_mode": config.get("training_mode", "lora"),
                "lora_merged": config.get("training_mode", "lora") == "lora",
                "claim_token": lease.token, "time": time.time(), "validation": validation,
            })
            if stopping.is_set() or (run_dir / "STOP").exists() or (run_dir / "FATAL.json").exists():
                lease.abandon_own()
                return
            if lease.publish(final_export):
                emit("export_complete", worker_id=worker_id, path=str(final_export))
                lease.abandon_own()
                return
            emit("export_claim_lost", worker_id=worker_id, token=lease.token)
        except Exception:
            lease.abandon_own()
            raise


def _prewarm_before_admission(config):
    enabled = config.get("worker_prewarm_before_admission", False)
    if type(enabled) is not bool:
        raise ValueError("worker_prewarm_before_admission must be a bool")
    if enabled and (config.get("training_mode") != "full"
                    or config.get("atomic_batching") is not True
                    or config.get("verify_first_task_replay") is not True):
        raise ValueError("worker_prewarm_before_admission requires full atomic training and verify_first_task_replay=true")
    return enabled


def _reducer_workers(config):
    """Validate the opt-in before clients, leases, threads or GPU work exist."""
    workers = config.get("reducer_workers", 1)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("reducer_workers must be an integer in 1..4")
    if "reducer_workers" in config:
        if (config.get("training_mode") != "full" or config.get("atomic_batching") is not True
                or config.get("gradient_transport") != "sharded_blob_v1"):
            raise ValueError("reducer_workers requires full atomic training and sharded_blob_v1")
        shard = config.get("gradient_shard_bytes", 256 * 1024 * 1024)
        if type(shard) is not int or not 4 <= shard <= 256 * 1024 * 1024 or shard % 4:
            raise ValueError("reducer_workers requires FP32-aligned shards no larger than 256 MiB")
    return workers


def _gradient_wire_dtype(config):
    """Reject an unsupported codec before model loading or worker admission."""
    mode = config.get("gradient_wire_dtype", "float32")
    from jade_lotus.array_blob import WIRE_DTYPES
    if mode not in WIRE_DTYPES:
        raise ValueError("Unsupported gradient_wire_dtype")
    if mode != "float32":
        from jade_lotus.sharded_transport import require_sharded

        require_sharded(config)
    from jade_lotus.gradient_export import gradient_export_mode

    gradient_export_mode(config)
    return mode


class _ReducerPool:
    """Launch bounded, independent instances of the existing reducer service.

    Each slot owns its current claim and renewal thread. Only slot zero uses
    the original GPU-worker session; extra slots have separate heartbeat files.
    No gradient tree is duplicated: each service reduces one bounded shard at
    a time with the existing ordered FP32 reduction implementation.
    """

    def __init__(self, run_dir, worker_id, session_id, config, stopping):
        from jade_lotus.protocol import safe_id
        from jade_lotus.sharded_transport import ReducerService

        workers = _reducer_workers(config)
        sessions = [session_id] + [safe_id(f"{session_id}-reducer-{slot}") for slot in range(1, workers)]
        self.services = [ReducerService(run_dir, worker_id, session) for session in sessions]
        self.stopping = stopping
        self.poll_seconds = float(config.get("reducer_poll_seconds", 0.2))
        self.errors = []
        self._error_lock = threading.Lock()
        self.threads = []
        self._renewal_threads = set()
        self._started = False
        for slot, service in enumerate(self.services):
            self._guard_heartbeat(slot, service)

    def _record_error(self, slot, service, error):
        with self._error_lock:
            self.errors.append(error)
        emit("reducer_fatal", worker_id=service.worker_id, session_id=service.session_id,
             reducer_slot=slot, error=repr(error), traceback=traceback.format_exc())
        self.stopping.set()

    def _guard_heartbeat(self, slot, service):
        original = service.heartbeat

        def heartbeat():
            current_thread = threading.current_thread()
            if current_thread.name == "raw-reducer-lease":
                with self._error_lock:
                    self._renewal_threads.add(current_thread)
            try:
                return original()
            except Exception as error:
                # The native renewal thread does not return its exceptions to
                # .run(). Record them here and stop every slot; .run() still
                # owns the renewal lifecycle and stop-fenced commit behavior.
                self._record_error(slot, service, error)

        service.heartbeat = heartbeat

    def _run(self, slot, service):
        try:
            service.run(self.stopping, poll_seconds=self.poll_seconds)
        except Exception as error:
            self._record_error(slot, service, error)

    def start(self):
        if self._started:
            raise RuntimeError("The bounded reducer pool can only be started once")
        self._started = True
        for slot, service in enumerate(self.services):
            thread = threading.Thread(target=self._run, args=(slot, service),
                                      name="gradient-reducer" if slot == 0 else f"gradient-reducer-{slot}",
                                      daemon=True)
            thread.start()
            self.threads.append(thread)

    def join(self):
        for thread in self.threads:
            thread.join()
        # Native .run() uses a timed renewal join. If a heartbeat was inside
        # filesystem I/O at shutdown, finish joining that known thread too.
        with self._error_lock:
            renewals = tuple(self._renewal_threads)
        for thread in renewals:
            thread.join()

    def raise_if_failed(self):
        with self._error_lock:
            error = self.errors[0] if self.errors else None
        if error is not None:
            raise RuntimeError("The background gradient reducer failed") from error


def _release_gradient_snapshot(gradients):
    from jade_lotus.native_snapshot import NativeSnapshot

    if isinstance(gradients, NativeSnapshot):
        gradients.close()


def _verify_gradient_replay(engine, params, version, task_id, sample_positions, seed,
                            gradients, metrics, *, training_mode):
    """Keep the existing replay thresholds while releasing its temporary tree."""
    import numpy as np

    repeated = repeated_metrics = None
    try:
        repeated, repeated_metrics = engine.compute(params, version, task_id, sample_positions, seed)
        from jade_lotus.native_snapshot import NativeSnapshot

        native = isinstance(gradients, NativeSnapshot) or isinstance(repeated, NativeSnapshot)
        if training_mode == "full":
            if native:
                from jade_lotus.native_numeric_validation import validate_gradient_replay
            else:
                from jade_lotus.numeric_validation import validate_gradient_replay

            replay_report = validate_gradient_replay(gradients, repeated, training_mode="full")
            if not replay_report["passed"]:
                raise FloatingPointError(f"Full task replay gradient mismatch: {replay_report}")
            metrics["replay_relative_l2_grad_error"] = replay_report["relative_l2_error"]
        if native:
            from jade_lotus.native_numeric_validation import iter_gradient_chunks

            max_error, exact, close = 0.0, True, True
            for _name, left, right in iter_gradient_chunks(gradients, repeated):
                max_error = max(max_error, float(np.max(np.abs(left - right))))
                exact &= np.array_equal(left, right)
                close &= np.allclose(left, right, rtol=1e-5, atol=1e-6)
        else:
            max_error = max(float(np.max(np.abs(gradients[key] - repeated[key]))) for key in gradients)
            exact = all(np.array_equal(gradients[key], repeated[key]) for key in gradients)
            close = all(np.allclose(gradients[key], repeated[key], rtol=1e-5, atol=1e-6) for key in gradients)
        if not close:
            raise FloatingPointError(f"Task replay gradient mismatch: max_abs={max_error}")
        metrics.update(replay_max_abs_grad_error=max_error, replay_gradients_bitwise_equal=int(exact),
                       replay_loss_abs_error=abs(metrics["loss"] - repeated_metrics["loss"]))
        return {"max_abs_grad_error": max_error, "gradients_bitwise_equal": exact}
    finally:
        # A full host gradient is 13.4 GB. Neither a successful replay nor this
        # helper's exception frame should keep an extra tree between tasks.
        _release_gradient_snapshot(repeated)
        del repeated, repeated_metrics, gradients


def _prewarm_worker(engine, initial, config, client, active):
    """Compute and replay local samples before publishing any bootstrap.

    Explicit sample positions do not advance the authoritative sampler. These
    gradients are discarded without claiming or submitting a training task.
    The first real assignment still loads and fences its current parameters.
    """
    capacity = engine.task_batch_capacity
    if type(capacity) is not int or capacity not in (4, 8, 16):
        raise ValueError("Worker prewarm requires a logical batch capacity of 4, 8 or 16")
    positions, seed = list(range(capacity)), config.get("seed", 42)
    task_id = "prewarm-" + client.session_id
    fields = {"worker_id": client.worker_id, "session_id": client.session_id,
              "task_id": task_id, "version": 0, "sample_positions": positions,
              "seed": seed, "training_mode": "full"}
    active.clear()
    active.update(stage="prewarming")
    client.heartbeat("prewarming")
    emit("worker_prewarm_started", **fields)
    began = time.monotonic()
    gradients = metrics = None
    try:
        gradients, metrics = engine.compute(initial, 0, task_id, positions, seed)
        _verify_gradient_replay(engine, initial, 0, task_id, positions, seed,
                                gradients, metrics, training_mode="full")
        emit("worker_prewarm_verified", **fields, wall_seconds=time.monotonic() - began, **metrics)
    except Exception as error:
        emit("worker_prewarm_failed", **fields, error=repr(error))
        raise
    finally:
        _release_gradient_snapshot(gradients)
        del gradients, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker-id")
    parser.add_argument("--lease-seconds", type=float, default=600)
    parser.add_argument("--heartbeat-seconds", type=float, default=20)
    parser.add_argument("--poll-seconds", type=float, help="Override run_config worker_poll_seconds (default 2)")
    parser.add_argument("--device-index", type=int)
    args = parser.parse_args()
    worker_id = args.worker_id or f"{socket.gethostname()}-gpu{os.environ.get('JADE_GPU_SLOT', '0')}-{os.getpid()}"
    run_dir = args.run_dir.resolve()
    stopping = threading.Event()

    def on_signal(signum, _frame):
        emit("stop_signal", signal=signum, worker_id=worker_id)
        stopping.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    worker_started_at = time.time()
    emit("worker_start", worker_id=worker_id, run_dir=str(run_dir),
         cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
         ascend_visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    while not (run_dir / "run_config.json").is_file():
        if stopping.wait(2 if args.poll_seconds is None else args.poll_seconds):
            return
    config = json.loads((run_dir / "run_config.json").read_text())
    args.poll_seconds = float(config.get("worker_poll_seconds", 2) if args.poll_seconds is None else args.poll_seconds)
    if not 0 < args.poll_seconds <= 60:
        raise ValueError("worker polling interval must be in (0, 60] seconds")
    prewarm_before_admission = _prewarm_before_admission(config)
    _reducer_workers(config)
    _gradient_wire_dtype(config)
    from jade_lotus.protocol import WorkerClient
    from jade_lotus.engine_factory import create_gradient_engine

    transport = config.get("gradient_transport", "npz")
    if transport == "sharded_blob_v1":
        from jade_lotus.sharded_transport import RawWorkerClient

        client = RawWorkerClient(run_dir, worker_id, lease_seconds=args.lease_seconds)
        reducers = _ReducerPool(run_dir, client.worker_id, client.session_id, config, stopping)
    elif transport == "npz":
        client = WorkerClient(run_dir, worker_id, lease_seconds=args.lease_seconds)
        reducers = None
    else:
        raise ValueError(f"Unknown gradient transport: {transport}")
    active = {"stage": "initializing"}

    def heartbeat_loop():
        while not stopping.is_set():
            try:
                snapshot = dict(active)
                stage = snapshot.pop("stage", "alive")
                client.heartbeat(stage, **snapshot)
            except Exception as exc:
                emit("heartbeat_error", worker_id=worker_id, error=str(exc))
            stopping.wait(args.heartbeat_seconds)

    heartbeat = threading.Thread(target=heartbeat_loop, name="lease-renewal", daemon=True)
    worker_error = None
    try:
        heartbeat.start()
        if reducers is not None:
            reducers.start()
        engine = create_gradient_engine(config, device_index=args.device_index)
        _publish_npu_launch(engine, client, config, run_dir, worker_id, worker_started_at)
        if config.get("atomic_batching", False):
            client.set_capacity(engine.task_batch_capacity, engine.device.device_kind)
        initial = engine.initial_params()
        replay_verified = False
        if prewarm_before_admission:
            if stopping.is_set() or client.done():
                return
            _prewarm_worker(engine, initial, config, client, active)
            replay_verified = True
            if stopping.is_set() or client.done():
                emit("worker_prewarm_admission_cancelled", worker_id=worker_id)
                return
        client.initialize(initial)
        emit("bootstrap_published", worker_id=worker_id, parameters=sum(value.size for value in initial.values()))
        del initial
        cached_token, cached_params = None, None
        consecutive_failures = 0
        while not stopping.is_set():
            active.clear()
            active.update(stage="polling")
            if client.done():
                emit("training_complete", worker_id=worker_id)
                finish_export(engine, client, config, run_dir, worker_id, active, stopping, args.poll_seconds)
                break
            task = client.claim()
            if task is None:
                stopping.wait(args.poll_seconds)
                continue
            active.clear()
            active.update(stage="computing", task_id=task["task_id"], attempt_id=task["attempt_id"], version=task["version"])
            emit("task_claimed", worker_id=worker_id, task_id=task["task_id"], attempt_id=task["attempt_id"],
                 version=task["version"], samples=len(task["sample_positions"]))
            started = time.monotonic()
            gradients = None
            try:
                parameter_load_started = time.monotonic()
                token = (task["generation"], task["version"], task["params_sha256"])
                if cached_token != token:
                    cached_params = client.load_params(task)
                    # A restarted coordinator may republish an equal step number
                    # under a new generation. Fence the engine's device cache
                    # with the authoritative checksum as well as its version.
                    engine.invalidate_parameter_cache()
                    cached_token = token
                parameter_load_seconds = time.monotonic() - parameter_load_started
                gradients, metrics = engine.compute(cached_params, task["version"], task["task_id"],
                                                    task["sample_positions"], task["seed"])
                metrics["parameter_load_seconds"] = parameter_load_seconds
                if config.get("verify_first_task_replay", False) and not replay_verified:
                    active["stage"] = "verifying_replay"
                    replay = _verify_gradient_replay(engine, cached_params, task["version"], task["task_id"],
                        task["sample_positions"], task["seed"], gradients, metrics,
                        training_mode=config.get("training_mode", "lora"))
                    replay_verified = True
                    emit("task_replay_verified", worker_id=worker_id, task_id=task["task_id"],
                         **replay)
                active["stage"] = "submitting"
                if reducers is not None:
                    reducers.raise_if_failed()
                submitted = client.submit(task, gradients, metrics)
                emit("task_result", worker_id=worker_id, task_id=task["task_id"], attempt_id=task["attempt_id"],
                     version=task["version"], submitted=submitted, wall_seconds=time.monotonic() - started, **metrics)
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                emit("task_error", worker_id=worker_id, task_id=task["task_id"], attempt_id=task["attempt_id"],
                     version=task["version"], error=repr(exc), traceback=traceback.format_exc(),
                     consecutive_failures=consecutive_failures)
                # A failed attempt naturally expires and is reassigned. Fatal
                # numerical/schema errors must be visible rather than hot-loop.
                if isinstance(exc, (FloatingPointError, ValueError, KeyError)) or consecutive_failures >= 3:
                    raise
                stopping.wait(min(30, consecutive_failures * 5))
            finally:
                _release_gradient_snapshot(gradients)
                del gradients
        if reducers is not None:
            reducers.raise_if_failed()
        active.clear()
        active.update(stage="finished")
        client.heartbeat("finished")
    except Exception as exc:
        worker_error = exc
        emit("worker_fatal", worker_id=worker_id, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        stopping.set()
        if heartbeat.ident is not None:
            heartbeat.join()
        if reducers is not None:
            reducers.join()
            # A late reducer failure must also defeat early returns (including
            # cancelled admission), without hiding an existing foreground error.
            if worker_error is None:
                reducers.raise_if_failed()
    emit("worker_exit", worker_id=worker_id)


if __name__ == "__main__":
    main()
