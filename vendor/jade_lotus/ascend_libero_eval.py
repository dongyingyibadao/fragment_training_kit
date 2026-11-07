"""Resumable LIBERO rollouts with one NPU policy and several CPU simulators.

Each simulator has a private subprocess and a durable per-episode POSIX lock.
Only the NPU owner's main thread calls the policy. Explicit episode noise makes
the result independent of simulator scheduling. Infrastructure exceptions never
become completed model failures.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future
import contextlib
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
import uuid

import numpy as np

from jade_lotus.evaluate import Simulator, SUITE_STEPS, _posix_file_lock, preprocess_observation
from jade_lotus.protocol import file_sha256
from jade_lotus.runtime import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def save_failure_video(path, frames, *, fps):
    """Encode in the simulator lane with one CPU thread and publish durably."""
    import imageio.v2 as imageio
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.stem, suffix=".mp4", dir=path.parent)
    os.close(descriptor)
    try:
        imageio.mimwrite(temporary, frames, fps=fps, codec="libx264", quality=7,
                         macro_block_size=16, ffmpeg_params=["-threads", "1", "-movflags", "+faststart"])
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        digest = file_sha256(temporary)
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return digest
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_sources(directory, config):
    unsigned = {k: v for k, v in config.items() if k != "config_digest"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() != config["config_digest"]:
        raise ValueError("Evaluation configuration changed")
    manifest = read(directory / "source_manifest.json")
    encoded = json.dumps(manifest, sort_keys=True).encode()
    if hashlib.sha256(encoded).hexdigest() != config["source_digest"]:
        raise ValueError("Evaluation source manifest changed")
    for name, digest in manifest.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory / "runtime") or file_sha256(path) != digest:
            raise ValueError(f"Frozen evaluation source changed: {name}")
    if file_sha256(Path(config["libero_config_path"]) / "config.yaml") != config["libero_config_sha256"]:
        raise ValueError("Frozen LIBERO asset-path configuration changed")
    for name, digest in config.get("simulator_overlay_sha256", {}).items():
        if file_sha256(name) != digest:
            raise ValueError(f"Isolated simulator dependency changed: {name}")


def prepare(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    training = args.training_run.resolve()
    train_config = read(training / "run_config.json")
    checkpoint = training / "exports/step-20000"
    receipt = read(checkpoint / "export_receipt.json")
    if receipt["status"] != "passed" or receipt["step"] != 20000:
        raise ValueError("Expected the verified final step-20000 export")
    if file_sha256(checkpoint / "model.safetensors") != receipt["model"]["sha256"]:
        raise ValueError("Export model checksum differs from its receipt")
    for name, digest in receipt["assets_sha256"].items():
        if file_sha256(checkpoint / name) != digest:
            raise ValueError(f"Export asset changed: {name}")
    base = Path(__file__).resolve().parents[2]
    for name in ("results", "videos", "rollouts", "errors", "episode_locks", "worker_logs", "workers"):
        (directory / name).mkdir()
    config = {
        "schema": "jade-ascend-libero-pipeline-v1", "created_at": stamp(),
        "mode": "smoke" if args.smoke else "benchmark", "training_run": str(training),
        "training_step": 20000, "checkpoint": str(checkpoint),
        "checkpoint_sha256": receipt["model"]["sha256"],
        "export_receipt_sha256": file_sha256(checkpoint / "export_receipt.json"),
        "openpi_root": train_config["openpi_root"], "openpi_config": train_config["openpi_config"],
        "tokenizer_path": train_config["tokenizer_path"],
        "simulator_python": str(base / "openpi/.venv-ascend/bin/python"),
        "libero_root": "/opt/ascend-vla/envs/lerobot-npu/lib/python3.12/site-packages",
        "libero_config_path": str(base / "workspace/config/libero"),
        "simulator_timeout_seconds": 300, "seed": 7,
        "simulator_seed_scheme": "constant seed 7 at each reset, official initial state index 0..49",
        "noise_scheme": "numpy PCG64 seeded by first 8 little-endian bytes of SHA256('7:<episode_id>'); explicit float32 50x32 noise per policy call",
        "settle_steps": 10, "replan_steps": 5, "denoise_steps": 10,
        "action_horizon": 50, "action_dimension": 32, "step_limits": SUITE_STEPS,
        "success_criterion": "LIBERO env.check_success() after each physical action",
        "trials_per_task": 1 if args.smoke else 50,
        "simulators_per_npu": args.simulators_per_npu,
        "cache_intermediate_images": True,
        "failure_video": "dual view, 224x448, policy input boundaries, 4 FPS",
        "infra_error_policy": "preserve attempt separately, retry same episode/initial-state/noise; never count as model failure",
        "model_invalid_action_policy": "record as model failure with evidence; no success-filtering or retries",
    }
    if args.simulator_overlay:
        overlay = args.simulator_overlay.resolve()
        config["libero_root"] = str(overlay)
        config["simulator_overlay_sha256"] = {str(p): file_sha256(p) for p in sorted(overlay.rglob("*.py"))}
    (directory / "libero_config").mkdir()
    shutil.copy2(Path(config["libero_config_path"]) / "config.yaml", directory / "libero_config/config.yaml")
    config["libero_config_path"] = str(directory / "libero_config")
    config["libero_config_sha256"] = file_sha256(directory / "libero_config/config.yaml")
    simulator = Simulator(config, directory, "prepare")
    try:
        description = simulator.request("describe", suites=list(SUITE_STEPS))
    finally:
        simulator.close()
    atomic_json(directory / "simulator_description.json", description)
    episodes = []
    for suite, tasks in description["suites"].items():
        for task_id in range(10):
            if args.smoke and task_id != 0:
                continue
            task = tasks[str(task_id)]
            for trial in range(config["trials_per_task"]):
                if trial >= task["initial_states"]:
                    raise ValueError("Insufficient official initial states")
                episode_id = f"{suite}-task-{task_id:02d}-episode-{trial:02d}"
                episodes.append({"episode_id": episode_id, "suite": suite, "task_id": task_id,
                                 "trial": trial, "prompt": task["language"], "max_steps": SUITE_STEPS[suite]})
    config["episodes"] = episodes
    config["planned_episodes"] = len(episodes)
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "jade_lotus", directory / "runtime/jade_lotus",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (directory / "runtime/scripts").mkdir()
    shutil.copy2(root / "scripts/evaluate_ascend_libero.py", directory / "runtime/scripts/evaluate_ascend_libero.py")
    manifest = {str(p.relative_to(directory)): file_sha256(p)
                for p in sorted((directory / "runtime").rglob("*.py"))}
    config["source_digest"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    config["config_digest"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    atomic_json(directory / "source_manifest.json", manifest)
    atomic_json(directory / "eval_config.json", config)
    print(json.dumps({"prepared": str(directory), "episodes": len(episodes), "mode": config["mode"]}))


class QueuedPolicy:
    def __init__(self, requests, stop):
        self.requests, self.stop = requests, stop

    def infer(self, observation, *, noise):
        if self.stop.is_set():
            raise InterruptedError("Policy owner stopped")
        future = Future()
        self.requests.put((observation, noise, future, time.monotonic()))
        return future.result(timeout=300)


def rollout(policy, simulator, item, config):
    started = time.monotonic()
    reset = simulator.request("reset", suite=item["suite"], task_id=item["task_id"],
                              trial=item["trial"], seed=config["seed"],
                              camera_cache=config["cache_intermediate_images"])
    if reset["prompt"] != item["prompt"]:
        raise ValueError("LIBERO task language changed")
    for i in range(config["settle_steps"]):
        settled = simulator.request("step", action=[0.] * 6 + [-1.],
                                    capture_images=i == config["settle_steps"] - 1)
    if settled["success"]:
        raise ValueError("Initial state was already successful before policy execution")
    obs = settled["observation"]
    noise_seed = int.from_bytes(hashlib.sha256(f"{config['seed']}:{item['episode_id']}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(noise_seed)
    frames, frame_steps, actions, chunks, infer_seconds = [], [], [], [], []
    success, status, invalid = False, "timeout", None
    for first_step in range(0, item["max_steps"], config["replan_steps"]):
        element = preprocess_observation(obs, item["prompt"])
        frames.append(np.concatenate([element["observation/image"], element["observation/wrist_image"]], axis=1))
        frame_steps.append(len(actions))
        before = time.monotonic()
        chunk = np.asarray(policy.infer(element, noise=rng.standard_normal((50, 32)).astype(np.float32)))
        infer_seconds.append(time.monotonic() - before)
        chunks.append(chunk)
        if chunk.shape != (50, 7) or not np.isfinite(chunk).all():
            status, invalid = "invalid_model_action", {"shape": list(chunk.shape), "finite": bool(np.isfinite(chunk).all()),
                                                       "policy_call": len(infer_seconds)}
            break
        amount = min(config["replan_steps"], item["max_steps"] - first_step)
        for i, action in enumerate(chunk[:amount]):
            result = simulator.request("step", action=action.tolist(), capture_images=i == amount - 1)
            actions.append(action)
            obs = result["observation"]
            success = bool(result["success"])
            if success or result["done"]:
                status = "success" if success else "terminated_without_success"
                break
        if success or status == "terminated_without_success":
            break
    record = {**item, "success": success, "status": status, "policy_steps": len(actions),
              "environment_steps": config["settle_steps"] + len(actions), "policy_calls": len(infer_seconds),
              "inference_and_queue_seconds": infer_seconds, "noise_seed": noise_seed,
              "renderer": reset["renderer"], "camera_cache": reset["camera_cache"],
              "rollout_seconds": time.monotonic() - started, "invalid_model_action": invalid,
              "video_sampling": {"fps": reset["control_frequency_hz"] / config["replan_steps"],
                                 "frame_policy_steps": frame_steps, "views": ["agentview", "wrist"]}}
    return record, frames, np.asarray(actions), chunks


def claim_next(directory, pending, stop):
    """Retry busy or interrupted work without losing it after a single scan."""
    while pending and not stop.is_set():
        remaining = []
        for item in pending:
            if stop.is_set() or (directory / "STOP").exists() or (directory / "FATAL.json").exists():
                return
            destination = directory / "results" / (item["episode_id"] + ".json")
            if destination.exists():
                continue
            with _posix_file_lock(directory / "episode_locks" / (item["episode_id"] + ".lock"), blocking=False) as acquired:
                if not acquired:
                    remaining.append(item)
                elif not destination.exists():
                    # Keep this lock across the caller's rollout and atomic commit.
                    yield item, destination
                    if not destination.exists():
                        remaining.append(item)
        pending = remaining
        if pending:
            stop.wait(2)


def worker(args):
    directory = args.output.resolve()
    config = read(directory / "eval_config.json")
    verify_sources(directory, config)
    if not 1 <= config["simulators_per_npu"] <= 4:
        raise ValueError("Expected 1..4 CPU simulators per NPU")
    import torch
    import torch_npu  # noqa: F401
    from jade_lotus.ascend_eval_policy import load_policy
    device = f"npu:{args.device_index}"
    if not torch.npu.is_available():
        raise RuntimeError("The assigned NPU is unavailable")
    torch.npu.set_device(device)
    worker_id = f"{args.worker_id}-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    worker_dir = directory / "workers" / worker_id
    worker_dir.mkdir(exist_ok=False)
    atomic_json(worker_dir / "started.json", {"at": stamp(), "pid": os.getpid(), "host": socket.gethostname(),
                "device": device, "visible_devices": torch.npu.device_count(),
                "device_name": torch.npu.get_device_name(args.device_index),
                "device_mapping": {k: os.environ.get(k) for k in ("ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")},
                "cpu_affinity": sorted(os.sched_getaffinity(0)), "simulators": config["simulators_per_npu"]})
    policy = load_policy(config["openpi_root"], config["checkpoint"], device,
                         expected_sha=config["checkpoint_sha256"],
                         tokenizer_path=config["tokenizer_path"], receipt_path=worker_dir / "load_receipt.json")
    atomic_json(worker_dir / "model_ready.json", {"at": stamp(), "device": device,
                "memory_allocated_bytes": torch.npu.memory_allocated(device)})
    load_receipt_sha256 = file_sha256(worker_dir / "load_receipt.json")
    requests = queue.Queue()
    stop = threading.Event()
    proxy = QueuedPolicy(requests, stop)
    errors = []

    def lane(number):
        simulator = None
        completed = 0
        try:
            simulator = Simulator(config, directory, f"{worker_id}-lane{number}")
            # Rotate by complete tasks so a lane can reuse its current simulator.
            offset = int.from_bytes(hashlib.sha256(f"{worker_id}:{number}".encode()).digest()[:4], "little")
            offset = (offset % 40) * config["trials_per_task"]
            items = config["episodes"][offset:] + config["episodes"][:offset]
            for item, destination in claim_next(directory, items, stop):
                attempt = uuid.uuid4().hex
                before = stamp()
                try:
                    record, frames, actions, chunks = rollout(proxy, simulator, item, config)
                    record.update(worker_id=worker_id, lane=number, attempt_id=attempt,
                                  started_at=before, finished_at=stamp(), checkpoint_sha256=config["checkpoint_sha256"],
                                      source_digest=config["source_digest"], config_digest=config["config_digest"],
                                      load_receipt_sha256=load_receipt_sha256)
                    if not record["success"]:
                        relative = Path("videos") / (item["episode_id"] + ".mp4")
                        record["video"] = str(relative)
                        record["video_sha256"] = save_failure_video(directory / relative, frames, fps=4)
                        record["video_frames"] = len(frames)
                        payload = {f"chunk_{i:03d}": chunk for i, chunk in enumerate(chunks)}
                        np.savez_compressed(directory / "rollouts" / (item["episode_id"] + ".npz"),
                                            actions=actions, **payload)
                    record["finished_at"] = stamp()
                    atomic_json(destination, record)
                    completed += 1
                    print(json.dumps({"event": "episode", "id": item["episode_id"], "success": record["success"],
                                      "steps": record["policy_steps"], "seconds": record["rollout_seconds"]}), flush=True)
                except Exception:
                    error = {**item, "worker_id": worker_id, "lane": number, "attempt_id": attempt,
                             "started_at": before, "at": stamp(), "kind": "infrastructure_exception",
                             "traceback": traceback.format_exc()}
                    atomic_json(directory / "errors" / f"{item['episode_id']}-{attempt}.json", error)
                    errors.append(error)
                    print(json.dumps({"event": "infrastructure_error", "id": item["episode_id"],
                                      "traceback": error["traceback"]}), flush=True)
                    # Restart a damaged simulator, retaining the episode as unfinished.
                    simulator.close()
                    simulator = Simulator(config, directory, f"{worker_id}-lane{number}")
                    if len(errors) >= 3:
                        stop.set()
                        break
        except Exception:
            errors.append({"lane": number, "traceback": traceback.format_exc()})
            stop.set()
        finally:
            if simulator is not None:
                simulator.close()
            atomic_json(worker_dir / f"lane{number}.json", {"completed": completed, "errors": len(errors), "at": stamp()})

    lanes = [threading.Thread(target=lane, args=(i,), name=f"simulator-{i}")
             for i in range(config["simulators_per_npu"])]
    for thread in lanes:
        thread.start()
    timings = []
    while any(thread.is_alive() for thread in lanes) or not requests.empty():
        try:
            obs, noise, future, enqueued = requests.get(timeout=.2)
        except queue.Empty:
            continue
        before = time.monotonic()
        if stop.is_set():
            future.set_exception(InterruptedError("Evaluation worker stopped"))
            continue
        try:
            prediction = policy.infer(obs, noise=noise)["actions"]
            future.set_result(prediction)
            timings.append({"queue_seconds": before - enqueued, "infer_seconds": time.monotonic() - before})
        except Exception as error:
            future.set_exception(error)
    for thread in lanes:
        thread.join()
    unresolved = [error for error in errors if not error.get("episode_id") or
                  not (directory / "results" / (error["episode_id"] + ".json")).exists()]
    atomic_json(worker_dir / "finished.json", {"at": stamp(), "inference_timings": timings, "errors": errors,
                "unresolved_errors": unresolved,
                "peak_memory_allocated_bytes": torch.npu.max_memory_allocated(device)})
    if unresolved:
        raise RuntimeError(f"Worker retained {len(unresolved)} unresolved infrastructure errors")


def aggregate(args):
    directory = args.output.resolve()
    config = read(directory / "eval_config.json")
    verify_sources(directory, config)
    expected = {item["episode_id"]: item for item in config["episodes"]}
    records = [read(p) for p in sorted((directory / "results").glob("*.json"))]
    seen = set()
    loaded_workers = {}
    for record in records:
        key = record["episode_id"]
        if key in seen or key not in expected or any(record.get(k) != v for k, v in expected[key].items()):
            raise ValueError("Duplicate, unexpected, or mismatched episode result")
        if (record["checkpoint_sha256"] != config["checkpoint_sha256"] or record["source_digest"] != config["source_digest"]
                or record.get("config_digest") != config["config_digest"]
                or type(record["success"]) is not bool or record["status"] not in
                {"success", "timeout", "terminated_without_success", "invalid_model_action"}
                or record["success"] != (record["status"] == "success")):
            raise ValueError("Result model/source/status does not match the frozen protocol")
        worker_id = record["worker_id"]
        if worker_id not in loaded_workers:
            receipt_file = directory / "workers" / worker_id / "load_receipt.json"
            receipt = read(receipt_file)
            if (receipt["status"] != "passed" or receipt["model_sha256"] != config["checkpoint_sha256"]
                    or receipt["post_load_dtype_conversion"] or not receipt["strict_safetensors_load"]
                    or not receipt["device_load_schema"]["all_shapes_and_dtypes_match"]
                    or not receipt["device_fp32_value_checks"]["all_byte_equal"]
                    or receipt["buffer_restoration"]["count"] != 3
                    or not receipt["buffer_restoration"]["all_restored_exactly"]):
                raise ValueError("Worker did not load the exact trained policy")
            loaded_workers[worker_id] = file_sha256(receipt_file)
        if record["load_receipt_sha256"] != loaded_workers[worker_id]:
            raise ValueError("Episode load receipt differs from its policy worker")
        steps = record.get("policy_steps")
        if (type(steps) is not int or not 0 <= steps <= expected[key]["max_steps"]
                or record.get("environment_steps") != config["settle_steps"] + steps
                or (record["success"] and steps == 0)
                or (record["status"] == "timeout" and steps != expected[key]["max_steps"])):
            raise ValueError("Result has an invalid rollout horizon or action count")
        if not record["success"] and args.verify_videos:
            if file_sha256(directory / record["video"]) != record["video_sha256"]:
                raise ValueError("Failure video missing or changed")
        seen.add(key)
    by_suite, by_task = {}, {}
    for suite in SUITE_STEPS:
        subset = [r for r in records if r["suite"] == suite]
        by_suite[suite] = {"completed": len(subset), "planned": sum(i["suite"] == suite for i in expected.values()),
                           "successes": sum(r["success"] for r in subset),
                           "success_rate": sum(r["success"] for r in subset) / len(subset) if subset else None}
        for task_id in range(10):
            task = [r for r in subset if r["task_id"] == task_id]
            if task:
                by_task[f"{suite}:{task_id}"] = {"suite": suite, "task_id": task_id, "prompt": task[0]["prompt"],
                    "completed": len(task), "successes": sum(r["success"] for r in task),
                    "success_rate": sum(r["success"] for r in task) / len(task)}
    attempts = [read(p) for p in (directory / "errors").glob("*.json")]
    missing = sorted(set(expected) - seen)
    summary = {"at": stamp(), "mode": config["mode"], "training_step": 20000,
               "checkpoint": config["checkpoint"], "checkpoint_sha256": config["checkpoint_sha256"],
               "planned": len(expected), "completed": len(records), "missing": missing,
               "completion_verified": not missing and bool(args.verify_videos),
               "successes": sum(r["success"] for r in records),
               "success_rate": sum(r["success"] for r in records) / len(records) if records else None,
               "by_suite": by_suite, "by_task": by_task,
               "statuses": dict(Counter(r["status"] for r in records)),
               "infrastructure_error_attempts": len(attempts),
               "unresolved_infrastructure_episodes": sorted({r["episode_id"] for r in attempts} - seen),
               "failed_videos": sum(not r["success"] for r in records),
               "limitations": ["No contemporaneous native-training baseline evaluated with this same noise protocol.",
                               "Explicit per-episode noise differs from legacy shared Torch RNG."]}
    atomic_json(directory / "progress.json", summary)
    print(json.dumps({k:summary[k] for k in ["completed", "planned", "successes", "success_rate", "by_suite", "infrastructure_error_attempts"]}, ensure_ascii=False))
    if not missing and args.verify_videos:
        if config["mode"] == "benchmark" and (len(records) != 2000 or len(by_task) != 40 or
                any(t["completed"] != 50 for t in by_task.values())):
            raise ValueError("Formal LIBERO evaluation requires 40 tasks x 50 trials")
        atomic_json(directory / "results.json", summary)
        atomic_json(directory / "COMPLETE.json", summary)
    elif not args.allow_incomplete:
        raise RuntimeError(f"Evaluation not verified complete; {len(missing)} episodes missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    p = modes.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--training-run", type=Path, required=True)
    p.add_argument("--simulators-per-npu", type=int, default=3)
    p.add_argument("--simulator-overlay", type=Path)
    p.add_argument("--smoke", action="store_true")
    p = modes.add_parser("worker")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker-id", required=True)
    p.add_argument("--device-index", type=int, default=0)
    p = modes.add_parser("aggregate")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--verify-videos", action="store_true")
    args = parser.parse_args()
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
