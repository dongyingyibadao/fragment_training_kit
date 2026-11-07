"""Pin exact training milestones and schedule resumable LIBERO evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

from jade_lotus.runtime import atomic_json, writer_lock


SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
EPISODE_LOCK_PROTOCOL = "posix-record-v1"
PROJECT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def fleet_budget(training):
    global_limit = training.get("fleet_global_limit", 16)
    gpu_limit = training.get("fleet_global_gpu_limit", global_limit)
    for name, value in (("fleet_global_limit", global_limit), ("fleet_global_gpu_limit", gpu_limit)):
        if type(value) is not int or not 1 <= value <= 128:
            raise ValueError(f"{name} must be an integer in 1..128")
    registry = training.get("fleet_registry_path")
    if registry is not None and (not isinstance(registry, str) or not Path(registry).is_absolute()):
        raise ValueError("fleet_registry_path must be an absolute shared registry path")
    return global_limit, gpu_limit, Path(registry).resolve() if registry is not None else None


def pin_checkpoint(training_run, step, mode="benchmark", simulator=None):
    """Keep the same inode after rolling checkpoint pruning; verify before use."""
    training_run = Path(training_run).resolve()
    directory = training_run / "evaluations" / (f"step-{step:08d}" if mode == "benchmark" else f"smoke-step-{step:08d}")
    directory.mkdir(parents=True, exist_ok=True)
    with writer_lock(directory / ".prepare-lock"):
        ready = directory / "eval_config.json"
        if ready.exists():
            config = read_json(ready)
            if config["step"] != step or config["mode"] != mode or Path(config["training_run"]) != training_run:
                raise ValueError("Existing evaluation configuration belongs to another request")
            return directory
        snapshot = directory / "snapshot" / f"step-{step:08d}.npz"
        snapshot.parent.mkdir(exist_ok=True)
        source_manifest = training_run / "checkpoints" / f"step-{step:08d}.json"
        metadata = (read_json(snapshot.with_suffix(".json"))
                    or read_json(directory / "pin_intent.json")
                    or read_json(source_manifest))
        if metadata is None:
            return None
        training_config = read_json(training_run / "run_config.json")
        if metadata["step"] != step or metadata["config_digest"] != digest_json(training_config):
            raise ValueError("Checkpoint does not match the immutable training configuration")
        source_path = metadata.get("source_checkpoint_path", metadata["path"])
        if not (directory / "pin_intent.json").exists():
            atomic_json(directory / "pin_intent.json", dict(metadata, source_checkpoint_path=source_path))
        if not snapshot.exists():
            temporary = snapshot.with_suffix(".pinning")
            # A killed preparer may have left the only surviving hard link.
            # Verify it before considering the now-pruned source checkpoint.
            if not temporary.exists() or sha256_file(temporary) != metadata["sha256"]:
                temporary.unlink(missing_ok=True)
                try:
                    os.link(source_path, temporary)
                except OSError:
                    shutil.copyfile(source_path, temporary)
                    with temporary.open("rb") as stream:
                        os.fsync(stream.fileno())
            if sha256_file(temporary) != metadata["sha256"]:
                temporary.unlink(missing_ok=True)
                raise ValueError("Milestone checkpoint checksum mismatch")
            os.replace(temporary, snapshot)
        elif sha256_file(snapshot) != metadata["sha256"]:
            raise ValueError("Pinned checkpoint checksum mismatch")
        pinned = dict(metadata, path=str(snapshot), source_checkpoint_path=source_path, pinned_at=time.time())
        atomic_json(snapshot.with_suffix(".json"), pinned)
        # A new evaluation has its own immutable code, separate from training.
        runtime = directory / "runtime"
        if runtime.exists():
            shutil.rmtree(runtime)
        shutil.copytree(training_run / "runtime", runtime,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("jade_lotus/evaluate.py", "jade_lotus/evaluation_sim.py", "scripts/run_fragment_evaluation.sh"):
            source = PROJECT / name
            if not source.is_file():
                raise FileNotFoundError("Evaluation implementation is not ready: " + name)
            shutil.copyfile(source, runtime / name)
        files = {str(p.relative_to(directory)): sha256_file(p)
                 for p in sorted(runtime.rglob("*")) if p.is_file()}
        source_digest = digest_json(files)
        atomic_json(directory / "eval_source_manifest.json", {"files": files, "sha256": source_digest})
        config = {
            "mode": mode, "training_run": str(training_run), "step": step,
            "snapshot": {"path": str(snapshot), "sha256": metadata["sha256"],
                         "schema": metadata["schema"], "format": "central_adam_npz",
                         "config_digest": metadata["config_digest"], "initial_digest": metadata["initial_digest"]},
            "suites": SUITES if mode == "benchmark" else [SUITES[0]],
            "trials_per_task": 10 if mode == "benchmark" else 1,
            "seed": 7, "replan_steps": 5, "num_steps_wait": 10,
            "num_steps": 10, "save_failure_videos_only": True,
            "source_digest": source_digest,
            "episode_lock_protocol": EPISODE_LOCK_PROTOCOL,
        }
        if mode == "smoke":
            config.update(task_ids=[0], max_steps=5, replan_steps=2)
        if simulator:
            config.update(simulator)
        atomic_json(directory / "run_config.json", {
            "skip_final_export": True, "evaluation": True, "mode": mode,
            "training_run": str(training_run), "step": step, "eval_config_digest": digest_json(config),
        })
        # Publish this last; evaluation workers never observe a half-frozen run.
        atomic_json(ready, config)
        return directory


def training_export_ready(training_run):
    training_run = Path(training_run)
    return ((training_run / "COMPLETE.json").is_file()
            and (training_run / "final_export/EXPORT_COMPLETE.json").is_file())


def active_evaluation_jobs(evaluation_dir):
    from jade_lotus.fleet import active

    state = read_json(Path(evaluation_dir) / "fleet/state.json", {"jobs": []})
    return sum(active(entry) for entry in state["jobs"])


def evaluation_instance_limit(config, allowance):
    """Only a frozen POSIX benchmark can use more than one instance."""
    return allowance if (config["mode"] == "benchmark"
                         and config.get("episode_lock_protocol") == EPISODE_LOCK_PROTOCOL) else 1


def publish_evaluation_completion(evaluation_dir):
    """Validate/publish completion locally, including when no API slot is free."""
    completion = read_json(evaluation_dir / "EVALUATION_COMPLETE.json")
    if completion is not None and not (evaluation_dir / "COMPLETE.json").exists():
        config = read_json(evaluation_dir / "eval_config.json")
        expected = len(config["suites"]) * len(config.get("task_ids", list(range(10)))) * config["trials_per_task"]
        overall = completion.get("overall", {})
        if (completion.get("complete") is not True or completion.get("step") != config["step"]
                or completion.get("mode") != config["mode"]
                or completion.get("config_digest") != digest_json(config)
                or completion.get("snapshot", {}).get("sha256") != config["snapshot"]["sha256"]
                or completion.get("training_run") != config["training_run"]
                or completion.get("expected_episodes") != expected
                or overall.get("episodes") != expected
                or overall.get("successes", -1) + overall.get("failures", -1) != expected
                or len(completion.get("failure_videos", [])) != overall.get("failures")):
            raise ValueError("Evaluation completion does not match the pinned evaluation plan")
        if config["mode"] == "benchmark":
            if expected != 400 or completion.get("benchmark_complete") is not True:
                raise ValueError("Benchmark completion must represent all 400 episodes")
            for suite in SUITES:
                statistics = completion.get("suites", {}).get(suite, {})
                if statistics.get("episodes") != 100 or any(
                        statistics.get("tasks", {}).get(str(task), {}).get("episodes") != 10 for task in range(10)):
                    raise ValueError("Benchmark completion must contain ten trials for every LIBERO task")
        atomic_json(evaluation_dir / "COMPLETE.json", completion)
    return read_json(evaluation_dir / "eval_config.json")


def reconcile_evaluation(client, evaluation_dir, fill=True, post_training_limit=1, training_evaluation_limit=1):
    from jade_lotus import fleet
    from jade_lotus.fleet import COMPLETION_MARKERS, reconcile

    evaluation_dir = Path(evaluation_dir).resolve()
    config = publish_evaluation_completion(evaluation_dir)
    training_run = Path(config["training_run"])
    # Preserve the training fleet's reserved evaluation slots. Once both the
    # optimizer and export have finished, validated POSIX episode locks allow
    # new evaluations to use released slots. Legacy frozen BSD-flock runtimes
    # remain serial per milestone; their configuration and results stay intact.
    limit = training_evaluation_limit if config["mode"] == "benchmark" else 1
    if config["mode"] == "benchmark" and training_export_ready(training_run):
        limit = post_training_limit
    limit = evaluation_instance_limit(config, limit)
    # Fleet recovery can submit durable planned entries even when the normal
    # placement limit has no free slots. Preserve them while an older fleet is
    # over its current cap; completed fleets must still be allowed to stop.
    if not any((evaluation_dir / marker).exists() for marker in COMPLETION_MARKERS):
        fill = fill and active_evaluation_jobs(evaluation_dir) <= limit
    command = f"bash {evaluation_dir}/runtime/scripts/run_fragment_evaluation.sh {{run_dir}} {{worker_id}} {{gpu_count}}"
    training = read_json(training_run / "run_config.json", {})
    global_limit, gpu_limit, registry = fleet_budget(training)
    if type(limit) is not int or not 1 <= limit <= min(global_limit, gpu_limit):
        raise ValueError("Evaluation instance limit exceeds the training run's shared instance/GPU ceiling")
    if registry is not None:
        fleet.REGISTRY_PATH = registry
    return reconcile(client, evaluation_dir, command, limit=limit, fill=fill,
                     inventory_seed=PROJECT / "artifacts/platform-discovery.json", allow_two=False,
                     diversify_groups=True, global_limit=global_limit, global_gpu_limit=gpu_limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("watch", "prepare", "reconcile"))
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1000, 2000, 4000])
    parser.add_argument("--mode", choices=("benchmark", "smoke"), default="benchmark")
    parser.add_argument("--simulator-settings", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=60)
    parser.add_argument("--fleet-interval-seconds", type=float, default=180)
    parser.add_argument("--training-evaluation-limit", type=int, default=1,
                        help="Total single-GPU evaluation instances during training, shared in milestone order")
    parser.add_argument("--post-training-limit", type=int, default=4,
                        help="Per-milestone single-GPU instances after export; the training run's shared global ceilings remain in force")
    parser.add_argument("--fill", action="store_true")
    parser.add_argument("--credentials", default="~/.local/share/jade-lotus/credentials.json")
    args = parser.parse_args()
    run = args.training_run.resolve()
    steps = sorted(set(args.steps))
    if not steps or any(step < 1 for step in steps):
        raise ValueError("Milestone steps must be positive")
    global_limit, gpu_limit, registry = fleet_budget(read_json(run / "run_config.json", {}))
    ceiling = min(global_limit, gpu_limit)
    if not 1 <= args.post_training_limit <= ceiling:
        raise ValueError(f"post-training-limit must be between 1 and the shared ceiling ({ceiling}, at most 128)")
    if not 1 <= args.training_evaluation_limit <= ceiling:
        raise ValueError(f"training-evaluation-limit must be between 1 and the shared ceiling ({ceiling}, at most 128)")
    if registry is not None:
        from jade_lotus import fleet
        fleet.REGISTRY_PATH = registry
    simulator = read_json(args.simulator_settings) if args.simulator_settings else None
    client = None
    if args.action != "prepare":
        from jade_lotus.platform import InspireClient, OAuthCredentials
        client = InspireClient(OAuthCredentials(Path(args.credentials).expanduser()))
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_reconcile = {}
    with writer_lock(run / "evaluations" / (".smoke-watch-lock" if args.mode == "smoke" else ".milestone-watch-lock")):
        while not stopping:
            summary, ready = [], []
            # Snapshot retention has priority over every platform request.
            for step in steps:
                directory = pin_checkpoint(run, step, args.mode, simulator)
                if directory is None:
                    active = read_json(run / "active.json", {})
                    if int(active.get("version", 0)) >= step + 60:
                        raise FileNotFoundError(f"Milestone {step} was not pinned before checkpoint pruning")
                    summary.append({"step": step, "status": "waiting_for_checkpoint"})
                    continue
                ready.append((step, directory))
            parallel_evaluations = training_export_ready(run)
            # Reserve every existing instance before granting any new slots.
            # Unknown submissions and stopping jobs remain active until the
            # platform confirms a terminal state. A reservation also covers a
            # planned fill if its API request fails or is not due on this tick.
            active_counts = {directory: active_evaluation_jobs(directory) for _, directory in ready}
            reservations = dict(active_counts)
            for step, directory in ready:
                cleanup = any((directory / name).is_file()
                              for name in ("COMPLETE.json", "EVALUATION_COMPLETE.json"))
                other_reservations = sum(reservations.values()) - reservations[directory]
                allowance = max(0, args.training_evaluation_limit - other_reservations)
                config = read_json(directory / "eval_config.json")
                training_limit = min(allowance, evaluation_instance_limit(config, args.training_evaluation_limit))
                budget_available = (training_limit > 0
                                    and sum(reservations.values()) <= args.training_evaluation_limit)
                if not active_counts[directory] and (cleanup or (not parallel_evaluations and not budget_available)):
                    # A completed empty fleet needs only local publication.
                    # Other empty fleets without a slot must not take the
                    # global fleet lock or query platform resources.
                    if cleanup:
                        try:
                            publish_evaluation_completion(directory)
                        except Exception as error:
                            summary.append({"step": step, "status": "completion_retry",
                                            "error_type": type(error).__name__, "error": str(error)})
                            continue
                    summary.append({"step": step, "status": "complete" if cleanup
                                    else "waiting_for_evaluation_slot", "active_jobs": 0,
                                    "running_jobs": 0, "fill": False})
                    continue
                within_directory_limit = active_counts[directory] <= evaluation_instance_limit(
                    config, args.post_training_limit if parallel_evaluations else args.training_evaluation_limit)
                evaluation_fill = args.fill and (cleanup or (within_directory_limit
                                                             and (parallel_evaluations or budget_available)))
                if not parallel_evaluations:
                    reservations[directory] = max(active_counts[directory], 0 if cleanup else training_limit)
                if client is not None and (time.monotonic() - last_reconcile.get(step, float('-inf')) >= args.fleet_interval_seconds):
                    last_reconcile[step] = time.monotonic()
                    try:
                        snapshot = reconcile_evaluation(client, directory, evaluation_fill, args.post_training_limit,
                                                        training_evaluation_limit=max(1, training_limit))
                    except Exception as error:
                        # The failed request may already have persisted plans
                        # or unknown submissions. Never lend those slots out.
                        reservations[directory] = max(reservations[directory], active_evaluation_jobs(directory))
                        if isinstance(error, RuntimeError) and "fleet-global-lock" in str(error):
                            # Lock contention did not contact the platform or
                            # complete a fleet pass. Retry on the next ordinary
                            # watch tick instead of waiting the full API interval.
                            last_reconcile.pop(step, None)
                        # Keep pinning exact future steps even if the platform
                        # API is temporarily unavailable. Do not treat an API
                        # error as a policy rollout failure or training fatal.
                        summary.append({"step": step, "status": "platform_retry",
                                        "error_type": type(error).__name__, "error": str(error)})
                        continue
                    active_counts[directory] = max(snapshot["active_jobs"], active_evaluation_jobs(directory))
                    cleanup = any((directory / name).is_file()
                                  for name in ("COMPLETE.json", "EVALUATION_COMPLETE.json"))
                    reservations[directory] = max(active_counts[directory], 0 if cleanup else training_limit)
                    summary.append({"step": step, "status": "complete" if (directory / "COMPLETE.json").exists() else "evaluating",
                                    "active_jobs": snapshot["active_jobs"], "running_jobs": snapshot["running_jobs"],
                                    "blocked_reason": snapshot["blocked_reason"], "fill": evaluation_fill})
                else:
                    summary.append({"step": step, "status": "complete" if (directory / "EVALUATION_COMPLETE.json").exists() else "pinned"})
            state = {"at": time.time(), "training_run": str(run), "mode": args.mode, "evaluations": summary}
            atomic_json(run / "evaluations" / ("smoke_watch.json" if args.mode == "smoke" else "milestone_watch.json"), state)
            print(json.dumps(state, ensure_ascii=False), flush=True)
            if args.action != "watch":
                return
            if len(ready) == len(steps) and all((directory / "EVALUATION_COMPLETE.json").is_file() for _, directory in ready):
                # Send final stops after the last worker commits its result.
                cleanup_complete = True
                for _, directory in ready:
                    try:
                        if not active_evaluation_jobs(directory):
                            publish_evaluation_completion(directory)
                            continue
                        final = reconcile_evaluation(client, directory, args.fill, args.post_training_limit,
                                                     training_evaluation_limit=args.training_evaluation_limit)
                        cleanup_complete &= final["active_jobs"] == 0
                    except Exception as error:
                        cleanup_complete = False
                        print(json.dumps({"event": "evaluation_cleanup_retry", "evaluation_dir": str(directory),
                                          "error_type": type(error).__name__, "error": str(error)}), flush=True)
                if cleanup_complete:
                    return
            # Short, interruptible timer: no model tokens are spent polling.
            deadline = time.monotonic() + args.interval_seconds
            while not stopping and time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
