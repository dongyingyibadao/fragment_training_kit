"""Independent, resumable LIBERO evaluation of immutable training snapshots.

One GPU owns inference; a local subprocess uses the existing LIBERO simulation
environment. Only a completed rollout with a durable failure video (if needed)
is committed. Interrupted rollouts and infrastructure errors remain unfinished.
"""

from __future__ import annotations

import argparse
from collections import deque
import contextlib
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid

import numpy as np

from jade_lotus.protocol import array_schema, file_sha256, read_json, safe_id, verify_sources
from jade_lotus.runtime import atomic_json


SUITE_STEPS = {"libero_spatial": 220, "libero_object": 280,
               "libero_goal": 300, "libero_10": 520}
DEFAULT_OPENPI = Path("/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/openpi")
EPISODE_LOCK_PROTOCOL = "posix-record-v1"
_PATH_LOCKS = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _reset_path_locks_after_fork():
    # POSIX record locks are not inherited by forked children. Neither should
    # their copies of a parent's possibly held thread locks remain locked.
    global _PATH_LOCKS, _PATH_LOCKS_GUARD
    _PATH_LOCKS, _PATH_LOCKS_GUARD = {}, threading.Lock()


os.register_at_fork(after_in_child=_reset_path_locks_after_fork)


@contextlib.contextmanager
def _posix_file_lock(path, *, blocking=True):
    """Hold a whole-file POSIX lock without a same-process close race.

    Record locks belong to a process, and closing *any* descriptor for their
    inode releases that process's locks. Serialize by resolved absolute path
    before opening the file, including nonblocking unsuccessful contenders.
    Lock files must never be unlinked, replaced, or opened through hard links.
    Cross-node filesystem behavior requires a separate acceptance probe.
    """
    path = Path(path).resolve()
    with _PATH_LOCKS_GUARD:
        local = _PATH_LOCKS.setdefault(str(path), threading.Lock())
    if not local.acquire(blocking=blocking):
        yield False
        return
    try:
        with path.open("a+") as stream:
            try:
                fcntl.lockf(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB),
                            0, 0, os.SEEK_SET)
            except OSError as error:
                if not blocking and error.errno in (errno.EACCES, errno.EAGAIN):
                    yield False
                    return
                raise
            try:
                yield True
            finally:
                fcntl.lockf(stream, fcntl.LOCK_UN, 0, 0, os.SEEK_SET)
    finally:
        local.release()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_config(config):
    """Keep abbreviated smoke tests separate from the requested 400 episodes."""
    if "episode_lock_protocol" in config and config["episode_lock_protocol"] != EPISODE_LOCK_PROTOCOL:
        raise ValueError(f"episode_lock_protocol must be {EPISODE_LOCK_PROTOCOL}")
    mode = config.get("mode", "benchmark")
    suites = config.get("suites", list(SUITE_STEPS))
    trials = config.get("trials_per_task", 10)
    if mode not in ("benchmark", "smoke"):
        raise ValueError("Evaluation mode must be benchmark or smoke")
    if not isinstance(suites, list) or not suites or len(suites) != len(set(suites)):
        raise ValueError("Suites must be a nonempty list without duplicates")
    if any(suite not in SUITE_STEPS for suite in suites):
        raise ValueError("Unsupported LIBERO suite")
    if type(trials) is not int or not 1 <= trials <= 10:
        raise ValueError("trials_per_task must be an integer in [1, 10]")
    task_ids = config.get("task_ids", list(range(10)))
    if (not isinstance(task_ids, list) or not task_ids or
            any(type(i) is not int or not 0 <= i < 10 for i in task_ids) or
            len(task_ids) != len(set(task_ids))):
        raise ValueError("task_ids must be distinct integers in [0, 9]")
    if mode == "benchmark" and (set(suites) != set(SUITE_STEPS) or trials != 10 or
                                task_ids != list(range(10)) or "max_steps" in config):
        raise ValueError("Benchmark requires four suites, all 10 tasks, 10 trials, and standard horizons")
    for key, default in (("replan_steps", 5), ("num_steps", 10)):
        value = config.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.get("replan_steps", 5) > 10:
        raise ValueError("replan_steps exceeds pi05 action_horizon=10")
    if type(config.get("num_steps_wait", 10)) is not int or config.get("num_steps_wait", 10) < 0:
        raise ValueError("num_steps_wait must be a nonnegative integer")
    if type(config.get("cache_intermediate_images", True)) is not bool:
        raise ValueError("cache_intermediate_images must be a boolean")
    if "max_steps" in config and (type(config["max_steps"]) is not int or config["max_steps"] < 1):
        raise ValueError("Smoke max_steps must be a positive integer")
    if type(config.get("step")) is not int or config["step"] < 0:
        raise ValueError("An explicit nonnegative snapshot step is required")
    snapshot = config.get("snapshot", {})
    if not snapshot.get("path") or len(snapshot.get("sha256", "")) != 64 or not snapshot.get("schema"):
        raise ValueError("snapshot.path, snapshot.sha256, and snapshot.schema are required")
    if not config.get("training_run"):
        raise ValueError("training_run is required")
    return config


def episode_plan(config):
    validate_config(config)
    return [{"episode_id": f"{suite}-task-{task:02d}-episode-{trial:02d}",
             "suite": suite, "task_id": task, "trial": trial,
             "max_steps": config.get("max_steps", SUITE_STEPS[suite])}
            for suite in config.get("suites", list(SUITE_STEPS))
            for task in config.get("task_ids", list(range(10)))
            for trial in range(config.get("trials_per_task", 10))]


def verify_evaluation_sources(evaluation_dir, config):
    directory = Path(evaluation_dir).resolve()
    manifest_path = directory / "eval_source_manifest.json"
    if not manifest_path.exists():
        if config.get("source_digest"):
            raise ValueError("Evaluation source manifest is missing")
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("sha256") != json_digest(manifest["files"]) or config.get("source_digest") != manifest["sha256"]:
        raise ValueError("Evaluation source manifest digest differs from eval_config")
    for name, digest in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory / "runtime") or file_sha256(path) != digest:
            raise ValueError(f"Frozen evaluation source changed: {name}")


def load_snapshot(config):
    """Read only Adam's parameter vector, restoring canonical sorted leaves."""
    snapshot = config["snapshot"]
    path = Path(snapshot["path"])
    if file_sha256(path) != snapshot["sha256"]:
        raise ValueError("Evaluation snapshot checksum mismatch")
    schema = snapshot["schema"]
    for key, spec in schema.items():
        if (not isinstance(key, str) or not isinstance(spec.get("shape"), list) or
                any(type(n) is not int or n < 1 for n in spec["shape"]) or spec.get("dtype") != "float32"):
            raise ValueError(f"Invalid FP32 trainable schema: {key}")
    with np.load(path, allow_pickle=False) as archive:
        if snapshot.get("format", "central_adam_npz") == "central_adam_npz":
            parameter = archive["parameter"]
            expected_size = sum(math.prod(spec["shape"]) for spec in schema.values())
            if parameter.dtype != np.float32 or parameter.shape != (expected_size,):
                raise ValueError("Snapshot parameter vector differs from its schema")
            if not np.isfinite(parameter).all():
                raise ValueError("Snapshot parameters are nonfinite")
            arrays, offset = {}, 0
            for key, spec in sorted(schema.items()):
                size = math.prod(spec["shape"])
                arrays[key] = parameter[offset:offset + size].reshape(spec["shape"])
                offset += size
        elif snapshot["format"] == "trainable_npz":
            arrays = {key: archive[key] for key in archive.files}
        else:
            raise ValueError("Unsupported evaluation snapshot format")
    if array_schema(arrays) != schema or any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Snapshot trainable leaves differ from schema or contain nonfinite values")
    return arrays


class EvaluationPolicy:
    """Reconstruct the exact local backbone plus snapshot adapters/heads."""

    def __init__(self, config):
        from jade_lotus.gradient_model import _flat, _imports, _unflat, make_configuration

        training_dir = Path(config["training_run"]).resolve()
        training = json.loads((training_dir / "run_config.json").read_text())
        if config["snapshot"].get("config_digest") != json_digest(training):
            raise ValueError("Snapshot belongs to a different training configuration")
        latest = read_json(training_dir / "checkpoint.json", {})
        initial_digest = config["snapshot"].get("initial_digest")
        if initial_digest and latest and initial_digest != latest.get("initial_digest"):
            raise ValueError("Snapshot initialization differs from the training run")
        verify_sources(training_dir, training)
        root = _imports(training)
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from openpi import transforms
        from openpi.models import model as model_module
        from openpi.policies.policy import Policy

        self.model_config, data_config, trainable_filter = make_configuration(training)
        devices = jax.local_devices()
        self.device = devices[0]
        if self.device.platform != "gpu":
            raise RuntimeError("Evaluation inference requires one scheduler-assigned GPU")
        arrays = load_snapshot(config)
        print(json.dumps({"event": "evaluation_loading_model", "step": config["step"],
                          "device": str(self.device), "trainable_parameters": sum(v.size for v in arrays.values())}), flush=True)
        with jax.default_device(self.device):
            shape_model = nnx.eval_shape(lambda: self.model_config.create(jax.random.key(int(training.get("seed", 42)))))
            graphdef, state = nnx.split(shape_model)
            train_shapes = _flat(state.filter(trainable_filter).to_pure_dict())
            expected = {key: {"shape": list(spec.shape), "dtype": "float32"}
                        for key, spec in train_shapes.items()}
            if expected != config["snapshot"]["schema"]:
                raise ValueError("Snapshot paths/shapes do not match the actual pi05 trainable model")
            base_path = Path(training.get("base_params", root / "outputs/robotwin_clean50_abs_eef/pretrained/pi05_base/params"))
            base = _flat(model_module.restore_params(base_path, restore_type=np.ndarray))
            pure = {}
            for path, spec in _flat(state.to_pure_dict()).items():
                if path in arrays:
                    value = arrays.pop(path)
                    # The native base also stores trainable non-LoRA heads.
                    base.pop(path, None)
                elif path in base:
                    value = np.asarray(base.pop(path)).astype(jnp.bfloat16)
                else:
                    raise ValueError(f"Base checkpoint is missing required frozen parameter {path}")
                if tuple(value.shape) != tuple(spec.shape):
                    raise ValueError(f"Evaluation parameter shape mismatch: {path}")
                pure[path] = jax.device_put(value, self.device)
            if base or arrays:
                raise ValueError("Unexpected base or snapshot parameters")
            state.replace_by_pure_dict(_unflat(pure))
            model = nnx.merge(graphdef, state)
            self.policy = Policy(
                model,
                # This upstream Policy uses ``rng or key(0)``; passing a typed
                # JAX key triggers an invalid truth-value conversion. Explicit
                # per-episode noise already determines every flow sample.
                transforms=[transforms.InjectDefaultPrompt(None), *data_config.data_transforms.inputs,
                            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                            *data_config.model_transforms.inputs],
                output_transforms=[*data_config.model_transforms.outputs,
                                   transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                                   *data_config.data_transforms.outputs],
                sample_kwargs={"num_steps": config.get("num_steps", 10)},
                metadata={"training_step": config["step"], "snapshot_sha256": config["snapshot"]["sha256"]},
            )

    def infer(self, element, *, noise):
        return self.policy.infer(element, noise=noise)["actions"]


def simulator_environment(parent, *, libero_root, libero_config_path):
    """Build an isolated CPU-renderer environment without touching JAX's env."""
    env = dict(parent)
    env["PYTHONPATH"] = str(libero_root)
    env["LIBERO_CONFIG_PATH"] = str(libero_config_path)
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["EGL_PLATFORM"] = "surfaceless"
    env["GALLIUM_DRIVER"] = "llvmpipe"
    # The subprocess creates a Mesa surfaceless display directly, bypassing the
    # legacy EGL device-selection code entirely. No GPU is visible to its torch
    # runtime; the inference parent's scheduler assignment remains unchanged.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
    return env


class Simulator:
    """Framed local RPC keeps the existing MuJoCo environment separate from JAX."""

    def __init__(self, config, evaluation_dir, worker_id):
        import pickle
        self.pickle = pickle
        self.timeout = float(config.get("simulator_timeout_seconds", 180))
        project = Path(config["training_run"]).resolve().parent.parent
        training = read_json(Path(config["training_run"]) / "run_config.json", {})
        openpi_root = Path(training.get("openpi_root", DEFAULT_OPENPI))
        python = Path(config.get("simulator_python", openpi_root.parent / "miniconda3/envs/fastwam/bin/python"))
        libero_root = Path(config.get("libero_root", openpi_root.parent / "LIBERO"))
        env = simulator_environment(
            os.environ, libero_root=libero_root,
            libero_config_path=Path(config.get("libero_config_path", project / "artifacts/evaluation-env/libero-config")),
        )
        self.log = (Path(evaluation_dir) / "worker_logs" / f"{safe_id(worker_id)}-simulator.log").open("ab")
        self.process = subprocess.Popen([str(python), "-u", str(Path(__file__).with_name("evaluation_sim.py"))],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        env=env, bufsize=0)

    def _read_exact(self, count, deadline):
        chunks = bytearray()
        while len(chunks) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], max(0, remaining))[0]:
                raise TimeoutError("LIBERO simulator RPC timed out")
            block = os.read(self.process.stdout.fileno(), count - len(chunks))
            if not block:
                raise RuntimeError(f"LIBERO simulator exited during RPC (exit={self.process.poll()})")
            chunks.extend(block)
        return chunks

    def request(self, operation, **fields):
        payload = self.pickle.dumps({"operation": operation, **fields}, protocol=4)
        self.process.stdin.write(struct.pack(">Q", len(payload)) + payload)
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout
        size, = struct.unpack(">Q", self._read_exact(8, deadline))
        if size > 128 * 1024 * 1024:
            raise RuntimeError("Invalid simulator response length")
        response = self.pickle.loads(self._read_exact(size, deadline))
        if not response.get("ok"):
            raise RuntimeError(f"LIBERO {operation} failed: {response.get('error')}\n{response.get('traceback', '')}")
        return response["result"]

    def close(self):
        with contextlib.suppress(Exception):
            if self.process.poll() is None:
                self.process.stdin.close()
                self.process.wait(timeout=5)
        if self.process.poll() is None:
            self.process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=5)
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
        self.log.close()


def quaternion_to_axisangle(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64).copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quaternion[3] ** 2)
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return quaternion[:3] * 2.0 * math.acos(float(quaternion[3])) / denominator


def preprocess_observation(observation, prompt):
    from openpi_client import image_tools

    image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
    state = np.concatenate((observation["robot0_eef_pos"],
                            quaternion_to_axisangle(observation["robot0_eef_quat"]),
                            observation["robot0_gripper_qpos"]))
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("LIBERO observation state is malformed or nonfinite")
    return {"observation/image": image, "observation/wrist_image": wrist,
            "observation/state": state, "prompt": str(prompt)}


def rollout(policy, simulator, descriptor, config, should_stop=lambda: False):
    if should_stop():
        raise InterruptedError("Evaluation interrupted before reset")
    reset = simulator.request("reset", suite=descriptor["suite"], task_id=descriptor["task_id"],
                              trial=descriptor["trial"], seed=int(config.get("seed", 7)),
                              camera_cache=config.get("cache_intermediate_images", True))
    observation, prompt = reset["observation"], reset["prompt"]
    simulation_hz = float(reset.get("control_frequency_hz", 20))
    if not math.isfinite(simulation_hz) or simulation_hz <= 0:
        raise ValueError("Simulator control frequency must be positive and finite")
    settling_steps = config.get("num_steps_wait", 10)
    for index in range(settling_steps):
        if should_stop():
            raise InterruptedError("Evaluation interrupted during settling")
        settled = simulator.request("step", action=[0.0] * 6 + [-1.0], capture_images=index == settling_steps - 1)
        observation = settled["observation"]
    plan, frames, frame_steps, timings = deque(), [], [], []
    completed_steps, plans, success = 0, 0, False
    noise_seed = int.from_bytes(hashlib.sha256(
        f"{config.get('seed', 7)}:{descriptor['episode_id']}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(noise_seed)
    for _ in range(descriptor["max_steps"]):
        if should_stop():
            raise InterruptedError("Evaluation interrupted during rollout")
        if not plan:
            element = preprocess_observation(observation, prompt)
            frames.append(element["observation/image"])
            frame_steps.append(completed_steps)
            started = time.monotonic()
            noise = rng.standard_normal((10, 32)).astype(np.float32)
            actions = np.asarray(policy.infer(element, noise=noise))
            timings.append(time.monotonic() - started)
            if (actions.ndim != 2 or actions.shape[0] < config.get("replan_steps", 5) or
                    actions.shape[1] != 7 or not np.isfinite(actions).all()):
                raise ValueError("Policy returned malformed or nonfinite action chunk")
            plan.extend(actions[:config.get("replan_steps", 5)])
            plans += 1
        action = plan.popleft().tolist()
        # The final action of a chunk renders the exact observation consumed by
        # the next policy call. Intermediate physics/control/success checks run
        # normally while their unused RGB values remain cached in the simulator.
        result = simulator.request("step", action=action, capture_images=not plan)
        observation = result["observation"]
        completed_steps += 1
        success = bool(result["success"])
        if success:
            break
    return {"success": success, "policy_steps": completed_steps,
            "environment_steps": completed_steps + config.get("num_steps_wait", 10),
            "policy_calls": plans, "inference_seconds": timings, "task_language": prompt,
            "noise_seed": noise_seed, "camera_cache": reset.get("camera_cache", {"active": False, "reason": "legacy simulator"}),
            "renderer": reset.get("renderer", {"backend": "legacy simulator; not diagnosed"}),
            "video_sampling": {"simulation_hz": simulation_hz, "stride_steps": config.get("replan_steps", 5),
                               "fps": simulation_hz / config.get("replan_steps", 5),
                               "cadence": "policy input boundaries after settling",
                               "frame_policy_steps": frame_steps, "frames": len(frames),
                               "rollout_seconds": completed_steps / simulation_hz,
                               "terminal_partial_chunk_steps": completed_steps % config.get("replan_steps", 5)}}, frames


def save_failure_video(path, frames, *, fps=10):
    import imageio.v2 as imageio

    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.stem + "-", suffix=".mp4", dir=path.parent)
    os.close(fd)
    try:
        imageio.mimwrite(temporary, frames, fps=fps, codec="libx264", quality=7, macro_block_size=16)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        digest = file_sha256(temporary)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return digest
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class EpisodeStore:
    """A POSIX record lock spans one rollout and its durable result commit.

    Per-path thread locks exclude stores in the same process; whole-file
    lockf locks exclude other processes and release on SIGKILL. There is no
    expiring lease. Cross-node use must pass the platform's filesystem probe;
    frozen evaluations using the older protocol remain separately serialized.
    """

    def __init__(self, evaluation_dir, worker_id):
        self.directory = Path(evaluation_dir).resolve()
        self.worker_id = safe_id(worker_id)
        self.session = uuid.uuid4().hex
        self.config = validate_config(json.loads((self.directory / "eval_config.json").read_text()))
        self.config_digest = json_digest(self.config)
        verify_evaluation_sources(self.directory, self.config)
        self.plan = episode_plan(self.config)
        self.by_id = {item["episode_id"]: item for item in self.plan}
        self._verified_results = {}
        self._heartbeat_lock = threading.Lock()
        self.current_episode = None
        for name in ("episode_locks", "results", "errors", "videos", "workers", "worker_logs"):
            (self.directory / name).mkdir(exist_ok=True)

    def check_configuration(self):
        if json_digest(json.loads((self.directory / "eval_config.json").read_text())) != self.config_digest:
            raise ValueError("Evaluation configuration changed during execution")

    def heartbeat(self, stage, **fields):
        with self._heartbeat_lock:
            atomic_json(self.directory / "workers" / (self.worker_id + ".json"),
                        {"worker_id": self.worker_id, "session_id": self.session, "pid": os.getpid(),
                         "time": time.time(), "stage": stage, "episode_id": self.current_episode,
                         "step": self.config["step"], "config_digest": self.config_digest, **fields})

    def result(self, descriptor):
        path = self.directory / "results" / (descriptor["episode_id"] + ".json")
        if not path.exists():
            return None
        def signature(target):
            stat = target.stat()
            return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

        result_signature = signature(path)
        cached = self._verified_results.get(descriptor["episode_id"])
        if cached is not None and cached[0] == result_signature:
            video = cached[2].get("failure_video")
            if video is None or (self.directory / video["path"]).is_file():
                if video is None or cached[1] == signature(self.directory / video["path"]):
                    return cached[2]
        result = json.loads(path.read_text())
        if (result.get("status") != "completed" or result.get("descriptor") != descriptor or
                result.get("config_digest") != self.config_digest or
                result.get("snapshot_sha256") != self.config["snapshot"]["sha256"] or
                result.get("step") != self.config["step"] or type(result.get("success")) is not bool or
                type(result.get("policy_steps")) is not int or not 1 <= result["policy_steps"] <= descriptor["max_steps"]):
            raise ValueError(f"Invalid or mismatched committed evaluation result: {path}")
        video = result.get("failure_video")
        if result["success"]:
            if video:
                raise ValueError("Successful episodes must not contain replay video")
        elif (not isinstance(video, dict) or video.get("path") != f"videos/{descriptor['episode_id']}.mp4" or
              not (self.directory / video["path"]).is_file() or
              file_sha256(self.directory / video["path"]) != video.get("sha256")):
            raise ValueError(f"Missing or invalid failure video: {path}")
        self._verified_results[descriptor["episode_id"]] = (
            result_signature, signature(self.directory / video["path"]) if video else None, result)
        return result

    @contextlib.contextmanager
    def claim(self, descriptor):
        self.check_configuration()
        path = self.directory / "episode_locks" / (descriptor["episode_id"] + ".lock")
        with _posix_file_lock(path, blocking=False) as locked:
            if not locked:
                yield False
                return
            try:
                if self.result(descriptor):
                    yield False
                else:
                    self.current_episode = descriptor["episode_id"]
                    self.heartbeat("rollout")
                    yield True
            finally:
                self.current_episode = None

    def commit(self, descriptor, outcome, frames):
        self.check_configuration()
        if self.current_episode != descriptor["episode_id"]:
            raise RuntimeError("Evaluation result requires the current episode lock")
        if self.result(descriptor) is not None:
            raise RuntimeError("Evaluation episode is already committed")
        if (type(outcome.get("success")) is not bool or type(outcome.get("policy_steps")) is not int or
                not 1 <= outcome["policy_steps"] <= descriptor["max_steps"]):
            raise ValueError("Only a valid completed policy rollout may be committed")
        # A preempted encoder can leave a temporary video, or a complete video
        # whose episode JSON was never published. This lock excludes another
        # live encoder, and the committed-result guard above protects valid
        # existing videos. A successful retry must leave no replay behind.
        video_directory = self.directory / "videos"
        stale = list(video_directory.glob(f".{descriptor['episode_id']}-*.mp4"))
        if outcome["success"]:
            stale.append(video_directory / (descriptor["episode_id"] + ".mp4"))
        removed = False
        for path in stale:
            try:
                path.unlink()
                removed = True
            except FileNotFoundError:
                pass
        if removed:
            directory_fd = os.open(video_directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        result = {**outcome, "descriptor": descriptor, "status": "completed", "step": self.config["step"],
                  "config_digest": self.config_digest, "snapshot_sha256": self.config["snapshot"]["sha256"],
                  "worker_id": self.worker_id, "session_id": self.session, "completed_at": time.time()}
        if not result["success"]:
            video = Path("videos") / (descriptor["episode_id"] + ".mp4")
            sampling = result.get("video_sampling", {"fps": 20 / self.config.get("replan_steps", 5),
                                                      "simulation_hz": 20,
                                                      "stride_steps": self.config.get("replan_steps", 5)})
            sha = save_failure_video(self.directory / video, frames, fps=sampling["fps"])
            result["failure_video"] = {"path": str(video), "sha256": sha, **sampling}
        atomic_json(self.directory / "results" / (descriptor["episode_id"] + ".json"), result)
        return result

    def record_error(self, descriptor, error):
        record = {"descriptor": descriptor, "config_digest": self.config_digest, "step": self.config["step"],
                  "worker_id": self.worker_id, "session_id": self.session, "time": time.time(),
                  "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                  "counts_as_episode": False}
        path = self.directory / "errors" / f"{descriptor['episode_id']}-{self.session}-{uuid.uuid4().hex}.json"
        atomic_json(path, record)

    def summarize(self):
        # Serialize the scan as well as publication: an older 399-result scan
        # must never overwrite a newer 400-result progress file. Episode locks
        # and rollouts remain independent of this short summary critical section.
        with _posix_file_lock(self.directory / "summary.lock"):
            return self._summarize_locked()

    def _summarize_locked(self):
        self.check_configuration()
        results = [result for item in self.plan if (result := self.result(item)) is not None]

        def statistics(group):
            successes = sum(item["success"] for item in group)
            return {"episodes": len(group), "successes": successes, "failures": len(group) - successes,
                    "success_rate": successes / len(group) if group else None}

        suites = {}
        for suite in self.config.get("suites", list(SUITE_STEPS)):
            selected = [item for item in results if item["descriptor"]["suite"] == suite]
            suites[suite] = {**statistics(selected), "tasks": {
                str(task): statistics([item for item in selected if item["descriptor"]["task_id"] == task])
                for task in self.config.get("task_ids", list(range(10)))}}
        complete = len(results) == len(self.plan)
        summary = {"mode": self.config.get("mode", "benchmark"), "complete": complete,
                   "benchmark_complete": complete and self.config.get("mode", "benchmark") == "benchmark",
                   "step": self.config["step"], "training_run": self.config["training_run"],
                   "snapshot": self.config["snapshot"], "config_digest": self.config_digest,
                   "expected_episodes": len(self.plan), "overall": statistics(results), "suites": suites,
                   "failure_videos": [item["failure_video"] for item in results if not item["success"]],
                   "infrastructure_errors": len(list((self.directory / "errors").glob("*.json"))),
                   "updated_at": time.time()}
        progress_path = self.directory / "progress.json"
        if progress_path.exists():
            previous = json.loads(progress_path.read_text())
            if (previous.get("config_digest") != self.config_digest or
                    previous.get("overall", {}).get("episodes", 0) > len(results)):
                raise ValueError("Evaluation progress configuration changed or completed episode count regressed")
        completion_path = self.directory / "EVALUATION_COMPLETE.json"
        if completion_path.exists():
            committed = json.loads(completion_path.read_text())
            stable_fields = ("mode", "complete", "benchmark_complete", "step", "training_run", "snapshot",
                             "config_digest", "expected_episodes", "overall", "suites", "failure_videos")
            if not complete or any(committed.get(key) != summary.get(key) for key in stable_fields):
                raise ValueError("Committed evaluation completion differs from verified episode results")
            # All concurrent finishers retain the original durable completion.
            # In particular, completed_at and the marker inode never change.
            summary = committed
        elif complete:
            summary = {**summary, "completed_at": time.time()}
        atomic_json(progress_path, summary)
        if complete and not completion_path.exists():
            # Progress reaches 400 before the terminal marker becomes visible.
            # A crash between these two writes is repaired by the next summary.
            atomic_json(completion_path, summary)
        return summary


def run(evaluation_dir, worker_id, *, policy_factory=EvaluationPolicy, simulator_factory=Simulator):
    store = EpisodeStore(evaluation_dir, worker_id)
    stopped = threading.Event()
    finished = threading.Event()
    simulator = None

    def handle_signal(_number, _frame):
        stopped.set()

    previous = {number: signal.signal(number, handle_signal) for number in (signal.SIGTERM, signal.SIGINT)}

    def heartbeat_loop():
        while not finished.wait(10):
            store.heartbeat("alive" if store.current_episode is None else "rollout")

    thread = threading.Thread(target=heartbeat_loop, daemon=True)
    thread.start()
    try:
        summary = store.summarize()
        if summary["complete"]:
            store.heartbeat("complete")
            return summary
        store.heartbeat("initializing_simulator")
        simulator = simulator_factory(store.config, store.directory, store.worker_id)
        metadata = simulator.request("describe", suites=store.config.get("suites", list(SUITE_STEPS)))
        for item in store.plan:
            task = metadata["suites"][item["suite"]][str(item["task_id"])]
            if task["initial_states"] <= item["trial"]:
                raise ValueError("LIBERO task has too few initial states for requested trials")
        atomic_json(store.directory / "workers" / (worker_id + "-environment.json"),
                    {"config_digest": store.config_digest, "metadata": metadata})
        store.heartbeat("loading_policy")
        policy = policy_factory(store.config)
        store.heartbeat("ready")
        while not stopped.is_set() and not (store.directory / "STOP").exists():
            processed = False
            for descriptor in store.plan:
                if stopped.is_set() or (store.directory / "STOP").exists():
                    break
                if store.result(descriptor):
                    continue
                with store.claim(descriptor) as owned:
                    if not owned:
                        continue
                    try:
                        outcome, frames = rollout(policy, simulator, descriptor, store.config,
                                                  lambda: stopped.is_set() or (store.directory / "STOP").exists())
                        if stopped.is_set():
                            raise InterruptedError("Evaluation stopped before result commit")
                        result = store.commit(descriptor, outcome, frames)
                    except InterruptedError:
                        stopped.set()
                        break
                    except Exception as error:
                        store.record_error(descriptor, error)
                        store.heartbeat("infrastructure_error", error=str(error))
                        # Recreate the entire process on infrastructure failure;
                        # this episode remains uncommitted for the next worker.
                        raise
                    print(json.dumps({"event": "evaluation_episode_completed", "episode_id": descriptor["episode_id"],
                                      "step": store.config["step"], "success": result["success"],
                                      "policy_steps": result["policy_steps"]}), flush=True)
                    processed = True
                summary = store.summarize()
                if summary["complete"]:
                    store.heartbeat("complete")
                    return summary
            summary = store.summarize()
            if summary["complete"]:
                store.heartbeat("complete")
                return summary
            if not processed:
                stopped.wait(2)
        store.heartbeat("interrupted")
        return store.summarize()
    finally:
        finished.set()
        thread.join(timeout=15)
        if simulator is not None:
            simulator.close()
        for number, handler in previous.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()
    run(args.evaluation_dir, args.worker_id)


if __name__ == "__main__":
    main()
