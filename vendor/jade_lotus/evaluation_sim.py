"""Python 3.8 compatible LIBERO subprocess; stdin/stdout are private local RPC.

MuJoCo and its existing environment never enter the JAX inference process.
Import chatter goes to stderr so it cannot corrupt the binary response stream.
"""

import os
import sys

# Executing this file directly otherwise puts jade_lotus/platform.py ahead of
# Python's standard platform module, breaking torch's import-time checks.
_module_directory = os.path.dirname(os.path.abspath(__file__))
sys.path = [entry for entry in sys.path if os.path.abspath(entry) != _module_directory]

import ctypes
import json
import pickle
import signal
import struct
import traceback


OBSERVATION_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
                    "robot0_eef_quat", "robot0_gripper_qpos")
CAMERA_KEYS = OBSERVATION_KEYS[:2]


def read_exact(stream, length):
    output = bytearray()
    while len(output) < length:
        data = stream.read(length - len(output))
        if not data:
            raise EOFError()
        output.extend(data)
    return output


def create_software_egl_display(egl, device_id=0):
    """Select Mesa's CPU surfaceless platform without enumerating any device."""
    del device_id  # Legacy context callers pass this GPU index; it is unused.
    display = egl.eglGetPlatformDisplayEXT(0x31DD, None, None)  # EGL_PLATFORM_SURFACELESS_MESA
    if display == egl.EGL_NO_DISPLAY:
        raise RuntimeError("Mesa surfaceless EGL returned no display (error=%s)" % egl.eglGetError())
    if egl.eglInitialize(display, None, None) != egl.EGL_TRUE:
        raise RuntimeError("Mesa surfaceless EGL initialization failed (error=%s)" % egl.eglGetError())
    return display


def inspect_software_renderer(gl):
    def value(constant):
        raw = gl.glGetString(constant)
        if raw is None:
            raise RuntimeError("No current OpenGL context for renderer diagnostics")
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)

    result = {"backend": "egl-surfaceless-mesa", "gl_vendor": value(gl.GL_VENDOR),
              "gl_renderer": value(gl.GL_RENDERER), "gl_version": value(gl.GL_VERSION),
              "uses_device_enumeration": False}
    renderer = result["gl_renderer"].lower()
    if not any(name in renderer for name in ("llvmpipe", "softpipe", "swrast", "software rasterizer")):
        raise RuntimeError("Expected a CPU renderer; refusing OpenGL renderer %r" % result)
    result["cpu_renderer_verified"] = True
    return result


def configure_software_renderer():
    """Patch only this child process's legacy display factory, then probe GL.

    Robosuite's original factory always selects EGL_PLATFORM_DEVICE_EXT even
    when EGL_PLATFORM=surfaceless. On GPU nodes its device 0 can be hardware,
    which conflicts with LIBGL_ALWAYS_SOFTWARE. A direct surfaceless display
    uses Mesa's software context and never queries the GPU device list.
    """
    import mujoco.egl as mujoco_egl
    from mujoco.egl import egl_ext as egl
    from OpenGL import GL
    from robosuite.renderers.context import egl_context

    if egl_context.EGL_DISPLAY is not None or mujoco_egl.EGL_DISPLAY is not None:
        raise RuntimeError("An EGL display was initialized before CPU renderer isolation")

    def factory(device_id=0):
        return create_software_egl_display(egl, device_id=device_id)

    def forbid_device_enumeration(*args, **kwargs):
        raise RuntimeError("The isolated CPU renderer must not enumerate EGL GPU devices")

    egl.eglQueryDevicesEXT = forbid_device_enumeration
    egl_context.create_initialized_egl_device_display = factory
    mujoco_egl.create_initialized_egl_device_display = factory
    context = egl_context.EGLGLContext(max_width=16, max_height=16)
    try:
        context.make_current()
        report = inspect_software_renderer(GL)
    finally:
        context.free()
    print(json.dumps({"event": "libero_renderer_probe", "renderer": report}), file=sys.stderr, flush=True)
    return report


class CachedCameraSensors:
    """Skip camera rendering while preserving every observable sampling clock.

    Camera values are consumed only at policy replanning boundaries. All other
    sensors, simulator steps, control updates, and task predicates run normally.
    Unknown observable implementations retain their original full rendering.
    """

    def __init__(self, np):
        self.np = np
        self.entries = []
        self.capture = True
        self.cache = {}
        self.reason = "not installed"

    @property
    def active(self):
        return len(self.entries) == len(CAMERA_KEYS)

    def remove(self):
        self.capture = True
        for observable, wrapped, original in self.entries:
            if observable._sensor is wrapped:
                observable._sensor = original
        self.entries = []
        self.cache = {}

    def install(self, environment, observation, enabled=True):
        self.remove()
        if not enabled:
            self.reason = "disabled by evaluation configuration"
            return
        observables = getattr(getattr(environment, "env", None), "_observables", {})
        targets = []
        for name in CAMERA_KEYS:
            observable = observables.get(name)
            original = getattr(observable, "_sensor", None)
            # A reset may replace Observable objects. Unwrap any cached wrapper
            # left on an implementation that retained its sensor definition.
            while hasattr(original, "_jade_original_sensor"):
                original = original._jade_original_sensor
            if (not callable(original) or not hasattr(original, "__modality__") or
                    name not in observation):
                self.reason = "unsupported camera observable; using full rendering"
                return
            targets.append((name, observable, original))
        for name, observable, original in targets:
            self.cache[name] = self.np.array(observation[name], copy=True)

            def sensor(obs_cache, name=name, original=original):
                if self.capture:
                    self.cache[name] = original(obs_cache)
                return self.cache[name]

            sensor.__modality__ = original.__modality__
            sensor._jade_original_sensor = original
            observable._sensor = sensor
            self.entries.append((observable, sensor, original))
        self.reason = "cached RGB between requested captures; sampling clocks unchanged"

    def metadata(self):
        return {"active": self.active, "cameras": list(CAMERA_KEYS), "reason": self.reason,
                "observable_clocks_unchanged": True}


class LiberoSimulator:
    def __init__(self):
        import numpy as np
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.np = np
        self.benchmarks = benchmark.get_benchmark_dict()
        self.get_libero_path = get_libero_path
        self.environment_type = OffScreenRenderEnv
        self.suites = {}
        self.initial_states = {}
        self.environment = None
        self.environment_key = None
        self.camera_sensors = CachedCameraSensors(np)
        # Fail before loading the inference model if this container cannot
        # create the explicitly selected CPU renderer.
        self.renderer = configure_software_renderer()

    def suite(self, name):
        if name not in self.suites:
            self.suites[name] = self.benchmarks[name]()
        return self.suites[name]

    def states(self, name, task):
        key = (name, task)
        if key not in self.initial_states:
            import torch
            fixture = self.suite(name).get_task(task)
            path = os.path.join(self.get_libero_path("init_states"), fixture.problem_folder, fixture.init_states_file)
            # These are the trusted existing LIBERO fixtures. torch 2.7's new
            # weights_only default rejects their NumPy initial-state payloads.
            self.initial_states[key] = torch.load(path, map_location="cpu", weights_only=False)
        return self.initial_states[key]

    def observation(self, value, include_images=True):
        keys = OBSERVATION_KEYS if include_images else OBSERVATION_KEYS[2:]
        return {key: self.np.ascontiguousarray(value[key]) for key in keys}

    def describe(self, suites):
        import mujoco
        import robosuite
        result = {}
        for name in suites:
            suite = self.suite(name)
            if suite.n_tasks != 10:
                raise ValueError("Expected ten tasks per requested LIBERO suite")
            result[name] = {str(task): {"language": suite.get_task(task).language,
                                       "initial_states": len(self.states(name, task))}
                            for task in range(suite.n_tasks)}
        return {"suites": result, "python": sys.version, "numpy": self.np.__version__,
                "mujoco": mujoco.__version__, "robosuite": robosuite.__version__,
                "libero_module": str(sys.modules["libero.libero"].__file__),
                "rendering": "egl-surfaceless-mesa", "renderer": self.renderer,
                "camera_sampling": "cached sensor between explicit captures; standard step API captures by default"}

    def reset(self, suite, task_id, trial, seed, camera_cache=True):
        key = (suite, task_id)
        task = self.suite(suite).get_task(task_id)
        if self.environment_key != key:
            self.close()
            bddl = os.path.join(self.get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            self.environment = self.environment_type(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            self.environment_key = key
        # Reset and init-state restoration always use native camera sensors.
        # This also prevents nesting wrappers or retaining an old episode cache.
        self.camera_sensors.remove()
        self.np.random.seed(seed)
        self.environment.seed(seed)
        self.environment.reset()
        observation = self.environment.set_init_state(self.states(suite, task_id)[trial])
        from OpenGL import GL
        context = self.environment.sim._render_context_offscreen.gl_ctx
        context.make_current()
        actual_renderer = inspect_software_renderer(GL)
        if actual_renderer != self.renderer:
            raise RuntimeError("LIBERO environment renderer differs from the preflight context")
        self.camera_sensors.install(self.environment, observation, enabled=camera_cache)
        return {"observation": self.observation(observation), "prompt": str(task.language),
                "camera_cache": self.camera_sensors.metadata(),
                "renderer": actual_renderer,
                "control_frequency_hz": float(self.environment.env.control_freq)}

    def step(self, action, capture_images=True):
        if self.environment is None:
            raise RuntimeError("Simulator step before reset")
        action = self.np.asarray(action, dtype=self.np.float64)
        if action.shape != (7,) or not self.np.isfinite(action).all():
            raise ValueError("Expected seven finite LIBERO action components")
        if type(capture_images) is not bool:
            raise ValueError("capture_images must be a boolean")
        self.camera_sensors.capture = capture_images
        observation, reward, done, info = self.environment.step(action.tolist())
        # Explicit task predicate avoids mistaking time-limit termination for success.
        success = bool(self.environment.check_success())
        return {"observation": self.observation(observation, include_images=capture_images), "reward": float(reward),
                "done": bool(done), "success": success}

    def close(self):
        self.camera_sensors.remove()
        if self.environment is not None:
            self.environment.close()
        self.environment = None
        self.environment_key = None
        return None


def main():
    # Kill the simulator if the preempted inference parent disappears.
    try:
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() == 1:
            return
    except AttributeError:
        pass
    # Preserve the RPC pipe separately, redirecting even native-library stdout
    # chatter to stderr before importing MuJoCo and LIBERO.
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    simulator = None
    while True:
        try:
            size, = struct.unpack(">Q", read_exact(sys.stdin.buffer, 8))
            if size > 128 * 1024 * 1024:
                raise ValueError("Invalid simulator request length")
            request = pickle.loads(read_exact(sys.stdin.buffer, size))
        except EOFError:
            break
        operation = request.pop("operation")
        try:
            if simulator is None:
                simulator = LiberoSimulator()
            if operation not in ("describe", "reset", "step", "close"):
                raise ValueError("Unsupported simulator operation")
            result = getattr(simulator, operation)(**request)
            response = {"ok": True, "result": result}
        except Exception as error:
            response = {"ok": False, "error": repr(error), "traceback": traceback.format_exc()}
        payload = pickle.dumps(response, protocol=4)
        channel.write(struct.pack(">Q", len(payload)) + payload)
        channel.flush()
        if operation == "close":
            break
    if simulator is not None:
        simulator.close()


if __name__ == "__main__":
    main()
