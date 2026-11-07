"""Bounded CPU-process prefetch of the existing deterministic atomic samples.

Children construct only the transformed dataset.  The parent still performs
``prepare_atomic_device``: NPU observation preprocessing, noise and flow time
are deliberately outside this module.  No background thread reads or changes
the parent's Python/NumPy/Torch random generators.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import copy
import dataclasses
import hashlib
import importlib.abc
import importlib.util
import multiprocessing
from numbers import Integral
import os
from pathlib import Path
import queue
import random
import sys
import threading
import time
from types import MappingProxyType

import numpy as np

from jade_lotus.atomic_batch import ATOMIC_SIZE, describe_units


_CPU_ENV = {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "CUDA_VISIBLE_DEVICES": "",
            "ASCEND_RT_VISIBLE_DEVICES": "", "ASCEND_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu"}
_SPAWN_ENV_LOCK = threading.Lock()
_READER = None


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class DeterministicCpuReader:
    """The CPU part of TorchNpuGradientEngine._sample, with the same RNGs."""

    def __init__(self, dataset, seed=42):
        import torch
        self.dataset, self.seed, self.torch = dataset, _integer(seed, "seed"), torch
        self.length = _integer(len(dataset), "dataset length", minimum=1)
        self._permutations = {}

    def sample(self, position):
        position = _integer(position, "sample position")
        epoch, offset = divmod(position, self.length)
        if epoch not in self._permutations:
            self._permutations[epoch] = np.random.default_rng(self.seed + epoch).permutation(self.length)
            for previous in list(self._permutations):
                if previous < epoch - 1:
                    self._permutations.pop(previous)
        index = int(self._permutations[epoch][offset])
        sample_seed = int.from_bytes(hashlib.blake2b(
            f"{self.seed}:{epoch}:{index}".encode(), digest_size=8).digest(), "big")
        torch = self.torch
        cpu_state = torch.random.get_rng_state()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            sample_seed &= 2**63 - 1
            # torch.manual_seed also addresses accelerators; use only the CPU
            # default generator, exactly like the existing engine.
            torch.random.default_generator.manual_seed(sample_seed)
            np.random.seed(sample_seed % 2**32)
            random.seed(sample_seed)
            return self.dataset[index]
        finally:
            torch.random.set_rng_state(cpu_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def build_openpi_dataset(config):
    """Reproduce the engine's CPU dataset/config construction, without a model."""
    from jade_lotus.npu_gradient_model import _checkpoint_file, _imports

    _imports(config)
    from openpi.training import config as upstream_config
    from openpi.training import data_loader

    recipe = upstream_config.get_config(config.get("openpi_config", "pi05_libero_lerobot"))
    checkpoint = _checkpoint_file(config)
    assets = dataclasses.replace(recipe.data.assets, assets_dir=str(checkpoint.parent / "assets"))
    model_config = dataclasses.replace(recipe.model, dtype=recipe.pytorch_training_precision)
    data_config = dataclasses.replace(recipe.data, assets=assets).create(recipe.assets_dirs, model_config)
    data_config = dataclasses.replace(data_config, dataset_root=str(Path(config["dataset_root"]).resolve()))
    if data_config.use_quantile_norm or data_config.norm_stats is None:
        raise ValueError("pi05_libero_lerobot must use its matching mean/std assets")
    dataset = data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)
    return data_loader.transform_dataset(dataset, data_config)


class _NoNpuBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch_npu" or fullname.startswith("torch_npu."):
            raise ImportError("resident_data CPU workers cannot import torch_npu")
        return None


def _hide_npu_backend():
    """Expose an unavailable backend to probes, while rejecting real imports.

    Accelerate and Transformers call ``importlib.util.find_spec`` before
    choosing a backend.  Raising from that availability probe prevents their
    CPU-only imports too.  The wrapper is installed only in a spawned reader;
    Python's actual import machinery still reaches the rejecting finder.
    """
    if any(name == "torch_npu" or name.startswith("torch_npu.") for name in sys.modules):
        raise RuntimeError("Spawn main imported torch_npu before CPU initializer; import NPU only inside main()")
    original_find_spec = importlib.util.find_spec

    def cpu_find_spec(name, package=None):
        absolute = importlib.util.resolve_name(name, package) if name.startswith(".") else name
        if absolute == "torch_npu" or absolute.startswith("torch_npu."):
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = cpu_find_spec
    sys.meta_path.insert(0, _NoNpuBackend())


def _initialize_worker(config, dataset_factory, startup_queue, startup_barrier):
    global _READER
    os.environ.update(_CPU_ENV)
    _hide_npu_backend()
    import torch

    torch.set_num_threads(_integer(config.get("resident_data_cpu_threads", 1), "CPU threads", minimum=1))
    dataset = dataset_factory(config)
    _READER = DeterministicCpuReader(dataset, config.get("seed", 42))
    startup_queue.put({"pid": os.getpid(), "dataset_length": _READER.length})
    startup_barrier.wait()


def _worker_ready():
    return True


def _validate_cpu_tree(value):
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_cpu_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _validate_cpu_tree(item)
    else:
        import torch
        if torch.is_tensor(value) and value.device.type != "cpu":
            raise ValueError("Dataset sample contains an accelerator tensor")


def _read_atomic_unit(step, positions):
    if _READER is None:
        raise RuntimeError("CPU dataset reader is not initialized")
    samples = {}
    for position in positions:
        sample = _READER.sample(position)
        _validate_cpu_tree(sample)
        # A dataset may reuse mutable read buffers; each returned sample owns
        # its values before this process proceeds to the next sample.
        samples[position] = copy.deepcopy(sample)
    return {"step": step, "positions": positions, "dataset_length": _READER.length,
            "samples": samples}


@dataclasses.dataclass(frozen=True)
class StepSamples(Mapping):
    """An exact, ordered position cache; missing positions fail explicitly."""

    step: int
    seed: int
    batch_size: int
    dataset_length: int
    positions: tuple
    _samples: Mapping = dataclasses.field(repr=False)

    def __getitem__(self, position):
        position = _integer(position, "sample position")
        try:
            return self._samples[position]
        except KeyError:
            raise KeyError(f"Position {position} is absent from prefetched step {self.step}; CPU fallback is forbidden") from None

    def __iter__(self):
        return iter(self.positions)

    def __len__(self):
        return len(self.positions)


@contextmanager
def use_cached_samples(engine, samples):
    """Temporarily supply only cached samples to the parent's original engine.

    Caller still invokes engine.prepare_atomic_device(..., version=samples.step)
    on the parent thread.  No CPU/NPU augmentation or random input is moved.
    """
    if not isinstance(samples, StepSamples):
        raise ValueError("A validated StepSamples cache is required")
    if (engine.seed != samples.seed or engine.config.get("batch_size", 256) != samples.batch_size
            or len(engine.dataset) != samples.dataset_length):
        raise ValueError("Prefetched sample identities differ from the parent engine")
    original = engine._sample
    engine._sample = samples.__getitem__
    try:
        yield samples
    finally:
        engine._sample = original


class ResidentDataProvider:
    """Spawn at most eight CPU readers and retain at most two step windows.

    Prefetch positions may cover any ordered collection of complete atomic-4
    units within one global step (usually a single card's eight positions).
    Steps are submitted and consumed in increasing order.  A full window, old
    step, duplicate, partial unit or mismatched result is an explicit error.

    CPU environment variables apply briefly while all child processes start;
    the original parent environment is restored before construction returns.
    Accelerator imports in the script entry point must be inside main(), so
    the spawn interpreter can import it without loading torch_npu.
    """

    def __init__(self, config, *, workers=2, prefetch_steps=2, dataset_factory=None, startup_timeout=60):
        self.config = copy.deepcopy(dict(config))
        self.seed = _integer(self.config.get("seed", 42), "seed")
        self.batch_size = _integer(self.config.get("batch_size", 256), "batch_size", minimum=ATOMIC_SIZE)
        if self.batch_size % ATOMIC_SIZE:
            raise ValueError("Global batch must contain complete atomic-4 units")
        self.workers = _integer(workers, "workers", minimum=1)
        self.prefetch_steps = _integer(prefetch_steps, "prefetch_steps", minimum=1)
        if self.workers > 8 or self.prefetch_steps > 2:
            raise ValueError("CPU readers are bounded to eight workers and two prefetched steps")
        if isinstance(startup_timeout, bool) or not isinstance(startup_timeout, (int, float)) or startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        self._pending, self._last_submitted, self._last_consumed = OrderedDict(), -1, -1
        self._closed = False
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(self.workers + 1, timeout=startup_timeout)
        startup_queue = context.Queue()
        self._executor = None
        self.worker_pids, self.dataset_length = (), None
        # ProcessPoolExecutor starts workers lazily.  The barrier keeps early
        # workers busy until all N have spawned under the CPU-only environment;
        # no later submit can spawn a worker with restored NPU visibility.
        with _SPAWN_ENV_LOCK:
            previous = {key: os.environ.get(key) for key in _CPU_ENV}
            os.environ.update(_CPU_ENV)
            try:
                self._executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=context,
                    initializer=_initialize_worker,
                    initargs=(self.config, dataset_factory or build_openpi_dataset, startup_queue, barrier))
                ready = [self._executor.submit(_worker_ready) for _ in range(self.workers)]
                deadline, metadata = time.monotonic() + startup_timeout, []
                while len(metadata) < self.workers:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("CPU data worker initialization timed out")
                    try:
                        metadata.append(startup_queue.get(timeout=min(remaining, 1.0)))
                    except queue.Empty:
                        for future in ready:
                            if future.done():
                                future.result()
                lengths = {item["dataset_length"] for item in metadata}
                pids = {item["pid"] for item in metadata}
                if len(lengths) != 1 or len(pids) != self.workers:
                    raise ValueError("CPU reader processes initialized different datasets")
                barrier.wait(timeout=max(0.01, deadline - time.monotonic()))
                for future in ready:
                    future.result(timeout=max(0.01, deadline - time.monotonic()))
                self.dataset_length = lengths.pop()
                self.worker_pids = tuple(sorted(pids))
            except BaseException:
                barrier.abort()
                if self._executor is not None:
                    self._executor.shutdown(wait=True, cancel_futures=True)
                self._closed = True
                raise
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                startup_queue.close()
                startup_queue.join_thread()

    def _ready(self):
        if self._closed:
            raise RuntimeError("CPU data provider is closed")

    def preload_step(self, step, positions):
        self._ready()
        step = _integer(step, "step")
        if step <= self._last_submitted or step <= self._last_consumed:
            raise ValueError("Prefetch step is old, repeated or out of order")
        if len(self._pending) >= self.prefetch_steps:
            raise ValueError("Prefetch window is full; consume its oldest step first")
        positions = tuple(_integer(position, "sample position") for position in positions)
        if not positions or len(positions) % ATOMIC_SIZE or len(positions) > self.batch_size:
            raise ValueError("Prefetch requires complete atomic-4 units within one global batch")
        if any(right <= left for left, right in zip(positions, positions[1:])):
            raise ValueError("Prefetch positions must be ordered and unique")
        units = [positions[start:start + ATOMIC_SIZE] for start in range(0, len(positions), ATOMIC_SIZE)]
        for unit in units:
            describe_units(unit, run_seed=self.seed, version=step, batch_size=self.batch_size)
        futures = [self._executor.submit(_read_atomic_unit, step, unit) for unit in units]
        self._pending[step] = (positions, units, futures)
        self._last_submitted = step

    def get_step(self, step):
        self._ready()
        step = _integer(step, "step")
        if not self._pending or step != next(iter(self._pending)):
            raise ValueError("Requested step is old, absent or not the oldest prefetched step")
        positions, units, futures = self._pending[step]
        samples = {}
        for unit, future in zip(units, futures, strict=True):
            result = future.result()
            if (result["step"] != step or result["positions"] != unit
                    or result["dataset_length"] != self.dataset_length
                    or set(result["samples"]) != set(unit)):
                raise ValueError("CPU sample result does not match its issued step/positions")
            samples.update(result["samples"])
        if tuple(samples) != positions:
            raise ValueError("CPU sample result has missing, repeated or reordered positions")
        self._pending.pop(step)
        self._last_consumed = step
        return StepSamples(step, self.seed, self.batch_size, self.dataset_length,
                           positions, MappingProxyType(samples))

    def close(self):
        if self._closed:
            return
        self._closed = True
        for _, _, futures in self._pending.values():
            for future in futures:
                future.cancel()
        self._pending.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *_):
        self.close()
