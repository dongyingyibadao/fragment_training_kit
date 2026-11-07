"""Bounded host accumulation of physical gradients within one logical task."""

from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np


class HostGradientAccumulator:
    """Average disjoint physical means in FP32, checking every input.

    The single-part path returns the original host arrays. Multiple parts use
    one owned FP32 accumulator and bounded temporary chunks. This does not
    combine parameter versions or submit partially completed tasks.
    """

    def __init__(self, samples, parts, *, workers=8):
        if (type(samples) is not int or samples not in (4, 8, 16)
                or type(parts) is not int or parts < 1 or parts > samples // 4):
            raise ValueError("Invalid logical sample/physical part count")
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError("local_accumulation_workers must be in 1..32")
        self.samples, self.parts = samples, parts
        self.seen_samples = self.seen_parts = 0
        self.arrays = None
        self.seconds = 0.0
        self.pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.pool is not None:
            self.pool.shutdown(wait=True)

    def add(self, arrays, samples):
        started = time.monotonic()
        if (type(samples) is not int or samples not in (4, 8, 16)
                or self.seen_samples + samples > self.samples
                or self.seen_parts >= self.parts):
            raise ValueError("Physical parts exceed the logical task")
        if not arrays or any(value.dtype != np.float32 for value in arrays.values()):
            raise ValueError("Local gradients must be nonempty FP32 arrays")
        if self.arrays is not None and (set(arrays) != set(self.arrays)
                or any(arrays[key].shape != self.arrays[key].shape for key in arrays)):
            raise ValueError("Physical gradient schema changed inside a task")
        first = self.arrays is None
        if first:
            self.arrays = arrays if self.parts == 1 else {key: np.empty_like(value) for key, value in arrays.items()}
        weight = np.float32(samples / self.samples)

        def chunks():
            for key in sorted(arrays):
                source = arrays[key].reshape(-1)
                destination = self.arrays[key].reshape(-1)
                for offset in range(0, source.size, 2 ** 21):
                    yield source[offset:offset + 2 ** 21], destination[offset:offset + 2 ** 21]

        def consume(pair):
            source, destination = pair
            if not np.isfinite(source).all():
                raise FloatingPointError("Nonfinite physical gradient")
            if self.parts == 1:
                return
            if first:
                np.multiply(source, weight, out=destination)
            else:
                np.add(destination, source * weight, out=destination)
            if not np.isfinite(destination).all():
                raise FloatingPointError("Nonfinite local gradient accumulation")

        if self.pool is None:
            for pair in chunks():
                consume(pair)
        else:
            # map consumes chunk views only; buffers stay bounded per worker.
            for _ in self.pool.map(consume, chunks()):
                pass
        self.seen_samples += samples
        self.seen_parts += 1
        self.seconds += time.monotonic() - started

    def finish(self):
        if self.seen_samples != self.samples or self.seen_parts != self.parts:
            raise ValueError("Incomplete logical task cannot publish a gradient")
        return self.arrays
