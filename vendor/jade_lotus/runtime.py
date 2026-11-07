"""Small, dependency-free pieces of the recovery protocol."""

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time


def emit(run_dir, event, **fields):
    record = {"time": time.time(), "pid": os.getpid(), "event": event, **fields}
    with (Path(run_dir) / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
    print(json.dumps(record, sort_keys=True), flush=True)


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class WriterLockBusy(RuntimeError):
    """Another live writer holds this lock; no guarded work has started."""

    def __init__(self, path):
        self.path = Path(path)
        super().__init__(f"Another trainer owns {self.path}; refusing concurrent writes")


@contextlib.contextmanager
def writer_lock(run_dir):
    """Hold a POSIX advisory lock for the entire writer lifetime.

    GPFS lock behavior across different platform partitions must be acceptance-tested.
    This deliberately has no expiring lease that could admit a second live writer.
    """
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "writer.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WriterLockBusy(path) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class ResumableBatchSampler:
    """Seek directly to a completed-update cursor without replaying decoded data.

    Each index carries its epoch so transforms can have reproducible per-sample RNG.
    Prefetch never advances the authoritative cursor: only optimizer updates do.
    """

    def __init__(self, size, batch_size, seed, completed_updates=0):
        if batch_size < 1 or size < batch_size or completed_updates < 0:
            raise ValueError("Invalid dataset size, batch size, or recovery cursor")
        self.size = size
        self.batch_size = batch_size
        self.seed = seed
        self.completed_updates = completed_updates

    def __iter__(self):
        batches_per_epoch = self.size // self.batch_size
        epoch, offset = divmod(self.completed_updates, batches_per_epoch)
        while True:
            indices = list(range(self.size))
            random.Random(self.seed + epoch).shuffle(indices)
            for batch in range(offset, batches_per_epoch):
                start = batch * self.batch_size
                yield [(epoch, index) for index in indices[start : start + self.batch_size]]
            epoch += 1
            offset = 0


def sample_seed(seed, epoch, index):
    value = f"{seed}:{epoch}:{index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")
