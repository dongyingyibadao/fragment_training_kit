"""Small bounded shared ledger for optional 16-card refill, without devices."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid
from contextlib import contextmanager

from .refill_protocol import atomic_write
from .runtime import writer_lock, WriterLockBusy

BATCH_SIZE = 16


@contextmanager
def bounded_lock(run, seconds=5):
    deadline = time.monotonic() + seconds
    while True:
        context = writer_lock(Path(run) / 'refill/lock')
        try:
            context.__enter__()
            break
        except WriterLockBusy:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.01)
    try:
        yield
    finally:
        context.__exit__(None, None, None)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def immutable(path, value):
    """Publish a complete write-once record before an external side effect."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read(path) != value:
                raise ValueError('Immutable refill identity/authorization changed')
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def load_state(run, config_sha256):
    value = read(Path(run) / 'refill/state.json', {
        'schema': 'fragment-refill-state-v1', 'run_dir': str(Path(run).resolve()),
        'config_sha256': config_sha256, 'next_batch': 1, 'current': None, 'history': []})
    if (value.get('schema') != 'fragment-refill-state-v1'
            or value.get('run_dir') != str(Path(run).resolve()) or value.get('config_sha256') != config_sha256):
        raise ValueError('Refill ledger belongs to another run/configuration')
    return value


def save_state(run, value):
    atomic_write(Path(run) / 'refill/state.json', value)


def cancel(run, state, batch, reason):
    """Caller holds refill/lock. Admission and cancellation are exclusive."""
    if batch['phase'] == 'admitted':
        raise ValueError('Cannot cancel an admitted batch')
    batch.update(phase='cancelled', reason=reason, cancelled_at=time.time())
    save_state(run, state)
    for prefix in batch['allocations']:
        mapping = read(Path(run) / 'refill/allocations' / f'{prefix}.json')
        if not mapping or mapping['batch_id'] != batch['batch_id'] or mapping['role'] != 'candidate':
            raise ValueError('Cancellation mapping does not match its candidate batch')
        immutable(Path(run) / 'refill/cancelled' / f'{prefix}.json', {
            'schema': 'fragment-refill-cancel-v1',
            **{key: mapping[key] for key in ('run_dir', 'config_sha256', 'intent_id', 'worker_prefix', 'batch_id')}})
