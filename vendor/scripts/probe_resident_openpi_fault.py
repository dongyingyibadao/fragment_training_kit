#!/usr/bin/env python3
"""Experimental OpenPI resident fault probe, separate from production training.

A CPU controller coordinates separately allocated LOW single-NPU supervisors.
Only a supervisor may kill its own child.  The probe preserves healthy local
contributions when a controlled loss occurs before optimizer application.  It
supports one detected loss before optimizer authorization through CPU-supervisor counters
and exits. It has no durable gradient acceptance or recovery for loss during
optimizer application. NPU recovery remains experimental until
this exact model/optimizer/gradient integration passes on real hardware.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "jade-resident-openpi-controlled-fault-v1"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _wait(readiness, timeout, label, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = readiness()
        if value is not None:
            return value
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for {label}")


def _control_poll_seconds(spec, *, legacy=0.1):
    """Speed up hot control gates; never slow an existing shorter default."""
    return min(spec.get("control_poll_seconds", 0.1), legacy)


def _identity(member):
    return {key: member[key] for key in ("worker_id", "session_id", "pid")}


def _same_identity(record, member):
    return isinstance(record, dict) and all(record.get(key) == value for key, value in _identity(member).items())


def _kill_matches(directive, *, run_id, worker_id, session_id, child_pid):
    return (isinstance(directive, dict) and directive.get("schema") == SCHEMA
            and directive.get("run_id") == run_id and directive.get("action") == "kill_own_child"
            and directive.get("worker_id") == worker_id and directive.get("session_id") == session_id
            and directive.get("pid") == child_pid and isinstance(directive.get("token"), str))


def assignments(members):
    """Partition exactly 64 canonical atomic-4 units, independent of card count."""
    ids = [member["worker_id"] for member in members]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Distinct, nonempty membership is required")
    result = {worker: [] for worker in ids}
    for unit in range(64):
        result[ids[unit % len(ids)]].append(unit)
    validate_coverage(result)
    return result


def validate_coverage(mapping):
    units = [unit for values in mapping.values() for unit in values]
    if (any(type(unit) is not int for unit in units) or len(units) != 64
            or sorted(units) != list(range(64))):
        raise ValueError("A global256 step must cover each atomic-4 unit exactly once")
    return True


def repair_assignments(original, lost_worker, survivors):
    """Replay only the lost rank's entire bundle on one healthy process."""
    ids = [member["worker_id"] for member in survivors]
    if lost_worker not in original or set(ids) != set(original) - {lost_worker}:
        raise ValueError("Recovery must preserve every original surviving worker")
    result = {worker: list(original[worker]) for worker in ids}
    replay = {worker: [] for worker in ids}
    replay[ids[0]] = list(original[lost_worker])
    result[ids[0]].extend(replay[ids[0]])
    validate_coverage(result)
    return result, replay


class StepDataPrefetch:
    """Cache current/next canonical positions; rebuild CPU readers on reassignment.

    Readers never own NPU tensors or preprocessing RNG. A changed membership
    invalidates a prefetched window, never the healthy model or its optimizer.
    """
    def __init__(self, config, *, workers, steps, provider_factory=None):
        if not 1 <= workers <= 8:
            raise ValueError("CPU data workers must be in 1..8")
        self.config, self.workers, self.steps = dict(config), workers, steps
        if provider_factory is None:
            from jade_lotus.resident_data import ResidentDataProvider
            provider_factory = ResidentDataProvider
        self.provider_factory = provider_factory
        self.provider, self.pending, self.generation = None, {}, 0

    @staticmethod
    def positions(step, units):
        units = tuple(units)
        if (not units or len(set(units)) != len(units)
                or any(type(unit) is not int or not 0 <= unit < 64 for unit in units)
                or list(units) != sorted(units)):
            raise ValueError("CPU prefetch requires ordered, distinct canonical units")
        return tuple(step * 256 + unit * 4 + offset for unit in units for offset in range(4))

    def acquire(self, step, units, next_units):
        if type(step) is not int or not 0 <= step < self.steps:
            raise ValueError("Prefetch step is outside this experiment")
        current = self.positions(step, units)
        upcoming = self.positions(step + 1, next_units) if step + 1 < self.steps else None
        desired = {step: current, **({step + 1: upcoming} if upcoming is not None else {})}
        mismatch = any(key not in desired or desired[key] != value for key, value in self.pending.items())
        reason, startup_seconds = None, 0.0
        if self.provider is None or mismatch:
            reason = "membership_positions_changed" if mismatch else "initial_cpu_readers"
            began = time.monotonic()
            self.close()
            self.provider = self.provider_factory(self.config, workers=self.workers, startup_timeout=120)
            self.generation += 1
            startup_seconds = time.monotonic() - began
        for target, positions in desired.items():
            if target not in self.pending:
                self.provider.preload_step(target, positions)
                self.pending[target] = positions
        began = time.monotonic()
        cached = self.provider.get_step(step)
        waited = time.monotonic() - began
        if tuple(cached.positions) != current or cached.step != step:
            raise ValueError("CPU reader returned a different membership's sample positions")
        self.pending.pop(step)
        return cached, {"enabled": True, "workers": self.workers, "prefetch_wait_seconds": waited,
                        "cpu_reader_generation": self.generation, "rebuild_reason": reason,
                        "reader_startup_seconds": startup_seconds, "prefetched_next_step": upcoming is not None,
                        "worker_pids": list(self.provider.worker_pids),
                        "positions_sha256": hashlib.sha256(json.dumps(current).encode()).hexdigest()}

    def close(self):
        if self.provider is not None:
            self.provider.close()
            self.provider = None
        self.pending = {}


def _report_path(run, member, token, phase):
    return Path(run) / "reports" / member["worker_id"] / member["session_id"] / token / f"{phase}.json"


def _record(run, spec, member, token, phase, **fields):
    value = {"schema": SCHEMA, "run_id": spec["run_id"], **_identity(member),
             "token": token, "phase": phase, "time": time.time(), **fields}
    _write(_report_path(run, member, token, phase), value)
    return value


def _reports(run, spec, members, token, phase):
    result = {}
    for member in members:
        value = _read(_report_path(run, member, token, phase))
        if value is None:
            return None
        if (not _same_identity(value, member) or value.get("run_id") != spec["run_id"]
                or value.get("token") != token or value.get("phase") != phase):
            raise ValueError("Report session/run/token fence mismatch")
        result[member["worker_id"]] = value
    return result


def _send(run, spec, members, action, *, token=None, **fields):
    token = token or uuid.uuid4().hex
    for member in members:
        _write(Path(run) / "commands" / f"{member['worker_id']}.json",
               {"schema": SCHEMA, "run_id": spec["run_id"], "action": action,
                "token": token, "recipient": _identity(member), **fields})
    return token


def _free_port():
    with socket.socket() as probe:
        probe.bind(("0.0.0.0", 0))
        return probe.getsockname()[1]


def _group(members, generation, *, port=None):
    leader = members[0]
    return {"generation": generation, "members": members,
            "endpoint": f"tcp://{leader['address']}:{port or leader['port']}"}


def _abort_matches(value, expected, member):
    return (isinstance(value, dict) and value.get("action") == "abort_collective"
            and (value.get("confirmed_peer_exit") is True or value.get("peer_loss_fenced") is True)
            and all(value.get(key) == expected[key] for key in ("run_id", "token", "base_step", "generation"))
            and any(_same_identity(item, member) for item in value.get("members", [])))


def _hold_matches(value, spec, member, command):
    return (isinstance(value, dict) and value.get("action") == "hold_own_child"
            and value.get("run_id") == spec["run_id"] and _same_identity(value, member)
            and all(value.get(key) == command[key] for key in ("token", "base_step"))
            and value.get("generation") == command["group"]["generation"])


class ControllerLiveness:
    """File adapter for monotonic supervisor counters and irreversible fences."""
    def __init__(self, run, spec, *, persisted_fences=(), used_tokens=(), generation_floor=0, epoch=0,
                 last_committed_step=None):
        from jade_lotus.resident_liveness import ResidentLiveness
        self.run, self.spec, self.keys = Path(run), spec, {}
        self.seen_exits = set()
        self._persisted_losses = {}
        self.used_tokens, self.generation_floor = set(used_tokens), generation_floor
        self.last_committed_step = last_committed_step
        self.detector = ResidentLiveness(spec["run_id"],
            heartbeat_timeout_seconds=spec.get("heartbeat_timeout", 15), persisted_fences=persisted_fences)
        self.detector.epoch = epoch

    def register_ready(self, members, receipt):
        from jade_lotus.resident_liveness import SessionKey, Heartbeat
        for member in members:
            record = _read(self.run / "supervisors" / f"{member['worker_id']}.json")
            if not _same_identity(record, member) or record.get("run_id") != self.spec["run_id"]:
                raise ValueError("READY does not match its exact CPU-supervisor registration")
            key = SessionKey.from_supervisor_record(record)
            self.detector.register(key)
            self.keys[member["worker_id"]] = key
            self.detector.observe_heartbeat(Heartbeat.from_supervisor_record(key, record))
            self.detector.mark_ready(key, full_state_receipt=receipt)

    def poll(self):
        from jade_lotus.resident_liveness import Heartbeat, SupervisorExit
        try:
            # A shared-filesystem scan can take longer than a heartbeat lease.
            # Complete all reads before giving any fresh counter an observation
            # timestamp, so later reads cannot expire earlier healthy members.
            observations = []
            fenced = self.detector.fenced_sessions
            for worker, key in tuple(self.keys.items()):
                record = (None if key in fenced else
                          _read(self.run / "supervisors" / f"{worker}.json"))
                path = self.run / "exits" / f"{worker}-{key.session_id}.json"
                observations.append((key, record, path, _read(path)))
            for key, record, _, _ in observations:
                if key not in fenced and key.matches_supervisor_record(record):
                    self.detector.observe_heartbeat(Heartbeat.from_supervisor_record(key, record))
            for key, _, path, ended in observations:
                if path in self.seen_exits or not key.matches_supervisor_record(ended):
                    continue
                if self.detector.phase == "idle" and ended.get("exit_code") == 0:
                    self.seen_exits.add(path)  # Normal final process termination.
                    continue
                self.detector.observe_exit(SupervisorExit.from_supervisor_record(key, ended, evidence_ref=str(path)))
                self.seen_exits.add(path)
            return self.detector.poll()
        finally:
            self._persist_losses()

    def _persist_losses(self):
        # Transition methods also expire leases and may raise before poll can
        # return a notice. A later confirmed exit updates the same event ID.
        latest = {notice.event_id: notice.as_record() for notice in self.detector.loss_history}
        for event_id, record in latest.items():
            if self._persisted_losses.get(event_id) != record:
                _write(self.run / "losses" / f"{event_id}.json", record)
                self._persisted_losses[event_id] = record

    def _transition(self, method, *args, **kwargs):
        try:
            # Readiness/report I/O or controller descheduling may have occurred
            # since the last poll. Refresh immediately before authorization.
            self.poll()
            return method(*args, **kwargs)
        finally:
            self._persist_losses()

    def begin_step(self, step_token, **kwargs):
        from jade_lotus.resident_liveness import FencedSession
        if step_token in self.used_tokens or kwargs["generation"] < self.generation_floor:
            raise FencedSession("Step token or communication generation was retired before this active-group view")
        result = self._transition(self.detector.begin_step, step_token, **kwargs)
        self.used_tokens.add(step_token)
        self.generation_floor = kwargs["generation"]
        return result

    def advance_phase(self, *args, **kwargs):
        return self._transition(self.detector.advance_phase, *args, **kwargs)

    def finish_step(self, *args, **kwargs):
        base_step = self.detector.base_step
        result = self._transition(self.detector.finish_step, *args, **kwargs)
        self.last_committed_step = base_step + 1
        return result

    def reconfigure_pre_apply(self, step_token, new_step_token, **kwargs):
        from jade_lotus.resident_liveness import FencedSession
        if new_step_token in self.used_tokens:
            raise FencedSession("A repaired communication token was already used in an earlier active-group view")
        result = self._transition(self.detector.reconfigure_pre_apply, step_token, new_step_token, **kwargs)
        self.used_tokens.add(new_step_token)
        self.generation_floor = kwargs["generation"]
        return result

    def rebuild_active(self, members, *, generation, committed_step, receipt):
        """Replace the monitored view only at a completed COMMIT boundary.

        Parked sessions leave the active detector, while every irreversible
        session/token/generation fence and the loss evidence remain in force.
        This never resets a pending step or revives a previously fenced worker.
        """
        from dataclasses import asdict
        from jade_lotus.resident_liveness import LivenessError
        if (self.detector.phase != "idle" or self.detector.step_token is not None
                or self.detector.frozen_sessions or type(generation) is not int
                or generation <= self.generation_floor or type(committed_step) is not int
                or committed_step != self.last_committed_step):
            raise LivenessError("Active liveness may rebuild only at an idle COMMIT and a newer communication generation")
        self.poll()  # Persist any final old-view losses before retiring that view.
        new = ControllerLiveness(self.run, self.spec, persisted_fences=self.detector.fenced_sessions,
            used_tokens=self.used_tokens, generation_floor=generation, epoch=self.detector.epoch,
            last_committed_step=self.last_committed_step)
        new.seen_exits = set(self.seen_exits)
        new._persisted_losses = dict(self._persisted_losses)
        new.register_ready(members, receipt=receipt)
        _write(self.run / "topology" / f"liveness-generation-{generation}.json", {
            "run_id": self.spec["run_id"], "committed_step": committed_step, "generation": generation,
            "previous_generation_floor": self.generation_floor, "prior_epoch": self.detector.epoch,
            "active_sessions": [asdict(key) for key in new.keys.values()],
            "retained_fenced_sessions": [asdict(key) for key in sorted(self.detector.fenced_sessions, key=repr)],
            "retained_step_tokens": sorted(self.used_tokens), "full_state_receipt": receipt,
            "parked_sessions_are_not_active_losses": True})
        return new


class FaultMonitor:
    """Always-on CPU polling while communication/apply permission is outstanding."""
    def __init__(self, path, expected, member, cleanup, *, initialize=None, interval=0.05):
        self.path, self.expected, self.member, self.cleanup = path, expected, member, cleanup
        self.initialize, self.ready = initialize, threading.Event()
        self.submission_lock = threading.Lock()
        self.interval, self.armed = interval, threading.Event()
        self.done, self.closed = threading.Event(), threading.Event()
        self.triggered, self.error = False, None
        self.thread = threading.Thread(target=self._run, name="resident-fault-control", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            if self.initialize is not None:
                self.initialize()
        except BaseException as error:
            self.error = error
            self.done.set()
            return
        finally:
            self.ready.set()
        while not self.closed.wait(self.interval):
            try:
                if self.armed.is_set() and _abort_matches(_read(self.path), self.expected, self.member):
                    self.triggered = True
                    with self.submission_lock:
                        self.cleanup()
                    self.done.set()
                    return
            except BaseException as error:
                self.error = error
                self.done.set()
                return

    def wait_ready(self, timeout):
        if not self.ready.wait(timeout):
            raise TimeoutError("Fault monitor did not initialize its device-thread context")
        if self.error is not None:
            raise RuntimeError("Fault monitor device-thread initialization failed") from self.error

    def finish(self, timeout):
        if not self.done.wait(timeout):
            raise TimeoutError("CPU fault monitor did not finish the exact collective recovery")
        if self.error is not None:
            raise RuntimeError("NPU fault cleanup failed") from self.error

    def close(self):
        self.closed.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise TimeoutError("Fault cleanup thread is still running")


def _cleanup_backend(backend, torch_npu, device_index, publish):
    """Experimental stop -> abort -> clear -> restart(False); destroy is main-thread only."""
    rows = []
    def call(name, function):
        began = time.monotonic()
        row = {"operation": name, "started_at": time.time(), "passed": False}
        rows.append(row)
        publish(rows)
        try:
            value = function()
            row["passed"] = True
            return value
        except BaseException as error:
            row["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            row["seconds"] = time.monotonic() - began
            publish(rows)
    call("pause_watchdog", lambda: backend.set_watchdog_status(2))
    stopped = call("stop_device", lambda: torch_npu.npu.stop_device(device_index))
    if stopped != 0:
        raise RuntimeError(f"stop_device failed: {stopped!r}")
    call("abort_hccl_comm", lambda: backend.abort_hccl_comm("controlled resident OpenPI peer loss"))
    call("clear_workmeta_list", backend.clear_workmeta_list)
    call("restart_device_false", lambda: torch_npu.npu.restart_device(device_index, rebuild_all_resources=False))
    return rows


def backward_units(engine, *, base_step, units, clear=True):
    """Accumulate SUM contributions with a fixed global256 denominator.

    Unequal rank loads after shrink never change a sample's loss weight. With
    clear=False, healthy existing local gradients stay in place during replay.
    """
    if engine.config.get("resident_adapter"):
        from fragment_training_kit.adapters import accumulate_units
        return accumulate_units(engine, base_step=base_step, units=units, clear=clear)
    from jade_lotus.npu_gradient_model import _slice_observation
    units = list(units)
    if len(units) != len(set(units)) or any(type(unit) is not int or not 0 <= unit < 64 for unit in units):
        raise ValueError("Invalid atomic unit assignment")
    engine._assert_no_checkpointing()
    if clear:
        engine.model.zero_grad(set_to_none=True)
    loss_total, calls, data_seconds, backward_seconds = 0.0, 0, 0.0, 0.0
    began = time.monotonic()
    with engine._checkpoint_guard():
        for unit_id in units:
            position = base_step * 256 + unit_id * 4
            phase = time.monotonic()
            prepared = engine.prepare_atomic_device(range(position, position + 4), version=base_step)
            engine._sync()
            data_seconds += time.monotonic() - phase
            unit = prepared.units[0]
            for start in range(0, 4, engine.microbatch_size):
                phase = time.monotonic()
                end = min(start + engine.microbatch_size, 4)
                observation = _slice_observation(unit.observation, start, end)
                loss = engine.model(observation, unit.actions[start:end], noise=unit.noise[start:end],
                                    time=unit.flow_time[start:end]).mean()
                if not bool(engine.torch.isfinite(loss).item()):
                    raise FloatingPointError("Nonfinite resident OpenPI loss")
                weight = (end - start) / 256
                (loss * weight).backward()
                loss_total += float(loss.detach().item()) * weight
                calls += 1
                del loss, observation
                engine._sync()
                backward_seconds += time.monotonic() - phase
            del prepared, unit
    engine._sync()
    if engine.checkpoint_call_count:
        raise AssertionError("Activation recomputation occurred")
    return {"loss_contribution": loss_total, "backward_calls": calls, "units": units,
            "samples": len(units) * 4, "seconds": time.monotonic() - began,
            "data_seconds": data_seconds, "backward_seconds": backward_seconds,
            "timing_scope": "synchronized sequential data preparation and microbatch forward/backward"}


def reduction_buffers(parameters, torch):
    """Pack independent native buffers; HCCL never writes a healthy local grad."""
    groups = {}
    for name, parameter in sorted(parameters.items()):
        gradient = parameter.grad
        if gradient is None or gradient.dtype != parameter.dtype:
            raise ValueError(f"Missing or mistyped native gradient: {name}")
        groups.setdefault(gradient.dtype, []).append((name, parameter))
    result = []
    for dtype, leaves in sorted(groups.items(), key=lambda item: str(item[0])):
        total = sum(parameter.numel() for _, parameter in leaves)
        flat = torch.empty(total, device=leaves[0][1].device, dtype=dtype)
        offset, entries = 0, []
        for name, parameter in leaves:
            size = parameter.numel()
            flat[offset:offset + size].copy_(parameter.grad.reshape(-1))
            entries.append((name, parameter, offset, size))
            offset += size
        result.append((flat, entries))
    return result


def install_reduced_gradients(buffers):
    for flat, entries in buffers:
        for _, parameter, offset, size in entries:
            parameter.grad.copy_(flat[offset:offset + size].view_as(parameter))


def _reject_active_dropout(model, torch):
    """A replay may move ranks; this probe requires a stateless forward RNG path."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.training and module.p > 0:
            raise ValueError("Active dropout is outside this probe's deterministic atomic replay contract")
        config = getattr(module, "config", None)
        for field in ("attention_dropout", "hidden_dropout_prob", "attn_pdrop", "dropout_rate"):
            value = getattr(config, field, 0)
            if isinstance(value, (int, float)) and value > 0:
                raise ValueError(f"Active {field} is outside deterministic replay")


def _memory_status(engine):
    npu, device = engine.torch.npu, engine.torch_device
    free, total = npu.mem_get_info(device)
    return {"allocated_bytes": npu.memory_allocated(device), "reserved_bytes": npu.memory_reserved(device),
            "peak_allocated_bytes": npu.max_memory_allocated(device), "device_free_bytes": free,
            "device_total_bytes": total}


def _admit_buffers(engine):
    memory = _memory_status(engine)
    required = sum(parameter.numel() * parameter.element_size() for parameter in engine.parameters.values())
    reusable = memory["device_free_bytes"] + max(0, memory["reserved_bytes"] - memory["allocated_bytes"])
    memory.update(independent_gradient_buffer_bytes=required, reusable_bytes=reusable,
                  minimum_headroom_bytes=2 * 1024**3)
    if reusable < required + memory["minimum_headroom_bytes"]:
        raise MemoryError(f"Independent reduction buffers do not fit with 2GiB headroom: {memory}")
    return memory


def _pointer_snapshot(engine, optimizer, *, gradients):
    pointers = {f"model/{name}": tensor.data_ptr()
                for name, tensor in engine.model.named_parameters(remove_duplicate=False)}
    pointers.update({f"buffer/{name}": tensor.data_ptr()
                     for name, tensor in engine.model.named_buffers(remove_duplicate=False)})
    for name, parameter in sorted(engine.parameters.items()):
        for key, value in sorted(optimizer.state.get(parameter, {}).items()):
            if not engine.torch.is_tensor(value):
                raise TypeError("Expected tensor-valued native AdamW state")
            pointers[f"optimizer/{name}/{key}"] = value.data_ptr()
        if gradients:
            if parameter.grad is None:
                raise ValueError(f"Missing local gradient: {name}")
            pointers[f"gradient/{name}"] = parameter.grad.data_ptr()
    return pointers


def _fingerprint(engine, optimizer, *, step, gradients):
    """Hash actual full native p/m/v bytes and storage pointers, chunked on CPU."""
    from jade_lotus.resident_state import hash_native_state
    torch = engine.torch
    native = hash_native_state(engine.model, optimizer, step=step)
    result = {key: native[key] for key in ("model_sha256", "optimizer_sha256", "state_sha256", "step", "precision_contract")}
    pointers = _pointer_snapshot(engine, optimizer, gradients=gradients)
    if gradients:
        digest = hashlib.sha256()
        for name, parameter in sorted(engine.parameters.items()):
            tensor = parameter.grad
            if tensor is None:
                raise ValueError(f"Missing gradient: {name}")
            metadata = {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            digest.update(json.dumps(metadata, sort_keys=True).encode())
            pointers[f"gradient/{name}"] = tensor.data_ptr()
            for chunk in tensor.detach().reshape(-1).split(4 * 1024**2):
                host = chunk.to("cpu", copy=True)
                digest.update(host.view(torch.uint8).numpy().tobytes())
        result["gradient_sha256"] = digest.hexdigest()
    result["optimizer_config_sha256"] = _optimizer_config_hash(optimizer)
    result["pointers"] = pointers
    return result


def _optimizer_config_hash(optimizer):
    groups = [{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups]
    return hashlib.sha256(json.dumps({"groups": groups, "defaults": optimizer.defaults}, sort_keys=True).encode()).hexdigest()


def _init_group(dist, group, member, timeout):
    rank = next(i for i, item in enumerate(group["members"]) if _same_identity(item, member))
    dist.init_process_group("hccl", init_method=group["endpoint"], rank=rank,
                            world_size=len(group["members"]), timeout=timedelta(seconds=timeout))


def _submit(dist, buffers, bucket_bytes, *, cancelled=None, works=None, submission_lock=None,
            diagnostics=None):
    works = [] if works is None else works
    if diagnostics is not None:
        started = time.monotonic()
        submission = {"started_at": time.time(), "buckets": [], "completed": False}
        diagnostics["submission"] = submission
    try:
        for flat, _ in buffers:
            elements = max(1, bucket_bytes // flat.element_size())
            for bucket in flat.split(elements):
                with submission_lock if submission_lock is not None else nullcontext():
                    if cancelled is not None and cancelled():
                        raise RuntimeError("Cancelled old-group collective submission")
                    if diagnostics is None:
                        works.append(dist.all_reduce(bucket, op=dist.ReduceOp.SUM, async_op=True))
                    else:
                        row = {"index": len(works), "numel": bucket.numel(),
                               "bytes": bucket.numel() * bucket.element_size(),
                               "dtype": str(bucket.dtype), "completed": False}
                        submission["buckets"].append(row)
                        bucket_started = time.monotonic()
                        row["started_at"] = time.time()
                        try:
                            works.append(dist.all_reduce(bucket, op=dist.ReduceOp.SUM, async_op=True))
                            row["completed"] = True
                        finally:
                            row.update(seconds=time.monotonic() - bucket_started, finished_at=time.time())
                if cancelled is not None and cancelled():
                    raise RuntimeError("Cancelled old-group collective submission")
        if diagnostics is not None:
            submission["completed"] = True
    finally:
        if diagnostics is not None:
            submission.update(seconds=time.monotonic() - started, finished_at=time.time(),
                              bucket_seconds=sum(row["seconds"] for row in submission["buckets"]))
    return works


def _wait_works(works, engine, timeout, *, cancelled=None, diagnostics=None):
    def check():
        if cancelled is not None and cancelled():
            raise RuntimeError("CPU fault monitor cancelled old-group Work waits")
    if diagnostics is not None:
        started = time.monotonic()
        waits = {"started_at": time.time(), "buckets": [], "completed": False}
        diagnostics["work_wait"] = waits
        metadata = {row["index"]: row for row in diagnostics.get("submission", {}).get("buckets", [])}
    try:
        for index, work in enumerate(works):
            check()
            if diagnostics is None:
                work.wait(timedelta(seconds=timeout))
            else:
                row = {key: value for key, value in metadata.get(index, {}).items()
                       if key in ("index", "numel", "bytes", "dtype")}
                row.update(index=index, completed=False)
                waits["buckets"].append(row)
                bucket_started = time.monotonic()
                row["started_at"] = time.time()
                try:
                    work.wait(timedelta(seconds=timeout))
                    row["completed"] = True
                finally:
                    row.update(seconds=time.monotonic() - bucket_started, finished_at=time.time())
            check()
        check()
        if diagnostics is not None:
            waits["completed"] = True
    finally:
        if diagnostics is not None:
            waits.update(seconds=time.monotonic() - started, finished_at=time.time(),
                         bucket_seconds=sum(row["seconds"] for row in waits["buckets"]))
    if diagnostics is None:
        engine._sync()
    else:
        started = time.monotonic()
        sync = {"started_at": time.time(), "completed": False}
        diagnostics["final_sync"] = sync
        try:
            engine._sync()
            sync["completed"] = True
        finally:
            sync.update(seconds=time.monotonic() - started, finished_at=time.time())
    check()


def _wait_apply_or_abort(run, token, timeout, monitor=None, *, interval=0.1):
    def readiness():
        if monitor is not None and monitor.triggered:
            return {"action": "recover_communication"}
        return _read(Path(run) / "apply" / f"{token}.json")
    return _wait(readiness, timeout, "optimizer permission or fenced communication loss", interval=interval)


def _worker_step(run, spec, member, command, engine, optimizer, group, progress, prefetch=None):
    import torch.distributed as dist
    import torch_npu
    from jade_lotus.coordinator import learning_rate
    token, step = command["token"], command["base_step"]
    step_started, step_started_at = time.monotonic(), time.time()
    timings = {"state_preservation_hash_seconds": 0.0, "gradient_buffer_pack_seconds": 0.0,
               "communication_seconds": 0.0, "fault_recovery_total_seconds": 0.0}
    communication_diagnostics = None
    diagnostic_steps = spec.get("communication_diagnostics_steps", 0)
    if (spec.get("communication_diagnostics", False)
            and (diagnostic_steps == 0 or 0 <= step - spec.get("start_step", 0) < diagnostic_steps)):
        communication_diagnostics = {
            "schema": "jade-resident-communication-timing-v1", "attempts": [],
            "clock_scope": "Durations use this worker's monotonic clock; wall timestamps require clock alignment across workers.",
            "timing_scope": "Submission includes host enqueue/backpressure; work waits include peer arrival and stream dependencies, not pure network transfer. Report timing includes serialization, write, fsync and rename.",
        }

    def begin_communication(attempt_phase):
        if communication_diagnostics is None:
            return None
        attempt = {"phase": attempt_phase, "generation": group["generation"],
                   "rank": next(i for i, item in enumerate(group["members"]) if _same_identity(item, member)),
                   "world_size": len(group["members"]), "started_at": time.time()}
        communication_diagnostics["attempts"].append(attempt)
        return attempt

    def finish_communication(attempt, seconds, error=None):
        if attempt is None:
            return
        attempt.update(seconds=seconds, finished_at=time.time(), error=error,
                       bucket_count=len(attempt.get("submission", {}).get("buckets", [])))
        attempt["bytes"] = sum(row["bytes"] for row in attempt.get("submission", {}).get("buckets", []))
        measured = sum(attempt.get(stage, {}).get("seconds", 0.0)
                       for stage in ("submission", "submitted_report", "work_wait", "final_sync"))
        # This remainder also contains error reporting/fault-monitor waits on
        # a failed original attempt. It is not charged to network transfer.
        attempt["unattributed_seconds"] = seconds - measured

    def submitted_report(attempt, report_phase, **fields):
        if attempt is None:
            return _record(run, spec, member, token, report_phase, **fields)
        started = time.monotonic()
        record_timing = {"phase": report_phase, "started_at": time.time(), "completed": False}
        attempt["submitted_report"] = record_timing
        try:
            result = _record(run, spec, member, token, report_phase, **fields)
            record_timing["completed"] = True
            return result
        finally:
            record_timing.update(seconds=time.monotonic() - started, finished_at=time.time())
    if command["group"] != group or step != command["expected_step"]:
        raise ValueError("Training group/base-step mismatch")
    units = command["assignments"][member["worker_id"]]
    context = nullcontext()
    prefetch_metrics = {"enabled": False, "prefetch_wait_seconds": 0.0, "reader_startup_seconds": 0.0}
    if prefetch is not None:
        from jade_lotus.resident_data import use_cached_samples
        progress("prefetch_current_and_next")
        next_units = assignments(group["members"])[member["worker_id"]]
        cache, prefetch_metrics = prefetch.acquire(step, units, next_units)
        context = use_cached_samples(engine, cache)
    progress("backward")
    with context:
        local = backward_units(engine, base_step=step, units=units)
    data_seconds, backward_seconds = local["data_seconds"], local["backward_seconds"]
    _record(run, spec, member, token, "local_ready", generation=group["generation"],
            base_step=step, complete_local_backward=True, local=local)
    planned_victim = command.get("fault_victim")
    planned_survivor = bool(planned_victim and not _same_identity(planned_victim, member))
    hold = _read(run / "holds" / f"{member['worker_id']}-{member['session_id']}.json")
    external_hold = _hold_matches(hold, spec, member, command)
    if (planned_victim and not planned_survivor) or external_hold:
        phase = "external_held" if external_hold else "submitted"
        _record(run, spec, member, token, phase, generation=group["generation"], base_step=step,
                collectives=0, local=local, held_before_collective=True, prefetch=prefetch_metrics,
                fault_injection_stage="full_backward_finished_before_collective",
                injector_is_independent_of_controller=external_hold)
        progress("external_hold_before_collective" if external_hold else "fault_victim_held_before_collective")
        while True:
            time.sleep(0.1)  # Only this job's supervisor may kill its own child.
    pointers = _pointer_snapshot(engine, optimizer, gradients=True)
    phase = time.monotonic()
    audit = planned_survivor or spec.get("audit_before_collective", False)
    baseline = _fingerprint(engine, optimizer, step=step, gradients=True) if audit else None
    timings["state_preservation_hash_seconds"] += time.monotonic() - phase
    phase = time.monotonic()
    admission = _admit_buffers(engine)
    buffers = reduction_buffers(engine.parameters, engine.torch)
    engine._sync()
    memory_after_buffers = _memory_status(engine)
    timings["gradient_buffer_pack_seconds"] += time.monotonic() - phase
    backend = dist.distributed_c10d._get_default_group()._get_backend(engine.torch_device)
    fence = {"run_id": spec["run_id"], "token": token, "base_step": step, "generation": group["generation"]}
    def initialize_monitor():
        engine.torch.npu.set_device(engine.torch_device)
        engine._sync()  # Before submissions, with this CPU thread's device context.
    monitor = FaultMonitor(run / "abort" / f"{token}.json", fence, member,
        lambda: _cleanup_backend(backend, torch_npu, engine.torch_device.index,
            lambda rows: _record(run, spec, member, token, "cleanup_progress", operations=rows)),
        initialize=initialize_monitor).start()
    decision, collective_error, recovered_this_step = None, None, False
    works = []
    progress("collective_wait")
    try:
        monitor.wait_ready(spec["phase_timeout"])
        monitor.armed.set()  # Also covers a loss during the submission loop.
        communication_started = time.monotonic()
        attempt = begin_communication("original")
        try:
            _submit(dist, buffers, spec["bucket_bytes"], cancelled=lambda: monitor.triggered,
                    works=works, submission_lock=monitor.submission_lock, diagnostics=attempt)
            submitted_report(attempt, "submitted", generation=group["generation"], base_step=step,
                    collectives=len(works), local=local, buffer_admission=admission,
                    memory_after_buffers=memory_after_buffers, held_before_collective=False,
                    prefetch=prefetch_metrics, always_on_cpu_monitor=True)
            _wait_works(works, engine, spec["collective_timeout"], cancelled=lambda: monitor.triggered,
                        diagnostics=attempt)
        except BaseException as error:
            collective_error = {"type": type(error).__name__, "message": str(error)}
            if not planned_survivor and not monitor.triggered:
                _record(run, spec, member, token, "collective_failed", generation=group["generation"],
                        base_step=step, error=collective_error)
                monitor.done.wait(spec.get("heartbeat_timeout", 15) + 2)
                if not monitor.triggered:
                    raise
        communication_seconds = time.monotonic() - communication_started
        timings["communication_seconds"] += communication_seconds
        finish_communication(attempt, communication_seconds, collective_error)
        if not planned_survivor and not monitor.triggered:
            _record(run, spec, member, token, "reduced", generation=group["generation"], base_step=step)
            progress("waiting_apply")
            phase = time.monotonic()
            decision = _wait_apply_or_abort(run, token, spec["phase_timeout"], monitor,
                                            interval=_control_poll_seconds(spec))
            timings["apply_wait_seconds"] = time.monotonic() - phase
        if planned_survivor or monitor.triggered or (decision and decision.get("action") == "recover_communication"):
            recovered_this_step = True
            recovery_started = time.monotonic()
            monitor.finish(spec["phase_timeout"])
            monitor.close()
            monitor = None
            dist.destroy_process_group()
            actual_pointers = _pointer_snapshot(engine, optimizer, gradients=True)
            if actual_pointers != pointers:
                raise AssertionError("Healthy native state or local-gradient storage changed")
            phase = time.monotonic()
            actual = _fingerprint(engine, optimizer, step=step, gradients=True) if audit else None
            timings["state_preservation_hash_seconds"] += time.monotonic() - phase
            if audit and actual != baseline:
                raise AssertionError("Healthy native p/m/v/gradient bytes changed during recovery")
            _record(run, spec, member, token, "recovered", generation=group["generation"], base_step=step,
                    same_pid=True, full_native_state_and_gradient_unchanged=bool(audit),
                    state_storage_pointers_unchanged=True, full_byte_audit_enabled=bool(audit),
                    state_before=baseline, state_after=actual, collective_error=collective_error,
                    next_port=_free_port())
            recovery = _wait(lambda: _read(run / "recovery" / f"{token}.json"), spec["phase_timeout"],
                             "shrink plan", interval=_control_poll_seconds(spec))
            if (recovery.get("run_id") != spec["run_id"] or recovery.get("token") != token
                    or recovery.get("base_step") != step or recovery["group"]["generation"] != group["generation"] + 1):
                raise ValueError("Stale shrink plan")
            group = recovery["group"]
            _init_group(dist, group, member, spec["collective_timeout"])
            del buffers, works
            replay = recovery["replay"][member["worker_id"]]
            replay_metrics = backward_units(engine, base_step=step, units=replay, clear=False)
            data_seconds += replay_metrics["data_seconds"]
            backward_seconds += replay_metrics["backward_seconds"]
            phase = time.monotonic()
            admission = _admit_buffers(engine)
            buffers = reduction_buffers(engine.parameters, engine.torch)
            engine._sync()
            memory_after_buffers = _memory_status(engine)
            timings["gradient_buffer_pack_seconds"] += time.monotonic() - phase
            communication_started = time.monotonic()
            attempt = begin_communication("recovery")
            works = _submit(dist, buffers, spec["bucket_bytes"], diagnostics=attempt)
            submitted_report(attempt, "repair_submitted", generation=group["generation"], base_step=step,
                    replay=replay_metrics, preserved_original_units=units, buffer_admission=admission,
                    memory_after_buffers=memory_after_buffers)
            _wait_works(works, engine, spec["collective_timeout"], diagnostics=attempt)
            communication_seconds = time.monotonic() - communication_started
            timings["communication_seconds"] += communication_seconds
            finish_communication(attempt, communication_seconds)
            timings["fault_recovery_total_seconds"] = time.monotonic() - recovery_started
            _record(run, spec, member, token, "reduced", generation=group["generation"], base_step=step)
            progress("waiting_apply")
            phase = time.monotonic()
            decision = _wait_apply_or_abort(run, token, spec["phase_timeout"],
                                            interval=_control_poll_seconds(spec))
            timings["apply_wait_seconds"] = time.monotonic() - phase
        if monitor is not None:
            # APPLY is serialized by the controller. It never sends an abort
            # after authorizing optimizer writes; such a loss requires recovery.
            monitor.close()
            if monitor.triggered:
                raise RuntimeError("Communication loss raced optimizer authorization; recovery_required")
            monitor = None
        if (decision.get("run_id") != spec["run_id"] or decision.get("token") != token
                or decision.get("base_step") != step or decision.get("generation") != group["generation"]
                or not any(_same_identity(item, member) for item in decision.get("members", []))):
            raise ValueError("Apply decision does not name this exact live step/session")
        phase = time.monotonic()
        install_reduced_gradients(buffers)
        norm = engine.torch.nn.utils.clip_grad_norm_(list(engine.parameters.values()),
                    float(engine.config.get("clip_gradient_norm", 1.0)), foreach=False)
        if not bool(engine.torch.isfinite(norm).item()):
            raise FloatingPointError("Nonfinite reduced gradient norm")
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate(engine.config, step)
        optimizer.step()
        engine._sync()
        optimizer.zero_grad(set_to_none=True)
        timings.update(install_clip_optimizer_seconds=time.monotonic() - phase,
                       data_seconds=data_seconds, backward_seconds=backward_seconds,
                       prefetch_wait_seconds=prefetch_metrics["prefetch_wait_seconds"],
                       cpu_reader_startup_seconds=prefetch_metrics["reader_startup_seconds"],
                       worker_step_seconds=time.monotonic() - step_started,
                       worker_step_started_at=step_started_at, update_completed_at=time.time(),
                       fault_recovery_total_overlaps_phase_timings=recovered_this_step)
        _record(run, spec, member, token, "updated", generation=group["generation"], completed_step=step + 1,
                grad_norm=float(norm.item()), local=local, checkpoint_calls=engine.checkpoint_call_count,
                memory=_memory_status(engine), prefetch=prefetch_metrics, timings=timings,
                **({"communication_diagnostics": communication_diagnostics} if communication_diagnostics is not None else {}))
        return group
    finally:
        if monitor is not None:
            monitor.close()


def child(args):
    run = args.run_dir.resolve()
    spec = _read(run / "spec.json")
    if spec is None:
        raise ValueError("Controller specification is missing")
    member = {"worker_id": args.worker_id, "session_id": args.session_id, "pid": os.getpid(),
              "hostname": socket.gethostname(), "address": socket.gethostbyname(socket.gethostname()),
              "port": _free_port(), "started_at": time.time(),
              "device_index": args.device_index, "allocated_devices": args.allocated_devices,
              "allocation_group": os.environ.get("JADE_ALLOCATION_GROUP", args.worker_id)}
    active_status = {"token": None, "base_step": None, "generation": None}
    def progress(phase, *, timeout_seconds=None):
        _write(run / "status" / f"{args.worker_id}-{args.session_id}.json",
               {**_identity(member), "run_id": spec["run_id"], "phase": phase,
                **active_status, "deadline": time.time() + (spec["phase_timeout"] if timeout_seconds is None else timeout_seconds)})
    prefetch = None
    try:
        progress("import_and_load_model")
        import torch
        import torch.distributed as dist
        import torch_npu  # noqa: F401 -- the allocated child alone owns an NPU
        from jade_lotus.coordinator import learning_rate
        config_path = Path(spec["config"])
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != spec["config_sha256"]:
            raise ValueError("Training config changed after controller startup")
        config = json.loads(config_path.read_text())
        if config.get("batch_size") != 256 or config.get("disable_activation_checkpointing") is not True:
            raise ValueError("Probe requires global256 with recomputation disabled")
        if torch.npu.device_count() != args.allocated_devices or not 0 <= args.device_index < args.allocated_devices:
            raise ValueError("Logical device and visible count must match the allocated job")
        config.update(npu_gradient_export="fp32_snapshot_v1", npu_gradient_transfer="leaf_copy_v1")
        if config.get("resident_adapter"):
            from fragment_training_kit.adapters import create_engine
            engine = create_engine(config, device_index=args.device_index)
        else:
            from jade_lotus.npu_gradient_model import TorchNpuGradientEngine
            engine = TorchNpuGradientEngine(config, device_index=args.device_index)
        _reject_active_dropout(engine.model, torch)
        torch.npu.reset_peak_memory_stats(engine.torch_device)
        optimizer = torch.optim.AdamW(list(engine.parameters.values()), lr=learning_rate(config, 0),
                                     betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
        initial_state = _restore_initial_state(engine, optimizer, spec, replacement=args.replacement,
                                               progress=progress)
        if spec.get("data_workers", 0):
            prefetch = StepDataPrefetch(config, workers=spec["data_workers"], steps=spec["steps"])
        engine._sync()
        _write(run / "ready" / f"{args.worker_id}.json", {"run_id": spec["run_id"], **member,
                "replacement": args.replacement, "loaded_at": time.time(), **initial_state,
                **({"adapter_device": {"resident_adapter": config["resident_adapter"],
                    "device": str(engine.torch_device), "visible_devices": torch.npu.device_count(),
                    "device_index": args.device_index, "allocated_devices": args.allocated_devices,
                    "device_name": torch.npu.get_device_name(args.device_index)}} if config.get("resident_adapter") else {})})
        from jade_lotus.topology_cohort_runtime import CommittedBoundaryLease
        boundary_lease = CommittedBoundaryLease(spec["phase_timeout"])
        group, completed_step, previous, parked = None, initial_state["completed_step"], None, False
        while True:
            idle_timeout = boundary_lease.remaining()
            progress("idle_boundary", timeout_seconds=idle_timeout)
            independent_standby = group is None and (parked or (
                spec.get("topology_slots", spec["cards"]) > spec["cards"] and args.replacement))
            next_idle_refresh = time.monotonic() + min(5.0, spec["phase_timeout"] / 3)
            def next_command():
                nonlocal next_idle_refresh
                if independent_standby and time.monotonic() >= next_idle_refresh:
                    progress("parked_idle" if parked else "standby_idle")
                    next_idle_refresh = time.monotonic() + min(5.0, spec["phase_timeout"] / 3)
                value = _read(run / "commands" / f"{args.worker_id}.json")
                if (value and value.get("run_id") == spec["run_id"] and value.get("token") != previous
                        and _same_identity(value.get("recipient"), member)):
                    return value
                return None
            try:
                command = _wait(next_command, idle_timeout, "fenced worker command",
                                interval=_control_poll_seconds(spec))
            except TimeoutError:
                if independent_standby:
                    continue  # A healthy parked/standby worker may wait many steps.
                raise
            previous, action = command["token"], command["action"]
            if action not in {"hold_committed_boundary", "release_committed_boundary"}:
                boundary_lease.release()
            active_status.update(token=previous, base_step=command.get("base_step"),
                                 generation=command.get("group", {}).get("generation"))
            if action == "init":
                if group is not None:
                    raise ValueError("Initial group already exists")
                group = command["group"]
                progress("init_group")
                _init_group(dist, group, member, spec["collective_timeout"])
                _record(run, spec, member, previous, "initialized", generation=group["generation"])
            elif action == "step":
                if group is None or command["base_step"] != completed_step:
                    raise ValueError("Cannot train an unaligned native optimizer")
                group = _worker_step(run, spec, member, command, engine, optimizer, group, progress, prefetch)
                completed_step += 1
            elif action == "export":
                if command["expected_step"] != completed_step:
                    raise ValueError("Export boundary step mismatch")
                progress("export_native_state")
                receipt = _export_state(engine, optimizer, command["path"], completed_step, command["state_metadata"])
                _record(run, spec, member, previous, "exported", completed_step=completed_step,
                        receipt=receipt, next_port=_free_port())
            elif action == "load":
                if group is not None or not args.replacement:
                    raise ValueError("Only an independent loading replacement may load a join snapshot")
                progress("load_native_state")
                receipt = _load_state(engine, optimizer, command["path"], command["expected_step"], command["state_metadata"])
                completed_step = command["expected_step"]
                _record(run, spec, member, previous, "loaded", completed_step=completed_step, receipt=receipt)
            elif action == "hold_committed_boundary":
                if group is None:
                    raise ValueError("Only an active committed group may receive a boundary hold")
                duration = boundary_lease.arm(command, completed_step=completed_step,
                                              generation=group["generation"])
                progress("holding_committed_boundary", timeout_seconds=boundary_lease.remaining())
                _record(run, spec, member, previous, "boundary_held", completed_step=completed_step,
                        generation=group["generation"], hold_seconds=duration,
                        bounded_controller_wait=True, lease_renewable=False)
            elif action == "release_committed_boundary":
                if (group is None or boundary_lease.deadline is None
                        or command["expected_step"] != completed_step
                        or command["expected_generation"] != group["generation"]):
                    raise ValueError("Boundary release requires the exact held committed group")
                boundary_lease.release()
                progress("released_committed_boundary")
                _record(run, spec, member, previous, "boundary_released", completed_step=completed_step,
                        generation=group["generation"])
            elif action == "park":
                if (group is None or command["expected_step"] != completed_step
                        or command["expected_generation"] != group["generation"]):
                    raise ValueError("Park requires the exact idle committed group/step boundary")
                progress("park_committed_group")
                engine._sync()
                dist.destroy_process_group()
                group, parked = None, True
                if prefetch is not None:
                    prefetch.close()
                state = _fingerprint(engine, optimizer, step=completed_step, gradients=False)
                _record(run, spec, member, previous, "parked", completed_step=completed_step,
                        model_sha256=state["model_sha256"], optimizer_sha256=state["optimizer_sha256"],
                        optimizer_config_sha256=state["optimizer_config_sha256"])
            elif action == "load_parked":
                if group is not None or not parked or command["expected_step"] < completed_step:
                    raise ValueError("Only an explicitly parked worker may load a current boundary state")
                progress("load_parked_native_state")
                receipt = _load_state(engine, optimizer, command["path"], command["expected_step"], command["state_metadata"])
                completed_step = command["expected_step"]
                _record(run, spec, member, previous, "loaded", completed_step=completed_step, receipt=receipt,
                        explicitly_parked=True)
            elif action == "describe_native_state":
                if completed_step != command["expected_step"]:
                    raise ValueError("Native blueprint requires the exact committed boundary")
                from jade_lotus.resident_state_transport import create_native_state_blueprint
                blueprint = create_native_state_blueprint(engine.model, optimizer, step=completed_step,
                    metadata=command["state_metadata"], source_rank=0, chunk_bytes=64 * 1024**2)
                _record(run, spec, member, previous, "described", completed_step=completed_step,
                        blueprint=blueprint, next_port=_free_port())
            elif action == "device_join":
                from jade_lotus.resident_state_transport import broadcast_native_state
                role = command["roles"][member["worker_id"]]
                boundary = command["expected_step"]
                if role == "replacement":
                    if not args.replacement or group is not None or completed_step > boundary:
                        raise ValueError("Device join replacement must be independently loading")
                elif completed_step != boundary:
                    raise ValueError("Healthy device join state is not at the committed boundary")
                progress("device_join_group")
                if group is not None:
                    dist.destroy_process_group()
                group = command["group"]
                _init_group(dist, group, member, spec["collective_timeout"])
                progress("broadcast_native_state")
                receipt = broadcast_native_state(engine.model, optimizer, command["blueprint"],
                    source_rank=0, role=role, expected_step=boundary,
                    expected_metadata=command["state_metadata"], transport_device=engine.torch_device,
                    verify=True)
                if (not receipt["full_native_state_hash_verified"] or not receipt["parameter_pointers_preserved"]
                        or (role == "replacement" and not receipt["verified_loaded_state"])
                        or (role != "replacement" and not receipt["existing_state_not_overwritten"])):
                    raise ValueError("Device join requires complete actual state byte verification")
                completed_step = boundary
                parked = False
                state = receipt["native_state"]
                _record(run, spec, member, previous, "joined", completed_step=completed_step,
                        model_sha256=state["model_sha256"], optimizer_sha256=state["optimizer_sha256"],
                        optimizer_config_sha256=_optimizer_config_hash(optimizer), transport_receipt=receipt)
            elif action == "join":
                if completed_step != command["expected_step"]:
                    raise ValueError("Join requires aligned full native p/m/v and step")
                progress("join_group")
                if group is not None:
                    dist.destroy_process_group()
                group = command["group"]
                _init_group(dist, group, member, spec["collective_timeout"])
                parked = False
                state = _fingerprint(engine, optimizer, step=completed_step, gradients=False)
                _record(run, spec, member, previous, "joined", completed_step=completed_step,
                        model_sha256=state["model_sha256"], optimizer_sha256=state["optimizer_sha256"],
                        optimizer_config_sha256=state["optimizer_config_sha256"])
            elif action in ("inspect", "finish"):
                if action == "inspect" and command["expected_step"] != completed_step:
                    raise ValueError("Inspection boundary step mismatch")
                progress("verify_final_native_state" if action == "finish" else "verify_committed_native_state")
                state = _fingerprint(engine, optimizer, step=completed_step, gradients=False)
                _record(run, spec, member, previous, "finished" if action == "finish" else "inspected", completed_step=completed_step,
                        checkpoint_calls=engine.checkpoint_call_count,
                        model_sha256=state["model_sha256"], optimizer_sha256=state["optimizer_sha256"],
                        optimizer_config_sha256=state["optimizer_config_sha256"])
                if action == "finish":
                    if group is not None:
                        dist.destroy_process_group()
                    return 0
            else:
                raise ValueError(f"Unknown worker action: {action}")
    except BaseException:
        _write(run / "errors" / f"{args.worker_id}-{args.session_id}.json",
               {"run_id": spec["run_id"], **_identity(member), "error": traceback.format_exc()})
        raise
    finally:
        if prefetch is not None:
            prefetch.close()


def _export_state(engine, optimizer, path, expected_step, metadata):
    from jade_lotus.resident_state import save_native_state, verify_native_state
    saved = save_native_state(engine.model, optimizer, path, step=expected_step, metadata=metadata)
    verified = verify_native_state(path, expected_step=expected_step, expected_metadata=metadata)
    if saved["manifest_sha256"] != verified["manifest_sha256"]:
        raise AssertionError("Native-state manifest changed during verification")
    return verified


def _load_state(engine, optimizer, path, expected_step, metadata):
    from jade_lotus.resident_state import load_native_state
    loaded = load_native_state(engine.model, optimizer, path, expected_step=expected_step, expected_metadata=metadata)
    if not loaded.get("verified_loaded_state") or not loaded.get("parameter_pointers_preserved"):
        raise AssertionError("Replacement complete native state was not verified")
    return loaded


def _restore_initial_state(engine, optimizer, spec, *, replacement=False, progress=lambda phase: None):
    """Restore an initial member before it can advertise readiness or join HCCL."""
    resume = spec.get("resume")
    if not resume or replacement:
        return {"completed_step": 0}
    progress("restore_native_checkpoint")
    loaded = _load_state(engine, optimizer, resume["checkpoint"], spec["start_step"],
                         resume["expected_metadata"])
    for field in ("manifest_sha256", "model_sha256", "optimizer_sha256"):
        if loaded[field] != resume[field]:
            raise ValueError(f"Initial restored checkpoint identity differs: {field}")
    return {"completed_step": spec["start_step"], "resume_receipt": loaded,
            "optimizer_config_sha256": _optimizer_config_hash(optimizer)}


def _validate_initial_states(members, spec):
    start = spec.get("start_step", 0)
    if any(member.get("completed_step") != start for member in members):
        raise ValueError("Initial worker native optimizer steps are not aligned")
    if not spec.get("resume"):
        return None
    resume = spec["resume"]
    configs = set()
    for member in members:
        receipt = member.get("resume_receipt", {})
        if (receipt.get("step") != start or not receipt.get("verified_loaded_state")
                or not receipt.get("parameter_pointers_preserved")
                or not receipt.get("payload_bytes_verified")):
            raise ValueError("Initial member lacks a complete verified native restore")
        for field in ("manifest_sha256", "model_sha256", "optimizer_sha256"):
            if receipt.get(field) != resume[field]:
                raise ValueError(f"Initial member restored the wrong checkpoint: {field}")
        digest = member.get("optimizer_config_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Initial optimizer configuration hash is missing")
        configs.add(digest)
    if len(configs) != 1:
        raise ValueError("Initial restored optimizer configurations disagree")
    return {"step": start, "members": len(members), "full_states_identical": True,
            "optimizer_config_sha256": next(iter(configs)), **resume,
            "verified_at": time.time()}


def supervisor(args):
    """Host-only owner of one child; never signal a PID taken from a remote record."""
    run = args.run_dir.resolve()
    spec = _wait(lambda: _read(run / "spec.json"), args.phase_timeout, "controller spec")
    if args.worker_id not in spec["workers"]:
        raise ValueError("Worker is outside this isolated probe")
    if getattr(args, "initial_standby", False) and args.worker_id in spec.get("initial_workers", spec["workers"]):
        raise ValueError("An initial active worker cannot be launched as independent standby")
    session, replacement, process = uuid.uuid4().hex, getattr(args, "initial_standby", False), None
    while True:
        logfile = run / "logs" / f"{args.worker_id}-{session}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        cli = [sys.executable, str(Path(__file__).resolve()), "--child", "--run-dir", str(run),
               "--worker-id", args.worker_id, "--session-id", session,
               "--device-index", str(args.device_index), "--allocated-devices", str(args.allocated_devices)]
        if replacement:
            cli.append("--replacement")
        with logfile.open("x") as output:
            process = subprocess.Popen(cli, stdout=output, stderr=subprocess.STDOUT)
            startup_deadline = time.time() + spec["phase_timeout"]
            own = {"schema": SCHEMA, "run_id": spec["run_id"], "worker_id": args.worker_id,
                   "session_id": session, "pid": process.pid, "supervisor_pid": os.getpid(),
                   "heartbeat_counter": 0}
            _write(run / "supervisors" / f"{args.worker_id}.json", own)
            killed, directive, timed_out = False, None, False
            next_heartbeat, forwarded_hold = time.monotonic() + 1, None
            while process.poll() is None:
                if time.monotonic() >= next_heartbeat:
                    own["heartbeat_counter"] += 1
                    _write(run / "supervisors" / f"{args.worker_id}.json", own)
                    next_heartbeat = time.monotonic() + 1
                directive = _read(run / "supervisor_commands" / f"{args.worker_id}.json")
                if (directive and directive.get("action") == "hold_own_child"
                        and directive.get("run_id") == spec["run_id"] and _same_identity(directive, own)
                        and directive.get("token") != forwarded_hold):
                    _write(run / "holds" / f"{args.worker_id}-{session}.json", directive)
                    forwarded_hold = directive["token"]
                if _kill_matches(directive, run_id=spec["run_id"], worker_id=args.worker_id,
                                 session_id=session, child_pid=process.pid):
                    process.kill()  # Only the exact Popen child owned by this supervisor.
                    killed = True
                    break
                status = _read(run / "status" / f"{args.worker_id}-{session}.json")
                deadline = status["deadline"] if status and _same_identity(status, own) else startup_deadline
                if time.time() > deadline:
                    process.kill()
                    timed_out = True
                    break
                time.sleep(0.1)
            code = process.wait(timeout=30)
        exit_record = {**own, "exit_code": code, "killed_by_controller": killed,
                       "kill_directive": directive if killed else None, "phase_timed_out": timed_out, "time": time.time()}
        _write(run / "exits" / f"{args.worker_id}-{session}.json", exit_record)
        if not killed:
            return code
        if directive.get("terminal_cancel"):
            return 1
        def after_kill():
            value = _read(run / "supervisor_commands" / f"{args.worker_id}.json")
            if (value and value.get("run_id") == spec["run_id"] and value.get("session_id") == session
                    and (value.get("action") in ("replace", "finish_supervisor") or value.get("terminal_cancel"))):
                return value
            return None
        directive = _wait(after_kill, spec["phase_timeout"], "replacement or finish supervisor")
        if directive["action"] == "finish_supervisor" or directive.get("terminal_cancel"):
            return 0
        session, replacement = uuid.uuid4().hex, True


def _external_reduced_or_recover(run, spec, members, group, mapping, token, step, watch, wait_reports, *, already_lost):
    """One unknown communication loss; never wait for the missing submission."""
    pending = None
    deadline = time.monotonic() + spec["phase_timeout"]
    while time.monotonic() < deadline:
        current_ids = {m["worker_id"] for m in members}
        notices = [n for n in watch.poll() if n.newly_fenced and n.session.worker_id in current_ids]
        if notices:
            if pending is not None or len(notices) != 1 or already_lost:
                raise RuntimeError("Only one communication-stage loss is supported by this component")
            pending = notices[0]
            if pending.action != "shrink_pre_apply" or pending.phase != "reducing":
                raise RuntimeError("recovery_required: loss outside pre-apply communication")
        if pending is not None:
            victim = next((m for m in members if m["worker_id"] == pending.session.worker_id
                           and m["session_id"] == pending.session.session_id and m["pid"] == pending.session.child_pid), None)
            if victim is None:
                raise ValueError("Loss notice is outside this exact frozen membership")
            victim_ready = _read(_report_path(run, victim, token, "local_ready"))
            lost_complete_backward_known = bool(victim_ready and victim_ready.get("complete_local_backward")
                and victim_ready.get("base_step") == step and victim_ready.get("generation") == group["generation"])
            survivors = [m for m in members if not _same_identity(m, victim)]
            ready = _reports(run, spec, survivors, token, "local_ready")
            if ready is None:
                time.sleep(_control_poll_seconds(spec, legacy=0.05))
                continue
            if any(not d.get("complete_local_backward") or d.get("base_step") != step
                   or d.get("generation") != group["generation"] for d in ready.values()):
                raise ValueError("Healthy local-gradient readiness fence mismatch")
            _write(run / "abort" / f"{token}.json", {"action": "abort_collective", "run_id": spec["run_id"],
                "token": token, "base_step": step, "generation": group["generation"], "members": survivors,
                "confirmed_peer_exit": pending.confirmed, "peer_loss_fenced": True,
                "loss_evidence": pending.as_record(), "victim": _identity(victim)})
            if spec["replacement"]:
                _write(run / "supervisor_commands" / f"{victim['worker_id']}.json",
                       {"run_id": spec["run_id"], "session_id": victim["session_id"], "action": "replace"})
            recovered = wait_reports(survivors, token, "recovered")
            if not all(d.get("state_storage_pointers_unchanged") for d in recovered.values()):
                raise AssertionError("Healthy native state/local-gradient storage was not preserved")
            if spec.get("audit_before_collective") and not all(d.get("full_native_state_and_gradient_unchanged") for d in recovered.values()):
                raise AssertionError("Requested complete byte audit did not pass")
            repaired, replay = repair_assignments(mapping, victim["worker_id"], survivors)
            new_group = _group(survivors, group["generation"] + 1,
                               port=recovered[survivors[0]["worker_id"]]["next_port"])
            detector_token = uuid.uuid4().hex
            watch.reconfigure_pre_apply(token, detector_token, generation=new_group["generation"],
                survivors=[watch.keys[m["worker_id"]] for m in survivors])
            watch.advance_phase(detector_token, "reducing")
            _write(run / "recovery" / f"{token}.json", {"run_id": spec["run_id"], "token": token,
                "detector_token": detector_token, "base_step": step, "group": new_group,
                "assignments": repaired, "replay": replay})
            def repaired_reduced():
                later = [n for n in watch.poll() if n.newly_fenced
                         and n.session.worker_id in {m["worker_id"] for m in survivors}]
                if later:
                    raise RuntimeError("Second loss during communication repair is unsupported")
                records = _reports(run, spec, survivors, token, "reduced")
                if records and all(d.get("generation") == new_group["generation"] and d.get("base_step") == step for d in records.values()):
                    return records
                return None
            _wait(repaired_reduced, spec["phase_timeout"], "repaired new-group gradients",
                  interval=_control_poll_seconds(spec))
            fault = {"kind": "unannounced_pre_apply_loss", "base_step": step, "victim": _identity(victim),
                     "controller_had_fault_victim": False, "loss_notice": pending.as_record(),
                     "lost_complete_local_backward_known": lost_complete_backward_known,
                     "lost_assignment_durably_accepted": False,
                     "healthy_processes_preserved": len(survivors), "only_lost_bundle_replayed": replay,
                     "healthy_state_and_gradient_storage_preserved": True,
                     "healthy_full_native_state_and_gradients_verified": all(d.get("full_native_state_and_gradient_unchanged") for d in recovered.values())}
            return survivors, new_group, repaired, detector_token, victim, fault
        records = _reports(run, spec, members, token, "reduced")
        if records is not None:
            return members, group, mapping, token, None, None
        time.sleep(_control_poll_seconds(spec, legacy=0.05))
    raise TimeoutError("No reduced gradients or fenced communication loss before deadline")



def _publish_commit(run, summary, commit, *, window=0):
    summary["committed_steps"] = commit["step"]
    summary["executed_steps"] = commit["step"] - summary.get("start_step", 0)
    if window:
        if type(window) is not int or window < 1:
            raise ValueError("Summary window must be a positive integer")
        path = Path(run) / "commits.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(commit, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        summary["committed_steps"] = commit["step"]
        summary["commit_journal"] = str(path)
        summary["commits"] = summary["commits"][-window:]
    _write(Path(run) / "summary.json", summary)


def _boundary_metadata(spec, boundary):
    return {"run_id": spec["run_id"], "boundary_step": boundary,
            "config_sha256": spec["config_sha256"], "global_batch": 256,
            "seed": spec["seed"], "rng_scheme": spec.get("rng_scheme", "torch-atomic4-preprocess-cpu-noise-inverse-beta-v1"),
            "activation_recomputation": False, "precision_contract": spec["precision_contract"]}


def _resume_contract(args, config):
    """Bind a resume to the original training identity without reading device state."""
    checkpoint = getattr(args, "resume_checkpoint", None)
    source_path = getattr(args, "resume_source_spec", None)
    if bool(checkpoint) != bool(source_path):
        raise ValueError("Resume requires both checkpoint and source specification")
    if checkpoint is None:
        return 0, None
    from jade_lotus.resident_state import _read_manifest
    source = _read(source_path)
    if not source:
        raise ValueError("Resume source specification is missing")
    original_bytes = Path(source["config"]).read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != source["config_sha256"]:
        raise ValueError("Resume source training config identity changed")
    original_config = json.loads(original_bytes)
    # Runtime scheduling/observability may change, but the numerical recipe may not.
    mutable = {"run_dir", "run_id", "name", "openpi_root"}
    for key in set(config) | set(original_config):
        if key.startswith("resident_") or key in mutable:
            continue
        if config.get(key) != original_config.get(key):
            raise ValueError(f"Resume numerical training configuration changed: {key}")
    directory, manifest = _read_manifest(checkpoint, None, None)
    start = manifest["step"]
    if not 0 < start < args.steps or source.get("steps") != args.steps or source.get("cards") != args.cards:
        raise ValueError("Resume must preserve source total target/cards and start before its target")
    metadata = _boundary_metadata(source, start)
    _read_manifest(directory, start, metadata)
    if (directory / "state.bin").stat().st_size != manifest["payload_bytes"]:
        raise ValueError("Resume checkpoint payload size differs")
    return start, {"checkpoint": str(directory), "source_spec": str(Path(source_path).resolve()),
                   "source_run_id": source["run_id"], "expected_metadata": metadata,
                   **{field: manifest[field] for field in
                      ("manifest_sha256", "model_sha256", "optimizer_sha256", "payload_bytes")}}


def _periodic_checkpoint(run, spec, members, boundary, summary, wait_reports):
    started = time.monotonic()
    path = Path(run) / "states" / f"checkpoint-{boundary}"
    token = _send(run, spec, [members[0]], "export", path=str(path), expected_step=boundary,
                  state_metadata=_boundary_metadata(spec, boundary))
    record = wait_reports([members[0]], token, "exported")[members[0]["worker_id"]]
    if record["completed_step"] != boundary or record["receipt"]["step"] != boundary:
        raise ValueError("Periodic checkpoint is not the committed boundary")
    summary.setdefault("checkpoints", []).append({"step": boundary, "path": str(path),
        "receipt": record["receipt"], "pause_seconds": time.monotonic() - started})
    _write(Path(run) / "summary.json", summary)


def _inspect_boundary(run, spec, members, boundary, summary, wait_reports):
    started = time.monotonic()
    token = _send(run, spec, members, "inspect", expected_step=boundary)
    records = wait_reports(members, token, "inspected")
    if (any(r["completed_step"] != boundary or r["checkpoint_calls"] != 0 for r in records.values())
            or len({(r["model_sha256"], r["optimizer_sha256"], r["optimizer_config_sha256"])
                    for r in records.values()}) != 1):
        raise AssertionError("Committed boundary complete native states disagree")
    summary.setdefault("state_inspections", []).append({"step": boundary, "token": token,
        "members": len(members), "full_states_identical": True,
        "pause_seconds": time.monotonic() - started})
    _write(Path(run) / "summary.json", summary)

def controller(args):
    """Pure CPU orchestration; this path never imports torch or torch_npu."""
    from jade_lotus.topology_cohort_runtime import (CandidateUnavailable, CohortBoundaryRuntime,
        CohortRecoveryRequired, validate_topology_config)
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "spec.json").exists() or any((run / name).exists() for name in ("run_config.json", "active.json", "COMPLETE.json")):
        raise ValueError("Use a fresh, isolated probe directory")
    with (run / "controller.owner").open("x") as owner:
        owner.write(f"{os.getpid()}\n")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    if config.get("batch_size") != 256 or config.get("disable_activation_checkpointing") is not True:
        raise ValueError("Probe requires global256 with recomputation disabled")
    start_step, resume = _resume_contract(args, config)
    data_workers = config.get("resident_data_workers", 0) if args.data_workers is None else args.data_workers
    if type(data_workers) is not int or not 0 <= data_workers <= 8:
        raise ValueError("resident_data_workers/--data-workers must be an integer in 0..8")
    if config.get("resident_adapter") and data_workers:
        raise ValueError("Custom resident adapters own data loading; set resident_data_workers=0")
    if config.get("resident_adapter") and any(not isinstance(config.get(key), str) or not config[key].strip()
            for key in ("resident_rng_scheme", "resident_precision_contract")):
        raise ValueError("Custom adapters must declare resident_rng_scheme and resident_precision_contract")
    for key in ("resident_checkpoint_every", "resident_verify_every", "resident_summary_window"):
        value = config.get(key, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    topology_slots = validate_topology_config(config, args.cards, start_step=start_step, steps=args.steps)
    workers = [f"w{rank + 1:06d}" for rank in range(topology_slots)]
    initial_workers = workers[:args.cards]
    spec = {"schema": SCHEMA, "run_id": uuid.uuid4().hex, "config": str(args.config.resolve()),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(), "seed": config.get("seed", 42),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "workers": workers, "initial_workers": initial_workers, "cards": args.cards,
            "topology_slots": topology_slots, "run_dir": str(run),
            "steps": args.steps, "fault_step": args.fault_step,
            "start_step": start_step, "resume": resume,
            "phase_timeout": args.phase_timeout, "collective_timeout": args.collective_timeout,
            "bucket_bytes": args.bucket_mib * 1024**2, "replacement": not args.no_replacement,
            "data_workers": data_workers,
            "resident_adapter": config.get("resident_adapter"),
            "rng_scheme": config.get("resident_rng_scheme", "torch-atomic4-preprocess-cpu-noise-inverse-beta-v1"),
            "join_transport": args.join_transport,
            "external_loss": args.external_loss, "audit_before_collective": args.audit_before_collective,
            "heartbeat_timeout": args.heartbeat_timeout,
            "communication_diagnostics": args.communication_diagnostics,
            "communication_diagnostics_steps": args.communication_diagnostics_steps,
            "control_poll_seconds": args.control_poll_seconds,
            "control_poll_scope": "hot gates only, capped by each gate's existing interval; heartbeat and supervisor unchanged",
            "external_recovery_scope": "one frozen-session loss before optimizer authorization; wait healthy local gradients, replay whole lost assignment",
            "second_loss_during_repair_supported": False,
            "experimental_component": True, "production_recovery_verified": False,
            "precision_contract": config.get("resident_precision_contract", "native BF16/FP32 parameters and matching Adam moments; no FP32 master"),
            "fault_injection_mode": "independent_external_injector" if args.external_loss else "hold_final_rank_after_complete_backward_then_kill",
            "kill_requires_all_healthy_collectives_submitted": True, "started_at": time.time()}
    _write(run / "spec.json", spec)
    summary = {"schema": SCHEMA, "run_id": spec["run_id"], "status": "running", "commits": [],
               "start_step": start_step, "committed_steps": start_step, "executed_steps": 0,
               "stage": "initial_model_loads", "resume": resume,
               "single_card_loss_during_optimizer_apply_verified": False,
               "durable_gradient_acceptance_verified": False}
    _write(run / "summary.json", summary)
    watch, topology = None, None
    members, nonactive_errors = [], set()
    def checked_wait(readiness, label, *, hot_control=False, fatal_members=None, timeout_seconds=None):
        def check():
            fatal_ids = set(initial_workers) if not members else {m["worker_id"] for m in members}
            fatal_ids.update(m["worker_id"] for m in (fatal_members or ()))
            if watch is not None:
                notices = [n for n in watch.poll() if n.newly_fenced and n.session.worker_id in fatal_ids]
                if notices:
                    raise RuntimeError("recovery_required: loss outside the supported communication collection phase")
            errors = list((run / "errors").glob("*.json"))
            for path in errors:
                record = _read(path)
                if topology_slots == args.cards or not record or record.get("worker_id") in fatal_ids:
                    raise RuntimeError(f"Worker failed: {path}")
                if str(path) not in nonactive_errors:
                    nonactive_errors.add(str(path))
                    summary.setdefault("topology_candidate_failures", []).append({"evidence": str(path),
                        "worker_id": record.get("worker_id"), "active_group_continued": True})
            for path in (run / "exits").glob("*.json"):
                record = _read(path)
                if watch is not None and record and any(key.matches_supervisor_record(record) for key in watch.detector.fenced_sessions):
                    continue  # An exact persisted loss is already being repaired.
                if record and record.get("run_id") == spec["run_id"] and record.get("exit_code") != 0 and not record.get("killed_by_controller"):
                    if topology_slots > args.cards and record.get("worker_id") not in fatal_ids:
                        continue
                    raise RuntimeError(f"Unexpected worker exit: {path}")
            return readiness()
        return _wait(check, args.phase_timeout if timeout_seconds is None else min(args.phase_timeout, timeout_seconds), label,
                     interval=_control_poll_seconds(spec) if hot_control else 0.1)
    def wait_reports(members, token, phase):
        return checked_wait(lambda: _reports(run, spec, members, token, phase), phase, hot_control=True,
                            fatal_members=members)
    def pool_ready(worker):
        record = _read(run / "ready" / f"{worker}.json")
        supervisor = _read(run / "supervisors" / f"{worker}.json")
        if (not record or record.get("run_id") != spec["run_id"] or record.get("worker_id") != worker
                or not supervisor or supervisor.get("run_id") != spec["run_id"]
                or not _same_identity(supervisor, record)):
            raise CandidateUnavailable("Candidate has not completed its independent initial model load")
        if (_read(run / "exits" / f"{worker}-{record['session_id']}.json") is not None
                or _read(run / "errors" / f"{worker}-{record['session_id']}.json") is not None):
            raise CandidateUnavailable("Candidate session has exited or reported an error")
        if watch is not None and any(k.worker_id == worker and k.session_id == record["session_id"]
                                     for k in watch.detector.fenced_sessions):
            raise CandidateUnavailable("Candidate session has already been irreversibly fenced")
        return record
    def wait_candidate_reports(candidates, token, phase, *, timeout_seconds=None):
        def ready_reports():
            for candidate in candidates:
                if not _same_identity(pool_ready(candidate["worker_id"]), candidate):
                    raise CandidateUnavailable("Candidate session changed during state alignment")
            return _reports(run, spec, candidates, token, phase)
        return checked_wait(ready_reports, f"candidate {phase}", hot_control=True, timeout_seconds=timeout_seconds)
    victim, replacement_member, joined = None, None, False
    try:
        def initial_ready():
            records = [_read(run / "ready" / f"{worker}.json") for worker in initial_workers]
            return records if all(record and record.get("run_id") == spec["run_id"]
                                  and record.get("worker_id") == worker and not record["replacement"]
                                  for worker, record in zip(initial_workers, records)) else None
        members = checked_wait(initial_ready, "all initial model loads")
        restored = _validate_initial_states(members, spec)
        if restored:
            summary["initial_restore"] = restored
        summary["stage"] = "initializing_group"
        _write(run / "summary.json", summary)
        original_members = members[:]
        victim = None if args.external_loss else members[-1]
        group = _group(members, 0)
        token = _send(run, spec, members, "init", group=group)
        wait_reports(members, token, "initialized")
        if args.external_loss:
            watch = ControllerLiveness(run, spec)
            watch.register_ready(members, receipt=f"initialized-group-{group['generation']}")
        if topology_slots > args.cards or config.get("resident_topology_switches"):
            topology = CohortBoundaryRuntime(spec, config.get("resident_topology_switches", []),
                send=lambda members, action, **fields: _send(run, spec, members, action, **fields),
                wait_active=wait_reports, wait_candidate=wait_candidate_reports, ready=pool_ready,
                make_group=_group, persist=lambda state: _write(run / "topology" / "state.json", state),
                metadata=lambda boundary: _boundary_metadata(spec, boundary))
            topology.start(members, group)
        summary.update(stage="training", training_started_at=time.time())
        _write(run / "summary.json", summary)
        previous_commit = time.monotonic()
        for step in range(start_step, args.steps):
            commit_started, commit_started_at = time.monotonic(), time.time()
            mapping = assignments(members)
            fault = not args.external_loss and step + 1 == args.fault_step
            token = uuid.uuid4().hex
            detector_token = token
            if watch is not None:
                if any(n.newly_fenced and n.session.worker_id in {m["worker_id"] for m in members}
                       for n in watch.poll()):
                    raise RuntimeError("Unsupported loss at the idle/update boundary")
                watch.begin_step(token, base_step=step, generation=group["generation"],
                                          sessions=[watch.keys[m["worker_id"]] for m in members])
                watch.advance_phase(token, "reducing")
            _send(run, spec, members, "step", token=token, group=group, base_step=step, expected_step=step,
                          assignments=mapping, fault_victim=victim if fault else None)
            if fault:
                submitted = wait_reports(members, token, "submitted")
                if any(record["generation"] != group["generation"] or record["base_step"] != step for record in submitted.values()):
                    raise ValueError("Collective submission fence mismatch")
                victim_submission = submitted[victim["worker_id"]]
                if (victim_submission.get("held_before_collective") is not True
                        or victim_submission["collectives"] != 0
                        or victim_submission["local"]["units"] != mapping[victim["worker_id"]]):
                    raise ValueError("Victim must finish its complete bundle and hold before HCCL")
                if any(record.get("held_before_collective") or record["collectives"] <= 0
                       for worker, record in submitted.items() if worker != victim["worker_id"]):
                    raise ValueError("Every healthy rank must submit its complete communication batch before kill")
                kill = {"schema": SCHEMA, "run_id": spec["run_id"], "action": "kill_own_child",
                        "token": token, **_identity(victim)}
                _write(run / "supervisor_commands" / f"{victim['worker_id']}.json", kill)
                exit_record = checked_wait(lambda: _read(run / "exits" / f"{victim['worker_id']}-{victim['session_id']}.json"), "victim exit")
                if not exit_record.get("killed_by_controller") or exit_record.get("kill_directive") != kill:
                    raise ValueError("Supervisor did not confirm the exact own-child kill")
                members = [member for member in members if not _same_identity(member, victim)]
                _write(run / "abort" / f"{token}.json",
                    {"action": "abort_collective", "run_id": spec["run_id"], "token": token,
                     "base_step": step, "generation": group["generation"], "members": members,
                     "confirmed_peer_exit": True, "victim": _identity(victim)})
                if not args.no_replacement:
                    _write(run / "supervisor_commands" / f"{victim['worker_id']}.json",
                           {"run_id": spec["run_id"], "session_id": victim["session_id"], "action": "replace"})
                recovered = wait_reports(members, token, "recovered")
                if not all(record.get("full_native_state_and_gradient_unchanged") for record in recovered.values()):
                    raise AssertionError("Healthy state preservation was not verified")
                mapping, replay = repair_assignments(mapping, victim["worker_id"], members)
                group = _group(members, group["generation"] + 1, port=recovered[members[0]["worker_id"]]["next_port"])
                _write(run / "recovery" / f"{token}.json", {"run_id": spec["run_id"], "token": token,
                       "base_step": step, "group": group, "assignments": mapping, "replay": replay})
                summary["fault"] = {"base_step": step, "victim": _identity(victim),
                    "healthy_processes_preserved": len(members), "healthy_full_native_state_and_gradients_verified": True,
                    "only_lost_bundle_replayed": replay,
                    "fault_injection_stage": "victim_full_backward_finished_held_before_collective",
                    "pending_collective_guaranteed_by_missing_rank": True}
            if watch is not None:
                members, group, mapping, detector_token, lost, fault_record = _external_reduced_or_recover(
                    run, spec, members, group, mapping, token, step, watch, wait_reports,
                    already_lost="fault" in summary)
                if lost is not None:
                    victim = lost
                    summary["fault"] = fault_record
                    _write(run / "summary.json", summary)
            reduced = wait_reports(members, token, "reduced")
            if any(record["base_step"] != step or record["generation"] != group["generation"] for record in reduced.values()):
                raise ValueError("Reduced-gradient generation/base-step mismatch")
            validate_coverage(mapping)
            if watch is not None:
                if watch.poll():
                    raise RuntimeError("Loss raced the prepared optimizer boundary; recovery_required")
                watch.advance_phase(detector_token, "prepared")
                watch.advance_phase(detector_token, "applying")
                _write(run / "controller_phase.json", {"run_id": spec["run_id"], "token": token,
                       "detector_token": detector_token, "base_step": step, "generation": group["generation"],
                       "phase": "applying", "loss_requires_full_state_recovery": True})
            _write(run / "apply" / f"{token}.json", {"run_id": spec["run_id"], "token": token,
                    "base_step": step, "generation": group["generation"], "members": members})
            updated = wait_reports(members, token, "updated")
            if any(record["completed_step"] != step + 1 or record["checkpoint_calls"] != 0 for record in updated.values()):
                raise AssertionError("Native optimizer step or recomputation ledger mismatch")
            if watch is not None:
                watch.advance_phase(detector_token, "committing")
                watch.finish_step(detector_token, committed_state_receipt=f"all-updated/{token}")
            committed_at = time.monotonic()
            summary["commits"].append({"step": step + 1, "generation": group["generation"],
                    "members": [_identity(member) for member in members], "assignments": mapping,
                    "atomic_units": 64, "samples": 256, "token": token, "time": time.time(),
                    "commit_seconds": committed_at - commit_started,
                    "commit_interval_seconds": committed_at - previous_commit,
                    "command_started_at": commit_started_at})
            previous_commit = committed_at
            _publish_commit(run, summary, summary["commits"][-1], window=config.get("resident_summary_window", 0))
            if topology is not None:
                topology.on_commit(step + 1, members)
            checkpoint_every = config.get("resident_checkpoint_every", 0)
            verify_every = config.get("resident_verify_every", 0)
            stability_boundary = bool(resume) and step + 1 == start_step + 2000
            if (checkpoint_every and (step + 1) % checkpoint_every == 0) or stability_boundary:
                _periodic_checkpoint(run, spec, members, step + 1, summary, wait_reports)
            if (verify_every and (step + 1) % verify_every == 0) or stability_boundary:
                _inspect_boundary(run, spec, members, step + 1, summary, wait_reports)
            if topology is not None and step + 1 < args.steps:
                try:
                    request = _read(run / "topology" / "switch_request.json")
                except (ValueError, OSError) as error:
                    request = {"request_id": "invalid-request-file", "reason": str(error)}
                previous_generation = group["generation"]
                members, group = topology.boundary(step + 1, members, group, request)
                if watch is not None and group["generation"] != previous_generation:
                    try:
                        watch = watch.rebuild_active(members, generation=group["generation"], committed_step=step + 1,
                            receipt=f"verified-cohort-boundary/{group['generation']}")
                    except Exception as error:
                        topology.recovery_required = True
                        _write(run / "topology" / "state.json", topology.snapshot())
                        raise CohortRecoveryRequired("New active group liveness could not be verified after boundary reconfiguration") from error
                summary["topology_switches"] = list(topology.requests.values())
                summary["topology_state"] = str(run / "topology" / "state.json")
                summary["topology_automatic_failover_verified"] = False
                _write(run / "summary.json", summary)
            # Independent loading does not block healthy updates. The join
            # boundary starts only after at least two further healthy commits.
            if (victim is not None and "fault" in summary and not args.no_replacement and not joined
                    and step + 1 >= summary["fault"]["base_step"] + 3 and step + 1 < args.steps):
                ready = _read(run / "ready" / f"{victim['worker_id']}.json")
                if (ready and ready.get("run_id") == spec["run_id"] and ready.get("replacement") is True
                        and ready.get("session_id") != victim["session_id"]):
                    replacement_member = ready
                    boundary = step + 1
                    join_started = time.monotonic()
                    state_path = run / "states" / f"join-{boundary}"
                    state_metadata = {"run_id": spec["run_id"], "boundary_step": boundary,
                                      "config_sha256": spec["config_sha256"], "global_batch": 256,
                                      "seed": spec["seed"],
                                      "rng_scheme": spec.get("rng_scheme", "torch-atomic4-preprocess-cpu-noise-inverse-beta-v1"),
                                      "precision_contract": spec["precision_contract"],
                                      "activation_recomputation": False}
                    if args.join_transport == "device":
                        describe = _send(run, spec, [members[0]], "describe_native_state",
                            expected_step=boundary, state_metadata=state_metadata)
                        described = wait_reports([members[0]], describe, "described")[members[0]["worker_id"]]
                        roles = {item["worker_id"]: "source" if index == 0 else "healthy"
                                 for index, item in enumerate(members)}
                        roles[ready["worker_id"]] = "replacement"
                        members = members + [ready]
                        group = _group(members, group["generation"] + 1, port=described["next_port"])
                        join_token = _send(run, spec, members, "device_join", group=group,
                            roles=roles, blueprint=described["blueprint"], expected_step=boundary,
                            state_metadata=state_metadata)
                    else:
                        export_token = _send(run, spec, [members[0]], "export", path=str(state_path),
                                             expected_step=boundary, state_metadata=state_metadata)
                        exported = wait_reports([members[0]], export_token, "exported")[members[0]["worker_id"]]
                        load_token = _send(run, spec, [ready], "load", path=str(state_path),
                                           expected_step=boundary, state_metadata=state_metadata)
                        loaded = wait_reports([ready], load_token, "loaded")[ready["worker_id"]]
                        for field in ("manifest_sha256", "model_sha256", "optimizer_sha256", "step"):
                            if exported["receipt"][field] != loaded["receipt"][field]:
                                raise AssertionError("Replacement did not load the exported exact native state")
                        members = members + [ready]
                        group = _group(members, group["generation"] + 1, port=exported["next_port"])
                        join_token = _send(run, spec, members, "join", group=group, expected_step=boundary)
                    states = wait_reports(members, join_token, "joined")
                    if args.join_transport == "device" and any(
                            not record.get("transport_receipt", {}).get("full_native_state_hash_verified")
                            for record in states.values()):
                        raise AssertionError("Device join report lacks complete native state verification")
                    digests = {(record["model_sha256"], record["optimizer_sha256"], record["optimizer_config_sha256"]) for record in states.values()}
                    if len(digests) != 1:
                        raise AssertionError("Joining ranks do not hold identical complete native model/AdamW states")
                    if watch is not None:
                        watch.register_ready([ready], receipt=f"complete-native-join/{join_token}")
                    joined = True
                    summary["replacement"] = {"joined_after_step": boundary,
                        "join_transport": args.join_transport,
                        "healthy_additional_commits_before_join": boundary - summary["fault"]["base_step"] - 1,
                        "full_native_state_alignment_verified": True, "session": _identity(ready),
                        "boundary_state_export_load_and_join_pause_seconds": time.monotonic() - join_started,
                        "initial_model_load_started_at": ready.get("started_at"),
                        "initial_model_loaded_at": ready.get("loaded_at"),
                        "healthy_commits_during_initial_model_load": [commit["step"] for commit in summary["commits"]
                            if ready.get("started_at", float("inf")) <= commit["time"] <= ready.get("loaded_at", 0)]}
                    _write(run / "summary.json", summary)
        if config.get("resident_final_state") == "checkpoint":
            state_metadata = {"run_id": spec["run_id"], "boundary_step": args.steps,
                              "config_sha256": spec["config_sha256"], "global_batch": 256,
                              "seed": spec["seed"], "rng_scheme": spec.get("rng_scheme", "torch-atomic4-preprocess-cpu-noise-inverse-beta-v1"),
                              "activation_recomputation": False, "precision_contract": spec["precision_contract"]}
            path = run / "states" / f"final-{args.steps}"
            token = _send(run, spec, [members[0]], "export", path=str(path),
                          expected_step=args.steps, state_metadata=state_metadata)
            record = wait_reports([members[0]], token, "exported")[members[0]["worker_id"]]
            summary["final_checkpoint"] = {"path": str(path), "receipt": record["receipt"]}
            _write(run / "summary.json", summary)
        finish_members, ready_standby = list(members), []
        if topology is not None:
            active_ids = {m["worker_id"] for m in members}
            for worker in workers:
                if worker in active_ids:
                    continue
                own = _read(run / "supervisors" / f"{worker}.json")
                if (not own or own.get("run_id") != spec["run_id"]
                        or _read(run / "exits" / f"{worker}-{own['session_id']}.json") is not None):
                    continue
                # A still-loading registered child receives finish when it
                # reaches its command loop; unallocated slots are never waited.
                finish_members.append(own)
                ready = _read(run / "ready" / f"{worker}.json")
                if ready and ready.get("run_id") == spec["run_id"] and _same_identity(ready, own):
                    ready_standby.append(ready)
        token = _send(run, spec, finish_members, "finish")
        final = wait_reports(members, token, "finished")
        if topology is not None:
            def standby_finished():
                result = {}
                for standby in ready_standby:
                    rows = _reports(run, spec, [standby], token, "finished")
                    if rows:
                        result.update(rows)
                    elif _read(run / "exits" / f"{standby['worker_id']}-{standby['session_id']}.json") is None:
                        return None
                return result
            standby_final = checked_wait(standby_finished, "ready parked/standby finish", hot_control=True)
            summary["topology_resident_finish"] = {"sent": [_identity(m) for m in finish_members],
                "standby_reports": standby_final, "waited_for_unallocated_slots": False}
        digests = {(record["model_sha256"], record["optimizer_sha256"], record["optimizer_config_sha256"]) for record in final.values()}
        if len(digests) != 1 or any(record["completed_step"] != args.steps for record in final.values()):
            raise AssertionError("Final complete native states or optimizer steps disagree")
        if "final_checkpoint" in summary:
            receipt = summary["final_checkpoint"]["receipt"]
            if any(receipt[field] != final[members[0]["worker_id"]][field]
                   for field in ("model_sha256", "optimizer_sha256")):
                raise AssertionError("Final checkpoint differs from the actual final resident state")
        if not args.no_replacement and not joined and (not args.external_loss or "fault" in summary):
            raise RuntimeError("Shrink training completed, but replacement never loaded in time to join; acceptance incomplete")
        summary.update(status="passed", completed_at=time.time(), final_states_identical=True,
                       final_members=len(members), healthy_rollback_count=0,
                       initial_members=[_identity(member) for member in original_members])
        _write(run / "summary.json", summary)
        return 0
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc(), failed_at=time.time())
        if ((watch is not None and watch.detector.phase in ("applying", "committing", "recovery_required"))
                or isinstance(sys.exc_info()[1], CohortRecoveryRequired)
                or (topology is not None and topology.recovery_required)):
            summary["recovery_required"] = True
        _write(run / "summary.json", summary)
        raise
    finally:
        if summary["status"] == "failed":
            # Cleanup is confined to supervisors registered in this private
            # run. Each still signals only the Popen child it actually owns.
            for worker in workers:
                own = _read(run / "supervisors" / f"{worker}.json")
                if own and own.get("run_id") == spec["run_id"]:
                    _write(run / "supervisor_commands" / f"{worker}.json",
                           {"schema": SCHEMA, "run_id": spec["run_id"], **_identity(own),
                            "action": "kill_own_child", "token": uuid.uuid4().hex, "terminal_cancel": True})
        if victim is not None and args.no_replacement and summary["status"] == "passed":
            _write(run / "supervisor_commands" / f"{victim['worker_id']}.json",
                   {"run_id": spec["run_id"], "session_id": victim["session_id"], "action": "finish_supervisor"})


def injector(args):
    """Independent CPU actor; sends only exact own-child hold/kill to supervisor."""
    run = args.run_dir.resolve()
    spec = _wait(lambda: _read(run / "spec.json"), args.phase_timeout, "controller specification")
    if not spec.get("external_loss") or args.victim_worker not in spec["workers"]:
        raise ValueError("Injector requires an external-loss run and one of its exact workers")
    ready = _wait(lambda: _read(run / "ready" / f"{args.victim_worker}.json"), spec["phase_timeout"], "victim model ready")
    def target_command():
        value = _read(run / "commands" / f"{args.victim_worker}.json")
        if (value and value.get("run_id") == spec["run_id"] and value.get("action") == "step"
                and value.get("base_step", -1) + 1 >= args.inject_after_step
                and _same_identity(value.get("recipient"), ready)):
            return value
        return None
    command = _wait(target_command, spec["phase_timeout"], "independently selected training step")
    if command.get("fault_victim") is not None:
        raise ValueError("External injection must not use the planned-fault command")
    member, token = command["recipient"], command["token"]
    if args.inject_stage == "backward":
        def backward_active():
            status = _read(run / "status" / f"{member['worker_id']}-{member['session_id']}.json")
            return status if (status and _same_identity(status, member) and status.get("run_id") == spec["run_id"]
                and status.get("token") == token and status.get("phase") == "backward") else None
        status = _wait(backward_active, spec["phase_timeout"], "victim's exact-step backward phase")
        kill = {"schema": SCHEMA, "run_id": spec["run_id"], **_identity(member), "action": "kill_own_child",
                "token": token, "injected_by_independent_actor": True}
        _write(run / "supervisor_commands" / f"{args.victim_worker}.json", kill)
        _write(run / "injection.json", {"run_id": spec["run_id"], "token": token, "victim": _identity(member),
            "base_step": command["base_step"], "controller_command_had_no_fault_victim": True,
            "requested_injection_stage": "backward", "observed_status_before_kill": status,
            "kill": kill, "time": time.time()})
        return 0
    hold = {"schema": SCHEMA, "run_id": spec["run_id"], **_identity(member), "action": "hold_own_child",
            "token": token, "base_step": command["base_step"], "generation": command["group"]["generation"]}
    _write(run / "supervisor_commands" / f"{args.victim_worker}.json", hold)
    held = _wait(lambda: _read(_report_path(run, member, token, "external_held")), spec["phase_timeout"], "victim hold acknowledgement")
    if not held.get("held_before_collective") or held.get("collectives") != 0:
        raise ValueError("External victim did not hold before HCCL")
    healthy = [m for m in command["group"]["members"] if not _same_identity(m, member)]
    submitted = _wait(lambda: _reports(run, spec, healthy, token, "submitted"), spec["phase_timeout"], "healthy async collective submissions")
    if any(record["collectives"] <= 0 for record in submitted.values()):
        raise ValueError("Healthy rank did not submit real communication")
    kill = {**hold, "action": "kill_own_child", "injected_by_independent_actor": True}
    _write(run / "supervisor_commands" / f"{args.victim_worker}.json", kill)
    _write(run / "injection.json", {"run_id": spec["run_id"], "token": token, "victim": _identity(member),
            "base_step": command["base_step"], "controller_command_had_no_fault_victim": True,
            "held": held, "kill": kill, "time": time.time()})
    return 0


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--controller", action="store_true")
    mode.add_argument("--worker", action="store_true", help="CPU-only supervisor for one allocated NPU")
    mode.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--injector", action="store_true", help="Independent CPU-only supervisor hold/kill actor")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-source-spec", type=Path)
    parser.add_argument("--worker-id")
    parser.add_argument("--session-id", help=argparse.SUPPRESS)
    parser.add_argument("--device-index", type=int, default=0, help="Logical device within the job's unchanged visibility mask")
    parser.add_argument("--allocated-devices", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--replacement", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--initial-standby", action="store_true",
                        help="Supervisor: load independently outside the initial active communication group")
    parser.add_argument("--cards", type=int, default=8)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--fault-step", type=int, default=3, help="One-based update at which to kill the last rank")
    parser.add_argument("--no-replacement", action="store_true", help="Only validate 8-to-7 shrink training")
    parser.add_argument("--phase-timeout", type=float, default=600)
    parser.add_argument("--collective-timeout", type=float, default=120)
    parser.add_argument("--bucket-mib", type=int, default=128)
    parser.add_argument("--data-workers", type=int, default=None,
                        help="CPU readers per NPU; defaults to config resident_data_workers, or zero")
    parser.add_argument("--external-loss", action="store_true", help="No planned victim; detect one unknown loss before optimizer authorization")
    parser.add_argument("--audit-before-collective", action="store_true", help="Short probe only: full native p/m/v/gradient hashes every step")
    parser.add_argument("--heartbeat-timeout", type=float, default=15)
    parser.add_argument("--communication-diagnostics", action="store_true",
                        help="Time collective submission, report writes, work waits and final sync separately")
    parser.add_argument("--communication-diagnostics-steps", type=int, default=0,
                        help="Diagnostic update window from start_step; zero records the full run")
    parser.add_argument("--control-poll-seconds", type=float, choices=(0.01, 0.02, 0.05, 0.1), default=0.1,
                        help="Hot command/APPLY/report polling; preserves each gate's shorter legacy interval")
    parser.add_argument("--join-transport", choices=("file", "device"), default="file")
    parser.add_argument("--victim-worker", default="w000008", help="Injector-only target within its isolated run")
    parser.add_argument("--inject-after-step", type=int, default=3, help="Injector-only earliest one-based step")
    parser.add_argument("--inject-stage", choices=("before-collective", "backward"), default="before-collective",
                        help="Injector-only: kill after exact backward-phase observation or hold before HCCL")
    args = parser.parse_args(argv)
    if not 0 <= args.device_index < args.allocated_devices:
        parser.error("--device-index must be inside the allocated visible device set")
    if args.controller and args.config is None:
        parser.error("--controller requires --config")
    if bool(args.resume_checkpoint) != bool(args.resume_source_spec) or (args.resume_checkpoint and not args.controller):
        parser.error("Resume requires controller mode, --resume-checkpoint and --resume-source-spec")
    if (args.worker or args.child) and (not args.worker_id or not _SAFE_ID.fullmatch(args.worker_id)):
        parser.error("A safe --worker-id is required")
    if args.child and (not args.session_id or not _SAFE_ID.fullmatch(args.session_id)):
        parser.error("Internal child requires a safe session")
    if args.initial_standby and not args.worker:
        parser.error("--initial-standby is a supervisor-only option")
    if not 2 <= args.cards <= 64 or not 1 <= args.fault_step < args.steps <= 20000:
        parser.error("Expected 2..64 cards and 1 <= fault-step < steps <= 20000")
    if min(args.phase_timeout, args.collective_timeout, args.bucket_mib, args.heartbeat_timeout) <= 0:
        parser.error("Timeouts and bucket size must be positive")
    if args.communication_diagnostics_steps < 0:
        parser.error("Communication diagnostics steps must be nonnegative")
    if args.communication_diagnostics_steps and not args.communication_diagnostics:
        parser.error("Communication diagnostics steps requires --communication-diagnostics")
    if args.data_workers is not None and not 0 <= args.data_workers <= 8:
        parser.error("--data-workers must be in 0..8")
    if not args.no_replacement and not args.external_loss and args.steps < args.fault_step + 3:
        parser.error("Replacement requires two healthy updates and a subsequent joined update")
    return args


def main(argv=None):
    args = options(argv)
    return controller(args) if args.controller else child(args) if args.child else injector(args) if args.injector else supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
