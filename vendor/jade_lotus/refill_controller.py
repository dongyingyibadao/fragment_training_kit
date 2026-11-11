"""Optional exact-16 admission at an optimizer COMMIT boundary.

Cold loading is asynchronous. Only complete native state synchronization pauses
the current group. Candidates never participate in a gradient before admission.
"""
from __future__ import annotations

import time
import math
from pathlib import Path

from .ascend_platform import RUNNING
from .pod_controller_runtime import PodRecoveryRequired
from .refill_protocol import atomic_write, mapping_digest, read_allocation, validate_permit
from .refill_state import BATCH_SIZE, bounded_lock, cancel, immutable, load_state, read, save_state
from .runtime import writer_lock, WriterLockBusy


class CandidateRejected(RuntimeError):
    pass


def fresh(record, key, maximum):
    value = record.get(key)
    return (type(value) in (int, float) and math.isfinite(value)
            and -5 <= time.time() - value <= maximum)


def verified_reservation_devices(mapping, reservation, official):
    count = len(mapping['members'])
    if official.get('allocation_verified') is True:
        return official.get('allocated_cards') == count
    if official.get('allocated_cards') is not None:
        return False  # A conflicting official allocation is never overridden.
    proof = reservation.get('device_visibility', {})
    devices = proof.get('devices')
    if (proof.get('verified') is not True or type(proof.get('count')) is not int
            or proof['count'] != count or not isinstance(devices, list)
            or len(devices) != count or len(set(devices)) != count
            or reservation.get('hostname') not in official.get('running_instance_names', [])):
        return False
    if proof.get('kind') == 'assigned_environment_mask':
        return (proof.get('source') in {'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES', 'NPU_VISIBLE_DEVICES'}
                and all(type(d) is int and d >= 0 for d in devices))
    if proof.get('kind') == 'accessible_character_devices':
        import re
        return all(isinstance(d, str) and re.fullmatch(r'davinci[0-9]+', d) for d in devices)
    return False


class RefillController:
    def __init__(self, pod, config):
        self.pod, self.api, self.spec = pod, pod.api, pod.spec
        self.run, self.root, self.config = pod.run, pod.run.parent, config
        self.sha = self.spec['config_sha256']
        self.target = config['resident_refill_target_cards']
        self.last_poll = 0.
        self.counters = {}

    def _compact(self, members, group, step):
        if set(self.pod.watch.keys) != {m['worker_id'] for m in members}:
            self.pod.compact_liveness(members, group, step)

    def _mappings(self, batch):
        mappings = []
        for prefix in batch['allocations']:
            value = read(self.root / 'refill/allocations' / f'{prefix}.json')
            if not value:
                raise CandidateRejected('Candidate allocation mapping disappeared')
            mapping = read_allocation(self.root, prefix, len(value['members']))
            if mapping['role'] != 'candidate' or mapping['batch_id'] != batch['batch_id']:
                raise CandidateRejected('Candidate allocation belongs to another batch')
            mappings.append(mapping)
        return mappings

    def _reservations(self, batch, mappings):
        records = []
        for mapping in mappings:
            prefix = mapping['worker_prefix']
            reservation = read(self.root / 'refill/reservations' / f'{prefix}.json')
            official = batch.get('official', {}).get(prefix, {})
            if (not reservation or reservation.get('state') != 'reserved'
                    or not fresh(reservation, 'updated_at', self.spec['heartbeat_timeout'])
                    or official.get('status') not in RUNNING or not official.get('priority_verified')
                    or not verified_reservation_devices(mapping, reservation, official)
                    or not fresh(official, 'observed_at', 120)):
                return None
            for field in ('run_dir', 'config_sha256', 'worker_prefix', 'intent_id', 'batch_id'):
                if reservation.get(field) != mapping[field]:
                    raise CandidateRejected('Reservation identity differs from its allocation')
            if reservation.get('mapping_sha256') != mapping_digest(mapping):
                raise CandidateRejected('Reservation mapping digest changed')
            records.append(reservation)
        return records

    def _permit(self, batch, mappings, reservations):
        members = [m['worker_id'] for a in mappings for m in a['members']]
        if len(members) != BATCH_SIZE or len(set(members)) != BATCH_SIZE:
            raise CandidateRejected('Loading requires exactly 16 distinct candidate members')
        batch.update(phase='loading', member_ids=members, load_started_at=time.time(),
                     wrapper_sessions={r['worker_prefix']: r['wrapper_session'] for r in reservations})
        # Persist the frozen batch before any model process may be started.
        for mapping, reservation in zip(mappings, reservations, strict=True):
            permit = {'schema': 'fragment-refill-load-v1',
                **{k: mapping[k] for k in ('run_dir', 'config_sha256', 'worker_prefix', 'intent_id', 'batch_id')},
                'mapping_sha256': mapping_digest(mapping), 'wrapper_session': reservation['wrapper_session'],
                'batch_member_ids': members}
            if not validate_permit(mapping, reservation, permit):
                raise CandidateRejected('Loading permit failed its own identity validation')
        return members

    def _write_permits(self, batch, mappings):
        for mapping in mappings:
            prefix = mapping['worker_prefix']
            immutable(self.root / 'refill/permits' / f'{prefix}.json', {
                'schema': 'fragment-refill-load-v1',
                **{k: mapping[k] for k in ('run_dir', 'config_sha256', 'worker_prefix', 'intent_id', 'batch_id')},
                'mapping_sha256': mapping_digest(mapping), 'wrapper_session': batch['wrapper_sessions'][prefix],
                'batch_member_ids': batch['member_ids']})

    def _candidates(self, batch, *, expected=None):
        try:
            return self._candidate_records(batch, expected=expected)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            raise CandidateRejected(f'Candidate evidence is invalid: {error}') from error

    def _candidate_records(self, batch, *, expected=None):
        candidates = []
        now = time.monotonic()
        for mapping in self._mappings(batch):
            prefix = mapping['worker_prefix']
            reservation = read(self.root / 'refill/reservations' / f'{prefix}.json')
            if (not reservation or reservation.get('wrapper_session') != batch['wrapper_sessions'][prefix]
                    or not fresh(reservation, 'updated_at', self.spec['heartbeat_timeout'])):
                raise CandidateRejected('Candidate wrapper disappeared or restarted')
            for item in mapping['members']:
                worker = item['worker_id']
                ready = self.api._read(self.run / 'ready' / f'{worker}.json')
                supervisor = self.api._read(self.run / 'supervisors' / f'{worker}.json')
                if not ready:
                    # Fail early for a loader that exited before READY.
                    if supervisor and self.api._read(self.run / 'exits' / f"{worker}-{supervisor['session_id']}.json"):
                        raise CandidateRejected('Candidate model loader exited before READY')
                    return None
                if (ready.get('run_id') != self.spec['run_id'] or ready.get('worker_id') != worker
                        or ready.get('allocation_group') != prefix or ready.get('device_index') != item['device_index']
                        or ready.get('refill_wrapper_session') != reservation['wrapper_session']
                        or ready.get('refill_batch_id') != batch['batch_id'] or ready.get('replacement') is not True
                        or not supervisor or supervisor.get('run_id') != self.spec['run_id']
                        or not self.api._same_identity(supervisor, ready)
                        or supervisor.get('refill_wrapper_session') != reservation['wrapper_session']):
                    raise CandidateRejected('Candidate READY has stale or mismatched authority')
                for directory in ('errors', 'exits'):
                    if self.api._read(self.run / directory / f"{worker}-{ready['session_id']}.json") is not None:
                        raise CandidateRejected('Candidate exited or reported a device error')
                counter = supervisor.get('heartbeat_counter')
                if type(counter) is not int or counter < 0:
                    raise CandidateRejected('Candidate lacks a monotonic CPU heartbeat')
                key = (worker, ready['session_id'], ready['pid'])
                previous, observed = self.counters.get(key, (counter, now))
                if counter < previous:
                    raise CandidateRejected('Candidate heartbeat counter moved backwards')
                if counter > previous:
                    observed = now
                self.counters[key] = (counter, observed)
                if now - observed > self.spec['heartbeat_timeout']:
                    raise CandidateRejected('Candidate CPU heartbeat expired')
                candidates.append(ready)
        if len(candidates) != BATCH_SIZE:
            return None
        if expected is not None and not self.pod.same(expected, candidates):
            raise CandidateRejected('Candidate sessions changed during boundary synchronization')
        return candidates

    def _wait_candidates(self, batch, candidates, token, phase, deadline):
        while time.monotonic() < deadline:
            self.pod._remaining()
            self.pod.live()  # Active execution failures still fail closed.
            if self._candidates(batch, expected=candidates) is None:
                raise CandidateRejected('Candidate readiness disappeared during admission')
            rows = self.api._reports(self.run, self.spec, candidates, token, phase)
            if rows is not None:
                return rows
            time.sleep(self.api._control_poll_seconds(self.spec))
        raise CandidateRejected('Bounded candidate alignment deadline expired')

    def on_commit(self, members, group, step, *, final=False):
        self._compact(members, group, step)
        atomic_write(self.root / 'refill/progress.json', {'run_id': self.spec['run_id'], 'at': time.time(),
            'completed_step': step, 'active_member_ids': [m['worker_id'] for m in members],
            'active_allocation_groups': sorted({m.get('allocation_group', m['worker_id']) for m in members}),
            'target_cards': self.target})
        if not final and time.monotonic() - self.last_poll < .5:
            return members, group
        self.last_poll = time.monotonic()
        admission = None
        try:
            with writer_lock(self.root / 'refill/lock'):
                state = load_state(self.root, self.sha)
                batch = state['current']
                if batch and batch['phase'] == 'admitted' and step > batch['admitted_step']:
                    batch.update(first_successful_commit=step,
                        candidate_survivors=[m['worker_id'] for m in members if m['worker_id'] in batch['member_ids']])
                    state['history'].append(batch)
                    state['current'], batch = None, None
                elif batch and batch['phase'] == 'cancelled':
                    state['history'].append(batch)
                    state['current'], batch = None, None
                if final:
                    if batch and batch['phase'] in {'reserving', 'loading'}:
                        cancel(self.root, state, batch, 'Training reached its final committed update')
                    save_state(self.root, state)
                    return members, group
                if batch is None and len(members) < self.target:
                    if state['next_batch'] > self.config['resident_refill_max_batches']:
                        state['refill_paused_reason'] = 'Configured bounded batch history exhausted'
                    else:
                        batch = {'batch_id': f"b{state['next_batch']:06d}", 'phase': 'reserving',
                                 'allocations': [], 'created_at': time.time(), 'created_step': step}
                        state['next_batch'] += 1
                        state['current'] = batch
                if batch and batch['phase'] in {'reserving', 'loading'}:
                    try:
                        if batch.get('failure'):
                            raise CandidateRejected(batch['failure'])
                        if time.time() - batch['created_at'] > self.config['resident_refill_candidate_timeout']:
                            raise CandidateRejected('Bounded candidate reservation/loading lifetime expired')
                        mappings = self._mappings(batch)
                        if batch['phase'] == 'reserving' and sum(len(m['members']) for m in mappings) == BATCH_SIZE:
                            reservations = self._reservations(batch, mappings)
                            if reservations is not None:
                                self._permit(batch, mappings, reservations)
                                save_state(self.root, state)
                                self._write_permits(batch, mappings)
                        elif batch['phase'] == 'loading':
                            candidates = self._candidates(batch)
                            if candidates and len(members) + BATCH_SIZE <= self.target:
                                batch.update(phase='admitting', boundary_step=step)
                                admission = batch.copy(), candidates
                    except (CandidateRejected, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                        cancel(self.root, state, batch, str(error))
                save_state(self.root, state)
        except WriterLockBusy:
            return members, group
        if admission is not None:
            batch, candidates = admission
            members, group, evidence = self._admit(batch, candidates, members, group, step)
            with bounded_lock(self.root):
                state = load_state(self.root, self.sha)
                current = state['current']
                if current['batch_id'] != batch['batch_id'] or current['phase'] != 'admitting':
                    raise PodRecoveryRequired('Admission authority was modified concurrently')
                if evidence['status'] == 'admitted':
                    current.update(phase='admitted', admitted_step=step, admission=evidence)
                    immutable(self.root / 'refill/admissions' / f"{batch['batch_id']}.json", evidence)
                else:
                    cancel(self.root, state, current, evidence['reason'])
                    current['rejection'] = evidence
                save_state(self.root, state)
            self.pod.summary.setdefault('refill_events', []).append(evidence)
            self._compact(members, group, step)
        return members, group

    def _admit(self, batch, candidates, members, group, step):
        started = time.monotonic()
        # One coherent budget, including rollback. Individual phases may not
        # renew a healthy worker's idle lease indefinitely.
        deadline = started + 2 * self.spec['phase_timeout']
        lease_end = started + 3 * self.spec['phase_timeout']
        self.pod.deadline = deadline
        current, current_group, reset_started = list(members), group, False

        def hold():
            remaining = lease_end - time.monotonic()
            if remaining <= 0:
                raise PodRecoveryRequired('Refill boundary hold deadline exhausted')
            token = self.pod.send(current, 'hold_committed_boundary', expected_step=step,
                expected_generation=current_group['generation'], hold_seconds=remaining)
            return self.pod.wait(current, token, 'boundary_held')[0]

        try:
            current = self.pod.live(current)
            if not self.pod.same(current, current_group['members']):
                current, current_group = self.pod.rebuild(current, current_group, step, preserve_gradients=False)
            current = hold()
            current, current_group, checkpoint, inspection = self.pod.checkpoint_boundary(
                current, current_group, step, export=True, deadline=deadline)
            authority = self.pod._state_hashes(inspection['members'], step)
            current = hold()  # Preserve the same absolute lease_end after inspect.
            if self._candidates(batch, expected=candidates) is None:
                raise CandidateRejected('Candidate left before native-state synchronization')
            load = self.pod.send(candidates, 'load', expected_step=step, path=checkpoint['path'],
                                 state_metadata=self.api._boundary_metadata(self.spec, step))
            loaded = self._wait_candidates(batch, candidates, load, 'loaded', min(deadline, time.monotonic() + self.spec['phase_timeout']))
            for row in loaded.values():
                try:
                    self.pod._validate_export(row, step)
                except PodRecoveryRequired as error:
                    raise CandidateRejected(str(error)) from error
                for field in ('model_sha256', 'optimizer_sha256', 'manifest_sha256'):
                    if row['receipt'][field] != checkpoint['receipt'][field]:
                        raise CandidateRejected('Candidate did not load the authoritative native checkpoint')
            inspect = self.pod.send(candidates, 'inspect', expected_step=step)
            inspected = self._wait_candidates(batch, candidates, inspect, 'inspected', deadline)
            try:
                candidate_hashes = self.pod._state_hashes(inspected, step)
            except PodRecoveryRequired as error:
                raise CandidateRejected(str(error)) from error
            if candidate_hashes != authority:
                raise CandidateRejected('Candidate model/Adam/configuration differs from committed authority')
            current = self.pod.live(current)
            if len(current) + BATCH_SIZE > min(64, self.target):
                raise CandidateRejected('Exact-16 admission would exceed the active-card target')
            if self._candidates(batch, expected=candidates) is None:
                raise CandidateRejected('Candidate left before communication reset')
            try:
                self.pod.watch.register_ready(candidates, receipt=f'full-native-alignment/{batch["batch_id"]}/{step}')
            except (ValueError, RuntimeError) as error:
                raise CandidateRejected(f'Candidate registration changed: {error}') from error
            reset_started = True
            try:
                union, new_group = self.pod.rebuild(current + candidates, current_group, step, preserve_gradients=False)
            except PodRecoveryRequired as error:
                # Only a concrete candidate device error is optional here.
                # Storage integrity, exhausted budgets and healthy failures
                # remain fatal even if another candidate happens to be lost.
                bad_candidates = [m for m in candidates if self.api._read(
                    self.run/'errors'/f"{m['worker_id']}-{m['session_id']}.json") is not None]
                if (bad_candidates and str(error).startswith('Device execution failed;')
                        and any(m['worker_id'] in str(error) for m in bad_candidates)):
                    raise CandidateRejected(str(error)) from error
                raise
            if not {m['worker_id'] for m in candidates}.issubset({m['worker_id'] for m in union}):
                current_group = new_group
                raise CandidateRejected('Candidate lost during group initialization; exact-16 admission withdrawn')
            self.pod.members = union
            self.pod._persist('committed_refill', completed_step=step, batch_id=batch['batch_id'])
            return union, new_group, {'batch_id': batch['batch_id'], 'status': 'admitted', 'step': step,
                'candidates': [self.api._identity(m) for m in candidates],
                'members': [self.api._identity(m) for m in union], 'candidate_count': BATCH_SIZE,
                'full_native_state_alignment_verified': True, 'state_hashes': list(next(iter(authority))),
                'checkpoint': checkpoint, 'boundary_pause_seconds': time.monotonic() - started,
                'optimizer_updates_during_admission': 0, 'healthy_rollback_count': 0}
        except CandidateRejected as error:
            # Numerical errors of healthy replicas remain fatal. Only isolated
            # candidate failures may be rejected with an unchanged COMMIT k.
            current = self.pod.live(current)
            self.pod.deadline = lease_end
            if reset_started or not self.pod.same(current, current_group['members']):
                current, current_group = self.pod.rebuild(current, current_group, step, preserve_gradients=False)
            else:
                release = self.pod.send(current, 'release_committed_boundary', expected_step=step,
                                       expected_generation=current_group['generation'])
                current, _ = self.pod.wait(current, release, 'boundary_released')
            self.pod.members = current
            return current, current_group, {'batch_id': batch['batch_id'], 'status': 'rejected', 'step': step,
                'reason': str(error), 'reset_started': reset_started, 'active_group_continued': True,
                'boundary_pause_seconds': time.monotonic() - started, 'healthy_rollback_count': 0,
                'optimizer_updates_during_admission': 0}
