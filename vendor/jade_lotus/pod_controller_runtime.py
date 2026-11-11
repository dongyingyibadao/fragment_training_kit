"""CPU-only, bounded repeated pod-loss orchestration for split worker phases.

Uses ResidentLiveness solely as the exact-session heartbeat/fencing authority.
The phase ledger here owns APPLY authorization: the legacy single-loss detector
is never reset or used to authorize a partially completed optimizer update.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid

from .pod_recovery import repair_ownership, validate_forward_update


class PodRecoveryRequired(RuntimeError):
    pass


class PodController:
    def __init__(self, api, run, spec, members, summary):
        self.api, self.run, self.spec, self.summary = api, run, spec, summary
        self.members = list(members)
        self.watch = api.ControllerLiveness(run, spec)
        self.watch.register_ready(members, receipt='split-phase-initial-native-state')
        self.generation = 0
        self.recoveries, self.events = 0, []
        self.deadline = None
        self.inflight, self.committed_step = False, spec.get('start_step', 0)
        self.max_recoveries = spec.get('pod_max_recoveries', 64)
        self._persist('ready', expected_step=spec.get('start_step', 0))

    def _persist(self, phase, **fields):
        value = {'schema': 'jade-pod-recovery-runtime-v1', 'run_id': self.spec['run_id'],
                 'config_sha256': self.spec['config_sha256'], 'phase': phase,
                 'at': time.time(), 'recoveries': self.recoveries, 'generation': self.generation,
                 'members': [self.api._identity(m) for m in self.members], **fields}
        self.api._write(self.run/'pod'/'phase.json', value)
        return value

    def compact_liveness(self, members, group, step):
        """Retire historical polling only after the split protocol's COMMIT.

        This detector owns heartbeats, not APPLY. Do not impersonate a legacy
        detector finish_step to reset its view. Keep every irreversible fence.
        """
        old = self.watch
        if self.inflight or step != self.committed_step or old.detector.phase != 'idle':
            raise PodRecoveryRequired('Pod liveness compaction requires its verified COMMIT boundary')
        if not self.same(members, group['members']):
            raise PodRecoveryRequired('Cannot compact a group with unacknowledged membership')
        from dataclasses import asdict
        new = self.api.ControllerLiveness(self.run, self.spec,
            persisted_fences=old.detector.fenced_sessions, used_tokens=old.used_tokens,
            generation_floor=max(old.generation_floor, group['generation']), epoch=old.detector.epoch,
            last_committed_step=step)
        new.seen_exits = set(old.seen_exits)
        new._persisted_losses = dict(old._persisted_losses)
        new.register_ready(members, receipt=f'pod-verified-commit/{step}')
        self.api._write(self.run/'pod'/'liveness'/f'{step:06d}-{group["generation"]:06d}.json', {
            'run_id': self.spec['run_id'], 'committed_step': step, 'generation': group['generation'],
            'members': [self.api._identity(m) for m in members],
            'retained_fenced_sessions': [asdict(k) for k in sorted(old.detector.fenced_sessions, key=repr)]})
        self.watch = new
        self.members = list(members)

    def live(self, members=None):
        members = list(self.members if members is None else members)
        # Numerical/model execution errors are never laundered into preemption,
        # even when a supervisor has subsequently recorded the process exit.
        for member in members:
            error = self.api._read(self.run/'errors'/f"{member['worker_id']}-{member['session_id']}.json")
            if error is not None:
                raise PodRecoveryRequired(f"Device execution failed; retain original error: {member['worker_id']}")
        # poll() persists every newly fenced exact session before returning.
        self.watch.poll()
        fenced = self.watch.detector.fenced_sessions
        survivors = [m for m in members if self.watch.keys[m['worker_id']] not in fenced]
        if not survivors:
            raise PodRecoveryRequired('No surviving current replica; explicit complete checkpoint recovery required')
        return survivors

    def _remaining(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise PodRecoveryRequired('Bounded pod recovery/update deadline expired')

    @staticmethod
    def _validate_export(record, step):
        receipt = record.get('receipt', {})
        if (type(record.get('completed_step')) is not int or record['completed_step'] != step
                or type(receipt.get('step')) is not int or receipt['step'] != step):
            raise PodRecoveryRequired('Export is not the committed checkpoint step')
        if (receipt.get('payload_bytes_verified') is not True
                or any(not isinstance(receipt.get(key), str)
                       or re.fullmatch(r'[0-9a-f]{64}', receipt[key]) is None
                       for key in ('model_sha256', 'optimizer_sha256', 'manifest_sha256'))):
            raise PodRecoveryRequired('Export lacks a verified complete checkpoint receipt')

    @staticmethod
    def _state_hashes(rows, step):
        fields = ('model_sha256', 'optimizer_sha256', 'optimizer_config_sha256')
        if (not rows or any(type(row.get('completed_step')) is not int or row['completed_step'] != step
                or type(row.get('checkpoint_calls')) is not int or row['checkpoint_calls'] != 0
                or any(not isinstance(row.get(key), str)
                       or re.fullmatch(r'[0-9a-f]{64}', row[key]) is None for key in fields)
                for row in rows.values())):
            raise PodRecoveryRequired('Complete state report does not match the committed boundary')
        hashes = {tuple(row[key] for key in fields) for row in rows.values()}
        if len(hashes) != 1:
            raise PodRecoveryRequired('Complete native model/Adam states disagree')
        return hashes

    def wait(self, members, token, phase):
        """Finish the serial phase of every still-live exact session.

        Lost members are omitted only through persisted liveness fences. A live
        device failure is not silently relabeled as a preempted job.
        """
        while True:
            self._remaining()
            survivors = self.live(members)
            for m in survivors:
                error = self.api._read(self.run/'errors'/f"{m['worker_id']}-{m['session_id']}.json")
                if error is not None:
                    # Give its supervisor one polling interval to publish exit,
                    # then the normal exact liveness fence will remove it.
                    exit_record = self.api._read(self.run/'exits'/f"{m['worker_id']}-{m['session_id']}.json")
                    if exit_record is None:
                        raise PodRecoveryRequired(f"Active device reported an execution error: {m['worker_id']}")
            reports = self.api._reports(self.run, self.spec, survivors, token, phase)
            if reports is not None:
                return survivors, reports
            time.sleep(self.api._control_poll_seconds(self.spec))

    def send(self, members, action, *, expected_step, root_token=None, **fields):
        return self.api._send(self.run, self.spec, members, action, expected_step=expected_step,
                              **({'root_token': root_token} if root_token is not None else {}), **fields)

    @staticmethod
    def same(left, right):
        return [(m['worker_id'], m['session_id'], m['pid']) for m in left] == [
            (m['worker_id'], m['session_id'], m['pid']) for m in right]

    def _charge(self, reason, step, old, new):
        self.recoveries += 1
        if self.recoveries > self.max_recoveries:
            raise PodRecoveryRequired('Configured maximum pod-recovery attempts exceeded')
        event = {'number': self.recoveries, 'reason': reason, 'step': step,
                 'lost': [self.api._identity(m) for m in old if not any(self.api._same_identity(m, n) for n in new)],
                 'survivors': [self.api._identity(m) for m in new]}
        self.events.append(event)
        self.api._write(self.run/'pod'/'events'/f'{self.recoveries:06d}.json', event)

    def rebuild(self, members, old_group, step, *, preserve_gradients):
        """Quiesce serial worker phases; rebuild with a fresh real rendezvous.

        Failures during init are bounded by HCCL's init timeout; no rank may
        start the next compute/reduction before all survivors acknowledge init.
        """
        current = list(members)
        while True:
            self._remaining()
            survivors = self.live(current)
            self._charge('rebuild_group', step, old_group['members'], survivors)
            self.members = survivors
            reset = self.send(survivors, 'pod_reset', expected_step=step, preserve_gradients=preserve_gradients)
            survivors, rows = self.wait(survivors, reset, 'pod_reset_done')
            if not all(row['state_storage_pointers_unchanged'] for row in rows.values()):
                raise PodRecoveryRequired('Resident storage changed while discarding old communication')
            self.generation = max(self.generation, old_group['generation']) + 1
            group = self.api._group(survivors, self.generation,
                                    port=rows[survivors[0]['worker_id']]['next_port'])
            self._persist('initializing_recovered_group', expected_step=step, group=group)
            init = self.send(survivors, 'pod_init', expected_step=step, group=group)
            remaining, initialized = self.wait(survivors, init, 'pod_initialized')
            if self.same(survivors, remaining) and all(row['status']=='ready' for row in initialized.values()):
                self.members = remaining
                return remaining, group
            current, old_group = remaining, group

    def settle(self, members, mapping, command_token, *, step, root_token):
        """Finish every healthy local accumulation, then replay only missing owners.

        On another loss while replaying, wait for surviving replay commands to
        complete before assigning any more units. Thus no partial accumulation
        can be appended twice; current mapping follows transferred ownership.
        """
        current = list(members)
        while True:
            surviving, rows = self.wait(current, command_token, 'pod_local')
            for m in surviving:
                row = rows[m['worker_id']]
                if (row['completed_step'] != step or not row.get('complete_local_backward')
                        or sorted(row['local']['units']) != sorted(mapping[m['worker_id']])):
                    raise PodRecoveryRequired('Local-gradient ownership/step fence mismatch')
            if set(mapping)=={m['worker_id'] for m in surviving}:
                return surviving, mapping
            repaired, replay = repair_ownership(mapping, [m['worker_id'] for m in surviving])
            self._charge('replay_missing_owners', step, current, surviving)
            self.members = surviving
            self._persist('replaying_missing_units', expected_step=step, root_token=root_token,
                          previous_assignments=mapping, assignments=repaired, replay=replay)
            command_token = self.send(surviving, 'pod_replay', expected_step=step, root_token=root_token,
                                      replay=replay)
            current, mapping = surviving, repaired

    def run_step(self, members, group, step):
        if self.inflight or step != self.committed_step:
            raise PodRecoveryRequired('Pod update must start at its last verified COMMIT')
        self.inflight = True
        self.deadline = time.monotonic() + self.spec.get('pod_step_timeout', self.spec['phase_timeout'] * 4)
        current = self.live(members)
        if not self.same(current, members):
            current, group = self.rebuild(current, group, step, preserve_gradients=False)
        self.members = current
        self.generation = group['generation']
        root_token = uuid.uuid4().hex
        mapping = self.api.assignments(current)
        self._persist('computing', expected_step=step, root_token=root_token, assignments=mapping)
        compute = self.send(current, 'pod_compute', expected_step=step, root_token=root_token,
                            group=group, assignments=mapping)
        current, mapping = self.settle(current, mapping, compute, step=step, root_token=root_token)
        if not self.same(current, group['members']):
            current, group = self.rebuild(current, group, step, preserve_gradients=True)
            # A reset/init may itself lose an owner. The completed gradients
            # for each surviving owner are still intact.
            current, mapping = self._repair_ready(current, mapping, step, root_token)
        while True:
            current, mapping = self._repair_ready(current, mapping, step, root_token)
            if not self.same(current, group['members']):
                current, group = self.rebuild(current, group, step, preserve_gradients=True)
                continue
            self.members = current
            token = self.send(current, 'pod_reduce', expected_step=step, root_token=root_token,
                              group=group, assignments=mapping)
            self._persist('reducing', expected_step=step, root_token=root_token, reduction_token=token,
                          group=group, assignments=mapping)
            reduced = None
            while True:
                self._remaining()
                remaining = self.live(current)
                if not self.same(current, remaining):
                    self.api._write(self.run/'abort'/f'{token}.json', {
                        'action': 'abort_collective', 'run_id': self.spec['run_id'], 'token': token,
                        'base_step': step, 'generation': group['generation'], 'members': current,
                        'peer_loss_fenced': True, 'confirmed_peer_exit': False})
                    remaining, _ = self.wait(remaining, token, 'pod_aborted')
                    current, mapping = self._repair_ready(remaining, mapping, step, root_token)
                    current, group = self.rebuild(current, group, step, preserve_gradients=True)
                    current, mapping = self._repair_ready(current, mapping, step, root_token)
                    break
                reduced = self.api._reports(self.run, self.spec, current, token, 'pod_reduced')
                if reduced is not None:
                    release = {'action':'release_reduction', 'run_id':self.spec['run_id'], 'token':token,
                               'base_step':step, 'generation':group['generation']}
                    self.api._write(self.run/'pod'/'reduce_release'/f'{token}.json', release)
                    remaining, _ = self.wait(current, token, 'pod_released')
                    if not self.same(current, remaining):
                        current, mapping = self._repair_ready(remaining, mapping, step, root_token)
                        current, group = self.rebuild(current, group, step, preserve_gradients=True)
                        current, mapping = self._repair_ready(current, mapping, step, root_token)
                        reduced = None
                    break
                time.sleep(self.api._control_poll_seconds(self.spec))
            if reduced is not None:
                break
        self.api.validate_coverage(mapping)
        prepared_members = list(current)
        prepared_loss = {w:r['local']['loss_contribution'] for w,r in reduced.items()}
        if any(sorted(reduced[m['worker_id']]['local']['units']) != sorted(mapping[m['worker_id']]) for m in current):
            raise PodRecoveryRequired('Prepared reduction does not cover exactly one global batch')
        update_token = uuid.uuid4().hex
        authorization = self._persist('applying', base_step=step, root_token=root_token,
            update_token=update_token, reduction_token=token, group=group, assignments=mapping,
            prepared_members=[self.api._identity(m) for m in current], prepared_loss=prepared_loss,
            input_positions={'first':step*256, 'last_exclusive':(step+1)*256, 'seed':self.spec['seed'],
                             'rng_scheme':self.spec['rng_scheme']}, optimizer_replay_forbidden=True)
        # Write-once immutable authorization before any device optimizer command.
        self.api._write(self.run/'pod'/'authorizations'/f'{update_token}.json', authorization)
        self.api._send(self.run, self.spec, current, 'pod_apply', token=update_token,
            expected_step=step, root_token=root_token, reduction_token=token, group=group)
        surviving, updated = self.wait(current, update_token, 'updated')
        for m in surviving:
            row = updated[m['worker_id']]
            if (row['completed_step'] != step+1 or row['checkpoint_calls'] != 0
                    or row['optimizer_applications'] != 1 or row['root_token'] != root_token
                    or row['reduction_token'] != token or row['generation'] != group['generation']):
                raise PodRecoveryRequired('Updated report does not match the one authorized optimizer application')
        forward = None
        if not self.same(surviving, current):
            self.members = surviving
            self._persist('verifying_forward_recovery', base_step=step, update_token=update_token,
                          optimizer_replay_forbidden=True)
            inspection = self.send(surviving, 'inspect', expected_step=step+1)
            surviving, inspections = self.wait(surviving, inspection, 'inspected')
            updated = {m['worker_id']: updated[m['worker_id']] for m in surviving}
            forward = validate_forward_update(self.spec, surviving, update_token, step,
                group['generation'], updated, inspections, inspection)
            self.api._write(self.run/'pod'/'forward'/f'{update_token}.json', forward)
            surviving, group = self.rebuild(surviving, group, step+1, preserve_gradients=False)
        self.members = surviving
        self._persist('updated_verified', completed_step=step+1, root_token=root_token,
                      update_token=update_token, forward_recovery=forward)
        self.inflight, self.committed_step = False, step + 1
        return surviving, group, {'token': update_token, 'root_token':root_token,
            'assignments':mapping, 'reduction_token':token, 'prepared_members':[self.api._identity(m) for m in prepared_members],
            'prepared_loss_contributions':prepared_loss, 'pod_recovery':True,
            'forward_recovery':forward, 'recovery_events':self.events[:],
            'samples':256, 'atomic_units':64}

    def _repair_ready(self, members, mapping, step, root_token):
        """Workers are serially idle with complete original/earlier replay grads."""
        current = self.live(members)
        while set(mapping) != {m['worker_id'] for m in current}:
            repaired, replay = repair_ownership(mapping, [m['worker_id'] for m in current])
            self._charge('replay_missing_ready_owners', step, [
                {'worker_id':w, 'session_id':self.watch.keys[w].session_id, 'pid':self.watch.keys[w].child_pid}
                for w in mapping], current)
            self.members = current
            self._persist('replaying_missing_units', expected_step=step, root_token=root_token,
                          assignments=repaired, replay=replay)
            token = self.send(current, 'pod_replay', expected_step=step, root_token=root_token, replay=replay)
            current, mapping = self.settle(current, repaired, token, step=step, root_token=root_token)
        return current, mapping

    def checkpoint_boundary(self, members, group, step, *, export=False, deadline=None):
        self.deadline = time.monotonic() + self.spec['phase_timeout'] * 4 if deadline is None else deadline
        current = self.live(members)
        record = None
        if export:
            while record is None:
                source = current[0]
                path = self.run/'states'/f'checkpoint-{step}-pod-{uuid.uuid4().hex}'
                token = self.send([source], 'export', expected_step=step, path=str(path),
                                  state_metadata=self.api._boundary_metadata(self.spec, step))
                while True:
                    self._remaining()
                    current = self.live(current)
                    if not any(self.api._same_identity(source,m) for m in current):
                        break
                    rows = self.api._reports(self.run,self.spec,[source],token,'exported')
                    if rows:
                        self._validate_export(rows[source['worker_id']], step)
                        record = {'step':step,'path':str(path),'receipt':rows[source['worker_id']]['receipt']}
                        break
                    time.sleep(self.api._control_poll_seconds(self.spec))
        inspect = self.send(current,'inspect',expected_step=step)
        current, rows = self.wait(current,inspect,'inspected')
        self._state_hashes(rows, step)
        if record and any(record['receipt'][k]!=rows[current[0]['worker_id']][k] for k in ('model_sha256','optimizer_sha256')):
            raise PodRecoveryRequired('Checkpoint and committed boundary state differ')
        if not self.same(current,group['members']):
            current,group=self.rebuild(current,group,step,preserve_gradients=False)
        self.members=current
        return current,group,record,{'step':step,'token':inspect,'members':rows,'full_states_identical':True}

    def finalize(self, members, group, step, *, checkpoint):
        """Export from a surviving source; retry another if its pod disappears.

        All parameters and optimizer states are already at a committed boundary.
        No further optimizer command is permitted in this method.
        """
        self.deadline = time.monotonic() + self.spec['phase_timeout'] * 4
        current = self.live(members)
        if not self.same(current, group['members']):
            current, group = self.rebuild(current, group, step, preserve_gradients=False)
        checkpoint_record = None
        if checkpoint:
            while True:
                current = self.live(current)
                source = current[0]
                path = self.run/'states'/f'final-{step}-pod-{uuid.uuid4().hex}'
                metadata = self.api._boundary_metadata(self.spec, step)
                token = self.send([source], 'export', expected_step=step, path=str(path), state_metadata=metadata)
                self._persist('exporting_checkpoint', completed_step=step, source=self.api._identity(source))
                while True:
                    self._remaining()
                    current = self.live(current)
                    if not any(self.api._same_identity(source, m) for m in current):
                        break
                    rows = self.api._reports(self.run, self.spec, [source], token, 'exported')
                    if rows:
                        self._validate_export(rows[source['worker_id']], step)
                        checkpoint_record = {'path':str(path), 'receipt':rows[source['worker_id']]['receipt']}
                        break
                    time.sleep(self.api._control_poll_seconds(self.spec))
                if checkpoint_record is not None:
                    break
        self.members = current
        self._persist('verifying_final_state', completed_step=step)
        inspect = self.send(current, 'inspect', expected_step=step)
        current, rows = self.wait(current, inspect, 'inspected')
        hashes = self._state_hashes(rows, step)
        if checkpoint_record and any(checkpoint_record['receipt'][field] != rows[current[0]['worker_id']][field]
                                      for field in ('model_sha256', 'optimizer_sha256')):
            raise PodRecoveryRequired('Final checkpoint differs from the surviving resident model/Adam')
        if not self.same(current, group['members']):
            current, group = self.rebuild(current, group, step, preserve_gradients=False)
        finish = self.send(current, 'finish', expected_step=step)
        current, finished = self.wait(current, finish, 'finished')
        if self._state_hashes(finished, step) != hashes:
            raise PodRecoveryRequired('Final finish state differs from the verified checkpoint boundary')
        self.members = current
        self._persist('finished', completed_step=step)
        return current, checkpoint_record, finished
