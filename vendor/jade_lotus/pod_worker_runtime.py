"""Split device phases for recoverable multi-member preemption.

The parent command loop owns exactly one child/session and executes these
commands serially. No optimizer write occurs before pod_apply. Communication
buffers never alias the original accumulated per-sample gradients.
"""
from __future__ import annotations

import time


class PodWorker:
    def __init__(self, api, run, spec, member, engine, optimizer, progress, prefetch=None):
        self.api, self.run, self.spec, self.member = api, run, spec, member
        self.engine, self.optimizer, self.progress, self.prefetch = engine, optimizer, progress, prefetch
        self.local, self.buffers, self.base_step = None, None, None
        self.root_token, self.contributions, self.reduction_token = None, [], None
        self.timings = {}

    def record(self, command, phase, **fields):
        return self.api._record(self.run, self.spec, self.member, command['token'], phase, **fields)

    def _discard_group(self, dist, torch_npu, group, command):
        if dist.is_initialized():
            backend = dist.distributed_c10d._get_default_group()._get_backend(self.engine.torch_device)
            self.api._cleanup_backend(backend, torch_npu, self.engine.torch_device.index,
                lambda rows: self.record(command, 'pod_cleanup', operations=rows))
            dist.destroy_process_group()
        return None

    def execute(self, command, group, completed_step):
        import torch.distributed as dist
        import torch_npu
        from jade_lotus.coordinator import learning_rate
        api, engine, optimizer, member = self.api, self.engine, self.optimizer, self.member
        action, step = command['action'], command['expected_step']
        if type(step) is not int or step != completed_step:
            raise ValueError('Pod command does not name the resident completed step')
        started = time.monotonic()
        self.progress(action)
        self._test_gate(command)
        if action == 'pod_compute':
            if group != command['group'] or self.local is not None:
                raise ValueError('Compute requires the exact initialized group and a fresh update')
            self.base_step, self.root_token = step, command['root_token']
            units = command['assignments'][member['worker_id']]
            from contextlib import nullcontext
            context = nullcontext()
            if self.prefetch is not None:
                from jade_lotus.resident_data import use_cached_samples
                cache, _ = self.prefetch.acquire(step, units, units)
                context = use_cached_samples(engine, cache)
            with context:
                self.local = api.backward_units(engine, base_step=step, units=units, clear=True)
            self.contributions = [{'token': command['token'], **self.local}]
            self.buffers, self.reduction_token = None, None
            self.timings = {'data_seconds': self.local['data_seconds'],
                'backward_seconds': self.local['backward_seconds'], 'communication_seconds': 0.,
                'gradient_buffer_pack_seconds': 0., 'state_preservation_hash_seconds': 0.}
            self.record(command, 'pod_local', completed_step=step, local=self.local,
                        complete_local_backward=True, generation=group['generation'])
        elif action == 'pod_replay':
            self._require_pending(command, step)
            units = command['replay'][member['worker_id']]
            if set(units).intersection(self.local['units']):
                raise ValueError('Replay would duplicate an already accumulated atomic unit')
            extra = api.backward_units(engine, base_step=step, units=units, clear=False)
            self.contributions.append({'token': command['token'], **extra})
            self.local = {**self.local, 'units': self.local['units'] + list(units),
                **{key: self.local[key] + extra[key] for key in
                   ('samples', 'loss_contribution', 'data_seconds', 'backward_seconds', 'seconds', 'backward_calls')}}
            for key in ('data_seconds', 'backward_seconds'):
                self.timings[key] += extra[key]
            self.buffers, self.reduction_token = None, None
            self.record(command, 'pod_local', completed_step=step, complete_local_backward=True,
                        local=self.local, replay=extra)
        elif action == 'pod_reset':
            preserve = command['preserve_gradients']
            pointers = api._pointer_snapshot(engine, optimizer, gradients=preserve)
            audit = self.spec.get('audit_before_collective', False)
            before = api._fingerprint(engine, optimizer, step=step, gradients=preserve) if audit else None
            group = self._discard_group(dist, torch_npu, group, command)
            if pointers != api._pointer_snapshot(engine, optimizer, gradients=preserve):
                raise AssertionError('Group reset changed resident storage')
            after = api._fingerprint(engine, optimizer, step=step, gradients=preserve) if audit else None
            if audit and before != after:
                raise AssertionError('Group reset changed model/Adam/raw gradient bytes')
            self.buffers, self.reduction_token = None, None
            self.record(command, 'pod_reset_done', completed_step=step, next_port=api._free_port(),
                        state_storage_pointers_unchanged=True, full_byte_audit_enabled=audit,
                        full_native_state_and_gradient_unchanged=bool(audit))
        elif action == 'pod_init':
            if group is not None:
                raise ValueError('Recovery init requires prior group cleanup')
            candidate = command['group']
            try:
                api._init_group(dist, candidate, member, self.spec['collective_timeout'])
            except Exception as error:
                group = self._discard_group(dist, torch_npu, candidate, command)
                self.record(command, 'pod_initialized', completed_step=step, status='failed',
                            generation=candidate['generation'], error=f'{type(error).__name__}: {error}')
            else:
                group = candidate
                self.record(command, 'pod_initialized', completed_step=step, status='ready',
                            generation=group['generation'])
        elif action == 'pod_reduce':
            self._require_pending(command, step)
            if group != command['group'] or sorted(self.local['units']) != sorted(command['assignments'][member['worker_id']]):
                raise ValueError('Reduction does not match resident gradient ownership/group')
            pointers = api._pointer_snapshot(engine, optimizer, gradients=True)
            audit = self.spec.get('audit_before_collective', False)
            before = api._fingerprint(engine, optimizer, step=step, gradients=True) if audit else None
            pack = time.monotonic()
            self.buffers = api.reduction_buffers(engine.parameters, engine.torch)
            engine._sync()
            self.timings['gradient_buffer_pack_seconds'] += time.monotonic() - pack
            backend = dist.distributed_c10d._get_default_group()._get_backend(engine.torch_device)
            fence = {'run_id': self.spec['run_id'], 'token': command['token'],
                     'base_step': step, 'generation': group['generation']}
            def initialize():
                engine.torch.npu.set_device(engine.torch_device)
                engine._sync()
            monitor = api.FaultMonitor(self.run/'abort'/f"{command['token']}.json", fence, member,
                lambda: api._cleanup_backend(backend, torch_npu, engine.torch_device.index,
                    lambda rows: self.record(command, 'pod_cleanup', operations=rows)), initialize=initialize).start()
            error = None
            try:
                monitor.wait_ready(self.spec['phase_timeout'])
                monitor.armed.set()
                works = []
                began = time.monotonic()
                try:
                    api._submit(dist, self.buffers, self.spec['bucket_bytes'], works=works,
                        cancelled=lambda: monitor.triggered, submission_lock=monitor.submission_lock)
                    self.record(command, 'pod_submitted', base_step=step, generation=group['generation'],
                                collectives=len(works))
                    api._wait_works(works, engine, self.spec['collective_timeout'], cancelled=lambda: monitor.triggered)
                except Exception as failure:
                    error = f'{type(failure).__name__}: {failure}'
                    self.record(command, 'pod_collective_failed', error=error, base_step=step,
                                generation=group['generation'])
                    if not monitor.done.wait(self.spec['heartbeat_timeout'] + 10) or not monitor.triggered:
                        raise
                self.timings['communication_seconds'] += time.monotonic() - began
                if not monitor.triggered:
                    # Stay armed until an explicit release, even after HCCL
                    # succeeds: a late peer loss must not overwrite raw grads.
                    self.record(command, 'pod_reduced', base_step=step, generation=group['generation'],
                                status='reduced', local=self.local, contributions=self.contributions)
                    def decision():
                        if monitor.triggered:
                            return 'abort'
                        value = api._read(self.run/'pod'/'reduce_release'/f"{command['token']}.json")
                        if (value and all(value.get(k)==v for k,v in fence.items())
                                and value.get('action')=='release_reduction'):
                            return 'release'
                        return None
                    choice = api._wait(decision, self.spec['phase_timeout'], 'reduce release or loss',
                                       interval=api._control_poll_seconds(self.spec))
                else:
                    choice = 'abort'
                if choice == 'abort':
                    monitor.finish(self.spec['phase_timeout'])
                    monitor.close()
                    dist.destroy_process_group()
                    group = None
                    if api._pointer_snapshot(engine, optimizer, gradients=True) != pointers:
                        raise AssertionError('Aborted reduction changed original gradient storage')
                    if audit and api._fingerprint(engine, optimizer, step=step, gradients=True) != before:
                        raise AssertionError('Aborted reduction changed original gradient or native state')
                    self.buffers, self.reduction_token = None, None
                    self.record(command, 'pod_aborted', completed_step=step,
                        state_storage_pointers_unchanged=True, full_byte_audit_enabled=audit,
                        full_native_state_and_gradient_unchanged=bool(audit), next_port=api._free_port())
                else:
                    monitor.close()
                    if monitor.triggered:
                        raise RuntimeError('Abort raced a reduction release')
                    self.reduction_token = command['token']
                    self.record(command, 'pod_released', completed_step=step, generation=group['generation'])
            finally:
                monitor.close()
        elif action == 'pod_apply':
            self._require_pending(command, step)
            if (group != command['group'] or self.buffers is None
                    or command['reduction_token'] != self.reduction_token):
                raise ValueError('APPLY requires this exact successfully released reduction')
            api.install_reduced_gradients(self.buffers)
            norm = engine.torch.nn.utils.clip_grad_norm_(list(engine.parameters.values()),
                float(engine.config.get('clip_gradient_norm', 1.)), foreach=False)
            if not bool(engine.torch.isfinite(norm).item()):
                raise FloatingPointError('Nonfinite reduced gradient norm')
            for options in optimizer.param_groups:
                options['lr'] = learning_rate(engine.config, step)
            optimizer.step()
            engine._sync()
            optimizer.zero_grad(set_to_none=True)
            completed_step += 1
            self.timings['install_clip_optimizer_seconds'] = time.monotonic() - started
            self.record(command, 'updated', generation=group['generation'], completed_step=completed_step,
                root_token=self.root_token, reduction_token=self.reduction_token, optimizer_applications=1,
                grad_norm=float(norm.item()), local=self.local, contributions=self.contributions,
                checkpoint_calls=engine.checkpoint_call_count, timings=self.timings)
            self.local, self.buffers, self.reduction_token = None, None, None
        else:
            raise ValueError(f'Unknown pod action {action}')
        return group, completed_step

    def _require_pending(self, command, step):
        if self.local is None or self.base_step != step or command['root_token'] != self.root_token:
            raise ValueError('Command belongs to a different/unprepared update')

    def _test_gate(self, command):
        """Opt-in external fault-injector gate; bounded and exact-session fenced."""
        if not self.spec.get('pod_allow_test_holds', False):
            return
        path = self.run/'pod'/'test_holds'/f"{self.member['worker_id']}.json"
        hold = self.api._read(path)
        if not (hold and hold.get('run_id') == self.spec['run_id']
                and self.api._same_identity(hold.get('recipient'), self.member)
                and hold.get('action') == command['action']
                and hold.get('expected_step') == command['expected_step']):
            return
        self.progress('pod_test_hold')
        self.record(command, 'pod_test_held', action=command['action'],
                    expected_step=command['expected_step'], independent_fault_gate=True)
        deadline = time.monotonic() + self.spec['phase_timeout']
        while time.monotonic() < deadline:
            release = self.api._read(path)
            if (release and release.get('release_token') == command['token']
                    and release.get('run_id') == self.spec['run_id']
                    and self.api._same_identity(release.get('recipient'), self.member)):
                self.progress(command['action'])
                return
            time.sleep(.02)
        raise TimeoutError('Independent test hold expired; no silently continued experiment')
