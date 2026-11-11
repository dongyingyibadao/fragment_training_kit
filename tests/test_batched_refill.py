"""Stateful serial-worker tests for exact-16 COMMIT admission, no devices."""
from copy import deepcopy
import hashlib
import time

import pytest

from fragment_training_kit import launch
from fragment_training_kit.tests.test_pod_controller_runtime import Actors, assert_commit
from jade_lotus.pod_controller_runtime import PodController, PodRecoveryRequired
from jade_lotus.refill_controller import RefillController, verified_reservation_devices
from jade_lotus.refill_protocol import mapping_digest
from jade_lotus.refill_state import cancel, load_state, save_state
from fragment_training_kit.tests.test_topology_cohort_runtime import controller_liveness


class RefillActors(Actors):
    def ControllerLiveness(self, run, spec, **kwargs):
        watch = super().ControllerLiveness(run, spec)
        watch.used_tokens, watch.seen_exits, watch._persisted_losses = set(), set(), {}
        watch.generation_floor = kwargs.get('generation_floor', 0)
        watch.detector.phase, watch.detector.epoch = 'idle', kwargs.get('epoch', 0)
        watch.detector.fenced_sessions.update(kwargs.get('persisted_fences', ()))
        return watch

    def _send(self, run, spec, members, action, **fields):
        if action not in {'load', 'hold_committed_boundary', 'release_committed_boundary'}:
            return super()._send(run, spec, members, action, **fields)
        self.sequence += 1
        token = f'command-{self.sequence}'
        self.command_counts[action] += 1
        self.log.append(('send', action, deepcopy(fields), [m['worker_id'] for m in members]))
        for member in members:
            worker, actor = member['worker_id'], self.actors[member['worker_id']]
            assert actor['alive'] and actor['units'] is None and actor['reducing'] is None
            if action == 'load':
                assert actor['group'] is None and actor['step'] <= fields['expected_step']
                actor['step'] = fields['expected_step']
                self.record(worker, token, 'loaded', completed_step=actor['step'], receipt=self.native_receipt(actor['step']))
            else:
                assert actor['group'] is not None and actor['step'] == fields['expected_step']
                self.record(worker, token, 'boundary_held' if action.startswith('hold') else 'boundary_released',
                            completed_step=actor['step'])
        return token


@pytest.fixture
def refill(tmp_path):
    root = tmp_path / 'run'
    run = root / 'probe'
    run.mkdir(parents=True)
    (root / 'training.json').write_text('{}')
    api = RefillActors(run, count=18)
    api.spec['config_sha256'] = hashlib.sha256((root / 'training.json').read_bytes()).hexdigest()
    api.spec['heartbeat_timeout'] = 60
    active, candidates = api.members[:2], api.members[2:]
    group = api._group(active, 0)
    for member in active:
        api.actors[member['worker_id']]['group'] = deepcopy(group)
    for member in candidates:
        api.actors[member['worker_id']]['group'] = None
    pod = PodController(api, run, api.spec, active, {})
    config = {'resident_refill_target_cards': 18, 'resident_refill_max_batches': 8,
              'resident_refill_candidate_timeout': 300}
    manager = RefillController(pod, config)
    batch = {'batch_id': 'b000001', 'phase': 'loading', 'created_at': time.time(), 'allocations': ['w000002'],
             'member_ids': [m['worker_id'] for m in candidates], 'wrapper_sessions': {'w000002': 'wrapper-1'}}
    mapping = {'schema': 'fragment-refill-allocation-v1', 'run_dir': str(root),
               'config_sha256': api.spec['config_sha256'], 'worker_prefix': 'w000002', 'intent_id': 'intent-2',
               'role': 'candidate', 'batch_id': batch['batch_id'],
               'members': [{'worker_id': m['worker_id'], 'device_index': i} for i, m in enumerate(candidates)]}
    launch.write(root / 'refill/allocations/w000002.json', mapping)
    launch.write(root / 'refill/reservations/w000002.json', {'wrapper_session': 'wrapper-1', 'updated_at': time.time()})
    for i, member in enumerate(candidates):
        ready = {**member, 'run_id': api.spec['run_id'], 'allocation_group': 'w000002', 'device_index': i,
                 'refill_wrapper_session': 'wrapper-1', 'refill_batch_id': batch['batch_id'], 'replacement': True}
        api.files[run / 'ready' / f"{member['worker_id']}.json"] = ready
        api.files[run / 'supervisors' / f"{member['worker_id']}.json"] = {**ready, 'heartbeat_counter': 1}
    state = load_state(root, api.spec['config_sha256'])
    state.update(current=batch, next_batch=2)
    save_state(root, state)
    return api, pod, manager, active, candidates, group, batch, mapping


def train_and_admit(fixture):
    api, pod, manager, active, _, group, batch, _ = fixture
    members, group, commit = pod.run_step(active, group, 0)
    assert_commit(api, members, commit, 0)
    return manager.on_commit(members, group, 1)


def test_exact16_native_sync_then_next_step_applies_once(refill):
    api, pod, manager, active, candidates, _, batch, _ = refill
    members, group = train_and_admit(refill)
    assert len(members) == 18 and api.command_counts['load'] == 1
    assert all(api.actors[m['worker_id']]['step'] == 1 for m in members)
    assert len(api.applications) == 2  # Loading/rebuild never invokes Adam.
    state = load_state(manager.root, manager.sha)
    assert state['current']['phase'] == 'admitted'
    assert state['current']['admission']['candidate_count'] == 16
    members, group, commit = pod.run_step(members, group, 1)
    assert_commit(api, members, commit, 1)
    assert set(api.applications.values()) == {1}


def test_bad_candidate_complete_adam_state_is_rejected_before_reset(refill):
    api, pod, manager, active, candidates, group, _, _ = refill
    api.mutations[candidates[-1]['worker_id'], 'inspected'] = {'optimizer_sha256': 'e' * 64}
    members, current = train_and_admit(refill)
    assert [m['worker_id'] for m in members] == [m['worker_id'] for m in active]
    assert current == group and api.command_counts['pod_reset'] == 0
    assert api.command_counts['release_committed_boundary'] == 1
    assert load_state(manager.root, manager.sha)['current']['phase'] == 'cancelled'
    members, _, commit = pod.run_step(members, current, 1)
    assert_commit(api, members, commit, 1)


def test_candidate_loss_during_init_withdraws_whole_batch(refill):
    api, pod, manager, active, candidates, _, _, _ = refill
    api.lose_on('pod_init', 1, [candidates[-1]['worker_id']])
    members, group = train_and_admit(refill)
    assert len(members) == 2  # Never quietly admit the remaining 15.
    evidence = load_state(manager.root, manager.sha)['current']['rejection']
    assert evidence['reset_started'] and evidence['active_group_continued']
    assert api.command_counts['pod_apply'] == 1
    members, _, commit = pod.run_step(members, group, 1)
    assert_commit(api, members, commit, 1)


def test_candidate_load_error_leaves_active_training_untouched(refill):
    api, pod, manager, active, candidates, group, _, _ = refill
    member = candidates[-1]
    api.files[manager.run / 'errors' / f"{member['worker_id']}-{member['session_id']}.json"] = {'error': 'loader'}
    members, current = train_and_admit(refill)
    assert members == active and current == group
    assert api.command_counts['load'] == 0 and api.command_counts['pod_reset'] == 0
    assert load_state(manager.root, manager.sha)['current']['phase'] == 'cancelled'


def test_active_native_state_disagreement_is_never_hidden_as_candidate_failure(refill):
    api, _, _, _, _, _, _, _ = refill
    api.mutations['w000001', 'inspected'] = {'optimizer_sha256': 'e' * 64}
    with pytest.raises(PodRecoveryRequired, match='disagree'):
        train_and_admit(refill)


def test_new_wrapper_cannot_reuse_old_batch_permit(refill):
    api, _, manager, _, _, _, _, _ = refill
    launch.write(manager.root / 'refill/reservations/w000002.json', {
        'wrapper_session': 'restarted-wrapper', 'updated_at': time.time()})
    members, _ = train_and_admit(refill)
    assert len(members) == 2 and api.command_counts['load'] == 0
    assert load_state(manager.root, manager.sha)['current']['phase'] == 'cancelled'


@pytest.mark.parametrize('count', [15, 16])
def test_reserve_threshold_and_exact_incarnation_load_authority(refill, count):
    api, _, manager, active, candidates, group, batch, mapping = refill
    sizes = [8, 4, 2, 1] if count == 15 else [16]
    batch.update(phase='reserving', allocations=[], official={}, official_at=time.time())
    offset = 0
    for number, size in enumerate(sizes, start=2):
        prefix = f'w{number:06d}'
        value = {**mapping, 'worker_prefix': prefix, 'intent_id': 'intent-' + prefix,
                 'members': [{'worker_id': m['worker_id'], 'device_index': i}
                             for i, m in enumerate(candidates[offset:offset + size]) ]}
        offset += size
        launch.write(manager.root / 'refill/allocations' / f'{prefix}.json', value)
        launch.write(manager.root / 'refill/reservations' / f'{prefix}.json', {
            **value, 'schema': 'fragment-refill-reservation-v1', 'mapping_sha256': mapping_digest(value),
            'wrapper_session': 'wrapper-' + prefix, 'updated_at': time.time(), 'state': 'reserved',
            'heartbeat_seq': 1, 'pid': 1000 + number, 'hostname': 'fixture-host'})
        batch['allocations'].append(prefix)
        batch['official'][prefix] = {'status': 'job_running', 'priority_verified': True,
            'allocation_verified': True, 'allocated_cards': size, 'observed_at': time.time()}
    state = load_state(manager.root, manager.sha)
    state['current'] = batch
    save_state(manager.root, state)
    manager.on_commit(active, group, 0)
    permits = list((manager.root / 'refill/permits').glob('*.json'))
    assert len(permits) == (0 if count == 15 else 1)
    assert api.command_counts['load'] == 0
    assert load_state(manager.root, manager.sha)['current']['phase'] == ('reserving' if count == 15 else 'loading')


def test_admitted_batch_cannot_be_cancelled(refill):
    _, _, manager, _, _, _, _, _ = refill
    train_and_admit(refill)
    state = load_state(manager.root, manager.sha)
    with pytest.raises(ValueError, match='admitted'):
        cancel(manager.root, state, state['current'], 'raced cancellation')
    assert not (manager.root / 'refill/cancelled/w000002.json').exists()


def test_real_pod_commit_compacts_after_shrink_without_legacy_finish_step(controller_liveness):
    module, watch, members = controller_liveness
    run = watch.run
    spec = {'run_id': 'liveness-run', 'config_sha256': 'c'*64, 'heartbeat_timeout': 3600}
    pod = PodController(module, run, spec, members[:4], {})
    lost = members[3]
    own = module._read(run/'supervisors'/f"{lost['worker_id']}.json")
    module._write(run/'exits'/f"{lost['worker_id']}-{lost['session_id']}.json", {**own, 'exit_code': 137})
    alive = pod.live()
    assert len(alive) == 3 and pod.watch.last_committed_step is None
    pod.inflight, pod.committed_step = False, 1
    group = {'members': alive, 'generation': 1}
    fences = set(pod.watch.detector.fenced_sessions)
    pod.compact_liveness(alive, group, 1)
    assert set(pod.watch.keys) == {m['worker_id'] for m in alive}
    assert set(pod.watch.detector.fenced_sessions) == fences
    assert pod.watch.last_committed_step == 1
    assert len(pod.live()) == 3
    with pytest.raises(Exception):
        pod.watch.register_ready([lost], receipt='cannot-resurrect-fenced-session')


@pytest.mark.parametrize('inflight,step', [(True, 1), (False, 2)])
def test_pod_compaction_rejects_pending_or_unverified_boundary(refill, inflight, step):
    _, pod, _, active, _, group, _, _ = refill
    pod.inflight, pod.committed_step = inflight, 1
    with pytest.raises(PodRecoveryRequired, match='COMMIT'):
        pod.compact_liveness(active, group, step)


def test_partial_candidate_registration_failure_is_isolated(refill):
    api, pod, manager, active, candidates, group, _, _ = refill
    register = pod.watch.register_ready
    def changed(members, *, receipt):
        register(members[:-1], receipt=receipt)
        raise ValueError('Last candidate restarted between validation and registration')
    pod.watch.register_ready = changed
    members, current = train_and_admit(refill)
    assert members == active and current == group
    assert set(pod.watch.keys) == {m['worker_id'] for m in active}
    assert load_state(manager.root, manager.sha)['current']['phase'] == 'cancelled'


def test_stale_official_allocation_is_not_refreshed_by_a_new_wrapper_record(refill):
    _, _, manager, _, _, _, batch, mapping = refill
    batch.update(phase='reserving', official_at=time.time(), official={'w000002': {
        'status': 'job_running', 'priority_verified': True, 'allocation_verified': True,
        'allocated_cards': 16, 'observed_at': time.time()-1000}})
    launch.write(manager.root/'refill/reservations/w000002.json', {
        **mapping, 'schema': 'fragment-refill-reservation-v1', 'mapping_sha256': mapping_digest(mapping),
        'wrapper_session': 'wrapper-1', 'pid': 1000, 'hostname': 'host', 'heartbeat_seq': 1,
        'updated_at': time.time(), 'state': 'reserved'})
    assert manager._reservations(batch, [mapping]) is None


@pytest.mark.parametrize('official_count,host,expected', [(None, 'host', True), (None, 'old-host', False),
                                                         (8, 'host', False)])
def test_qz_missing_device_ids_requires_independent_container_assignment(refill, official_count, host, expected):
    *_, mapping = refill
    reservation = {'hostname': 'host', 'device_visibility': {'verified': True, 'kind': 'assigned_environment_mask',
                   'source': 'ASCEND_RT_VISIBLE_DEVICES', 'count': 16, 'devices': list(range(16))}}
    official = {'allocation_verified': False, 'allocated_cards': official_count, 'running_instance_names': [host]}
    assert verified_reservation_devices(mapping, reservation, official) is expected
