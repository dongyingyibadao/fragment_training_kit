"""Allocate optional refill intents under the fleet's shared physical budget.

The controller alone issues load/admission/cancellation authority. This module
never infers a released card from a lost training process or stops an active pod.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time
import uuid

from .ascend_fleet import active, cards, IntentDeferred
from .ascend_platform import RUNNING
from .refill_protocol import atomic_write, read_allocation, validate_cancel
from .refill_state import BATCH_SIZE, immutable, load_state, read, save_state
from .runtime import writer_lock, WriterLockBusy


class RefillFleet:
    def __init__(self, client, run, site, *, initial_planner):
        self.client, self.run, self.site = client, Path(run).resolve(), site
        self.initial_planner = initial_planner
        self.config_sha = hashlib.sha256((self.run / 'training.json').read_bytes()).hexdigest()
        self.selected_batch = None

    def restore(self, fleet):
        seen = set()
        for entry in fleet['jobs']:
            mapping = read_allocation(self.run, entry['worker_id'], cards(entry['payload']))
            members = [m['worker_id'] for m in mapping['members']]
            if seen.intersection(members):
                raise ValueError('Durable allocations reuse a historical training identity')
            seen.update(members)
            entry.update(refill_role=mapping['role'], refill_batch_id=mapping['batch_id'],
                         intent_id=mapping['intent_id'], member_ids=members)
        fleet['next_member'] = max(fleet.get('next_member', 1), max((int(m[1:]) + 1 for m in seen), default=1))
        fleet['next_worker'] = max(fleet.get('next_worker', 1),
                                 max((int(e['worker_id'][1:]) + 1 for e in fleet['jobs']), default=1))

    def retire(self, entry):
        if entry.get('refill_role') != 'candidate':
            return False
        prefix = entry['worker_id']
        record = read(self.run / 'refill/cancelled' / f'{prefix}.json')
        if record is None:
            return False
        mapping = read_allocation(self.run, prefix, cards(entry['payload']))
        # A controller-only immutable cancellation can never coexist with an
        # admitted authorization for the same batch.
        return validate_cancel(mapping, record)

    def plan(self, fleet, inventory, reserved, available, backoff, now):
        if len(fleet['jobs']) < len(self.site['job_layout']):
            return self.initial_planner(inventory, reserved, available, backoff, now)
        try:
            with writer_lock(self.run / 'refill/lock'):
                state = load_state(self.run, self.config_sha)
                batch = state['current']
                if not batch or batch['phase'] not in {'reserving', 'loading'}:
                    return []
                entries = {e['worker_id']: e for e in fleet['jobs']}
                observed = {}
                unfinished = None
                for prefix in batch['allocations']:
                    entry = entries.get(prefix)
                    if entry is None:
                        unfinished = read(self.run / 'refill/allocations' / f'{prefix}.json')
                        continue
                    observation = entry.get('last_observation', {})
                    observed[prefix] = {'status': entry['status'], 'official_terminal': entry.get('official_terminal', False),
                        'job_id': entry.get('job_id'), 'requested_cards': cards(entry['payload']),
                        'observed_at': observation.get('time', now if observation.get('snapshot_id') else 0),
                        'priority_verified': observation.get('priority_verified', False),
                        'allocation_verified': observation.get('allocation_verified', False),
                        'allocated_cards': observation.get('allocated_cards'),
                        'running_instance_names': [i.get('name') for i in observation.get('instances', [])
                            if i.get('instance_status') == 'instance_running' and i.get('name')],
                        'running_round': observation.get('running_round')}
                    if not active(entry):
                        batch['failure'] = 'Candidate job ended before admission'
                    if entry.get('refill_priority_error'):
                        batch['failure'] = 'Candidate allocation failed verified LOW priority'
                batch['official'] = observed
                batch['official_at'] = now
                save_state(self.run, state)
                if batch['phase'] != 'reserving' or batch.get('failure') or not available:
                    return []
                # The immutable map may precede the fleet intent by a local
                # crash. No POST is possible before that intent is durable;
                # replay exactly its original mapping and selection.
                next_prefix = f"w{fleet['next_worker']:06d}"
                unfinished = unfinished or read(self.run / 'refill/allocations' / f'{next_prefix}.json')
                if unfinished:
                    if (unfinished['worker_prefix'] != next_prefix or unfinished['batch_id'] != batch['batch_id']
                            or unfinished['role'] != 'candidate'):
                        raise ValueError('Unfinished mapping does not match the next durable job identity')
                    size = len(unfinished['members'])
                    if size > available:
                        return []
                    self.selected_batch = batch['batch_id']
                    return [{'selection': unfinished['placement_selection'], 'cards': size,
                             'reason': unfinished['placement_reason']}]
                count = sum(cards(entries[p]['payload']) for p in batch['allocations'] if p in entries)
                room = min(available, BATCH_SIZE - count)
                if room <= 0:
                    return []
                if len(fleet['jobs']) >= self.site['training_config']['resident_refill_max_intents']:
                    batch['failure'] = 'Configured bounded intent history exhausted'
                    save_state(self.run, state)
                    return []
                from .ascend_nodes import discover_nodes
                from .topology_placement import plan_topology, reservation_ledger
                nodes = discover_nodes(self.client, inventory)
                used = sum(r['cards'] for r in reservation_ledger(reserved))
                planned = plan_topology(inventory, nodes, reserved,
                    global_card_limit=min(64, used + room), local_card_limit=min(64, used + room),
                    group_backoff=backoff, now=time.time(), queue_when_idle_exhausted=True)
                placements = planned['placements']
                atomic_write(self.run / 'refill/placement-latest.json', {
                    'at': time.time(), 'batch_id': batch['batch_id'], 'available_cards': available,
                    'batch_room': room, 'placements': placements, 'node_snapshot': nodes,
                    'planned_nodes_are_not_allocations': True})
                self.selected_batch = batch['batch_id']
                return placements
        except WriterLockBusy:
            return []  # Optional bookkeeping must not hold up active training.
        except Exception as error:
            atomic_write(self.run / 'refill/planning-error.json', {
                'at': time.time(), 'error': f'{type(error).__name__}: {error}', 'training_stop_requested': False})
            return []

    def intent(self, fleet, entry, placement):
        try:
            return self._intent(fleet, entry, placement)
        except WriterLockBusy as error:
            raise IntentDeferred('Candidate authority is busy; retry without submitting') from error

    def _intent(self, fleet, entry, placement):
        prefix, size = entry['worker_id'], cards(entry['payload'])
        initial = len(fleet['jobs']) < len(self.site['job_layout'])
        with writer_lock(self.run / 'refill/lock'):
            state = load_state(self.run, self.config_sha)
            batch = state['current']
            if not initial and (not batch or batch['batch_id'] != self.selected_batch or batch['phase'] != 'reserving'):
                raise IntentDeferred('Candidate allocation raced its batch authority')
            path = self.run / 'refill/allocations' / f'{prefix}.json'
            existing = read(path)
            start = fleet.get('next_member', 1)
            value = existing or {'schema': 'fragment-refill-allocation-v1',
                'run_dir': str(self.run), 'config_sha256': self.config_sha,
                'worker_prefix': prefix, 'intent_id': uuid.uuid4().hex,
                'role': 'initial' if initial else 'candidate',
                'batch_id': None if initial else batch['batch_id'],
                'placement_selection': entry['selection'], 'placement_reason': placement['reason'],
                'members': [{'worker_id': f'w{start + i:06d}', 'device_index': i} for i in range(size)]}
            if (len(value['members']) != size or value['role'] != ('initial' if initial else 'candidate')
                    or (not initial and value['batch_id'] != batch['batch_id'])):
                raise ValueError('Unfinished intent cannot be reused for a different allocation')
            immutable(path, value)
            read_allocation(self.run, prefix, size)
            fleet['next_member'] = max(start, max(int(m['worker_id'][1:]) for m in value['members']) + 1)
            entry.update(refill_role=value['role'], refill_batch_id=value['batch_id'], intent_id=value['intent_id'],
                         member_ids=[m['worker_id'] for m in value['members']])
            if not initial:
                if prefix not in batch['allocations']:
                    batch['allocations'].append(prefix)
                save_state(self.run, state)
