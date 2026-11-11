"""Explicit refill configuration freezes a finite initial layout, no jobs."""
import json
import sys

import pytest

from fragment_training_kit import launch
from fragment_training_kit.tests.test_topology_placement import selection


@pytest.fixture
def site(tmp_path):
    cann = tmp_path/'cann.sh'
    cann.write_text('# source fixture\n')
    inventory = tmp_path/'inventory.json'
    launch.write(inventory, {'backend':'ascend', 'user':{'id':'fixture'},
        'selections':[selection(n) for n in (1,2,4,8,16)]})
    value = {'schema':launch.SCHEMA, 'python':sys.executable, 'cann_env':str(cann),
        'inventory':str(inventory), 'registry':str(tmp_path/'registry.json'),
        'cards':64, 'active_cards':48, 'job_layout':[16]*3, 'steps':20,
        'training_config':{'resident_pod_recovery':True, 'resident_pod_refill':True}}
    path = tmp_path/'site-config.json'
    launch.write(path, value)
    return path, value


def test_refill_freezes_only_initial_layout_but_retains_fallback_specs(site, tmp_path):
    path, value = site
    run = tmp_path/'run'
    result = launch.prepare(path, run)
    assert result['arbitrary_fleet_refill_joins_supported'] and result['submitted'] is False
    assert result['job_layout'] == [16]*3
    assert {s['resource_spec_price']['gpu_count'] for s in launch.read(run/'inventory.json')['selections']} == {1,2,4,8,16}
    config = launch.read(run/'training.json')
    assert config['resident_refill_target_cards'] == 64 and 'resident_topology_slots' not in config
    assert '{npu_count}' in (run/'fleet_command.txt').read_text()
    assert launch.verify(run)['verified']


def test_refill_always_uses_group_wrapper_even_for_single_card_initial_jobs(site, tmp_path):
    path, value = site
    value.update(active_cards=64,job_layout=[1]*64)
    launch.write(path, value)
    run = tmp_path/'run'
    launch.prepare(path, run)
    assert 'group_worker.py' in (run/'launch.sh').read_text()
    assert '{npu_count}' in (run/'fleet_command.txt').read_text()


@pytest.mark.parametrize('mutation', ['no_recovery','wrong_layout','legacy_slots','legacy_defer','invalid_bool'])
def test_unsupported_config_is_rejected_before_run_creation(site, tmp_path, mutation):
    path, value = site
    if mutation == 'no_recovery':
        value['training_config']['resident_pod_recovery'] = False
    elif mutation == 'wrong_layout':
        value['job_layout'] = [16]*4
    elif mutation == 'legacy_slots':
        value['training_config']['resident_topology_slots'] = 64
    elif mutation == 'legacy_defer':
        value['defer_standby_until_idle'] = True
    else:
        value['training_config']['resident_pod_refill'] = 'true'
    launch.write(path, value)
    with pytest.raises(ValueError):
        launch.prepare(path, tmp_path/'must-not-exist')
    assert not (tmp_path/'must-not-exist').exists()


def test_refill_preserves_explicit_resource_group_scope(site, tmp_path):
    path, value = site
    inventory = launch.read(value['inventory'])
    inventory['selections'].append(selection(16, group='unselected-group'))
    launch.write(value['inventory'], inventory)
    value['allowed_group_ids'] = ['lcg-fixture']
    launch.write(path, value)
    run = tmp_path/'run'
    launch.prepare(path, run)
    assert {s['logic_compute_group_id'] for s in launch.read(run/'inventory.json')['selections']} == {'lcg-fixture'}


def test_openpi_source_root_and_dataset_cache_are_run_owned(site, tmp_path):
    path, value = site
    source = tmp_path/'model-source'
    (source/'src/openpi').mkdir(parents=True)
    (source/'src/openpi/__init__.py').write_text('# frozen source fixture\n')
    value.update(source_dirs=[str(source)], isolate_model_cache=True)
    value['training_config']['openpi_root'] = str(source)
    launch.write(path, value)
    run = tmp_path/'run'
    launch.prepare(path, run)
    assert launch.read(run/'training.json')['openpi_root'] == str(run/'runtime/project_sources/source0')
    assert 'HF_DATASETS_CACHE="$HF_HOME/datasets/$kit_worker"' in (run/'launch.sh').read_text()
    assert launch.verify(run)['verified']
