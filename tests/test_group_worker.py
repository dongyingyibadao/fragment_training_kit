"""Packing keeps one global worker identity per allocated logical device."""
import copy
import json
import signal
import subprocess
import sys

import pytest

from fragment_training_kit.group_worker import worker_slots
from fragment_training_kit import launch
from fragment_training_kit import group_worker


def test_mixed_job_slots_cover_every_device_once():
    layout = [16, 8, 4, 2, 1]
    slots = [worker_slots(layout, f"w{i+1:06d}", size) for i, size in enumerate(layout)]
    ids = [worker for group in slots for worker, index in group]
    assert ids == [f"w{i+1:06d}" for i in range(31)]
    assert slots[1] == [(f"w{i+17:06d}", i) for i in range(8)]


@pytest.mark.parametrize("layout,prefix,count", [([16],"w000001",8), ([2],"w000002",2),
    ([32],"w000001",32), ([True],"w000001",1), ([16]*5,"w000001",16), ([2],"../w000001",2)])
def test_slot_mismatch_or_unbounded_ids_are_rejected(layout, prefix, count):
    with pytest.raises(ValueError):
        worker_slots(layout, prefix, count)


def selection(size):
    return {"workspace_id":launch.WORKSPACES[1],"project_id":"fixture","logic_compute_group_id":"group",
        "image":"fixture","image_type":"SOURCE_PRIVATE",
        "priority":{"verified":True,"level":"LOW","name":"1","task_priority":1},
        "resource_spec_price":{"gpu_count":size,"gpu_type":"ASCEND_910B","quota_id":"fixture",
                               "cpu_count":4*size,"memory_size_gib":16*size}}


def test_layout_planning_honors_rank_ownership_after_partial_submission():
    inventory = {"selections":[selection(n) for n in (1,2,4,8,16)],"groups":[{
        "id":"group","workspace_id":launch.WORKSPACES[1],"resource_query":{"status":"fresh"},
        "resources":{"gpu_available":32,"cpu_available":128,"memory_gi_available":512}}]}
    planner = launch.layout_planner([16,8,4,2,1], 1)
    result = planner(inventory, [], 15, {}, 1)
    assert [r['cards'] for r in result] == [8,4,2,1]
    # A partial budget cannot silently replace a fixed eight-card identity by
    # seven single-card jobs and assign overlapping global ranks.
    assert planner(inventory, [], 7, {}, 1) == []


def test_prepare_mixed_jobs_freezes_mapping_and_preserves_allocated_mask(tmp_path):
    cann=tmp_path/'cann.sh';cann.write_text('# fixture\n')
    inventory=tmp_path/'inventory.json'
    inventory.write_text(json.dumps({'backend':'ascend','user':{'id':'fixture'},'selections':[selection(2),selection(1)]}))
    site={'schema':launch.SCHEMA,'python':sys.executable,'cann_env':str(cann),'inventory':str(inventory),
        'registry':str(tmp_path/'registry.json'),'cards':3,'job_layout':[2,1],'steps':20,
        'training_config':{'resident_adapter':'fixture:create_engine','resident_rng_scheme':'fixture',
                           'resident_precision_contract':'native FP32'}}
    path=tmp_path/'site.json';path.write_text(json.dumps(site))
    run=tmp_path/'run'
    result=launch.prepare(path,run)
    assert result['initial_intent_limit']==2 and result['job_layout']==[2,1]
    assert launch.verify(run)['verified']
    shell=(run/'launch.sh').read_text()
    assert 'group_worker.py' in shell and '--allocated-devices' in shell
    assert 'ASCEND_VISIBLE_DEVICES=' not in shell and 'ASCEND_RT_VISIBLE_DEVICES=' not in shell
    assert '{npu_count}' in (run/'fleet_command.txt').read_text()


@pytest.mark.parametrize("vanishes_before_kill", [False, True])
def test_sigterm_enters_bounded_cleanup_even_when_children_ignore_it(tmp_path, monkeypatch, vanishes_before_kill):
    run = tmp_path / "run"
    (run / "worker_logs").mkdir(parents=True)
    (run / "site.json").write_text(json.dumps({"cards": 2, "job_layout": [2], "active_cards": 2}))
    handlers, original, children, signals, sleeps = {}, {}, [], [], []
    for sig in (signal.SIGTERM, signal.SIGINT):
        handlers[sig] = original[sig] = object()
    def install(sig, handler):
        previous = handlers[sig]
        handlers[sig] = handler
        return previous
    monkeypatch.setattr(group_worker.signal, "signal", install)
    class Child:
        def __init__(self, pid):
            self.pid, self.returncode, self.waits = pid, None, []
        def poll(self):
            return self.returncode
        def wait(self, timeout):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                raise subprocess.TimeoutExpired("stubborn-fixture-supervisor", timeout)
            self.returncode = -signal.SIGKILL
            return self.returncode
    def spawn(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        child = Child(100000 + len(children))
        children.append(child)
        return child
    def send(pid, sig):
        assert pid in {child.pid for child in children}
        signals.append((pid, sig))
        if sig == signal.SIGKILL and vanishes_before_kill:
            raise ProcessLookupError("owned group exited just before escalation")
    def idle(seconds):
        sleeps.append(seconds)
        assert len(sleeps) == 1, "SIGTERM must leave the polling loop even if every child remains alive"
        handlers[signal.SIGTERM](signal.SIGTERM, None)
    monkeypatch.setattr(group_worker.subprocess, "Popen", spawn)
    monkeypatch.setattr(group_worker.os, "killpg", send)
    monkeypatch.setattr(group_worker.time, "sleep", idle)
    assert group_worker.main(["--run-dir", str(run), "--worker-prefix", "w000001", "--allocated-devices", "2"]) == 1
    assert len(children) == 2 and all(child.waits == [5, 5] for child in children)
    assert {pid for pid, sig in signals if sig == signal.SIGKILL} == {child.pid for child in children}
    assert handlers == original
