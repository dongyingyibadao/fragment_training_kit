"""CPU tensor oracles for split-phase device safety; no NPU allocation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from fragment_training_kit.examples.toy_regression import ToyEngine, config
from fragment_training_kit.vendor.jade_lotus.pod_worker_runtime import PodWorker


def probe_module():
    path=Path(__file__).resolve().parents[1]/'vendor/scripts/probe_resident_openpi_fault.py'
    spec=importlib.util.spec_from_file_location('pod_worker_probe',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def worker(tmp_path,monkeypatch):
    api=probe_module()
    engine=ToyEngine(config(20),device='cpu')
    optimizer=torch.optim.AdamW(engine.parameters.values(),lr=.05,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    member={'worker_id':'w000001','session_id':'session','pid':101}
    group={'generation':0,'members':[member]}
    spec={'run_id':'run','phase_timeout':1,'collective_timeout':1,'heartbeat_timeout':.1,
          'bucket_bytes':512,'control_poll_seconds':.01}
    subject=PodWorker(api,tmp_path,spec,member,engine,optimizer,lambda *_:None)
    monkeypatch.setitem(sys.modules,'torch_npu',SimpleNamespace())
    backend=SimpleNamespace()
    monkeypatch.setattr(torch.distributed.distributed_c10d,'_get_default_group',lambda:SimpleNamespace(_get_backend=lambda _:backend))
    class Monitor:
        triggered=False
        done=SimpleNamespace(wait=lambda _:True)
        def __init__(self,*_args,**_kwargs):
            self.submission_lock=__import__('threading').Lock()
            self.armed=SimpleNamespace(set=lambda:None)
        def start(self):return self
        def wait_ready(self,_):pass
        def finish(self,_):pass
        def close(self):pass
    monkeypatch.setattr(api,'FaultMonitor',Monitor)
    monkeypatch.setattr(api,'_submit',lambda *_args,**_kwargs:[])
    monkeypatch.setattr(api,'_wait_works',lambda *_args,**_kwargs:None)
    return subject,group,api


def command(action,token='command',**fields):
    return {'action':action,'token':token,'expected_step':0,'root_token':'root',**fields}


def test_one_optimizer_application_and_independent_raw_gradients(worker,monkeypatch):
    subject,group,api=worker
    mapping={'w000001':list(range(64))}
    subject.execute(command('pod_compute',group=group,assignments=mapping),group,0)
    original={name:p.grad.clone() for name,p in subject.engine.parameters.items()}
    def release(readiness,*_args,**_kwargs):return 'release'
    monkeypatch.setattr(api,'_wait',release)
    subject.execute(command('pod_reduce',token='reduce',group=group,assignments=mapping),group,0)
    assert all(torch.equal(p.grad,original[name]) for name,p in subject.engine.parameters.items())
    before={name:p.clone() for name,p in subject.engine.parameters.items()}
    _,step=subject.execute(command('pod_apply',token='update',group=group,reduction_token='reduce'),group,0)
    assert step==1 and all(subject.optimizer.state[p]['step'].item()==1 for p in subject.engine.parameters.values())
    assert any(not torch.equal(p,before[name]) for name,p in subject.engine.parameters.items())
    updated=api._read(subject.run/'reports/w000001/session/update/updated.json')
    assert updated['optimizer_applications']==1 and updated['local']['samples']==256
    with pytest.raises(ValueError,match='unprepared'):
        subject.execute(command('pod_apply',token='update-again',group=group,reduction_token='reduce'),group,0)
    assert all(subject.optimizer.state[p]['step'].item()==1 for p in subject.engine.parameters.values())


def test_replay_rejects_duplicate_unit_without_mutating_gradient(worker):
    subject,group,_=worker
    subject.execute(command('pod_compute',group=group,assignments={'w000001':[0,1]}),group,0)
    saved={name:p.grad.clone() for name,p in subject.engine.parameters.items()}
    with pytest.raises(ValueError,match='duplicate'):
        subject.execute(command('pod_replay',replay={'w000001':[1,2]}),group,0)
    assert all(torch.equal(p.grad,saved[name]) for name,p in subject.engine.parameters.items())


def test_failed_reduction_can_never_authorize_apply(worker,monkeypatch):
    subject,group,api=worker
    mapping={'w000001':list(range(64))}
    subject.execute(command('pod_compute',group=group,assignments=mapping),group,0)
    with pytest.raises(ValueError,match='released reduction'):
        subject.execute(command('pod_apply',group=group,reduction_token='nonexistent'),group,0)
    assert not subject.optimizer.state


def test_stale_root_replay_cannot_add_gradient(worker):
    subject,group,_=worker
    subject.execute(command('pod_compute',group=group,assignments={'w000001':[0]}),group,0)
    stale=command('pod_replay',replay={'w000001':[1]});stale['root_token']='retired'
    with pytest.raises(ValueError,match='different/unprepared'):
        subject.execute(stale,group,0)
    assert subject.local['units']==[0]
