"""Synthetic CPU checks, not evidence of trained reliability performance."""
from copy import deepcopy
import json
import pytest
import torch
from crane_project.utils import port_reliability_candidate_fit_v1 as f
from crane_project.tools import fit_port_reliability_candidates_v1 as tool
from crane_project.utils.port_structure_reliability_v1 import component_probes_original, quality_targets_original


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture():
    gt=torch.tensor([32.,32.,32.,16.,0.])
    probes,names=component_probes_original(gt)
    boxes=torch.cat((gt[None],probes))
    target,mask,_=quality_targets_original(boxes,gt)
    return dict(image='fixture',domain='real',descriptor=torch.eye(14),target=target,mask=mask,
                genuine_count=1,probe_names=names)


def test_capture_exact_and_no_original_gradients():
    arms=tool.branch.make_arms('cpu')
    gt=torch.tensor([32.,32.,32.,16.,0.]); probes,_=component_probes_original(gt)
    boxes=torch.cat((gt[None],probes)); p3=torch.randn(1,256,8,8)
    meta=dict(scale_factor=[1.]*4,ori_shape=(64,64,3),img_shape=(64,64,3),pad_shape=(64,64,3),flip=False)
    for arm in arms.values():
        arm.eval(); before=tool.prior.state_digest(arm)
        d,q,mlp,delta=f.capture(arm,p3,boxes,torch.full((14,),.7),meta)
        assert not d.requires_grad and d.device.type=='cpu' and delta<=1e-6
        row=fixture(); row['descriptor']=d
        f.fit(mlp,[row],steps=2,milestones=(1,2))
        assert tool.prior.state_digest(arm)==before
        assert all(p.grad is None for p in arm.parameters())
        assert len(arm.quality[0]._forward_pre_hooks)==0


def test_pairs_denominators_masks_and_gap():
    row=fixture()
    model=torch.nn.Linear(14,3,bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.logit(row['target'].clamp(.00001,.99999)).T)
    report=f.assess(model,[row])
    assert len(report['pairs'])==12
    assert all(p['correct_sign'] for p in report['pairs'])
    assert max(abs(p['gap_residual']) for p in report['pairs'])<.00002
    assert report['pair_summary']['all:angle']['count']==2
    row['mask'][:,2]=0
    assert len(f.assess(model,[row])['pairs'])==10


def test_fit_improves_learnable_fixture_preserves_initial_and_is_repeatable():
    torch.manual_seed(17)
    model=torch.nn.Sequential(torch.nn.Linear(14,16),torch.nn.ReLU(),torch.nn.Linear(16,3))
    before=deepcopy(model.state_dict()); rows=[fixture()]
    result,trained=f.fit(model,rows,steps=100,lr=.02,milestones=(50,100))
    repeat,_=f.fit(model,rows,steps=100,lr=.02,milestones=(50,100))
    assert result==repeat and result['final']['loss']<result['before']['loss']*.1
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert any(not torch.equal(v,trained.state_dict()[k]) for k,v in before.items())


def test_objective_preserves_equal_view_and_genuine_probe_weight():
    row=fixture(); model=torch.nn.Linear(14,3)
    other=deepcopy(row); other['target']*=0
    expected=(f.objective(model,[row])+f.objective(model,[other]))/2
    assert torch.allclose(f.objective(model,[row,other]),expected)
    row['descriptor'][0,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):
        f.fit(model,[row],steps=1)


def test_cache_hash_failure_and_valid_roundtrip(tmp_path):
    path=tmp_path/'descriptors.pt'; sources={'sha':'x'}; protocol={'steps':2}
    with path.open('xb') as stream:
        torch.save(dict(sources=sources,protocol=protocol),stream)
    tool.branch.write_new(path.with_suffix('.sha.json'),dict(sha256=tool.branch.sha(path),sources=sources,protocol=protocol))
    assert tool.load_cache(path,sources,protocol)['protocol']==protocol
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='identity'):
        tool.load_cache(path,sources,protocol)


def test_source_contract_and_budget():
    _,protocol,_=tool.checked_sources()
    assert protocol['steps']==2000 and protocol['lr']==.001 and len(protocol['cases'])==8
    assert protocol['optimizer']=='Adam' and protocol['val_or_test_read'] is False


def test_cpu_cache_fit_reporting_and_no_weight_export(tmp_path,monkeypatch):
    arms=tool.branch.make_arms('cpu'); sources={'fixed':'synthetic'}
    protocol={'steps':2,'lr':.001,'milestones':[1,2]}
    cache=dict(sources=sources,protocol=protocol,arms={})
    for name,arm in arms.items():
        row=fixture(); row['descriptor']=torch.randn(14,arm.quality[0].in_features)
        cache['arms'][name]=dict(initial=deepcopy(arm.quality.state_dict()),rows=[row])
    monkeypatch.setattr(tool,'checked_sources',lambda:(sources,protocol,{}))
    tool.fit_cache(cache,tmp_path)
    report=json.loads((tmp_path/'fit_report.json').read_text())
    assert report['val_or_test_read'] is False and report['deployable_weights_saved'] is False
    assert all(r['optimizer_steps']==2 for r in report['results'].values())
    assert list(tmp_path.iterdir())==[tmp_path/'fit_report.json']
    with pytest.raises(FileExistsError):
        tool.fit_cache(cache,tmp_path)
