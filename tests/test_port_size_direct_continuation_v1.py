"""S048 changed D continuation checks; no dataset or TEST/depth execution."""
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from crane_project.utils import port_size_direct_continuation_v1 as o
from crane_project.utils import port_size_structure_v1 as m
from crane_project.tools import run_port_size_direct_continuation_v1 as run
from crane_project.tools import run_port_size_structure_v1 as old
from crane_project.utils import port_size_a20_protection_train_v1 as historic
from test_port_size_native_p3_v1 import fixture
from test_port_size_a20_protection_train_v1 import row,eligible_fixture


def batch():
    x=list(fixture()[:6]);gt=x[5][:,:5].clone();gt[:,2]*=.97;gt[:,3]*=1.02;x.append(gt);return x


def test_direct_C_original_two_update_loss_gradients_and_adam_exact():
    torch.manual_seed(1703);a=m.DirectSizeHead();b=deepcopy(a);oa,ob=[run.optimizer(h,torch) for h in (a,b)];x=batch()
    assert not oa.state and not ob.state
    for _ in range(2):
        reference=old.update(a,oa,x,'D_direct',torch);got,_=run.update(b,ob,x,'C',torch)
        assert reference['loss']==got['total']
        assert run.equal(a.state_dict(),b.state_dict(),torch) and run.equal(oa.state_dict(),ob.state_dict(),torch)
        assert all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()))


@pytest.mark.parametrize('swap,aniso',[(False,False),(True,True)])
def test_R_ratio_is_exact_historical_definition_and_only_difference(swap,aniso):
    x=list(fixture(swap,aniso)[:6]);gt=x[-1][:,:5].clone();gt[:,2]*=.97;gt[:,3]*=1.02;h=m.DirectSizeHead()
    with torch.no_grad():h.terminal.bias.copy_(torch.tensor([.01,-.02]))
    out=h(*x);c=o.losses(out,gt,x[-1],x[3],'C',torch);r=o.losses(out,gt,x[-1],x[3],'R',torch)
    # Testing formula equivalence only; no surrogate vector validity in D forward.
    prior=historic.losses(dict(out,vector_valid=out['regression_valid']),gt,x[-1],x[3],dict(active=torch.zeros(1,2,dtype=torch.bool),reference=torch.zeros(1,2,dtype=torch.float64)),'R',torch)
    assert torch.equal(r['ratio'],prior['ratio']) and torch.equal(r['total'],prior['total'])
    assert torch.equal(r['total'],c['total']+.25*c['ratio'])
    assert set(c)=={'total','size','ratio','point_size','point_ratio'}
    with pytest.raises(ValueError):o.losses(out,gt,x[-1],x[3],'P',torch)


@pytest.mark.parametrize('arm',o.ARMS)
def test_frozen_inputs_and_exact_next_step_state_resume(arm):
    x=batch();x[0].requires_grad_(True);a=m.DirectSizeHead();oa=run.optimizer(a,torch);run.update(a,oa,x,arm,torch)
    assert x[0].grad is None
    stream=io.BytesIO();torch.save(dict(head=a.state_dict(),opt=oa.state_dict(),rng=torch.get_rng_state()),stream);stream.seek(0);saved=torch.load(stream)
    b=m.DirectSizeHead();b.load_state_dict(saved['head']);ob=run.optimizer(b,torch);assert not ob.state;ob.load_state_dict(saved['opt'])
    run.update(a,oa,x,arm,torch);run.update(b,ob,x,arm,torch)
    assert run.equal(a.state_dict(),b.state_dict(),torch) and run.equal(oa.state_dict(),ob.state_dict(),torch)


def test_D_reference_sets_only_offline_retention_and_erosion():
    r=row(m=90.,a=95.);table=o.reference_table([r]);changed=deepcopy(r);changed['edge_residual']=[0.,0.,93.,50.,0.,.8];changed['pre_guard']=changed['edge_residual']
    report=o.protection_report([changed],table)['sim']['edges']['long']['delivered']
    assert report['D_gain_weakened_still_beneficial']==1 and report['D_gain_preserved_or_better']==0
    assert report['D_gain_erosion_sum']==pytest.approx(.02) and report['useful']['fraction']==pytest.approx(.6)
    changed['edge_residual']=[0.,0.,89.,50.,0.,.8];changed['pre_guard']=changed['edge_residual']
    report=o.protection_report([changed],table)['sim']['edges']['long']['delivered'];assert report['useful_lost']==1 and report['useful_direction_reversed']==1


def test_ratio_intersections_keep_magnitude_not_just_net_count():
    c=row(m=100.,a=110.,split='val');c['b']=list(c['midpoint']);c['edge_residual']=[0.,0.,110.,50.,0.,.8]
    r=deepcopy(c);r['edge_residual']=[0.,0.,110.,54.,0.,.8]
    report=o.ratio_intersections([c],[r])['sim']
    assert report['counts']['ratio_improved']==1 and report['counts']['ratio_improved_short_worse']==1
    assert report['counts']['ratio_improved_any_Mnear_harmed']==1 and report['counts']['ratio_improved_scale_worse']==1
    assert report['amounts_on_ratio_improved']['short_harm']==pytest.approx(.08)
    bad=deepcopy(r);bad['image']='other'
    with pytest.raises(ValueError):o.ratio_intersections([c],[bad])


def test_historical_conditions_source_binding_D02_not_new_thresholds():
    s,d,start=eligible_fixture();current=o.eligibility(s,d,start);old=historic.eligibility(s,d,start)
    assert current['checks']=={k.replace('A20','D02'):v for k,v in old['checks'].items()}
    p=o.protocol();assert p['epochs_per_arm']==6 and p['formal_updates']==7668 and p['val_evaluations']==12
    assert p['shuffle']=='Random(1703+20+continuation_epoch)' and p['losses']['ratio_weight']==.25 and 'Lprotect' not in p['losses']
    assert not any(k.startswith('sim_A20') for k in p['conditions'])


def test_invalid_prediction_stops_no_effect_gate_or_fake_vector_checks():
    x=batch();h=m.DirectSizeHead();out=h(*x[:-1]);out['candidate_valid'][0]=False
    with pytest.raises(ValueError):o.losses(out,x[-1],x[-2],x[3],'C',torch)
    out=h(*x[:-1]);out['regression_valid'][0]=False
    with pytest.raises(ValueError):o.losses(out,x[-1],x[-2],x[3],'R',torch)


def test_resource_pick_requested_card_and_low_free_fail(monkeypatch,tmp_path):
    from crane_project.utils import port_size_direct_continuation_resource_v1 as r
    class NoThread:
        def __init__(self,**kwargs):pass
        def start(self):pass
    monkeypatch.setattr(r.threading,'Thread',NoThread);monkeypatch.setattr(r,'smi',lambda *args:[['0','card0','6500'],['1','card1','4000'],['2','card2','4500'],['3','card3','5000']]);monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    state,_,_=r.setup_gpu(tmp_path);assert state['physical_gpu']==0
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3');state,_,_=r.setup_gpu(tmp_path);assert state['physical_gpu']==3
    monkeypatch.setattr(r,'smi',lambda *args:[['3','card3','3000']])
    with pytest.raises(RuntimeError):r.setup_gpu(tmp_path)


def test_checkpoint_source_is_D_only_and_adam_fresh():
    state=m.DirectSizeHead().state_dict();other=m.DirectSizeHead();other.load_state_dict(state)
    opt=run.optimizer(other,torch);assert not opt.state
    pg=opt.param_groups[0];assert (pg['lr'],pg['betas'],pg['eps'],pg['weight_decay'],pg['amsgrad'])==(.0001,(.9,.999),1e-8,0,False)
