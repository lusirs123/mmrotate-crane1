"""Axis identity, natural-path equivalence and repeated-slot update checks."""
from copy import deepcopy
import pytest
import torch
from crane_project.utils import port_geometry_size_proposal_v1 as p
from tests.test_port_geometry_size_condition_v1 import inputs

torch.set_num_threads(1)


def batch():
    x=inputs();gt=x[-1][:,:5].clone();gt[:,2:4]*=torch.tensor([1.04,.96]);return x,gt


def test_seven_slots_exact_single_axis_targets():
    x,gt=batch();q,t=p.proposals(x[-1],gt,x[3],'proposal7')
    assert q.shape==(4,7,6) and t.shape==(4,7,2)
    assert torch.equal(q[:,0],x[-1])
    long_raw=(x[-1][:,2]<x[-1][:,3]).long();short_raw=1-long_raw
    for i in range(4):
        for slot,raw in [(5,long_raw[i]),(6,short_raw[i])]:
            roi_raw=1-int(raw) if x[3][i,2]<x[3][i,3] else int(raw)
            assert t[i,slot,roi_raw]==0
        assert torch.equal(q[i,1,[0,1,4,5]],x[-1][i,[0,1,4,5]])
    # Near-square proposals may cross. Assignment remains natural-M based.
    assert (q[2,3,2]<q[2,3,3])!=(x[-1][2,2]<x[-1][2,3])


def test_empty_and_no_input_mutation():
    x,gt=batch();before=deepcopy(x);p.proposals(x[-1],gt,x[3],'proposal7')
    assert all(torch.equal(a,b) for a,b in zip(x,before))
    q,t=p.proposals(x[-1][:0],gt[:0],x[3][:0],'proposal7');assert q.shape==(0,7,6)


def test_neutral_and_non_neutral_natural_readout_exact():
    x,gt=batch();h=p.GeometryConditionSizeHead()
    for nonneutral in (False,True):
        if nonneutral:
            with torch.no_grad():h.u.weight.normal_(0,.02);h.v.weight.normal_(0,.02)
        natural=h(*x);got=p.readout(h,*x,x[-1][:,None])
        for key in ('delta_roi','bucket_logits','border_support','geometry_inputs'):
            assert torch.equal(natural[key],got[key][:,0]),key
        if not nonneutral:assert torch.equal(p.delivery(got,*x[2:],x[-1][:,None])['boxes_original'],x[-1])


def test_q_outside_M_guard_not_filtered_and_targets_not_clipped():
    x,gt=batch();gt[0,2]*=1.8;q,t=p.proposals(x[-1],gt,x[3],'proposal7')
    h=p.GeometryConditionSizeHead();out=p.readout(h,*x,q)
    assert torch.equal(out['delta_roi'],torch.zeros_like(t))
    assert q[0,5,2]>x[2][0,2]*1.25
    delivered=p.delivery(out,*x[2:],q)
    assert torch.equal(delivered['boxes_original'][5],x[-1][0])
    assert t[0,0].abs().max()>.5


def test_repeat7_loss_gradients_and_two_adam_updates():
    x,gt=batch();torch.manual_seed(1703);original=p.GeometryConditionSizeHead();repeat=deepcopy(original)
    oa=torch.optim.Adam(original.parameters(),lr=.001);ob=torch.optim.Adam(repeat.parameters(),lr=.001)
    for _ in range(2):
        oa.zero_grad();ob.zero_grad();out=original(*x)
        target=p.base.size_targets(gt,x[-1],x[3])['log_residual']
        la=torch.nn.functional.smooth_l1_loss(out['delta_roi'],target,beta=.1)
        q,t=p.proposals(x[-1],gt,x[3],'repeat7');lb=p.loss(p.readout(repeat,*x,q),t)['total']
        assert torch.allclose(la,lb,rtol=1e-6,atol=1e-10)
        la.backward();lb.backward()
        for a,b in zip(original.parameters(),repeat.parameters()):assert torch.allclose(a.grad,b.grad,rtol=1e-5,atol=1e-8)
        for h in (original,repeat):torch.nn.utils.clip_grad_norm_(h.parameters(),10.)
        oa.step();ob.step()
        # Common logit biases have zero mathematical gradient, but Adam can
        # amplify roundoff. Full update equivalence is NOT asserted/reused.
        with torch.no_grad():
            left,right=original(*x),repeat(*x)
            assert torch.allclose(left['delta_roi'],right['delta_roi'],rtol=1e-5,atol=1e-6)
            assert torch.allclose(left['boxes_original'],right['boxes_original'],rtol=1e-5,atol=1e-4)


def test_no_gt_readout_and_frozen_visual_inputs():
    x,gt=batch();x[0].requires_grad_(True);x[1].requires_grad_(True)
    q,t=p.proposals(x[-1],gt,x[3],'proposal7');h=p.GeometryConditionSizeHead()
    p.loss(p.readout(h,*x,q),t)['total'].backward()
    assert x[0].grad is None and x[1].grad is None
    bad=q.clone();bad[0,0,0]+=1
    with pytest.raises(ValueError):p.readout(h,*x,bad)


def test_frozen_selection_no_train_gate_and_near_reference():
    from crane_project.tools import run_port_geometry_size_proposal_v1 as t
    rs=[dict(epoch=e,selection_key=[1/e,1/e,e],adoption=dict(pass_all=False),train_pass_diagnostic_only=False) for e in range(1,25)]
    assert t.condition.choose(rs)['representative']['epoch']==24
    rs[3]['adoption']['pass_all']=True
    assert t.condition.choose(rs)['selected']['epoch']==4
    with pytest.raises(ValueError):t.condition.choose(rs[:-1])
    assert 'fixed-M' in t.protocol()['adoption']['near']
    assert t.protocol()['limits']['test_access'] is False and t.protocol()['limits']['depth_access'] is False
