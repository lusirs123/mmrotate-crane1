"""Synthetic CPU math/axis tests. No optimizer, real head update or inference."""
import math
import pytest
import torch
from crane_project.utils import port_size_a20_protection_review_v1 as p
from crane_project.utils import port_geometry_midpoint_edge_residual_v1 as old
from crane_project.tools.review_port_size_a20_protection_v1 import summary


def test_static_membership_overlap_uses_tighter_error():
    x=p.fixed_references([.02,.2],[.01,.1])
    assert x==dict(near=[True,False],gain=[True,True],active=[True,True],reference=[.01,.1])
    x=p.fixed_references([.02,.2],[.04,.21])
    assert x['reference']==[.02,0.] and x['active']==[True,False]


@pytest.mark.parametrize('direction',[-1,1])
def test_harmed_near_has_nonzero_initial_repair_penalty(direction):
    z=torch.tensor([[math.log(1+direction*.04),math.log(.9)]],dtype=torch.float64,requires_grad=True)
    m=torch.tensor([[.02,.2]],dtype=z.dtype);a=torch.tensor([[.04,.1]],dtype=z.dtype)
    v=p.prototype_losses(z,m,a,'P');assert v['protect'].item()==pytest.approx(.01)
    v['protect'].backward();assert z.grad[0,0]*direction>0 and z.grad[0,1]==0


def test_better_than_teacher_is_allowed_and_references_detached():
    z=torch.tensor([[math.log(.99),math.log(1.01)]],dtype=torch.float64,requires_grad=True)
    m=torch.tensor([[.15,.2]],dtype=z.dtype,requires_grad=True)
    a=torch.tensor([[.05,.05]],dtype=z.dtype,requires_grad=True)
    v=p.prototype_losses(z,m,a,'P');assert v['protect']==0
    v['total'].backward();assert m.grad is None and a.grad is None
    # This prediction crosses the teacher's side of GT; no coordinate distillation.
    assert z.grad[0,0]<0 and z.grad[0,1]>0


def test_inactive_zero_fixed_two_n_denominator():
    z=torch.full((2,2),math.log(1.2),dtype=torch.float64,requires_grad=True)
    m=torch.tensor([[.02,.1],[.1,.1]],dtype=z.dtype)
    a=torch.tensor([[.1,.2],[.2,.2]],dtype=z.dtype)
    v=p.prototype_losses(z,m,a,'P')
    assert v['active'].sum()==1 and v['protect'].item()==pytest.approx((.2-.02)/4)


def test_exact_gt_global_minimum_does_not_pull_back_to_teacher():
    z=torch.zeros(2,2,dtype=torch.float64,requires_grad=True)
    m=torch.full_like(z,.02);a=torch.full_like(z,.01)
    v=p.prototype_losses(z,m,a,'P');assert v['total']==0
    v['total'].backward();assert torch.equal(z.grad,torch.zeros_like(z))


def test_protection_is_reactive_not_per_step_guarantee():
    z=torch.tensor([[0.,-.1]],dtype=torch.float64,requires_grad=True)
    m=torch.tensor([[0.,.15]],dtype=z.dtype);a=torch.expm1(z.detach()).abs()
    v=p.prototype_losses(z,m,a,'P');assert v['protect']==0
    v['total'].backward();new=z.detach()-.001*z.grad
    assert new[0,0]!=0 and abs(torch.expm1(new[0,0]))>0
    assert p.prototype_losses(new,m,a,'P')['protect']>0


@pytest.mark.parametrize('m_swap,b_swap',[(False,False),(True,False),(False,True),(True,True)])
def test_axis_mapping_and_exact_control_graph(m_swap,b_swap):
    gt=torch.tensor([[10.,10.,100.,50.,.2]],dtype=torch.float64)
    m=torch.tensor([[10.,10.,90.,55.,.2,.8]],dtype=torch.float64)
    bm=torch.tensor([[10.,10.,95.,54.,.2]],dtype=torch.float64)
    if m_swap:m[:,2:4]=m[:,2:4].flip(-1)
    if b_swap:bm[:,2:4]=bm[:,2:4].flip(-1)
    target=old.size_targets(gt,m,bm)['log_residual']
    canonical_error=torch.tensor([[.02,-.03]],dtype=m.dtype)
    delta=(target+(canonical_error.flip(-1) if m_swap^b_swap else canonical_error)).requires_grad_()
    em=torch.tensor([[.1,.1]],dtype=m.dtype);ea=torch.tensor([[.05,.05]],dtype=m.dtype)
    result=p.from_delta(delta,gt,m,bm,em,ea,'C');base=old.size_loss(delta,gt,m,bm)
    assert torch.equal(result['total'],base)
    g1=torch.autograd.grad(result['total'],delta,retain_graph=True)[0]
    g2=torch.autograd.grad(base,delta)[0];assert torch.equal(g1,g2)
    assert torch.allclose(result['error'],torch.expm1(canonical_error).abs(),atol=1e-15)


def test_ratio_formula_is_same_as_old_ratio_control():
    z=torch.tensor([[.05,-.08],[-.02,.01]],dtype=torch.float64)
    m=torch.full_like(z,.1);a=torch.full_like(z,.05)
    r=p.prototype_losses(z,m,a,'R')
    expected=torch.nn.functional.smooth_l1_loss(z[:,0]-z[:,1],torch.zeros(2,dtype=z.dtype),beta=.1)
    assert torch.equal(r['ratio'],expected)
    assert torch.equal(r['total'],r['size']+.25*expected)


@pytest.mark.parametrize('value',[float('nan'),1000.])
def test_invalid_or_overflow_not_silently_filtered(value):
    z=torch.tensor([[value,0.]],dtype=torch.float64);ref=torch.full_like(z,.02)
    with pytest.raises(ValueError):p.prototype_losses(z,ref,ref,'P')


def test_saved_stats_overlap_near_harm_and_missing_separate():
    row=dict(gt=[0,0,100,50,0],midpoint=[0,0,98,40,0,.8],edge_residual=[0,0,96,45,0,.8],size_delivery='accepted')
    missing=dict(gt=[0,0,100,50,0],midpoint=None,edge_residual=None,size_delivery='no_output')
    x=summary([row,missing]);assert x['views']==2 and x['outputs']==1 and x['missing']==1
    assert x['edges']['long']['near']==1 and x['edges']['long']['near_harmed']==1
    assert x['edges']['short']['gain']==1 and x['edges']['short']['near']==0
    assert x['edges']['long']['initial_hinge_sum']==pytest.approx(.02)


def test_source_identity_accepts_relative_cli_file(monkeypatch):
    from crane_project.tools import review_port_size_a20_protection_v1 as review
    monkeypatch.setattr(review,'__file__','crane_project/tools/review_port_size_a20_protection_v1.py')
    proof=review.source_identity()
    assert len(proof)==4 and all(len(h)==64 for h in proof.values())


def test_capped_retention_cannot_hide_lost_gains_with_larger_other_gain():
    x=p.capped_retention([.1,.1],[.05,.05],[0.,.1]);assert x['fraction']==pytest.approx(.5)
    assert p.capped_retention([.1],[.1],[0.])['fraction'] is None


def test_removing_all_correction_fails_gain_reference():
    m=torch.tensor([[.2,.1]],dtype=torch.float64);a=torch.tensor([[.1,.05]],dtype=m.dtype)
    z=torch.log1p(m).requires_grad_();v=p.prototype_losses(z,m,a,'P')
    assert v['protect'].item()>0
    assert p.capped_retention(m[0].tolist(),a[0].tolist(),m[0].tolist())['fraction']==0
