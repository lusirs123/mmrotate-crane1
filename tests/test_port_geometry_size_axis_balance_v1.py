"""Axis permutation, normalization and unchanged-I control checks."""
import pytest
import torch
from crane_project.utils import port_geometry_size_axis_balance_v1 as model
from crane_project.utils.port_geometry_size_condition_v1 import GeometryConditionSizeHead
from crane_project.utils import port_geometry_size_distribution_v1 as original
from tests.test_port_geometry_size_condition_v1 import inputs

torch.set_num_threads(1)


def test_axes_all_four_swap_combinations():
    m = torch.tensor([[0.,0.,100.,50.],[0.,0.,50.,100.],[0.,0.,100.,50.],[0.,0.,50.,100.]])
    b = torch.tensor([[0.,0.,100.,50.],[0.,0.,100.,50.],[0.,0.,50.,100.],[0.,0.,50.,100.]])
    v = torch.tensor([[2.,3.]]*4)
    assert torch.equal(model.permute_axes(v,m,b),torch.tensor([[2.,3.],[3.,2.],[3.,2.],[2.,3.]]))
    assert torch.equal(model.permute_axes(model.permute_axes(v,m,b),m,b),v)


@pytest.mark.parametrize('n',[dict(undersized=38,oversized=130,near3=580),dict(undersized=1,oversized=99,near3=1)])
def test_mean_one_cap_and_no_near_reduction(n):
    w = model.weights_for_counts(n)
    assert w['near3']==1 and max(w.values())<=4
    assert sum(n[g]*w[g] for g in n)==pytest.approx(sum(n.values()))
    with pytest.raises(ValueError):model.weights_for_counts(dict(undersized=0,oversized=2,near3=1))


def test_weighted_loss_not_batch_normalized_and_gradient_mapping():
    x=inputs();gt=x[-1][:,:5].clone();gt[:,2:4]*=1.1
    h=GeometryConditionSizeHead();out=h(*x);w=torch.tensor([[2.,1.],[3.,1.],[1.,1.],[1.,1.]])
    result=model.loss(out,gt,x[-1],x[3],w)
    expected=(result['canonical_pointwise']*w).sum()/8
    assert torch.equal(result['total'],expected)
    assert not torch.isclose(result['total'],(result['canonical_pointwise']*w).sum()/w.sum())
    result['total'].backward();assert h.u.weight.grad is not None


def test_all_one_loss_gradients_adam_and_clip_exact_two_updates():
    x=inputs();gt=x[-1][:,:5].clone();gt[:,2:4]*=torch.tensor([1.04,.98])
    torch.manual_seed(1703);a=GeometryConditionSizeHead()
    torch.manual_seed(1703);b=GeometryConditionSizeHead()
    oa,ob=[torch.optim.Adam(h.parameters(),lr=.001) for h in (a,b)]
    for _ in range(2):
        oa.zero_grad();ob.zero_grad()
        la=original.loss(a(*x),gt,x[-1],x[3],'decoded_control')['total']
        lb=model.loss(b(*x),gt,x[-1],x[3],torch.ones(4,2))['total']
        assert torch.equal(la,lb)
        la.backward();lb.backward()
        assert all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()))
        torch.nn.utils.clip_grad_norm_(a.parameters(),10);torch.nn.utils.clip_grad_norm_(b.parameters(),10)
        oa.step();ob.step()
        assert all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters()))


def test_label_is_fixed_m_single_axis_not_candidate():
    assert model.labels([0,0,100,50,0],[0,0,95,51,0,.9])==['undersized','near3']
    assert model.labels([0,0,50,100,0],[0,0,51,95,0,.9])==['undersized','near3']


def test_all24_choice_never_uses_train_gate():
    from crane_project.tools import run_port_geometry_size_axis_balance_v1 as tool
    values=[dict(epoch=e,selection_key=[1/e,.1,e],adoption={'pass_all':False},train_pass_diagnostic_only=False) for e in range(1,25)]
    c=tool.choose(values)
    assert c['representative']['epoch']==24 and c['selected'] is None
    values[2]['adoption']['pass_all']=True
    assert tool.choose(values)['selected']['epoch']==3
    with pytest.raises(ValueError):tool.choose(values[:-1])


def test_strict_adoption_and_fixed_near_are_not_train_gate():
    from copy import deepcopy
    from crane_project.tools import run_port_geometry_size_axis_balance_v1 as tool
    def method(mae,riou,a):
        return dict(sizes=dict(edges={e:dict(mae=mae) for e in ('long','short')},a={'abs_mae':a}),
            mean_riou_full_frame=riou,center_correct_full_frame={'numerator':1,'denominator':1},pure_angle_rmse_deg=1.,
            temporal={d:dict(mcml_max=0,mcml_mean=0.,tdr_w10={'numerator':1,'denominator':1},aci=.9,aci_adjacent_pairs=1) for d in ('real','sim')})
    reference={d:{'edge_residual':method(.05,.8,.05)} for d in ('real','sim')}
    val={d:dict(b=method(.04,.8,.04),midpoint=method(.05,.8,.04),edge_residual=method(.03,.81,.03),
               paired=dict(frozen_identity=True,new_output=0,lost_output=0)) for d in ('real','sim')}
    near={d:{e:dict(count=2,methods={'edge_residual':{'mae':.01}}) for e in ('long','short')} for d in ('real','sim')}
    assert tool.adoption(val,near,reference,near)['pass_all']
    changed=deepcopy(near);changed['sim']['short']['methods']['edge_residual']['mae']=.02
    assert 'sim/near_short' in tool.adoption(val,changed,reference,near)['failed']
