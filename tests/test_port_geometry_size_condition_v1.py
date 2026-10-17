"""Geometry conditioning, inherited delivery and complete-VAL selection checks."""
import math
import pytest
import torch
from crane_project.utils import port_geometry_size_distribution_v1 as base
from crane_project.utils.port_geometry_size_condition_v1 import GeometryConditionSizeHead
from crane_project.tools import run_port_geometry_size_condition_v1 as tool

torch.set_num_threads(1)


def inputs(n=4):
    b=torch.tensor([[100.,100.,100.,50.,.2,.9],[100.,100.,50.,100.,-.4,.8],
                    [100.,100.,100.,100.1,.72,.7],[100.,100.,7.,4.,.3,.6]])[:n]
    xy=torch.tensor([[1.,1.],[.5,.5],[1.002,1.],[1.,1.]])[:n]
    bm=b[:,:5].clone();bm[:,:4]*=xy[:,[0,1,0,1]]
    m=b.clone()
    if n:m[0,2:4]*=torch.tensor([1.02,.98])
    return [torch.randn(n,256,9,9),torch.ones(n,1,9,9),b,bm,xy,m]


@pytest.mark.parametrize('n',[0,1,4])
def test_exact_initial_state_neutral_and_frame(n):
    torch.manual_seed(1703);r=base.DistributionSizeHead()
    torch.manual_seed(1703);c=GeometryConditionSizeHead()
    assert sum(p.numel() for p in c.parameters())==2712
    assert all(torch.equal(r.state_dict()[k],v) for k,v in c.state_dict().items())
    x=inputs(n);out=c(*x)
    assert torch.equal(out['delta_roi'],torch.zeros(n,2))
    assert torch.equal(out['boxes_original'],x[-1])


def test_only_geometry_interaction_and_no_gt_forward():
    x=inputs();r=base.DistributionSizeHead();c=GeometryConditionSizeHead()
    with torch.no_grad():
        for layer in (r.u,r.v):layer.weight.normal_(0,.1);layer.bias.fill_(.2)
    c.load_state_dict(r.state_dict())
    def interaction(module,args):
        a=args[0];return (torch.cat((a[...,:16],a[...,16:20]*a[...,-1:],a[...,20:]),-1),)
    handles=[l.register_forward_pre_hook(interaction) for l in (r.u,r.v)]
    expected=r(*x)
    for handle in handles:handle.remove()
    out=c(*x)
    for key in ('bucket_logits','delta_roi','boxes_original','border_support','geometry_inputs'):
        assert torch.equal(expected[key],out[key])
    assert torch.equal(out['boxes_original'][:,[0,1,4,5]],x[-1][:,[0,1,4,5]])


def test_geometry_gradient_and_frozen_input():
    x=inputs();x[0].requires_grad_(True);x[1].requires_grad_(True)
    h=GeometryConditionSizeHead();out=h(*x);out['delta_roi'].sum().backward()
    assert max(l.weight.grad[:,16:20].abs().max() for l in (h.u,h.v))>1e-3
    assert x[0].grad is None and x[1].grad is None
    gt=x[-1][:,:5].clone();gt[:,2:4]*=1.04
    opt=torch.optim.Adam(h.parameters(),lr=.001)
    packed=dict(tensors=x+[gt],indices=list(range(4)))
    events=[tool.old.update(h,opt,packed,list(range(4)),'decoded_control',torch) for _ in range(2)]
    assert events[0]['stem_grad_norm']==0 and events[1]['stem_grad_norm']>0


def test_components_missing_eligibility_and_strict_center():
    gt=[0.,0.,100.,50.,0.];near_square=[0.,0.,100.,90.,0.]
    def row(g,p,i):return dict(gt=g,b=p,midpoint=p,edge_residual=p,domain='real',sequence='real_seq01',image=str(i))
    rows=[row(gt,[15.,0.,100.,50.,0.,.9],0),row(gt,None,1),row(near_square,[0.,0.,100.,90.,.1,.9],2)]
    v=tool.components(rows)['overall']['edge_residual']
    assert v==dict(frames=3,outputs=2,missing=1,center_correct=1,size_correct=2,
                   angle_eligible_frames=2,angle_eligible_outputs=1,angle_correct=1)


def test_complete_epoch_choice_no_train_gate():
    rs=[dict(epoch=e,selection_key=[1./e,.1,e],adoption={'pass_all':False},train_pass_diagnostic_only=False) for e in range(1,25)]
    choice=tool.choose(rs)
    assert choice['representative']['epoch']==24 and choice['selected'] is None
    rs[5]['adoption']['pass_all']=True
    assert tool.choose(rs)['selected']['epoch']==6
    with pytest.raises(ValueError):tool.choose(rs[:-1])


def test_adoption_diagnostics_are_not_extra_gates():
    def method(mae,riou):return dict(sizes={'edges':{e:{'mae':mae,'p95':9.,'bias':8.} for e in ('long','short')},'joint10':0,'a':{'abs_mae':9.}},
        mean_riou_full_frame=riou,center_correct_full_frame={'numerator':1,'denominator':1},pure_angle_rmse_deg=1.,
        temporal={d:dict(mcml_max=0,mcml_mean=0.,tdr_w10={'numerator':1,'denominator':1},aci=.9,aci_adjacent_pairs=1) for d in ('real','sim')})
    val={d:dict(midpoint=method(.08,.8),b=method(.08,.8),edge_residual=method(.07,.81),paired=dict(frozen_identity=True,new_output=0,lost_output=0)) for d in ('real','sim')}
    assert tool.adoption(val)['pass_all']
    val['sim']['edge_residual']['sizes']['edges']['long']['mae']=.0801
    assert not tool.adoption(val)['pass_all']


def test_frozen_sources():assert tool.checked_sources()['parent']
