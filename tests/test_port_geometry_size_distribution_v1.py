"""Meaningful size codec, gradient, frame and selection contracts; synthetic only."""
from copy import deepcopy
import math
import tempfile
from pathlib import Path
import pytest
import torch
from crane_project.utils import port_geometry_size_distribution_v1 as model
from crane_project.utils import port_geometry_size_distribution_metrics_v1 as metrics
from crane_project.tools import run_port_geometry_size_distribution_v1 as tool

torch.set_num_threads(1)


def inputs(n=8):
    b=torch.tensor([[100.,100.,100.,50.,.2,.9]]).repeat(n,1)
    gt=b[:,:5].clone();gt[:,2:4]*=1.04
    return [torch.randn(n,256,9,9),torch.ones(n,1,9,9),b,b[:,:5].clone(),torch.ones(n,2),b.clone(),gt]


@pytest.mark.parametrize('n',[0,1,8])
def test_full_neutral_and_missing(n):
    x=inputs(n);h=model.DistributionSizeHead();y=h(*x[:-1])
    assert sum(p.numel() for p in h.parameters())==2712
    assert torch.equal(y['boxes_original'],x[-2])
    assert torch.equal(y['delta_roi'],torch.zeros(n,2))
    assert y['border_support'].shape==(n,2,33,2)


def test_soft_labels_reconstruct_continuous_targets_and_endpoints():
    grid=torch.linspace(-1,1,33,dtype=torch.float64)
    target=torch.tensor([[-1.,1.],[-.8402781227487,.275788762177],[.0003,-.0031]],dtype=torch.float64)
    logits=torch.zeros(3,2,33,dtype=torch.float64)
    label=model.distribution_targets(target,logits,grid)
    assert torch.allclose(label['reconstructed'],target,atol=2e-16,rtol=0)
    assert torch.equal(label['weight_left']+label['weight_right'],torch.ones_like(target))
    assert bool((label['weight_left']>=0).all() and (label['weight_right']>=0).all())
    with pytest.raises(ValueError,match='no clipping'):
        model.distribution_targets(torch.tensor([[1.01,0.]],dtype=torch.float64),logits[:1],grid)


@pytest.mark.parametrize('arm',model.ARMS)
def test_actual_scalar_bidirectional_gradient_and_correct_size_protection(arm):
    # Independent logits make target-response direction observable; not a
    # claim that shared network parameters will succeed on natural TRAIN.
    logits=torch.zeros(3,2,33,requires_grad=True)
    b=torch.tensor([[0.,0.,100.,50.,0.,.9]]).repeat(3,1)
    gt=b[:,:5].clone();gt[0,2:4]*=1.08;gt[1,2:4]*=.92
    grid=model.bucket_grid(b);out=dict(model.decode(logits,grid),bucket_logits=logits,grid=grid)
    first=model.loss(out,gt,b,b[:,:5],arm)['total'];first.backward()
    with torch.no_grad():changed=logits-.1*logits.grad
    delta=model.decode(changed,grid)['delta_roi']
    assert bool((delta[0]>0).all() and (delta[1]<0).all())
    assert bool((delta[2].abs()<1e-7).all())
    target=model.size_targets(gt,b,b[:,:5])['log_residual']
    assert bool(((delta-target).abs()<(target.abs()+1e-7)).all())


def test_soft_distribution_penalizes_wrong_same_mean():
    b=torch.tensor([[0.,0.,100.,50.,0.,.9]])
    grid=model.bucket_grid(b);gt=b[:,:5].clone()
    logits=torch.full((1,2,33),-5.);logits[:,:,0]=logits[:,:,-1]=5.
    out=dict(model.decode(logits,grid),bucket_logits=logits,grid=grid)
    assert float(model.loss(out,gt,b,b[:,:5],'decoded_control')['total'])<1e-10
    assert float(model.loss(out,gt,b,b[:,:5],'distribution')['total'])>5


@pytest.mark.parametrize('arm',model.ARMS)
def test_only_new_head_gradients_second_step_stem(arm):
    torch.manual_seed(1703);x=inputs();x[0].requires_grad_(True);x[1].requires_grad_(True)
    h=model.DistributionSizeHead();opt=torch.optim.Adam(h.parameters(),lr=.001)
    packed=dict(tensors=x,indices=list(range(8)))
    first=tool.update(h,opt,packed,list(range(8)),arm,torch)
    second=tool.update(h,opt,packed,list(range(8)),arm,torch)
    assert first['stem_grad_norm']==0 and first['terminal_grad_norm']>0
    assert second['stem_grad_norm']>0
    assert x[0].grad is None and x[1].grad is None


def test_original_model_swap_and_gt_center_angle_independence():
    x=inputs(1);x[2]=torch.tensor([[23.,41.,100.,100.1,.72,.7]])
    x[4]=torch.tensor([[1.002,1.]])
    x[3]=x[2][:,:5].clone();x[3][:,:4]*=x[4][:,[0,1,0,1]];x[5]=x[2].clone()
    assert x[2][0,2]<x[2][0,3] and x[3][0,2]>=x[3][0,3]
    y=model.DistributionSizeHead()(*x[:-1]);assert torch.equal(y['boxes_original'],x[5])
    gt=x[5][:,:5].clone();gt[:,2:4]*=1.03
    before=model.size_targets(gt,x[5],x[3])['log_residual']
    gt[:,:2]+=1000;gt[:,4]+=.8;gt[:,2:4]=gt[:,2:4].flip(-1)
    assert torch.equal(before,model.size_targets(gt,x[5],x[3])['log_residual'])
    points=model.border_points(x[5],x[3],model.bucket_grid(x[5]))
    u=torch.tensor([math.cos(.72),math.sin(.72)])
    assert torch.allclose(points[0,0,0,16,4],x[5][0,:2]+50*u,atol=1e-5)
    assert torch.allclose(points[0,:,0,:,4]+points[0,:,1,:,4],2*x[5][0,:2],atol=2e-5)


def test_border_support_and_whole_pair_guard():
    x=inputs(1);b,cb,xy,sides,rotation,m=model.residual.checked_inputs(*x[:-1])
    points=model.border_points(m,x[3],model.bucket_grid(m))
    features=torch.ones(1,8,9,9)
    value,cover=model.pool_borders(features,torch.zeros_like(x[1]),points,cb,xy,sides,rotation)
    assert bool((value==0).all() and (cover==0).all())
    value,cover=model.pool_borders(features,x[1],points+10000,cb,xy,sides,rotation)
    assert bool((value==0).all() and (cover==0).all())
    result=model.residual.deliver_sizes(torch.tensor([[.02,-.02]]),x[2],x[3],x[4],x[5])
    assert bool(result['accepted'][0])
    assert torch.equal(result['boxes_original'][:,[0,1,4,5]],m[:,[0,1,4,5]])
    for delta in (torch.tensor([[2.,0.]]),torch.tensor([[float('nan'),0.]])):
        result=model.residual.deliver_sizes(delta,x[2],x[3],x[4],x[5])
        assert torch.equal(result['boxes_original'],m)


def test_schedule_complete_natural_exposure_and_same_arms():
    a=tool.epoch_batches(5112,1);b=tool.epoch_batches(5112,2)
    assert len(a)==639 and all(len(x)==8 for x in a)
    assert sorted(i for batch in a for i in batch)==list(range(5112))
    assert a==tool.epoch_batches(5112,1) and a!=b
    assert [len(x) for x in tool.epoch_batches(10,1)]==[8,2]


def row(gt,m,c,domain='real',scale=1.,sequence='real_seq01',frame=1):
    return dict(gt=gt,b=m,midpoint=m,edge_residual=c,domain=domain,scale=scale,
                sequence=sequence,frame_id=frame,image=sequence+'_'+str(frame),size_delivery='accepted',failed_checks=[])


def test_train_gate_rejects_single_direction_and_near_damage():
    rows=[]
    for d in ('real','sim'):
        for s in (1.,.5):
            for w,c in ((92.,96.),(108.,104.),(100.,100.),(92.,92.)):
                gt=[0.,0.,100.,50.,0.];m=[0.,0.,w,w/2,0.,.9];p=[0.,0.,c,c/2,0.,.9]
                rows.append(row(gt,m,p,d,s))
    assert metrics.train_report(rows)['train_pass']
    rows[0]['edge_residual'][2:4]=[90.,45.]
    assert not metrics.train_report(rows)['train_pass']
    rows[0]['edge_residual'][2:4]=[96.,48.];rows[2]['edge_residual'][2:4]=[89.,44.5]
    assert not metrics.train_report(rows)['train_pass']


def test_metric_denominators_and_transition_identity():
    gt=[0.,0.,100.,50.,0.]
    rows=[row(gt,[0.,0.,80.,40.,0.,.9],[0.,0.,100.,50.,0.,.9]),row(gt,None,None)]
    before=metrics.size_summary(rows,'midpoint');after=metrics.size_summary(rows,'edge_residual')
    assert before['frames']==after['frames']==2 and after['outputs']==1 and after['missing']==1
    assert before['joint10']==0 and after['joint10']==1
    assert metrics.paired(rows)['wrong_to_right']==1 and metrics.paired(rows)['new_output']==0
    assert after['a']['abs_mae']==after['r']['abs_mae']==0


def test_source_and_static_check_no_tensor_load(tmp_path,monkeypatch):
    assert tool.checked_sources()['sources']
    out=tmp_path/'check';monkeypatch.setattr(tool,'check_out',lambda x:out)
    monkeypatch.setattr(tool,'load_records',lambda *a:(_ for _ in ()).throw(AssertionError('tensor load')))
    tool.run(tool.parser().parse_args(['--stage','check','--out-dir',str(out)]))
    assert tool.sealed.read(out/'completion.json')['head_updates']==0
    for arg in ('--epochs','--seed','--resume'):
        with pytest.raises(SystemExit):tool.parser().parse_args(['--stage','run','--out-dir',str(out),arg,'1'])


def test_val_gate_requires_size_and_detection_gains_and_protection():
    def method(mae=.08,riou=.8):
        return dict(sizes=dict(edges={e:dict(mae=mae,p95=.2) for e in ('long','short')},
            a={'abs_mae':.04},r={'abs_mae':.03},joint10=10),
            mean_riou_full_frame=riou,center_correct_full_frame={'numerator':10,'denominator':10},
            pure_angle_rmse_deg=1.,temporal={d:dict(mcml_max=0,mcml_mean=0.,
                tdr_w10={'numerator':1,'denominator':1},aci=.9,aci_adjacent_pairs=9) for d in ('real','sim')})
    summary={d:dict(midpoint=method(),b=method(),edge_residual=method(.07,.81),
        paired=dict(frozen_identity=True,new_output=0,lost_output=0)) for d in ('real','sim')}
    assert metrics.val_gate(summary)['val_pass']
    summary['real']['edge_residual']['mean_riou_full_frame']=.8
    assert 'real/riou' in metrics.val_gate(summary)['failed_checks']
    summary['real']['edge_residual']['mean_riou_full_frame']=.81
    summary['sim']['edge_residual']['sizes']['edges']['short']['p95']=.21
    assert 'sim/short/p95' in metrics.val_gate(summary)['failed_checks']
