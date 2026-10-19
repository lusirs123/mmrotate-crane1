import math
from copy import deepcopy
import pytest
import torch
from crane_project.utils import port_size_qalign_v1 as a
from test_port_size_native_p3_v1 import fixture

torch.set_num_threads(1)

@pytest.mark.parametrize('swap,aniso',[(False,False),(True,True)])
def test_identity_sampling_and_zero_output(swap,aniso):
    x=list(fixture(swap,aniso));x[5]=x[2].clone()
    p=a.read_pair(*x[:6]);assert torch.allclose(p['features']['fixed_b'],p['features']['q_aligned'],atol=4e-6)
    for arm in a.ARMS:
        h=a.AlignedSizeHead(arm);assert sum(v.numel() for v in h.parameters())==2800
        assert torch.equal(h(*x[:6])['boxes_original'],x[5])

@pytest.mark.parametrize('angle',[-1.55,-.3,.7,1.58])
def test_independent_scalar_affine_and_raw_model_flip(angle):
    x=list(fixture());x[2]=torch.tensor([[128.,128.,100.,99.9,angle,.9]],dtype=torch.float64)
    x[4]=torch.tensor([[.998,1.]],dtype=torch.float64);x[3]=x[2][:,:5].clone();x[3][:,:4]*=x[4][:,[0,1,0,1]]
    x[5]=x[2].clone();x[5][:,:2]+=torch.tensor([1.,-2.]);x[5][:,2:4]*=torch.tensor([1.02,.98]);x[5][:,4]+=.04
    yy,xx=torch.meshgrid(torch.linspace(-.5,.5,9,dtype=torch.float64),torch.linspace(-.5,.5,9,dtype=torch.float64))
    x[0]=(3+2*xx+4*yy)[None,None].expand(1,256,9,9).clone();x[1]=torch.ones(1,1,9,9,dtype=torch.float64)
    p=a.read_pair(*x[:6]);src=p['original_points']['fixed_b'][0].tolist();expected=[]
    b,q=x[2][0].tolist(),x[5][0].tolist()
    for px,py in src:
        dx,dy=px-b[0],py-b[1];u=(dx*math.cos(b[4])+dy*math.sin(b[4]))*q[2]/b[2];v=(-dx*math.sin(b[4])+dy*math.cos(b[4]))*q[3]/b[3]
        expected.append([q[0]+u*math.cos(q[4])-v*math.sin(q[4]),q[1]+u*math.sin(q[4])+v*math.cos(q[4])])
    assert torch.allclose(p['original_points']['q_aligned'][0],torch.tensor(expected,dtype=torch.float64),atol=1e-10)
    g=p['source_local']['q_aligned'];target=(3+2*g[:,:,0]+4*g[:,:,1]).reshape(1,1,9,9).expand_as(x[0]);valid=p['inside']['q_aligned'].expand_as(x[0])
    assert torch.allclose(p['features']['q_aligned'][valid],target[valid],atol=1e-10)
    assert x[2][0,2]>x[2][0,3] and x[3][0,2]<x[3][0,3]

def test_support_missing_and_constant():
    x=list(fixture());x[0].fill_(7);p=a.read_pair(*x[:6])
    for arm in a.ARMS:assert torch.allclose(p['features'][arm][p['support'].expand_as(x[0])>0],torch.tensor(7.),atol=1e-6)
    x[1][:,:,:,0]=0;p=a.read_pair(*x[:6]);assert bool((p['support']==0).any())
    x[1].zero_()
    for arm in a.ARMS:
        z=a.AlignedSizeHead(arm)(*x[:6]);assert not z['vector_valid'].any() and torch.equal(z['boxes_original'],x[5])
        e=fixture(empty=True);assert a.AlignedSizeHead(arm)(*e[:6])['boxes_original'].shape==(0,6)

@pytest.mark.parametrize('arm',a.ARMS)
def test_gradients_resume_and_fallback(arm):
    x=list(fixture());x[0].requires_grad_();gt=x[5][:,:5].clone();gt[:,2]*=.95;gt[:,3]*=1.04
    h=a.AlignedSizeHead(arm);opt=torch.optim.Adam(h.parameters(),lr=.001)
    def step(h,opt):
        opt.zero_grad();o=h(*x[:6]);loss=a.size_loss(o,gt,x[5],x[3]);loss.backward();assert x[0].grad is None
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters());opt.step()
    step(h,opt);state=deepcopy(h.state_dict());os=deepcopy(opt.state_dict());clone=a.AlignedSizeHead(arm);clone.load_state_dict(state);other=torch.optim.Adam(clone.parameters(),lr=.001);other.load_state_dict(os)
    step(h,opt);step(clone,other);assert all(torch.equal(v,clone.state_dict()[k]) for k,v in h.state_dict().items())
    with torch.no_grad():h.terminal.bias.fill_(float('nan'))
    o=h(*x[:6]);assert torch.equal(o['boxes_original'],x[5])

def test_cross_edge_retains_nonintersection_harm():
    rows=[dict(domain='sim',gt=[0,0,100,50,0],midpoint=[0,0,90,50,0,1],edge_residual=[0,0,95,51,0,1]),dict(domain='sim',gt=[0,0,100,50,0],midpoint=[0,0,90,50,0,1],edge_residual=[0,0,85,52,0,1])]
    cell=a.cross_edge(rows)['sim'];v=cell['edges']['long']
    assert v['denominator']==2 and v['corrected_and_other_harmed']==1 and cell['both_harmed']==1
    assert v['all_other_positive_harm_sum_in_subset']>v['intersection_other_harm_sum'] and v['beneficial_target_count']==1

def test_nonvisual_identical_and_no_gt_forward_argument():
    import inspect
    assert 'gt' not in inspect.signature(a.read_pair).parameters and 'gt' not in inspect.signature(a.AlignedSizeHead.forward).parameters
    x=fixture();seen={}
    for arm in a.ARMS:
        h=a.AlignedSizeHead(arm)
        handle=h.stem.register_forward_pre_hook(lambda module,args:seen.update({arm:args[0].detach().clone()}))
        h(*x[:6]);handle.remove()
    assert torch.equal(seen['fixed_b'][:,256:],seen['q_aligned'][:,256:])
