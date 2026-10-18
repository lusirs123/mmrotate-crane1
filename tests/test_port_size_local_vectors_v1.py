import io
import math
from copy import deepcopy
import pytest
import torch
from crane_project.utils import port_size_local_vectors_v1 as v
from test_port_size_native_p3_v1 import fixture

torch.set_num_threads(1)


@pytest.mark.parametrize('swap,anisotropic', [(False,False),(True,True)])
def test_neutral_identity_and_missing(swap, anisotropic):
    x=fixture(swap,anisotropic);h=v.LocalVectorSizeHead()
    out=h(*x[:6]);assert torch.equal(out['boxes_original'],x[5])
    assert torch.equal(out['delta_roi'],torch.zeros(1,2))
    z=fixture(empty=True);assert h(*z[:6])['boxes_original'].shape==(0,6)


@pytest.mark.parametrize('angle', [-math.pi/2-1e-5,-1.1,0.,math.pi/2+1e-5])
def test_original_gt_oracle_and_d4(angle):
    x=fixture(True,True);b,bm,xy=x[2:5]
    gt=torch.tensor([[142.,111.,113.,55.,angle]])
    vec=v.encode(gt,b,bm,xy);d=v.decode(vec,x[1],b,bm,xy)
    assert torch.allclose(d['sizes'],torch.tensor([[113.,55.]]),atol=2e-5,rtol=1e-6)
    for p in v.PERMUTATIONS:
        perm=vec[:,:,list(p)]
        assert v.geometric_loss(perm,gt,x[1],b,bm,xy)[0]<1e-12
        assert torch.allclose(v.decode(perm,x[1],b,bm,xy)['sizes'],d['sizes'],atol=2e-5)
    exchanged=gt.clone();exchanged[:,2:4]=gt[:,2:4].flip(-1);exchanged[:,4]+=math.pi/2
    assert v.geometric_loss(vec,exchanged,x[1],b,bm,xy)[0]<1e-10


def test_translation_outside_window_and_near_square():
    x=fixture();b,bm,xy=x[2:5]
    for wh in ((100.,99.999),(100.,100.)):
        gt=torch.tensor([[900.,-300.,*wh,.73]])
        vec=v.encode(gt,b,bm,xy)
        assert vec.abs().max()>1 and torch.isfinite(vec).all()
        assert torch.allclose(v.decode(vec,x[1],b,bm,xy)['sizes'],torch.tensor([wh]),atol=2e-4)


def test_inverse_resize_before_norm_and_axis_flip():
    x=list(fixture());x[2]=torch.tensor([[128.,128.,100.,99.9,.2,.9]])
    x[4]=torch.tensor([[.998,1.]])
    x[3]=x[2][:,:5].clone();x[3][:,:4]*=x[4][:,[0,1,0,1]];x[5]=x[2].clone()
    assert x[2][0,2]>x[2][0,3] and x[3][0,2]<x[3][0,3]
    gt=torch.tensor([[130.,131.,95.,42.,.7]])
    vec=v.encode(gt,*x[2:5]);assert torch.allclose(v.decode(vec,x[1],*x[2:5])['sizes'],gt[:,2:4],atol=2e-5)
    assert torch.equal(v.LocalVectorSizeHead()(*x[:6])['boxes_original'],x[5])


def test_invalid_support_and_vectors_fallback():
    x=list(fixture());h=v.LocalVectorSizeHead();x[1]=torch.zeros_like(x[1])
    out=h(*x[:6]);assert not out['vector_valid'].any() and torch.equal(out['boxes_original'],x[5])
    x=list(fixture());base=v.encode(x[5],*x[2:5])
    for bad in (float('nan'),0.):
        vec=base.clone();vec[:,0]=bad
        assert not v.decode(vec,x[1],*x[2:5])['valid'].any()
    x[1].reshape(1,81)[:,0]=0
    vec=base.clone();vec[:,0]=float('nan')
    assert v.decode(vec,x[1],*x[2:5])['valid'].all()


def test_local_conflict_mean_and_gradient():
    x=fixture();base=v.encode(x[5],*x[2:5]).requires_grad_()
    changed=base.clone();changed[:,0,0,0]+=0.1
    d=v.decode(changed,x[1],*x[2:5]);w=x[1].reshape(1,81)
    expected=(d['local_sizes']*w[:,:,None]).sum(1)/w.sum(1)[:,None]
    assert torch.equal(expected,d['sizes']) and d['variance'].max()>0
    d['sizes'].sum().backward();assert torch.isfinite(base.grad).all()
    zero=torch.zeros_like(base,requires_grad=True)
    v.decode(zero,x[1],*x[2:5])['sizes'].sum().backward()
    assert torch.isfinite(zero.grad).all()


@pytest.mark.parametrize('arm',v.ARMS)
def test_adam_gradients_and_exact_resume(arm):
    x=list(fixture());x[0].requires_grad_();h=v.LocalVectorSizeHead();opt=torch.optim.Adam(h.parameters(),lr=.001)
    gt=x[5][:,:5].clone();gt[:,2]*=.95;gt[:,3]*=1.04
    def step(head,optimizer):
        optimizer.zero_grad();out=head(*x[:6]);loss=v.losses(out,gt,x[1],*x[2:6],arm)[0]
        loss.backward();grads=[p.grad.clone() for p in head.parameters()]
        assert all(torch.isfinite(g).all() for g in grads) and x[0].grad is None
        optimizer.step();return grads
    first=step(h,opt);assert sum(g.abs().sum() for g in first[:4])==0 and first[-2].abs().sum()>0
    buf=io.BytesIO();torch.save(dict(h=h.state_dict(),o=opt.state_dict(),rng=torch.get_rng_state()),buf);buf.seek(0)
    saved=torch.load(buf);clone=v.LocalVectorSizeHead();clone.load_state_dict(saved['h'])
    op=torch.optim.Adam(clone.parameters(),lr=.001);op.load_state_dict(saved['o'])
    torch.set_rng_state(saved['rng']);second=step(h,opt)
    torch.set_rng_state(saved['rng']);step(clone,op)
    assert sum(g.abs().sum() for g in second[:4])>0
    assert all(torch.equal(h.state_dict()[k],clone.state_dict()[k]) for k in h.state_dict())
    assert torch.equal(h(*x[:6])['vectors'],clone(*x[:6])['vectors'])
