import math
import torch
from crane_project.utils import port_size_local_vectors_v1 as old
from crane_project.utils import port_size_auxiliary_conflict_v1 as aux
from test_port_size_native_p3_v1 import fixture

torch.set_num_threads(1)


def test_original_gradient_projection_reconstructs_and_separates_geometry():
    x=fixture(True,True);vectors=old.encode(x[5],*x[2:5]);g=torch.randn_like(vectors)
    parts=aux.project_geometry_gradient(vectors,g,*x[2:5])
    assert torch.allclose(sum(parts.values()),g,atol=2e-6,rtol=2e-6)
    # In original coordinates the common translation has identical4 point
    # gradients and no opposite-pair difference; remainder has equal opposites.
    for key in ('translation','pair_center'):
        assert torch.allclose(parts[key][:,:,0],parts[key][:,:,2],atol=2e-6)
        assert torch.allclose(parts[key][:,:,1],parts[key][:,:,3],atol=2e-6)


def test_local_loss_zero_for_correct_sizes_despite_center_and_angle():
    x=fixture(True,True);b,bm,xy,m=x[2:6]
    gt=torch.tensor([[145.,112.,100.,45.,-.8]])
    different=gt.clone();different[:,:2]+=torch.tensor([[35.,-20.]]);different[:,4]+=.6
    vectors=old.encode(different,b,bm,xy).requires_grad_()
    output=dict(vectors=vectors)
    # The exact neutral roundoff denominator is part of the original decoder.
    ref=old.decode(old.encode(m,b,bm,xy),x[1],b,bm,xy)['sizes']
    targets=gt.clone();targets[:,2:4]*=m[:,2:4].sort(descending=True)[0]/ref
    local=aux.local_size_loss(output,targets,x[1],b,bm,xy,m)
    geometry=old.geometric_loss(vectors,gt,x[1],b,bm,xy)[0]
    assert local<1e-11 and geometry>.01
    grad=torch.autograd.grad(local,vectors)[0];assert grad.abs().max()<1e-6


def test_local_supervision_detects_cancellation_hidden_by_aggregation():
    x=fixture();b,bm,xy,m=x[2:6];base=old.encode(m,b,bm,xy)
    # 40 positions shrink,40 enlarge,1 neutral; average size stays neutral.
    factor=torch.ones(1,81,1,1);factor[:,0:40]=.9;factor[:,40:80]=1.1
    centers=base.mean(2,keepdim=True);vectors=centers+(base-centers)*factor
    support=torch.ones_like(x[1]);gt=m[:,:5]
    ref=old.decode(base,support,b,bm,xy)['sizes'];decoded=old.decode(vectors,support,b,bm,xy)['sizes']
    assert torch.allclose(ref,decoded,atol=2e-5)
    assert aux.local_size_loss(dict(vectors=vectors),gt,support,b,bm,xy,m)>.02


def test_full_geometry_can_reduce_loss_while_harming_exact_lengths():
    x=[t.double() if torch.is_tensor(t) else t for t in fixture()]
    b,bm,xy,m=x[2:6];gt=m[:,:5].clone();pose=gt.clone();pose[:,4]+=.3
    vectors=old.encode(pose,b,bm,xy).requires_grad_()
    loss=old.geometric_loss(vectors,gt,x[1],b,bm,xy)[0]
    gradient=torch.autograd.grad(loss,vectors)[0];after=vectors-.1*gradient
    before_sizes=old.decode(vectors,x[1],b,bm,xy)['sizes'];after_sizes=old.decode(after,x[1],b,bm,xy)['sizes']
    target=gt[:,2:4].sort(descending=True)[0]
    assert (before_sizes/target-1).abs().max()<1e-12
    assert old.geometric_loss(after,gt,x[1],b,bm,xy)[0]<loss
    assert (after_sizes/target-1).abs().mean()>1e-6


def test_control_loss_and_adam_update_exactly_equal_original_V():
    x=fixture();gt=x[5][:,:5].clone();gt[:,2:4]*=torch.tensor([[.95,1.05]])
    torch.manual_seed(1703);a=old.LocalVectorSizeHead();b=old.LocalVectorSizeHead();b.load_state_dict(a.state_dict())
    oa=torch.optim.Adam(a.parameters(),lr=.001);ob=torch.optim.Adam(b.parameters(),lr=.001)
    for _ in range(2):
        oa.zero_grad();ob.zero_grad();u=a(*x[:6]);v=b(*x[:6])
        la=old.losses(u,gt,x[1],*x[2:6],'size_only')[0]
        lb=aux.losses(v,gt,x[1],*x[2:6],'size_only')[0]
        assert torch.equal(la,lb);la.backward();lb.backward()
        assert all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()))
        oa.step();ob.step();assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())


def test_local_size_loss_masks_only_geometric_support_and_has_finite_gradient():
    x=fixture();b,bm,xy,m=x[2:6];v=old.encode(m,b,bm,xy).requires_grad_();gt=m[:,:5].clone();gt[:,2]*=.9
    loss=aux.local_size_loss(dict(vectors=v),gt,x[1],b,bm,xy,m);loss.backward()
    assert torch.isfinite(v.grad).all() and v.grad.abs().sum()>0
