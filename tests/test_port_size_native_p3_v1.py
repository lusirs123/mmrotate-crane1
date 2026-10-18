import io
import math
import pytest
import torch
from crane_project.utils import port_size_native_p3_v1 as n
from crane_project.utils import port_geometry_refine_g_v1 as g

torch.set_num_threads(1)


def fixture(swap=False, anisotropic=False, empty=False):
    b = torch.tensor([[128., 128., 112., 56., .2, .9]])
    if swap:
        b[:, 2:4] = b[:, 2:4].flip(-1); b[:, 4] -= math.pi/2
    xy = torch.tensor([[.998, 1.]]) if anisotropic else torch.ones(1, 2)
    bm = b[:, :5].clone(); bm[:, :4] *= xy[:, [0, 1, 0, 1]]
    m = b.clone(); m[:, :2] += 1.; m[:, 2:4] *= 1.01; m[:, 4] += .01
    meta = dict(ori_shape=(256, 256, 3), img_shape=(256, 256, 3), pad_shape=(256, 256, 3),
                scale_factor=[float(xy[0, 0]), float(xy[0, 1])]*2, flip=False)
    p3 = torch.randn(1, 256, 32, 32)
    if empty: b, bm, xy, m = b[:0], bm[:0], xy[:0], m[:0]
    roi, support, _ = g.sample_local(p3, bm, meta, 'aligned')
    return roi, support, b, bm, xy, m, p3, meta


@pytest.mark.parametrize('swap,anisotropic,empty', [(False,False,False),(True,True,False),(False,False,True)])
def test_both_arms_neutral_and_identical_support(swap, anisotropic, empty):
    x = fixture(swap, anisotropic, empty); a = n.read_points(*x[:6], 'cached_p3')
    b = n.read_points(*x[:6], 'native_p3', *x[6:]); h = n.CommonPointSizeHead()
    assert sum(p.numel() for p in h.parameters()) == 2328
    for key in ('masks','local','geometry','model_points','inside'): assert torch.equal(a[key], b[key])
    for r in (a,b):
        out=h(r,*x[2:6]); assert torch.equal(out['boxes_original'],x[5])
        assert torch.equal(out['delta_roi'],torch.zeros(len(x[2]),2))


def test_constant_and_affine_equivalence_inside():
    x=list(fixture()); yy,xx=torch.meshgrid(torch.arange(32.),torch.arange(32.))
    x[6]=(2*xx+yy+3)[None,None].expand(1,256,32,32).clone()
    x[0],x[1],_=g.sample_local(x[6],x[3],x[7],'aligned')
    a=n.read_points(*x[:6],'cached_p3');b=n.read_points(*x[:6],'native_p3',*x[6:])
    valid=a['inside'].expand_as(a['samples'])
    assert torch.allclose(a['samples'][valid],b['samples'][valid],atol=2e-5,rtol=1e-6)


def test_native_has_evidence_not_recoverable_from_roi():
    x=list(fixture()); x[6].zero_(); x[0],x[1],_=g.sample_local(x[6],x[3],x[7],'aligned')
    _,oldpoints=g.sampling_grid(x[3],(32,32),'aligned')
    oldcells=set()
    for a,b in (oldpoints.reshape(-1,2)/8).floor().long().tolist():
        oldcells.update((a+dx,b+dy) for dx in (0,1) for dy in (0,1))
    native=n.read_points(*x[:6],'native_p3',*x[6:]); chosen=None
    for pt,valid in zip(native['model_points'][0].tolist(),native['inside'].reshape(-1).tolist()):
        if not valid:continue
        a,b=[math.floor(v/8) for v in pt]
        for dx,dy in ((0,0),(0,1),(1,0),(1,1)):
            if (a+dx,b+dy) not in oldcells and 0<=a+dx<32 and 0<=b+dy<32:
                chosen=(a+dx,b+dy);break
        if chosen:break
    assert chosen
    x[6][0,0,chosen[1],chosen[0]]=1
    changed,_,_=g.sample_local(x[6],x[3],x[7],'aligned')
    assert torch.equal(changed,x[0])
    direct=n.read_points(*x[:6],'native_p3',*x[6:])
    assert direct['samples'].abs().max()>0


def test_no_detector_gradient_and_exact_reload():
    x=fixture(); r=n.read_points(*x[:6],'native_p3',*x[6:]);r['samples'].requires_grad_(True)
    h=n.CommonPointSizeHead()
    with torch.no_grad(): h.u.weight.fill_(.01);h.v.weight.fill_(.02)
    out=h(r,*x[2:6]); grads=torch.autograd.grad(out['delta_roi'].sum(),list(h.parameters()))
    assert all(torch.isfinite(v).all() for v in grads)
    assert sum(v.abs().sum() for v in grads[:4])>0 and r['samples'].grad is None
    buffer=io.BytesIO();torch.save(h.state_dict(),buffer);buffer.seek(0)
    clone=n.CommonPointSizeHead();clone.load_state_dict(torch.load(buffer))
    assert torch.equal(clone(r,*x[2:6])['bucket_logits'],out['bucket_logits'])


def test_delivery_fallback_and_missing_are_inherited():
    x=fixture(); b,bm,xy,m=x[2:6]
    for delta in (torch.tensor([[float('nan'),0.]]),torch.tensor([[10.,-10.]])):
        out=n.old.residual.deliver_sizes(delta,b,bm,xy,m)
        assert not bool(out['accepted'][0]) and torch.equal(out['boxes_original'],m)
    out=n.old.residual.deliver_sizes(torch.tensor([[.001,-.001]]),b,bm,xy,m)
    assert torch.equal(out['boxes_original'][:,[0,1,4,5]],m[:,[0,1,4,5]])


def test_frozen_native_graph_rejected():
    x=list(fixture());x[6].requires_grad_(True)
    with pytest.raises(ValueError):n.read_points(*x[:6],'native_p3',*x[6:])


def test_protocol_no_training_or_test():
    p=n.protocol(); assert p['limits']['training_updates']==0 and p['limits']['test_reads']==0
    assert p['training']['present_views']==5112 and p['parameter_count']==2328


def test_same_cells_do_not_imply_added_information():
    from crane_project.tools.audit_port_size_native_p3_v1 import linear_evidence
    # Four source nodes are fully recoverable from the old samples. The new
    # center interpolation changes coefficients but adds no independent value.
    old=torch.tensor([[0.,0.],[8.,0.],[0.,8.],[8.,8.]])
    result=linear_evidence(old,torch.tensor([[4.,4.]]),4,4,(32,32))
    assert result['old_rank']==4 and not result['witness']
    assert not result['valid_witness_constructed'] and result['old_null_response'] is None and result['native_response'] is None
    # Two diagonal samples touch ALL four nodes, yet only constrain two DOFs.
    # A new coefficient combination can still add evidence inside same cells.
    partial=torch.tensor([[2.,2.],[6.,6.]])
    result=linear_evidence(partial,torch.tensor([[2.,6.]]),4,4,(32,32))
    assert result['old_rank']==2 and result['witness']


def test_original_model_swap_differs_independent_coordinates():
    b=torch.tensor([[128.,128.,100.,100.1,.3,.9]])
    xy=torch.tensor([[1.002,1.]])
    bm=b[:,:5].clone();bm[:,:4]*=xy[:,[0,1,0,1]];m=b.clone()
    assert bool(b[0,2]<b[0,3]) and not bool(bm[0,2]<bm[0,3])
    roi=torch.zeros(1,256,9,9);sup=torch.ones(1,1,9,9)
    a=n.read_points(roi,sup,b,bm,xy,m,'cached_p3')
    # Axis0 is M RAW width here because MODEL-B is not swapped. Independently
    # compute bucket16(log0), plus side0, tangent4(center), no shared helper.
    expected=torch.tensor([(128.+50.*math.cos(.3))*1.002,128.+50.*math.sin(.3)])
    got=a['model_points'].reshape(1,2,2,33,9,2)[0,0,0,16,4]
    assert torch.allclose(got,expected,atol=2e-5,rtol=1e-6)
    dx=float(expected[0])-float(bm[0,0]);dy=float(expected[1])-float(bm[0,1])
    local=torch.tensor([(dx*math.cos(.3)+dy*math.sin(.3))/(float(bm[0,2])*1.5),
                        (-dx*math.sin(.3)+dy*math.cos(.3))/(float(bm[0,3])*1.5)])
    gotlocal=a['local'].permute(0,2,3,1).reshape(1,2,2,33,9,2)[0,0,0,16,4]
    assert torch.allclose(gotlocal,local,atol=2e-6,rtol=1e-6)
