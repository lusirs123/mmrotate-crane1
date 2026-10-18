"""S035 offline interventions and size-only local auxiliary objective.

No historical model is modified. GT is used by loss/diagnosis only.
"""
import torch
from torch.nn import functional as F
from crane_project.utils import port_size_local_vectors_v1 as old


def project_geometry_gradient(vectors, gradient, b, bm, xy):
    """Decompose the ACTUAL full-geometry gradient, not four new objectives.

    Gradient covectors are transformed to original image coordinates before
    Euclidean translation/radial/tangential projection and then pulled back.
    Tangential includes independent pair rotations, not only one OBB angle.
    Pair-center remainder permits nonrectangular opposite-midpoint conflicts.
    The components sum to the original gradient; finite parameter steps need
    not add because the common network/decoder is nonlinear.
    """
    n=len(b)
    eye=torch.eye(2,dtype=b.dtype,device=b.device)[None].expand(n,-1,-1)
    inverse_affine=old.residual.original.roi_delta_to_original(eye,b,bm,xy)
    forward_affine=torch.inverse(inverse_affine)
    physical=old.residual.original.roi_delta_to_original(vectors.detach().reshape(n,324,2),b,bm,xy).reshape(n,81,4,2)
    g=torch.bmm(gradient.detach().reshape(n,324,2),forward_affine.transpose(1,2)).reshape(n,81,4,2)
    translation=g.mean(2,keepdim=True).expand_as(g)
    a,c=(g[:,:,0]-g[:,:,2])/2,(g[:,:,1]-g[:,:,3])/2
    antisymmetric=torch.stack((a,c,-a,-c),2)
    d,e=physical[:,:,0]-physical[:,:,2],physical[:,:,1]-physical[:,:,3]
    u=d/d.norm(dim=-1,keepdim=True).clamp(min=1e-12)
    v=e/e.norm(dim=-1,keepdim=True).clamp(min=1e-12)
    ar,cr=(a*u).sum(-1,keepdim=True)*u,(c*v).sum(-1,keepdim=True)*v
    radial=torch.stack((ar,cr,-ar,-cr),2)
    parts=dict(translation=translation,radial=radial,tangential=antisymmetric-radial,
               pair_center=g-translation-antisymmetric)
    result={k:torch.bmm(x.reshape(n,324,2),inverse_affine.transpose(1,2)).reshape(n,81,4,2) for k,x in parts.items()}
    return result


def local_size_loss(output,gt,support,b,bm,xy,m):
    """Per-location canonical sizes, exactly the aggregate decoder's units.

    Each supported position predicts the frame's two annotated dimensions;
    support is geometric availability, not an object/foreground selector.
    Mean2 local SmoothL1 is normalized by support mass per frame, then meanN.
    No midpoint-center or vector-angle target, mask, weighting or new input.
    """
    decoded=old.decode(output['vectors'],support,b,bm,xy)
    base=old.decode(old.encode(m,b,bm,xy),support,b,bm,xy)['sizes'].detach()
    local_delta=(decoded['local_sizes'].clamp(min=1e-6)/base[:,None].clamp(min=1e-6)).log()
    target=(gt[:,2:4].detach().sort(descending=True)[0]/m[:,2:4].detach().sort(descending=True)[0]).log()
    loss=F.smooth_l1_loss(local_delta,target[:,None].expand_as(local_delta),beta=.1,reduction='none').mean(-1)
    w=support.detach().reshape(len(b),81).clamp(0,1)
    if not bool(decoded['valid'].all()) or bool((w.sum(-1)<=0).any()):raise ValueError('Invalid local auxiliary sizes')
    return ((loss*w).sum(-1)/w.sum(-1)).mean()


def losses(output,gt,support,b,bm,xy,m,arm):
    if arm not in ('size_only','size_plus_local'):raise ValueError('Unknown local-size arm')
    size=old.residual.size_loss(output['delta_roi'],gt,m,bm)
    local=local_size_loss(output,gt,support,b,bm,xy,m)
    return size+(0.1*local if arm=='size_plus_local' else 0.),size,local
