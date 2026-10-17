"""TRAIN-only seven single-axis proposals on the unchanged 2712-parameter I.

No slot/type/perturbation/GT flag enters readout. Natural inference is the
original GeometryConditionSizeHead.forward, unchanged. GT only constructs
OFFLINE inputs/targets. Q axes are NEVER sorted or rematched.
"""
import math
import torch
from torch.nn import functional as F
from crane_project.utils.port_geometry_size_condition_v1 import GeometryConditionSizeHead
from crane_project.utils import port_geometry_size_distribution_v1 as base

VERSION='port_geometry_size_proposal_v1'
H=.0625
SLOTS=('natural','long_plus','long_minus','short_plus','short_minus','long_exact','short_exact')


def proposals(m,gt,bm,arm):
    if arm not in ('repeat7','proposal7'):raise ValueError('Unknown arm')
    fixed=base.size_targets(gt,m,bm)  # Assign GT by NATURAL M once.
    q=m.detach()[:,None,:].repeat(1,7,1)
    if arm=='proposal7':
        long_axis=(m[:,2]>=m[:,3]).long()  # canonical long -> raw column
        long_axis=1-long_axis;short_axis=1-long_axis
        for slot,axis,sign in [(1,long_axis,1),(2,long_axis,-1),(3,short_axis,1),(4,short_axis,-1)]:
            indices=torch.arange(len(m),device=m.device)
            q[indices,slot,2+axis]*=math.exp(sign*H)
        for slot,axis in ((5,long_axis),(6,short_axis)):
            indices=torch.arange(len(m),device=m.device)
            q[indices,slot,2+axis]=fixed['raw_edges'][indices,axis]
    goal=fixed['roi_edges'][:,None,:].expand(-1,7,-1)
    flat_q=q.reshape(-1,6);flat_bm=bm[:,None,:].expand(-1,7,-1).reshape(-1,5)
    roi_q=base.residual.raw_roi_pair(flat_q[:,2:4],flat_bm).reshape(len(m),7,2)
    target=(goal/roi_q).log().detach()
    return q,target


def readout(head,roi,support,b,bm,xy,m,q):
    """Same I scoring; shared deterministic stem once/view, seven Q states.

    Natural M is validated, Q gets finite/identity checks, not M-size guards.
    Shared stem is algebraically the repeated-slot stem: no BN/dropout/state.
    """
    b,cb,xy,sides,rotation,natural=base.residual.checked_inputs(roi,support,b,bm,xy,m)
    n,k=len(m),base.SETTINGS['bins'];slots=q.shape[1]
    if q.shape!=(n,slots,6) or slots<1 or q.device!=m.device or q.dtype!=m.dtype:
        raise ValueError('Proposal shape/device/dtype')
    if not bool(torch.isfinite(q).all()) or bool((q[:,:,2:4]<=0).any()):raise ValueError('Invalid Q edges')
    if not torch.equal(q[:,:,[0,1,4,5]],natural[:,None,[0,1,4,5]].expand(-1,slots,-1)):
        raise ValueError('Q changed natural center/angle/score')
    q=q.detach();expand=lambda x:x[:,None].expand(-1,slots,*x.shape[1:]).reshape(n*slots,*x.shape[1:])
    r=base.SETTINGS['roi_size'];coord=torch.linspace(-.5,.5,r,device=b.device,dtype=b.dtype)
    yy,xx=torch.meshgrid(coord,coord)
    fixed=torch.stack((xx,yy),0)[None].expand(n,-1,-1,-1)
    feature=head.stem(torch.cat((roi.detach(),support.detach(),fixed),1))
    qb,qbm,qxy,qcb,qsides,qrotation=[expand(x) for x in (b,bm,xy,cb,sides,rotation)]
    qm=q.reshape(n*slots,6);grid=base.bucket_grid(qm)
    mu,bu=(base.residual.raw_roi_pair(x[:,2:4],qbm) for x in (qm,qb))
    geometry=torch.cat(((mu/bu).log(),(qcb[:,2:4]/qsides).log()),-1).detach()
    paired,coverage=base.pool_borders(expand(feature),expand(support),base.border_points(qm,qbm,grid),qcb,qxy,qsides,qrotation)
    conditioned=geometry[:,None,None]*grid[None,None,:,None]
    inputs=torch.cat((paired,conditioned.expand(n*slots,2,k,4),coverage,grid[None,None,:,None].expand(n*slots,2,k,1)),-1)
    logits=torch.stack((head.u(inputs[:,0]).squeeze(-1),head.v(inputs[:,1]).squeeze(-1)),1)
    decoded=base.decode(logits,grid)
    return dict(delta_roi=decoded['delta_roi'].reshape(n,slots,2),bucket_logits=logits.reshape(n,slots,2,k),grid=grid,
                border_support=coverage.reshape(n,slots,2,k,2),geometry_inputs=geometry.reshape(n,slots,4))


def loss(output,target):
    delta=output['delta_roi']
    if delta.shape!=target.shape or not delta.numel() or not bool(torch.isfinite(delta).all() and torch.isfinite(target).all()):
        raise ValueError('Finite unfiltered proposal targets required')
    pointwise=F.smooth_l1_loss(delta,target,beta=.1,reduction='none')
    # Two axes, then seven slots, then logical batch. No weight normalization.
    return dict(total=pointwise.mean(-1).mean(-1).mean(),pointwise=pointwise)


def delivery(output,b,bm,xy,m,q):
    """Diagnostic Q candidate delivered with ORIGINAL natural-M whole-pair guards.

    Online never uses Q. Effective delta is expressed relative to natural M so
    inherited checks stay active, including natural raw order/direction.
    """
    n,slots=q.shape[:2];expand=lambda x:x[:,None].expand(-1,slots,*x.shape[1:]).reshape(n*slots,*x.shape[1:])
    qm=q.reshape(n*slots,6);nm=expand(m);model=expand(bm)
    offset=base.residual.raw_roi_pair((qm[:,2:4]/nm[:,2:4]).log(),model)
    return base.residual.deliver_sizes(output['delta_roi'].reshape(n*slots,2)+offset,expand(b),model,expand(xy),nm)
