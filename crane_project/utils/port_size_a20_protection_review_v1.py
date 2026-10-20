"""Offline A20 review and loss prototype; not connected to any trainer/model.

GT and frozen errors are training-only. This module adds no inference rule.
All error quantities are dimensionless; .001 means .1 percentage points.
"""
import math

VERSION = 'port_size_a20_protection_review_v1'
NEAR = .03
GAIN_TOL = 1e-6
RATIO_WEIGHT = .25  # Already used by A048; this is a control, not novelty.
PROTECT_WEIGHT = .1
EPSILON = 0.


def relative_errors(box, gt):
    sides, target = sorted(box[2:4], reverse=True), sorted(gt[2:4], reverse=True)
    if any(not math.isfinite(x) or x <= 0 for x in sides+target):
        raise ValueError('Positive finite canonical sizes required')
    signed = [x/y-1 for x,y in zip(sides,target)]
    logs = [math.log(x/y) for x,y in zip(sides,target)]
    return dict(signed=signed, absolute=list(map(abs,signed)),
                a=sum(logs)/2, r=logs[0]-logs[1])


def fixed_references(m_errors, a_errors):
    """Static per-view/edge memberships; the tighter error wins on overlap."""
    if len(m_errors) != 2 or len(a_errors) != 2 or any(
            not math.isfinite(x) or x < 0 for x in list(m_errors)+list(a_errors)):
        raise ValueError('Two finite nonnegative frozen error pairs required')
    near = [x <= NEAR for x in m_errors]
    gain = [y < x-GAIN_TOL for x,y in zip(m_errors,a_errors)]
    active = [n or g for n,g in zip(near,gain)]
    refs = [min(([m] if n else [])+([a] if g else [])) if on else 0.
            for m,a,n,g,on in zip(m_errors,a_errors,near,gain,active)]
    return dict(near=near,gain=gain,active=active,reference=refs)


def capped_retention(m_errors,a_errors,new_errors):
    """Offline evaluation: gains elsewhere cannot cancel lost teacher gains."""
    if len(m_errors)!=len(a_errors) or len(m_errors)!=len(new_errors) or any(
            not math.isfinite(x) or x<0 for seq in (m_errors,a_errors,new_errors) for x in seq):
        raise ValueError('Aligned finite nonnegative error lists required')
    triples=[(m,a,n) for m,a,n in zip(m_errors,a_errors,new_errors) if a<m-GAIN_TOL]
    baseline=sum(m-a for m,a,n in triples)
    retained=sum(min(max(m-n,0.),m-a) for m,a,n in triples)
    return dict(count=len(triples),baseline_gain=baseline,retained_gain=retained,
                fraction=retained/baseline if baseline else None)


def prototype_losses(log_errors, m_errors, a_errors, arm, base_loss=None):
    """Synthetic CPU autograd prototype, NO training/selection API.

    log_errors: pre-guard canonical log(predicted_edge / GT_edge).
    Reference memberships are frozen from DELIVERED M/A20, detached here.
    Protection uses mean2N (inactive entries zero), never active-set mean.
    Callers must preserve original base_loss graph when integrating in future.
    """
    import torch
    from torch.nn import functional as F
    if arm not in ('C','R','P'):
        raise ValueError('Expected C/R/P')
    if (log_errors.ndim != 2 or log_errors.shape[1] != 2 or not len(log_errors)
            or not log_errors.is_floating_point() or not bool(torch.isfinite(log_errors).all())):
        raise ValueError('Nonempty finite N x 2 log errors required')
    for value in (m_errors,a_errors):
        if (value.shape != log_errors.shape or value.dtype != log_errors.dtype
                or value.device != log_errors.device or not bool(torch.isfinite(value).all())
                or bool((value < 0).any())):
            raise ValueError('Frozen reference shape/dtype/device/values differ')
    m,a = m_errors.detach(),a_errors.detach()
    near = m <= NEAR
    gain = a < m-GAIN_TOL
    active = near | gain
    reference = torch.where(near & gain,torch.minimum(m,a),torch.where(near,m,a))
    reference = torch.where(active,reference,torch.zeros_like(reference)).detach()
    size = F.smooth_l1_loss(log_errors,torch.zeros_like(log_errors),beta=.1) if base_loss is None else base_loss
    if size.ndim or not bool(torch.isfinite(size)):
        raise ValueError('Finite scalar base loss required')
    ratio = F.smooth_l1_loss(log_errors[:,0]-log_errors[:,1],torch.zeros_like(log_errors[:,0]),beta=.1)
    error = torch.expm1(log_errors).abs()
    if not bool(torch.isfinite(error).all()):
        raise ValueError('Error overflow; do not clip targets or silently mask')
    hinge = torch.where(active,F.relu(error-reference-EPSILON),torch.zeros_like(error))
    protect = hinge.mean()
    # C reuses the supplied original graph exactly; no algebraic zero branch.
    total = size if arm == 'C' else size+RATIO_WEIGHT*ratio
    if arm == 'P':
        total = total+PROTECT_WEIGHT*protect
    return dict(total=total,size=size,ratio=ratio,protect=protect,error=error,
                near=near,gain=gain,active=active,reference=reference,hinge=hinge)


def from_delta(delta_roi,gt,m,bm,m_errors,a_errors,arm):
    """CPU-tested axis/graph adapter only; never patches the sealed head."""
    import torch
    from crane_project.utils import port_geometry_midpoint_edge_residual_v1 as original
    target = original.size_targets(gt,m,bm)['log_residual']
    errors = delta_roi-target
    swap = (m[:,2] < m[:,3]) ^ (bm[:,2] < bm[:,3])
    canonical = torch.where(swap[:,None],errors.flip(-1),errors)
    base = original.size_loss(delta_roi,gt,m,bm)
    return prototype_losses(canonical,m_errors,a_errors,arm,base_loss=base)
