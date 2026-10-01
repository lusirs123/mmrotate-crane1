"""TRAIN-only covariance candidates; no established accuracy claim or centre term."""
import math

import torch
import torch.nn as nn
from mmdet.models.losses.utils import weight_reduce_loss

from mmrotate.models.builder import ROTATED_LOSSES
from .sym_kld_calculator import _xywha_to_gaussian, _inv2x2_safe


def covariance_shape_terms(pred, target, eps=1e-6, return_guards=False):
    """Same Gaussian map as B. Double precision is local to the extra loss.

    T is the Bhattacharyya covariance term, NOT the full box distance.
    H is a zero-at-match, epsilon-smoothed same-centre Hellinger distance.
    The smoothing changes its near-zero response and is explicitly recorded.
    """
    if pred.ndim != 2 or pred.shape[-1] != 5 or target.shape != pred.shape:
        raise ValueError('Expected matching Nx5 decoded OBBs')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    if (not torch.isfinite(pred[:, 2:]).all()
            or not torch.isfinite(target[:, 2:]).all()
            or (pred[:, 2:4] <= 0).any() or (target[:, 2:4] <= 0).any()):
        raise ValueError('Shape supervision requires finite angles and positive edges')
    _, p = _xywha_to_gaussian(pred.double())
    _, q = _xywha_to_gaussian(target.detach().double())
    ip, iq = _inv2x2_safe(p, eps), _inv2x2_safe(q, eps)
    raw_s = .5 * (torch.einsum('nij,nji->n', iq, p)
              + torch.einsum('nij,nji->n', ip, q) - 4.)
    s = raw_s.clamp(min=0., max=1e4)

    def det(x):
        return (x[:, 0, 0]*x[:, 1, 1] - x[:, 0, 1]*x[:, 1, 0]).clamp_min(eps)

    raw_t = .5 * (det((p+q)*.5).log() - .5*(det(p).log()+det(q).log()))
    t = raw_t.clamp(min=0., max=1e4)
    h = ((-torch.expm1(-t)+eps).sqrt() - math.sqrt(eps)).clamp_min(0.)
    terms = dict(symkl=s, bhattacharyya=t, hellinger=h)
    if not return_guards:
        return terms
    with torch.no_grad():
        inverse_count,det_count = 0,0
        for cov in (p,q):
            a,b,d = cov[:,0,0],cov[:,0,1],cov[:,1,1]
            raw_det = a*d-b*b
            inverse = torch.stack([d,-b,-b,a],-1)/raw_det.clamp_min(eps)[:,None]
            inverse_count += int((inverse.abs()>=1e4).any(-1).sum())
            det_count += int((raw_det<=eps).sum())
        m = (p+q)*.5
        mdet = m[:,0,0]*m[:,1,1]-m[:,0,1]*m[:,1,0]
        guards = dict(n=len(pred),edge_floor_boxes=int((pred[:,2:4]<1.).any(-1).sum()),
            target_edge_floor_boxes=int((target[:,2:4]<1.).any(-1).sum()),
            covariance_determinant_floors=det_count,mean_determinant_floors=int((mdet<=eps).sum()),
            inverse_limits=inverse_count,symkl_negative_roundoff=int((raw_s<0).sum()),
            symkl_raw_caps=int((raw_s>=1e4).sum()),symkl_loss_caps=int((s>120).sum()),
            bhattacharyya_negative_roundoff=int((raw_t<0).sum()),
            bhattacharyya_raw_caps=int((raw_t>=1e4).sum()),
            hellinger_exponential_tail_gt30=int((t>30).sum()))
    return terms,guards


@ROTATED_LOSSES.register_module()
class CovarianceShapeLoss(nn.Module):
    """E0 or E-H candidate. No parameters, assignment changes or inference use."""
    def __init__(self, mode='symkl', loss_weight=.25, eps=1e-6, reduction='mean'):
        super().__init__()
        if (mode not in ('symkl', 'hellinger') or reduction not in ('none', 'mean', 'sum')
                or not math.isfinite(loss_weight) or loss_weight <= 0
                or not math.isfinite(eps) or eps <= 0):
            raise ValueError('Invalid covariance candidate settings')
        self.mode, self.loss_weight = mode, float(loss_weight)
        self.eps, self.reduction = float(eps), reduction

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        reduction = reduction_override or self.reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Invalid reduction')
        terms = covariance_shape_terms(pred, target, self.eps)
        if pred.shape[0] == 0:
            return pred[:, 2:].sum()*0. if reduction != 'none' else pred[:, 2:].sum(-1)*0.
        value = ((torch.sqrt(1.+terms['symkl'])-1.).clamp(max=10.)
                 if self.mode == 'symkl' else terms['hellinger'])
        if weight is not None and weight.ndim > 1:
            if weight.shape != pred.shape:
                raise ValueError('Expected N or Nx5 weights')
            weight = weight[:, 0]
        return self.loss_weight*weight_reduce_loss(
            value.to(pred.dtype), weight, reduction, avg_factor)
