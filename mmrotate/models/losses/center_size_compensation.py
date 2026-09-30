"""Positive-only, scale-normalized centre/size supervision for decoded OBBs."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.losses.utils import weight_reduce_loss

from mmrotate.models.builder import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class CenterSizeCompensationLoss(nn.Module):
    """No angle target, learnable parameters or inference dependency.

    Per box: mean SmoothL1((pred_xy - gt_xy) / gt_short)
           + mean SmoothL1(log(pred_sorted_edges / gt_sorted_edges)).
    Sorting makes sizes invariant to the equivalent w/h + 90deg OBB encoding.
    Both terms are invariant to a common positive coordinate scale (above eps).
    The GT is detached; gradients enter only the prediction's first 4 columns.
    """

    def __init__(self, beta=0.1, loss_weight=0.25, eps=1e-6,
                 reduction='mean'):
        super().__init__()
        if (not all(math.isfinite(float(x)) for x in (beta, loss_weight, eps))
                or beta <= 0 or loss_weight <= 0 or eps <= 0
                or reduction not in ('none', 'mean', 'sum')):
            raise ValueError('Invalid center-size compensation settings')
        self.beta = float(beta)
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        reduction = reduction_override or self.reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Invalid reduction')
        if pred.ndim != 2 or pred.shape[-1] != 5 or target.shape != pred.shape:
            raise ValueError('Expected matching Nx5 decoded OBBs')
        if pred.shape[0] == 0:
            return pred[:, :4].sum() * 0.0
        target = target.detach()
        if (not torch.isfinite(pred[:, :4]).all()
                or not torch.isfinite(target[:, :4]).all()
                or (pred[:, 2:4] <= 0).any() or (target[:, 2:4] <= 0).any()):
            raise ValueError('Compensation requires finite centres and positive edges')
        pred_edges = pred[:, 2:4].sort(dim=-1, descending=True).values.clamp_min(self.eps)
        gt_edges = target[:, 2:4].sort(dim=-1, descending=True).values.clamp_min(self.eps)
        center = (pred[:, :2] - target[:, :2]) / gt_edges[:, 1:2]
        size = pred_edges.log() - gt_edges.log()
        center_loss = F.smooth_l1_loss(center, torch.zeros_like(center),
                                     reduction='none', beta=self.beta).mean(-1)
        size_loss = F.smooth_l1_loss(size, torch.zeros_like(size),
                                   reduction='none', beta=self.beta).mean(-1)
        if weight is not None and weight.ndim > 1:
            if weight.shape != pred.shape:
                raise ValueError('Expected N or Nx5 regression weights')
            weight = weight[:, 0]
        return self.loss_weight * weight_reduce_loss(
            center_loss + size_loss, weight, reduction, avg_factor)
