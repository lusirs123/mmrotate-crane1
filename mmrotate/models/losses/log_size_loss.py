"""Independent decoded edge supervision; project candidate, not a new OBB loss."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.losses.utils import weight_reduce_loss

from mmrotate.models.builder import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class LogSizeLoss(nn.Module):
    """Mean SmoothL1 of sorted log edges. No centre or angle dependency.

    Reuses the main regression positives, weights and global normalizer.
    The decoder and its guards remain upstream and unchanged. Exact edge
    ties use torch.sort's branch derivative; this is not everywhere smooth.
    """

    def __init__(self, beta=0.1, loss_weight=0.1, eps=1e-6, reduction='mean'):
        super().__init__()
        if (not all(math.isfinite(float(v)) and float(v) > 0
                    for v in (beta, loss_weight, eps))
                or reduction not in ('none', 'mean', 'sum')):
            raise ValueError('Invalid log-size settings')
        self.beta, self.loss_weight, self.eps = map(float, (beta, loss_weight, eps))
        self.reduction = reduction

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        reduction = reduction_override or self.reduction
        if (reduction not in ('none', 'mean', 'sum') or pred.ndim != 2
                or pred.shape[-1] != 5 or target.shape != pred.shape):
            raise ValueError('Expected matching Nx5 OBBs and valid reduction')
        if avg_factor is not None and (not math.isfinite(float(avg_factor))
                                       or float(avg_factor) <= 0):
            raise ValueError('Expected a finite positive global normalizer')
        if weight is not None:
            if weight.ndim == 2 and weight.shape == pred.shape:
                weight = weight[:, 0]
            if (weight.ndim != 1 or len(weight) != len(pred)
                    or not torch.isfinite(weight).all() or (weight < 0).any()):
                raise ValueError('Expected finite nonnegative N or Nx5 weights')
        if not len(pred):
            zero = pred[:, 2:4].sum(-1) * 0.
            return zero if reduction == 'none' else zero.sum()
        pe, te = pred[:, 2:4], target.detach()[:, 2:4]
        if (not torch.isfinite(pe).all() or not torch.isfinite(te).all()
                or (pe <= 0).any() or (te <= 0).any()):
            raise ValueError('Log-size requires finite positive edges')
        pe = pe.sort(dim=-1, descending=True).values.clamp_min(self.eps)
        te = te.sort(dim=-1, descending=True).values.clamp_min(self.eps)
        error = pe.log() - te.log()
        value = F.smooth_l1_loss(error, torch.zeros_like(error),
                                 reduction='none', beta=self.beta).mean(-1)
        return self.loss_weight * weight_reduce_loss(value, weight, reduction, avg_factor)
