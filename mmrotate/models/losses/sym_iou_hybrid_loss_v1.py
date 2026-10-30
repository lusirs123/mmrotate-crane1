"""B2: preserve overlap supervision and retain symmetric-KLD geometry guidance."""
import math

import torch
import torch.nn as nn

from mmrotate.models.builder import ROTATED_LOSSES
from .rotated_iou_loss import RotatedIoULoss
from .sym_kld_loss import SymKLDLoss


@ROTATED_LOSSES.register_module()
class SymIoUHybridLossV1(nn.Module):
    """2 * (linear rotated DIoU + lambda * compressed symmetric KLD).

    Both terms supervise the same assigned positive boxes, using the same
    weights and positive normalizer. lambda=0 is supported only for identity
    tests; the B2 experiment requires lambda>0 and therefore retains B.
    This loss adds no learned parameter or inference branch.
    """

    def __init__(self, geometry_weight=0.25, loss_weight=2.0, eps=1e-6,
                 reduction='mean', strict_finite=True):
        super().__init__()
        if not math.isfinite(geometry_weight) or geometry_weight < 0:
            raise ValueError('geometry_weight must be finite and nonnegative')
        if not math.isfinite(loss_weight) or loss_weight <= 0:
            raise ValueError('loss_weight must be finite and positive')
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Unknown reduction')
        self.geometry_weight = float(geometry_weight)
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.strict_finite = bool(strict_finite)
        self.overlap = RotatedIoULoss(mode='linear', diou=True, eps=eps,
                                     reduction=reduction, loss_weight=1.0)
        self.geometry = SymKLDLoss(eps=eps, reduction=reduction,
                                   loss_weight=1.0, strict_finite=strict_finite)

    def components(self, pred, target, weight=None, avg_factor=None,
                   reduction_override=None):
        """Return actual reduced terms, also usable by zero-update diagnostics."""
        if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 5:
            raise ValueError('Expected matching (N,5) decoded OBBs')
        reduction = self.reduction if reduction_override is None else reduction_override
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Unknown reduction override')
        if weight is not None and weight.ndim > 1:
            if weight.shape != pred.shape:
                raise ValueError('OBB weights must match predictions')
            # The detector passes one positive weight per box. Preserve this
            # contract rather than silently blending different edge weights.
            if not torch.equal(weight, weight[:, :1].expand_as(weight)):
                raise ValueError('The two terms require a common per-box weight')
            weight = weight[:, 0]
        if pred.shape[0] == 0:
            zero = pred.sum(dim=1) * 0 if reduction == 'none' else pred.sum() * 0
            return dict(overlap=zero, geometry=zero)
        if weight is not None and not torch.any(weight > 0):
            zero = pred.sum(dim=1) * 0 if reduction == 'none' else pred.sum() * 0
            return dict(overlap=zero, geometry=zero)
        result = dict(overlap=self.overlap(pred, target, weight=weight,
                      avg_factor=avg_factor, reduction_override=reduction),
                      geometry=self.geometry(pred, target, weight=weight,
                      avg_factor=avg_factor, reduction_override=reduction))
        if self.strict_finite:
            for name, value in result.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError('Nonfinite B2 ' + name)
        return result

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        terms = self.components(pred, target, weight, avg_factor, reduction_override)
        return self.loss_weight * (terms['overlap'] + self.geometry_weight * terms['geometry'])
