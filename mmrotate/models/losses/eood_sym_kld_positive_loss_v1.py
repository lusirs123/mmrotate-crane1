"""Original B in EOOD's decoded-box regression slot, positive boxes only."""
import torch
from ..builder import ROTATED_LOSSES
from .sym_kld_loss import SymKLDLoss


@ROTATED_LOSSES.register_module()
class EOODSymKLDPositiveLossV1(SymKLDLoss):
    """Keep B's formula/normalizer while adapting EOOD's dense loss interface.

    EOOD supplies decoded predictions and zero GT/weight for negatives to its
    ``loss_iou`` slot. B supervises assigned positives only. This adapter adds
    no parameter, inference operation, IoU term or loss-weight adjustment.
    """

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        if pred.shape != target.shape or pred.ndim != 2 or pred.shape[-1] != 5:
            raise ValueError('Expected matching decoded (N,5) rotated boxes')
        reduction = self.reduction if reduction_override is None else reduction_override
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Unknown reduction')
        if weight is not None:
            if weight.ndim == 2:
                if weight.shape != pred.shape or not torch.equal(weight, weight[:, :1].expand_as(weight)):
                    raise ValueError('Expected a common weight for all five box coordinates')
                weight = weight[:, 0]
            if weight.shape != pred.shape[:1] or not torch.isfinite(weight).all() or (weight < 0).any():
                raise ValueError('Invalid per-box regression weight')
            positive = weight > 0
        else:
            positive = torch.ones(len(pred), dtype=torch.bool, device=pred.device)
        if not positive.any():
            return pred.sum(dim=1)*0 if reduction == 'none' else pred.sum()*0
        result = super().forward(pred[positive], target[positive],
            weight=None if weight is None else weight[positive], avg_factor=avg_factor,
            reduction_override=reduction)
        if reduction == 'none':
            output = pred.sum(dim=1)*0
            return output.index_copy(0, positive.nonzero(as_tuple=False).reshape(-1), result)
        return result
