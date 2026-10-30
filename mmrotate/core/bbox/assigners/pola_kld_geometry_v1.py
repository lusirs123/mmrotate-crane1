"""Replace POLA's geometry only, retaining the effective EOOD skeleton."""
import torch
from mmdet.core.bbox.assigners.assign_result import AssignResult

from ..builder import ROTATED_BBOX_ASSIGNERS
from .pola import PolaAssigner
from mmrotate.models.losses.sym_kld_calculator import sym_kld


@ROTATED_BBOX_ASSIGNERS.register_module()
class PolaKLDGeometryV1(PolaAssigner):
    """Same POLA interface, class cost, one-positive rule and IoU metadata.

    Only the geometry in the matching cost changes to the existing raw
    symmetric KLD. No temperature, positive-count schedule, loss modification,
    trainable tensor or inference module is introduced. ``geometry_mode=iou``
    is a neutral replay check, not an additional planned training arm.
    """

    def __init__(self, cost_class=1., cost_bbox=1., cost_riou=1., o2m=False,
                 geometry_mode='kld', eps=1e-6):
        if o2m:
            raise ValueError('This experiment retains the original O2O POLA')
        if geometry_mode not in ('iou', 'kld'):
            raise ValueError('Unknown geometry mode')
        super().__init__(cost_class, cost_bbox, cost_riou, o2m=False)
        self.geometry_mode = geometry_mode
        self.eps = eps

    @torch.no_grad()
    def assign(self, pred_logits, pred_bboxes, gt_labels, gt_bboxes, img_metas):
        for value in (pred_logits, pred_bboxes, gt_bboxes):
            if not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite POLA geometry-only input')
        # Original POLA supplies exactly its existing IoU metadata/empty-case
        # contract. The additional KLD computation changes assignment only.
        original = super().assign(pred_logits, pred_bboxes, gt_labels,
                                  gt_bboxes, img_metas)
        if (self.geometry_mode == 'iou' or len(gt_bboxes) == 0
                or len(pred_bboxes) == 0):
            return original
        probability = pred_logits.sigmoid()
        class_cost = (self.focal_loss_alpha
                      * (1 - probability) ** self.focal_loss_gamma
                      * -(probability + 1e-8).log())[:, gt_labels]
        geometry = sym_kld(pred_bboxes[:, None, :],
                           gt_bboxes[None, :, :], eps=self.eps)
        cost = self.cost_class * class_cost + self.cost_riou * geometry
        if not torch.isfinite(cost).all():
            raise FloatingPointError('Nonfinite POLA KLD matching cost')
        indices = cost.argmin(dim=0)
        assigned = original.gt_inds.new_zeros(len(pred_bboxes))
        # Preserve the original independent-per-GT semantics, including its
        # conflict behaviour; this is not a Hungarian/global matching change.
        assigned[indices] = torch.arange(len(gt_bboxes), device=indices.device) + 1
        labels = assigned.new_full((len(pred_bboxes),), -1)
        positives = torch.nonzero(assigned > 0, as_tuple=False).reshape(-1)
        labels[positives] = gt_labels[assigned[positives] - 1]
        return AssignResult(len(gt_bboxes), assigned, original.max_overlaps,
                            labels=labels)
