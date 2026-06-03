# Copyright (c) OpenMMLab. All rights reserved.
# AGAS: Anisotropic Gaussian Auxiliary Supervision for ATSS auxiliary head.

import torch

from mmrotate.models.builder import ROTATED_HEADS
from .rotated_atss_head import RotatedATSSHead


@ROTATED_HEADS.register_module(force=True)
class AGASHead(RotatedATSSHead):
    """ATSS auxiliary head with anisotropic Gaussian regression weighting.

    AGAS keeps the stable ATSS assignment/classification path, but computes the
    auxiliary bbox loss on decoded OBBs with SymKLDLoss and reweights positive
    regression samples by a rotated anisotropic Gaussian aligned to each target
    OBB. It is intended to replace the M2 RotatedATSS auxiliary head, not to add
    another independent auxiliary branch.
    """

    def __init__(self,
                 *args,
                 agas_alpha=2.0,
                 agas_beta=0.5,
                 agas_min_weight=0.05,
                 **kwargs):
        super(AGASHead, self).__init__(*args, **kwargs)
        self.agas_alpha = agas_alpha
        self.agas_beta = agas_beta
        self.agas_min_weight = agas_min_weight

    def _anisotropic_gaussian_weight(self, anchor_centers, target_bboxes):
        """Compute object-aligned scalar weights for positive anchors.

        Args:
            anchor_centers (Tensor): Positive anchor centers, shape (N, 2).
            target_bboxes (Tensor): Decoded target OBBs in ``cx, cy, w, h, a``
                format, shape (N, 5).

        Returns:
            Tensor: Scalar regression weights, shape (N,).
        """
        if anchor_centers.numel() == 0:
            return anchor_centers.new_zeros((0,))

        ctr = target_bboxes[:, 0:2]
        wh = target_bboxes[:, 2:4].clamp(min=1.0)
        theta = target_bboxes[:, 4]

        offset = anchor_centers - ctr
        dx = offset[:, 0]
        dy = offset[:, 1]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Rotate offsets into the local OBB coordinate system.
        dx_rot = cos_t * dx + sin_t * dy
        dy_rot = -sin_t * dx + cos_t * dy

        # Give the longer local axis a broader support and the shorter local
        # axis a sharper decay. This keeps supervision object-aligned without
        # changing ATSS positive/negative assignment.
        w = wh[:, 0]
        h = wh[:, 1]
        w_is_long = w >= h
        sigma_x = torch.where(w_is_long, w * self.agas_alpha,
                              w * self.agas_beta)
        sigma_y = torch.where(w_is_long, h * self.agas_beta,
                              h * self.agas_alpha)
        sigma_x = sigma_x.clamp(min=1.0)
        sigma_y = sigma_y.clamp(min=1.0)

        weight = torch.exp(-((dx_rot / sigma_x).pow(2) +
                             (dy_rot / sigma_y).pow(2)))
        if self.agas_min_weight is not None and self.agas_min_weight > 0:
            weight = weight.clamp(min=float(self.agas_min_weight))
        return weight

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute one-level AGAS loss.

        Classification follows the original ATSS auxiliary head. Regression is
        computed only on positives after decoding both predictions and targets to
        physical OBBs, then applying anisotropic Gaussian sample weights.
        """
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        anchors = anchors.reshape(-1, 5)

        pos_inds = (labels >= 0) & (labels < self.num_classes) & \
            (bbox_weights[:, 0] > 0)

        if pos_inds.any():
            pos_anchors = anchors[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]

            pos_pred_bboxes = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_target_bboxes = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)

            agas_weights = self._anisotropic_gaussian_weight(
                pos_anchors[:, 0:2], pos_target_bboxes)
            pos_weights = bbox_weights[pos_inds][:, 0] * agas_weights

            loss_bbox = self.loss_bbox(
                pos_pred_bboxes,
                pos_target_bboxes,
                weight=pos_weights,
                avg_factor=num_total_samples)
        else:
            loss_bbox = bbox_pred.sum() * 0.0

        if torch.isnan(loss_cls) or torch.isinf(loss_cls):
            loss_cls = cls_score.sum() * 0.0
        if torch.isnan(loss_bbox) or torch.isinf(loss_bbox):
            loss_bbox = bbox_pred.sum() * 0.0

        return loss_cls, loss_bbox
