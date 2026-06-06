# Copyright (c) OpenMMLab. All rights reserved.
# M2 + weak SymKLD add-on: keep ATSS SmoothL1 regression, add lightweight
# SymKLD on decoded OBBs as a geometric consistency term.
#
# Design rationale (see AGAS ablation):
#   - AGAS fully replaces SmoothL1 with SymKLD + anisotropic Gaussian weighting,
#     which proved too aggressive and hurt real-domain recall.
#   - M2's broad SmoothL1 + ATSS coverage provides the most stable real-domain
#     supervision.
#   - SymKLD retains value for angular / aspect-ratio geometry but should only
#     act as a lightweight add-on, not a replacement.
#
# Therefore this head:
#   1. Computes the standard SmoothL1 loss on delta-coded predictions (M2 path).
#   2. Decodes positive predictions and targets to physical OBBs.
#   3. Computes SymKLD on decoded positives and adds it to the bbox loss.
#   4. No anisotropic Gaussian weighting is applied.

import torch

from mmrotate.models.builder import ROTATED_HEADS, build_loss
from .rotated_atss_head import RotatedATSSHead


@ROTATED_HEADS.register_module(force=True)
class RotatedATSSSymKLDHead(RotatedATSSHead):
    """RotatedATSS head with an additional weak SymKLD geometric term.

    This head keeps the standard ATSS assignment, FocalLoss classification, and
    SmoothL1 delta regression identical to M2. On top of that, it decodes
    positive predictions and targets to physical OBBs and computes a SymKLD loss
    as a lightweight geometric consistency add-on.

    Args:
        loss_kld (dict): Config for the SymKLD loss. Default weight is 0.001
            (10% of M2's SmoothL1 weight of 0.01).
    """

    def __init__(self, *args, loss_kld=None, **kwargs):
        super(RotatedATSSSymKLDHead, self).__init__(*args, **kwargs)
        if loss_kld is not None:
            self.loss_kld = build_loss(loss_kld)
        else:
            self.loss_kld = None

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Classification and delta-coded SmoothL1 regression are identical to the
        parent ``RotatedAnchorHead.loss_single``. An additional SymKLD term is
        computed on decoded positive OBBs and added to the bbox loss.
        """
        # ---- classification (unchanged from M2) ----
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # ---- SmoothL1 regression on deltas (unchanged from M2) ----
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        anchors_flat = anchors.reshape(-1, 5)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        # ---- weak SymKLD on decoded positive OBBs ----
        if self.loss_kld is not None:
            pos_inds = (labels >= 0) & (labels < self.num_classes) & \
                (bbox_weights[:, 0] > 0)

            if pos_inds.any():
                pos_anchors = anchors_flat[pos_inds]
                pos_pred = bbox_pred[pos_inds]
                pos_target = bbox_targets[pos_inds]

                # Decode delta-coded predictions/targets to physical OBBs
                pred_decoded = self.bbox_coder.decode(pos_anchors, pos_pred)
                target_decoded = self.bbox_coder.decode(pos_anchors, pos_target)

                loss_kld = self.loss_kld(
                    pred_decoded,
                    target_decoded,
                    avg_factor=num_total_samples)

                loss_bbox = loss_bbox + loss_kld

        return loss_cls, loss_bbox
