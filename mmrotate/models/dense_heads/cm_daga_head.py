# Copyright (c) OpenMMLab. All rights reserved.
# CM-DAGA-W: confidence-modulated ATSS auxiliary head.

import torch
from mmcv.runner import force_fp32
from mmdet.core import images_to_levels, multi_apply, unmap

from mmrotate.core import obb2hbb, rotated_anchor_inside_flags
from mmrotate.models.builder import ROTATED_HEADS
from .rotated_atss_head import RotatedATSSHead
from .utils import get_num_level_anchors_inside


@ROTATED_HEADS.register_module(force=True)
class CMDAGAHead(RotatedATSSHead):
    """Confidence-modulated ATSS auxiliary head.

    This first CM-DAGA-W version intentionally keeps the M2 auxiliary path
    conservative: standard ATSS assignment, FocalLoss classification, and
    SmoothL1 regression are preserved. The only change is a detached GT-level
    main-head confidence weight applied to positive auxiliary bbox regression
    samples, so GTs already learned confidently by the main head send weaker
    redundant auxiliary regression gradients.
    """

    def __init__(self,
                 *args,
                 cm_min_weight=0.5,
                 cm_warmup_iters=1000,
                 cm_modulate_cls=False,
                 cm_modulate_bbox=True,
                 cm_topk=9,
                 **kwargs):
        super(CMDAGAHead, self).__init__(*args, **kwargs)
        self.cm_min_weight = cm_min_weight
        self.cm_warmup_iters = cm_warmup_iters
        self.cm_modulate_cls = cm_modulate_cls
        self.cm_modulate_bbox = cm_modulate_bbox
        self.cm_topk = cm_topk
        self.register_buffer('_cm_iter', torch.tensor(0, dtype=torch.long))

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            num_level_anchors,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute ATSS targets and additionally return assigned GT indices."""
        inside_flags = rotated_anchor_inside_flags(
            flat_anchors, valid_flags, img_meta['img_shape'][:2],
            self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 8

        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        if self.assign_by_circumhbbox is not None:
            gt_bboxes_assign = obb2hbb(gt_bboxes, self.assign_by_circumhbbox)
            assign_result = self.assigner.assign(
                anchors, num_level_anchors_inside, gt_bboxes_assign,
                gt_bboxes_ignore, None if self.sampling else gt_labels)
        else:
            assign_result = self.assigner.assign(
                anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore,
                None if self.sampling else gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        assigned_gt_inds = anchors.new_full((num_valid_anchors, ),
                                            -1,
                                            dtype=torch.long)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            assigned_gt_inds[pos_inds] = sampling_result.pos_assigned_gt_inds
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            assigned_gt_inds = unmap(assigned_gt_inds, num_total_anchors,
                                     inside_flags, fill=-1)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result, assigned_gt_inds)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      main_outs=None,
                      main_bbox_head=None,
                      **kwargs):
        """Forward and compute CM-DAGA-W losses."""
        outs = self(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(
            *loss_inputs,
            gt_bboxes_ignore=gt_bboxes_ignore,
            main_outs=main_outs,
            main_bbox_head=main_bbox_head)
        return losses

    def _get_main_gt_confidences(self, main_outs, main_bbox_head, gt_bboxes,
                                 gt_labels, img_metas):
        """Estimate detached per-GT main confidence by nearest main anchors."""
        if main_outs is None or main_bbox_head is None:
            return None

        main_cls_scores = main_outs[0]
        featmap_sizes = [featmap.size()[-2:] for featmap in main_cls_scores]
        device = main_cls_scores[0].device
        anchor_list, _ = main_bbox_head.get_anchors(
            featmap_sizes, img_metas, device=device)
        num_imgs = len(img_metas)
        gt_conf_list = []

        with torch.no_grad():
            for img_id in range(num_imgs):
                if gt_bboxes[img_id].numel() == 0:
                    gt_conf_list.append(gt_bboxes[img_id].new_zeros((0,)))
                    continue

                scores_per_level = []
                for cls_score in main_cls_scores:
                    scores = cls_score[img_id].detach().permute(1, 2, 0).reshape(
                        -1, main_bbox_head.cls_out_channels)
                    scores = scores.sigmoid()
                    if main_bbox_head.num_classes == 1:
                        scores = scores[:, 0]
                    else:
                        labels = gt_labels[img_id].clamp(
                            min=0, max=main_bbox_head.num_classes - 1)
                        scores = scores[:, labels]
                    scores_per_level.append(scores)

                main_scores = torch.cat(scores_per_level, dim=0)
                main_anchors = torch.cat(anchor_list[img_id], dim=0).to(device)
                anchor_centers = main_anchors[:, :2]
                gt_centers = gt_bboxes[img_id][:, :2]
                num_gts = gt_centers.size(0)

                if main_bbox_head.num_classes == 1:
                    dist = (gt_centers[:, None, :] -
                            anchor_centers[None, :, :]).pow(2).sum(dim=-1)
                    topk = min(max(int(self.cm_topk), 1), dist.size(1))
                    nearest = dist.topk(topk, dim=1, largest=False).indices
                    gt_scores = main_scores[nearest].max(dim=1).values
                else:
                    gt_scores = gt_centers.new_zeros((num_gts,))
                    for gt_idx in range(num_gts):
                        dist = (anchor_centers - gt_centers[gt_idx]).pow(2).sum(dim=-1)
                        topk = min(max(int(self.cm_topk), 1), dist.size(0))
                        nearest = dist.topk(topk, largest=False).indices
                        gt_scores[gt_idx] = main_scores[nearest, gt_idx].max()
                gt_conf_list.append(gt_scores.clamp(0.0, 1.0).detach())

        return gt_conf_list

    def _build_cm_weight_list(self, assigned_gt_inds_list, gt_conf_list):
        """Build per-anchor CM weights for each feature level."""
        if gt_conf_list is None:
            return None

        warmup = 1.0
        if self.cm_warmup_iters is not None and self.cm_warmup_iters > 0:
            warmup = min(1.0, float(self._cm_iter.item()) /
                         float(self.cm_warmup_iters))
        min_weight = float(self.cm_min_weight)
        cm_weight_list = []
        for assigned_gt_inds in assigned_gt_inds_list:
            weights = assigned_gt_inds.new_ones(
                assigned_gt_inds.shape, dtype=torch.float)
            for img_id, gt_conf in enumerate(gt_conf_list):
                pos_mask = assigned_gt_inds[img_id] >= 0
                if pos_mask.any() and gt_conf.numel() > 0:
                    gt_inds = assigned_gt_inds[img_id][pos_mask].clamp(
                        max=gt_conf.numel() - 1)
                    raw_weight = min_weight + (1.0 - min_weight) * \
                        (1.0 - gt_conf[gt_inds])
                    weights[img_id][pos_mask] = \
                        (1.0 - warmup) + warmup * raw_weight
            cm_weight_list.append(weights.detach())
        return cm_weight_list

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, cm_weights, num_total_samples):
        """Compute one-level CM-DAGA-W loss."""
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)

        if self.cm_modulate_cls and cm_weights is not None:
            flat_cm_weights = cm_weights.reshape(-1).to(label_weights.dtype)
            pos_inds = (labels >= 0) & (labels < self.num_classes)
            label_weights = label_weights.clone()
            label_weights[pos_inds] = label_weights[pos_inds] * \
                flat_cm_weights[pos_inds]

        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        if self.cm_modulate_bbox and cm_weights is not None:
            flat_cm_weights = cm_weights.reshape(-1).to(bbox_weights.dtype)
            bbox_weights = bbox_weights * flat_cm_weights[:, None]

        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        if torch.isnan(loss_cls) or torch.isinf(loss_cls):
            loss_cls = cls_score.sum() * 0.0
        if torch.isnan(loss_bbox) or torch.isinf(loss_bbox):
            loss_bbox = bbox_pred.sum() * 0.0
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             main_outs=None,
             main_bbox_head=None):
        """Compute CM-DAGA-W losses."""
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, assigned_gt_inds_list) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        gt_conf_list = self._get_main_gt_confidences(
            main_outs, main_bbox_head, gt_bboxes, gt_labels, img_metas)
        cm_weight_list = self._build_cm_weight_list(
            assigned_gt_inds_list, gt_conf_list)
        if cm_weight_list is None:
            cm_weight_list = [None for _ in labels_list]

        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            cm_weight_list,
            num_total_samples=num_total_samples)

        if self.training:
            self._cm_iter += 1
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
