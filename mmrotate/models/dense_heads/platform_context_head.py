import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32

from mmrotate.models.builder import ROTATED_HEADS


@ROTATED_HEADS.register_module(force=True)
class PlatformContextHead(BaseModule):
    """Seq-K platform context auxiliary supervision head.

    This head predicts a one-channel platform context map on selected FPN
    levels. Targets are generated online from current transformed beam GT boxes
    and per-sequence rigid K values. It is training-only and does not participate
    in inference.
    """

    def __init__(self,
                 in_channels=256,
                 feat_channels=128,
                 stacked_convs=2,
                 levels=(0, 1, 2),
                 seq_platform_k=None,
                 default_platform_k=None,
                 loss_weight=0.05,
                 pos_weight=5.0,
                 neg_weight=0.05,
                 min_pos_pixels=1,
                 use_dtheta=False,
                 use_dice_loss=False,
                 dice_weight=1.0,
                 target_dilate_k=1.0,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = int(in_channels)
        self.feat_channels = int(feat_channels)
        self.stacked_convs = int(stacked_convs)
        self.levels = tuple(int(x) for x in levels)
        self.seq_platform_k = seq_platform_k or {}
        self.default_platform_k = default_platform_k
        self.loss_weight = float(loss_weight)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.min_pos_pixels = int(min_pos_pixels)
        self.use_dtheta = bool(use_dtheta)
        self.use_dice_loss = bool(use_dice_loss)
        self.dice_weight = float(dice_weight)
        self.target_dilate_k = float(target_dilate_k)

        layers = []
        in_ch = self.in_channels
        for _ in range(self.stacked_convs):
            layers.extend([
                nn.Conv2d(in_ch, self.feat_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            ])
            in_ch = self.feat_channels
        self.context_convs = nn.Sequential(*layers)
        self.context_logits = nn.Conv2d(in_ch, 1, 3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        outs = []
        for lvl in self.levels:
            feat = feats[lvl]
            outs.append(self.context_logits(self.context_convs(feat)))
        return outs

    @staticmethod
    def _parse_seq(meta):
        name = str(meta.get('filename', meta.get('ori_filename', '')))
        match = re.search(r'(?:^|/)((?:real|sim)_seq\d+)_\d{5}', name)
        if match:
            return match.group(1)
        match = re.search(r'((?:real|sim)_seq\d+)', name)
        return match.group(1) if match else None

    def _seq_k_for_meta(self, meta):
        seq = self._parse_seq(meta)
        if seq is not None and seq in self.seq_platform_k:
            return self.seq_platform_k[seq]
        return self.default_platform_k

    @staticmethod
    def _long_frame_from_gt(gt_bboxes):
        center = gt_bboxes[:, 0:2]
        wh = gt_bboxes[:, 2:4].clamp(min=1e-6)
        theta = gt_bboxes[:, 4]
        width_is_long = wh[:, 0] >= wh[:, 1]
        long_len = torch.where(width_is_long, wh[:, 0], wh[:, 1])
        short_len = torch.where(width_is_long, wh[:, 1], wh[:, 0])
        long_theta = torch.where(width_is_long, theta, theta + math.pi / 2)
        ux = torch.stack([torch.cos(long_theta), torch.sin(long_theta)], dim=1)
        flip = (ux[:, 0] < 0) | ((ux[:, 0].abs() < 1e-6) & (ux[:, 1] < 0))
        ux = torch.where(flip[:, None], -ux, ux)
        uy = torch.stack([-ux[:, 1], ux[:, 0]], dim=1)
        return center, ux, uy, long_len, short_len, long_theta

    def _platform_boxes_from_gt(self, gt_bboxes, seq_k):
        if gt_bboxes.numel() == 0 or seq_k is None:
            return gt_bboxes.new_zeros((0, 5))

        center, ux, uy, long_len, short_len, long_theta = (
            self._long_frame_from_gt(gt_bboxes))
        width_k = float(seq_k['width_k'])
        height_k = float(seq_k['height_k'])
        offset_long_k = float(seq_k.get('offset_long_k', 0.0))
        offset_short_k = float(seq_k.get('offset_short_k', 0.0))
        dtheta = float(seq_k.get('dtheta', 0.0)) if self.use_dtheta else 0.0

        plat_center = (
            center
            + ux * (offset_long_k * long_len)[:, None]
            + uy * (offset_short_k * short_len)[:, None])
        # Dilate platform box for target generation to give context head
        # a larger positive region (helps with extreme class imbalance).
        plat_w = (width_k * long_len * self.target_dilate_k).clamp(min=1e-6)
        plat_h = (height_k * short_len * self.target_dilate_k).clamp(min=1e-6)
        plat_theta = long_theta + dtheta
        return torch.stack([
            plat_center[:, 0], plat_center[:, 1], plat_w, plat_h, plat_theta
        ], dim=1)

    @staticmethod
    def _points_in_obb(xs, ys, boxes):
        if boxes.numel() == 0:
            return torch.zeros_like(xs, dtype=torch.bool)
        mask = torch.zeros_like(xs, dtype=torch.bool)
        for box in boxes:
            cx, cy, w, h, theta = box
            dx = xs - cx
            dy = ys - cy
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            local_x = dx * cos_t + dy * sin_t
            local_y = -dx * sin_t + dy * cos_t
            inside = (
                (local_x.abs() <= w * 0.5)
                & (local_y.abs() <= h * 0.5))
            mask = mask | inside
        return mask

    def _target_single_level(self, logit, img_metas, gt_bboxes):
        batch, _, feat_h, feat_w = logit.shape
        targets = logit.new_zeros((batch, 1, feat_h, feat_w))
        valid = logit.new_zeros((batch, 1, feat_h, feat_w))

        for i in range(batch):
            seq_k = self._seq_k_for_meta(img_metas[i])
            platform_boxes = self._platform_boxes_from_gt(gt_bboxes[i], seq_k)
            if platform_boxes.numel() == 0:
                continue

            pad_shape = img_metas[i].get('pad_shape', img_metas[i]['img_shape'])
            pad_h, pad_w = int(pad_shape[0]), int(pad_shape[1])
            stride_y = float(pad_h) / float(feat_h)
            stride_x = float(pad_w) / float(feat_w)
            ys = (torch.arange(feat_h, device=logit.device,
                               dtype=logit.dtype) + 0.5) * stride_y
            xs = (torch.arange(feat_w, device=logit.device,
                               dtype=logit.dtype) + 0.5) * stride_x
            yy, xx = torch.meshgrid(ys, xs)
            inside = self._points_in_obb(xx, yy, platform_boxes)
            if int(inside.sum()) < self.min_pos_pixels:
                continue
            targets[i, 0] = inside.to(logit.dtype)
            valid[i, 0] = 1.0
        return targets, valid

    @force_fp32(apply_to=('preds',))
    def loss(self, preds, img_metas, gt_bboxes):
        losses = []
        total_pos = preds[0].new_tensor(0.0)
        for pred in preds:
            target, valid = self._target_single_level(pred, img_metas, gt_bboxes)
            if valid.sum() <= 0:
                losses.append(pred.sum() * 0.0)
                continue
            weights = torch.where(
                target > 0,
                pred.new_tensor(self.pos_weight),
                pred.new_tensor(self.neg_weight))
            weights = weights * valid
            bce = F.binary_cross_entropy_with_logits(
                pred, target, reduction='none')
            denom = torch.clamp((weights * (target > 0).float()).sum(), min=1.0)
            level_loss = (bce * weights).sum() / denom

            # Dice loss component: handles extreme class imbalance by
            # optimizing IoU of predicted vs target activation regions.
            if self.use_dice_loss:
                pred_sig = torch.sigmoid(pred)
                inter = ((pred_sig * target) * valid).sum()
                union = ((pred_sig + target) * valid).sum()
                dice = 1.0 - (2.0 * inter + 1.0) / (union.clamp(min=1.0) + 1.0)
                level_loss = level_loss + self.dice_weight * dice

            losses.append(level_loss)
            total_pos = total_pos + target.sum()

        if not losses:
            return dict(loss_platform_context=preds[0].sum() * 0.0)
        loss = sum(losses) / max(len(losses), 1)
        return dict(
            loss_platform_context=loss * self.loss_weight,
            platform_pos_pixels=total_pos.detach())

    def forward_train(self, feats, img_metas, gt_bboxes, gt_labels=None,
                      gt_bboxes_ignore=None):
        preds = self(feats)
        return self.loss(preds, img_metas, gt_bboxes)
