"""Pixel-level quality assessment head for oriented detections.

This module implements the representation proposed by PQA: every FPN level
predicts a dense GT-relative position heatmap.  At inference, the heatmap is
compared with a Gaussian position encoding generated from each decoded OBB;
their Volume-IoU is the localization quality used for ranking.

The implementation deliberately keeps candidate geometry explicit.  It does
not directly regress one scalar IoU per anchor, which was the failure mode of
``RegQualityHead`` in this project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob

from mmrotate.models.builder import ROTATED_HEADS


@ROTATED_HEADS.register_module(force=True)
class PQAHeatmapHead(nn.Module):
    """PQA heatmap predictor shared across FPN levels.

    ``stacked_convs`` defaults to zero for checkpoint compatibility with the
    first experiment.  V2 uses a small private localization tower because the
    SymEOOD Retina head in this fork has no hidden regression subnet.
    """

    def __init__(self, in_channels=256, feat_channels=256, stacked_convs=0,
                 prior_prob=0.01, init_cfg=None):
        super().__init__()
        if stacked_convs < 0:
            raise ValueError('stacked_convs must be non-negative')
        if not 0.0 < prior_prob < 1.0:
            raise ValueError('prior_prob must be in (0, 1)')
        self.in_channels = int(in_channels)
        self.feat_channels = int(feat_channels)
        self.stacked_convs = int(stacked_convs)
        self.prior_prob = float(prior_prob)
        self.init_cfg = init_cfg
        layers = []
        for index in range(self.stacked_convs):
            in_channels_i = (self.in_channels if index == 0
                             else self.feat_channels)
            layers.append(nn.Conv2d(
                in_channels_i, self.feat_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.localization_tower = nn.Sequential(*layers)
        output_channels = (self.feat_channels if self.stacked_convs > 0
                           else self.in_channels)
        self.heatmap_pred = nn.Conv2d(
            output_channels, 1, kernel_size=3, padding=1)
        self.init_weights()

    def init_weights(self):
        for module in self.localization_tower.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        nn.init.normal_(self.heatmap_pred.weight, std=0.01)
        nn.init.constant_(
            self.heatmap_pred.bias, bias_init_with_prob(self.prior_prob))

    def forward(self, feats):
        return tuple(
            self.heatmap_pred(self.localization_tower(feat))
            for feat in feats)

    @staticmethod
    def build_targets(heatmap_logits, img_metas, gt_bboxes, strides,
                      eps=1e-6):
        """Build the oriented Gaussian label H* on every FPN level.

        The Gaussian follows the PQA definition: inside each GT OBB,
        ``Sigma^(1/2)`` has eigenvalues ``w/4`` and ``h/4``.  Multiple GTs
        are combined by point-wise maximum.  Pixels outside the unpadded image
        are excluded from the loss through a separate valid mask.
        """
        if not (len(heatmap_logits) == len(strides)):
            raise ValueError('PQA heatmap/stride level mismatch')
        targets = []
        valid_masks = []
        for level, logits in enumerate(heatmap_logits):
            batch, channels, height, width = logits.shape
            if channels != 1:
                raise ValueError('PQA heatmap must have one channel')
            stride = strides[level]
            if isinstance(stride, (tuple, list)):
                stride_x, stride_y = float(stride[0]), float(stride[1])
            else:
                stride_x = stride_y = float(stride)
            work_dtype = torch.float32
            xs = ((torch.arange(width, device=logits.device,
                                dtype=work_dtype) + 0.5) * stride_x)
            ys = ((torch.arange(height, device=logits.device,
                                dtype=work_dtype) + 0.5) * stride_y)
            try:
                yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            except TypeError:
                yy, xx = torch.meshgrid(ys, xs)

            level_target = logits.new_zeros(
                (batch, 1, height, width), dtype=work_dtype)
            level_valid = logits.new_zeros(
                (batch, 1, height, width), dtype=work_dtype)
            for image_index in range(batch):
                image_h, image_w = img_metas[image_index]['img_shape'][:2]
                valid = ((xx < float(image_w)) & (yy < float(image_h)))
                level_valid[image_index, 0] = valid.to(work_dtype)
                if gt_bboxes[image_index].numel() == 0:
                    continue
                target = torch.zeros_like(xx)
                for gt in gt_bboxes[image_index].detach().float():
                    cx, cy, box_w, box_h, angle = gt[:5]
                    box_w = box_w.clamp(min=eps)
                    box_h = box_h.clamp(min=eps)
                    dx = xx - cx
                    dy = yy - cy
                    cos_a = torch.cos(angle)
                    sin_a = torch.sin(angle)
                    local_x = cos_a * dx + sin_a * dy
                    local_y = -sin_a * dx + cos_a * dy
                    inside = ((local_x.abs() <= box_w * 0.5)
                              & (local_y.abs() <= box_h * 0.5))
                    mahal = ((local_x / (box_w * 0.25)) ** 2
                             + (local_y / (box_h * 0.25)) ** 2)
                    gaussian = torch.exp(-0.5 * mahal) * inside.float()
                    target = torch.maximum(target, gaussian)
                level_target[image_index, 0] = target * valid.float()
            targets.append(level_target.to(dtype=logits.dtype))
            valid_masks.append(level_valid.to(dtype=logits.dtype))
        return tuple(targets), tuple(valid_masks)

    @staticmethod
    def ld_loss(heatmap_logits, targets, valid_masks, gamma=2.0,
                loss_weight=1.0):
        """Localization-distribution loss with continuous Gaussian labels."""
        if not (len(heatmap_logits) == len(targets) == len(valid_masks)):
            raise ValueError('PQA LD loss level mismatch')
        total = heatmap_logits[0].sum() * 0.0
        positive_count = 0
        target_sum = total.detach()
        pred_sum = total.detach()
        for logits, target, valid in zip(
                heatmap_logits, targets, valid_masks):
            probabilities = logits.sigmoid()
            bce = F.binary_cross_entropy_with_logits(
                logits, target, reduction='none')
            focal_weight = (target - probabilities).abs().pow(float(gamma))
            total = total + (bce * focal_weight * valid).sum()
            positive = (target > 0.0) & (valid > 0.0)
            count = int(positive.sum().item())
            positive_count += count
            if count:
                target_sum = target_sum + target[positive].sum().detach()
                pred_sum = pred_sum + probabilities[positive].sum().detach()
        normalizer = max(positive_count, 1)
        loss = total / float(normalizer) * float(loss_weight)
        stats = dict(
            pqa_positive=total.new_tensor(float(positive_count)),
            pqa_target_mean=target_sum / float(normalizer),
            pqa_pred_positive_mean=pred_sum / float(normalizer))
        return loss, stats

    @staticmethod
    def consistency_loss(clean_logits, dark_logits, targets, valid_masks,
                         loss_weight=1.0):
        """Clean-teacher/dark-student heatmap consistency.

        Positive and background regions are averaged separately so the large
        background does not overwhelm the object-region consistency signal.
        """
        if not (len(clean_logits) == len(dark_logits) == len(targets)
                == len(valid_masks)):
            raise ValueError('PQA consistency level mismatch')
        losses = []
        for clean, dark, target, valid in zip(
                clean_logits, dark_logits, targets, valid_masks):
            difference = (dark.sigmoid() - clean.sigmoid().detach()).abs()
            positive = (target > 0.0) & (valid > 0.0)
            negative = (target <= 0.0) & (valid > 0.0)
            terms = []
            if positive.any():
                terms.append(difference[positive].mean())
            if negative.any():
                terms.append(difference[negative].mean() * 0.1)
            if terms:
                losses.append(torch.stack(terms).sum())
        if not losses:
            return clean_logits[0].sum() * 0.0
        return torch.stack(losses).mean() * float(loss_weight)

    @staticmethod
    def _local_grid(grid_size, device, dtype):
        if grid_size < 3:
            raise ValueError('PQA grid_size must be at least 3')
        axis = ((torch.arange(grid_size, device=device, dtype=dtype) + 0.5)
                * (2.0 / float(grid_size)) - 1.0)
        try:
            yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        except TypeError:
            yy, xx = torch.meshgrid(axis, axis)
        return xx, yy

    @classmethod
    def quality_from_boxes(cls, heatmap_logits, boxes, levels, pad_shape,
                           grid_size=9, batch_size=512,
                           canonical_level=None, eps=1e-12):
        """Compute differentiable PQA Volume-IoU for decoded OBBs.

        Args:
            heatmap_logits: per-level tensors for one image, each [1,1,H,W].
            boxes: [N,5] decoded OBBs in padded-image coordinates.
            levels: [N] FPN level index for every box.
            pad_shape: padded image shape used by the FPN.
        """
        if boxes.ndim != 2 or boxes.shape[1] != 5:
            raise ValueError('PQA boxes must have shape [N, 5]')
        if levels.ndim != 1 or levels.numel() != boxes.shape[0]:
            raise ValueError('PQA levels must align with boxes')
        if batch_size <= 0:
            raise ValueError('PQA batch_size must be positive')
        if boxes.shape[0] == 0:
            return boxes.new_zeros((0,))
        pad_h, pad_w = float(pad_shape[0]), float(pad_shape[1])
        if pad_h <= 0 or pad_w <= 0:
            raise ValueError('PQA pad_shape must be positive')

        work_boxes = boxes.float()
        qualities = work_boxes.new_zeros(work_boxes.shape[0])
        xx, yy = cls._local_grid(
            int(grid_size), work_boxes.device, work_boxes.dtype)
        candidate_map = torch.exp(-2.0 * (xx.square() + yy.square()))

        if canonical_level is not None:
            canonical_level = int(canonical_level)
            if not 0 <= canonical_level < len(heatmap_logits):
                raise ValueError('canonical PQA heatmap level is out of range')
            heatmap_items = [(
                canonical_level, heatmap_logits[canonical_level],
                torch.arange(boxes.shape[0], device=boxes.device))]
        else:
            heatmap_items = []
            for level_index, level_heatmap in enumerate(heatmap_logits):
                indices = torch.nonzero(
                    levels == int(level_index), as_tuple=False).reshape(-1)
                heatmap_items.append((level_index, level_heatmap, indices))

        for _, level_heatmap, indices in heatmap_items:
            if indices.numel() == 0:
                continue
            if level_heatmap.shape[0] != 1 or level_heatmap.shape[1] != 1:
                raise ValueError(
                    'quality_from_boxes expects one-image one-channel maps')
            probability_map = level_heatmap.float().sigmoid()
            for start in range(0, indices.numel(), int(batch_size)):
                batch_indices = indices[start:start + int(batch_size)]
                selected = work_boxes[batch_indices]
                cx, cy, width, height, angle = selected.unbind(dim=1)
                local_x = xx[None] * width.clamp(min=1e-6)[:, None, None] * 0.5
                local_y = yy[None] * height.clamp(min=1e-6)[:, None, None] * 0.5
                cos_a = torch.cos(angle)[:, None, None]
                sin_a = torch.sin(angle)[:, None, None]
                points_x = (cx[:, None, None] + cos_a * local_x
                            - sin_a * local_y)
                points_y = (cy[:, None, None] + sin_a * local_x
                            + cos_a * local_y)
                grid_x = points_x * (2.0 / pad_w) - 1.0
                grid_y = points_y * (2.0 / pad_h) - 1.0
                grid = torch.stack([grid_x, grid_y], dim=-1)
                # Keep a single heatmap input and concatenate candidate grids
                # along the output-height axis to avoid repeating HxW maps.
                count = selected.shape[0]
                merged_grid = grid.reshape(
                    1, count * int(grid_size), int(grid_size), 2)
                sampled = F.grid_sample(
                    probability_map, merged_grid, mode='bilinear',
                    padding_mode='zeros', align_corners=False)
                sampled = sampled.reshape(
                    count, int(grid_size), int(grid_size))
                reference = candidate_map[None].expand_as(sampled)
                intersection = torch.minimum(sampled, reference).sum((1, 2))
                union = torch.maximum(sampled, reference).sum((1, 2))
                qualities[batch_indices] = intersection / union.clamp(min=eps)
        return qualities.to(dtype=boxes.dtype)
