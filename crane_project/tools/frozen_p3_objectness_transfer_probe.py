#!/usr/bin/env python3
"""Source-trained Frozen-P3 spatial objectness transfer diagnosis.

The detector, backbone, FPN, classification head, and regression head remain
frozen.  Only one 3x3 ``P3: 256 -> 1`` convolution is optimized from source
annotations.  Source validation selects the probe checkpoint.  The labelled
target-dev slice is read only after checkpoint selection and is used solely
for the final paired diagnosis.

This tool never fuses scores into detector inference and never writes a main
detector checkpoint.
"""

import argparse
import copy
import glob
import json
import math
import os
import random
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import candidate_pool_oracle_probe as pool_probe  # noqa: E402
from crane_project.tools import ctx_entry_probe as entry_probe  # noqa: E402
from crane_project.tools import retina_cls_contribution_probe as contribution  # noqa: E402


PROBE_NAME = 'Frozen-P3 Spatial Objectness Transfer Probe'
PROTOCOL_VERSION = 1
CANONICAL_CONFIG = 'crane_symeood_k1_brightaug.py'
CANONICAL_CHECKPOINT = 'epoch_20.pth'
TARGET_SPLIT = 'test'
TARGET_SEQ = 'real_seq02'
TARGET_START = 137
TARGET_END = 169
EXPECTED_GEOMETRY_MISSES = [164, 167]
EXPECTED_ELIGIBLE = 31


def parse_args():
    parser = argparse.ArgumentParser(description=PROBE_NAME)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--source-indexes', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--feature-level', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--positive-weight', type=float, default=20.0)
    parser.add_argument('--gaussian-sigma-scale', type=float, default=0.25)
    parser.add_argument('--gaussian-min-sigma-strides', type=float, default=1.0)
    parser.add_argument('--max-train-samples-per-source', type=int, default=0,
                        help='0 means all samples; nonzero is smoke-test only')
    parser.add_argument('--max-val-samples', type=int, default=0,
                        help='0 means all source-val samples')
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--false-iou-thr', type=float, default=0.1)
    parser.add_argument('--source-val-min-accuracy', type=float, default=0.8)
    parser.add_argument('--target-min-wins', type=int, default=26)
    parser.add_argument('--target-start', type=int, default=TARGET_START)
    parser.add_argument('--target-end', type=int, default=TARGET_END)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--allow-noncanonical', action='store_true')
    return parser.parse_args()


def validate_args(args) -> bool:
    if args.seed != 0:
        raise ValueError('The unified diagnostic protocol requires --seed 0')
    if not args.source_indexes or len(set(args.source_indexes)) != len(
            args.source_indexes):
        raise ValueError('--source-indexes requires unique values')
    if min(args.source_indexes) < 0:
        raise ValueError('--source-indexes must be non-negative')
    if args.feature_level < 0:
        raise ValueError('--feature-level must be non-negative')
    if args.epochs <= 0 or args.lr <= 0.0 or args.weight_decay < 0.0:
        raise ValueError('Invalid optimizer schedule')
    if args.positive_weight < 0.0:
        raise ValueError('--positive-weight must be non-negative')
    if args.gaussian_sigma_scale <= 0.0:
        raise ValueError('--gaussian-sigma-scale must be positive')
    if args.gaussian_min_sigma_strides <= 0.0:
        raise ValueError('--gaussian-min-sigma-strides must be positive')
    if args.max_train_samples_per_source < 0 or args.max_val_samples < 0:
        raise ValueError('Sample limits must be non-negative')
    if not 0.0 <= args.false_iou_thr < args.riou_thr <= 1.0:
        raise ValueError('Require 0 <= false-iou-thr < riou-thr <= 1')
    if not 0.0 < args.source_val_min_accuracy <= 1.0:
        raise ValueError('--source-val-min-accuracy must be in (0, 1]')
    if args.target_min_wins <= 0:
        raise ValueError('--target-min-wins must be positive')

    canonical_checks = dict(
        config=os.path.basename(args.config) == CANONICAL_CONFIG,
        checkpoint=os.path.basename(args.checkpoint) == CANONICAL_CHECKPOINT,
        source_indexes=list(args.source_indexes) == [0, 1],
        feature_level=int(args.feature_level) == 0,
        target_slice=(int(args.target_start) == TARGET_START
                      and int(args.target_end) == TARGET_END),
        thresholds=(math.isclose(args.riou_thr, 0.5)
                    and math.isclose(args.false_iou_thr, 0.1)),
        target_gate=int(args.target_min_wins) == 26,
        source_gate=math.isclose(args.source_val_min_accuracy, 0.8),
        full_source_data=(args.max_train_samples_per_source == 0
                          and args.max_val_samples == 0),
    )
    canonical = all(canonical_checks.values())
    if not canonical and not args.allow_noncanonical:
        failed = [key for key, value in canonical_checks.items() if not value]
        raise ValueError(
            'Canonical probe protocol mismatch: {}. Use '
            '--allow-noncanonical only for smoke tests; those results cannot '
            'authorize training.'.format(', '.join(failed)))
    return canonical


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FrozenP3SpatialObjectness(nn.Module):
    """The complete trainable surface of the diagnosis."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.objectness = nn.Conv2d(in_channels, 1, 3, padding=1)
        nn.init.normal_(self.objectness.weight, std=0.01)
        prior = 0.01
        nn.init.constant_(self.objectness.bias, math.log(prior / (1.0 - prior)))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.objectness(feature)


def stride_value(stride) -> float:
    if isinstance(stride, (tuple, list)):
        if len(stride) != 2 or float(stride[0]) != float(stride[1]):
            raise ValueError('Probe requires an isotropic FPN stride')
        return float(stride[0])
    return float(stride)


def valid_grid_mask(height: int, width: int, img_shape,
                    stride: float, device) -> torch.Tensor:
    img_h, img_w = [int(value) for value in img_shape[:2]]
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
    valid_y = (ys >= 0.0) & (ys < float(img_h))
    valid_x = (xs >= 0.0) & (xs < float(img_w))
    return (valid_y[:, None] & valid_x[None, :]).unsqueeze(0)


def oriented_gaussian_heatmap(boxes: torch.Tensor, height: int, width: int,
                              stride: float, sigma_scale: float,
                              min_sigma_strides: float,
                              valid_mask: Optional[torch.Tensor] = None
                              ) -> torch.Tensor:
    """Build an anchor-independent, orientation-aware source target."""
    device = boxes.device
    heatmap = torch.zeros((1, height, width), device=device, dtype=torch.float32)
    if boxes.numel() == 0:
        return heatmap
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    minimum_sigma = float(min_sigma_strides) * float(stride)
    valid_float = None if valid_mask is None else valid_mask[0].float()

    for box in boxes.float():
        cx, cy, box_w, box_h, angle = box[:5]
        dx = grid_x - cx
        dy = grid_y - cy
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        along_w = cos_a * dx + sin_a * dy
        along_h = -sin_a * dx + cos_a * dy
        sigma_w = torch.clamp(box_w.abs() * sigma_scale,
                              min=minimum_sigma)
        sigma_h = torch.clamp(box_h.abs() * sigma_scale,
                              min=minimum_sigma)
        inside = ((along_w.abs() <= box_w.abs() * 0.5)
                  & (along_h.abs() <= box_h.abs() * 0.5))
        gaussian = torch.exp(-0.5 * (
            (along_w / sigma_w) ** 2 + (along_h / sigma_h) ** 2))
        gaussian = gaussian * inside.float()
        if valid_float is not None:
            gaussian = gaussian * valid_float
        peak = gaussian.max()
        if float(peak.item()) > 0.0:
            gaussian = gaussian / peak
        heatmap[0] = torch.maximum(heatmap[0], gaussian)
    return heatmap


def weighted_heatmap_bce(logits: torch.Tensor, target: torch.Tensor,
                         valid_mask: torch.Tensor,
                         positive_weight: float) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('Expected objectness logits [B,1,H,W]')
    if target.shape != logits[:, 0].shape:
        raise ValueError('Target/logit shape mismatch')
    if valid_mask.shape != target.shape:
        raise ValueError('Valid-mask/target shape mismatch')
    loss = F.binary_cross_entropy_with_logits(
        logits[:, 0], target, reduction='none')
    weights = 1.0 + float(positive_weight) * target
    weights = weights * valid_mask.float()
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def nearest_grid_location(box: Sequence[float], height: int, width: int,
                          stride: float, img_shape) -> Tuple[int, int]:
    img_h, img_w = [int(value) for value in img_shape[:2]]
    valid_height = min(height, int(math.ceil(img_h / float(stride))))
    valid_width = min(width, int(math.ceil(img_w / float(stride))))
    if valid_height <= 0 or valid_width <= 0:
        raise ValueError('Image has no valid feature-grid locations')
    col = int(round(float(box[0]) / float(stride) - 0.5))
    row = int(round(float(box[1]) / float(stride) - 0.5))
    return (max(0, min(row, valid_height - 1)),
            max(0, min(col, valid_width - 1)))


def select_level_candidate(scores: torch.Tensor, ious: torch.Tensor,
                           layout: Sequence[Dict], level: int,
                           min_iou: Optional[float] = None,
                           max_iou: Optional[float] = None) -> Optional[int]:
    """Select the highest-main-score candidate under level/IoU constraints."""
    if scores.ndim != 1 or ious.ndim != 1:
        raise ValueError('scores and ious must be one-dimensional')
    if scores.numel() != ious.numel() or scores.numel() != len(layout):
        raise ValueError('Candidate tensors/layout are not aligned')
    eligible = []
    for index, location in enumerate(layout):
        if int(location['level']) != int(level):
            continue
        iou = float(ious[index].item())
        if min_iou is not None and iou < float(min_iou):
            continue
        if max_iou is not None and iou >= float(max_iou):
            continue
        eligible.append(index)
    if not eligible:
        return None
    index_tensor = torch.tensor(eligible, device=scores.device, dtype=torch.long)
    local = int(torch.argmax(scores[index_tensor]).item())
    return int(eligible[local])


def _number(value) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def candidate_record(index: int, scores: torch.Tensor, ious: torch.Tensor,
                     layout: Sequence[Dict], objectness_logits: torch.Tensor,
                     stride: float) -> Dict:
    location = layout[int(index)]
    row = int(location['row'])
    col = int(location['col'])
    logit = float(objectness_logits[0, 0, row, col].item())
    return dict(
        candidate_index=int(index),
        level=int(location['level']),
        row=row,
        col=col,
        anchor_id=int(location['anchor_id']),
        source_grid_center=[
            _number((col + 0.5) * stride),
            _number((row + 0.5) * stride),
        ],
        main_cls_score=_number(scores[index].item()),
        riou=_number(ious[index].item()),
        objectness_logit=_number(logit),
        objectness_probability=_number(torch.sigmoid(
            torch.tensor(logit)).item()),
    )


def probability_stats(logits: torch.Tensor,
                      valid_mask: torch.Tensor) -> Dict:
    probabilities = logits[:, 0].sigmoid()[valid_mask]
    if probabilities.numel() == 0:
        raise ValueError('No valid feature locations for probability stats')
    quantiles = torch.quantile(
        probabilities.float(), torch.tensor(
            [0.01, 0.5, 0.99], device=probabilities.device))
    minimum = float(probabilities.min().item())
    maximum = float(probabilities.max().item())
    return dict(
        count=int(probabilities.numel()),
        min=_number(minimum),
        max=_number(maximum),
        mean=_number(probabilities.mean().item()),
        p01=_number(quantiles[0].item()),
        p50=_number(quantiles[1].item()),
        p99=_number(quantiles[2].item()),
        fraction_above_0_9=_number((probabilities > 0.9).float().mean().item()),
        fraction_below_0_1=_number((probabilities < 0.1).float().mean().item()),
        all_high=bool(minimum >= 0.95),
        all_low=bool(maximum <= 0.05),
    )


def paired_summary(rows: Sequence[Dict]) -> Dict:
    margins = np.asarray([
        float(row['margin']) for row in rows
        if row.get('margin') is not None
    ], dtype=np.float64)
    if margins.size == 0:
        return dict(count=0, wins=0, accuracy=0.0, median_margin=None,
                    positive_logits={}, negative_logits={})
    positive = np.asarray([
        float(row['positive_logit']) for row in rows
    ], dtype=np.float64)
    negative = np.asarray([
        float(row['negative_logit']) for row in rows
    ], dtype=np.float64)

    def distribution(values):
        return dict(
            min=_number(values.min()), max=_number(values.max()),
            mean=_number(values.mean()), median=_number(np.median(values)),
            p10=_number(np.quantile(values, 0.1)),
            p90=_number(np.quantile(values, 0.9)))

    return dict(
        count=int(margins.size),
        wins=int((margins > 0.0).sum()),
        accuracy=_number((margins > 0.0).mean()),
        median_margin=_number(np.median(margins)),
        positive_logits=distribution(positive),
        negative_logits=distribution(negative),
    )


def target_gate(rows: Sequence[Dict], source_val_summary: Dict,
                required_eligible: int, min_wins: int,
                source_min_accuracy: float) -> Dict:
    eligible = [row for row in rows if row.get('eligible', False)]
    margins = np.asarray([float(row['margin']) for row in eligible],
                         dtype=np.float64)
    wins = int((margins > 0.0).sum()) if margins.size else 0
    median_margin = None if not margins.size else _number(np.median(margins))

    leave_one_out_accuracy = []
    leave_one_out_median = []
    if margins.size >= 2:
        for index in range(margins.size):
            subset = np.delete(margins, index)
            leave_one_out_accuracy.append(float((subset > 0.0).mean()))
            leave_one_out_median.append(float(np.median(subset)))
    robust = bool(
        leave_one_out_accuracy
        and min(leave_one_out_accuracy) >= float(source_min_accuracy)
        and min(leave_one_out_median) > 0.0)
    no_collapse = bool(source_val_summary.get('heatmap_degenerate_images', 0) == 0)
    checks = dict(
        eligible_count=len(eligible) == int(required_eligible),
        target_wins=wins >= int(min_wins),
        target_median_margin=(median_margin is not None
                              and median_margin > 0.0),
        source_val_accuracy=float(source_val_summary.get('accuracy', 0.0)) >= (
            float(source_min_accuracy)),
        source_heatmap_non_degenerate=no_collapse,
        single_frame_robust=robust,
    )
    return dict(
        decision='GO_SUPPORTS_A' if all(checks.values()) else 'STOP_A_EVIDENCE_INSUFFICIENT',
        interpretation=(
            'PASS supports explanation A and authorizes designing a formal '
            'classification/regression-decoupled experiment.'
            if all(checks.values()) else
            'FAIL only says this shallow source-trained P3 objectness route '
            'does not support explanation A; it does not prove explanation B.'),
        checks=checks,
        eligible_count=len(eligible),
        wins=wins,
        accuracy=(None if not margins.size
                  else _number((margins > 0.0).mean())),
        median_margin=median_margin,
        leave_one_out_min_accuracy=(
            None if not leave_one_out_accuracy
            else _number(min(leave_one_out_accuracy))),
        leave_one_out_min_median_margin=(
            None if not leave_one_out_median
            else _number(min(leave_one_out_median))),
    )


def _as_list(value):
    if isinstance(value, tuple):
        return list(value)
    return value if isinstance(value, list) else [value]


def normalize_scattered_batch(data: Dict) -> Tuple[
        torch.Tensor, List[Dict], List[torch.Tensor]]:
    img = data['img']
    if isinstance(img, (list, tuple)) and len(img) == 1:
        img = img[0]
    img_metas = _as_list(data['img_metas'])
    gt_bboxes = _as_list(data['gt_bboxes'])
    if not isinstance(img, torch.Tensor) or img.ndim != 4:
        raise RuntimeError('Expected scattered img tensor [B,C,H,W]')
    if not all(isinstance(item, dict) for item in img_metas):
        raise RuntimeError('Expected scattered img_metas list[dict]')
    if not all(isinstance(item, torch.Tensor) for item in gt_bboxes):
        raise RuntimeError('Expected scattered gt_bboxes list[tensor]')
    return img, img_metas, gt_bboxes


def detector_parameter_versions(model) -> Dict[str, int]:
    return {name: int(parameter._version)
            for name, parameter in model.named_parameters()}


def freeze_detector(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def discover_labeled_records(data_root: str, split: str,
                             limit: int = 0) -> List[Dict]:
    ann_dir = os.path.join(data_root, split, 'annfiles')
    img_dir = os.path.join(data_root, split, 'images')
    records = []
    for ann_path in sorted(glob.glob(os.path.join(ann_dir, '*.txt'))):
        base = os.path.splitext(os.path.basename(ann_path))[0]
        match = re.match(r'(.+_seq\d+)_(\d{5})$', base)
        if match is None:
            continue
        img_path = None
        for extension in ('.jpg', '.png', '.bmp', '.tif'):
            candidate = os.path.join(img_dir, base + extension)
            if os.path.isfile(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue
        records.append(dict(
            split=split, seq=match.group(1), frame=int(match.group(2)),
            image=img_path, annotation=ann_path,
            domain=base.split('_', 1)[0]))
    if limit > 0:
        records = records[:limit]
    return records


def scaled_gt_tensors(record: Dict, meta: Dict, device) -> torch.Tensor:
    diag = entry_probe.get_diag()
    gts = [gt for gt in diag.parse_dota_ann(record['annotation'])
           if gt.get('cls') == 'grab']
    scaled = [pool_probe.scale_gt_to_img(gt, meta) for gt in gts]
    if not scaled:
        return torch.empty((0, 5), device=device, dtype=torch.float32)
    return torch.cat([
        entry_probe.gt_to_tensor(gt, device) for gt in scaled
    ], dim=0)


def forward_main_candidates(model, features, img_shape):
    head, cls_scores, bbox_preds = entry_probe.forward_candidate_head(
        model, features, 'main')
    boxes, scores, levels, _centers, alignment = (
        entry_probe.flatten_decode_candidates(
            head, cls_scores, bbox_preds, img_shape))
    layout = contribution.candidate_layout(cls_scores, head, img_shape)
    if len(layout) != int(scores.numel()):
        raise RuntimeError(
            'Candidate layout mismatch: {} vs {}'.format(
                len(layout), scores.numel()))
    if any(int(layout[index]['level']) != int(levels[index].item())
           for index in range(len(layout))):
        raise RuntimeError('Candidate level alignment failed')
    return head, boxes, scores, layout, alignment


def train_epoch(model, probe, loaders: Sequence[Tuple[int, Iterable]],
                optimizer, args, stride: float) -> Dict:
    from mmcv.parallel import scatter

    probe.train()
    losses = []
    used = 0
    skipped_empty = 0
    by_source = {}
    for source_index, loader in loaders:
        source_used = 0
        for sample_index, raw_data in enumerate(loader):
            if (args.max_train_samples_per_source > 0
                    and sample_index >= args.max_train_samples_per_source):
                break
            data = scatter(raw_data, [int(args.gpu)])[0]
            img, img_metas, gt_bboxes = normalize_scattered_batch(data)
            if len(img_metas) != 1 or len(gt_bboxes) != 1:
                raise RuntimeError('Probe training requires batch size 1')
            if gt_bboxes[0].numel() == 0:
                skipped_empty += 1
                continue
            with torch.no_grad():
                features = model.extract_feat(img)
                p3 = features[args.feature_level].detach()
            logits = probe(p3)
            height, width = [int(value) for value in logits.shape[-2:]]
            mask = valid_grid_mask(
                height, width, img_metas[0]['img_shape'], stride,
                logits.device)
            target = oriented_gaussian_heatmap(
                gt_bboxes[0], height, width, stride,
                args.gaussian_sigma_scale,
                args.gaussian_min_sigma_strides, mask)
            loss = weighted_heatmap_bce(
                logits, target, mask, args.positive_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
            used += 1
            source_used += 1
            if source_used % 200 == 0:
                print('[source-train {}] {} samples'.format(
                    source_index, source_used))
        by_source[str(source_index)] = source_used
    if not losses:
        raise RuntimeError('No source training samples were usable')
    return dict(
        samples=used, skipped_empty=skipped_empty, samples_by_source=by_source,
        mean_loss=_number(np.mean(losses)),
        median_loss=_number(np.median(losses)),
        max_loss=_number(np.max(losses)))


def evaluate_source_validation(model, probe, records: Sequence[Dict],
                               transforms, img_scale, flip,
                               args, stride: float) -> Dict:
    from mmcv.ops import box_iou_rotated

    diag = entry_probe.get_diag()
    probe.eval()
    paired_rows = []
    image_rows = []
    for record_index, record in enumerate(records, start=1):
        img_tensor, meta, _stats = diag.preprocess_image(
            record['image'], transforms, img_scale, flip)
        if img_tensor is None:
            continue
        img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))
        with torch.no_grad():
            features = model.extract_feat(img_tensor)
            objectness = probe(features[args.feature_level])
            head, boxes, scores, layout, alignment = forward_main_candidates(
                model, features, meta['img_shape'])
            gt_boxes = scaled_gt_tensors(record, meta, boxes.device)
            if gt_boxes.numel() == 0:
                continue
            ious = box_iou_rotated(
                boxes.float(), gt_boxes.float()).max(dim=1).values
            false_index = select_level_candidate(
                scores, ious, layout, args.feature_level,
                max_iou=args.false_iou_thr)
            height, width = [int(value) for value in objectness.shape[-2:]]
            mask = valid_grid_mask(
                height, width, meta['img_shape'], stride,
                objectness.device)
            heatmap_stats = probability_stats(objectness, mask)
            image_rows.append(dict(
                split=record['split'], seq=record['seq'],
                frame=int(record['frame']), domain=record['domain'],
                heatmap=heatmap_stats, decode_alignment=alignment))
            if false_index is None:
                continue
            negative = candidate_record(
                false_index, scores, ious, layout, objectness, stride)
            for gt_index, gt_box in enumerate(gt_boxes):
                row, col = nearest_grid_location(
                    gt_box, height, width, stride, meta['img_shape'])
                positive_logit = float(objectness[0, 0, row, col].item())
                margin = positive_logit - float(negative['objectness_logit'])
                paired_rows.append(dict(
                    split=record['split'], seq=record['seq'],
                    frame=int(record['frame']), domain=record['domain'],
                    gt_index=int(gt_index), positive_row=int(row),
                    positive_col=int(col),
                    positive_source_grid_center=[
                        _number((col + 0.5) * stride),
                        _number((row + 0.5) * stride)],
                    positive_logit=_number(positive_logit),
                    negative_logit=negative['objectness_logit'],
                    margin=_number(margin), win=bool(margin > 0.0),
                    hard_negative=negative))
        if record_index % 100 == 0:
            print('[source-val] {}/{} images'.format(
                record_index, len(records)))

    summary = paired_summary(paired_rows)
    degenerate = sum(
        int(row['heatmap']['all_high'] or row['heatmap']['all_low'])
        for row in image_rows)
    summary.update(dict(
        images=len(image_rows),
        heatmap_degenerate_images=int(degenerate),
        heatmap_all_high_images=int(sum(
            row['heatmap']['all_high'] for row in image_rows)),
        heatmap_all_low_images=int(sum(
            row['heatmap']['all_low'] for row in image_rows))))
    return dict(summary=summary, paired_rows=paired_rows,
                image_rows=image_rows)


def evaluate_target_dev(model, probe, transforms, img_scale, flip,
                        args, stride: float) -> Dict:
    from mmcv.ops import box_iou_rotated

    diag = entry_probe.get_diag()
    probe.eval()
    rows = []
    for frame in range(args.target_start, args.target_end + 1):
        img_path, ann_path = diag.find_files(
            args.data_root, TARGET_SPLIT, TARGET_SEQ, frame)
        if img_path is None or ann_path is None:
            raise RuntimeError('Missing target-dev frame {}'.format(frame))
        record = dict(
            split=TARGET_SPLIT, seq=TARGET_SEQ, frame=frame,
            image=img_path, annotation=ann_path, domain='real')
        img_tensor, meta, image_stats = diag.preprocess_image(
            img_path, transforms, img_scale, flip)
        if img_tensor is None:
            raise RuntimeError('Target preprocessing failed: {}'.format(frame))
        img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))
        with torch.no_grad():
            features = model.extract_feat(img_tensor)
            objectness = probe(features[args.feature_level])
            head, boxes, scores, layout, alignment = forward_main_candidates(
                model, features, meta['img_shape'])
            gt_boxes = scaled_gt_tensors(record, meta, boxes.device)
            if gt_boxes.numel() == 0:
                raise RuntimeError('Missing target GT: {}'.format(frame))
            ious = box_iou_rotated(
                boxes.float(), gt_boxes.float()).max(dim=1).values
            usable_index = select_level_candidate(
                scores, ious, layout, args.feature_level,
                min_iou=args.riou_thr)
            false_index = select_level_candidate(
                scores, ious, layout, args.feature_level,
                max_iou=args.false_iou_thr)
            if false_index is None:
                raise RuntimeError(
                    'No matched level0 false candidate for frame {}'.format(
                        frame))
            false = candidate_record(
                false_index, scores, ious, layout, objectness, stride)
            usable = None
            margin = None
            if usable_index is not None:
                usable = candidate_record(
                    usable_index, scores, ious, layout, objectness, stride)
                margin = (float(usable['objectness_logit'])
                          - float(false['objectness_logit']))
            rows.append(dict(
                role='target_dev_diagnosis_only', split=TARGET_SPLIT,
                seq=TARGET_SEQ, frame=int(frame), image_stats=image_stats,
                eligible=usable is not None,
                geometry_miss=usable is None,
                dense_best_riou=_number(ious.max().item()),
                level0_best_riou=_number(max(
                    float(ious[index].item()) for index, location in enumerate(layout)
                    if int(location['level']) == args.feature_level)),
                usable=usable, matched_level0_false=false,
                positive_logit=(None if usable is None
                                else usable['objectness_logit']),
                negative_logit=false['objectness_logit'],
                margin=_number(margin) if margin is not None else None,
                win=bool(margin is not None and margin > 0.0),
                decode_alignment=alignment))
        print('[target-dev] frame {} eligible={} margin={}'.format(
            frame, usable_index is not None,
            None if margin is None else _number(margin)))
    return dict(rows=rows)


def build_source_loaders(cfg, source_indexes: Sequence[int], seed: int):
    from mmdet.datasets import build_dataloader, build_dataset

    train_cfgs = cfg.data.train
    if not isinstance(train_cfgs, (list, tuple)):
        train_cfgs = [train_cfgs]
    if max(source_indexes) >= len(train_cfgs):
        raise ValueError(
            'Requested source index exceeds cfg.data.train length {}'.format(
                len(train_cfgs)))
    loaders = []
    metadata = []
    for source_index in source_indexes:
        source_cfg = train_cfgs[source_index]
        dataset = build_dataset(source_cfg)
        loader = build_dataloader(
            dataset, samples_per_gpu=1, workers_per_gpu=0,
            num_gpus=1, dist=False, shuffle=True, seed=seed)
        loaders.append((int(source_index), loader))
        metadata.append(dict(
            source_index=int(source_index), samples=len(dataset),
            ann_file=str(source_cfg.get('ann_file')),
            img_prefix=str(source_cfg.get('img_prefix'))))
    return loaders, metadata


def save_probe_checkpoint(path: str, probe, metadata: Dict):
    torch.save(dict(
        state_dict=probe.state_dict(), metadata=metadata,
        contains_detector_parameters=False), path)


def main():
    args = parse_args()
    canonical = validate_args(args)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    model, cfg = entry_probe.load_model(
        args.config, args.checkpoint, args.gpu)
    freeze_detector(model)
    versions_before = detector_parameter_versions(model)

    candidate_head = entry_probe.get_candidate_head(model, 'main')
    if args.feature_level >= len(candidate_head.anchor_generator.strides):
        raise ValueError('Feature level exceeds anchor-generator strides')
    stride = stride_value(
        candidate_head.anchor_generator.strides[args.feature_level])
    in_channels = int(cfg.model.neck.out_channels)
    if in_channels != 256:
        raise RuntimeError(
            'Canonical probe requires FPN out_channels=256, got {}'.format(
                in_channels))
    probe = FrozenP3SpatialObjectness(in_channels).cuda(
        'cuda:{}'.format(args.gpu))
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    detector_param_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_param_ids = {
        id(parameter) for group in optimizer.param_groups
        for parameter in group['params']}
    if detector_param_ids & optimizer_param_ids:
        raise RuntimeError('Detector parameters leaked into probe optimizer')

    source_loaders, source_metadata = build_source_loaders(
        cfg, args.source_indexes, args.seed)
    diag = entry_probe.get_diag()
    transforms, img_scale, flip = diag.build_test_transforms(cfg)
    val_records = discover_labeled_records(
        args.data_root, 'val', args.max_val_samples)
    if not val_records:
        raise RuntimeError('No source-validation records found')

    history = []
    best_key = None
    best_state = None
    best_validation = None
    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(
            model, probe, source_loaders, optimizer, args, stride)
        validation = evaluate_source_validation(
            model, probe, val_records, transforms, img_scale, flip,
            args, stride)
        summary = validation['summary']
        key = (
            float(summary.get('accuracy', 0.0)),
            float(summary.get('median_margin') or -float('inf')),
            -int(summary.get('heatmap_degenerate_images', 0)),
            -float(train_stats['mean_loss']))
        selected = best_key is None or key > best_key
        if selected:
            best_key = key
            best_state = copy.deepcopy(probe.state_dict())
            best_validation = validation
        history.append(dict(
            epoch=int(epoch), selected=bool(selected),
            train=train_stats, source_validation=summary))
        print(
            '[epoch {}/{}] loss={:.6f} source-val={}/{} ({:.3f}) '
            'median_margin={} selected={}'.format(
                epoch, args.epochs, train_stats['mean_loss'],
                summary['wins'], summary['count'], summary['accuracy'],
                summary['median_margin'], selected))

    if best_state is None or best_validation is None:
        raise RuntimeError('Source validation failed to select a probe')
    probe.load_state_dict(best_state)
    best_epoch = next(
        row['epoch'] for row in reversed(history) if row['selected'])
    probe_path = os.path.join(args.out_dir, 'probe_best_source_only.pth')
    save_probe_checkpoint(probe_path, probe, dict(
        probe=PROBE_NAME, protocol_version=PROTOCOL_VERSION,
        source_trained=True, target_used_for_training=False,
        target_used_for_checkpoint_selection=False,
        selected_epoch=int(best_epoch), config=os.path.abspath(args.config),
        detector_checkpoint=os.path.abspath(args.checkpoint)))

    # This is intentionally the first target-dev access in the process.
    target = evaluate_target_dev(
        model, probe, transforms, img_scale, flip, args, stride)
    geometry_misses = [
        int(row['frame']) for row in target['rows'] if row['geometry_miss']]
    gate = target_gate(
        target['rows'], best_validation['summary'], EXPECTED_ELIGIBLE,
        args.target_min_wins, args.source_val_min_accuracy)
    canonical_geometry = geometry_misses == EXPECTED_GEOMETRY_MISSES
    if not canonical_geometry:
        gate['checks']['expected_geometry_misses'] = False
        gate['decision'] = 'STOP_A_EVIDENCE_INSUFFICIENT'
        gate['interpretation'] = (
            'The level0 geometry baseline changed from the frozen protocol. '
            'This run cannot distinguish A from B and must not authorize '
            'training.')
    else:
        gate['checks']['expected_geometry_misses'] = True

    versions_after = detector_parameter_versions(model)
    detector_unchanged = versions_before == versions_after
    if not detector_unchanged:
        raise RuntimeError('Frozen detector parameters changed during probe')

    payload = dict(
        probe=PROBE_NAME,
        protocol_version=PROTOCOL_VERSION,
        canonical_protocol=bool(canonical),
        authorization_eligible=bool(canonical),
        data_role='source_trained_target_dev_diagnosis_only',
        config=os.path.abspath(args.config),
        detector_checkpoint=os.path.abspath(args.checkpoint),
        probe_checkpoint=os.path.abspath(probe_path),
        protocol=dict(
            feature_level=int(args.feature_level), fpn_name='P3',
            stride=_number(stride), in_channels=in_channels,
            trainable_module='Conv2d(256,1,kernel_size=3,padding=1)',
            gaussian_sigma_scale=_number(args.gaussian_sigma_scale),
            gaussian_min_sigma_strides=_number(
                args.gaussian_min_sigma_strides),
            positive_weight=_number(args.positive_weight),
            riou_thr=_number(args.riou_thr),
            false_iou_thr=_number(args.false_iou_thr),
            target_gate='usable_logit > matched_level0_false_logit',
            target_slice='real_seq02[137..169]',
            target_expected_eligible=EXPECTED_ELIGIBLE,
            target_expected_geometry_misses=EXPECTED_GEOMETRY_MISSES),
        isolation=dict(
            detector_frozen=True,
            detector_eval_mode=True,
            detector_parameters_in_optimizer=False,
            detector_parameter_versions_unchanged=detector_unchanged,
            regression_unchanged=True,
            main_classification_unchanged=True,
            score_fusion_used=False,
            main_checkpoint_written=False,
            probe_checkpoint_contains_detector_parameters=False,
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_evaluated_after_source_selection=True),
        source_train=dict(
            sources=source_metadata, epochs=int(args.epochs),
            optimizer='AdamW', lr=_number(args.lr),
            weight_decay=_number(args.weight_decay), history=history,
            selected_epoch=int(best_epoch)),
        source_validation=best_validation,
        target_dev=dict(
            geometry_misses=geometry_misses,
            eligible_count=sum(row['eligible'] for row in target['rows']),
            rows=target['rows']),
        gate=gate)
    if not canonical:
        payload['gate']['decision'] = 'NONCANONICAL_NO_AUTHORIZATION'
        payload['gate']['interpretation'] = (
            'Smoke-test result only; rerun the canonical full-data protocol '
            'before drawing a model-design conclusion.')

    out_json = os.path.join(args.out_dir, 'result.json')
    with open(out_json, 'w') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False,
                  allow_nan=False)
    print('[gate] {} wins={}/{} median_margin={} source_val_accuracy={}'.format(
        payload['gate']['decision'], payload['gate']['wins'],
        payload['gate']['eligible_count'],
        payload['gate']['median_margin'],
        best_validation['summary']['accuracy']))
    print('[out] {}'.format(out_json))


if __name__ == '__main__':
    main()
