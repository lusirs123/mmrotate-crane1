#!/usr/bin/env python3
"""Read-only audit of source/target P3 feature alignment.

This tool is the follow-up to ``Frozen-P3 Spatial Objectness Transfer Probe``.
It does not train a detector or a probe.  It compares the frozen P3 patches at
source positives/backgrounds and target usable/false candidates.

No raw FPN patch is serialized.  Source-only prototypes are frozen before the
target-dev slice is read.
"""

import argparse
import copy
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import frozen_p3_objectness_transfer_probe as transfer  # noqa: E402


PROBE_NAME = 'Frozen-P3 Feature Alignment Audit'
PROTOCOL_VERSION = 1
CANONICAL_CONFIG = 'crane_symeood_k1_brightaug.py'
CANONICAL_DETECTOR = 'epoch_20.pth'
CANONICAL_PROBE = 'probe_best_source_only.pth'
SOURCE_SPLIT = 'val'
SOURCE_SEQ = 'real_seq07'
TARGET_SPLIT = 'test'
TARGET_SEQ = 'real_seq02'
TARGET_START = 137
TARGET_END = 169
EXPECTED_GEOMETRY_MISSES = [164, 167]
EXPECTED_ELIGIBLE = 31


def parse_args():
    parser = argparse.ArgumentParser(description=PROBE_NAME)
    parser.add_argument('--config', required=True)
    parser.add_argument('--detector-checkpoint', required=True)
    parser.add_argument('--probe-checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-seq', default=SOURCE_SEQ)
    parser.add_argument('--feature-level', type=int, default=0)
    parser.add_argument('--max-source-samples', type=int, default=0,
                        help='0 means all source-real validation samples')
    parser.add_argument('--source-control-modulus', type=int, default=5)
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--false-iou-thr', type=float, default=0.1)
    parser.add_argument('--reconstruction-atol', type=float, default=1e-4)
    parser.add_argument('--source-min-accuracy', type=float, default=0.8)
    parser.add_argument('--target-min-count', type=int, default=26)
    parser.add_argument('--min-size-matched-source', type=int, default=10)
    parser.add_argument('--target-start', type=int, default=TARGET_START)
    parser.add_argument('--target-end', type=int, default=TARGET_END)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--allow-noncanonical', action='store_true')
    return parser.parse_args()


def validate_args(args) -> bool:
    if args.seed != 0:
        raise ValueError('The unified protocol requires --seed 0')
    if args.feature_level < 0:
        raise ValueError('--feature-level must be non-negative')
    if args.max_source_samples < 0:
        raise ValueError('--max-source-samples must be non-negative')
    if args.source_control_modulus < 2:
        raise ValueError('--source-control-modulus must be at least 2')
    if not 0.0 <= args.false_iou_thr < args.riou_thr <= 1.0:
        raise ValueError('Require 0 <= false-iou-thr < riou-thr <= 1')
    if args.reconstruction_atol <= 0.0:
        raise ValueError('--reconstruction-atol must be positive')
    if not 0.0 < args.source_min_accuracy <= 1.0:
        raise ValueError('--source-min-accuracy must be in (0, 1]')
    if args.target_min_count <= 0 or args.min_size_matched_source <= 0:
        raise ValueError('Gate counts must be positive')

    checks = dict(
        config=os.path.basename(args.config) == CANONICAL_CONFIG,
        detector=(os.path.basename(args.detector_checkpoint)
                  == CANONICAL_DETECTOR),
        probe=os.path.basename(args.probe_checkpoint) == CANONICAL_PROBE,
        source_seq=args.source_seq == SOURCE_SEQ,
        feature_level=int(args.feature_level) == 0,
        full_source=args.max_source_samples == 0,
        source_split=int(args.source_control_modulus) == 5,
        target_slice=(int(args.target_start) == TARGET_START
                      and int(args.target_end) == TARGET_END),
        thresholds=(math.isclose(args.riou_thr, 0.5)
                    and math.isclose(args.false_iou_thr, 0.1)),
        target_gate=int(args.target_min_count) == 26,
        source_gate=math.isclose(args.source_min_accuracy, 0.8),
        size_control=int(args.min_size_matched_source) == 10,
    )
    canonical = all(checks.values())
    if not canonical and not args.allow_noncanonical:
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(
            'Canonical feature-audit protocol mismatch: {}. '
            'Use --allow-noncanonical only for smoke tests.'.format(
                ', '.join(failed)))
    return canonical


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _number(value) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def module_parameter_versions(module) -> Dict[str, int]:
    return {name: int(parameter._version)
            for name, parameter in module.named_parameters()}


def load_frozen_probe(path: str, in_channels: int, gpu: int):
    checkpoint = torch.load(path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise RuntimeError('Invalid source-only probe checkpoint')
    if checkpoint.get('contains_detector_parameters') is not False:
        raise RuntimeError('Probe checkpoint isolation metadata is missing')
    metadata = checkpoint.get('metadata', {})
    if metadata.get('target_used_for_training') is not False:
        raise RuntimeError('Probe checkpoint does not prove target-free training')
    if metadata.get('target_used_for_checkpoint_selection') is not False:
        raise RuntimeError(
            'Probe checkpoint does not prove target-free checkpoint selection')
    probe = transfer.FrozenP3SpatialObjectness(in_channels)
    probe.load_state_dict(checkpoint['state_dict'], strict=True)
    probe = probe.cuda('cuda:{}'.format(gpu))
    probe.eval()
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    return probe, metadata


def extract_conv_patch(feature: torch.Tensor, row: int,
                       col: int) -> torch.Tensor:
    """Return the exact Cx3x3 patch used by a padding=1 convolution."""
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError('Expected one P3 feature map [1,C,H,W]')
    height, width = [int(value) for value in feature.shape[-2:]]
    if not 0 <= row < height or not 0 <= col < width:
        raise ValueError('Feature-grid location is out of bounds')
    padded = F.pad(feature[0], (1, 1, 1, 1))
    return padded[:, row:row + 3, col:col + 3].detach().float()


def exact_patch_audit(patch: torch.Tensor, conv,
                      actual_logit: float, atol: float,
                      top_channels: int = 12) -> Dict:
    if conv.kernel_size != (3, 3) or conv.padding != (1, 1):
        raise RuntimeError('Feature audit requires a 3x3 padding=1 probe')
    if conv.weight.shape[0] != 1 or patch.shape != conv.weight[0].shape:
        raise RuntimeError('Probe weight/P3 patch shape mismatch')
    weight = conv.weight[0].detach().float()
    contribution = patch * weight
    bias = 0.0 if conv.bias is None else float(conv.bias[0].detach().item())
    reconstructed = float(contribution.sum().item()) + bias
    error = abs(reconstructed - float(actual_logit))
    if error > float(atol):
        raise RuntimeError(
            'Probe logit reconstruction failed: {:.8g} > {:.8g}'.format(
                error, atol))
    per_channel = contribution.sum(dim=(1, 2))
    count = min(int(top_channels), int(per_channel.numel()))
    positive_indices = torch.topk(
        per_channel, k=count, largest=True, sorted=True).indices
    negative_indices = torch.topk(
        per_channel, k=count, largest=False, sorted=True).indices

    def channel_rows(indices):
        return [dict(
            channel=int(index.item()),
            contribution=_number(per_channel[index].item()),
            patch_norm=_number(patch[index].norm().item()),
            weight_norm=_number(weight[index].norm().item()))
            for index in indices]

    return dict(
        actual_logit=_number(actual_logit),
        reconstructed_logit=_number(reconstructed),
        reconstruction_abs_error=_number(error),
        bias=_number(bias),
        patch_norm=_number(patch.norm().item()),
        weight_norm=_number(weight.norm().item()),
        patch_weight_cosine=_number(F.cosine_similarity(
            patch.reshape(1, -1), weight.reshape(1, -1), dim=1).item()),
        positive_contribution=_number(contribution.clamp_min(0).sum().item()),
        negative_contribution=_number(contribution.clamp_max(0).sum().item()),
        spatial_contribution=[
            [_number(value) for value in row]
            for row in contribution.sum(dim=0).tolist()],
        top_positive_channels=channel_rows(positive_indices),
        top_negative_channels=channel_rows(negative_indices))


def normalized_mean(vectors: torch.Tensor) -> torch.Tensor:
    return F.normalize(F.normalize(vectors, dim=1).mean(dim=0), dim=0)


def build_source_prototypes(positive: torch.Tensor,
                            negative: torch.Tensor) -> Dict[str, torch.Tensor]:
    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError('Prototype inputs must be [N,D]')
    if positive.shape[1] != negative.shape[1]:
        raise ValueError('Positive/negative prototype dimensions differ')
    if positive.shape[0] < 2 or negative.shape[0] < 2:
        raise ValueError('Need at least two source samples per prototype')
    positive = positive.float()
    negative = negative.float()
    combined = torch.cat([positive, negative], dim=0)
    center = combined.mean(dim=0)
    scale = combined.std(dim=0, unbiased=False).clamp_min(1e-6)
    positive_white = (positive - center) / scale
    negative_white = (negative - center) / scale
    return dict(
        raw_positive=positive.mean(dim=0),
        raw_negative=negative.mean(dim=0),
        cosine_positive=normalized_mean(positive),
        cosine_negative=normalized_mean(negative),
        whiten_center=center,
        whiten_scale=scale,
        whiten_positive=normalized_mean(positive_white),
        whiten_negative=normalized_mean(negative_white))


def prototype_scores(vector: torch.Tensor,
                     prototypes: Dict[str, torch.Tensor]) -> Dict:
    vector = vector.detach().float().cpu().reshape(-1)
    raw_pos = torch.norm(vector - prototypes['raw_positive']).item()
    raw_neg = torch.norm(vector - prototypes['raw_negative']).item()
    unit = F.normalize(vector, dim=0)
    cosine_pos = torch.dot(unit, prototypes['cosine_positive']).item()
    cosine_neg = torch.dot(unit, prototypes['cosine_negative']).item()
    whitened = ((vector - prototypes['whiten_center'])
                / prototypes['whiten_scale'])
    whitened = F.normalize(whitened, dim=0)
    whiten_pos = torch.dot(
        whitened, prototypes['whiten_positive']).item()
    whiten_neg = torch.dot(
        whitened, prototypes['whiten_negative']).item()
    return dict(
        vector_norm=_number(vector.norm().item()),
        raw_distance_positive=_number(raw_pos),
        raw_distance_negative=_number(raw_neg),
        raw_preference_positive=_number(raw_neg - raw_pos),
        cosine_positive=_number(cosine_pos),
        cosine_negative=_number(cosine_neg),
        cosine_preference_positive=_number(cosine_pos - cosine_neg),
        whitened_cosine_positive=_number(whiten_pos),
        whitened_cosine_negative=_number(whiten_neg),
        whitened_preference_positive=_number(whiten_pos - whiten_neg))


def _candidate_patch_record(feature, objectness, index, scores, ious,
                            layout, conv, stride, atol):
    candidate = transfer.candidate_record(
        index, scores, ious, layout, objectness, stride)
    row, col = int(candidate['row']), int(candidate['col'])
    patch = extract_conv_patch(feature, row, col)
    audit = exact_patch_audit(
        patch, conv, candidate['objectness_logit'], atol)
    candidate['feature_audit'] = audit
    return candidate, patch.reshape(-1).detach().cpu()


def collect_source(model, probe, records: Sequence[Dict], transforms,
                   img_scale, flip, args, stride: float):
    from mmcv.ops import box_iou_rotated

    diag = transfer.entry_probe.get_diag()
    samples = []
    rows = []
    for record_index, record in enumerate(records):
        img_tensor, meta, image_stats = diag.preprocess_image(
            record['image'], transforms, img_scale, flip)
        if img_tensor is None:
            continue
        img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))
        with torch.no_grad():
            features = model.extract_feat(img_tensor)
            feature = features[args.feature_level]
            objectness = probe(feature)
            _head, boxes, scores, layout, alignment = (
                transfer.forward_main_candidates(
                    model, features, meta['img_shape']))
            gt_boxes = transfer.scaled_gt_tensors(record, meta, boxes.device)
            if gt_boxes.numel() == 0:
                continue
            ious = box_iou_rotated(
                boxes.float(), gt_boxes.float()).max(dim=1).values
            false_index = transfer.select_level_candidate(
                scores, ious, layout, args.feature_level,
                max_iou=args.false_iou_thr)
            if false_index is None:
                continue
            negative, negative_vector = _candidate_patch_record(
                feature, objectness, false_index, scores, ious, layout,
                probe.objectness, stride, args.reconstruction_atol)
            height, width = [int(value) for value in feature.shape[-2:]]
            for gt_index, gt_box in enumerate(gt_boxes):
                row, col = transfer.nearest_grid_location(
                    gt_box, height, width, stride, meta['img_shape'])
                positive_logit = float(objectness[0, 0, row, col].item())
                positive_patch = extract_conv_patch(feature, row, col)
                positive_audit = exact_patch_audit(
                    positive_patch, probe.objectness, positive_logit,
                    args.reconstruction_atol)
                long_side = float(max(gt_box[2].item(), gt_box[3].item()))
                short_side = float(min(gt_box[2].item(), gt_box[3].item()))
                row_record = dict(
                    role='source_real_validation_control',
                    split=record['split'], seq=record['seq'],
                    frame=int(record['frame']), gt_index=int(gt_index),
                    image_stats=image_stats,
                    gt_long_side=_number(long_side),
                    gt_short_side=_number(short_side),
                    positive=dict(
                        level=int(args.feature_level), row=int(row), col=int(col),
                        source_grid_center=[
                            _number((col + 0.5) * stride),
                            _number((row + 0.5) * stride)],
                        objectness_logit=_number(positive_logit),
                        feature_audit=positive_audit),
                    hard_negative=copy.deepcopy(negative),
                    paired_margin=_number(
                        positive_logit - negative['objectness_logit']),
                    paired_win=bool(
                        positive_logit > negative['objectness_logit']),
                    decode_alignment=alignment)
                rows.append(row_record)
                samples.append(dict(
                    order=len(samples), row=row_record,
                    positive_vector=positive_patch.reshape(-1).detach().cpu(),
                    negative_vector=negative_vector.clone()))
        if (record_index + 1) % 50 == 0:
            print('[source-readonly] {}/{} images'.format(
                record_index + 1, len(records)))
    if not samples:
        raise RuntimeError('No source controls were collected')
    return samples, rows


def collect_target(model, probe, transforms, img_scale, flip,
                   args, stride: float):
    from mmcv.ops import box_iou_rotated

    diag = transfer.entry_probe.get_diag()
    samples = []
    rows = []
    for frame_id in range(args.target_start, args.target_end + 1):
        img_path, ann_path = diag.find_files(
            args.data_root, TARGET_SPLIT, TARGET_SEQ, frame_id)
        if img_path is None or ann_path is None:
            raise RuntimeError('Missing target-dev frame {}'.format(frame_id))
        record = dict(
            split=TARGET_SPLIT, seq=TARGET_SEQ, frame=frame_id,
            image=img_path, annotation=ann_path, domain='real')
        img_tensor, meta, image_stats = diag.preprocess_image(
            img_path, transforms, img_scale, flip)
        if img_tensor is None:
            raise RuntimeError('Target preprocessing failed')
        img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))
        with torch.no_grad():
            features = model.extract_feat(img_tensor)
            feature = features[args.feature_level]
            objectness = probe(feature)
            _head, boxes, scores, layout, alignment = (
                transfer.forward_main_candidates(
                    model, features, meta['img_shape']))
            gt_boxes = transfer.scaled_gt_tensors(record, meta, boxes.device)
            if gt_boxes.numel() == 0:
                raise RuntimeError('Missing target GT')
            ious = box_iou_rotated(
                boxes.float(), gt_boxes.float()).max(dim=1).values
            usable_index = transfer.select_level_candidate(
                scores, ious, layout, args.feature_level,
                min_iou=args.riou_thr)
            false_index = transfer.select_level_candidate(
                scores, ious, layout, args.feature_level,
                max_iou=args.false_iou_thr)
            if false_index is None:
                raise RuntimeError('No target level0 matched false candidate')
            false, false_vector = _candidate_patch_record(
                feature, objectness, false_index, scores, ious, layout,
                probe.objectness, stride, args.reconstruction_atol)
            usable = None
            usable_vector = None
            gaussian_value = None
            if usable_index is not None:
                usable, usable_vector = _candidate_patch_record(
                    feature, objectness, usable_index, scores, ious, layout,
                    probe.objectness, stride, args.reconstruction_atol)
                height, width = [int(value) for value in feature.shape[-2:]]
                valid = transfer.valid_grid_mask(
                    height, width, meta['img_shape'], stride, feature.device)
                gaussian = transfer.oriented_gaussian_heatmap(
                    gt_boxes, height, width, stride, 0.25, 1.0, valid)
                gaussian_value = float(gaussian[
                    0, int(usable['row']), int(usable['col'])].item())
            gt_long = float(max(gt_boxes[0, 2].item(), gt_boxes[0, 3].item()))
            gt_short = float(min(gt_boxes[0, 2].item(), gt_boxes[0, 3].item()))
            row_record = dict(
                role='target_dev_diagnosis_only', split=TARGET_SPLIT,
                seq=TARGET_SEQ, frame=int(frame_id), image_stats=image_stats,
                eligible=usable is not None,
                geometry_miss=usable is None,
                dense_best_riou=_number(ious.max().item()),
                gt_long_side=_number(gt_long),
                gt_short_side=_number(gt_short),
                gaussian_target_at_usable=_number(gaussian_value)
                if gaussian_value is not None else None,
                usable=usable,
                matched_level0_false=false,
                probe_paired_margin=(
                    None if usable is None else _number(
                        usable['objectness_logit']
                        - false['objectness_logit'])),
                probe_paired_win=bool(
                    usable is not None
                    and usable['objectness_logit']
                    > false['objectness_logit']),
                decode_alignment=alignment)
            rows.append(row_record)
            if usable is not None:
                samples.append(dict(
                    row=row_record, usable_vector=usable_vector,
                    false_vector=false_vector))
        print('[target-readonly] frame {} eligible={}'.format(
            frame_id, usable is not None))
    return samples, rows


def attach_prototype_scores(samples: Sequence[Dict], prototypes: Dict,
                            positive_key: str, negative_key: str,
                            positive_record_key: str,
                            negative_record_key: str):
    for sample in samples:
        positive_scores = prototype_scores(sample[positive_key], prototypes)
        negative_scores = prototype_scores(sample[negative_key], prototypes)
        sample['row'][positive_record_key]['source_prototype'] = positive_scores
        sample['row'][negative_record_key]['source_prototype'] = negative_scores


def accuracy(values: Sequence[bool]) -> float:
    return 0.0 if not values else float(sum(values)) / float(len(values))


def summarize_source_controls(control_samples: Sequence[Dict]) -> Dict:
    positive_cosine = [
        sample['row']['positive']['source_prototype'][
            'cosine_preference_positive'] > 0.0
        for sample in control_samples]
    negative_cosine = [
        sample['row']['hard_negative']['source_prototype'][
            'cosine_preference_positive'] < 0.0
        for sample in control_samples]
    positive_white = [
        sample['row']['positive']['source_prototype'][
            'whitened_preference_positive'] > 0.0
        for sample in control_samples]
    negative_white = [
        sample['row']['hard_negative']['source_prototype'][
            'whitened_preference_positive'] < 0.0
        for sample in control_samples]
    return dict(
        count=len(control_samples),
        positive_cosine_accuracy=_number(accuracy(positive_cosine)),
        negative_cosine_accuracy=_number(accuracy(negative_cosine)),
        positive_whitened_accuracy=_number(accuracy(positive_white)),
        negative_whitened_accuracy=_number(accuracy(negative_white)),
        probe_paired_accuracy=_number(accuracy([
            sample['row']['paired_win'] for sample in control_samples])))


def summarize_target(samples: Sequence[Dict]) -> Dict:
    cosine_negative_like = []
    cosine_pair_inverted = []
    whitened_negative_like = []
    whitened_pair_inverted = []
    probe_wins = []
    for sample in samples:
        usable = sample['row']['usable']['source_prototype']
        false = sample['row']['matched_level0_false']['source_prototype']
        cosine_negative_like.append(
            usable['cosine_preference_positive'] < 0.0)
        cosine_pair_inverted.append(
            usable['cosine_preference_positive']
            <= false['cosine_preference_positive'])
        whitened_negative_like.append(
            usable['whitened_preference_positive'] < 0.0)
        whitened_pair_inverted.append(
            usable['whitened_preference_positive']
            <= false['whitened_preference_positive'])
        probe_wins.append(sample['row']['probe_paired_win'])
    return dict(
        eligible_count=len(samples),
        usable_cosine_negative_like=sum(cosine_negative_like),
        cosine_pair_inverted=sum(cosine_pair_inverted),
        usable_whitened_negative_like=sum(whitened_negative_like),
        whitened_pair_inverted=sum(whitened_pair_inverted),
        probe_paired_wins=sum(probe_wins),
        usable_cosine_negative_like_fraction=_number(
            accuracy(cosine_negative_like)),
        cosine_pair_inverted_fraction=_number(
            accuracy(cosine_pair_inverted)),
        usable_whitened_negative_like_fraction=_number(
            accuracy(whitened_negative_like)),
        whitened_pair_inverted_fraction=_number(
            accuracy(whitened_pair_inverted)))


def size_matched_source_summary(source_rows: Sequence[Dict],
                                target_rows: Sequence[Dict]) -> Dict:
    eligible = [row for row in target_rows if row['eligible']]
    target_long = [float(row['gt_long_side']) for row in eligible]
    target_short = [float(row['gt_short_side']) for row in eligible]
    if not target_long:
        return dict(count=0, probe_paired_accuracy=0.0)
    matched = [
        row for row in source_rows
        if min(target_long) <= float(row['gt_long_side']) <= max(target_long)
        and min(target_short) <= float(row['gt_short_side']) <= max(target_short)]
    return dict(
        target_long_side_range=[_number(min(target_long)),
                                _number(max(target_long))],
        target_short_side_range=[_number(min(target_short)),
                                 _number(max(target_short))],
        count=len(matched),
        probe_paired_wins=sum(row['paired_win'] for row in matched),
        probe_paired_accuracy=_number(accuracy([
            row['paired_win'] for row in matched])),
        frames=[int(row['frame']) for row in matched])


def reconstruction_max_error(source_rows: Sequence[Dict],
                             target_rows: Sequence[Dict]) -> float:
    errors = []
    for row in source_rows:
        errors.append(row['positive']['feature_audit'][
            'reconstruction_abs_error'])
        errors.append(row['hard_negative']['feature_audit'][
            'reconstruction_abs_error'])
    for row in target_rows:
        errors.append(row['matched_level0_false']['feature_audit'][
            'reconstruction_abs_error'])
        if row['usable'] is not None:
            errors.append(row['usable']['feature_audit'][
                'reconstruction_abs_error'])
    return float(max(errors)) if errors else float('inf')


def make_gate(source_control: Dict, target_summary: Dict,
              size_control: Dict, geometry_misses: Sequence[int],
              max_reconstruction_error: float, args) -> Dict:
    checks = dict(
        eligible_count=(target_summary['eligible_count'] == EXPECTED_ELIGIBLE),
        expected_geometry_misses=(list(geometry_misses)
                                  == EXPECTED_GEOMETRY_MISSES),
        exact_reconstruction=(max_reconstruction_error
                              <= float(args.reconstruction_atol)),
        source_positive_cosine=(
            source_control['positive_cosine_accuracy']
            >= float(args.source_min_accuracy)),
        source_negative_cosine=(
            source_control['negative_cosine_accuracy']
            >= float(args.source_min_accuracy)),
        source_positive_whitened=(
            source_control['positive_whitened_accuracy']
            >= float(args.source_min_accuracy)),
        source_negative_whitened=(
            source_control['negative_whitened_accuracy']
            >= float(args.source_min_accuracy)),
        enough_size_matched_source=(
            size_control['count'] >= int(args.min_size_matched_source)),
        size_matched_source_works=(
            size_control['probe_paired_accuracy']
            >= float(args.source_min_accuracy)),
        target_usable_cosine_negative_like=(
            target_summary['usable_cosine_negative_like']
            >= int(args.target_min_count)),
        target_cosine_pair_inverted=(
            target_summary['cosine_pair_inverted']
            >= int(args.target_min_count)),
        target_usable_whitened_negative_like=(
            target_summary['usable_whitened_negative_like']
            >= int(args.target_min_count)),
        target_whitened_pair_inverted=(
            target_summary['whitened_pair_inverted']
            >= int(args.target_min_count)))
    supported = all(checks.values())
    return dict(
        decision=('B_STRONGLY_SUPPORTED' if supported
                  else 'B_NOT_CONFIRMED'),
        checks=checks,
        plain_interpretation=(
            'Target correct P3 features look like source background even '
            'after removing feature magnitude, and matched false locations '
            'look more like source targets than the usable locations. This '
            'strongly supports a P3 feature-domain shift, but is still a '
            'diagnosis rather than a deployable gain.'
            if supported else
            'The independent source prototypes do not consistently reproduce '
            'the inversion. Do not claim that P3 feature-domain shift is the '
            'sole cause; inspect which failed check remains confounded.'))


def main():
    args = parse_args()
    canonical = validate_args(args)
    set_seed(args.seed)

    model, cfg = transfer.entry_probe.load_model(
        args.config, args.detector_checkpoint, args.gpu)
    transfer.freeze_detector(model)
    detector_versions_before = module_parameter_versions(model)
    candidate_head = transfer.entry_probe.get_candidate_head(model, 'main')
    if args.feature_level >= len(candidate_head.anchor_generator.strides):
        raise ValueError('Feature level exceeds anchor-generator strides')
    stride = transfer.stride_value(
        candidate_head.anchor_generator.strides[args.feature_level])
    in_channels = int(cfg.model.neck.out_channels)
    if in_channels != 256:
        raise RuntimeError('Canonical P3 audit requires 256 channels')
    probe, probe_metadata = load_frozen_probe(
        args.probe_checkpoint, in_channels, args.gpu)
    probe_versions_before = module_parameter_versions(probe)

    diag = transfer.entry_probe.get_diag()
    transforms, img_scale, flip = diag.build_test_transforms(cfg)
    source_records = [
        record for record in transfer.discover_labeled_records(
            args.data_root, SOURCE_SPLIT, 0)
        if record['seq'] == args.source_seq]
    if args.max_source_samples > 0:
        source_records = source_records[:args.max_source_samples]
    if not source_records:
        raise RuntimeError('No source-real validation records found')

    source_samples, source_rows = collect_source(
        model, probe, source_records, transforms, img_scale, flip,
        args, stride)
    reference = [
        sample for sample in source_samples
        if sample['order'] % args.source_control_modulus != 0]
    controls = [
        sample for sample in source_samples
        if sample['order'] % args.source_control_modulus == 0]
    if len(reference) < 2 or len(controls) < 2:
        raise RuntimeError('Source reference/control split is too small')
    prototypes = build_source_prototypes(
        torch.stack([sample['positive_vector'] for sample in reference]),
        torch.stack([sample['negative_vector'] for sample in reference]))
    # Freeze source-only statistics before target-dev is read.
    prototypes = {key: value.detach().clone().cpu()
                  for key, value in prototypes.items()}
    attach_prototype_scores(
        source_samples, prototypes,
        'positive_vector', 'negative_vector',
        'positive', 'hard_negative')
    source_control_summary = summarize_source_controls(controls)

    # First target-dev access happens only after source prototypes are frozen.
    target_samples, target_rows = collect_target(
        model, probe, transforms, img_scale, flip, args, stride)
    attach_prototype_scores(
        target_samples, prototypes,
        'usable_vector', 'false_vector',
        'usable', 'matched_level0_false')
    target_summary = summarize_target(target_samples)
    size_control = size_matched_source_summary(source_rows, target_rows)
    geometry_misses = [
        int(row['frame']) for row in target_rows if row['geometry_miss']]
    max_error = reconstruction_max_error(source_rows, target_rows)
    gate = make_gate(
        source_control_summary, target_summary, size_control,
        geometry_misses, max_error, args)

    detector_unchanged = (
        detector_versions_before == module_parameter_versions(model))
    probe_unchanged = probe_versions_before == module_parameter_versions(probe)
    if not detector_unchanged or not probe_unchanged:
        raise RuntimeError('Read-only parameter invariant failed')

    payload = dict(
        audit=PROBE_NAME,
        protocol_version=PROTOCOL_VERSION,
        canonical_protocol=bool(canonical),
        data_role='source_reference_target_dev_diagnosis_only',
        config=os.path.abspath(args.config),
        detector_checkpoint=os.path.abspath(args.detector_checkpoint),
        probe_checkpoint=os.path.abspath(args.probe_checkpoint),
        probe_checkpoint_metadata=probe_metadata,
        protocol=dict(
            feature_level=int(args.feature_level), fpn_name='P3',
            stride=_number(stride), patch_shape=[256, 3, 3],
            source_split='real_seq07 modulo-5 reference/control',
            source_reference_count=len(reference),
            source_control_count=len(controls),
            target_slice='real_seq02[137..169]',
            target_expected_eligible=EXPECTED_ELIGIBLE,
            target_min_count=int(args.target_min_count),
            source_min_accuracy=_number(args.source_min_accuracy)),
        isolation=dict(
            performs_optimizer_step=False,
            creates_optimizer=False,
            writes_checkpoint=False,
            detector_frozen=True,
            probe_frozen=True,
            detector_parameters_unchanged=detector_unchanged,
            probe_parameters_unchanged=probe_unchanged,
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            source_prototypes_frozen_before_target=True,
            target_geometry_used_for_source_prototypes=False,
            target_geometry_used_for_diagnostic_size_control=True,
            raw_fpn_patches_serialized=False),
        prototype_metadata=dict(
            dimension=int(prototypes['raw_positive'].numel()),
            raw_positive_norm=_number(
                prototypes['raw_positive'].norm().item()),
            raw_negative_norm=_number(
                prototypes['raw_negative'].norm().item()),
            vectors_serialized=False),
        reconstruction_max_abs_error=_number(max_error),
        source=dict(
            summary=source_control_summary,
            size_matched_target_geometry=size_control,
            rows=source_rows),
        target_dev=dict(
            geometry_misses=geometry_misses,
            summary=target_summary,
            rows=target_rows),
        gate=gate)
    if not canonical:
        payload['gate']['decision'] = 'NONCANONICAL_NO_CONCLUSION'
        payload['gate']['plain_interpretation'] = (
            'Smoke test only; run the full canonical audit before concluding.')
    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, 'w') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False,
                  allow_nan=False)
    print('[audit] {} cosine_negative_like={}/{} cosine_inverted={}/{} '
          'whitened_negative_like={}/{} whitened_inverted={}/{}'.format(
              payload['gate']['decision'],
              target_summary['usable_cosine_negative_like'],
              target_summary['eligible_count'],
              target_summary['cosine_pair_inverted'],
              target_summary['eligible_count'],
              target_summary['usable_whitened_negative_like'],
              target_summary['eligible_count'],
              target_summary['whitened_pair_inverted'],
              target_summary['eligible_count']))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
