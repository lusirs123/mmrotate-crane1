#!/usr/bin/env python3
"""Read-only P3/P4 neighborhood feature rescue audit.

The audit answers which bounded architecture direction is justified after the
canonical Frozen-P3 feature-alignment failure:

1. a nearby P3 location contains source-compatible target semantics;
2. the corresponding P4 neighborhood contains them; or
3. neither ordinary FPN neighborhood contains them.

No optimizer is created, no parameter is updated, and no checkpoint or raw
FPN patch is written.  Source-only prototypes are frozen before target-dev is
read.
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


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import frozen_p3_feature_alignment_audit as alignment  # noqa: E402
from crane_project.tools import frozen_p3_objectness_transfer_probe as transfer  # noqa: E402


AUDIT_NAME = 'P3/P4 Neighborhood Feature Rescue Audit'
PROTOCOL_VERSION = 2
CANONICAL_CONFIG = 'crane_symeood_k1_brightaug.py'
CANONICAL_CHECKPOINT = 'epoch_20.pth'
SOURCE_SPLIT = 'val'
SOURCE_SEQ = 'real_seq07'
TARGET_SPLIT = 'test'
TARGET_SEQ = 'real_seq02'
TARGET_START = 137
TARGET_END = 169
EXPECTED_GEOMETRY_MISSES = [164, 167]
EXPECTED_ELIGIBLE = 31
CANONICAL_LEVELS = [0, 1]


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument('--config', required=True)
    parser.add_argument('--detector-checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-seq', default=SOURCE_SEQ)
    parser.add_argument('--levels', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--physical-radius-px', type=float, default=16.0)
    parser.add_argument('--min-target-gaussian', type=float, default=0.1)
    parser.add_argument('--source-folds', type=int, default=5)
    parser.add_argument('--min-fold-votes', type=int, default=4)
    parser.add_argument('--positive-quantile', type=float, default=0.1)
    parser.add_argument('--max-source-samples', type=int, default=0,
                        help='0 means all source-real validation samples')
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--false-iou-thr', type=float, default=0.1)
    parser.add_argument('--source-min-accuracy', type=float, default=0.8)
    parser.add_argument('--target-min-rescues', type=int, default=26)
    parser.add_argument('--target-start', type=int, default=TARGET_START)
    parser.add_argument('--target-end', type=int, default=TARGET_END)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--allow-noncanonical', action='store_true')
    return parser.parse_args()


def validate_args(args) -> bool:
    levels = [int(value) for value in args.levels]
    if args.seed != 0:
        raise ValueError('The unified protocol requires --seed 0')
    if not levels or len(levels) != len(set(levels)) or min(levels) < 0:
        raise ValueError('--levels requires unique non-negative values')
    if args.physical_radius_px <= 0.0:
        raise ValueError('--physical-radius-px must be positive')
    if not 0.0 <= args.min_target_gaussian <= 1.0:
        raise ValueError('--min-target-gaussian must be in [0, 1]')
    if args.source_folds < 2:
        raise ValueError('--source-folds must be at least 2')
    if not 1 <= args.min_fold_votes <= args.source_folds:
        raise ValueError('--min-fold-votes must be in [1, source-folds]')
    if not 0.0 < args.positive_quantile < 0.5:
        raise ValueError('--positive-quantile must be in (0, 0.5)')
    if args.max_source_samples < 0:
        raise ValueError('--max-source-samples must be non-negative')
    if not 0.0 <= args.false_iou_thr < args.riou_thr <= 1.0:
        raise ValueError('Require 0 <= false-iou-thr < riou-thr <= 1')
    if not 0.0 < args.source_min_accuracy <= 1.0:
        raise ValueError('--source-min-accuracy must be in (0, 1]')
    if args.target_min_rescues <= 0:
        raise ValueError('--target-min-rescues must be positive')

    checks = dict(
        config=os.path.basename(args.config) == CANONICAL_CONFIG,
        checkpoint=(os.path.basename(args.detector_checkpoint)
                    == CANONICAL_CHECKPOINT),
        source_seq=args.source_seq == SOURCE_SEQ,
        levels=levels == CANONICAL_LEVELS,
        radius=math.isclose(float(args.physical_radius_px), 16.0),
        target_gaussian=math.isclose(float(args.min_target_gaussian), 0.1),
        source_folds=int(args.source_folds) == 5,
        fold_votes=int(args.min_fold_votes) == 4,
        positive_quantile=math.isclose(float(args.positive_quantile), 0.1),
        full_source=args.max_source_samples == 0,
        thresholds=(math.isclose(args.riou_thr, 0.5)
                    and math.isclose(args.false_iou_thr, 0.1)),
        source_gate=math.isclose(args.source_min_accuracy, 0.8),
        target_gate=int(args.target_min_rescues) == 26,
        target_slice=(int(args.target_start) == TARGET_START
                      and int(args.target_end) == TARGET_END))
    canonical = all(checks.values())
    if not canonical and not args.allow_noncanonical:
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(
            'Canonical neighborhood-audit mismatch: {}. '
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


def stride_value(stride) -> float:
    return transfer.stride_value(stride)


def radius_in_cells(physical_radius_px: float, stride: float) -> int:
    cells = float(physical_radius_px) / float(stride)
    rounded = int(round(cells))
    if rounded < 1 or not math.isclose(cells, rounded, abs_tol=1e-6):
        raise ValueError(
            'Physical radius must be a positive integer number of cells')
    return rounded


def physical_to_grid(x: float, y: float, stride: float,
                     height: int, width: int,
                     img_shape) -> Tuple[int, int]:
    img_h, img_w = [int(value) for value in img_shape[:2]]
    valid_height = min(height, int(math.ceil(img_h / float(stride))))
    valid_width = min(width, int(math.ceil(img_w / float(stride))))
    if valid_height <= 0 or valid_width <= 0:
        raise ValueError('No valid grid locations')
    row = int(round(float(y) / float(stride) - 0.5))
    col = int(round(float(x) / float(stride) - 0.5))
    return (max(0, min(row, valid_height - 1)),
            max(0, min(col, valid_width - 1)))


def grid_center(row: int, col: int, stride: float) -> List[float]:
    return [_number((col + 0.5) * stride),
            _number((row + 0.5) * stride)]


def enumerate_neighborhood(center_row: int, center_col: int,
                           radius_cells: int, height: int, width: int,
                           img_shape, stride: float) -> List[Tuple[int, int]]:
    img_h, img_w = [int(value) for value in img_shape[:2]]
    rows = []
    for row in range(center_row - radius_cells,
                     center_row + radius_cells + 1):
        for col in range(center_col - radius_cells,
                         center_col + radius_cells + 1):
            if not (0 <= row < height and 0 <= col < width):
                continue
            x, y = grid_center(row, col, stride)
            if x < float(img_w) and y < float(img_h):
                rows.append((int(row), int(col)))
    return rows


def main_candidate_record(index: int, scores: torch.Tensor,
                          ious: torch.Tensor, layout: Sequence[Dict],
                          stride: float) -> Dict:
    location = layout[int(index)]
    row, col = int(location['row']), int(location['col'])
    return dict(
        candidate_index=int(index), level=int(location['level']),
        row=row, col=col, anchor_id=int(location['anchor_id']),
        source_grid_center=grid_center(row, col, stride),
        main_cls_score=_number(scores[index].item()),
        riou=_number(ious[index].item()))


def patch_vector(feature: torch.Tensor, row: int, col: int) -> torch.Tensor:
    return alignment.extract_conv_patch(
        feature, row, col).reshape(-1).detach().cpu()


def ensemble_scores(vector: torch.Tensor,
                    fold_models: Sequence[Dict]) -> Dict:
    folds = []
    for model in fold_models:
        scores = alignment.prototype_scores(vector, model['prototypes'])
        folds.append(dict(
            fold_id=int(model['fold_id']),
            cosine_positive=scores['cosine_positive'],
            whitened_cosine_positive=scores[
                'whitened_cosine_positive'],
            cosine_positive_threshold=model[
                'cosine_positive_threshold'],
            whitened_positive_threshold=model[
                'whitened_positive_threshold']))
    return dict(
        fold_count=len(folds), folds=folds,
        mean_cosine_positive=_number(np.mean([
            fold['cosine_positive'] for fold in folds])),
        mean_whitened_cosine_positive=_number(np.mean([
            fold['whitened_cosine_positive'] for fold in folds])))


def location_scores(feature: torch.Tensor, row: int, col: int,
                    stride: float, fold_models: Sequence[Dict]) -> Dict:
    vector = patch_vector(feature, row, col)
    return dict(
        row=int(row), col=int(col),
        source_grid_center=grid_center(row, col, stride),
        source_ensemble=ensemble_scores(vector, fold_models))


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError('Cannot calculate a quantile from no values')
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def contiguous_fold_ids(count: int, folds: int) -> List[int]:
    if count < folds:
        raise ValueError('Source sample count is smaller than source folds')
    return [min((index * folds) // count, folds - 1)
            for index in range(count)]


def source_level_control_summary(records: Sequence[Dict]) -> Dict:
    positive_cosine = [record['positive_cosine_pass'] for record in records]
    negative_cosine = [record['negative_cosine_pass'] for record in records]
    positive_white = [record['positive_whitened_pass'] for record in records]
    negative_white = [record['negative_whitened_pass'] for record in records]
    return dict(
        count=len(records),
        positive_cosine_accuracy=_number(
            alignment.accuracy(positive_cosine)),
        negative_cosine_accuracy=_number(
            alignment.accuracy(negative_cosine)),
        positive_whitened_accuracy=_number(
            alignment.accuracy(positive_white)),
        negative_whitened_accuracy=_number(
            alignment.accuracy(negative_white)))


def collect_source(model, records: Sequence[Dict], transforms,
                   img_scale, flip, args, strides: Dict[int, float]):
    from mmcv.ops import box_iou_rotated

    diag = transfer.entry_probe.get_diag()
    samples_by_level = {int(level): [] for level in args.levels}
    rows = []
    for record_index, record in enumerate(records):
        img_tensor, meta, _stats = diag.preprocess_image(
            record['image'], transforms, img_scale, flip)
        if img_tensor is None:
            continue
        img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))
        with torch.no_grad():
            features = model.extract_feat(img_tensor)
            _head, boxes, scores, layout, _alignment = (
                transfer.forward_main_candidates(
                    model, features, meta['img_shape']))
            gt_boxes = transfer.scaled_gt_tensors(record, meta, boxes.device)
            if gt_boxes.numel() == 0:
                continue
            ious = box_iou_rotated(
                boxes.float(), gt_boxes.float()).max(dim=1).values
            false_by_level = {
                int(level): transfer.select_level_candidate(
                    scores, ious, layout, int(level),
                    max_iou=args.false_iou_thr)
                for level in args.levels}
            if any(value is None for value in false_by_level.values()):
                continue
            for gt_index, gt_box in enumerate(gt_boxes):
                source_order = len(rows)
                row_record = dict(
                    role='source_real_validation_control',
                    split=SOURCE_SPLIT, seq=record['seq'],
                    frame=int(record['frame']), gt_index=int(gt_index),
                    levels={})
                for level in args.levels:
                    level = int(level)
                    feature = features[level]
                    stride = strides[level]
                    height, width = [int(value)
                                     for value in feature.shape[-2:]]
                    pos_row, pos_col = transfer.nearest_grid_location(
                        gt_box, height, width, stride, meta['img_shape'])
                    negative_index = int(false_by_level[level])
                    negative_location = layout[negative_index]
                    neg_row = int(negative_location['row'])
                    neg_col = int(negative_location['col'])
                    positive_vector = patch_vector(
                        feature, pos_row, pos_col)
                    negative_vector = patch_vector(
                        feature, neg_row, neg_col)
                    level_record = dict(
                        positive=dict(
                            row=int(pos_row), col=int(pos_col),
                            source_grid_center=grid_center(
                                pos_row, pos_col, stride)),
                        hard_negative=main_candidate_record(
                            negative_index, scores, ious, layout, stride))
                    row_record['levels'][str(level)] = level_record
                    samples_by_level[level].append(dict(
                        order=source_order, row=row_record,
                        level_record=level_record,
                        positive_vector=positive_vector,
                        negative_vector=negative_vector))
                rows.append(row_record)
        if (record_index + 1) % 50 == 0:
            print('[source-neighborhood] {}/{} images'.format(
                record_index + 1, len(records)))
    if not rows:
        raise RuntimeError('No source controls collected')
    return samples_by_level, rows


def build_level_ensembles(samples_by_level: Dict[int, Sequence[Dict]],
                          folds: int, positive_quantile: float):
    ensembles = {}
    summaries = {}
    metadata = {}
    for level, samples in samples_by_level.items():
        fold_ids = contiguous_fold_ids(len(samples), int(folds))
        fold_models = []
        control_records = []
        for fold_id in range(int(folds)):
            reference = [sample for index, sample in enumerate(samples)
                         if fold_ids[index] != fold_id]
            controls = [sample for index, sample in enumerate(samples)
                        if fold_ids[index] == fold_id]
            if len(reference) < 2 or not controls:
                raise RuntimeError(
                    'Source fold {} is too small at level {}'.format(
                        fold_id, level))
            prototypes = alignment.build_source_prototypes(
                torch.stack([
                    sample['positive_vector'] for sample in reference]),
                torch.stack([
                    sample['negative_vector'] for sample in reference]))
            prototypes = {key: value.detach().clone().cpu()
                          for key, value in prototypes.items()}
            reference_positive_scores = [
                alignment.prototype_scores(
                    sample['positive_vector'], prototypes)
                for sample in reference]
            cosine_threshold = quantile([
                score['cosine_positive']
                for score in reference_positive_scores], positive_quantile)
            whitened_threshold = quantile([
                score['whitened_cosine_positive']
                for score in reference_positive_scores], positive_quantile)
            fold_model = dict(
                fold_id=int(fold_id), prototypes=prototypes,
                cosine_positive_threshold=_number(cosine_threshold),
                whitened_positive_threshold=_number(whitened_threshold),
                reference_count=len(reference), control_count=len(controls))
            fold_models.append(fold_model)
            for sample in controls:
                positive_scores = alignment.prototype_scores(
                    sample['positive_vector'], prototypes)
                negative_scores = alignment.prototype_scores(
                    sample['negative_vector'], prototypes)
                calibration = dict(
                    fold_id=int(fold_id),
                    cosine_positive_threshold=_number(cosine_threshold),
                    whitened_positive_threshold=_number(whitened_threshold),
                    positive_scores=positive_scores,
                    negative_scores=negative_scores,
                    positive_cosine_pass=bool(
                        positive_scores['cosine_positive']
                        >= cosine_threshold),
                    negative_cosine_pass=bool(
                        negative_scores['cosine_positive']
                        < cosine_threshold),
                    positive_whitened_pass=bool(
                        positive_scores['whitened_cosine_positive']
                        >= whitened_threshold),
                    negative_whitened_pass=bool(
                        negative_scores['whitened_cosine_positive']
                        < whitened_threshold))
                sample['level_record']['source_calibration'] = calibration
                control_records.append(calibration)
        ensembles[level] = fold_models
        summaries[level] = source_level_control_summary(control_records)
        metadata[level] = dict(
            folds=int(folds), sample_count=len(samples),
            dimension=int(fold_models[0]['prototypes'][
                'raw_positive'].numel()),
            positive_quantile=_number(positive_quantile),
            fold_thresholds=[dict(
                fold_id=model['fold_id'],
                reference_count=model['reference_count'],
                control_count=model['control_count'],
                cosine_positive_threshold=model[
                    'cosine_positive_threshold'],
                whitened_positive_threshold=model[
                    'whitened_positive_threshold'])
                for model in fold_models])
    return ensembles, summaries, metadata


def neighborhood_rescue(feature: torch.Tensor, base_x: float, base_y: float,
                        false_x: float, false_y: float, img_shape,
                        stride: float, physical_radius_px: float,
                        fold_models: Sequence[Dict],
                        min_fold_votes: int,
                        target_heatmap: Optional[torch.Tensor] = None,
                        min_target_gaussian: float = 0.0) -> Dict:
    height, width = [int(value) for value in feature.shape[-2:]]
    center_row, center_col = physical_to_grid(
        base_x, base_y, stride, height, width, img_shape)
    false_row, false_col = physical_to_grid(
        false_x, false_y, stride, height, width, img_shape)
    radius_cells = radius_in_cells(physical_radius_px, stride)
    false_record = location_scores(
        feature, false_row, false_col, stride, fold_models)
    false_folds = {
        int(fold['fold_id']): fold
        for fold in false_record['source_ensemble']['folds']}
    locations = []
    for row, col in enumerate_neighborhood(
            center_row, center_col, radius_cells,
            height, width, img_shape, stride):
        gaussian_value = None
        if target_heatmap is not None:
            if target_heatmap.shape != (1, height, width):
                raise ValueError('Target heatmap/feature shape mismatch')
            gaussian_value = float(target_heatmap[0, row, col].item())
            if gaussian_value < float(min_target_gaussian):
                continue
        item = location_scores(feature, row, col, stride, fold_models)
        item['offset_cells'] = [
            int(row - center_row), int(col - center_col)]
        item['is_mapped_center'] = bool(
            row == center_row and col == center_col)
        item['target_gaussian'] = (
            None if gaussian_value is None else _number(gaussian_value))
        fold_votes = 0
        threshold_margins = []
        for fold in item['source_ensemble']['folds']:
            false_fold = false_folds[int(fold['fold_id'])]
            fold['false_cosine_positive'] = false_fold['cosine_positive']
            fold['false_whitened_cosine_positive'] = false_fold[
                'whitened_cosine_positive']
            fold['passes'] = bool(
                fold['cosine_positive']
                >= fold['cosine_positive_threshold']
                and fold['whitened_cosine_positive']
                >= fold['whitened_positive_threshold']
                and fold['cosine_positive']
                > false_fold['cosine_positive']
                and fold['whitened_cosine_positive']
                > false_fold['whitened_cosine_positive'])
            fold_votes += int(fold['passes'])
            threshold_margins.append(min(
                fold['cosine_positive']
                - fold['cosine_positive_threshold'],
                fold['whitened_cosine_positive']
                - fold['whitened_positive_threshold']))
        item['fold_votes'] = int(fold_votes)
        item['required_fold_votes'] = int(min_fold_votes)
        item['mean_threshold_margin'] = _number(
            np.mean(threshold_margins))
        item['rescues'] = bool(fold_votes >= int(min_fold_votes))
        locations.append(item)
    if not locations:
        return dict(
            stride=_number(stride), radius_cells=int(radius_cells),
            mapped_center=dict(
                row=int(center_row), col=int(center_col),
                source_grid_center=grid_center(
                    center_row, center_col, stride)),
            matched_false=false_record,
            location_count=0, rescued=False,
            best_location=None, locations=[])
    ranked = sorted(
        locations,
        key=lambda item: (
            bool(item['rescues']),
            int(item['fold_votes']),
            float(item['mean_threshold_margin'])),
        reverse=True)
    rescued = any(item['rescues'] for item in locations)
    return dict(
        stride=_number(stride), radius_cells=int(radius_cells),
        mapped_center=dict(
            row=int(center_row), col=int(center_col),
            source_grid_center=grid_center(
                center_row, center_col, stride)),
        matched_false=false_record,
        location_count=len(locations),
        rescued=bool(rescued),
        best_location=copy.deepcopy(ranked[0]),
        locations=locations)


def collect_target(model, transforms, img_scale, flip, args,
                   strides: Dict[int, float], ensembles: Dict):
    from mmcv.ops import box_iou_rotated

    diag = transfer.entry_probe.get_diag()
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
            _head, boxes, scores, layout, alignment_rows = (
                transfer.forward_main_candidates(
                    model, features, meta['img_shape']))
            gt_boxes = transfer.scaled_gt_tensors(record, meta, boxes.device)
            if gt_boxes.numel() == 0:
                raise RuntimeError('Missing target GT')
            ious = box_iou_rotated(
                boxes.float(), gt_boxes.float()).max(dim=1).values
            usable_index = transfer.select_level_candidate(
                scores, ious, layout, 0, min_iou=args.riou_thr)
            false_index = transfer.select_level_candidate(
                scores, ious, layout, 0, max_iou=args.false_iou_thr)
            if false_index is None:
                raise RuntimeError('No target P3 matched false candidate')
            false_record = main_candidate_record(
                false_index, scores, ious, layout, strides[0])
            usable_record = None
            level_results = {}
            if usable_index is not None:
                usable_record = main_candidate_record(
                    usable_index, scores, ious, layout, strides[0])
                base_x, base_y = usable_record['source_grid_center']
                false_x, false_y = false_record['source_grid_center']
                for level in args.levels:
                    level = int(level)
                    height, width = [
                        int(value) for value in features[level].shape[-2:]]
                    valid = transfer.valid_grid_mask(
                        height, width, meta['img_shape'],
                        strides[level], features[level].device)
                    target_heatmap = transfer.oriented_gaussian_heatmap(
                        gt_boxes, height, width, strides[level],
                        0.25, 1.0, valid)
                    level_results[str(level)] = neighborhood_rescue(
                        features[level], base_x, base_y,
                        false_x, false_y, meta['img_shape'],
                        strides[level], args.physical_radius_px,
                        ensembles[level], args.min_fold_votes,
                        target_heatmap,
                        args.min_target_gaussian)
            rows.append(dict(
                role='target_dev_diagnosis_only', split=TARGET_SPLIT,
                seq=TARGET_SEQ, frame=int(frame_id), image_stats=image_stats,
                eligible=usable_record is not None,
                geometry_miss=usable_record is None,
                dense_best_riou=_number(ious.max().item()),
                usable_p3_candidate=usable_record,
                matched_p3_false=false_record,
                levels=level_results,
                decode_alignment=alignment_rows))
        print('[target-neighborhood] frame {} eligible={}'.format(
            frame_id, usable_record is not None))
    return rows


def summarize_target(rows: Sequence[Dict], levels: Sequence[int]) -> Dict:
    eligible = [row for row in rows if row['eligible']]
    level_rows = {}
    for level in levels:
        level = int(level)
        rescued = [row['levels'][str(level)]['rescued'] for row in eligible]
        center_positive = [
            any(item['is_mapped_center']
                and item['fold_votes'] >= item['required_fold_votes']
                for item in row['levels'][str(level)]['locations'])
            for row in eligible]
        level_rows[str(level)] = dict(
            eligible_count=len(eligible),
            rescue_count=int(sum(rescued)),
            rescue_fraction=_number(alignment.accuracy(rescued)),
            mapped_center_cosine_positive_count=int(sum(center_positive)))
    return dict(eligible_count=len(eligible), levels=level_rows)


def source_level_valid(summary: Dict, minimum: float) -> bool:
    return bool(
        summary['positive_cosine_accuracy'] >= minimum
        and summary['negative_cosine_accuracy'] >= minimum
        and summary['positive_whitened_accuracy'] >= minimum
        and summary['negative_whitened_accuracy'] >= minimum)


def make_gate(source_summaries: Dict[int, Dict], target_summary: Dict,
              geometry_misses: Sequence[int], args) -> Dict:
    common = dict(
        eligible_count=(target_summary['eligible_count'] == EXPECTED_ELIGIBLE),
        expected_geometry_misses=(list(geometry_misses)
                                  == EXPECTED_GEOMETRY_MISSES))
    level_valid = {
        int(level): source_level_valid(
            source_summaries[int(level)], float(args.source_min_accuracy))
        for level in args.levels}
    rescue_counts = {
        int(level): int(target_summary['levels'][str(level)]['rescue_count'])
        for level in args.levels}
    common_valid = all(common.values())
    p3_valid = level_valid.get(0, False)
    p4_valid = level_valid.get(1, False)
    p3_pass = bool(
        common_valid and p3_valid
        and rescue_counts.get(0, 0) >= int(args.target_min_rescues))
    p4_pass = bool(
        common_valid and p4_valid
        and rescue_counts.get(1, 0) >= int(args.target_min_rescues))
    if not common_valid:
        decision = 'AUDIT_INVALID'
        interpretation = (
            'Target geometry invariants failed. Do not select an architecture '
            'from this run.')
    elif not p3_valid:
        decision = 'P3_INCONCLUSIVE_SOURCE_CONTROL'
        interpretation = (
            'The P3 source-only calibration failed, so the smallest sampling '
            'route is unresolved. Do not authorize P4 or close the FPN route.')
    elif p3_pass:
        decision = 'AUTHORIZE_P3_LOCAL_SAMPLING'
        interpretation = (
            'A fixed local P3 neighborhood recovers source-compatible '
            'semantics. Authorize one bounded task-specific/deformable '
            'sampling experiment; this audit is not a deployment gain.')
    elif p4_valid and p4_pass:
        decision = 'AUTHORIZE_P4_CROSS_LEVEL_OBJECTNESS'
        interpretation = (
            'P3 local sampling is insufficient, but the corresponding P4 '
            'neighborhood recovers source-compatible semantics. Authorize '
            'one cross-level objectness/classification experiment while '
            'keeping P3 regression geometry unchanged.')
    elif p4_valid:
        decision = 'NO_FPN_RESCUE'
        interpretation = (
            'Neither bounded P3 sampling nor P4 cross-level reading recovers '
            'source-compatible semantics. Close ordinary FPN-only objectness '
            'and sampling; next use a read-only VFM/domain-representation '
            'probe before training.')
    else:
        decision = 'P3_NO_RESCUE_P4_INCONCLUSIVE'
        interpretation = (
            'The valid P3 audit found no local rescue, but P4 source '
            'calibration failed. Close P3 local sampling and repair only the '
            'P4 source calibration before choosing the next architecture.')
    return dict(
        decision=decision,
        common_checks=common,
        source_level_valid={str(key): value
                            for key, value in level_valid.items()},
        rescue_counts={str(key): value
                       for key, value in rescue_counts.items()},
        target_min_rescues=int(args.target_min_rescues),
        p3_pass=p3_pass, p4_pass=p4_pass,
        interpretation=interpretation)


def main():
    args = parse_args()
    canonical = validate_args(args)
    set_seed(args.seed)

    model, cfg = transfer.entry_probe.load_model(
        args.config, args.detector_checkpoint, args.gpu)
    transfer.freeze_detector(model)
    versions_before = alignment.module_parameter_versions(model)
    candidate_head = transfer.entry_probe.get_candidate_head(model, 'main')
    if max(args.levels) >= len(candidate_head.anchor_generator.strides):
        raise ValueError('Requested level exceeds detector FPN levels')
    strides = {
        int(level): stride_value(
            candidate_head.anchor_generator.strides[int(level)])
        for level in args.levels}
    radius_cells = {
        level: radius_in_cells(args.physical_radius_px, stride)
        for level, stride in strides.items()}
    if canonical and radius_cells != {0: 2, 1: 1}:
        raise RuntimeError('Canonical P3/P4 search grids changed')

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
    samples_by_level, source_rows = collect_source(
        model, source_records, transforms, img_scale, flip,
        args, strides)
    ensembles, source_summaries, prototype_metadata = (
        build_level_ensembles(
            samples_by_level, args.source_folds,
            args.positive_quantile))

    # First target access occurs only after all source level prototypes freeze.
    target_rows = collect_target(
        model, transforms, img_scale, flip, args, strides, ensembles)
    target_summary = summarize_target(target_rows, args.levels)
    geometry_misses = [
        int(row['frame']) for row in target_rows if row['geometry_miss']]
    gate = make_gate(
        source_summaries, target_summary, geometry_misses, args)

    parameters_unchanged = (
        versions_before == alignment.module_parameter_versions(model))
    if not parameters_unchanged:
        raise RuntimeError('Read-only detector parameter invariant failed')
    payload = dict(
        audit=AUDIT_NAME,
        protocol_version=PROTOCOL_VERSION,
        canonical_protocol=bool(canonical),
        data_role='source_reference_target_dev_diagnosis_only',
        config=os.path.abspath(args.config),
        detector_checkpoint=os.path.abspath(args.detector_checkpoint),
        protocol=dict(
            levels=[int(value) for value in args.levels],
            strides={str(key): _number(value)
                     for key, value in strides.items()},
            physical_radius_px=_number(args.physical_radius_px),
            physical_window_metric='Chebyshev_per_axis',
            min_target_gaussian=_number(args.min_target_gaussian),
            radius_cells={str(key): int(value)
                          for key, value in radius_cells.items()},
            neighborhood_shapes={
                str(key): [2 * int(value) + 1, 2 * int(value) + 1]
                for key, value in radius_cells.items()},
            source_seq=args.source_seq,
            source_calibration='contiguous_fold_source_positive_thresholds',
            source_folds=int(args.source_folds),
            min_fold_votes=int(args.min_fold_votes),
            positive_quantile=_number(args.positive_quantile),
            source_min_accuracy=_number(args.source_min_accuracy),
            target_slice='real_seq02[137..169]',
            target_min_rescues=int(args.target_min_rescues)),
        isolation=dict(
            creates_optimizer=False,
            performs_optimizer_step=False,
            writes_checkpoint=False,
            detector_frozen=True,
            detector_parameters_unchanged=parameters_unchanged,
            source_prototypes_frozen_before_target=True,
            target_used_for_source_prototypes=False,
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_labels_used_for_diagnosis_only=True,
            raw_fpn_patches_serialized=False),
        prototype_metadata={str(key): value
                            for key, value in prototype_metadata.items()},
        source=dict(
            summaries={str(key): value
                       for key, value in source_summaries.items()},
            rows=source_rows),
        target_dev=dict(
            geometry_misses=geometry_misses,
            summary=target_summary,
            rows=target_rows),
        gate=gate)
    if not canonical:
        payload['gate']['decision'] = 'NONCANONICAL_NO_AUTHORIZATION'
        payload['gate']['interpretation'] = (
            'Smoke test only; run the canonical full audit.')
    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, 'w') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False,
                  allow_nan=False)
    counts = ' '.join(
        'P{}={}/{}'.format(
            int(level) + 3,
            target_summary['levels'][str(level)]['rescue_count'],
            target_summary['eligible_count'])
        for level in args.levels)
    print('[rescue] {} {}'.format(
        payload['gate']['decision'], counts))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
