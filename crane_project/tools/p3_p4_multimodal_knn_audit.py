#!/usr/bin/env python3
"""Read-only multimodal P3/P4 source-representation audit.

This audit follows the failed single-prototype neighborhood audit. It keeps a
source-only bank of real positive/background FPN patches and uses fixed k-NN
similarity instead of collapsing each class to one mean direction. The goal is
to distinguish a single-prototype artifact from a genuine lack of transferable
semantics in the ordinary P3/P4 neighborhoods.

No optimizer is created, no parameter is updated, and no checkpoint or raw FPN
patch is written. All source banks are frozen before target-dev is read.
"""

import argparse
import copy
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import frozen_p3_feature_alignment_audit as alignment
from crane_project.tools import frozen_p3_objectness_transfer_probe as transfer
from crane_project.tools import p3_p4_neighborhood_rescue_audit as neighborhood


AUDIT_NAME = 'P3/P4 Multimodal Source k-NN Audit'
PROTOCOL_VERSION = 2
CANONICAL_NEIGHBORS = 5
CANONICAL_POSITIVE_QUANTILE = 0.1
CANONICAL_NEGATIVE_QUANTILE = 0.9


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument('--config', required=True)
    parser.add_argument('--detector-checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-seq', default=neighborhood.SOURCE_SEQ)
    parser.add_argument('--levels', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--physical-radius-px', type=float, default=16.0)
    parser.add_argument('--min-target-gaussian', type=float, default=0.1)
    parser.add_argument('--source-folds', type=int, default=5)
    parser.add_argument('--neighbors', type=int, default=CANONICAL_NEIGHBORS)
    parser.add_argument('--positive-quantile', type=float,
                        default=CANONICAL_POSITIVE_QUANTILE)
    parser.add_argument('--negative-quantile', type=float,
                        default=CANONICAL_NEGATIVE_QUANTILE)
    parser.add_argument('--min-fold-votes', type=int, default=4)
    parser.add_argument('--max-source-samples', type=int, default=0)
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--false-iou-thr', type=float, default=0.1)
    parser.add_argument('--source-min-accuracy', type=float, default=0.8)
    parser.add_argument('--target-min-rescues', type=int, default=26)
    parser.add_argument('--target-start', type=int,
                        default=neighborhood.TARGET_START)
    parser.add_argument('--target-end', type=int,
                        default=neighborhood.TARGET_END)
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
    if 0 not in levels:
        raise ValueError('--levels must include P3 level 0')
    if args.physical_radius_px <= 0.0:
        raise ValueError('--physical-radius-px must be positive')
    if not 0.0 <= args.min_target_gaussian <= 1.0:
        raise ValueError('--min-target-gaussian must be in [0, 1]')
    if args.source_folds < 2:
        raise ValueError('--source-folds must be at least 2')
    if args.neighbors <= 0:
        raise ValueError('--neighbors must be positive')
    if not 0.0 < args.positive_quantile < 0.5:
        raise ValueError('--positive-quantile must be in (0, 0.5)')
    if not 0.5 < args.negative_quantile < 1.0:
        raise ValueError('--negative-quantile must be in (0.5, 1)')
    if not 1 <= args.min_fold_votes <= args.source_folds:
        raise ValueError('--min-fold-votes must be in [1, source-folds]')
    if args.max_source_samples < 0:
        raise ValueError('--max-source-samples must be non-negative')
    if not 0.0 <= args.false_iou_thr < args.riou_thr <= 1.0:
        raise ValueError('Require 0 <= false-iou-thr < riou-thr <= 1')
    if not 0.0 < args.source_min_accuracy <= 1.0:
        raise ValueError('--source-min-accuracy must be in (0, 1]')
    if args.target_min_rescues <= 0:
        raise ValueError('--target-min-rescues must be positive')

    checks = dict(
        config=(os.path.basename(args.config)
                == neighborhood.CANONICAL_CONFIG),
        checkpoint=(os.path.basename(args.detector_checkpoint)
                    == neighborhood.CANONICAL_CHECKPOINT),
        source_seq=args.source_seq == neighborhood.SOURCE_SEQ,
        levels=levels == neighborhood.CANONICAL_LEVELS,
        radius=math.isclose(float(args.physical_radius_px), 16.0),
        target_gaussian=math.isclose(float(args.min_target_gaussian), 0.1),
        source_folds=int(args.source_folds) == 5,
        neighbors=int(args.neighbors) == CANONICAL_NEIGHBORS,
        positive_quantile=math.isclose(
            float(args.positive_quantile), CANONICAL_POSITIVE_QUANTILE),
        negative_quantile=math.isclose(
            float(args.negative_quantile), CANONICAL_NEGATIVE_QUANTILE),
        fold_votes=int(args.min_fold_votes) == 4,
        full_source=args.max_source_samples == 0,
        thresholds=(math.isclose(args.riou_thr, 0.5)
                    and math.isclose(args.false_iou_thr, 0.1)),
        source_gate=math.isclose(args.source_min_accuracy, 0.8),
        target_gate=int(args.target_min_rescues) == 26,
        target_slice=(int(args.target_start) == neighborhood.TARGET_START
                      and int(args.target_end) == neighborhood.TARGET_END))
    canonical = all(checks.values())
    if not canonical and not args.allow_noncanonical:
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(
            'Canonical multimodal-audit mismatch: {}. '
            'Use --allow-noncanonical only for smoke tests.'.format(
                ', '.join(failed)))
    return canonical


def _number(value) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def _normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
    return F.normalize(vectors.float(), dim=1)


def build_knn_bank(positive: torch.Tensor, negative: torch.Tensor,
                   neighbors: int) -> Dict[str, torch.Tensor]:
    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError('k-NN bank inputs must be [N,D]')
    if positive.shape[1] != negative.shape[1]:
        raise ValueError('Positive/negative bank dimensions differ')
    if min(positive.shape[0], negative.shape[0]) < int(neighbors):
        raise ValueError('Each source bank needs at least k samples')
    positive = positive.detach().float().cpu()
    negative = negative.detach().float().cpu()
    combined = torch.cat([positive, negative], dim=0)
    center = combined.mean(dim=0)
    scale = combined.std(dim=0, unbiased=False).clamp_min(1e-6)
    return dict(
        neighbors=int(neighbors),
        raw_positive=_normalize_rows(positive),
        raw_negative=_normalize_rows(negative),
        whiten_center=center,
        whiten_scale=scale,
        whiten_positive=_normalize_rows((positive - center) / scale),
        whiten_negative=_normalize_rows((negative - center) / scale))


def _topk_mean(query: torch.Tensor, bank: torch.Tensor,
               neighbors: int) -> Dict[str, float]:
    similarities = torch.mv(bank, query)
    values = torch.topk(
        similarities, k=int(neighbors), largest=True, sorted=True).values
    return dict(mean=_number(values.mean().item()),
                maximum=_number(values[0].item()))


def knn_scores(vector: torch.Tensor, bank: Dict[str, torch.Tensor]) -> Dict:
    vector = vector.detach().float().cpu().reshape(-1)
    raw_query = F.normalize(vector, dim=0)
    whitened = ((vector - bank['whiten_center'])
                / bank['whiten_scale'])
    white_query = F.normalize(whitened, dim=0)
    neighbors = int(bank['neighbors'])
    raw_positive = _topk_mean(
        raw_query, bank['raw_positive'], neighbors)
    raw_negative = _topk_mean(
        raw_query, bank['raw_negative'], neighbors)
    white_positive = _topk_mean(
        white_query, bank['whiten_positive'], neighbors)
    white_negative = _topk_mean(
        white_query, bank['whiten_negative'], neighbors)
    return dict(
        vector_norm=_number(vector.norm().item()),
        cosine_positive=raw_positive['mean'],
        cosine_negative=raw_negative['mean'],
        cosine_preference_positive=_number(
            raw_positive['mean'] - raw_negative['mean']),
        cosine_nearest_positive=raw_positive['maximum'],
        cosine_nearest_negative=raw_negative['maximum'],
        whitened_cosine_positive=white_positive['mean'],
        whitened_cosine_negative=white_negative['mean'],
        whitened_preference_positive=_number(
            white_positive['mean'] - white_negative['mean']),
        whitened_nearest_positive=white_positive['maximum'],
        whitened_nearest_negative=white_negative['maximum'])


def ensemble_scores(vector: torch.Tensor,
                    fold_models: Sequence[Dict]) -> Dict:
    folds = []
    for model in fold_models:
        scores = knn_scores(vector, model['bank'])
        scores['fold_id'] = int(model['fold_id'])
        scores['cosine_preference_threshold'] = model[
            'cosine_preference_threshold']
        scores['whitened_preference_threshold'] = model[
            'whitened_preference_threshold']
        folds.append(scores)
    return dict(
        fold_count=len(folds), folds=folds,
        mean_cosine_preference_positive=_number(np.mean([
            fold['cosine_preference_positive'] for fold in folds])),
        mean_whitened_preference_positive=_number(np.mean([
            fold['whitened_preference_positive'] for fold in folds])))


def location_scores(feature: torch.Tensor, row: int, col: int,
                    stride: float, fold_models: Sequence[Dict]) -> Dict:
    vector = neighborhood.patch_vector(feature, row, col)
    return dict(
        row=int(row), col=int(col),
        source_grid_center=neighborhood.grid_center(row, col, stride),
        source_ensemble=ensemble_scores(vector, fold_models))


def calibrated_threshold(positive_values: Sequence[float],
                         negative_values: Sequence[float],
                         positive_quantile: float,
                         negative_quantile: float) -> Dict[str, float]:
    positive_floor = neighborhood.quantile(
        positive_values, positive_quantile)
    negative_ceiling = neighborhood.quantile(
        negative_values, negative_quantile)
    return dict(
        positive_floor=_number(positive_floor),
        negative_ceiling=_number(negative_ceiling),
        threshold=_number(max(positive_floor, negative_ceiling)))


def zero_margin_control_summary(records: Sequence[Dict]) -> Dict:
    return dict(
        count=len(records),
        positive_cosine_accuracy=_number(alignment.accuracy([
            record['zero_margin_positive_cosine_pass']
            for record in records])),
        negative_cosine_accuracy=_number(alignment.accuracy([
            record['zero_margin_negative_cosine_pass']
            for record in records])),
        positive_whitened_accuracy=_number(alignment.accuracy([
            record['zero_margin_positive_whitened_pass']
            for record in records])),
        negative_whitened_accuracy=_number(alignment.accuracy([
            record['zero_margin_negative_whitened_pass']
            for record in records])))


def build_level_ensembles(samples_by_level: Dict[int, Sequence[Dict]],
                          folds: int, neighbors: int,
                          positive_quantile: float,
                          negative_quantile: float):
    ensembles = {}
    summaries = {}
    calibrated_summaries = {}
    metadata = {}
    for level, samples in samples_by_level.items():
        fold_ids = neighborhood.contiguous_fold_ids(len(samples), int(folds))
        fold_models = []
        control_records = []
        for fold_id in range(int(folds)):
            reference = [sample for index, sample in enumerate(samples)
                         if fold_ids[index] != fold_id]
            controls = [sample for index, sample in enumerate(samples)
                        if fold_ids[index] == fold_id]
            if len(reference) < int(neighbors) or not controls:
                raise RuntimeError(
                    'Source fold {} is too small at level {}'.format(
                        fold_id, level))
            bank = build_knn_bank(
                torch.stack([
                    sample['positive_vector'] for sample in reference]),
                torch.stack([
                    sample['negative_vector'] for sample in reference]),
                neighbors)
            pending_controls = []
            for sample in controls:
                positive_scores = knn_scores(
                    sample['positive_vector'], bank)
                negative_scores = knn_scores(
                    sample['negative_vector'], bank)
                pending_controls.append(
                    (sample, positive_scores, negative_scores))
            raw_calibration = calibrated_threshold(
                [item[1]['cosine_preference_positive']
                 for item in pending_controls],
                [item[2]['cosine_preference_positive']
                 for item in pending_controls],
                positive_quantile, negative_quantile)
            white_calibration = calibrated_threshold(
                [item[1]['whitened_preference_positive']
                 for item in pending_controls],
                [item[2]['whitened_preference_positive']
                 for item in pending_controls],
                positive_quantile, negative_quantile)
            fold_model = dict(
                fold_id=int(fold_id), bank=bank,
                reference_count=len(reference), control_count=len(controls),
                cosine_preference_threshold=raw_calibration['threshold'],
                whitened_preference_threshold=white_calibration['threshold'],
                raw_calibration=raw_calibration,
                whitened_calibration=white_calibration)
            fold_models.append(fold_model)
            for sample, positive_scores, negative_scores in pending_controls:
                calibration = dict(
                    fold_id=int(fold_id),
                    cosine_preference_threshold=raw_calibration['threshold'],
                    whitened_preference_threshold=(
                        white_calibration['threshold']),
                    positive_scores=positive_scores,
                    negative_scores=negative_scores,
                    positive_cosine_pass=bool(
                        positive_scores['cosine_preference_positive']
                        >= raw_calibration['threshold']),
                    negative_cosine_pass=bool(
                        negative_scores['cosine_preference_positive']
                        < raw_calibration['threshold']),
                    positive_whitened_pass=bool(
                        positive_scores[
                            'whitened_preference_positive']
                        >= white_calibration['threshold']),
                    negative_whitened_pass=bool(
                        negative_scores[
                            'whitened_preference_positive']
                        < white_calibration['threshold']),
                    zero_margin_positive_cosine_pass=bool(
                        positive_scores['cosine_preference_positive'] > 0.0),
                    zero_margin_negative_cosine_pass=bool(
                        negative_scores['cosine_preference_positive'] < 0.0),
                    zero_margin_positive_whitened_pass=bool(
                        positive_scores[
                            'whitened_preference_positive'] > 0.0),
                    zero_margin_negative_whitened_pass=bool(
                        negative_scores[
                            'whitened_preference_positive'] < 0.0))
                sample['level_record']['source_knn_control'] = calibration
                control_records.append(calibration)
        ensembles[level] = fold_models
        summaries[level] = zero_margin_control_summary(control_records)
        calibrated_summaries[level] = (
            neighborhood.source_level_control_summary(control_records))
        metadata[level] = dict(
            folds=int(folds), sample_count=len(samples),
            dimension=int(fold_models[0]['bank'][
                'raw_positive'].shape[1]),
            neighbors=int(neighbors),
            positive_quantile=_number(positive_quantile),
            negative_quantile=_number(negative_quantile),
            fold_sizes=[dict(
                fold_id=model['fold_id'],
                reference_count=model['reference_count'],
                control_count=model['control_count'],
                cosine_preference_threshold=model[
                    'cosine_preference_threshold'],
                whitened_preference_threshold=model[
                    'whitened_preference_threshold'],
                raw_calibration=model['raw_calibration'],
                whitened_calibration=model['whitened_calibration'])
                for model in fold_models])
    return ensembles, summaries, calibrated_summaries, metadata


def multimodal_neighborhood_rescue(
        feature: torch.Tensor, base_x: float, base_y: float,
        false_x: float, false_y: float, img_shape,
        stride: float, physical_radius_px: float,
        fold_models: Sequence[Dict], min_fold_votes: int,
        target_heatmap: Optional[torch.Tensor] = None,
        min_target_gaussian: float = 0.0) -> Dict:
    height, width = [int(value) for value in feature.shape[-2:]]
    center_row, center_col = neighborhood.physical_to_grid(
        base_x, base_y, stride, height, width, img_shape)
    false_row, false_col = neighborhood.physical_to_grid(
        false_x, false_y, stride, height, width, img_shape)
    radius_cells = neighborhood.radius_in_cells(
        physical_radius_px, stride)
    false_record = location_scores(
        feature, false_row, false_col, stride, fold_models)
    false_folds = {
        int(fold['fold_id']): fold
        for fold in false_record['source_ensemble']['folds']}
    locations = []
    for row, col in neighborhood.enumerate_neighborhood(
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
        decision_margins = []
        raw_margins = []
        white_margins = []
        for fold in item['source_ensemble']['folds']:
            false_fold = false_folds[int(fold['fold_id'])]
            fold['false_cosine_preference_positive'] = false_fold[
                'cosine_preference_positive']
            fold['false_whitened_preference_positive'] = false_fold[
                'whitened_preference_positive']
            raw_margin = (
                fold['cosine_preference_positive']
                - fold['cosine_preference_threshold'])
            white_margin = (
                fold['whitened_preference_positive']
                - fold['whitened_preference_threshold'])
            fold['calibrated_cosine_margin'] = _number(raw_margin)
            fold['calibrated_whitened_margin'] = _number(white_margin)
            fold['passes'] = bool(
                raw_margin >= 0.0
                and white_margin >= 0.0
                and fold['cosine_preference_positive']
                > false_fold['cosine_preference_positive']
                and fold['whitened_preference_positive']
                > false_fold['whitened_preference_positive'])
            fold_votes += int(fold['passes'])
            raw_margins.append(raw_margin)
            white_margins.append(white_margin)
            decision_margins.append(min(raw_margin, white_margin))
        item['fold_votes'] = int(fold_votes)
        item['required_fold_votes'] = int(min_fold_votes)
        item['mean_decision_margin'] = _number(
            np.mean(decision_margins))
        item['mean_calibrated_cosine_margin'] = _number(
            np.mean(raw_margins))
        item['mean_calibrated_whitened_margin'] = _number(
            np.mean(white_margins))
        item['rescues'] = bool(fold_votes >= int(min_fold_votes))
        locations.append(item)
    if not locations:
        return dict(
            stride=_number(stride), radius_cells=int(radius_cells),
            mapped_center=dict(
                row=int(center_row), col=int(center_col),
                source_grid_center=neighborhood.grid_center(
                    center_row, center_col, stride)),
            matched_false=false_record, location_count=0,
            rescued=False, best_location=None, locations=[])
    ranked = sorted(
        locations,
        key=lambda item: (
            bool(item['rescues']), int(item['fold_votes']),
            float(item['mean_decision_margin'])),
        reverse=True)
    return dict(
        stride=_number(stride), radius_cells=int(radius_cells),
        mapped_center=dict(
            row=int(center_row), col=int(center_col),
            source_grid_center=neighborhood.grid_center(
                center_row, center_col, stride)),
        matched_false=false_record,
        location_count=len(locations),
        rescued=bool(any(item['rescues'] for item in locations)),
        best_location=copy.deepcopy(ranked[0]), locations=locations)


def collect_target(model, transforms, img_scale, flip, args,
                   strides: Dict[int, float], ensembles: Dict):
    from mmcv.ops import box_iou_rotated

    diag = transfer.entry_probe.get_diag()
    rows = []
    for frame_id in range(args.target_start, args.target_end + 1):
        img_path, ann_path = diag.find_files(
            args.data_root, neighborhood.TARGET_SPLIT,
            neighborhood.TARGET_SEQ, frame_id)
        if img_path is None or ann_path is None:
            raise RuntimeError('Missing target-dev frame {}'.format(frame_id))
        record = dict(
            split=neighborhood.TARGET_SPLIT,
            seq=neighborhood.TARGET_SEQ, frame=frame_id,
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
            gt_boxes = transfer.scaled_gt_tensors(
                record, meta, boxes.device)
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
            false_record = neighborhood.main_candidate_record(
                false_index, scores, ious, layout, strides[0])
            usable_record = None
            level_results = {}
            if usable_index is not None:
                usable_record = neighborhood.main_candidate_record(
                    usable_index, scores, ious, layout, strides[0])
                base_x, base_y = usable_record['source_grid_center']
                false_x, false_y = false_record['source_grid_center']
                for level in args.levels:
                    level = int(level)
                    height, width = [
                        int(value) for value
                        in features[level].shape[-2:]]
                    valid = transfer.valid_grid_mask(
                        height, width, meta['img_shape'],
                        strides[level], features[level].device)
                    target_heatmap = transfer.oriented_gaussian_heatmap(
                        gt_boxes, height, width, strides[level],
                        0.25, 1.0, valid)
                    level_results[str(level)] = (
                        multimodal_neighborhood_rescue(
                            features[level], base_x, base_y,
                            false_x, false_y, meta['img_shape'],
                            strides[level], args.physical_radius_px,
                            ensembles[level], args.min_fold_votes,
                            target_heatmap, args.min_target_gaussian))
            rows.append(dict(
                role='target_dev_diagnosis_only',
                split=neighborhood.TARGET_SPLIT,
                seq=neighborhood.TARGET_SEQ, frame=int(frame_id),
                image_stats=image_stats,
                eligible=usable_record is not None,
                geometry_miss=usable_record is None,
                dense_best_riou=_number(ious.max().item()),
                usable_p3_candidate=usable_record,
                matched_p3_false=false_record,
                levels=level_results,
                decode_alignment=alignment_rows))
        print('[target-multimodal] frame {} eligible={}'.format(
            frame_id, usable_record is not None))
    return rows


def _margin_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return dict(minimum=None, median=None, maximum=None)
    return dict(
        minimum=_number(min(values)),
        median=_number(np.median(values)),
        maximum=_number(max(values)))


def summarize_target(rows: Sequence[Dict], levels: Sequence[int]) -> Dict:
    eligible = [row for row in rows if row['eligible']]
    level_rows = {}
    for level in levels:
        level = int(level)
        results = [row['levels'][str(level)] for row in eligible]
        rescued = [result['rescued'] for result in results]
        best_locations = [result['best_location'] for result in results]
        fold_count = (int(
            best_locations[0]['source_ensemble']['fold_count'])
            if best_locations else 0)
        vote_histogram = {
            str(votes): int(sum(
                location['fold_votes'] == votes
                for location in best_locations))
            for votes in range(fold_count + 1)}
        center_rescued = [
            any(item['is_mapped_center'] and item['rescues']
                for item in result['locations'])
            for result in results]
        level_rows[str(level)] = dict(
            eligible_count=len(eligible),
            rescue_count=int(sum(rescued)),
            rescue_fraction=_number(alignment.accuracy(rescued)),
            best_vote_histogram=vote_histogram,
            mapped_center_calibrated_rescue_count=int(sum(center_rescued)),
            best_decision_margin=_margin_summary([
                location['mean_decision_margin']
                for location in best_locations]),
            best_cosine_margin=_margin_summary([
                location['mean_calibrated_cosine_margin']
                for location in best_locations]),
            best_whitened_margin=_margin_summary([
                location['mean_calibrated_whitened_margin']
                for location in best_locations]))
    return dict(eligible_count=len(eligible), levels=level_rows)


def make_gate(source_summaries: Dict[int, Dict], target_summary: Dict,
              geometry_misses: Sequence[int], args) -> Dict:
    common = dict(
        eligible_count=(target_summary['eligible_count']
                        == neighborhood.EXPECTED_ELIGIBLE),
        expected_geometry_misses=(list(geometry_misses)
                                  == neighborhood.EXPECTED_GEOMETRY_MISSES))
    level_valid = {
        int(level): neighborhood.source_level_valid(
            source_summaries[int(level)], float(args.source_min_accuracy))
        for level in args.levels}
    rescue_counts = {
        int(level): int(
            target_summary['levels'][str(level)]['rescue_count'])
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
            'Target geometry invariants failed. Do not select an '
            'architecture from this run.')
    elif not p3_valid:
        decision = 'P3_MULTIMODAL_INCONCLUSIVE'
        interpretation = (
            'The source-calibrated P3 k-NN control failed. The multimodal '
            'representation question remains unresolved.')
    elif p3_pass:
        decision = 'MULTIMODAL_P3_RESCUE'
        interpretation = (
            'Real source modes recover target semantics at source-positive '
            'confidence in the P3 '
            'neighborhood. The previous single-prototype failure was too '
            'restrictive; authorize one bounded multimodal P3 readout.')
    elif p4_valid and p4_pass:
        decision = 'MULTIMODAL_P4_RESCUE'
        interpretation = (
            'P3 source modes do not rescue target semantics, but P4 source '
            'modes reach source-positive confidence. Authorize one bounded '
            'P4 representation readout while keeping P3 regression unchanged.')
    elif p4_valid:
        decision = 'CLOSE_ORDINARY_FPN_REPRESENTATION'
        interpretation = (
            'Even real multimodal source banks cannot recover target '
            'semantics at source-positive confidence in P3 or P4. Close '
            'ordinary FPN-only sampling and classification; move to a '
            'read-only external/domain representation probe before training.')
    else:
        decision = 'P3_CLOSED_P4_MULTIMODAL_INCONCLUSIVE'
        interpretation = (
            'The valid multimodal P3 audit found no rescue, but P4 source '
            'control failed. Close P3 and repair only the P4 diagnosis.')
    return dict(
        decision=decision, common_checks=common,
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
    neighborhood.set_seed(args.seed)

    model, cfg = transfer.entry_probe.load_model(
        args.config, args.detector_checkpoint, args.gpu)
    transfer.freeze_detector(model)
    versions_before = alignment.module_parameter_versions(model)
    candidate_head = transfer.entry_probe.get_candidate_head(model, 'main')
    if max(args.levels) >= len(candidate_head.anchor_generator.strides):
        raise ValueError('Requested level exceeds detector FPN levels')
    strides = {
        int(level): neighborhood.stride_value(
            candidate_head.anchor_generator.strides[int(level)])
        for level in args.levels}
    radius_cells = {
        level: neighborhood.radius_in_cells(
            args.physical_radius_px, stride)
        for level, stride in strides.items()}
    if canonical and radius_cells != {0: 2, 1: 1}:
        raise RuntimeError('Canonical P3/P4 search grids changed')

    diag = transfer.entry_probe.get_diag()
    transforms, img_scale, flip = diag.build_test_transforms(cfg)
    source_records = [
        record for record in transfer.discover_labeled_records(
            args.data_root, neighborhood.SOURCE_SPLIT, 0)
        if record['seq'] == args.source_seq]
    if args.max_source_samples > 0:
        source_records = source_records[:args.max_source_samples]
    if not source_records:
        raise RuntimeError('No source-real validation records found')
    samples_by_level, source_rows = neighborhood.collect_source(
        model, source_records, transforms, img_scale, flip, args, strides)
    (ensembles, source_summaries,
     calibrated_source_summaries, bank_metadata) = build_level_ensembles(
         samples_by_level, args.source_folds, args.neighbors,
         args.positive_quantile, args.negative_quantile)

    # First target access occurs only after all source k-NN banks freeze.
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
            source_representation='fixed_topk_mean_cosine',
            source_folds=int(args.source_folds),
            neighbors=int(args.neighbors),
            positive_quantile=_number(args.positive_quantile),
            negative_quantile=_number(args.negative_quantile),
            source_gate_definition='cross_fold_zero_margin_accuracy',
            target_threshold_calibration=(
                'source_fold_controls_positive_p10_negative_p90'),
            target_vote_definition=(
                'source_calibrated_raw_and_whitened_preferences'),
            min_fold_votes=int(args.min_fold_votes),
            source_min_accuracy=_number(args.source_min_accuracy),
            target_slice='real_seq02[137..169]',
            target_min_rescues=int(args.target_min_rescues)),
        isolation=dict(
            creates_optimizer=False,
            performs_optimizer_step=False,
            writes_checkpoint=False,
            detector_frozen=True,
            detector_parameters_unchanged=parameters_unchanged,
            source_banks_frozen_before_target=True,
            target_used_for_source_banks=False,
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_labels_used_for_diagnosis_only=True,
            raw_fpn_patches_serialized=False),
        bank_metadata={str(key): value
                       for key, value in bank_metadata.items()},
        source=dict(
            summaries={str(key): value
                       for key, value in source_summaries.items()},
            calibrated_summaries={
                str(key): value
                for key, value in calibrated_source_summaries.items()},
            rows=source_rows),
        target_dev=dict(
            geometry_misses=geometry_misses,
            summary=target_summary, rows=target_rows),
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
    print('[multimodal] {} {}'.format(
        payload['gate']['decision'], counts))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
