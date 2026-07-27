#!/usr/bin/env python3
"""Read-only token-scale and RPN coverage audit for frozen DINOv2.

The audit separates three questions without optimizer steps or checkpoint
writes:

1. How large is each rotated GT in the actual resized DINO patch grid?
2. Can the configured RPN anchors cover its horizontal envelope in theory?
3. Does the source-selected RPN produce a rotated proposal with usable IoU?

Source-train records define the token-size bins.  Source validation checks the
trained RPN, while target slices are diagnosis-only and never influence a
threshold, checkpoint, or model parameter.
"""

import argparse
import math
import os
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch

from crane_project.tools import dino_teacher_common as common
from crane_project.tools import (
    dino_teacher_far_distance_candidate_audit as far_audit)
from crane_project.tools import dino_teacher_rotated_labeller as labeller


AUDIT_NAME = 'DINO Token-Scale and RPN Coverage Audit V1'
PROTOCOL_VERSION = 1
DEFAULT_TARGET_SLICES = (
    'seq02_far:test:real_seq02:2:41',
    'seq02_dark:test:real_seq02:137:169',
    'seq03_small:test:real_seq03:129:192',
)
DEFAULT_RECALL_KS = (20, 100, 1000, 2000)
DEFAULT_ANCHOR_IOU_THRESHOLDS = (0.3, 0.5, 0.7)


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument(
        '--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument(
        '--source-scale-datasets', nargs='+',
        default=['train:train', 'train_sim:train'],
        help=(
            'Source-train annotation_split:image_split datasets defining '
            'bins.'))
    parser.add_argument(
        '--source-rpn-datasets', nargs='+', default=['val:val'],
        help=(
            'Source validation datasets used for trained-RPN control.'))
    parser.add_argument(
        '--target-slice', action='append', dest='target_slices',
        help='NAME:SPLIT:SEQ:START:END; repeat for multiple diagnosis slices.')
    parser.add_argument('--source-rpn-limit', type=int, default=0)
    parser.add_argument('--labeller-checkpoint', required=True)
    parser.add_argument('--dinov2-repo', required=True)
    parser.add_argument('--dinov2-checkpoint', required=True)
    parser.add_argument('--dinov2-model', default='dinov2_vitl14')
    parser.add_argument('--dino-gpus', type=int, nargs='+', required=True)
    parser.add_argument('--head-gpu', type=int, default=0)
    parser.add_argument('--legacy-sdpa-query-chunk', type=int, default=512)
    parser.add_argument('--dino-height', type=int, default=600)
    parser.add_argument('--dino-max-long-side', type=int, default=1333)
    parser.add_argument('--patch-size', type=int, default=14)
    parser.add_argument('--rpn-feat-channels', type=int, default=256)
    parser.add_argument('--roi-fc-channels', type=int, default=1024)
    parser.add_argument('--roi-samples', type=int, default=256)
    parser.add_argument('--proposal-count', type=int, default=2000)
    parser.add_argument('--max-detections', type=int, default=2000)
    parser.add_argument('--recall-ks', type=int, nargs='+',
                        default=list(DEFAULT_RECALL_KS))
    parser.add_argument('--anchor-iou-thresholds', type=float, nargs='+',
                        default=list(DEFAULT_ANCHOR_IOU_THRESHOLDS))
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--feature-cache-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def parse_target_slice(value: str) -> Dict:
    parts = str(value).split(':')
    if len(parts) != 5 or not all(parts):
        raise ValueError(
            'Target slices must use NAME:SPLIT:SEQ:START:END: {}'.format(
                value))
    name, split, seq, start_text, end_text = parts
    try:
        start, end = int(start_text), int(end_text)
    except ValueError as exc:
        raise ValueError('Target slice frame bounds must be integers') from exc
    if start < 0 or end < start:
        raise ValueError('Target slice must satisfy 0 <= START <= END')
    return dict(name=name, split=split, seq=seq, start=start, end=end)


def validate_args(args):
    if args.seed != 0:
        raise ValueError('The read-only protocol requires --seed 0')
    labeller.parse_dataset_specs(args.source_scale_datasets)
    labeller.parse_dataset_specs(args.source_rpn_datasets)
    target_values = (DEFAULT_TARGET_SLICES if args.target_slices is None
                     else args.target_slices)
    args.parsed_target_slices = [parse_target_slice(value)
                                 for value in target_values]
    names = [row['name'] for row in args.parsed_target_slices]
    if len(names) != len(set(names)):
        raise ValueError('Target slice names must be unique')
    if not args.dino_gpus or len(args.dino_gpus) != len(set(args.dino_gpus)):
        raise ValueError('DINO GPU ids must be non-empty and unique')
    if args.head_gpu in args.dino_gpus:
        raise ValueError('Head GPU must be separate from DINO GPUs')
    positive = (
        args.patch_size, args.rpn_feat_channels, args.roi_fc_channels,
        args.roi_samples, args.proposal_count, args.max_detections,
        args.dino_height, args.dino_max_long_side)
    if any(int(value) <= 0 for value in positive):
        raise ValueError('Architecture and image sizes must be positive')
    if args.source_rpn_limit < 0:
        raise ValueError('--source-rpn-limit must be non-negative')
    if not args.recall_ks or any(int(value) <= 0 for value in args.recall_ks):
        raise ValueError('--recall-ks must contain positive integers')
    args.recall_ks = sorted(set(int(value) for value in args.recall_ks))
    if max(args.recall_ks) > int(args.proposal_count):
        raise ValueError('Recall K cannot exceed --proposal-count')
    if (not args.anchor_iou_thresholds
            or any(not 0.0 < float(value) <= 1.0
                   for value in args.anchor_iou_thresholds)):
        raise ValueError('Anchor IoU thresholds must be in (0, 1]')
    args.anchor_iou_thresholds = sorted(set(
        float(value) for value in args.anchor_iou_thresholds))
    if 0.5 not in args.anchor_iou_thresholds:
        raise ValueError(
            '--anchor-iou-thresholds must include 0.5 for diagnosis')
    if not 0.0 < float(args.riou_thr) <= 1.0:
        raise ValueError('--riou-thr must be in (0, 1]')
    for path in (args.labeller_checkpoint, args.dinov2_checkpoint):
        if not os.path.isfile(path):
            raise ValueError('Required checkpoint does not exist: {}'.format(
                path))
    if os.path.exists(args.out_json):
        raise ValueError('Refusing to overwrite a completed audit result')


def protocol_args(args):
    # Required by the existing frozen-head constructor/checkpoint validator.
    args.valid_content_tolerance = 1e-3
    args.deployment_score_thr = 0.05
    args.border_margin_ratio = 0.02
    return args


def discover_dataset_records(data_root: str,
                             specs: Sequence[str]) -> List[Dict]:
    records = []
    for annotation_split, image_split in labeller.parse_dataset_specs(specs):
        records.extend(labeller.discover_labeled_records_with_image_split(
            data_root, annotation_split, image_split))
    records = sorted(records, key=lambda row: (
        row['split'], row['seq'], int(row['frame'])))
    paths = [os.path.realpath(row['image']) for row in records]
    if len(paths) != len(set(paths)):
        raise RuntimeError('Dataset specs contain duplicate source images')
    return records


def discover_target_records(data_root: str, spec: Dict) -> List[Dict]:
    rows = [
        row for row in common.discover_labeled_records(
            data_root, spec['split'], 0)
        if row['seq'] == spec['seq']
        and spec['start'] <= int(row['frame']) <= spec['end']]
    expected = int(spec['end'] - spec['start'] + 1)
    if len(rows) != expected:
        raise RuntimeError(
            'Incomplete target slice {}: expected {}, found {}'.format(
                spec['name'], expected, len(rows)))
    return rows


def dino_resize_meta(image: np.ndarray, args) -> Dict:
    if image is None or image.ndim != 3:
        raise RuntimeError('Cannot read image for token-scale audit')
    ori_h, ori_w = [int(value) for value in image.shape[:2]]
    scale = min(
        float(args.dino_height) / float(min(ori_h, ori_w)),
        float(args.dino_max_long_side) / float(max(ori_h, ori_w)))
    resized_h = max(1, int(round(float(ori_h) * scale)))
    resized_w = max(1, int(round(float(ori_w) * scale)))
    pad_h = int(math.ceil(resized_h / float(args.patch_size))
                * args.patch_size)
    pad_w = int(math.ceil(resized_w / float(args.patch_size))
                * args.patch_size)
    return dict(
        ori_shape=[ori_h, ori_w], resized_shape=[resized_h, resized_w],
        padded_shape=[pad_h, pad_w], scale=float(scale),
        feature_shape=[pad_h // args.patch_size,
                       pad_w // args.patch_size])


def token_objects(gt_original: np.ndarray, resize_meta: Dict,
                  patch_size: int) -> List[Dict]:
    scale_per_token = float(resize_meta['scale']) / float(patch_size)
    rows = []
    for box in np.asarray(gt_original, dtype=np.float32).reshape((-1, 5)):
        width, height = abs(float(box[2])), abs(float(box[3]))
        short_px, long_px = sorted((width, height))
        short_token = short_px * scale_per_token
        long_token = long_px * scale_per_token
        rows.append(dict(
            width_px=width, height_px=height,
            short_px=short_px, long_px=long_px,
            short_token=float(short_token), long_token=float(long_token),
            area_token2=float(short_token * long_token),
            aspect_ratio=float(long_px / max(short_px, 1e-12)),
            angle_rad=float(box[4]), angle_deg=float(math.degrees(box[4]))))
    return rows


def static_scale_rows(records: Sequence[Dict], args, role: str) -> List[Dict]:
    rows = []
    for index, record in enumerate(records):
        image = cv2.imread(record['image'])
        meta = dino_resize_meta(image, args)
        original = labeller.parse_original_gt(record['annotation'])
        rows.append(dict(
            role=role, split=record['split'], seq=record['seq'],
            frame=int(record['frame']), image=os.path.abspath(record['image']),
            resize=meta, objects=token_objects(
                original, meta, args.patch_size)))
        if (index + 1) % 250 == 0 or index + 1 == len(records):
            print('[token-scale] role={} {}/{}'.format(
                role, index + 1, len(records)))
    return rows


def percentile(values: Iterable[float], quantiles: Sequence[float]) -> Dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {str(value): None for value in quantiles}
    return {
        str(value): float(np.percentile(array, float(value)))
        for value in quantiles}


def summarize_scale(rows: Sequence[Dict]) -> Dict:
    objects = [obj for row in rows for obj in row['objects']]
    dimensions = sorted({
        (int(row['resize']['ori_shape'][0]),
         int(row['resize']['ori_shape'][1])) for row in rows})
    scales = [float(row['resize']['scale']) for row in rows]
    quantiles = (0, 5, 25, 50, 75, 95, 100)
    return dict(
        frame_count=int(len(rows)), object_count=int(len(objects)),
        original_image_shapes=[list(value) for value in dimensions],
        resize_scale_range=(None if not scales else
                            [float(min(scales)), float(max(scales))]),
        short_token_percentiles=percentile(
            (obj['short_token'] for obj in objects), quantiles),
        long_token_percentiles=percentile(
            (obj['long_token'] for obj in objects), quantiles),
        area_token2_percentiles=percentile(
            (obj['area_token2'] for obj in objects), quantiles),
        aspect_ratio_percentiles=percentile(
            (obj['aspect_ratio'] for obj in objects), quantiles))


def source_token_boundaries(rows: Sequence[Dict]) -> Dict:
    values = [float(obj['short_token'])
              for row in rows for obj in row['objects']]
    if not values:
        raise RuntimeError('Source scale dataset contains no grab boxes')
    lower, upper = np.percentile(
        np.asarray(values, dtype=np.float64), [100.0 / 3.0, 200.0 / 3.0])
    return dict(
        definition='source_train_short_token_tertiles',
        lower=float(lower), upper=float(upper),
        labels=['source_small', 'source_medium', 'source_large'])


def token_bin(value: float, boundaries: Dict) -> str:
    if float(value) <= float(boundaries['lower']):
        return 'source_small'
    if float(value) <= float(boundaries['upper']):
        return 'source_medium'
    return 'source_large'


def pairwise_hbb_iou(anchors: torch.Tensor,
                     gt_hbboxes: torch.Tensor) -> torch.Tensor:
    if anchors.ndim != 2 or anchors.shape[1] != 4:
        raise ValueError('Anchors must have shape [N,4]')
    if gt_hbboxes.ndim != 2 or gt_hbboxes.shape[1] != 4:
        raise ValueError('GT horizontal boxes must have shape [M,4]')
    if anchors.shape[0] == 0 or gt_hbboxes.shape[0] == 0:
        return anchors.new_zeros((anchors.shape[0], gt_hbboxes.shape[0]))
    left_top = torch.maximum(anchors[:, None, :2], gt_hbboxes[None, :, :2])
    right_bottom = torch.minimum(
        anchors[:, None, 2:], gt_hbboxes[None, :, 2:])
    intersection = (right_bottom - left_top).clamp(min=0)
    intersection = intersection[..., 0] * intersection[..., 1]
    anchor_area = ((anchors[:, 2] - anchors[:, 0]).clamp(min=0)
                   * (anchors[:, 3] - anchors[:, 1]).clamp(min=0))
    gt_area = ((gt_hbboxes[:, 2] - gt_hbboxes[:, 0]).clamp(min=0)
               * (gt_hbboxes[:, 3] - gt_hbboxes[:, 1]).clamp(min=0))
    union = anchor_area[:, None] + gt_area[None, :] - intersection
    return intersection / union.clamp(min=torch.finfo(union.dtype).eps)


def score_rank(scores: torch.Tensor, index: int) -> int:
    value = scores[int(index)]
    return int((scores > value).sum().item()) + 1


def anchor_metrics(rpn_head, feature: torch.Tensor, img_meta: Dict,
                   gt_boxes: torch.Tensor, cls_score: torch.Tensor,
                   thresholds: Sequence[float]) -> List[Dict]:
    from mmdet.core import anchor_inside_flags
    from mmrotate.core import obb2xyxy

    featmap_sizes = [tuple(int(value) for value in feature.shape[-2:])]
    anchor_list, flag_list = rpn_head.get_anchors(
        featmap_sizes, [img_meta], device=feature.device)
    anchors = anchor_list[0][0]
    valid_flags = flag_list[0][0]
    inside = anchor_inside_flags(
        anchors, valid_flags, img_meta['img_shape'][:2],
        rpn_head.train_cfg.allowed_border)
    anchors = anchors[inside]
    flat_scores = cls_score[0].permute(1, 2, 0).reshape(-1).sigmoid()
    if flat_scores.numel() != inside.numel():
        raise RuntimeError('RPN score/anchor ordering mismatch')
    flat_scores = flat_scores[inside]
    gt_hbboxes = obb2xyxy(gt_boxes, rpn_head.version)
    overlaps = pairwise_hbb_iou(anchors, gt_hbboxes)
    assignment = rpn_head.assigner.assign(
        anchors, gt_hbboxes, None, None)
    rows = []
    for gt_index in range(int(gt_boxes.shape[0])):
        gt_overlaps = overlaps[:, gt_index]
        best_index = int(torch.argmax(gt_overlaps).item())
        positive = assignment.gt_inds == int(gt_index + 1)
        positive_indices = torch.nonzero(positive, as_tuple=False).reshape(-1)
        positive_best_rank = None
        if positive_indices.numel():
            local = flat_scores[positive_indices]
            chosen = int(positive_indices[int(torch.argmax(local).item())])
            positive_best_rank = score_rank(flat_scores, chosen)
        rows.append(dict(
            inside_anchor_count=int(anchors.shape[0]),
            max_hbb_iou=float(gt_overlaps[best_index].item()),
            best_iou_anchor_score=float(flat_scores[best_index].item()),
            best_iou_anchor_score_rank=score_rank(flat_scores, best_index),
            exact_assigned_positive_count=int(positive.sum().item()),
            exact_positive_best_score_rank=positive_best_rank,
            anchors_above_iou={
                str(float(threshold)): int(
                    (gt_overlaps >= float(threshold)).sum().item())
                for threshold in thresholds}))
    return rows


def proposal_metrics(proposals: torch.Tensor, gt_boxes: torch.Tensor,
                     recall_ks: Sequence[int], riou_thr: float) -> List[Dict]:
    from mmcv.ops import box_iou_rotated

    if proposals.ndim != 2 or proposals.shape[1] < 5:
        raise ValueError('RPN proposals must have shape [N,>=5]')
    if gt_boxes.shape[0] == 0:
        return []
    if proposals.shape[0] == 0:
        return [dict(
            proposal_count=0, best_riou=0.0, best_usable_rank=None,
            recall_at={str(value): False for value in recall_ks})
                for _ in range(int(gt_boxes.shape[0]))]
    overlaps = box_iou_rotated(
        proposals[:, :5].float(), gt_boxes.float())
    rows = []
    for gt_index in range(int(gt_boxes.shape[0])):
        values = overlaps[:, gt_index]
        usable = torch.nonzero(
            values >= float(riou_thr), as_tuple=False).reshape(-1)
        rows.append(dict(
            proposal_count=int(proposals.shape[0]),
            best_riou=float(values.max().item()),
            best_usable_rank=(None if not usable.numel()
                              else int(usable[0].item()) + 1),
            recall_at={
                str(int(value)): bool(
                    values[:min(int(value), values.numel())].max().item()
                    >= float(riou_thr))
                for value in recall_ks}))
    return rows


def evaluate_rpn_records(dino, heads, records: Sequence[Dict], args,
                         dino_device, head_device, role: str,
                         boundaries: Dict) -> Tuple[List[Dict], Dict]:
    heads.eval()
    rows = []
    cache_hits = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for index, record in enumerate(records):
            feature, img_meta, gt_boxes, _labels, original, cached = (
                labeller.prepare_record(
                    dino, record, args, dino_device, head_device))
            cache_hits += int(cached)
            rpn_outputs = heads.rpn_head([feature])
            cls_scores, bbox_preds = rpn_outputs
            anchor_rows = anchor_metrics(
                heads.rpn_head, feature, img_meta, gt_boxes,
                cls_scores[0], args.anchor_iou_thresholds)
            proposals = heads.rpn_head.get_bboxes(
                cls_scores, bbox_preds, img_metas=[img_meta],
                cfg=heads.proposal_cfg, rescale=False)[0]
            proposal_rows = proposal_metrics(
                proposals, gt_boxes, args.recall_ks, args.riou_thr)
            resize = dict(
                ori_shape=[int(value) for value in img_meta['ori_shape'][:2]],
                resized_shape=[int(value)
                               for value in img_meta['img_shape'][:2]],
                padded_shape=[int(value)
                              for value in img_meta['pad_shape'][:2]],
                scale=float(img_meta['scale_factor'][0]),
                feature_shape=[int(value) for value in feature.shape[-2:]])
            objects = token_objects(original, resize, args.patch_size)
            if not (len(objects) == len(anchor_rows) == len(proposal_rows)):
                raise RuntimeError('GT metric row count mismatch')
            for object_row, anchor_row, proposal_row in zip(
                    objects, anchor_rows, proposal_rows):
                object_row['source_token_bin'] = token_bin(
                    object_row['short_token'], boundaries)
                object_row['anchor'] = anchor_row
                object_row['rpn'] = proposal_row
            rows.append(dict(
                role=role, split=record['split'], seq=record['seq'],
                frame=int(record['frame']), feature_cache_hit=bool(cached),
                resize=resize, objects=objects))
            if (index + 1) % 25 == 0 or index + 1 == len(records):
                recalled = sum(
                    obj['rpn']['best_usable_rank'] is not None
                    for row in rows for obj in row['objects'])
                total = sum(len(row['objects']) for row in rows)
                print('[rpn-coverage] role={} {}/{} recall={}/{} cache={}/{}'
                      .format(role, index + 1, len(records), recalled, total,
                              cache_hits, index + 1))
            del feature, gt_boxes, _labels, proposals
    return rows, dict(
        elapsed_seconds=float(time.perf_counter() - start_time),
        feature_cache_hits=int(cache_hits), frame_count=int(len(records)))


def _summarize_rpn_objects(objects: Sequence[Dict], args) -> Dict:
    count = len(objects)
    if not count:
        return dict(object_count=0)
    best_ranks = [obj['rpn']['best_usable_rank'] for obj in objects
                  if obj['rpn']['best_usable_rank'] is not None]
    return dict(
        object_count=int(count),
        anchor_max_hbb_iou_percentiles=percentile(
            (obj['anchor']['max_hbb_iou'] for obj in objects),
            (0, 5, 25, 50, 75, 95, 100)),
        anchor_coverage_rate={
            str(float(threshold)): float(sum(
                obj['anchor']['max_hbb_iou'] >= float(threshold)
                for obj in objects) / count)
            for threshold in args.anchor_iou_thresholds},
        exact_positive_assignment_rate=float(sum(
            obj['anchor']['exact_assigned_positive_count'] > 0
            for obj in objects) / count),
        rpn_best_riou_percentiles=percentile(
            (obj['rpn']['best_riou'] for obj in objects),
            (0, 5, 25, 50, 75, 95, 100)),
        rpn_recall_at={
            str(int(value)): float(sum(
                obj['rpn']['recall_at'][str(int(value))]
                for obj in objects) / count)
            for value in args.recall_ks},
        rpn_usable_rank_median=(None if not best_ranks else
                                float(np.median(best_ranks))),
        rpn_usable_count=int(len(best_ranks)))


def summarize_rpn(rows: Sequence[Dict], args, boundaries: Dict) -> Dict:
    objects = [obj for row in rows for obj in row['objects']]
    summary = _summarize_rpn_objects(objects, args)
    summary['frame_count'] = int(len(rows))
    summary['token_scale'] = summarize_scale(rows)
    summary['source_token_bins'] = {
        label: _summarize_rpn_objects(
            [obj for obj in objects if obj['source_token_bin'] == label], args)
        for label in boundaries['labels']}
    return summary


def diagnose_slice(source_summary: Dict, target_summary: Dict,
                   args) -> str:
    source_recall = float(source_summary.get(
        'rpn_recall_at', {}).get(str(max(args.recall_ks)), 0.0))
    if source_recall < 0.8:
        return 'SOURCE_RPN_CONTROL_INSUFFICIENT'
    anchor_rate = float(target_summary.get(
        'anchor_coverage_rate', {}).get('0.5', 0.0))
    proposal_rate = float(target_summary.get(
        'rpn_recall_at', {}).get(str(max(args.recall_ks)), 0.0))
    if proposal_rate >= 0.8:
        return 'RPN_GEOMETRY_PRESENT_CHECK_ROI_ORDERING'
    if anchor_rate >= 0.8 and proposal_rate < 0.5:
        return 'ANCHORS_COVER_BUT_TRAINED_RPN_FAILS'
    if anchor_rate < 0.8:
        return 'ANCHOR_OR_ASSIGNMENT_GEOMETRY_LIMITED'
    return 'MIXED_RPN_COVERAGE'


def parameter_counts(dino, heads) -> Dict:
    def count(module):
        return int(sum(parameter.numel() for parameter in module.parameters()))
    return dict(
        frozen_dinov2=count(dino), current_heads=count(heads),
        current_rpn=count(heads.rpn_head), current_roi=count(heads.roi_head),
        audit_trainable_parameters=0)


def assert_disjoint(source_records: Sequence[Dict],
                    target_groups: Dict[str, Sequence[Dict]]):
    source = {os.path.realpath(row['image']) for row in source_records}
    for name, records in target_groups.items():
        overlap = source & {
            os.path.realpath(row['image']) for row in records}
        if overlap:
            raise RuntimeError(
                'Target diagnosis leaked into source records for {}'.format(
                    name))


def main():
    args = protocol_args(parse_args())
    validate_args(args)
    labeller.set_seed(args.seed)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)

    source_scale_records = discover_dataset_records(
        args.data_root, args.source_scale_datasets)
    source_rpn_records = discover_dataset_records(
        args.data_root, args.source_rpn_datasets)
    if args.source_rpn_limit:
        source_rpn_records = source_rpn_records[:args.source_rpn_limit]
    target_groups = {
        spec['name']: discover_target_records(args.data_root, spec)
        for spec in args.parsed_target_slices}
    assert_disjoint(
        list(source_scale_records) + list(source_rpn_records), target_groups)

    source_scale_rows = static_scale_rows(
        source_scale_records, args, 'source_train_scale_definition')
    boundaries = source_token_boundaries(source_scale_rows)

    dino_devices = [torch.device('cuda:{}'.format(gpu))
                    for gpu in args.dino_gpus]
    dino_device = dino_devices[0]
    head_device = torch.device('cuda:{}'.format(args.head_gpu))
    dino, heads = far_audit.load_frozen_labeller(
        args, dino_devices, head_device)
    dino_versions = common.module_parameter_versions(dino)
    head_versions = common.module_parameter_versions(heads)

    source_rows, source_runtime = evaluate_rpn_records(
        dino, heads, source_rpn_records, args, dino_device, head_device,
        'source_validation_rpn_control', boundaries)
    source_summary = summarize_rpn(source_rows, args, boundaries)
    targets = {}
    for name, records in target_groups.items():
        rows, runtime = evaluate_rpn_records(
            dino, heads, records, args, dino_device, head_device,
            'target_diagnosis_only', boundaries)
        summary = summarize_rpn(rows, args, boundaries)
        targets[name] = dict(
            specification=next(
                row for row in args.parsed_target_slices
                if row['name'] == name),
            summary=summary, runtime=runtime, rows=rows,
            diagnosis=diagnose_slice(source_summary, summary, args))

    dino_unchanged = (
        dino_versions == common.module_parameter_versions(dino))
    heads_unchanged = (
        head_versions == common.module_parameter_versions(heads))
    if not dino_unchanged or not heads_unchanged:
        raise RuntimeError('Read-only parameter invariant failed')

    payload = dict(
        audit=AUDIT_NAME, protocol_version=PROTOCOL_VERSION,
        labeller_checkpoint=os.path.abspath(args.labeller_checkpoint),
        labeller_checkpoint_sha256=common.file_sha256(
            args.labeller_checkpoint),
        dinov2_checkpoint=os.path.abspath(args.dinov2_checkpoint),
        dinov2_checkpoint_sha256=common.file_sha256(
            args.dinov2_checkpoint),
        protocol=dict(
            source_scale_datasets=list(args.source_scale_datasets),
            source_rpn_datasets=list(args.source_rpn_datasets),
            target_role='diagnosis_only',
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            source_defined_token_bins=boundaries,
            token_definition='resized_obb_side_pixels_divided_by_patch_size',
            theoretical_anchor_definition=(
                'exact_inside_RPN_anchors_vs_scaled_GTs_horizontal_envelope'),
            trained_rpn_definition=(
                'decoded_rotated_proposals_after_current_RPN_NMS'),
            recall_ks=list(args.recall_ks), riou_thr=float(args.riou_thr)),
        isolation=dict(
            optimizer_steps=0, checkpoint_writes=0,
            dino_frozen=True, dino_parameters_unchanged=dino_unchanged,
            labeller_heads_frozen=True,
            labeller_parameters_unchanged=heads_unchanged,
            target_labels_used_for_evaluation_only=True),
        parameter_counts=parameter_counts(dino, heads),
        source_train_scale=dict(
            summary=summarize_scale(source_scale_rows),
            rows=source_scale_rows),
        source_rpn_control=dict(
            summary=source_summary, runtime=source_runtime,
            rows=source_rows),
        target_diagnoses=targets)
    replacements = common.write_json_atomic(args.out_json, payload)
    print('[audit] source_rpn_r{}={:.3f}'.format(
        max(args.recall_ks), source_summary['rpn_recall_at'][
            str(max(args.recall_ks))]))
    for name, result in targets.items():
        summary = result['summary']
        print('[audit] {} {} anchor@0.5={:.3f} rpn_r{}={:.3f}'.format(
            name, result['diagnosis'],
            summary['anchor_coverage_rate'].get('0.5', 0.0),
            max(args.recall_ks),
            summary['rpn_recall_at'][str(max(args.recall_ks))]))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
