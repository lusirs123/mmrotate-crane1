#!/usr/bin/env python3
"""Read-only RPN-to-ROI attrition and DINO latency audit.

The tool traces every source-selected RPN proposal through raw ROI logits,
class-agnostic rotated-box regression, rotated NMS, and the formal
valid-content filter.  It performs no optimizer step and writes no checkpoint.
Target
slices are diagnosis-only; source-train token bins are imported from the
completed token-scale/RPN coverage audit.
"""

import argparse
import json
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
from crane_project.tools import (
    dino_teacher_token_scale_rpn_coverage_audit as coverage)


AUDIT_NAME = 'DINO RPN-to-ROI Attrition and Latency Audit V1'
# Version 2 fixes terminal attribution after an ROI regression recovery.
# Keep AUDIT_NAME stable so the standalone V2 report can also repair V1 JSON.
PROTOCOL_VERSION = 2
DEFAULT_TARGET_SLICES = coverage.DEFAULT_TARGET_SLICES
DEFAULT_RECALL_KS = coverage.DEFAULT_RECALL_KS


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument(
        '--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument(
        '--source-rpn-datasets', nargs='+', default=['val:val'])
    parser.add_argument(
        '--target-slice', action='append', dest='target_slices',
        help='NAME:SPLIT:SEQ:START:END; repeat for multiple diagnosis slices.')
    parser.add_argument('--source-rpn-limit', type=int, default=0)
    parser.add_argument('--coverage-audit-json', required=True)
    parser.add_argument(
        '--source-selection-json',
        help=(
            'Source-only epoch-selection proof. Required when the coverage '
            'audit used a classifier-different but frozen-head-equivalent '
            'checkpoint; only its source token bins are reused.'))
    parser.add_argument(
        '--expected-source-retention-rate', type=float, default=0.985,
        help='Fixed source-only selection gate expected in the proof JSON.')
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
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--feature-cache-dir', required=True)
    parser.add_argument(
        '--source-fresh-latency-samples', type=int, default=10,
        help='Source frames recomputed without cache for latency only.')
    parser.add_argument(
        '--target-feature-mode', choices=['fresh_fp32', 'cache_fp16'],
        default='fresh_fp32')
    parser.add_argument('--latency-warmup', type=int, default=1)
    parser.add_argument('--reconstruction-check-count', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def protocol_args(args):
    args.valid_content_tolerance = 1e-3
    args.deployment_score_thr = 0.05
    args.border_margin_ratio = 0.02
    return args


def validate_args(args):
    if args.seed != 0:
        raise ValueError('The read-only protocol requires --seed 0')
    labeller.parse_dataset_specs(args.source_rpn_datasets)
    target_values = (DEFAULT_TARGET_SLICES if args.target_slices is None
                     else args.target_slices)
    args.parsed_target_slices = [
        coverage.parse_target_slice(value) for value in target_values]
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
    nonnegative = (
        args.source_rpn_limit, args.source_fresh_latency_samples,
        args.latency_warmup, args.reconstruction_check_count)
    if any(int(value) < 0 for value in nonnegative):
        raise ValueError('Limits and sample counts must be non-negative')
    if not args.recall_ks or any(int(value) <= 0 for value in args.recall_ks):
        raise ValueError('--recall-ks must contain positive integers')
    args.recall_ks = sorted(set(int(value) for value in args.recall_ks))
    if max(args.recall_ks) > int(args.proposal_count):
        raise ValueError('Recall K cannot exceed --proposal-count')
    if not 0.0 < float(args.riou_thr) <= 1.0:
        raise ValueError('--riou-thr must be in (0, 1]')
    if not 0.0 < float(args.expected_source_retention_rate) <= 1.0:
        raise ValueError(
            '--expected-source-retention-rate must be in (0, 1]')
    for path in (
            args.coverage_audit_json, args.labeller_checkpoint,
            args.dinov2_checkpoint):
        if not os.path.isfile(path):
            raise ValueError('Required input does not exist: {}'.format(path))
    if (args.source_selection_json
            and not os.path.isfile(args.source_selection_json)):
        raise ValueError(
            'Source-selection proof does not exist: {}'.format(
                args.source_selection_json))
    if os.path.exists(args.out_json):
        raise ValueError('Refusing to overwrite a completed audit result')


def load_source_selection_contract(path: str, args,
                                   current_checkpoint_sha256: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if payload.get('decision') != 'SOURCE_ONLY_EPOCH_SELECTED_TARGET_NOT_READ':
        raise RuntimeError('Source-selection proof is not source-only')
    if (payload.get('source_only') is not True
            or payload.get('target_data_read') is not False):
        raise RuntimeError('Source-selection proof used target data')
    selected = payload.get('selected', {})
    invariants = payload.get('parameter_invariants', {})
    expected_rate = float(args.expected_source_retention_rate)
    proof_rate = float(payload.get('min_exact_retention_rate', -1.0))
    if not abs(proof_rate - expected_rate) <= 1e-12:
        raise RuntimeError('Source-selection retention gate mismatch')
    retention = selected.get('source_exact_retention', {})
    baseline_count = int(retention.get('baseline_correct_count', 0))
    retained_count = int(retention.get('retained_correct_count', -1))
    if (baseline_count <= 0 or retained_count < 0
            or float(retained_count / baseline_count) < expected_rate):
        raise RuntimeError(
            'Selected checkpoint does not pass source-retention gate')
    if selected.get('output_checkpoint_sha256') != current_checkpoint_sha256:
        raise RuntimeError(
            'Source-selection proof does not match current checkpoint')
    if invariants.get('frozen_tensors_bit_identical') is not True:
        raise RuntimeError(
            'Source-selection proof lacks frozen-head equivalence')
    changed = set(invariants.get('changed_parameter_names', ()))
    expected = {
        'roi_head.bbox_head.fc_cls.weight',
        'roi_head.bbox_head.fc_cls.bias',
    }
    if not changed or not changed <= expected:
        raise RuntimeError(
            'Source-selection proof changed unauthorized parameters')
    return dict(
        path=os.path.abspath(path), sha256=common.file_sha256(path),
        selected_epoch=int(selected['epoch']),
        current_checkpoint_sha256=current_checkpoint_sha256,
        frozen_tensors_bit_identical=True,
        changed_parameter_names=sorted(changed),
        min_exact_retention_rate=proof_rate,
        exact_retention_rate=float(retained_count / baseline_count),
        target_data_read=False)


def load_coverage_contract(path: str, args) -> Tuple[Dict, Dict]:
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if payload.get('audit') != coverage.AUDIT_NAME:
        raise RuntimeError('Input is not the token-scale/RPN coverage audit')
    isolation = payload.get('isolation', {})
    required_true = (
        'dino_parameters_unchanged', 'labeller_parameters_unchanged',
        'target_labels_used_for_evaluation_only')
    if any(isolation.get(key) is not True for key in required_true):
        raise RuntimeError('Coverage audit isolation contract is invalid')
    if int(isolation.get('optimizer_steps', -1)) != 0:
        raise RuntimeError('Coverage audit performed optimizer steps')
    current_labeller_sha = common.file_sha256(args.labeller_checkpoint)
    current_dino_sha = common.file_sha256(args.dinov2_checkpoint)
    if payload.get('dinov2_checkpoint_sha256') != current_dino_sha:
        raise RuntimeError(
            'Coverage audit checkpoint mismatch: dinov2_checkpoint_sha256')
    coverage_labeller_sha = payload.get('labeller_checkpoint_sha256')
    if coverage_labeller_sha == current_labeller_sha:
        reuse_contract = dict(
            mode='exact_checkpoint_match',
            coverage_checkpoint_sha256=coverage_labeller_sha,
            current_checkpoint_sha256=current_labeller_sha,
            source_selection_proof=None)
    else:
        if not args.source_selection_json:
            raise RuntimeError(
                'Coverage audit labeller differs; provide the source-only '
                'selection proof to reuse source token bins')
        proof = load_source_selection_contract(
            args.source_selection_json, args, current_labeller_sha)
        reuse_contract = dict(
            mode='source_token_bins_only_with_fc_cls_change_proof',
            coverage_checkpoint_sha256=coverage_labeller_sha,
            current_checkpoint_sha256=current_labeller_sha,
            source_selection_proof=proof)
    boundaries = payload.get(
        'protocol', {}).get('source_defined_token_bins')
    if (not isinstance(boundaries, dict)
            or not {'lower', 'upper', 'labels'} <= set(boundaries)):
        raise RuntimeError('Coverage audit lacks source-defined token bins')
    payload = dict(payload)
    payload['_reuse_contract'] = reuse_contract
    return payload, boundaries


def synchronize(devices: Sequence[torch.device]):
    seen = set()
    for device in devices:
        key = str(device)
        if device.type == 'cuda' and key not in seen:
            torch.cuda.synchronize(device)
            seen.add(key)


def timed_call(function, devices: Sequence[torch.device]):
    synchronize(devices)
    start = time.perf_counter()
    result = function()
    synchronize(devices)
    return result, float((time.perf_counter() - start) * 1000.0)


def fresh_feature(dino, record: Dict, args,
                  dino_devices: Sequence[torch.device],
                  head_device: torch.device):
    start = time.perf_counter()
    image = cv2.imread(record['image'], cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('Cannot read {}'.format(record['image']))
    tensor, dino_meta = common.resize_and_normalize_bgr(
        image, args.dino_height, args.patch_size,
        args.dino_max_long_side)
    preprocess_ms = float((time.perf_counter() - start) * 1000.0)

    dino_device = dino_devices[0]

    def run_dino():
        input_tensor = tensor.to(dino_device)
        return common.extract_patch_grid(
            dino, input_tensor, args.patch_size)

    feature, dino_ms = timed_call(run_dino, dino_devices)

    def transfer_feature():
        return feature.to(head_device, dtype=torch.float32)

    head_feature, transfer_ms = timed_call(
        transfer_feature, [dino_devices[-1], head_device])
    img_meta = labeller.feature_meta(record['image'], dino_meta)
    gt_boxes, gt_labels, original = labeller.scaled_gt_tensors(
        record['annotation'], float(dino_meta['scale']), head_device)
    return (
        head_feature, img_meta, gt_boxes, gt_labels, original,
        dict(preprocess_ms=preprocess_ms, dino_ms=dino_ms,
             dino_to_head_ms=transfer_ms, feature_cache_hit=False,
             feature_precision='fp32_fresh'))


def cached_feature(dino, record: Dict, args, dino_device, head_device):
    feature, img_meta, gt_boxes, labels, original, cached = (
        labeller.prepare_record(
            dino, record, args, dino_device, head_device))
    return (
        feature, img_meta, gt_boxes, labels, original,
        dict(preprocess_ms=None, dino_ms=None, dino_to_head_ms=None,
             feature_cache_hit=bool(cached),
             feature_precision='fp16_cache_to_fp32_head'))


def rotated_ious(boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    if boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return boxes.new_zeros((boxes.shape[0], gt_boxes.shape[0]))
    from mmcv.ops import box_iou_rotated

    return box_iou_rotated(boxes[:, :5].float(), gt_boxes.float())


def first_usable_rank(values: torch.Tensor, threshold: float):
    indices = torch.nonzero(
        values >= float(threshold), as_tuple=False).reshape(-1)
    return None if not indices.numel() else int(indices[0].item()) + 1


def probability_rank(probabilities: torch.Tensor, index: int) -> int:
    value = probabilities[int(index)]
    return int((probabilities > value).sum().item()) + 1


def original_nms_indices(probabilities: torch.Tensor, score_thr: float,
                         filtered_keep: torch.Tensor) -> torch.Tensor:
    """Map MMRotate's post-threshold NMS indices back to proposal rows."""
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise RuntimeError(
            'Attrition audit requires the trained one-class ROI head')
    score_valid = probabilities[:, 0] > float(score_thr)
    original_indices = torch.nonzero(
        score_valid, as_tuple=False).reshape(-1)
    keep = filtered_keep.long()
    if keep.numel() and int(keep.max().item()) >= original_indices.numel():
        raise RuntimeError('NMS index exceeds score-filtered proposal count')
    return original_indices[keep]


def attrition_cause_from_object(obj: Dict) -> str:
    """Classify the terminal failure, after allowing ROI recovery.

    RPN IoU is an intermediate diagnostic, not a terminal failure: a coarse
    proposal can still decode into a usable ROI box.  The previous priority
    checked RPN first and therefore mislabeled recovered frames as RPN misses.
    """
    post_valid = obj['post_valid_content']
    post_nms = obj['post_nms']
    roi = obj['roi_regression']
    rpn = obj['rpn']
    if bool(post_valid['top1_hit']):
        return 'ROI_TOP1_RESTORED'
    if post_valid['best_usable_rank'] is None:
        if post_nms['best_usable_rank'] is None:
            if int(roi['decoded_usable_count']) > 0:
                return 'ROI_ORDERING_OR_NMS_REMOVES_GEOMETRY'
            if rpn['best_usable_rank'] is not None:
                return 'ROI_REGRESSION_DESTROYS_RPN_GEOMETRY'
            return 'RPN_MISS'
        return 'VALID_CONTENT_FILTER_REMOVES_GEOMETRY'
    return 'ROI_GEOMETRY_SURVIVES_BUT_RANKING_FAILS'


def object_attrition_rows(
        proposals: torch.Tensor, decoded: torch.Tensor,
        probabilities: torch.Tensor, nms_keep: torch.Tensor,
        final_detections: torch.Tensor, valid_detections: np.ndarray,
        gt_scaled: torch.Tensor, gt_original: np.ndarray,
        threshold: float) -> List[Dict]:
    rpn_iou = rotated_ious(proposals, gt_scaled)
    decoded_iou = rotated_ious(decoded, gt_scaled)
    final_iou = rotated_ious(final_detections, gt_scaled)
    valid_tensor = torch.as_tensor(
        valid_detections[:, :5], dtype=torch.float32,
        device=gt_scaled.device)
    original_tensor = torch.as_tensor(
        gt_original, dtype=torch.float32, device=gt_scaled.device)
    valid_iou = rotated_ious(valid_tensor, original_tensor)
    foreground = probabilities[:, 0]
    background = probabilities[:, -1]
    kept = torch.zeros(
        (decoded.shape[0],), dtype=torch.bool, device=decoded.device)
    if nms_keep.numel():
        kept[nms_keep.long()] = True

    rows = []
    for gt_index in range(int(gt_scaled.shape[0])):
        rpn_values = rpn_iou[:, gt_index]
        decoded_values = decoded_iou[:, gt_index]
        rpn_usable = rpn_values >= float(threshold)
        decoded_usable = decoded_values >= float(threshold)
        rpn_indices = torch.nonzero(
            rpn_usable, as_tuple=False).reshape(-1)
        rpn_to_decoded = rpn_usable & decoded_usable
        best_rpn_index = (None if not rpn_values.numel() else
                          int(torch.argmax(rpn_values).item()))

        best_usable_by_fg = None
        if rpn_indices.numel():
            local = foreground[rpn_indices]
            best_usable_by_fg = int(
                rpn_indices[int(torch.argmax(local).item())].item())

        decoded_order = torch.argsort(foreground, descending=True)
        decoded_ranked_iou = decoded_values[decoded_order]
        decoded_usable_score_rank = first_usable_rank(
            decoded_ranked_iou, threshold)

        post_nms_rank = (None if final_iou.shape[0] == 0 else
                         first_usable_rank(
                             final_iou[:, gt_index], threshold))
        post_valid_rank = (None if valid_iou.shape[0] == 0 else
                           first_usable_rank(
                               valid_iou[:, gt_index], threshold))
        final_top1 = bool(
            valid_iou.shape[0] > 0
            and float(valid_iou[0, gt_index].item()) >= float(threshold))

        best_record = None
        if best_rpn_index is not None:
            best_record = dict(
                index=best_rpn_index,
                rpn_riou=float(rpn_values[best_rpn_index].item()),
                decoded_riou=float(decoded_values[best_rpn_index].item()),
                foreground_probability=float(
                    foreground[best_rpn_index].item()),
                background_probability=float(
                    background[best_rpn_index].item()),
                foreground_over_background=bool(
                    foreground[best_rpn_index] > background[best_rpn_index]),
                foreground_rank=probability_rank(
                    foreground, best_rpn_index),
                kept_after_nms=bool(kept[best_rpn_index].item()))
        best_fg_record = None
        if best_usable_by_fg is not None:
            index = best_usable_by_fg
            best_fg_record = dict(
                index=index,
                rpn_riou=float(rpn_values[index].item()),
                decoded_riou=float(decoded_values[index].item()),
                foreground_probability=float(foreground[index].item()),
                background_probability=float(background[index].item()),
                foreground_over_background=bool(
                    foreground[index] > background[index]),
                foreground_rank=probability_rank(foreground, index),
                kept_after_nms=bool(kept[index].item()))
        row = dict(
            attrition_cause=None,
            rpn=dict(
                proposal_count=int(proposals.shape[0]),
                best_riou=(0.0 if not rpn_values.numel() else
                           float(rpn_values.max().item())),
                best_usable_rank=first_usable_rank(rpn_values, threshold),
                usable_count=int(rpn_usable.sum().item())),
            best_rpn_geometry=best_record,
            best_foreground_among_rpn_usable=best_fg_record,
            roi_regression=dict(
                best_decoded_riou=(0.0 if not decoded_values.numel() else
                                   float(decoded_values.max().item())),
                rpn_usable_survives_count=int(rpn_to_decoded.sum().item()),
                decoded_usable_count=int(decoded_usable.sum().item()),
                decoded_usable_foreground_rank=decoded_usable_score_rank),
            post_nms=dict(
                detection_count=int(final_detections.shape[0]),
                best_usable_rank=post_nms_rank,
                best_riou=(0.0 if final_iou.shape[0] == 0 else
                           float(final_iou[:, gt_index].max().item()))),
            post_valid_content=dict(
                detection_count=int(valid_detections.shape[0]),
                best_usable_rank=post_valid_rank,
                top1_hit=final_top1,
                best_riou=(0.0 if valid_iou.shape[0] == 0 else
                           float(valid_iou[:, gt_index].max().item()))))
        row['attrition_cause'] = attrition_cause_from_object(row)
        rows.append(row)
    return rows


def manual_rpn_roi_trace(heads, feature: torch.Tensor, img_meta: Dict,
                         gt_scaled: torch.Tensor, gt_original: np.ndarray,
                         args, timed: bool, devices: Sequence[torch.device]):
    from mmrotate.core import multiclass_nms_rotated, rbbox2roi

    def run_rpn():
        cls_scores, bbox_preds = heads.rpn_head([feature])
        proposals = heads.rpn_head.get_bboxes(
            cls_scores, bbox_preds, img_metas=[img_meta],
            cfg=heads.proposal_cfg, rescale=False)[0]
        return proposals

    if timed:
        proposals, rpn_ms = timed_call(run_rpn, devices)
    else:
        proposals, rpn_ms = run_rpn(), None

    if proposals.shape[0] == 0:
        decoded = proposals.new_zeros((0, 5))
        probabilities = proposals.new_zeros((0, 2))
        keep = torch.zeros((0,), dtype=torch.long, device=proposals.device)
        final_scaled = proposals.new_zeros((0, 6))
        valid = np.zeros((0, 6), dtype=np.float32)
        objects = object_attrition_rows(
            proposals, decoded, probabilities, keep, final_scaled, valid,
            gt_scaled, gt_original, args.riou_thr)
        return dict(
            objects=objects,
            counts=dict(
                rpn_proposals=0, roi_decoded=0,
                post_nms=0, post_valid=0),
            filter_stats=dict(
                raw_detection_count=0,
                invalid_border_filtered_count=0,
                valid_detection_count=0),
            valid_detections=valid,
            latency=dict(
                rpn_ms=rpn_ms,
                roi_head_ms=0.0 if timed else None,
                roi_decode_ms=0.0 if timed else None,
                roi_nms_ms=0.0 if timed else None,
                valid_filter_ms=0.0 if timed else None))

    rois = rbbox2roi([proposals[:, :5]])

    def run_roi_head():
        return heads.roi_head._bbox_forward([feature], rois)

    if timed:
        bbox_results, roi_head_ms = timed_call(run_roi_head, devices)
    else:
        bbox_results, roi_head_ms = run_roi_head(), None
    cls_score = bbox_results['cls_score']
    bbox_pred = bbox_results['bbox_pred']

    def decode_roi():
        return heads.roi_head.bbox_head.get_bboxes(
            rois, cls_score, bbox_pred, img_meta['img_shape'],
            img_meta['scale_factor'], rescale=False, cfg=None)

    if timed:
        decoded_and_scores, decode_ms = timed_call(decode_roi, devices)
    else:
        decoded_and_scores, decode_ms = decode_roi(), None
    decoded, probabilities = decoded_and_scores

    def run_nms():
        cfg = heads.roi_head.test_cfg
        return multiclass_nms_rotated(
            decoded, probabilities, cfg.score_thr, cfg.nms,
            cfg.max_per_img, return_inds=True)

    if timed:
        nms_result, nms_ms = timed_call(run_nms, devices)
    else:
        nms_result, nms_ms = run_nms(), None
    final_scaled, _labels, filtered_keep = nms_result
    keep = original_nms_indices(
        probabilities, heads.roi_head.test_cfg.score_thr, filtered_keep)

    start = time.perf_counter()
    final_original = final_scaled.detach().cpu().numpy().astype(
        np.float32, copy=True)
    if final_original.size:
        scale = float(img_meta['scale_factor'][0])
        final_original[:, :4] /= scale
    valid, filter_stats = labeller.filter_valid_rotated_detections(
        final_original, img_meta, args.valid_content_tolerance)
    valid_ms = float((time.perf_counter() - start) * 1000.0) if timed else None

    objects = object_attrition_rows(
        proposals, decoded, probabilities, keep, final_scaled, valid,
        gt_scaled, gt_original, args.riou_thr)
    return dict(
        objects=objects,
        counts=dict(
            rpn_proposals=int(proposals.shape[0]),
            roi_decoded=int(decoded.shape[0]),
            post_nms=int(final_scaled.shape[0]),
            post_valid=int(valid.shape[0])),
        filter_stats=filter_stats,
        valid_detections=valid,
        latency=dict(
            rpn_ms=rpn_ms, roi_head_ms=roi_head_ms,
            roi_decode_ms=decode_ms, roi_nms_ms=nms_ms,
            valid_filter_ms=valid_ms))


def reconstruction_check(heads, feature, img_meta, manual_valid,
                         args) -> Dict:
    expected = heads.simple_test(feature, img_meta)
    expected, _stats = labeller.filter_valid_rotated_detections(
        expected, img_meta, args.valid_content_tolerance)
    shape_equal = expected.shape == manual_valid.shape
    max_abs_diff = None
    close = False
    if shape_equal:
        if expected.size:
            max_abs_diff = float(np.max(np.abs(expected - manual_valid)))
        else:
            max_abs_diff = 0.0
        close = bool(np.allclose(
            expected, manual_valid, rtol=0.0, atol=1e-4))
    return dict(
        shape_equal=bool(shape_equal), allclose_atol_1e_4=close,
        max_abs_diff=max_abs_diff,
        expected_count=int(expected.shape[0]),
        manual_count=int(manual_valid.shape[0]))


def reset_peak_memory(devices: Sequence[torch.device]):
    for device in devices:
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)


def peak_memory(devices: Sequence[torch.device]) -> Dict:
    result = {}
    for device in devices:
        if device.type == 'cuda':
            result[str(device)] = dict(
                allocated_mib=float(
                    torch.cuda.max_memory_allocated(device) / 1024.0 ** 2),
                reserved_mib=float(
                    torch.cuda.max_memory_reserved(device) / 1024.0 ** 2))
    return result


def evaluate_records(dino, heads, records: Sequence[Dict], args,
                     dino_devices: Sequence[torch.device],
                     head_device: torch.device, role: str,
                     boundaries: Dict, force_fresh: bool,
                     fresh_latency_samples: int):
    rows = []
    all_devices = list(dino_devices) + [head_device]
    reset_peak_memory(all_devices)
    for index, record in enumerate(records):
        timed_fresh = bool(force_fresh or index < fresh_latency_samples)
        if timed_fresh:
            prepared = fresh_feature(
                dino, record, args, dino_devices, head_device)
        else:
            prepared = cached_feature(
                dino, record, args, dino_devices[0], head_device)
        (feature, img_meta, gt_boxes, labels,
         original, feature_timing) = prepared
        trace = manual_rpn_roi_trace(
            heads, feature, img_meta, gt_boxes, original, args,
            timed=timed_fresh, devices=all_devices)
        resize = dict(
            ori_shape=[int(value) for value in img_meta['ori_shape'][:2]],
            resized_shape=[int(value)
                           for value in img_meta['img_shape'][:2]],
            scale=float(img_meta['scale_factor'][0]),
            feature_shape=[int(value) for value in feature.shape[-2:]])
        token_rows = coverage.token_objects(
            original, resize, args.patch_size)
        if not (len(token_rows) == len(trace['objects'])):
            raise RuntimeError('GT/attrition row count mismatch')
        for token_row, attrition in zip(token_rows, trace['objects']):
            attrition['token_scale'] = token_row
            attrition['source_token_bin'] = coverage.token_bin(
                token_row['short_token'], boundaries)
        reconstruction = None
        if index < int(args.reconstruction_check_count):
            reconstruction = reconstruction_check(
                heads, feature, img_meta, trace['valid_detections'], args)
            if not reconstruction['allclose_atol_1e_4']:
                raise RuntimeError(
                    'Manual ROI trace does not reconstruct formal inference')
        latency = dict(feature_timing)
        latency.update(trace['latency'])
        timed_values = [
            value for key, value in latency.items()
            if key.endswith('_ms') and value is not None]
        latency['dino_branch_total_ms'] = (
            float(sum(timed_values)) if timed_values else None)
        rows.append(dict(
            role=role, split=record['split'], seq=record['seq'],
            frame=int(record['frame']), resize=resize,
            counts=trace['counts'], filter_stats=trace['filter_stats'],
            objects=trace['objects'], latency=latency,
            reconstruction=reconstruction))
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            causes = {}
            for row in rows:
                for obj in row['objects']:
                    cause = obj['attrition_cause']
                    causes[cause] = causes.get(cause, 0) + 1
            print('[roi-attrition] role={} {}/{} causes={}'.format(
                role, index + 1, len(records), causes))
        del feature, gt_boxes, labels
    return rows, peak_memory(all_devices)


def percentile(values: Iterable[float], quantiles=(50, 95)) -> Dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {str(value): None for value in quantiles}
    return {str(value): float(np.percentile(array, value))
            for value in quantiles}


def summarize_latency(rows: Sequence[Dict]) -> Dict:
    keys = (
        'preprocess_ms', 'dino_ms', 'dino_to_head_ms', 'rpn_ms',
        'roi_head_ms', 'roi_decode_ms', 'roi_nms_ms', 'valid_filter_ms',
        'dino_branch_total_ms')
    summary = {}
    for key in keys:
        values = [float(row['latency'][key]) for row in rows
                  if row['latency'].get(key) is not None]
        summary[key] = dict(
            sample_count=int(len(values)),
            mean=(None if not values else float(np.mean(values))),
            percentiles=percentile(values))
    return summary


def summarize_attrition(rows: Sequence[Dict],
                        include_token_bins: bool = True) -> Dict:
    objects = [obj for row in rows for obj in row['objects']]
    count = len(objects)
    frame_outcomes = [dict(
        seq=row.get('seq', '__summary__'),
        frame=int(index if row.get('frame') is None else row['frame']),
        hit=bool(any(
            obj['post_valid_content']['top1_hit']
            for obj in row['objects'])))
        for index, row in enumerate(rows)]
    frame_top1_hits = int(sum(row['hit'] for row in frame_outcomes))
    frame_top1_mcml = labeller.longest_miss(frame_outcomes, 'hit')
    causes = {}
    for obj in objects:
        cause = obj['attrition_cause']
        causes[cause] = causes.get(cause, 0) + 1
    terminal_failures = [
        obj for obj in objects
        if not bool(obj['post_valid_content']['top1_hit'])]
    terminal_failure_causes = {}
    for obj in terminal_failures:
        cause = obj['attrition_cause']
        terminal_failure_causes[cause] = (
            terminal_failure_causes.get(cause, 0) + 1)
    if count == 0:
        summary = dict(
            frame_count=len(rows), object_count=0, causes=causes,
            final_top1_hits=frame_top1_hits,
            final_top1_mcml=frame_top1_mcml,
            terminal_failure_count=0, terminal_failure_causes={})
        if include_token_bins:
            summary['source_token_bins'] = {
                label: dict(frame_count=0, object_count=0, causes={})
                for label in (
                    'source_small', 'source_medium', 'source_large')}
        return summary

    def rate(predicate):
        return float(sum(bool(predicate(obj)) for obj in objects) / count)

    fg_records = [obj['best_foreground_among_rpn_usable'] for obj in objects
                  if obj['best_foreground_among_rpn_usable'] is not None]
    fg_ranks = [row['foreground_rank'] for row in fg_records]
    summary = dict(
        frame_count=int(len(rows)), object_count=int(count), causes=causes,
        final_top1_hits=frame_top1_hits,
        final_top1_mcml=frame_top1_mcml,
        terminal_failure_count=int(len(terminal_failures)),
        terminal_failure_causes=terminal_failure_causes,
        rpn_recall=rate(
            lambda obj: obj['rpn']['best_usable_rank'] is not None),
        rpn_geometry_survives_roi_regression=rate(
            lambda obj: obj['roi_regression'][
                'rpn_usable_survives_count'] > 0),
        decoded_geometry_exists=rate(
            lambda obj: obj['roi_regression']['decoded_usable_count'] > 0),
        post_nms_recall=rate(
            lambda obj: obj['post_nms']['best_usable_rank'] is not None),
        post_valid_recall=rate(
            lambda obj: obj['post_valid_content'][
                'best_usable_rank'] is not None),
        final_top1_recall=rate(
            lambda obj: obj['post_valid_content']['top1_hit']),
        best_usable_foreground_over_background_rate=(
            None if not fg_records else float(sum(
                row['foreground_over_background'] for row in fg_records)
                / len(fg_records))),
        best_usable_foreground_rank_percentiles=percentile(
            fg_ranks, (0, 25, 50, 75, 95, 100)))
    if include_token_bins:
        summary['source_token_bins'] = {}
        for label in ('source_small', 'source_medium', 'source_large'):
            grouped_rows = []
            for row in rows:
                grouped = [
                    obj for obj in row['objects']
                    if obj['source_token_bin'] == label]
                if grouped:
                    grouped_row = dict(objects=grouped)
                    if row.get('seq') is not None:
                        grouped_row['seq'] = row['seq']
                    if row.get('frame') is not None:
                        grouped_row['frame'] = row['frame']
                    grouped_rows.append(grouped_row)
            summary['source_token_bins'][label] = summarize_attrition(
                grouped_rows, include_token_bins=False)
    return summary


def diagnose(summary: Dict) -> str:
    if int(summary.get('object_count', 0)) == 0:
        return 'INVALID_EMPTY_GROUP'
    terminal_failure_causes = summary.get('terminal_failure_causes')
    if terminal_failure_causes is not None:
        if not terminal_failure_causes:
            return 'ROI_PIPELINE_PRESERVES_GEOMETRY_AND_ORDERING'
        primary = max(
            terminal_failure_causes.items(), key=lambda item: item[1])[0]
        if primary == 'RPN_MISS':
            return 'RPN_COVERAGE_PRIMARY_LIMIT'
        if primary == 'ROI_REGRESSION_DESTROYS_RPN_GEOMETRY':
            return 'ROI_REGRESSION_PRIMARY_ATTRITION'
        if primary == 'ROI_ORDERING_OR_NMS_REMOVES_GEOMETRY':
            return 'ROI_CLASSIFICATION_ORDERING_OR_NMS_PRIMARY_ATTRITION'
        if primary == 'VALID_CONTENT_FILTER_REMOVES_GEOMETRY':
            return 'VALID_CONTENT_PRIMARY_ATTRITION'
        return 'ROI_ORDERING_PRIMARY_ATTRITION'
    if float(summary['rpn_recall']) < 0.8:
        return 'RPN_COVERAGE_PRIMARY_LIMIT'
    if float(summary['rpn_geometry_survives_roi_regression']) < 0.8:
        return 'ROI_REGRESSION_PRIMARY_ATTRITION'
    if float(summary['post_nms_recall']) < 0.8:
        return 'ROI_CLASSIFICATION_ORDERING_OR_NMS_PRIMARY_ATTRITION'
    if float(summary['post_valid_recall']) < 0.8:
        return 'VALID_CONTENT_PRIMARY_ATTRITION'
    if float(summary['final_top1_recall']) < 0.8:
        return 'ROI_ORDERING_PRIMARY_ATTRITION'
    return 'ROI_PIPELINE_PRESERVES_GEOMETRY_AND_ORDERING'


def warmup(dino, heads, record: Dict, args, dino_devices, head_device):
    for _ in range(int(args.latency_warmup)):
        feature, img_meta, gt_boxes, labels, original, _timing = fresh_feature(
            dino, record, args, dino_devices, head_device)
        manual_rpn_roi_trace(
            heads, feature, img_meta, gt_boxes, original, args,
            timed=False, devices=list(dino_devices) + [head_device])
        del feature, gt_boxes, labels
    synchronize(list(dino_devices) + [head_device])


def main():
    args = protocol_args(parse_args())
    validate_args(args)
    labeller.set_seed(args.seed)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    coverage_payload, boundaries = load_coverage_contract(
        args.coverage_audit_json, args)

    source_records = coverage.discover_dataset_records(
        args.data_root, args.source_rpn_datasets)
    if args.source_rpn_limit:
        source_records = source_records[:args.source_rpn_limit]
    target_groups = {
        spec['name']: coverage.discover_target_records(args.data_root, spec)
        for spec in args.parsed_target_slices}
    coverage.assert_disjoint(source_records, target_groups)

    dino_devices = [torch.device('cuda:{}'.format(gpu))
                    for gpu in args.dino_gpus]
    head_device = torch.device('cuda:{}'.format(args.head_gpu))
    dino, heads = far_audit.load_frozen_labeller(
        args, dino_devices, head_device)
    dino_versions = common.module_parameter_versions(dino)
    head_versions = common.module_parameter_versions(heads)
    warmup(
        dino, heads, source_records[0], args, dino_devices, head_device)

    source_rows, source_memory = evaluate_records(
        dino, heads, source_records, args, dino_devices, head_device,
        'source_validation_roi_control', boundaries, force_fresh=False,
        fresh_latency_samples=args.source_fresh_latency_samples)
    source_summary = summarize_attrition(source_rows)
    source_latency = summarize_latency(source_rows)

    targets = {}
    force_fresh_target = args.target_feature_mode == 'fresh_fp32'
    for name, records in target_groups.items():
        rows, memory = evaluate_records(
            dino, heads, records, args, dino_devices, head_device,
            'target_diagnosis_only', boundaries,
            force_fresh=force_fresh_target,
            fresh_latency_samples=0)
        summary = summarize_attrition(rows)
        targets[name] = dict(
            specification=next(
                row for row in args.parsed_target_slices
                if row['name'] == name),
            diagnosis=diagnose(summary), summary=summary,
            latency=summarize_latency(rows), peak_memory_mib=memory,
            rows=rows)

    dino_unchanged = (
        dino_versions == common.module_parameter_versions(dino))
    heads_unchanged = (
        head_versions == common.module_parameter_versions(heads))
    if not dino_unchanged or not heads_unchanged:
        raise RuntimeError('Read-only parameter invariant failed')

    payload = dict(
        audit=AUDIT_NAME, protocol_version=PROTOCOL_VERSION,
        coverage_audit_json=os.path.abspath(args.coverage_audit_json),
        coverage_audit_sha256=common.file_sha256(args.coverage_audit_json),
        labeller_checkpoint=os.path.abspath(args.labeller_checkpoint),
        labeller_checkpoint_sha256=common.file_sha256(
            args.labeller_checkpoint),
        dinov2_checkpoint=os.path.abspath(args.dinov2_checkpoint),
        dinov2_checkpoint_sha256=common.file_sha256(
            args.dinov2_checkpoint),
        protocol=dict(
            target_role='diagnosis_only',
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            coverage_reuse_contract=coverage_payload['_reuse_contract'],
            coverage_reuse_scope=(
                'source_defined_token_bins_only; current RPN, ROI '
                'regression, classification, NMS, and valid filtering are '
                'recomputed'),
            source_defined_token_bins=boundaries,
            source_feature_mode=(
                'training_fp16_cache_except_explicit_fresh_latency_samples'),
            target_feature_mode=args.target_feature_mode,
            latency_scope=(
                'DINO_branch_only_excludes_BrightAug_and_includes_'
                'preprocess_DINO_RPN_ROI_NMS_valid_filter'),
            riou_thr=float(args.riou_thr),
            recall_ks=list(args.recall_ks),
            reconstruction_atol=1e-4),
        isolation=dict(
            optimizer_steps=0, checkpoint_writes=0,
            dino_frozen=True, dino_parameters_unchanged=dino_unchanged,
            labeller_heads_frozen=True,
            labeller_parameters_unchanged=heads_unchanged,
            target_labels_used_for_evaluation_only=True),
        parameter_counts=coverage.parameter_counts(dino, heads),
        source_roi_control=dict(
            diagnosis=diagnose(source_summary), summary=source_summary,
            latency=source_latency, peak_memory_mib=source_memory,
            rows=source_rows),
        target_diagnoses=targets)
    replacements = common.write_json_atomic(args.out_json, payload)
    print('[attrition] source {}'.format(diagnose(source_summary)))
    for name, result in targets.items():
        summary = result['summary']
        total_latency = result['latency']['dino_branch_total_ms']
        print('[attrition] {} {} rpn={:.3f} reg={:.3f} nms={:.3f} '
              'valid={:.3f} top1={}/{} mcml={} latency_p50_ms={}'.format(
                  name, result['diagnosis'], summary['rpn_recall'],
                  summary['rpn_geometry_survives_roi_regression'],
                  summary['post_nms_recall'],
                  summary['post_valid_recall'],
                  summary['final_top1_hits'], summary['frame_count'],
                  summary['final_top1_mcml'],
                  total_latency['percentiles']['50']))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
