#!/usr/bin/env python3
"""Audit the frozen Base-V3 history contribution and key-frame geometry."""

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from crane_project.tools.eval_crane_offline import parse_dota_txt
from crane_project.tools.symeood_dino_seq11_v2_baseline_compare import (
    _audit_boxes, _evaluate, _identity, _json, _load_result_boxes, _segments,
    _sha256, _write_exact)


PROTOCOL = 'symeood_dino_seq11_v2_history_contribution_audit_v1'
CONTRACT_PROTOCOL = 'real_seq11_base_v3_history_contribution_ablation_v1'
THREE_WAY_PROTOCOL = 'symeood_dino_seq11_v2_three_way_baseline_comparison_v1'
RECEIPT_PROTOCOL = 'mmdet_runtime_result_order_identity_v1'
RUNTIME_PROTOCOL = 'mmdet_runtime_inference_resource_audit_v1'
CURRENT_ONLY_PROTOCOL = (
    'base_v3_epoch9_seq11_v2_full251_current_only_inference_v1')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--three-way-report', required=True)
    parser.add_argument('--ablation-contract', required=True)
    parser.add_argument('--current-only-results', required=True)
    parser.add_argument('--current-only-result-receipt', required=True)
    parser.add_argument('--current-only-runtime-audit', required=True)
    parser.add_argument('--current-only-config', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument(
        '--source-split', default='extra_source_real_seq11_pilot_k1p9_v2')
    parser.add_argument('--overlay-dir', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--top-history-delta-frames', type=int, default=12)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--center-threshold-px', type=float, default=25.0)
    parser.add_argument('--angle-limit-deg', type=float, default=35.0)
    parser.add_argument('--high-confidence-threshold', type=float, default=0.5)
    return parser.parse_args()


def _verify_recorded_identity(item, role):
    if not isinstance(item, dict) or not item.get('path') or not item.get('sha256'):
        raise RuntimeError('Missing recorded identity for ' + role)
    observed = _identity(item['path'])
    if observed['sha256'] != item['sha256']:
        raise RuntimeError('Recorded identity changed for ' + role)
    return observed


def _metric_delta(full, current):
    full_temporal = full['temporal']
    current_temporal = current['temporal']
    return dict(
        hit_count=full['hit_count'] - current['hit_count'],
        rescued=full['rescue']['rescued'] - current['rescue']['rescued'],
        old_correct_lost=(full['old_correct_lost']
                          - current['old_correct_lost']),
        mean_RIoU=full['mean_RIoU'] - current['mean_RIoU'],
        valid_pair_fraction=(full_temporal['valid_pair_fraction']
                             - current_temporal['valid_pair_fraction']),
        DFR_percent_per_frame=(
            None if full_temporal['DFR_percent_per_frame'] is None
            or current_temporal['DFR_percent_per_frame'] is None else
            full_temporal['DFR_percent_per_frame']
            - current_temporal['DFR_percent_per_frame']),
        ACI=(None if full_temporal['ACI'] is None
             or current_temporal['ACI'] is None else
             full_temporal['ACI'] - current_temporal['ACI']),
        MCML_max=full_temporal['MCML_max'] - current_temporal['MCML_max'],
        both_bad_false_output_count=(
            full['both_bad']['false_output_count']
            - current['both_bad']['false_output_count']))


def _angle_error_deg(box, gt):
    if box is None:
        return None
    delta = float(box[4]) - float(gt[4])
    delta = math.atan2(math.sin(delta), math.cos(delta))
    if delta > math.pi / 2:
        delta -= math.pi
    if delta < -math.pi / 2:
        delta += math.pi
    return abs(math.degrees(delta))


def _geometry(box, gt, per_frame):
    if box is None:
        return dict(present=False, score=per_frame.get('score'),
                    riou=0.0, center_error_px=None,
                    diagonal_ratio_to_gt=None, angle_error_deg=None)
    gt_diag = float(np.linalg.norm(gt[2:4]))
    return dict(
        present=True, score=per_frame.get('score'),
        riou=float(per_frame['riou']),
        center_error_px=float(np.linalg.norm(box[:2] - gt[:2])),
        diagonal_ratio_to_gt=float(np.linalg.norm(box[2:4]) / gt_diag),
        angle_error_deg=_angle_error_deg(box, gt))


def _polygon(box):
    if box is None:
        return None
    rect = ((float(box[0]), float(box[1])),
            (float(box[2]), float(box[3])),
            math.degrees(float(box[4])))
    return np.rint(cv2.boxPoints(rect)).astype(np.int32)


def _draw_overlay(image, boxes, labels):
    colors = dict(
        GT=(0, 220, 0), K1=(255, 255, 0), DINO=(0, 165, 255),
        deterministic=(255, 0, 255), current_only=(0, 255, 255),
        full=(0, 0, 255))
    canvas = image.copy()
    for label in labels:
        polygon = _polygon(boxes.get(label))
        if polygon is not None:
            cv2.polylines(canvas, [polygon], True, colors[label], 2,
                          lineType=cv2.LINE_AA)
    y = 24
    for label in labels:
        state = 'present' if boxes.get(label) is not None else 'missing'
        cv2.putText(canvas, '{}: {}'.format(label, state), (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, colors[label], 2,
                    cv2.LINE_AA)
        y += 21
    return canvas


def audit(args):
    contract_id, contract = _json(
        args.ablation_contract, CONTRACT_PROTOCOL)
    if (contract.get('status') !=
            'preregistered_before_current_only_inference'
            or contract.get('three_fold_training_authorized') is not False):
        raise RuntimeError('Invalid history ablation contract')
    report_id, three_way = _json(args.three_way_report, THREE_WAY_PROTOCOL)
    if (three_way.get('training_run') is not False
            or three_way.get('eligible_for_three_fold_training') is not False):
        raise RuntimeError('Three-way input has an invalid evidence role')
    recorded = three_way.get('inputs') or {}
    verified_inputs = {}
    for role in (
            'frame_order_manifest', 'formal_k1_identity', 'k1_results',
            'dino_cache_identity', 'dino_all_lane_audit',
            'formal_support_audit', 'history_dependency_audit',
            'base_v3_config', 'base_v3_checkpoint', 'base_v3_promotion',
            'base_v3_results', 'base_v3_result_receipt',
            'base_v3_runtime_audit', 'simple_fallback_results'):
        verified_inputs[role] = _verify_recorded_identity(recorded.get(role), role)

    with open(verified_inputs['frame_order_manifest']['path'], 'r',
              encoding='utf-8') as handle:
        frame_manifest = json.load(handle)
    frame_rows = frame_manifest.get('frames') or []
    frame_keys = [row['frame_key'] for row in frame_rows]
    frames = [int(row['frame']) for row in frame_rows]
    if len(frame_keys) != 251 or len(set(frame_keys)) != 251:
        raise RuntimeError('Frame-order manifest must bind 251 unique frames')
    with open(verified_inputs['formal_support_audit']['path'], 'r',
              encoding='utf-8') as handle:
        support = json.load(handle)
    support_rows = support.get('rows') or []
    if [row.get('frame_key') for row in support_rows] != frame_keys:
        raise RuntimeError('Support rows differ from runtime frame order')
    with open(verified_inputs['history_dependency_audit']['path'], 'r',
              encoding='utf-8') as handle:
        history = json.load(handle)
    history_counts = Counter(
        edge['current'] for edge in history.get('history_dependency_edges') or [])

    full_id, full_boxes, full_scores = _load_result_boxes(
        verified_inputs['base_v3_results']['path'])
    simple_id, simple_boxes, simple_scores = _load_result_boxes(
        verified_inputs['simple_fallback_results']['path'])
    k1_id, k1_boxes, k1_scores = _load_result_boxes(
        verified_inputs['k1_results']['path'])
    dino_id, dino_boxes, dino_scores = _audit_boxes(
        verified_inputs['dino_all_lane_audit']['path'], frame_keys)
    current_id, current_boxes, current_scores = _load_result_boxes(
        args.current_only_results)

    current_config_id = _identity(args.current_only_config)
    config_text = Path(current_config_id['path']).read_text(encoding='utf-8')
    required_config_tokens = (
        "inference_component_mode='current_only'",
        'history_output_contribution=False',
        'same_setting_all_frames=True',
        'domain_routing=False', 'sequence_frame_routing=False',
        'optimizer_steps=0', 'fixed_test_read=False')
    if not all(token in config_text for token in required_config_tokens):
        raise RuntimeError('Current-only config contract is incomplete')
    receipt_id, receipt = _json(
        args.current_only_result_receipt, RECEIPT_PROTOCOL)
    receipt_order = [
        row.get('frame_key') for row in receipt.get('runtime_dataset_order') or []]
    receipt_contract = receipt.get('evidence_contract') or {}
    checkpoint_sha = verified_inputs['base_v3_checkpoint']['sha256']
    if (receipt.get('result_count') != 251
            or receipt_order != frame_keys
            or receipt.get('results_sha256') != current_id['sha256']
            or receipt.get('checkpoint_sha256') != checkpoint_sha
            or receipt.get('config_sha256') != current_config_id['sha256']
            or receipt_contract.get('protocol') != CURRENT_ONLY_PROTOCOL
            or receipt_contract.get('history_output_contribution') is not False
            or receipt_contract.get('same_setting_all_frames') is not True):
        raise RuntimeError('Current-only result receipt mismatch')
    runtime_id, runtime = _json(
        args.current_only_runtime_audit, RUNTIME_PROTOCOL)
    runtime_dino = dict(runtime.get('runtime_input_files') or {}).get(
        'dino_all_lane_audit') or {}
    if (runtime.get('result_count') != 251
            or runtime.get('checkpoint_sha256') != checkpoint_sha
            or runtime.get('config_sha256') != current_config_id['sha256']
            or runtime_dino.get('sha256') != dino_id['sha256']
            or (runtime.get('forward_counts') or {}).get(
                'inference_forward') != 251
            or (runtime.get('forward_counts') or {}).get(
                'dino_detector_forward') != 0
            or (runtime.get('cuda') or {}).get('cuda_visible_devices') != '0'):
        raise RuntimeError('Current-only runtime audit mismatch')

    ann_root = Path(args.data_root).resolve() / args.source_split / 'annfiles'
    image_root = Path(args.data_root).resolve() / args.source_split / 'images'
    gt_boxes = []
    for row in frame_rows:
        annotation = ann_root / row['annotation_filename']
        if not annotation.is_file() or _sha256(annotation) != row[
                'annotation_sha256']:
            raise RuntimeError('Annotation identity mismatch: ' + str(annotation))
        parsed = parse_dota_txt(os.fspath(annotation))
        if len(parsed) != 1:
            raise RuntimeError('Every frame must contain one GT OBB')
        gt_boxes.append(np.asarray(parsed[0], dtype=np.float64)[:5])

    evaluation_kwargs = dict(
        gt_boxes=gt_boxes, frames=frames, segments=_segments(frames),
        category_rows=support_rows, iou_threshold=args.iou_threshold,
        center_threshold=args.center_threshold_px,
        angle_limit_deg=args.angle_limit_deg,
        high_confidence_threshold=args.high_confidence_threshold)
    current_metric = _evaluate(
        'base_v3_epoch9_current_only', current_boxes, current_scores,
        **evaluation_kwargs)
    full_metric = three_way['metrics']['base_v3_epoch9_refiner']
    simple_metric = three_way['metrics']['deterministic_k1_else_dino']
    full_by_key = {row['frame_key']: row for row in full_metric['per_frame']}
    simple_by_key = {row['frame_key']: row for row in simple_metric['per_frame']}
    current_by_key = {
        row['frame_key']: row for row in current_metric['per_frame']}

    no_history_equality = []
    max_no_history_box_delta = 0.0
    for index, key in enumerate(frame_keys):
        if history_counts[key] != 0:
            continue
        first, second = full_boxes[index], current_boxes[index]
        if first is None or second is None:
            equal = first is None and second is None
            delta = None
        else:
            delta = float(np.max(np.abs(first - second)))
            max_no_history_box_delta = max(max_no_history_box_delta, delta)
            equal = delta <= 1e-6
        if not equal:
            no_history_equality.append(dict(frame_key=key, max_box_delta=delta))

    reasons = defaultdict(list)
    for key in frame_keys:
        if simple_by_key[key]['hit'] != full_by_key[key]['hit']:
            reasons[key].append('deterministic_fallback_vs_full_hit_change')
        if current_by_key[key]['hit'] != full_by_key[key]['hit']:
            reasons[key].append('current_only_vs_full_hit_change')
    ranked = sorted(
        frame_keys,
        key=lambda key: abs(
            full_by_key[key]['riou'] - current_by_key[key]['riou']),
        reverse=True)
    for key in ranked[:max(0, args.top_history_delta_frames)]:
        reasons[key].append('top_absolute_history_riou_delta')
    overlay_root = Path(args.overlay_dir).resolve()
    overlay_root.mkdir(parents=True, exist_ok=True)
    geometry_rows = []
    labels = ('GT', 'K1', 'DINO', 'deterministic', 'current_only', 'full')
    for index, row in enumerate(frame_rows):
        key = row['frame_key']
        if key not in reasons:
            continue
        box_map = dict(
            GT=gt_boxes[index], K1=k1_boxes[index], DINO=dino_boxes[index],
            deterministic=simple_boxes[index],
            current_only=current_boxes[index], full=full_boxes[index])
        per_frame_map = dict(
            K1=dict(score=k1_scores[index], riou=(
                0.0 if k1_boxes[index] is None else
                float(three_way['metrics']['ordinary_k1']['per_frame'][index][
                    'riou']))),
            DINO=dict(score=dino_scores[index], riou=(
                float(support_rows[index].get('dino_riou', 0.0)))),
            deterministic=simple_by_key[key],
            current_only=current_by_key[key], full=full_by_key[key])
        methods = {
            name: _geometry(box_map[name], gt_boxes[index], per_frame_map[name])
            for name in ('K1', 'DINO', 'deterministic', 'current_only', 'full')}
        image_path = image_root / row['image_filename']
        image = cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Cannot read key-frame image: ' + str(image_path))
        output_path = overlay_root / (key + '_geometry_overlay.png')
        if output_path.exists():
            raise RuntimeError('Refusing to overwrite overlay: ' + str(output_path))
        if not cv2.imwrite(
                os.fspath(output_path), _draw_overlay(image, box_map, labels)):
            raise RuntimeError('Failed to write overlay: ' + str(output_path))
        geometry_rows.append(dict(
            frame_key=key, frame=int(row['frame']),
            category=support_rows[index]['category'],
            valid_history_count=int(history_counts[key]),
            selection_reasons=reasons[key], methods=methods,
            full_minus_current_only_riou=(
                full_by_key[key]['riou'] - current_by_key[key]['riou']),
            overlay=dict(path=os.fspath(output_path),
                         sha256=_sha256(output_path))))

    history_changed = [
        row for row in geometry_rows
        if 'current_only_vs_full_hit_change' in row['selection_reasons']]
    return dict(
        protocol=PROTOCOL,
        evidence_boundary='seq11_same_video_source_only_component_ablation',
        inputs=dict(
            ablation_contract=contract_id, three_way_report=report_id,
            verified_three_way_inputs=verified_inputs,
            current_only_config=current_config_id,
            current_only_results=current_id,
            current_only_result_receipt=receipt_id,
            current_only_runtime_audit=runtime_id,
            full_results=full_id, deterministic_results=simple_id,
            k1_results=k1_id, dino_audit=dino_id),
        design=dict(
            checkpoint_sha256=checkpoint_sha,
            same_checkpoint=True, same_frame_order=True,
            current_only_all_frames=True,
            history_tensors_loaded=True,
            history_output_contribution=False,
            domain_routing=False, sequence_frame_routing=False),
        metrics=dict(
            deterministic_k1_else_dino=three_way['metrics'][
                'deterministic_k1_else_dino'],
            base_v3_epoch9_current_only=current_metric,
            base_v3_epoch9_full=full_metric),
        full_minus_current_only=_metric_delta(full_metric, current_metric),
        no_valid_history_invariance=dict(
            frame_count=sum(history_counts[key] == 0 for key in frame_keys),
            tolerance=1e-6,
            max_box_delta=max_no_history_box_delta,
            violation_count=len(no_history_equality),
            violations=no_history_equality),
        key_frame_summary=dict(
            selected_frame_count=len(geometry_rows),
            history_changed_hit_count=len(history_changed),
            history_changed_hit_frames=[row['frame_key'] for row in history_changed],
            rows=geometry_rows),
        interpretation_policy=dict(
            history_claim_requires_paired_improvement=True,
            no_single_metric_promotion=True,
            no_automatic_cv_training_authorization=True),
        target_data_read=False, fixed_test_read=False,
        training_run=False, eligible_for_checkpoint_promotion=False,
        eligible_for_three_fold_training=False,
        decision='HISTORY_CONTRIBUTION_AUDIT_READY_FOR_REVIEW_NO_TRAINING_AUTHORIZED')


def main():
    args = parse_args()
    report = audit(args)
    raw = (json.dumps(report, indent=2, ensure_ascii=False) + '\n').encode(
        'utf-8')
    output, _ = _write_exact(args.out_json, raw)
    delta = report['full_minus_current_only']
    print('[seq11-history-ablation] output={}'.format(output))
    print('[seq11-history-ablation] decision={}'.format(report['decision']))
    print('[seq11-history-ablation] full-current_only: hits={hit_count} '
          'rescue={rescued} old_lost={old_correct_lost} RIoU={mean_RIoU} '
          'DFR={DFR_percent_per_frame} ACI={ACI} MCML={MCML_max}'.format(
              **delta))
    print('[seq11-history-ablation] no-valid-history violations={}'.format(
        report['no_valid_history_invariance']['violation_count']))
    print('[seq11-history-ablation] selected_overlays={}'.format(
        report['key_frame_summary']['selected_frame_count']))


if __name__ == '__main__':
    main()
