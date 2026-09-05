#!/usr/bin/env python3
"""Compare K1, deterministic DINO fallback, and frozen Base V3 on seq11-v2.

This source-only diagnostic is deliberately separate from the block-CV gate.
It creates no training authorization and never reads the fixed TEST split.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np

from crane_project.tools.eval_crane_offline import compute_riou, parse_dota_txt


PROTOCOL = 'symeood_dino_seq11_v2_three_way_baseline_comparison_v1'
FRAME_PROTOCOL = 'seq11_v2_full251_frame_order_manifest_v2'
K1_IDENTITY_PROTOCOL = 'formal_k1_seq11_v2_full251_identity_v2'
DINO_IDENTITY_PROTOCOL = 'seq11_v2_dino_cache_identity_v1'
HISTORY_PROTOCOL = 'seq11_v2_history_dependency_audit_v1'
SUPPORT_PROTOCOL = 'k1_dino_seq11_event_block_cv_support_audit_v2'
RESULT_RECEIPT_PROTOCOL = 'mmdet_runtime_result_order_identity_v1'
COMPARISON_CONTRACT_PROTOCOL = (
    'real_seq11_k1_dino_base_v3_three_way_baseline_v1')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument(
        '--source-split', default='extra_source_real_seq11_pilot_k1p9_v2')
    parser.add_argument('--k1-results', required=True)
    parser.add_argument('--formal-k1-identity', required=True)
    parser.add_argument('--frame-order-json', required=True)
    parser.add_argument('--dino-audit-json', required=True)
    parser.add_argument('--dino-cache-identity', required=True)
    parser.add_argument('--support-audit', required=True)
    parser.add_argument('--history-dependency-audit', required=True)
    parser.add_argument('--base-v3-results', required=True)
    parser.add_argument('--base-v3-result-receipt', required=True)
    parser.add_argument('--base-v3-runtime-audit', required=True)
    parser.add_argument('--base-v3-config', required=True)
    parser.add_argument('--base-v3-checkpoint', required=True)
    parser.add_argument('--base-v3-promotion', required=True)
    parser.add_argument('--comparison-contract', required=True)
    parser.add_argument('--simple-fallback-results', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--center-threshold-px', type=float, default=25.0)
    parser.add_argument('--angle-limit-deg', type=float, default=35.0)
    parser.add_argument('--high-confidence-threshold', type=float, default=0.5)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path):
    absolute = Path(path).resolve()
    if not absolute.is_file():
        raise RuntimeError('Missing required input: ' + os.fspath(absolute))
    return dict(path=os.fspath(absolute), sha256=_sha256(absolute),
                size_bytes=absolute.stat().st_size)


def _json(path, protocol=None, require_unread_flags=True):
    identity = _identity(path)
    with open(identity['path'], 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if protocol is not None and payload.get('protocol') != protocol:
        raise RuntimeError(
            '{} has protocol {!r}, expected {!r}'.format(
                identity['path'], payload.get('protocol'), protocol))
    if require_unread_flags and payload.get('target_data_read') is not False:
        raise RuntimeError('Target data was read: ' + identity['path'])
    if require_unread_flags and payload.get('fixed_test_read') is not False:
        raise RuntimeError(
            'Input does not prove fixed_test_read=false: ' + identity['path'])
    return identity, payload


def _load_result_boxes(path, expected_count=251):
    identity = _identity(path)
    with open(identity['path'], 'rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise RuntimeError('Result PKL must contain exactly 251 rows')
    boxes, scores = [], []
    for index, result in enumerate(payload):
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise RuntimeError('Result row {} must contain one class'.format(index))
        detections = np.asarray(result[0], dtype=np.float64)
        if detections.size == 0:
            boxes.append(None)
            scores.append(None)
            continue
        detections = detections.reshape((-1, 6))
        if detections.shape[0] != 1:
            raise RuntimeError(
                'Result row {} must contain at most one OBB'.format(index))
        if (not np.isfinite(detections).all()
                or np.any(detections[0, 2:4] <= 0.0)):
            raise RuntimeError('Invalid OBB at result row {}'.format(index))
        boxes.append(detections[0, :5].copy())
        scores.append(float(detections[0, 5]))
    return identity, boxes, scores


def _audit_boxes(path, frame_keys):
    identity, payload = _json(path, require_unread_flags=False)
    if payload.get('protocol') != 'source_owned_geometry_union_v2':
        raise RuntimeError('Input is not a complete all-lane audit')
    records = payload.get('records') or []
    if len(records) != 251:
        raise RuntimeError('DINO all-lane audit must contain 251 records')
    by_key = {}
    for record in records:
        key = Path(record.get('filename', '')).stem
        if not key or key in by_key:
            raise RuntimeError('Invalid or duplicate DINO audit frame')
        if (record.get('raw_selected_source') == 'not_computed'
                or ('dino_invoked' in record
                    and record.get('dino_invoked') not in (True, 1))):
            raise RuntimeError('DINO not computed for ' + key)
        value = record.get('dino_native_box')
        if value is None:
            box, score = None, None
        else:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            if (array.size < 5 or not np.isfinite(array).all()
                    or np.any(array[2:4] <= 0.0)):
                raise RuntimeError('Invalid DINO OBB for ' + key)
            box = array[:5].copy()
            score = float(array[5]) if array.size >= 6 else None
        by_key[key] = (box, score)
    if set(by_key) != set(frame_keys):
        raise RuntimeError('DINO audit and frame-order sets differ')
    return identity, [by_key[key][0] for key in frame_keys], [
        by_key[key][1] for key in frame_keys]


def _angle_diff(first, second):
    delta = float(first) - float(second)
    delta = math.atan2(math.sin(delta), math.cos(delta))
    if delta > math.pi / 2:
        delta -= math.pi
    if delta < -math.pi / 2:
        delta += math.pi
    return delta


def _segments(frames):
    segments, current = [], []
    for index, frame in enumerate(frames):
        if current and frame != frames[current[-1]] + 1:
            segments.append(current)
            current = []
        current.append(index)
    if current:
        segments.append(current)
    return segments


def _failure_runs(flags):
    runs, start = [], None
    for index, hit in enumerate(flags):
        if not hit and start is None:
            start = index
        elif hit and start is not None:
            runs.append((start, index - 1, index - start, True))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1, len(flags) - start, False))
    return runs


def _evaluate(name, boxes, scores, gt_boxes, frames, segments,
              category_rows, iou_threshold, center_threshold, angle_limit_deg,
              high_confidence_threshold):
    rious, hits, center_hits = [], [], []
    for box, gt in zip(boxes, gt_boxes):
        riou = 0.0 if box is None else float(compute_riou(box, gt))
        rious.append(riou)
        hits.append(riou >= iou_threshold)
        center_hits.append(
            False if box is None else
            float(np.linalg.norm(box[:2] - gt[:2])) < center_threshold)

    dfr_sum = aci_sum = 0.0
    dfr_count = aci_count = valid_pairs = contiguous_pairs = 0
    segment_mcml, all_runs, terminal_runs, block_rows = [], [], [], []
    angle_limit = math.radians(angle_limit_deg)
    for block_id, indices in enumerate(segments, 1):
        contiguous_pairs += max(0, len(indices) - 1)
        block_dfr_sum = block_aci_sum = 0.0
        block_valid = 0
        for previous, current in zip(indices[:-1], indices[1:]):
            if frames[current] != frames[previous] + 1:
                raise RuntimeError('Internal metric block is not contiguous')
            if boxes[previous] is None or boxes[current] is None:
                continue
            block_valid += 1
            previous_diag = float(np.linalg.norm(boxes[previous][2:4]))
            current_diag = float(np.linalg.norm(boxes[current][2:4]))
            block_dfr_sum += abs(current_diag - previous_diag) / previous_diag
            delta = abs(_angle_diff(boxes[current][4], boxes[previous][4]))
            block_aci_sum += float(np.clip(
                1.0 - delta / (angle_limit + 1e-9), 0.0, 1.0))
        valid_pairs += block_valid
        dfr_count += block_valid
        aci_count += block_valid
        dfr_sum += block_dfr_sum
        aci_sum += block_aci_sum
        block_hits = [hits[index] for index in indices]
        runs = _failure_runs(block_hits)
        maximum = max([run[2] for run in runs] or [0])
        segment_mcml.append(maximum)
        all_runs.extend(run[2] for run in runs)
        terminal = runs[-1][2] if runs and not runs[-1][3] else 0
        terminal_runs.append(terminal)
        block_rows.append(dict(
            block_id=block_id, start_frame=frames[indices[0]],
            end_frame=frames[indices[-1]], frame_count=len(indices),
            contiguous_gt_pair_count=max(0, len(indices) - 1),
            valid_prediction_pair_count=block_valid,
            dfr_valid_pair_count=block_valid,
            aci_valid_pair_count=block_valid,
            valid_pair_fraction=(block_valid / max(1, len(indices) - 1)),
            missing_prediction_count=sum(boxes[index] is None for index in indices),
            DFR_percent_per_frame=(
                None if block_valid == 0 else 100.0 * block_dfr_sum / block_valid),
            ACI=(None if block_valid == 0 else block_aci_sum / block_valid),
            MCML_max=maximum,
            unrecovered_terminal_run_length=terminal))

    categories = {row['frame_key']: row['category'] for row in category_rows}
    rescue_indices = [
        index for index, row in enumerate(category_rows)
        if row['category'] in (
            'k1_missing_dino_hit', 'k1_present_wrong_dino_hit')]
    missing_rescue_indices = [
        index for index, row in enumerate(category_rows)
        if row['category'] == 'k1_missing_dino_hit']
    old_correct_indices = [
        index for index, row in enumerate(category_rows)
        if row['category'] == 'k1_hit']
    both_bad_indices = [
        index for index, row in enumerate(category_rows)
        if row['category'] == 'both_bad']
    false_output = [
        index for index in both_bad_indices
        if boxes[index] is not None and not hits[index]]
    false_runs = []
    current = []
    false_set = set(false_output)
    for index in both_bad_indices:
        if index not in false_set:
            if current:
                false_runs.append(current)
                current = []
            continue
        if current and frames[index] != frames[current[-1]] + 1:
            false_runs.append(current)
            current = []
        current.append(index)
    if current:
        false_runs.append(current)

    return dict(
        method=name,
        frame_count=len(boxes),
        prediction_count=sum(box is not None for box in boxes),
        missing_prediction_count=sum(box is None for box in boxes),
        hit_count=sum(hits),
        mean_RIoU=float(np.mean(rious)),
        R_center_percent=100.0 * float(np.mean(center_hits)),
        rescue=dict(
            denominator=len(rescue_indices),
            rescued=sum(hits[index] for index in rescue_indices),
            rate=(sum(hits[index] for index in rescue_indices)
                  / max(1, len(rescue_indices))),
            both_bad_excluded=True),
        missing_rescue=dict(
            denominator=len(missing_rescue_indices),
            rescued=sum(hits[index] for index in missing_rescue_indices),
            rate=(sum(hits[index] for index in missing_rescue_indices)
                  / max(1, len(missing_rescue_indices)))),
        old_correct_lost=sum(not hits[index] for index in old_correct_indices),
        temporal=dict(
            block_count=len(segments),
            contiguous_gt_pair_count=contiguous_pairs,
            valid_prediction_pair_count=valid_pairs,
            dfr_valid_pair_count=dfr_count,
            aci_valid_pair_count=aci_count,
            valid_pair_fraction=valid_pairs / max(1, contiguous_pairs),
            DFR_percent_per_frame=(
                None if dfr_count == 0 else 100.0 * dfr_sum / dfr_count),
            ACI=None if aci_count == 0 else aci_sum / aci_count,
            MCML_max=max(segment_mcml or [0]),
            MCML_segment_max_mean=float(np.mean(segment_mcml or [0])),
            failure_run_length_mean=float(np.mean(all_runs or [0])),
            failure_run_count=len(all_runs),
            unrecovered_terminal_run_length=max(terminal_runs or [0]),
            aggregation='sum_block_numerators_and_denominators_once',
            blocks=block_rows),
        both_bad=dict(
            frame_count=len(both_bad_indices),
            continuous_event_count=len(_segments(
                [frames[index] for index in both_bad_indices])),
            false_output_count=len(false_output),
            high_confidence_false_output_count=sum(
                scores[index] is not None
                and scores[index] >= high_confidence_threshold
                for index in false_output),
            confidence_unavailable_false_output_count=sum(
                scores[index] is None for index in false_output),
            longest_false_output_run=max(
                [len(run) for run in false_runs] or [0]),
            unavailable_count=sum(boxes[index] is None
                                  for index in both_bad_indices),
            abstention_count=sum(boxes[index] is None
                                 for index in both_bad_indices)),
        per_frame=[dict(
            frame_key=row['frame_key'], frame=frames[index],
            category=categories[row['frame_key']],
            prediction_present=boxes[index] is not None,
            score=scores[index], riou=rious[index], hit=hits[index])
            for index, row in enumerate(category_rows)])


def _write_exact(path, raw):
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != raw:
        raise RuntimeError('Refusing to overwrite different output: '
                           + os.fspath(output))
    if not output.exists():
        output.write_bytes(raw)
    return os.fspath(output), hashlib.sha256(raw).hexdigest()


def compare(args):
    comparison_contract_id, comparison_contract = _json(
        args.comparison_contract, COMPARISON_CONTRACT_PROTOCOL)
    if (comparison_contract.get('status') !=
            'preregistered_before_base_v3_full251_inference'
            or comparison_contract.get('three_fold_training_authorized')
            is not False):
        raise RuntimeError('Invalid three-way comparison contract')
    frame_id, frame_manifest = _json(args.frame_order_json, FRAME_PROTOCOL)
    frame_rows = frame_manifest.get('frames') or []
    if len(frame_rows) != 251:
        raise RuntimeError('Frame-order manifest must contain 251 rows')
    frame_keys = [row['frame_key'] for row in frame_rows]
    frames = [int(row['frame']) for row in frame_rows]
    if len(set(frame_keys)) != 251:
        raise RuntimeError('Frame-order manifest contains duplicates')

    k1_id_file, k1_identity = _json(
        args.formal_k1_identity, K1_IDENTITY_PROTOCOL)
    if (k1_identity.get('passed') is not True
            or k1_identity['inputs'].get('frame_order_manifest_sha256')
            != frame_id['sha256']):
        raise RuntimeError('Formal K1 identity is not bound to frame order')
    k1_result_id, k1_boxes, k1_scores = _load_result_boxes(args.k1_results)
    if k1_result_id['sha256'] != k1_identity['inputs'].get('results_sha256'):
        raise RuntimeError('K1 result hash differs from formal identity')

    dino_id_file, dino_identity = _json(
        args.dino_cache_identity, DINO_IDENTITY_PROTOCOL)
    if dino_identity.get('passed') is not True:
        raise RuntimeError('DINO cache identity did not pass')
    dino_audit_id, dino_boxes, dino_scores = _audit_boxes(
        args.dino_audit_json, frame_keys)
    if dino_audit_id['sha256'] != dino_identity['inputs'][
            'audit_json']['sha256']:
        raise RuntimeError('DINO audit hash differs from cache identity')

    support_id, support = _json(args.support_audit, SUPPORT_PROTOCOL)
    support_rows = support.get('rows') or []
    if ([row.get('frame_key') for row in support_rows] != frame_keys
            or support.get('category_counts') != {
                'k1_hit': 205, 'k1_missing_dino_hit': 32,
                'k1_present_wrong_dino_hit': 0, 'both_bad': 14}):
        raise RuntimeError('Support audit is not the formal 205/32/0/14 set')

    history_id, history = _json(
        args.history_dependency_audit, HISTORY_PROTOCOL)
    if (history['inputs'].get('audit_sha256') != dino_audit_id['sha256']
            or history['inputs'].get('frame_order_sha256') != frame_id['sha256']):
        raise RuntimeError('History dependency audit identity mismatch')
    derived_pair_edges = [
        (frame_keys[index - 1], frame_keys[index])
        for index in range(1, len(frames))
        if frames[index] == frames[index - 1] + 1]
    declared_pair_edges = [
        (row.get('previous'), row.get('current'))
        for row in history.get('metric_pair_candidates') or []]
    if declared_pair_edges != derived_pair_edges:
        raise RuntimeError('Metric pair edges differ from the frozen audit')

    base_config_id = _identity(args.base_v3_config)
    base_checkpoint_id = _identity(args.base_v3_checkpoint)
    promotion_id, promotion = _json(args.base_v3_promotion)
    if (promotion.get('decision') !=
            'ALLOW_K1_RETENTIVE_CAUSAL_PHASE_FIXED_BENCHMARK_TEST'
            or promotion.get('output', {}).get('checkpoint_sha256') !=
            base_checkpoint_id['sha256']):
        raise RuntimeError('Base V3 checkpoint is not the promoted epoch9')
    base_result_id, base_boxes, base_scores = _load_result_boxes(
        args.base_v3_results)
    base_receipt_id, base_receipt = _json(
        args.base_v3_result_receipt, RESULT_RECEIPT_PROTOCOL)
    receipt_order = [
        row.get('frame_key') for row in
        base_receipt.get('runtime_dataset_order') or []]
    contract = base_receipt.get('evidence_contract') or {}
    if (base_receipt.get('result_count') != 251
            or receipt_order != frame_keys
            or base_receipt.get('results_sha256') != base_result_id['sha256']
            or base_receipt.get('checkpoint_sha256') !=
            base_checkpoint_id['sha256']
            or base_receipt.get('config_sha256') != base_config_id['sha256']
            or contract.get('protocol') !=
            'base_v3_epoch9_seq11_v2_full251_inference_v1'):
        raise RuntimeError('Base V3 runtime result receipt mismatch')
    runtime_id, runtime = _json(
        args.base_v3_runtime_audit,
        'mmdet_runtime_inference_resource_audit_v1')
    forward_counts = runtime.get('forward_counts') or {}
    cuda = runtime.get('cuda') or {}
    runtime_dino = dict(runtime.get('runtime_input_files') or {}).get(
        'dino_all_lane_audit') or {}
    if (runtime.get('result_count') != 251
            or runtime.get('checkpoint_sha256') !=
            base_checkpoint_id['sha256']
            or runtime.get('config_sha256') != base_config_id['sha256']
            or runtime_dino.get('sha256') != dino_audit_id['sha256']
            or forward_counts.get('inference_forward') != 251
            or cuda.get('available') is not True
            or int(cuda.get('peak_allocated_bytes', 0)) <= 0
            or int(cuda.get('peak_reserved_bytes', 0)) <= 0):
        raise RuntimeError('Base V3 runtime resource audit mismatch')

    ann_root = Path(args.data_root).resolve() / args.source_split / 'annfiles'
    gt_boxes = []
    for row in frame_rows:
        path = ann_root / row['annotation_filename']
        if not path.is_file() or _sha256(path) != row['annotation_sha256']:
            raise RuntimeError('GT annotation identity mismatch: ' + str(path))
        parsed = parse_dota_txt(os.fspath(path))
        if len(parsed) != 1:
            raise RuntimeError('Every seq11-v2 frame must contain one GT OBB')
        gt_boxes.append(np.asarray(parsed[0], dtype=np.float64)[:5])

    fallback_boxes, fallback_scores, fallback_payload = [], [], []
    for k1_box, k1_score, dino_box, dino_score in zip(
            k1_boxes, k1_scores, dino_boxes, dino_scores):
        box = k1_box if k1_box is not None else dino_box
        score = k1_score if k1_box is not None else dino_score
        fallback_boxes.append(None if box is None else box.copy())
        fallback_scores.append(score)
        if box is None:
            detection = np.zeros((0, 6), dtype=np.float32)
        else:
            serialized_score = 1.0 if score is None else score
            detection = np.concatenate([
                box.astype(np.float32),
                np.asarray([serialized_score], dtype=np.float32)]).reshape(1, 6)
        fallback_payload.append([detection])
    fallback_raw = pickle.dumps(fallback_payload, protocol=4)
    fallback_path, fallback_sha = _write_exact(
        args.simple_fallback_results, fallback_raw)

    blocks = _segments(frames)
    evaluation_kwargs = dict(
        gt_boxes=gt_boxes, frames=frames, segments=blocks,
        category_rows=support_rows, iou_threshold=args.iou_threshold,
        center_threshold=args.center_threshold_px,
        angle_limit_deg=args.angle_limit_deg,
        high_confidence_threshold=args.high_confidence_threshold)
    metrics = {
        'ordinary_k1': _evaluate(
            'ordinary_k1', k1_boxes, k1_scores, **evaluation_kwargs),
        'deterministic_k1_else_dino': _evaluate(
            'deterministic_k1_else_dino', fallback_boxes, fallback_scores,
            **evaluation_kwargs),
        'base_v3_epoch9_refiner': _evaluate(
            'base_v3_epoch9_refiner', base_boxes, base_scores,
            **evaluation_kwargs)}
    fallback = metrics['deterministic_k1_else_dino']
    base = metrics['base_v3_epoch9_refiner']
    deltas = dict(
        mean_RIoU=base['mean_RIoU'] - fallback['mean_RIoU'],
        rescued=base['rescue']['rescued'] - fallback['rescue']['rescued'],
        old_correct_lost=(base['old_correct_lost']
                          - fallback['old_correct_lost']),
        valid_pair_fraction=(base['temporal']['valid_pair_fraction']
                             - fallback['temporal']['valid_pair_fraction']),
        DFR_percent_per_frame=(
            None if base['temporal']['DFR_percent_per_frame'] is None
            or fallback['temporal']['DFR_percent_per_frame'] is None else
            base['temporal']['DFR_percent_per_frame']
            - fallback['temporal']['DFR_percent_per_frame']),
        ACI=(None if base['temporal']['ACI'] is None
             or fallback['temporal']['ACI'] is None else
             base['temporal']['ACI'] - fallback['temporal']['ACI']),
        MCML_max=base['temporal']['MCML_max']
        - fallback['temporal']['MCML_max'],
        both_bad_false_output_count=(
            base['both_bad']['false_output_count']
            - fallback['both_bad']['false_output_count']))

    return dict(
        protocol=PROTOCOL,
        evidence_boundary='seq11_same_video_source_only_baseline_diagnostic',
        purpose=('test whether the frozen refiner adds measured value beyond '
                 'the deterministic K1-present-else-DINO fallback'),
        inputs=dict(
            comparison_contract=comparison_contract_id,
            frame_order_manifest=frame_id,
            formal_k1_identity=k1_id_file,
            k1_results=k1_result_id,
            dino_cache_identity=dino_id_file,
            dino_all_lane_audit=dino_audit_id,
            formal_support_audit=support_id,
            history_dependency_audit=history_id,
            base_v3_config=base_config_id,
            base_v3_checkpoint=base_checkpoint_id,
            base_v3_promotion=promotion_id,
            base_v3_results=base_result_id,
            base_v3_result_receipt=base_receipt_id,
            base_v3_runtime_audit=runtime_id,
            simple_fallback_results=dict(
                path=fallback_path, sha256=fallback_sha,
                score_fill_value_when_dino_score_absent=1.0,
                metric_confidence_preserves_absence=True)),
        metric_contract=dict(
            frame_count=251, iou_threshold=args.iou_threshold,
            center_threshold_px=args.center_threshold_px,
            angle_limit_deg=args.angle_limit_deg,
            high_confidence_threshold=args.high_confidence_threshold,
            static_metrics_pool_all_frames=True,
            temporal_pairs_require_original_frame_difference_one=True,
            temporal_state_resets_at_frame_gaps=True,
            missing_prediction_breaks_continuity=True,
            DFR_ACI_report_pair_coverage=True,
            both_bad_excluded_from_rescue_denominator=True,
            terminal_failure_included_in_MCML=True,
            cross_checkpoint_pairs=False),
        formal_category_counts=support['category_counts'],
        metric_pair_edge_count=len(declared_pair_edges),
        metrics=metrics,
        base_v3_minus_deterministic_fallback=deltas,
        interpretation_policy=dict(
            frozen_before_results=True,
            deterministic_fallback_is_required_baseline=True,
            base_v3_value_requires_joint_review=(
                'rescue, old-correct loss, RIoU, pair coverage, DFR, ACI, '
                'MCML, and both-bad behavior'),
            no_single_metric_promotion=True,
            no_automatic_cv_training_authorization=True),
        target_data_read=False,
        fixed_test_read=False,
        training_run=False,
        eligible_for_checkpoint_promotion=False,
        eligible_for_three_fold_training=False,
        decision='BASELINE_COMPARISON_READY_FOR_REVIEW_NO_TRAINING_AUTHORIZED')


def main():
    args = parse_args()
    report = compare(args)
    raw = (json.dumps(report, indent=2, ensure_ascii=False) + '\n').encode(
        'utf-8')
    output, _ = _write_exact(args.out_json, raw)
    print('[seq11-v2-baselines] output={}'.format(output))
    print('[seq11-v2-baselines] decision={}'.format(report['decision']))
    for name, metric in report['metrics'].items():
        print('[seq11-v2-baselines] {}: rescue={}/{} old_lost={} '
              'RIoU={:.4f} DFR={} ACI={} MCML={} pair_fraction={:.4f}'.format(
                  name, metric['rescue']['rescued'],
                  metric['rescue']['denominator'],
                  metric['old_correct_lost'], metric['mean_RIoU'],
                  metric['temporal']['DFR_percent_per_frame'],
                  metric['temporal']['ACI'], metric['temporal']['MCML_max'],
                  metric['temporal']['valid_pair_fraction']))


if __name__ == '__main__':
    main()
