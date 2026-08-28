"""Audit the single authorized Dual-Tower V2.1 fixed TEST run.

The candidate is compared with both native DINO and SymEOOD boxes already
stored in the complete all-lane audit.  The pass rule is preregistered before
TEST: positive relaxed-composite improvement over native DINO, all existing
guardrails, TDR >= 99%, and MCML_max <= 5 in both domains.  Results never
select a different epoch or update parameters.
"""

import argparse
import glob
import hashlib
import json
import os
import pickle

import numpy as np

from crane_project.tools.eval_crane_offline import (
    CraneOfflineEvaluator, parse_dota_txt, parse_seq_frame)
from crane_project.utils.geometry_refiner_source_gate import (
    relaxed_composite_gate)
from mmrotate.datasets.pipelines.loading import dino_record_was_computed


EXPECTED_FRAME_COUNT = 992
PROMOTION_PROTOCOL = 'source_gated_dual_tower_v21_promotion_v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-results', required=True)
    parser.add_argument('--candidate-checkpoint', required=True)
    parser.add_argument('--promotion-json', required=True)
    parser.add_argument('--all-lane-audit', required=True)
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_key(filename):
    return os.path.splitext(os.path.basename(os.fspath(filename)))[0]


def _as_box(value, frame_key, name):
    if value is None:
        return None
    box = np.asarray(value, dtype=np.float64).reshape(-1)
    if (box.size < 5 or not np.isfinite(box[:5]).all()
            or np.any(box[2:4] <= 0.0)):
        raise RuntimeError('Invalid {} box at {}'.format(name, frame_key))
    return box[:5].copy()


def _load_results(path):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or len(payload) != EXPECTED_FRAME_COUNT:
        raise RuntimeError('Candidate results must contain exactly 992 frames')
    boxes = []
    for index, result in enumerate(payload):
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise RuntimeError(
                'Invalid class result at frame {}'.format(index))
        detections = np.asarray(result[0], dtype=np.float64)
        if detections.size == 0:
            boxes.append(None)
            continue
        detections = detections.reshape((-1, 6))
        if detections.shape[0] != 1:
            raise RuntimeError('Expected one OBB at frame {}'.format(index))
        boxes.append(_as_box(detections[0], str(index), 'candidate'))
    return absolute, boxes


def _annotations(ann_dir):
    absolute = os.path.abspath(os.fspath(ann_dir))
    paths = sorted(glob.glob(os.path.join(absolute, '*.txt')))
    if len(paths) != EXPECTED_FRAME_COUNT:
        raise RuntimeError('Fixed TEST annotations must contain 992 frames')
    records = []
    keys = []
    for path in paths:
        boxes = parse_dota_txt(path)
        if len(boxes) != 1:
            raise RuntimeError('TEST frame must contain one GT: ' + path)
        domain, sequence, frame = parse_seq_frame(path)
        records.append(dict(
            domain=domain, seq_id=sequence, frame_id=frame,
            gt_box=boxes[0], score=1.0, plc_rope=None))
        keys.append(_frame_key(path))
    counts = dict(
        real=sum(item['domain'] == 'real' for item in records),
        sim=sum(item['domain'] == 'sim' for item in records))
    if counts != {'real': 420, 'sim': 572}:
        raise RuntimeError('Unexpected fixed TEST domain counts')
    return absolute, records, keys, counts


def _all_lane_boxes(path, ordered_keys):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if payload.get('protocol') != 'source_owned_geometry_union_v2':
        raise RuntimeError('Fixed TEST requires a complete all-lane audit')
    records = list(payload.get('records') or [])
    if len(records) != EXPECTED_FRAME_COUNT:
        raise RuntimeError('All-lane audit must contain exactly 992 frames')
    indexed = {}
    for record in records:
        key = _frame_key(record.get('filename', ''))
        if not key or key in indexed:
            raise RuntimeError('Invalid or duplicate all-lane frame key')
        if 'dino_native_box' not in record or 'sym_eood_box' not in record:
            raise RuntimeError('All-lane record lacks a required box key')
        if not dino_record_was_computed(record):
            raise RuntimeError('DINO was not computed at ' + key)
        indexed[key] = record
    if set(indexed) != set(ordered_keys):
        raise RuntimeError('All-lane audit and TEST annotations disagree')
    dino = [_as_box(indexed[key]['dino_native_box'], key, 'DINO')
            for key in ordered_keys]
    sym = [_as_box(indexed[key]['sym_eood_box'], key, 'SymEOOD')
           for key in ordered_keys]
    return absolute, dino, sym


def _metrics(metadata, boxes):
    records = []
    for meta, box in zip(metadata, boxes):
        record = dict(meta)
        record['pred_box'] = box
        records.append(record)
    evaluator = CraneOfflineEvaluator(
        mode='test', center_thresh_px=15.0,
        sim_angle_center_thresh_px=10.0,
        ekf_window=10, mcml_limit=5, iou_thresh=0.5)
    return evaluator.evaluate_records(records)


def _promotion(path, checkpoint_path):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'r', encoding='utf-8') as handle:
        report = json.load(handle)
    required = dict(
        protocol=PROMOTION_PROTOCOL,
        evidence_boundary='source_gate_only',
        target_data_read=False,
        fixed_test_read=False,
        passed=True,
        eligible_for_one_fixed_test=True,
        decision='ALLOW_ONE_DUAL_TOWER_V21_FIXED_TEST')
    failures = [
        '{}={!r}'.format(key, report.get(key))
        for key, expected in required.items()
        if report.get(key) != expected]
    output = dict(report.get('output') or {})
    checkpoint_hash = _sha256(checkpoint_path)
    if output.get('checkpoint_sha256') != checkpoint_hash:
        failures.append('promoted checkpoint SHA256 mismatch')
    contract = dict(output.get('contract') or {})
    if (contract.get('source_gate_passed') is not True
            or contract.get('selected_source_epoch') != 7):
        failures.append('promoted checkpoint contract is not epoch 7')
    if failures:
        raise RuntimeError('Promotion validation failed: ' + '; '.join(
            failures))
    return absolute, _sha256(absolute), checkpoint_hash, report


def _fixed_gate(candidate, dino):
    composite = relaxed_composite_gate(candidate, dino)
    checks = dict(composite['checks'])
    checks.update(dict(
        real_tdr_ge_99=float(candidate['real/TDR_w10(%)']) >= 99.0,
        sim_tdr_ge_99=float(candidate['sim/TDR_w10(%)']) >= 99.0,
        real_mcml_max_le_5=int(candidate['real/MCML_max(frames)']) <= 5,
        sim_mcml_max_le_5=int(candidate['sim/MCML_max(frames)']) <= 5))
    return dict(
        reference='native_dino_fixed_test',
        preregistered_before_fixed_test=True,
        relaxed_composite=composite,
        checks=checks,
        passed=all(checks.values()))


def main():
    args = parse_args()
    candidate_path, candidate_boxes = _load_results(args.candidate_results)
    ann_dir, metadata, keys, counts = _annotations(args.ann_dir)
    audit_path, dino_boxes, sym_boxes = _all_lane_boxes(
        args.all_lane_audit, keys)
    checkpoint_path = os.path.abspath(args.candidate_checkpoint)
    promotion_path, promotion_hash, checkpoint_hash, promotion = _promotion(
        args.promotion_json, checkpoint_path)
    candidate_metrics = _metrics(metadata, candidate_boxes)
    dino_metrics = _metrics(metadata, dino_boxes)
    sym_metrics = _metrics(metadata, sym_boxes)
    fixed_gate = _fixed_gate(candidate_metrics, dino_metrics)
    report = dict(
        protocol='dual_tower_v21_one_time_fixed_test_v1',
        metric_protocol_version=2,
        evidence_boundary='fixed_test_once_after_source_promotion',
        source_selected_epoch=7,
        parameter_update_after_test=False,
        epoch_reselection_after_test=False,
        domain_routing=False,
        sequence_frame_routing=False,
        temporal_inference_state=False,
        dino_detector_rerun=False,
        input=dict(
            candidate_results=candidate_path,
            candidate_results_sha256=_sha256(candidate_path),
            candidate_checkpoint=checkpoint_path,
            candidate_checkpoint_sha256=checkpoint_hash,
            promotion_json=promotion_path,
            promotion_json_sha256=promotion_hash,
            promotion=promotion,
            all_lane_audit=audit_path,
            all_lane_audit_sha256=_sha256(audit_path),
            ann_dir=ann_dir,
            frame_count=EXPECTED_FRAME_COUNT,
            domain_counts=counts),
        metrics=dict(
            dual_tower_v21=candidate_metrics,
            native_dino=dino_metrics,
            sym_eood=sym_metrics),
        fixed_test_gate=fixed_gate,
        passed=fixed_gate['passed'],
        eligible_for_unknown_sequence_claim=False,
        eligible_for_parameter_tuning_from_this_report=False,
        decision=(
            'PASS_DUAL_TOWER_V21_ONE_TIME_FIXED_TEST'
            if fixed_gate['passed'] else
            'STOP_DUAL_TOWER_V21_FIXED_TEST_FAILED'))
    output = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not fixed_gate['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
