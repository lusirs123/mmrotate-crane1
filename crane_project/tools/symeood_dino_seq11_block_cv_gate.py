#!/usr/bin/env python3
"""Joint legacy non-regression and pooled OOF mechanism gate for block CV."""

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from crane_project.tools.eval_crane_offline import compute_riou, parse_dota_txt
from crane_project.tools.symeood_dino_causal_history_source_gate import (
    _seq11_blocksplit_legacy_preservation)
from crane_project.tools.symeood_dino_dual_tower_v2_audit import (
    _annotations as _source_val_annotations,
    _load_results as _source_val_results,
    _metrics)
from crane_project.tools.symeood_dino_seq11_block_cv_materialize import (
    load_manifest)
from crane_project.tools.symeood_dino_seq11_block_cv_support_audit import (
    PROTOCOL as SUPPORT_PROTOCOL, _audit as _source_audit,
    _box as _dino_box, _category, _load_results as _full59_results)
from crane_project.utils.geometry_refiner_source_gate import (
    relaxed_composite_gate)


PROTOCOL = 'k1_retentive_seq11_three_window_block_cv_gate_v1'
CHECKPOINT_PROTOCOL = 'source_only_k1_retentive_v3_seq11_blockcv_v1'
MANIFEST_SHA256 = (
    '09474f2145803498b4651dd0ea4431de15402a3c74ba230214a7a9e0a651f7ac')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--candidate-source-val-results', nargs=3, required=True)
    parser.add_argument('--candidate-oof-results', nargs=3, required=True)
    parser.add_argument('--candidate-checkpoints', nargs=3, required=True)
    parser.add_argument('--k1-source-val-results', required=True)
    parser.add_argument('--k1-full59-results', required=True)
    parser.add_argument('--source-val-audit', required=True)
    parser.add_argument('--source-audit', required=True)
    parser.add_argument('--support-audit', required=True)
    parser.add_argument(
        '--cv-manifest',
        default=('crane_project/data_contracts/'
                 'real_seq11_pilot_k1p9_three_window_block_cv_v1.json'))
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument(
        '--source-split',
        default='extra_source_real_seq11_pilot_k1p9_cv_full59_v1')
    parser.add_argument('--min-present-wrong-rescue-rate', type=float,
                        default=0.25)
    parser.add_argument('--min-missing-rescue-rate', type=float, default=0.80)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_variable_results(path, expected_count):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise RuntimeError('OOF result count mismatch: ' + absolute)
    boxes = []
    for index, result in enumerate(payload):
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise RuntimeError('OOF result must contain one class')
        detections = np.asarray(result[0], dtype=np.float64)
        if detections.size == 0:
            boxes.append(None)
            continue
        detections = detections.reshape((-1, 6))
        if detections.shape[0] != 1:
            raise RuntimeError('OOF frame contains multiple detections')
        box = detections[0, :5].copy()
        if not np.isfinite(box).all() or np.any(box[2:4] <= 0.0):
            raise RuntimeError('Invalid OOF OBB at index {}'.format(index))
        boxes.append(box)
    return absolute, boxes


def _checkpoint(path, fold_id, manifest_sha256):
    absolute = os.path.abspath(os.fspath(path))
    payload = torch.load(absolute, map_location='cpu')
    contract = dict((payload.get('meta') or {}).get(
        'geometry_refiner_checkpoint_contract') or {})
    train_count, val_count = {1: (49, 10), 2: (47, 12), 3: (48, 11)}[
        fold_id]
    required = dict(
        protocol=CHECKPOINT_PROTOCOL,
        architecture='k1_retentive_causal_phase_refiner_v3',
        frozen_baseline_variant='symeood_k1_epoch24',
        source_train_frames=2781 + train_count,
        original_source_train_frames=2781,
        auxiliary_source_frames=59,
        auxiliary_source_train_frames=train_count,
        auxiliary_source_val_frames=val_count,
        auxiliary_cv_protocol=(
            'real_seq11_auxiliary_three_window_block_cv_v1'),
        auxiliary_cv_fold=fold_id,
        auxiliary_cv_manifest_sha256=manifest_sha256,
        auxiliary_train_val_overlap=0,
        auxiliary_validation_temporal_metrics=False,
        target_data_read=False, fixed_test_read=False,
        source_gate_passed=False,
        domain_routing=False, sequence_frame_routing=False,
        temporal_state=False)
    failures = [
        '{}={!r} expected {!r}'.format(key, contract.get(key), expected)
        for key, expected in required.items()
        if contract.get(key) != expected]
    if failures:
        raise RuntimeError('Fold checkpoint contract failed: ' + '; '.join(
            failures))
    return absolute, _sha256(absolute), contract


def _epoch(path):
    stem = Path(path).stem
    if not stem.startswith('epoch_') or not stem[6:].isdigit():
        raise RuntimeError('Checkpoint must be named epoch_<N>.pth')
    return int(stem[6:])


def _support(path, manifest_sha256):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        raw = handle.read()
    payload = json.loads(raw.decode('utf-8'))
    required = dict(
        protocol=SUPPORT_PROTOCOL,
        evidence_boundary='same_video_three_window_source_support_only',
        target_data_read=False, fixed_test_read=False,
        temporal_metrics_computed=False, passed=True,
        eligible_for_three_fold_training=True)
    failures = [
        '{}={!r}'.format(key, payload.get(key))
        for key, expected in required.items()
        if payload.get(key) != expected]
    if ((payload.get('input') or {}).get('cv_manifest_sha256')
            != manifest_sha256):
        failures.append('manifest hash mismatch')
    if failures:
        raise RuntimeError('Support audit contract failed: ' + '; '.join(
            failures))
    return absolute, hashlib.sha256(raw).hexdigest(), payload


def _gt_map(data_root, source_split):
    ann_dir = (Path(data_root).resolve()
               / source_split / 'annfiles')
    result = {}
    for path in sorted(ann_dir.glob('*.txt')):
        if path.name.startswith('._'):
            continue
        parsed = parse_dota_txt(os.fspath(path))
        if len(parsed) != 1:
            raise RuntimeError('Every seq11 annotation must have one GT')
        result[path.stem] = np.asarray(parsed[0], dtype=np.float64)[:5]
    if len(result) != 59:
        raise RuntimeError('Expected 59 seq11 annotations')
    return os.fspath(ann_dir), result


def _mean(values):
    return float(np.mean(values)) if values else None


def audit(args):
    manifest = load_manifest(args.cv_manifest)
    if manifest['sha256'] != MANIFEST_SHA256:
        raise RuntimeError('Unexpected block-CV manifest hash')
    support_path, support_sha, support = _support(
        args.support_audit, manifest['sha256'])
    source_val_ann_dir, source_val_meta, domain_counts = (
        _source_val_annotations(os.path.join(
            os.path.abspath(args.data_root), 'val', 'annfiles')))
    k1_source_path, k1_source_boxes = _source_val_results(
        args.k1_source_val_results)
    k1_source_metrics = _metrics(source_val_meta, k1_source_boxes)
    k1_full_path, k1_full_boxes = _full59_results(
        args.k1_full59_results)
    ann_dir, gt_map = _gt_map(args.data_root, args.source_split)
    stems59 = sorted(gt_map)
    k1_map = dict(zip(stems59, k1_full_boxes))
    source_audit_path, source_audit_sha, source_index = _source_audit(
        args.source_audit, stems59)

    fold_reports = []
    oof_rows = []
    epochs = set()
    for offset, fold in enumerate(manifest['folds']):
        fold_id = fold['fold_id']
        source_path, source_boxes = _source_val_results(
            args.candidate_source_val_results[offset])
        oof_path, oof_boxes = _load_variable_results(
            args.candidate_oof_results[offset],
            len(fold['validation_stems']))
        checkpoint_path, checkpoint_sha, checkpoint_contract = _checkpoint(
            args.candidate_checkpoints[offset], fold_id,
            manifest['sha256'])
        epochs.add(_epoch(checkpoint_path))
        source_metrics = _metrics(source_val_meta, source_boxes)
        preservation = _seq11_blocksplit_legacy_preservation(
            source_metrics, k1_source_metrics)
        composite = relaxed_composite_gate(
            source_metrics, k1_source_metrics, min_composite_gain=0.0,
            reference_policy='formal_k1_source_val_738')
        legacy_passed = bool(preservation['passed'] and composite['passed'])
        validation_stems = sorted(fold['validation_stems'])
        fold_rows = []
        for stem, candidate_box in zip(validation_stems, oof_boxes):
            gt_box = gt_map[stem]
            k1_box = k1_map[stem]
            dino_box = _dino_box(
                source_index[stem].get('dino_native_box'))
            k1_riou = (0.0 if k1_box is None else
                       float(compute_riou(k1_box, gt_box)))
            dino_riou = (0.0 if dino_box is None else
                         float(compute_riou(dino_box, gt_box)))
            candidate_riou = (
                0.0 if candidate_box is None else
                float(compute_riou(candidate_box, gt_box)))
            row = dict(
                fold_id=fold_id, frame_key=stem,
                category=_category(k1_box, k1_riou, dino_riou),
                k1_present=k1_box is not None, k1_riou=k1_riou,
                k1_hit=k1_riou >= 0.5,
                dino_riou=dino_riou, dino_hit=dino_riou >= 0.5,
                candidate_present=candidate_box is not None,
                candidate_riou=candidate_riou,
                candidate_hit=candidate_riou >= 0.5)
            fold_rows.append(row)
            oof_rows.append(row)
        fold_reports.append(dict(
            fold_id=fold_id,
            input=dict(
                source_val_results=source_path,
                source_val_results_sha256=_sha256(source_path),
                oof_results=oof_path,
                oof_results_sha256=_sha256(oof_path),
                checkpoint=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_contract=checkpoint_contract),
            source_val_metrics=source_metrics,
            legacy_preservation=preservation,
            legacy_average_gain=composite,
            legacy_passed=legacy_passed,
            oof_frame_count=len(fold_rows)))
    if len(epochs) != 1:
        raise RuntimeError('All fold checkpoints must be from one epoch')
    if (len(oof_rows) != 33
            or len({row['frame_key'] for row in oof_rows}) != 33):
        raise RuntimeError('Pooled OOF rows must contain 33 unique frames')

    present = [row for row in oof_rows
               if row['category'] == 'k1_present_wrong_dino_hit']
    missing = [row for row in oof_rows
               if row['category'] == 'k1_missing_dino_hit']
    k1_good = [row for row in oof_rows if row['k1_hit']]
    present_rescued = sum(row['candidate_hit'] for row in present)
    missing_rescued = sum(row['candidate_hit'] for row in missing)
    k1_good_lost = sum(not row['candidate_hit'] for row in k1_good)
    present_rate = present_rescued / len(present) if present else 0.0
    missing_rate = missing_rescued / len(missing) if missing else 0.0
    candidate_mean = _mean([row['candidate_riou'] for row in oof_rows])
    k1_mean = _mean([row['k1_riou'] for row in oof_rows])
    present_candidate_mean = _mean(
        [row['candidate_riou'] for row in present])
    present_k1_mean = _mean([row['k1_riou'] for row in present])
    checks = dict(
        support_audit_passed=support['passed'] is True,
        all_three_legacy_source_val_gates_passed=all(
            row['legacy_passed'] for row in fold_reports),
        present_wrong_support_ge_6=len(present) >= 6,
        present_wrong_rescue_rate_ge_0p25=(
            present_rate >= float(args.min_present_wrong_rescue_rate)),
        present_wrong_mean_riou_improved=(
            bool(present) and present_candidate_mean > present_k1_mean),
        missing_support_ge_3=len(missing) >= 3,
        missing_rescue_rate_ge_0p80=(
            missing_rate >= float(args.min_missing_rescue_rate)),
        no_k1_good_hit_lost=(k1_good_lost == 0),
        pooled_mean_riou_within_0p01=(
            candidate_mean >= k1_mean - 0.01),
        sparse_oof_temporal_metrics_not_computed=True,
        fixed_test_not_read=True)
    passed = all(checks.values())
    return dict(
        protocol=PROTOCOL, metric_protocol_version=2,
        evidence_boundary=(
            'three_legacy_source_val_streams_plus_pooled_sparse_oof_33'),
        target_data_read=False, fixed_test_read=False,
        temporal_metrics_computed_on_oof=False,
        input=dict(
            cv_manifest=manifest['path'],
            cv_manifest_sha256=manifest['sha256'],
            support_audit=support_path,
            support_audit_sha256=support_sha,
            source_audit=source_audit_path,
            source_audit_sha256=source_audit_sha,
            source_val_audit=os.path.abspath(args.source_val_audit),
            source_val_audit_sha256=_sha256(args.source_val_audit),
            k1_source_val_results=k1_source_path,
            k1_source_val_results_sha256=_sha256(k1_source_path),
            k1_full59_results=k1_full_path,
            k1_full59_results_sha256=_sha256(k1_full_path),
            source_val_ann_dir=source_val_ann_dir,
            source_val_domain_counts=domain_counts,
            seq11_ann_dir=ann_dir),
        epoch=next(iter(epochs)),
        formal_k1_source_val_metrics=k1_source_metrics,
        folds=fold_reports,
        pooled_oof=dict(
            frame_count=len(oof_rows),
            present_wrong_support_count=len(present),
            present_wrong_rescued_count=int(present_rescued),
            present_wrong_rescue_rate=float(present_rate),
            present_wrong_candidate_mean_riou=present_candidate_mean,
            present_wrong_k1_mean_riou=present_k1_mean,
            missing_support_count=len(missing),
            missing_rescued_count=int(missing_rescued),
            missing_rescue_rate=float(missing_rate),
            k1_good_count=len(k1_good),
            k1_good_lost_count=int(k1_good_lost),
            candidate_mean_riou=candidate_mean,
            k1_mean_riou=k1_mean,
            rows=oof_rows),
        thresholds=dict(
            min_present_wrong_rescue_rate=float(
                args.min_present_wrong_rescue_rate),
            min_missing_rescue_rate=float(args.min_missing_rescue_rate),
            pooled_mean_riou_drop_limit=0.01),
        checks=checks, passed=passed,
        eligible_for_epoch_selection=passed,
        eligible_for_checkpoint_promotion=False,
        eligible_for_fixed_test=False,
        eligible_for_unknown_sequence_claim=False,
        decision=(
            'ALLOW_SEQ11_BLOCK_CV_EPOCH_SELECTION_CANDIDATE' if passed else
            'STOP_SEQ11_BLOCK_CV_GATE_FAILED'))


def main():
    args = parse_args()
    report = audit(args)
    output = os.path.abspath(os.fspath(args.out_json))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print('[seq11-cv-gate] epoch={}'.format(report['epoch']))
    print('[seq11-cv-gate] pooled={}'.format({
        key: value for key, value in report['pooled_oof'].items()
        if key != 'rows'}))
    print('[seq11-cv-gate] decision={}'.format(report['decision']))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
