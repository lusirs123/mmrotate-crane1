#!/usr/bin/env python3
"""Inventory both seq11 rescue mechanisms before block-CV training."""

import argparse
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from crane_project.tools.eval_crane_offline import compute_riou, parse_dota_txt
from crane_project.tools.symeood_dino_seq11_block_cv_materialize import (
    load_manifest)
from crane_project.tools.symeood_dino_seq11_block_split import (
    ALL_LANE_PROTOCOL)


PROTOCOL = 'k1_dino_seq11_three_window_block_cv_support_audit_v1'
PROTOCOL_V2 = 'k1_dino_seq11_event_block_cv_support_audit_v2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-version', choices=('v1', 'v2'),
                        default='v1')
    parser.add_argument('--k1-results', required=True)
    parser.add_argument('--audit-json', required=True)
    parser.add_argument(
        '--cv-manifest',
        default=('crane_project/data_contracts/'
                 'real_seq11_pilot_k1p9_three_window_block_cv_v1.json'))
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument(
        '--source-split',
        default='extra_source_real_seq11_pilot_k1p9_cv_full59_v1')
    parser.add_argument('--min-pooled-present-wrong', type=int, default=6)
    parser.add_argument('--min-present-wrong-folds', type=int, default=2)
    parser.add_argument('--min-pooled-missing', type=int, default=3)
    parser.add_argument('--min-missing-folds', type=int, default=1)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--formal-k1-identity')
    parser.add_argument('--dino-cache-identity')
    parser.add_argument('--frame-order-json')
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_results(path, expected_count=59):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise RuntimeError('Formal K1 PKL must contain exactly {} frames'.format(
            expected_count))
    boxes = []
    for index, result in enumerate(payload):
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise RuntimeError('K1 frame {} must contain one class'.format(
                index))
        detections = np.asarray(result[0], dtype=np.float64)
        if detections.size == 0:
            boxes.append(None)
            continue
        detections = detections.reshape((-1, 6))
        if detections.shape[0] != 1:
            raise RuntimeError(
                'K1 frame {} must contain at most one OBB'.format(index))
        box = detections[0, :5].copy()
        if not np.isfinite(box).all() or np.any(box[2:4] <= 0.0):
            raise RuntimeError('Invalid K1 OBB at index {}'.format(index))
        boxes.append(box)
    return absolute, boxes


def _annotations(data_root, split):
    ann_dir = Path(data_root).resolve() / split / 'annfiles'
    paths = sorted(
        path for path in ann_dir.glob('*.txt')
        if not path.name.startswith('._'))
    if len(paths) != 59:
        raise RuntimeError('Original seq11 split must contain 59 annotations')
    stems, boxes = [], []
    for path in paths:
        parsed = parse_dota_txt(os.fspath(path))
        if len(parsed) != 1:
            raise RuntimeError('Every seq11 frame must have exactly one GT')
        stems.append(path.stem)
        boxes.append(np.asarray(parsed[0], dtype=np.float64)[:5])
    return os.fspath(ann_dir), stems, boxes


def _audit(path, stems):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        raw = handle.read()
    payload = json.loads(raw.decode('utf-8'))
    if payload.get('protocol') != ALL_LANE_PROTOCOL:
        raise RuntimeError('Input is not a complete all-lane audit')
    index = {}
    for record in payload.get('records') or []:
        stem = Path(record.get('filename', '')).stem
        if not stem or stem in index:
            raise RuntimeError('Invalid/duplicate seq11 audit frame')
        if record.get('raw_selected_source') == 'not_computed':
            raise RuntimeError('DINO was not computed for ' + stem)
        if ('dino_invoked' in record
                and record.get('dino_invoked') not in (True, 1)):
            raise RuntimeError('DINO was not computed for ' + stem)
        if 'dino_native_box' not in record:
            raise RuntimeError('DINO key absent for ' + stem)
        index[stem] = record
    if set(index) != set(stems):
        raise RuntimeError('Audit and formal K1 frame sets differ')
    return absolute, hashlib.sha256(raw).hexdigest(), index


def _box(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if (array.size < 5 or not np.isfinite(array[:5]).all()
            or np.any(array[2:4] <= 0.0)):
        raise RuntimeError('Invalid cached DINO OBB')
    return array[:5].copy()


def _category(k1_box, k1_riou, dino_riou):
    if k1_riou >= 0.5:
        return 'k1_hit'
    if k1_box is None:
        return ('k1_missing_dino_hit' if dino_riou >= 0.5
                else 'k1_missing_dino_not_hit')
    return ('k1_present_wrong_dino_hit' if dino_riou >= 0.5
            else 'k1_present_wrong_dino_not_hit')


def _summary(rows):
    counts = Counter(row['category'] for row in rows)
    categories = (
        'k1_hit', 'k1_missing_dino_hit', 'k1_missing_dino_not_hit',
        'k1_present_wrong_dino_hit',
        'k1_present_wrong_dino_not_hit')
    return dict(
        frame_count=len(rows),
        category_counts={key: int(counts.get(key, 0)) for key in categories},
        category_frames={
            key: [row['frame_key'] for row in rows
                  if row['category'] == key]
            for key in categories})


def _identity_json(path, expected_protocol):
    if not path:
        raise RuntimeError('Missing required V2 identity JSON')
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if (payload.get('protocol') != expected_protocol
            or payload.get('passed') is not True
            or payload.get('target_data_read') is not False
            or payload.get('fixed_test_read') is not False):
        raise RuntimeError('Invalid V2 identity: ' + absolute)
    return absolute, _sha256(absolute), payload


def _v2_annotations(data_root, split, frame_order):
    ann_dir = Path(data_root).resolve() / split / 'annfiles'
    stems, boxes = [], []
    for row in frame_order:
        stem = row.get('frame_key')
        path = ann_dir / (stem + '.txt')
        if not path.is_file() or _sha256(path) != row.get('annotation_sha256'):
            raise RuntimeError('Frame-order annotation mismatch: ' + str(path))
        parsed = parse_dota_txt(os.fspath(path))
        if len(parsed) != 1:
            raise RuntimeError('Every seq11-v2 frame must have exactly one GT')
        stems.append(stem)
        boxes.append(np.asarray(parsed[0], dtype=np.float64)[:5])
    if len(stems) != 251 or len(set(stems)) != 251:
        raise RuntimeError('V2 frame order must contain exactly 251 frames')
    return os.fspath(ann_dir), stems, boxes


def _hard_events(rows):
    events = []
    current = []
    for row in rows:
        if row['category'] == 'k1_hit':
            if current:
                events.append(current)
                current = []
            continue
        if current and row['frame'] != current[-1]['frame'] + 1:
            events.append(current)
            current = []
        current.append(row)
    if current:
        events.append(current)
    result = []
    for event_id, event in enumerate(events, 1):
        counts = Counter(row['category'] for row in event)
        result.append(dict(
            event_id=event_id, start_frame=event[0]['frame'],
            end_frame=event[-1]['frame'], frame_count=len(event),
            frame_keys=[row['frame_key'] for row in event],
            category_counts=dict(counts)))
    return result


def audit_v2(args):
    k1_id_path, k1_id_sha, k1_identity = _identity_json(
        args.formal_k1_identity,
        'formal_k1_seq11_v2_full251_identity_v2')
    dino_id_path, dino_id_sha, dino_identity = _identity_json(
        args.dino_cache_identity, 'seq11_v2_dino_cache_identity_v1')
    frame_order_path = os.path.abspath(os.fspath(args.frame_order_json))
    with open(frame_order_path, 'r', encoding='utf-8') as handle:
        frame_manifest = json.load(handle)
    if (frame_manifest.get('protocol') !=
            'seq11_v2_full251_frame_order_manifest_v2'
            or _sha256(frame_order_path) !=
            k1_identity['inputs']['frame_order_manifest_sha256']):
        raise RuntimeError('Formal K1 frame-order identity mismatch')
    k1_path, k1_boxes = _load_results(args.k1_results, expected_count=251)
    if _sha256(k1_path) != k1_identity['inputs']['results_sha256']:
        raise RuntimeError('K1 result hash differs from formal identity')
    ann_dir, stems, gt_boxes = _v2_annotations(
        args.data_root, args.source_split, frame_manifest['frames'])
    audit_path, audit_sha, audit_index = _audit(args.audit_json, stems)
    if audit_sha != dino_identity['inputs']['audit_json']['sha256']:
        raise RuntimeError('DINO audit hash differs from cache identity')
    rows = []
    for stem, gt_box, k1_box in zip(stems, gt_boxes, k1_boxes):
        frame = int(stem.rsplit('_', 1)[1])
        dino_box = _box(audit_index[stem].get('dino_native_box'))
        k1_riou = (0.0 if k1_box is None else
                   float(compute_riou(k1_box, gt_box)))
        dino_riou = (0.0 if dino_box is None else
                     float(compute_riou(dino_box, gt_box)))
        if k1_riou >= 0.5:
            category = 'k1_hit'
        elif dino_riou < 0.5:
            category = 'both_bad'
        elif k1_box is None:
            category = 'k1_missing_dino_hit'
        else:
            category = 'k1_present_wrong_dino_hit'
        rows.append(dict(
            frame_key=stem, frame=frame, k1_present=k1_box is not None,
            k1_riou=k1_riou, dino_present=dino_box is not None,
            dino_riou=dino_riou, category=category))
    counts = Counter(row['category'] for row in rows)
    events = _hard_events(rows)
    event_support = Counter()
    for event in events:
        for category, count in event['category_counts'].items():
            if count:
                event_support[category] += 1
    thresholds = dict(
        pooled_k1_missing_dino_hit_min=12,
        pooled_k1_present_wrong_dino_hit_min=6,
        k1_missing_dino_hit_event_min=6,
        k1_present_wrong_dino_hit_event_min=3,
        per_fold_k1_missing_dino_hit_min=3,
        per_fold_k1_present_wrong_dino_hit_min=1,
        per_fold_each_rescue_event_min=1,
        worst_fold_valid_pair_fraction_min=0.70,
        worst_fold_mcml_max=5,
        pooled_old_correct_lost_max=0)
    checks = dict(
        exact_251_frames=len(rows) == 251,
        formal_k1_identity_bound=True,
        dino_cache_identity_bound=True,
        pooled_missing_support=(
            counts['k1_missing_dino_hit'] >=
            thresholds['pooled_k1_missing_dino_hit_min']),
        pooled_present_wrong_support=(
            counts['k1_present_wrong_dino_hit'] >=
            thresholds['pooled_k1_present_wrong_dino_hit_min']),
        missing_event_support=(
            event_support['k1_missing_dino_hit'] >=
            thresholds['k1_missing_dino_hit_event_min']),
        present_wrong_event_support=(
            event_support['k1_present_wrong_dino_hit'] >=
            thresholds['k1_present_wrong_dino_hit_event_min']),
        fixed_test_not_read=True)
    passed = all(checks.values())
    return dict(
        protocol=PROTOCOL_V2,
        evidence_boundary='seq11_same_video_source_support_pre_fold',
        inputs=dict(
            k1_results=k1_path, k1_results_sha256=_sha256(k1_path),
            formal_k1_identity=k1_id_path,
            formal_k1_identity_sha256=k1_id_sha,
            dino_audit=audit_path, dino_audit_sha256=audit_sha,
            dino_cache_identity=dino_id_path,
            dino_cache_identity_sha256=dino_id_sha,
            frame_order_manifest=frame_order_path,
            frame_order_manifest_sha256=_sha256(frame_order_path),
            ann_dir=ann_dir),
        classification_threshold_riou=0.5,
        category_counts={key: int(counts.get(key, 0)) for key in (
            'k1_hit', 'k1_missing_dino_hit',
            'k1_present_wrong_dino_hit', 'both_bad')},
        hard_event_count=len(events), hard_event_support=dict(event_support),
        hard_events=events, preregistered_thresholds=thresholds,
        checks=checks, rows=rows,
        target_data_read=False, fixed_test_read=False,
        passed=passed, eligible_for_fold_construction=passed,
        eligible_for_three_fold_training=False,
        decision=('ALLOW_SEQ11_V2_EVENT_FOLD_CONSTRUCTION' if passed else
                  'STOP_SEQ11_V2_MECHANISM_SUPPORT_INSUFFICIENT'))


def audit(args):
    manifest = load_manifest(args.cv_manifest)
    k1_path, k1_boxes = _load_results(args.k1_results)
    ann_dir, stems, gt_boxes = _annotations(
        args.data_root, args.source_split)
    audit_path, audit_sha, audit_index = _audit(args.audit_json, stems)
    fold_by_stem = {}
    for fold in manifest['folds']:
        for stem in fold['validation_stems']:
            fold_by_stem[stem] = fold['fold_id']
    rows = []
    for stem, gt_box, k1_box in zip(stems, gt_boxes, k1_boxes):
        dino_box = _box(audit_index[stem].get('dino_native_box'))
        k1_riou = (0.0 if k1_box is None else
                   float(compute_riou(k1_box, gt_box)))
        dino_riou = (0.0 if dino_box is None else
                     float(compute_riou(dino_box, gt_box)))
        rows.append(dict(
            frame_key=stem, oof_fold=fold_by_stem.get(stem),
            k1_present=k1_box is not None, k1_riou=k1_riou,
            k1_hit=k1_riou >= 0.5,
            dino_present=dino_box is not None, dino_riou=dino_riou,
            dino_hit=dino_riou >= 0.5,
            category=_category(k1_box, k1_riou, dino_riou)))
    fold_summaries = {}
    for fold in manifest['folds']:
        fold_summaries[str(fold['fold_id'])] = _summary([
            row for row in rows if row['oof_fold'] == fold['fold_id']])
    oof_rows = [row for row in rows if row['oof_fold'] is not None]
    pooled = _summary(oof_rows)
    present_key = 'k1_present_wrong_dino_hit'
    missing_key = 'k1_missing_dino_hit'
    present_folds = sum(
        fold_summaries[str(fold_id)]['category_counts'][present_key] > 0
        for fold_id in range(1, 4))
    missing_folds = sum(
        fold_summaries[str(fold_id)]['category_counts'][missing_key] > 0
        for fold_id in range(1, 4))
    checks = dict(
        formal_k1_frame_count_59=len(k1_boxes) == 59,
        pooled_oof_frame_count_33=len(oof_rows) == 33,
        pooled_present_wrong_support_ge_6=(
            pooled['category_counts'][present_key]
            >= int(args.min_pooled_present_wrong)),
        present_wrong_supported_folds_ge_2=(
            present_folds >= int(args.min_present_wrong_folds)),
        pooled_missing_support_ge_3=(
            pooled['category_counts'][missing_key]
            >= int(args.min_pooled_missing)),
        missing_supported_folds_ge_1=(
            missing_folds >= int(args.min_missing_folds)),
        sparse_oof_temporal_metrics_not_computed=True,
        fixed_test_not_read=True)
    passed = all(checks.values())
    return dict(
        protocol=PROTOCOL,
        evidence_boundary='same_video_three_window_source_support_only',
        target_data_read=False, fixed_test_read=False,
        temporal_metrics_computed=False,
        input=dict(
            k1_results=k1_path, k1_results_sha256=_sha256(k1_path),
            audit_json=audit_path, audit_json_sha256=audit_sha,
            cv_manifest=manifest['path'],
            cv_manifest_sha256=manifest['sha256'], ann_dir=ann_dir),
        thresholds=dict(
            min_pooled_present_wrong=int(args.min_pooled_present_wrong),
            min_present_wrong_folds=int(args.min_present_wrong_folds),
            min_pooled_missing=int(args.min_pooled_missing),
            min_missing_folds=int(args.min_missing_folds)),
        all_59=_summary(rows), pooled_oof_33=pooled,
        folds=fold_summaries,
        present_wrong_supported_fold_count=int(present_folds),
        missing_supported_fold_count=int(missing_folds),
        checks=checks, rows=rows, passed=passed,
        eligible_for_three_fold_training=passed,
        eligible_for_router_claim=False,
        eligible_for_independent_sequence_claim=False,
        eligible_for_fixed_test=False,
        decision=(
            'ALLOW_SEQ11_THREE_WINDOW_BLOCK_CV_TRAINING' if passed else
            'STOP_SEQ11_THREE_WINDOW_BLOCK_CV_SUPPORT_INSUFFICIENT'))


def main():
    args = parse_args()
    report = audit_v2(args) if args.protocol_version == 'v2' else audit(args)
    output = os.path.abspath(os.fspath(args.out_json))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    if args.protocol_version == 'v1':
        print('[seq11-cv-support] pooled={}'.format(
            report['pooled_oof_33']['category_counts']))
        print('[seq11-cv-support] folds={}'.format({
            key: value['category_counts']
            for key, value in report['folds'].items()}))
    else:
        print('[seq11-cv-support] categories={}'.format(
            report['category_counts']))
    print('[seq11-cv-support] decision={}'.format(report['decision']))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
