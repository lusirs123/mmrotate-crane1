#!/usr/bin/env python3
"""Build the frozen score-only OBB reliability baseline for Base V3 epoch 9.

The tool is deliberately source-val only.  It reconstructs Base V3's actual
anchor branch (K1 when present, native DINO only when K1 is absent), rather
than reusing the older proposal-union audit's selected source.  Base V3's
constant final output score is audited but never used as confidence.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np

from crane_project.tools.eval_crane_offline import (
    angle_diff, compute_riou, parse_dota_txt, parse_seq_frame)


PROTOCOL = 'base_v3_obb_reliability_baseline_v1'
CONTRACT_PROTOCOL = 'base_v3_obb_reliability_baseline_contract_v1'
AUDIT_PROTOCOL = 'source_owned_geometry_union_v2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True)
    parser.add_argument('--promoted-refiner', required=True)
    parser.add_argument('--promotion-report', required=True)
    parser.add_argument('--source-gate', required=True)
    parser.add_argument('--base-v3-results', required=True)
    parser.add_argument('--k1-checkpoint', required=True)
    parser.add_argument('--k1-results', required=True)
    parser.add_argument('--source-val-audit', required=True)
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _identity(path):
    absolute = Path(path).resolve()
    if not absolute.is_file():
        raise RuntimeError('Missing required input: ' + os.fspath(absolute))
    digest = hashlib.sha256()
    with absolute.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=os.fspath(absolute), sha256=digest.hexdigest(),
                size_bytes=absolute.stat().st_size)


def _read_json(path, protocol=None):
    identity = _identity(path)
    with open(identity['path'], 'r', encoding='utf-8') as stream:
        payload = json.load(stream)
    if protocol is not None and payload.get('protocol') != protocol:
        raise RuntimeError(
            'Unexpected protocol in {}: {!r}'.format(
                identity['path'], payload.get('protocol')))
    return identity, payload


def _require_hash(identity, expected, role):
    if identity['sha256'] != expected:
        raise RuntimeError(
            '{} identity mismatch: expected {}, got {}'.format(
                role, expected, identity['sha256']))


def _load_result_boxes(path, expected_count):
    identity = _identity(path)
    with open(identity['path'], 'rb') as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise RuntimeError(
            'Result PKL must contain exactly {} frames'.format(
                expected_count))
    boxes = []
    for index, result in enumerate(payload):
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise RuntimeError(
                'Frame {} must contain exactly one class'.format(index))
        detections = np.asarray(result[0], dtype=np.float64)
        if detections.size == 0:
            boxes.append(None)
            continue
        detections = detections.reshape((-1, 6))
        if detections.shape[0] != 1:
            raise RuntimeError(
                'Frame {} must contain zero or one Top-1 OBB'.format(index))
        box = detections[0].copy()
        if (not np.isfinite(box).all() or np.any(box[2:4] <= 0.0)
                or not 0.0 <= box[5] <= 1.0):
            raise RuntimeError('Invalid OBB at frame {}'.format(index))
        boxes.append(box)
    return identity, boxes


def _load_annotations(path, expected_count, expected_domains):
    directory = Path(path).resolve()
    paths = sorted(directory.glob('*.txt'))
    if len(paths) != expected_count:
        raise RuntimeError(
            'Annotation directory must contain exactly {} frames'.format(
                expected_count))
    rows = []
    for ann_path in paths:
        boxes = parse_dota_txt(os.fspath(ann_path))
        if len(boxes) != 1:
            raise RuntimeError(
                'Each source-val frame must contain exactly one GT: '
                + os.fspath(ann_path))
        domain, sequence, frame = parse_seq_frame(os.fspath(ann_path))
        rows.append(dict(
            frame_key=ann_path.stem, domain=domain, sequence=sequence,
            frame=int(frame), gt=np.asarray(boxes[0], dtype=np.float64)))
    counts = {
        domain: sum(row['domain'] == domain for row in rows)
        for domain in sorted(expected_domains)}
    if counts != expected_domains:
        raise RuntimeError(
            'Unexpected source-val domain counts: {!r}'.format(counts))
    return dict(path=os.fspath(directory), frame_count=len(rows)), rows


def _load_dino_audit(path, frame_keys):
    identity, payload = _read_json(path, AUDIT_PROTOCOL)
    if payload.get('metadata', {}).get('fusion_policy') != (
            'sym_eood_proposal_dino_roi_union'):
        raise RuntimeError('Unexpected source audit fusion policy')
    records = payload.get('records') or []
    by_key = {}
    for record in records:
        key = Path(record.get('filename', '')).stem
        if not key or key in by_key:
            raise RuntimeError('Invalid or duplicate audit frame identity')
        dino = record.get('dino_native_box')
        if dino is None:
            box = None
        else:
            box = np.asarray(dino, dtype=np.float64).reshape(-1)
            if (box.size < 6 or not np.isfinite(box[:6]).all()
                    or np.any(box[2:4] <= 0.0)
                    or not 0.0 <= box[5] <= 1.0):
                raise RuntimeError('Invalid DINO box for ' + key)
            common_score = record.get('dino_native_common_score')
            if common_score is None or not math.isclose(
                    float(common_score), float(box[5]),
                    rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError('DINO score mismatch for ' + key)
            box = box[:6].copy()
        by_key[key] = dict(
            dino=box,
            legacy_selected_source=record.get('selected_source'),
            legacy_sym_eood_box=record.get('sym_eood_box'))
    if set(by_key) != set(frame_keys):
        raise RuntimeError('Source audit frame set differs from annotations')
    return identity, payload, by_key


def _component_errors(pred, gt):
    pred_long, pred_short = sorted(
        (float(pred[2]), float(pred[3])), reverse=True)
    gt_long, gt_short = sorted(
        (float(gt[2]), float(gt[3])), reverse=True)
    center = float(np.linalg.norm(pred[:2] - gt[:2]))
    angle = abs(math.degrees(float(angle_diff(
        np.asarray([pred[4]]), np.asarray([gt[4]]))[0])))
    return dict(
        center_error_px=center,
        long_relative_error=abs(pred_long - gt_long) / gt_long,
        short_relative_error=abs(pred_short - gt_short) / gt_short,
        diagonal_relative_error=(
            abs(math.hypot(pred_long, pred_short)
                - math.hypot(gt_long, gt_short))
            / math.hypot(gt_long, gt_short)),
        log_aspect_ratio_error=abs(math.log(
            (pred_long / pred_short) / (gt_long / gt_short))),
        angle_error_deg=angle,
        riou=float(compute_riou(pred[:5], gt[:5])))


ERROR_KEYS = (
    'center_error_px', 'long_relative_error', 'short_relative_error',
    'diagonal_relative_error', 'log_aspect_ratio_error',
    'angle_error_deg', 'riou')


def _mean(values):
    return None if not values else float(sum(values) / len(values))


def _pearson(xs, ys):
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _risk_summary(rows):
    result = dict(frame_count=len(rows))
    for key in ERROR_KEYS:
        result['mean_' + key] = _mean([row['errors'][key] for row in rows])
    result['anchor_source_counts'] = {
        source: sum(row['anchor_source'] == source for row in rows)
        for source in ('k1', 'dino_fallback')}
    result['domain_counts'] = {
        domain: sum(row['domain'] == domain for row in rows)
        for domain in ('real', 'sim')}
    return result


def _risk_coverage(rows, coverages):
    ordered = sorted(rows, key=lambda row: (
        -row['anchor_score'], row['frame_key']))
    curves = []
    for coverage in coverages:
        retained = max(1, int(round(len(ordered) * float(coverage))))
        selected = ordered[:retained]
        item = dict(
            target_coverage=float(coverage), retained_count=retained,
            realized_coverage=float(retained / len(ordered)),
            minimum_retained_score=float(selected[-1]['anchor_score']))
        item.update(_risk_summary(selected))
        curves.append(item)
    return curves


def _threshold_support(rows, thresholds):
    support = {}
    for key, values in thresholds.items():
        support[key] = [{
            'threshold': float(threshold),
            'exceedance_count': sum(
                row['errors'][key] > float(threshold) for row in rows),
            'frame_count': len(rows)
        } for threshold in values]
    return support


def _json_box(box):
    return None if box is None else [float(value) for value in box]


def _write_exact(path, payload):
    raw = (json.dumps(payload, indent=2, sort_keys=True,
                      ensure_ascii=False) + '\n').encode('utf-8')
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != raw:
        raise RuntimeError(
            'Refusing to overwrite different output: ' + os.fspath(output))
    if not output.exists():
        output.write_bytes(raw)
    return _identity(output)


def build(args):
    contract_id, contract = _read_json(args.contract, CONTRACT_PROTOCOL)
    expected = contract['expected_inputs']
    scope = contract['source_val_scope']
    frame_count = int(scope['frame_count'])

    promoted_id = _identity(args.promoted_refiner)
    promotion_id, promotion = _read_json(args.promotion_report)
    gate_id, gate = _read_json(args.source_gate)
    k1_checkpoint_id = _identity(args.k1_checkpoint)
    base_id, base_boxes = _load_result_boxes(
        args.base_v3_results, frame_count)
    k1_id, k1_boxes = _load_result_boxes(args.k1_results, frame_count)
    for identity, key, role in (
            (promoted_id, 'promoted_refiner_sha256', 'promoted refiner'),
            (promotion_id, 'promotion_report_sha256', 'promotion report'),
            (gate_id, 'source_gate_sha256', 'source gate'),
            (base_id, 'base_v3_results_sha256', 'Base V3 results'),
            (k1_checkpoint_id, 'k1_checkpoint_sha256', 'K1 checkpoint'),
            (k1_id, 'k1_results_sha256', 'K1 results')):
        _require_hash(identity, expected[key], role)

    if (promotion.get('evidence_boundary') != 'source_gate_only'
            or promotion.get('fixed_test_read') is not False
            or promotion.get('target_data_read') is not False
            or promotion.get('output', {}).get('checkpoint_sha256')
            != promoted_id['sha256']
            or promotion.get('input', {}).get('candidate_results_sha256')
            != base_id['sha256']):
        raise RuntimeError('Promotion report does not bind the supplied model')
    if (gate.get('evidence_boundary') != 'source_val_only'
            or gate.get('fixed_test_read') is not False
            or gate.get('target_data_read') is not False
            or gate.get('input', {}).get('candidate_results_sha256')
            != base_id['sha256']
            or gate.get('input', {}).get('sym_reference_results_sha256')
            != k1_id['sha256']
            or gate.get('input', {}).get('source_val_audit_sha256')
            != expected['source_val_audit_sha256']):
        raise RuntimeError('Source gate does not bind the supplied inputs')

    ann_id, annotations = _load_annotations(
        args.ann_dir, frame_count, scope['domain_counts'])
    frame_keys = [row['frame_key'] for row in annotations]
    audit_id, audit, audit_by_key = _load_dino_audit(
        args.source_val_audit, frame_keys)
    _require_hash(
        audit_id, expected['source_val_audit_sha256'], 'source-val audit')

    rows = []
    legacy_sym_exact_k1 = 0
    legacy_sym_comparable = 0
    for index, meta in enumerate(annotations):
        key = meta['frame_key']
        base = base_boxes[index]
        k1 = k1_boxes[index]
        dino = audit_by_key[key]['dino']
        if base is None:
            raise RuntimeError('Base V3 prediction is missing for ' + key)
        if k1 is not None:
            anchor_source = 'k1'
            anchor = k1
        elif dino is not None:
            anchor_source = 'dino_fallback'
            anchor = dino
        else:
            raise RuntimeError('No valid Base V3 anchor for ' + key)
        legacy = audit_by_key[key]['legacy_sym_eood_box']
        if legacy is not None and k1 is not None:
            legacy_array = np.asarray(legacy, dtype=np.float64).reshape(-1)
            if legacy_array.size >= 5 and np.isfinite(legacy_array[:5]).all():
                legacy_sym_comparable += 1
                legacy_sym_exact_k1 += int(np.array_equal(
                    legacy_array[:5], k1[:5]))
        rows.append(dict(
            frame_key=key, domain=meta['domain'], sequence=meta['sequence'],
            frame=meta['frame'], anchor_source=anchor_source,
            anchor_score=float(anchor[5]),
            base_v3_box=_json_box(base[:5]),
            base_v3_final_score=float(base[5]),
            k1_box=_json_box(None if k1 is None else k1[:5]),
            k1_score=None if k1 is None else float(k1[5]),
            dino_box=_json_box(None if dino is None else dino[:5]),
            dino_score=None if dino is None else float(dino[5]),
            gt_box=_json_box(meta['gt']),
            errors=_component_errors(base[:5], meta['gt'])))

    source_counts = {
        source: sum(row['anchor_source'] == source for row in rows)
        for source in ('k1', 'dino_fallback')}
    if (len(rows) != scope['expected_base_v3_present']
            or source_counts['k1'] != scope['expected_k1_present']
            or source_counts['dino_fallback']
            != scope['expected_dino_fallback']):
        raise RuntimeError(
            'Actual Base V3 branch counts violate the frozen contract')

    final_scores = [row['base_v3_final_score'] for row in rows]
    exact_one = sum(score == 1.0 for score in final_scores)
    if exact_one < frame_count - 1:
        raise RuntimeError(
            'Base V3 final-score saturation evidence unexpectedly changed')

    score_cfg = contract['score_baseline']
    domains = {
        domain: [row for row in rows if row['domain'] == domain]
        for domain in ('real', 'sim')}
    correlation_keys = (
        'center_error_px', 'long_relative_error',
        'short_relative_error', 'angle_error_deg', 'riou')

    def correlations(selected):
        scores = [row['anchor_score'] for row in selected]
        return {
            key: _pearson(
                scores, [row['errors'][key] for row in selected])
            for key in correlation_keys}

    fallback_count = source_counts['dino_fallback']
    min_fallback = int(
        score_cfg['minimum_fallback_samples_for_cross_source_calibration_claim'])
    output = dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        fixed_test_read=False,
        target_data_read=False,
        decision='ESTABLISH_SOURCE_VAL_SCORE_ONLY_BASELINE',
        claim_status=dict(
            score_only_baseline_established=True,
            calibrated_reliability_established=False,
            cross_source_score_calibration_supported=(
                fallback_count >= min_fallback),
            unknown_sequence_generalization_established=False,
            reason='Only {} DINO fallback frames; source scores remain '
                    'uncalibrated and are not yet a reliability method.'.format(
                        fallback_count)),
        inputs=dict(
            contract=contract_id, promoted_refiner=promoted_id,
            promotion_report=promotion_id, source_gate=gate_id,
            base_v3_results=base_id, k1_checkpoint=k1_checkpoint_id,
            k1_results=k1_id, source_val_audit=audit_id,
            annotations=ann_id),
        provenance_checks=dict(
            promotion_and_gate_bound=True,
            actual_anchor_policy='k1_present_else_native_dino',
            legacy_union_selected_source_used=False,
            legacy_union_sym_eood_box_used=False,
            legacy_union_sym_eood_box_comparable_count=(
                legacy_sym_comparable),
            legacy_union_sym_eood_box_exact_k1_count=legacy_sym_exact_k1,
            audit_record_count=len(audit.get('records') or [])),
        availability=dict(
            frame_count=len(rows), anchor_source_counts=source_counts,
            dino_present=sum(row['dino_box'] is not None for row in rows),
            base_v3_present=len(rows)),
        final_score_audit=dict(
            role='prohibited_as_confidence',
            exact_one_count=exact_one,
            unique_count=len(set(final_scores)),
            minimum=float(min(final_scores)), maximum=float(max(final_scores))),
        anchor_score_summary=dict(
            minimum=float(min(row['anchor_score'] for row in rows)),
            maximum=float(max(row['anchor_score'] for row in rows)),
            quantiles={str(q): float(np.quantile(
                [row['anchor_score'] for row in rows], q))
                for q in (0.0, 0.01, 0.05, 0.1, 0.25, 0.5,
                          0.75, 0.9, 0.95, 0.99, 1.0)}),
        all_frames_summary=_risk_summary(rows),
        anchor_score_error_pearson=dict(
            all=correlations(rows),
            real=correlations(domains['real']),
            sim=correlations(domains['sim'])),
        risk_coverage=dict(
            global_rank=_risk_coverage(
                rows, score_cfg['coverage_targets']),
            within_domain_rank={
                domain: _risk_coverage(
                    selected, score_cfg['coverage_targets'])
                for domain, selected in domains.items()}),
        component_error_support={
            name: _threshold_support(selected, score_cfg[
                'component_error_thresholds'])
            for name, selected in [('all', rows)] + list(domains.items())},
        records=rows)
    return output


def main():
    args = parse_args()
    payload = build(args)
    identity = _write_exact(args.out_json, payload)
    print(json.dumps(dict(
        protocol=payload['protocol'], decision=payload['decision'],
        output=identity, availability=payload['availability'],
        claim_status=payload['claim_status']), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
