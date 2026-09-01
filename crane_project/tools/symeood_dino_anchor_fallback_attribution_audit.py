"""Attribute K1 anchor-lock and DINO fallback failures without inference.

The audit consumes frozen per-frame result PKLs, a complete all-lane audit,
and DOTA annotations.  It does not run either detector, train a parameter,
search a threshold, or create a runtime routing rule.  Its bounded purpose is
to separate two structural failure mechanisms:

* K1 is absent and the output changes to a native-DINO fallback, which may
  preserve detection while discontinuously changing size or angle; and
* K1 is present but geometrically wrong, so a missing-only fallback remains
  locked to the wrong K1 anchor even when native DINO would be correct.

Ground-truth rescue streams are explicitly diagnostic oracles.  Sequence and
frame identifiers are used only to reconstruct contiguous temporal metrics.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from crane_project.tools import (
    symeood_dino_application_domain_v4_audit as base)
from crane_project.tools.eval_crane_offline import (
    METRIC_PROTOCOL_VERSION, angle_diff, compute_riou, obb_diag,
    parse_dota_txt, parse_seq_frame)


PROTOCOL = 'k1_anchor_fallback_attribution_audit_v1'
ROLE_FRAME_COUNTS = dict(base.ROLE_FRAME_COUNTS)
MODE_NAMES = ('full', 'current_only', 'center_only', 'k1_identity')
ANGLE_LIMIT_DEG = 35.0
IOU_THRESHOLD = 0.5
IDENTITY_TOLERANCE = 1e-4


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Attribute K1 anchor-lock and DINO-fallback failures from '
            'existing result PKLs and all-lane JSON'))
    parser.add_argument('--audit-json', required=True)
    parser.add_argument('--k1-results', required=True)
    parser.add_argument('--full-results', required=True)
    parser.add_argument('--current-only-results', required=True)
    parser.add_argument('--center-only-results', required=True)
    parser.add_argument('--k1-identity-results', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument('--split', required=True, choices=['val', 'test'])
    parser.add_argument(
        '--evidence-role', required=True,
        choices=sorted(ROLE_FRAME_COUNTS))
    parser.add_argument('--require-frame-count', type=int)
    parser.add_argument('--measurement-validity-json')
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _read_bytes(path):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        raw = handle.read()
    return absolute, raw


def _copy_box(box):
    if box is None:
        return None
    return np.asarray(box, dtype=np.float64).reshape(-1)[:6].copy()


def _result_boxes(path, expected_count):
    absolute, raw = _read_bytes(path)
    payload = pickle.loads(raw)
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise RuntimeError(
            '{} must contain exactly {} frames'.format(
                absolute, expected_count))
    boxes = []
    for index, result in enumerate(payload):
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise RuntimeError(
                'Invalid single-class result at frame {}'.format(index))
        detections = np.asarray(result[0], dtype=np.float64)
        if detections.size == 0:
            boxes.append(None)
            continue
        if detections.ndim != 2 or detections.shape[1] < 6:
            raise RuntimeError(
                'Invalid OBB result shape at frame {}'.format(index))
        if detections.shape[0] != 1:
            raise RuntimeError(
                'Expected exactly one selected OBB at frame {}'.format(
                    index))
        box = detections[0, :6].copy()
        if (not np.isfinite(box).all() or np.any(box[2:4] <= 0.0)):
            raise RuntimeError(
                'Invalid selected OBB at frame {}'.format(index))
        boxes.append(box)
    return absolute, raw, boxes


def _ordered_annotations(split_root, expected_count):
    annotation_dir = Path(split_root) / 'annfiles'
    paths = sorted(annotation_dir.glob('*.txt'))
    if len(paths) != expected_count:
        raise RuntimeError(
            'Expected {} annotations, got {}'.format(
                expected_count, len(paths)))
    output = []
    for path in paths:
        boxes = parse_dota_txt(os.fspath(path))
        if len(boxes) != 1:
            raise RuntimeError(
                'Each frame must contain exactly one GT: {}'.format(path))
        domain, seq_id, frame = parse_seq_frame(os.fspath(path))
        output.append(dict(
            frame_key=path.stem,
            sequence='{}_{}'.format(domain, seq_id),
            domain=domain, seq_id=seq_id, frame=int(frame),
            gt=np.asarray(boxes[0], dtype=np.float64)))
    return output


def _indexed_lanes(payload, ordered_keys):
    records = list(payload.get('records') or [])
    indexed = {}
    sym_key = base._resolve_sym_key(records)
    for record in records:
        key = Path(record.get('filename', '')).stem
        if not key or key in indexed:
            raise RuntimeError('Invalid or duplicate all-lane frame key')
        indexed[key] = record
    if set(indexed) != set(ordered_keys):
        raise RuntimeError('All-lane audit and annotations disagree')
    return {
        key: dict(
            dino=_copy_box(base._box(indexed[key], 'dino_native_box')),
            sym=_copy_box(base._box(indexed[key], sym_key)))
        for key in ordered_keys}, sym_key


def _riou(box, gt):
    if box is None:
        return 0.0
    return float(compute_riou(box[:5], gt[:5]))


def _hit(box, gt):
    return bool(_riou(box, gt) >= IOU_THRESHOLD)


def _box_equivalent(first, second, tolerance=IDENTITY_TOLERANCE):
    if first is None or second is None:
        return first is None and second is None
    if np.linalg.norm(first[:2] - second[:2]) > tolerance:
        return False
    return compute_riou(first[:5], second[:5]) >= 1.0 - tolerance


def _source_for_identity(k1, dino):
    if k1 is not None:
        return 'k1_anchor'
    if dino is not None:
        return 'dino_fallback'
    return 'missing'


def _rows(metadata, lanes, streams):
    output = []
    for index, meta in enumerate(metadata):
        lane = lanes[meta['frame_key']]
        k1 = streams['k1_reference'][index]
        dino = lane['dino']
        boxes = {
            name: streams[name][index]
            for name in MODE_NAMES}
        output.append(dict(
            **meta, k1=k1, dino=dino, sym=lane['sym'], boxes=boxes,
            identity_source=_source_for_identity(k1, dino),
            k1_riou=_riou(k1, meta['gt']),
            dino_riou=_riou(dino, meta['gt']),
            report=dict(
                frame_key=meta['frame_key'], sequence=meta['sequence'],
                frame=int(meta['frame']), domain=meta['domain'],
                k1_present=k1 is not None,
                k1_riou=_riou(k1, meta['gt']),
                k1_hit=_hit(k1, meta['gt']),
                dino_present=dino is not None,
                dino_riou=_riou(dino, meta['gt']),
                dino_hit=_hit(dino, meta['gt']),
                identity_source=_source_for_identity(k1, dino),
                mode_riou={
                    name: _riou(box, meta['gt'])
                    for name, box in boxes.items()},
                mode_hit={
                    name: _hit(box, meta['gt'])
                    for name, box in boxes.items()})))
    return output


def _temporal_metrics(rows, box_getter):
    records = []
    for row in rows:
        box = box_getter(row)
        records.append(dict(
            domain=row['domain'], seq_id=row['seq_id'],
            frame_id=int(row['frame']),
            pred_box=None if box is None else box[:5],
            gt_box=row['gt'][:5],
            score=0.0 if box is None else float(box[5]),
            plc_rope=None))
    return base._offline_metrics(records)


def _mode_metrics(rows):
    output = {
        'k1_reference': _temporal_metrics(rows, lambda row: row['k1']),
        'dino_native': _temporal_metrics(rows, lambda row: row['dino'])}
    output.update({
        name: _temporal_metrics(
            rows, lambda row, mode=name: row['boxes'][mode])
        for name in MODE_NAMES})
    return output


def _component_contract(rows):
    checks = Counter()
    failures = []
    for row in rows:
        full = row['boxes']['full']
        center = row['boxes']['center_only']
        identity = row['boxes']['k1_identity']
        expected_identity = (
            row['k1'] if row['k1'] is not None else row['dino'])
        checks['frame_count'] += 1
        if _box_equivalent(identity, expected_identity):
            checks['identity_matches_k1_else_dino'] += 1
        else:
            failures.append('{}:identity'.format(row['frame_key']))
        if full is None or center is None:
            if full is None and center is None:
                checks['full_center_presence_match'] += 1
            else:
                failures.append('{}:full_center_presence'.format(
                    row['frame_key']))
        elif np.linalg.norm(full[:2] - center[:2]) <= IDENTITY_TOLERANCE:
            checks['full_center_coordinates_match'] += 1
        else:
            failures.append('{}:full_center_coordinates'.format(
                row['frame_key']))
        if center is None or identity is None:
            if center is None and identity is None:
                checks['center_identity_presence_match'] += 1
            else:
                failures.append('{}:center_identity_presence'.format(
                    row['frame_key']))
        elif _box_equivalent(
                np.asarray([0.0, 0.0, *center[2:6]]),
                np.asarray([0.0, 0.0, *identity[2:6]])):
            checks['center_identity_size_angle_match'] += 1
        else:
            failures.append('{}:center_identity_size_angle'.format(
                row['frame_key']))
    return dict(
        tolerance=IDENTITY_TOLERANCE,
        counts=dict(checks),
        failure_count=len(failures),
        first_failures=failures[:20],
        passed=not failures)


def _contiguous_pairs(rows):
    ordered = sorted(rows, key=lambda row: (
        row['domain'], row['seq_id'], int(row['frame'])))
    previous = None
    for row in ordered:
        if previous is not None and (
                previous['domain'] == row['domain']
                and previous['seq_id'] == row['seq_id']
                and int(row['frame']) == int(previous['frame']) + 1):
            yield previous, row
        previous = row


def _transition_attribution(rows):
    buckets = defaultdict(list)
    for previous, current in _contiguous_pairs(rows):
        first = previous['boxes']['k1_identity']
        second = current['boxes']['k1_identity']
        if first is None or second is None:
            continue
        transition = '{}->{}'.format(
            previous['identity_source'], current['identity_source'])
        dfr = abs(obb_diag(second) - obb_diag(first)) / max(
            obb_diag(first), 1e-9)
        angle = abs(float(angle_diff(
            np.asarray([second[4]]), np.asarray([first[4]]))[0]))
        aci = float(np.clip(
            1.0 - math.degrees(angle) / ANGLE_LIMIT_DEG, 0.0, 1.0))
        buckets[(current['domain'], transition)].append((dfr, aci))

    output = {}
    for domain in ('all', 'real', 'sim'):
        selected = defaultdict(list)
        for (item_domain, transition), values in buckets.items():
            if domain == 'all' or item_domain == domain:
                selected[transition].extend(values)
        total_dfr = sum(value[0] for values in selected.values()
                        for value in values)
        output[domain] = {}
        for transition, values in sorted(selected.items()):
            dfr_values = np.asarray(
                [value[0] for value in values], dtype=np.float64)
            aci_values = np.asarray(
                [value[1] for value in values], dtype=np.float64)
            dfr_sum = float(dfr_values.sum())
            output[domain][transition] = dict(
                transition_count=len(values),
                mean_dfr_fraction=float(dfr_values.mean()),
                p90_dfr_fraction=float(np.quantile(dfr_values, 0.9)),
                dfr_sum=dfr_sum,
                share_of_identity_dfr_sum=(
                    None if total_dfr <= 0.0 else dfr_sum / total_dfr),
                mean_aci=float(aci_values.mean()))
    return output


def _miss_runs(rows, mode):
    runs = []
    current = []
    previous = None
    ordered = sorted(rows, key=lambda row: (
        row['domain'], row['seq_id'], int(row['frame'])))
    for row in ordered:
        continuous = bool(
            previous is not None
            and previous['domain'] == row['domain']
            and previous['seq_id'] == row['seq_id']
            and int(row['frame']) == int(previous['frame']) + 1)
        miss = not _hit(row['boxes'][mode], row['gt'])
        if miss:
            if current and not continuous:
                runs.append(current)
                current = []
            current.append(row)
        elif current:
            runs.append(current)
            current = []
        previous = row
    if current:
        runs.append(current)
    return runs


def _failure_cause(row):
    if row['k1'] is not None:
        if _hit(row['k1'], row['gt']):
            return 'refiner_changed_correct_k1_to_miss'
        if _hit(row['dino'], row['gt']):
            return 'k1_present_wrong_anchor_lock_dino_hit'
        return 'k1_present_wrong_both_lanes_miss'
    if row['dino'] is not None:
        return ('dino_fallback_hit_but_output_miss'
                if _hit(row['dino'], row['gt'])
                else 'dino_fallback_wrong')
    return 'both_lanes_missing'


def _run_report(run, mode):
    return dict(
        domain=run[0]['domain'], sequence=run[0]['sequence'],
        start_frame=int(run[0]['frame']), end_frame=int(run[-1]['frame']),
        length=len(run),
        cause_counts=dict(Counter(_failure_cause(row) for row in run)),
        dino_hit_count=sum(_hit(row['dino'], row['gt']) for row in run),
        k1_present_count=sum(row['k1'] is not None for row in run),
        frames=[dict(
            frame=int(row['frame']), cause=_failure_cause(row),
            k1_riou=float(row['k1_riou']),
            dino_riou=float(row['dino_riou']),
            mode_riou=_riou(row['boxes'][mode], row['gt']))
            for row in run])


def _failure_attribution(rows):
    output = {}
    for mode in MODE_NAMES:
        runs = _miss_runs(rows, mode)
        reports = sorted(
            [_run_report(run, mode) for run in runs],
            key=lambda item: (-item['length'], item['sequence'],
                              item['start_frame']))
        output[mode] = dict(
            miss_run_count=len(reports),
            max_run_length=max([item['length'] for item in reports] or [0]),
            over_limit_run_count=sum(
                item['length'] > 5 for item in reports),
            longest_runs=reports[:10])
    return output


def _present_wrong_rescue_oracle(rows):
    boxes = []
    rescue_count = 0
    for row in rows:
        selected = row['boxes']['k1_identity']
        if (row['k1'] is not None
                and not _hit(row['k1'], row['gt'])
                and _hit(row['dino'], row['gt'])):
            selected = row['dino']
            rescue_count += 1
        boxes.append(_copy_box(selected))
    metrics = _temporal_metrics(
        [dict(row, oracle=box) for row, box in zip(rows, boxes)],
        lambda row: row['oracle'])
    return dict(
        non_deployable_gt_oracle=True,
        rescued_frame_count=rescue_count,
        metrics=metrics)


def _causal_geometry_preserving_fallback(rows, horizon=4):
    selected = {}
    states = {}
    fallback_count = 0
    history_geometry_count = 0
    for row in sorted(rows, key=lambda item: (
            item['domain'], item['seq_id'], int(item['frame']))):
        identity = (row['domain'], row['seq_id'])
        state = states.setdefault(identity, dict(
            previous_frame=None, k1_history=[]))
        if (state['previous_frame'] is not None
                and int(row['frame']) != state['previous_frame'] + 1):
            state['k1_history'][:] = []
        past = state['k1_history']
        if row['k1'] is not None:
            box = _copy_box(row['k1'])
            past.append((int(row['frame']), _copy_box(row['k1'])))
            past[:] = past[-horizon:]
        elif row['dino'] is not None:
            fallback_count += 1
            box = _copy_box(row['dino'])
            usable = [item for item in past
                      if int(row['frame']) - int(item[0]) <= horizon]
            if usable:
                box[2:5] = usable[-1][1][2:5]
                history_geometry_count += 1
        else:
            box = None
        state['previous_frame'] = int(row['frame'])
        selected[row['frame_key']] = box
    metrics = _temporal_metrics(
        rows, lambda row: selected[row['frame_key']])
    return dict(
        fixed_history_horizon=int(horizon),
        current_frame_dino_center_with_recent_k1_size_angle=True,
        dino_fallback_frame_count=fallback_count,
        recent_k1_geometry_used_count=history_geometry_count,
        native_dino_geometry_used_count=(
            fallback_count - history_geometry_count),
        uses_gt_for_selection=False,
        fixed_target_result_is_diagnostic_not_parameter_selection=True,
        metrics=metrics)


def _support_counts(rows):
    categories = Counter()
    for row in rows:
        k1_present = row['k1'] is not None
        k1_hit = _hit(row['k1'], row['gt'])
        dino_present = row['dino'] is not None
        dino_hit = _hit(row['dino'], row['gt'])
        if not k1_present:
            categories['k1_missing_dino_hit' if dino_hit
                       else 'k1_missing_dino_not_hit'] += 1
        elif k1_hit:
            categories['k1_present_hit'] += 1
        elif dino_hit:
            categories['k1_present_wrong_dino_hit'] += 1
        else:
            categories['k1_present_wrong_dino_not_hit'] += 1
        if not dino_present:
            categories['dino_missing'] += 1
    return dict(categories)


def _measurement_slice(manifest, manifest_bytes, audit_records, rows,
                       evidence_role):
    invalid, normalized = base._validate_measurement_validity(
        manifest, audit_records, evidence_role)
    kept = [row for row in rows
            if (row['sequence'], int(row['frame'])) not in invalid]
    excluded = [row for row in rows
                if (row['sequence'], int(row['frame'])) in invalid]
    return dict(
        use='evaluation_scope_only_never_model_input',
        input=dict(
            sha256=_sha256_bytes(manifest_bytes),
            normalized_sequences=normalized),
        kept_frame_count=len(kept),
        excluded_real_frame_count=len(excluded),
        mode_metrics=_mode_metrics(kept),
        transition_attribution=_transition_attribution(kept),
        failure_attribution=_failure_attribution(kept),
        eligible_for_primary_decision_override=False)


def audit_payload(payload, audit_bytes, split_root, evidence_role,
                  result_payloads, required_frame_count=None,
                  measurement_validity=None,
                  measurement_validity_bytes=None):
    records = list(payload.get('records') or [])
    expected = (ROLE_FRAME_COUNTS[evidence_role]
                if required_frame_count is None else
                int(required_frame_count))
    split_root = Path(split_root)
    sym_key, domains = base._validate(
        payload, records, split_root, evidence_role, expected)
    metadata = _ordered_annotations(split_root, expected)
    ordered_keys = [item['frame_key'] for item in metadata]
    lanes, indexed_sym_key = _indexed_lanes(payload, ordered_keys)
    if indexed_sym_key != sym_key:
        raise RuntimeError('SymEOOD lane-key validation disagrees')

    streams = {}
    inputs = {}
    for name, item in result_payloads.items():
        if isinstance(item, tuple) and len(item) == 3:
            path, raw, boxes = item
        else:
            path, raw, boxes = _result_boxes(item, expected)
        if len(boxes) != expected:
            raise RuntimeError('{} result count disagrees'.format(name))
        streams[name] = boxes
        inputs[name] = dict(
            path=os.path.abspath(os.fspath(path)),
            sha256=_sha256_bytes(raw), frame_count=len(boxes))
    required_streams = {'k1_reference', *MODE_NAMES}
    if set(streams) != required_streams:
        raise RuntimeError(
            'Result streams must be exactly {}'.format(
                sorted(required_streams)))

    rows = _rows(metadata, lanes, streams)
    component = _component_contract(rows)
    if not component['passed']:
        raise RuntimeError(
            'Component-mode contract failed: {}'.format(
                component['first_failures']))
    measurement = None
    if measurement_validity is not None:
        measurement = _measurement_slice(
            measurement_validity, measurement_validity_bytes,
            records, rows, evidence_role)

    return dict(
        protocol=PROTOCOL,
        metric_protocol_version=METRIC_PROTOCOL_VERSION,
        evidence_role=evidence_role,
        evidence_boundary=(
            'source_only_structural_support_audit'
            if evidence_role == 'source-val' else
            'fixed_target_posthoc_failure_attribution_not_model_selection'),
        input=dict(
            audit_json_sha256=_sha256_bytes(audit_bytes),
            audit_protocol=payload.get('protocol'),
            frame_count=len(rows), domain_counts=domains,
            sym_eood_box_key=sym_key,
            result_streams=inputs),
        audit_contract=dict(
            detector_forward=False, dino_detector_rerun=False,
            parameter_update=False, threshold_search=False,
            epoch_selection=False, checkpoint_selection=False,
            domain_routing=False, sequence_frame_slice_routing=False,
            sequence_frame_used_for_temporal_metrics_only=True,
            gt_rescue_is_non_deployable_oracle=True,
            fixed_target_cannot_authorize_runtime_policy=True,
            eligible_for_unknown_sequence_claim=False),
        component_mode_contract=component,
        support_counts={
            name: _support_counts(group)
            for name, group in (
                ('all', rows),
                ('real', [row for row in rows if row['domain'] == 'real']),
                ('sim', [row for row in rows if row['domain'] == 'sim']))},
        stream_metrics=_mode_metrics(rows),
        identity_transition_attribution=_transition_attribution(rows),
        failure_attribution=_failure_attribution(rows),
        diagnostic_capacity=dict(
            present_wrong_dino_rescue_oracle=(
                _present_wrong_rescue_oracle(rows)),
            causal_geometry_preserving_fallback_h4=(
                _causal_geometry_preserving_fallback(rows, horizon=4))),
        measurement_validity=measurement,
        frame_rows=[row['report'] for row in rows],
        next_stage=dict(
            recommended_action=(
                'USE_SOURCE_VAL_SUPPORT_TO_PREREGISTER_NEW_STRUCTURE'
                if evidence_role == 'source-val' else
                'DIAGNOSTIC_ONLY_RETURN_TO_SOURCE_BEFORE_NEW_STRUCTURE'),
            continue_component_mask_tuning=False,
            continue_same_residual_head_tuning=False,
            runtime_quality_gate_implemented=False,
            eligible_for_runtime_policy=False),
        decision=(
            'SOURCE_VAL_ANCHOR_FALLBACK_ATTRIBUTION_COMPLETE'
            if evidence_role == 'source-val' else
            'FIXED_TARGET_ANCHOR_FALLBACK_ATTRIBUTION_DIAGNOSTIC_ONLY'))


def main():
    args = parse_args()
    audit_path, audit_bytes = _read_bytes(args.audit_json)
    payload = json.loads(audit_bytes.decode('utf-8'))
    expected = (ROLE_FRAME_COUNTS[args.evidence_role]
                if args.require_frame_count is None else
                int(args.require_frame_count))
    paths = dict(
        k1_reference=args.k1_results,
        full=args.full_results,
        current_only=args.current_only_results,
        center_only=args.center_only_results,
        k1_identity=args.k1_identity_results)
    result_payloads = {
        name: _result_boxes(path, expected)
        for name, path in paths.items()}
    manifest = None
    manifest_bytes = None
    if args.measurement_validity_json:
        _, manifest_bytes = _read_bytes(args.measurement_validity_json)
        manifest = json.loads(manifest_bytes.decode('utf-8'))
    result = audit_payload(
        payload, audit_bytes, Path(args.data_root) / args.split,
        args.evidence_role, result_payloads,
        required_frame_count=args.require_frame_count,
        measurement_validity=manifest,
        measurement_validity_bytes=manifest_bytes)
    result['input']['audit_json_path'] = audit_path
    base._write_json_atomic(args.out_json, result)
    print('[anchor-fallback] role={}'.format(result['evidence_role']))
    print('[anchor-fallback] frames={}'.format(result['input']['frame_count']))
    print('[anchor-fallback] support={}'.format(result['support_counts']))
    print('[anchor-fallback] transitions={}'.format(
        result['identity_transition_attribution']))
    print('[anchor-fallback] max_runs={}'.format({
        name: item['max_run_length']
        for name, item in result['failure_attribution'].items()}))
    print('[anchor-fallback] capacity={}'.format(
        result['diagnostic_capacity']))
    print('[anchor-fallback] decision={}'.format(result['decision']))


if __name__ == '__main__':
    main()
