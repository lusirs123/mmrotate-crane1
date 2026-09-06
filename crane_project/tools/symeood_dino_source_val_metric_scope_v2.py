#!/usr/bin/env python3
"""Audit source-val metric scopes from existing Base-V3 prediction streams.

This is a supplementary, post-gate audit.  It never changes the official
all-frame gate and never exposes operation phase to the detector.  A draft
manifest can be materialized first, reviewed against video/simulator events,
and then frozen before the scoped report is generated.
"""

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from crane_project.tools.eval_crane_offline import (
    CraneOfflineEvaluator, parse_dota_txt)
from crane_project.tools.symeood_dino_dual_tower_v2_audit import _load_results


PROTOCOL = 'base_v3_source_val_metric_scope_audit_v2'
CONTRACT_PROTOCOL = 'base_v3_source_val_metric_scope_contract_v2'
ATTRIBUTION_PROTOCOL = (
    'base_v3_current_only_source_val_failure_attribution_v1')
MANIFEST_PROTOCOL = 'source_val_operation_metric_scope_manifest_v2'
MANIFEST_DRAFT_STATUS = 'DRAFT_REQUIRES_OPERATION_PHASE_REVIEW'
MANIFEST_FROZEN_STATUS = 'FROZEN_BEFORE_SCOPED_METRIC_OUTPUT'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attribution-report', required=True)
    parser.add_argument('--contract')
    parser.add_argument('--scope-manifest')
    parser.add_argument('--out-json')
    parser.add_argument('--materialize-scope-template')
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


def _json(path, protocol=None):
    identity = _identity(path)
    with open(identity['path'], 'r', encoding='utf-8') as stream:
        payload = json.load(stream)
    if protocol is not None and payload.get('protocol') != protocol:
        raise RuntimeError(
            'Unexpected protocol in {}: {!r}'.format(
                identity['path'], payload.get('protocol')))
    return identity, payload


def _write_exact(path, payload):
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, ensure_ascii=False) + '\n').encode(
        'utf-8')
    if output.exists() and output.read_bytes() != raw:
        raise RuntimeError('Refusing to overwrite different output: '
                           + os.fspath(output))
    if not output.exists():
        output.write_bytes(raw)
    return _identity(output)


def _canonical_frame_set_sha256(rows):
    keys = [row['frame_key'] for row in rows]
    raw = json.dumps(
        keys, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _sequence_key(row):
    return '{}_{}'.format(row['domain'], row['sequence'])


def materialize_template(attribution, output):
    rows = list(attribution.get('per_frame') or [])
    if len(rows) != 738 or len({row['frame_key'] for row in rows}) != 738:
        raise RuntimeError('Attribution report must contain exactly 738 frames')
    grouped = defaultdict(list)
    for row in rows:
        grouped[_sequence_key(row)].append(int(row['frame']))
    sequences = {}
    for key, frames in sorted(grouped.items()):
        sequences[key] = dict(
            observed_frame_count=len(frames),
            first_observed_frame=min(frames),
            last_observed_frame=max(frames),
            default_perception_valid=True,
            default_control_valid=True,
            override_intervals=[])
    template = dict(
        protocol=MANIFEST_PROTOCOL,
        status=MANIFEST_DRAFT_STATUS,
        evidence_role='official_source_val_738_supplementary_diagnostic',
        selection_basis=(
            'review_from_operation_phase_evidence_without_model_outputs'),
        instructions=(
            'Review every sequence, add only physically justified override '
            'intervals, then set status to FROZEN_BEFORE_SCOPED_METRIC_OUTPUT.'),
        frame_set_sha256=_canonical_frame_set_sha256(rows),
        frame_count=len(rows),
        sequences=sequences,
        confirmed_valid_intervals=[dict(
            sequence='sim_seq10', start_frame=207, end_frame=221,
            perception_valid=True, control_valid=True,
            reason='verified_non_material_contact_operation')],
        operation_phase_used_as_model_input=False,
        official_gate_override=False,
        target_data_read=False,
        fixed_test_read=False)
    return _write_exact(output, template)


def _validate_contract(contract):
    if (contract.get('status') !=
            'supplementary_metric_scope_frozen_before_output'):
        raise RuntimeError('Metric-scope contract is not frozen')
    if (contract.get('training') is not False
            or contract.get('target_data_read') is not False
            or contract.get('fixed_test_read') is not False):
        raise RuntimeError('Metric-scope evidence boundary changed')
    prohibited = dict(contract.get('prohibited_uses') or {})
    required = (
        'official_gate_override', 'threshold_tuning', 'epoch_selection',
        'checkpoint_promotion', 'model_routing',
        'training_authorization', 'fixed_test_authorization')
    if any(prohibited.get(key) is not True for key in required):
        raise RuntimeError('Metric-scope prohibited uses changed')


def _expand_scope(rows, manifest, contract):
    if manifest.get('status') != MANIFEST_FROZEN_STATUS:
        raise RuntimeError(
            'Scope manifest is still a draft; review and freeze it first')
    if (manifest.get('evidence_role') !=
            'official_source_val_738_supplementary_diagnostic'
            or manifest.get('selection_basis') !=
            'review_from_operation_phase_evidence_without_model_outputs'):
        raise RuntimeError('Scope manifest provenance is invalid')
    if (manifest.get('frame_count') != len(rows)
            or manifest.get('frame_set_sha256') !=
            _canonical_frame_set_sha256(rows)):
        raise RuntimeError('Scope manifest frame set differs from source-val')
    if (manifest.get('operation_phase_used_as_model_input') is not False
            or manifest.get('official_gate_override') is not False
            or manifest.get('target_data_read') is not False
            or manifest.get('fixed_test_read') is not False):
        raise RuntimeError('Scope manifest violates evidence isolation')

    grouped = defaultdict(list)
    for row in rows:
        grouped[_sequence_key(row)].append(row)
    specs = manifest.get('sequences')
    if not isinstance(specs, dict) or set(specs) != set(grouped):
        raise RuntimeError('Scope manifest must enumerate every sequence')

    scope = {}
    reason_counts = Counter()
    for key, sequence_rows in grouped.items():
        spec = specs[key]
        if (spec.get('observed_frame_count') != len(sequence_rows)
                or spec.get('default_perception_valid') is not True
                or spec.get('default_control_valid') is not True):
            raise RuntimeError('Invalid sequence scope defaults for ' + key)
        overrides = list(spec.get('override_intervals') or [])
        previous_end = None
        for interval in overrides:
            start, end = int(interval['start_frame']), int(interval['end_frame'])
            if start > end or (previous_end is not None and start <= previous_end):
                raise RuntimeError('Overlapping/invalid scope intervals for ' + key)
            if not str(interval.get('reason', '')).strip():
                raise RuntimeError('Scope override requires a reason')
            selected = [row for row in sequence_rows
                        if start <= int(row['frame']) <= end]
            if not selected:
                raise RuntimeError('Scope interval selects no frames: ' + key)
            perception = bool(interval['perception_valid'])
            control = bool(interval['control_valid'])
            if control and not perception:
                raise RuntimeError('control_valid requires perception_valid')
            for row in selected:
                scope[row['frame_key']] = dict(
                    perception_valid=perception, control_valid=control,
                    reason=str(interval['reason']))
                reason_counts[str(interval['reason'])] += 1
            previous_end = end
        for row in sequence_rows:
            scope.setdefault(row['frame_key'], dict(
                perception_valid=True, control_valid=True,
                reason='default_valid'))

    required = list(contract.get('required_confirmed_valid_intervals') or [])
    confirmed = list(manifest.get('confirmed_valid_intervals') or [])
    for item in required:
        if item not in confirmed:
            raise RuntimeError('Required confirmed-valid interval is missing')
        key = str(item['sequence'])
        selected = [row for row in rows if _sequence_key(row) == key
                    and int(item['start_frame']) <= int(row['frame'])
                    <= int(item['end_frame'])]
        if not selected or any(
                not scope[row['frame_key']]['perception_valid']
                or not scope[row['frame_key']]['control_valid']
                for row in selected):
            raise RuntimeError('Confirmed-valid interval was excluded: ' + key)
    return scope, dict(reason_counts)


def _verify_bound_input(recorded, role):
    if not isinstance(recorded, dict):
        raise RuntimeError('Missing input identity for ' + role)
    observed = _identity(recorded.get('path'))
    if observed['sha256'] != recorded.get('sha256'):
        raise RuntimeError('Input identity mismatch for ' + role)
    return observed


def _records(rows, boxes, gt_boxes, scope, field):
    selected = []
    for index, row in enumerate(rows):
        if not scope[row['frame_key']][field]:
            continue
        selected.append(dict(
            domain=row['domain'], seq_id=row['sequence'],
            frame_id=int(row['frame']), pred_box=boxes[index],
            gt_box=gt_boxes[index], score=0.0))
    return selected


def _metric_subset(metrics, suffixes):
    return {
        key: value for key, value in metrics.items()
        if any(key.endswith('/' + suffix) for suffix in suffixes)}


def _pair_coverage(rows, boxes, scope):
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(row['domain'], row['sequence'])].append((row, boxes[index]))
    by_domain = {}
    for domain in ('real', 'sim'):
        contiguous = valid = missing = 0
        for (row_domain, _sequence), items in grouped.items():
            if row_domain != domain:
                continue
            ordered = sorted(items, key=lambda item: int(item[0]['frame']))
            missing += sum(
                scope[row['frame_key']]['control_valid'] and box is None
                for row, box in ordered)
            for (previous, previous_box), (current, current_box) in zip(
                    ordered, ordered[1:]):
                if (int(current['frame']) != int(previous['frame']) + 1
                        or not scope[previous['frame_key']]['control_valid']
                        or not scope[current['frame_key']]['control_valid']):
                    continue
                contiguous += 1
                if previous_box is not None and current_box is not None:
                    valid += 1
        by_domain[domain] = dict(
            contiguous_control_valid_gt_pair_count=contiguous,
            valid_prediction_pair_count=valid,
            dfr_valid_pair_count=valid,
            aci_valid_pair_count=valid,
            valid_pair_fraction=(
                0.0 if contiguous == 0 else valid / contiguous),
            missing_prediction_count=missing)
    return by_domain


def _angle_summary(rows, method, scope):
    selected = [row for row in rows
                if row['domain'] == 'sim'
                and scope[row['frame_key']]['perception_valid']]
    present_raw = [row[method]['raw_angle_error_deg'] for row in selected
                   if row[method]['prediction_present']]
    direct = [row[method]['raw_angle_error_deg'] for row in selected
              if row[method]['angle_metric_state'] ==
              'direct_periodic_angle_error']
    penalized = [row[method]['angle_metric_error_deg'] for row in selected]
    center_penalties = sum(
        row[method]['angle_metric_state'] == 'center_error_penalty'
        for row in selected)
    missing = sum(
        row[method]['angle_metric_state'] == 'missing_prediction_penalty'
        for row in selected)
    return dict(
        perception_valid_sim_frame_count=len(selected),
        penalized_A_RMSE_deg=(None if not penalized else math.sqrt(
            sum(value ** 2 for value in penalized) / len(penalized))),
        raw_angle_RMSE_present_predictions_deg=(
            None if not present_raw else math.sqrt(
                sum(value ** 2 for value in present_raw) / len(present_raw))),
        direct_angle_RMSE_center_lt_10px_deg=(
            None if not direct else math.sqrt(
                sum(value ** 2 for value in direct) / len(direct))),
        center_penalty_count=center_penalties,
        center_penalty_fraction=(
            0.0 if not selected else center_penalties / len(selected)),
        missing_prediction_count=missing)


def audit(contract_id, contract, attribution_id, attribution,
          manifest_id, manifest):
    _validate_contract(contract)
    if (attribution.get('decision') !=
            'CURRENT_ONLY_FAILURE_ATTRIBUTION_READY_NO_MODEL_CHANGE_AUTHORIZED'
            or attribution.get('fixed_test_read') is not False):
        raise RuntimeError('Metric-scope audit requires the frozen attribution')
    rows = list(attribution.get('per_frame') or [])
    if len(rows) != 738:
        raise RuntimeError('Official source-val attribution must have 738 rows')
    scope, reason_counts = _expand_scope(rows, manifest, contract)

    inputs = attribution.get('inputs') or {}
    full_id = _verify_bound_input(inputs.get('full_results'), 'full results')
    current_id = _verify_bound_input(
        inputs.get('current_only_results'), 'current-only results')
    _, full_boxes = _load_results(full_id['path'])
    _, current_boxes = _load_results(current_id['path'])
    if len(full_boxes) != len(rows) or len(current_boxes) != len(rows):
        raise RuntimeError('Result count differs from attribution rows')

    ann_root = Path(inputs.get('ann_dir', '')).resolve()
    ann_paths = sorted(ann_root.glob('*.txt'), key=lambda path: path.name)
    if [path.stem for path in ann_paths] != [row['frame_key'] for row in rows]:
        raise RuntimeError('Annotation order differs from attribution rows')
    gt_boxes = []
    for path in ann_paths:
        boxes = parse_dota_txt(os.fspath(path))
        if len(boxes) != 1:
            raise RuntimeError('Expected one GT box: ' + os.fspath(path))
        gt_boxes.append(boxes[0])

    methods = {}
    for method, boxes in (('full', full_boxes),
                          ('current_only', current_boxes)):
        perception_records = _records(
            rows, boxes, gt_boxes, scope, 'perception_valid')
        control_records = _records(
            rows, boxes, gt_boxes, scope, 'control_valid')
        perception_metrics = CraneOfflineEvaluator(
            mode='test').evaluate_records(perception_records)
        control_metrics = CraneOfflineEvaluator(
            mode='test').evaluate_records(control_records)
        methods[method] = dict(
            perception_valid_metrics=_metric_subset(
                perception_metrics,
                ('A-RMSE(deg)', 'R_center(%)', 'mean_RIoU')),
            control_valid_metrics=_metric_subset(
                control_metrics,
                ('DFR(%/frame)', 'ACI', 'TDR_w10(%)',
                 'MCML_max(frames)', 'MCML_mean(frames)',
                 'MCML_pass(limit=5)', 'MRF(frames)')),
            angle_decomposition=_angle_summary(rows, method, scope))
        methods[method]['temporal_pair_coverage'] = _pair_coverage(
            rows, boxes, scope)

    counts = {}
    for domain in ('real', 'sim'):
        domain_rows = [row for row in rows if row['domain'] == domain]
        counts[domain] = dict(
            all_frame_count=len(domain_rows),
            perception_valid_count=sum(
                scope[row['frame_key']]['perception_valid']
                for row in domain_rows),
            control_valid_count=sum(
                scope[row['frame_key']]['control_valid']
                for row in domain_rows))

    return dict(
        protocol=PROTOCOL,
        evidence_role='official_source_val_738_supplementary_diagnostic',
        inputs=dict(contract=contract_id, attribution=attribution_id,
                    scope_manifest=manifest_id, full_results=full_id,
                    current_only_results=current_id,
                    annotation_dir=os.fspath(ann_root)),
        scope_coverage=dict(by_domain=counts,
                            override_reason_counts=reason_counts),
        methods=methods,
        confirmed_valid_interval_audit=dict(
            sim_seq10_207_221_perception_valid=True,
            sim_seq10_207_221_control_valid=True),
        interpretation_policy=dict(
            official_all_frame_gate_unchanged=True,
            current_only_failure_unchanged=True,
            scoped_metrics_are_supplementary=True,
            operation_phase_used_as_model_input=False,
            threshold_or_epoch_reselection_authorized=False,
            training_authorized=False),
        training_run=False, optimizer_steps=0,
        target_data_read=False, fixed_test_read=False,
        decision=(
            'SOURCE_VAL_METRIC_SCOPE_V2_READY_SUPPLEMENTARY_ONLY_'
            'NO_GATE_OVERRIDE'))


def main():
    args = parse_args()
    attribution_id, attribution = _json(
        args.attribution_report, ATTRIBUTION_PROTOCOL)
    if args.materialize_scope_template:
        output = materialize_template(
            attribution, args.materialize_scope_template)
        print('[metric-scope-v2] template={}'.format(output['path']))
        print('[metric-scope-v2] decision=' + MANIFEST_DRAFT_STATUS)
        return
    if not args.contract or not args.scope_manifest or not args.out_json:
        raise RuntimeError(
            'Audit mode requires --contract, --scope-manifest and --out-json')
    contract_id, contract = _json(args.contract, CONTRACT_PROTOCOL)
    manifest_id, manifest = _json(args.scope_manifest, MANIFEST_PROTOCOL)
    report = audit(contract_id, contract, attribution_id, attribution,
                   manifest_id, manifest)
    output = _write_exact(args.out_json, report)
    print('[metric-scope-v2] output={}'.format(output['path']))
    print('[metric-scope-v2] decision={}'.format(report['decision']))
    for method in ('full', 'current_only'):
        angle = report['methods'][method]['angle_decomposition']
        print('[metric-scope-v2] {} penalized={:.4f} raw={:.4f} '
              'center_penalty={} missing={}'.format(
                  method, angle['penalized_A_RMSE_deg'],
                  angle['raw_angle_RMSE_present_predictions_deg'],
                  angle['center_penalty_count'],
                  angle['missing_prediction_count']))


if __name__ == '__main__':
    main()
