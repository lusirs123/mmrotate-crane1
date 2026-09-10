#!/usr/bin/env python3
"""Compare Base V3 with its actual same-frame anchor component by component.

The frozen TEST is already exposed.  This diagnostic reconstructs all 992
frames, verifies the prior V5.1 and geometry reports, and compares Base V3
against formal K1 when present or DINO when K1 is missing.  GT and manually
reviewed operation groups are used only for offline analysis.
"""

import argparse
import json
import math
from collections import Counter

import numpy as np

from crane_project.tools import base_v3_obb_top_beam_geometry_audit_v1 as v1
from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    _error_from_value, _geometry, _measurement_value)
from crane_project.tools.base_v3_obb_fixed_test_eval import (
    ATTRIBUTION_PROTOCOL, _read_json, _require_identity, build_records)
from crane_project.tools.base_v3_obb_hybrid_fixed_test_eval_v51 import (
    PROTOCOL as V51_PROTOCOL)
from crane_project.tools.base_v3_obb_reliability_baseline import _write_exact
from crane_project.tools.eval_crane_offline import angle_diff


PROTOCOL = 'base_v3_obb_top_beam_paired_diagnostic_v2'
CONTRACT_PROTOCOL = 'base_v3_obb_top_beam_paired_diagnostic_contract_v2'
GEOMETRY_V1_PROTOCOL = v1.PROTOCOL
MANIFEST_PROTOCOL = v1.MANIFEST_PROTOCOL
ABSOLUTE_METRICS = (
    'center_error_px', 'long_side_relative_error',
    'short_side_relative_error', 'max_side_relative_error',
    'angle_error_deg')
SIGNED_METRICS = (
    'center_dx_px', 'center_dy_px', 'long_side_bias_px',
    'long_side_relative_bias', 'short_side_bias_px',
    'short_side_relative_bias', 'angle_signed_error_deg')
COMPARISON_COMPONENTS = {
    'center': 'center_error_px',
    'long_side': 'long_side_relative_error',
    'short_side': 'short_side_relative_error',
    'scale': 'max_side_relative_error',
    'angle': 'angle_error_deg'}


def _valid_sha256(value):
    text = str(value)
    return len(text) == 64 and all(char in '0123456789abcdef' for char in text)


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected top-beam paired diagnostic contract')
    if contract.get('fixed_test_read') is not True:
        raise ValueError('Paired diagnostic must declare fixed TEST read')
    if contract.get('fixed_test_previously_exposed') is not True:
        raise ValueError('Paired diagnostic must preserve prior TEST exposure')
    if contract.get('operation_labels_used_online') is not False:
        raise ValueError('Operation labels must remain offline-only')
    if contract.get('parameter_tuning_authorized') is not False:
        raise ValueError('Paired diagnostic cannot authorize tuning')
    expected = contract.get('expected_inputs', {})
    protocols = {
        'fixed_test_report_protocol': V51_PROTOCOL,
        'geometry_v1_report_protocol': GEOMETRY_V1_PROTOCOL,
        'attribution_protocol': ATTRIBUTION_PROTOCOL,
        'phase_manifest_protocol': MANIFEST_PROTOCOL}
    for field, protocol in protocols.items():
        if expected.get(field) != protocol:
            raise ValueError('Unexpected bound protocol: ' + field)
    for field in ('fixed_test_report_sha256', 'geometry_v1_report_sha256',
                  'attribution_sha256', 'phase_manifest_sha256'):
        if not _valid_sha256(expected.get(field)):
            raise ValueError(field + ' must be a lowercase SHA256')
    scope = contract.get('fixed_test_scope', {})
    if int(scope.get('frame_count', 0)) <= 0:
        raise ValueError('Fixed TEST frame count must be positive')
    if sum(scope.get('domain_counts', {}).values()) != scope['frame_count']:
        raise ValueError('Domain counts must sum to fixed TEST frame count')
    if scope.get('annotation_status') != 'complete':
        raise ValueError('Paired diagnostic requires complete annotations')
    for field in ('expected_operation_group_counts',
                  'expected_anchor_source_counts'):
        counts = contract.get(field, {})
        if sum(counts.values()) != scope['frame_count']:
            raise ValueError(field + ' must sum to fixed TEST frame count')
    evaluation = contract.get('evaluation', {})
    for field in ('center_error_threshold_px',
                  'scale_relative_error_threshold',
                  'angle_error_threshold_deg', 'comparison_tolerance',
                  'identity_tolerance'):
        if float(evaluation.get(field, 0.0)) <= 0.0:
            raise ValueError(field + ' must be positive')


def _signed_angle_error_deg(predicted, target):
    value = float(angle_diff(
        np.asarray([predicted]), np.asarray([target]))[0])
    return math.degrees(value)


def box_metrics(box, gt_box):
    center, long_side, short_side, angle = _geometry(
        np.asarray(box, dtype=np.float64))
    gt_center, gt_long, gt_short, gt_angle = _geometry(
        np.asarray(gt_box, dtype=np.float64))
    center_delta = center-gt_center
    long_bias = long_side-gt_long
    short_bias = short_side-gt_short
    angle_signed = _signed_angle_error_deg(angle, gt_angle)
    long_relative = long_bias/max(gt_long, 1e-12)
    short_relative = short_bias/max(gt_short, 1e-12)
    return dict(
        center_error_px=float(np.linalg.norm(center_delta)),
        center_dx_px=float(center_delta[0]),
        center_dy_px=float(center_delta[1]),
        long_side_bias_px=float(long_bias),
        long_side_relative_bias=float(long_relative),
        long_side_relative_error=float(abs(long_relative)),
        short_side_bias_px=float(short_bias),
        short_side_relative_bias=float(short_relative),
        short_side_relative_error=float(abs(short_relative)),
        max_side_relative_error=float(max(
            abs(long_relative), abs(short_relative))),
        angle_signed_error_deg=float(angle_signed),
        angle_error_deg=float(abs(angle_signed)))


def _outcome(delta, tolerance):
    if delta < -tolerance:
        return 'improved'
    if delta > tolerance:
        return 'degraded'
    return 'tie'


def _thresholds(contract):
    evaluation = contract['evaluation']
    return dict(
        center=float(evaluation['center_error_threshold_px']),
        long_side=float(evaluation['scale_relative_error_threshold']),
        short_side=float(evaluation['scale_relative_error_threshold']),
        scale=float(evaluation['scale_relative_error_threshold']),
        angle=float(evaluation['angle_error_threshold_deg']))


def _transition(anchor_error, base_error, threshold):
    anchor = 'correct' if anchor_error <= threshold else 'bad'
    base = 'correct' if base_error <= threshold else 'bad'
    return 'anchor_{}_to_base_{}'.format(anchor, base)


def paired_row(row, contract):
    source = row['anchor_source']
    anchor_box = row['k1_box'] if source == 'k1' else row['dino_box']
    if anchor_box is None:
        raise RuntimeError('Missing actual anchor for ' + row['frame_key'])
    base = box_metrics(row['base_v3_box'], row['gt_box'])
    anchor = box_metrics(anchor_box, row['gt_box'])
    tolerance = float(contract['evaluation']['comparison_tolerance'])
    thresholds = _thresholds(contract)
    deltas = {}
    outcomes = {}
    transitions = {}
    for component, metric in COMPARISON_COMPONENTS.items():
        delta = base[metric]-anchor[metric]
        deltas[component] = float(delta)
        outcomes[component] = _outcome(delta, tolerance)
        transitions[component] = _transition(
            anchor[metric], base[metric], thresholds[component])

    base_center, base_long, base_short, base_angle = _geometry(
        np.asarray(row['base_v3_box'], dtype=np.float64))
    anchor_center, anchor_long, anchor_short, anchor_angle = _geometry(
        np.asarray(anchor_box, dtype=np.float64))
    correction = dict(
        box_changed=not bool(np.allclose(
            np.asarray(row['base_v3_box']), np.asarray(anchor_box),
            rtol=0.0, atol=tolerance)),
        center_shift_px=float(np.linalg.norm(base_center-anchor_center)),
        long_side_relative_change=float(
            (base_long-anchor_long)/max(anchor_long, 1e-12)),
        short_side_relative_change=float(
            (base_short-anchor_short)/max(anchor_short, 1e-12)),
        angle_shift_deg=float(abs(_signed_angle_error_deg(
            base_angle, anchor_angle))))
    return dict(
        frame_key=row['frame_key'], domain=row['domain'],
        sequence=row['sequence'], frame=int(row['frame']),
        operation_group=row['operation_group'], anchor_source=source,
        anchor_score=float(row['anchor_score']),
        base_v3_metrics=base, anchor_metrics=anchor,
        base_minus_anchor_error=deltas,
        component_outcomes=outcomes,
        correctness_transitions=transitions,
        refiner_correction=correction)


def _distribution(values):
    if not values:
        return dict(mean=None, median=None, p90=None, minimum=None,
                    maximum=None)
    data = np.asarray(values, dtype=np.float64)
    return dict(
        mean=float(np.mean(data)), median=float(np.median(data)),
        p90=float(np.quantile(data, .9)), minimum=float(np.min(data)),
        maximum=float(np.max(data)))


def summarize_pairs(rows):
    summary = dict(frame_count=len(rows))
    summary['base_v3_absolute_error'] = {
        metric: _distribution([row['base_v3_metrics'][metric]
                               for row in rows])
        for metric in ABSOLUTE_METRICS}
    summary['anchor_absolute_error'] = {
        metric: _distribution([row['anchor_metrics'][metric] for row in rows])
        for metric in ABSOLUTE_METRICS}
    summary['signed_bias'] = {
        method: {metric: _distribution([
            row[method + '_metrics'][metric] for row in rows])
            for metric in SIGNED_METRICS}
        for method in ('base_v3', 'anchor')}
    summary['component_comparison'] = {}
    for component in COMPARISON_COMPONENTS:
        deltas = [row['base_minus_anchor_error'][component] for row in rows]
        outcomes = Counter(row['component_outcomes'][component] for row in rows)
        transitions = Counter(
            row['correctness_transitions'][component] for row in rows)
        summary['component_comparison'][component] = dict(
            base_minus_anchor_error=_distribution(deltas),
            outcome_counts={key: int(outcomes.get(key, 0))
                            for key in ('improved', 'degraded', 'tie')},
            correctness_transition_counts=dict(transitions))
    correction = [row['refiner_correction'] for row in rows]
    summary['refiner_correction'] = dict(
        changed_count=sum(item['box_changed'] for item in correction),
        unchanged_count=sum(not item['box_changed'] for item in correction),
        center_shift_px=_distribution([
            item['center_shift_px'] for item in correction]),
        long_side_relative_change=_distribution([
            item['long_side_relative_change'] for item in correction]),
        short_side_relative_change=_distribution([
            item['short_side_relative_change'] for item in correction]),
        angle_shift_deg=_distribution([
            item['angle_shift_deg'] for item in correction]))
    return summary


def grouped_summaries(rows):
    groups = {'all': rows}
    operations = sorted({row['operation_group'] for row in rows})
    sources = sorted({row['anchor_source'] for row in rows})
    for operation in operations:
        groups['operation:' + operation] = [
            row for row in rows if row['operation_group'] == operation]
    for source in sources:
        groups['anchor:' + source] = [
            row for row in rows if row['anchor_source'] == source]
    for operation in operations:
        for source in sources:
            selected = [row for row in rows
                        if row['operation_group'] == operation
                        and row['anchor_source'] == source]
            if selected:
                groups['operation:{}|anchor:{}'.format(
                    operation, source)] = selected
    return {name: summarize_pairs(selected)
            for name, selected in groups.items()}


def verify_v51_report(report, rows, contract):
    if report.get('protocol') != V51_PROTOCOL:
        raise ValueError('Unexpected frozen V5.1 TEST report')
    records = report.get('records')
    if not isinstance(records, list) or len(records) != len(rows):
        raise ValueError('Frozen V5.1 TEST records are incomplete')
    by_key = {record['frame_key']: record for record in records}
    if len(by_key) != len(records):
        raise ValueError('Frozen V5.1 TEST report has duplicate frame keys')
    tolerance = float(contract['evaluation']['identity_tolerance'])
    max_value_difference = 0.0
    max_error_difference = 0.0
    for row in rows:
        record = by_key.get(row['frame_key'])
        if record is None:
            raise RuntimeError('V5.1 report is missing ' + row['frame_key'])
        if record.get('anchor_source') != row['anchor_source']:
            raise RuntimeError('Anchor source changed for ' + row['frame_key'])
        raw = record['online_observations']['raw']
        old_errors = record['offline_errors']['raw']
        for component in ('center', 'scale', 'angle'):
            expected_value = np.asarray(
                _measurement_value(row, component), dtype=np.float64)
            actual_value = np.asarray(raw[component]['value'], dtype=np.float64)
            difference = float(np.max(np.abs(expected_value-actual_value)))
            max_value_difference = max(max_value_difference, difference)
            expected_error = float(_error_from_value(
                _measurement_value(row, component), row, component))
            error_difference = abs(expected_error-float(old_errors[component]))
            max_error_difference = max(max_error_difference, error_difference)
            if difference > tolerance or error_difference > tolerance:
                raise RuntimeError(
                    'V5.1 raw observation mismatch for {} {}'.format(
                        row['frame_key'], component))
    return dict(
        verified_frame_count=len(rows),
        maximum_raw_value_difference=max_value_difference,
        maximum_raw_error_difference=max_error_difference)


def verify_geometry_v1_report(report, rows, contract):
    if report.get('protocol') != GEOMETRY_V1_PROTOCOL:
        raise ValueError('Unexpected top-beam geometry V1 report')
    operation_counts = dict(Counter(
        row['operation_group'] for row in rows))
    anchor_counts = dict(Counter(row['anchor_source'] for row in rows))
    if report.get('frame_count') != len(rows):
        raise RuntimeError('Geometry V1 frame count changed')
    if report.get('operation_group_counts') != operation_counts:
        raise RuntimeError('Geometry V1 operation groups changed')
    if report.get('anchor_source_counts') != anchor_counts:
        raise RuntimeError('Geometry V1 anchor sources changed')
    recomputed = v1.full_set_summaries(v1.add_geometry_errors(rows))
    recorded = report.get('full_set_geometry_summary', {})
    tolerance = float(contract['evaluation']['identity_tolerance'])
    maximum_difference = 0.0
    if set(recomputed) != set(recorded):
        raise RuntimeError('Geometry V1 summary groups changed')
    for group in recomputed:
        for source in v1.SOURCES:
            if recomputed[group][source]['available_count'] != (
                    recorded[group][source]['available_count']):
                raise RuntimeError('Geometry V1 availability changed')
            for metric in ABSOLUTE_METRICS:
                for statistic in ('mean', 'median', 'p90', 'maximum'):
                    first = recomputed[group][source]['metrics'][metric][statistic]
                    second = recorded[group][source]['metrics'][metric][statistic]
                    if first is None and second is None:
                        continue
                    difference = abs(float(first)-float(second))
                    maximum_difference = max(maximum_difference, difference)
                    if difference > tolerance:
                        raise RuntimeError(
                            'Geometry V1 summary mismatch: {} {} {} {}'.format(
                                group, source, metric, statistic))
    return dict(
        verified_frame_count=len(rows),
        maximum_summary_difference=maximum_difference)


def run(rows, v51_report, geometry_v1_report, manifest, contract,
        identities=None):
    validate_contract(contract)
    v1.validate_manifest(manifest)
    annotated = v1.attach_operation_groups(rows, manifest)
    operation_counts = dict(Counter(
        row['operation_group'] for row in annotated))
    anchor_counts = dict(Counter(row['anchor_source'] for row in annotated))
    if operation_counts != contract['expected_operation_group_counts']:
        raise RuntimeError('Operation group counts changed')
    if anchor_counts != contract['expected_anchor_source_counts']:
        raise RuntimeError('Anchor source counts changed')
    verification = dict(
        v51_report=verify_v51_report(v51_report, annotated, contract),
        geometry_v1_report=verify_geometry_v1_report(
            geometry_v1_report, annotated, contract))
    records = [paired_row(row, contract) for row in annotated]
    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        claim_status='POST_EXPOSURE_SAME_FRAME_COMPONENT_DIAGNOSTIC_ONLY',
        fixed_test_read=True, fixed_test_previously_exposed=True,
        target_definition=manifest['target_definition'],
        comparison_baseline='actual_same_frame_k1_else_dino_anchor',
        operation_labels_attached_after_inference=True,
        operation_labels_used_in_online_decisions=False,
        gt_used_only_for_offline_evaluation=True,
        threshold_fitting_performed=False,
        feature_selection_performed=False,
        parameter_tuning_authorized=False,
        inputs=identities, input_reverification=verification,
        frame_count=len(records), operation_group_counts=operation_counts,
        anchor_source_counts=anchor_counts,
        paired_summaries=grouped_summaries(records),
        records=records,
        eligible_for_parameter_tuning_from_this_report=False,
        eligible_for_phase_conditioned_online_policy_claim=False,
        eligible_for_fresh_sequence_generalization_claim=False,
        claim_limit=contract['claim_limit'])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-test-attribution', required=True)
    parser.add_argument('--fixed-test-v51-report', required=True)
    parser.add_argument('--geometry-v1-report', required=True)
    parser.add_argument('--phase-manifest', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--base-v3-results')
    parser.add_argument('--k1-results')
    parser.add_argument('--all-lane-audit')
    parser.add_argument('--ann-dir')
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    contract_id, contract = _read_json(args.contract, 'contract')
    validate_contract(contract)
    attribution_id, attribution = _read_json(
        args.fixed_test_attribution, 'fixed-test attribution')
    v51_id, v51_report = _read_json(
        args.fixed_test_v51_report, 'frozen V5.1 TEST report')
    geometry_id, geometry_report = _read_json(
        args.geometry_v1_report, 'top-beam geometry V1 report')
    manifest_id, manifest = _read_json(args.phase_manifest, 'phase manifest')
    expected = contract['expected_inputs']
    for identity, field, role in (
            (attribution_id, 'attribution_sha256', 'fixed-test attribution'),
            (v51_id, 'fixed_test_report_sha256', 'frozen V5.1 TEST report'),
            (geometry_id, 'geometry_v1_report_sha256',
             'top-beam geometry V1 report'),
            (manifest_id, 'phase_manifest_sha256', 'phase manifest V2')):
        _require_identity(identity, expected[field], role)
    rows, reconstruction = build_records(attribution, args, contract)
    identities = dict(
        contract=contract_id, attribution=attribution_id,
        fixed_test_v51_report=v51_id, geometry_v1_report=geometry_id,
        phase_manifest=manifest_id, reconstruction=reconstruction)
    payload = run(rows, v51_report, geometry_report, manifest, contract,
                  identities)
    output = _write_exact(args.out_json, payload)
    all_summary = payload['paired_summaries']['all']
    concise = {component: dict(
        mean_delta=all_summary['component_comparison'][component][
            'base_minus_anchor_error']['mean'],
        outcomes=all_summary['component_comparison'][component][
            'outcome_counts']) for component in COMPARISON_COMPONENTS}
    print(json.dumps(dict(
        output=output, frame_count=payload['frame_count'],
        operation_group_counts=payload['operation_group_counts'],
        anchor_source_counts=payload['anchor_source_counts'],
        input_reverification=payload['input_reverification'],
        all_frame_component_comparison=concise,
        claim_status=payload['claim_status'],
        parameter_tuning_authorized=False), indent=2))


if __name__ == '__main__':
    main()
