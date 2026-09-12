#!/usr/bin/env python3
"""Build one compact, table-ready metric report from frozen true-online TEST output.

This module is a reporting endpoint. It never runs a detector, fits a threshold,
selects a policy, or changes the frozen V5.1 observation calibration.
"""

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

from crane_project.tools import base_v3_obb_hybrid_fixed_test_diagnostic_v51 as diagnostic
from crane_project.tools.base_v3_obb_component_reliability_continuous import COMPONENTS
from crane_project.tools.base_v3_obb_paper_final_report_v1 import METHODS, _recovery_report
from crane_project.tools.base_v3_obb_reliability_baseline import _identity, _write_exact
from crane_project.tools.base_v3_obb_true_online_finalization_v2 import PROTOCOL as V2_PROTOCOL
from crane_project.tools.base_v3_obb_true_online_pipeline_v1 import PROTOCOL as ONLINE_PROTOCOL

PROTOCOL = 'base_v3_obb_true_online_paper_metrics_v3'
CONTRACT_PROTOCOL = 'base_v3_obb_true_online_paper_metrics_contract_v3'


def _require_identity(identity, expected, role):
    if identity['sha256'] != expected:
        raise RuntimeError('{} identity mismatch: expected {}, got {}'.format(
            role, expected, identity['sha256']))


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected true-online paper-metrics contract')
    required_flags = {
        'fixed_test_read': True,
        'fixed_test_previously_exposed': True,
        'threshold_fitting_performed': False,
        'parameter_tuning_authorized': False}
    for field, expected in required_flags.items():
        if contract.get(field) is not expected:
            raise ValueError('Paper-metrics contract mismatch: ' + field)
    expected = contract.get('expected_inputs', {})
    if expected.get('true_online_finalization_protocol') != V2_PROTOCOL:
        raise ValueError('V3 must bind the V2 finalization protocol')
    if expected.get('true_online_pipeline_protocol') != ONLINE_PROTOCOL:
        raise ValueError('V3 must bind the true-online pipeline protocol')
    if contract.get('methods') != list(METHODS):
        raise ValueError('V3 method inventory changed')
    if contract.get('components') != list(COMPONENTS):
        raise ValueError('V3 component inventory changed')
    targets = contract.get('matched_coverage_targets', [])
    if not targets or any(not 0.0 < float(value) <= 1.0 for value in targets):
        raise ValueError('Matched coverage targets must lie in (0, 1]')


def _validate_thresholds(v2, contract):
    for field, value in contract['evaluation'].items():
        if field == 'tail_quantiles':
            if [float(x) for x in v2['evaluation_thresholds'][field]] != [
                    float(x) for x in value]:
                raise RuntimeError('Evaluation threshold changed: ' + field)
        elif float(v2['evaluation_thresholds'][field]) != float(value):
            raise RuntimeError('Evaluation threshold changed: ' + field)


def validate_inputs(v2, online, contract):
    if v2.get('protocol') != V2_PROTOCOL:
        raise ValueError('Unexpected V2 finalization report')
    if online.get('protocol') != ONLINE_PROTOCOL:
        raise ValueError('Unexpected true-online pipeline report')
    if v2.get('claim_status') != 'POST_EXPOSURE_TRUE_ONLINE_FINALIZATION_V2':
        raise ValueError('V2 finalization status changed')
    if online.get('execution_mode') != 'true_online_model_inference':
        raise ValueError('Input is not true-online model inference')
    for report in (v2,):
        flags = {
            'fixed_test_read': True,
            'fixed_test_previously_exposed': True,
            'threshold_fitting_performed': False,
            'parameter_tuning_authorized': False}
        for field, expected in flags.items():
            if report.get(field) is not expected:
                raise ValueError('V2 evidence flag changed: ' + field)
    leakage = online.get('leakage_controls', {})
    if leakage.get('annotations_or_gt_read') is not False:
        raise ValueError('True-online inference read annotations or GT')
    _validate_thresholds(v2, contract)
    scope = contract['fixed_test_scope']
    online_rows = online.get('records') or []
    v2_rows = v2.get('records') or []
    expected_count = int(scope['frame_count'])
    if len(online_rows) != expected_count or len(v2_rows) != expected_count:
        raise ValueError('V3 input frame count changed')
    if len({row['frame_key'] for row in online_rows}) != expected_count:
        raise ValueError('Duplicate true-online frame keys')
    if len({row['frame_key'] for row in v2_rows}) != expected_count:
        raise ValueError('Duplicate V2 frame keys')
    online_keys = {row['frame_key'] for row in online_rows}
    if online_keys != {row['frame_key'] for row in v2_rows}:
        raise RuntimeError('V2 and true-online frame sets differ')
    if dict(Counter(row['domain'] for row in online_rows)) != scope['domain_counts']:
        raise ValueError('V3 domain counts changed')
    if dict(Counter(row['sequence'] for row in online_rows)) != scope['sequence_counts']:
        raise ValueError('V3 sequence counts changed')
    if online.get('pipeline') != contract['pipeline']:
        raise ValueError('V3 pipeline order changed')
    for row in online_rows:
        if row.get('timestamp_seconds') is not None:
            raise ValueError('Unexpected partial timestamp population')
        if 'gt_box' in row or row.get('online_gt_fields_consumed'):
            raise ValueError('True-online row consumed ground truth')
        detector = row.get('detector_components', {})
        if not math.isfinite(float(detector.get('anchor_score', float('nan')))):
            raise ValueError('Every frame requires a finite anchor score')
        if detector.get('base_v3_output_score_semantics') != contract[
                'base_v3_output_score_semantics']:
            raise ValueError('Base V3 output-score semantics changed')
        if set(row.get('observations', {})) != set(METHODS):
            raise ValueError('Incomplete method observations')
        for method in METHODS:
            if set(row['observations'][method]) != set(COMPONENTS):
                raise ValueError('Incomplete component observations')
    return online_rows, v2_rows


def build_diagnostic_records(online_rows, v2_rows):
    offline = {row['frame_key']: row for row in v2_rows}
    records = []
    for row in online_rows:
        key = row['frame_key']
        detector = row['detector_components']
        metrics = offline[key]
        records.append(dict(
            frame_key=key, domain=row['domain'], sequence=row['sequence'],
            frame=int(row['frame']), anchor_source=detector['anchor_source'],
            anchor_score=float(detector['anchor_score']),
            online_observations=copy.deepcopy(row['observations']),
            offline_errors=copy.deepcopy(metrics['offline_errors']),
            offline_riou=copy.deepcopy(metrics['offline_riou'])))
    return records


def _paper_groups():
    return ('all', 'real', 'sim')


def _obb_table(v2):
    rows = []
    keys = ('frame_count', 'available_obb_count', 'available_obb_coverage',
            'mean_available_riou', 'riou_hit_count', 'riou_hit_coverage',
            'riou_hit_rate_given_available', 'bad_available_obb_rate')
    joint_keys = ('all_components_valid_count',
                  'all_components_valid_coverage', 'jointly_correct_count',
                  'jointly_correct_coverage',
                  'false_all_components_valid_count',
                  'false_all_components_valid_rate',
                  'longest_incomplete_obb_run')
    for method in METHODS:
        for group in _paper_groups():
            row = dict(method=method, group=group)
            row.update({key: v2['obb_riou_metrics'][method][group].get(key)
                        for key in keys})
            row.update({key: v2['joint_component_metrics'][method][group].get(key)
                        for key in joint_keys})
            rows.append(row)
    return rows


def _component_table(v2):
    rows = []
    keys = ('frame_count', 'measurement_count', 'prediction_count',
            'unavailable_count', 'measurement_coverage', 'output_coverage',
            'mean_available_output_error', 'bad_available_output_rate',
            'correct_output_coverage', 'longest_unavailable_run')
    for method in METHODS:
        for group in _paper_groups():
            for component in COMPONENTS:
                metrics = v2['component_metrics'][method][group][component]
                row = dict(method=method, group=group, component=component)
                row.update({key: metrics.get(key) for key in keys})
                rows.append(row)
    return rows


def _anchor_table(v2):
    rows = []
    for method in METHODS:
        for group in ('anchor:k1', 'anchor:dino_fallback'):
            riou = v2['obb_riou_metrics'][method][group]
            joint = v2['joint_component_metrics'][method][group]
            rows.append(dict(
                method=method, group=group,
                frame_count=riou['frame_count'],
                available_obb_coverage=riou['available_obb_coverage'],
                mean_available_riou=riou['mean_available_riou'],
                riou_hit_coverage=riou['riou_hit_coverage'],
                bad_available_obb_rate=riou['bad_available_obb_rate'],
                jointly_correct_coverage=joint['jointly_correct_coverage'],
                false_all_components_valid_rate=joint[
                    'false_all_components_valid_rate']))
    return rows


def _compact_scale(report):
    result = {}
    keep = ('retained_count', 'realized_coverage', 'mean_error',
            'failure_rate', 'correct_retained_count',
            'correct_output_coverage')
    for group in _paper_groups():
        entry = report[group]
        result[group] = dict(
            frame_count=entry['frame_count'],
            risk_wins_both_count=entry['risk_wins_both_count'],
            comparison_count=len(entry['points']),
            points=[dict(
                target_coverage=point['target_coverage'],
                anchor_score={key: point['anchor_score'][key] for key in keep},
                learned_scale_risk={key: point['learned_scale_risk'][key]
                                    for key in keep},
                learned_minus_score=copy.deepcopy(
                    point['learned_minus_score']))
                for point in entry['points']])
    return result


def _compact_angle(report):
    keys = ('frame_count', 'prediction_count', 'correctness_transitions',
            'improved_count', 'degraded_count', 'tied_count',
            'mean_error_delta_vs_raw', 'error_delta_quantiles',
            'by_full_rejection_run_length')
    return {group: {key: report[group][key] for key in keys}
            for group in _paper_groups()}


def _recovery(records):
    selected = diagnostic.groups(records)
    return {method: {group: {component: _recovery_report(
        rows, method, component) for component in COMPONENTS}
        for group, rows in selected if group in _paper_groups()}
        for method in METHODS}


def _derived_comparisons(v2, angle):
    result = {}
    for group in _paper_groups():
        score_riou = v2['obb_riou_metrics']['score_rejection_only'][group]
        hybrid_riou = v2['obb_riou_metrics']['v51_hybrid'][group]
        score_joint = v2['joint_component_metrics']['score_rejection_only'][group]
        hybrid_joint = v2['joint_component_metrics']['v51_hybrid'][group]
        result[group] = dict(
            v51_minus_score_complete_obb_coverage=(
                hybrid_riou['available_obb_coverage']-
                score_riou['available_obb_coverage']),
            v51_minus_score_mean_available_riou=(
                hybrid_riou['mean_available_riou']-
                score_riou['mean_available_riou']),
            v51_minus_score_riou_hit_coverage=(
                hybrid_riou['riou_hit_coverage']-
                score_riou['riou_hit_coverage']),
            v51_minus_score_jointly_correct_coverage=(
                hybrid_joint['jointly_correct_coverage']-
                score_joint['jointly_correct_coverage']),
            angle_hold_prediction_count=angle[group]['prediction_count'],
            angle_hold_mean_error_delta_vs_raw=angle[group][
                'mean_error_delta_vs_raw'])
    return result


def run(v2, online, contract, inputs=None):
    validate_contract(contract)
    online_rows, v2_rows = validate_inputs(v2, online, contract)
    records = build_diagnostic_records(online_rows, v2_rows)
    diagnostic_contract = dict(
        evaluation=contract['evaluation'],
        matched_coverage_targets=contract['matched_coverage_targets'])
    scale = diagnostic.scale_matched_coverage(records, diagnostic_contract)
    angle = diagnostic.angle_pair_audit(records, diagnostic_contract)
    joint = diagnostic.joint_availability(records, diagnostic_contract)
    if joint != v2['joint_component_metrics']:
        raise RuntimeError('V3 joint metrics do not reproduce V2 exactly')
    export = v2.get('online_observation_export') or {}
    if export.get('sha256') != contract['expected_inputs'][
            'online_observation_csv_sha256']:
        raise RuntimeError('V2 does not bind the expected observation CSV')
    summary = online['summary']
    source_semantics = Counter(
        row['detector_components']['base_v3_output_score_semantics']
        for row in online_rows)
    completeness = dict(
        true_online_pipeline_bound=True,
        frozen_metric_thresholds_bound=True,
        front_end_candidate_counts_reported=True,
        component_metrics_reported=True,
        complete_obb_metrics_reported=True,
        anchor_source_metrics_reported=True,
        matched_coverage_scale_comparison_reported=True,
        angle_hold_cost_reported=True,
        rejection_recovery_reported=True,
        runtime_reported=v2.get('runtime_measurement') is not None,
        observation_csv_bound=True,
        timestamps_available=False,
        fresh_unknown_sequence_generalization_available=False,
        physical_state_metrics_available=False)
    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        claim_status='POST_EXPOSURE_TRUE_ONLINE_PAPER_METRICS_V3',
        inputs=inputs,
        fixed_test_read=True, fixed_test_previously_exposed=True,
        frozen_reporting_only=True, threshold_fitting_performed=False,
        feature_selection_performed=False, policy_selection_performed=False,
        parameter_tuning_authorized=False,
        pipeline=copy.deepcopy(online['pipeline']),
        leakage_controls=dict(
            gt_used_in_model_inference=False,
            gt_used_in_online_observation=False,
            gt_attached_after_true_online_output=True,
            gt_used_only_for_offline_metrics=True,
            future_frames_used=False,
            domain_identity_used_in_model_decisions=False),
        dataset_summary=copy.deepcopy(v2['dataset_summary']),
        evaluation_thresholds=copy.deepcopy(contract['evaluation']),
        front_end_summary=dict(
            anchor_source_counts=copy.deepcopy(summary['anchor_source_counts']),
            detector_present_counts=copy.deepcopy(
                summary['detector_present_counts']),
            base_v3_output_score_semantics_counts=dict(source_semantics),
            candidate_and_geometry_reproduction=copy.deepcopy(
                v2['candidate_and_geometry_reproduction'])),
        paper_tables=dict(
            complete_obb_metrics=_obb_table(v2),
            component_metrics=_component_table(v2),
            anchor_source_metrics=_anchor_table(v2)),
        supporting_diagnostics=dict(
            scale_matched_coverage=_compact_scale(scale),
            angle_hold_pair_audit=_compact_angle(angle),
            rejection_recovery=_recovery(records),
            derived_v51_minus_score_comparisons=_derived_comparisons(v2, angle)),
        runtime_measurement=copy.deepcopy(v2['runtime_measurement']),
        online_observation_export=copy.deepcopy(export),
        timestamp_status='UNAVAILABLE_NOT_SYNTHESIZED',
        completeness_audit=completeness,
        final_detection_metrics_complete=all(value for key, value in
            completeness.items() if key not in (
                'timestamps_available',
                'fresh_unknown_sequence_generalization_available',
                'physical_state_metrics_available')),
        overall_winner_claimed=False,
        eligible_for_parameter_tuning_from_this_report=False,
        eligible_for_fresh_unknown_sequence_claim=False,
        eligible_for_physical_state_claim=False,
        claim_limit=contract['claim_limit'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--true-online-finalization-v2', required=True)
    parser.add_argument('--true-online-report', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    paths = dict(
        true_online_finalization_v2=args.true_online_finalization_v2,
        true_online_report=args.true_online_report,
        contract=args.contract)
    identities = {role: _identity(path) for role, path in paths.items()}
    contract = json.loads(Path(args.contract).read_text(encoding='utf-8'))
    validate_contract(contract)
    expected = contract['expected_inputs']
    _require_identity(identities['true_online_finalization_v2'],
                      expected['true_online_finalization_sha256'],
                      'true_online_finalization_v2')
    _require_identity(identities['true_online_report'],
                      expected['true_online_pipeline_sha256'],
                      'true_online_report')
    v2 = json.loads(Path(args.true_online_finalization_v2).read_text(
        encoding='utf-8'))
    online = json.loads(Path(args.true_online_report).read_text(
        encoding='utf-8'))
    payload = run(v2, online, contract, identities)
    output = _write_exact(args.out_json, payload)
    print(json.dumps(dict(
        output=output,
        frame_count=payload['dataset_summary']['frame_count'],
        front_end_summary={key: payload['front_end_summary'][key] for key in (
            'anchor_source_counts', 'detector_present_counts',
            'base_v3_output_score_semantics_counts')},
        derived_v51_minus_score_comparisons=payload[
            'supporting_diagnostics'][
                'derived_v51_minus_score_comparisons'],
        final_detection_metrics_complete=payload[
            'final_detection_metrics_complete'],
        completeness_audit=payload['completeness_audit'],
        claim_status=payload['claim_status'],
        parameter_tuning_authorized=False), indent=2))


if __name__ == '__main__':
    main()
