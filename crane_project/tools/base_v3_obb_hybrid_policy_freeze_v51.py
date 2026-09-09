#!/usr/bin/env python3
"""Freeze the post-V5 hybrid OBB observation policy on source-val only.

Center uses the frozen anchor-score gate and no prediction. Scale uses the
already-fitted V5 scale-error ranker and no prediction. Angle uses the frozen
anchor-score gate and at most one causal last-measurement hold. No parameter is
refit here, and fixed TEST is never read.
"""

import argparse
import copy
import json
from pathlib import Path

from crane_project.tools import (
    base_v3_obb_heterogeneous_policy_development_v5 as v5)
from crane_project.tools.base_v3_obb_component_policy_audit_v4 import _groups
from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    COMPONENTS, _measurement_value, assign_split, build_feature_rows)
from crane_project.tools.base_v3_obb_external_sequence_eval import state_report
from crane_project.tools.base_v3_obb_observation_interface_v3 import (
    CALIBRATION_PROTOCOL)
from crane_project.tools.base_v3_obb_reliability_baseline import (
    PROTOCOL as BASELINE_PROTOCOL, _identity, _write_exact)
from crane_project.tools.base_v3_reliability_failure_audit import (
    portable_digest, validate)


PROTOCOL = 'base_v3_obb_hybrid_policy_freeze_v51'
CONTRACT_PROTOCOL = 'base_v3_obb_hybrid_policy_freeze_contract_v51'
RUNTIME_PROTOCOL = 'base_v3_obb_hybrid_observation_calibration_v51'


def _require_identity(identity, expected, role):
    if identity['sha256'] != expected:
        raise RuntimeError('{} identity mismatch: expected {}, got {}'.format(
            role, expected, identity['sha256']))


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected V5.1 freeze contract')
    if contract.get('fixed_test_read') is not False:
        raise ValueError('V5.1 must not authorize fixed TEST')
    if contract.get('target_data_read') is not False:
        raise ValueError('V5.1 must remain source-only')
    if contract.get('post_fixed_test_hypothesis_generation') is not True:
        raise ValueError('V5.1 must preserve its post-TEST status')
    expected = contract.get('expected_inputs', {})
    if expected.get('baseline_protocol') != BASELINE_PROTOCOL:
        raise ValueError('V5.1 must bind the Base V3 baseline')
    if expected.get('runtime_calibration_v3_protocol') != CALIBRATION_PROTOCOL:
        raise ValueError('V5.1 must bind the V3 score calibration')
    if expected.get('v5_report_protocol') != v5.PROTOCOL:
        raise ValueError('V5.1 must bind the V5 report')
    policy = contract.get('frozen_component_policy', {})
    required = {
        'center': ('anchor_score', 'unavailable', 0),
        'scale': ('v5_supervised_scale_error_risk', 'unavailable', 0),
        'angle': ('anchor_score', 'bounded_last_measurement_hold', 1)}
    for component, (gate, rejected, maximum) in required.items():
        item = policy.get(component, {})
        if (item.get('gate'), item.get('rejected_or_missing'),
                int(item.get('max_prediction_frames', -1))) != (
                    gate, rejected, maximum):
            raise ValueError('Unexpected frozen {} policy'.format(component))
    if contract.get('matched_coverage_groups') != ['all', 'real', 'sim']:
        raise ValueError('V5.1 must disclose all/real/sim risk curves')
    if contract.get('risk_coverage_targets') != [0.95, 0.9, 0.8, 0.7, 0.5]:
        raise ValueError('V5.1 risk coverage targets changed')


def validate_v5_report(report, contract):
    if report.get('protocol') != v5.PROTOCOL:
        raise ValueError('Unexpected V5 report protocol')
    if report.get('fixed_test_read') is not False:
        raise ValueError('V5 report violates the fixed-TEST boundary')
    if report.get('decision') != 'KEEP_V5_AS_SOURCE_DEVELOPMENT_AND_REVISE_ONCE':
        raise ValueError('V5.1 expects the one-revision V5 decision')
    verdict = report.get('component_risk_development_verdict', {})
    if verdict.get('center', {}).get('passes_development_gate') is not False:
        raise ValueError('V5.1 expects the center-risk negative result')
    for component in ('scale', 'angle'):
        if verdict.get(component, {}).get('passes_development_gate') is not True:
            raise ValueError('V5.1 expects {} risk support'.format(component))
    if report.get('heterogeneous_policy_development_verdict', {}).get(
            'passes_all') is not False:
        raise ValueError('V5.1 expects the failed V5 heterogeneous policy')
    if report.get('inputs', {}).get('baseline_portable_sha256') != contract[
            'expected_inputs']['baseline_portable_sha256']:
        raise ValueError('V5 report and V5.1 baseline identity disagree')
    fitted = report.get('fitted_component_risk_challenger', {})
    if fitted.get('online_gt_fields_consumed') != []:
        raise ValueError('V5 risk model must not consume online GT')
    if fitted.get('components', {}).get('scale', {}).get('converged') is not True:
        raise ValueError('V5 scale risk model did not converge')


def hybrid_decisions(rows, score_decisions, learned_decisions):
    decisions = {}
    for row in rows:
        key = row['frame_key']
        decisions[key] = dict(
            center=bool(score_decisions[key]['center']),
            scale=bool(learned_decisions[key]['scale']),
            angle=bool(score_decisions[key]['angle']))
    return decisions


def replay_hybrid(rows, decisions, risks):
    """Run the frozen V5.1 policy with angle-only one-frame hold."""
    outputs = {}
    last_angle_measurement = None
    last_identity = None
    last_frame = None
    for row in v5._ordered(rows):
        identity = (row['domain'], row['sequence'])
        if (identity != last_identity or last_frame is None or
                int(row['frame']) != last_frame+1):
            last_angle_measurement = None
        observations = {}
        for component in COMPONENTS:
            accepted = decisions[row['frame_key']][component]
            if accepted:
                value = _measurement_value(row, component)
                state, source, age = 'measurement', 'base_v3_measurement', 0
                if component == 'angle':
                    last_angle_measurement = dict(
                        frame=int(row['frame']), value=copy.deepcopy(value))
            else:
                state, source, value = 'unavailable', None, None
                age = (None if component != 'angle' or
                       last_angle_measurement is None else
                       int(row['frame'])-last_angle_measurement['frame'])
                if (component == 'angle' and
                        last_angle_measurement is not None and age == 1):
                    state = 'prediction'
                    source = 'one_frame_angle_hold'
                    value = copy.deepcopy(last_angle_measurement['value'])
            observations[component] = dict(
                value=value, valid=state != 'unavailable', state=state,
                source=source, age_since_measurement_frames=age,
                measurement_available=row.get('base_v3_box') is not None,
                measurement_accepted=bool(accepted),
                risk=(risks[row['frame_key']][component]
                      if component == 'scale' else None),
                gate=('v5_supervised_scale_error_risk'
                      if component == 'scale' else 'anchor_score'),
                reason=(('accepted_measurement' if accepted else
                         'detector_missing' if row.get('base_v3_box') is None
                         else 'rejected_measurement')))
        outputs[row['frame_key']] = dict(
            frame_key=row['frame_key'], domain=row['domain'],
            sequence=row['sequence'], frame=int(row['frame']),
            components=observations)
        last_identity, last_frame = identity, int(row['frame'])
    return outputs


def _equal_metrics(first, second, keys, role):
    for key in keys:
        left, right = first[key], second[key]
        if left is None or right is None:
            equal = left is right
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            equal = abs(float(left)-float(right)) <= 1e-12
        else:
            equal = left == right
        if not equal:
            raise RuntimeError('{} reproduction failed at {}'.format(role, key))


def _verify_v5_reproduction(v5_report, reports):
    keys = ('measurement_count', 'prediction_count', 'unavailable_count',
            'mean_available_output_error', 'correct_output_coverage')
    for component in COMPONENTS:
        _equal_metrics(
            v5_report['method_reports']['raw_rejection_only']['all'][component],
            reports['raw_rejection_only']['all'][component], keys,
            'raw_rejection_only.' + component)
        _equal_metrics(
            v5_report['method_reports']['score_rejection_only']['all'][component],
            reports['score_rejection_only']['all'][component], keys,
            'score_rejection_only.' + component)
    references = {
        'center': ('score_rejection_only', 'center'),
        'scale': ('learned_risk_rejection_only', 'scale'),
        'angle': ('score_heterogeneous', 'angle')}
    for component, (method, source_component) in references.items():
        _equal_metrics(
            v5_report['method_reports'][method]['all'][source_component],
            reports['v51_hybrid']['all'][component], keys,
            'v51_hybrid.' + component)
    return dict(passed=True, hybrid_component_references={
        component: method for component, (method, _) in references.items()})


def _policy_verdict(reports, v5_report):
    raw = reports['raw_rejection_only']['all']
    score = reports['score_rejection_only']['all']
    hybrid = reports['v51_hybrid']['all']
    center_keys = ('measurement_count', 'prediction_count',
                   'unavailable_count', 'mean_available_output_error',
                   'correct_output_coverage')
    checks = {
        'center': {
            'equals_score_rejection_only': all(
                hybrid['center'][key] == score['center'][key]
                for key in center_keys),
            'prediction_count_is_zero': hybrid['center']['prediction_count'] == 0},
        'scale': {
            'v5_matched_coverage_gate_passed': v5_report[
                'component_risk_development_verdict']['scale'][
                    'passes_development_gate'] is True,
            'mean_error_not_worse_than_raw': (
                hybrid['scale']['mean_available_output_error'] <=
                raw['scale']['mean_available_output_error']),
            'correct_output_coverage_not_worse_than_score_rejection': (
                hybrid['scale']['correct_output_coverage'] >=
                score['scale']['correct_output_coverage']),
            'prediction_count_is_zero': hybrid['scale']['prediction_count'] == 0},
        'angle': {
            'mean_error_not_worse_than_raw': (
                hybrid['angle']['mean_available_output_error'] <=
                raw['angle']['mean_available_output_error']),
            'correct_output_coverage_not_worse_than_raw': (
                hybrid['angle']['correct_output_coverage'] >=
                raw['angle']['correct_output_coverage']),
            'prediction_improved_not_fewer_than_degraded': (
                hybrid['angle']['prediction_improved_count'] >=
                hybrid['angle']['prediction_degraded_count']),
            'longest_prediction_is_one_frame_by_construction': True}}
    return dict(
        checks=checks,
        passes_all=all(value for group in checks.values()
                       for value in group.values()))


def build_runtime_calibration(runtime_v3, v5_report, contract, identities,
                              eligible_for_fresh_sequence):
    fitted = v5_report['fitted_component_risk_challenger']
    scale = copy.deepcopy(fitted['components']['scale'])
    return dict(
        protocol=RUNTIME_PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        source_v5_report_sha256=identities['v5_report']['sha256'],
        source_v5_contract_sha256=identities['v5_contract']['sha256'],
        baseline_portable_sha256=contract['expected_inputs'][
            'baseline_portable_sha256'],
        component_policy=copy.deepcopy(contract['frozen_component_policy']),
        score_gate=dict(
            score_threshold=float(runtime_v3['score_threshold']),
            accepted_when='anchor_score >= score_threshold'),
        scale_error_ranker=dict(
            model=fitted['model'],
            feature_names=copy.deepcopy(fitted['feature_names']),
            expanded_feature_names=copy.deepcopy(
                fitted['expanded_feature_names']),
            preprocessing=copy.deepcopy(fitted['preprocessing']),
            weights=scale['weights'],
            risk_threshold=float(scale['risk_threshold']),
            accepted_when='scale_error_risk <= risk_threshold',
            online_gt_fields_consumed=[]),
        online_contract=dict(
            components=list(COMPONENTS),
            states=['measurement', 'prediction', 'unavailable'],
            domain_identity_used_in_decisions=False,
            future_frames_used=False,
            gt_fields_consumed=[],
            reset_on_sequence_or_frame_gap=True),
        eligible_for_fresh_sequence_evaluation=bool(
            eligible_for_fresh_sequence),
        eligible_for_fixed_test_reuse_as_final_validation=False,
        claim_limit=contract['claim_limit'])


def run(baseline, baseline_contract, runtime_v3, v5_report, contract):
    validate_contract(contract)
    validate(baseline, baseline_contract)
    if runtime_v3.get('protocol') != CALIBRATION_PROTOCOL:
        raise ValueError('Unexpected V3 runtime calibration protocol')
    validate_v5_report(v5_report, contract)
    rows = build_feature_rows(baseline['records'])
    split_counts = assign_split(
        rows, float(contract['split']['calibration_prefix_fraction']),
        int(contract['split']['minimum_calibration_frames_per_segment']))
    expected = dict(
        calibration=int(contract['split']['expected_calibration_frames']),
        evaluation=int(contract['split']['expected_evaluation_frames']))
    if split_counts != expected or v5_report.get('split_counts') != expected:
        raise RuntimeError('V5.1 source split counts changed')
    evaluation_rows = [row for row in rows if row['split'] == 'evaluation']
    fitted = v5_report['fitted_component_risk_challenger']
    risks = v5.risk_scores(evaluation_rows, fitted)
    score, learned = v5.gate_decisions(
        evaluation_rows, runtime_v3, fitted, risks)
    hybrid = hybrid_decisions(evaluation_rows, score, learned)
    raw = {row['frame_key']: {
        component: row.get('base_v3_box') is not None
        for component in COMPONENTS} for row in evaluation_rows}
    outputs = {
        'raw_rejection_only': v5.replay_policy(
            evaluation_rows, raw, None, 'rejection_only'),
        'score_rejection_only': v5.replay_policy(
            evaluation_rows, score, None, 'rejection_only'),
        'v51_hybrid': replay_hybrid(evaluation_rows, hybrid, risks)}
    reports = {name: {
        group: state_report(selected, output, contract, annotated=True)
        for group, selected in _groups(evaluation_rows)}
        for name, output in outputs.items()}
    curves = {}
    for group in contract['matched_coverage_groups']:
        selected = (evaluation_rows if group == 'all' else
                    [row for row in evaluation_rows if row['domain'] == group])
        curves[group] = v5.matched_coverage(selected, risks, {
            'component_risk_challenger': {
                'risk_coverage_targets': contract['risk_coverage_targets']},
            'evaluation': contract['evaluation']})
    reproduction = _verify_v5_reproduction(v5_report, reports)
    verdict = _policy_verdict(reports, v5_report)
    output_rows = []
    for row in rows:
        item = {key: copy.deepcopy(row.get(key)) for key in (
            'frame_key', 'domain', 'sequence', 'frame', 'split',
            'anchor_source', 'anchor_score')}
        if row['split'] == 'evaluation':
            item['online_features'] = row['online_features']
            item['scale_error_risk'] = risks[row['frame_key']]['scale']
            item['hybrid_components'] = outputs['v51_hybrid'][
                row['frame_key']]['components']
            item['offline_errors'] = row['errors']
        output_rows.append(item)
    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        claim_status='SOURCE_VAL_POST_V5_POLICY_FREEZE_ONLY',
        fixed_test_read=False, target_data_read=False,
        post_fixed_test_hypothesis_generation=True,
        threshold_fitting_performed=False,
        feature_selection_performed=False,
        policy_selection_source='bound_v5_source_development_report',
        leakage_controls=dict(
            gt_used_in_online_features=False,
            gt_used_in_online_decisions=False,
            evaluation_gt_used_for_fitting=False,
            gt_attached_to_online_outputs_after_replay=True,
            future_frames_used=False,
            domain_identity_used_in_decisions=False,
            state_reset_on_sequence_or_frame_gap=True),
        split_counts=split_counts,
        method_inventory=list(outputs),
        method_reports=reports,
        matched_coverage_by_group=curves,
        v5_reproduction_check=reproduction,
        hybrid_policy_freeze_verdict=verdict,
        records=output_rows,
        decision=('FREEZE_V51_FOR_FRESH_SEQUENCE_EVALUATION'
                  if verdict['passes_all'] else
                  'STOP_V51_POLICY_FREEZE'),
        eligible_for_fresh_sequence_evaluation=verdict['passes_all'],
        eligible_for_fixed_test_reuse_as_final_validation=False,
        eligible_for_unknown_sequence_claim=False,
        claim_limit=contract['claim_limit'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--baseline-contract', required=True)
    parser.add_argument('--runtime-calibration-v3', required=True)
    parser.add_argument('--v5-report', required=True)
    parser.add_argument('--v5-contract', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--out-runtime-calibration', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    identities = {name: _identity(path) for name, path in (
        ('baseline', args.baseline),
        ('baseline_contract', args.baseline_contract),
        ('runtime_calibration_v3', args.runtime_calibration_v3),
        ('v5_report', args.v5_report), ('v5_contract', args.v5_contract),
        ('contract', args.contract))}
    baseline = json.loads(Path(args.baseline).read_text())
    baseline_contract = json.loads(Path(args.baseline_contract).read_text())
    runtime_v3 = json.loads(Path(args.runtime_calibration_v3).read_text())
    v5_report = json.loads(Path(args.v5_report).read_text())
    contract = json.loads(Path(args.contract).read_text())
    validate_contract(contract)
    expected = contract['expected_inputs']
    for role, field in (
            ('baseline_contract', 'baseline_contract_sha256'),
            ('runtime_calibration_v3', 'runtime_calibration_v3_sha256'),
            ('v5_report', 'v5_report_sha256'),
            ('v5_contract', 'v5_contract_sha256')):
        _require_identity(identities[role], expected[field], role)
    portable = portable_digest(baseline)
    if portable != expected['baseline_portable_sha256']:
        raise RuntimeError('Portable source baseline identity mismatch')
    payload = run(
        baseline, baseline_contract, runtime_v3, v5_report, contract)
    runtime = build_runtime_calibration(
        runtime_v3, v5_report, contract, identities,
        payload['eligible_for_fresh_sequence_evaluation'])
    runtime_output = _write_exact(args.out_runtime_calibration, runtime)
    payload['inputs'] = identities
    payload['inputs']['baseline_portable_sha256'] = portable
    payload['frozen_runtime_calibration'] = runtime_output
    output = _write_exact(args.out_json, payload)
    print(json.dumps(dict(
        output=output, runtime_calibration=runtime_output,
        split_counts=payload['split_counts'],
        method_count=len(payload['method_inventory']),
        decision=payload['decision'],
        hybrid_policy_passed=payload['hybrid_policy_freeze_verdict'][
            'passes_all'],
        eligible_for_fresh_sequence_evaluation=payload[
            'eligible_for_fresh_sequence_evaluation'],
        claim_status=payload['claim_status'], fixed_test_read=False), indent=2))


if __name__ == '__main__':
    main()
