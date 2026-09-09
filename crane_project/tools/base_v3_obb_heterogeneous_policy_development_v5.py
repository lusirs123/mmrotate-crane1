#!/usr/bin/env python3
"""Develop a component-specific OBB observation policy on source-val only.

V5 keeps the frozen V3 score threshold as the operational baseline, applies a
different causal output policy to center, scale and angle, and fits a small
component error-ranking challenger on calibration-prefix labels only.  It
never reads fixed TEST and never exposes ground truth to the online replay.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from crane_project.tools.base_v3_obb_component_policy_audit_v4 import (
    PROTOCOL as V4_PROTOCOL, _groups)
from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    COMPONENTS, _measurement_value, _raw_component_error,
    assign_split, build_feature_rows)
from crane_project.tools.base_v3_obb_external_sequence_eval import state_report
from crane_project.tools.base_v3_obb_observation_interface_v3 import (
    CALIBRATION_PROTOCOL)
from crane_project.tools.base_v3_obb_reliability_baseline import (
    PROTOCOL as BASELINE_PROTOCOL, _identity, _write_exact)
from crane_project.tools.base_v3_reliability_failure_audit import (
    portable_digest, validate)


PROTOCOL = 'base_v3_obb_heterogeneous_policy_development_v5'
CONTRACT_PROTOCOL = 'base_v3_obb_heterogeneous_policy_development_contract_v5'


def _require_identity(identity, expected, role):
    if identity['sha256'] != expected:
        raise RuntimeError('{} identity mismatch: expected {}, got {}'.format(
            role, expected, identity['sha256']))


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected V5 development contract')
    if contract.get('fixed_test_read') is not False:
        raise ValueError('V5 must not authorize fixed TEST')
    if contract.get('target_data_read') is not False:
        raise ValueError('V5 must remain source-only')
    if contract.get('post_fixed_test_hypothesis_generation') is not True:
        raise ValueError('V5 must preserve its post-TEST development status')
    expected = contract.get('expected_inputs', {})
    if expected.get('baseline_protocol') != BASELINE_PROTOCOL:
        raise ValueError('V5 must bind the Base V3 baseline protocol')
    if expected.get('runtime_calibration_protocol') != CALIBRATION_PROTOCOL:
        raise ValueError('V5 must bind the V3 runtime calibration')
    if expected.get('v4_report_protocol') != V4_PROTOCOL:
        raise ValueError('V5 must bind the V4 development report')
    policy = contract.get('score_policy', {})
    expected_policy = {
        'center': 'one_frame_constant_velocity',
        'scale': 'unavailable',
        'angle': 'bounded_last_measurement_hold'}
    for component, name in expected_policy.items():
        if policy.get(component, {}).get('rejected_or_missing') != name:
            raise ValueError('Unexpected {} V5 output policy'.format(component))
    if int(policy['center']['max_prediction_frames']) != 1:
        raise ValueError('Center prediction must remain one-frame bounded')
    if int(policy['scale']['max_prediction_frames']) != 0:
        raise ValueError('Scale must become unavailable after rejection')
    if int(policy['angle']['max_prediction_frames']) != 1:
        raise ValueError('Angle hold must remain one-frame bounded')


def _ordered(rows):
    return sorted(rows, key=lambda row: (
        row['domain'], row['sequence'], row['frame'], row['frame_key']))


def _matrix(rows, feature_names, preprocessing=None):
    values = np.asarray([[np.nan if row['online_features'].get(name) is None
                          else float(row['online_features'][name])
                          for name in feature_names]
                         for row in rows], dtype=np.float64)
    missing = np.isnan(values).astype(np.float64)
    if preprocessing is None:
        medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    else:
        medians = np.asarray(preprocessing['medians'], dtype=np.float64)
    values = np.where(np.isnan(values), medians, values)
    combined = np.concatenate([values, missing], axis=1)
    if preprocessing is None:
        means = combined.mean(axis=0)
        scales = combined.std(axis=0)
        scales = np.where(scales < 1e-8, 1.0, scales)
        preprocessing = dict(
            medians=medians.tolist(), means=means.tolist(),
            scales=scales.tolist())
    else:
        means = np.asarray(preprocessing['means'], dtype=np.float64)
        scales = np.asarray(preprocessing['scales'], dtype=np.float64)
    standardized = (combined-means)/scales
    return np.concatenate([
        np.ones((len(rows), 1), dtype=np.float64), standardized], axis=1), \
        preprocessing


def _fit_logistic(features, labels, l2, maximum_iterations, tolerance):
    labels = np.asarray(labels, dtype=np.float64)
    if labels.min() == labels.max():
        raise RuntimeError('A component has only one calibration label class')
    weights = np.zeros(features.shape[1], dtype=np.float64)
    converged = False
    iterations = 0
    for iterations in range(1, maximum_iterations+1):
        logits = np.clip(features.dot(weights), -30.0, 30.0)
        probabilities = 1.0/(1.0+np.exp(-logits))
        curvature = probabilities*(1.0-probabilities)+1e-6
        gradient = features.T.dot(probabilities-labels)
        gradient[1:] += l2*weights[1:]
        hessian = (features.T*curvature).dot(features)
        hessian[1:, 1:] += l2*np.eye(features.shape[1]-1)
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.max(np.abs(step))) < tolerance:
            converged = True
            break
    return weights, iterations, converged


def fit_risk_challenger(rows, contract):
    """Fit component error rankers from calibration-prefix rows only."""
    calibration_rows = [row for row in rows if row['split'] == 'calibration']
    spec = contract['component_risk_challenger']
    names = list(spec['features'])
    matrix, preprocessing = _matrix(calibration_rows, names)
    thresholds = {
        'center': float(contract['evaluation']['center_error_threshold_px']),
        'scale': float(contract['evaluation']['scale_relative_error_threshold']),
        'angle': float(contract['evaluation']['angle_error_threshold_deg'])}
    models = {}
    target = float(spec['target_measurement_coverage'])
    for component in COMPONENTS:
        labels = np.asarray([
            _raw_component_error(row, component) > thresholds[component]
            for row in calibration_rows], dtype=np.float64)
        weights, iterations, converged = _fit_logistic(
            matrix, labels, float(spec['l2_regularization']),
            int(spec['maximum_newton_iterations']),
            float(spec['convergence_tolerance']))
        risks = 1.0/(1.0+np.exp(-np.clip(matrix.dot(weights), -30.0, 30.0)))
        models[component] = dict(
            weights=weights.tolist(), iterations=iterations,
            converged=converged, positive_count=int(labels.sum()),
            negative_count=int(len(labels)-labels.sum()),
            risk_threshold=float(np.quantile(risks, target)))
    return dict(
        model=spec['model'], feature_names=names,
        expanded_feature_names=(names + [name + '__missing' for name in names]),
        preprocessing=preprocessing,
        l2_regularization=float(spec['l2_regularization']),
        target_measurement_coverage=target, components=models,
        gt_role='calibration_prefix_error_labels_only',
        online_gt_fields_consumed=[])


def risk_scores(rows, fitted):
    matrix, _ = _matrix(
        rows, fitted['feature_names'], fitted['preprocessing'])
    output = {row['frame_key']: {} for row in rows}
    for component in COMPONENTS:
        weights = np.asarray(
            fitted['components'][component]['weights'], dtype=np.float64)
        logits = np.clip(matrix.dot(weights), -30.0, 30.0)
        values = 1.0/(1.0+np.exp(-logits))
        for row, value in zip(rows, values):
            output[row['frame_key']][component] = float(value)
    return output


def gate_decisions(rows, runtime_calibration, fitted_risk, learned_risks):
    score = {}
    learned = {}
    for row in rows:
        available = row.get('base_v3_box') is not None
        score_value = (available and row.get('anchor_score') is not None and
                       float(row['anchor_score']) >=
                       float(runtime_calibration['score_threshold']))
        score[row['frame_key']] = {
            component: bool(score_value) for component in COMPONENTS}
        learned[row['frame_key']] = {
            component: bool(available and learned_risks[row['frame_key']][
                component] <= float(fitted_risk['components'][component][
                    'risk_threshold'])) for component in COMPONENTS}
    return score, learned


def replay_policy(rows, decisions, risks, policy_mode):
    """Replay rejection-only or the frozen heterogeneous causal policy."""
    if policy_mode not in ('rejection_only', 'heterogeneous'):
        raise ValueError('Unknown V5 policy mode')
    outputs = {}
    histories = {component: [] for component in COMPONENTS}
    last_identity = None
    last_frame = None
    for row in _ordered(rows):
        identity = (row['domain'], row['sequence'])
        if (identity != last_identity or last_frame is None or
                int(row['frame']) != last_frame+1):
            histories = {component: [] for component in COMPONENTS}
        observations = {}
        for component in COMPONENTS:
            accepted = decisions[row['frame_key']][component]
            history = histories[component]
            if accepted:
                value = _measurement_value(row, component)
                state, source, age = 'measurement', 'base_v3_measurement', 0
                history.append((int(row['frame']), copy.deepcopy(value)))
                histories[component] = history[-2:]
            else:
                state, source, value = 'unavailable', None, None
                age = (None if not history else
                       int(row['frame'])-history[-1][0])
                if policy_mode == 'heterogeneous' and component == 'angle':
                    if history and age == 1:
                        state, source = 'prediction', 'one_frame_angle_hold'
                        value = copy.deepcopy(history[-1][1])
                elif policy_mode == 'heterogeneous' and component == 'center':
                    consecutive = (len(history) == 2 and
                                   history[-2][0] == int(row['frame'])-2 and
                                   history[-1][0] == int(row['frame'])-1)
                    if consecutive:
                        old = np.asarray(history[-2][1], dtype=np.float64)
                        recent = np.asarray(history[-1][1], dtype=np.float64)
                        value = (recent+(recent-old)).tolist()
                        state = 'prediction'
                        source = 'one_frame_center_constant_velocity'
                # Scale deliberately has no prediction branch.
            observations[component] = dict(
                value=value, valid=state != 'unavailable', state=state,
                source=source, age_since_measurement_frames=age,
                measurement_available=row.get('base_v3_box') is not None,
                measurement_accepted=bool(accepted),
                risk=(None if risks is None else
                      risks[row['frame_key']][component]),
                reason=(('accepted_measurement' if accepted else
                         'detector_missing' if row.get('base_v3_box') is None
                         else 'rejected_measurement')))
        outputs[row['frame_key']] = dict(
            frame_key=row['frame_key'], domain=row['domain'],
            sequence=row['sequence'], frame=int(row['frame']),
            components=observations)
        last_identity, last_frame = identity, int(row['frame'])
    return outputs


def matched_coverage(rows, risks, contract):
    result = {}
    thresholds = {
        'center': float(contract['evaluation']['center_error_threshold_px']),
        'scale': float(contract['evaluation']['scale_relative_error_threshold']),
        'angle': float(contract['evaluation']['angle_error_threshold_deg'])}
    for component in COMPONENTS:
        orderings = {
            'raw_anchor_score': sorted(
                rows, key=lambda row: (-float(row['anchor_score']),
                                       row['frame_key'])),
            'v5_supervised_error_risk': sorted(
                rows, key=lambda row: (
                    risks[row['frame_key']][component], row['frame_key']))}
        result[component] = {}
        for method, ordered in orderings.items():
            curve = []
            for coverage in contract['component_risk_challenger'][
                    'risk_coverage_targets']:
                retained = ordered[:max(1, round(len(ordered)*coverage))]
                errors = [_raw_component_error(row, component)
                          for row in retained]
                curve.append(dict(
                    target_coverage=float(coverage),
                    retained_count=len(retained),
                    mean_error=float(np.mean(errors)),
                    failure_rate=sum(error > thresholds[component]
                                     for error in errors)/len(errors)))
            result[component][method] = curve
    return result


def _risk_verdict(curves, minimum_wins):
    verdict = {}
    for component in COMPONENTS:
        anchor = {item['target_coverage']: item
                  for item in curves[component]['raw_anchor_score']}
        challenger = curves[component]['v5_supervised_error_risk']
        wins = []
        for item in challenger:
            other = anchor[item['target_coverage']]
            if (item['mean_error'] <= other['mean_error'] and
                    item['failure_rate'] <= other['failure_rate'] and
                    (item['mean_error'] < other['mean_error'] or
                     item['failure_rate'] < other['failure_rate'])):
                wins.append(item['target_coverage'])
        verdict[component] = dict(
            matched_coverage_win_count=len(wins), winning_coverages=wins,
            passes_development_gate=len(wins) >= minimum_wins)
    return verdict


def _heterogeneous_verdict(reports):
    raw = reports['raw_rejection_only']['all']
    score_reject = reports['score_rejection_only']['all']
    candidate = reports['score_heterogeneous']['all']
    angle = candidate['angle']
    center = candidate['center']
    scale = candidate['scale']
    checks = {
        'angle': {
            'mean_error_not_worse_than_raw': (
                angle['mean_available_output_error'] <=
                raw['angle']['mean_available_output_error']),
            'correct_output_coverage_not_worse_than_raw': (
                angle['correct_output_coverage'] >=
                raw['angle']['correct_output_coverage']),
            'prediction_improved_not_fewer_than_degraded': (
                angle['prediction_improved_count'] >=
                angle['prediction_degraded_count'])},
        'center': {
            'correct_output_coverage_better_than_score_rejection': (
                center['correct_output_coverage'] >
                score_reject['center']['correct_output_coverage']),
            'prediction_improved_not_fewer_than_degraded': (
                center['prediction_improved_count'] >=
                center['prediction_degraded_count']),
            'mean_error_not_worse_than_raw': (
                center['mean_available_output_error'] <=
                raw['center']['mean_available_output_error'])},
        'scale': {
            'must_equal_score_rejection_only': all(
                scale[key] == score_reject['scale'][key] for key in (
                    'measurement_count', 'prediction_count',
                    'unavailable_count', 'mean_available_output_error',
                    'correct_output_coverage'))}}
    return dict(
        checks=checks,
        passes_all=all(value for group in checks.values()
                       for value in group.values()))


def _verify_v4_reproduction(v4, reports):
    comparisons = [
        ('raw_rejection_only', 'raw_rejection_only', 'center'),
        ('raw_rejection_only', 'raw_rejection_only', 'scale'),
        ('raw_rejection_only', 'raw_rejection_only', 'angle'),
        ('score_rejection_only', 'score_rejection_only', 'center'),
        ('score_rejection_only', 'score_rejection_only', 'scale'),
        ('score_rejection_only', 'score_rejection_only', 'angle'),
        ('score_bounded_hold', 'score_heterogeneous', 'angle')]
    keys = ('measurement_count', 'prediction_count', 'unavailable_count',
            'mean_available_output_error', 'correct_output_coverage')
    checked = []
    for v4_name, v5_name, component in comparisons:
        old = v4['method_reports'][v4_name]['all'][component]
        new = reports[v5_name]['all'][component]
        for key in keys:
            a, b = old[key], new[key]
            if a is None or b is None:
                equal = a is b
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                equal = abs(float(a)-float(b)) <= 1e-12
            else:
                equal = a == b
            if not equal:
                raise RuntimeError(
                    'V4 reproduction failed for {}.{}.{}'.format(
                        v5_name, component, key))
        checked.append(dict(v4_method=v4_name, v5_method=v5_name,
                            component=component))
    return dict(passed=True, comparisons=checked)


def run(baseline, baseline_contract, runtime_calibration, v4, contract):
    validate_contract(contract)
    validate(baseline, baseline_contract)
    if runtime_calibration.get('protocol') != CALIBRATION_PROTOCOL:
        raise ValueError('Unexpected runtime calibration protocol')
    if v4.get('protocol') != V4_PROTOCOL:
        raise ValueError('Unexpected V4 report protocol')
    if v4.get('fixed_test_read') is not False:
        raise ValueError('Supplied V4 report violates the source-only boundary')
    if v4.get('post_fixed_test_hypothesis_generation') is not True:
        raise ValueError('Supplied V4 report lost its development-only status')
    if v4.get('split_counts') != dict(
            calibration=int(contract['split']['expected_calibration_frames']),
            evaluation=int(contract['split']['expected_evaluation_frames'])):
        raise ValueError('Supplied V4 report has unexpected split counts')
    if v4.get('inputs', {}).get('baseline_portable_sha256') != contract[
            'expected_inputs']['baseline_portable_sha256']:
        raise ValueError('V4 report and V5 baseline identity disagree')
    rows = build_feature_rows(baseline['records'])
    split_counts = assign_split(
        rows, float(contract['split']['calibration_prefix_fraction']),
        int(contract['split']['minimum_calibration_frames_per_segment']))
    expected_counts = dict(
        calibration=int(contract['split']['expected_calibration_frames']),
        evaluation=int(contract['split']['expected_evaluation_frames']))
    if split_counts != expected_counts:
        raise RuntimeError('Source split counts changed: {!r}'.format(
            split_counts))
    fitted = fit_risk_challenger(rows, contract)
    evaluation_rows = [row for row in rows if row['split'] == 'evaluation']
    learned_risks = risk_scores(evaluation_rows, fitted)
    score_decisions, learned_decisions = gate_decisions(
        evaluation_rows, runtime_calibration, fitted, learned_risks)
    raw_decisions = {row['frame_key']: {
        component: row.get('base_v3_box') is not None
        for component in COMPONENTS} for row in evaluation_rows}
    specifications = {
        'raw_rejection_only': (raw_decisions, None, 'rejection_only'),
        'score_rejection_only': (
            score_decisions, None, 'rejection_only'),
        'score_heterogeneous': (
            score_decisions, learned_risks, 'heterogeneous'),
        'learned_risk_rejection_only': (
            learned_decisions, learned_risks, 'rejection_only'),
        'learned_risk_heterogeneous': (
            learned_decisions, learned_risks, 'heterogeneous')}
    outputs = {name: replay_policy(
        evaluation_rows, decisions, risks, policy)
        for name, (decisions, risks, policy) in specifications.items()}
    reports = {name: {
        group: state_report(selected, output, contract, annotated=True)
        for group, selected in _groups(evaluation_rows)}
        for name, output in outputs.items()}
    curves = matched_coverage(evaluation_rows, learned_risks, contract)
    minimum_wins = int(contract['component_risk_challenger'][
        'minimum_matched_coverage_wins'])
    reproduction = _verify_v4_reproduction(v4, reports)
    output_rows = []
    for row in rows:
        item = {key: copy.deepcopy(row.get(key)) for key in (
            'frame_key', 'domain', 'sequence', 'frame', 'split',
            'anchor_source', 'anchor_score')}
        item['online_features'] = row['online_features']
        item['offline_errors'] = row['errors']
        if row['split'] == 'evaluation':
            item['learned_component_risks'] = learned_risks[row['frame_key']]
            item['method_states'] = {
                name: output[row['frame_key']]['components']
                for name, output in outputs.items()}
        output_rows.append(item)
    heterogeneous = _heterogeneous_verdict(reports)
    risk_verdict = _risk_verdict(curves, minimum_wins)
    risk_passed_component_count = sum(
        item['passes_development_gate'] for item in risk_verdict.values())
    ready_to_freeze = (heterogeneous['passes_all'] and
                       risk_passed_component_count >= 2)
    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        claim_status='SOURCE_VAL_DEVELOPMENT_V5_ONLY',
        fixed_test_read=False, target_data_read=False,
        post_fixed_test_hypothesis_generation=True,
        eligible_for_fixed_test_reuse_as_final_validation=False,
        eligible_for_unknown_sequence_claim=False,
        threshold_fitting_performed=True,
        threshold_fitting_scope='calibration_prefix_only',
        feature_selection_performed=False,
        leakage_controls=dict(
            gt_used_in_online_features=False,
            gt_used_in_online_decisions=False,
            gt_used_for_calibration_prefix_error_labels=True,
            evaluation_gt_used_for_fitting=False,
            gt_attached_to_online_outputs_after_replay=True,
            domain_identity_used_in_decisions=False,
            future_frames_used=False,
            state_reset_on_sequence_or_frame_gap=True),
        split_counts=split_counts,
        fitted_component_risk_challenger=fitted,
        method_inventory=list(specifications),
        method_reports=reports,
        matched_coverage=curves,
        component_risk_development_verdict=risk_verdict,
        component_risk_passed_component_count=risk_passed_component_count,
        heterogeneous_policy_development_verdict=heterogeneous,
        v4_reproduction_check=reproduction,
        records=output_rows,
        decision=('FREEZE_V5_FOR_FRESH_SEQUENCE_EVALUATION'
                  if ready_to_freeze else
                  'KEEP_V5_AS_SOURCE_DEVELOPMENT_AND_REVISE_ONCE'),
        claim_limit=contract['claim_limit'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--baseline-contract', required=True)
    parser.add_argument('--runtime-calibration', required=True)
    parser.add_argument('--v4-report', required=True)
    parser.add_argument('--v4-contract', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    identities = {name: _identity(path) for name, path in (
        ('baseline', args.baseline),
        ('baseline_contract', args.baseline_contract),
        ('runtime_calibration', args.runtime_calibration),
        ('v4_report', args.v4_report), ('v4_contract', args.v4_contract),
        ('contract', args.contract))}
    baseline = json.loads(Path(args.baseline).read_text())
    baseline_contract = json.loads(Path(args.baseline_contract).read_text())
    runtime_calibration = json.loads(Path(args.runtime_calibration).read_text())
    v4 = json.loads(Path(args.v4_report).read_text())
    contract = json.loads(Path(args.contract).read_text())
    validate_contract(contract)
    expected = contract['expected_inputs']
    for role, field in (
            ('baseline_contract', 'baseline_contract_sha256'),
            ('runtime_calibration', 'runtime_calibration_sha256'),
            ('v4_report', 'v4_report_sha256'),
            ('v4_contract', 'v4_contract_sha256')):
        _require_identity(identities[role], expected[field], role)
    portable = portable_digest(baseline)
    if portable != expected['baseline_portable_sha256']:
        raise RuntimeError('Portable source baseline identity mismatch')
    payload = run(
        baseline, baseline_contract, runtime_calibration, v4, contract)
    payload['inputs'] = identities
    payload['inputs']['baseline_portable_sha256'] = portable
    output = _write_exact(args.out_json, payload)
    print(json.dumps(dict(
        output=output, split_counts=payload['split_counts'],
        method_count=len(payload['method_inventory']),
        decision=payload['decision'],
        heterogeneous_policy_passed=payload[
            'heterogeneous_policy_development_verdict']['passes_all'],
        component_risk_passed={component: value[
            'passes_development_gate'] for component, value in payload[
                'component_risk_development_verdict'].items()},
        claim_status=payload['claim_status'],
        fixed_test_read=False), indent=2))


if __name__ == '__main__':
    main()
