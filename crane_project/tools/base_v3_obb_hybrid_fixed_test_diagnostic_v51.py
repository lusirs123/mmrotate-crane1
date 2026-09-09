#!/usr/bin/env python3
"""Complete diagnostic comparisons for the frozen V5.1 fixed-TEST report.

This audit measures scale ranking at matched coverage, pairs every angle hold
with the raw current-frame angle, and reports joint component availability.
It is descriptive: it fits and selects nothing from TEST.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from crane_project.tools.base_v3_obb_reliability_baseline import (
    _identity, _write_exact)


PROTOCOL = 'base_v3_obb_hybrid_fixed_test_diagnostic_v51'
CONTRACT_PROTOCOL = 'base_v3_obb_hybrid_fixed_test_diagnostic_contract_v51'
INPUT_PROTOCOL = 'base_v3_obb_hybrid_fixed_test_eval_v51'
COMPONENTS = ('center', 'scale', 'angle')


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected V5.1 TEST diagnostic contract')
    if contract.get('fixed_test_read') is not True:
        raise ValueError('Diagnostic contract must declare fixed TEST read')
    if contract.get('fixed_test_previously_exposed') is not True:
        raise ValueError('Diagnostic contract must preserve prior exposure')
    if contract.get('parameter_tuning_authorized') is not False:
        raise ValueError('TEST diagnostic must not authorize parameter tuning')
    expected = contract.get('expected_input', {})
    if expected.get('protocol') != INPUT_PROTOCOL:
        raise ValueError('Contract must bind the V5.1 TEST report protocol')
    digest = str(expected.get('sha256', ''))
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Input report SHA256 must be lowercase hexadecimal')
    targets = contract.get('matched_coverage_targets', [])
    if not targets or any(not 0.0 < float(value) <= 1.0 for value in targets):
        raise ValueError('Matched coverage targets must lie in (0, 1]')


def validate_report(report, contract):
    if report.get('protocol') != INPUT_PROTOCOL:
        raise ValueError('Unexpected V5.1 fixed-TEST report')
    required_flags = {
        'fixed_test_read': True,
        'fixed_test_previously_exposed': True,
        'frozen_calibration': True,
        'threshold_fitting_performed': False,
        'policy_selection_performed': False,
        'eligible_for_parameter_tuning_from_this_report': False}
    for field, expected in required_flags.items():
        if report.get(field) is not expected:
            raise ValueError('Unexpected input report flag: ' + field)
    if set(report.get('method_inventory', [])) != {
            'raw', 'score_rejection_only', 'v51_hybrid'}:
        raise ValueError('Input report must contain all three methods')
    records = report.get('records')
    scope = contract['fixed_test_scope']
    if not isinstance(records, list) or len(records) != scope['frame_count']:
        raise ValueError('Fixed TEST frame count changed')
    if len({row['frame_key'] for row in records}) != len(records):
        raise ValueError('Duplicate frame keys in fixed TEST report')
    counts = Counter(row['domain'] for row in records)
    if dict(counts) != scope['domain_counts']:
        raise ValueError('Fixed TEST domain counts changed')
    for row in records:
        if not math.isfinite(float(row['anchor_score'])):
            raise ValueError('Every fixed TEST row needs a finite score')
        observations = row.get('online_observations', {})
        errors = row.get('offline_errors', {})
        for method in ('raw', 'score_rejection_only', 'v51_hybrid'):
            if set(observations.get(method, {})) != set(COMPONENTS):
                raise ValueError('Incomplete online observations')
            if set(errors.get(method, {})) != set(COMPONENTS):
                raise ValueError('Incomplete offline errors')
        risk = observations['v51_hybrid']['scale'].get('risk')
        if risk is None or not math.isfinite(float(risk)):
            raise ValueError('Every fixed TEST row needs a finite scale risk')


def groups(records):
    output = [('all', records)]
    output.extend((domain, [r for r in records if r['domain'] == domain])
                  for domain in sorted({r['domain'] for r in records}))
    output.extend(('sequence:' + sequence,
                   [r for r in records if r['sequence'] == sequence])
                  for sequence in sorted({r['sequence'] for r in records}))
    output.extend(('anchor:' + source,
                   [r for r in records if r['anchor_source'] == source])
                  for source in sorted({r['anchor_source'] for r in records}))
    return output


def _mean(values):
    return None if not values else float(np.mean(values))


def _quantiles(values):
    if not values:
        return None
    return {str(q): float(np.quantile(values, q))
            for q in (0.5, 0.9, 0.95, 1.0)}


def _ranking_point(ordered, retained_count, total, threshold):
    retained = ordered[:retained_count]
    errors = [float(row['offline_errors']['raw']['scale'])
              for row in retained]
    correct = sum(error <= threshold for error in errors)
    return dict(
        retained_count=retained_count,
        realized_coverage=retained_count/total,
        mean_error=_mean(errors), error_quantiles=_quantiles(errors),
        failure_rate=sum(error > threshold for error in errors)/retained_count,
        correct_retained_count=correct,
        correct_output_coverage=correct/total,
        retained_frame_keys=[row['frame_key'] for row in retained])


def scale_matched_coverage(records, contract):
    threshold = float(contract['evaluation']['scale_relative_error_threshold'])
    result = {}
    for name, selected in groups(records):
        score_order = sorted(
            selected, key=lambda row: (-float(row['anchor_score']),
                                       row['frame_key']))
        risk_order = sorted(selected, key=lambda row: (
            float(row['online_observations']['v51_hybrid']['scale']['risk']),
            row['frame_key']))
        points = []
        for target in contract['matched_coverage_targets']:
            count = max(1, round(len(selected)*float(target)))
            score = _ranking_point(score_order, count, len(selected), threshold)
            risk = _ranking_point(risk_order, count, len(selected), threshold)
            points.append(dict(
                target_coverage=float(target), anchor_score=score,
                learned_scale_risk=risk,
                learned_minus_score=dict(
                    mean_error=risk['mean_error']-score['mean_error'],
                    failure_rate=risk['failure_rate']-score['failure_rate'],
                    correct_output_coverage=(
                        risk['correct_output_coverage']-
                        score['correct_output_coverage']),
                    wins_mean_and_failure=(
                        risk['mean_error'] <= score['mean_error'] and
                        risk['failure_rate'] <= score['failure_rate']))))
        result[name] = dict(frame_count=len(selected), points=points,
                            risk_wins_both_count=sum(
                                point['learned_minus_score'][
                                    'wins_mean_and_failure']
                                for point in points))
    return result


def _ordered(records):
    return sorted(records, key=lambda row: (
        row['domain'], row['sequence'], int(row['frame']), row['frame_key']))


def angle_rejection_run_lengths(records):
    lengths = {}
    run = []
    last_identity = None
    last_frame = None

    def finish():
        for item in run:
            lengths[item['frame_key']] = len(run)

    for row in _ordered(records):
        identity = (row['domain'], row['sequence'])
        contiguous = identity == last_identity and (
            last_frame is not None and int(row['frame']) == last_frame+1)
        state = row['online_observations']['v51_hybrid']['angle']['state']
        if not contiguous or state == 'measurement':
            finish()
            run = []
        if state != 'measurement':
            run.append(row)
        last_identity, last_frame = identity, int(row['frame'])
    finish()
    return lengths


def _run_bucket(length):
    if length == 1:
        return '1'
    if length <= 3:
        return '2-3'
    if length <= 9:
        return '4-9'
    return '10+'


def _paired_angle_report(selected, run_lengths, threshold):
    predictions = [row for row in selected if row['online_observations'][
        'v51_hybrid']['angle']['state'] == 'prediction']
    confusion = Counter()
    deltas = []
    buckets = defaultdict(list)
    cases = []
    for row in predictions:
        raw = float(row['offline_errors']['raw']['angle'])
        held = float(row['offline_errors']['v51_hybrid']['angle'])
        raw_label = 'raw_correct' if raw <= threshold else 'raw_bad'
        held_label = 'hold_correct' if held <= threshold else 'hold_bad'
        confusion[raw_label + '_to_' + held_label] += 1
        delta = held-raw
        deltas.append(delta)
        relation = ('improved' if delta < -1e-12 else
                    'degraded' if delta > 1e-12 else 'tied')
        length = run_lengths[row['frame_key']]
        buckets[_run_bucket(length)].append(delta)
        cases.append(dict(
            frame_key=row['frame_key'], anchor_source=row['anchor_source'],
            raw_error=raw, hold_error=held, delta_vs_raw=delta,
            relation=relation, full_rejection_run_length=length))
    return dict(
        frame_count=len(selected), prediction_count=len(predictions),
        correctness_transitions=dict(confusion),
        improved_count=sum(delta < -1e-12 for delta in deltas),
        degraded_count=sum(delta > 1e-12 for delta in deltas),
        tied_count=sum(abs(delta) <= 1e-12 for delta in deltas),
        mean_error_delta_vs_raw=_mean(deltas),
        error_delta_quantiles=_quantiles(deltas),
        by_full_rejection_run_length={name: dict(
            prediction_count=len(values), mean_error_delta_vs_raw=_mean(values),
            improved_count=sum(value < -1e-12 for value in values),
            degraded_count=sum(value > 1e-12 for value in values))
            for name, values in sorted(buckets.items())},
        prediction_cases=cases)


def angle_pair_audit(records, contract):
    threshold = float(contract['evaluation']['angle_error_threshold_deg'])
    run_lengths = angle_rejection_run_lengths(records)
    return {name: _paired_angle_report(selected, run_lengths, threshold)
            for name, selected in groups(records)}


def _longest_run(selected, predicate):
    longest = current = 0
    last_identity = None
    last_frame = None
    for row in _ordered(selected):
        identity = (row['domain'], row['sequence'])
        if identity != last_identity or last_frame is None or (
                int(row['frame']) != last_frame+1):
            current = 0
        current = current+1 if predicate(row) else 0
        longest = max(longest, current)
        last_identity, last_frame = identity, int(row['frame'])
    return longest


def _joint_report(selected, method, thresholds):
    masks = Counter()
    state_tuples = Counter()
    all_valid = all_measurement = all_correct = false_all_valid = 0
    partial = none_valid = 0
    for row in selected:
        observations = row['online_observations'][method]
        valid = [component for component in COMPONENTS
                 if observations[component]['valid']]
        masks['+'.join(valid) if valid else 'none'] += 1
        state_tuples['|'.join(observations[component]['state']
                              for component in COMPONENTS)] += 1
        if len(valid) == 3:
            all_valid += 1
            all_measurement += all(observations[component]['state'] ==
                                   'measurement' for component in COMPONENTS)
            correct = all(
                float(row['offline_errors'][method][component]) <=
                thresholds[component] for component in COMPONENTS)
            all_correct += correct
            false_all_valid += not correct
        elif not valid:
            none_valid += 1
        else:
            partial += 1

    def all_valid_predicate(row):
        return all(row['online_observations'][method][component]['valid']
                   for component in COMPONENTS)

    def none_valid_predicate(row):
        return not any(row['online_observations'][method][component]['valid']
                       for component in COMPONENTS)

    count = len(selected)
    return dict(
        frame_count=count, all_components_valid_count=all_valid,
        all_components_valid_coverage=all_valid/count,
        all_components_measurement_count=all_measurement,
        partial_valid_count=partial, all_components_unavailable_count=none_valid,
        jointly_correct_count=all_correct,
        jointly_correct_coverage=all_correct/count,
        false_all_components_valid_count=false_all_valid,
        false_all_components_valid_rate=(
            None if not all_valid else false_all_valid/all_valid),
        longest_incomplete_obb_run=_longest_run(
            selected, lambda row: not all_valid_predicate(row)),
        longest_all_components_unavailable_run=_longest_run(
            selected, none_valid_predicate),
        validity_mask_counts=dict(masks), state_tuple_counts=dict(state_tuples))


def joint_availability(records, contract):
    evaluation = contract['evaluation']
    thresholds = dict(
        center=float(evaluation['center_error_threshold_px']),
        scale=float(evaluation['scale_relative_error_threshold']),
        angle=float(evaluation['angle_error_threshold_deg']))
    return {method: {name: _joint_report(selected, method, thresholds)
                     for name, selected in groups(records)}
            for method in ('raw', 'score_rejection_only', 'v51_hybrid')}


def run(report, contract, input_identity=None):
    validate_contract(contract)
    validate_report(report, contract)
    records = report['records']
    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        claim_status='POST_EXPOSURE_FIXED_TEST_DIAGNOSTIC_ONLY',
        fixed_test_read=True, fixed_test_previously_exposed=True,
        descriptive_analysis_only=True,
        threshold_fitting_performed=False,
        feature_selection_performed=False,
        policy_selection_performed=False,
        parameter_tuning_authorized=False,
        input=input_identity,
        evaluation_thresholds=contract['evaluation'],
        scale_matched_coverage=scale_matched_coverage(records, contract),
        angle_hold_pair_audit=angle_pair_audit(records, contract),
        joint_component_availability=joint_availability(records, contract),
        eligible_for_parameter_tuning_from_this_report=False,
        eligible_for_final_independent_validation_claim=False,
        claim_limit=contract['claim_limit'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-test-v51-report', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    report_id = _identity(args.fixed_test_v51_report)
    contract_id = _identity(args.contract)
    report = json.loads(Path(args.fixed_test_v51_report).read_text())
    contract = json.loads(Path(args.contract).read_text())
    validate_contract(contract)
    if report_id['sha256'] != contract['expected_input']['sha256']:
        raise RuntimeError('V5.1 fixed-TEST report identity mismatch')
    payload = run(report, contract, report_id)
    payload['contract'] = contract_id
    output = _write_exact(args.out_json, payload)
    all_scale = payload['scale_matched_coverage']['all']
    all_angle = payload['angle_hold_pair_audit']['all']
    joint = {method: payload['joint_component_availability'][method]['all']
             for method in ('raw', 'score_rejection_only', 'v51_hybrid')}
    print(json.dumps(dict(
        output=output,
        scale_all_risk_wins_both_count=all_scale['risk_wins_both_count'],
        scale_all_comparison_count=len(all_scale['points']),
        angle_all_summary={key: all_angle[key] for key in (
            'prediction_count', 'improved_count', 'degraded_count',
            'mean_error_delta_vs_raw', 'correctness_transitions')},
        joint_all_summary={method: {key: value[key] for key in (
            'all_components_valid_coverage', 'jointly_correct_coverage',
            'false_all_components_valid_rate',
            'longest_incomplete_obb_run')}
            for method, value in joint.items()},
        claim_status=payload['claim_status'],
        parameter_tuning_authorized=False), indent=2))


if __name__ == '__main__':
    main()
