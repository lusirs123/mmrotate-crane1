#!/usr/bin/env python3
"""Ablate OBB reliability features and compare bounded output policies."""

import argparse
import bisect
import json
import math
from pathlib import Path

import numpy as np

from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    COMPONENTS, _contiguous_segments, _error_from_value, _geometry,
    _longest_run, _measurement_value, _raw_component_error,
    _rejection_runs, assign_split, build_feature_rows)
from crane_project.tools.base_v3_obb_reliability_baseline import (
    _identity, _write_exact)
from crane_project.tools.base_v3_reliability_failure_audit import validate


PROTOCOL = 'base_v3_obb_component_reliability_ablation_v2'
CONTRACT_PROTOCOL = 'base_v3_obb_component_reliability_ablation_contract_v2'


def _cdf(value, values):
    if value is None or not values:
        return None
    return bisect.bisect_right(values, value)/len(values)


def calibrate(rows, contract):
    calibration = [row for row in rows if row['split'] == 'calibration']
    names = sorted({name for groups in contract['feature_groups'].values()
                    for features in groups.values() for name in features})
    distributions = {name: sorted(
        row['online_features'][name] for row in calibration
        if row['online_features'][name] is not None) for name in names}
    for row in rows:
        row['risk_ablation'] = {}
        for component, groups in contract['feature_groups'].items():
            row['risk_ablation'][component] = {}
            for group, features in groups.items():
                ranks = [_cdf(row['online_features'][name], distributions[name])
                         for name in features]
                available = [rank for rank in ranks if rank is not None]
                row['risk_ablation'][component][group] = (
                    float(np.mean(available)) if available else 1.0)
    target = float(contract['target_measurement_coverage'])
    selected = contract['selected_gate_group']
    thresholds = {
        component: float(np.quantile([
            row['risk_ablation'][component][selected]
            for row in calibration], target))
        for component in COMPONENTS}
    score_threshold = float(np.quantile(
        [row['anchor_score'] for row in calibration], 1.0-target))
    return dict(
        empirical_cdf_values=distributions,
        score_threshold=score_threshold,
        selected_group=selected,
        component_risk_thresholds=thresholds)


def gate_decisions(rows, calibration, mode):
    decisions = {}
    for row in rows:
        if mode == 'raw':
            accepted = {component: True for component in COMPONENTS}
        elif mode == 'score':
            value = row['anchor_score'] >= calibration['score_threshold']
            accepted = {component: value for component in COMPONENTS}
        else:
            selected = calibration['selected_group']
            accepted = {component:
                row['risk_ablation'][component][selected]
                <= calibration['component_risk_thresholds'][component]
                for component in COMPONENTS}
        decisions[row['frame_key']] = accepted
    return decisions


def output_states(rows, accepted, max_gap, policy, forced_missing=None):
    forced_missing = set(forced_missing or ())
    states = {}
    histories = {component: [] for component in COMPONENTS}
    previous_identity = None
    previous_frame = None
    for row in sorted(rows, key=lambda item: (
            item['domain'], item['sequence'], item['frame'], item['frame_key'])):
        identity = (row['domain'], row['sequence'])
        if (identity != previous_identity or previous_frame is None
                or row['frame'] != previous_frame+1):
            histories = {component: [] for component in COMPONENTS}
        states[row['frame_key']] = {}
        for component in COMPONENTS:
            history = histories[component]
            measurement = (accepted[row['frame_key']][component]
                           and row['frame_key'] not in forced_missing)
            if measurement:
                state = 'measurement'
                value = _measurement_value(row, component)
                history.append((row['frame'], value))
                histories[component] = history[-2:]
            elif (policy != 'none' and history
                  and row['frame']-history[-1][0] <= max_gap):
                state = 'prediction'
                value = history[-1][1]
                if (policy == 'center_constant_velocity'
                        and component == 'center' and len(history) >= 2):
                    old_frame, old_value = history[-2]
                    new_frame, new_value = history[-1]
                    velocity = ((np.asarray(new_value)-np.asarray(old_value))
                                / max(new_frame-old_frame, 1))
                    value = (np.asarray(new_value)
                             + velocity*(row['frame']-new_frame)).tolist()
            else:
                state, value = 'unavailable', None
            states[row['frame_key']][component] = dict(
                state=state, value=value,
                error=None if value is None else _error_from_value(
                    value, row, component),
                forced_missing=row['frame_key'] in forced_missing)
        previous_identity, previous_frame = identity, row['frame']
    return states


def _quantiles(values, requested):
    return ({str(value): float(np.quantile(values, value))
             for value in requested} if values else {})


def evaluate(rows, accepted, states, thresholds, quantiles):
    report = {}
    for component in COMPONENTS:
        threshold = float(thresholds[component])
        observations = [(row, states[row['frame_key']][component])
                        for row in rows]
        measurements = [(row, state) for row, state in observations
                        if state['state'] == 'measurement']
        predictions = [(row, state) for row, state in observations
                       if state['state'] == 'prediction']
        available = [(row, state) for row, state in observations
                     if state['error'] is not None]
        bad_available = sum(state['error'] > threshold
                            for _, state in available)
        correct_available = len(available)-bad_available
        prediction_deltas = [
            state['error']-_raw_component_error(row, component)
            for row, state in predictions]
        effective_accepted = {
            row['frame_key']: dict(accepted[row['frame_key']], **{
                component: state['state'] == 'measurement'})
            for row, state in observations}
        rejection_runs = _rejection_runs(rows, effective_accepted, component)
        recovered = [run['length'] for run in rejection_runs if run['recovered']]
        report[component] = dict(
            frame_count=len(rows), measurement_count=len(measurements),
            prediction_count=len(predictions), available_count=len(available),
            unavailable_count=len(rows)-len(available),
            measurement_coverage=len(measurements)/len(rows),
            output_coverage=len(available)/len(rows),
            bad_available_output_rate=(None if not available else
                                       bad_available/len(available)),
            correct_output_coverage=correct_available/len(rows),
            measurement_error_quantiles=_quantiles(
                [state['error'] for _, state in measurements], quantiles),
            prediction_error_quantiles=_quantiles(
                [state['error'] for _, state in predictions], quantiles),
            available_error_quantiles=_quantiles(
                [state['error'] for _, state in available], quantiles),
            mean_measurement_error=(None if not measurements else float(np.mean(
                [state['error'] for _, state in measurements]))),
            mean_prediction_error=(None if not predictions else float(np.mean(
                [state['error'] for _, state in predictions]))),
            mean_available_output_error=(None if not available else float(np.mean(
                [state['error'] for _, state in available]))),
            prediction_improved_count=sum(delta < 0 for delta in prediction_deltas),
            prediction_degraded_count=sum(delta > 0 for delta in prediction_deltas),
            mean_prediction_error_delta_vs_raw=(None if not prediction_deltas else
                                                float(np.mean(prediction_deltas))),
            rejection_run_count=len(rejection_runs),
            mean_recovery_delay_frames=(None if not recovered else
                                        float(np.mean(recovered))),
            longest_unavailable_run=_longest_run(
                rows, states, component, 'unavailable'))
    return report


def risk_curves(rows, contract):
    output = {}
    thresholds = dict(
        center=contract['evaluation']['center_error_threshold_px'],
        scale=contract['evaluation']['scale_relative_error_threshold'],
        angle=contract['evaluation']['angle_error_threshold_deg'])
    for component, groups in contract['feature_groups'].items():
        output[component] = {}
        for group in groups:
            ordered = sorted(rows, key=lambda row: (
                row['risk_ablation'][component][group], row['frame_key']))
            curve = []
            for coverage in contract['risk_coverage_targets']:
                retained = ordered[:max(1, round(len(ordered)*coverage))]
                errors = [_raw_component_error(row, component) for row in retained]
                curve.append(dict(
                    target_coverage=float(coverage),
                    retained_count=len(retained), mean_error=float(np.mean(errors)),
                    failure_rate=sum(error > thresholds[component]
                                     for error in errors)/len(errors)))
            output[component][group] = curve
    return output


def synthetic_dropout_keys(rows, dropout):
    keys = set()
    runs = []
    stride = int(dropout['stride_frames'])
    offset = int(dropout['start_offset_frames'])
    lengths = list(map(int, dropout['run_lengths']))
    for segment in _contiguous_segments(rows):
        start = offset
        run_index = 0
        while start < len(segment):
            length = lengths[run_index % len(lengths)]
            selected = segment[start:start+length]
            selected_keys = [row['frame_key'] for row in selected]
            keys.update(selected_keys)
            if selected_keys:
                runs.append(dict(sequence=segment[0]['sequence'],
                                 domain=segment[0]['domain'],
                                 requested_length=length,
                                 frame_keys=selected_keys))
            run_index += 1
            start += stride
    return keys, runs


def run(baseline, baseline_contract, contract):
    validate(baseline, baseline_contract)
    rows = build_feature_rows(baseline['records'])
    split_counts = assign_split(
        rows, contract['split']['calibration_prefix_fraction'],
        contract['split']['minimum_calibration_frames_per_segment'])
    calibration = calibrate(rows, contract)
    evaluation_rows = [row for row in rows if row['split'] == 'evaluation']
    thresholds = dict(
        center=contract['evaluation']['center_error_threshold_px'],
        scale=contract['evaluation']['scale_relative_error_threshold'],
        angle=contract['evaluation']['angle_error_threshold_deg'])
    max_gap = int(contract['continuous_output']['max_prediction_gap_frames'])
    quantiles = contract['evaluation']['tail_quantiles']
    reports = {}
    states_by_method = {}
    for gate in ('raw', 'score', 'component'):
        accepted = gate_decisions(evaluation_rows, calibration, gate)
        policies = ('none',) if gate == 'raw' else contract['continuous_output']['policies']
        for policy in policies:
            name = '{}_{}'.format(gate, policy)
            states = output_states(evaluation_rows, accepted, max_gap, policy)
            reports[name] = {group: evaluate(
                selected, accepted, states, thresholds, quantiles)
                for group, selected in [('all', evaluation_rows)] + [
                    (domain, [row for row in evaluation_rows
                              if row['domain'] == domain])
                    for domain in ('real', 'sim')]}
            states_by_method[name] = states

    dropout = contract['continuous_output']['synthetic_dropout']
    dropout_keys, dropout_runs = synthetic_dropout_keys(evaluation_rows, dropout)
    dropout_reports = {}
    raw_accepted = gate_decisions(evaluation_rows, calibration, 'raw')
    if dropout['enabled']:
        for policy in dropout.get('policies', ('none', 'hold',
                                                'center_constant_velocity')):
            states = output_states(evaluation_rows, raw_accepted, max_gap, policy,
                                   forced_missing=dropout_keys)
            dropout_reports[policy] = evaluate(
                evaluation_rows, raw_accepted, states, thresholds, quantiles)

    curves = {group: risk_curves(selected, contract)
              for group, selected in [('all', evaluation_rows)] + [
                  (domain, [row for row in evaluation_rows
                            if row['domain'] == domain])
                  for domain in ('real', 'sim')]}
    output_rows = []
    for row in rows:
        item = {key: row[key] for key in (
            'frame_key', 'domain', 'sequence', 'frame', 'split',
            'anchor_source', 'anchor_score')}
        item['online_features'] = row['online_features']
        item['risk_ablation'] = row['risk_ablation']
        item['offline_errors'] = row['errors']
        if row['split'] == 'evaluation':
            item['method_states'] = {name: states[row['frame_key']]
                                     for name, states in states_by_method.items()}
            item['synthetic_dropout'] = row['frame_key'] in dropout_keys
        output_rows.append(item)
    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        fixed_test_read=False, target_data_read=False,
        claim_status='EXPLORATORY_ABLATION_AND_DROPOUT_STRESS_ONLY',
        leakage_controls=dict(
            gt_used_in_online_features=False,
            gt_used_in_risk_calibration=False,
            domain_identity_used_in_decisions=False,
            future_frames_used=False,
            synthetic_dropout_independent_of_gt=True),
        split_counts=split_counts, calibration=calibration,
        risk_ablation=curves, method_reports=reports,
        synthetic_dropout=dict(
            frame_count=len(dropout_keys), runs=dropout_runs,
            reports=dropout_reports,
            interpretation='Artificial observation-loss stress test; not real miss evidence.'),
        records=output_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--baseline-contract', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text())
    baseline_contract = json.loads(Path(args.baseline_contract).read_text())
    contract = json.loads(Path(args.contract).read_text())
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected V2 ablation contract')
    payload = run(baseline, baseline_contract, contract)
    payload['inputs'] = dict(
        baseline=_identity(args.baseline),
        baseline_contract=_identity(args.baseline_contract),
        contract=_identity(args.contract))
    output = _write_exact(args.out_json, payload)
    print(json.dumps(dict(
        output=output, split_counts=payload['split_counts'],
        method_count=len(payload['method_reports']),
        synthetic_dropout_frames=payload['synthetic_dropout']['frame_count'],
        claim_status=payload['claim_status'],
        leakage_controls=payload['leakage_controls']), indent=2))


if __name__ == '__main__':
    main()
