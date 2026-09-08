#!/usr/bin/env python3
"""Explore component reliability and bounded continuous OBB observations.

Online features use only current/past predictions. GT is attached afterwards
for offline evaluation. Calibration uses pooled sequence prefixes; evaluation
uses the held-out suffixes and reports real/sim separately.
"""

import argparse
import bisect
import json
import math
from pathlib import Path

import numpy as np

from crane_project.tools.base_v3_obb_reliability_baseline import (
    _identity, _write_exact)
from crane_project.tools.base_v3_reliability_failure_audit import validate
from crane_project.tools.eval_crane_offline import angle_diff


PROTOCOL = 'base_v3_obb_component_reliability_continuous_v1'
CONTRACT_PROTOCOL = (
    'base_v3_obb_component_reliability_continuous_contract_v1')
COMPONENTS = ('center', 'scale', 'angle')


def _periodic_error_deg(first, second):
    return abs(math.degrees(float(angle_diff(
        np.asarray([first]), np.asarray([second]))[0])))


def _geometry(box):
    long_side, short_side = sorted(
        (float(box[2]), float(box[3])), reverse=True)
    return np.asarray(box[:2], dtype=np.float64), long_side, short_side, float(box[4])


def _frame_features(row, history):
    """Create inference-time features. This function never reads GT/errors."""
    current = np.asarray(row['base_v3_box'], dtype=np.float64)
    anchor = (np.asarray(row['k1_box'], dtype=np.float64)
              if row['anchor_source'] == 'k1'
              else np.asarray(row['dino_box'], dtype=np.float64))
    center, long_side, short_side, angle = _geometry(current)
    anchor_center, anchor_long, anchor_short, anchor_angle = _geometry(anchor)
    diagonal = max(math.hypot(long_side, short_side), 1e-9)
    features = dict(
        anchor_uncertainty=1.0-float(row['anchor_score']),
        refiner_center_correction_norm=float(
            np.linalg.norm(center-anchor_center)/diagonal),
        refiner_scale_correction_log=max(
            abs(math.log(long_side/anchor_long)),
            abs(math.log(short_side/anchor_short))),
        refiner_angle_correction_norm=(
            _periodic_error_deg(angle, anchor_angle)/90.0),
        aspect_ambiguity=short_side/long_side)

    if row.get('k1_box') is not None and row.get('dino_box') is not None:
        k_center, k_long, k_short, k_angle = _geometry(
            np.asarray(row['k1_box'], dtype=np.float64))
        d_center, d_long, d_short, d_angle = _geometry(
            np.asarray(row['dino_box'], dtype=np.float64))
        norm = max((math.hypot(k_long, k_short)
                    + math.hypot(d_long, d_short))/2.0, 1e-9)
        features.update(
            k1_dino_center_disagreement_norm=float(
                np.linalg.norm(k_center-d_center)/norm),
            k1_dino_scale_disagreement_log=max(
                abs(math.log(k_long/d_long)),
                abs(math.log(k_short/d_short))),
            k1_dino_angle_disagreement_norm=(
                _periodic_error_deg(k_angle, d_angle)/90.0))
    else:
        features.update(
            k1_dino_center_disagreement_norm=None,
            k1_dino_scale_disagreement_log=None,
            k1_dino_angle_disagreement_norm=None)

    features.update(
        causal_center_residual_norm=None,
        causal_scale_residual_log=None,
        causal_angle_residual_norm=None)
    if history:
        previous = history[-1]
        p_center, p_long, p_short, p_angle = _geometry(
            np.asarray(previous['base_v3_box'], dtype=np.float64))
        predicted_center = p_center
        predicted_long, predicted_short, predicted_angle = p_long, p_short, p_angle
        if len(history) >= 2:
            older = history[-2]
            o_center, o_long, o_short, o_angle = _geometry(
                np.asarray(older['base_v3_box'], dtype=np.float64))
            step = max(previous['frame']-older['frame'], 1)
            horizon = row['frame']-previous['frame']
            predicted_center = p_center+(p_center-o_center)*(horizon/step)
            predicted_long = math.exp(
                math.log(p_long)+(math.log(p_long)-math.log(o_long))*(horizon/step))
            predicted_short = math.exp(
                math.log(p_short)+(math.log(p_short)-math.log(o_short))*(horizon/step))
            predicted_angle = p_angle+float(angle_diff(
                np.asarray([p_angle]), np.asarray([o_angle]))[0])*(horizon/step)
        features.update(
            causal_center_residual_norm=float(
                np.linalg.norm(center-predicted_center)/diagonal),
            causal_scale_residual_log=max(
                abs(math.log(long_side/predicted_long)),
                abs(math.log(short_side/predicted_short))),
            causal_angle_residual_norm=(
                _periodic_error_deg(angle, predicted_angle)/90.0))
    return features


def build_feature_rows(rows):
    ordered = sorted(rows, key=lambda row: (
        row['domain'], row['sequence'], row['frame'], row['frame_key']))
    history = []
    last_identity = None
    output = []
    for row in ordered:
        identity = (row['domain'], row['sequence'])
        contiguous = (last_identity == identity and history
                      and row['frame'] == history[-1]['frame']+1)
        if not contiguous:
            history = []
        copied = dict(row)
        copied['online_features'] = _frame_features(row, history)
        output.append(copied)
        history.append(row)
        history = history[-2:]
        last_identity = identity
    return output


def _contiguous_segments(rows):
    segments = []
    current = []
    for row in sorted(rows, key=lambda item: (
            item["domain"], item["sequence"], item["frame"], item["frame_key"])):
        identity = (row["domain"], row["sequence"])
        if (current and (identity != (current[-1]["domain"], current[-1]["sequence"])
                         or row["frame"] != current[-1]["frame"]+1)):
            segments.append(current)
            current = []
        current.append(row)
    if current:
        segments.append(current)
    return segments


def assign_split(rows, fraction, minimum):
    counts = dict(calibration=0, evaluation=0)
    for segment in _contiguous_segments(rows):
        boundary = max(minimum, int(math.floor(len(segment)*fraction)))
        boundary = min(boundary, max(len(segment)-1, 0))
        for index, row in enumerate(segment):
            row["split"] = "calibration" if index < boundary else "evaluation"
            counts[row["split"]] += 1
    return counts


def _cdf(value, sorted_values):
    if value is None or not sorted_values:
        return None
    return bisect.bisect_right(sorted_values, value)/len(sorted_values)


def calibrate_risk(rows, risk_contract):
    calibration = [row for row in rows if row['split'] == 'calibration']
    feature_names = sorted({name for component in COMPONENTS
                            for name in risk_contract[component+'_features']})
    distributions = {name: sorted(
        row['online_features'][name] for row in calibration
        if row['online_features'][name] is not None) for name in feature_names}
    for row in rows:
        row['component_risk'] = {}
        for component in COMPONENTS:
            ranks = [_cdf(row['online_features'][name], distributions[name])
                     for name in risk_contract[component+'_features']]
            available = [rank for rank in ranks if rank is not None]
            row['component_risk'][component] = float(sum(available)/len(available))
    target = float(risk_contract['target_measurement_coverage'])
    score_values = sorted(float(row['anchor_score']) for row in calibration)
    score_threshold = float(np.quantile(score_values, 1.0-target))
    risk_thresholds = {}
    for component in COMPONENTS:
        values = [row['component_risk'][component] for row in calibration]
        risk_thresholds[component] = float(np.quantile(values, target))
    return dict(
        feature_distributions={name: dict(
            count=len(values), minimum=min(values), maximum=max(values))
            for name, values in distributions.items()},
        score_threshold=score_threshold, risk_thresholds=risk_thresholds)


def _raw_component_error(row, component):
    if component == 'center':
        return float(row['errors']['center_error_px'])
    if component == 'scale':
        return max(float(row['errors']['long_relative_error']),
                   float(row['errors']['short_relative_error']))
    return float(row['errors']['angle_error_deg'])


def _error_from_value(value, row, component):
    gt_center, gt_long, gt_short, gt_angle = _geometry(
        np.asarray(row['gt_box'], dtype=np.float64))
    if component == 'center':
        return float(np.linalg.norm(np.asarray(value)-gt_center))
    if component == 'scale':
        return max(abs(value[0]-gt_long)/gt_long,
                   abs(value[1]-gt_short)/gt_short)
    return _periodic_error_deg(float(value), gt_angle)


def _measurement_value(row, component):
    center, long_side, short_side, angle = _geometry(
        np.asarray(row['base_v3_box'], dtype=np.float64))
    return center.tolist() if component == 'center' else (
        [long_side, short_side] if component == 'scale' else angle)


def decisions(rows, calibration, mode):
    result = {}
    for row in rows:
        if mode == 'raw':
            result[row['frame_key']] = {component: True for component in COMPONENTS}
        elif mode == 'score':
            accepted = row['anchor_score'] >= calibration['score_threshold']
            result[row['frame_key']] = {component: accepted for component in COMPONENTS}
        else:
            result[row['frame_key']] = {
                component: row['component_risk'][component]
                <= calibration['risk_thresholds'][component]
                for component in COMPONENTS}
    return result


def continuous_states(rows, accepted, max_gap, enable_prediction):
    states = {}
    histories = {component: [] for component in COMPONENTS}
    last_identity = None
    last_frame = None
    for row in sorted(rows, key=lambda item: (
            item['domain'], item['sequence'], item['frame'], item['frame_key'])):
        identity = (row['domain'], row['sequence'])
        if identity != last_identity or (last_frame is not None
                                         and row['frame'] != last_frame+1):
            histories = {component: [] for component in COMPONENTS}
        states[row['frame_key']] = {}
        for component in COMPONENTS:
            history = histories[component]
            if accepted[row['frame_key']][component]:
                value = _measurement_value(row, component)
                state = 'measurement'
                history.append((row['frame'], value))
                histories[component] = history[-2:]
            elif enable_prediction and history and row['frame']-history[-1][0] <= max_gap:
                state = 'prediction'
                value = history[-1][1]
                if component == 'center' and len(history) >= 2:
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
                    value, row, component))
        last_identity, last_frame = identity, row['frame']
    return states


def _longest_run(rows, states, component, state_name):
    longest = current = 0
    previous = None
    for row in sorted(rows, key=lambda item: (
            item['domain'], item['sequence'], item['frame'], item['frame_key'])):
        identity = (row['domain'], row['sequence'])
        contiguous = previous is not None and previous[0] == identity and row['frame'] == previous[1]+1
        if not contiguous:
            current = 0
        current = current+1 if states[row['frame_key']][component]['state'] == state_name else 0
        longest = max(longest, current)
        previous = (identity, row['frame'])
    return longest


def _rejection_runs(rows, accepted, component):
    runs = []
    current = 0
    previous = None
    for row in sorted(rows, key=lambda item: (
            item["domain"], item["sequence"], item["frame"], item["frame_key"])):
        identity = (row["domain"], row["sequence"])
        contiguous = (previous is not None and previous[0] == identity
                      and row["frame"] == previous[1]+1)
        if not contiguous:
            if current:
                runs.append(dict(length=current, recovered=False))
            current = 0
        if accepted[row["frame_key"]][component]:
            if current:
                runs.append(dict(length=current, recovered=True))
                current = 0
        else:
            current += 1
        previous = (identity, row["frame"])
    if current:
        runs.append(dict(length=current, recovered=False))
    return runs


def evaluate_method(rows, accepted, states, thresholds):
    report = {}
    for component in COMPONENTS:
        threshold = float(thresholds[component])
        raw_errors = [_raw_component_error(row, component) for row in rows]
        component_states = [states[row['frame_key']][component] for row in rows]
        measurements = [state for state in component_states if state['state'] == 'measurement']
        predictions = [state for state in component_states if state['state'] == 'prediction']
        available = [state for state in component_states if state['error'] is not None]
        bad_indices = [index for index, error in enumerate(raw_errors) if error > threshold]
        accepted_bad = sum(
            accepted[row['frame_key']][component] and raw_errors[index] > threshold
            for index, row in enumerate(rows))
        accepted_good = sum(
            accepted[row['frame_key']][component] and raw_errors[index] <= threshold
            for index, row in enumerate(rows))
        good_count = len(rows)-len(bad_indices)
        rejection_runs = _rejection_runs(rows, accepted, component)
        recovered_runs = [run for run in rejection_runs if run["recovered"]]
        report[component] = dict(
            frame_count=len(rows), measurement_count=len(measurements),
            prediction_count=len(predictions),
            unavailable_count=len(rows)-len(available),
            measurement_coverage=len(measurements)/len(rows),
            output_coverage=len(available)/len(rows),
            mean_measurement_error=(None if not measurements else float(np.mean(
                [state['error'] for state in measurements]))),
            mean_prediction_error=(None if not predictions else float(np.mean(
                [state['error'] for state in predictions]))),
            mean_available_output_error=(None if not available else float(np.mean(
                [state['error'] for state in available]))),
            false_acceptance_rate=(None if not measurements else
                accepted_bad/len(measurements)),
            bad_observation_rejection_recall=(None if not bad_indices else
                (len(bad_indices)-accepted_bad)/len(bad_indices)),
            correct_measurement_retention=(None if not good_count else
                accepted_good/good_count),
            rejection_run_count=len(rejection_runs),
            mean_recovery_delay_frames=(None if not recovered_runs else
                float(np.mean([run["length"] for run in recovered_runs]))),
            maximum_recovery_delay_frames=(None if not recovered_runs else
                max(run["length"] for run in recovered_runs)),
            longest_unavailable_run=_longest_run(
                rows, states, component, 'unavailable'))
    return report


def _risk_curve(rows, component, coverages, method):
    if method == 'score':
        ordered = sorted(rows, key=lambda row: (-row['anchor_score'], row['frame_key']))
    else:
        ordered = sorted(rows, key=lambda row: (
            row['component_risk'][component], row['frame_key']))
    curve = []
    for coverage in coverages:
        kept = ordered[:max(1, int(round(len(ordered)*coverage)))]
        errors = [_raw_component_error(row, component) for row in kept]
        curve.append(dict(target_coverage=float(coverage), retained_count=len(kept),
                          mean_error=float(np.mean(errors))))
    return curve


def run(baseline, baseline_contract, contract):
    validate(baseline, baseline_contract)
    rows = build_feature_rows(baseline['records'])
    split_counts = assign_split(
        rows, float(contract['split']['calibration_prefix_fraction']),
        int(contract['split']['minimum_calibration_frames_per_segment']))
    calibration = calibrate_risk(rows, contract['risk'])
    evaluation = [row for row in rows if row['split'] == 'evaluation']
    thresholds = dict(
        center=contract['evaluation']['center_error_threshold_px'],
        scale=contract['evaluation']['scale_relative_error_threshold'],
        angle=contract['evaluation']['angle_error_threshold_deg'])
    method_reports = {}
    record_states = {}
    for name, decision_mode, prediction in (
            ('raw_base_v3', 'raw', False),
            ('score_gate', 'score', False),
            ('component_gate', 'component', False),
            ('component_gate_continuous', 'component', True)):
        accepted = decisions(evaluation, calibration, decision_mode)
        states = continuous_states(
            evaluation, accepted,
            int(contract['continuous_output']['max_prediction_gap_frames']),
            prediction)
        method_reports[name] = {
            group: evaluate_method(
                selected, accepted, states, thresholds)
            for group, selected in [('all', evaluation)] + [
                (domain, [row for row in evaluation if row['domain'] == domain])
                for domain in ('real', 'sim')]}
        record_states[name] = states
    coverages = contract['risk']['risk_coverage_targets']
    risk_curves = {}
    for group, selected in [('all', evaluation)] + [
            (domain, [row for row in evaluation if row['domain'] == domain])
            for domain in ('real', 'sim')]:
        risk_curves[group] = {component: {
            method: _risk_curve(selected, component, coverages, method)
            for method in ('score', 'component_risk')}
            for component in COMPONENTS}
    output_rows = []
    for row in rows:
        output = {key: row[key] for key in (
            'frame_key', 'domain', 'sequence', 'frame', 'split',
            'anchor_source', 'anchor_score')}
        output['online_features'] = row['online_features']
        output['component_risk'] = row['component_risk']
        output['offline_errors'] = row['errors']
        if row['split'] == 'evaluation':
            output['method_states'] = {
                name: states[row['frame_key']]
                for name, states in record_states.items()}
        output_rows.append(output)
    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        fixed_test_read=False, target_data_read=False,
        claim_status='EXPLORATORY_SAME_SEQUENCE_SUFFIX_ONLY',
        leakage_controls=dict(
            gt_used_in_online_features=False,
            gt_used_in_risk_calibration=False,
            domain_identity_used_in_decisions=False,
            future_frames_used=False),
        split_counts=split_counts, calibration=calibration,
        thresholds=thresholds, risk_coverage=risk_curves,
        method_reports=method_reports, records=output_rows)


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
        raise ValueError('Unexpected component reliability contract')
    payload = run(baseline, baseline_contract, contract)
    payload['inputs'] = dict(
        baseline=_identity(args.baseline),
        baseline_contract=_identity(args.baseline_contract),
        contract=_identity(args.contract))
    identity = _write_exact(args.out_json, payload)
    print(json.dumps(dict(
        output=identity, split_counts=payload['split_counts'],
        claim_status=payload['claim_status'],
        leakage_controls=payload['leakage_controls']), indent=2))


if __name__ == '__main__':
    main()
