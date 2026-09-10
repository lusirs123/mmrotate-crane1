#!/usr/bin/env python3
"""Develop a split long/short-side reliability policy on source-val only.

V5.2 keeps the Base V3 K1-first/DINO-fallback detector and the V5.1 center
and angle decisions frozen.  It replaces the single scale-error ranker with
separate long-side and short-side error rankers, combines their calibration
percentiles into one scale-validity decision, and evaluates rejection before
an explicitly separate one-frame scale hold.  Fixed TEST is never read.
"""

import argparse
import bisect
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from crane_project.tools import (
    base_v3_obb_heterogeneous_policy_development_v5 as v5)
from crane_project.tools import base_v3_obb_hybrid_policy_freeze_v51 as v51
from crane_project.tools.base_v3_obb_component_policy_audit_v4 import _groups
from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    COMPONENTS, _geometry, _measurement_value, assign_split,
    build_feature_rows)
from crane_project.tools.base_v3_obb_external_sequence_eval import state_report
from crane_project.tools.base_v3_obb_observation_interface_v3 import (
    CALIBRATION_PROTOCOL)
from crane_project.tools.base_v3_obb_reliability_baseline import (
    PROTOCOL as BASELINE_PROTOCOL, _identity, _write_exact)
from crane_project.tools.base_v3_reliability_failure_audit import (
    portable_digest, validate)


PROTOCOL = 'base_v3_obb_split_scale_policy_development_v52'
CONTRACT_PROTOCOL = 'base_v3_obb_split_scale_policy_development_contract_v52'
SIDE_COMPONENTS = ('long_side', 'short_side')


def _require_identity(identity, expected, role):
    if identity['sha256'] != expected:
        raise RuntimeError('{} identity mismatch: expected {}, got {}'.format(
            role, expected, identity['sha256']))


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected V5.2 development contract')
    if contract.get('fixed_test_read') is not False:
        raise ValueError('V5.2 must not authorize fixed TEST')
    if contract.get('target_data_read') is not False:
        raise ValueError('V5.2 must remain source-only')
    if contract.get('post_fixed_test_hypothesis_generation') is not True:
        raise ValueError('V5.2 must preserve post-TEST development status')
    expected = contract.get('expected_inputs', {})
    if expected.get('baseline_protocol') != BASELINE_PROTOCOL:
        raise ValueError('V5.2 must bind the Base V3 source baseline')
    if expected.get('runtime_calibration_v3_protocol') != CALIBRATION_PROTOCOL:
        raise ValueError('V5.2 must bind the V3 score calibration')
    if expected.get('v5_report_protocol') != v5.PROTOCOL:
        raise ValueError('V5.2 must bind the V5 source report')
    if expected.get('v51_report_protocol') != v51.PROTOCOL:
        raise ValueError('V5.2 must bind the V5.1 source report')
    if contract.get('primary_candidate') != 'split_existing_features':
        raise ValueError('V5.2 primary candidate changed')
    if contract.get('automatic_ablation_promotion') is not False:
        raise ValueError('V5.2 ablations must not be promoted automatically')
    sets = contract.get('feature_sets', {})
    if set(sets) != {
            'split_existing_features', 'split_geometry_ablation',
            'split_image_quality_ablation'}:
        raise ValueError('Unexpected V5.2 feature-set inventory')
    existing = list(sets['split_existing_features'])
    if not existing:
        raise ValueError('V5.2 existing feature set must not be empty')
    geometry = list(sets['split_geometry_ablation'])
    image = list(sets['split_image_quality_ablation'])
    if not set(existing).issubset(geometry) or not set(existing).issubset(image):
        raise ValueError('Each ablation must preserve existing online features')
    if (set(geometry)-set(existing)) & (set(image)-set(existing)):
        raise ValueError('Geometry and image-quality ablations must stay separate')
    if int(contract['continuous_output']['scale_hold_max_frames']) != 1:
        raise ValueError('V5.2 scale hold must be exactly one-frame bounded')
    if contract['continuous_output'].get('hold_used_for_ranker_selection') is not False:
        raise ValueError('Hold results must not select the V5.2 ranker')


def validate_source_inputs(runtime_v3, v5_report, v51_report, contract):
    if runtime_v3.get('protocol') != CALIBRATION_PROTOCOL:
        raise ValueError('Unexpected V3 runtime calibration protocol')
    if v5_report.get('protocol') != v5.PROTOCOL:
        raise ValueError('Unexpected V5 report protocol')
    if v51_report.get('protocol') != v51.PROTOCOL:
        raise ValueError('Unexpected V5.1 report protocol')
    for name, report in (('V5', v5_report), ('V5.1', v51_report)):
        if report.get('fixed_test_read') is not False:
            raise ValueError('{} report violates fixed-TEST boundary'.format(name))
        if report.get('target_data_read') is not False:
            raise ValueError('{} report violates source-only boundary'.format(name))
        if report.get('split_counts') != {
                'calibration': int(contract['split'][
                    'expected_calibration_frames']),
                'evaluation': int(contract['split'][
                    'expected_evaluation_frames'])}:
            raise ValueError('{} report has unexpected split counts'.format(name))
    if v51_report.get('decision') != 'FREEZE_V51_FOR_FRESH_SEQUENCE_EVALUATION':
        raise ValueError('V5.2 expects the frozen V5.1 source policy')


def _side_error(row, side):
    field = '{}_relative_error'.format(
        'long' if side == 'long_side' else 'short')
    return float(row['errors'][field])


def _scale_error(row):
    return max(_side_error(row, side) for side in SIDE_COMPONENTS)


def _geometry_features(row):
    """Online box-size features; never access GT or offline errors."""
    _, long_side, short_side, _ = _geometry(
        np.asarray(row['base_v3_box'], dtype=np.float64))
    return {
        'obb_log_long_side_px': float(math.log(max(long_side, 1e-6))),
        'obb_log_short_side_px': float(math.log(max(short_side, 1e-6))),
        'obb_log_area_px2': float(math.log(max(long_side*short_side, 1e-6))),
    }


def _obb_axis_aligned_roi(box, width, height):
    cx, cy, side_a, side_b, angle = map(float, box)
    cosine, sine = abs(math.cos(angle)), abs(math.sin(angle))
    half_w = 0.5*(cosine*side_a+sine*side_b)
    half_h = 0.5*(sine*side_a+cosine*side_b)
    x0 = max(0, int(math.floor(cx-half_w)))
    y0 = max(0, int(math.floor(cy-half_h)))
    x1 = min(width, int(math.ceil(cx+half_w)))
    y1 = min(height, int(math.ceil(cy+half_h)))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError('Base V3 box produced an empty image ROI')
    return x0, y0, x1, y1


def _image_quality_features(row, image_dir):
    """Current-frame ROI statistics, independent of GT and phase labels."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('Pillow is required for image-quality ablation') from exc
    path = Path(image_dir)/('{}.jpg'.format(row['frame_key']))
    if not path.is_file():
        raise RuntimeError('Missing source-val image: {}'.format(path))
    with Image.open(str(path)) as handle:
        gray = np.asarray(handle.convert('L'), dtype=np.float64)
    x0, y0, x1, y1 = _obb_axis_aligned_roi(
        row['base_v3_box'], gray.shape[1], gray.shape[0])
    roi = gray[y0:y1, x0:x1]
    return {
        'roi_gray_mean_norm': float(np.mean(roi)/255.0),
        'roi_gray_std_norm': float(np.std(roi)/255.0),
        'roi_dark_fraction': float(np.mean(roi <= 32.0)),
        'roi_bright_fraction': float(np.mean(roi >= 240.0)),
        'roi_clipped_fraction': float(np.mean((roi <= 8.0) | (roi >= 247.0))),
    }


def enrich_online_features(rows, image_dir=None):
    """Append predeclared geometry and optional image-quality features."""
    output = build_feature_rows(rows)
    image_digest = hashlib.sha256()
    for row in output:
        row['online_features'].update(_geometry_features(row))
        if image_dir is not None:
            image_features = _image_quality_features(row, image_dir)
            row['online_features'].update(image_features)
            image_digest.update(json.dumps(
                [row['frame_key'], image_features], sort_keys=True,
                separators=(',', ':')).encode('utf-8'))
    return output, (None if image_dir is None else image_digest.hexdigest())


def _fit_side_ranker(rows, feature_names, contract):
    calibration = [row for row in rows if row['split'] == 'calibration']
    matrix, preprocessing = v5._matrix(calibration, feature_names)
    spec = contract['split_scale_ranker']
    threshold = float(contract['evaluation']['side_relative_error_threshold'])
    models = {}
    calibration_probabilities = {}
    for side in SIDE_COMPONENTS:
        labels = np.asarray([
            _side_error(row, side) > threshold for row in calibration],
            dtype=np.float64)
        weights, iterations, converged = v5._fit_logistic(
            matrix, labels, float(spec['l2_regularization']),
            int(spec['maximum_newton_iterations']),
            float(spec['convergence_tolerance']))
        probability = 1.0/(1.0+np.exp(-np.clip(
            matrix.dot(weights), -30.0, 30.0)))
        calibration_probabilities[side] = probability
        models[side] = {
            'weights': weights.tolist(), 'iterations': iterations,
            'converged': converged, 'positive_count': int(labels.sum()),
            'negative_count': int(len(labels)-labels.sum()),
            'calibration_probability_distribution': sorted(
                map(float, probability)),
        }
    combined = []
    for index in range(len(calibration)):
        combined.append(max(
            bisect.bisect_right(
                models[side]['calibration_probability_distribution'],
                float(calibration_probabilities[side][index]))/len(calibration)
            for side in SIDE_COMPONENTS))
    target = float(spec['target_measurement_coverage'])
    return {
        'model': spec['model'], 'feature_names': list(feature_names),
        'expanded_feature_names': list(feature_names)+[
            name+'__missing' for name in feature_names],
        'preprocessing': preprocessing,
        'l2_regularization': float(spec['l2_regularization']),
        'target_measurement_coverage': target,
        'combined_scale_risk_threshold': float(np.quantile(combined, target)),
        'components': models,
        'gt_role': 'calibration_prefix_long_short_error_labels_only',
        'online_gt_fields_consumed': [],
    }


def fit_candidates(rows, contract):
    fitted = {}
    for name, features in contract['feature_sets'].items():
        if any(feature not in rows[0]['online_features'] for feature in features):
            missing = [feature for feature in features
                       if feature not in rows[0]['online_features']]
            raise RuntimeError(
                '{} features unavailable: {}'.format(name, missing))
        fitted[name] = _fit_side_ranker(rows, list(features), contract)
    return fitted


def split_risk_scores(rows, fitted):
    matrix, _ = v5._matrix(
        rows, fitted['feature_names'], fitted['preprocessing'])
    output = {}
    for row_index, row in enumerate(rows):
        side_risks = {}
        for side in SIDE_COMPONENTS:
            model = fitted['components'][side]
            weights = np.asarray(model['weights'], dtype=np.float64)
            logit = float(np.clip(matrix[row_index].dot(weights), -30.0, 30.0))
            probability = float(1.0/(1.0+math.exp(-logit)))
            distribution = model['calibration_probability_distribution']
            percentile = bisect.bisect_right(distribution, probability)/len(
                distribution)
            side_risks[side] = {
                'probability_score': probability,
                'calibration_percentile_risk': float(percentile),
            }
        output[row['frame_key']] = {
            'long_side': side_risks['long_side'],
            'short_side': side_risks['short_side'],
            'combined_scale_risk': max(
                side_risks[side]['calibration_percentile_risk']
                for side in SIDE_COMPONENTS),
        }
    return output


def _candidate_decisions(rows, score_decisions, risks, fitted):
    output = {}
    threshold = float(fitted['combined_scale_risk_threshold'])
    for row in rows:
        key = row['frame_key']
        available = row.get('base_v3_box') is not None
        output[key] = {
            'center': bool(score_decisions[key]['center']),
            'scale': bool(available and risks[key][
                'combined_scale_risk'] <= threshold),
            'angle': bool(score_decisions[key]['angle']),
        }
    return output


def replay_candidate(rows, decisions, risks, scale_hold):
    """Replay V5.1 center/angle with rejection-only or one-frame scale hold."""
    if scale_hold not in (False, True):
        raise ValueError('scale_hold must be boolean')
    outputs = {}
    last_angle = None
    last_scale = None
    last_identity = None
    last_frame = None
    for row in v5._ordered(rows):
        identity = (row['domain'], row['sequence'])
        if (identity != last_identity or last_frame is None or
                int(row['frame']) != last_frame+1):
            last_angle = None
            last_scale = None
        observations = {}
        for component in COMPONENTS:
            accepted = decisions[row['frame_key']][component]
            if accepted:
                value = _measurement_value(row, component)
                state, source, age = 'measurement', 'base_v3_measurement', 0
                if component == 'angle':
                    last_angle = (int(row['frame']), copy.deepcopy(value))
                if component == 'scale':
                    last_scale = (int(row['frame']), copy.deepcopy(value))
            else:
                state, source, value, age = 'unavailable', None, None, None
                history = last_angle if component == 'angle' else (
                    last_scale if component == 'scale' else None)
                if history is not None:
                    age = int(row['frame'])-history[0]
                if component == 'angle' and history is not None and age == 1:
                    state, source = 'prediction', 'one_frame_angle_hold'
                    value = copy.deepcopy(history[1])
                if (component == 'scale' and scale_hold and
                        history is not None and age == 1):
                    state, source = 'prediction', 'one_frame_scale_hold'
                    value = copy.deepcopy(history[1])
            risk = None
            if component == 'scale':
                risk = risks[row['frame_key']]['combined_scale_risk']
            observations[component] = {
                'value': value, 'valid': state != 'unavailable',
                'state': state, 'source': source,
                'age_since_measurement_frames': age,
                'measurement_available': row.get('base_v3_box') is not None,
                'measurement_accepted': bool(accepted),
                'risk': risk,
                'gate': ('v52_split_long_short_scale_risk'
                         if component == 'scale' else 'anchor_score'),
                'reason': ('accepted_measurement' if accepted else
                           'detector_missing' if row.get('base_v3_box') is None
                           else 'rejected_measurement'),
            }
        outputs[row['frame_key']] = {
            'frame_key': row['frame_key'], 'domain': row['domain'],
            'sequence': row['sequence'], 'frame': int(row['frame']),
            'components': observations,
        }
        last_identity, last_frame = identity, int(row['frame'])
    return outputs


def _risk_value(row, target, method, v5_risks, candidate_risks):
    if method == 'anchor_score':
        return -float(row['anchor_score'])
    if method == 'v51_single_scale_risk':
        return float(v5_risks[row['frame_key']]['scale'])
    item = candidate_risks[method][row['frame_key']]
    if target in SIDE_COMPONENTS:
        return float(item[target]['calibration_percentile_risk'])
    return float(item['combined_scale_risk'])


def matched_coverage(rows, methods, v5_risks, candidate_risks, contract):
    threshold = float(contract['evaluation']['side_relative_error_threshold'])
    result = {}
    for target in ('long_side', 'short_side', 'scale'):
        result[target] = {}
        for method in methods:
            ordered = sorted(rows, key=lambda row: (
                _risk_value(row, target, method, v5_risks,
                            candidate_risks), row['frame_key']))
            curve = []
            for coverage in contract['evaluation']['matched_coverages']:
                retained = ordered[:max(1, round(len(ordered)*coverage))]
                errors = [(_scale_error(row) if target == 'scale' else
                           _side_error(row, target)) for row in retained]
                curve.append({
                    'target_coverage': float(coverage),
                    'retained_count': len(retained),
                    'mean_error': float(np.mean(errors)),
                    'failure_rate': sum(error > threshold for error in errors)/
                    len(errors),
                    'correct_retained_count': sum(
                        error <= threshold for error in errors),
                    'bad_retained_count': sum(error > threshold
                                              for error in errors),
                })
            result[target][method] = curve
    return result


def _curve_wins(curves, challenger, reference):
    result = {}
    for target in ('long_side', 'short_side', 'scale'):
        baseline = {item['target_coverage']: item
                    for item in curves[target][reference]}
        wins, losses, ties = [], [], []
        for item in curves[target][challenger]:
            other = baseline[item['target_coverage']]
            no_worse = (item['mean_error'] <= other['mean_error'] and
                        item['failure_rate'] <= other['failure_rate'])
            strictly = (item['mean_error'] < other['mean_error'] or
                        item['failure_rate'] < other['failure_rate'])
            no_better = (item['mean_error'] >= other['mean_error'] and
                         item['failure_rate'] >= other['failure_rate'])
            if no_worse and strictly:
                wins.append(item['target_coverage'])
            elif no_better and (item['mean_error'] > other['mean_error'] or
                                item['failure_rate'] > other['failure_rate']):
                losses.append(item['target_coverage'])
            else:
                ties.append(item['target_coverage'])
        result[target] = {'wins': wins, 'losses': losses, 'ties_or_tradeoffs': ties}
    return result


def _hold_verdict(rejection, hold):
    rejection_scale = rejection['all']['scale']
    hold_scale = hold['all']['scale']
    checks = {
        'prediction_improved_not_fewer_than_degraded': (
            hold_scale['prediction_improved_count'] >=
            hold_scale['prediction_degraded_count']),
        'correct_output_coverage_not_worse_than_rejection': (
            hold_scale['correct_output_coverage'] >=
            rejection_scale['correct_output_coverage']),
        'longest_prediction_is_one_frame_by_construction': True,
    }
    return {'checks': checks, 'passes_all': all(checks.values())}


def _source_slices(rows, outputs, contract):
    result = {}
    minimum = int(contract['evaluation']['minimum_source_slice_count'])
    for source in ('k1', 'dino_fallback'):
        selected = [row for row in rows if row['anchor_source'] == source]
        item = {'frame_count': len(selected),
                'sufficient_for_comparative_claim': len(selected) >= minimum}
        if selected:
            item['components'] = state_report(
                selected, outputs, contract, annotated=True)
            for component in item['components'].values():
                component.pop('longest_unavailable_run', None)
        else:
            item['components'] = {}
        result[source] = item
    return result


def run(baseline, baseline_contract, runtime_v3, v5_report, v51_report,
        contract, image_dir):
    validate_contract(contract)
    validate(baseline, baseline_contract)
    validate_source_inputs(runtime_v3, v5_report, v51_report, contract)
    rows, image_digest = enrich_online_features(baseline['records'], image_dir)
    split_counts = assign_split(
        rows, float(contract['split']['calibration_prefix_fraction']),
        int(contract['split']['minimum_calibration_frames_per_segment']))
    expected = {
        'calibration': int(contract['split']['expected_calibration_frames']),
        'evaluation': int(contract['split']['expected_evaluation_frames'])}
    if split_counts != expected:
        raise RuntimeError('V5.2 source split counts changed')
    fitted = fit_candidates(rows, contract)
    evaluation = [row for row in rows if row['split'] == 'evaluation']
    v5_fitted = v5_report['fitted_component_risk_challenger']
    v5_risks = v5.risk_scores(evaluation, v5_fitted)
    score, learned = v5.gate_decisions(
        evaluation, runtime_v3, v5_fitted, v5_risks)
    v51_decisions = v51.hybrid_decisions(evaluation, score, learned)
    candidate_risks = {name: split_risk_scores(evaluation, model)
                       for name, model in fitted.items()}
    raw_decisions = {row['frame_key']: {
        component: row.get('base_v3_box') is not None
        for component in COMPONENTS} for row in evaluation}
    outputs = {
        'raw_direct': v5.replay_policy(
            evaluation, raw_decisions, None, 'rejection_only'),
        'confidence_only_rejection': v5.replay_policy(
            evaluation, score, None, 'rejection_only'),
        'v51_hybrid': v51.replay_hybrid(
            evaluation, v51_decisions, v5_risks),
    }
    for name, risks in candidate_risks.items():
        decisions = _candidate_decisions(evaluation, score, risks, fitted[name])
        outputs[name+'_rejection_only'] = replay_candidate(
            evaluation, decisions, risks, scale_hold=False)
        outputs[name+'_one_frame_scale_hold'] = replay_candidate(
            evaluation, decisions, risks, scale_hold=True)
    reports = {name: {
        group: state_report(selected, output, contract, annotated=True)
        for group, selected in _groups(evaluation)}
        for name, output in outputs.items()}
    curve_methods = ['anchor_score', 'v51_single_scale_risk']+list(fitted)
    curves = {}
    curve_verdicts = {}
    for group, selected in _groups(evaluation):
        if group not in contract['evaluation']['matched_coverage_groups']:
            continue
        curves[group] = matched_coverage(
            selected, curve_methods, v5_risks, candidate_risks, contract)
        curve_verdicts[group] = {
            name: _curve_wins(curves[group], name,
                              'v51_single_scale_risk')
            for name in fitted}
    primary = contract['primary_candidate']
    minimum_wins = int(contract['promotion_gates'][
        'minimum_matched_coverage_wins_per_target'])
    primary_comparison = curve_verdicts['all'][primary]
    primary_risk_passed = all(
        len(primary_comparison[target]['wins']) >= minimum_wins
        for target in ('long_side', 'short_side', 'scale'))
    hold = _hold_verdict(
        reports[primary+'_rejection_only'],
        reports[primary+'_one_frame_scale_hold'])
    v51_reproduced = all(
        abs(reports['v51_hybrid']['all'][component][key]-
            v51_report['method_reports']['v51_hybrid']['all'][component][key])
        <= 1e-12
        for component in COMPONENTS
        for key in ('measurement_count', 'prediction_count',
                    'unavailable_count', 'mean_available_output_error',
                    'correct_output_coverage'))
    if not v51_reproduced:
        raise RuntimeError('V5.1 source metrics were not reproduced')
    output_rows = []
    for row in rows:
        item = {key: copy.deepcopy(row.get(key)) for key in (
            'frame_key', 'domain', 'sequence', 'frame', 'split',
            'anchor_source', 'anchor_score')}
        if row['split'] == 'evaluation':
            item['online_features'] = row['online_features']
            item['v51_scale_risk'] = v5_risks[row['frame_key']]['scale']
            item['v52_split_scale_risks'] = {
                name: risks[row['frame_key']]
                for name, risks in candidate_risks.items()}
            item['method_states'] = {
                name: output[row['frame_key']]['components']
                for name, output in outputs.items()}
            item['offline_errors'] = row['errors']
        output_rows.append(item)
    source_slices = {
        name: _source_slices(evaluation, output, contract)
        for name, output in outputs.items()}
    return {
        'protocol': PROTOCOL,
        'evidence_boundary': contract['evidence_boundary'],
        'claim_status': 'SOURCE_VAL_POST_V51_DEVELOPMENT_V52_ONLY',
        'fixed_test_read': False, 'target_data_read': False,
        'post_fixed_test_hypothesis_generation': True,
        'threshold_fitting_performed': True,
        'threshold_fitting_scope': 'source_calibration_prefix_only',
        'feature_selection_performed': False,
        'parameter_selection_authorized_from_fixed_test': False,
        'automatic_ablation_promotion': False,
        'operation_or_phase_labels_used': False,
        'image_quality_ablation_enabled': image_dir is not None,
        'image_quality_feature_digest': image_digest,
        'leakage_controls': {
            'gt_used_in_online_features': False,
            'gt_used_in_online_decisions': False,
            'calibration_gt_used_for_side_error_labels': True,
            'evaluation_gt_used_for_fitting': False,
            'fixed_test_used_for_fitting_or_selection': False,
            'domain_identity_used_in_decisions': False,
            'operation_phase_used_in_decisions': False,
            'future_frames_used': False,
            'state_reset_on_sequence_or_frame_gap': True,
        },
        'split_counts': split_counts,
        'anchor_source_counts_evaluation': dict(Counter(
            row['anchor_source'] for row in evaluation)),
        'fitted_split_scale_candidates': fitted,
        'method_inventory': list(outputs),
        'method_reports': reports,
        'source_slice_reports': source_slices,
        'matched_coverage_by_group': curves,
        'matched_coverage_comparison_vs_v51': curve_verdicts,
        'primary_candidate': primary,
        'primary_split_risk_passed': primary_risk_passed,
        'primary_scale_hold_verdict': hold,
        'v51_reproduction_check': {'passed': True},
        'decision': ('KEEP_V52_PRIMARY_SPLIT_RISK_FOR_SOURCE_FREEZE'
                     if primary_risk_passed else
                     'STOP_V52_PRIMARY_AND_REPORT_ABLATIONS_ONLY'),
        'scale_hold_decision': ('KEEP_ONE_FRAME_SCALE_HOLD'
                                if hold['passes_all'] else
                                'DISABLE_SCALE_HOLD'),
        'eligible_for_fixed_test_reuse_as_final_validation': False,
        'eligible_for_unknown_sequence_claim': False,
        'records': output_rows,
        'claim_limit': contract['claim_limit'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--baseline-contract', required=True)
    parser.add_argument('--runtime-calibration-v3', required=True)
    parser.add_argument('--v5-report', required=True)
    parser.add_argument('--v5-contract', required=True)
    parser.add_argument('--v51-report', required=True)
    parser.add_argument('--v51-contract', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--image-dir', default=None)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    paths = {
        'baseline': args.baseline,
        'baseline_contract': args.baseline_contract,
        'runtime_calibration_v3': args.runtime_calibration_v3,
        'v5_report': args.v5_report, 'v5_contract': args.v5_contract,
        'v51_report': args.v51_report, 'v51_contract': args.v51_contract,
        'contract': args.contract,
    }
    identities = {name: _identity(path) for name, path in paths.items()}
    payloads = {name: json.loads(Path(path).read_text()) for name, path in
                paths.items() if name not in ('v5_contract', 'v51_contract')}
    contract = payloads['contract']
    validate_contract(contract)
    expected = contract['expected_inputs']
    for role, field in (
            ('baseline_contract', 'baseline_contract_sha256'),
            ('runtime_calibration_v3', 'runtime_calibration_v3_sha256'),
            ('v5_report', 'v5_report_sha256'),
            ('v5_contract', 'v5_contract_sha256'),
            ('v51_report', 'v51_report_sha256'),
            ('v51_contract', 'v51_contract_sha256')):
        _require_identity(identities[role], expected[field], role)
    portable = portable_digest(payloads['baseline'])
    if portable != expected['baseline_portable_sha256']:
        raise RuntimeError('Portable source baseline identity mismatch')
    image_required = bool(contract['image_quality_ablation']['enabled'])
    if image_required and args.image_dir is None:
        raise RuntimeError('Contract requires --image-dir for image ablation')
    result = run(
        payloads['baseline'], payloads['baseline_contract'],
        payloads['runtime_calibration_v3'], payloads['v5_report'],
        payloads['v51_report'], contract, args.image_dir)
    result['inputs'] = identities
    result['inputs']['baseline_portable_sha256'] = portable
    if args.image_dir is not None:
        result['inputs']['image_dir'] = str(Path(args.image_dir).resolve())
    output = _write_exact(args.out_json, result)
    print(json.dumps({
        'output': output,
        'split_counts': result['split_counts'],
        'method_count': len(result['method_inventory']),
        'primary_candidate': result['primary_candidate'],
        'primary_split_risk_passed': result['primary_split_risk_passed'],
        'scale_hold_decision': result['scale_hold_decision'],
        'decision': result['decision'],
        'claim_status': result['claim_status'],
        'fixed_test_read': False,
    }, indent=2))


if __name__ == '__main__':
    main()
