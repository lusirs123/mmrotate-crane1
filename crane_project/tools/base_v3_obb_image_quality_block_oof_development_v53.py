#!/usr/bin/env python3
"""Evaluate one image-quality addition with source-val block OOF.

The experiment compares anchor confidence, an existing-feature scale ranker,
and the same ranker plus current-frame OBB ROI image statistics.  Every source
frame is evaluated once by a model that excludes its contiguous validation
block and a two-frame label embargo.  Scale output is rejection-only.
"""

import argparse
import copy
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from crane_project.tools import (
    base_v3_obb_heterogeneous_policy_development_v5 as v5)
from crane_project.tools import (
    base_v3_obb_split_scale_policy_development_v52 as v52)
from crane_project.tools import (
    base_v3_obb_split_scale_policy_diagnostic_v521 as v521)
from crane_project.tools.base_v3_obb_reliability_baseline import (
    PROTOCOL as BASELINE_PROTOCOL, _identity, _write_exact)
from crane_project.tools.base_v3_reliability_failure_audit import (
    portable_digest, validate)


PROTOCOL = 'base_v3_obb_image_quality_block_oof_development_v53'
CONTRACT_PROTOCOL = (
    'base_v3_obb_image_quality_block_oof_development_contract_v53')
METHODS = ('anchor_score', 'existing_scale_risk',
           'existing_plus_image_quality')


def _require_identity(identity, expected, role):
    if identity['sha256'] != expected:
        raise RuntimeError('{} identity mismatch: expected {}, got {}'.format(
            role, expected, identity['sha256']))


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected V5.3 contract')
    if contract.get('fixed_test_read') is not False:
        raise ValueError('V5.3 must not authorize fixed TEST')
    if contract.get('target_data_read') is not False:
        raise ValueError('V5.3 must remain source-only')
    if contract.get('post_fixed_test_hypothesis_generation') is not True:
        raise ValueError('V5.3 must retain post-exposure status')
    if contract.get('methods') != list(METHODS):
        raise ValueError('V5.3 method inventory changed')
    if contract['expected_inputs'].get(
            'baseline_protocol') != BASELINE_PROTOCOL:
        raise ValueError('V5.3 must bind the Base V3 baseline')
    if contract['expected_inputs'].get(
            'v521_report_protocol') != v521.PROTOCOL:
        raise ValueError('V5.3 must bind the V5.2.1 diagnosis')
    if int(contract['block_oof']['fold_count']) != 3:
        raise ValueError('V5.3 requires three block folds')
    if int(contract['block_oof']['history_embargo_frames']) != 2:
        raise ValueError('V5.3 history embargo changed')
    if contract['output_policy'].get('scale_hold_enabled') is not False:
        raise ValueError('V5.3 scale output must remain rejection-only')
    if contract['development_gate'].get(
            'automatic_runtime_freeze') is not False:
        raise ValueError('V5.3 must not automatically freeze a runtime')


def validate_v521(report):
    if report.get('protocol') != v521.PROTOCOL:
        raise ValueError('Unexpected V5.2.1 report protocol')
    if report.get('fixed_test_read') is not False:
        raise ValueError('V5.2.1 report violates fixed-TEST boundary')
    if report.get('target_data_read') is not False:
        raise ValueError('V5.2.1 report violates source-only boundary')
    if report.get('scale_hold_decision') != (
            'DISABLE_V52_SCALE_HOLD_AS_VALID_OBSERVATION'):
        raise ValueError('V5.3 requires the disabled V5.2 scale hold')
    if report.get('next_development_candidate') != (
            'IMAGE_QUALITY_ABLATION_HYPOTHESIS_ONLY'):
        raise ValueError('V5.3 requires the image-quality hypothesis')
    if report.get('automatic_ablation_promotion') is not False:
        raise ValueError('V5.2.1 report permits automatic promotion')


def _ordered(rows):
    return sorted(rows, key=lambda row: (
        row['domain'], row['sequence'], int(row['frame']), row['frame_key']))


def build_block_folds(rows, fold_count, embargo):
    """Return non-overlapping cores and label-embargoed training indices."""
    segments = {}
    for row in _ordered(rows):
        segments.setdefault((row['domain'], row['sequence']), []).append(row)
    fold_cores = [[] for _ in range(fold_count)]
    excluded = [set() for _ in range(fold_count)]
    for identity, segment in segments.items():
        count = len(segment)
        boundaries = [round(count*index/fold_count)
                      for index in range(fold_count+1)]
        for fold in range(fold_count):
            start, end = boundaries[fold], boundaries[fold+1]
            core = segment[start:end]
            fold_cores[fold].extend(row['frame_key'] for row in core)
            left, right = max(0, start-embargo), min(count, end+embargo)
            excluded[fold].update(
                row['frame_key'] for row in segment[left:right])
    all_keys = {row['frame_key'] for row in rows}
    folds = []
    seen = []
    for fold in range(fold_count):
        validation = set(fold_cores[fold])
        training = all_keys-excluded[fold]
        if validation & training:
            raise RuntimeError('V5.3 training/validation overlap')
        folds.append({'fold': fold, 'training_keys': training,
                      'validation_keys': validation,
                      'embargoed_keys': excluded[fold]-validation})
        seen.extend(validation)
    if len(seen) != len(rows) or len(set(seen)) != len(rows):
        raise RuntimeError('Each V5.3 frame must be evaluated exactly once')
    return folds


def _scale_error(row):
    return max(float(row['errors']['long_relative_error']),
               float(row['errors']['short_relative_error']))


def _fit_ranker(training, feature_names, contract):
    matrix, preprocessing = v5._matrix(training, feature_names)
    threshold = float(contract['evaluation'][
        'scale_relative_error_threshold'])
    labels = np.asarray([_scale_error(row) > threshold for row in training],
                        dtype=np.float64)
    spec = contract['ranker']
    weights, iterations, converged = v5._fit_logistic(
        matrix, labels, float(spec['l2_regularization']),
        int(spec['maximum_newton_iterations']),
        float(spec['convergence_tolerance']))
    training_risks = 1.0/(1.0+np.exp(-np.clip(
        matrix.dot(weights), -30.0, 30.0)))
    risk_threshold = float(np.quantile(
        training_risks, float(spec['target_measurement_coverage'])))
    return {
        'feature_names': list(feature_names),
        'expanded_feature_names': list(feature_names)+[
            name+'__missing' for name in feature_names],
        'preprocessing': preprocessing,
        'weights': weights.tolist(),
        'iterations': iterations,
        'converged': converged,
        'positive_count': int(labels.sum()),
        'negative_count': int(len(labels)-labels.sum()),
        'risk_threshold': risk_threshold,
        'online_gt_fields_consumed': [],
    }


def _ranker_risks(rows, model):
    matrix, _ = v5._matrix(
        rows, model['feature_names'], model['preprocessing'])
    weights = np.asarray(model['weights'], dtype=np.float64)
    return 1.0/(1.0+np.exp(-np.clip(matrix.dot(weights), -30.0, 30.0)))


def fit_and_predict_fold(training, validation, contract):
    started = time.perf_counter()
    existing = list(contract['existing_features'])
    image = existing+list(contract['image_quality_features'])
    models = {
        'existing_scale_risk': _fit_ranker(training, existing, contract),
        'existing_plus_image_quality': _fit_ranker(
            training, image, contract)}
    target = float(contract['ranker']['target_measurement_coverage'])
    anchor_scores = np.asarray(
        [float(row['anchor_score']) for row in training], dtype=np.float64)
    score_threshold = float(np.quantile(anchor_scores, 1.0-target))
    predictions = {}
    learned = {name: _ranker_risks(validation, model)
               for name, model in models.items()}
    for index, row in enumerate(validation):
        key = row['frame_key']
        predictions[key] = {
            'anchor_score': {
                'ranking_risk': -float(row['anchor_score']),
                'accepted': float(row['anchor_score']) >= score_threshold},
        }
        for name, model in models.items():
            risk = float(learned[name][index])
            predictions[key][name] = {
                'ranking_risk': risk,
                'accepted': risk <= float(model['risk_threshold'])}
    return {
        'models': models,
        'anchor_score_threshold': score_threshold,
        'predictions': predictions,
        'elapsed_seconds': time.perf_counter()-started,
    }


def _longest_unavailable_run(rows, predictions, method):
    longest = current = 0
    previous = None
    for row in _ordered(rows):
        identity = (row['domain'], row['sequence'])
        contiguous = (previous is not None and previous[0] == identity and
                      int(row['frame']) == previous[1]+1)
        if not contiguous:
            current = 0
        if predictions[row['frame_key']][method]['accepted']:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
        previous = (identity, int(row['frame']))
    return longest


def rejection_report(rows, predictions, method, contract):
    threshold = float(contract['evaluation'][
        'scale_relative_error_threshold'])
    errors = [_scale_error(row) for row in rows]
    accepted = [bool(predictions[row['frame_key']][method]['accepted'])
                for row in rows]
    measurement_errors = [error for error, keep in zip(errors, accepted)
                          if keep]
    good_count = sum(error <= threshold for error in errors)
    bad_count = len(errors)-good_count
    accepted_good = sum(keep and error <= threshold
                        for error, keep in zip(errors, accepted))
    accepted_bad = sum(keep and error > threshold
                       for error, keep in zip(errors, accepted))
    return {
        'frame_count': len(rows),
        'measurement_count': sum(accepted),
        'unavailable_count': len(rows)-sum(accepted),
        'measurement_coverage': sum(accepted)/len(rows),
        'mean_measurement_error': (
            None if not measurement_errors else
            float(np.mean(measurement_errors))),
        'false_acceptance_rate': (
            None if not measurement_errors else
            accepted_bad/len(measurement_errors)),
        'correct_measurement_retention': (
            None if not good_count else accepted_good/good_count),
        'bad_observation_rejection_recall': (
            None if not bad_count else (bad_count-accepted_bad)/bad_count),
        'longest_unavailable_run': _longest_unavailable_run(
            rows, predictions, method),
        'prediction_count': 0,
        'scale_states': ['measurement', 'unavailable'],
    }


def matched_coverage_curves(rows, predictions, contract):
    threshold = float(contract['evaluation'][
        'scale_relative_error_threshold'])
    output = {}
    for method in METHODS:
        ordered = sorted(rows, key=lambda row: (
            float(predictions[row['frame_key']][method]['ranking_risk']),
            row['frame_key']))
        curve = []
        for coverage in contract['evaluation']['matched_coverages']:
            kept = ordered[:max(1, round(len(ordered)*coverage))]
            errors = [_scale_error(row) for row in kept]
            curve.append({
                'target_coverage': float(coverage),
                'retained_count': len(kept),
                'mean_error': float(np.mean(errors)),
                'failure_rate': sum(error > threshold for error in errors)/
                len(errors),
                'correct_retained_count': sum(
                    error <= threshold for error in errors),
                'bad_retained_count': sum(error > threshold
                                          for error in errors),
            })
        output[method] = curve
    return output


def _groups(rows, contract):
    output = [('all', rows)]
    output.extend((domain, [row for row in rows if row['domain'] == domain])
                  for domain in contract['evaluation']['groups']
                  if domain != 'all')
    return output


def _development_verdict(curves, fold_curves, contract):
    gate = contract['development_gate']
    comparisons = {}
    for group in contract['evaluation']['groups']:
        comparisons[group] = {
            'image_vs_existing': v521._curve_outcome(
                curves[group]['existing_plus_image_quality'],
                curves[group]['existing_scale_risk']),
            'image_vs_anchor_score': v521._curve_outcome(
                curves[group]['existing_plus_image_quality'],
                curves[group]['anchor_score']),
        }
    fold_comparisons = []
    for item in fold_curves:
        outcome = v521._curve_outcome(
            item['curves']['existing_plus_image_quality'],
            item['curves']['existing_scale_risk'])
        fold_comparisons.append({'fold': item['fold'], 'outcome': outcome})
    folds_better = sum(
        item['outcome']['win_count'] > item['outcome']['loss_count']
        for item in fold_comparisons)
    checks = {
        'all_win_count_vs_existing': comparisons['all'][
            'image_vs_existing']['win_count'] >= int(
                gate['minimum_all_win_count_vs_existing']),
        'real_win_count_vs_existing': comparisons['real'][
            'image_vs_existing']['win_count'] >= int(
                gate['minimum_real_win_count_vs_existing']),
        'real_win_count_vs_anchor_score': comparisons['real'][
            'image_vs_anchor_score']['win_count'] >= int(
                gate['minimum_real_win_count_vs_anchor_score']),
        'sim_loss_count_vs_existing': comparisons['sim'][
            'image_vs_existing']['loss_count'] <= int(
                gate['maximum_sim_loss_count_vs_existing']),
        'fold_consistency': folds_better >= int(
            gate['minimum_folds_with_more_wins_than_losses_vs_existing']),
    }
    return {'checks': checks, 'passes_all': all(checks.values()),
            'matched_coverage_comparisons': comparisons,
            'fold_comparisons': fold_comparisons,
            'folds_with_more_wins_than_losses': folds_better}


def run(baseline, baseline_contract, v521_report, contract, image_dir):
    validate_contract(contract)
    validate(baseline, baseline_contract)
    validate_v521(v521_report)
    started = time.perf_counter()
    rows, image_digest = v52.enrich_online_features(
        baseline['records'], image_dir)
    feature_seconds = time.perf_counter()-started
    expected_segments = contract['source_scope']['segments']
    actual_segments = Counter(
        '{}:{}'.format(row['domain'], row['sequence']) for row in rows)
    if dict(actual_segments) != expected_segments:
        raise RuntimeError('V5.3 source segments changed')
    if len(rows) != int(contract['source_scope']['frame_count']):
        raise RuntimeError('V5.3 source frame count changed')
    folds = build_block_folds(
        rows, int(contract['block_oof']['fold_count']),
        int(contract['block_oof']['history_embargo_frames']))
    by_key = {row['frame_key']: row for row in rows}
    all_predictions = {}
    fold_reports = []
    fold_curves = []
    for fold in folds:
        training = _ordered([by_key[key] for key in fold['training_keys']])
        validation = _ordered([by_key[key] for key in fold['validation_keys']])
        fitted = fit_and_predict_fold(training, validation, contract)
        overlap = set(all_predictions) & set(fitted['predictions'])
        if overlap:
            raise RuntimeError('Duplicate V5.3 OOF predictions')
        all_predictions.update(fitted['predictions'])
        curves = matched_coverage_curves(validation, fitted['predictions'],
                                         contract)
        fold_curves.append({'fold': fold['fold'], 'curves': curves})
        fold_reports.append({
            'fold': fold['fold'],
            'training_frame_count': len(training),
            'validation_frame_count': len(validation),
            'embargoed_frame_count': len(fold['embargoed_keys']),
            'training_error_counts': {
                'bad': sum(_scale_error(row) > float(contract['evaluation'][
                    'scale_relative_error_threshold']) for row in training),
                'good': sum(_scale_error(row) <= float(contract['evaluation'][
                    'scale_relative_error_threshold']) for row in training)},
            'validation_error_counts': {
                'bad': sum(_scale_error(row) > float(contract['evaluation'][
                    'scale_relative_error_threshold']) for row in validation),
                'good': sum(_scale_error(row) <= float(contract['evaluation'][
                    'scale_relative_error_threshold']) for row in validation)},
            'anchor_score_threshold': fitted['anchor_score_threshold'],
            'models': fitted['models'],
            'fit_and_predict_seconds': fitted['elapsed_seconds'],
            'matched_coverage_curves': curves,
        })
    if set(all_predictions) != set(by_key):
        raise RuntimeError('Incomplete V5.3 OOF predictions')
    expected_train = contract['block_oof'][
        'expected_training_frames_per_fold']
    expected_val = contract['block_oof'][
        'expected_validation_frames_per_fold']
    if [item['training_frame_count'] for item in fold_reports] != expected_train:
        raise RuntimeError('V5.3 fold training counts changed')
    if [item['validation_frame_count'] for item in fold_reports] != expected_val:
        raise RuntimeError('V5.3 fold validation counts changed')
    curves = {}
    reports = {}
    for group, selected in _groups(rows, contract):
        curves[group] = matched_coverage_curves(
            selected, all_predictions, contract)
        reports[group] = {method: rejection_report(
            selected, all_predictions, method, contract)
            for method in METHODS}
    verdict = _development_verdict(curves, fold_curves, contract)
    records = []
    for row in _ordered(rows):
        item = {key: copy.deepcopy(row.get(key)) for key in (
            'frame_key', 'domain', 'sequence', 'frame', 'anchor_source',
            'anchor_score')}
        item['scale_error'] = _scale_error(row)
        item['image_quality_features'] = {
            name: row['online_features'][name]
            for name in contract['image_quality_features']}
        item['oof_methods'] = copy.deepcopy(all_predictions[row['frame_key']])
        item['scale_outputs'] = {method: {
            'state': ('measurement' if result['accepted'] else 'unavailable'),
            'valid': bool(result['accepted']),
            'prediction_used': False}
            for method, result in all_predictions[row['frame_key']].items()}
        records.append(item)
    return {
        'protocol': PROTOCOL,
        'evidence_boundary': contract['evidence_boundary'],
        'claim_status': 'SOURCE_VAL_BLOCK_OOF_IMAGE_QUALITY_V53_ONLY',
        'fixed_test_read': False, 'target_data_read': False,
        'post_fixed_test_hypothesis_generation': True,
        'threshold_fitting_scope': 'fold_training_only',
        'evaluation_gt_used_for_fitting': False,
        'domain_or_sequence_identity_used_in_decisions': False,
        'operation_phase_used_in_decisions': False,
        'future_frames_used_in_online_features': False,
        'scale_hold_enabled': False,
        'each_frame_evaluated_once': True,
        'image_quality_feature_digest': image_digest,
        'source_counts': dict(actual_segments),
        'error_counts': {
            'bad': sum(_scale_error(row) > float(contract['evaluation'][
                'scale_relative_error_threshold']) for row in rows),
            'good': sum(_scale_error(row) <= float(contract['evaluation'][
                'scale_relative_error_threshold']) for row in rows)},
        'method_inventory': list(METHODS),
        'fold_reports': fold_reports,
        'oof_rejection_reports': reports,
        'oof_matched_coverage_curves': curves,
        'image_quality_development_verdict': verdict,
        'decision': ('FREEZE_IMAGE_QUALITY_CANDIDATE_FOR_FRESH_HOLDOUT'
                     if verdict['passes_all'] else
                     'STOP_IMAGE_QUALITY_CANDIDATE'),
        'automatic_runtime_freeze_performed': False,
        'eligible_for_fixed_test_reuse_as_final_validation': False,
        'eligible_for_unknown_sequence_claim': False,
        'timing': {
            'image_feature_extraction_seconds': feature_seconds,
            'fold_fit_and_predict_seconds': sum(
                item['fit_and_predict_seconds'] for item in fold_reports)},
        'records': records,
        'claim_limit': contract['claim_limit'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--baseline-contract', required=True)
    parser.add_argument('--v521-report', required=True)
    parser.add_argument('--v521-contract', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    paths = {
        'baseline': args.baseline,
        'baseline_contract': args.baseline_contract,
        'v521_report': args.v521_report,
        'v521_contract': args.v521_contract,
        'contract': args.contract}
    identities = {name: _identity(path) for name, path in paths.items()}
    baseline = json.loads(Path(args.baseline).read_text())
    baseline_contract = json.loads(Path(args.baseline_contract).read_text())
    v521_report = json.loads(Path(args.v521_report).read_text())
    contract = json.loads(Path(args.contract).read_text())
    validate_contract(contract)
    expected = contract['expected_inputs']
    for role, field in (
            ('baseline_contract', 'baseline_contract_sha256'),
            ('v521_report', 'v521_report_sha256'),
            ('v521_contract', 'v521_contract_sha256')):
        _require_identity(identities[role], expected[field], role)
    portable = portable_digest(baseline)
    if portable != expected['baseline_portable_sha256']:
        raise RuntimeError('Portable source baseline identity mismatch')
    payload = run(
        baseline, baseline_contract, v521_report, contract, args.image_dir)
    payload['inputs'] = identities
    payload['inputs']['baseline_portable_sha256'] = portable
    payload['inputs']['image_dir'] = str(Path(args.image_dir).resolve())
    output = _write_exact(args.out_json, payload)
    verdict = payload['image_quality_development_verdict']
    print(json.dumps({
        'output': output,
        'source_counts': payload['source_counts'],
        'fold_counts': [{
            'fold': item['fold'],
            'training': item['training_frame_count'],
            'validation': item['validation_frame_count'],
            'embargoed': item['embargoed_frame_count']}
            for item in payload['fold_reports']],
        'method_count': len(payload['method_inventory']),
        'image_quality_gate_passed': verdict['passes_all'],
        'gate_checks': verdict['checks'],
        'decision': payload['decision'],
        'claim_status': payload['claim_status'],
        'fixed_test_read': False,
    }, indent=2))


if __name__ == '__main__':
    main()
