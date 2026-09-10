import copy

import pytest

from crane_project.tools import (
    base_v3_obb_image_quality_block_oof_development_v53 as v53)


FEATURES = ['anchor_uncertainty', 'roi_gray_mean_norm']


def _row(domain, sequence, frame, error=0.0):
    return {
        'frame_key': '{}_{}_{:05d}'.format(domain, sequence, frame),
        'domain': domain, 'sequence': sequence, 'frame': frame,
        'anchor_score': 0.9-frame/1000.0,
        'errors': {'long_relative_error': error,
                   'short_relative_error': error/2.0},
        'online_features': {
            'anchor_uncertainty': frame/100.0,
            'roi_gray_mean_norm': frame/200.0}}


def test_three_block_folds_cover_each_frame_once_with_embargo():
    rows = ([_row('real', 'a', frame) for frame in range(1, 10)]+
            [_row('sim', 'b', frame) for frame in range(1, 13)])
    folds = v53.build_block_folds(rows, fold_count=3, embargo=2)
    validation = [key for fold in folds for key in fold['validation_keys']]
    assert len(validation) == len(rows)
    assert len(set(validation)) == len(rows)
    for fold in folds:
        assert not fold['training_keys'] & fold['validation_keys']
        assert not fold['training_keys'] & fold['embargoed_keys']


def test_ranker_fit_ignores_validation_labels():
    contract = {
        'evaluation': {'scale_relative_error_threshold': 0.1},
        'ranker': {'l2_regularization': 1.0,
                   'maximum_newton_iterations': 50,
                   'convergence_tolerance': 1e-8,
                   'target_measurement_coverage': 0.9}}
    training = [_row('real', 'a', frame, 0.2 if frame in (2, 5) else 0.0)
                for frame in range(1, 9)]
    validation = [_row('real', 'a', 9), _row('real', 'a', 10)]
    first = v53.fit_and_predict_fold(training, validation, {
        **contract, 'existing_features': ['anchor_uncertainty'],
        'image_quality_features': ['roi_gray_mean_norm']})
    changed = copy.deepcopy(validation)
    for row in changed:
        row['errors']['long_relative_error'] = 999.0
    second = v53.fit_and_predict_fold(training, changed, {
        **contract, 'existing_features': ['anchor_uncertainty'],
        'image_quality_features': ['roi_gray_mean_norm']})
    assert first['models'] == second['models']
    assert first['predictions'] == second['predictions']


def test_matched_coverage_keeps_equal_counts_and_no_prediction_state():
    rows = [_row('real', 'a', frame, frame/100.0)
            for frame in range(1, 11)]
    predictions = {row['frame_key']: {
        method: {'ranking_risk': index/10.0,
                 'accepted': index < 8}
        for method in v53.METHODS}
        for index, row in enumerate(rows)}
    contract = {'evaluation': {
        'scale_relative_error_threshold': 0.1,
        'matched_coverages': [0.5, 0.8]}}
    curves = v53.matched_coverage_curves(rows, predictions, contract)
    for index in range(2):
        assert len({curve[index]['retained_count']
                    for curve in curves.values()}) == 1
    report = v53.rejection_report(rows, predictions, 'anchor_score', contract)
    assert report['prediction_count'] == 0
    assert report['scale_states'] == ['measurement', 'unavailable']


def test_contract_rejects_scale_hold():
    with pytest.raises(ValueError, match='rejection-only'):
        v53.validate_contract({
            'protocol': v53.CONTRACT_PROTOCOL,
            'fixed_test_read': False, 'target_data_read': False,
            'post_fixed_test_hypothesis_generation': True,
            'methods': list(v53.METHODS),
            'expected_inputs': {
                'baseline_protocol': v53.BASELINE_PROTOCOL,
                'v521_report_protocol': v53.v521.PROTOCOL},
            'block_oof': {'fold_count': 3, 'history_embargo_frames': 2},
            'output_policy': {'scale_hold_enabled': True},
            'development_gate': {'automatic_runtime_freeze': False}})
