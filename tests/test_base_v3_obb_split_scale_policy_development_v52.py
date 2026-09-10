import copy

import numpy as np
import pytest

from crane_project.tools import (
    base_v3_obb_split_scale_policy_development_v52 as v52)


EXISTING = [
    'anchor_uncertainty', 'aspect_ambiguity',
    'causal_angle_residual_norm', 'causal_center_residual_norm',
    'causal_scale_residual_log', 'k1_dino_angle_disagreement_norm',
    'k1_dino_center_disagreement_norm',
    'k1_dino_scale_disagreement_log', 'refiner_angle_correction_norm',
    'refiner_center_correction_norm', 'refiner_scale_correction_log']


def _row(frame, long_error=0.0, short_error=0.0):
    box = [float(frame), 2.0, 10.0, 5.0, 0.1]
    return {
        'frame_key': 'real_seq_{:05d}'.format(frame), 'domain': 'real',
        'sequence': 'seq', 'frame': frame, 'anchor_source': 'k1',
        'anchor_score': 0.9, 'base_v3_box': copy.deepcopy(box),
        'k1_box': copy.deepcopy(box), 'dino_box': copy.deepcopy(box),
        'gt_box': copy.deepcopy(box),
        'errors': {'center_error_px': 0.0,
                   'long_relative_error': long_error,
                   'short_relative_error': short_error,
                   'angle_error_deg': 0.0},
        'online_features': {name: float(frame)/10.0 for name in EXISTING},
    }


def _fit_contract():
    return {
        'split_scale_ranker': {
            'model': 'separate_standardized_l2_logistic_side_error_rankers',
            'l2_regularization': 1.0, 'maximum_newton_iterations': 50,
            'convergence_tolerance': 1e-8,
            'target_measurement_coverage': 0.9},
        'evaluation': {'side_relative_error_threshold': 0.1}}


def test_geometry_features_do_not_depend_on_gt_or_errors():
    row = _row(1)
    first = v52._geometry_features(row)
    row['gt_box'] = [999.0]*5
    row['errors'] = {'long_relative_error': 999.0,
                     'short_relative_error': 999.0}
    assert v52._geometry_features(row) == first


def test_long_and_short_side_models_are_fitted_separately():
    rows = []
    for frame in range(1, 13):
        row = _row(frame, long_error=0.2 if frame in (2, 4, 6) else 0.0,
                   short_error=0.2 if frame in (7, 9, 11) else 0.0)
        row['split'] = 'calibration' if frame <= 10 else 'evaluation'
        rows.append(row)
    fitted = v52._fit_side_ranker(rows, EXISTING, _fit_contract())
    assert set(fitted['components']) == {'long_side', 'short_side'}
    assert fitted['components']['long_side']['weights'] != (
        fitted['components']['short_side']['weights'])
    assert fitted['online_gt_fields_consumed'] == []


def test_fitting_ignores_evaluation_errors():
    rows = []
    for frame in range(1, 13):
        row = _row(frame, long_error=0.2 if frame in (2, 5) else 0.0,
                   short_error=0.2 if frame in (3, 6) else 0.0)
        row['split'] = 'calibration' if frame <= 10 else 'evaluation'
        rows.append(row)
    first = v52._fit_side_ranker(rows, EXISTING, _fit_contract())
    changed = copy.deepcopy(rows)
    for row in changed:
        if row['split'] == 'evaluation':
            row['errors']['long_relative_error'] = 999.0
            row['errors']['short_relative_error'] = 999.0
    second = v52._fit_side_ranker(changed, EXISTING, _fit_contract())
    assert first == second


def test_combined_scale_risk_is_maximum_of_side_percentiles():
    rows = []
    for frame in range(1, 13):
        row = _row(frame, long_error=0.2 if frame in (2, 5) else 0.0,
                   short_error=0.2 if frame in (3, 6) else 0.0)
        row['split'] = 'calibration' if frame <= 10 else 'evaluation'
        rows.append(row)
    fitted = v52._fit_side_ranker(rows, EXISTING, _fit_contract())
    scores = v52.split_risk_scores(rows[-2:], fitted)
    for item in scores.values():
        expected = max(item[side]['calibration_percentile_risk']
                       for side in v52.SIDE_COMPONENTS)
        assert item['combined_scale_risk'] == expected


def test_scale_hold_is_one_frame_bounded_and_resets_at_sequence_boundary():
    rows = [_row(1), _row(2), _row(3)]
    decisions = {
        rows[0]['frame_key']: {'center': True, 'scale': True, 'angle': True},
        rows[1]['frame_key']: {'center': True, 'scale': False, 'angle': True},
        rows[2]['frame_key']: {'center': True, 'scale': False, 'angle': True}}
    risks = {row['frame_key']: {'combined_scale_risk': 0.9} for row in rows}
    output = v52.replay_candidate(rows, decisions, risks, scale_hold=True)
    assert output[rows[1]['frame_key']]['components']['scale']['state'] == (
        'prediction')
    assert output[rows[2]['frame_key']]['components']['scale']['state'] == (
        'unavailable')

    new = _row(1)
    new.update(frame_key='real_new_00001', sequence='new')
    boundary_decisions = {rows[0]['frame_key']:
                          {'center': True, 'scale': True, 'angle': True},
                          new['frame_key']:
                          {'center': True, 'scale': False, 'angle': True}}
    boundary_risks = {key: {'combined_scale_risk': 0.9}
                      for key in boundary_decisions}
    reset = v52.replay_candidate(
        [rows[0], new], boundary_decisions, boundary_risks, scale_hold=True)
    assert reset[new['frame_key']]['components']['scale']['state'] == (
        'unavailable')


def test_matched_coverage_retains_equal_counts_for_all_methods():
    rows = [_row(frame, long_error=frame/100.0,
                 short_error=(11-frame)/100.0) for frame in range(1, 11)]
    v5_risks = {row['frame_key']: {'scale': row['frame']/10.0}
                for row in rows}
    candidate = {row['frame_key']: {
        'long_side': {'calibration_percentile_risk': row['frame']/10.0},
        'short_side': {'calibration_percentile_risk': (11-row['frame'])/10.0},
        'combined_scale_risk': max(row['frame'], 11-row['frame'])/10.0}
        for row in rows}
    contract = {'evaluation': {
        'side_relative_error_threshold': 0.1,
        'matched_coverages': [0.5, 0.8]}}
    curves = v52.matched_coverage(
        rows, ['anchor_score', 'v51_single_scale_risk', 'candidate'],
        v5_risks, {'candidate': candidate}, contract)
    for target in curves.values():
        for index in range(2):
            assert len({items[index]['retained_count']
                        for items in target.values()}) == 1


def test_contract_rejects_fixed_test_authorization():
    with pytest.raises(ValueError, match='fixed TEST'):
        v52.validate_contract({
            'protocol': v52.CONTRACT_PROTOCOL, 'fixed_test_read': True,
            'target_data_read': False,
            'post_fixed_test_hypothesis_generation': True})
