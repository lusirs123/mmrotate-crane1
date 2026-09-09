import copy

import pytest

from crane_project.tools import (
    base_v3_obb_heterogeneous_policy_development_v5 as v5)


def _row(frame, accepted=True):
    box = [float(frame), 2.0, 10.0, 5.0, 0.1]
    return dict(
        frame_key='real_seq_{:05d}'.format(frame), domain='real',
        sequence='seq', frame=frame, anchor_source='k1',
        anchor_score=0.9 if accepted else 0.1,
        base_v3_box=copy.deepcopy(box), gt_box=copy.deepcopy(box),
        errors=dict(center_error_px=0.0, long_relative_error=0.0,
                    short_relative_error=0.0, angle_error_deg=0.0),
        online_features={name: 0.1 for name in (
            'anchor_uncertainty', 'aspect_ambiguity',
            'causal_angle_residual_norm', 'causal_center_residual_norm',
            'causal_scale_residual_log',
            'k1_dino_angle_disagreement_norm',
            'k1_dino_center_disagreement_norm',
            'k1_dino_scale_disagreement_log',
            'refiner_angle_correction_norm',
            'refiner_center_correction_norm',
            'refiner_scale_correction_log')})


def _decisions(rows):
    return {row['frame_key']: {
        component: row['anchor_score'] >= 0.5
        for component in ('center', 'scale', 'angle')} for row in rows}


def test_heterogeneous_policy_has_one_frame_center_prediction_only():
    rows = [_row(1), _row(2), _row(3, False), _row(4, False)]
    output = v5.replay_policy(
        rows, _decisions(rows), risks=None, policy_mode='heterogeneous')

    third = output[rows[2]['frame_key']]['components']
    fourth = output[rows[3]['frame_key']]['components']
    assert third['center']['state'] == 'prediction'
    assert third['center']['value'] == [3.0, 2.0]
    assert third['center']['source'] == 'one_frame_center_constant_velocity'
    assert fourth['center']['state'] == 'unavailable'


def test_scale_rejects_while_angle_holds_for_one_frame():
    rows = [_row(1), _row(2, False), _row(3, False)]
    output = v5.replay_policy(
        rows, _decisions(rows), risks=None, policy_mode='heterogeneous')

    second = output[rows[1]['frame_key']]['components']
    third = output[rows[2]['frame_key']]['components']
    assert second['scale']['state'] == 'unavailable'
    assert second['angle']['state'] == 'prediction'
    assert second['angle']['source'] == 'one_frame_angle_hold'
    assert third['angle']['state'] == 'unavailable'


def test_sequence_boundary_resets_all_prediction_history():
    rows = [_row(1), _row(2)]
    new = _row(1, False)
    new.update(frame_key='real_new_00001', sequence='new')
    rows.append(new)
    output = v5.replay_policy(
        rows, _decisions(rows), risks=None, policy_mode='heterogeneous')

    components = output[new['frame_key']]['components']
    assert all(item['state'] == 'unavailable'
               for item in components.values())


def test_contract_rejects_fixed_test_authorization():
    contract = dict(
        protocol=v5.CONTRACT_PROTOCOL, fixed_test_read=True,
        target_data_read=False, post_fixed_test_hypothesis_generation=True,
        expected_inputs=dict(
            baseline_protocol=v5.BASELINE_PROTOCOL,
            runtime_calibration_protocol=v5.CALIBRATION_PROTOCOL,
            v4_report_protocol=v5.V4_PROTOCOL))
    with pytest.raises(ValueError, match='fixed TEST'):
        v5.validate_contract(contract)


def test_score_and_learned_gates_use_their_own_frozen_thresholds():
    rows = [_row(1)]
    learned_risks = {rows[0]['frame_key']: {
        'center': 0.1, 'scale': 0.6, 'angle': 0.2}}
    runtime = {'score_threshold': 0.5}
    fitted = {'components': {component: {'risk_threshold': 0.5}
                             for component in ('center', 'scale', 'angle')}}
    score, learned = v5.gate_decisions(
        rows, runtime, fitted, learned_risks)

    assert all(score[rows[0]['frame_key']].values())
    assert learned[rows[0]['frame_key']] == {
        'center': True, 'scale': False, 'angle': True}


def test_risk_fit_ignores_evaluation_labels():
    feature_names = list(_row(1)['online_features'])
    contract = dict(
        component_risk_challenger=dict(
            model='standardized_l2_logistic_error_ranker',
            features=feature_names, l2_regularization=1.0,
            maximum_newton_iterations=50, convergence_tolerance=1e-8,
            target_measurement_coverage=0.9),
        evaluation=dict(center_error_threshold_px=5.0,
                        scale_relative_error_threshold=0.1,
                        angle_error_threshold_deg=3.0))
    rows = []
    for frame in range(1, 9):
        row = _row(frame)
        row['split'] = 'calibration' if frame <= 6 else 'evaluation'
        bad = frame in (2, 5)
        row['errors'] = dict(
            center_error_px=8.0 if bad else 0.0,
            long_relative_error=0.2 if bad else 0.0,
            short_relative_error=0.0,
            angle_error_deg=5.0 if bad else 0.0)
        row['online_features'][feature_names[0]] = float(frame)
        rows.append(row)
    first = v5.fit_risk_challenger(rows, contract)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row['split'] == 'evaluation':
            row['errors'] = dict(center_error_px=999.0,
                                 long_relative_error=999.0,
                                 short_relative_error=999.0,
                                 angle_error_deg=999.0)
    second = v5.fit_risk_challenger(changed, contract)
    assert first == second
