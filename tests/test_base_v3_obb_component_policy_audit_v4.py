import copy

from crane_project.tools import (
    base_v3_obb_component_policy_audit_v4 as audit)
from crane_project.tools.base_v3_obb_observation_interface_v3 import (
    CALIBRATION_PROTOCOL)


def _calibration():
    return dict(
        protocol=CALIBRATION_PROTOCOL,
        component_features={
            'center': ['anchor_uncertainty',
                       'k1_dino_center_disagreement_norm'],
            'scale': ['refiner_scale_correction_log'],
            'angle': ['k1_dino_angle_disagreement_norm']},
        empirical_cdf_values={
            'anchor_uncertainty': [0., .5, 1.],
            'k1_dino_center_disagreement_norm': [0., .1, .2],
            'refiner_scale_correction_log': [0., .1, .2],
            'k1_dino_angle_disagreement_norm': [0., .1, .2]},
        component_risk_thresholds={
            'center': .8, 'scale': .8, 'angle': .8},
        score_threshold=.5,
        component_output_policy={
            'center': dict(rejected_or_missing='unavailable',
                           max_hold_frames=0),
            'scale': dict(rejected_or_missing='bounded_last_measurement_hold',
                          max_hold_frames=1),
            'angle': dict(rejected_or_missing='bounded_last_measurement_hold',
                          max_hold_frames=1)})


def _row(frame, source='k1', score=.9):
    box = [float(frame), 2., 10., 5., .1]
    return dict(
        frame_key='real_seq_{:05d}'.format(frame), domain='real',
        sequence='seq', frame=frame, anchor_source=source,
        anchor_score=score, base_v3_box=copy.deepcopy(box),
        k1_box=copy.deepcopy(box) if source == 'k1' else None,
        dino_box=[float(frame)+1., 2., 10., 5., .2],
        gt_box=copy.deepcopy(box),
        errors=dict(center_error_px=0., long_relative_error=0.,
                    short_relative_error=0., angle_error_deg=0.))


def test_availability_gate_exposes_missing_disagreement_features():
    rows = [_row(1), _row(2, source='dino_fallback')]
    features = audit._features(rows)
    decisions = audit.gate_decisions(
        rows, features, _calibration(), 'feature_availability')

    assert all(decisions[rows[0]['frame_key']].values())
    fallback = decisions[rows[1]['frame_key']]
    assert fallback['center'] is False
    assert fallback['angle'] is False
    assert fallback['scale'] is True


def test_rejection_only_and_bounded_hold_share_gate_but_not_state():
    rows = [_row(1), _row(2, score=.1)]
    calibration = _calibration()
    features = audit._features(rows)
    decisions = audit.gate_decisions(rows, features, calibration, 'score')
    reject = audit.replay_policy(
        rows, features, decisions, calibration, 'score', 'rejection_only')
    hold = audit.replay_policy(
        rows, features, decisions, calibration, 'score', 'bounded_hold')

    key = rows[1]['frame_key']
    assert reject[key]['components']['scale']['state'] == 'unavailable'
    assert hold[key]['components']['scale']['state'] == 'prediction'
    assert hold[key]['components']['center']['state'] == 'unavailable'
    assert reject[key]['components']['scale']['measurement_accepted'] is False
    assert hold[key]['components']['scale']['measurement_accepted'] is False


def test_policy_replay_resets_hold_at_sequence_boundary():
    rows = [_row(1), _row(2, score=.1)]
    rows[1].update(frame_key='real_new_00001', sequence='new', frame=1)
    calibration = _calibration()
    features = audit._features(rows)
    decisions = audit.gate_decisions(rows, features, calibration, 'score')
    output = audit.replay_policy(
        rows, features, decisions, calibration, 'score', 'bounded_hold')

    assert output['real_new_00001']['components']['scale']['state'] == (
        'unavailable')
