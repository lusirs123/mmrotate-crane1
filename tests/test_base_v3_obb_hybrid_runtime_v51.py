import copy

import pytest

from crane_project.tools.base_v3_obb_hybrid_runtime_v51 import (
    HybridObservationManagerV51, scale_error_risk,
    validate_runtime_calibration)


FEATURES = ['anchor_uncertainty']


def _calibration(scale_threshold=0.6):
    return dict(
        protocol='base_v3_obb_hybrid_observation_calibration_v51',
        component_policy={
            'center': dict(gate='anchor_score',
                           rejected_or_missing='unavailable',
                           max_prediction_frames=0),
            'scale': dict(gate='v5_supervised_scale_error_risk',
                          rejected_or_missing='unavailable',
                          max_prediction_frames=0),
            'angle': dict(gate='anchor_score',
                          rejected_or_missing='bounded_last_measurement_hold',
                          max_prediction_frames=1)},
        score_gate=dict(score_threshold=0.5),
        scale_error_ranker=dict(
            feature_names=FEATURES,
            expanded_feature_names=['anchor_uncertainty',
                                    'anchor_uncertainty__missing'],
            preprocessing=dict(
                medians=[0.0], means=[0.0, 0.0], scales=[1.0, 1.0]),
            weights=[0.0, 0.0, 0.0], risk_threshold=scale_threshold,
            online_gt_fields_consumed=[]),
        online_contract=dict(
            gt_fields_consumed=[], future_frames_used=False,
            domain_identity_used_in_decisions=False))


def _row(frame, score=0.9, sequence='seq'):
    box = [float(frame), 2.0, 10.0, 5.0, 0.1]
    return dict(
        frame_key='real_{}_{:05d}'.format(sequence, frame), domain='real',
        sequence=sequence, frame=frame, anchor_source='k1',
        anchor_score=score, base_v3_box=copy.deepcopy(box),
        k1_box=copy.deepcopy(box),
        dino_box=[float(frame)+1.0, 2.0, 10.0, 5.0, 0.2])


def test_scale_risk_uses_frozen_preprocessing_and_weights():
    calibration = _calibration()
    calibration['scale_error_ranker']['weights'] = [0.0, 2.0, 0.0]
    assert scale_error_risk({'anchor_uncertainty': 0.0}, calibration) == 0.5
    assert scale_error_risk({'anchor_uncertainty': 1.0}, calibration) > 0.5


def test_online_manager_never_predicts_center_or_scale():
    manager = HybridObservationManagerV51(_calibration(scale_threshold=0.4))
    first = manager.update(_row(1))
    second = manager.update(_row(2, score=0.1))
    third = manager.update(_row(3, score=0.1))

    assert first['components']['center']['state'] == 'measurement'
    assert first['components']['scale']['state'] == 'unavailable'
    assert second['components']['center']['state'] == 'unavailable'
    assert second['components']['scale']['state'] == 'unavailable'
    assert second['components']['angle']['state'] == 'prediction'
    assert third['components']['angle']['state'] == 'unavailable'


def test_online_manager_resets_hold_on_sequence_change():
    manager = HybridObservationManagerV51(_calibration())
    manager.update(_row(1))
    changed = manager.update(_row(1, score=0.1, sequence='new'))
    assert changed['components']['angle']['state'] == 'unavailable'


def test_runtime_validation_rejects_gt_inputs():
    calibration = _calibration()
    calibration['online_contract']['gt_fields_consumed'] = ['gt_box']
    with pytest.raises(ValueError, match='ground truth'):
        validate_runtime_calibration(calibration)
