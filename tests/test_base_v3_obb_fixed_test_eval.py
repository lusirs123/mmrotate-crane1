import copy
import json
from pathlib import Path

from crane_project.tools import base_v3_obb_fixed_test_eval as fixed
from crane_project.tools.base_v3_obb_observation_interface_v3 import (
    CALIBRATION_PROTOCOL)


CONTRACT_PATH = Path(
    'crane_project/data_contracts/base_v3_obb_fixed_test_eval_v1.json')


def _contract():
    contract = json.loads(CONTRACT_PATH.read_text())
    contract['fixed_test_scope'] = dict(
        frame_count=3, domain_counts={'real': 3},
        annotation_status='complete')
    return contract


def _calibration():
    return dict(
        protocol=CALIBRATION_PROTOCOL,
        component_features={
            'center': ['anchor_uncertainty'],
            'scale': ['refiner_scale_correction_log'],
            'angle': ['k1_dino_angle_disagreement_norm']},
        empirical_cdf_values={
            'anchor_uncertainty': [0., .5, 1.],
            'refiner_scale_correction_log': [0., .1, .2],
            'k1_dino_angle_disagreement_norm': [0., .1, .2]},
        component_risk_thresholds={
            'center': 1., 'scale': 1., 'angle': 1.},
        score_threshold=.5,
        component_output_policy={
            'center': dict(rejected_or_missing='unavailable',
                           max_hold_frames=0),
            'scale': dict(rejected_or_missing='bounded_last_measurement_hold',
                          max_hold_frames=1),
            'angle': dict(rejected_or_missing='bounded_last_measurement_hold',
                          max_hold_frames=1)})


def _row(frame, detected=True):
    box = [float(frame), 2., 10., 5., .1]
    return dict(
        frame_key='real_test_{:05d}'.format(frame), domain='real',
        sequence='real_test', frame=frame, anchor_source='k1',
        anchor_score=.9, base_v3_box=copy.deepcopy(box) if detected else None,
        k1_box=copy.deepcopy(box), dino_box=copy.deepcopy(box),
        gt_box=copy.deepcopy(box))


def test_frozen_fixed_test_run_reports_real_missing_and_no_refit():
    calibration = _calibration()
    before = copy.deepcopy(calibration)
    result = fixed.run(
        [_row(1), _row(2, detected=False), _row(3)], calibration,
        _contract(), dict(frame_count=3))

    assert calibration == before
    assert result['claim_status'] == (
        'FROZEN_CALIBRATION_FIXED_TEST_EVALUATION')
    assert result['fixed_test_read'] is True
    assert result['threshold_fitting_performed'] is False
    assert result['eligible_for_parameter_tuning_from_this_report'] is False
    missing = result['records'][1]['online_observations']['component']
    assert missing['center']['state'] == 'unavailable'
    assert missing['scale']['state'] == 'prediction'
    assert missing['angle']['state'] == 'prediction'
    assert result['method_reports']['component']['sequence:real_test'][
        'center']['detector_missing_frame_count'] == 1


def test_contract_requires_explicit_fixed_test_authorization():
    contract = _contract()
    contract['fixed_test_read'] = False
    try:
        fixed.validate_contract(contract)
    except ValueError as error:
        assert 'authorize fixed TEST' in str(error)
    else:
        raise AssertionError('fixed TEST must require explicit authorization')
