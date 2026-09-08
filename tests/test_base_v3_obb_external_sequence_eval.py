import copy
import json
from pathlib import Path

import pytest

from crane_project.tools import base_v3_obb_external_sequence_eval as external
from crane_project.tools.base_v3_obb_observation_interface_v3 import (
    CALIBRATION_PROTOCOL)


CONTRACT_PATH = Path(
    'crane_project/data_contracts/base_v3_obb_external_sequence_eval_v1.json')


def _contract():
    return json.loads(CONTRACT_PATH.read_text())


def _calibration():
    return dict(
        protocol=CALIBRATION_PROTOCOL,
        component_features={
            'center': ['anchor_uncertainty'],
            'scale': ['refiner_scale_correction_log'],
            'angle': ['k1_dino_angle_disagreement_norm']},
        empirical_cdf_values={
            'anchor_uncertainty': [0., .25, .5, .75, 1.],
            'refiner_scale_correction_log': [0., .1, .2, .3, .4],
            'k1_dino_angle_disagreement_norm': [0., .1, .2, .3, .4]},
        component_risk_thresholds={
            'center': 1., 'scale': 1., 'angle': 1.},
        score_threshold=.5,
        component_output_policy={
            'center': dict(rejected_or_missing='unavailable',
                           max_hold_frames=0),
            'scale': dict(
                rejected_or_missing='bounded_last_measurement_hold',
                max_hold_frames=1),
            'angle': dict(
                rejected_or_missing='bounded_last_measurement_hold',
                max_hold_frames=1)})


def _row(frame, base=True, gt=True, score=.9, sequence='external_01'):
    box = [float(frame), 2., 10., 5., .1]
    row = dict(
        frame_key='new_{}_{}'.format(sequence, frame), domain='real',
        sequence=sequence, frame=frame, anchor_source='k1',
        anchor_score=score,
        base_v3_box=copy.deepcopy(box) if base else None,
        k1_box=copy.deepcopy(box),
        dino_box=[float(frame)+1., 2., 10., 5., .2])
    if gt:
        row['gt_box'] = copy.deepcopy(box)
    return row


def _payload(rows, annotation='complete'):
    return dict(
        protocol=external.RECORDS_PROTOCOL,
        dataset_id='independent_video_01',
        capture_session_id='capture_20260908_01',
        frame_manifest_sha256='a'*64,
        annotation_status=annotation,
        not_used_for_calibration_or_feature_selection=True,
        records=rows)


def _development():
    return dict(protocol='base_v3_obb_observation_interface_v3', records=[
        dict(frame_key='old_1', domain='real', sequence='source_seq', frame=1)])


def test_frozen_annotated_replay_handles_real_missing_and_reacquisition():
    records = [_row(1), _row(2, base=False), _row(3)]
    calibration = _calibration()
    before = copy.deepcopy(calibration)

    result = external.run(
        _payload(records), calibration, _development(), _contract())

    assert calibration == before
    assert result['frozen_calibration'] is True
    assert result['threshold_fitting_performed'] is False
    assert result['feature_selection_performed'] is False
    assert result['claim_status'] == 'INDEPENDENT_SEQUENCE_EVALUATION_CANDIDATE'
    missing = result['records'][1]['online_observations']['component']
    assert missing['center']['state'] == 'unavailable'
    assert missing['scale']['state'] == 'prediction'
    assert missing['angle']['state'] == 'prediction'
    assert result['records'][2]['online_observations']['component'][
        'center']['state'] == 'measurement'
    assert result['method_reports']['component']['all']['center'][
        'detector_missing_frame_count'] == 1
    assert result['matched_coverage'] is not None


def test_interface_only_has_no_offline_errors_or_accuracy_claim():
    records = [_row(1, gt=False), _row(2, base=False, gt=False)]

    result = external.run(
        _payload(records, annotation='none'), _calibration(),
        _development(), _contract())

    assert result['claim_status'] == 'INTERFACE_ONLY_NO_ACCURACY_CLAIM'
    assert result['matched_coverage'] is None
    assert all('offline_errors' not in row for row in result['records'])
    center = result['method_reports']['raw']['all']['center']
    assert 'mean_available_output_error' not in center


def test_development_sequence_or_frame_overlap_is_rejected():
    same_sequence = _payload([_row(1, sequence='source_seq')])
    with pytest.raises(ValueError, match='domain/sequence identities overlap'):
        external.validate_external_records(
            same_sequence, _development(), _contract())

    same_frame = _payload([_row(1)])
    same_frame['records'][0]['frame_key'] = 'old_1'
    with pytest.raises(ValueError, match='frame keys overlap'):
        external.validate_external_records(
            same_frame, _development(), _contract())


def test_partial_annotations_and_unattested_provenance_are_rejected():
    partial = _payload([_row(1), _row(2, gt=False)])
    with pytest.raises(ValueError, match='gt_box on every frame'):
        external.validate_external_records(
            partial, _development(), _contract())

    unattested = _payload([_row(1)])
    unattested['not_used_for_calibration_or_feature_selection'] = False
    with pytest.raises(ValueError, match='unused for development'):
        external.validate_external_records(
            unattested, _development(), _contract())
