import copy

from crane_project.tools import base_v3_obb_observation_interface_v3 as v3


def _calibration(score_threshold=.5):
    return dict(
        protocol=v3.CALIBRATION_PROTOCOL,
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
        score_threshold=score_threshold,
        component_output_policy={
            'center': dict(rejected_or_missing='unavailable',
                           max_hold_frames=0),
            'scale': dict(
                rejected_or_missing='bounded_last_measurement_hold',
                max_hold_frames=1),
            'angle': dict(
                rejected_or_missing='bounded_last_measurement_hold',
                max_hold_frames=1)})


def _row(frame, score=.9, sequence='seq', base=True):
    box = [float(frame), 2., 10., 5., .1]
    return dict(
        frame_key='real_{}_{}'.format(sequence, frame), domain='real',
        sequence=sequence, frame=frame, anchor_source='k1',
        anchor_score=score,
        base_v3_box=copy.deepcopy(box) if base else None,
        k1_box=copy.deepcopy(box),
        dino_box=[float(frame)+1., 2., 10., 5., .2])


def test_online_output_is_independent_of_gt_and_offline_errors():
    first = _row(1)
    with_gt = copy.deepcopy(first)
    with_gt['gt_box'] = [999., 999., 99., 1., 1.]
    with_gt['errors'] = dict(center_error_px=999.)

    plain_output = v3.ComponentObservationManager(
        _calibration(), 'component').update(first)
    gt_output = v3.ComponentObservationManager(
        _calibration(), 'component').update(with_gt)

    assert plain_output == gt_output
    assert all(key not in plain_output for key in ('gt_box', 'errors'))


def test_center_unavailable_but_scale_and_angle_hold_for_one_frame():
    manager = v3.ComponentObservationManager(_calibration(), 'score')
    accepted = manager.update(_row(1, score=.9))
    rejected = manager.update(_row(2, score=.1))
    expired = manager.update(_row(3, score=.1))

    assert all(accepted['components'][name]['state'] == 'measurement'
               for name in v3.COMPONENTS)
    assert rejected['components']['center']['state'] == 'unavailable'
    assert rejected['components']['center']['value'] is None
    assert rejected['components']['scale']['state'] == 'prediction'
    assert rejected['components']['angle']['state'] == 'prediction'
    assert rejected['components']['scale']['age_since_measurement_frames'] == 1
    assert expired['components']['scale']['state'] == 'unavailable'
    assert expired['components']['angle']['state'] == 'unavailable'
    assert expired['components']['scale']['age_since_measurement_frames'] == 2


def test_real_missing_reset_and_reacquisition_are_explicit():
    manager = v3.ComponentObservationManager(_calibration(), 'score')
    manager.update(_row(1, score=.9))
    missing = manager.update(_row(2, base=False))
    reset_missing = manager.update(_row(
        1, score=.9, sequence='new_seq', base=False))
    reacquired = manager.update(_row(2, score=.9, sequence='new_seq'))

    assert missing['components']['center']['reason'] == 'detector_missing'
    assert missing['components']['center']['state'] == 'unavailable'
    assert missing['components']['scale']['state'] == 'prediction'
    assert reset_missing['components']['scale']['state'] == 'unavailable'
    assert reset_missing['components']['scale'][
        'age_since_measurement_frames'] is None
    assert all(reacquired['components'][name]['state'] == 'measurement'
               for name in v3.COMPONENTS)
    assert all(reacquired['components'][name][
        'age_since_measurement_frames'] == 0 for name in v3.COMPONENTS)


def test_score_curve_uses_raw_score_order_without_cdf_ties():
    rows = []
    for key, score, error in [('high', .91, 1.), ('mid', .90, 10.),
                              ('low', .10, 2.)]:
        row = _row(len(rows)+1, score=score)
        row.update(frame_key=key,
                   online_features=dict(
                       anchor_uncertainty=1.-score,
                       refiner_scale_correction_log=0.,
                       k1_dino_angle_disagreement_norm=0.),
                   errors=dict(center_error_px=error,
                               long_relative_error=error/100.,
                               short_relative_error=0., angle_error_deg=error))
        rows.append(row)
    contract = dict(
        risk_coverage_targets=[1/3],
        evaluation=dict(center_error_threshold_px=5.,
                        scale_relative_error_threshold=.1,
                        angle_error_threshold_deg=3.))

    curves = v3._direct_risk_curves(rows, _calibration(), contract)

    selected = curves['center']['raw_anchor_score'][0]
    assert selected['retained_frame_keys'] == ['high']
    assert selected['mean_error'] == 1.
