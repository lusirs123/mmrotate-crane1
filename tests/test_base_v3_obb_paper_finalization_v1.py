import copy

import pytest

from crane_project.tools import base_v3_obb_paper_final_report_v1 as final
from crane_project.tools.base_v3_obb_paper_runtime_v1 import (
    PaperObservationManagerV1)


def _calibration():
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
            feature_names=['anchor_uncertainty'],
            expanded_feature_names=['anchor_uncertainty',
                                    'anchor_uncertainty__missing'],
            preprocessing=dict(medians=[0.0], means=[0.0, 0.0],
                               scales=[1.0, 1.0]),
            weights=[0.0, 0.0, 0.0], risk_threshold=0.6,
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
        k1_box=copy.deepcopy(box), dino_box=copy.deepcopy(box),
        gt_box=[999.0, 999.0, 1.0, 1.0, 0.0])


def test_runtime_emits_three_methods_and_strips_gt():
    manager = PaperObservationManagerV1(_calibration())
    first = manager.update(_row(1))
    second = manager.update(_row(2, score=0.1))
    assert set(first['observations']) == {
        'raw', 'score_rejection_only', 'v51_hybrid'}
    assert first['online_gt_fields_consumed'] == []
    assert 'gt_box' not in first
    assert first['detector']['raw_obb'] == _row(1)['base_v3_box']
    assert second['observations']['raw']['center']['state'] == 'measurement'
    assert second['observations']['score_rejection_only']['center'][
        'state'] == 'unavailable'
    assert second['observations']['v51_hybrid']['angle'][
        'state'] == 'prediction'


def test_runtime_resets_prediction_on_gap():
    manager = PaperObservationManagerV1(_calibration())
    manager.update(_row(1))
    gap = manager.update(_row(3, score=0.1))
    assert gap['observations']['v51_hybrid']['angle'][
        'state'] == 'unavailable'


def test_contract_rejects_tuning_and_overall_winner():
    base = dict(
        protocol=final.CONTRACT_PROTOCOL,
        fixed_test_read=True, fixed_test_previously_exposed=True,
        parameter_tuning_authorized=False,
        methods=list(final.METHODS), components=list(final.COMPONENTS),
        runtime_interface={'protocol': final.RUNTIME_PROTOCOL},
        expected_inputs={
            'v51_report_protocol': final.V51_REPORT_PROTOCOL,
            'v51_diagnostic_protocol': final.diagnostic.PROTOCOL,
            'v53_report_protocol': final.v53.PROTOCOL,
            'v51_eval_contract_protocol':
                final.V51_EVAL_CONTRACT_PROTOCOL,
            'runtime_calibration_protocol':
                final.V51_RUNTIME_CALIBRATION_PROTOCOL},
        evaluation={
            'riou_hit_threshold': 0.5,
            'runtime_benchmark_repeats': 1},
        selection_closure={
            'overall_winner_claimed': False,
            'future_parameter_selection_from_fixed_test': False})
    changed = copy.deepcopy(base)
    changed['parameter_tuning_authorized'] = True
    with pytest.raises(ValueError, match='authorize tuning'):
        final.validate_contract(changed)
    changed = copy.deepcopy(base)
    changed['selection_closure']['overall_winner_claimed'] = True
    with pytest.raises(ValueError, match='overall winner'):
        final.validate_contract(changed)


def test_recovery_stops_at_sequence_boundary():
    records = []
    for frame, accepted in ((1, True), (2, False), (3, True)):
        row = _row(frame)
        row['online_observations'] = {'raw': {
            component: {'measurement_accepted': accepted}
            for component in final.COMPONENTS}}
        records.append(row)
    row = _row(1, sequence='next')
    row['online_observations'] = {'raw': {
        component: {'measurement_accepted': False}
        for component in final.COMPONENTS}}
    records.append(row)
    report = final._recovery_report(records, 'raw', 'center')
    assert report['recovered_run_count'] == 1
    assert report['unrecovered_run_count'] == 1
    assert report['maximum_recovery_delay_frames'] == 1


def test_reconstruct_obb_preserves_raw_short_long_axis_order():
    row = _row(1)
    row['base_v3_box'] = [1.0, 2.0, 5.0, 10.0, 0.2]
    observations = {
        'center': dict(valid=True, value=[3.0, 4.0], state='measurement'),
        'scale': dict(valid=True, value=[12.0, 6.0], state='measurement'),
        'angle': dict(valid=True, value=0.3, state='prediction')}
    assert final._reconstruct_obb(row, observations) == [
        3.0, 4.0, 6.0, 12.0, 0.3]


def test_reconstruct_obb_returns_none_if_any_component_unavailable():
    row = _row(1)
    observations = {
        'center': dict(valid=True, value=[1.0, 2.0], state='measurement'),
        'scale': dict(valid=False, value=None, state='unavailable'),
        'angle': dict(valid=True, value=0.1, state='measurement')}
    assert final._reconstruct_obb(row, observations) is None


def test_riou_metrics_separate_availability_from_accuracy():
    row = _row(1)
    row['gt_box'] = copy.deepcopy(row['base_v3_box'])
    measurement = {
        'center': dict(valid=True, value=[1.0, 2.0], state='measurement'),
        'scale': dict(valid=True, value=[10.0, 5.0], state='measurement'),
        'angle': dict(valid=True, value=0.1, state='measurement')}
    unavailable = {
        component: dict(valid=False, value=None, state='unavailable')
        for component in final.COMPONENTS}
    outputs = {row['frame_key']: {'observations': {
        'raw': copy.deepcopy(measurement),
        'score_rejection_only': unavailable,
        'v51_hybrid': copy.deepcopy(measurement)}}}
    report, per_frame = final._riou_metrics([row], outputs, 0.5)
    assert per_frame[row['frame_key']]['raw'] == pytest.approx(1.0)
    assert report['raw']['all']['available_obb_coverage'] == 1.0
    assert report['raw']['all']['riou_hit_coverage'] == 1.0
    assert report['score_rejection_only']['all'][
        'available_obb_coverage'] == 0.0
    assert report['score_rejection_only']['all'][
        'mean_available_riou'] is None
