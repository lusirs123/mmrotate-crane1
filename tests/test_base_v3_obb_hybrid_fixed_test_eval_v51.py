import copy

import pytest

from crane_project.tools import base_v3_obb_hybrid_fixed_test_eval_v51 as v51


def _contract():
    return dict(
        protocol=v51.CONTRACT_PROTOCOL,
        evidence_boundary='post-exposure-test',
        fixed_test_read=True, target_data_read=True,
        fixed_test_previously_exposed=True,
        post_fixed_test_hypothesis_generation=True,
        fixed_test_reuse_role='post_exposure_frozen_policy_reevaluation',
        expected_inputs=dict(
            runtime_calibration_protocol=(
                'base_v3_obb_hybrid_observation_calibration_v51'),
            policy_freeze_report_protocol=(
                'base_v3_obb_hybrid_policy_freeze_v51'),
            previous_fixed_test_report_protocol=(
                'base_v3_obb_fixed_test_eval_v1'),
            attribution_protocol='k1_anchor_fallback_attribution_audit_v1',
            attribution_evidence_role='fixed-target',
            all_lane_audit_protocol='source_owned_geometry_union_v2'),
        fixed_test_scope=dict(
            frame_count=3, domain_counts={'real': 3},
            annotation_status='complete'),
        evaluation=dict(
            center_error_threshold_px=5.0,
            scale_relative_error_threshold=0.1,
            angle_error_threshold_deg=3.0,
            tail_quantiles=[0.5, 0.9, 1.0]),
        claim_limit='post-exposure comparison only')


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
            feature_names=['anchor_uncertainty'],
            expanded_feature_names=[
                'anchor_uncertainty', 'anchor_uncertainty__missing'],
            preprocessing=dict(
                medians=[0.0], means=[0.0, 0.0], scales=[1.0, 1.0]),
            weights=[0.0, 0.0, 0.0], risk_threshold=scale_threshold,
            online_gt_fields_consumed=[]),
        online_contract=dict(
            gt_fields_consumed=[], future_frames_used=False,
            domain_identity_used_in_decisions=False))


def _row(frame, score=0.9):
    box = [float(frame), 2.0, 10.0, 5.0, 0.1]
    return dict(
        frame_key='real_seq_{:05d}'.format(frame), domain='real',
        sequence='seq', frame=frame, anchor_source='k1',
        anchor_score=score, base_v3_box=copy.deepcopy(box),
        k1_box=copy.deepcopy(box), dino_box=copy.deepcopy(box),
        gt_box=copy.deepcopy(box))


def test_run_compares_raw_score_and_frozen_v51_without_refit():
    rows = [_row(1), _row(2, score=0.1), _row(3, score=0.1)]
    calibration = _calibration()
    before = copy.deepcopy(calibration)
    result = v51.run(rows, calibration, _contract(), {'frame_count': 3})

    assert calibration == before
    assert result['method_inventory'] == [
        'raw', 'score_rejection_only', 'v51_hybrid']
    assert result['threshold_fitting_performed'] is False
    assert result['eligible_for_parameter_tuning_from_this_report'] is False
    second = result['records'][1]['online_observations']
    assert second['raw']['center']['state'] == 'measurement'
    assert second['score_rejection_only']['angle']['state'] == 'unavailable'
    assert second['v51_hybrid']['center']['state'] == 'unavailable'
    assert second['v51_hybrid']['scale']['state'] == 'measurement'
    assert second['v51_hybrid']['angle']['state'] == 'prediction'
    assert result['method_reports']['v51_hybrid']['anchor:k1'][
        'angle']['prediction_count'] == 1


def test_contract_requires_explicit_prior_test_exposure():
    contract = _contract()
    contract['fixed_test_previously_exposed'] = False
    with pytest.raises(ValueError, match='prior TEST exposure'):
        v51.validate_contract(contract)
