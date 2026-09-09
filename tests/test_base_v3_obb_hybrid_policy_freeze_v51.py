import copy

import pytest

from crane_project.tools import base_v3_obb_hybrid_policy_freeze_v51 as v51


def _row(frame):
    box = [float(frame), 2.0, 10.0, 5.0, 0.1]
    return dict(
        frame_key='real_seq_{:05d}'.format(frame), domain='real',
        sequence='seq', frame=frame, anchor_source='k1', anchor_score=0.9,
        base_v3_box=copy.deepcopy(box), gt_box=copy.deepcopy(box))


def _risks(rows):
    return {row['frame_key']: {
        'center': 0.1, 'scale': 0.2, 'angle': 0.3} for row in rows}


def test_hybrid_uses_score_for_center_and_angle_but_risk_for_scale():
    rows = [_row(1)]
    key = rows[0]['frame_key']
    score = {key: {'center': True, 'scale': True, 'angle': False}}
    learned = {key: {'center': False, 'scale': False, 'angle': True}}

    result = v51.hybrid_decisions(rows, score, learned)

    assert result[key] == {'center': True, 'scale': False, 'angle': False}


def test_rejected_center_and_scale_are_unavailable_while_angle_holds_once():
    rows = [_row(1), _row(2), _row(3)]
    decisions = {
        rows[0]['frame_key']: {'center': True, 'scale': True, 'angle': True},
        rows[1]['frame_key']: {'center': False, 'scale': False, 'angle': False},
        rows[2]['frame_key']: {'center': False, 'scale': False, 'angle': False}}
    output = v51.replay_hybrid(rows, decisions, _risks(rows))

    second = output[rows[1]['frame_key']]['components']
    third = output[rows[2]['frame_key']]['components']
    assert second['center']['state'] == 'unavailable'
    assert second['scale']['state'] == 'unavailable'
    assert second['angle']['state'] == 'prediction'
    assert second['angle']['source'] == 'one_frame_angle_hold'
    assert third['angle']['state'] == 'unavailable'


def test_sequence_boundary_clears_angle_hold():
    rows = [_row(1)]
    new = _row(1)
    new.update(frame_key='real_new_00001', sequence='new')
    rows.append(new)
    decisions = {
        rows[0]['frame_key']: {'center': True, 'scale': True, 'angle': True},
        rows[1]['frame_key']: {'center': False, 'scale': False, 'angle': False}}
    output = v51.replay_hybrid(rows, decisions, _risks(rows))

    assert output[new['frame_key']]['components']['angle']['state'] == (
        'unavailable')


def test_runtime_artifact_contains_only_selected_scale_ranker():
    fitted = dict(
        model='ranker', feature_names=['a'], expanded_feature_names=['a'],
        preprocessing={'medians': [0.0], 'means': [0.0], 'scales': [1.0]},
        components={
            'center': {'weights': [1.0], 'risk_threshold': 0.1},
            'scale': {'weights': [2.0], 'risk_threshold': 0.2},
            'angle': {'weights': [3.0], 'risk_threshold': 0.3}})
    report = {'fitted_component_risk_challenger': fitted}
    contract = dict(
        evidence_boundary='source',
        expected_inputs={'baseline_portable_sha256': 'a'*64},
        frozen_component_policy={'center': {}, 'scale': {}, 'angle': {}},
        claim_limit='development only')
    identities = {
        'v5_report': {'sha256': 'b'*64},
        'v5_contract': {'sha256': 'c'*64}}

    runtime = v51.build_runtime_calibration(
        {'score_threshold': 0.5}, report, contract, identities, True)

    assert runtime['scale_error_ranker']['weights'] == [2.0]
    assert 'center_error_ranker' not in runtime
    assert 'angle_error_ranker' not in runtime
    assert runtime['online_contract']['gt_fields_consumed'] == []


def test_contract_rejects_fixed_test_authorization():
    with pytest.raises(ValueError, match='fixed TEST'):
        v51.validate_contract(dict(
            protocol=v51.CONTRACT_PROTOCOL, fixed_test_read=True,
            target_data_read=False, post_fixed_test_hypothesis_generation=True))
