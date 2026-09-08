import copy

from crane_project.tools import (
    base_v3_obb_component_reliability_continuous as reliability)


def _row(frame, center_x=None):
    center_x = float(frame) if center_x is None else float(center_x)
    return dict(
        frame_key='real_seq_{}'.format(frame), domain='real',
        sequence='seq', frame=frame, anchor_source='k1', anchor_score=0.8,
        base_v3_box=[center_x, 0.0, 10.0, 5.0, 0.0],
        k1_box=[center_x, 0.0, 10.0, 5.0, 0.0],
        dino_box=[center_x+1.0, 0.0, 10.0, 5.0, 0.0],
        gt_box=[center_x, 0.0, 10.0, 5.0, 0.0],
        errors=dict(center_error_px=0.0, long_relative_error=0.0,
                    short_relative_error=0.0, angle_error_deg=0.0))


def test_online_features_do_not_read_gt_or_offline_errors():
    row = _row(1)
    changed_labels = copy.deepcopy(row)
    changed_labels['gt_box'] = [999.0, 999.0, 2.0, 1.0, 1.2]
    changed_labels['errors'] = dict(center_error_px=999.0,
                                    long_relative_error=9.0,
                                    short_relative_error=9.0,
                                    angle_error_deg=80.0)
    assert reliability._frame_features(row, []) == reliability._frame_features(
        changed_labels, [])


def test_causal_history_resets_at_frame_gap():
    rows = reliability.build_feature_rows([_row(1), _row(2), _row(4)])
    assert rows[0]['online_features']['causal_center_residual_norm'] is None
    assert rows[1]['online_features']['causal_center_residual_norm'] is not None
    assert rows[2]['online_features']['causal_center_residual_norm'] is None


def test_continuous_output_expires_and_becomes_unavailable():
    rows = [_row(frame) for frame in range(1, 6)]
    accepted = {
        row['frame_key']: {component: row['frame'] <= 2
                           for component in reliability.COMPONENTS}
        for row in rows}
    states = reliability.continuous_states(
        rows, accepted, max_gap=2, enable_prediction=True)
    assert [states[row['frame_key']]['center']['state'] for row in rows] == [
        'measurement', 'measurement', 'prediction', 'prediction',
        'unavailable']
    assert states[rows[2]['frame_key']]['center']['value'] == [3.0, 0.0]
    assert states[rows[3]['frame_key']]['scale']['value'] == [10.0, 5.0]
