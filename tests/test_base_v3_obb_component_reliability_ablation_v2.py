import copy

from crane_project.tools import (
    base_v3_obb_component_reliability_ablation_v2 as ablation)
from crane_project.tools import (
    base_v3_obb_component_reliability_continuous as reliability)


def _row(frame, center=None):
    center = float(frame if center is None else center)
    return dict(
        frame_key='real_seq_{}'.format(frame), domain='real', sequence='seq',
        frame=frame, split='evaluation', anchor_source='k1', anchor_score=.8,
        base_v3_box=[center, 0., 10., 5., 0.],
        k1_box=[center, 0., 10., 5., 0.],
        dino_box=[center+1., 0., 10., 5., 0.],
        gt_box=[float(frame), 0., 10., 5., 0.],
        errors=dict(center_error_px=abs(center-frame),
                    long_relative_error=0., short_relative_error=0.,
                    angle_error_deg=0.))


def test_hold_and_constant_velocity_share_measurement_gate():
    rows = [_row(frame) for frame in range(1, 5)]
    accepted = {row['frame_key']: {component: row['frame'] <= 2
                                   for component in reliability.COMPONENTS}
                for row in rows}
    hold = ablation.output_states(rows, accepted, 2, 'hold')
    velocity = ablation.output_states(
        rows, accepted, 2, 'center_constant_velocity')

    for row in rows:
        assert hold[row['frame_key']]['center']['state'] == velocity[
            row['frame_key']]['center']['state']
    assert hold['real_seq_3']['center']['value'] == [2., 0.]
    assert velocity['real_seq_3']['center']['value'] == [3., 0.]


def test_available_error_rate_counts_bad_predictions():
    rows = [_row(1), _row(2), _row(3)]
    accepted = {row['frame_key']: {component: row['frame'] <= 2
                                   for component in reliability.COMPONENTS}
                for row in rows}
    states = ablation.output_states(rows, accepted, 2, 'hold')
    states['real_seq_3']['center']['error'] = 10.
    report = ablation.evaluate(
        rows, accepted, states,
        dict(center=5., scale=.1, angle=3.), [.5, 1.])

    assert report['center']['prediction_count'] == 1
    assert report['center']['bad_available_output_rate'] == 1/3
    assert report['center']['correct_output_coverage'] == 2/3


def test_synthetic_dropout_is_deterministic_and_gt_independent():
    rows = [_row(frame) for frame in range(1, 15)]
    changed = copy.deepcopy(rows)
    for row in changed:
        row['gt_box'][0] += 999.
    contract = dict(stride_frames=5, start_offset_frames=2,
                    run_lengths=[1, 2, 3])

    keys, runs = ablation.synthetic_dropout_keys(rows, contract)
    changed_keys, changed_runs = ablation.synthetic_dropout_keys(
        changed, contract)

    assert keys == changed_keys
    assert runs == changed_runs
    assert [len(run['frame_keys']) for run in runs] == [1, 2, 2]
