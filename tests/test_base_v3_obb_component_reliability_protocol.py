from crane_project.tools import (
    base_v3_obb_component_reliability_continuous as reliability)


def _row(frame):
    return dict(frame_key='real_seq_{}'.format(frame), domain='real',
                sequence='seq', frame=frame)


def test_split_is_local_to_each_contiguous_segment():
    rows = [_row(1), _row(2), _row(3), _row(10), _row(11), _row(12)]
    counts = reliability.assign_split(rows, fraction=0.5, minimum=1)

    assert counts == {'calibration': 2, 'evaluation': 4}
    assert [row['split'] for row in rows] == [
        'calibration', 'evaluation', 'evaluation',
        'calibration', 'evaluation', 'evaluation']


def test_recovery_delay_counts_only_contiguous_rejections():
    rows = [_row(frame) for frame in range(1, 7)]
    pattern = [True, False, False, True, False, True]
    accepted = {row['frame_key']: {'center': pattern[index]}
                for index, row in enumerate(rows)}

    assert reliability._rejection_runs(rows, accepted, 'center') == [
        {'length': 2, 'recovered': True},
        {'length': 1, 'recovered': True}]
