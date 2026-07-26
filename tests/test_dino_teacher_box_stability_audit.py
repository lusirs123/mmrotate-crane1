import numpy as np

from crane_project.tools import dino_teacher_box_stability_audit as audit


def _detections(*boxes):
    if not boxes:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray([
        list(box) + [0.9] for box in boxes], dtype=np.float32)


def _records():
    return [dict(
        split='test', seq='real_seq02', frame=frame,
        image='image', annotation='annotation', domain='real')
            for frame in (1, 2, 3, 4)]


def _scope():
    return dict(values={
        ('test', 'real_seq02', frame): frame >= 2
        for frame in (1, 2, 3, 4)})


def test_transition_metrics_match_dfr_definition():
    previous = np.asarray([0, 0, 3, 4, 0], dtype=np.float32)
    current = np.asarray([0, 0, 6, 8, 0], dtype=np.float32)
    metrics = audit.transition_metrics(previous, current, frame_gap=1)
    assert metrics['dfr'] == 1.0
    assert metrics['diagonal_previous'] == 5.0
    assert metrics['diagonal_current'] == 10.0


def test_report_separates_new_transitions_from_common_output():
    baseline = [
        _detections((0, 0, 10, 10, 0)),
        _detections(),
        _detections(),
        _detections((0, 0, 10, 10, 0)),
    ]
    scoped = [
        _detections((0, 0, 10, 10, 0)),
        _detections((0, 0, 10, 10, 0)),
        _detections((0, 0, 20, 20, 0)),
        _detections((0, 0, 20, 20, 0)),
    ]
    report = audit.build_report(
        _records(), baseline, scoped, _scope(), top_transitions=2)
    seq = report['by_sequence']['real_seq02']
    assert seq['baseline']['count'] == 0
    assert seq['scoped_dino']['count'] == 3
    assert seq['common_output_baseline']['count'] == 0
    assert seq['newly_observed_scoped_dino']['count'] == 3
    assert seq['scope_internal_scoped_dino']['count'] == 2
    assert report['changed_frames']['changed_outside_scope_count'] == 0


def test_report_rejects_changes_outside_scope():
    baseline = [_detections() for _ in range(4)]
    scoped = list(baseline)
    scoped[0] = _detections((0, 0, 10, 10, 0))
    try:
        audit.build_report(
            _records(), baseline, scoped, _scope(), top_transitions=2)
    except RuntimeError as error:
        assert 'outside the declared scope' in str(error)
    else:
        raise AssertionError('Expected outside-scope change rejection')
