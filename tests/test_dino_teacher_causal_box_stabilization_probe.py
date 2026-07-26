import numpy as np
import pytest

from crane_project.tools import (
    dino_teacher_causal_box_stabilization_probe as probe,
)


def _record(frame, seq='real_seq02'):
    return dict(
        split='test', seq=seq, frame=frame,
        annotation='annotation', image='image')


def _detection(cx, width, height, angle, score=0.9):
    return np.asarray(
        [[cx, 0, width, height, angle, score]], dtype=np.float32)


def test_smooth_box_preserves_center_and_uses_log_size():
    previous = np.asarray([1, 2, 10, 20, 0], dtype=np.float32)
    current = np.asarray([3, 4, 40, 80, 0], dtype=np.float32)
    smoothed = probe.smooth_box(previous, current, alpha=0.5)
    assert smoothed[:2] == pytest.approx([3, 4])
    assert smoothed[2:4] == pytest.approx([20, 40])


def test_angle_smoothing_respects_pi_periodicity():
    previous = np.asarray([0, 0, 10, 10, np.deg2rad(89)], dtype=np.float32)
    current = np.asarray([0, 0, 10, 10, np.deg2rad(-89)], dtype=np.float32)
    smoothed = probe.smooth_box(previous, current, alpha=0.5)
    assert abs(abs(np.rad2deg(smoothed[4])) - 90.0) < 1e-3


def test_smoothing_handles_equivalent_edge_swap_without_box_distortion():
    previous = np.asarray([0, 0, 10, 20, 0], dtype=np.float32)
    current = np.asarray([0, 0, 20, 10, np.pi / 2], dtype=np.float32)
    smoothed = probe.smooth_box(previous, current, alpha=0.5)
    assert smoothed[2:4] == pytest.approx([10, 20])
    assert smoothed[4] == pytest.approx(0.0, abs=1e-6)


def test_causal_smoothing_changes_only_enabled_top1_geometry():
    records = [_record(frame) for frame in (1, 2, 3)]
    raw = [
        _detection(1, 10, 10, 0),
        _detection(2, 20, 20, 0),
        _detection(3, 40, 40, 0),
    ]
    scope = {
        ('test', 'real_seq02', 1): False,
        ('test', 'real_seq02', 2): True,
        ('test', 'real_seq02', 3): True,
    }
    output = probe.apply_causal_smoothing(
        records, raw, alpha=0.5, scope_values=scope)
    assert np.array_equal(output[0], raw[0])
    assert np.array_equal(output[1], raw[1])
    assert output[2][0, 0] == raw[2][0, 0]
    assert output[2][0, 2] == pytest.approx(np.sqrt(20 * 40))
    assert output[2][0, 5] == raw[2][0, 5]


def test_silence_resets_causal_state():
    records = [_record(frame) for frame in (1, 2, 3)]
    raw = [
        _detection(1, 10, 10, 0),
        np.zeros((0, 6), dtype=np.float32),
        _detection(3, 40, 40, 0),
    ]
    output = probe.apply_causal_smoothing(records, raw, alpha=0.25)
    assert np.array_equal(output[2], raw[2])


def test_alpha_one_is_exact_identity():
    records = [_record(frame) for frame in (1, 2)]
    raw = [
        _detection(10, 30, 40, -1.2, score=0.7),
        _detection(11, 35, 37, 1.4, score=0.8),
    ]
    output = probe.apply_causal_smoothing(records, raw, alpha=1.0)
    assert all(np.array_equal(before, after)
               for before, after in zip(raw, output))
