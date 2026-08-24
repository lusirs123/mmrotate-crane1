import numpy as np

from crane_project.tools import symeood_dino_distillation_support_audit as audit


def _box():
    return np.asarray([10.0, 10.0, 8.0, 4.0, 0.0, 0.9])


def test_present_wrong_teacher_gain_is_separate_from_missing_rescue():
    gt = _box()
    dino = _box()
    wrong = _box()
    wrong[0] = 100.0
    present = audit._classify_frame(
        wrong, dino, gt, hit_iou=0.5, gain_delta=0.1)
    missing = audit._classify_frame(
        None, dino, gt, hit_iou=0.5, gain_delta=0.1)
    assert present['hard_teacher_gain'] is True
    assert present['present_wrong_teacher_gain'] is True
    assert missing['hard_teacher_gain'] is True
    assert missing['present_wrong_teacher_gain'] is False


def test_student_preserve_is_not_teacher_gain():
    gt = _box()
    wrong = _box()
    wrong[0] = 100.0
    result = audit._classify_frame(
        gt, wrong, gt, hit_iou=0.5, gain_delta=0.1)
    assert result['student_preserve'] is True
    assert result['teacher_gain'] is False
    assert result['category'] == 'student_preserve'


class _Args:
    min_distill_gain_frames = 4
    min_distill_real_sequences = 2
    min_distill_gain_per_real_sequence = 2
    min_present_wrong_frames = 4
    min_present_wrong_per_sequence = 2
    min_router_real_sequences = 2


def test_support_gate_requires_cross_sequence_present_wrong_support():
    frames = []
    sequences = []
    for sequence in ('real_seq01', 'real_seq04'):
        sequences.append(dict(
            sequence=sequence, domain='real', teacher_gain_count=2,
            present_wrong_teacher_gain_count=2))
        for _ in range(2):
            frames.append(dict(
                teacher_gain=True, present_wrong_teacher_gain=True))
    gate = audit._support_gate(frames, sequences, _Args())
    assert gate['eligible_for_instance_distillation'] is True
    assert gate['eligible_for_learned_router'] is True

    sequences[1]['present_wrong_teacher_gain_count'] = 1
    gate = audit._support_gate(frames, sequences, _Args())
    assert gate['eligible_for_instance_distillation'] is True
    assert gate['eligible_for_learned_router'] is False
