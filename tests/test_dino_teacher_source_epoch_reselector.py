import pytest
import torch

from crane_project.tools import (
    dino_teacher_source_epoch_reselector as selector)


def _summary(top1, mcml, frames=100):
    return dict(top1_hits=top1, top1_mcml=mcml, frame_count=frames)


def _history(epoch, retained, gained, full_top1, small_top1):
    return dict(
        epoch=epoch, selection_eligible=True, checkpoint_saved=True,
        source_val=_summary(full_top1, 4),
        source_small_val=_summary(small_top1, 4, frames=50),
        source_exact_retention=dict(
            baseline_correct_count=662,
            retained_correct_count=retained,
            lost_correct_count=662 - retained,
            gained_correct_count=gained))


def _report():
    return dict(source=dict(
        baseline_validation_summary=_summary(662, 7, frames=738),
        baseline_small_validation_summary=_summary(293, 7, frames=350),
        history=[
            _history(1, 653, 25, 678, 302),
            _history(2, 652, 25, 677, 306),
            _history(3, 652, 27, 679, 307),
            _history(4, 652, 25, 677, 306)]))


def _checkpoint(classifier_value=1.0, frozen_value=3.0):
    return dict(
        source_only=True, frozen_dinov2=True,
        heads_state_dict={
            'roi_head.bbox_head.fc_cls.weight': torch.full(
                (2, 3), classifier_value),
            'roi_head.bbox_head.fc_cls.bias': torch.full(
                (2,), classifier_value),
            'rpn_head.conv.weight': torch.full((1,), frozen_value)})


def test_fixed_985_retention_selects_epoch_one():
    selected, candidates = selector.select_source_epoch(
        _report(), min_retention_rate=0.985)
    assert selected['epoch'] == 1
    assert candidates[0]['gate']['passed'] is True
    assert candidates[0]['gate']['exact_retention_rate'] == pytest.approx(
        653 / 662)
    assert all(not row['gate']['passed'] for row in candidates[1:])


def test_relaxed_98_retention_would_select_epoch_three():
    selected, _candidates = selector.select_source_epoch(
        _report(), min_retention_rate=0.98)
    assert selected['epoch'] == 3


def test_gate_requires_both_source_improvement_and_saved_checkpoint():
    baseline_full = _summary(662, 7)
    baseline_small = _summary(293, 7)
    row = _history(1, 653, 25, 662, 302)
    row['checkpoint_saved'] = False
    gate = selector.candidate_gate(
        row, baseline_full, baseline_small, min_retention_rate=0.985)
    assert gate['passed'] is False
    assert gate['checks']['checkpoint_saved'] is False
    assert gate['checks']['full_top1_strictly_higher'] is False


def test_classifier_only_invariant_accepts_fc_cls_update():
    result = selector.validate_classifier_only_change(
        _checkpoint(1.0), _checkpoint(2.0))
    assert result['changed_parameter_count'] == 8
    assert result['frozen_tensors_bit_identical'] is True


def test_classifier_only_invariant_rejects_frozen_parameter_change():
    with pytest.raises(RuntimeError, match='Frozen head tensor changed'):
        selector.validate_classifier_only_change(
            _checkpoint(1.0, 3.0), _checkpoint(2.0, 4.0))


def test_retention_rate_rejects_invalid_counts():
    with pytest.raises(RuntimeError, match='Invalid'):
        selector.retention_rate(dict(
            baseline_correct_count=10, retained_correct_count=11))
