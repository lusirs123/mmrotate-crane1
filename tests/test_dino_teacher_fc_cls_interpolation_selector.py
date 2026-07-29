import argparse

import pytest
import torch

from crane_project.tools import (
    dino_teacher_fc_cls_interpolation_selector as selector)


def _checkpoint(classifier_value, non_classifier_value=3.0):
    state = {
        'roi_head.bbox_head.fc_cls.weight': torch.full((2, 3),
                                                       classifier_value),
        'roi_head.bbox_head.fc_cls.bias': torch.full((2,),
                                                     classifier_value),
        'rpn_head.conv.weight': torch.full((1,), non_classifier_value),
    }
    return dict(
        source_only=True, frozen_dinov2=True, in_channels=1024,
        patch_size=14, rpn_feat_channels=256, roi_fc_channels=1024,
        heads_state_dict=state,
        source_sampling=dict(
            definition='source_train_short_token_lower_tertile',
            short_token_threshold=1.5),
        training_protocol=dict(train_components='roi_cls'))


def _args():
    return argparse.Namespace(
        patch_size=14, rpn_feat_channels=256, roi_fc_channels=1024)


def _row(frame, hit):
    return dict(
        seq='real_seq07', frame=frame,
        metrics=dict(top1_hit=hit, best_usable_rank=(1 if hit else None),
                     top1_riou=(0.6 if hit else 0.1), top1_score=0.9))


def _summary(top1, mcml):
    return dict(top1_hits=top1, top1_mcml=mcml)


def test_checkpoint_pair_allows_only_classifier_difference():
    old = _checkpoint(1.0)
    updated = _checkpoint(2.0)
    sampling = selector.validate_checkpoint_pair(old, updated, 1024, _args())
    assert sampling['short_token_threshold'] == pytest.approx(1.5)


def test_checkpoint_pair_rejects_frozen_head_difference():
    old = _checkpoint(1.0)
    updated = _checkpoint(2.0, non_classifier_value=4.0)
    with pytest.raises(RuntimeError, match='frozen head tensor'):
        selector.validate_checkpoint_pair(old, updated, 1024, _args())


def test_interpolation_preserves_non_classifier_and_endpoints():
    old = _checkpoint(1.0)['heads_state_dict']
    updated = _checkpoint(3.0)['heads_state_dict']
    middle = selector.interpolated_state(old, updated, 0.25)
    assert middle['roi_head.bbox_head.fc_cls.weight'] == pytest.approx(
        torch.full((2, 3), 1.5))
    assert torch.equal(middle['rpn_head.conv.weight'],
                       old['rpn_head.conv.weight'])
    assert torch.equal(
        selector.interpolated_state(old, updated, 0.0)[
            'roi_head.bbox_head.fc_cls.bias'],
        old['roi_head.bbox_head.fc_cls.bias'])
    assert torch.equal(
        selector.interpolated_state(old, updated, 1.0)[
            'roi_head.bbox_head.fc_cls.bias'],
        updated['roi_head.bbox_head.fc_cls.bias'])


def test_retention_counts_detects_gains_and_regressions():
    baseline = [_row(1, True), _row(2, True), _row(3, False)]
    candidate = [_row(1, True), _row(2, False), _row(3, True)]
    result = selector.retention_counts(baseline, candidate)
    assert result['baseline_correct_count'] == 2
    assert result['retained_correct_count'] == 1
    assert result['newly_correct_count'] == 1
    assert result['newly_incorrect_count'] == 1
    assert result['old_correct_retention_rate'] == pytest.approx(0.5)


def test_source_gate_rejects_net_gain_that_breaks_an_old_correct_frame():
    gate = selector.source_gate(
        _summary(10, 7), _summary(4, 7),
        _summary(11, 4), _summary(5, 4),
        dict(newly_incorrect_count=1))
    assert gate['passed'] is False
    assert gate['checks']['full_top1_not_lower'] is True
    assert gate['checks']['small_top1_strictly_higher'] is True
    assert gate['checks']['old_correct_frames_fully_retained'] is False


def test_source_gate_accepts_strict_small_gain_without_regression():
    gate = selector.source_gate(
        _summary(10, 7), _summary(4, 7),
        _summary(11, 6), _summary(5, 6),
        dict(newly_incorrect_count=0))
    assert gate['passed'] is True
