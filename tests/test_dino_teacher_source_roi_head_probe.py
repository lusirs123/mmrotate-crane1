import argparse

import pytest
import torch

from crane_project.tools import dino_teacher_source_roi_head_probe as probe


def _args(**overrides):
    values = dict(
        seed=0, source_folds=5, source_negatives_per_image=3,
        hidden_dim=8, epochs=2, batch_size=4, lr=0.01,
        momentum=0.9, weight_decay=1e-4,
        target_candidate_limit=10000, roi_chunk_size=4,
        min_roi_in_bounds=0.9, source_min_accuracy=0.8,
        pool_resolution=2, target_min_wins=26)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_two_fc_head_preserves_spatial_input_and_returns_two_logits():
    head = probe.TwoFCObjectnessHead(
        channels=4, pool_resolution=2, hidden_dim=8)
    features = torch.randn(3, 4, 2, 2)
    logits = head(features)
    assert logits.shape == (3, 2)
    assert head.fc1.in_features == 16
    assert head.objectness_logit(features).shape == (3,)


def test_sample_spatial_rois_keeps_channels_and_pool_grid():
    feature = torch.ones(1, 4, 8, 8)
    boxes = [
        torch.tensor([56.0, 56.0, 28.0, 28.0, 0.0]),
        torch.tensor([70.0, 70.0, 28.0, 28.0, 0.0])]
    sampled = probe.sample_spatial_rois(
        feature, boxes, patch_size=14, pool_resolution=3)
    assert sampled.shape == (2, 4, 3, 3)
    assert torch.allclose(sampled, torch.ones_like(sampled))


def test_valid_candidate_selection_skips_out_of_bounds_box():
    feature = torch.ones(1, 4, 8, 8)
    boxes = torch.tensor([
        [0.0, 0.0, 28.0, 28.0, 0.0],
        [56.0, 56.0, 28.0, 28.0, 0.0]])
    selected = probe.valid_candidate_selections(
        [0, 1], boxes,
        {'scale_factor': [1.0, 1.0, 1.0, 1.0]}, {'scale': 1.0},
        feature, patch_size=14, pool_resolution=7,
        min_in_bounds=0.9, limit=1)
    assert len(selected) == 1
    assert selected[0]['index'] == 1
    assert selected[0]['detector_rank'] == 2


def test_grouped_folds_keep_all_candidates_from_frame_together():
    samples = []
    for frame in range(10):
        for label in (0, 1):
            samples.append(dict(
                feature=torch.zeros(1, 1, 1), label=label,
                row=dict(seq='real_seq07', frame=frame)))
    fold_ids = probe.grouped_fold_ids(samples, folds=5)
    for frame in range(10):
        frame_folds = {
            fold for sample, fold in zip(samples, fold_ids)
            if sample['row']['frame'] == frame}
        assert len(frame_folds) == 1


def test_train_one_head_runs_source_validation_and_selects_checkpoint():
    samples = []
    for index in range(12):
        label = index % 2
        value = 1.0 if label else -1.0
        samples.append(dict(
            feature=torch.full((2, 2, 2), value, dtype=torch.float16),
            label=label,
            row=dict(seq='real_seq07', frame=index // 2)))
    head, result = probe.train_one_head(
        samples, list(range(8)), list(range(8, 12)),
        _args(), torch.device('cpu'), seed=0, epochs=2)
    assert isinstance(head, probe.TwoFCObjectnessHead)
    assert result['best_epoch'] in (1, 2)
    assert result['best_metrics']['count'] == 4
    assert len(result['history']) == 2


def test_target_decision_requires_global_top1_not_only_pairwise_signal():
    source = dict(valid=True)
    target = dict(
        geometry_eligible_count=31, geometry_misses=[164, 167],
        evaluable_count=31, top1_wins=0, paired_wins=31,
        paired_margin=dict(median=1.0))
    assert probe.target_decision(source, target, _args()) == (
        'PAIRWISE_SIGNAL_ONLY_GLOBAL_RANK_NOT_RESTORED')
    target['top1_wins'] = 26
    assert probe.target_decision(source, target, _args()) == (
        'SOURCE_ONLY_DINO_ROI_HEAD_RESTORES_ORDERING')


def test_validate_args_rejects_nonzero_seed():
    with pytest.raises(ValueError, match='seed 0'):
        probe.validate_args(_args(seed=1))
