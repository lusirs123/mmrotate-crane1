import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from crane_project.tools import guided_background_rank_probe as guided


def _args(root, **overrides):
    values = dict(
        seed=0, split='val', seq='real_seq07', start=100, end=115,
        dark_severity=0.45, steps=2, learning_rate=0.1,
        latent_size=8, pixel_epsilon=24.0, region_size_ratio=0.28,
        region_aspect=1.5, drift_ratio=0.025, target_margin_ratio=0.35,
        background_topm=8, smoothmax_temperature=0.5,
        rank_margin=3.0, background_score_ceiling=0.02,
        score_thr=0.05, ceiling_weight=1.0, tv_weight=0.05,
        l2_weight=0.01, false_iou_thr=0.1, riou_thr=0.5,
        min_top1_error_run=16, topks=[1, 10000], pool_size=10000,
        data_root=root)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_protocol_accepts_only_16_source_validation_frames():
    with tempfile.TemporaryDirectory() as root:
        frames, topks = guided.validate_args(_args(root))
        assert frames == list(range(100, 116))
        assert topks == [1, 10000]
        with pytest.raises(ValueError, match='exactly 16'):
            guided.validate_args(_args(root, end=114))
        with pytest.raises(ValueError, match='source validation'):
            guided.validate_args(_args(root, split='test'))


def test_smooth_region_trajectory_is_deterministic_and_continuous():
    first = guided.smooth_region_centers(
        (0.5, 0.5), 16, 0.025, 0, 'real_seq07')
    second = guided.smooth_region_centers(
        (0.5, 0.5), 16, 0.025, 0, 'real_seq07')
    assert first == second
    for left, right in zip(first, first[1:]):
        assert abs(left[0] - right[0]) < 0.01
        assert abs(left[1] - right[1]) < 0.01


def test_region_alpha_excludes_target_and_padding():
    exclusion = np.zeros((80, 120), dtype=np.uint8)
    exclusion[30:50, 45:75] = 255
    alpha = guided.build_region_alpha(
        pad_shape=(128, 128, 3), valid_shape=(80, 120, 3),
        region=(20, 10, 80, 60), target_exclusion=exclusion)
    assert tuple(alpha.shape) == (1, 1, 128, 128)
    assert float(alpha[0, 0, 35, 55]) == 0.0
    assert float(alpha[0, 0, 100, 100]) == 0.0
    assert float(alpha.max()) > 0.0


def test_relative_loss_uses_soft_ceiling_and_margin():
    scores = torch.tensor([0.015, 0.010, 0.005, 0.004])
    background = torch.tensor([True, True, False, False])
    usable = torch.tensor([False, False, True, False])
    margin_loss, components = guided.guided_rank_loss(
        scores, background, usable, topm=2, temperature=0.5,
        rank_margin=3.0, score_ceiling=0.02)
    assert margin_loss.item() > 0.0
    assert components['ceiling_loss'].item() == 0.0

    high_scores = scores.clone()
    high_scores[0] = 0.04
    _, high = guided.guided_rank_loss(
        high_scores, background, usable, topm=2, temperature=0.5,
        rank_margin=3.0, score_ceiling=0.02)
    assert high['ceiling_loss'].item() > 0.0


def test_fixed_carrier_survives_later_geometry_mask_changes():
    baseline_scores = torch.tensor([0.01, 0.60, 0.02, 0.03])
    baseline_ious = torch.tensor([0.0, 0.75, 0.0, 0.55])
    region = torch.tensor([True, False, True, False])
    background, carrier, record = guided.select_fixed_objective_masks(
        baseline_scores, baseline_ious, region,
        false_iou_thr=0.1, riou_thr=0.5)
    assert background.tolist() == [True, False, True, False]
    assert carrier.tolist() == [False, True, False, False]
    assert record['carrier_index'] == 1

    later_scores = torch.tensor([0.015, 0.20, 0.010, 0.04])
    margin_loss, components = guided.guided_rank_loss(
        later_scores, background, carrier, topm=2, temperature=0.5,
        rank_margin=3.0, score_ceiling=0.02)
    assert torch.isfinite(margin_loss)
    assert components['usable_candidates'] == 1


def test_model_input_and_replay_preserve_excluded_target_pixels():
    base = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    latent = torch.full((1, 3, 8, 8), 0.5, requires_grad=True)
    alpha = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    alpha[:, :, 8:40, 8:48] = 1.0
    alpha[:, :, 20:32, 24:36] = 0.0
    mean = torch.zeros((1, 3, 1, 1))
    std = torch.ones((1, 3, 1, 1))
    modified, _ = guided.make_guided_input(
        base, latent, alpha, (8, 8, 40, 32), mean, std, 24.0)
    assert torch.equal(modified[:, :, 20:32, 24:36],
                       base[:, :, 20:32, 24:36])
    assert float(
        modified[:, :, 10:18, 10:18].detach().abs().sum()) > 0.0
    modified.sum().backward()
    assert latent.grad is not None
    assert float(latent.grad.abs().sum()) > 0.0

    image = np.full((64, 64, 3), 80, dtype=np.uint8)
    gts = [dict(cx=30.0, cy=26.0, w=12.0, h=8.0, angle=0.0)]
    replay, stats = guided.render_guided_bgr(
        image, latent.detach(), alpha, (8, 8, 40, 32),
        (64, 64, 3), 24.0, gts, target_margin_ratio=0.35)
    assert stats['target_pixels_changed'] == 0
    assert stats['changed_pixels'] > 0
    assert stats['delta_abs_max'] <= 24.0
    target = guided.target_exclusion_mask(
        image.shape, gts, margin_ratio=0.35)
    np.testing.assert_array_equal(replay[target > 0], image[target > 0])


def test_extract_normalization_requires_rgb_test_pipeline():
    cfg = SimpleNamespace(test_pipeline=[dict(
        type='MultiScaleFlipAug', transforms=[dict(
            type='Normalize', mean=[1, 2, 3], std=[4, 5, 6],
            to_rgb=True)])])
    assert guided.extract_normalization(cfg) == ([1, 2, 3], [4, 5, 6])
