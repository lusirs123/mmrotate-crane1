"""Checks for the bounded corrected-mask source probe."""

import math

import pytest
import torch

from crane_project.tools import probe_k1_dino_corrected_mask_source_v4 as probe


def test_gradient_comparison_reports_norm_ratio_and_direction():
    det = [torch.tensor([3.0, 4.0]), None]
    dist = [torch.tensor([0.0, 2.0]), torch.tensor([0.0])]
    result = probe.gradient_comparison(det, dist)
    assert result['detection_l2'] == 5.0
    assert result['distillation_l2'] == 2.0
    assert result['distillation_to_detection_ratio'] == 0.4
    assert result['cosine'] == pytest.approx(0.8)


def test_gradient_comparison_rejects_zero_or_nonfinite_gradients():
    with pytest.raises(ValueError, match='gradient is zero'):
        probe.gradient_comparison([torch.ones(1)], [None])
    with pytest.raises(ValueError, match='Nonfinite'):
        probe.gradient_comparison([torch.tensor([math.nan])],
                                  [torch.ones(1)])


def test_corrected_teacher_mask_keeps_single_odd_source_cell():
    mask = torch.zeros((1, 1, 4, 4))
    mask[0, 0, 1, 1] = 1
    tokens = probe._foreground_teacher_tokens(mask, (2, 2))
    assert tokens.sum().item() == 1
    assert tokens[0, 0, 0].item() is True
