from types import SimpleNamespace

import pytest
import torch

from crane_project.tools import frozen_p3_objectness_transfer_probe as probe


def _args(**overrides):
    values = dict(
        seed=0,
        source_indexes=[0, 1],
        feature_level=0,
        epochs=4,
        lr=1e-3,
        weight_decay=1e-4,
        positive_weight=20.0,
        gaussian_sigma_scale=0.25,
        gaussian_min_sigma_strides=1.0,
        max_train_samples_per_source=0,
        max_val_samples=0,
        riou_thr=0.5,
        false_iou_thr=0.1,
        source_val_min_accuracy=0.8,
        target_min_wins=26,
        target_start=137,
        target_end=169,
        config='crane_project/configs/crane_symeood_k1_brightaug.py',
        checkpoint='work_dirs/crane_symeood_k1_brightaug/epoch_20.pth',
        allow_noncanonical=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_canonical_protocol_locks_epoch20_p3_and_full_data():
    assert probe.validate_args(_args()) is True
    with pytest.raises(ValueError, match='Canonical probe protocol mismatch'):
        probe.validate_args(_args(checkpoint='work/epoch_24.pth'))
    with pytest.raises(ValueError, match='Canonical probe protocol mismatch'):
        probe.validate_args(_args(max_val_samples=4))
    assert probe.validate_args(_args(
        max_val_samples=4, allow_noncanonical=True)) is False


def test_oriented_gaussian_follows_obb_axes_and_normalizes_peak():
    horizontal = torch.tensor([[32.0, 32.0, 48.0, 16.0, 0.0]])
    vertical = horizontal.clone()
    vertical[:, 4] = torch.pi / 2
    mask = torch.ones((1, 8, 8), dtype=torch.bool)
    heat_h = probe.oriented_gaussian_heatmap(
        horizontal, 8, 8, stride=8.0, sigma_scale=0.25,
        min_sigma_strides=0.25, valid_mask=mask)
    heat_v = probe.oriented_gaussian_heatmap(
        vertical, 8, 8, stride=8.0, sigma_scale=0.25,
        min_sigma_strides=0.25, valid_mask=mask)

    assert heat_h.max().item() == pytest.approx(1.0)
    assert heat_v.max().item() == pytest.approx(1.0)
    # Around the peak, horizontal boxes decay slower along columns; rotating
    # the OBB by 90 degrees swaps that relation.
    assert heat_h[0, 3, 5] > heat_h[0, 5, 3]
    assert heat_v[0, 3, 5] < heat_v[0, 5, 3]


def test_valid_grid_mask_and_weighted_loss_exclude_padding():
    mask = probe.valid_grid_mask(
        height=4, width=4, img_shape=(16, 24, 3),
        stride=8.0, device=torch.device('cpu'))
    assert mask.shape == (1, 4, 4)
    assert mask.sum().item() == 6

    logits = torch.zeros((1, 1, 4, 4), requires_grad=True)
    target = torch.zeros((1, 4, 4))
    target[0, 0, 0] = 1.0
    loss = probe.weighted_heatmap_bce(
        logits, target, mask, positive_weight=10.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0, 0, 3, 3].item() == 0.0


def test_candidate_pairing_is_level0_and_uses_highest_main_score():
    scores = torch.tensor([0.40, 0.90, 0.70, 0.95, 0.80])
    ious = torch.tensor([0.70, 0.05, 0.02, 0.00, 0.60])
    layout = [
        dict(level=0, row=0, col=0, anchor_id=2),
        dict(level=0, row=0, col=1, anchor_id=0),
        dict(level=0, row=0, col=2, anchor_id=1),
        dict(level=1, row=0, col=0, anchor_id=0),
        dict(level=1, row=0, col=1, anchor_id=0),
    ]
    usable = probe.select_level_candidate(
        scores, ious, layout, level=0, min_iou=0.5)
    false = probe.select_level_candidate(
        scores, ious, layout, level=0, max_iou=0.1)
    assert usable == 0
    assert false == 1
    assert layout[false]['level'] == 0


def _target_rows(wins):
    rows = []
    eligible_frames = [
        frame for frame in range(137, 170) if frame not in (164, 167)]
    for index, frame in enumerate(eligible_frames):
        margin = 1.0 if index < wins else -0.25
        rows.append(dict(
            frame=frame, eligible=True, margin=margin,
            positive_logit=margin, negative_logit=0.0))
    rows.extend([
        dict(frame=164, eligible=False, margin=None),
        dict(frame=167, eligible=False, margin=None),
    ])
    return rows


def test_target_gate_accepts_26_of_31_with_leave_one_out_robustness():
    source = dict(accuracy=0.8, heatmap_degenerate_images=0)
    result = probe.target_gate(
        _target_rows(26), source, required_eligible=31,
        min_wins=26, source_min_accuracy=0.8)
    assert result['decision'] == 'GO_SUPPORTS_A'
    assert result['checks']['single_frame_robust'] is True
    assert result['leave_one_out_min_accuracy'] == pytest.approx(25 / 30)


def test_target_gate_failure_does_not_claim_explanation_b():
    source = dict(accuracy=0.8, heatmap_degenerate_images=0)
    result = probe.target_gate(
        _target_rows(25), source, required_eligible=31,
        min_wins=26, source_min_accuracy=0.8)
    assert result['decision'] == 'STOP_A_EVIDENCE_INSUFFICIENT'
    assert 'does not prove explanation B' in result['interpretation']
