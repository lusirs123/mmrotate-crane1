from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from crane_project.tools import frozen_p3_feature_alignment_audit as audit


def _args(**overrides):
    values = dict(
        seed=0,
        feature_level=0,
        max_source_samples=0,
        source_control_modulus=5,
        riou_thr=0.5,
        false_iou_thr=0.1,
        reconstruction_atol=1e-4,
        source_min_accuracy=0.8,
        target_min_count=26,
        min_size_matched_source=10,
        target_start=137,
        target_end=169,
        source_seq='real_seq07',
        config='crane_project/configs/crane_symeood_k1_brightaug.py',
        detector_checkpoint='work_dirs/crane_symeood_k1_brightaug/epoch_20.pth',
        probe_checkpoint=(
            'work_dirs/frozen_p3_objectness_transfer_epoch20/'
            'probe_best_source_only.pth'),
        allow_noncanonical=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_canonical_protocol_locks_existing_detector_and_probe():
    assert audit.validate_args(_args()) is True
    with pytest.raises(ValueError, match='Canonical feature-audit'):
        audit.validate_args(_args(detector_checkpoint='work/epoch_24.pth'))
    with pytest.raises(ValueError, match='Canonical feature-audit'):
        audit.validate_args(_args(max_source_samples=10))
    assert audit.validate_args(_args(
        max_source_samples=10, allow_noncanonical=True)) is False


def test_boundary_patch_exactly_reconstructs_padding_conv_logit():
    feature = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    conv = nn.Conv2d(2, 1, 3, padding=1)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(18).reshape(1, 2, 3, 3) / 10)
        conv.bias.fill_(0.25)
        logits = conv(feature)
    patch = audit.extract_conv_patch(feature, row=0, col=0)
    result = audit.exact_patch_audit(
        patch, conv, float(logits[0, 0, 0, 0]), atol=1e-6)
    assert patch.shape == (2, 3, 3)
    assert result['reconstruction_abs_error'] < 1e-6
    assert result['reconstructed_logit'] == pytest.approx(
        logits[0, 0, 0, 0].item())


def test_source_prototypes_separate_positive_and_negative_directions():
    positive = torch.tensor([
        [2.0, 1.0, 0.2],
        [1.8, 1.2, 0.1],
        [2.2, 0.8, 0.3],
    ])
    negative = torch.tensor([
        [-2.0, -1.0, -0.2],
        [-1.8, -1.2, -0.1],
        [-2.2, -0.8, -0.3],
    ])
    prototypes = audit.build_source_prototypes(positive, negative)
    positive_score = audit.prototype_scores(
        torch.tensor([2.1, 1.0, 0.2]), prototypes)
    negative_score = audit.prototype_scores(
        torch.tensor([-2.1, -1.0, -0.2]), prototypes)
    assert positive_score['raw_preference_positive'] > 0
    assert positive_score['cosine_preference_positive'] > 0
    assert positive_score['whitened_preference_positive'] > 0
    assert negative_score['raw_preference_positive'] < 0
    assert negative_score['cosine_preference_positive'] < 0
    assert negative_score['whitened_preference_positive'] < 0


def _strong_source_control():
    return dict(
        positive_cosine_accuracy=0.95,
        negative_cosine_accuracy=0.95,
        positive_whitened_accuracy=0.90,
        negative_whitened_accuracy=0.90)


def _strong_target_summary():
    return dict(
        eligible_count=31,
        usable_cosine_negative_like=28,
        cosine_pair_inverted=27,
        usable_whitened_negative_like=29,
        whitened_pair_inverted=26)


def test_gate_requires_independent_normalized_inversion_and_controls():
    result = audit.make_gate(
        _strong_source_control(), _strong_target_summary(),
        dict(count=22, probe_paired_accuracy=1.0),
        geometry_misses=[164, 167], max_reconstruction_error=1e-6,
        args=_args())
    assert result['decision'] == 'B_STRONGLY_SUPPORTED'
    assert all(result['checks'].values())


def test_gate_does_not_confirm_b_from_probe_failure_alone():
    target = _strong_target_summary()
    target['cosine_pair_inverted'] = 10
    result = audit.make_gate(
        _strong_source_control(), target,
        dict(count=22, probe_paired_accuracy=1.0),
        geometry_misses=[164, 167], max_reconstruction_error=1e-6,
        args=_args())
    assert result['decision'] == 'B_NOT_CONFIRMED'
    assert result['checks']['target_cosine_pair_inverted'] is False


def test_size_matched_control_uses_target_range_without_training():
    source = [
        dict(frame=1, gt_long_side=70.0, gt_short_side=35.0,
             paired_win=True),
        dict(frame=2, gt_long_side=90.0, gt_short_side=35.0,
             paired_win=False),
    ]
    target = [
        dict(eligible=True, gt_long_side=65.0, gt_short_side=30.0),
        dict(eligible=True, gt_long_side=80.0, gt_short_side=40.0),
    ]
    result = audit.size_matched_source_summary(source, target)
    assert result['count'] == 1
    assert result['probe_paired_accuracy'] == 1.0
    assert result['frames'] == [1]
