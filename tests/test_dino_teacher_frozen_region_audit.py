import argparse

import numpy as np
import pytest
import torch

from crane_project.tools import dino_teacher_frozen_region_audit as audit


def _canonical_args(**overrides):
    values = dict(
        config='/tmp/crane_symeood_k1_brightaug.py',
        detector_checkpoint='/tmp/epoch_20.pth',
        source_seq='real_seq07', dinov2_model='dinov2_vitl14',
        dino_height=600, patch_size=14, pool_resolution=7,
        source_folds=5, neighbors=5, positive_quantile=0.1,
        negative_quantile=0.9, min_fold_votes=4,
        max_source_samples=0, riou_thr=0.5, false_iou_thr=0.1,
        source_min_accuracy=0.8, target_min_wins=26,
        target_start=137, target_end=169, seed=0,
        allow_noncanonical=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def _source_summary(value=1.0):
    return dict(
        positive_cosine_accuracy=value,
        negative_cosine_accuracy=value,
        positive_whitened_accuracy=value,
        negative_whitened_accuracy=value)


def _target_summary(wins=26, eligible=31, median=0.1,
                    loo_accuracy=0.8, loo_median=0.05):
    return dict(
        eligible_count=eligible, win_count=wins,
        decision_margin=dict(median=median),
        leave_one_out_min_accuracy=loo_accuracy,
        leave_one_out_min_median_margin=loo_median)


def test_validate_args_accepts_canonical_protocol():
    assert audit.validate_args(_canonical_args()) is True


def test_validate_args_requires_explicit_noncanonical_override():
    with pytest.raises(ValueError, match='dino_height'):
        audit.validate_args(_canonical_args(dino_height=588))
    args = _canonical_args(dino_height=588, allow_noncanonical=True)
    assert audit.validate_args(args) is False


def test_unwrap_state_dict_strips_common_teacher_prefixes():
    tensor = torch.ones(2)
    state = audit._unwrap_state_dict({
        'teacher': {'module.backbone.encoder.weight': tensor}})
    assert list(state) == ['weight']
    assert torch.equal(state['weight'], tensor)


def test_resize_normalize_preserves_aspect_and_pads_to_patch_grid():
    image = np.zeros((20, 41, 3), dtype=np.uint8)
    tensor, meta = audit.resize_and_normalize_bgr(image, 28, 14)
    assert tensor.shape == (1, 3, 28, 70)
    assert meta['ori_shape'] == [20, 41]
    assert meta['resized_shape'] == [28, 57]
    assert meta['padded_shape'] == [28, 70]
    assert meta['scale'] == pytest.approx(1.4)


def test_resize_normalize_uses_short_side_for_portrait_images():
    image = np.zeros((41, 20, 3), dtype=np.uint8)
    tensor, meta = audit.resize_and_normalize_bgr(image, 28, 14)
    assert tensor.shape == (1, 3, 70, 28)
    assert meta['resized_shape'] == [57, 28]
    assert meta['padded_shape'] == [70, 28]
    assert meta['scale'] == pytest.approx(1.4)


class _FakeDino:
    def get_intermediate_layers(self, tensor, n=1):
        batch = tensor.shape[0]
        tokens = torch.arange(
            batch * 8 * 3, dtype=tensor.dtype,
            device=tensor.device).reshape(batch, 8, 3)
        return (tokens,)


class _FakeHubDino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.patch_embed = argparse.Namespace(patch_size=torch.Size([14, 14]))


def test_load_frozen_dinov2_strictly_loads_and_freezes(
        tmp_path, monkeypatch):
    repo = tmp_path / 'dinov2'
    repo.mkdir()
    (repo / 'hubconf.py').write_text('# fake local hub\n')
    checkpoint = tmp_path / 'dinov2_vitl14_pretrain.pth'
    torch.save(_FakeHubDino().state_dict(), checkpoint)
    monkeypatch.setattr(
        torch.hub, 'load', lambda *args, **kwargs: _FakeHubDino())
    model, patch_size = audit.load_frozen_dinov2(
        str(repo), str(checkpoint), 'dinov2_vitl14',
        torch.device('cpu'))
    assert patch_size == 14
    assert model.training is False
    assert all(not parameter.requires_grad
               for parameter in model.parameters())


def test_load_frozen_dinov2_rejects_weight_mismatch(
        tmp_path, monkeypatch):
    repo = tmp_path / 'dinov2'
    repo.mkdir()
    (repo / 'hubconf.py').write_text('# fake local hub\n')
    checkpoint = tmp_path / 'wrong.pth'
    torch.save({'other': torch.ones(2)}, checkpoint)
    monkeypatch.setattr(
        torch.hub, 'load', lambda *args, **kwargs: _FakeHubDino())
    with pytest.raises(RuntimeError, match='checkpoint/model mismatch'):
        audit.load_frozen_dinov2(
            str(repo), str(checkpoint), 'dinov2_vitl14',
            torch.device('cpu'))


def test_extract_patch_grid_reconstructs_spatial_token_map():
    tensor = torch.zeros(1, 3, 4, 8)
    feature = audit.extract_patch_grid(_FakeDino(), tensor, patch_size=2)
    assert feature.shape == (1, 3, 2, 4)
    assert torch.equal(feature.flatten(2).transpose(1, 2),
                       _FakeDino().get_intermediate_layers(tensor)[0])


def test_detector_box_mapping_undoes_detector_scale_then_applies_dino_scale():
    box = torch.tensor([20.0, 40.0, 8.0, 12.0, 0.25])
    mapped = audit.detector_box_to_dino(
        box, {'scale_factor': [2.0, 2.0, 2.0, 2.0]},
        {'scale': 0.5})
    assert torch.allclose(
        mapped, torch.tensor([5.0, 10.0, 2.0, 3.0, 0.25]))


def test_detector_box_mapping_rejects_anisotropic_detector_resize():
    with pytest.raises(RuntimeError, match='isotropic'):
        audit.detector_box_to_dino(
            [10, 10, 4, 4, 0],
            {'scale_factor': [2.0, 3.0, 2.0, 3.0]},
            {'scale': 1.0})


def test_oriented_roi_pool_is_constant_on_constant_feature_map():
    feature = torch.ones(1, 4, 8, 8)
    vector, metadata = audit.oriented_roi_vector(
        feature, torch.tensor([56.0, 56.0, 28.0, 14.0, 0.4]),
        patch_size=14, output_size=7)
    assert torch.allclose(vector, torch.ones(4), atol=1e-6)
    assert metadata['in_bounds_fraction'] == pytest.approx(1.0)
    assert metadata['sampled_points'] == 49


def _fold_models():
    positive = torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.8, 0.2],
        [1.0, 0.1], [0.9, -0.1]])
    negative = -positive
    models = []
    for fold_id in range(5):
        models.append(dict(
            fold_id=fold_id,
            bank=audit.multimodal.build_knn_bank(
                positive, negative, neighbors=3),
            cosine_preference_threshold=0.0,
            whitened_preference_threshold=0.0))
    return models


def test_score_region_requires_source_calibration_and_false_separation():
    result = audit.score_region(
        torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]),
        _fold_models(), min_fold_votes=4)
    assert result['rescued'] is True
    assert result['fold_votes'] == 5
    assert result['mean_decision_margin'] > 0.0


@pytest.mark.parametrize(
    'source,target,misses,expected', [
        (_source_summary(), _target_summary(), [164, 167],
         'AUTHORIZE_DINO_TEACHER_SOURCE_LABELLER'),
        (_source_summary(0.7), _target_summary(), [164, 167],
         'DINO_SOURCE_CONTROL_INCONCLUSIVE'),
        (_source_summary(), _target_summary(wins=25), [164, 167],
         'DINO_TEACHER_FEATURES_INSUFFICIENT'),
        (_source_summary(), _target_summary(), [164], 'AUDIT_INVALID'),
    ])
def test_gate_decisions(source, target, misses, expected):
    gate = audit.make_gate(
        source, source, target, misses, _canonical_args())
    assert gate['decision'] == expected


def test_gate_rejects_invalid_calibrated_source_control():
    gate = audit.make_gate(
        _source_summary(), _source_summary(0.7), _target_summary(),
        [164, 167], _canonical_args())
    assert gate['decision'] == 'DINO_SOURCE_CONTROL_INCONCLUSIVE'
    assert gate['source_zero_margin_valid'] is True
    assert gate['source_calibrated_valid'] is False
