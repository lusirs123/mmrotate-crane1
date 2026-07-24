import argparse
import types

import numpy as np
import pytest
import torch

from crane_project.tools import dino_teacher_frozen_region_audit as audit


def _canonical_args(**overrides):
    values = dict(
        config='/tmp/crane_symeood_k1_brightaug.py',
        detector_checkpoint='/tmp/epoch_20.pth',
        source_seq='real_seq07', dinov2_model='dinov2_vitl14',
        dino_height=600, dino_max_long_side=1333,
        patch_size=14, pool_resolution=7,
        legacy_sdpa_query_chunk=512,
        source_folds=5, source_calibration_folds=4,
        neighbors=5, positive_quantile=0.1,
        negative_quantile=0.9, min_fold_votes=4,
        min_roi_in_bounds=0.9,
        max_source_samples=0, riou_thr=0.5, false_iou_thr=0.1,
        source_min_accuracy=0.8, target_min_wins=26,
        target_start=137, target_end=169, seed=0, dino_gpus=None,
        allow_noncanonical=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def _source_summary(value=1.0):
    return dict(
        positive_cosine_accuracy=value,
        negative_cosine_accuracy=value,
        positive_whitened_accuracy=value,
        negative_whitened_accuracy=value)


def _source_collection(coverage=1.0):
    return dict(proposal_coverage=coverage)


def _source_paired_summary(value=1.0, median=0.1):
    return dict(
        count=100,
        cosine_accuracy=value,
        whitened_accuracy=value,
        joint_accuracy=value,
        minimum_fold_joint_accuracy=value,
        cosine_margin=dict(minimum=median, median=median, maximum=median),
        whitened_margin=dict(
            minimum=median, median=median, maximum=median),
        decision_margin=dict(
            minimum=median, median=median, maximum=median))


def _target_summary(wins=26, eligible=31, median=0.1,
                    loo_accuracy=0.8, loo_median=0.05,
                    paired_wins=None):
    criterion = dict(
        win_count=wins,
        decision_margin=dict(median=median),
        leave_one_out_min_accuracy=loo_accuracy,
        leave_one_out_min_median_margin=loo_median)
    paired = dict(criterion)
    paired['win_count'] = wins if paired_wins is None else paired_wins
    return dict(
        eligible_count=eligible, absolute=criterion,
        paired=paired, joint=dict(criterion))


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


def test_legacy_sdpa_matches_explicit_attention():
    torch.manual_seed(0)
    query = torch.randn(1, 2, 4, 3)
    key = torch.randn(1, 2, 4, 3)
    value = torch.randn(1, 2, 4, 5)
    previous = audit._LEGACY_SDPA_QUERY_CHUNK
    audit.configure_legacy_sdpa_query_chunk(2)
    try:
        actual = audit.legacy_scaled_dot_product_attention(query, key, value)
    finally:
        audit.configure_legacy_sdpa_query_chunk(previous)
    weights = torch.softmax(
        torch.matmul(query, key.transpose(-2, -1)) / (3.0 ** 0.5),
        dim=-1)
    expected = torch.matmul(weights, value)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_install_sdpa_compatibility_only_when_missing():
    functional = types.SimpleNamespace()
    assert audit.install_torch_sdpa_compatibility(functional) is True
    assert functional.scaled_dot_product_attention is (
        audit.legacy_scaled_dot_product_attention)
    assert audit.install_torch_sdpa_compatibility(functional) is False


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


def test_resize_normalize_caps_extreme_long_side():
    image = np.zeros((100, 1000, 3), dtype=np.uint8)
    tensor, meta = audit.resize_and_normalize_bgr(
        image, target_height=600, patch_size=14, max_long_side=1333)
    assert meta['resized_shape'] == [133, 1333]
    assert meta['scale'] == pytest.approx(1.333)
    assert tensor.shape[-2:] == (140, 1344)


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


class _FakeShardedDino(torch.nn.Module):
    def __init__(self, blocks=6):
        super().__init__()
        self.patch_embed = torch.nn.Conv2d(3, 4, 1)
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4) for _ in range(blocks)])
        self.norm = torch.nn.LayerNorm(4)


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
    assert model._sym_annotation_compatibility['operation'] == 'not_needed'


def test_load_frozen_dinov2_auto_patches_py38_annotations_and_retries(
        tmp_path, monkeypatch):
    repo = tmp_path / 'dinov2'
    package = repo / 'dinov2'
    package.mkdir(parents=True)
    (repo / 'hubconf.py').write_text('# fake local hub\n')
    attention = package / 'attention.py'
    attention.write_text(
        'class Attention:\n    value: float | None = None\n')
    checkpoint = tmp_path / 'dinov2_vitl14_pretrain.pth'
    torch.save(_FakeHubDino().state_dict(), checkpoint)
    calls = []

    def load_once_then_succeed(*_args, **_kwargs):
        calls.append(len(calls))
        if len(calls) == 1:
            raise TypeError(
                "unsupported operand type(s) for |: 'type' and 'NoneType'")
        return _FakeHubDino()

    monkeypatch.setattr(audit, '_load_local_dinov2',
                        load_once_then_succeed)
    monkeypatch.setattr(audit, '_legacy_annotation_error',
                        lambda _error: True)
    model, patch_size = audit.load_frozen_dinov2(
        str(repo), str(checkpoint), 'dinov2_vitl14',
        torch.device('cpu'))
    assert len(calls) == 2
    assert patch_size == 14
    assert audit.py38_patcher.FUTURE_IMPORT in attention.read_text()
    assert model._sym_annotation_compatibility['attempted'] is True
    assert model._sym_annotation_compatibility['changed_count'] == 1


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


def test_shard_frozen_dinov2_records_contiguous_block_partition():
    model = _FakeShardedDino(blocks=6)
    metadata = audit.shard_frozen_dinov2(
        model, [torch.device('cpu'), torch.device('cpu')])
    assert metadata['block_count'] == 6
    assert metadata['blocks_per_device'] == {'cpu': 6}
    assert len(model._sym_dino_device_hooks) == 6


def test_contiguous_device_indices_balance_vitl_blocks():
    assert audit.contiguous_device_indices(24, 2) == [0] * 12 + [1] * 12
    assert audit.contiguous_device_indices(24, 3) == (
        [0] * 8 + [1] * 8 + [2] * 8)


def test_validate_args_rejects_duplicate_dino_gpus():
    with pytest.raises(ValueError, match='duplicates'):
        audit.validate_args(_canonical_args(dino_gpus=[1, 1]))


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


def test_candidate_selection_skips_higher_score_out_of_bounds_roi():
    feature = torch.ones(1, 4, 8, 8)
    boxes = torch.tensor([
        [0.0, 0.0, 28.0, 28.0, 0.0],
        [56.0, 56.0, 28.0, 28.0, 0.0]])
    scores = torch.tensor([0.9, 0.8])
    ious = torch.tensor([0.0, 0.0])
    layout = [
        dict(level=0, row=0, col=0, anchor_id=0),
        dict(level=0, row=1, col=1, anchor_id=0)]
    selected = audit.select_valid_dino_candidate(
        boxes, scores, ious, layout, 0,
        {'scale_factor': [1.0, 1.0, 1.0, 1.0]}, {'scale': 1.0},
        feature, patch_size=14, output_size=7, min_in_bounds=0.9,
        max_iou=0.1)
    assert selected['index'] == 1
    assert selected['in_bounds_fraction'] == pytest.approx(1.0)
    assert selected['score_rank_before_roi_filter'] == 2
    assert selected['rejected_higher_score_count'] == 1


def _source_sample(frame, gt_index=0):
    offset = float(frame) * 0.001 + float(gt_index) * 0.0001
    return dict(
        row=dict(seq='real_seq07', frame=frame, gt_index=gt_index),
        positive_vector=torch.tensor([1.0, offset]),
        negative_vector=torch.tensor([-1.0, offset]))


def test_grouped_source_folds_keep_all_gt_from_a_frame_together():
    samples = [
        _source_sample(frame, gt_index)
        for frame in range(10) for gt_index in range(2)]
    fold_ids = audit.grouped_source_fold_ids(samples, folds=5)
    for frame in range(10):
        frame_folds = {
            fold_id for sample, fold_id in zip(samples, fold_ids)
            if sample['row']['frame'] == frame}
        assert len(frame_folds) == 1


def test_source_three_way_split_has_no_frame_overlap():
    samples = [
        _source_sample(frame, gt_index)
        for frame in range(10) for gt_index in range(2)]
    fold_ids = audit.grouped_source_fold_ids(samples, folds=5)
    split = audit.split_source_fold(
        samples, fold_ids, fold_id=0, calibration_folds=4)
    groups = split['group_sets']
    assert groups['bank'].isdisjoint(groups['calibration'])
    assert groups['bank'].isdisjoint(groups['validation'])
    assert groups['calibration'].isdisjoint(groups['validation'])
    assert len(groups['validation']) == 2
    for group_samples in (
            split['bank'], split['calibration'], split['validation']):
        counts = {}
        for sample in group_samples:
            frame = sample['row']['frame']
            counts[frame] = counts.get(frame, 0) + 1
        assert all(count == 2 for count in counts.values())


def test_source_ensemble_reports_independent_calibration_validation():
    samples = [_source_sample(frame) for frame in range(20)]
    models, _zero, calibrated, paired, metadata = audit.build_source_ensemble(
        samples, folds=5, calibration_folds=4, neighbors=1,
        positive_quantile=0.1, negative_quantile=0.9)
    assert len(models) == 5
    assert calibrated['count'] == 20
    assert paired['count'] == 20
    assert paired['joint_accuracy'] == 1.0
    assert paired['minimum_fold_joint_accuracy'] == 1.0
    assert metadata['split_unit'] == 'seq_frame'
    assert metadata['calibration_and_validation_disjoint'] is True
    assert all(not item['group_overlap']
               for item in metadata['fold_sizes'])
    assert all(item['calibration_count'] > 0
               and item['validation_count'] > 0
               for item in metadata['fold_sizes'])


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
    assert result['absolute_rescued'] is True
    assert result['paired_rescued'] is True
    assert result['joint_rescued'] is True
    assert result['absolute_fold_votes'] == 5
    assert result['paired_fold_votes'] == 5
    assert result['mean_absolute_decision_margin'] > 0.0
    assert result['mean_paired_decision_margin'] > 0.0


@pytest.mark.parametrize(
    'source,paired,target,misses,expected', [
        (_source_summary(), _source_paired_summary(),
         _target_summary(), [164, 167],
         'AUTHORIZE_DINO_TEACHER_SOURCE_LABELLER'),
        (_source_summary(0.7), _source_paired_summary(0.7),
         _target_summary(), [164, 167],
         'DINO_SOURCE_CONTROL_INCONCLUSIVE'),
        (_source_summary(), _source_paired_summary(),
         _target_summary(wins=25, paired_wins=25), [164, 167],
         'DINO_TEACHER_FEATURES_INSUFFICIENT'),
        (_source_summary(), _source_paired_summary(),
         _target_summary(), [164], 'AUDIT_INVALID'),
    ])
def test_gate_decisions(source, paired, target, misses, expected):
    gate = audit.make_gate(
        source, source, paired, _source_collection(), target, misses, [],
        _canonical_args())
    assert gate['decision'] == expected


def test_gate_rejects_invalid_calibrated_source_control():
    gate = audit.make_gate(
        _source_summary(), _source_summary(0.7),
        _source_paired_summary(0.7), _source_collection(),
        _target_summary(),
        [164, 167], [], _canonical_args())
    assert gate['decision'] == 'DINO_SOURCE_CONTROL_INCONCLUSIVE'
    assert gate['source_zero_margin_valid'] is True
    assert gate['source_calibrated_valid'] is False


def test_gate_treats_zero_margin_source_control_as_diagnostic_only():
    gate = audit.make_gate(
        _source_summary(0.7), _source_summary(), _source_paired_summary(),
        _source_collection(),
        _target_summary(),
        [164, 167], [], _canonical_args())
    assert gate['decision'] == 'AUTHORIZE_DINO_TEACHER_SOURCE_LABELLER'
    assert gate['source_zero_margin_valid'] is False
    assert gate['source_valid'] is True


def test_gate_authorizes_only_roi_head_when_paired_transfer_passes():
    gate = audit.make_gate(
        _source_summary(), _source_summary(), _source_paired_summary(),
        _source_collection(),
        _target_summary(wins=25, paired_wins=26),
        [164, 167], [], _canonical_args())
    assert gate['decision'] == 'AUTHORIZE_SOURCE_ONLY_DINO_ROI_HEAD'


def test_gate_uses_paired_source_control_when_absolute_control_fails():
    gate = audit.make_gate(
        _source_summary(), _source_summary(0.7), _source_paired_summary(),
        _source_collection(), _target_summary(wins=0, paired_wins=31),
        [164, 167], [], _canonical_args())
    assert gate['decision'] == 'AUTHORIZE_SOURCE_ONLY_DINO_ROI_HEAD'
    assert gate['source_absolute_valid'] is False
    assert gate['source_paired_valid'] is True


def test_gate_rejects_any_target_roi_invalidity():
    gate = audit.make_gate(
        _source_summary(), _source_summary(), _source_paired_summary(),
        _source_collection(),
        _target_summary(),
        [164, 167], [155], _canonical_args())
    assert gate['decision'] == 'AUDIT_INVALID'


def test_gate_rejects_biased_low_source_proposal_coverage():
    gate = audit.make_gate(
        _source_summary(), _source_summary(), _source_paired_summary(),
        _source_collection(0.79),
        _target_summary(), [164, 167], [], _canonical_args())
    assert gate['decision'] == 'DINO_SOURCE_CONTROL_INCONCLUSIVE'
    assert gate['source_proposal_coverage_valid'] is False
