import argparse
import json

import numpy as np
import pytest
import torch

from crane_project.tools import dino_teacher_rotated_labeller as labeller


def _args(tmp_path, **overrides):
    checkpoint = tmp_path / 'dino.pth'
    checkpoint.write_bytes(b'dino')
    values = dict(
        seed=0, source_val_modulus=5, dino_gpus=[1, 2], head_gpu=0,
        patch_size=14, rpn_feat_channels=256, roi_fc_channels=1024,
        roi_samples=256, proposal_count=2000, max_detections=2000,
        epochs=12, lr=0.001, momentum=0.9, weight_decay=1e-4,
        max_grad_norm=10.0, riou_thr=0.5, target_min_wins=26,
        max_mcml=5, resume_checkpoint=None, eval_only_checkpoint=None,
        dinov2_checkpoint=str(checkpoint), dinov2_model='dinov2_vitl14',
        dino_height=600, dino_max_long_side=1333,
        feature_cache_dir=str(tmp_path / 'cache'),
        target_start=137, target_end=169)
    values.update(overrides)
    return argparse.Namespace(**values)


def _record(tmp_path, frame):
    image = tmp_path / 'real_seq07_{:05d}.jpg'.format(frame)
    image.write_bytes('image-{}'.format(frame).encode('ascii'))
    return dict(
        split='train', seq='real_seq07', frame=frame,
        image=str(image), annotation=str(tmp_path / 'unused.txt'))


def test_validate_requires_disjoint_head_and_dino_gpus(tmp_path):
    with pytest.raises(ValueError, match='separate'):
        labeller.validate_args(_args(tmp_path, head_gpu=1))


def test_source_split_is_deterministic_and_disjoint(tmp_path):
    records = [_record(tmp_path, frame) for frame in range(1, 11)]
    train, val = labeller.split_source_records(records, modulus=5)
    assert [row['frame'] for row in val] == [5, 10]
    assert set(row['frame'] for row in train).isdisjoint(
        row['frame'] for row in val)


def test_target_image_cannot_enter_source_training(tmp_path):
    source = [_record(tmp_path, 1)]
    target = [dict(source[0], split='test', seq='real_seq02', frame=137)]
    with pytest.raises(RuntimeError, match='leaked'):
        labeller.assert_training_target_isolation(source, target)


def test_rpn_config_uses_single_dino_stride_and_canonical_sizes(tmp_path):
    args = _args(tmp_path)
    config = labeller.rpn_config(1024, args)
    anchors = config['anchor_generator']
    assert config['type'] == 'OrientedRPNHead'
    assert anchors['strides'] == [14]
    assert np.asarray(anchors['scales']) * 14 == pytest.approx(
        [32, 64, 128, 256, 512])
    assert len(anchors['scales']) * len(anchors['ratios']) == 15


def test_roi_config_is_paper_style_two_fc_with_obb_regression(tmp_path):
    config = labeller.roi_config(1024, _args(tmp_path))
    assert config['type'] == 'OrientedStandardRoIHead'
    assert config['bbox_roi_extractor']['roi_layer']['out_size'] == 7
    assert config['bbox_roi_extractor']['featmap_strides'] == [14]
    assert config['bbox_head']['type'] == 'RotatedShared2FCBBoxHead'
    assert config['bbox_head']['num_classes'] == 1
    assert config['bbox_head']['reg_class_agnostic'] is True


def test_scaled_gt_preserves_angle_and_scales_first_four_values(monkeypatch):
    monkeypatch.setattr(
        labeller, 'parse_original_gt',
        lambda _path: np.asarray([[1, 2, 3, 4, 0.25]], dtype=np.float32))
    boxes, labels, original = labeller.scaled_gt_tensors(
        'unused', 2.0, torch.device('cpu'))
    assert boxes.cpu().numpy() == pytest.approx(
        np.asarray([[2, 4, 6, 8, 0.25]], dtype=np.float32))
    assert labels.tolist() == [0]
    assert original == pytest.approx(
        np.asarray([[1, 2, 3, 4, 0.25]], dtype=np.float32))


def test_loss_total_reduces_rpn_lists_and_roi_tensors():
    total = labeller.loss_total(dict(
        loss_rpn_cls=[torch.tensor([1.0, 3.0])],
        loss_rpn_bbox=[torch.tensor(2.0)],
        loss_cls=torch.tensor(4.0),
        loss_bbox=torch.tensor(5.0)))
    assert float(total.item()) == pytest.approx(13.0)


def test_source_selection_prioritizes_top1_before_oracle_recall():
    a = dict(top1_hits=5, recall_at_20=5,
             recall_at_100=5, mean_top1_riou=0.5)
    b = dict(top1_hits=4, recall_at_20=20,
             recall_at_100=20, mean_top1_riou=0.9)
    assert labeller.source_selection_key(a) > labeller.source_selection_key(b)


def test_target_decision_requires_top1_and_mcml(tmp_path):
    args = _args(tmp_path)
    summary = dict(frame_count=33, top1_hits=26, top1_mcml=5,
                   recall_at_100=26)
    assert labeller.make_target_decision(summary, args) == (
        'FROZEN_DINO_ROTATED_LABELLER_RESTORES_ORDERING')
    summary['top1_hits'] = 3
    summary['top1_mcml'] = 16
    assert labeller.make_target_decision(summary, args) == (
        'DINO_LABELLER_GEOMETRY_ONLY_RANKING_INSUFFICIENT')


def test_checkpoint_rejects_non_source_payload(tmp_path):
    args = _args(tmp_path)
    payload = dict(
        source_only=False, frozen_dinov2=True, in_channels=1024,
        patch_size=14, rpn_feat_channels=256, roi_fc_channels=1024,
        heads_state_dict={})
    with pytest.raises(RuntimeError, match='source-only'):
        labeller.validate_checkpoint(payload, 1024, args)


def test_cache_signature_changes_with_dino_checkpoint(tmp_path):
    record = _record(tmp_path, 1)
    first_args = _args(tmp_path)
    second_checkpoint = tmp_path / 'dino_other.pth'
    second_checkpoint.write_bytes(b'other')
    second_args = _args(tmp_path, dinov2_checkpoint=str(second_checkpoint))
    first = json.dumps(
        labeller.cache_signature(record, first_args), sort_keys=True)
    second = json.dumps(
        labeller.cache_signature(record, second_args), sort_keys=True)
    assert first != second
