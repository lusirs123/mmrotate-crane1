import argparse

import pytest
import torch

from crane_project.tools import (
    dino_teacher_token_scale_rpn_coverage_audit as audit)


def _args(tmp_path, **overrides):
    labeller_checkpoint = tmp_path / 'labeller.pth'
    dino_checkpoint = tmp_path / 'dino.pth'
    labeller_checkpoint.write_bytes(b'labeller')
    dino_checkpoint.write_bytes(b'dino')
    values = dict(
        seed=0, source_scale_datasets=['train:train', 'train_sim:train'],
        source_rpn_datasets=['val:val'], target_slices=None,
        dino_gpus=[1, 2], head_gpu=0,
        patch_size=14, rpn_feat_channels=256, roi_fc_channels=1024,
        roi_samples=256, proposal_count=2000, max_detections=2000,
        dino_height=600, dino_max_long_side=1333, source_rpn_limit=0,
        recall_ks=[20, 100, 2000],
        anchor_iou_thresholds=[0.3, 0.5, 0.7], riou_thr=0.5,
        labeller_checkpoint=str(labeller_checkpoint),
        dinov2_checkpoint=str(dino_checkpoint),
        out_json=str(tmp_path / 'result.json'))
    values.update(overrides)
    return argparse.Namespace(**values)


def test_parse_target_slice():
    assert audit.parse_target_slice('far:test:real_seq02:2:41') == dict(
        name='far', split='test', seq='real_seq02', start=2, end=41)
    with pytest.raises(ValueError, match='START'):
        audit.parse_target_slice('far:test:real_seq02:41:2')


def test_validate_sets_default_slices_and_refuses_overlap_gpu(tmp_path):
    args = _args(tmp_path)
    audit.validate_args(args)
    assert [row['name'] for row in args.parsed_target_slices] == [
        'seq02_far', 'seq02_dark', 'seq03_small']
    with pytest.raises(ValueError, match='separate'):
        audit.validate_args(_args(tmp_path, head_gpu=1))


def test_validate_refuses_recall_beyond_proposal_count(tmp_path):
    with pytest.raises(ValueError, match='proposal-count'):
        audit.validate_args(_args(tmp_path, recall_ks=[2001]))


def test_token_objects_use_actual_uniform_resize_scale():
    boxes = torch.tensor([[0.0, 0.0, 42.0, 21.0, 0.25]]).numpy()
    rows = audit.token_objects(
        boxes, dict(scale=0.5), patch_size=14)
    assert rows[0]['short_token'] == pytest.approx(0.75)
    assert rows[0]['long_token'] == pytest.approx(1.5)
    assert rows[0]['aspect_ratio'] == pytest.approx(2.0)


def test_pairwise_hbb_iou():
    anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                            [0.0, 0.0, 5.0, 10.0]])
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    overlap = audit.pairwise_hbb_iou(anchors, gt)
    assert overlap[:, 0].tolist() == pytest.approx([1.0, 0.5])


def test_feature_map_contract_uses_first_level_device_and_all_shapes():
    features = [torch.zeros((1, 4, 8, 12)),
                torch.zeros((1, 4, 4, 6))]
    sizes, device = audit.feature_map_contract(features)
    assert sizes == [(8, 12), (4, 6)]
    assert device == features[0].device
    with pytest.raises(ValueError, match='at least one'):
        audit.feature_map_contract([])


def test_flatten_rpn_scores_preserves_all_feature_levels():
    scores = [torch.zeros((1, 3, 4, 5)),
              torch.ones((1, 3, 2, 3)),
              torch.full((1, 3, 1, 2), 2.0)]
    flattened = audit.flatten_rpn_scores(scores, expected_levels=3)
    assert flattened.shape == (84,)
    assert flattened[:60].tolist() == pytest.approx([0.5] * 60)
    with pytest.raises(RuntimeError, match='level count'):
        audit.flatten_rpn_scores(scores[0], expected_levels=3)


def test_source_bins_are_source_defined():
    rows = [dict(objects=[dict(short_token=value)])
            for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    boundaries = audit.source_token_boundaries(rows)
    assert audit.token_bin(1.0, boundaries) == 'source_small'
    assert audit.token_bin(6.0, boundaries) == 'source_large'


def test_diagnosis_separates_anchor_and_trained_rpn_failures():
    args = argparse.Namespace(recall_ks=[20, 2000])
    source = dict(rpn_recall_at={'2000': 0.9})
    target = dict(
        anchor_coverage_rate={'0.5': 0.9},
        rpn_recall_at={'2000': 0.1})
    assert audit.diagnose_slice(source, target, args) == (
        'ANCHORS_COVER_BUT_TRAINED_RPN_FAILS')
    target['anchor_coverage_rate']['0.5'] = 0.5
    assert audit.diagnose_slice(source, target, args) == (
        'ANCHOR_OR_ASSIGNMENT_GEOMETRY_LIMITED')
