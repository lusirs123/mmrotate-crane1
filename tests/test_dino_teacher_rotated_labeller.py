import argparse
import json
import pathlib

import numpy as np
import pytest
import torch

from crane_project.tools import dino_teacher_rotated_labeller as labeller


def _args(tmp_path, **overrides):
    checkpoint = tmp_path / 'dino.pth'
    checkpoint.write_bytes(b'dino')
    values = dict(
        seed=0, source_val_modulus=5, dino_gpus=[1, 2], head_gpu=0,
        source_train_datasets=None, source_val_datasets=None,
        patch_size=14, rpn_feat_channels=256, roi_fc_channels=1024,
        roi_samples=256, proposal_count=2000, max_detections=2000,
        s7_residual=False, s7_channels=128, s7_rpn_feat_channels=128,
        s7_proposal_count=500, s7_nms_pre=2000,
        s7_anchor_sizes=[16, 32, 64, 128, 256],
        roi_nms_iou_thr=0.1,
        valid_content_tolerance=1e-3,
        deployment_score_thr=0.05, border_margin_ratio=0.02,
        epochs=8, lr=0.001, momentum=0.9, weight_decay=1e-4,
        max_grad_norm=10.0, riou_thr=0.5, target_min_wins=26,
        warmup_iters=1000, warmup_ratio=0.001,
        lr_steps=[5, 7], lr_gamma=0.1,
        checkpoint_interval=1,
        selection_epochs=[1, 2, 3, 4, 5, 6, 7, 8],
        max_mcml=5, source_min_top1_rate=0.8,
        train_components='all', init_checkpoint=None,
        source_small_repeat=1, source_retain_max_top1_drop=0,
        pairwise_margin=0.5, pairwise_cls_loss_weight=0.25,
        pairwise_loss_weight=1.0,
        retention_loss_weight=1.0, retention_temperature=1.0,
        pairwise_negative_riou_thr=0.1, pairwise_nms_iou_thr=0.1,
        pairwise_negatives_per_positive=2,
        resume_checkpoint=None, eval_only_checkpoint=None,
        source_val_results_out=None,
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


def test_validate_rejects_selection_outside_checkpoint_epochs(tmp_path):
    with pytest.raises(ValueError, match='validation checkpoints'):
        labeller.validate_args(_args(
            tmp_path, checkpoint_interval=2, selection_epochs=[1, 2, 4, 6, 8]))


def test_validate_rejects_init_and_resume_together(tmp_path):
    init_path = tmp_path / 'init.pth'
    init_path.write_bytes(b'checkpoint')
    with pytest.raises(ValueError, match='cannot be combined'):
        labeller.validate_args(_args(
            tmp_path, init_checkpoint=str(init_path),
            resume_checkpoint='resume.pth'))


def test_validate_requires_checkpoint_for_roi_classifier_mode(tmp_path):
    with pytest.raises(ValueError, match='requires an init/resume/eval-only'):
        labeller.validate_args(_args(tmp_path, train_components='roi_cls'))


def test_validate_requires_checkpoint_for_pairwise_roi_mode(tmp_path):
    with pytest.raises(ValueError, match='requires an init/resume/eval-only'):
        labeller.validate_args(_args(
            tmp_path, train_components='roi_cls_pairwise'))


def test_validate_s7_requires_explicit_mode_and_native_init(tmp_path):
    with pytest.raises(ValueError, match='enabled together'):
        labeller.validate_args(_args(tmp_path, s7_residual=True))
    with pytest.raises(ValueError, match='requires an init/resume/eval-only'):
        labeller.validate_args(_args(
            tmp_path, s7_residual=True, train_components='s7_rpn',
            epochs=4, lr_steps=None, selection_epochs=[1, 2, 3, 4]))
    init_path = tmp_path / 'init.pth'
    init_path.write_bytes(b'checkpoint')
    args = _args(
        tmp_path, s7_residual=True, train_components='s7_rpn',
        init_checkpoint=str(init_path), epochs=4, lr_steps=None,
        selection_epochs=[1, 2, 3, 4])
    labeller.validate_args(args)
    assert args.lr_steps == [2, 3]


def test_validate_pairwise_v2_requires_matching_nms_policy(tmp_path):
    init_path = tmp_path / 'init.pth'
    init_path.write_bytes(b'checkpoint')
    with pytest.raises(ValueError, match='same NMS IoU threshold'):
        labeller.validate_args(_args(
            tmp_path, train_components='roi_cls_pairwise_v2',
            init_checkpoint=str(init_path), roi_nms_iou_thr=0.5,
            pairwise_nms_iou_thr=0.1, epochs=4, lr_steps=[2, 3],
            selection_epochs=[1, 2, 3, 4]))


def test_validate_pairwise_v2_requires_exact_source_retention(tmp_path):
    init_path = tmp_path / 'init.pth'
    init_path.write_bytes(b'checkpoint')
    with pytest.raises(ValueError, match='exact source retention'):
        labeller.validate_args(_args(
            tmp_path, train_components='roi_cls_pairwise_v2',
            init_checkpoint=str(init_path), roi_nms_iou_thr=0.5,
            pairwise_nms_iou_thr=0.5,
            source_retain_max_top1_drop=1, epochs=4, lr_steps=[2, 3],
            selection_epochs=[1, 2, 3, 4]))


def test_validate_pairwise_v2_limits_schedule_to_four_epochs(tmp_path):
    init_path = tmp_path / 'init.pth'
    init_path.write_bytes(b'checkpoint')
    with pytest.raises(ValueError, match='at most 4 epochs'):
        labeller.validate_args(_args(
            tmp_path, train_components='roi_cls_pairwise_v2',
            init_checkpoint=str(init_path), roi_nms_iou_thr=0.5,
            pairwise_nms_iou_thr=0.5))


def test_validate_pairwise_v2_accepts_four_epoch_schedule(tmp_path):
    init_path = tmp_path / 'init.pth'
    init_path.write_bytes(b'checkpoint')
    args = _args(
        tmp_path, train_components='roi_cls_pairwise_v2',
        init_checkpoint=str(init_path), roi_nms_iou_thr=0.5,
        pairwise_nms_iou_thr=0.5, epochs=4, lr_steps=None,
        selection_epochs=[1, 2, 3, 4])
    labeller.validate_args(args)
    assert args.lr_steps == [2, 3]
    assert args.selection_epochs == [1, 2, 3, 4]


def test_validate_rejects_pairwise_negative_overlap_with_positive_band(
        tmp_path):
    with pytest.raises(ValueError, match='negative-riou-thr'):
        labeller.validate_args(_args(
            tmp_path, pairwise_negative_riou_thr=0.5))


def test_source_split_is_deterministic_and_disjoint(tmp_path):
    records = [_record(tmp_path, frame) for frame in range(1, 11)]
    train, val = labeller.split_source_records(records, modulus=5)
    assert [row['frame'] for row in val] == [5, 10]
    assert set(row['frame'] for row in train).isdisjoint(
        row['frame'] for row in val)


def test_formal_source_records_support_train_sim_image_mapping(tmp_path):
    root = tmp_path / 'data'
    for split in ('train', 'train_sim', 'val'):
        (root / split / 'annfiles').mkdir(parents=True)
    (root / 'train' / 'images').mkdir(parents=True)
    (root / 'val' / 'images').mkdir(parents=True)
    fixtures = (
        ('train', 'train', 'real_seq01_00001'),
        ('train_sim', 'train', 'sim_seq08_00001'),
        ('val', 'val', 'real_seq07_00001'))
    for annotation_split, image_split, name in fixtures:
        (root / annotation_split / 'annfiles' / (name + '.txt')).write_text(
            '', encoding='ascii')
        (root / image_split / 'images' / (name + '.jpg')).write_bytes(b'image')
    args = _args(
        tmp_path, data_root=str(root),
        source_train_datasets=['train:train', 'train_sim:train'],
        source_val_datasets=['val:val'])
    train, val = labeller.formal_source_records(args)
    assert [row['seq'] for row in train] == ['real_seq01', 'sim_seq08']
    assert [row['seq'] for row in val] == ['real_seq07']


def test_target_image_cannot_enter_source_training(tmp_path):
    source = [_record(tmp_path, 1)]
    target = [dict(source[0], split='test', seq='real_seq02', frame=137)]
    with pytest.raises(RuntimeError, match='leaked'):
        labeller.assert_training_target_isolation(source, target)


def test_source_val_result_export_preserves_one_class_structure(tmp_path):
    rows = [dict(detections=[[1, 2, 3, 4, 0, 0.9]]),
            dict(detections=[])]
    path = tmp_path / 'source_val.pkl'
    labeller.write_detection_rows_pickle(rows, str(path))
    import pickle
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    assert payload[0][0].shape == (1, 6)
    assert payload[1][0].shape == (0, 6)


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


def test_roi_config_propagates_explicit_nms_policy(tmp_path):
    config = labeller.roi_config(
        1024, _args(tmp_path, roi_nms_iou_thr=0.5))
    assert config['test_cfg']['nms']['iou_thr'] == pytest.approx(0.5)


def test_interpolated_feature_levels_preserve_default_and_add_scales(tmp_path):
    args = _args(tmp_path, feature_strides=[7, 14, 28])
    heads = object.__new__(labeller.FrozenDinoRotatedHeads)
    heads._args = args
    feature = torch.zeros((1, 4, 6, 8))
    levels = heads.feature_levels(feature)
    assert [tuple(level.shape[-2:]) for level in levels] == [
        (12, 16), (6, 8), (3, 4)]


def test_multiscale_rpn_and_roi_share_feature_stride_contract(tmp_path):
    args = _args(tmp_path, feature_strides=[7, 14, 28])
    rpn = labeller.rpn_config(1024, args)
    roi = labeller.roi_config(1024, args)
    assert rpn['anchor_generator']['strides'] == [7, 14, 28]
    assert roi['bbox_roi_extractor']['featmap_strides'] == [7, 14, 28]


def test_s7_rpn_uses_audited_stride_and_physical_anchor_sizes(tmp_path):
    args = _args(tmp_path, s7_residual=True)
    config = labeller.s7_rpn_config(128, args)
    anchors = config['anchor_generator']
    assert anchors['strides'] == [7]
    assert np.asarray(anchors['scales']) * 7 == pytest.approx(
        [16, 32, 64, 128, 256])
    assert config['test_cfg']['max_per_img'] == 500
    assert labeller.roi_candidate_budget(args) == 2500


def test_s7_readout_projects_before_upsampling_and_zero_gates_refinement():
    readout = labeller.ResidualS7Readout(8, 4)
    feature = torch.randn(1, 8, 3, 5)
    output = readout(feature)
    expected = torch.nn.functional.interpolate(
        readout.projection(feature), scale_factor=2.0,
        mode='bilinear', align_corners=False)
    assert output.shape == (1, 4, 6, 10)
    assert torch.equal(output, expected)
    assert float(readout.residual_gate.detach()) == pytest.approx(0.0)


def test_s7_proposal_merge_keeps_native_rows_before_bounded_supplement():
    class FakeRpn:
        def __init__(self, rows):
            self.rows = rows

        def simple_test_rpn(self, _features, _metas):
            return [self.rows]

    heads = object.__new__(labeller.FrozenDinoRotatedHeads)
    torch.nn.Module.__init__(heads)
    heads._args = argparse.Namespace(patch_size=14, feature_strides=None)
    heads.s7_enabled = True
    heads._s7_inference_enabled = True
    native = torch.tensor([[1, 2, 3, 4, 0, 0.9]], dtype=torch.float32)
    extra = torch.tensor([[5, 6, 7, 8, 0, 0.8]], dtype=torch.float32)
    heads.rpn_head = FakeRpn(native)
    heads.s7_rpn_head = FakeRpn(extra)
    heads.s7_readout = torch.nn.Identity()
    feature = torch.zeros((1, 4, 2, 3))
    _features, proposals = heads.simple_test_proposals(feature, {})
    assert torch.equal(proposals[0], torch.cat([native, extra], dim=0))


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
    losses = dict(
        loss_rpn_cls=[torch.tensor([1.0, 3.0])],
        loss_rpn_bbox=[torch.tensor(2.0)],
        loss_cls=torch.tensor(4.0),
        loss_bbox=torch.tensor(5.0),
        acc=torch.tensor(99.0))
    total = labeller.loss_total(losses)
    assert float(total.item()) == pytest.approx(13.0)
    assert labeller.loss_component_means(losses) == pytest.approx(dict(
        loss_rpn_cls=2.0, loss_rpn_bbox=2.0,
        loss_cls=4.0, loss_bbox=5.0))


def test_roi_cls_mode_trains_only_final_classifier():
    class TinyBBoxHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_fcs = torch.nn.ModuleList([
                torch.nn.Linear(3, 3)])
            self.fc_cls = torch.nn.Linear(3, 2)
            self.fc_reg = torch.nn.Linear(3, 5)

    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rpn_head = torch.nn.Linear(3, 3)
            self.roi_head = torch.nn.Module()
            self.roi_head.bbox_head = TinyBBoxHead()

    heads = TinyHeads()
    names = labeller.configure_trainable_components(heads, 'roi_cls')
    assert names == [
        'roi_head.bbox_head.fc_cls.weight',
        'roi_head.bbox_head.fc_cls.bias']
    assert all(
        parameter.requires_grad == name.startswith(
            'roi_head.bbox_head.fc_cls.')
        for name, parameter in heads.named_parameters())
    pairwise_names = labeller.configure_trainable_components(
        heads, 'roi_cls_pairwise')
    assert pairwise_names == names
    pairwise_v2_names = labeller.configure_trainable_components(
        heads, 'roi_cls_pairwise_v2')
    assert pairwise_v2_names == names


def test_s7_mode_trains_only_readout_and_s7_rpn():
    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.s7_enabled = True
            self.s7_readout = torch.nn.Conv2d(3, 2, 1)
            self.s7_rpn_head = torch.nn.Conv2d(2, 2, 1)
            self.rpn_head = torch.nn.Conv2d(3, 2, 1)
            self.roi_head = torch.nn.Conv2d(3, 2, 1)

        def s7_inference_enabled(self):
            return True

    heads = TinyHeads()
    names = labeller.configure_trainable_components(heads, 's7_rpn')
    assert names == [
        's7_readout.weight', 's7_readout.bias',
        's7_rpn_head.weight', 's7_rpn_head.bias']
    assert all(
        parameter.requires_grad == (
            name.startswith('s7_readout.')
            or name.startswith('s7_rpn_head.'))
        for name, parameter in heads.named_parameters())


def test_s7_base_checkpoint_load_allows_only_new_branch_keys():
    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = torch.nn.Linear(2, 2)
            self.s7_readout = torch.nn.Linear(2, 2)
            self.s7_rpn_head = torch.nn.Linear(2, 2)
            self.enabled = True

        def set_s7_inference_enabled(self, enabled):
            self.enabled = bool(enabled)

    heads = TinyHeads()
    native = torch.nn.Linear(2, 2)
    payload = {'heads_state_dict': {
        'native.weight': native.weight.detach().clone(),
        'native.bias': native.bias.detach().clone()}}
    labeller.load_heads_checkpoint_state(
        heads, payload, allow_s7_base_initialization=True)
    assert heads.enabled is False
    assert torch.equal(heads.native.weight, native.weight)


def test_s7_source_gate_requires_absolute_small_and_mcml_targets(tmp_path):
    args = _args(tmp_path)
    baseline = dict(top1_hits=677, top1_mcml=3)
    baseline_small = dict(top1_hits=303, top1_mcml=3)
    retention = dict(
        baseline_correct_count=677, retained_correct_count=677,
        lost_correct_count=0)
    candidate = dict(top1_hits=678, top1_mcml=3)
    candidate_small = dict(top1_hits=303, top1_mcml=3)
    result = labeller.s7_source_selection_gate(
        baseline, baseline_small, candidate, candidate_small, retention, args)
    assert result['passed']
    candidate_small['top1_hits'] = 302
    assert not labeller.s7_source_selection_gate(
        baseline, baseline_small, candidate, candidate_small,
        retention, args)['passed']


def test_s7_config_declares_explicit_supplement_checkpoint_and_head():
    import runpy
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(str(root /
        'crane_project/configs/crane_symeood_scoped_dino_lowlight_s7_v1.py'))
    assert config['model']['dino_rescue']['head']['s7_residual'] is True
    assert config['model']['dino_rescue']['head']['s7_channels'] == 128
    assert config['model']['dino_head_checkpoint'].endswith(
        'dino_teacher_s7_residual_v1/labeller_best_source_only.pth')


def test_roi_cls_mode_optimizes_only_roi_classification_loss():
    losses = dict(
        loss_rpn_cls=torch.tensor(10.0),
        loss_rpn_bbox=torch.tensor(20.0),
        loss_cls=torch.tensor(3.0, requires_grad=True),
        loss_bbox=torch.tensor(30.0))
    total = labeller.optimization_loss_total(losses, 'roi_cls')
    assert float(total.item()) == pytest.approx(3.0)


def test_pairwise_roi_mode_optimizes_all_three_authorized_losses():
    losses = dict(
        loss_cls=torch.tensor(3.0, requires_grad=True),
        loss_roi_pairwise=torch.tensor(2.0, requires_grad=True),
        loss_roi_retention=torch.tensor(0.5, requires_grad=True),
        loss_bbox=torch.tensor(30.0))
    total = labeller.optimization_loss_total(losses, 'roi_cls_pairwise')
    assert float(total.item()) == pytest.approx(5.5)
    total_v2 = labeller.optimization_loss_total(
        losses, 'roi_cls_pairwise_v2')
    assert float(total_v2.item()) == pytest.approx(5.5)


def test_pairwise_margin_loss_rewards_correct_relative_order():
    good = labeller.roi_pairwise_margin_loss(
        torch.tensor([2.0]), torch.tensor([0.0]), margin=0.5)
    bad = labeller.roi_pairwise_margin_loss(
        torch.tensor([0.0]), torch.tensor([1.0]), margin=0.5)
    assert float(good.item()) == pytest.approx(0.0)
    assert float(bad.item()) == pytest.approx(1.5)


def test_paired_margin_loss_does_not_form_cartesian_pairs():
    loss = labeller.roi_paired_margin_loss(
        torch.tensor([2.0, 0.0]), torch.tensor([0.0, 1.0]), margin=0.5)
    assert float(loss.item()) == pytest.approx(0.75)
    with pytest.raises(ValueError, match='same shape'):
        labeller.roi_paired_margin_loss(
            torch.tensor([1.0, 2.0]), torch.tensor([0.0]), margin=0.5)


def test_classifier_retention_loss_is_zero_for_identical_logits():
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.2]])
    loss = labeller.roi_classifier_retention_loss(
        logits, logits.clone(), temperature=1.0)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-7)


def test_pairwise_mining_prioritizes_nms_competitor_before_global_false():
    overlap = torch.tensor([0.8, 0.0, 0.0, 0.3])
    scores = torch.tensor([0.2, 0.7, 0.99, 0.8])
    competitor_iou = torch.tensor([1.0, 0.2, 0.0, 0.15])
    positive, negative = labeller.select_hard_pairwise_indices(
        overlap, scores, competitor_iou, max_samples=3,
        positive_fraction=0.5, positive_riou_thr=0.5,
        negative_riou_thr=0.1, nms_iou_thr=0.1)
    assert positive.tolist() == [0]
    assert negative.tolist() == [3, 1]


def test_pairwise_v2_mines_only_actual_higher_scoring_competitors():
    overlap = torch.tensor([0.8, 0.0, 0.3, 0.0])
    scores = torch.tensor([0.2, 0.9, 0.95, 0.1])
    positive = torch.tensor([0])
    candidate_iou = torch.tensor([[1.0], [0.6], [0.1], [0.8]])
    pair_pos, pair_neg, suppressor = (
        labeller.mine_actual_roi_competitor_pairs(
            overlap, scores, positive, candidate_iou,
            positive_riou_thr=0.5, nms_iou_thr=0.5,
            negatives_per_positive=2))
    assert pair_pos.tolist() == [0, 0]
    assert pair_neg.tolist() == [2, 1]
    assert suppressor.tolist() == [False, True]


def test_pairwise_v2_ignores_lower_scoring_nms_overlap():
    overlap = torch.tensor([0.8, 0.0])
    scores = torch.tensor([0.7, 0.6])
    pair_pos, pair_neg, suppressor = (
        labeller.mine_actual_roi_competitor_pairs(
            overlap, scores, torch.tensor([0]),
            torch.tensor([[1.0], [0.9]]),
            positive_riou_thr=0.5, nms_iou_thr=0.5,
            negatives_per_positive=1))
    assert pair_pos.numel() == 0
    assert pair_neg.numel() == 0
    assert suppressor.numel() == 0


def test_pairwise_v2_uses_best_scoring_usable_roi_per_gt():
    overlap_by_gt = torch.tensor([
        [0.8], [0.7], [0.2]])
    scores = torch.tensor([0.4, 0.9, 1.0])
    selected = labeller.select_representative_usable_rois(
        overlap_by_gt, scores, positive_riou_thr=0.5, max_positives=1)
    assert selected.tolist() == [1]


def test_rotated_box_corners_and_valid_content_filter_preserve_order():
    detections = np.asarray([
        [50.0, 50.0, 20.0, 10.0, 0.0, 0.9],
        [0.0, 0.0, 100.0, 100.0, 0.0, 0.8],
        [100.0, 100.0, 20.0, 10.0, np.pi / 4.0, 0.7],
    ], dtype=np.float32)
    corners = labeller.rotated_box_corners(detections[:, :5])
    assert corners.shape == (3, 4, 2)
    filtered, stats = labeller.filter_valid_rotated_detections(
        detections, {'ori_shape': (200, 200, 3)})
    assert filtered[:, 5].tolist() == pytest.approx([0.9, 0.7])
    assert stats == dict(raw_detection_count=3,
                         invalid_border_filtered_count=1,
                         valid_detection_count=2)


def test_valid_content_filter_rejects_rotated_corner_outside_image():
    detections = np.asarray([
        [5.0, 5.0, 20.0, 10.0, np.pi / 4.0, 0.9],
    ], dtype=np.float32)
    filtered, stats = labeller.filter_valid_rotated_detections(
        detections, {'ori_shape': (100, 100, 3)})
    assert filtered.shape == (0, 6)
    assert stats['invalid_border_filtered_count'] == 1


def test_gt_border_metrics_distinguish_interior_and_near_border():
    image_meta = {'ori_shape': (100, 100, 3)}
    interior = np.asarray([[50, 50, 10, 10, 0]], dtype=np.float32)
    border = np.asarray([[5, 5, 10, 10, 0]], dtype=np.float32)
    interior_metrics = labeller.gt_border_metrics(
        interior, image_meta, margin_ratio=0.02)
    border_metrics = labeller.gt_border_metrics(
        border, image_meta, margin_ratio=0.02)
    assert interior_metrics['gt_near_border'] is False
    assert interior_metrics['gt_min_border_distance_px'] == pytest.approx(45)
    assert border_metrics['gt_near_border'] is True
    assert border_metrics['gt_all_corners_inside'] is True


def test_summary_reports_deployment_threshold_and_filter_statistics():
    rows = []
    values = [(True, True, False), (True, False, True),
              (False, False, False)]
    for frame, (top1, deployed, near_border) in enumerate(values, start=1):
        raw_rank = 2 if frame == 1 else (5 if frame == 2 else 4)
        rows.append(dict(
            seq='seq', frame=frame,
            metrics=dict(
                top1_hit=top1, deployment_top1_hit=deployed,
                deployment_silence=not deployed,
                best_usable_rank=(1 if top1 else None),
                top1_riou=(0.6 if top1 else 0.1), top1_score=0.1,
                raw_detection_count=10,
                invalid_border_filtered_count=6,
                valid_detection_count=4,
                raw_unfiltered=dict(
                    top1_hit=False, best_usable_rank=raw_rank,
                    top1_riou=0.0, top1_score=0.2),
                filter_effect=dict(
                    removed_usable_geometry=(frame == 3),
                    promoted_to_top1=(frame <= 2),
                    demoted_from_top1=False),
                gt_near_border=near_border)))
    summary = labeller.summarize_rows(rows)
    assert summary['top1_hits'] == 2
    assert summary['top1_mcml'] == 1
    assert summary['deployment_top1_hits'] == 1
    assert summary['deployment_top1_mcml'] == 2
    assert summary['invalid_border_filtered_fraction'] == pytest.approx(0.6)
    assert summary['near_border_frame_count'] == 1
    assert summary['near_border_top1_hits'] == 1
    assert summary['raw_unfiltered_top1_hits'] == 0
    assert summary['raw_unfiltered_geometry_eligible_count'] == 3
    assert summary['filter_removed_usable_geometry_count'] == 1
    assert summary['filter_promoted_to_top1_count'] == 2
    assert summary['filter_demoted_from_top1_count'] == 0


def test_longest_miss_resets_at_sequence_boundary():
    rows = [
        dict(seq='a', frame=1, hit=False),
        dict(seq='a', frame=2, hit=False),
        dict(seq='b', frame=3, hit=False),
    ]
    assert labeller.longest_miss(rows, 'hit') == 2


def test_source_selection_prioritizes_top1_before_oracle_recall():
    a = dict(top1_hits=5, recall_at_20=5,
             recall_at_100=5, mean_top1_riou=0.5)
    b = dict(top1_hits=4, recall_at_20=20,
             recall_at_100=20, mean_top1_riou=0.9)
    assert labeller.source_selection_key(a) > labeller.source_selection_key(b)


def test_roi_cls_selection_prioritizes_source_small_validation():
    full_a = dict(top1_hits=10, recall_at_20=10,
                  recall_at_100=10, mean_top1_riou=0.6)
    small_a = dict(top1_hits=2, recall_at_20=2,
                   recall_at_100=2, mean_top1_riou=0.6)
    full_b = dict(top1_hits=9, recall_at_20=9,
                  recall_at_100=9, mean_top1_riou=0.6)
    small_b = dict(top1_hits=3, recall_at_20=3,
                   recall_at_100=3, mean_top1_riou=0.6)
    assert labeller.roi_cls_selection_key(full_b, small_b) > (
        labeller.roi_cls_selection_key(full_a, small_a))


def test_exact_source_retention_detects_swapped_correct_frames():
    baseline = ['val|seq|1', 'val|seq|2']
    candidate = [
        dict(split='val', seq='seq', frame=1,
             metrics=dict(top1_hit=True)),
        dict(split='val', seq='seq', frame=2,
             metrics=dict(top1_hit=False)),
        dict(split='val', seq='seq', frame=3,
             metrics=dict(top1_hit=True)),
    ]
    summary = labeller.source_top1_retention_summary(baseline, candidate)
    assert summary['candidate_correct_count'] == 2
    assert summary['retained_correct_count'] == 1
    assert summary['lost_frame_keys'] == ['val|seq|2']
    assert summary['gained_frame_keys'] == ['val|seq|3']


def test_pairwise_v2_source_gate_requires_strict_small_improvement():
    baseline_full = dict(top1_hits=662, top1_mcml=7)
    baseline_small = dict(top1_hits=293, top1_mcml=7)
    retention = dict(
        baseline_correct_count=662, retained_correct_count=662,
        lost_correct_count=0)
    unchanged = labeller.pairwise_v2_source_selection_gate(
        baseline_full, baseline_small,
        dict(top1_hits=662, top1_mcml=7),
        dict(top1_hits=293, top1_mcml=7), retention)
    assert not unchanged['passed']
    improved = labeller.pairwise_v2_source_selection_gate(
        baseline_full, baseline_small,
        dict(top1_hits=663, top1_mcml=7),
        dict(top1_hits=294, top1_mcml=6), retention)
    assert improved['passed']


def test_source_training_progress_is_atomic_and_target_free(tmp_path):
    args = argparse.Namespace(
        out_json=str(tmp_path / 'train_result.json'),
        train_components='roi_cls_pairwise_v2', epochs=4)
    output_path, replacements = labeller.write_source_training_progress(
        args, completed_epoch=1, best_epoch=0,
        best_path=str(tmp_path / 'best.pth'),
        latest_path=str(tmp_path / 'latest.pth'),
        baseline_summary=dict(top1_hits=662),
        baseline_small_summary=dict(top1_hits=293),
        best_summary=dict(top1_hits=662),
        best_small_summary=dict(top1_hits=293),
        history=[dict(epoch=1, pairwise_v2_source_gate=dict(
            passed=False, checks=dict(exact_old_correct_retention=False)))])
    with open(output_path, 'r') as handle:
        payload = json.load(handle)
    assert output_path.endswith('train_result.partial.json')
    assert replacements == 0
    assert payload['target_read'] is False
    assert payload['completed_epoch'] == 1
    assert payload['best_epoch'] == 0
    assert payload['history'][0]['pairwise_v2_source_gate']['passed'] is False


def test_source_small_sampling_uses_source_train_threshold_only(
        monkeypatch, tmp_path):
    args = _args(tmp_path, source_small_repeat=2)
    train = [dict(name='small'), dict(name='medium'), dict(name='large')]
    val = [dict(name='val_small'), dict(name='val_large')]
    scale = dict(small=1.0, medium=2.0, large=3.0,
                 val_small=1.5, val_large=2.5)
    monkeypatch.setattr(
        labeller, 'record_short_token',
        lambda record, _args: scale[record['name']])
    balanced, sampling = labeller.source_small_balanced_records(train, args)
    selected_val = labeller.source_small_records(
        val, args, sampling['short_token_threshold'])
    assert sampling['short_token_threshold'] == pytest.approx(5.0 / 3.0)
    assert [row['name'] for row in balanced] == [
        'small', 'small', 'medium', 'large']
    assert [row['name'] for row in selected_val] == ['val_small']


def test_lr_schedule_uses_independent_dino_head_warmup_and_steps(tmp_path):
    args = _args(tmp_path)
    assert labeller.scheduled_lr(
        args, epoch=1, global_step=0) == pytest.approx(0.001 * 0.001)
    assert labeller.scheduled_lr(
        args, epoch=1, global_step=1000) == pytest.approx(0.001)
    assert labeller.scheduled_lr(
        args, epoch=6, global_step=1000) == pytest.approx(0.0001)
    assert labeller.scheduled_lr(
        args, epoch=8, global_step=1000) == pytest.approx(0.00001)


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


def test_target_decision_rejects_broken_source_control(tmp_path):
    args = _args(tmp_path)
    target = dict(frame_count=33, top1_hits=32, top1_mcml=1,
                  recall_at_100=32)
    source = dict(frame_count=45, top1_hits=35)
    assert labeller.make_target_decision(target, args, source) == (
        'AUDIT_INVALID_SOURCE_CONTROL')


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
