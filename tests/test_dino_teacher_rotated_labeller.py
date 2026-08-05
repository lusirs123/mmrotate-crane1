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
        s7_component_checkpoint=None, s7_protected_merge=False,
        s7_merge_init_bias=-2.0, s7_merge_margin=0.5,
        s7_merge_retention_weight=2.0, s7_merge_gain_weight=1.0,
        s7_merge_prior_weight=0.01,
        s7_lane_hidden=32, s7_lane_max_adjustment=2.0,
        s7_lane_base_epoch=1,
        s7_lane_hard_negatives=4, s7_lane_gain_repeat=8,
        s7_lane_arbitration=False,
        s7_quality_suppression=False,
        s7_quality_hidden=32, s7_quality_max_suppression=2.0,
        s7_quality_init_risk_bias=0.0, s7_quality_margin=0.5,
        s7_quality_risk_weight=1.0, s7_quality_preserve_weight=1.0,
        s7_quality_retention_weight=2.0, s7_quality_prior_weight=0.01,
        s7_quality_base_epoch=1,
        s7_temporal_association=False, s7_temporal_base_epoch=1,
        s7_temporal_quality_head=False, s7_temporal_quality_hidden=128,
        s7_temporal_quality_loss_weight=1.0,
        s7_temporal_relative_quality=False,
        s7_temporal_relative_quality_weight=0.5,
        s7_temporal_relative_quality_margin=0.25,
        s7_temporal_relative_quality_min_gap=0.10,
        s7_temporal_relative_quality_max_pairs=128,
        s7_temporal_relative_base_epoch=4,
        s7_temporal_margin=0.5, s7_temporal_retention_weight=2.0,
        s7_temporal_gain_weight=1.0, s7_temporal_prior_weight=0.01,
        s7_temporal_max_candidates=100,
        s7_temporal_min_confirmations=2,
        s7_temporal_override_margin=0.25,
        s7_temporal_max_center_distance=3.0,
        s7_temporal_min_riou=0.05,
        s7_temporal_min_appearance=0.20,
        s7_source_min_full_top1=677, s7_source_min_small_top1=303,
        s7_source_max_mcml=3,
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
        skip_target_eval=False,
        source_conflict_result_json=None, source_conflict_epoch=1,
        source_temporal_attribution_audit=False,
        source_temporal_attribution_epoch=4,
        source_temporal_immediate_override_audit=False,
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


def test_validate_s7_merge_requires_native_and_component_checkpoints(tmp_path):
    native = tmp_path / 'native.pth'
    component = tmp_path / 's7_epoch2.pth'
    native.write_bytes(b'native')
    component.write_bytes(b's7')
    with pytest.raises(ValueError, match='component-checkpoint'):
        labeller.validate_args(_args(
            tmp_path, s7_residual=True, train_components='s7_merge',
            init_checkpoint=str(native), epochs=4, lr_steps=None,
            selection_epochs=[1, 2, 3, 4]))
    args = _args(
        tmp_path, s7_residual=True, train_components='s7_merge',
        init_checkpoint=str(native), s7_component_checkpoint=str(component),
        epochs=4, lr_steps=None, selection_epochs=[1, 2, 3, 4])
    labeller.validate_args(args)
    assert args.lr_steps == [2, 3]


def test_validate_s7_lane_arbitration_requires_complete_epoch1_checkpoint(
        tmp_path):
    checkpoint = tmp_path / 'epoch01.pth'
    checkpoint.write_bytes(b'checkpoint')
    args = _args(
        tmp_path, s7_residual=True,
        s7_lane_arbitration=True,
        train_components='s7_lane_arbitration',
        init_checkpoint=str(checkpoint), epochs=4, lr_steps=None,
        selection_epochs=[1, 2, 3, 4])
    labeller.validate_args(args)
    assert args.lr_steps == [2, 3]
    with pytest.raises(ValueError, match='component-checkpoint'):
        labeller.validate_args(
            _args(tmp_path, s7_residual=True,
                  s7_lane_arbitration=True,
                  train_components='s7_lane_arbitration',
                  init_checkpoint=str(checkpoint),
                  s7_component_checkpoint=str(checkpoint), epochs=4,
                  lr_steps=None, selection_epochs=[1, 2, 3, 4]))
    with pytest.raises(ValueError, match='locked to the audited epoch-1'):
        labeller.validate_args(
            _args(tmp_path, s7_residual=True,
                  s7_lane_arbitration=True,
                  train_components='s7_lane_arbitration',
                  init_checkpoint=str(checkpoint), s7_lane_base_epoch=2,
                  epochs=4, lr_steps=None,
                  selection_epochs=[1, 2, 3, 4]))
    with pytest.raises(ValueError, match='hard-negatives'):
        labeller.validate_args(
            _args(tmp_path, s7_residual=True,
                  s7_lane_arbitration=True,
                  train_components='s7_lane_arbitration',
                  init_checkpoint=str(checkpoint), s7_lane_hard_negatives=0,
                  epochs=4, lr_steps=None,
                  selection_epochs=[1, 2, 3, 4]))
    with pytest.raises(ValueError, match='gain-repeat'):
        labeller.validate_args(
            _args(tmp_path, s7_residual=True,
                  s7_lane_arbitration=True,
                  train_components='s7_lane_arbitration',
                  init_checkpoint=str(checkpoint), s7_lane_gain_repeat=33,
                  epochs=4, lr_steps=None,
                  selection_epochs=[1, 2, 3, 4]))


def test_validate_s7_quality_suppression_locks_source_only_formal_gate(
        tmp_path):
    checkpoint = tmp_path / 'affine_epoch01.pth'
    checkpoint.write_bytes(b'checkpoint')
    common = dict(
        s7_residual=True, train_components='s7_quality_suppression',
        init_checkpoint=str(checkpoint), epochs=4, lr_steps=None,
        selection_epochs=[1, 2, 3, 4],
        s7_source_min_full_top1=688,
        s7_source_min_small_top1=311)
    with pytest.raises(ValueError, match='source-only'):
        labeller.validate_args(_args(tmp_path, **common))
    args = _args(tmp_path, skip_target_eval=True, **common)
    labeller.validate_args(args)
    assert args.lr_steps == [2, 3]
    with pytest.raises(ValueError, match='min-full-top1'):
        labeller.validate_args(_args(
            tmp_path, skip_target_eval=True,
            **dict(common, s7_source_min_full_top1=687)))
    with pytest.raises(ValueError, match='min-small-top1'):
        labeller.validate_args(_args(
            tmp_path, skip_target_eval=True,
            **dict(common, s7_source_min_small_top1=310)))


def test_validate_s7_temporal_association_requires_continuous_source_protocol(
        tmp_path):
    checkpoint = tmp_path / 'affine_epoch01.pth'
    checkpoint.write_bytes(b'checkpoint')
    common = dict(
        s7_residual=True, s7_temporal_association=True,
        train_components='s7_temporal_association',
        init_checkpoint=str(checkpoint), skip_target_eval=True,
        epochs=4, lr_steps=None, selection_epochs=[1, 2, 3, 4],
        s7_source_min_full_top1=688,
        s7_source_min_small_top1=311)
    with pytest.raises(ValueError, match='formal source train/val'):
        labeller.validate_args(_args(tmp_path, **common))
    args = _args(
        tmp_path,
        source_train_datasets=['train:train', 'train_sim:train'],
        source_val_datasets=['val:val'], **common)
    labeller.validate_args(args)
    assert args.lr_steps == [2, 3]
    with pytest.raises(ValueError, match='frame repetition'):
        labeller.validate_args(_args(
            tmp_path,
            source_train_datasets=['train:train'],
            source_val_datasets=['val:val'], source_small_repeat=2,
            **common))


def test_validate_relative_quality_uses_pointwise_quality_checkpoint(
        tmp_path):
    checkpoint = tmp_path / 'quality_epoch04.pth'
    checkpoint.write_bytes(b'checkpoint')
    common = dict(
        s7_residual=True, s7_temporal_association=True,
        s7_temporal_quality_head=True,
        s7_temporal_relative_quality=True,
        train_components='s7_temporal_association',
        init_checkpoint=str(checkpoint), skip_target_eval=True,
        source_train_datasets=['train:train'],
        source_val_datasets=['val:val'], source_small_repeat=1,
        s7_source_min_full_top1=688, s7_source_min_small_top1=311,
        s7_temporal_min_confirmations=1,
        epochs=4, lr_steps=[2, 3], selection_epochs=[1, 2, 3, 4])
    args = _args(tmp_path, **common)
    labeller.validate_args(args)
    assert args.lr_steps == [2, 3]
    with pytest.raises(ValueError, match='quality-head'):
        labeller.validate_args(_args(
            tmp_path, **dict(common, s7_temporal_quality_head=False)))
    with pytest.raises(ValueError, match='S7 mode requires'):
        labeller.validate_args(_args(
            tmp_path, **dict(common, init_checkpoint=None)))


def test_relative_quality_is_included_in_temporal_loss_metadata():
    assert labeller.optimization_loss_component_names(
        's7_temporal_association', quality_head=True,
        relative_quality=True) == [
            'loss_s7_candidate_quality',
            'loss_s7_candidate_quality_relative']


def test_validate_relative_quality_allows_fixed_source_attribution(
        tmp_path):
    checkpoint = tmp_path / 'relative_epoch04.pth'
    checkpoint.write_bytes(b'checkpoint')
    args = _args(
        tmp_path,
        s7_residual=True, s7_temporal_association=True,
        s7_temporal_quality_head=True, s7_temporal_relative_quality=True,
        train_components='s7_temporal_association',
        source_train_datasets=['train:train'],
        source_val_datasets=['val:val'], source_small_repeat=1,
        s7_source_min_full_top1=688, s7_source_min_small_top1=311,
        s7_temporal_min_confirmations=1,
        epochs=4, lr_steps=[2, 3], selection_epochs=[1, 2, 3, 4],
        eval_only_checkpoint=str(checkpoint), skip_target_eval=True,
        source_temporal_attribution_audit=True,
        source_temporal_attribution_epoch=4)
    labeller.validate_args(args)
    assert args.eval_only_checkpoint == str(checkpoint)


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


def test_s7_score_calibrator_is_monotonic_and_conservatively_initialized():
    calibrator = labeller.S7ScoreCalibrator(initial_bias=-2.0)
    values = calibrator(torch.tensor([-1.0, 0.0, 1.0]))
    assert values.tolist() == pytest.approx([-3.0, -2.0, -1.0])
    assert float(calibrator.scale().item()) == pytest.approx(1.0)
    assert float(calibrator.prior_loss().item()) == pytest.approx(0.0)


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


def test_protected_merge_runs_nms_per_source_and_records_provenance():
    heads = object.__new__(labeller.FrozenDinoRotatedHeads)
    torch.nn.Module.__init__(heads)
    heads._args = argparse.Namespace(
        roi_nms_iou_thr=0.5, max_detections=10)
    heads.s7_score_calibrator = labeller.S7ScoreCalibrator(-2.0)
    native = torch.tensor([[1, 2, 3, 4, 0, 0.9]], dtype=torch.float32)
    supplement = torch.tensor([[5, 6, 7, 8, 0, 0.8]], dtype=torch.float32)
    heads.proposal_sources = lambda _feature, _meta: ([], dict(
        native_s14=native, supplement_s7=supplement))
    heads._decode_roi_candidates = lambda _feature, _meta, proposals, rescale: (
        proposals[:, :5], torch.logit(proposals[:, 5]), proposals[:, 5],
        torch.zeros((proposals.shape[0], 4)))
    calls = []

    def lane_nms(boxes, scores):
        calls.append(boxes.clone())
        return (torch.cat([boxes, scores[:, None]], dim=1),
                torch.arange(boxes.shape[0]))

    heads._nms_candidate_lane = lane_nms
    detections = heads._protected_merge_detections(
        torch.zeros(1), {'img_shape': (10, 10, 3)})
    assert len(calls) == 2
    assert torch.equal(calls[0], native[:, :5])
    assert torch.equal(calls[1], supplement[:, :5])
    assert detections.shape == (2, 6)
    assert heads._last_candidate_merge['proposal_source_counts'] == {
        'native_s14': 1, 'supplement_s7': 1}
    assert heads._last_candidate_merge['raw_top1_source'] == 'native_s14'
    assert heads._last_candidate_merge['source_top1_detections'] == {
        'native_s14': pytest.approx([1, 2, 3, 4, 0, 0.9]),
        'supplement_s7': pytest.approx(
            [5, 6, 7, 8, 0, torch.sigmoid(torch.logit(
                torch.tensor(0.8)) - 2.0).item()])}
    assert heads._last_candidate_merge['s7_affine_scale'] == pytest.approx(1.0)
    assert heads._last_candidate_merge['s7_affine_bias'] == pytest.approx(-2.0)


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


def test_s7_merge_mode_trains_only_affine_calibrator():
    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = torch.nn.Linear(2, 2)
            self.s7_readout = torch.nn.Linear(2, 2)
            self.s7_rpn_head = torch.nn.Linear(2, 2)
            self.s7_score_calibrator = labeller.S7ScoreCalibrator(-2.0)
            self.s7_protected_merge = True

    heads = TinyHeads()
    names = labeller.configure_trainable_components(heads, 's7_merge')
    assert names == [
        's7_score_calibrator.raw_scale', 's7_score_calibrator.bias']
    assert all(
        parameter.requires_grad == name.startswith('s7_score_calibrator.')
        for name, parameter in heads.named_parameters())


def test_s7_lane_arbitrator_starts_at_zero_and_is_bounded():
    arbitrator = labeller.S7LaneArbitrator(
        embedding_channels=4, hidden=3, max_adjustment=1.5)
    embedding = torch.randn(5, 4)
    raw = torch.randn(5)
    output = arbitrator(embedding, raw, torch.tensor(0.2))
    assert torch.equal(output, torch.zeros(5))
    with torch.no_grad():
        arbitrator.output.bias.fill_(100.0)
    output = arbitrator(embedding, raw, torch.tensor(0.2))
    assert bool(torch.all(output <= 1.5))
    assert bool(torch.all(output >= -1.5))


def test_s7_quality_suppressor_is_lane_wide_non_positive_and_near_zero():
    suppressor = labeller.S7QualitySuppressor(
        embedding_channels=4, hidden=3, max_suppression=2.0,
        initial_risk_bias=0.0)
    embedding = torch.randn(5, 4)
    raw = torch.tensor([-1.0, 0.4, 1.2, 0.2, -0.5])
    affine = raw - 0.3
    delta, risk_logit, top_index = suppressor(
        embedding, raw, affine, torch.tensor(0.8))
    assert int(top_index) == 2
    assert -2.0 <= float(delta.item()) <= 0.0
    assert float(delta.item()) == pytest.approx(0.0)
    assert float(risk_logit.item()) == pytest.approx(0.0)
    adjusted = affine + delta.expand_as(affine)
    assert torch.equal(torch.argsort(adjusted), torch.argsort(affine))
    torch.nn.functional.binary_cross_entropy_with_logits(
        risk_logit, torch.ones_like(risk_logit)).backward()
    assert float(suppressor.output.bias.grad.item()) < 0.0
    with torch.no_grad():
        suppressor.output.bias.fill_(100.0)
    delta, _risk, _top = suppressor(
        embedding, raw, affine, torch.tensor(0.8))
    assert float(delta.item()) == pytest.approx(-2.0)


def test_s7_lane_arbitration_trains_only_lane_module():
    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = torch.nn.Linear(2, 2)
            self.s7_readout = torch.nn.Linear(2, 2)
            self.s7_rpn_head = torch.nn.Linear(2, 2)
            self.s7_score_calibrator = labeller.S7ScoreCalibrator(-2.0)
            self.s7_lane_arbitrator = labeller.S7LaneArbitrator(2, 3)
            self.s7_protected_merge = True

    heads = TinyHeads()
    names = labeller.configure_trainable_components(
        heads, 's7_lane_arbitration')
    assert all(name.startswith('s7_lane_arbitrator.') for name in names)
    assert not any(name.startswith('s7_score_calibrator.') for name in names)


def test_s7_quality_suppression_trains_only_quality_module():
    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = torch.nn.Linear(2, 2)
            self.s7_readout = torch.nn.Linear(2, 2)
            self.s7_rpn_head = torch.nn.Linear(2, 2)
            self.s7_score_calibrator = labeller.S7ScoreCalibrator(-2.0)
            self.s7_quality_suppressor = labeller.S7QualitySuppressor(2, 3)
            self.s7_protected_merge = True

    heads = TinyHeads()
    names = labeller.configure_trainable_components(
        heads, 's7_quality_suppression')
    assert names
    assert all(name.startswith('s7_quality_suppressor.') for name in names)
    assert not any(name.startswith('s7_score_calibrator.') for name in names)


def test_s7_temporal_association_trains_only_six_cue_weights():
    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = torch.nn.Linear(2, 2)
            self.s7_score_calibrator = labeller.S7ScoreCalibrator(-2.0)
            self.s7_temporal_scorer = (
                labeller.temporal.S7TemporalAssociationScorer())
            self.s7_protected_merge = True

    heads = TinyHeads()
    names = labeller.configure_trainable_components(
        heads, 's7_temporal_association')
    assert names == ['s7_temporal_scorer.raw_weights']
    assert heads.s7_temporal_scorer.raw_weights.numel() == 6
    assert not heads.native.weight.requires_grad
    assert not heads.s7_score_calibrator.bias.requires_grad


def test_s7_temporal_quality_trains_only_candidate_quality_head():
    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = torch.nn.Linear(2, 2)
            self.s7_score_calibrator = labeller.S7ScoreCalibrator(-2.0)
            self.s7_temporal_scorer = (
                labeller.temporal.S7TemporalAssociationScorer(
                    cue_names=labeller.temporal.QUALITY_CUE_NAMES))
            self.s7_candidate_quality_head = (
                labeller.temporal.S7CandidateQualityHead(2, 4))
            self.s7_temporal_quality_head_enabled = True
            self.s7_protected_merge = True

    heads = TinyHeads()
    names = labeller.configure_trainable_components(
        heads, 's7_temporal_association')
    assert names
    assert all(name.startswith('s7_candidate_quality_head.') for name in names)
    assert not any(name.startswith('s7_temporal_scorer.') for name in names)
    assert not any(name.startswith('s7_score_calibrator.') for name in names)


def test_s7_lane_architecture_records_bounded_arbitration(tmp_path):
    args = _args(
        tmp_path, s7_residual=True,
        train_components='s7_lane_arbitration',
        s7_protected_merge=True, s7_lane_arbitration=True,
        s7_lane_hidden=24, s7_lane_max_adjustment=1.25)
    architecture = labeller.s7_architecture(args)
    assert architecture['lane_arbitration'] is True
    assert architecture['lane_hidden'] == 24
    assert architecture['lane_max_adjustment'] == pytest.approx(1.25)


def test_s7_quality_architecture_records_non_positive_lane_mode(tmp_path):
    args = _args(
        tmp_path, s7_residual=True,
        train_components='s7_quality_suppression',
        s7_protected_merge=True, s7_quality_suppression=True,
        s7_quality_hidden=24, s7_quality_max_suppression=1.25,
        s7_quality_init_risk_bias=-7.0)
    architecture = labeller.s7_architecture(args)
    assert architecture['quality_suppression'] is True
    assert architecture['lane_arbitration'] is False
    assert architecture['quality_hidden'] == 24
    assert architecture['quality_max_suppression'] == pytest.approx(1.25)
    assert architecture['quality_initial_risk_bias'] == pytest.approx(-7.0)


def test_s7_temporal_architecture_records_causal_candidate_policy(tmp_path):
    args = _args(
        tmp_path, s7_residual=True,
        train_components='s7_temporal_association',
        s7_protected_merge=True, s7_temporal_association=True,
        s7_temporal_max_candidates=100,
        s7_temporal_min_confirmations=2)
    architecture = labeller.s7_architecture(args)
    assert architecture['temporal_association'] is True
    assert architecture['temporal_cues'] == list(labeller.temporal.CUE_NAMES)
    assert architecture['temporal_max_candidates'] == 100
    assert architecture['temporal_min_confirmations'] == 2


def test_s7_temporal_quality_architecture_records_dense_candidate_head(tmp_path):
    args = _args(
        tmp_path, s7_residual=True,
        train_components='s7_temporal_association',
        s7_protected_merge=True, s7_temporal_association=True,
        s7_temporal_quality_head=True, s7_temporal_quality_hidden=24)
    architecture = labeller.s7_architecture(args)
    assert architecture['temporal_cues'] == list(
        labeller.temporal.QUALITY_CUE_NAMES)
    assert architecture['temporal_quality_head'] is True
    assert architecture['temporal_quality_hidden'] == 24


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


def test_s7_lane_checkpoint_initialization_allows_only_new_arbitrator_keys():
    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = torch.nn.Linear(2, 2)
            self.s7_readout = torch.nn.Linear(2, 2)
            self.s7_rpn_head = torch.nn.Linear(2, 2)
            self.s7_score_calibrator = labeller.S7ScoreCalibrator(-2.0)
            self.s7_lane_arbitrator = labeller.S7LaneArbitrator(2, 3)
            self.enabled = True

        def set_s7_inference_enabled(self, enabled):
            self.enabled = bool(enabled)

    heads = TinyHeads()
    stored_state = {
        name: tensor.detach().clone()
        for name, tensor in heads.state_dict().items()
        if not name.startswith('s7_lane_arbitrator.')}
    stored_state['native.weight'].fill_(0.25)
    payload = {'heads_state_dict': stored_state}
    labeller.load_heads_checkpoint_state(
        heads, payload, allow_lane_arbitration_initialization=True)
    assert heads.enabled is False
    assert torch.all(heads.native.weight == 0.25)
    adjustment = heads.s7_lane_arbitrator(
        torch.ones((1, 2)), torch.ones(1), torch.ones(()))
    assert torch.equal(adjustment, torch.zeros_like(adjustment))


def test_s7_quality_checkpoint_initialization_allows_only_new_suppressor_keys():
    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = torch.nn.Linear(2, 2)
            self.s7_readout = torch.nn.Linear(2, 2)
            self.s7_rpn_head = torch.nn.Linear(2, 2)
            self.s7_score_calibrator = labeller.S7ScoreCalibrator(-2.0)
            self.s7_quality_suppressor = labeller.S7QualitySuppressor(2, 3)
            self.enabled = True

        def set_s7_inference_enabled(self, enabled):
            self.enabled = bool(enabled)

    heads = TinyHeads()
    stored_state = {
        name: tensor.detach().clone()
        for name, tensor in heads.state_dict().items()
        if not name.startswith('s7_quality_suppressor.')}
    stored_state['native.weight'].fill_(0.25)
    payload = {'heads_state_dict': stored_state}
    labeller.load_heads_checkpoint_state(
        heads, payload, allow_quality_suppression_initialization=True)
    assert heads.enabled is False
    assert torch.all(heads.native.weight == 0.25)
    delta, _risk, _top = heads.s7_quality_suppressor(
        torch.ones((1, 2)), torch.ones(1), torch.ones(1), torch.ones(()))
    assert float(delta.item()) <= 0.0


def test_s7_temporal_checkpoint_initialization_allows_only_new_scorer_keys():
    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = torch.nn.Linear(2, 2)
            self.s7_readout = torch.nn.Linear(2, 2)
            self.s7_rpn_head = torch.nn.Linear(2, 2)
            self.s7_score_calibrator = labeller.S7ScoreCalibrator(-2.0)
            self.s7_temporal_scorer = (
                labeller.temporal.S7TemporalAssociationScorer())
            self.enabled = True

        def set_s7_inference_enabled(self, enabled):
            self.enabled = bool(enabled)

    heads = TinyHeads()
    stored_state = {
        name: tensor.detach().clone()
        for name, tensor in heads.state_dict().items()
        if not name.startswith('s7_temporal_scorer.')}
    stored_state['native.weight'].fill_(0.25)
    payload = {'heads_state_dict': stored_state}
    labeller.load_heads_checkpoint_state(
        heads, payload, allow_temporal_association_initialization=True)
    assert heads.enabled is False
    assert torch.all(heads.native.weight == 0.25)
    assert heads.s7_temporal_scorer.raw_weights.numel() == 6


def test_frozen_s7_component_loader_does_not_replace_native_or_calibrator(
        tmp_path):
    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = torch.nn.Linear(2, 2)
            self.s7_readout = torch.nn.Linear(2, 2)
            self.s7_rpn_head = torch.nn.Linear(2, 2)
            self.s7_score_calibrator = labeller.S7ScoreCalibrator(-2.0)
            self.enabled = False

        def set_s7_inference_enabled(self, enabled):
            self.enabled = bool(enabled)

    heads = TinyHeads()
    native_before = heads.native.weight.detach().clone()
    bias_before = heads.s7_score_calibrator.bias.detach().clone()
    component_readout = torch.nn.Linear(2, 2)
    component_rpn = torch.nn.Linear(2, 2)
    args = _args(
        tmp_path, s7_residual=True, s7_protected_merge=True,
        train_components='s7_merge')
    component_args = _args(
        tmp_path, s7_residual=True, train_components='s7_rpn')
    payload = dict(
        source_only=True, frozen_dinov2=True, in_channels=1024,
        epoch=2, best_epoch=0, s7_inference_enabled=True,
        training_protocol=dict(train_components='s7_rpn'),
        s7_architecture=labeller.s7_architecture(component_args),
        heads_state_dict={
            's7_readout.weight': component_readout.weight.detach().clone(),
            's7_readout.bias': component_readout.bias.detach().clone(),
            's7_rpn_head.weight': component_rpn.weight.detach().clone(),
            's7_rpn_head.bias': component_rpn.bias.detach().clone()})
    summary = labeller.load_frozen_s7_component(
        heads, payload, in_channels=1024, args=args)
    assert summary['epoch'] == 2
    assert heads.enabled is True
    assert torch.equal(heads.native.weight, native_before)
    assert torch.equal(heads.s7_score_calibrator.bias, bias_before)
    assert torch.equal(heads.s7_readout.weight, component_readout.weight)


def test_frozen_s7_component_loader_rejects_epoch0_fallback(tmp_path):
    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.s7_readout = torch.nn.Linear(2, 2)
            self.s7_rpn_head = torch.nn.Linear(2, 2)

    args = _args(
        tmp_path, s7_residual=True, s7_protected_merge=True,
        train_components='s7_merge')
    component_args = _args(
        tmp_path, s7_residual=True, train_components='s7_rpn')
    payload = dict(
        source_only=True, frozen_dinov2=True, in_channels=1024,
        epoch=0, s7_inference_enabled=False,
        training_protocol=dict(train_components='s7_rpn'),
        s7_architecture=labeller.s7_architecture(component_args),
        heads_state_dict={})
    with pytest.raises(RuntimeError, match='epoch-0 fallback'):
        labeller.load_frozen_s7_component(
            TinyHeads(), payload, in_channels=1024, args=args)


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


def test_source_selected_checkpoint_gate_requires_positive_selected_epoch():
    payload = dict(
        best_epoch=0,
        best_source_val_summary=dict(top1_hits=677, top1_mcml=3),
        best_source_small_val_summary=dict(top1_hits=303, top1_mcml=3),
        source_selection_gate_passed=False,
        source_exact_retention=None,
        training_protocol=dict(train_components='s7_temporal_association'))
    result = labeller.source_selected_checkpoint_gate(payload)
    assert result['passed'] is False
    assert result['checks']['positive_best_epoch'] is False
    assert result['checks']['stored_source_selection_gate'] is False


def test_source_selected_checkpoint_gate_requires_exact_retention_and_gate():
    payload = dict(
        best_epoch=1,
        best_source_val_summary=dict(top1_hits=688, top1_mcml=3),
        best_source_small_val_summary=dict(top1_hits=311, top1_mcml=3),
        source_selection_gate_passed=True,
        source_exact_retention=dict(
            baseline_correct_count=677, retained_correct_count=677,
            lost_correct_count=0),
        training_protocol=dict(train_components='s7_temporal_association'))
    result = labeller.source_selected_checkpoint_gate(payload)
    assert result['passed'] is True


def test_temporal_association_audit_reports_blocking_conditions():
    rows = [
        dict(
            metrics=dict(raw_unfiltered=dict(best_usable_rank=4)),
            candidate_merge=dict(source_top1_metrics=dict(
                native_s14=dict(top1_hit=False))),
            temporal_selection=dict(
                reason='native_fallback_pending_confirmation', reset=False,
                override=False, selected_source='native_s14',
                native_fallback_index=0, candidate_index=1,
                candidate_margin_ok=True, candidate_continuity_ok=True,
                candidate_override_ok=True)),
        dict(
            metrics=dict(raw_unfiltered=dict(best_usable_rank=None)),
            candidate_merge=dict(source_top1_metrics=dict(
                native_s14=dict(top1_hit=True))),
            temporal_selection=dict(
                reason='native_fallback_no_override_evidence', reset=False,
                override=False, selected_source='native_s14',
                native_fallback_index=0, candidate_index=1,
                candidate_margin_ok=False, candidate_continuity_ok=True,
                candidate_override_ok=False)),
    ]
    audit = labeller.summarize_temporal_association_audit(rows)
    assert audit['frame_count'] == 2
    assert audit['native_top1_wrong_count'] == 1
    assert audit['usable_candidate_when_native_wrong_count'] == 1
    assert audit['pending_confirmation_count'] == 1
    assert audit['candidate_override_ok_count'] == 1


def test_temporal_readonly_attribution_reports_stage_gains_and_losses():
    def evidence(hit):
        return dict(top1_hit=hit)

    rows = [
        dict(temporal_attribution=dict(
            fallback=evidence(False), quality_only=evidence(True),
            quality_ranked=dict(
                top1_hit=True, best_usable_rank=1,
                recall_at_20=True, recall_at_100=True),
            fused_candidate=evidence(True),
            margin_counterfactual=evidence(True),
            preconfirmation_counterfactual=evidence(True),
            final_selected=evidence(False), candidate_margin_ok=True,
            candidate_continuity_ok=True, candidate_override_ok=True,
            pending_confirmation=True)),
        dict(temporal_attribution=dict(
            fallback=evidence(True), quality_only=evidence(False),
            quality_ranked=dict(
                top1_hit=False, best_usable_rank=3,
                recall_at_20=True, recall_at_100=True),
            fused_candidate=evidence(False),
            margin_counterfactual=evidence(False),
            preconfirmation_counterfactual=evidence(False),
            final_selected=evidence(True), candidate_margin_ok=True,
            candidate_continuity_ok=True, candidate_override_ok=True,
            pending_confirmation=True)),
    ]
    audit = labeller.summarize_temporal_readonly_attribution(rows)
    preconfirmation = audit['stages']['preconfirmation_counterfactual']
    assert preconfirmation['top1_hits'] == 1
    assert preconfirmation['gained_vs_fallback_count'] == 1
    assert preconfirmation['lost_vs_fallback_count'] == 1
    assert audit['pending_confirmation']['candidate_gain_count'] == 1
    assert audit['pending_confirmation']['candidate_loss_count'] == 1
    assert audit['quality_ranked']['recall_at_100'] == 2
    assert audit['quality_ranked']['median_best_usable_rank'] == 2.0


def test_temporal_attribution_decision_requires_absolute_and_zero_loss(
        tmp_path):
    def summary(frame_count, fallback, final, preconfirmation, lost):
        stages = {
            name: dict(
                evaluated_count=frame_count, top1_hits=value,
                gained_vs_fallback_count=max(0, value - fallback),
                lost_vs_fallback_count=(lost if name ==
                                        'preconfirmation_counterfactual'
                                        else 0),
                net_gain_vs_fallback=value - fallback)
            for name, value in (
                ('fallback', fallback), ('quality_only', fallback),
                ('fused_candidate', fallback),
                ('margin_counterfactual', preconfirmation),
                ('preconfirmation_counterfactual', preconfirmation),
                ('final_selected', final))}
        attribution = dict(
            stages=stages, pending_confirmation={}, conditions={},
            quality_ranked={})
        return dict(
            frame_count=frame_count,
            temporal_association_audit=dict(
                readonly_attribution=attribution))

    args = _args(
        tmp_path, s7_source_min_full_top1=688,
        s7_source_min_small_top1=311)
    passed = labeller.build_source_temporal_attribution_audit(
        summary(738, 677, 681, 688, 0),
        summary(350, 303, 306, 311, 0), args,
        str(tmp_path / 'epoch04.pth'), 4)
    assert passed['confirmation_rule_revision_supported'] is True
    assert passed['final_shortfall'] == dict(full=7, small=5)
    failed = labeller.build_source_temporal_attribution_audit(
        summary(738, 677, 681, 688, 1),
        summary(350, 303, 306, 311, 0), args,
        str(tmp_path / 'epoch04.pth'), 4)
    assert failed['confirmation_rule_revision_supported'] is False
    assert failed['recommendation'].startswith('CLOSE_CURRENT')


def test_s7_merge_pair_loss_separates_retention_and_gain_cases():
    calibrator = labeller.S7ScoreCalibrator(initial_bias=0.0)
    retain = labeller.s7_merge_pair_losses(
        torch.tensor([0.2]), torch.tensor([0.8]),
        torch.tensor([0.7, -1.0]), torch.tensor([0.1, 0.8]),
        calibrator, margin=0.5)
    assert float(retain['retention'].item()) == pytest.approx(1.0)
    assert float(retain['gain'].item()) == pytest.approx(0.0)
    assert retain['retain_pair_count'] == 1
    assert retain['retention_active'] == 1
    assert retain['gain_active'] == 0
    gain = labeller.s7_merge_pair_losses(
        torch.tensor([0.8]), torch.tensor([0.1]),
        torch.tensor([0.2, 1.5]), torch.tensor([0.1, 0.8]),
        calibrator, margin=0.5)
    assert float(gain['retention'].item()) == pytest.approx(0.0)
    assert float(gain['gain'].item()) == pytest.approx(0.0)
    assert gain['gain_pair_count'] == 1
    assert gain['retention_active'] == 0
    assert gain['gain_active'] == 0


def test_s7_lane_retention_mines_topk_current_adjusted_wrong_candidates():
    adjusted = torch.tensor([1.2, 0.9, 2.0], requires_grad=True)
    result = labeller.s7_lane_pair_losses(
        torch.tensor([1.0]), torch.tensor([0.8]), adjusted,
        torch.tensor([0.1, 0.2, 0.8]), margin=0.5,
        hard_negatives=2)
    assert result['retain_pair_count'] == 2
    assert result['retention_active'] == 2
    assert float(result['retention'].item()) == pytest.approx(0.55)
    result['retention'].backward()
    assert adjusted.grad.tolist() == pytest.approx([0.5, 0.5, 0.0])


def test_s7_lane_gain_beats_native_and_strongest_wrong_s7():
    adjusted = torch.tensor([1.2, 1.5], requires_grad=True)
    result = labeller.s7_lane_pair_losses(
        torch.tensor([1.0]), torch.tensor([0.1]), adjusted,
        torch.tensor([0.8, 0.1]), margin=0.5,
        hard_negatives=4)
    assert result['gain_pair_count'] == 1
    assert result['gain_s7_competitor_count'] == 1
    assert result['gain_active'] == 1
    assert float(result['gain'].item()) == pytest.approx(0.8)
    result['gain'].backward()
    assert adjusted.grad.tolist() == pytest.approx([-1.0, 1.0])


def test_s7_quality_losses_suppress_only_competitive_wrong_lane():
    delta = torch.tensor(-0.1, requires_grad=True)
    risk_logit = torch.tensor(-2.0, requires_grad=True)
    result = labeller.s7_quality_suppression_losses(
        torch.tensor([1.0]), torch.tensor([0.8]),
        torch.tensor([1.2, 0.5]), torch.tensor([0.1, 0.8]),
        delta, risk_logit, margin=0.5)
    assert result['risk_pair_count'] == 1
    assert result['preserve_pair_count'] == 0
    assert result['retention_active'] == 1
    assert result['native_top1_riou'] == pytest.approx(0.8)
    assert result['s7_top1_riou'] == pytest.approx(0.1)
    assert float(result['retention'].item()) == pytest.approx(0.6)
    (result['risk'] + result['retention']).backward()
    assert float(delta.grad.item()) > 0.0
    assert float(risk_logit.grad.item()) < 0.0


def test_s7_quality_losses_preserve_usable_lane_without_promotion():
    delta = torch.tensor(-0.1, requires_grad=True)
    risk_logit = torch.tensor(-2.0, requires_grad=True)
    result = labeller.s7_quality_suppression_losses(
        torch.tensor([1.0]), torch.tensor([0.1]),
        torch.tensor([0.8]), torch.tensor([0.8]),
        delta, risk_logit, margin=0.5)
    assert result['risk_pair_count'] == 0
    assert result['preserve_pair_count'] == 1
    assert float(result['retention'].item()) == pytest.approx(0.0)
    result['preserve'].backward()
    assert float(risk_logit.grad.item()) > 0.0
    assert delta.grad is None


def test_s7_quality_support_audit_reports_zero_risk_and_exclusions():
    rows = [
        dict(
            frame_key='train|seq_a|1', split='train', seq='seq_a',
            risk_pair=False, preserve_pair=True,
            native_top1_correct=False, s7_top1_correct=True,
            native_top1_riou=0.1, s7_top1_riou=0.8,
            base_gap=0.2, s7_candidate_count=10),
        dict(
            frame_key='train|seq_a|2', split='train', seq='seq_a',
            risk_pair=False, preserve_pair=False,
            native_top1_correct=True, s7_top1_correct=False,
            native_top1_riou=0.7, s7_top1_riou=0.2,
            base_gap=-0.6, s7_candidate_count=10),
        dict(
            frame_key='train|seq_b|3', split='train', seq='seq_b',
            risk_pair=False, preserve_pair=False,
            native_top1_correct=False, s7_top1_correct=False,
            native_top1_riou=0.2, s7_top1_riou=0.1,
            base_gap=0.9, s7_candidate_count=10),
    ]
    result = labeller.summarize_s7_quality_support_rows(
        rows, margin=0.5, riou_thr=0.5)
    assert result['status'] == 'FAIL_ZERO_RISK_SUPPORT'
    assert result['training_allowed'] is False
    assert result['training_skipped'] is True
    assert result['risk_pair_count'] == 0
    assert result['preserve_pair_count'] == 1
    assert result['s7_top1_wrong_count'] == 2
    assert result['native_correct_s7_wrong_count'] == 1
    assert (
        result['native_correct_s7_wrong_excluded_by_margin_count'] == 1)
    assert result['s7_wrong_sequence_counts'] == {
        'train|seq_a': 1, 'train|seq_b': 1}
    assert result['s7_wrong_frames'][0]['excluded_by_margin'] is True
    assert result['s7_wrong_frames'][0][
        'suppression_to_match_native'] == pytest.approx(0.0)


def test_s7_quality_support_audit_allows_a_matching_risk_pair():
    rows = [dict(
        frame_key='train|seq|1', split='train', seq='seq',
        risk_pair=True, preserve_pair=False,
        native_top1_correct=True, s7_top1_correct=False,
        native_top1_riou=0.8, s7_top1_riou=0.1,
        base_gap=-0.4, s7_candidate_count=10)]
    result = labeller.summarize_s7_quality_support_rows(
        rows, margin=0.5, riou_thr=0.5)
    assert result['status'] == 'PASS'
    assert result['training_allowed'] is True
    assert result['risk_pair_count'] == 1
    assert result['s7_wrong_frames'][0][
        'suppression_for_margin'] == pytest.approx(0.1)


def test_s7_lane_uses_canonical_optimization_loss_metadata():
    assert labeller.optimization_loss_component_names(
        's7_lane_arbitration') == [
            'loss_s7_lane_retention', 'loss_s7_lane_gain',
            'loss_s7_lane_prior']


def test_s7_quality_uses_no_gain_or_promotion_loss():
    names = labeller.optimization_loss_component_names(
        's7_quality_suppression')
    assert names == [
        'loss_s7_quality_risk', 'loss_s7_quality_preserve',
        'loss_s7_quality_retention', 'loss_s7_quality_prior']
    assert not any('gain' in name or 'promotion' in name for name in names)
    losses = {name: torch.tensor(1.0, requires_grad=True) for name in names}
    total = labeller.optimization_loss_total(
        losses, 's7_quality_suppression')
    assert float(total.item()) == pytest.approx(4.0)


def test_s7_lane_train_epoch_replays_gain_frames_source_only(
        tmp_path, monkeypatch):
    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rpn_head = torch.nn.Identity()
            self.roi_head = torch.nn.Identity()
            self.s7_readout = torch.nn.Identity()
            self.s7_rpn_head = torch.nn.Identity()
            self.s7_score_calibrator = torch.nn.Identity()
            self.s7_lane_arbitrator = torch.nn.Linear(1, 1, bias=False)

        def forward_s7_lane_arbitration_train(
                self, _feature, img_meta, _gt_boxes, **_kwargs):
            parameter = self.s7_lane_arbitrator.weight.sum()
            zero = parameter * 0.0
            gain_pair = int(img_meta['frame'] == 1)
            gain = (parameter - 1.0).square() if gain_pair else zero
            return dict(
                loss_s7_lane_retention=zero,
                loss_s7_lane_gain=gain,
                loss_s7_lane_prior=zero,
                s7_lane_retain_pair_count=1 - gain_pair,
                s7_lane_gain_pair_count=gain_pair,
                s7_lane_retention_active=0,
                s7_lane_gain_active=gain_pair)

    def fake_prepare(_dino, record, _args, _dino_device, _head_device):
        return (
            torch.zeros(1), {'frame': int(record['frame'])},
            torch.zeros((0, 5)), torch.zeros(0, dtype=torch.long),
            np.zeros((0, 5), dtype=np.float32), True)

    monkeypatch.setattr(labeller, 'prepare_record', fake_prepare)
    heads = TinyHeads()
    optimizer = torch.optim.SGD(
        heads.s7_lane_arbitrator.parameters(), lr=0.01)
    args = _args(
        tmp_path, train_components='s7_lane_arbitration',
        warmup_iters=0, s7_lane_gain_repeat=3,
        s7_lane_hard_negatives=4)
    records = [
        dict(split='train', seq='seq', frame=1),
        dict(split='train', seq='seq', frame=2)]
    summary = labeller.train_epoch(
        None, heads, optimizer, records, epoch=1, global_step=0,
        args=args, dino_device=torch.device('cpu'),
        head_device=torch.device('cpu'))
    assert summary['count'] == 4
    assert summary['s7_lane_gain_replay'] == dict(
        repeat=3, unique_gain_frame_count=1, extra_record_count=2,
        source_train_only=True)
    assert summary['optimized_components'] == [
        'loss_s7_lane_retention', 'loss_s7_lane_gain',
        'loss_s7_lane_prior']


def test_s7_conflict_summary_keeps_compact_per_source_evidence():
    merge = dict(
        raw_top1_source='supplement_s7',
        source_top1_metrics={
            'native_s14': dict(top1_hit=True, top1_riou=0.8, top1_score=0.7),
            'supplement_s7': dict(
                top1_hit=False, top1_riou=0.1, top1_score=0.8)},
        source_pre_nms_top_log_odds={
            'native_s14': 0.9, 'supplement_s7_raw': 2.0,
            'supplement_s7_calibrated': 1.2},
        s7_affine_scale=0.9, s7_affine_bias=-0.6)
    rows = [dict(
        split='val', seq='seq', frame=1, candidate_merge=merge,
        metrics=dict(
            top1_hit=False, top1_riou=0.1, top1_score=0.8))]
    summary = labeller.source_merge_conflict_summary(['val|seq|1'], rows)
    assert summary['gained'] == []
    assert summary['lost'][0]['frame_key'] == 'val|seq|1'
    assert summary['lost'][0]['merged_top1']['source'] == 'supplement_s7'
    assert summary['lost'][0]['source_top1_metrics']['native_s14'][
        'top1_hit'] is True


def test_s7_calibration_state_records_effective_parameters():
    heads = argparse.Namespace(
        s7_score_calibrator=labeller.S7ScoreCalibrator(-2.0))
    state = labeller.s7_calibration_state(heads)
    assert state == pytest.approx(dict(scale=1.0, bias=-2.0, prior_loss=0.0))


def test_source_conflict_spec_reads_only_one_epoch_changed_frames(tmp_path):
    result_path = tmp_path / 'train_result.json'
    result_path.write_text(json.dumps(dict(source=dict(history=[
        dict(epoch=1, source_exact_retention=dict(
            lost_frame_keys=['val|seq|2'],
            gained_frame_keys=['val|seq|3', 'val|seq|1']))]))))
    spec = labeller.load_source_conflict_spec(str(result_path), epoch=1)
    assert spec['lost_frame_keys'] == ['val|seq|2']
    assert spec['gained_frame_keys'] == ['val|seq|1', 'val|seq|3']
    assert spec['frame_keys'] == ['val|seq|1', 'val|seq|2', 'val|seq|3']


def test_validate_source_conflict_audit_is_eval_only_and_target_free(tmp_path):
    result_path = tmp_path / 'train_result.json'
    result_path.write_text('{}')
    eval_path = tmp_path / 'epoch01.pth'
    eval_path.write_bytes(b'checkpoint')
    args = _args(
        tmp_path, s7_residual=True, train_components='s7_merge', epochs=4,
        lr_steps=[2, 3], selection_epochs=[1, 2, 3, 4],
        eval_only_checkpoint=str(eval_path), skip_target_eval=True,
        source_conflict_result_json=str(result_path), source_conflict_epoch=1)
    labeller.validate_args(args)
    args.skip_target_eval = False
    with pytest.raises(ValueError, match='skip-target-eval'):
        labeller.validate_args(args)


def test_validate_temporal_attribution_is_fixed_eval_only_and_target_free(
        tmp_path):
    checkpoint = tmp_path / 'labeller_epoch_04_source_only.pth'
    checkpoint.write_bytes(b'checkpoint')
    common = dict(
        s7_residual=True, s7_temporal_association=True,
        s7_temporal_quality_head=True,
        train_components='s7_temporal_association',
        source_train_datasets=['train:train'],
        source_val_datasets=['val:val'],
        source_small_repeat=1, s7_source_min_full_top1=688,
        s7_source_min_small_top1=311,
        epochs=4, lr_steps=[2, 3], selection_epochs=[1, 2, 3, 4],
        eval_only_checkpoint=str(checkpoint), skip_target_eval=True,
        source_temporal_attribution_audit=True,
        source_temporal_attribution_epoch=4)
    labeller.validate_args(_args(tmp_path, **common))
    with pytest.raises(ValueError, match='eval-only-checkpoint'):
        labeller.validate_args(_args(
            tmp_path, **dict(common, eval_only_checkpoint=None)))
    with pytest.raises(ValueError, match='skip-target-eval'):
        labeller.validate_args(_args(
            tmp_path, **dict(common, skip_target_eval=False)))
    with pytest.raises(ValueError, match='quality-head'):
        labeller.validate_args(_args(
            tmp_path, **dict(common, s7_temporal_quality_head=False)))


def test_validate_immediate_override_audit_is_explicit_and_readonly(tmp_path):
    checkpoint = tmp_path / 'labeller_epoch_04_source_only.pth'
    checkpoint.write_bytes(b'checkpoint')
    args = _args(
        tmp_path, s7_residual=True, s7_temporal_association=True,
        s7_temporal_quality_head=True,
        train_components='s7_temporal_association',
        source_train_datasets=['train:train'],
        source_val_datasets=['val:val'], source_small_repeat=1,
        s7_source_min_full_top1=688, s7_source_min_small_top1=311,
        epochs=4, lr_steps=[2, 3], selection_epochs=[1, 2, 3, 4],
        eval_only_checkpoint=str(checkpoint), skip_target_eval=True,
        source_temporal_immediate_override_audit=True,
        source_temporal_attribution_epoch=4)
    labeller.validate_args(args)
    assert labeller.temporal_runtime_min_confirmations(args) == 1
    with pytest.raises(ValueError, match='mutually exclusive'):
        conflicting = vars(args).copy()
        conflicting['source_temporal_attribution_audit'] = True
        labeller.validate_args(_args(tmp_path, **conflicting))


def test_s7_config_declares_explicit_supplement_checkpoint_and_head():
    import runpy
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(str(root /
        'crane_project/configs/crane_symeood_scoped_dino_lowlight_s7_v1.py'))
    assert config['model']['dino_rescue']['head']['s7_residual'] is True
    assert config['model']['dino_rescue']['head']['s7_channels'] == 128
    assert config['model']['dino_head_checkpoint'].endswith(
        'dino_teacher_s7_residual_v1/labeller_best_source_only.pth')


def test_s7_retention_merge_config_uses_new_source_gated_checkpoint():
    import runpy
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(str(root / 'crane_project/configs/'
        'crane_symeood_scoped_dino_lowlight_s7_retention_merge_v1.py'))
    head = config['model']['dino_rescue']['head']
    assert head['s7_protected_merge'] is True
    assert head['s7_merge_init_bias'] == pytest.approx(-2.0)
    assert config['model']['dino_head_checkpoint'].endswith(
        'dino_teacher_s7_retention_merge_v1/labeller_best_source_only.pth')


def test_s7_temporal_relative_quality_config_declares_phase_two_b_protocol():
    import runpy
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(str(root / 'crane_project/configs/'
        'crane_symeood_scoped_dino_lowlight_s7_temporal_relative_quality_v1.py'))
    head = config['model']['dino_rescue']['head']
    training = config['s7_temporal_quality_training']
    assert head['s7_temporal_quality_head'] is True
    assert head['s7_temporal_relative_quality'] is True
    assert head['s7_temporal_min_confirmations'] == 1
    assert training['base_epoch'] == 4
    assert training['relative_quality'] is True
    assert training['source_only'] is True
    assert training['target_read'] is False
    assert training['positive_promotion'] is False
    assert training['gain_replay'] is False


def test_s7_lane_arbitration_config_freezes_epoch1_merge_base():
    import runpy
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(str(root / 'crane_project/configs/'
        'crane_symeood_scoped_dino_lowlight_s7_lane_arbitration_v1.py'))
    head = config['model']['dino_rescue']['head']
    assert head['s7_protected_merge'] is True
    assert head['s7_lane_arbitration'] is True
    assert head['s7_lane_hidden'] == 32
    assert config['model']['dino_head_checkpoint'].endswith(
        'dino_teacher_s7_lane_arbitration_v1/'
        'labeller_best_source_only.pth')
    assert config['s7_lane_training']['base_checkpoint'].endswith(
        'dino_teacher_s7_retention_merge_v1/'
        'labeller_epoch_01_source_only.pth')
    assert config['s7_lane_training']['source_gate'] == 'exact_retention'


def test_s7_lane_arbitration_v2_config_uses_dynamic_source_only_mining():
    import runpy
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(str(root / 'crane_project/configs/'
        'crane_symeood_scoped_dino_lowlight_s7_lane_arbitration_v2.py'))
    training = config['s7_lane_training']
    assert training['hard_negative_ranking'] == (
        'current_adjusted_s7_log_odds')
    assert training['hard_negatives'] == 4
    assert training['gain_repeat'] == 8
    assert training['source_train_only'] is True
    assert config['model']['dino_head_checkpoint'].endswith(
        'dino_teacher_s7_lane_arbitration_v2/'
        'labeller_best_source_only.pth')


def test_s7_quality_suppression_config_locks_affine_base_and_formal_gate():
    import runpy
    root = pathlib.Path(__file__).resolve().parents[1]
    config = runpy.run_path(str(root / 'crane_project/configs/'
        'crane_symeood_scoped_dino_lowlight_s7_quality_suppression_v1.py'))
    head = config['model']['dino_rescue']['head']
    training = config['s7_quality_training']
    assert head['s7_quality_suppression'] is True
    assert head['s7_lane_arbitration'] is False
    assert training['base_checkpoint'].endswith(
        'dino_teacher_s7_retention_merge_v1/'
        'labeller_epoch_01_source_only.pth')
    assert training['source_gate'] == dict(
        exact_retention=True, min_full_top1=688,
        min_small_top1=311, max_mcml=3)
    assert training['adjustment_range'] == [-2.0, 0.0]
    assert training['source_support_preflight'] == dict(
        exact_training_risk_miner=True, minimum_risk_pairs=1,
        zero_risk_action='skip_optimization_and_keep_epoch_0',
        report_all_s7_wrong_frames=True)
    assert training['positive_promotion'] is False
    assert training['gain_replay'] is False
    assert training['target_gate'] == 'formal_source_gate_only'


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


def test_top1_temporal_geometry_metrics_match_project_dfr_aci_definitions():
    rows = [
        dict(seq='a', frame=1, detections=[
            [0, 0, 6, 8, 0.0, 0.9]]),
        dict(seq='a', frame=2, detections=[
            [0, 0, 12, 16, np.deg2rad(17.5), 0.9]]),
        dict(seq='a', frame=4, detections=[
            [0, 0, 24, 32, np.deg2rad(35.0), 0.9]]),
    ]
    metrics = labeller.top1_temporal_geometry_metrics(rows, 35.0)
    assert metrics['dfr_fraction_per_frame'] == pytest.approx(1.0)
    assert metrics['dfr_percent_per_frame'] == pytest.approx(100.0)
    assert metrics['aci'] == pytest.approx(0.5)
    assert metrics['transition_count'] == 1


def test_temporal_source_gate_rejects_dfr_or_aci_regression(tmp_path):
    args = _args(
        tmp_path, train_components='s7_temporal_association',
        s7_source_min_full_top1=688, s7_source_min_small_top1=311)
    baseline = dict(
        top1_hits=677, top1_mcml=3,
        top1_dfr_fraction_per_frame=0.02, top1_aci=0.95)
    baseline_small = dict(top1_hits=303, top1_mcml=3)
    retention = dict(
        baseline_correct_count=677, retained_correct_count=677,
        lost_correct_count=0)
    candidate = dict(
        top1_hits=688, top1_mcml=3,
        top1_dfr_fraction_per_frame=0.03, top1_aci=0.94)
    candidate_small = dict(top1_hits=311, top1_mcml=3)
    result = labeller.s7_source_selection_gate(
        baseline, baseline_small, candidate, candidate_small,
        retention, args)
    assert result['checks']['source_temporal_metrics_available'] is True
    assert result['checks']['source_dfr_nonregression'] is False
    assert result['checks']['source_aci_nonregression'] is False
    assert result['passed'] is False


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


def test_source_training_progress_records_zero_risk_preflight_stop(tmp_path):
    args = argparse.Namespace(
        out_json=str(tmp_path / 'train_result.json'),
        train_components='s7_quality_suppression', epochs=4)
    audit = dict(
        status='FAIL_ZERO_RISK_SUPPORT', training_allowed=False,
        training_skipped=True, risk_pair_count=0, target_read=False)
    output_path, replacements = labeller.write_source_training_progress(
        args, completed_epoch=0, best_epoch=0,
        best_path=str(tmp_path / 'best.pth'),
        latest_path=str(tmp_path / 'best.pth'),
        baseline_summary=dict(top1_hits=677),
        baseline_small_summary=dict(top1_hits=303),
        best_summary=dict(top1_hits=677),
        best_small_summary=dict(top1_hits=303), history=[],
        status='SOURCE_ONLY_TRAINING_SKIPPED_ZERO_S7_QUALITY_RISK_SUPPORT',
        s7_quality_support_audit=audit)
    with open(output_path, 'r') as handle:
        payload = json.load(handle)
    assert replacements == 0
    assert payload['completed_epoch'] == 0
    assert payload['best_epoch'] == 0
    assert payload['target_read'] is False
    assert payload['status'].endswith('ZERO_S7_QUALITY_RISK_SUPPORT')
    assert payload['s7_quality_support_audit'] == audit


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
