import copy
import argparse
import json

from crane_project.tools import dino_teacher_s7_relaxed_target_audit as audit


def _source_result(lost=1, gained=12, retained=676):
    return dict(
        target_dev=None,
        isolation=dict(
            train_components='s7_merge',
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_labels_used_for_evaluation_only=False),
        source=dict(history=[dict(
            epoch=1,
            source_val=dict(top1_hits=688, top1_mcml=3),
            source_small_val=dict(top1_hits=311, top1_mcml=3),
            source_exact_retention=dict(
                lost_correct_count=lost, gained_correct_count=gained,
                retained_correct_count=retained))]))


def _row(frame, hit, mcml_score=0.9):
    metrics = dict(
        top1_hit=hit, top1_riou=(0.7 if hit else 0.2),
        top1_score=mcml_score, best_usable_rank=(1 if hit else None),
        deployment_top1_hit=hit, deployment_silence=False,
        raw_detection_count=1, invalid_border_filtered_count=0,
        valid_detection_count=1, gt_near_border=False)
    metrics['raw_unfiltered'] = dict(
        top1_hit=metrics['top1_hit'], top1_riou=metrics['top1_riou'],
        top1_score=metrics['top1_score'],
        best_usable_rank=metrics['best_usable_rank'])
    return dict(
        split='test', seq='seq', frame=frame,
        candidate_merge=None, metrics=metrics)


def test_relaxed_source_gate_accepts_fixed_epoch_one_frontier():
    result = audit.relaxed_source_gate(_source_result(), epoch=1)
    assert result['passed']
    assert result['gain_loss_ratio'] == 12.0
    assert result['checks']['source_result_target_not_read']


def test_relaxed_source_gate_does_not_replace_source_isolation():
    result = _source_result()
    result['isolation']['target_used_for_checkpoint_selection'] = True
    gated = audit.relaxed_source_gate(result, epoch=1)
    assert not gated['passed']
    assert not gated['checks']['source_result_target_not_read']


def test_relaxed_source_gate_still_rejects_two_lost_frames():
    gated = audit.relaxed_source_gate(
        _source_result(lost=2, gained=12, retained=675), epoch=1)
    assert not gated['passed']
    assert not gated['checks']['lost_correct_within_bound']


def test_small_slice_requires_strict_gain_but_far_only_nonregression():
    baseline = [_row(1, True), _row(2, False)]
    unchanged = copy.deepcopy(baseline)
    far = audit.compare_slice('seq02_far', baseline, unchanged)
    small = audit.compare_slice('seq03_small', baseline, unchanged)
    assert far['passed']
    assert not small['passed']

    improved = [_row(1, True), _row(2, True)]
    small = audit.compare_slice('seq03_small', baseline, improved)
    assert small['passed']
    assert small['delta_top1'] == 1
    assert small['gained_frame_keys'] == ['test|seq|2']
    assert small['lost_frame_keys'] == []
    assert small['baseline_rows'] == baseline
    assert small['candidate_rows'] == improved


def test_validate_args_rejects_duplicate_target_slice_names(tmp_path):
    source_result = tmp_path / 'train_result.json'
    source_result.write_text('{}')
    paths = {}
    for name in ('baseline_checkpoint', 'candidate_checkpoint',
                 'dinov2_checkpoint'):
        path = tmp_path / name
        path.write_bytes(b'x')
        paths[name] = str(path)
    duplicate = 'seq02_far:test:real_seq02:2:41'
    args = argparse.Namespace(
        source_result_json=str(source_result), source_epoch=1,
        baseline_checkpoint=paths['baseline_checkpoint'],
        candidate_checkpoint=paths['candidate_checkpoint'],
        dinov2_checkpoint=paths['dinov2_checkpoint'],
        out_json=str(tmp_path / 'result.json'), dino_gpus=[1, 2], head_gpu=0,
        seed=0, dino_height=600, dino_max_long_side=1333, patch_size=14,
        rpn_feat_channels=256, roi_fc_channels=1024, roi_samples=256,
        proposal_count=2000, max_detections=2000, roi_nms_iou_thr=0.5,
        s7_channels=128, s7_rpn_feat_channels=128, s7_proposal_count=500,
        s7_nms_pre=2000, s7_anchor_sizes=[16, 32, 64, 128, 256],
        s7_lane_hidden=32, s7_lane_max_adjustment=2.0,
        target_slices=[duplicate, duplicate, duplicate])
    try:
        audit.validate_args(args)
    except ValueError as error:
        assert 'three unique target slices' in str(error)
    else:
        raise AssertionError('duplicate target slices were accepted')


def test_validate_args_detects_lane_candidate_from_source_result(tmp_path):
    source_result = tmp_path / 'train_result.json'
    result = _source_result(lost=0, gained=13, retained=677)
    result['isolation']['train_components'] = 's7_lane_arbitration'
    source_result.write_text(json.dumps(result))
    paths = {}
    for name in ('baseline_checkpoint', 'candidate_checkpoint',
                 'dinov2_checkpoint'):
        path = tmp_path / name
        path.write_bytes(b'x')
        paths[name] = str(path)
    args = argparse.Namespace(
        source_result_json=str(source_result), source_epoch=1,
        baseline_checkpoint=paths['baseline_checkpoint'],
        candidate_checkpoint=paths['candidate_checkpoint'],
        dinov2_checkpoint=paths['dinov2_checkpoint'],
        out_json=str(tmp_path / 'result.json'), dino_gpus=[1, 2], head_gpu=0,
        seed=0, dino_height=600, dino_max_long_side=1333, patch_size=14,
        rpn_feat_channels=256, roi_fc_channels=1024, roi_samples=256,
        proposal_count=2000, max_detections=2000, roi_nms_iou_thr=0.5,
        s7_channels=128, s7_rpn_feat_channels=128, s7_proposal_count=500,
        s7_nms_pre=2000, s7_anchor_sizes=[16, 32, 64, 128, 256],
        s7_lane_hidden=32, s7_lane_max_adjustment=2.0,
        target_slices=None)
    audit.validate_args(args)
    assert args.train_components == 's7_lane_arbitration'
    assert args.s7_lane_arbitration is True
