import argparse

import pytest

from crane_project.tools import dino_teacher_far_distance_candidate_audit as audit


def _args(tmp_path, **overrides):
    labeller_checkpoint = tmp_path / 'labeller.pth'
    dino_checkpoint = tmp_path / 'dino.pth'
    labeller_checkpoint.write_bytes(b'labeller')
    dino_checkpoint.write_bytes(b'dino')
    values = dict(
        seed=0, source_split='val', source_seq='real_seq07',
        source_val_modulus=5, target_split='test',
        target_seq='real_seq02', target_start=2, target_end=41,
        dino_gpus=[1, 2], head_gpu=0,
        patch_size=14, rpn_feat_channels=256, roi_fc_channels=1024,
        roi_samples=256, proposal_count=2000, max_detections=2000,
        dino_height=600, dino_max_long_side=1333,
        labeller_checkpoint=str(labeller_checkpoint),
        dinov2_checkpoint=str(dino_checkpoint),
        out_json=str(tmp_path / 'result.json'))
    values.update(overrides)
    return argparse.Namespace(**values)


def test_validate_requires_fixed_far_distance_slice(tmp_path):
    audit.validate_args(_args(tmp_path))
    with pytest.raises(ValueError, match='2..41'):
        audit.validate_args(_args(tmp_path, target_start=1))


def test_validate_refuses_existing_result(tmp_path):
    args = _args(tmp_path)
    with open(args.out_json, 'w', encoding='utf-8') as handle:
        handle.write('{}')
    with pytest.raises(ValueError, match='overwrite'):
        audit.validate_args(args)


def test_decision_requires_source_control():
    source = dict(frame_count=45, top1_hits=35)
    target = dict(frame_count=40, geometry_eligible_count=40,
                  raw_unfiltered_geometry_eligible_count=40)
    assert audit.make_decision(source, target) == (
        'AUDIT_INVALID_SOURCE_CONTROL')


def test_decision_authorizes_ranking_training_only_with_valid_geometry():
    source = dict(frame_count=45, top1_hits=45)
    target = dict(frame_count=40, geometry_eligible_count=32,
                  raw_unfiltered_geometry_eligible_count=32)
    assert audit.make_decision(source, target) == (
        'AUTHORIZE_SOURCE_ONLY_FAR_SCALE_RANKING_TRAINING')


def test_decision_detects_border_filter_conflict():
    source = dict(frame_count=45, top1_hits=45)
    target = dict(frame_count=40, geometry_eligible_count=20,
                  raw_unfiltered_geometry_eligible_count=32)
    assert audit.make_decision(source, target) == (
        'FAR_GEOMETRY_BORDER_FILTER_CONFLICT')


def test_decision_rejects_classifier_training_without_candidates():
    source = dict(frame_count=45, top1_hits=45)
    target = dict(frame_count=40, geometry_eligible_count=10,
                  raw_unfiltered_geometry_eligible_count=12)
    assert audit.make_decision(source, target) == (
        'DINO_FAR_DISTANCE_CANDIDATE_GENERATION_INSUFFICIENT')
