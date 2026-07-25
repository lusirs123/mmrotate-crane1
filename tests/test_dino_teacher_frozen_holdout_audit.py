import argparse

import pytest

from crane_project.tools import dino_teacher_frozen_holdout_audit as holdout


def _args(tmp_path, **overrides):
    labeller_checkpoint = tmp_path / 'labeller.pth'
    dino_checkpoint = tmp_path / 'dino.pth'
    labeller_checkpoint.write_bytes(b'labeller')
    dino_checkpoint.write_bytes(b'dino')
    values = dict(
        confirm_frozen_holdout=True, seed=0,
        source_split='val', source_seq='real_seq07',
        source_val_modulus=5, holdout_split='test',
        holdout_seqs=['real_seq03', 'sim_seq09'],
        dino_gpus=[1, 2], head_gpu=0,
        patch_size=14, rpn_feat_channels=256, roi_fc_channels=1024,
        roi_samples=256, proposal_count=2000, max_detections=2000,
        dino_height=600, dino_max_long_side=1333,
        labeller_checkpoint=str(labeller_checkpoint),
        dinov2_checkpoint=str(dino_checkpoint),
        out_json=str(tmp_path / 'result.json'))
    values.update(overrides)
    return argparse.Namespace(**values)


def test_holdout_protocol_requires_both_frozen_sequences(tmp_path):
    holdout.validate_args(_args(tmp_path))
    with pytest.raises(ValueError, match='together'):
        holdout.validate_args(_args(
            tmp_path, holdout_seqs=['real_seq03']))
    with pytest.raises(ValueError, match='confirm'):
        holdout.validate_args(_args(
            tmp_path, confirm_frozen_holdout=False))


def test_complete_holdout_rejects_missing_frame():
    records = [dict(frame=frame) for frame in range(1, 201)]
    holdout.validate_complete_holdout('real_seq03', records)
    with pytest.raises(RuntimeError, match='incomplete'):
        holdout.validate_complete_holdout('real_seq03', records[:-1])


def test_fixed_holdout_gate_uses_rate_and_mcml():
    assert holdout.holdout_passes(dict(
        frame_count=200, top1_hits=160, top1_mcml=5))
    assert not holdout.holdout_passes(dict(
        frame_count=200, top1_hits=159, top1_mcml=5))
    assert not holdout.holdout_passes(dict(
        frame_count=200, top1_hits=190, top1_mcml=6))


def test_source_control_gate_is_fixed_to_eighty_percent():
    assert holdout.source_control_passes(dict(
        frame_count=45, top1_hits=36))
    assert not holdout.source_control_passes(dict(
        frame_count=45, top1_hits=35))
