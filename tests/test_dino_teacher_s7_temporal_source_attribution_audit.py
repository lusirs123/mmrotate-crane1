import argparse
import json
import sys

import pytest

from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import (
    dino_teacher_s7_temporal_source_attribution_audit as audit)


def _source_result(candidate_path):
    checks = dict(
        exact_old_correct_retention=True,
        full_top1_nonregression=True,
        full_top1_absolute=True,
        small_top1_nonregression=True,
        small_top1_absolute=True,
        full_mcml_absolute=True,
        small_mcml_absolute=True,
        source_temporal_metrics_available=True,
        source_dfr_nonregression=True,
        source_aci_nonregression=True)
    retention = dict(
        baseline_correct_count=677, retained_correct_count=677,
        lost_correct_count=0, gained_correct_count=14,
        candidate_correct_count=691, lost_frame_keys=[],
        gained_frame_keys=[])
    return dict(
        protocol_version=20,
        source_selected_checkpoint=str(candidate_path),
        protocol=dict(s7_temporal_association=dict(
            candidate_quality_head=True, relative_quality=True,
            min_confirmations=1, target_read=False)),
        isolation=dict(
            train_components='s7_temporal_association',
            dino_parameters_unchanged=True,
            frozen_head_parameters_unchanged=True,
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_labels_used_for_evaluation_only=False),
        source=dict(
            best_epoch=4,
            history=[dict(
                epoch=4, selected_as_best=True, checkpoint_saved=True,
                source_selection_gate_passed=True,
                source_retention_passed=True,
                source_selection_gate=dict(checks=checks, passed=True),
                s7_source_gate=dict(checks=checks, passed=True),
                source_val=dict(
                    frame_count=738, top1_hits=691, top1_mcml=3,
                    recall_at_100=738, top1_dfr_fraction_per_frame=0.07,
                    top1_aci=0.83),
                source_small_val=dict(
                    frame_count=350, top1_hits=312, top1_mcml=3,
                    recall_at_100=350),
                source_exact_retention=retention)]),
        target_dev=None)


def _checkpoint_payload():
    return dict(
        source_only=True, frozen_dinov2=True, epoch=4, best_epoch=4,
        s7_inference_enabled=True, source_selection_gate_passed=True,
        best_source_val_summary=dict(top1_hits=691, top1_mcml=3),
        best_source_small_val_summary=dict(top1_hits=312, top1_mcml=3),
        source_exact_retention=dict(
            baseline_correct_count=677, retained_correct_count=677,
            lost_correct_count=0),
        s7_architecture=dict(
            temporal_association=True, temporal_quality_head=True,
            temporal_min_confirmations=1),
        training_protocol=dict(
            train_components='s7_temporal_association',
            s7_temporal_association=dict(
                relative_quality=True, candidate_quality_head=True,
                target_read=False)))


def _args(tmp_path, candidate):
    source = tmp_path / 'train_result.json'
    source.write_text(json.dumps(_source_result(candidate)))
    checkpoint = tmp_path / 'candidate.pth'
    checkpoint.write_bytes(b'checkpoint')
    dino = tmp_path / 'dino.pth'
    dino.write_bytes(b'dino')
    return argparse.Namespace(
        seed=0, dinov2_model='dinov2_vitl14',
        source_result_json=str(source), eval_only_checkpoint=str(checkpoint),
        dinov2_repo=str(tmp_path), dinov2_checkpoint=str(dino),
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=512,
        out_json=str(tmp_path / 'out' / 'attribution_result.json'),
        data_root='data', feature_cache_dir=str(tmp_path / 'cache'))


def test_validate_args_accepts_only_source_selected_epoch_four(tmp_path,
                                                                monkeypatch):
    candidate = tmp_path / 'candidate.pth'
    candidate.write_bytes(b'checkpoint')
    args = _args(tmp_path, candidate)
    monkeypatch.setattr(
        audit.fixed_target.torch, 'load',
        lambda path, map_location: _checkpoint_payload())
    audit.validate_args(args)
    assert args.source_gate['passed'] is True
    assert args.checkpoint_gate['passed'] is True


def test_validate_args_rejects_target_result_or_nonselected_checkpoint(
        tmp_path, monkeypatch):
    candidate = tmp_path / 'candidate.pth'
    candidate.write_bytes(b'checkpoint')
    args = _args(tmp_path, candidate)
    monkeypatch.setattr(
        audit.fixed_target.torch, 'load',
        lambda path, map_location: _checkpoint_payload())
    result = _source_result(candidate)
    result['target_dev'] = dict(summary={})
    with open(args.source_result_json, 'w') as handle:
        json.dump(result, handle)
    with pytest.raises(ValueError, match='source_result_target_not_read'):
        audit.validate_args(args)

    args = _args(tmp_path, candidate)
    args.eval_only_checkpoint = str(tmp_path / 'other.pth')
    (tmp_path / 'other.pth').write_bytes(b'checkpoint')
    with pytest.raises(ValueError, match='source_selected_checkpoint'):
        audit.validate_args(args)


def test_locked_argv_is_readonly_and_target_free(tmp_path):
    candidate = tmp_path / 'candidate.pth'
    args = _args(tmp_path, candidate)
    argv = audit.build_locked_labeller_argv(args)
    assert '--eval-only-checkpoint' in argv
    assert '--skip-target-eval' in argv
    assert '--source-temporal-attribution-audit' in argv
    assert '--source-temporal-attribution-epoch' in argv
    assert '--init-checkpoint' not in argv
    assert '--resume-checkpoint' not in argv
    assert '--target-start' not in argv
    assert '--target-end' not in argv
    assert argv[argv.index('--train-components') + 1] == \
        's7_temporal_association'


def test_locked_argv_is_accepted_by_labeller_parser(tmp_path, monkeypatch):
    args = _args(tmp_path, tmp_path / 'candidate.pth')
    monkeypatch.setattr(sys, 'argv', audit.build_locked_labeller_argv(args))
    parsed = labeller.parse_args()
    assert parsed.eval_only_checkpoint == args.eval_only_checkpoint
    assert parsed.skip_target_eval is True
    assert parsed.source_temporal_attribution_audit is True
    assert parsed.source_temporal_attribution_epoch == 4
    assert parsed.s7_temporal_min_confirmations == 1
