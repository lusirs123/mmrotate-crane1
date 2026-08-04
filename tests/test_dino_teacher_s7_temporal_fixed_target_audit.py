import argparse
import json

from crane_project.tools import (
    dino_teacher_s7_temporal_fixed_target_audit as audit)


def _source_result(candidate_path, lost=0, relative=True, target_read=False):
    gate_checks = dict(
        exact_old_correct_retention=(lost == 0),
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
        baseline_correct_count=677,
        retained_correct_count=677 - lost,
        lost_correct_count=lost,
        gained_correct_count=14,
        candidate_correct_count=691 - lost,
        lost_frame_keys=[], gained_frame_keys=[])
    return dict(
        protocol_version=20,
        source_selected_checkpoint=str(candidate_path),
        protocol=dict(s7_temporal_association=dict(
            candidate_quality_head=True,
            relative_quality=relative,
            min_confirmations=1,
            target_read=target_read)),
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
                epoch=4,
                train=dict(),
                source_val=dict(
                    frame_count=738, top1_hits=691, top1_mcml=3,
                    recall_at_100=738,
                    top1_dfr_fraction_per_frame=0.078,
                    top1_aci=0.837),
                source_small_val=dict(
                    frame_count=350, top1_hits=312, top1_mcml=3,
                    recall_at_100=350),
                source_exact_retention=retention,
                source_selection_gate_passed=(lost == 0),
                source_retention_passed=(lost == 0),
                source_selection_gate=dict(
                    checks=gate_checks, passed=(lost == 0)),
                s7_source_gate=dict(
                    checks=gate_checks, passed=(lost == 0)),
                selected_as_best=True,
                selection_eligible=True,
                checkpoint_saved=True)]),
        target_dev=None,
        decision='SOURCE_ONLY_TRAINING_COMPLETE_TARGET_NOT_READ')


def _checkpoint_payload(relative=True):
    return dict(
        source_only=True,
        frozen_dinov2=True,
        epoch=4,
        best_epoch=4,
        s7_inference_enabled=True,
        source_selection_gate_passed=True,
        best_source_val_summary=dict(top1_hits=691, top1_mcml=3),
        best_source_small_val_summary=dict(top1_hits=312, top1_mcml=3),
        source_exact_retention=dict(
            baseline_correct_count=677,
            retained_correct_count=677,
            lost_correct_count=0),
        s7_architecture=dict(
            temporal_association=True,
            temporal_quality_head=True,
            temporal_min_confirmations=1),
        training_protocol=dict(
            train_components='s7_temporal_association',
            s7_temporal_association=dict(
                relative_quality=relative,
                candidate_quality_head=True,
                target_read=False)))


def _summary(frame_count, top1, mcml, recall_at_100):
    return dict(
        frame_count=frame_count,
        top1_hits=top1,
        top1_mcml=mcml,
        recall_at_100=recall_at_100)


def _row(frame, hit):
    return dict(
        role='target_dev_diagnosis_only', split='test', seq='seq',
        frame=frame, feature_cache_hit=True,
        metrics=dict(top1_hit=hit), detections=[])


def test_strict_source_gate_accepts_selected_relative_epoch_four(tmp_path):
    result = audit.strict_source_gate(
        _source_result(tmp_path / 'candidate.pth'))
    assert result['passed'] is True
    assert result['best_epoch'] == 4
    assert result['retention']['lost_correct_count'] == 0


def test_strict_source_gate_rejects_retention_or_target_leak(tmp_path):
    lost = audit.strict_source_gate(
        _source_result(tmp_path / 'candidate.pth', lost=1))
    assert lost['passed'] is False
    assert lost['checks']['exact_old_correct_retention'] is False
    leaked = _source_result(tmp_path / 'candidate.pth', target_read=True)
    leaked['target_dev'] = dict(summary={})
    gated = audit.strict_source_gate(leaked)
    assert gated['passed'] is False
    assert gated['checks']['source_result_target_not_read'] is False


def test_candidate_checkpoint_gate_requires_relative_quality():
    source_gate = dict(best_epoch=4)
    accepted = audit.candidate_checkpoint_gate(
        _checkpoint_payload(relative=True), source_gate)
    rejected = audit.candidate_checkpoint_gate(
        _checkpoint_payload(relative=False), source_gate)
    assert accepted['passed'] is True
    assert rejected['passed'] is False
    assert rejected['checks']['relative_quality_checkpoint'] is False


def test_fixed_slice_gates_match_preregistered_thresholds(monkeypatch):
    baseline = dict(
        seq02_far=_summary(40, 38, 1, 40),
        seq02_dark=_summary(33, 29, 1, 33),
        seq03_small=_summary(64, 50, 6, 55))
    candidate = dict(
        seq02_far=_summary(40, 38, 1, 40),
        seq02_dark=_summary(33, 29, 1, 33),
        seq03_small=_summary(64, 51, 6, 64))
    summaries = iter([
        baseline['seq02_far'], candidate['seq02_far'],
        baseline['seq02_dark'], candidate['seq02_dark'],
        baseline['seq03_small'], candidate['seq03_small']])
    monkeypatch.setattr(audit.labeller, 'summarize_rows', lambda rows: next(
        summaries))
    rows = [_row(1, True)]
    assert audit.fixed_slice_result('seq02_far', rows, rows)['passed']
    assert audit.fixed_slice_result('seq02_dark', rows, rows)['passed']
    small = audit.fixed_slice_result('seq03_small', rows, rows)
    assert small['passed']
    assert small['checks']['strict_top1_gain']
    assert small['checks']['candidate_recall_at_100_floor']


def test_fixed_small_slice_rejects_no_gain_or_incomplete_recall(monkeypatch):
    summaries = iter([
        _summary(64, 50, 6, 55),
        _summary(64, 50, 6, 63)])
    monkeypatch.setattr(audit.labeller, 'summarize_rows', lambda rows: next(
        summaries))
    result = audit.fixed_slice_result(
        'seq03_small', [_row(1, True)], [_row(1, True)])
    assert result['passed'] is False
    assert result['checks']['strict_top1_gain'] is False
    assert result['checks']['candidate_recall_at_100_floor'] is False


def test_validate_args_locks_selected_checkpoint_and_model(tmp_path):
    baseline = tmp_path / 'baseline.pth'
    candidate = tmp_path / 'candidate.pth'
    dino = tmp_path / 'dino.pth'
    for path in (baseline, candidate, dino):
        path.write_bytes(b'x')
    source_result = tmp_path / 'train_result.json'
    source_result.write_text(json.dumps(_source_result(candidate)))
    args = argparse.Namespace(
        data_root=str(tmp_path),
        source_result_json=str(source_result),
        baseline_checkpoint=str(baseline),
        candidate_checkpoint=str(candidate),
        dinov2_repo=str(tmp_path),
        dinov2_checkpoint=str(dino),
        dinov2_model='dinov2_vitl14',
        dino_gpus=[1, 2], head_gpu=0,
        legacy_sdpa_query_chunk=512,
        feature_cache_dir=str(tmp_path / 'cache'),
        seed=0, out_json=str(tmp_path / 'result.json'))
    audit.validate_args(args)
    assert args.train_components == 's7_temporal_association'
    assert args.s7_temporal_relative_quality is True
    assert args.s7_temporal_min_confirmations == 1
    assert [row['name'] for row in args.parsed_target_slices] == [
        'seq02_far', 'seq02_dark', 'seq03_small']

    other = tmp_path / 'other.pth'
    other.write_bytes(b'x')
    args.candidate_checkpoint = str(other)
    try:
        audit.validate_args(args)
    except ValueError as error:
        assert 'source_selected_checkpoint' in str(error)
    else:
        raise AssertionError('A non-selected checkpoint was accepted')
