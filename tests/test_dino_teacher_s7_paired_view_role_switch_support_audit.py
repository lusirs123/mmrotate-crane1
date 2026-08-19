import json
from argparse import Namespace

import numpy as np

from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import (
    dino_teacher_s7_paired_view_role_switch_support_audit as audit)


def _unified_payload(work_dir):
    checks = dict(
        exact_old_correct_retention=False,
        full_top1_nonregression=True, full_top1_absolute=True,
        small_top1_nonregression=True, small_top1_absolute=True,
        full_mcml_absolute=True, small_mcml_absolute=True,
        source_temporal_metrics_available=True,
        source_dfr_nonregression=True, source_aci_nonregression=True)
    return dict(
        protocol_version=23,
        decision='SOURCE_ONLY_HIGHRES_ROI_RANKER_FALLBACK_TARGET_NOT_READ',
        source_selected_checkpoint=str(
            work_dir / 'labeller_best_source_only.pth'),
        target_dev=None,
        protocol=dict(s7_highres_roi_ranker=dict(
            target_read=False, source_only=True, unified_ranking=True,
            exact_source_retention=True, promotion_margin=0.25)),
        architecture=dict(s7=dict(highres_unified_ranking=True)),
        isolation=dict(
            train_components='s7_highres_roi_ranker',
            dino_parameters_unchanged=True,
            frozen_head_parameters_unchanged=True,
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False),
        source=dict(
            baseline_validation_summary=dict(top1_hits=677, top1_mcml=3),
            baseline_small_validation_summary=dict(
                top1_hits=303, top1_mcml=3),
            small_sampling=dict(short_token_threshold=1.67),
            history=[dict(
                epoch=3, checkpoint_saved=True, selection_eligible=True,
                source_selection_gate_passed=False,
                source_val=dict(top1_hits=696, top1_mcml=3),
                source_small_val=dict(top1_hits=320, top1_mcml=3),
                source_exact_retention=dict(
                    baseline_correct_count=677, retained_correct_count=676,
                    lost_correct_count=1, gained_correct_count=20,
                    lost_frame_keys=['val|real_seq07|215']),
                source_selection_gate=dict(checks=checks))]))


def _view(name, native_correct, s7_correct):
    return dict(
        view=name, candidate_count=3, eligible=True,
        native_index=0, native_riou=0.8 if native_correct else 0.2,
        native_correct=native_correct, s7_correct_count=int(s7_correct),
        best_s7_riou=0.8 if s7_correct else 0.2,
        best_s7_rank_by_roi_score=2 if s7_correct else None,
        native_rank_by_roi_score=1, feature_cache_hit=False)


def _frame(seq, domain, frame, photo_gain=False, degrade_gain=False):
    views = dict(
        clean=_view('clean', True, False),
        photometric=_view('photometric', not photo_gain, photo_gain),
        degradation=_view('degradation', not degrade_gain, degrade_gain))
    return labeller._paired_view_frame_row(dict(
        split='train', seq=seq, frame=frame,
        annotation='unused'), views)


def test_paired_views_are_deterministic_and_keep_geometry():
    image = np.full((24, 32, 3), 128, dtype=np.uint8)
    record = dict(split='train', seq='real_seq01', frame=7)
    clean = labeller.paired_view_image(image, record)
    photometric_a = labeller.paired_view_image(
        image, dict(record, paired_view='photometric', paired_view_version=1))
    photometric_b = labeller.paired_view_image(
        image, dict(record, paired_view='photometric', paired_view_version=1))
    degradation = labeller.paired_view_image(
        image, dict(record, paired_view='degradation', paired_view_version=1))
    assert clean.shape == image.shape
    assert np.array_equal(clean, image)
    assert np.array_equal(photometric_a, photometric_b)
    assert degradation.shape == image.shape
    assert not np.array_equal(degradation, image)


def test_role_switch_summary_counts_each_original_frame_once():
    rows = [
        _frame('real_seq01', 'real', 1, photo_gain=True),
        _frame('sim_seq02', 'sim', 2, degrade_gain=True),
        _frame('sim_seq03', 'sim', 3, photo_gain=True, degrade_gain=True),
    ]
    for row, domain in zip(rows, ('real', 'sim', 'sim')):
        row['domain'] = domain
    summary = labeller.summarize_paired_view_role_switch_support(rows)
    assert summary['paired_gain_role_switch_frame_count'] == 3
    assert summary['augmented_native_wrong_s7_correct_frame_count'] == 3
    assert summary['paired_gain_role_switch_domains'] == ['real', 'sim']
    assert summary['paired_gain_role_switch_sequences'] == [
        'real_seq01', 'sim_seq02', 'sim_seq03']
    assert summary['view_summaries']['photometric'][
        'native_wrong_s7_correct_frame_count'] == 2


def test_support_gate_requires_cross_domain_and_non_dominant_sequences():
    summary = dict(
        paired_gain_role_switch_frame_count=32,
        paired_gain_role_switch_domains=['real', 'sim'],
        paired_gain_role_switch_sequences=['real_seq01', 'sim_seq02', 'sim_seq03'],
        paired_gain_role_switch_max_sequence_fraction=0.50,
        retention_support_domains=['real', 'sim'])
    args = Namespace(
        source_paired_view_min_gain_frames=32,
        source_paired_view_min_gain_sequences=3,
        source_paired_view_max_sequence_fraction=0.50)
    assert labeller._paired_view_role_switch_support_gate(
        summary, args)['passed'] is True
    summary['paired_gain_role_switch_domains'] = ['sim']
    assert labeller._paired_view_role_switch_support_gate(
        summary, args)['passed'] is False


def test_wrapper_locks_epoch3_and_never_reads_target(tmp_path):
    work_dir = tmp_path / 'work'
    work_dir.mkdir()
    checkpoint = work_dir / 'labeller_epoch_03_source_only.pth'
    checkpoint.write_bytes(b'checkpoint')
    result_json = work_dir / 'train_result.json'
    result_json.write_text(
        json.dumps(_unified_payload(work_dir)), encoding='utf-8')
    args = Namespace(
        data_root='data', source_result_json=str(result_json),
        eval_only_checkpoint=str(checkpoint), dinov2_repo='dinov2',
        dinov2_checkpoint='dino.pth', dinov2_model='dinov2_vitl14',
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=512,
        feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'audit'),
        out_json=str(tmp_path / 'audit' / 'result.json'),
        min_gain_frames=32, min_gain_sequences=3,
        max_sequence_fraction=0.50, seed=0)
    argv = audit.build_locked_labeller_argv(args)
    assert '--source-paired-view-role-switch-support-audit' in argv
    assert '--source-paired-view-source-result-json' in argv
    assert '--eval-only-checkpoint' in argv
    assert '--skip-target-eval' in argv
    assert '--init-checkpoint' not in argv
    assert '--resume-checkpoint' not in argv
    assert argv[argv.index('--s7-highres-base-epoch') + 1] == '3'


def test_wrapper_arguments_pass_labeller_validation(tmp_path, monkeypatch):
    work_dir = tmp_path / 'work'
    work_dir.mkdir()
    checkpoint = work_dir / 'labeller_epoch_03_source_only.pth'
    checkpoint.write_bytes(b'checkpoint')
    result_json = work_dir / 'train_result.json'
    result_json.write_text(
        json.dumps(_unified_payload(work_dir)), encoding='utf-8')
    dino_repo = tmp_path / 'dinov2'
    dino_repo.mkdir()
    dino_checkpoint = tmp_path / 'dino.pth'
    dino_checkpoint.write_bytes(b'dino')
    args = Namespace(
        data_root='data', source_result_json=str(result_json),
        eval_only_checkpoint=str(checkpoint), dinov2_repo=str(dino_repo),
        dinov2_checkpoint=str(dino_checkpoint), dinov2_model='dinov2_vitl14',
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=512,
        feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'audit'),
        out_json=str(tmp_path / 'audit' / 'result.json'),
        min_gain_frames=32, min_gain_sequences=3,
        max_sequence_fraction=0.50, seed=0)
    monkeypatch.setattr('sys.argv', audit.build_locked_labeller_argv(args))
    parsed = labeller.parse_args()
    labeller.validate_args(parsed)
    assert parsed.source_paired_view_role_switch_support_audit is True
    assert parsed.skip_target_eval is True
    assert parsed.source_paired_view_min_gain_frames == 32
