import json
from argparse import Namespace

import torch

from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import (
    dino_teacher_s7_dense_temporal_separability_audit as audit)


def _state(positive_roi, negative_roi, positive_highres, negative_highres,
           positive_box=None, negative_box=None):
    positive_box = positive_box or [10.0, 10.0, 4.0, 4.0, 0.0, 0.8]
    negative_box = negative_box or [30.0, 30.0, 4.0, 4.0, 0.0, 0.9]
    return dict(
        detections=torch.tensor([positive_box, negative_box]),
        source_ids=torch.tensor([1, 0]),
        roi_embeddings=torch.tensor([positive_roi, negative_roi]),
        highres_embeddings=torch.tensor([positive_highres, negative_highres]),
        overlaps=torch.tensor([0.8, 0.1]),
        positive_indices=torch.tensor([0]),
        negative_indices=torch.tensor([1]),
        representative_index=0)


def _comparison(seq, domain, success=True):
    margin = 0.5 if success else -0.1
    return dict(
        seq=seq, domain=domain, combined_margin=margin,
        appearance_success=success,
        roi=dict(margin=margin), highres=dict(margin=margin),
        motion=None, motion_success=None)


def _gate_args():
    return Namespace(
        source_dense_temporal_min_pairs_per_sequence=2,
        source_dense_temporal_min_real_sequences=2,
        source_dense_temporal_min_sim_sequences=1,
        source_dense_temporal_min_success_fraction=0.60,
        source_dense_temporal_min_cosine_margin=0.02)


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


def test_dense_comparison_prefers_gt_matched_candidate():
    previous = _state(
        [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0])
    current = _state(
        [0.9, 0.1], [0.0, 1.0], [0.8, 0.2], [0.0, 1.0])
    row = labeller._dense_temporal_comparison_row(
        previous, current, minimum_margin=0.02)
    assert row is not None
    assert row['combined_margin'] > 0.5
    assert row['appearance_success'] is True


def test_motion_comparison_uses_two_prior_positive_boxes():
    older = _state(
        [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0],
        positive_box=[8.0, 10.0, 4.0, 4.0, 0.0, 0.8])
    previous = _state(
        [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0],
        positive_box=[10.0, 10.0, 4.0, 4.0, 0.0, 0.8])
    current = _state(
        [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0],
        positive_box=[12.0, 10.0, 4.0, 4.0, 0.0, 0.8])
    motion = labeller._dense_temporal_motion_comparison(
        older, previous, current)
    assert motion is not None
    assert motion['best_positive_cost'] == 0.0
    assert motion['margin'] > 0.0


def test_dense_gate_requires_qualified_real_and_sim_sequences():
    rows = []
    for sequence, domain in (
            ('real_seq01', 'real'), ('real_seq04', 'real'),
            ('sim_seq08', 'sim')):
        rows.extend([_comparison(sequence, domain),
                     _comparison(sequence, domain)])
    summary = labeller.summarize_dense_temporal_separability(
        rows, list(rows), _gate_args())
    assert summary['support_gate']['passed'] is True
    assert summary['support_gate']['qualified_real_sequences'] == [
        'real_seq01', 'real_seq04']
    failed = [row for row in rows if row['seq'] != 'real_seq04']
    summary = labeller.summarize_dense_temporal_separability(
        failed, list(failed), _gate_args())
    assert summary['support_gate']['passed'] is False


def test_wrapper_is_eval_only_and_locks_protocol33_inputs(tmp_path):
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
        min_pairs_per_sequence=32, min_real_sequences=2,
        min_sim_sequences=1, min_success_fraction=0.60,
        min_cosine_margin=0.02, negative_riou_max=0.30,
        hard_negatives=8, seed=0)
    argv = audit.build_locked_labeller_argv(args)
    assert '--source-dense-temporal-separability-audit' in argv
    assert '--source-dense-temporal-source-result-json' in argv
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
        min_pairs_per_sequence=32, min_real_sequences=2,
        min_sim_sequences=1, min_success_fraction=0.60,
        min_cosine_margin=0.02, negative_riou_max=0.30,
        hard_negatives=8, seed=0)
    monkeypatch.setattr('sys.argv', audit.build_locked_labeller_argv(args))
    parsed = labeller.parse_args()
    labeller.validate_args(parsed)
    assert parsed.source_dense_temporal_separability_audit is True
    assert parsed.skip_target_eval is True
    assert parsed.source_dense_temporal_min_pairs_per_sequence == 32
