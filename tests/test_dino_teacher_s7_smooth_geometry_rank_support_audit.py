import json
from argparse import Namespace

import torch

from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import (
    dino_teacher_s7_smooth_geometry_rank_support_audit as audit)
from crane_project.utils import rotated_geometry_quality as geometry


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


def _row(seq, domain, native_correct, s7_correct, rankings):
    return dict(
        split='val', seq=seq, domain=domain, frame=1,
        eligible=True, candidate_count=3, native_index=0,
        native_riou=0.2 if not native_correct else 0.8,
        native_correct=native_correct,
        s7_correct_count=int(s7_correct),
        gain_pair_count=int(s7_correct if not native_correct else 0),
        rankings=rankings)


def _ranking(hit, gain, loss, agreement=0.5, takeover=False):
    return dict(
        top1_hit=hit, gain_vs_native=gain, loss_vs_native=loss,
        pair_agreement_with_riou=agreement,
        s7_takeover_supported=takeover)


def test_smooth_geometry_is_finite_and_identity_is_zero():
    boxes = torch.tensor([
        [10.0, 20.0, 8.0, 4.0, 0.3],
        [10.0, 20.0, 8.0, 4.0, 0.3]], dtype=torch.float32)
    for fn in (
            geometry.symmetric_gaussian_kl,
            geometry.gaussian_wasserstein_distance,
            geometry.normalized_gaussian_wasserstein_distance):
        values = fn(boxes[:1], boxes[1:])
        assert torch.isfinite(values).all()
        assert float(values.item()) < 1e-5


def test_smooth_geometry_respects_pi_angle_periodicity():
    first = torch.tensor([[0.0, 0.0, 12.0, 5.0, 0.2]])
    equivalent = torch.tensor([[0.0, 0.0, 12.0, 5.0, 0.2 + 3.14159265]])
    covariance_first = geometry.rotated_box_covariance(first)
    covariance_equivalent = geometry.rotated_box_covariance(equivalent)
    assert torch.allclose(covariance_first, covariance_equivalent, atol=1e-5)
    distance = geometry.gaussian_wasserstein_distance(first, equivalent)
    assert float(distance.item()) < 1e-4


def test_pair_agreement_and_source_support_summary_are_deterministic():
    rows = [
        _row('real_seq01', 'real', False, 1, dict(
            sym_kld=_ranking(True, True, False, 1.0, True),
            gwd=_ranking(True, True, False, 1.0, True),
            normalized_gwd=_ranking(True, True, False, 1.0, True))),
        _row('sim_seq02', 'sim', False, 1, dict(
            sym_kld=_ranking(False, False, False, 0.0, False),
            gwd=_ranking(False, False, False, 0.0, False),
            normalized_gwd=_ranking(False, False, False, 0.0, False))),
    ]
    summary = labeller.summarize_smooth_geometry_rank_support(rows)
    assert summary['native_wrong_s7_correct_pair_count'] == 2
    assert summary['gain_domains'] == ['real', 'sim']
    assert summary['gain_sequences'] == ['real_seq01', 'sim_seq02']
    assert summary['metrics']['sym_kld']['top1_gains'] == 1
    assert summary['metrics']['sym_kld']['net_top1_gain'] == 1

    values = [3.0, 2.0, 1.0]
    riou = [0.1, 0.4, 0.9]
    assert labeller._smooth_geometry_pair_agreement(
        values, riou, descending=False) == 1.0


def test_wrapper_is_read_only_and_locks_unified_source_epoch3(tmp_path):
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
        min_gain_domains=2, min_gain_sequences=2, seed=0)
    argv = audit.build_locked_labeller_argv(args)
    assert '--source-smooth-geometry-rank-support-audit' in argv
    assert '--source-smooth-geometry-source-result-json' in argv
    assert '--eval-only-checkpoint' in argv
    assert '--skip-target-eval' in argv
    assert '--init-checkpoint' not in argv
    assert '--resume-checkpoint' not in argv
    assert argv[argv.index('--source-smooth-geometry-min-gain-domains') + 1] == '2'
    assert argv[argv.index('--source-smooth-geometry-min-gain-sequences') + 1] == '2'


def test_wrapper_arguments_pass_labeller_validation(tmp_path, monkeypatch):
    work_dir = tmp_path / 'trained'
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
        min_gain_domains=2, min_gain_sequences=2, seed=0)
    monkeypatch.setattr('sys.argv', audit.build_locked_labeller_argv(args))
    parsed = labeller.parse_args()
    labeller.validate_args(parsed)
    assert parsed.source_smooth_geometry_rank_support_audit is True
    assert parsed.s7_highres_unified_ranking is True
    assert parsed.skip_target_eval is True
    assert parsed.source_smooth_geometry_audit_spec[
        'audit_variant'] == 'unified_bounded_risk'

