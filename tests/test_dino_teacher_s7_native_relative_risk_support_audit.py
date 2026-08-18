import json
from argparse import Namespace

from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import (
    dino_teacher_s7_native_relative_risk_support_audit as audit)


def _risk_payload(work_dir):
    history = []
    for epoch, lost, gained, active in (
            (1, 1, 20, 10.0), (2, 4, 23, 12.0),
            (3, 4, 24, 11.0), (4, 4, 26, 16.0)):
        history.append(dict(
            epoch=epoch, source_selection_gate_passed=False,
            source_exact_retention=dict(
                lost_correct_count=lost, gained_correct_count=gained),
            train=dict(mean_training_metrics=dict(
                s7_highres_relative_risk_scale=0.0,
                s7_highres_relative_risk_nonzero_count=0.0,
                s7_highres_relative_risk_retention_active_count=active))))
    return dict(
        protocol_version=29,
        decision='SOURCE_ONLY_HIGHRES_ROI_RANKER_FALLBACK_TARGET_NOT_READ',
        source_selected_checkpoint=str(work_dir / 'labeller_best_source_only.pth'),
        target_dev=None,
        protocol=dict(s7_highres_roi_ranker=dict(
            source_only=True, target_read=False, unified_ranking=True,
            smooth_geometry_ranking=True,
            native_relative_risk_residual=True)),
        architecture=dict(s7=dict(
            highres_native_relative_risk_residual=True)),
        isolation=dict(
            train_components='s7_highres_roi_ranker',
            dino_parameters_unchanged=True,
            frozen_head_parameters_unchanged=True,
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False),
        source=dict(best_epoch=0, history=history))


def test_support_summary_reports_zero_effective_penalty_and_coverage():
    rows = [
        dict(split='train', seq='real_seq01', domain='real', frame=1,
             eligible=True, candidate_count=5, native_correct=True,
             wrong_s7_count=3, usable_s7_count=0,
             active_required_penalty_count=2, required_penalty_sum=0.4,
             required_penalty_max=0.3,
             required_penalty_histogram=[1, 0, 1, 1, 0, 0, 0, 0, 0],
             oracle_s7=False, oracle_rank_by_base_fused=1,
             effective_penalty_nonzero_count=0, effective_penalty_max=0.0),
        dict(split='train_sim', seq='sim_seq10', domain='sim', frame=2,
             eligible=True, candidate_count=4, native_correct=False,
             wrong_s7_count=1, usable_s7_count=1,
             active_required_penalty_count=0, required_penalty_sum=0.0,
             required_penalty_max=0.0,
             required_penalty_histogram=[1, 0, 0, 0, 0, 0, 0, 0, 0],
             oracle_s7=True, oracle_rank_by_base_fused=2,
             effective_penalty_nonzero_count=0, effective_penalty_max=0.0),
    ]
    summary = labeller.summarize_native_relative_risk_support(rows)
    assert summary['wrong_s7_candidate_count'] == 4
    assert summary['active_required_penalty_candidate_count'] == 2
    assert summary['active_required_penalty_domains'] == ['real']
    assert summary['native_wrong_usable_s7_domains'] == ['sim']
    assert summary['effective_penalty_nonzero_count'] == 0
    assert summary['oracle_s7_not_top1_count'] == 1


def test_wrapper_locks_rejected_fallback_and_never_reads_target(tmp_path):
    work_dir = tmp_path / 'risk'
    work_dir.mkdir()
    checkpoint = work_dir / 'labeller_best_source_only.pth'
    checkpoint.write_bytes(b'checkpoint')
    result_json = work_dir / 'train_result.json'
    result_json.write_text(json.dumps(_risk_payload(work_dir)), encoding='utf-8')
    args = Namespace(
        data_root='data', source_result_json=str(result_json),
        eval_only_checkpoint=str(checkpoint),
        smooth_geometry_support_result_json='support.json',
        dinov2_repo='dinov2', dinov2_checkpoint='dino.pth',
        dinov2_model='dinov2_vitl14', dino_gpus=[1, 2], head_gpu=0,
        legacy_sdpa_query_chunk=512, feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'audit'),
        out_json=str(tmp_path / 'audit' / 'result.json'), seed=0)
    argv = audit.build_locked_labeller_argv(args)
    assert '--source-native-relative-risk-support-audit' in argv
    assert '--source-native-relative-risk-source-result-json' in argv
    assert '--s7-highres-native-relative-risk-residual' in argv
    assert '--eval-only-checkpoint' in argv
    assert '--skip-target-eval' in argv
    assert '--init-checkpoint' not in argv
    assert '--resume-checkpoint' not in argv
    spec = labeller.load_native_relative_risk_support_audit_spec(
        str(result_json), str(checkpoint))
    assert spec['observed_failure']['epochs'][0]['lost_correct_count'] == 1


def test_wrapper_arguments_pass_labeller_validation(tmp_path, monkeypatch):
    work_dir = tmp_path / 'risk'
    work_dir.mkdir()
    checkpoint = work_dir / 'labeller_best_source_only.pth'
    checkpoint.write_bytes(b'checkpoint')
    result_json = work_dir / 'train_result.json'
    result_json.write_text(json.dumps(_risk_payload(work_dir)), encoding='utf-8')
    support = tmp_path / 'support.json'
    support.write_text('{}', encoding='utf-8')
    dino_repo = tmp_path / 'dinov2'
    dino_repo.mkdir()
    dino_checkpoint = tmp_path / 'dino.pth'
    dino_checkpoint.write_bytes(b'dino')
    args = Namespace(
        data_root='data', source_result_json=str(result_json),
        eval_only_checkpoint=str(checkpoint),
        smooth_geometry_support_result_json=str(support),
        dinov2_repo=str(dino_repo), dinov2_checkpoint=str(dino_checkpoint),
        dinov2_model='dinov2_vitl14', dino_gpus=[1, 2], head_gpu=0,
        legacy_sdpa_query_chunk=512, feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'audit'),
        out_json=str(tmp_path / 'audit' / 'result.json'), seed=0)
    monkeypatch.setattr('sys.argv', audit.build_locked_labeller_argv(args))
    parsed = labeller.parse_args()
    labeller.validate_args(parsed)
    assert parsed.source_native_relative_risk_support_audit is True
    assert parsed.skip_target_eval is True
    assert parsed.s7_highres_native_relative_risk_residual is True
