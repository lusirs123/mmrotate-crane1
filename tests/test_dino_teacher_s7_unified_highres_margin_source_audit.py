import json
from argparse import Namespace

from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import (
    dino_teacher_s7_unified_highres_margin_source_audit as audit)


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


def test_unified_margin_spec_locks_epoch3_one_frame_loss(tmp_path):
    work_dir = tmp_path / 'work'
    work_dir.mkdir()
    checkpoint = work_dir / 'labeller_epoch_03_source_only.pth'
    checkpoint.write_bytes(b'checkpoint')
    result_json = work_dir / 'train_result.json'
    result_json.write_text(
        json.dumps(_unified_payload(work_dir)), encoding='utf-8')
    spec = labeller.load_unified_highres_margin_audit_spec(
        str(result_json), str(checkpoint), 3)
    assert spec['audit_variant'] == 'unified_bounded_risk'
    assert spec['history_row']['source_val']['top1_hits'] == 696
    assert spec['history_row']['source_exact_retention'][
        'lost_frame_keys'] == ['val|real_seq07|215']


def test_unified_margin_wrapper_is_read_only_and_uses_conservative_grid(
        tmp_path):
    args = Namespace(
        data_root='data', source_result_json='source.json',
        eval_only_checkpoint='epoch3.pth', dinov2_repo='dinov2',
        dinov2_checkpoint='dino.pth', dinov2_model='dinov2_vitl14',
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=512,
        feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'work'),
        out_json=str(tmp_path / 'work' / 'margin_result.json'), seed=0)
    argv = audit.build_locked_labeller_argv(args)
    assert '--s7-highres-unified-ranking' in argv
    index = argv.index('--source-highres-margin-values')
    assert argv[index + 1:index + 4] == ['0.25', '0.275', '0.3']
    assert argv[argv.index('--source-highres-margin-epoch') + 1] == '3'
    assert '--eval-only-checkpoint' in argv
    assert '--init-checkpoint' not in argv
    assert '--skip-target-eval' in argv


def test_locked_unified_margin_arguments_pass_labeller_validation(
        tmp_path, monkeypatch):
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
        dinov2_checkpoint=str(dino_checkpoint),
        dinov2_model='dinov2_vitl14', dino_gpus=[1, 2], head_gpu=0,
        legacy_sdpa_query_chunk=512,
        feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'audit'),
        out_json=str(tmp_path / 'audit' / 'margin_result.json'), seed=0)
    monkeypatch.setattr('sys.argv', audit.build_locked_labeller_argv(args))
    parsed = labeller.parse_args()
    labeller.validate_args(parsed)
    assert parsed.s7_highres_unified_ranking is True
    assert parsed.source_highres_margin_values == [0.25, 0.275, 0.3]
    assert parsed.source_highres_margin_audit_spec[
        'audit_variant'] == 'unified_bounded_risk'


def test_bounded_risk_gate_does_not_relabel_exact_retention():
    baseline = dict(
        top1_hits=677, top1_mcml=3,
        top1_dfr_fraction_per_frame=0.084,
        top1_aci=0.826)
    baseline_small = dict(top1_hits=303, top1_mcml=3)
    candidate = dict(
        top1_hits=696, top1_mcml=3,
        top1_dfr_fraction_per_frame=0.078,
        top1_aci=0.839)
    candidate_small = dict(top1_hits=320, top1_mcml=3)
    retention = dict(
        baseline_correct_count=677, retained_correct_count=676,
        lost_correct_count=1, gained_correct_count=20)
    args = Namespace(
        train_components='s7_highres_roi_ranker',
        s7_source_min_full_top1=688,
        s7_source_min_small_top1=311,
        s7_source_max_mcml=3)
    gate = labeller.unified_highres_bounded_risk_source_gate(
        baseline, baseline_small, candidate, candidate_small,
        retention, dict(passed=True), args)
    assert gate['passed'] is True
    assert gate['original_formal_gate_passed'] is False
    assert gate['source_safe_claim_allowed'] is False
    assert gate['deployment_claim_allowed'] is False

    retention['retained_correct_count'] = 675
    retention['lost_correct_count'] = 2
    rejected = labeller.unified_highres_bounded_risk_source_gate(
        baseline, baseline_small, candidate, candidate_small,
        retention, dict(passed=True), args)
    assert rejected['passed'] is False
    assert rejected['checks']['bounded_lost_correct_count'] is False
    assert rejected['checks']['bounded_lost_correct_fraction'] is False
