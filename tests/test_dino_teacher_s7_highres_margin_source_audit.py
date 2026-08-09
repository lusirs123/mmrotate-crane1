import json
from argparse import Namespace

from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import (
    dino_teacher_s7_highres_margin_source_audit as audit)


def _near_pass_payload(work_dir):
    checks = dict(
        exact_old_correct_retention=True,
        full_top1_nonregression=True,
        full_top1_absolute=False,
        small_top1_nonregression=True,
        small_top1_absolute=False,
        full_mcml_absolute=True,
        small_mcml_absolute=True,
        source_temporal_metrics_available=True,
        source_dfr_nonregression=True,
        source_aci_nonregression=True)
    return dict(
        protocol_version=23,
        decision='SOURCE_ONLY_HIGHRES_ROI_RANKER_FALLBACK_TARGET_NOT_READ',
        source_selected_checkpoint=str(
            work_dir / 'labeller_best_source_only.pth'),
        target_dev=None,
        protocol=dict(s7_highres_roi_ranker=dict(
            target_read=False, source_only=True,
            exact_source_retention=True, promotion_margin=0.25)),
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
            small_sampling=dict(short_token_threshold=1.5),
            history=[dict(
                epoch=3, checkpoint_saved=True, selection_eligible=True,
                source_selection_gate_passed=False,
                source_val=dict(top1_hits=687, top1_mcml=3),
                source_small_val=dict(top1_hits=310, top1_mcml=3),
                source_exact_retention=dict(
                    baseline_correct_count=677, lost_correct_count=0,
                    gained_correct_count=10),
                source_selection_gate=dict(checks=checks))]))


def test_margin_audit_spec_locks_epoch3_near_pass_checkpoint(tmp_path):
    work_dir = tmp_path / 'work'
    work_dir.mkdir()
    checkpoint = work_dir / 'labeller_epoch_03_source_only.pth'
    checkpoint.write_bytes(b'checkpoint')
    result_json = work_dir / 'train_result.json'
    result_json.write_text(
        json.dumps(_near_pass_payload(work_dir)), encoding='utf-8')
    spec = labeller.load_highres_margin_audit_spec(
        str(result_json), str(checkpoint), 3)
    assert spec['epoch'] == 3
    assert spec['history_row']['source_val']['top1_hits'] == 687
    assert spec['history_row']['source_small_val']['top1_hits'] == 310


def test_margin_audit_wrapper_locks_shared_forward_source_only_grid(tmp_path):
    args = Namespace(
        data_root='data', source_result_json='source.json',
        eval_only_checkpoint='epoch3.pth', dinov2_repo='dinov2',
        dinov2_checkpoint='dino.pth', dinov2_model='dinov2_vitl14',
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=512,
        feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'work'),
        out_json=str(tmp_path / 'work' / 'margin_result.json'), seed=0)
    argv = audit.build_locked_labeller_argv(args)
    assert '--source-highres-margin-audit' in argv
    index = argv.index('--source-highres-margin-values')
    assert argv[index + 1:index + 4] == ['0.2', '0.225', '0.25']
    assert argv[argv.index('--source-highres-margin-epoch') + 1] == '3'
    assert argv[argv.index('--eval-only-checkpoint') + 1] == 'epoch3.pth'
    assert '--init-checkpoint' not in argv
    assert '--skip-target-eval' in argv
    assert argv[argv.index('--s7-source-min-full-top1') + 1] == '688'
    assert argv[argv.index('--s7-source-min-small-top1') + 1] == '311'


def test_locked_margin_audit_arguments_pass_labeller_validation(
        tmp_path, monkeypatch):
    work_dir = tmp_path / 'trained'
    work_dir.mkdir()
    checkpoint = work_dir / 'labeller_epoch_03_source_only.pth'
    checkpoint.write_bytes(b'checkpoint')
    result_json = work_dir / 'train_result.json'
    result_json.write_text(
        json.dumps(_near_pass_payload(work_dir)), encoding='utf-8')
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
    monkeypatch.setattr(
        'sys.argv', audit.build_locked_labeller_argv(args))
    parsed = labeller.parse_args()
    labeller.validate_args(parsed)
    assert parsed.source_highres_margin_values == [0.2, 0.225, 0.25]
    assert parsed.source_highres_margin_audit_spec['epoch'] == 3
