from argparse import Namespace

from crane_project.tools import (
    dino_teacher_s7_pairwise_takeover_ranker_train as trainer)


def test_pairwise_takeover_wrapper_locks_v2_source_only_protocol(tmp_path):
    args = Namespace(
        data_root='data', source_margin_result_json='margin.json',
        init_checkpoint='epoch3.pth', dinov2_repo='dinov2',
        dinov2_checkpoint='dino.pth', dinov2_model='dinov2_vitl14',
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=512,
        feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'work'),
        out_json=str(tmp_path / 'work' / 'train_result.json'), seed=0)
    argv = trainer.build_locked_labeller_argv(args)
    assert '--s7-highres-pairwise-takeover-v2' in argv
    assert '--s7-highres-unified-ranking' not in argv
    assert argv[argv.index('--s7-highres-base-epoch') + 1] == '3'
    assert argv[argv.index('--s7-highres-max-candidates') + 1] == '64'
    assert argv[argv.index('--s7-takeover-uncertainty-multiplier') + 1] == '2.0'
    assert argv[argv.index('--s7-takeover-margin') + 1] == '0.05'
    assert argv[argv.index('--deployment-score-thr') + 1] == '0.05'
    assert argv[argv.index('--source-retain-max-top1-drop') + 1] == '0'
    assert '--skip-target-eval' in argv


def test_protocol26_gate_rejects_deployable_or_target_read_payload():
    checkpoint = '/tmp/labeller_epoch_03_source_only.pth'
    payload = dict(
        protocol_version=26,
        decision=('SOURCE_ONLY_UNIFIED_HIGHRES_BOUNDED_RISK_'
                  'RESEARCH_GATE_PASSED_TARGET_NOT_READ'),
        source_research_candidate_checkpoint=checkpoint,
        research_candidate_promotion_margin=0.3,
        source_safe=False, eligible_for_deployment=False,
        eligible_for_full_test=False, target_dev=None,
        source_highres_margin_audit=dict(
            checkpoint_epoch=3, audit_variant='unified_bounded_risk'),
        protocol=dict(source_only=True, target_read=False),
        isolation=dict(
            read_only_evaluation=True, parameter_updates_performed=False,
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False))
    assert trainer.margin_result_gate(payload, checkpoint)['passed'] is True
    payload['target_dev'] = {}
    assert trainer.margin_result_gate(payload, checkpoint)['passed'] is False
