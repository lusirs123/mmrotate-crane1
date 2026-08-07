import argparse

from crane_project.tools import (
    dino_teacher_s7_selective_promotion_train as selective)


def test_selective_wrapper_locks_source_only_native_protected_protocol(
        tmp_path):
    args = argparse.Namespace(
        data_root='data', source_result_json='source.json',
        init_checkpoint='phase2.pth', dinov2_repo='dinov2',
        dinov2_checkpoint='dino.pth', dinov2_model='dinov2_vitl14',
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=512,
        feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'work'),
        out_json=str(tmp_path / 'work' / 'train_result.json'), seed=0)
    argv = selective.build_locked_labeller_argv(args)
    assert argv[argv.index('--train-components') + 1] == (
        's7_selective_promotion')
    assert '--s7-selective-promotion' in argv
    assert '--s7-temporal-association' not in argv
    assert '--s7-temporal-student' not in argv
    assert '--s7-static-domain-ranker' not in argv
    assert '--skip-target-eval' in argv
    assert argv[argv.index('--source-small-repeat') + 1] == '1'
    assert argv[argv.index('--source-retain-max-top1-drop') + 1] == '0'
    assert argv[argv.index('--s7-selective-min-gain-sequences') + 1] == '2'
    assert argv[argv.index('--s7-selective-teacher-result-json') + 1] == (
        'source.json')
