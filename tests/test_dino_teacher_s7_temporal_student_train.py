import argparse

from crane_project.tools import dino_teacher_s7_temporal_student_train as stage3


def test_stage3_wrapper_locks_source_only_student_protocol(tmp_path):
    args = argparse.Namespace(
        data_root='data', source_result_json='source.json',
        init_checkpoint='phase2.pth', dinov2_repo='dinov2',
        dinov2_checkpoint='dino.pth', dinov2_model='dinov2_vitl14',
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=512,
        feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'work'),
        out_json=str(tmp_path / 'work' / 'train_result.json'), seed=0)
    argv = stage3.build_locked_labeller_argv(args)
    assert argv[argv.index('--train-components') + 1] == (
        's7_temporal_student')
    assert '--s7-temporal-student' in argv
    assert '--skip-target-eval' in argv
    assert argv[argv.index('--source-small-repeat') + 1] == '1'
    assert argv[argv.index('--s7-temporal-min-confirmations') + 1] == '1'
    assert argv[argv.index('--s7-student-small-token-thr') + 1] == '4.0'
    assert argv[argv.index('--s7-student-teacher-result-json') + 1] == (
        'source.json')
