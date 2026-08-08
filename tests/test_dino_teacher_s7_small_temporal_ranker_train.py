from argparse import Namespace

from crane_project.tools import (
    dino_teacher_s7_small_temporal_ranker_train as trainer)


def test_small_temporal_wrapper_locks_lightweight_source_only_protocol(tmp_path):
    args = Namespace(
        data_root='data', source_result_json='source.json',
        init_checkpoint='phase2.pth', dinov2_repo='dinov2',
        dinov2_checkpoint='dino.pth', dinov2_model='dinov2_vitl14',
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=512,
        feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'work'),
        out_json=str(tmp_path / 'work' / 'train_result.json'), seed=0)
    argv = trainer.build_locked_labeller_argv(args)
    assert '--s7-selective-promotion' in argv
    assert '--s7-selective-two-frame' in argv
    assert argv[argv.index('--s7-selective-hidden') + 1] == '16'
    assert argv[argv.index('--s7-selective-max-candidates') + 1] == '20'
    assert argv[argv.index('--train-components') + 1] == (
        's7_selective_promotion')
    assert argv[argv.index('--source-small-repeat') + 1] == '1'
    assert argv[argv.index('--source-retain-max-top1-drop') + 1] == '0'
    assert '--skip-target-eval' in argv
    assert argv[argv.index('--epochs') + 1] == '4'
    assert argv[argv.index('--selection-epochs') + 1:][:4] == [
        '1', '2', '3', '4']
