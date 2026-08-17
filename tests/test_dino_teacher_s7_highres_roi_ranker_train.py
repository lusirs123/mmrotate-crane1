from argparse import Namespace

from crane_project.tools import (
    dino_teacher_s7_highres_roi_ranker_train as trainer)


def test_highres_wrapper_locks_static_lightweight_source_only_protocol(tmp_path):
    args = Namespace(
        data_root='data', source_result_json='source.json',
        init_checkpoint='phase2.pth', dinov2_repo='dinov2',
        dinov2_checkpoint='dino.pth', dinov2_model='dinov2_vitl14',
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=512,
        feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'work'),
        out_json=str(tmp_path / 'work' / 'train_result.json'), seed=0)
    argv = trainer.build_locked_labeller_argv(args)
    assert '--s7-highres-roi-ranker' in argv
    assert argv[argv.index('--s7-highres-hidden') + 1] == '32'
    assert argv[argv.index('--s7-highres-channels') + 1] == '32'
    assert argv[argv.index('--s7-highres-max-candidates') + 1] == '32'
    assert argv[argv.index('--train-components') + 1] == (
        's7_highres_roi_ranker')
    assert argv[argv.index('--source-small-repeat') + 1] == '1'
    assert argv[argv.index('--source-retain-max-top1-drop') + 1] == '0'
    assert '--skip-target-eval' in argv
    assert argv[argv.index('--epochs') + 1] == '4'
    assert argv[argv.index('--selection-epochs') + 1:][:4] == [
        '1', '2', '3', '4']


def test_highres_wrapper_can_lock_unified_whole_pool_mode(tmp_path):
    args = Namespace(
        data_root='data', source_result_json='source.json',
        init_checkpoint='phase2.pth', dinov2_repo='dinov2',
        dinov2_checkpoint='dino.pth', dinov2_model='dinov2_vitl14',
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=512,
        feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'work'),
        out_json=str(tmp_path / 'work' / 'train_result.json'), seed=0,
        unified_ranking=True)
    argv = trainer.build_locked_labeller_argv(args)
    assert '--s7-highres-unified-ranking' in argv
    assert argv[argv.index('--s7-highres-unified-hard-pairs') + 1] == '8'
    assert argv[argv.index('--s7-highres-unified-aug-prob') + 1] == '0.75'


def test_highres_wrapper_locks_geometry_guided_source_stage(tmp_path):
    args = Namespace(
        data_root='data', source_result_json='source.json',
        init_checkpoint='unified_epoch03.pth', dinov2_repo='dinov2',
        dinov2_checkpoint='dino.pth', dinov2_model='dinov2_vitl14',
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=512,
        feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'work'),
        out_json=str(tmp_path / 'work' / 'train_result.json'), seed=0,
        unified_ranking=True, smooth_geometry_ranking=True,
        geometry_support_json='protocol28_support.json',
        geometry_metric='sym_kld', geometry_loss_weight=0.25,
        geometry_min_gap=0.05, geometry_max_pairs=64)
    argv = trainer.build_locked_labeller_argv(args)
    assert argv[argv.index('--s7-highres-base-epoch') + 1] == '3'
    assert '--s7-highres-smooth-geometry-ranking' in argv
    assert '--s7-highres-smooth-geometry-support-result-json' in argv
    assert '--s7-highres-teacher-result-json' not in argv
    assert argv[argv.index(
        '--s7-highres-worst-case-retention-weight') + 1] == '0.0'


def test_highres_wrapper_locks_native_relative_risk_source_stage(tmp_path):
    args = Namespace(
        data_root='data', source_result_json='source.json',
        init_checkpoint='unified_epoch03.pth', dinov2_repo='dinov2',
        dinov2_checkpoint='dino.pth', dinov2_model='dinov2_vitl14',
        dino_gpus=[1, 2], head_gpu=0, legacy_sdpa_query_chunk=512,
        feature_cache_dir=str(tmp_path / 'cache'),
        work_dir=str(tmp_path / 'work'),
        out_json=str(tmp_path / 'work' / 'train_result.json'), seed=0,
        unified_ranking=True, smooth_geometry_ranking=True,
        native_relative_risk_residual=True,
        geometry_support_json='protocol28_support.json',
        geometry_metric='sym_kld', geometry_loss_weight=0.25,
        worst_case_retention_weight=0.0,
        geometry_min_gap=0.05, geometry_max_pairs=64)
    argv = trainer.build_locked_labeller_argv(args)
    assert '--s7-highres-native-relative-risk-residual' in argv
    assert argv[argv.index(
        '--s7-highres-relative-risk-retention-weight') + 1] == '4.0'
    assert argv[argv.index(
        '--s7-highres-relative-risk-preserve-weight') + 1] == '2.0'
