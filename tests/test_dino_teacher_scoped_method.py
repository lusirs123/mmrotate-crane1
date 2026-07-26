from crane_project.tools import dino_teacher_scoped_method as runner


def _method():
    return dict(
        baseline=dict(
            config='baseline.py', checkpoint='baseline.pth', gpu=0,
            selection_tool='crane_project/tools/ckpt_sweep.py',
            selection_split='val',
            selection_epochs=[16, 18, 20, 22, 24],
            selection_file=None),
        data=dict(root='data', source_split='val', source_seq='source',
                  source_val_modulus=5,
                  source_train_datasets=['train:train', 'train_sim:train'],
                  source_val_datasets=['val:val'], target_dev_split='test',
                  target_dev_seq='target', target_dev_start=137,
                  target_dev_end=169),
        dinov2=dict(repo='dinov2', checkpoint='dino.pth', model='model',
                    gpus=[1, 2], legacy_sdpa_query_chunk=512,
                    height=600, max_long_side=1333, patch_size=14),
        head=dict(gpu=0, rpn_feat_channels=256, roi_fc_channels=1024,
                  roi_samples=256, proposal_count=2000,
                  max_detections=2000),
        train=dict(epochs=8, lr=0.001, momentum=0.9,
                   weight_decay=0.0001, max_grad_norm=10.0,
                   warmup_iters=1000, warmup_ratio=0.001,
                   lr_steps=[5, 7], lr_gamma=0.1,
                   checkpoint_interval=1,
                   selection_epochs=[1, 2, 3, 4, 5, 6, 7, 8],
                   feature_cache_dir='cache', work_dir='work', seed=0,
                   out_json='train.json'),
        test=dict(labeller_checkpoint='head.pth',
                  feature_cache_dir='cache', scope_manifest='scope.json',
                  out_json='test.json'))


def test_train_command_uses_source_only_labeller_entrypoint():
    script, argv = runner.build_stage_command(_method(), 'train')
    assert script.endswith('dino_teacher_rotated_labeller.py')
    assert '--epochs' in argv
    assert argv[argv.index('--epochs') + 1] == '8'
    assert '--selection-epochs' in argv
    assert '--warmup-iters' in argv
    assert '--source-train-datasets' in argv
    assert '--source-val-datasets' in argv
    assert '--skip-target-eval' in argv
    assert '--baseline-checkpoint' not in argv


def test_test_command_uses_scoped_rescue_and_fixed_checkpoints():
    script, argv = runner.build_stage_command(_method(), 'test')
    assert script.endswith('dino_teacher_baseline_first_rescue_audit.py')
    assert '--baseline-checkpoint' in argv
    assert '--labeller-checkpoint' in argv
    assert '--scope-manifest' in argv


def test_method_requires_seed_zero_and_scope_manifest():
    method = _method()
    method['train']['seed'] = 1
    try:
        runner.require_sections(method)
    except ValueError as error:
        assert 'seed=0' in str(error)
    else:
        raise AssertionError('Expected seed validation failure')
    method = _method()
    method['test']['scope_manifest'] = None
    try:
        runner.require_sections(method)
    except ValueError as error:
        assert 'scope_manifest' in str(error)
    else:
        raise AssertionError('Expected scope validation failure')


def test_method_rejects_brightaug_schedule_for_dino_head():
    method = _method()
    method['train']['epochs'] = 24
    try:
        runner.require_sections(method)
    except ValueError as error:
        assert 'epochs=8' in str(error)
    else:
        raise AssertionError('Expected independent DINO schedule validation')


def test_method_rejects_sparse_dino_checkpoint_selection():
    method = _method()
    method['train']['checkpoint_interval'] = 2
    try:
        runner.require_sections(method)
    except ValueError as error:
        assert 'every epoch' in str(error)
    else:
        raise AssertionError('Expected per-epoch DINO validation')


def test_resolve_selected_baseline_checkpoint_uses_sweep_file(tmp_path):
    checkpoint = tmp_path / 'epoch_20.pth'
    checkpoint.write_bytes(b'checkpoint')
    selection = tmp_path / 'selected_checkpoint.txt'
    selection.write_text(str(checkpoint) + '\n', encoding='utf-8')
    method = _method()
    method['baseline']['selection_file'] = str(selection)
    assert runner.resolve_selected_baseline_checkpoint(method) == str(
        checkpoint)
