#!/usr/bin/env python3
"""Run the scope-gated frozen-DINO paper method from one config file."""

import argparse
import os
import runpy
import shlex
import sys
from typing import Dict, List, Tuple


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DINO_HEAD_EPOCHS = 8
DINO_HEAD_SELECTION_EPOCHS = tuple(range(1, DINO_HEAD_EPOCHS + 1))
DINO_HEAD_LR_STEPS = (5, 7)
DINO_HEAD_LR = 0.001
DINO_HEAD_CHECKPOINT_INTERVAL = 1
BRIGHTAUG_SELECTION_EPOCHS = (16, 18, 20, 22, 24)
PAPER_SOURCE_TRAIN_DATASETS = ('train:train', 'train_sim:train')
PAPER_SOURCE_VAL_DATASETS = ('val:val',)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Scope-Gated Frozen DINO Semantic Rescue runner')
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', required=True, choices=('train', 'test'))
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def load_method_config(path: str) -> Dict:
    try:
        from mmcv import Config
    except ModuleNotFoundError:
        method = runpy.run_path(path).get('dino_rescue')
    else:
        method = Config.fromfile(path).get('dino_rescue')
    if method is None:
        raise ValueError('Config has no dino_rescue section')
    return method.to_dict() if hasattr(method, 'to_dict') else dict(method)


def require_sections(method: Dict):
    required = ('baseline', 'data', 'dinov2', 'head', 'train', 'test')
    missing = [name for name in required if name not in method]
    if missing:
        raise ValueError('Missing dino_rescue sections: {}'.format(missing))
    if int(method['train'].get('seed', -1)) != 0:
        raise ValueError('Paper protocol requires train.seed=0')
    train = method['train']
    if int(train.get('epochs', -1)) != DINO_HEAD_EPOCHS:
        raise ValueError('DINO head protocol requires train.epochs=8')
    if tuple(train.get('selection_epochs', ())) != (
            DINO_HEAD_SELECTION_EPOCHS):
        raise ValueError(
            'DINO head protocol requires selection epochs 1 through 8')
    if tuple(train.get('lr_steps', ())) != DINO_HEAD_LR_STEPS:
        raise ValueError('DINO head protocol requires LR steps 5 and 7')
    if float(train.get('lr', -1.0)) != DINO_HEAD_LR:
        raise ValueError('DINO head protocol requires train.lr=0.001')
    if int(train.get('checkpoint_interval', -1)) != (
            DINO_HEAD_CHECKPOINT_INTERVAL):
        raise ValueError('DINO head protocol validates every epoch')
    data = method['data']
    if tuple(data.get('source_train_datasets', ())) != (
            PAPER_SOURCE_TRAIN_DATASETS):
        raise ValueError(
            'Paper protocol requires train and train_sim source training')
    if tuple(data.get('source_val_datasets', ())) != (
            PAPER_SOURCE_VAL_DATASETS):
        raise ValueError('Paper protocol requires official val selection')
    baseline = method['baseline']
    if baseline.get('selection_tool') != 'crane_project/tools/ckpt_sweep.py':
        raise ValueError('BrightAug checkpoint must come from ckpt_sweep.py')
    if baseline.get('selection_split') != 'val':
        raise ValueError('BrightAug checkpoint selection must use val only')
    if tuple(baseline.get('selection_epochs', ())) != (
            BRIGHTAUG_SELECTION_EPOCHS):
        raise ValueError(
            'BrightAug sweep must inspect epochs 16 18 20 22 24')
    if method['test'].get('scope_manifest') is None:
        raise ValueError('Paper test requires an explicit scope_manifest')


def add_arg(argv: List[str], name: str, value):
    argv.extend([name, str(value)])


def resolve_selected_baseline_checkpoint(method: Dict) -> str:
    """Resolve the checkpoint written by the source-only BrightAug sweep."""
    selection_file = method['baseline'].get('selection_file')
    if not selection_file:
        return method['baseline']['checkpoint']
    path = (selection_file if os.path.isabs(selection_file)
            else os.path.join(PROJ_ROOT, selection_file))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            'Run ckpt_sweep.py first; selection file is missing: {}'.format(
                path))
    with open(path, 'r', encoding='utf-8') as handle:
        selected = next((line.strip() for line in handle if line.strip()), '')
    if not selected:
        raise RuntimeError('BrightAug selection file is empty: {}'.format(path))
    checkpoint = (selected if os.path.isabs(selected)
                  else os.path.join(PROJ_ROOT, selected))
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            'Selected BrightAug checkpoint does not exist: {}'.format(
                checkpoint))
    return checkpoint


def common_args(method: Dict) -> List[str]:
    data, dino, head = (
        method['data'], method['dinov2'], method['head'])
    argv = []
    add_arg(argv, '--data-root', data['root'])
    add_arg(argv, '--source-split', data['source_split'])
    add_arg(argv, '--source-seq', data['source_seq'])
    add_arg(argv, '--source-val-modulus', data['source_val_modulus'])
    add_arg(argv, '--target-split', data['target_dev_split'])
    add_arg(argv, '--target-seq', data['target_dev_seq'])
    add_arg(argv, '--target-start', data['target_dev_start'])
    add_arg(argv, '--target-end', data['target_dev_end'])
    add_arg(argv, '--dinov2-repo', dino['repo'])
    add_arg(argv, '--dinov2-checkpoint', dino['checkpoint'])
    add_arg(argv, '--dinov2-model', dino['model'])
    argv.append('--dino-gpus')
    argv.extend(str(value) for value in dino['gpus'])
    add_arg(argv, '--head-gpu', head['gpu'])
    add_arg(argv, '--legacy-sdpa-query-chunk',
            dino['legacy_sdpa_query_chunk'])
    add_arg(argv, '--dino-height', dino['height'])
    add_arg(argv, '--dino-max-long-side', dino['max_long_side'])
    add_arg(argv, '--patch-size', dino['patch_size'])
    add_arg(argv, '--rpn-feat-channels', head['rpn_feat_channels'])
    add_arg(argv, '--roi-fc-channels', head['roi_fc_channels'])
    add_arg(argv, '--roi-samples', head['roi_samples'])
    add_arg(argv, '--proposal-count', head['proposal_count'])
    add_arg(argv, '--max-detections', head['max_detections'])
    return argv


def build_stage_command(method: Dict, stage: str) -> Tuple[str, List[str]]:
    require_sections(method)
    argv = common_args(method)
    if stage == 'train':
        train = method['train']
        data = method['data']
        argv.append('--source-train-datasets')
        argv.extend(str(value) for value in data['source_train_datasets'])
        argv.append('--source-val-datasets')
        argv.extend(str(value) for value in data['source_val_datasets'])
        add_arg(argv, '--epochs', train['epochs'])
        add_arg(argv, '--lr', train['lr'])
        add_arg(argv, '--momentum', train['momentum'])
        add_arg(argv, '--weight-decay', train['weight_decay'])
        add_arg(argv, '--max-grad-norm', train['max_grad_norm'])
        add_arg(argv, '--warmup-iters', train['warmup_iters'])
        add_arg(argv, '--warmup-ratio', train['warmup_ratio'])
        argv.append('--lr-steps')
        argv.extend(str(value) for value in train['lr_steps'])
        add_arg(argv, '--lr-gamma', train['lr_gamma'])
        add_arg(argv, '--checkpoint-interval',
                train['checkpoint_interval'])
        argv.append('--selection-epochs')
        argv.extend(str(value) for value in train['selection_epochs'])
        add_arg(argv, '--feature-cache-dir', train['feature_cache_dir'])
        add_arg(argv, '--work-dir', train['work_dir'])
        add_arg(argv, '--seed', train['seed'])
        add_arg(argv, '--out-json', train['out_json'])
        argv.append('--skip-target-eval')
        script = 'crane_project/tools/dino_teacher_rotated_labeller.py'
    elif stage == 'test':
        baseline, test, train = (
            method['baseline'], method['test'], method['train'])
        add_arg(argv, '--baseline-config', baseline['config'])
        add_arg(argv, '--baseline-checkpoint', baseline['checkpoint'])
        add_arg(argv, '--baseline-gpu', baseline['gpu'])
        add_arg(argv, '--labeller-checkpoint',
                test['labeller_checkpoint'])
        add_arg(argv, '--feature-cache-dir', test['feature_cache_dir'])
        add_arg(argv, '--scope-manifest', test['scope_manifest'])
        add_arg(argv, '--seed', train['seed'])
        add_arg(argv, '--out-json', test['out_json'])
        script = (
            'crane_project/tools/'
            'dino_teacher_baseline_first_rescue_audit.py')
    else:
        raise ValueError('Unsupported stage: {}'.format(stage))
    return script, argv


def main():
    args = parse_args()
    method = load_method_config(args.config)
    script, stage_argv = build_stage_command(method, args.stage)
    if args.stage == 'test' and not args.dry_run:
        selected = resolve_selected_baseline_checkpoint(method)
        checkpoint_index = stage_argv.index('--baseline-checkpoint') + 1
        stage_argv[checkpoint_index] = selected
    command = [sys.executable, os.path.join(PROJ_ROOT, script)] + stage_argv
    if args.dry_run:
        print(shlex.join(command))
        return
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
