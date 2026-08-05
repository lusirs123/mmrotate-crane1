#!/usr/bin/env python3
"""Locked stage-3 source-only temporal candidate student training."""

import argparse
import json
import os
import sys
from typing import List

PROJ_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import (
    dino_teacher_s7_temporal_fixed_target_audit as fixed_target)


TRAINING_NAME = 'DINO S7 Temporal Source Student Training V1'
EXPECTED_BASE_EPOCH = 4


def parse_args():
    parser = argparse.ArgumentParser(description=TRAINING_NAME)
    parser.add_argument(
        '--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-result-json', required=True)
    parser.add_argument('--init-checkpoint', required=True)
    parser.add_argument('--dinov2-repo', required=True)
    parser.add_argument('--dinov2-checkpoint', required=True)
    parser.add_argument('--dinov2-model', default='dinov2_vitl14')
    parser.add_argument('--dino-gpus', type=int, nargs='+', required=True)
    parser.add_argument('--head-gpu', type=int, default=0)
    parser.add_argument('--legacy-sdpa-query-chunk', type=int, default=512)
    parser.add_argument('--feature-cache-dir', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def validate_args(args):
    if args.seed != 0:
        raise ValueError('Stage-3 training requires --seed 0')
    if args.dinov2_model != 'dinov2_vitl14':
        raise ValueError('Stage-3 training requires dinov2_vitl14')
    for name in ('source_result_json', 'init_checkpoint',
                 'dinov2_checkpoint'):
        if not os.path.isfile(getattr(args, name)):
            raise ValueError('{} does not exist: {}'.format(
                name, getattr(args, name)))
    if not os.path.isdir(args.dinov2_repo):
        raise ValueError('dinov2_repo does not exist: {}'.format(
            args.dinov2_repo))
    if os.path.exists(args.out_json):
        raise ValueError('Refusing to overwrite result: {}'.format(
            args.out_json))
    if not args.dino_gpus or len(args.dino_gpus) != len(set(args.dino_gpus)):
        raise ValueError('DINO GPU ids must be non-empty and unique')
    if args.head_gpu in args.dino_gpus:
        raise ValueError('Head GPU must be separate from DINO GPUs')

    with open(args.source_result_json, 'r') as handle:
        source_result = json.load(handle)
    source_gate = fixed_target.strict_source_gate(source_result)
    if not source_gate['passed']:
        failed = sorted(name for name, passed
                        in source_gate['checks'].items() if not passed)
        raise ValueError('Phase-2 source gate failed: {}'.format(
            ', '.join(failed)))
    selected = source_gate.get('selected_checkpoint')
    if not selected or os.path.realpath(selected) != os.path.realpath(
            args.init_checkpoint):
        raise ValueError(
            'Stage-3 init checkpoint must equal the phase-2 '
            'source_selected_checkpoint')


def build_locked_labeller_argv(args) -> List[str]:
    return [
        'dino_teacher_rotated_labeller.py',
        '--data-root', args.data_root,
        '--source-train-datasets', 'train:train', 'train_sim:train',
        '--source-val-datasets', 'val:val',
        '--source-small-repeat', '1',
        '--source-retain-max-top1-drop', '0',
        '--dinov2-repo', args.dinov2_repo,
        '--dinov2-checkpoint', args.dinov2_checkpoint,
        '--dinov2-model', 'dinov2_vitl14',
        '--dino-gpus', *[str(gpu) for gpu in args.dino_gpus],
        '--head-gpu', str(args.head_gpu),
        '--legacy-sdpa-query-chunk', str(args.legacy_sdpa_query_chunk),
        '--dino-height', '600', '--dino-max-long-side', '1333',
        '--patch-size', '14', '--rpn-feat-channels', '256',
        '--roi-fc-channels', '1024', '--roi-samples', '256',
        '--proposal-count', '2000', '--max-detections', '2000',
        '--roi-nms-iou-thr', '0.5', '--riou-thr', '0.5',
        '--deployment-score-thr', '0.05', '--border-margin-ratio', '0.02',
        '--s7-residual', '--s7-channels', '128',
        '--s7-rpn-feat-channels', '128', '--s7-proposal-count', '500',
        '--s7-nms-pre', '2000',
        '--s7-anchor-sizes', '16', '32', '64', '128', '256',
        '--s7-merge-init-bias', '-2.0',
        '--s7-temporal-association', '--s7-temporal-quality-head',
        '--s7-temporal-quality-hidden', '128',
        '--s7-temporal-quality-loss-weight', '1.0',
        '--s7-temporal-relative-quality',
        '--s7-temporal-relative-quality-weight', '0.5',
        '--s7-temporal-relative-quality-margin', '0.25',
        '--s7-temporal-relative-quality-min-gap', '0.10',
        '--s7-temporal-relative-quality-max-pairs', '128',
        '--s7-temporal-relative-base-epoch', '4',
        '--s7-temporal-student',
        '--s7-temporal-max-candidates', '100',
        '--s7-temporal-min-confirmations', '1',
        '--s7-temporal-override-margin', '0.25',
        '--s7-temporal-max-center-distance', '3.0',
        '--s7-temporal-min-riou', '0.05',
        '--s7-temporal-min-appearance', '0.20',
        '--s7-student-base-epoch', str(EXPECTED_BASE_EPOCH),
        '--s7-student-hidden', '128',
        '--s7-student-quality-loss-weight', '1.0',
        '--s7-student-relative-loss-weight', '0.5',
        '--s7-student-distillation-weight', '1.0',
        '--s7-student-distillation-temperature', '1.0',
        '--s7-student-small-loss-weight', '2.0',
        '--s7-student-small-token-thr', '4.0',
        '--s7-student-teacher-result-json', args.source_result_json,
        '--s7-source-min-full-top1', '688',
        '--s7-source-min-small-top1', '311',
        '--s7-source-max-mcml', '3',
        '--train-components', 's7_temporal_student',
        '--epochs', '4', '--lr', '0.001', '--momentum', '0.9',
        '--weight-decay', '0.0001', '--max-grad-norm', '10',
        '--warmup-iters', '1000', '--warmup-ratio', '0.001',
        '--lr-steps', '2', '3', '--selection-epochs', '1', '2', '3', '4',
        '--checkpoint-interval', '1',
        '--init-checkpoint', args.init_checkpoint,
        '--feature-cache-dir', args.feature_cache_dir,
        '--work-dir', args.work_dir,
        '--skip-target-eval', '--seed', '0', '--out-json', args.out_json,
    ]


def main():
    args = parse_args()
    validate_args(args)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    original_argv = sys.argv
    try:
        sys.argv = build_locked_labeller_argv(args)
        labeller.main()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
