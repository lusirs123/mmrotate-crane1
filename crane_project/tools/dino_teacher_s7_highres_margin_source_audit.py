#!/usr/bin/env python3
"""Locked one-pass source-only audit of three high-resolution margins."""

import argparse
import os
import sys
from typing import List


PROJ_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import dino_teacher_rotated_labeller as labeller


AUDIT_NAME = 'DINO S7 High-Resolution Shared-Forward Margin Audit'
LOCKED_EPOCH = 3
LOCKED_MARGINS = (0.20, 0.225, 0.25)


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-result-json', required=True)
    parser.add_argument('--eval-only-checkpoint', required=True)
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
        raise ValueError('High-resolution margin audit requires --seed 0')
    if args.dinov2_model != 'dinov2_vitl14':
        raise ValueError('High-resolution margin audit requires dinov2_vitl14')
    for name in ('source_result_json', 'eval_only_checkpoint',
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
    labeller.load_highres_margin_audit_spec(
        args.source_result_json, args.eval_only_checkpoint, LOCKED_EPOCH)


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
        '--s7-highres-roi-ranker',
        '--s7-highres-base-epoch', '4',
        '--s7-highres-hidden', '32', '--s7-highres-channels', '32',
        '--s7-highres-max-candidates', '32',
        '--s7-highres-score-weight', '1.0',
        '--s7-highres-rank-margin', '0.25',
        '--s7-highres-promotion-margin', '0.25',
        '--s7-highres-quality-loss-weight', '1.0',
        '--s7-highres-relative-loss-weight', '0.5',
        '--s7-highres-relative-min-gap', '0.10',
        '--s7-highres-relative-max-pairs', '128',
        '--s7-highres-retention-weight', '2.0',
        '--s7-highres-gain-weight', '1.0',
        '--s7-highres-prior-weight', '0.01',
        '--source-highres-margin-audit',
        '--source-highres-margin-source-result-json',
        args.source_result_json,
        '--source-highres-margin-values',
        *[str(value) for value in LOCKED_MARGINS],
        '--source-highres-margin-epoch', str(LOCKED_EPOCH),
        '--s7-source-min-full-top1', '688',
        '--s7-source-min-small-top1', '311',
        '--s7-source-max-mcml', '3',
        '--train-components', 's7_highres_roi_ranker',
        '--epochs', '4', '--lr', '0.001', '--momentum', '0.9',
        '--weight-decay', '0.0001', '--max-grad-norm', '10',
        '--warmup-iters', '1000', '--warmup-ratio', '0.001',
        '--lr-steps', '2', '3', '--selection-epochs', '1', '2', '3', '4',
        '--checkpoint-interval', '1',
        '--eval-only-checkpoint', args.eval_only_checkpoint,
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
