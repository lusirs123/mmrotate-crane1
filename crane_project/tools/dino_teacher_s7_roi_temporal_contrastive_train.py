#!/usr/bin/env python3
"""Run protocol-34 source-only ROI temporal contrastive adapter training."""

import argparse
import os
import sys
from typing import List


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import dino_teacher_rotated_labeller as labeller


LOCKED_EPOCH = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description='Source-only ROI Temporal Contrastive Candidate Adapter V1')
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--support-result-json', required=True)
    parser.add_argument('--eval-only-checkpoint', required=True)
    parser.add_argument('--dinov2-repo', required=True)
    parser.add_argument('--dinov2-checkpoint', required=True)
    parser.add_argument('--dino-gpus', type=int, nargs='+', required=True)
    parser.add_argument('--head-gpu', type=int, required=True)
    parser.add_argument('--legacy-sdpa-query-chunk', type=int, default=256)
    parser.add_argument('--feature-cache-dir', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--promotion-margin', type=float, default=0.05)
    parser.add_argument('--motion-weight', type=float, default=0.25)
    parser.add_argument('--empty-cache-interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def validate_args(args):
    if args.seed != 0:
        raise ValueError('Protocol-34 requires --seed 0')
    for name in ('support_result_json', 'eval_only_checkpoint',
                 'dinov2_checkpoint'):
        if not os.path.isfile(getattr(args, name)):
            raise ValueError('{} does not exist: {}'.format(
                name, getattr(args, name)))
    if not os.path.isdir(args.dinov2_repo):
        raise ValueError('dinov2_repo does not exist: {}'.format(
            args.dinov2_repo))
    if os.path.exists(args.out_json):
        raise ValueError('Refusing to overwrite result: {}'.format(args.out_json))
    for name in ('roi_temporal_source_train_cache_fp16.pt',
                 'roi_temporal_adapter_source_only.pth'):
        path = os.path.join(args.work_dir, name)
        if os.path.exists(path):
            raise ValueError('Refusing to overwrite protocol-34 artifact: {}'.format(
                path))
    if (not args.dino_gpus
            or len(args.dino_gpus) != len(set(args.dino_gpus))):
        raise ValueError('DINO GPU ids must be non-empty and unique')
    if args.head_gpu in args.dino_gpus:
        raise ValueError('Head GPU must be separate from DINO GPUs')
    if not 1 <= args.batch_size <= 256:
        raise ValueError('--batch-size must be in [1, 256] for the 8G route')
    if not 1 <= args.legacy_sdpa_query_chunk <= 256:
        raise ValueError('--legacy-sdpa-query-chunk must be in [1, 256]')
    if args.epochs < 1 or args.epochs > 8:
        raise ValueError('--epochs must be in [1, 8]')
    if float(args.promotion_margin) != 0.05:
        raise ValueError('Protocol-34 locks --promotion-margin 0.05')
    if float(args.motion_weight) != 0.25:
        raise ValueError('Protocol-34 locks --motion-weight 0.25')
    labeller.load_roi_temporal_support_spec(
        args.support_result_json, args.eval_only_checkpoint)


def build_locked_labeller_argv(args) -> List[str]:
    return [
        'dino_teacher_rotated_labeller.py',
        '--data-root', args.data_root,
        '--source-train-datasets', 'train:train', 'train_sim:train',
        '--source-val-datasets', 'val:val',
        '--source-small-repeat', '1', '--source-retain-max-top1-drop', '0',
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
        '--s7-highres-roi-ranker', '--s7-highres-unified-ranking',
        '--s7-highres-base-epoch', str(LOCKED_EPOCH),
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
        '--s7-highres-gain-weight', '1.0', '--s7-highres-prior-weight', '0.01',
        '--s7-highres-unified-hard-pairs', '8',
        '--s7-highres-unified-aug-prob', '0.75',
        '--s7-highres-unified-aug-strength', '0.15',
        '--source-roi-temporal-contrastive-train',
        '--source-roi-temporal-support-result-json', args.support_result_json,
        '--source-roi-temporal-hidden', '128',
        '--source-roi-temporal-output', '64',
        '--source-roi-temporal-batch-size', str(args.batch_size),
        '--source-roi-temporal-temperature', str(args.temperature),
        '--source-roi-temporal-promotion-margin', str(args.promotion_margin),
        '--source-roi-temporal-motion-weight', str(args.motion_weight),
        '--source-roi-temporal-empty-cache-interval',
        str(args.empty_cache_interval),
        '--source-roi-temporal-holdout-seq', 'real_seq05',
        '--source-dense-temporal-negative-riou-max', '0.30',
        '--source-dense-temporal-hard-negatives', '8',
        '--s7-source-min-full-top1', '688',
        '--s7-source-min-small-top1', '311', '--s7-source-max-mcml', '3',
        '--train-components', 's7_highres_roi_ranker',
        '--epochs', str(args.epochs), '--lr', str(args.lr),
        '--weight-decay', '0.0001', '--max-grad-norm', '10',
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
