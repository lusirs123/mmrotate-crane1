#!/usr/bin/env python3
"""Locked source-only stride-7 ROI quality/ranking experiment."""

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


TRAINING_NAME = 'DINO S7 Lightweight High-Resolution ROI Ranker A'
EXPECTED_BASE_EPOCH = 4
GEOMETRY_BASE_EPOCH = 3


def parse_args():
    parser = argparse.ArgumentParser(description=TRAINING_NAME)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
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
    parser.add_argument(
        '--unified-ranking', action='store_true',
        help=('Run the next-stage whole-pool hard-pair ranker with native '
              'protection and source feature-view augmentation.'))
    parser.add_argument(
        '--smooth-geometry-ranking', action='store_true',
        help=('Add the new source-only smooth-geometry auxiliary ranking '
              'objective. Requires unified ranking and a protocol-28 support '
              'artifact from the preceding read-only audit.'))
    parser.add_argument('--geometry-support-json', default=None)
    parser.add_argument('--geometry-metric', default='sym_kld',
                        choices=['sym_kld', 'gwd', 'normalized_gwd'])
    parser.add_argument('--geometry-loss-weight', type=float, default=0.25)
    parser.add_argument('--worst-case-retention-weight', type=float, default=4.0)
    parser.add_argument('--geometry-min-gap', type=float, default=0.05)
    parser.add_argument('--geometry-max-pairs', type=int, default=64)
    return parser.parse_args()


def validate_args(args):
    if args.seed != 0:
        raise ValueError('High-resolution ranker training requires --seed 0')
    if args.dinov2_model != 'dinov2_vitl14':
        raise ValueError(
            'High-resolution ranker training requires dinov2_vitl14')
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
    if bool(getattr(args, 'smooth_geometry_ranking', False)):
        if not args.unified_ranking:
            raise ValueError('--smooth-geometry-ranking requires '
                             '--unified-ranking')
        if not args.geometry_support_json:
            raise ValueError('--smooth-geometry-ranking requires '
                             '--geometry-support-json')
        with open(args.geometry_support_json, 'r') as handle:
            support = json.load(handle)
        checks = dict(
            protocol_version=int(support.get('protocol_version', -1)) == 28,
            target_not_read=support.get('target_dev') is None,
            eligible_for_training=(support.get(
                'eligible_for_training') is True),
            support_gate=((support.get('source') or {}).get(
                'support_gate') or {}).get('passed') is True,
            quality_gate=((support.get('source') or {}).get(
                'quality_gate') or {}).get('passed') is True,
            checkpoint_epoch=int((support.get('checkpoint_used') or {}).get(
                'epoch', -1)) == GEOMETRY_BASE_EPOCH,
            checkpoint_identity=os.path.realpath(str(
                (support.get('checkpoint_used') or {}).get('path', '')))
            == os.path.realpath(args.init_checkpoint))
        if not all(checks.values()):
            raise ValueError('Smooth-geometry support provenance failed: '
                             + ', '.join(sorted(
                                 name for name, passed in checks.items()
                                 if not passed)))
        if args.geometry_loss_weight <= 0.0 or args.geometry_min_gap <= 0.0 \
                or args.geometry_max_pairs <= 0:
            raise ValueError('Smooth-geometry training settings must be positive')
        if args.worst_case_retention_weight <= 0.0:
            raise ValueError(
                'Smooth-geometry training requires positive worst-case '
                'retention weight')
    else:
        with open(args.source_result_json, 'r') as handle:
            source_result = json.load(handle)
        source_gate = fixed_target.strict_source_gate(source_result)
        if not source_gate['passed']:
            failed = sorted(name for name, passed in source_gate['checks'].items()
                            if not passed)
            raise ValueError('Phase-2 source gate failed: {}'.format(
                ', '.join(failed)))
        provenance = labeller.phase2_selected_checkpoint_provenance_gate(
            source_result, args.init_checkpoint,
            expected_epoch=EXPECTED_BASE_EPOCH,
            min_full_top1=688, min_small_top1=311, max_mcml=3)
        if not provenance['passed']:
            failed = sorted(name for name, passed in provenance['checks'].items()
                            if not passed)
            raise ValueError(
                'High-resolution phase-2 checkpoint provenance gate failed: '
                + ', '.join(failed))


def build_locked_labeller_argv(args) -> List[str]:
    smooth_geometry = bool(getattr(args, 'smooth_geometry_ranking', False))
    base_epoch = (GEOMETRY_BASE_EPOCH if smooth_geometry
                  else EXPECTED_BASE_EPOCH)
    argv = [
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
        '--s7-highres-base-epoch', str(base_epoch),
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
        *(['--s7-highres-unified-ranking']
          if bool(getattr(args, 'unified_ranking', False)) else []),
        '--s7-highres-unified-hard-pairs', '8',
        '--s7-highres-unified-aug-prob', '0.75',
        '--s7-highres-unified-aug-strength', '0.15',
        '--s7-source-min-full-top1', '688',
        '--s7-source-min-small-top1', '311',
        '--s7-source-max-mcml', '3',
        '--train-components', 's7_highres_roi_ranker',
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
    if smooth_geometry:
        argv.extend([
            '--s7-highres-smooth-geometry-ranking',
            '--s7-highres-smooth-geometry-support-result-json',
            args.geometry_support_json,
            '--s7-highres-smooth-geometry-metric', args.geometry_metric,
            '--s7-highres-smooth-geometry-loss-weight',
            str(args.geometry_loss_weight),
            '--s7-highres-worst-case-retention-weight',
            str(getattr(args, 'worst_case_retention_weight', 4.0)),
            '--s7-highres-smooth-geometry-min-gap',
            str(args.geometry_min_gap),
            '--s7-highres-smooth-geometry-max-pairs',
            str(args.geometry_max_pairs)])
    else:
        argv.extend(['--s7-highres-teacher-result-json',
                     args.source_result_json])
    return argv


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
