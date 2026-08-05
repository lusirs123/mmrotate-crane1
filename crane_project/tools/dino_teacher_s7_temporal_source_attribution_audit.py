#!/usr/bin/env python3
"""Strict source-only attribution audit for the selected temporal S7 epoch.

This is a small locked wrapper around the labeller's existing attribution
implementation.  It prevents accidentally running the attribution command as
training, reading target-dev annotations, changing the checkpoint epoch, or
silently evaluating a checkpoint that was not selected by the source-only
relative-quality run.
"""

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


AUDIT_NAME = 'DINO S7 Temporal Source Attribution Audit V1'
EXPECTED_EPOCH = 4


def parse_args(description=None):
    parser = argparse.ArgumentParser(
        description=(AUDIT_NAME if description is None else description))
    parser.add_argument(
        '--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-result-json', required=True)
    parser.add_argument('--eval-only-checkpoint', required=True)
    parser.add_argument('--dinov2-repo', required=True)
    parser.add_argument('--dinov2-checkpoint', required=True)
    parser.add_argument('--dinov2-model', default='dinov2_vitl14')
    parser.add_argument('--dino-gpus', type=int, nargs='+', required=True)
    parser.add_argument('--head-gpu', type=int, default=0)
    parser.add_argument('--legacy-sdpa-query-chunk', type=int, default=512)
    parser.add_argument('--feature-cache-dir', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def _load_source_result(path: str):
    with open(path, 'r') as handle:
        return json.load(handle)


def validate_args(args):
    if args.seed != 0:
        raise ValueError('The fixed attribution protocol requires --seed 0')
    if args.dinov2_model != 'dinov2_vitl14':
        raise ValueError(
            'The fixed attribution protocol requires dinov2_vitl14')
    for name in (
            'source_result_json', 'eval_only_checkpoint',
            'dinov2_repo', 'dinov2_checkpoint'):
        value = getattr(args, name)
        if name != 'dinov2_repo' and not os.path.isfile(value):
            raise ValueError('{} does not exist: {}'.format(name, value))
        if name == 'dinov2_repo' and not os.path.isdir(value):
            raise ValueError('{} does not exist: {}'.format(name, value))
    if os.path.exists(args.out_json):
        raise ValueError(
            'Refusing to overwrite completed attribution: {}'.format(
                args.out_json))
    if not args.dino_gpus or len(args.dino_gpus) != len(set(args.dino_gpus)):
        raise ValueError('DINO GPU ids must be non-empty and unique')
    if args.head_gpu in args.dino_gpus:
        raise ValueError('Head GPU must be separate from DINO GPUs')
    if int(args.legacy_sdpa_query_chunk) <= 0:
        raise ValueError('--legacy-sdpa-query-chunk must be positive')

    source_result = _load_source_result(args.source_result_json)
    source_gate = fixed_target.strict_source_gate(source_result)
    if not source_gate['passed']:
        failed = sorted(
            name for name, passed in source_gate['checks'].items()
            if not passed)
        raise ValueError(
            'Strict source gate failed: {}'.format(', '.join(failed)))
    selected = source_gate.get('selected_checkpoint')
    if not selected or os.path.realpath(selected) != os.path.realpath(
            args.eval_only_checkpoint):
        raise ValueError(
            'Attribution checkpoint must equal source_selected_checkpoint')

    import torch
    payload = torch.load(args.eval_only_checkpoint, map_location='cpu')
    checkpoint_gate = fixed_target.candidate_checkpoint_gate(
        payload, source_gate)
    if not checkpoint_gate['passed']:
        failed = sorted(
            name for name, passed in checkpoint_gate['checks'].items()
            if not passed)
        raise ValueError(
            'Attribution checkpoint gate failed: {}'.format(
                ', '.join(failed)))
    if int(payload.get('epoch', -1)) != EXPECTED_EPOCH:
        raise ValueError(
            'Attribution is locked to epoch {}, got {}'.format(
                EXPECTED_EPOCH, payload.get('epoch')))
    if source_result.get('target_dev') is not None:
        raise ValueError('Source result already contains target-dev output')
    args.source_gate = source_gate
    args.checkpoint_gate = checkpoint_gate
    args.source_result = source_result


def build_locked_labeller_argv(args) -> List[str]:
    """Build all labeller flags instead of exposing tunable audit knobs."""
    work_dir = os.path.abspath(os.path.dirname(args.out_json))
    return [
        'dino_teacher_rotated_labeller.py',
        '--data-root', args.data_root,
        '--source-train-datasets', 'train:train', 'train_sim:train',
        '--source-val-datasets', 'val:val',
        '--source-small-repeat', '1',
        '--dinov2-repo', args.dinov2_repo,
        '--dinov2-checkpoint', args.dinov2_checkpoint,
        '--dinov2-model', 'dinov2_vitl14',
        '--dino-gpus', *[str(gpu) for gpu in args.dino_gpus],
        '--head-gpu', str(args.head_gpu),
        '--legacy-sdpa-query-chunk', str(args.legacy_sdpa_query_chunk),
        '--dino-height', '600',
        '--dino-max-long-side', '1333',
        '--patch-size', '14',
        '--rpn-feat-channels', '256',
        '--roi-fc-channels', '1024',
        '--roi-samples', '256',
        '--proposal-count', '2000',
        '--max-detections', '2000',
        '--roi-nms-iou-thr', '0.5',
        '--riou-thr', '0.5',
        '--deployment-score-thr', '0.05',
        '--border-margin-ratio', '0.02',
        '--s7-residual',
        '--s7-channels', '128',
        '--s7-rpn-feat-channels', '128',
        '--s7-proposal-count', '500',
        '--s7-nms-pre', '2000',
        '--s7-anchor-sizes', '16', '32', '64', '128', '256',
        '--s7-merge-init-bias', '-2.0',
        '--s7-temporal-association',
        '--s7-temporal-quality-head',
        '--s7-temporal-quality-hidden', '128',
        '--s7-temporal-quality-loss-weight', '1.0',
        '--s7-temporal-relative-quality',
        '--s7-temporal-relative-quality-weight', '0.5',
        '--s7-temporal-relative-quality-margin', '0.25',
        '--s7-temporal-relative-quality-min-gap', '0.10',
        '--s7-temporal-relative-quality-max-pairs', '128',
        '--s7-temporal-relative-base-epoch', '4',
        '--s7-temporal-max-candidates', '100',
        '--s7-temporal-min-confirmations', '1',
        '--s7-temporal-override-margin', '0.25',
        '--s7-temporal-max-center-distance', '3.0',
        '--s7-temporal-min-riou', '0.05',
        '--s7-temporal-min-appearance', '0.20',
        '--s7-source-min-full-top1', '688',
        '--s7-source-min-small-top1', '311',
        '--s7-source-max-mcml', '3',
        '--train-components', 's7_temporal_association',
        '--epochs', '4',
        '--lr-steps', '2', '3',
        '--selection-epochs', '1', '2', '3', '4',
        '--checkpoint-interval', '1',
        '--eval-only-checkpoint', args.eval_only_checkpoint,
        '--source-temporal-attribution-audit',
        '--source-temporal-attribution-epoch', str(EXPECTED_EPOCH),
        '--skip-target-eval',
        '--feature-cache-dir', args.feature_cache_dir,
        '--work-dir', work_dir,
        '--seed', '0',
        '--out-json', args.out_json,
    ]


def main():
    args = parse_args()
    validate_args(args)
    os.makedirs(os.path.abspath(os.path.dirname(args.out_json)), exist_ok=True)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    locked_argv = build_locked_labeller_argv(args)
    original_argv = sys.argv
    try:
        sys.argv = locked_argv
        labeller.main()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
