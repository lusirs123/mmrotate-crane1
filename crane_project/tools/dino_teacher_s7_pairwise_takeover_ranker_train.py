#!/usr/bin/env python3
"""Locked source-only Pairwise Takeover Ranker V2 training entry."""

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


TRAINING_NAME = 'DINO S7 Pairwise Takeover Ranker V2'
EXPECTED_PROTOCOL = 26
EXPECTED_BASE_EPOCH = 3


def parse_args():
    parser = argparse.ArgumentParser(description=TRAINING_NAME)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-margin-result-json', required=True)
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


def margin_result_gate(payload, checkpoint: str):
    audit = payload.get('source_highres_margin_audit') or {}
    protocol = payload.get('protocol') or {}
    isolation = payload.get('isolation') or {}
    candidate = payload.get('source_research_candidate_checkpoint')
    checks = dict(
        protocol_version=int(payload.get('protocol_version', -1))
        == EXPECTED_PROTOCOL,
        decision=(payload.get('decision') ==
                  'SOURCE_ONLY_UNIFIED_HIGHRES_BOUNDED_RISK_'
                  'RESEARCH_GATE_PASSED_TARGET_NOT_READ'),
        checkpoint_epoch=int(audit.get('checkpoint_epoch', -1))
        == EXPECTED_BASE_EPOCH,
        bounded_risk_variant=audit.get('audit_variant')
        == 'unified_bounded_risk',
        research_margin=float(payload.get(
            'research_candidate_promotion_margin', -1.0)) == 0.3,
        source_only=protocol.get('source_only') is True,
        target_not_read=(protocol.get('target_read') is False
                         and payload.get('target_dev') is None),
        read_only_audit=(isolation.get('read_only_evaluation') is True
                         and isolation.get('parameter_updates_performed')
                         is False),
        no_target_training=(isolation.get('target_used_for_training') is False
                            and isolation.get(
                                'target_used_for_checkpoint_selection')
                            is False),
        bounded_candidate_not_deployment=(
            payload.get('source_safe') is False
            and payload.get('eligible_for_deployment') is False
            and payload.get('eligible_for_full_test') is False),
        checkpoint_identity=bool(candidate) and os.path.realpath(
            str(candidate)) == os.path.realpath(checkpoint))
    return dict(checks=checks, passed=all(checks.values()))


def validate_args(args):
    if args.seed != 0:
        raise ValueError('Pairwise Takeover V2 requires --seed 0')
    if args.dinov2_model != 'dinov2_vitl14':
        raise ValueError('Pairwise Takeover V2 requires dinov2_vitl14')
    for name in ('source_margin_result_json', 'init_checkpoint',
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
    with open(args.source_margin_result_json, 'r') as handle:
        gate = margin_result_gate(json.load(handle), args.init_checkpoint)
    if not gate['passed']:
        failed = sorted(name for name, passed in gate['checks'].items()
                        if not passed)
        raise ValueError('Protocol-26 initialization gate failed: {}'.format(
            ', '.join(failed)))


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
        '--s7-highres-roi-ranker', '--s7-highres-pairwise-takeover-v2',
        '--s7-highres-base-epoch', str(EXPECTED_BASE_EPOCH),
        '--s7-highres-hidden', '32', '--s7-highres-channels', '32',
        '--s7-highres-max-candidates', '64',
        '--s7-takeover-initial-uncertainty', '0.25',
        '--s7-takeover-uncertainty-multiplier', '2.0',
        '--s7-takeover-margin', '0.05',
        '--s7-takeover-retention-margin', '0.10',
        '--s7-takeover-delta-weight', '1.0',
        '--s7-takeover-classification-weight', '1.0',
        '--s7-takeover-ranking-weight', '0.5',
        '--s7-takeover-retention-weight', '4.0',
        '--s7-takeover-gain-weight', '2.0',
        '--s7-takeover-consistency-weight', '0.5',
        '--s7-takeover-prior-weight', '0.01',
        '--s7-takeover-ranking-min-gap', '0.05',
        '--s7-takeover-max-ranking-pairs', '64',
        '--s7-takeover-group-dro-eta', '0.01',
        '--s7-highres-unified-aug-prob', '0.75',
        '--s7-highres-unified-aug-strength', '0.15',
        '--s7-highres-teacher-result-json',
        args.source_margin_result_json,
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
        '--skip-target-eval', '--seed', '0', '--out-json', args.out_json]


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
