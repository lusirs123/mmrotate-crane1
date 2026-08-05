#!/usr/bin/env python3
"""Train a source-only rotated detector on a frozen DINOv2 feature map.

This is the bounded SymEOOD adaptation of the CVPR 2025 DINO Teacher
labeller.  A frozen single-scale DINOv2 feature map feeds an Oriented RPN and
an oriented two-FC ROI box head.  Only the RPN and ROI head are optimized.
Source validation selects the checkpoint; target-dev annotations are first
read after the source-selected checkpoint has been fixed.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


PROJ_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import dino_teacher_common as common  # noqa: E402
from crane_project.utils import s7_temporal_association as temporal  # noqa: E402


LABELLER_NAME = 'Frozen DINOv2 Oriented RPN/ROI Source Labeller V1'
PROTOCOL_VERSION = 20
PAIRWISE_V2_MAX_EPOCHS = 4
S7_QUALITY_MIN_FULL_TOP1 = 688
S7_QUALITY_MIN_SMALL_TOP1 = 311
S7_QUALITY_MIN_RISK_PAIRS = 1
SOURCE_TEMPORAL_ANGLE_LIMIT_DEG = 35.0
PAPER_URL = (
    'https://openaccess.thecvf.com/content/CVPR2025/html/'
    'Lavoie_Large_Self-Supervised_Models_Bridge_the_Gap_in_Domain_Adaptive_'
    'Object_CVPR_2025_paper.html')
PAPER_CODE_URL = 'https://github.com/TRAILab/DINO_Teacher'
LIDERE_PAPER_URL = (
    'https://openaccess.thecvf.com/content/CVPR2026/html/'
    'Luddecke_LiDeRe_A_Lightweight_Readout_for_Fast_and_Data-Efficient_'
    'Dense_Prediction_CVPR_2026_paper.html')

def parse_args():
    parser = argparse.ArgumentParser(description=LABELLER_NAME)
    parser.add_argument(
        '--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-split', default=common.SOURCE_SPLIT)
    parser.add_argument('--source-seq', default=common.SOURCE_SEQ)
    parser.add_argument('--source-val-modulus', type=int, default=5)
    parser.add_argument(
        '--source-train-datasets', nargs='+',
        help='Formal source train specs: annotation_split:image_split')
    parser.add_argument(
        '--source-val-datasets', nargs='+',
        help='Formal source validation specs: annotation_split:image_split')
    parser.add_argument('--target-split', default=common.TARGET_SPLIT)
    parser.add_argument('--target-seq', default=common.TARGET_SEQ)
    parser.add_argument('--target-start', type=int,
                        default=common.TARGET_START)
    parser.add_argument('--target-end', type=int,
                        default=common.TARGET_END)
    parser.add_argument('--dinov2-repo', required=True)
    parser.add_argument('--dinov2-checkpoint', required=True)
    parser.add_argument('--dinov2-model', default=common.CANONICAL_MODEL)
    parser.add_argument('--dino-gpus', type=int, nargs='+', required=True)
    parser.add_argument('--head-gpu', type=int, default=0)
    parser.add_argument('--legacy-sdpa-query-chunk', type=int, default=512)
    parser.add_argument('--dino-height', type=int,
                        default=common.CANONICAL_DINO_HEIGHT)
    parser.add_argument('--dino-max-long-side', type=int,
                        default=common.CANONICAL_DINO_MAX_LONG_SIDE)
    parser.add_argument('--patch-size', type=int, default=14)
    parser.add_argument('--rpn-feat-channels', type=int, default=256)
    parser.add_argument('--roi-fc-channels', type=int, default=1024)
    parser.add_argument('--roi-samples', type=int, default=256)
    parser.add_argument('--proposal-count', type=int, default=2000)
    parser.add_argument('--max-detections', type=int, default=2000)
    parser.add_argument(
        '--feature-strides', type=int, nargs='+', default=None,
        help=('Optional DINO feature pyramid strides. Default is the original '
              'single stride equal to --patch-size.'))
    parser.add_argument(
        '--s7-residual', action='store_true',
        help=('Enable the protected stride-7 residual readout and its '
              'proposal-only Oriented RPN. The native stride-14 path stays '
              'unchanged.'))
    parser.add_argument('--s7-channels', type=int, default=128)
    parser.add_argument('--s7-rpn-feat-channels', type=int, default=128)
    parser.add_argument('--s7-proposal-count', type=int, default=500)
    parser.add_argument('--s7-nms-pre', type=int, default=2000)
    parser.add_argument(
        '--s7-anchor-sizes', type=float, nargs='+',
        default=[16.0, 32.0, 64.0, 128.0, 256.0])
    parser.add_argument('--s7-source-min-full-top1', type=int, default=677)
    parser.add_argument('--s7-source-min-small-top1', type=int, default=303)
    parser.add_argument('--s7-source-max-mcml', type=int, default=3)
    parser.add_argument(
        '--s7-component-checkpoint',
        help=('Rejected full S7 checkpoint used only as a frozen source for '
              's7_readout.* and s7_rpn_head.* in s7_merge mode.'))
    parser.add_argument('--s7-merge-init-bias', type=float, default=-2.0)
    parser.add_argument('--s7-merge-margin', type=float, default=0.5)
    parser.add_argument('--s7-merge-retention-weight', type=float, default=2.0)
    parser.add_argument('--s7-merge-gain-weight', type=float, default=1.0)
    parser.add_argument('--s7-merge-prior-weight', type=float, default=0.01)
    parser.add_argument(
        '--roi-nms-iou-thr', type=float, default=0.1,
        help='Frozen DINO ROI rotated-NMS IoU threshold.')
    parser.add_argument('--valid-content-tolerance', type=float, default=1e-3)
    parser.add_argument('--deployment-score-thr', type=float, default=0.05)
    parser.add_argument('--border-margin-ratio', type=float, default=0.02)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-grad-norm', type=float, default=10.0)
    parser.add_argument('--warmup-iters', type=int, default=1000)
    parser.add_argument('--warmup-ratio', type=float, default=0.001)
    parser.add_argument(
        '--lr-steps', type=int, nargs='+', default=None,
        help=('Defaults to [5, 7] for the formal eight-epoch head training '
              'and [2, 3] for the at-most-four-epoch Pairwise V2 fine-tune.'))
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    parser.add_argument('--checkpoint-interval', type=int, default=1)
    parser.add_argument('--selection-epochs', type=int, nargs='+')
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--target-min-wins', type=int, default=26)
    parser.add_argument('--max-mcml', type=int, default=5)
    parser.add_argument('--source-min-top1-rate', type=float, default=0.8)
    parser.add_argument(
        '--train-components',
        choices=['all', 'roi_cls', 'roi_cls_pairwise',
                 'roi_cls_pairwise_v2', 's7_rpn', 's7_merge',
                 's7_lane_arbitration', 's7_quality_suppression',
                 's7_temporal_association'],
        default='all',
        help=('Train all RPN/ROI heads, or only the final ROI classifier '
              'fc_cls while keeping RPN, shared ROI FCs, and bbox regression '
              'fixed. roi_cls_pairwise additionally uses source-only '
              'NMS-aware pairwise ranking and the initialized classifier as '
              'a frozen retention teacher. roi_cls_pairwise_v2 mines only '
              'false ROIs that actually outrank a usable ROI. s7_rpn trains '
              'only the residual stride-7 readout and proposal head. s7_merge '
              'freezes both proposal paths and trains only a pre-NMS affine '
              'S7 score calibrator with native-retention pairs. '
              's7_lane_arbitration freezes the epoch-1 merge and trains only '
              'a bounded source-aware S7 lane arbitrator. '
              's7_quality_suppression freezes the epoch-1 affine merge and '
              'trains one lane-wide non-positive source-quality penalty. '
              's7_temporal_association freezes every detector component and '
              'fits only six positive causal association cue weights. The '
              'optional quality-head flag adds dense continuous candidate '
              'max-RIoU supervision. The relative-quality flag adds a '
              'source-only same-frame pairwise ranking term on that head.'))
    parser.add_argument(
        '--source-small-repeat', type=int, default=1,
        help='Repeat source-train frames in the lower short-token tertile.')
    parser.add_argument(
        '--source-retain-max-top1-drop', type=int, default=0,
        help='Maximum source-val top1 drop allowed for ROI-cls selection.')
    parser.add_argument('--pairwise-margin', type=float, default=0.5)
    parser.add_argument(
        '--pairwise-cls-loss-weight', type=float, default=0.25,
        help='Weight of CE on the V2 paired ROI samples.')
    parser.add_argument('--pairwise-loss-weight', type=float, default=1.0)
    parser.add_argument('--retention-loss-weight', type=float, default=1.0)
    parser.add_argument('--retention-temperature', type=float, default=1.0)
    parser.add_argument('--s7-lane-hidden', type=int, default=32)
    parser.add_argument('--s7-lane-max-adjustment', type=float, default=2.0)
    parser.add_argument('--s7-lane-base-epoch', type=int, default=1)
    parser.add_argument(
        '--s7-lane-hard-negatives', type=int, default=4,
        help=('Number of current adjusted-logit S7 false candidates used by '
              'each native-retention frame.'))
    parser.add_argument(
        '--s7-lane-gain-repeat', type=int, default=8,
        help=('Within each source-train epoch, process each mined gain frame '
              'this many times in total.'))
    parser.add_argument('--s7-quality-hidden', type=int, default=32)
    parser.add_argument(
        '--s7-quality-max-suppression', type=float, default=2.0,
        help='Maximum lane-wide non-positive S7 log-odds adjustment.')
    parser.add_argument(
        '--s7-quality-init-risk-bias', type=float, default=0.0,
        help=('Initial risk logit. Zero reproduces the audited affine '
              'epoch-1 behavior exactly while BCE gradients remain active.'))
    parser.add_argument('--s7-quality-margin', type=float, default=0.5)
    parser.add_argument('--s7-quality-risk-weight', type=float, default=1.0)
    parser.add_argument(
        '--s7-quality-preserve-weight', type=float, default=1.0,
        help=('Keep suppression near zero when the fixed-affine S7 lane '
              'winner is source-GT usable; this never promotes S7.'))
    parser.add_argument(
        '--s7-quality-retention-weight', type=float, default=2.0)
    parser.add_argument('--s7-quality-prior-weight', type=float, default=0.01)
    parser.add_argument('--s7-quality-base-epoch', type=int, default=1)
    parser.add_argument(
        '--s7-temporal-association', action='store_true',
        help=('Enable source-only causal multi-cue candidate association on '
              'the fixed affine epoch-1 native/S7 pool.'))
    parser.add_argument(
        '--s7-temporal-quality-head', action='store_true',
        help=('Add a source-only candidate-level continuous max-RIoU head as '
              'the seventh temporal cue.'))
    parser.add_argument('--s7-temporal-quality-hidden', type=int, default=128)
    parser.add_argument(
        '--s7-temporal-quality-loss-weight', type=float, default=1.0)
    parser.add_argument(
        '--s7-temporal-relative-quality', action='store_true',
        help=(
            'Add source-only same-frame candidate relative-ranking loss to '
            'the continuous max-RIoU quality head.'))
    parser.add_argument(
        '--s7-temporal-relative-quality-weight', type=float, default=0.5)
    parser.add_argument(
        '--s7-temporal-relative-quality-margin', type=float, default=0.25)
    parser.add_argument(
        '--s7-temporal-relative-quality-min-gap', type=float, default=0.10)
    parser.add_argument(
        '--s7-temporal-relative-quality-max-pairs', type=int, default=128)
    parser.add_argument(
        '--s7-temporal-relative-base-epoch', type=int, default=4,
        help='Epoch of the source-gated pointwise quality checkpoint.')
    parser.add_argument('--s7-temporal-base-epoch', type=int, default=1)
    parser.add_argument('--s7-temporal-margin', type=float, default=0.5)
    parser.add_argument(
        '--s7-temporal-retention-weight', type=float, default=2.0)
    parser.add_argument('--s7-temporal-gain-weight', type=float, default=1.0)
    parser.add_argument('--s7-temporal-prior-weight', type=float, default=0.01)
    parser.add_argument('--s7-temporal-max-candidates', type=int, default=100)
    parser.add_argument('--s7-temporal-min-confirmations', type=int, default=2)
    parser.add_argument('--s7-temporal-override-margin', type=float, default=0.25)
    parser.add_argument(
        '--s7-temporal-max-center-distance', type=float, default=3.0)
    parser.add_argument('--s7-temporal-min-riou', type=float, default=0.05)
    parser.add_argument('--s7-temporal-min-appearance', type=float, default=0.20)
    parser.add_argument(
        '--pairwise-negative-riou-thr', type=float, default=0.1,
        help='Maximum GT RIoU for a decoded ROI to be a pairwise negative.')
    parser.add_argument(
        '--pairwise-nms-iou-thr', type=float, default=0.1,
        help='Prioritize false ROIs that can suppress a positive at NMS.')
    parser.add_argument(
        '--pairwise-negatives-per-positive', type=int, default=2,
        help=('Maximum actual higher-scoring false competitors paired with '
              'each usable ROI in pairwise V2.'))
    parser.add_argument('--feature-cache-dir', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument(
        '--init-checkpoint',
        help='Initialize head weights without resuming epoch or optimizer.')
    parser.add_argument('--resume-checkpoint')
    parser.add_argument('--eval-only-checkpoint')
    parser.add_argument(
        '--source-conflict-result-json',
        help=('Existing source-only train_result.json whose lost/gained keys '
              'define a bounded S7 merge conflict audit.'))
    parser.add_argument('--source-conflict-epoch', type=int, default=1)
    parser.add_argument(
        '--source-temporal-attribution-audit', action='store_true',
        help=(
            'Read-only source-val attribution of a fixed rejected temporal '
            'checkpoint.  This never trains, selects a checkpoint, or reads '
            'target data.'))
    parser.add_argument(
        '--source-temporal-attribution-epoch', type=int, default=4,
        help='Exact rejected checkpoint epoch authorized for attribution.')
    parser.add_argument(
        '--source-temporal-immediate-override-audit', action='store_true',
        help=(
            'Run one fixed-checkpoint source-only inference audit with a '
            'one-frame confirmation for candidates that pass margin and '
            'continuity.  No training, checkpoint selection, or target read.'))
    parser.add_argument(
        '--source-val-results-out',
        help='Optional one-class result pickle for read-only source probes.')
    parser.add_argument(
        '--skip-target-eval', action='store_true',
        help='Train/select on source only; run target evaluation separately.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def validate_args(args):
    if args.seed != 0:
        raise ValueError('The protocol requires --seed 0')
    if args.source_val_modulus < 2:
        raise ValueError('--source-val-modulus must be at least 2')
    if bool(args.source_train_datasets) != bool(args.source_val_datasets):
        raise ValueError(
            'Formal source train and validation specs must be supplied '
            'together')
    if args.source_train_datasets:
        parse_dataset_specs(args.source_train_datasets)
        parse_dataset_specs(args.source_val_datasets)
    if not args.dino_gpus:
        raise ValueError('At least one DINO GPU is required')
    if args.head_gpu in args.dino_gpus:
        raise ValueError(
            'Head GPU must be separate from sharded DINO GPUs on 8GB cards')
    s7_enabled = bool(getattr(args, 's7_residual', False))
    s7_positive = (
        getattr(args, 's7_channels', 128),
        getattr(args, 's7_rpn_feat_channels', 128),
        getattr(args, 's7_proposal_count', 500),
        getattr(args, 's7_nms_pre', 2000))
    if any(int(value) <= 0 for value in s7_positive):
        raise ValueError('S7 channel and proposal settings must be positive')
    if (int(getattr(args, 's7_source_min_full_top1', 677)) < 0
            or int(getattr(args, 's7_source_min_small_top1', 303)) < 0
            or int(getattr(args, 's7_source_max_mcml', 3)) < 0):
        raise ValueError('S7 source gate settings must be non-negative')
    s7_anchor_sizes = [float(value) for value in getattr(
        args, 's7_anchor_sizes', [16, 32, 64, 128, 256])]
    if not s7_anchor_sizes or any(value <= 0.0 for value in s7_anchor_sizes):
        raise ValueError('--s7-anchor-sizes must be positive')
    args.s7_anchor_sizes = sorted(set(s7_anchor_sizes))
    s7_training_mode = args.train_components in (
        's7_rpn', 's7_merge', 's7_lane_arbitration',
        's7_quality_suppression', 's7_temporal_association')
    if s7_enabled != s7_training_mode:
        raise ValueError(
            '--s7-residual and an S7 train-components mode must be enabled '
            'together during S7 training')
    merge_positive = (
        getattr(args, 's7_merge_margin', 0.5),
        getattr(args, 's7_merge_retention_weight', 2.0),
        getattr(args, 's7_merge_gain_weight', 1.0),
        getattr(args, 's7_merge_prior_weight', 0.01))
    if any(float(value) <= 0.0 for value in merge_positive):
        raise ValueError('S7 merge margin and loss weights must be positive')
    if (int(args.s7_lane_hidden) <= 0
            or float(args.s7_lane_max_adjustment) <= 0.0):
        raise ValueError('S7 lane arbitration settings must be positive')
    if not 1 <= int(args.s7_lane_hard_negatives) <= 32:
        raise ValueError('--s7-lane-hard-negatives must be within [1, 32]')
    if not 1 <= int(args.s7_lane_gain_repeat) <= 32:
        raise ValueError('--s7-lane-gain-repeat must be within [1, 32]')
    if int(args.s7_lane_base_epoch) != 1:
        raise ValueError(
            's7_lane_arbitration is locked to the audited epoch-1 base')
    quality_positive = (
        getattr(args, 's7_quality_hidden', 32),
        getattr(args, 's7_quality_max_suppression', 2.0),
        getattr(args, 's7_quality_margin', 0.5),
        getattr(args, 's7_quality_risk_weight', 1.0),
        getattr(args, 's7_quality_preserve_weight', 1.0),
        getattr(args, 's7_quality_retention_weight', 2.0),
        getattr(args, 's7_quality_prior_weight', 0.01))
    if any(float(value) <= 0.0 for value in quality_positive):
        raise ValueError('S7 quality suppression settings must be positive')
    if not math.isfinite(float(getattr(
            args, 's7_quality_init_risk_bias', 0.0))):
        raise ValueError('--s7-quality-init-risk-bias must be finite')
    if int(getattr(args, 's7_quality_base_epoch', 1)) != 1:
        raise ValueError(
            's7_quality_suppression is locked to the audited affine epoch-1 '
            'base')
    temporal_positive = (
        getattr(args, 's7_temporal_margin', 0.5),
        getattr(args, 's7_temporal_retention_weight', 2.0),
        getattr(args, 's7_temporal_gain_weight', 1.0),
        getattr(args, 's7_temporal_prior_weight', 0.01),
        getattr(args, 's7_temporal_max_candidates', 100),
        getattr(args, 's7_temporal_min_confirmations', 2),
        getattr(args, 's7_temporal_override_margin', 0.25),
        getattr(args, 's7_temporal_max_center_distance', 3.0))
    if any(float(value) <= 0.0 for value in temporal_positive):
        raise ValueError('S7 temporal association settings must be positive')
    if int(getattr(args, 's7_temporal_base_epoch', 1)) != 1:
        raise ValueError(
            's7_temporal_association is locked to the audited affine epoch-1 '
            'base')
    if not 0.0 <= float(getattr(args, 's7_temporal_min_riou', 0.05)) <= 1.0:
        raise ValueError('--s7-temporal-min-riou must be in [0, 1]')
    if not -1.0 <= float(getattr(
            args, 's7_temporal_min_appearance', 0.20)) <= 1.0:
        raise ValueError('--s7-temporal-min-appearance must be in [-1, 1]')
    if int(getattr(args, 's7_temporal_quality_hidden', 128)) <= 0:
        raise ValueError('--s7-temporal-quality-hidden must be positive')
    if float(getattr(args, 's7_temporal_quality_loss_weight', 1.0)) <= 0.0:
        raise ValueError('--s7-temporal-quality-loss-weight must be positive')
    relative_quality = bool(getattr(
        args, 's7_temporal_relative_quality', False))
    relative_quality_positive = (
        getattr(args, 's7_temporal_relative_quality_weight', 0.5),
        getattr(args, 's7_temporal_relative_quality_margin', 0.25),
        getattr(args, 's7_temporal_relative_quality_min_gap', 0.10),
        getattr(args, 's7_temporal_relative_quality_max_pairs', 128))
    if relative_quality and any(
            float(value) <= 0.0 for value in relative_quality_positive):
        raise ValueError(
            'Relative candidate quality weights, margin, gap and pair '
            'count must be positive')
    if relative_quality and not bool(getattr(
            args, 's7_temporal_quality_head', False)):
        raise ValueError(
            '--s7-temporal-relative-quality requires '
            '--s7-temporal-quality-head')
    if relative_quality and int(
            getattr(args, 's7_temporal_min_confirmations', 2)) != 1:
        raise ValueError(
            'Relative candidate quality training is locked to the audited '
            'one-frame immediate-override policy')
    if relative_quality and int(getattr(
            args, 's7_temporal_relative_base_epoch', 4)) <= 0:
        raise ValueError(
            '--s7-temporal-relative-base-epoch must be positive')
    if (bool(getattr(args, 's7_lane_arbitration', False))
            and (args.train_components == 's7_quality_suppression'
                 or bool(getattr(args, 's7_quality_suppression', False)))):
        raise ValueError(
            'S7 positive lane arbitration and non-positive quality '
            'suppression are mutually exclusive')
    temporal_enabled = bool(getattr(args, 's7_temporal_association', False))
    if temporal_enabled != (args.train_components == 's7_temporal_association'):
        raise ValueError(
            '--s7-temporal-association and its train-components mode must be '
            'enabled together during temporal fitting')
    quality_head_enabled = bool(getattr(
        args, 's7_temporal_quality_head', False))
    if quality_head_enabled and not temporal_enabled:
        raise ValueError(
            '--s7-temporal-quality-head requires '
            '--s7-temporal-association and its train-components mode')
    if temporal_enabled and (bool(getattr(args, 's7_lane_arbitration', False))
                             or bool(getattr(
                                 args, 's7_quality_suppression', False))):
        raise ValueError(
            'Temporal association cannot be combined with learned lane or '
            'quality adjustment')
    if not math.isfinite(float(getattr(args, 's7_merge_init_bias', -2.0))):
        raise ValueError('--s7-merge-init-bias must be finite')
    positive = (
        args.patch_size, args.rpn_feat_channels, args.roi_fc_channels,
        args.roi_samples, args.proposal_count, args.max_detections,
        args.epochs, args.lr, args.max_grad_norm, args.lr_gamma,
        args.checkpoint_interval)
    if any(float(value) <= 0.0 for value in positive):
        raise ValueError(
            'Head, optimizer, and count settings must be positive')
    if args.warmup_iters < 0:
        raise ValueError('--warmup-iters must be non-negative')
    if args.source_small_repeat < 1:
        raise ValueError('--source-small-repeat must be at least 1')
    if args.source_retain_max_top1_drop < 0:
        raise ValueError('--source-retain-max-top1-drop must be non-negative')
    pairwise_positive = (
        args.pairwise_margin, args.pairwise_cls_loss_weight,
        args.pairwise_loss_weight,
        args.retention_loss_weight, args.retention_temperature)
    if any(float(value) <= 0.0 for value in pairwise_positive):
        raise ValueError('Pairwise and retention settings must be positive')
    if not 0.0 <= args.pairwise_negative_riou_thr < args.riou_thr:
        raise ValueError(
            '--pairwise-negative-riou-thr must be in [0, --riou-thr)')
    if not 0.0 <= args.pairwise_nms_iou_thr <= 1.0:
        raise ValueError('--pairwise-nms-iou-thr must be in [0, 1]')
    if args.pairwise_negatives_per_positive < 1:
        raise ValueError(
            '--pairwise-negatives-per-positive must be at least 1')
    if not 0.0 < args.warmup_ratio <= 1.0:
        raise ValueError('--warmup-ratio must be in (0, 1]')
    if args.lr_steps is None:
        args.lr_steps = (
            [2, 3] if args.train_components in (
                'roi_cls_pairwise_v2', 's7_rpn', 's7_merge',
                's7_lane_arbitration', 's7_quality_suppression',
                's7_temporal_association') else [5, 7])
    lr_steps = sorted(set(int(value) for value in args.lr_steps))
    if any(value <= 0 or value >= args.epochs for value in lr_steps):
        raise ValueError('--lr-steps must be within the training schedule')
    args.lr_steps = lr_steps
    selection = (list(range(1, args.epochs + 1))
                 if args.selection_epochs is None
                 else sorted(set(int(value)
                                 for value in args.selection_epochs)))
    if not selection or any(
            value <= 0 or value > args.epochs for value in selection):
        raise ValueError(
            '--selection-epochs must be non-empty and within training')
    if any(value % args.checkpoint_interval != 0
           and value != args.epochs for value in selection):
        raise ValueError(
            '--selection-epochs must coincide with validation checkpoints')
    args.selection_epochs = selection
    if not 0.0 < args.riou_thr <= 1.0:
        raise ValueError('--riou-thr must be in (0, 1]')
    roi_nms_iou_thr = float(getattr(args, 'roi_nms_iou_thr', 0.1))
    if not 0.0 < roi_nms_iou_thr <= 1.0:
        raise ValueError('--roi-nms-iou-thr must be in (0, 1]')
    strides = getattr(args, 'feature_strides', None)
    if strides is not None:
        strides = [int(value) for value in strides]
        if (not strides or any(value <= 0 for value in strides)
                or len(set(strides)) != len(strides)):
            raise ValueError('--feature-strides must be positive and unique')
        if args.patch_size not in strides:
            raise ValueError('--feature-strides must include --patch-size')
        args.feature_strides = sorted(strides)
    if s7_enabled and feature_strides(args) != [int(args.patch_size)]:
        raise ValueError(
            'S7 residual training protects the native single-scale path and '
            'cannot be combined with interpolation-only feature strides')
    if args.valid_content_tolerance < 0.0:
        raise ValueError('--valid-content-tolerance must be non-negative')
    if args.deployment_score_thr < 0.0:
        raise ValueError('--deployment-score-thr must be non-negative')
    if not 0.0 <= args.border_margin_ratio <= 0.5:
        raise ValueError('--border-margin-ratio must be in [0, 0.5]')
    if not 0.0 <= args.source_min_top1_rate <= 1.0:
        raise ValueError('--source-min-top1-rate must be in [0, 1]')
    if args.resume_checkpoint and args.eval_only_checkpoint:
        raise ValueError(
            'Resume and eval-only checkpoints are mutually exclusive')
    if args.init_checkpoint and (
            args.resume_checkpoint or args.eval_only_checkpoint):
        raise ValueError(
            '--init-checkpoint cannot be combined with resume/eval-only')
    conflict_json = getattr(args, 'source_conflict_result_json', None)
    if conflict_json:
        if not os.path.isfile(conflict_json):
            raise ValueError(
                'Source conflict result does not exist: {}'.format(
                    conflict_json))
        if not args.eval_only_checkpoint or not args.skip_target_eval:
            raise ValueError(
                'Source conflict audit requires --eval-only-checkpoint and '
                '--skip-target-eval')
        if args.train_components != 's7_merge':
            raise ValueError(
                'Source conflict audit requires --train-components s7_merge')
        if int(getattr(args, 'source_conflict_epoch', 1)) <= 0:
            raise ValueError('--source-conflict-epoch must be positive')
    attribution_audit = bool(getattr(
        args, 'source_temporal_attribution_audit', False))
    immediate_override_audit = bool(getattr(
        args, 'source_temporal_immediate_override_audit', False))
    if attribution_audit and immediate_override_audit:
        raise ValueError(
            'Temporal attribution and immediate-override audits are '
            'mutually exclusive')
    readonly_temporal_audit = attribution_audit or immediate_override_audit
    if readonly_temporal_audit:
        if conflict_json:
            raise ValueError(
                'Temporal attribution and source conflict audits are '
                'mutually exclusive')
        if not args.eval_only_checkpoint or not args.skip_target_eval:
            raise ValueError(
                'Temporal attribution audit requires '
                '--eval-only-checkpoint and --skip-target-eval')
        if not os.path.isfile(args.eval_only_checkpoint):
            raise ValueError(
                'Temporal attribution checkpoint does not exist: {}'.format(
                    args.eval_only_checkpoint))
        if args.train_components != 's7_temporal_association':
            raise ValueError(
                'Temporal attribution audit requires --train-components '
                's7_temporal_association')
        if not bool(getattr(args, 's7_temporal_association', False)):
            raise ValueError(
                'Temporal attribution audit requires '
                '--s7-temporal-association')
        if not bool(getattr(args, 's7_temporal_quality_head', False)):
            raise ValueError(
                'Temporal attribution audit requires the trained '
                '--s7-temporal-quality-head')
        if int(getattr(args, 'source_temporal_attribution_epoch', 4)) <= 0:
            raise ValueError(
                '--source-temporal-attribution-epoch must be positive')
    if args.init_checkpoint and not os.path.isfile(args.init_checkpoint):
        raise ValueError('Init checkpoint does not exist: {}'.format(
            args.init_checkpoint))
    roi_cls_modes = ('roi_cls', 'roi_cls_pairwise',
                     'roi_cls_pairwise_v2')
    if (args.train_components in roi_cls_modes
            and not (args.init_checkpoint or args.resume_checkpoint
                     or args.eval_only_checkpoint)):
        raise ValueError(
            '--train-components {} requires an init/resume/eval-only '
            'checkpoint so the validated RPN and OBB regressor are retained'
            .format(args.train_components))
    if (args.train_components in (
            's7_rpn', 's7_merge', 's7_lane_arbitration',
            's7_quality_suppression', 's7_temporal_association')
            and not (args.init_checkpoint or args.resume_checkpoint
                     or args.eval_only_checkpoint)):
        raise ValueError(
            'S7 mode requires an init/resume/eval-only checkpoint; training '
            'must initialize from the retained native S14 heads')
    if args.train_components == 's7_merge':
        component = getattr(args, 's7_component_checkpoint', None)
        if args.init_checkpoint and not component:
            raise ValueError(
                's7_merge initialization requires --s7-component-checkpoint')
        if component and not os.path.isfile(component):
            raise ValueError(
                'S7 component checkpoint does not exist: {}'.format(component))
        if args.source_retain_max_top1_drop != 0:
            raise ValueError('s7_merge requires exact source retention')
    if args.train_components == 's7_lane_arbitration':
        if args.source_retain_max_top1_drop != 0:
            raise ValueError(
                's7_lane_arbitration requires exact source retention')
        if args.s7_component_checkpoint:
            raise ValueError(
                's7_lane_arbitration loads one complete epoch-1 checkpoint; '
                'do not pass --s7-component-checkpoint')
    if args.train_components == 's7_quality_suppression':
        if args.source_retain_max_top1_drop != 0:
            raise ValueError(
                's7_quality_suppression requires exact source retention')
        if args.s7_component_checkpoint:
            raise ValueError(
                's7_quality_suppression loads one complete affine epoch-1 '
                'checkpoint; do not pass --s7-component-checkpoint')
        if not getattr(args, 'skip_target_eval', False):
            raise ValueError(
                's7_quality_suppression is source-only; pass '
                '--skip-target-eval and authorize target diagnosis only '
                'after the formal gate passes')
        if not math.isclose(
                float(args.s7_quality_init_risk_bias), 0.0,
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                's7_quality_suppression requires zero risk-bias '
                'initialization to reproduce affine epoch 1 exactly')
        if int(getattr(args, 's7_source_min_full_top1', 677)) < int(
                S7_QUALITY_MIN_FULL_TOP1):
            raise ValueError(
                's7_quality_suppression requires '
                '--s7-source-min-full-top1 >= {}'.format(
                    S7_QUALITY_MIN_FULL_TOP1))
        if int(getattr(args, 's7_source_min_small_top1', 303)) < int(
                S7_QUALITY_MIN_SMALL_TOP1):
            raise ValueError(
                's7_quality_suppression requires '
                '--s7-source-min-small-top1 >= {}'.format(
                    S7_QUALITY_MIN_SMALL_TOP1))
        if int(getattr(args, 's7_source_max_mcml', 3)) > 3:
            raise ValueError(
                's7_quality_suppression requires '
                '--s7-source-max-mcml <= 3')
    if args.train_components == 's7_temporal_association':
        if args.source_retain_max_top1_drop != 0:
            raise ValueError(
                's7_temporal_association requires exact source retention')
        if args.s7_component_checkpoint:
            raise ValueError(
                's7_temporal_association loads one complete affine epoch-1 '
                'checkpoint; do not pass --s7-component-checkpoint')
        if not getattr(args, 'skip_target_eval', False):
            raise ValueError(
                's7_temporal_association is source-only; pass '
                '--skip-target-eval until the formal source gate passes')
        if not args.source_train_datasets or not args.source_val_datasets:
            raise ValueError(
                's7_temporal_association requires formal source train/val '
                'datasets; modulus-split frames are not continuous video')
        if relative_quality:
            if readonly_temporal_audit:
                if args.init_checkpoint or args.resume_checkpoint:
                    raise ValueError(
                        'Relative candidate quality attribution is '
                        'eval-only and cannot initialize or resume training')
            elif not args.init_checkpoint or args.resume_checkpoint \
                    or args.eval_only_checkpoint:
                raise ValueError(
                    'Relative candidate quality training requires an '
                    'init checkpoint and cannot resume or run eval-only')
            if int(getattr(args, 'source_small_repeat', 1)) != 1:
                raise ValueError(
                    'Relative candidate quality training forbids frame '
                    'repetition')
        if int(args.source_small_repeat) != 1:
            raise ValueError(
                's7_temporal_association forbids frame repetition because it '
                'would break causal sequence order')
        if int(getattr(args, 's7_source_min_full_top1', 677)) < 688:
            raise ValueError(
                's7_temporal_association requires '
                '--s7-source-min-full-top1 >= 688')
        if int(getattr(args, 's7_source_min_small_top1', 303)) < 311:
            raise ValueError(
                's7_temporal_association requires '
                '--s7-source-min-small-top1 >= 311')
        if int(getattr(args, 's7_source_max_mcml', 3)) > 3:
            raise ValueError(
                's7_temporal_association requires '
                '--s7-source-max-mcml <= 3')
    if args.train_components in (
            's7_rpn', 's7_merge', 's7_lane_arbitration',
            's7_quality_suppression', 's7_temporal_association') and args.epochs > 4:
        raise ValueError(
            'S7 source-only stages are limited to 4 epochs; extend only after '
            'source validation shows the selected component is improving')
    if args.train_components == 'roi_cls_pairwise_v2':
        if args.epochs > PAIRWISE_V2_MAX_EPOCHS:
            raise ValueError(
                'Pairwise V2 fine-tuning is limited to at most {} epochs; '
                'the eight-epoch schedule is reserved for formal full-head '
                'training'.format(PAIRWISE_V2_MAX_EPOCHS))
        if args.source_retain_max_top1_drop != 0:
            raise ValueError(
                'Pairwise V2 requires exact source retention: '
                '--source-retain-max-top1-drop must be 0')
        if not math.isclose(
                float(args.pairwise_nms_iou_thr),
                float(args.roi_nms_iou_thr), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                'Pairwise V2 mining and source validation must use the same '
                'NMS IoU threshold')
    if (args.source_val_results_out and
            os.path.exists(args.source_val_results_out)):
        raise ValueError('Refusing to overwrite source-val results: {}'.format(
            args.source_val_results_out))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_trainable_components(heads, train_components: str) -> List[str]:
    """Set the exact head parameters authorized for optimization."""
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    if train_components == 'all':
        for parameter in heads.parameters():
            parameter.requires_grad_(True)
    elif train_components in (
            'roi_cls', 'roi_cls_pairwise', 'roi_cls_pairwise_v2'):
        bbox_head = heads.roi_head.bbox_head
        if not hasattr(bbox_head, 'fc_cls'):
            raise RuntimeError('ROI bbox head has no fc_cls classifier')
        for parameter in bbox_head.fc_cls.parameters():
            parameter.requires_grad_(True)
    elif train_components == 's7_rpn':
        if not getattr(heads, 's7_enabled', False):
            raise RuntimeError('S7 train mode requires an S7-enabled head')
        for module in (heads.s7_readout, heads.s7_rpn_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    elif train_components == 's7_merge':
        if (not getattr(heads, 's7_protected_merge', False)
                or heads.s7_score_calibrator is None):
            raise RuntimeError(
                'S7 merge mode requires the protected score calibrator')
        for parameter in heads.s7_score_calibrator.parameters():
            parameter.requires_grad_(True)
    elif train_components == 's7_lane_arbitration':
        if (not getattr(heads, 's7_protected_merge', False)
                or getattr(heads, 's7_lane_arbitrator', None) is None):
            raise RuntimeError(
                'S7 lane arbitration requires the protected merge arbitrator')
        for parameter in heads.s7_lane_arbitrator.parameters():
            parameter.requires_grad_(True)
    elif train_components == 's7_quality_suppression':
        if (not getattr(heads, 's7_protected_merge', False)
                or getattr(heads, 's7_quality_suppressor', None) is None):
            raise RuntimeError(
                'S7 quality suppression requires the protected affine merge '
                'and lane-wide suppressor')
        for parameter in heads.s7_quality_suppressor.parameters():
            parameter.requires_grad_(True)
    elif train_components == 's7_temporal_association':
        if (not getattr(heads, 's7_protected_merge', False)
                or getattr(heads, 's7_temporal_scorer', None) is None):
            raise RuntimeError(
                'S7 temporal association requires the protected affine '
                'candidate pool and temporal scorer')
        if bool(getattr(heads, 's7_temporal_quality_head_enabled', False)):
            quality_head = getattr(heads, 's7_candidate_quality_head', None)
            if quality_head is None:
                raise RuntimeError('Temporal quality head is not configured')
            for parameter in quality_head.parameters():
                parameter.requires_grad_(True)
        else:
            for parameter in heads.s7_temporal_scorer.parameters():
                parameter.requires_grad_(True)
    else:
        raise ValueError('Unsupported train-components: {}'.format(
            train_components))
    return [
        name for name, parameter in heads.named_parameters()
        if parameter.requires_grad]


def optimization_loss_total(losses: Dict, train_components: str):
    if train_components == 'all':
        return loss_total(losses)
    if train_components == 'roi_cls':
        if 'loss_cls' not in losses:
            raise RuntimeError('ROI classification loss is missing')
        return loss_total({'loss_cls': losses['loss_cls']})
    if train_components == 's7_rpn':
        required = ('loss_s7_rpn_cls', 'loss_s7_rpn_bbox')
        missing = [name for name in required if name not in losses]
        if missing:
            raise RuntimeError('S7 RPN losses missing: {}'.format(
                ', '.join(missing)))
        return loss_total({name: losses[name] for name in required})
    if train_components == 's7_merge':
        required = (
            'loss_s7_merge_retention', 'loss_s7_merge_gain',
            'loss_s7_merge_prior')
        missing = [name for name in required if name not in losses]
        if missing:
            raise RuntimeError('S7 merge losses missing: {}'.format(
                ', '.join(missing)))
        return loss_total({name: losses[name] for name in required})
    if train_components == 's7_lane_arbitration':
        required = (
            'loss_s7_lane_retention', 'loss_s7_lane_gain',
            'loss_s7_lane_prior')
        missing = [name for name in required if name not in losses]
        if missing:
            raise RuntimeError('S7 lane losses missing: {}'.format(
                ', '.join(missing)))
        return loss_total({name: losses[name] for name in required})
    if train_components == 's7_quality_suppression':
        required = (
            'loss_s7_quality_risk', 'loss_s7_quality_preserve',
            'loss_s7_quality_retention', 'loss_s7_quality_prior')
        missing = [name for name in required if name not in losses]
        if missing:
            raise RuntimeError('S7 quality losses missing: {}'.format(
                ', '.join(missing)))
        return loss_total({name: losses[name] for name in required})
    if train_components == 's7_temporal_association':
        if 'loss_s7_candidate_quality' in losses:
            quality_losses = {
                'loss_s7_candidate_quality':
                    losses['loss_s7_candidate_quality']}
            if 'loss_s7_candidate_quality_relative' in losses:
                quality_losses['loss_s7_candidate_quality_relative'] = (
                    losses['loss_s7_candidate_quality_relative'])
            return loss_total(quality_losses)
        required = (
            'loss_s7_temporal_retention', 'loss_s7_temporal_gain',
            'loss_s7_temporal_prior')
        missing = [name for name in required if name not in losses]
        if missing:
            raise RuntimeError('S7 temporal losses missing: {}'.format(
                ', '.join(missing)))
        return loss_total({name: losses[name] for name in required})
    if train_components in ('roi_cls_pairwise', 'roi_cls_pairwise_v2'):
        required = ('loss_cls', 'loss_roi_pairwise', 'loss_roi_retention')
        missing = [name for name in required if name not in losses]
        if missing:
            raise RuntimeError('Pairwise ROI losses missing: {}'.format(
                ', '.join(missing)))
        return loss_total({name: losses[name] for name in required})
    raise ValueError('Unsupported train-components: {}'.format(
        train_components))


def optimization_loss_component_names(
        train_components: str, quality_head: bool = False,
        relative_quality: bool = False) -> List[str]:
    """Return the canonical loss-name list for logs and checkpoints."""
    if train_components == 'all':
        return ['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox']
    if train_components == 's7_rpn':
        return ['loss_s7_rpn_cls', 'loss_s7_rpn_bbox']
    if train_components == 's7_merge':
        return [
            'loss_s7_merge_retention', 'loss_s7_merge_gain',
            'loss_s7_merge_prior']
    if train_components == 's7_lane_arbitration':
        return [
            'loss_s7_lane_retention', 'loss_s7_lane_gain',
            'loss_s7_lane_prior']
    if train_components == 's7_quality_suppression':
        return [
            'loss_s7_quality_risk', 'loss_s7_quality_preserve',
            'loss_s7_quality_retention', 'loss_s7_quality_prior']
    if train_components == 's7_temporal_association':
        if quality_head:
            names = ['loss_s7_candidate_quality']
            if relative_quality:
                names.append('loss_s7_candidate_quality_relative')
            return names
        return [
            'loss_s7_temporal_retention', 'loss_s7_temporal_gain',
            'loss_s7_temporal_prior']
    if train_components in ('roi_cls_pairwise', 'roi_cls_pairwise_v2'):
        return ['loss_cls', 'loss_roi_pairwise', 'loss_roi_retention']
    if train_components == 'roi_cls':
        return ['loss_cls']
    raise ValueError('Unsupported train-components: {}'.format(
        train_components))


def roi_foreground_log_odds(cls_score: torch.Tensor) -> torch.Tensor:
    """Return the foreground-vs-background logit used for ROI ordering."""
    if cls_score.ndim != 2 or cls_score.shape[1] != 2:
        raise ValueError('One-class ROI logits must have shape [N, 2]')
    return cls_score[:, 0] - cls_score[:, 1]


def roi_pairwise_margin_loss(positive_log_odds: torch.Tensor,
                             negative_log_odds: torch.Tensor,
                             margin: float) -> torch.Tensor:
    """Require every selected positive to outrank selected false ROIs."""
    import torch.nn.functional as functional

    if positive_log_odds.numel() == 0 or negative_log_odds.numel() == 0:
        return (positive_log_odds.sum() + negative_log_odds.sum()) * 0.0
    violations = (float(margin) - positive_log_odds[:, None]
                  + negative_log_odds[None, :])
    return functional.relu(violations).mean()


def roi_paired_margin_loss(positive_log_odds: torch.Tensor,
                           negative_log_odds: torch.Tensor,
                           margin: float) -> torch.Tensor:
    """Rank aligned positive/negative pairs without a Cartesian product."""
    import torch.nn.functional as functional

    if positive_log_odds.ndim != 1 or negative_log_odds.ndim != 1:
        raise ValueError('Paired ROI log odds must be one-dimensional')
    if positive_log_odds.shape != negative_log_odds.shape:
        raise ValueError('Paired ROI log odds must have the same shape')
    if positive_log_odds.numel() == 0:
        return (positive_log_odds.sum() + negative_log_odds.sum()) * 0.0
    return functional.relu(
        float(margin) - positive_log_odds + negative_log_odds).mean()


def roi_classifier_retention_loss(student_logits: torch.Tensor,
                                  teacher_logits: torch.Tensor,
                                  temperature: float) -> torch.Tensor:
    """Keep the updated classifier close to the fixed source classifier."""
    import torch.nn.functional as functional

    if student_logits.shape != teacher_logits.shape:
        raise ValueError('Student and teacher ROI logits must have one shape')
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError('Retention temperature must be positive')
    return functional.kl_div(
        functional.log_softmax(student_logits / temperature, dim=1),
        functional.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean') * (temperature ** 2)


def s7_merge_pair_losses(native_log_odds: torch.Tensor,
                         native_gt_overlap: torch.Tensor,
                         s7_log_odds: torch.Tensor,
                         s7_gt_overlap: torch.Tensor,
                         calibrator: nn.Module, margin: float,
                         riou_thr: float = 0.5) -> Dict:
    """Mine one source-only retain or gain pair for monotonic calibration."""
    if not (native_log_odds.ndim == native_gt_overlap.ndim
            == s7_log_odds.ndim == s7_gt_overlap.ndim == 1):
        raise ValueError('S7 merge logits and overlaps must be one-dimensional')
    if native_log_odds.shape != native_gt_overlap.shape:
        raise ValueError('Native S14 logits and overlaps must share one shape')
    if s7_log_odds.shape != s7_gt_overlap.shape:
        raise ValueError('Supplement S7 logits and overlaps must share one shape')
    calibrated = calibrator(s7_log_odds)
    zero = (native_log_odds.sum() + calibrated.sum()) * 0.0
    retention = zero
    gain = zero
    retain_pair_count = 0
    gain_pair_count = 0
    native_top1_correct = False
    if native_log_odds.numel():
        native_top = torch.argmax(native_log_odds)
        native_top1_correct = bool(
            native_gt_overlap[native_top] >= float(riou_thr))
        if native_top1_correct:
            wrong = torch.nonzero(
                s7_gt_overlap < float(riou_thr),
                as_tuple=False).reshape(-1)
            if wrong.numel():
                competitor = wrong[torch.argmax(s7_log_odds[wrong])]
                retention = torch.relu(
                    float(margin) + calibrated[competitor]
                    - native_log_odds[native_top])
                retain_pair_count = 1
        else:
            usable = torch.nonzero(
                s7_gt_overlap >= float(riou_thr),
                as_tuple=False).reshape(-1)
            if usable.numel():
                supplement = usable[torch.argmax(s7_log_odds[usable])]
                gain = torch.relu(
                    float(margin) + native_log_odds[native_top]
                    - calibrated[supplement])
                gain_pair_count = 1
    return dict(
        retention=retention, gain=gain,
        retain_pair_count=retain_pair_count,
        gain_pair_count=gain_pair_count,
        retention_active=int(float(retention.detach().item()) > 0.0),
        gain_active=int(float(gain.detach().item()) > 0.0),
        native_top1_correct=int(native_top1_correct))


def s7_lane_pair_losses(native_log_odds: torch.Tensor,
                        native_gt_overlap: torch.Tensor,
                        adjusted_s7_log_odds: torch.Tensor,
                        s7_gt_overlap: torch.Tensor,
                        margin: float, hard_negatives: int,
                        riou_thr: float = 0.5) -> Dict:
    """Mine current-logit lane pairs without using source-val conflicts."""
    if not (native_log_odds.ndim == native_gt_overlap.ndim
            == adjusted_s7_log_odds.ndim == s7_gt_overlap.ndim == 1):
        raise ValueError('S7 lane logits and overlaps must be one-dimensional')
    if native_log_odds.shape != native_gt_overlap.shape:
        raise ValueError('Native lane logits and overlaps must share one shape')
    if adjusted_s7_log_odds.shape != s7_gt_overlap.shape:
        raise ValueError('S7 lane logits and overlaps must share one shape')
    hard_negatives = int(hard_negatives)
    if hard_negatives <= 0:
        raise ValueError('S7 lane hard-negative count must be positive')
    zero = (native_log_odds.sum() + adjusted_s7_log_odds.sum()) * 0.0
    retention = zero
    gain = zero
    retain_pair_count = 0
    retention_active = 0
    gain_pair_count = 0
    gain_s7_competitor_count = 0
    native_top1_correct = False
    if native_log_odds.numel():
        native_top = torch.argmax(native_log_odds)
        native_top_logit = native_log_odds[native_top]
        native_top1_correct = bool(
            native_gt_overlap[native_top] >= float(riou_thr))
        wrong = torch.nonzero(
            s7_gt_overlap < float(riou_thr),
            as_tuple=False).reshape(-1)
        if native_top1_correct and wrong.numel():
            count = min(hard_negatives, int(wrong.numel()))
            order = torch.topk(
                adjusted_s7_log_odds[wrong].detach(), count).indices
            competitors = wrong[order]
            violations = torch.relu(
                float(margin) + adjusted_s7_log_odds[competitors]
                - native_top_logit)
            retention = violations.mean()
            retain_pair_count = int(count)
            retention_active = int(
                (violations.detach() > 0.0).sum().item())
        elif not native_top1_correct:
            usable = torch.nonzero(
                s7_gt_overlap >= float(riou_thr),
                as_tuple=False).reshape(-1)
            if usable.numel():
                usable_order = torch.argmax(
                    adjusted_s7_log_odds[usable].detach())
                supplement = usable[usable_order]
                strongest_competitor = native_top_logit
                if wrong.numel():
                    wrong_order = torch.argmax(
                        adjusted_s7_log_odds[wrong].detach())
                    strongest_wrong = adjusted_s7_log_odds[wrong[wrong_order]]
                    strongest_competitor = torch.maximum(
                        strongest_competitor, strongest_wrong)
                    gain_s7_competitor_count = 1
                gain = torch.relu(
                    float(margin) + strongest_competitor
                    - adjusted_s7_log_odds[supplement])
                gain_pair_count = 1
    return dict(
        retention=retention, gain=gain,
        retain_pair_count=retain_pair_count,
        gain_pair_count=gain_pair_count,
        retention_active=retention_active,
        gain_active=int(float(gain.detach().item()) > 0.0),
        gain_s7_competitor_count=gain_s7_competitor_count,
        native_top1_correct=int(native_top1_correct))


def s7_quality_suppression_losses(
        native_log_odds: torch.Tensor, native_gt_overlap: torch.Tensor,
        base_s7_log_odds: torch.Tensor, s7_gt_overlap: torch.Tensor,
        delta: torch.Tensor, risk_logit: torch.Tensor,
        margin: float, riou_thr: float = 0.5) -> Dict:
    """Label one lane-wide source risk without any positive promotion."""
    import torch.nn.functional as functional

    if not (native_log_odds.ndim == native_gt_overlap.ndim
            == base_s7_log_odds.ndim == s7_gt_overlap.ndim == 1):
        raise ValueError('S7 quality logits and overlaps must be one-dimensional')
    if native_log_odds.shape != native_gt_overlap.shape:
        raise ValueError('Native quality logits and overlaps must share shape')
    if base_s7_log_odds.shape != s7_gt_overlap.shape:
        raise ValueError('S7 quality logits and overlaps must share shape')
    if delta.numel() != 1 or risk_logit.numel() != 1:
        raise ValueError('S7 quality delta and risk logit must be scalar')
    zero = (delta.reshape(()) + risk_logit.reshape(())) * 0.0
    native_top_correct = False
    s7_top_correct = False
    risk_pair = False
    preserve_pair = False
    retention = zero
    risk = zero
    preserve = zero
    base_gap = 0.0
    native_top_riou = 0.0
    s7_top_riou = 0.0
    if native_log_odds.numel():
        native_top = torch.argmax(native_log_odds.detach())
        native_top_logit = native_log_odds[native_top]
        native_top_riou = float(
            native_gt_overlap[native_top].detach().item())
        native_top_correct = bool(
            native_gt_overlap[native_top] >= float(riou_thr))
    else:
        native_top_logit = zero
    if base_s7_log_odds.numel():
        s7_top = torch.argmax(base_s7_log_odds.detach())
        s7_top_riou = float(s7_gt_overlap[s7_top].detach().item())
        s7_top_correct = bool(s7_gt_overlap[s7_top] >= float(riou_thr))
        preserve_pair = bool(s7_top_correct)
        if native_log_odds.numel():
            base_top = base_s7_log_odds[s7_top]
            base_gap = float((base_top - native_top_logit).detach().item())
            risk_pair = bool(
                native_top_correct and not s7_top_correct
                and base_gap + float(margin) > 0.0)
            if risk_pair:
                retention = torch.relu(
                    float(margin) + base_top + delta.reshape(())
                    - native_top_logit)
    if risk_pair:
        risk = functional.binary_cross_entropy_with_logits(
            risk_logit.reshape(()), torch.ones_like(risk_logit.reshape(())))
    if preserve_pair:
        preserve = functional.binary_cross_entropy_with_logits(
            risk_logit.reshape(()), torch.zeros_like(risk_logit.reshape(())))
    return dict(
        risk=risk, preserve=preserve, retention=retention,
        risk_pair_count=int(risk_pair),
        preserve_pair_count=int(preserve_pair),
        retention_active=int(float(retention.detach().item()) > 0.0),
        native_top1_correct=int(native_top_correct),
        s7_top1_correct=int(s7_top_correct),
        native_top1_riou=float(native_top_riou),
        s7_top1_riou=float(s7_top_riou), base_gap=float(base_gap))


def _finite_distribution(values: Sequence[float]) -> Dict:
    """Return compact, JSON-safe evidence for a finite scalar sample."""
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64)
    if not finite.size:
        return dict(
            count=0, minimum=None, p25=None, median=None, p75=None,
            maximum=None)
    return dict(
        count=int(finite.size), minimum=float(np.min(finite)),
        p25=float(np.percentile(finite, 25.0)),
        median=float(np.median(finite)),
        p75=float(np.percentile(finite, 75.0)),
        maximum=float(np.max(finite)))


def summarize_s7_quality_support_rows(
        rows: Sequence[Dict], margin: float, riou_thr: float,
        minimum_risk_pairs: int = S7_QUALITY_MIN_RISK_PAIRS) -> Dict:
    """Audit whether source train can supervise the quality suppressor."""
    if int(minimum_risk_pairs) < 1:
        raise ValueError('S7 quality support requires at least one risk pair')
    ordered = sorted(rows, key=lambda row: str(row['frame_key']))
    risk = [row for row in ordered if bool(row['risk_pair'])]
    preserve = [row for row in ordered if bool(row['preserve_pair'])]
    s7_wrong = [row for row in ordered if not bool(row['s7_top1_correct'])]
    native_correct_s7_wrong = [
        row for row in s7_wrong if bool(row['native_top1_correct'])]
    excluded_by_margin = [
        row for row in native_correct_s7_wrong
        if float(row['base_gap']) + float(margin) <= 0.0]
    expected_risk_keys = {
        str(row['frame_key']) for row in native_correct_s7_wrong
        if int(row.get('s7_candidate_count', 0)) > 0
        and float(row['base_gap']) + float(margin) > 0.0}
    actual_risk_keys = {str(row['frame_key']) for row in risk}
    if expected_risk_keys != actual_risk_keys:
        raise RuntimeError(
            'S7 quality support audit disagrees with training risk labels')
    sequence_counts = {}
    for row in s7_wrong:
        sequence = '{}|{}'.format(row.get('split', ''), row.get('seq', ''))
        sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
    training_allowed = len(risk) >= int(minimum_risk_pairs)
    return dict(
        status=('PASS' if training_allowed else 'FAIL_ZERO_RISK_SUPPORT'),
        source_train_only=True, target_read=False,
        frame_count=int(len(ordered)),
        minimum_risk_pairs=int(minimum_risk_pairs),
        training_allowed=bool(training_allowed),
        training_skipped=bool(not training_allowed),
        failure_reason=(
            None if training_allowed else
            'No source-train frame satisfies the fixed risk label'),
        riou_threshold=float(riou_thr), margin=float(margin),
        risk_pair_count=int(len(risk)),
        preserve_pair_count=int(len(preserve)),
        native_top1_correct_count=int(sum(
            bool(row['native_top1_correct']) for row in ordered)),
        s7_top1_correct_count=int(sum(
            bool(row['s7_top1_correct']) for row in ordered)),
        s7_top1_wrong_count=int(len(s7_wrong)),
        native_correct_s7_wrong_count=int(len(native_correct_s7_wrong)),
        native_correct_s7_wrong_excluded_by_margin_count=int(
            len(excluded_by_margin)),
        no_s7_candidate_count=int(sum(
            int(row.get('s7_candidate_count', 0)) == 0 for row in ordered)),
        base_gap_distribution_all=_finite_distribution(
            [row['base_gap'] for row in ordered]),
        base_gap_distribution_s7_wrong=_finite_distribution(
            [row['base_gap'] for row in s7_wrong]),
        base_gap_distribution_native_correct_s7_wrong=(
            _finite_distribution(
                [row['base_gap'] for row in native_correct_s7_wrong])),
        s7_wrong_sequence_counts=dict(sorted(sequence_counts.items())),
        s7_wrong_frames=[dict(
            frame_key=str(row['frame_key']),
            native_top1_correct=bool(row['native_top1_correct']),
            s7_top1_correct=bool(row['s7_top1_correct']),
            risk_pair=bool(row['risk_pair']),
            excluded_by_margin=bool(
                row['native_top1_correct']
                and float(row['base_gap']) + float(margin) <= 0.0),
            native_top1_riou=float(row['native_top1_riou']),
            s7_top1_riou=float(row['s7_top1_riou']),
            base_gap=float(row['base_gap']),
            suppression_to_match_native=float(max(
                0.0, float(row['base_gap']))),
            suppression_for_margin=float(max(
                0.0, float(row['base_gap']) + float(margin))),
            s7_candidate_count=int(row.get('s7_candidate_count', 0)))
            for row in s7_wrong])


def select_hard_pairwise_indices(
        gt_overlap: torch.Tensor, foreground_score: torch.Tensor,
        positive_competitor_iou: torch.Tensor, max_samples: int,
        positive_fraction: float, positive_riou_thr: float,
        negative_riou_thr: float, nms_iou_thr: float):
    """Select low-score positives and NMS-capable, high-score negatives."""
    if not (gt_overlap.ndim == foreground_score.ndim
            == positive_competitor_iou.ndim == 1):
        raise ValueError('Pairwise mining inputs must be one-dimensional')
    if not (gt_overlap.shape == foreground_score.shape
            == positive_competitor_iou.shape):
        raise ValueError('Pairwise mining inputs must have one shape')
    positive_indices = torch.nonzero(
        gt_overlap >= float(positive_riou_thr),
        as_tuple=False).reshape(-1)
    negative_eligible = (
        (gt_overlap < float(positive_riou_thr))
        & ((gt_overlap <= float(negative_riou_thr))
           | (positive_competitor_iou >= float(nms_iou_thr))))
    negative_indices = torch.nonzero(
        negative_eligible, as_tuple=False).reshape(-1)
    max_positive = max(1, int(max_samples * positive_fraction))
    if positive_indices.numel() > max_positive:
        order = torch.argsort(foreground_score[positive_indices])
        positive_indices = positive_indices[order[:max_positive]]
    max_negative = max(0, int(max_samples) - positive_indices.numel())
    if negative_indices.numel() > max_negative:
        suppressor_mask = (
            positive_competitor_iou[negative_indices]
            >= float(nms_iou_thr))
        suppressors = negative_indices[suppressor_mask]
        other_negatives = negative_indices[~suppressor_mask]
        suppressors = suppressors[torch.argsort(
            foreground_score[suppressors], descending=True)]
        other_negatives = other_negatives[torch.argsort(
            foreground_score[other_negatives], descending=True)]
        negative_indices = torch.cat(
            [suppressors, other_negatives], dim=0)[:max_negative]
    elif negative_indices.numel():
        suppressor_mask = (
            positive_competitor_iou[negative_indices]
            >= float(nms_iou_thr))
        suppressors = negative_indices[suppressor_mask]
        other_negatives = negative_indices[~suppressor_mask]
        suppressors = suppressors[torch.argsort(
            foreground_score[suppressors], descending=True)]
        other_negatives = other_negatives[torch.argsort(
            foreground_score[other_negatives], descending=True)]
        negative_indices = torch.cat([suppressors, other_negatives], dim=0)
    return positive_indices, negative_indices


def mine_actual_roi_competitor_pairs(
        gt_overlap: torch.Tensor, foreground_score: torch.Tensor,
        positive_indices: torch.Tensor,
        candidate_to_positive_iou: torch.Tensor,
        positive_riou_thr: float, nms_iou_thr: float,
        negatives_per_positive: int):
    """Pair each usable ROI only with false ROIs that currently outrank it.

    A pair is marked as an NMS suppressor only when the false ROI has both a
    higher score and rotated IoU above the deployment NMS threshold.  Other
    higher-scoring false ROIs are ordering competitors.  Already-correct
    positives generate no pair and therefore no unnecessary ranking update.
    """
    if gt_overlap.ndim != 1 or foreground_score.ndim != 1:
        raise ValueError('Competitor mining vectors must be one-dimensional')
    if gt_overlap.shape != foreground_score.shape:
        raise ValueError('Competitor mining vectors must have one shape')
    if positive_indices.ndim != 1:
        raise ValueError('Positive indices must be one-dimensional')
    expected_shape = (gt_overlap.numel(), positive_indices.numel())
    if tuple(candidate_to_positive_iou.shape) != expected_shape:
        raise ValueError(
            'Candidate-to-positive IoU must have shape {}'.format(
                expected_shape))
    if int(negatives_per_positive) < 1:
        raise ValueError('negatives_per_positive must be at least 1')

    pair_positive = []
    pair_negative = []
    pair_is_suppressor = []
    false_mask = gt_overlap < float(positive_riou_thr)
    for column, positive_index in enumerate(positive_indices.tolist()):
        positive_score = foreground_score[positive_index]
        outranker_mask = false_mask & (foreground_score > positive_score)
        competitors = torch.nonzero(
            outranker_mask, as_tuple=False).reshape(-1)
        if competitors.numel() == 0:
            continue
        order = torch.argsort(
            foreground_score[competitors], descending=True)
        competitors = competitors[order[:int(negatives_per_positive)]]
        suppressors = (
            candidate_to_positive_iou[competitors, column]
            > float(nms_iou_thr))
        pair_positive.extend(
            [int(positive_index)] * int(competitors.numel()))
        pair_negative.extend(int(value) for value in competitors.tolist())
        pair_is_suppressor.extend(
            bool(value) for value in suppressors.tolist())

    device = gt_overlap.device
    return (
        torch.as_tensor(pair_positive, dtype=torch.long, device=device),
        torch.as_tensor(pair_negative, dtype=torch.long, device=device),
        torch.as_tensor(
            pair_is_suppressor, dtype=torch.bool, device=device))


def select_representative_usable_rois(
        overlap_by_gt: torch.Tensor, foreground_score: torch.Tensor,
        positive_riou_thr: float, max_positives: int) -> torch.Tensor:
    """Select the highest-scoring usable ROI for each source GT."""
    if overlap_by_gt.ndim != 2 or foreground_score.ndim != 1:
        raise ValueError('ROI/GT overlaps must be [N,G] and scores must be [N]')
    if overlap_by_gt.shape[0] != foreground_score.numel():
        raise ValueError('ROI/GT overlaps and scores must share N')
    if int(max_positives) < 1:
        raise ValueError('max_positives must be at least 1')
    representatives = []
    for gt_index in range(overlap_by_gt.shape[1]):
        usable = torch.nonzero(
            overlap_by_gt[:, gt_index] >= float(positive_riou_thr),
            as_tuple=False).reshape(-1)
        if usable.numel():
            best = usable[torch.argmax(foreground_score[usable])]
            representatives.append(int(best))
    unique = sorted(set(representatives))
    if len(unique) > int(max_positives):
        ordered = sorted(
            unique, key=lambda index: float(foreground_score[index]),
            reverse=False)
        unique = ordered[:int(max_positives)]
    return torch.as_tensor(
        unique, dtype=torch.long, device=foreground_score.device)


def file_identity(path: str) -> Dict:
    stat = os.stat(path)
    return dict(
        path=os.path.abspath(path), size=int(stat.st_size),
        mtime_ns=int(getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9))))


def cache_signature(record: Dict, args) -> Dict:
    return dict(
        image=file_identity(record['image']),
        dinov2_checkpoint=file_identity(args.dinov2_checkpoint),
        dinov2_model=args.dinov2_model,
        dino_height=int(args.dino_height),
        dino_max_long_side=int(args.dino_max_long_side),
        patch_size=int(args.patch_size))


def cache_path(record: Dict, args) -> str:
    signature = json.dumps(
        cache_signature(record, args), sort_keys=True).encode('utf-8')
    digest = hashlib.sha256(signature).hexdigest()[:16]
    name = '{}_{:05d}_{}.pth'.format(
        record['seq'], int(record['frame']), digest)
    return os.path.join(args.feature_cache_dir, record['split'], name)


def atomic_torch_save(payload: Dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + '.tmp'
    torch.save(payload, temporary)
    os.replace(temporary, path)


def source_progress_path(out_json: str) -> str:
    root, extension = os.path.splitext(os.path.abspath(out_json))
    if extension.lower() != '.json':
        return os.path.abspath(out_json) + '.partial.json'
    return root + '.partial.json'


def write_source_training_progress(
        args, completed_epoch: int, best_epoch: int, best_path: str,
        latest_path: str, baseline_summary: Dict,
        baseline_small_summary: Dict, best_summary: Dict,
        best_small_summary: Dict, history: Sequence[Dict],
        status: str = 'SOURCE_ONLY_TRAINING_IN_PROGRESS',
        s7_quality_support_audit: Optional[Dict] = None
        ) -> Tuple[str, int]:
    """Persist source-only selection evidence before target is ever read."""
    output_path = source_progress_path(args.out_json)
    payload = dict(
        labeller=LABELLER_NAME,
        protocol_version=PROTOCOL_VERSION,
        status=str(status),
        target_read=False,
        train_components=str(args.train_components),
        configured_epochs=int(args.epochs),
        completed_epoch=int(completed_epoch),
        best_epoch=int(best_epoch),
        best_checkpoint=os.path.abspath(best_path),
        latest_checkpoint=os.path.abspath(latest_path),
        source_baseline_validation_summary=baseline_summary,
        source_baseline_small_validation_summary=baseline_small_summary,
        source_best_validation_summary=best_summary,
        source_best_small_validation_summary=best_small_summary,
        s7_quality_support_audit=s7_quality_support_audit,
        history=list(history))
    replacements = common.write_json_atomic(output_path, payload)
    return output_path, replacements


def write_detection_rows_pickle(rows: Sequence[Dict], path: str):
    payload = []
    for row in rows:
        detections = np.asarray(row['detections'], dtype=np.float32)
        if detections.size == 0:
            detections = np.zeros((0, 6), dtype=np.float32)
        payload.append([detections.reshape((-1, 6))])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def feature_meta(image_path: str, dino_meta: Dict) -> Dict:
    resized_h, resized_w = [int(value)
                            for value in dino_meta['resized_shape']]
    padded_h, padded_w = [int(value)
                          for value in dino_meta['padded_shape']]
    ori_h, ori_w = [int(value) for value in dino_meta['ori_shape']]
    scale = float(dino_meta['scale'])
    return dict(
        filename=image_path,
        ori_filename=os.path.basename(image_path),
        ori_shape=(ori_h, ori_w, 3),
        img_shape=(resized_h, resized_w, 3),
        pad_shape=(padded_h, padded_w, 3),
        scale_factor=np.asarray(
            [scale, scale, scale, scale], dtype=np.float32),
        flip=False, flip_direction=None,
        img_norm_cfg=dict(
            mean=np.asarray([123.675, 116.280, 103.530]),
            std=np.asarray([58.395, 57.120, 57.375]),
            to_rgb=True))


def extract_or_load_feature(dino, record: Dict, args,
                            dino_device: torch.device):
    path = cache_path(record, args)
    signature = cache_signature(record, args)
    if os.path.isfile(path):
        payload = torch.load(path, map_location='cpu')
        if payload.get('signature') == signature:
            feature = payload.get('feature')
            if (isinstance(feature, torch.Tensor) and feature.ndim == 4
                    and feature.shape[0] == 1
                    and bool(torch.isfinite(feature.float()).all().item())):
                return feature, payload['dino_meta'], True

    image = cv2.imread(record['image'])
    if image is None:
        raise RuntimeError('Cannot read {}'.format(record['image']))
    tensor, dino_meta = common.resize_and_normalize_bgr(
        image, args.dino_height, args.patch_size,
        args.dino_max_long_side)
    tensor = tensor.to(dino_device)
    feature = common.extract_patch_grid(dino, tensor, args.patch_size)
    if not bool(torch.isfinite(feature).all().item()):
        raise RuntimeError('Non-finite DINO feature')
    feature_cpu = feature.detach().cpu().half()
    atomic_torch_save(dict(
        signature=signature, feature=feature_cpu,
        dino_meta=dino_meta, frozen_dinov2=True), path)
    del tensor, feature
    return feature_cpu, dino_meta, False


def parse_original_gt(annotation: str) -> np.ndarray:
    diag = common.entry_probe.get_diag()
    boxes = []
    for gt in diag.parse_dota_ann(annotation):
        if gt.get('cls') != 'grab':
            continue
        boxes.append([
            float(gt['cx']), float(gt['cy']),
            float(gt['w']), float(gt['h']),
            math.radians(float(gt['angle']))])
    if not boxes:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def scaled_gt_tensors(annotation: str, scale: float,
                      device: torch.device):
    original = parse_original_gt(annotation)
    scaled = original.copy()
    if scaled.size:
        scaled[:, :4] *= float(scale)
    boxes = torch.as_tensor(scaled, dtype=torch.float32, device=device)
    labels = torch.zeros((boxes.shape[0],), dtype=torch.long, device=device)
    return boxes, labels, original


def record_short_token(record: Dict, args) -> float:
    image = cv2.imread(record['image'], cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('Cannot read {}'.format(record['image']))
    height, width = image.shape[:2]
    scale = min(
        float(args.dino_height) / float(min(height, width)),
        float(args.dino_max_long_side) / float(max(height, width)))
    boxes = parse_original_gt(record['annotation'])
    if boxes.shape[0] == 0:
        return float('inf')
    short_pixels = np.minimum(np.abs(boxes[:, 2]), np.abs(boxes[:, 3]))
    return float(np.min(short_pixels) * scale / float(args.patch_size))


def source_small_balanced_records(records: Sequence[Dict], args):
    """Repeat source-small frames using a source-train-only threshold."""
    values = [record_short_token(record, args) for record in records]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        raise RuntimeError('Source train has no labeled object scale')
    threshold = float(np.percentile(
        np.asarray(finite, dtype=np.float64), 100.0 / 3.0))
    repeat = int(args.source_small_repeat)
    balanced = []
    small_count = 0
    for record, value in zip(records, values):
        is_small = bool(math.isfinite(value) and value <= threshold)
        small_count += int(is_small)
        balanced.extend([record] * (repeat if is_small else 1))
    return balanced, dict(
        definition='source_train_short_token_lower_tertile',
        short_token_threshold=threshold,
        original_count=int(len(records)), small_frame_count=small_count,
        repeat=int(repeat), balanced_count=int(len(balanced)))


def source_small_records(records: Sequence[Dict], args,
                         threshold: float) -> List[Dict]:
    return [
        record for record in records
        if record_short_token(record, args) <= float(threshold)]


def split_source_records(records: Sequence[Dict], modulus: int):
    ordered = sorted(records, key=lambda row: (row['seq'], int(row['frame'])))
    train = [row for row in ordered if int(row['frame']) % modulus != 0]
    val = [row for row in ordered if int(row['frame']) % modulus == 0]
    if not train or not val:
        raise RuntimeError('Source train/validation split is empty')
    train_keys = {(row['seq'], int(row['frame'])) for row in train}
    val_keys = {(row['seq'], int(row['frame'])) for row in val}
    if train_keys & val_keys:
        raise RuntimeError('Source train/validation overlap')
    return train, val


def parse_dataset_specs(values: Sequence[str]) -> List[Tuple[str, str]]:
    """Parse ``annotation_split:image_split`` source dataset specs."""
    specs = []
    for value in values:
        parts = str(value).split(':')
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                'Dataset specs must use annotation_split:image_split: {}'
                .format(value))
        specs.append((parts[0], parts[1]))
    if not specs:
        raise ValueError('At least one source dataset spec is required')
    return specs


def discover_labeled_records_with_image_split(
        data_root: str, annotation_split: str, image_split: str) -> List[Dict]:
    """Read labels and images from independently configured splits."""
    ann_dir = os.path.join(data_root, annotation_split, 'annfiles')
    img_dir = os.path.join(data_root, image_split, 'images')
    records = []
    for annotation in sorted(glob.glob(os.path.join(ann_dir, '*.txt'))):
        base = os.path.splitext(os.path.basename(annotation))[0]
        match = re.match(r'(.+_seq\d+)_(\d{5})$', base)
        if match is None:
            continue
        image = None
        for extension in ('.jpg', '.png', '.bmp', '.tif'):
            candidate = os.path.join(img_dir, base + extension)
            if os.path.isfile(candidate):
                image = candidate
                break
        if image is None:
            continue
        records.append(dict(
            split=annotation_split, image_split=image_split,
            seq=match.group(1), frame=int(match.group(2)),
            image=image, annotation=annotation,
            domain=base.split('_', 1)[0]))
    if not records:
        raise RuntimeError(
            'No labeled records found for {}:{}'
            .format(annotation_split, image_split))
    return records


def formal_source_records(args):
    if not args.source_train_datasets or not args.source_val_datasets:
        return None
    train = []
    for annotation_split, image_split in parse_dataset_specs(
            args.source_train_datasets):
        train.extend(discover_labeled_records_with_image_split(
            args.data_root, annotation_split, image_split))
    val = []
    for annotation_split, image_split in parse_dataset_specs(
            args.source_val_datasets):
        val.extend(discover_labeled_records_with_image_split(
            args.data_root, annotation_split, image_split))
    train_keys = {os.path.realpath(row['image']) for row in train}
    val_keys = {os.path.realpath(row['image']) for row in val}
    if train_keys & val_keys:
        raise RuntimeError('Formal source train/validation overlap')
    train = sorted(
        train, key=lambda row: (row['split'], row['seq'], row['frame']))
    val = sorted(
        val, key=lambda row: (row['split'], row['seq'], row['frame']))
    return train, val


def target_records(args) -> List[Dict]:
    diag = common.entry_probe.get_diag()
    records = []
    for frame in range(args.target_start, args.target_end + 1):
        image, annotation = diag.find_files(
            args.data_root, args.target_split, args.target_seq, frame)
        if image is None or annotation is None:
            raise RuntimeError('Missing target-dev frame {}'.format(frame))
        records.append(dict(
            split=args.target_split, seq=args.target_seq, frame=int(frame),
            image=image, annotation=annotation))
    return records


def assert_training_target_isolation(source_records: Sequence[Dict],
                                     targets: Sequence[Dict]):
    source_paths = {os.path.realpath(row['image']) for row in source_records}
    target_paths = {os.path.realpath(row['image']) for row in targets}
    overlap = sorted(source_paths & target_paths)
    if overlap:
        raise RuntimeError(
            'Target-dev image leaked into source training: {}'.format(
                overlap[0]))


def feature_strides(args) -> List[int]:
    values = getattr(args, 'feature_strides', None)
    if values is None:
        return [int(args.patch_size)]
    values = sorted(set(int(value) for value in values))
    if not values or int(args.patch_size) not in values:
        raise ValueError('Feature strides must include the patch size')
    return values


def rpn_config(in_channels: int, args) -> Dict:
    strides = feature_strides(args)
    scales = [size / float(args.patch_size)
              for size in (32, 64, 128, 256, 512)]
    return dict(
        type='OrientedRPNHead', in_channels=int(in_channels),
        feat_channels=int(args.rpn_feat_channels), version='le90',
        anchor_generator=dict(
            type='AnchorGenerator', scales=scales,
            ratios=[0.5, 1.0, 2.0], strides=strides),
        bbox_coder=dict(
            type='MidpointOffsetCoder', angle_range='le90',
            target_means=[0.0] * 6,
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner', pos_iou_thr=0.7,
                neg_iou_thr=0.3, min_pos_iou=0.3,
                match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler', num=256, pos_fraction=0.5,
                neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=0, pos_weight=-1, debug=False),
        test_cfg=rpn_proposal_config(args))


def rpn_proposal_config(args) -> Dict:
    return dict(
        nms_pre=4000, max_per_img=int(args.proposal_count),
        nms=dict(type='nms', iou_threshold=0.8), min_bbox_size=0)


def s7_rpn_proposal_config(args) -> Dict:
    return dict(
        nms_pre=int(getattr(args, 's7_nms_pre', 2000)),
        max_per_img=int(getattr(args, 's7_proposal_count', 500)),
        nms=dict(type='nms', iou_threshold=0.8), min_bbox_size=0)


def s7_rpn_config(in_channels: int, args) -> Dict:
    """Proposal-only stride-7 RPN with source-defined small anchors."""
    stride = int(args.patch_size) // 2
    if int(args.patch_size) % 2 != 0 or stride <= 0:
        raise ValueError('S7 readout requires an even positive patch size')
    anchor_sizes = [float(value) for value in getattr(
        args, 's7_anchor_sizes', [16, 32, 64, 128, 256])]
    return dict(
        type='OrientedRPNHead', in_channels=int(in_channels),
        feat_channels=int(getattr(args, 's7_rpn_feat_channels', 128)),
        version='le90',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[size / float(stride) for size in anchor_sizes],
            ratios=[0.5, 1.0, 2.0], strides=[stride]),
        bbox_coder=dict(
            type='MidpointOffsetCoder', angle_range='le90',
            target_means=[0.0] * 6,
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner', pos_iou_thr=0.7,
                neg_iou_thr=0.3, min_pos_iou=0.3,
                match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler', num=256, pos_fraction=0.5,
                neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=0, pos_weight=-1, debug=False),
        test_cfg=s7_rpn_proposal_config(args))


def roi_candidate_budget(args) -> int:
    budget = int(args.proposal_count)
    if bool(getattr(args, 's7_residual', False)):
        budget += int(getattr(args, 's7_proposal_count', 500))
    return budget


def roi_config(in_channels: int, args) -> Dict:
    strides = feature_strides(args)
    return dict(
        type='OrientedStandardRoIHead', version='le90',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated', out_size=7,
                sample_num=2, clockwise=True),
            out_channels=int(in_channels),
            featmap_strides=strides),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=int(in_channels),
            fc_out_channels=int(args.roi_fc_channels),
            roi_feat_size=7, num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder', angle_range='le90',
                norm_factor=None, edge_swap=True, proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner', pos_iou_thr=0.5,
                neg_iou_thr=0.5, min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler', num=int(args.roi_samples),
                pos_fraction=0.25, neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1, debug=False),
        test_cfg=dict(
            nms_pre=roi_candidate_budget(args), min_bbox_size=0,
            score_thr=0.0,
            nms=dict(iou_thr=float(getattr(args, 'roi_nms_iou_thr', 0.1))),
            max_per_img=int(args.max_detections)))


class ResidualS7Readout(nn.Module):
    """Low-channel local readout from native DINO S14 tokens to S7."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        out_channels = int(out_channels)
        groups = min(32, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.projection = nn.Conv2d(
            int(in_channels), out_channels, kernel_size=1, bias=False)
        self.refinement = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1,
                groups=out_channels, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=1, bias=False))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as functional

        base = self.projection(feature)
        base = functional.interpolate(
            base, scale_factor=2.0, mode='bilinear', align_corners=False)
        residual = self.refinement(base)
        return base + torch.tanh(self.residual_gate) * residual


class S7ScoreCalibrator(nn.Module):
    """Monotonic affine calibration for supplement ROI foreground logits."""

    def __init__(self, initial_bias: float = -2.0):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(
            math.log(math.expm1(1.0)), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(
            float(initial_bias), dtype=torch.float32))
        self.register_buffer(
            'initial_bias', torch.tensor(float(initial_bias), dtype=torch.float32))

    def scale(self) -> torch.Tensor:
        import torch.nn.functional as functional

        return functional.softplus(self.raw_scale)

    def forward(self, foreground_log_odds: torch.Tensor) -> torch.Tensor:
        return self.scale() * foreground_log_odds + self.bias

    def prior_loss(self) -> torch.Tensor:
        return ((self.scale() - 1.0).square()
                + (self.bias - self.initial_bias).square())


class S7LaneArbitrator(nn.Module):
    """Bounded source-aware residual for the supplemental S7 lane."""

    def __init__(self, embedding_channels: int, hidden: int = 32,
                 max_adjustment: float = 2.0):
        super().__init__()
        self.max_adjustment = float(max_adjustment)
        if self.max_adjustment <= 0.0:
            raise ValueError('S7 lane max adjustment must be positive')
        self.embedding_projection = nn.Sequential(
            nn.Linear(int(embedding_channels), int(hidden)),
            nn.LayerNorm(int(hidden)), nn.GELU())
        self.scalar_projection = nn.Sequential(
            nn.Linear(3, int(hidden)), nn.GELU())
        self.output = nn.Linear(int(hidden) * 2, 1)
        # The new mode starts exactly at the already-audited epoch-1 merge.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, embedding: torch.Tensor,
                s7_raw_log_odds: torch.Tensor,
                native_context: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2 or s7_raw_log_odds.ndim != 1:
            raise ValueError('S7 lane features have invalid dimensions')
        if embedding.shape[0] != s7_raw_log_odds.shape[0]:
            raise ValueError('S7 lane features have mismatched candidate count')
        if embedding.shape[0] == 0:
            return s7_raw_log_odds.new_zeros((0,))
        context = native_context.reshape(()).expand_as(s7_raw_log_odds)
        scalars = torch.stack((
            s7_raw_log_odds, context, s7_raw_log_odds - context), dim=1)
        hidden = torch.cat((
            self.embedding_projection(embedding),
            self.scalar_projection(scalars)), dim=1)
        return (self.max_adjustment * torch.tanh(
            self.output(hidden).squeeze(1)))

    def prior_loss(self, adjustment: torch.Tensor) -> torch.Tensor:
        if adjustment.numel() == 0:
            return adjustment.sum() * 0.0
        return adjustment.square().mean()


class S7QualitySuppressor(nn.Module):
    """Predict one bounded non-positive adjustment for the whole S7 lane."""

    def __init__(self, embedding_channels: int, hidden: int = 32,
                 max_suppression: float = 2.0,
                 initial_risk_bias: float = 0.0):
        super().__init__()
        self.max_suppression = float(max_suppression)
        if self.max_suppression <= 0.0:
            raise ValueError('S7 quality max suppression must be positive')
        self.embedding_projection = nn.Sequential(
            nn.Linear(int(embedding_channels), int(hidden)),
            nn.LayerNorm(int(hidden)), nn.GELU())
        self.scalar_projection = nn.Sequential(
            nn.Linear(4, int(hidden)), nn.GELU())
        self.output = nn.Linear(int(hidden) * 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, float(initial_risk_bias))

    def forward(self, embedding: torch.Tensor,
                s7_raw_log_odds: torch.Tensor,
                s7_affine_log_odds: torch.Tensor,
                native_context: torch.Tensor):
        if (embedding.ndim != 2 or s7_raw_log_odds.ndim != 1
                or s7_affine_log_odds.ndim != 1):
            raise ValueError('S7 quality features have invalid dimensions')
        if (embedding.shape[0] != s7_raw_log_odds.shape[0]
                or s7_raw_log_odds.shape != s7_affine_log_odds.shape):
            raise ValueError('S7 quality features have mismatched counts')
        if embedding.shape[0] == 0:
            risk_logit = self.output.bias.reshape(())
            strength = torch.clamp(torch.relu(risk_logit), max=1.0)
            delta = -self.max_suppression * strength
            return delta, risk_logit, None
        top_index = torch.argmax(s7_affine_log_odds.detach())
        raw_top = s7_raw_log_odds[top_index]
        affine_top = s7_affine_log_odds[top_index]
        native_top = native_context.reshape(())
        scalars = torch.stack((
            raw_top, affine_top, native_top, affine_top - native_top), dim=0)
        hidden = torch.cat((
            self.embedding_projection(
                embedding[top_index].reshape(1, -1)),
            self.scalar_projection(scalars.reshape(1, 4))), dim=1)
        risk_logit = self.output(hidden).reshape(())
        strength = torch.clamp(torch.relu(risk_logit), max=1.0)
        delta = -self.max_suppression * strength
        return delta, risk_logit, top_index

    @staticmethod
    def prior_loss(delta: torch.Tensor) -> torch.Tensor:
        return delta.square()


class FrozenDinoRotatedHeads(nn.Module):
    def __init__(self, in_channels: int, args):
        super().__init__()
        from mmcv import ConfigDict
        from mmrotate.models.builder import build_head

        self.in_channels = int(in_channels)
        self._args = args
        self.rpn_head = build_head(ConfigDict(rpn_config(in_channels, args)))
        self.roi_head = build_head(ConfigDict(roi_config(in_channels, args)))
        self.rpn_head.init_weights()
        self.roi_head.init_weights()
        self.proposal_cfg = ConfigDict(rpn_proposal_config(args))
        self.s7_enabled = bool(getattr(args, 's7_residual', False))
        self.s7_protected_merge = bool(getattr(
            args, 's7_protected_merge', False) or getattr(
            args, 'train_components', '') in (
                    's7_merge', 's7_lane_arbitration',
                    's7_quality_suppression', 's7_temporal_association'))
        self.s7_lane_arbitration = bool(getattr(
            args, 's7_lane_arbitration', False) or getattr(
                args, 'train_components', '') == 's7_lane_arbitration')
        self.s7_quality_suppression = bool(getattr(
            args, 's7_quality_suppression', False) or getattr(
                args, 'train_components', '') == 's7_quality_suppression')
        self.s7_temporal_association = bool(getattr(
            args, 's7_temporal_association', False) or getattr(
            args, 'train_components', '') == 's7_temporal_association')
        self.s7_temporal_quality_head_enabled = bool(getattr(
            args, 's7_temporal_quality_head', False))
        if (self.s7_temporal_quality_head_enabled
                and not self.s7_temporal_association):
            raise ValueError(
                'Temporal quality head requires temporal association')
        self._last_candidate_merge = None
        self._last_temporal_pool = None
        self._s7_inference_enabled = self.s7_enabled
        if self.s7_enabled:
            s7_channels = int(getattr(args, 's7_channels', 128))
            self.s7_readout = ResidualS7Readout(
                int(in_channels), s7_channels)
            self.s7_rpn_head = build_head(ConfigDict(
                s7_rpn_config(s7_channels, args)))
            self.s7_rpn_head.init_weights()
            self.s7_proposal_cfg = ConfigDict(s7_rpn_proposal_config(args))
            self.s7_score_calibrator = (
                S7ScoreCalibrator(float(getattr(
                    args, 's7_merge_init_bias', -2.0)))
                if self.s7_protected_merge else None)
            self.s7_lane_arbitrator = (
                S7LaneArbitrator(
                    int(getattr(args, 'roi_fc_channels', 1024)),
                    int(getattr(args, 's7_lane_hidden', 32)),
                    float(getattr(args, 's7_lane_max_adjustment', 2.0)))
                if self.s7_lane_arbitration else None)
            self.s7_quality_suppressor = (
                S7QualitySuppressor(
                    int(getattr(args, 'roi_fc_channels', 1024)),
                    int(getattr(args, 's7_quality_hidden', 32)),
                    float(getattr(
                        args, 's7_quality_max_suppression', 2.0)),
                    float(getattr(
                        args, 's7_quality_init_risk_bias', 0.0)))
                if self.s7_quality_suppression else None)
            self.s7_temporal_scorer = (
                temporal.S7TemporalAssociationScorer(
                    cue_names=(temporal.QUALITY_CUE_NAMES
                               if self.s7_temporal_quality_head_enabled
                               else temporal.CUE_NAMES))
                if self.s7_temporal_association else None)
            self.s7_candidate_quality_head = (
                temporal.S7CandidateQualityHead(
                    int(getattr(args, 'roi_fc_channels', 1024)),
                    int(getattr(args, 's7_temporal_quality_hidden', 128)))
                if self.s7_temporal_quality_head_enabled else None)
        else:
            self.s7_readout = None
            self.s7_rpn_head = None
            self.s7_proposal_cfg = None
            self.s7_score_calibrator = None
            self.s7_lane_arbitrator = None
            self.s7_quality_suppressor = None
            self.s7_temporal_scorer = None
            self.s7_candidate_quality_head = None
        self._roi_cls_teacher_weight = None
        self._roi_cls_teacher_bias = None

    def set_s7_inference_enabled(self, enabled: bool):
        if enabled and not self.s7_enabled:
            raise RuntimeError('Cannot enable S7 proposals on a native head')
        self._s7_inference_enabled = bool(enabled)

    def s7_inference_enabled(self) -> bool:
        return bool(self.s7_enabled and self._s7_inference_enabled)

    def s7_feature(self, feature: torch.Tensor) -> torch.Tensor:
        if not self.s7_enabled:
            raise RuntimeError('S7 readout is not configured')
        return self.s7_readout(feature)

    def proposal_sources(self, feature: torch.Tensor, img_meta: Dict):
        """Return native and supplement proposals without losing provenance."""
        native_features = self.feature_levels(feature)
        native = self.rpn_head.simple_test_rpn(
            native_features, [img_meta])[0]
        if not self.s7_inference_enabled():
            return native_features, dict(native_s14=native)
        s7 = self.s7_feature(feature)
        supplement = self.s7_rpn_head.simple_test_rpn(
            [s7], [img_meta])[0]
        return native_features, dict(
            native_s14=native, supplement_s7=supplement)

    def simple_test_proposals(self, feature: torch.Tensor, img_meta: Dict):
        """Compatibility path for the original unprotected S7 experiment."""
        native_features, sources = self.proposal_sources(feature, img_meta)
        proposals = [sources['native_s14']]
        if 'supplement_s7' in sources:
            proposals = [torch.cat(
                [sources['native_s14'], sources['supplement_s7']], dim=0)]
        return native_features, proposals

    def _decode_roi_candidates(self, feature: torch.Tensor, img_meta: Dict,
                               proposals: torch.Tensor, rescale: bool):
        """Run the shared ROI head without NMS and preserve proposal order."""
        from mmrotate.core import rbbox2roi

        if proposals.shape[0] == 0:
            return (proposals.new_zeros((0, 5)),
                    proposals.new_zeros((0,)),
                    proposals.new_zeros((0,)),
                    proposals.new_zeros((0, int(
                        getattr(self._args, 'roi_fc_channels', 1024)))))
        rois = rbbox2roi([proposals])
        cls_score, bbox_pred, embedding = self._forward_pairwise_roi(
            feature, rois)
        decoded, _scores = self.roi_head.bbox_head.get_bboxes(
            rois, cls_score, bbox_pred, img_meta['img_shape'],
            img_meta['scale_factor'], rescale=rescale, cfg=None)
        return (decoded, roi_foreground_log_odds(cls_score),
                torch.softmax(cls_score, dim=1)[:, 0], embedding)

    def _nms_candidate_lane(self, boxes: torch.Tensor,
                            foreground_scores: torch.Tensor):
        """Apply ROI NMS within one source so S7 never deletes native boxes."""
        from mmcv.ops import nms_rotated

        if boxes.shape[0] == 0:
            return (boxes.new_zeros((0, 6)),
                    torch.zeros((0,), dtype=torch.long, device=boxes.device))
        detections, keep = nms_rotated(
            boxes, foreground_scores,
            float(getattr(self._args, 'roi_nms_iou_thr', 0.1)))
        limit = int(self._args.max_detections)
        return detections[:limit], keep[:limit]

    def _protected_merge_detections(self, feature: torch.Tensor,
                                    img_meta: Dict, rescale: bool = True):
        """Decode, calibrate, and merge two independently NMSed ROI lanes."""
        if self.s7_score_calibrator is None:
            raise RuntimeError('Protected S7 merge has no score calibrator')
        _features, sources = self.proposal_sources(feature, img_meta)
        native_boxes, native_logits, native_scores, native_embedding = (
            self._decode_roi_candidates(
            feature, img_meta, sources['native_s14'], rescale=rescale)
        )
        supplement = sources.get('supplement_s7')
        if supplement is None:
            raise RuntimeError('Protected S7 merge requires supplement proposals')
        s7_boxes, s7_logits, _s7_raw_scores, s7_embedding = (
            self._decode_roi_candidates(
                feature, img_meta, supplement, rescale=rescale))
        calibrated_s7_logits = self.s7_score_calibrator(s7_logits)
        lane_adjustment = s7_logits.new_zeros(s7_logits.shape)
        quality_delta = s7_logits.new_zeros(())
        quality_risk_logit = s7_logits.new_zeros(())
        lane_arbitrator = getattr(self, 's7_lane_arbitrator', None)
        quality_suppressor = getattr(self, 's7_quality_suppressor', None)
        if lane_arbitrator is not None and quality_suppressor is not None:
            raise RuntimeError(
                'Positive lane arbitration and quality suppression are '
                'mutually exclusive')
        if lane_arbitrator is not None:
            native_context = (native_logits.max() if native_logits.numel()
                              else s7_logits.new_zeros(()))
            lane_adjustment = lane_arbitrator(
                s7_embedding, s7_logits, native_context)
            calibrated_s7_logits = calibrated_s7_logits + lane_adjustment
        elif quality_suppressor is not None:
            native_context = (native_logits.max() if native_logits.numel()
                              else s7_logits.new_zeros(()))
            quality_delta, quality_risk_logit, _top_index = (
                quality_suppressor(
                    s7_embedding, s7_logits, calibrated_s7_logits,
                    native_context))
            lane_adjustment = quality_delta.expand_as(s7_logits)
            calibrated_s7_logits = calibrated_s7_logits + lane_adjustment
        native_detections, native_keep = self._nms_candidate_lane(
            native_boxes, native_scores)
        s7_detections, s7_keep = self._nms_candidate_lane(
            s7_boxes, torch.sigmoid(calibrated_s7_logits))
        native_post_embedding = native_embedding[native_keep]
        s7_post_embedding = s7_embedding[s7_keep]
        detections = torch.cat([native_detections, s7_detections], dim=0)
        embeddings = torch.cat(
            [native_post_embedding, s7_post_embedding], dim=0)
        source_ids = torch.cat([
            torch.zeros(
                native_detections.shape[0], dtype=torch.long,
                device=detections.device),
            torch.ones(
                s7_detections.shape[0], dtype=torch.long,
                device=detections.device)], dim=0)
        if detections.shape[0]:
            order = torch.argsort(detections[:, 5], descending=True)
            order = order[:int(self._args.max_detections)]
            detections = detections[order]
            embeddings = embeddings[order]
            source_ids = source_ids[order]
        quality_logits = None
        if (bool(getattr(self, 's7_temporal_quality_head_enabled', False))
                and getattr(self, 's7_candidate_quality_head', None) is not None):
            quality_logits = self.s7_candidate_quality_head(
                embeddings, detections, source_ids).detach()
        self._last_temporal_pool = dict(
            detections=detections.detach(),
            embeddings=embeddings.detach(),
            source_ids=source_ids.detach(),
            quality_logits=quality_logits,
            candidate_limit=int(getattr(
                self._args, 's7_temporal_max_candidates', 100)))
        self._last_candidate_merge = dict(
            proposal_source_counts=dict(
                native_s14=int(sources['native_s14'].shape[0]),
                supplement_s7=int(supplement.shape[0])),
            post_nms_source_counts=dict(
                native_s14=int(native_detections.shape[0]),
                supplement_s7=int(s7_detections.shape[0])),
            raw_top1_source=(
                None if source_ids.numel() == 0 else
                ('native_s14' if int(source_ids[0].item()) == 0
                 else 'supplement_s7')),
            source_top1_detections=dict(
                native_s14=(
                    None if native_detections.shape[0] == 0 else
                    [float(value) for value in
                     native_detections[0].detach().cpu().tolist()]),
                supplement_s7=(
                    None if s7_detections.shape[0] == 0 else
                    [float(value) for value in
                     s7_detections[0].detach().cpu().tolist()])),
            source_pre_nms_top_log_odds=dict(
                native_s14=(
                    None if native_logits.numel() == 0 else
                    float(native_logits.max().detach().item())),
                supplement_s7_raw=(
                    None if s7_logits.numel() == 0 else
                    float(s7_logits.max().detach().item())),
                supplement_s7_calibrated=(
                    None if calibrated_s7_logits.numel() == 0 else
                    float(calibrated_s7_logits.max().detach().item()))),
            s7_affine_scale=float(
                self.s7_score_calibrator.scale().detach().item()),
            s7_affine_bias=float(
                self.s7_score_calibrator.bias.detach().item()),
            s7_lane_adjustment_max=float(
                lane_adjustment.detach().abs().max().item())
            if lane_adjustment.numel() else 0.0,
            s7_lane_adjustment_mean=float(
                lane_adjustment.detach().mean().item())
            if lane_adjustment.numel() else 0.0,
            s7_quality_delta=float(quality_delta.detach().item()),
            s7_quality_risk_logit=float(
                quality_risk_logit.detach().item()),
            s7_quality_risk_probability=float(
                torch.sigmoid(quality_risk_logit.detach()).item()))
        return detections

    def feature_levels(self, feature: torch.Tensor):
        """Build optional spatial levels from the frozen patch grid.

        The default path returns the original tensor unchanged.  Non-default
        levels are interpolation-only and add no trainable parameters; their
        RPN/ROI weights are still trained source-only when this option is
        enabled.
        """
        strides = feature_strides(self._args)
        if strides == [int(self._args.patch_size)]:
            return [feature]
        import torch.nn.functional as functional
        base_stride = float(self._args.patch_size)
        levels = []
        for stride in strides:
            scale = base_stride / float(stride)
            height = max(1, int(round(feature.shape[-2] * scale)))
            width = max(1, int(round(feature.shape[-1] * scale)))
            if height == feature.shape[-2] and width == feature.shape[-1]:
                levels.append(feature)
            else:
                levels.append(functional.interpolate(
                    feature, size=(height, width), mode='bilinear',
                    align_corners=False))
        return levels

    def capture_roi_cls_teacher(self):
        """Freeze the initialized source classifier as a retention teacher."""
        classifier = self.roi_head.bbox_head.fc_cls
        self._roi_cls_teacher_weight = classifier.weight.detach().clone()
        self._roi_cls_teacher_bias = (
            None if classifier.bias is None
            else classifier.bias.detach().clone())

    def roi_cls_teacher_state(self):
        if self._roi_cls_teacher_weight is None:
            return None
        return dict(
            weight=self._roi_cls_teacher_weight.detach().cpu(),
            bias=(None if self._roi_cls_teacher_bias is None else
                  self._roi_cls_teacher_bias.detach().cpu()))

    def load_roi_cls_teacher_state(self, state: Dict):
        if not state or 'weight' not in state:
            raise RuntimeError('Pairwise checkpoint lacks ROI teacher state')
        classifier = self.roi_head.bbox_head.fc_cls
        weight = state['weight'].to(
            device=classifier.weight.device, dtype=classifier.weight.dtype)
        bias = state.get('bias')
        if weight.shape != classifier.weight.shape:
            raise RuntimeError('ROI teacher weight shape mismatch')
        if bias is not None:
            bias = bias.to(
                device=classifier.weight.device,
                dtype=classifier.weight.dtype)
            if classifier.bias is None or bias.shape != classifier.bias.shape:
                raise RuntimeError('ROI teacher bias shape mismatch')
        self._roi_cls_teacher_weight = weight.detach().clone()
        self._roi_cls_teacher_bias = (
            None if bias is None else bias.detach().clone())

    def _forward_pairwise_roi(self, feature: torch.Tensor,
                              rois: torch.Tensor):
        """Expose the exact Shared2FC representation entering ``fc_cls``."""
        bbox_head = self.roi_head.bbox_head
        if (bbox_head.num_shared_convs != 0
                or bbox_head.num_shared_fcs != 2
                or bbox_head.num_cls_convs != 0
                or bbox_head.num_cls_fcs != 0
                or bbox_head.num_reg_convs != 0
                or bbox_head.num_reg_fcs != 0):
            raise RuntimeError(
                'Pairwise mode requires the configured Shared2FC ROI head')
        x = self.roi_head.bbox_roi_extractor(
            self.feature_levels(feature), rois)
        if self.roi_head.with_shared_head:
            x = self.roi_head.shared_head(x)
        if bbox_head.with_avg_pool:
            x = bbox_head.avg_pool(x)
        x = x.flatten(1)
        for layer in bbox_head.shared_fcs:
            x = bbox_head.relu(layer(x))
        cls_score = bbox_head.fc_cls(x)
        bbox_pred = bbox_head.fc_reg(x)
        return cls_score, bbox_pred, x

    def forward_train(self, feature: torch.Tensor, img_meta: Dict,
                      gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        features = self.feature_levels(feature)
        img_metas = [img_meta]
        gt_bboxes = [gt_boxes]
        rpn_losses, proposals = self.rpn_head.forward_train(
            features, img_metas, gt_bboxes, gt_labels=None,
            gt_bboxes_ignore=None, proposal_cfg=self.proposal_cfg)
        roi_losses = self.roi_head.forward_train(
            features, img_metas, proposals, gt_bboxes, [gt_labels],
            gt_bboxes_ignore=None, gt_masks=None)
        losses = dict(rpn_losses)
        losses.update(roi_losses)
        return losses

    def forward_s7_rpn_train(self, feature: torch.Tensor, img_meta: Dict,
                             gt_boxes: torch.Tensor) -> Dict:
        """Train only the S7 readout/RPN; native RPN and ROI stay frozen."""
        if not self.s7_enabled:
            raise RuntimeError('S7 RPN training requires an S7-enabled head')
        losses, _proposals = self.s7_rpn_head.forward_train(
            [self.s7_feature(feature)], [img_meta], [gt_boxes],
            gt_labels=None, gt_bboxes_ignore=None,
            proposal_cfg=self.s7_proposal_cfg)
        renamed = {}
        for name, value in losses.items():
            if name == 'loss_rpn_cls':
                name = 'loss_s7_rpn_cls'
            elif name == 'loss_rpn_bbox':
                name = 'loss_s7_rpn_bbox'
            renamed[name] = value
        return renamed

    def forward_s7_merge_train(self, feature: torch.Tensor, img_meta: Dict,
                               gt_boxes: torch.Tensor, riou_thr: float,
                               margin: float, retention_weight: float,
                               gain_weight: float,
                               prior_weight: float) -> Dict:
        """Train only the S7 pre-NMS score calibration on source pairs."""
        from mmcv.ops import box_iou_rotated

        if not self.s7_protected_merge or self.s7_score_calibrator is None:
            raise RuntimeError('S7 merge training requires protected merge')
        with torch.no_grad():
            _features, sources = self.proposal_sources(feature, img_meta)
            supplement = sources.get('supplement_s7')
            if supplement is None:
                raise RuntimeError('S7 merge training has no supplement lane')
            native_boxes, native_logits, _native_scores, _native_embedding = (
                self._decode_roi_candidates(
                feature, img_meta, sources['native_s14'], rescale=False)
            )
            s7_boxes, s7_logits, _s7_scores, _s7_embedding = (
                self._decode_roi_candidates(
                    feature, img_meta, supplement, rescale=False))
            if gt_boxes.shape[0] and native_boxes.shape[0]:
                native_overlap = box_iou_rotated(
                    native_boxes.float(), gt_boxes.float()).max(dim=1).values
            else:
                native_overlap = native_logits.new_zeros(native_logits.shape)
            if gt_boxes.shape[0] and s7_boxes.shape[0]:
                s7_overlap = box_iou_rotated(
                    s7_boxes.float(), gt_boxes.float()).max(dim=1).values
            else:
                s7_overlap = s7_logits.new_zeros(s7_logits.shape)
        pairs = s7_merge_pair_losses(
            native_logits.detach(), native_overlap.detach(),
            s7_logits.detach(), s7_overlap.detach(),
            self.s7_score_calibrator, margin=float(margin),
            riou_thr=float(riou_thr))
        prior = self.s7_score_calibrator.prior_loss()
        return dict(
            loss_s7_merge_retention=(
                pairs['retention'] * float(retention_weight)),
            loss_s7_merge_gain=(pairs['gain'] * float(gain_weight)),
            loss_s7_merge_prior=(prior * float(prior_weight)),
            s7_merge_retain_pair_count=pairs['retain_pair_count'],
            s7_merge_gain_pair_count=pairs['gain_pair_count'],
            s7_merge_retention_active=pairs['retention_active'],
            s7_merge_gain_active=pairs['gain_active'],
            s7_merge_native_top1_correct=pairs['native_top1_correct'],
            s7_merge_native_candidate_count=int(native_logits.numel()),
            s7_merge_supplement_candidate_count=int(s7_logits.numel()))

    def forward_s7_lane_arbitration_train(
            self, feature: torch.Tensor, img_meta: Dict,
            gt_boxes: torch.Tensor, riou_thr: float, margin: float,
            retention_weight: float, gain_weight: float,
            prior_weight: float, hard_negatives: int) -> Dict:
        """Train only a bounded source-aware residual on the fixed S7 lane."""
        from mmcv.ops import box_iou_rotated

        if (not self.s7_lane_arbitration
                or self.s7_score_calibrator is None
                or self.s7_lane_arbitrator is None):
            raise RuntimeError(
                'S7 lane arbitration requires the configured lane arbitrator')
        with torch.no_grad():
            _features, sources = self.proposal_sources(feature, img_meta)
            supplement = sources.get('supplement_s7')
            if supplement is None:
                raise RuntimeError('S7 lane training has no supplement lane')
            native_boxes, native_logits, _native_scores, _native_embedding = (
                self._decode_roi_candidates(
                    feature, img_meta, sources['native_s14'], rescale=False))
            s7_boxes, s7_logits, _s7_scores, s7_embedding = (
                self._decode_roi_candidates(
                    feature, img_meta, supplement, rescale=False))
            if gt_boxes.shape[0] and native_boxes.shape[0]:
                native_overlap = box_iou_rotated(
                    native_boxes.float(), gt_boxes.float()).max(dim=1).values
            else:
                native_overlap = native_logits.new_zeros(native_logits.shape)
            if gt_boxes.shape[0] and s7_boxes.shape[0]:
                s7_overlap = box_iou_rotated(
                    s7_boxes.float(), gt_boxes.float()).max(dim=1).values
            else:
                s7_overlap = s7_logits.new_zeros(s7_logits.shape)
        native_top_logit = (native_logits.max().detach()
                            if native_logits.numel() else
                            s7_logits.new_zeros(()))
        base_s7_logits = self.s7_score_calibrator(
            s7_logits.detach()).detach()
        adjustment = self.s7_lane_arbitrator(
            s7_embedding.detach(), s7_logits.detach(), native_top_logit)
        adjusted_s7_logits = base_s7_logits + adjustment
        pairs = s7_lane_pair_losses(
            native_logits.detach(), native_overlap.detach(),
            adjusted_s7_logits, s7_overlap.detach(), margin,
            hard_negatives, riou_thr=riou_thr)
        prior = self.s7_lane_arbitrator.prior_loss(adjustment)
        return dict(
            loss_s7_lane_retention=(
                pairs['retention'] * float(retention_weight)),
            loss_s7_lane_gain=(pairs['gain'] * float(gain_weight)),
            loss_s7_lane_prior=(prior * float(prior_weight)),
            s7_lane_retain_pair_count=pairs['retain_pair_count'],
            s7_lane_gain_pair_count=pairs['gain_pair_count'],
            s7_lane_retention_active=pairs['retention_active'],
            s7_lane_gain_active=pairs['gain_active'],
            s7_lane_gain_s7_competitor_count=(
                pairs['gain_s7_competitor_count']),
            s7_lane_native_top1_correct=pairs['native_top1_correct'],
            s7_lane_adjustment_mean=float(adjustment.detach().mean().item())
            if adjustment.numel() else 0.0,
            s7_lane_candidate_count=int(s7_logits.numel()))

    def forward_s7_quality_suppression_train(
            self, feature: torch.Tensor, img_meta: Dict,
            gt_boxes: torch.Tensor, riou_thr: float, margin: float,
            risk_weight: float, preserve_weight: float,
            retention_weight: float, prior_weight: float) -> Dict:
        """Train one source-only non-positive adjustment for the whole lane."""
        from mmcv.ops import box_iou_rotated

        suppressor = getattr(self, 's7_quality_suppressor', None)
        if (not self.s7_protected_merge
                or self.s7_score_calibrator is None
                or suppressor is None):
            raise RuntimeError(
                'S7 quality training requires the fixed affine merge and '
                'lane-wide suppressor')
        with torch.no_grad():
            _features, sources = self.proposal_sources(feature, img_meta)
            supplement = sources.get('supplement_s7')
            if supplement is None:
                raise RuntimeError('S7 quality training has no supplement lane')
            native_boxes, native_logits, _native_scores, _native_embedding = (
                self._decode_roi_candidates(
                    feature, img_meta, sources['native_s14'], rescale=False))
            s7_boxes, s7_logits, _s7_scores, s7_embedding = (
                self._decode_roi_candidates(
                    feature, img_meta, supplement, rescale=False))
            if gt_boxes.shape[0] and native_boxes.shape[0]:
                native_overlap = box_iou_rotated(
                    native_boxes.float(), gt_boxes.float()).max(dim=1).values
            else:
                native_overlap = native_logits.new_zeros(native_logits.shape)
            if gt_boxes.shape[0] and s7_boxes.shape[0]:
                s7_overlap = box_iou_rotated(
                    s7_boxes.float(), gt_boxes.float()).max(dim=1).values
            else:
                s7_overlap = s7_logits.new_zeros(s7_logits.shape)
        base_s7_logits = self.s7_score_calibrator(
            s7_logits.detach()).detach()
        native_top_logit = (native_logits.max().detach()
                            if native_logits.numel()
                            else s7_logits.new_zeros(()))
        delta, risk_logit, _s7_top = suppressor(
            s7_embedding.detach(), s7_logits.detach(), base_s7_logits,
            native_top_logit)
        pairs = s7_quality_suppression_losses(
            native_logits.detach(), native_overlap.detach(),
            base_s7_logits, s7_overlap.detach(), delta, risk_logit,
            margin=float(margin), riou_thr=float(riou_thr))
        prior = suppressor.prior_loss(delta)
        return dict(
            loss_s7_quality_risk=pairs['risk'] * float(risk_weight),
            loss_s7_quality_preserve=(
                pairs['preserve'] * float(preserve_weight)),
            loss_s7_quality_retention=(
                pairs['retention'] * float(retention_weight)),
            loss_s7_quality_prior=prior * float(prior_weight),
            s7_quality_risk_pair_count=pairs['risk_pair_count'],
            s7_quality_preserve_pair_count=pairs['preserve_pair_count'],
            s7_quality_retention_active=pairs['retention_active'],
            s7_quality_native_top1_correct=pairs['native_top1_correct'],
            s7_quality_s7_top1_correct=pairs['s7_top1_correct'],
            s7_quality_native_top1_riou=pairs['native_top1_riou'],
            s7_quality_s7_top1_riou=pairs['s7_top1_riou'],
            s7_quality_delta=float(delta.detach().item()),
            s7_quality_risk_probability=float(
                torch.sigmoid(risk_logit.detach()).item()),
            s7_quality_base_gap=pairs['base_gap'],
            s7_quality_candidate_count=int(s7_logits.numel()))

    def forward_s7_temporal_association_train(
            self, feature: torch.Tensor, img_meta: Dict,
            gt_boxes: torch.Tensor,
            previous_box: Optional[torch.Tensor],
            previous_embedding: Optional[torch.Tensor],
            riou_thr: float, margin: float,
            retention_weight: float, gain_weight: float,
            prior_weight: float, max_candidates: int) -> Dict:
        """Fit only causal cue weights on consecutive source-frame pairs."""
        from mmcv.ops import box_iou_rotated

        scorer = getattr(self, 's7_temporal_scorer', None)
        if (not self.s7_protected_merge
                or self.s7_score_calibrator is None or scorer is None):
            raise RuntimeError(
                'S7 temporal training requires the fixed affine pool and '
                'temporal scorer')
        with torch.no_grad():
            self._protected_merge_detections(
                feature, img_meta, rescale=False)
            pool = self._last_temporal_pool
            if pool is None:
                raise RuntimeError('Temporal candidate pool was not produced')
            limit = min(int(max_candidates), int(pool['detections'].shape[0]))
            detections = pool['detections'][:limit]
            embeddings = pool['embeddings'][:limit]
            source_ids = pool['source_ids'][:limit]
            if gt_boxes.shape[0] and detections.shape[0]:
                overlap = box_iou_rotated(
                    detections[:, :5].float(), gt_boxes.float()).max(
                        dim=1).values
            else:
                overlap = detections.new_zeros((detections.shape[0],))
            teacher_index = (
                int(torch.argmax(overlap).item()) if overlap.numel() else None)
            teacher_usable = bool(
                teacher_index is not None
                and overlap[teacher_index] >= float(riou_thr))
            teacher_box = (
                detections[teacher_index, :5].detach().clone()
                if teacher_usable else None)
            teacher_embedding = (
                embeddings[teacher_index].detach().clone()
                if teacher_usable else None)

        if (previous_box is None or previous_embedding is None
                or detections.shape[0] == 0):
            zero = scorer.raw_weights.sum() * 0.0
            pairs = dict(
                loss_s7_temporal_retention=zero,
                loss_s7_temporal_gain=zero,
                loss_s7_temporal_prior=(
                    scorer.prior_loss() * float(prior_weight)),
                s7_temporal_retention_pair_count=0,
                s7_temporal_gain_pair_count=0,
                s7_temporal_native_top1_correct=0,
                s7_temporal_usable_candidate_count=int(
                    (overlap >= float(riou_thr)).sum().item()),
                s7_temporal_candidate_count=int(detections.shape[0]))
        else:
            cues = temporal.build_temporal_cues(
                detections, embeddings, previous_box, previous_embedding)
            pairs = temporal.temporal_pair_losses(
                scorer, cues, overlap.detach(), source_ids.detach(),
                riou_threshold=riou_thr, margin=margin,
                retention_weight=retention_weight,
                gain_weight=gain_weight, prior_weight=prior_weight)
        pairs['_temporal_teacher_box'] = teacher_box
        pairs['_temporal_teacher_embedding'] = teacher_embedding
        pairs['_temporal_teacher_usable'] = bool(teacher_usable)
        return pairs

    def forward_s7_temporal_quality_train(
            self, feature: torch.Tensor, img_meta: Dict,
            gt_boxes: torch.Tensor, riou_thr: float,
            quality_weight: float, max_candidates: int,
            relative_weight: float = 0.0,
            relative_margin: float = 0.25,
            relative_min_gap: float = 0.10,
            relative_max_pairs: int = 128) -> Dict:
        """Fit pointwise and optional relative quality on fixed affine pool."""
        from mmcv.ops import box_iou_rotated

        quality_head = getattr(self, 's7_candidate_quality_head', None)
        if (not self.s7_protected_merge
                or self.s7_score_calibrator is None
                or quality_head is None):
            raise RuntimeError(
                'Temporal quality training requires the fixed affine pool '
                'and candidate quality head')
        with torch.no_grad():
            self._protected_merge_detections(
                feature, img_meta, rescale=False)
            pool = self._last_temporal_pool
            if pool is None:
                raise RuntimeError('Temporal candidate pool was not produced')
            limit = min(int(max_candidates), int(pool['detections'].shape[0]))
            detections = pool['detections'][:limit]
            embeddings = pool['embeddings'][:limit]
            source_ids = pool['source_ids'][:limit]
            if gt_boxes.shape[0] and detections.shape[0]:
                overlap = box_iou_rotated(
                    detections[:, :5].float(), gt_boxes.float()).max(
                        dim=1).values
            else:
                overlap = detections.new_zeros((detections.shape[0],))
        losses = temporal.candidate_quality_losses(
            quality_head, embeddings, detections, source_ids, overlap,
            riou_threshold=riou_thr,
            relative_margin=(
                float(relative_margin) if float(relative_weight) > 0.0
                else None),
            relative_min_gap=float(relative_min_gap),
            relative_max_pairs=int(relative_max_pairs))
        losses['loss_s7_candidate_quality'] = (
            losses['loss_s7_candidate_quality'] * float(quality_weight))
        if 'loss_s7_candidate_quality_relative' in losses:
            losses['loss_s7_candidate_quality_relative'] = (
                losses['loss_s7_candidate_quality_relative']
                * float(relative_weight))
        return losses

    def forward_roi_cls_hard_train(
            self, feature: torch.Tensor, img_meta: Dict,
            gt_boxes: torch.Tensor, max_samples: int,
            positive_fraction: float = 0.25,
            riou_thr: float = 0.5) -> Dict:
        """Mine decoded-ROI positives and hard negatives for ``fc_cls``."""
        import torch.nn.functional as functional
        from mmcv.ops import box_iou_rotated
        from mmrotate.core import rbbox2roi

        features, proposal_list = self.simple_test_proposals(
            feature, img_meta)
        with torch.no_grad():
            proposals = proposal_list[0][:, :5]
            if gt_boxes.shape[0]:
                proposals = torch.cat([proposals, gt_boxes], dim=0)
        rois = rbbox2roi([proposals])
        bbox_results = self.roi_head._bbox_forward(features, rois)
        cls_score = bbox_results['cls_score']
        with torch.no_grad():
            decoded, _scores = self.roi_head.bbox_head.get_bboxes(
                rois, cls_score.detach(), bbox_results['bbox_pred'],
                img_meta['img_shape'], img_meta['scale_factor'],
                rescale=False, cfg=None)
            if decoded.shape[1] != 5:
                raise RuntimeError(
                    'ROI-cls hard mining requires class-agnostic regression')
            if gt_boxes.shape[0]:
                overlap_by_gt = box_iou_rotated(
                    decoded.float(), gt_boxes.float())
                overlap = overlap_by_gt.max(dim=1).values
            else:
                overlap_by_gt = decoded.new_zeros((decoded.shape[0], 0))
                overlap = decoded.new_zeros((decoded.shape[0],))
            positive = overlap >= float(riou_thr)
            foreground_score = torch.softmax(
                cls_score.detach(), dim=1)[:, 0]
            positive_indices = torch.nonzero(
                positive, as_tuple=False).reshape(-1)
            negative_indices = torch.nonzero(
                ~positive, as_tuple=False).reshape(-1)
            max_positive = max(1, int(max_samples * positive_fraction))
            if positive_indices.numel() > max_positive:
                hard_positive_order = torch.argsort(
                    foreground_score[positive_indices])
                positive_indices = positive_indices[
                    hard_positive_order[:max_positive]]
            max_negative = max(0, int(max_samples) - positive_indices.numel())
            if negative_indices.numel() > max_negative:
                hard_negative_order = torch.argsort(
                    foreground_score[negative_indices], descending=True)
                negative_indices = negative_indices[
                    hard_negative_order[:max_negative]]
            selected = torch.cat(
                [positive_indices, negative_indices], dim=0)
            labels = torch.cat([
                torch.zeros(
                    positive_indices.numel(), dtype=torch.long,
                    device=cls_score.device),
                torch.ones(
                    negative_indices.numel(), dtype=torch.long,
                    device=cls_score.device)], dim=0)
        if selected.numel() == 0:
            raise RuntimeError('ROI-cls hard mining selected no samples')
        loss = functional.cross_entropy(cls_score[selected], labels)
        accuracy = (cls_score[selected].argmax(dim=1) == labels).float().mean()
        return dict(
            loss_cls=loss, roi_cls_accuracy=accuracy,
            roi_cls_positive_count=int(positive_indices.numel()),
            roi_cls_hard_negative_count=int(negative_indices.numel()),
            roi_cls_candidate_count=int(proposals.shape[0]))

    def forward_roi_cls_pairwise_train(
            self, feature: torch.Tensor, img_meta: Dict,
            gt_boxes: torch.Tensor, max_samples: int,
            positive_fraction: float = 0.25, riou_thr: float = 0.5,
            negative_riou_thr: float = 0.1, nms_iou_thr: float = 0.1,
            pairwise_margin: float = 0.5,
            pairwise_loss_weight: float = 1.0,
            retention_loss_weight: float = 1.0,
            retention_temperature: float = 1.0) -> Dict:
        """Train ``fc_cls`` with source-only NMS-aware pair ranking."""
        import torch.nn.functional as functional
        from mmcv.ops import box_iou_rotated
        from mmrotate.core import rbbox2roi

        if self._roi_cls_teacher_weight is None:
            raise RuntimeError('ROI retention teacher was not initialized')
        features, proposal_list = self.simple_test_proposals(
            feature, img_meta)
        with torch.no_grad():
            proposals = proposal_list[0][:, :5]
            if gt_boxes.shape[0]:
                proposals = torch.cat([proposals, gt_boxes], dim=0)
        rois = rbbox2roi([proposals])
        cls_score, bbox_pred, cls_features = self._forward_pairwise_roi(
            feature, rois)
        with torch.no_grad():
            teacher_logits = functional.linear(
                cls_features.detach(), self._roi_cls_teacher_weight,
                self._roi_cls_teacher_bias)
            decoded, _scores = self.roi_head.bbox_head.get_bboxes(
                rois, cls_score.detach(), bbox_pred.detach(),
                img_meta['img_shape'], img_meta['scale_factor'],
                rescale=False, cfg=None)
            if decoded.shape[1] != 5:
                raise RuntimeError(
                    'Pairwise ROI training requires class-agnostic regression')
            if gt_boxes.shape[0]:
                overlap = box_iou_rotated(
                    decoded.float(), gt_boxes.float()).max(dim=1).values
            else:
                overlap = decoded.new_zeros((decoded.shape[0],))
            foreground_score = torch.softmax(
                cls_score.detach(), dim=1)[:, 0]
            all_positive = torch.nonzero(
                overlap >= float(riou_thr),
                as_tuple=False).reshape(-1)
            competitor_iou = decoded.new_zeros((decoded.shape[0],))
            if all_positive.numel():
                competitor_iou = box_iou_rotated(
                    decoded.float(), decoded[all_positive].float()).max(
                        dim=1).values
            positive_indices, negative_indices = (
                select_hard_pairwise_indices(
                    overlap, foreground_score, competitor_iou,
                    max_samples=max_samples,
                    positive_fraction=positive_fraction,
                    positive_riou_thr=riou_thr,
                    negative_riou_thr=negative_riou_thr,
                    nms_iou_thr=nms_iou_thr))
            selected = torch.cat(
                [positive_indices, negative_indices], dim=0)
            labels = torch.cat([
                torch.zeros(
                    positive_indices.numel(), dtype=torch.long,
                    device=cls_score.device),
                torch.ones(
                    negative_indices.numel(), dtype=torch.long,
                    device=cls_score.device)], dim=0)
        if selected.numel() == 0:
            raise RuntimeError('Pairwise ROI mining selected no samples')
        loss_cls = functional.cross_entropy(cls_score[selected], labels)
        log_odds = roi_foreground_log_odds(cls_score)
        raw_pairwise = roi_pairwise_margin_loss(
            log_odds[positive_indices], log_odds[negative_indices],
            pairwise_margin)
        raw_retention = roi_classifier_retention_loss(
            cls_score, teacher_logits, retention_temperature)
        pairwise_accuracy = cls_score.new_tensor(0.0)
        pair_count = int(
            positive_indices.numel() * negative_indices.numel())
        if pair_count:
            pairwise_accuracy = (
                log_odds[positive_indices, None]
                > log_odds[None, negative_indices]).float().mean()
        selected_accuracy = (
            cls_score[selected].argmax(dim=1) == labels).float().mean()
        suppressor_count = int((
            competitor_iou[negative_indices] >= float(nms_iou_thr)
        ).sum().item())
        return dict(
            loss_cls=loss_cls,
            loss_roi_pairwise=(raw_pairwise * float(pairwise_loss_weight)),
            loss_roi_retention=(raw_retention
                                * float(retention_loss_weight)),
            roi_cls_accuracy=selected_accuracy,
            roi_pairwise_accuracy=pairwise_accuracy,
            roi_cls_positive_count=int(positive_indices.numel()),
            roi_cls_hard_negative_count=int(negative_indices.numel()),
            roi_cls_nms_competitor_count=suppressor_count,
            roi_pair_count=pair_count,
            roi_cls_candidate_count=int(proposals.shape[0]))

    def forward_roi_cls_pairwise_v2_train(
            self, feature: torch.Tensor, img_meta: Dict,
            gt_boxes: torch.Tensor, max_samples: int,
            positive_fraction: float = 0.25, riou_thr: float = 0.5,
            nms_iou_thr: float = 0.5, pairwise_margin: float = 0.5,
            cls_loss_weight: float = 0.25,
            pairwise_loss_weight: float = 1.0,
            retention_loss_weight: float = 1.0,
            retention_temperature: float = 1.0,
            negatives_per_positive: int = 2) -> Dict:
        """Train ``fc_cls`` on actual source ordering/NMS failures only."""
        import torch.nn.functional as functional
        from mmcv.ops import box_iou_rotated
        from mmrotate.core import rbbox2roi

        if self._roi_cls_teacher_weight is None:
            raise RuntimeError('ROI retention teacher was not initialized')
        features, proposal_list = self.simple_test_proposals(
            feature, img_meta)
        with torch.no_grad():
            proposals = proposal_list[0][:, :5]
            # V2 is a deployment-faithful ordering probe.  Do not append GT
            # boxes: an injected GT ROI is an oracle positive and can make
            # every source frame have zero actual competitors.
        rois = rbbox2roi([proposals])
        cls_score, bbox_pred, cls_features = self._forward_pairwise_roi(
            feature, rois)
        with torch.no_grad():
            teacher_logits = functional.linear(
                cls_features.detach(), self._roi_cls_teacher_weight,
                self._roi_cls_teacher_bias)
            decoded, _scores = self.roi_head.bbox_head.get_bboxes(
                rois, cls_score.detach(), bbox_pred.detach(),
                img_meta['img_shape'], img_meta['scale_factor'],
                rescale=False, cfg=None)
            if decoded.shape[1] != 5:
                raise RuntimeError(
                    'Pairwise ROI training requires class-agnostic regression')
            if gt_boxes.shape[0]:
                overlap_by_gt = box_iou_rotated(
                    decoded.float(), gt_boxes.float())
                overlap = overlap_by_gt.max(dim=1).values
            else:
                overlap_by_gt = decoded.new_zeros((decoded.shape[0], 0))
                overlap = decoded.new_zeros((decoded.shape[0],))
            foreground_score = torch.softmax(
                cls_score.detach(), dim=1)[:, 0]
            positive_indices = select_representative_usable_rois(
                overlap_by_gt, foreground_score,
                positive_riou_thr=riou_thr,
                max_positives=max(1, int(max_samples * positive_fraction)))
            if positive_indices.numel():
                candidate_to_positive_iou = box_iou_rotated(
                    decoded.float(), decoded[positive_indices].float())
            else:
                candidate_to_positive_iou = decoded.new_zeros(
                    (decoded.shape[0], 0))
            (pair_positive_indices, pair_negative_indices,
             pair_is_suppressor) = mine_actual_roi_competitor_pairs(
                 overlap, foreground_score, positive_indices,
                 candidate_to_positive_iou,
                 positive_riou_thr=riou_thr,
                 nms_iou_thr=nms_iou_thr,
                 negatives_per_positive=negatives_per_positive)
            selected_positive = torch.unique(pair_positive_indices)
            selected_negative = torch.unique(pair_negative_indices)
            selected = torch.cat(
                [selected_positive, selected_negative], dim=0)
            labels = torch.cat([
                torch.zeros(
                    selected_positive.numel(), dtype=torch.long,
                    device=cls_score.device),
                torch.ones(
                    selected_negative.numel(), dtype=torch.long,
                    device=cls_score.device)], dim=0)

        zero = cls_score.sum() * 0.0
        if selected.numel():
            raw_cls = functional.cross_entropy(cls_score[selected], labels)
            selected_accuracy = (
                cls_score[selected].argmax(dim=1) == labels).float().mean()
        else:
            raw_cls = zero
            selected_accuracy = cls_score.new_tensor(1.0)
        log_odds = roi_foreground_log_odds(cls_score)
        raw_pairwise = roi_paired_margin_loss(
            log_odds[pair_positive_indices],
            log_odds[pair_negative_indices], pairwise_margin)
        raw_retention = roi_classifier_retention_loss(
            cls_score, teacher_logits, retention_temperature)
        if pair_positive_indices.numel():
            pairwise_accuracy = (
                log_odds[pair_positive_indices]
                > log_odds[pair_negative_indices]).float().mean()
        else:
            pairwise_accuracy = cls_score.new_tensor(1.0)
        pair_count = int(pair_positive_indices.numel())
        suppressor_count = int(pair_is_suppressor.sum().item())
        return dict(
            loss_cls=(raw_cls * float(cls_loss_weight)),
            loss_roi_pairwise=(raw_pairwise * float(pairwise_loss_weight)),
            loss_roi_retention=(raw_retention
                                * float(retention_loss_weight)),
            roi_cls_accuracy=selected_accuracy,
            roi_pairwise_accuracy=pairwise_accuracy,
            roi_cls_positive_count=int(selected_positive.numel()),
            roi_cls_hard_negative_count=int(selected_negative.numel()),
            roi_cls_nms_competitor_count=suppressor_count,
            roi_cls_ordering_competitor_count=(
                pair_count - suppressor_count),
            roi_pair_count=pair_count,
            roi_pairwise_failure_frame=int(pair_count > 0),
            roi_pairwise_unopposed_positive_count=(
                int(positive_indices.numel())
                - int(selected_positive.numel())),
            roi_cls_candidate_count=int(proposals.shape[0]))

    def simple_test(self, feature: torch.Tensor, img_meta: Dict):
        self._last_candidate_merge = None
        self._last_temporal_pool = None
        if (self.s7_protected_merge and self.s7_inference_enabled()
                and self.s7_score_calibrator is not None):
            detections = self._protected_merge_detections(feature, img_meta)
            return detections.detach().cpu().numpy().astype(
                np.float32, copy=False)
        features, proposals = self.simple_test_proposals(feature, img_meta)
        results = self.roi_head.simple_test(
            features, proposals, [img_meta], rescale=True)
        if len(results) != 1 or len(results[0]) != 1:
            raise RuntimeError('Unexpected one-image/one-class ROI result')
        return np.asarray(results[0][0], dtype=np.float32)


def loss_total(losses: Dict) -> torch.Tensor:
    terms = []
    for name, value in losses.items():
        # MMRotate loss dictionaries also contain metrics such as ``acc``.
        if 'loss' not in str(name).lower():
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        terms.extend(item.mean() for item in values
                     if isinstance(item, torch.Tensor))
    if not terms:
        raise RuntimeError('Detector heads returned no tensor losses')
    total = sum(terms)
    if not bool(torch.isfinite(total).item()):
        raise RuntimeError('Non-finite detector-head loss')
    return total


def loss_component_means(losses: Dict) -> Dict[str, float]:
    """Return scalar means for loss entries, excluding metrics."""
    components = {}
    for name, value in losses.items():
        if 'loss' not in str(name).lower():
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        tensors = [item for item in values if isinstance(item, torch.Tensor)]
        if tensors:
            component = sum(item.mean() for item in tensors)
            if not bool(torch.isfinite(component).item()):
                raise RuntimeError('Non-finite detector-head loss component')
            components[str(name)] = float(component.detach().item())
    return components


def prepare_record(dino, record: Dict, args, dino_device, head_device):
    feature_cpu, dino_meta, cached = extract_or_load_feature(
        dino, record, args, dino_device)
    feature = feature_cpu.to(device=head_device, dtype=torch.float32)
    img_meta = feature_meta(record['image'], dino_meta)
    gt_boxes, gt_labels, original = scaled_gt_tensors(
        record['annotation'], float(dino_meta['scale']), head_device)
    return feature, img_meta, gt_boxes, gt_labels, original, cached


def audit_s7_quality_training_support(
        dino, heads, records: Sequence[Dict], args,
        dino_device, head_device) -> Dict:
    """Run the exact quality miner without optimization or target access."""
    heads.eval()
    rows = []
    cache_hits = 0
    with torch.no_grad():
        for index, record in enumerate(records):
            feature, img_meta, gt_boxes, gt_labels, _original, cached = (
                prepare_record(
                    dino, record, args, dino_device, head_device))
            cache_hits += int(cached)
            output = heads.forward_s7_quality_suppression_train(
                feature, img_meta, gt_boxes,
                riou_thr=args.riou_thr,
                margin=args.s7_quality_margin,
                risk_weight=args.s7_quality_risk_weight,
                preserve_weight=args.s7_quality_preserve_weight,
                retention_weight=args.s7_quality_retention_weight,
                prior_weight=args.s7_quality_prior_weight)
            rows.append(dict(
                frame_key='{}|{}|{}'.format(
                    record.get('split', ''), record.get('seq', ''),
                    int(record.get('frame', -1))),
                split=str(record.get('split', '')),
                seq=str(record.get('seq', '')),
                frame=int(record.get('frame', -1)),
                risk_pair=bool(output['s7_quality_risk_pair_count']),
                preserve_pair=bool(
                    output['s7_quality_preserve_pair_count']),
                native_top1_correct=bool(
                    output['s7_quality_native_top1_correct']),
                s7_top1_correct=bool(
                    output['s7_quality_s7_top1_correct']),
                native_top1_riou=float(
                    output['s7_quality_native_top1_riou']),
                s7_top1_riou=float(output['s7_quality_s7_top1_riou']),
                base_gap=float(output['s7_quality_base_gap']),
                s7_candidate_count=int(
                    output['s7_quality_candidate_count'])))
            if (index + 1) % 100 == 0 or index + 1 == len(records):
                print('[s7-quality-preflight] {}/{} cache={}/{}'.format(
                    index + 1, len(records), cache_hits, index + 1))
            del feature, gt_boxes, gt_labels, output
    summary = summarize_s7_quality_support_rows(
        rows, margin=args.s7_quality_margin, riou_thr=args.riou_thr)
    summary['cache_hits'] = int(cache_hits)
    print('[s7-quality-support] status={} risk_pairs={} preserve_pairs={} '
          's7_wrong={} native_correct_s7_wrong={} excluded_by_margin={}'
          .format(
              summary['status'], summary['risk_pair_count'],
              summary['preserve_pair_count'],
              summary['s7_top1_wrong_count'],
              summary['native_correct_s7_wrong_count'],
              summary[
                  'native_correct_s7_wrong_excluded_by_margin_count']))
    return summary


def scheduled_lr(args, epoch: int, global_step: int) -> float:
    decay_count = sum(int(epoch) > int(step) for step in args.lr_steps)
    regular_lr = float(args.lr) * (float(args.lr_gamma) ** decay_count)
    if args.warmup_iters <= 0 or global_step >= args.warmup_iters:
        return regular_lr
    progress = float(global_step) / float(args.warmup_iters)
    warmup_factor = (float(args.warmup_ratio)
                     + (1.0 - float(args.warmup_ratio)) * progress)
    return regular_lr * warmup_factor


def train_epoch(dino, heads, optimizer, records: Sequence[Dict], epoch: int,
                global_step: int, args, dino_device, head_device) -> Dict:
    heads.train()
    if args.train_components in (
            's7_rpn', 's7_merge', 's7_lane_arbitration',
            's7_quality_suppression', 's7_temporal_association'):
        heads.rpn_head.eval()
        heads.roi_head.eval()
    if args.train_components in (
            's7_merge', 's7_lane_arbitration', 's7_quality_suppression',
            's7_temporal_association'):
        heads.s7_readout.eval()
        heads.s7_rpn_head.eval()
        heads.s7_score_calibrator.eval()
    if args.train_components == 's7_lane_arbitration':
        heads.s7_lane_arbitrator.train()
    elif args.train_components == 's7_merge':
        heads.s7_score_calibrator.train()
    elif args.train_components == 's7_quality_suppression':
        heads.s7_quality_suppressor.train()
    elif args.train_components == 's7_temporal_association':
        if bool(getattr(args, 's7_temporal_quality_head', False)):
            heads.s7_temporal_scorer.eval()
            heads.s7_candidate_quality_head.train()
        else:
            heads.s7_temporal_scorer.train()
    if head_device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(head_device)
    ordered = list(records)
    if args.train_components == 's7_temporal_association':
        ordered = sorted(
            ordered, key=lambda row: (
                str(row.get('split', '')), str(row.get('seq', '')),
                int(row.get('frame', -1))))
    else:
        random.Random(args.seed + epoch).shuffle(ordered)
    losses = []
    component_sums = {}
    metric_sums = {}
    cache_hits = 0
    gain_replayed_keys = set()
    gain_replay_extra_count = 0
    temporal_previous_box = None
    temporal_previous_embedding = None
    temporal_previous_key = None
    for index, record in enumerate(ordered):
        current_lr = scheduled_lr(args, epoch, global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        feature, img_meta, gt_boxes, gt_labels, _original, cached = (
            prepare_record(
                dino, record, args, dino_device, head_device))
        cache_hits += int(cached)
        optimizer.zero_grad()
        if args.train_components == 's7_temporal_association':
            current_key = (
                str(record.get('split', '')), str(record.get('seq', '')),
                int(record.get('frame', -1)))
            if (temporal_previous_key is None
                    or current_key[:2] != temporal_previous_key[:2]
                    or current_key[2] != temporal_previous_key[2] + 1):
                temporal_previous_box = None
                temporal_previous_embedding = None
        if args.train_components == 's7_rpn':
            output = heads.forward_s7_rpn_train(
                feature, img_meta, gt_boxes)
        elif args.train_components == 's7_merge':
            output = heads.forward_s7_merge_train(
                feature, img_meta, gt_boxes,
                riou_thr=args.riou_thr,
                margin=args.s7_merge_margin,
                retention_weight=args.s7_merge_retention_weight,
                gain_weight=args.s7_merge_gain_weight,
                prior_weight=args.s7_merge_prior_weight)
        elif args.train_components == 's7_lane_arbitration':
            output = heads.forward_s7_lane_arbitration_train(
                feature, img_meta, gt_boxes,
                riou_thr=args.riou_thr,
                margin=args.s7_merge_margin,
                retention_weight=args.s7_merge_retention_weight,
                gain_weight=args.s7_merge_gain_weight,
                prior_weight=args.s7_merge_prior_weight,
                hard_negatives=args.s7_lane_hard_negatives)
        elif args.train_components == 's7_quality_suppression':
            output = heads.forward_s7_quality_suppression_train(
                feature, img_meta, gt_boxes,
                riou_thr=args.riou_thr,
                margin=args.s7_quality_margin,
                risk_weight=args.s7_quality_risk_weight,
                preserve_weight=args.s7_quality_preserve_weight,
                retention_weight=args.s7_quality_retention_weight,
                prior_weight=args.s7_quality_prior_weight)
        elif args.train_components == 's7_temporal_association':
            if bool(getattr(args, 's7_temporal_quality_head', False)):
                output = heads.forward_s7_temporal_quality_train(
                    feature, img_meta, gt_boxes,
                    riou_thr=args.riou_thr,
                    quality_weight=args.s7_temporal_quality_loss_weight,
                    max_candidates=args.s7_temporal_max_candidates,
                    relative_weight=(
                        args.s7_temporal_relative_quality_weight
                        if bool(getattr(
                            args, 's7_temporal_relative_quality', False))
                        else 0.0),
                    relative_margin=args.s7_temporal_relative_quality_margin,
                    relative_min_gap=args.s7_temporal_relative_quality_min_gap,
                    relative_max_pairs=(
                        args.s7_temporal_relative_quality_max_pairs))
            else:
                output = heads.forward_s7_temporal_association_train(
                    feature, img_meta, gt_boxes,
                    previous_box=temporal_previous_box,
                    previous_embedding=temporal_previous_embedding,
                    riou_thr=args.riou_thr,
                    margin=args.s7_temporal_margin,
                    retention_weight=args.s7_temporal_retention_weight,
                    gain_weight=args.s7_temporal_gain_weight,
                    prior_weight=args.s7_temporal_prior_weight,
                    max_candidates=args.s7_temporal_max_candidates)
        elif args.train_components == 'roi_cls':
            output = heads.forward_roi_cls_hard_train(
                feature, img_meta, gt_boxes, args.roi_samples,
                riou_thr=args.riou_thr)
        elif args.train_components == 'roi_cls_pairwise':
            output = heads.forward_roi_cls_pairwise_train(
                feature, img_meta, gt_boxes, args.roi_samples,
                riou_thr=args.riou_thr,
                negative_riou_thr=args.pairwise_negative_riou_thr,
                nms_iou_thr=args.pairwise_nms_iou_thr,
                pairwise_margin=args.pairwise_margin,
                pairwise_loss_weight=args.pairwise_loss_weight,
                retention_loss_weight=args.retention_loss_weight,
                retention_temperature=args.retention_temperature)
        elif args.train_components == 'roi_cls_pairwise_v2':
            output = heads.forward_roi_cls_pairwise_v2_train(
                feature, img_meta, gt_boxes, args.roi_samples,
                riou_thr=args.riou_thr,
                nms_iou_thr=args.pairwise_nms_iou_thr,
                pairwise_margin=args.pairwise_margin,
                cls_loss_weight=args.pairwise_cls_loss_weight,
                pairwise_loss_weight=args.pairwise_loss_weight,
                retention_loss_weight=args.retention_loss_weight,
                retention_temperature=args.retention_temperature,
                negatives_per_positive=(
                    args.pairwise_negatives_per_positive))
        else:
            output = heads.forward_train(
                feature, img_meta, gt_boxes, gt_labels)
        if (args.train_components == 's7_temporal_association'
                and not bool(getattr(
                    args, 's7_temporal_quality_head', False))):
            teacher_usable = bool(output.pop('_temporal_teacher_usable'))
            temporal_previous_box = output.pop('_temporal_teacher_box')
            temporal_previous_embedding = output.pop(
                '_temporal_teacher_embedding')
            if not teacher_usable:
                temporal_previous_box = None
                temporal_previous_embedding = None
            temporal_previous_key = current_key
        if (args.train_components == 's7_lane_arbitration'
                and int(output.get('s7_lane_gain_pair_count', 0)) > 0):
            replay_key = (
                str(record.get('split', '')), str(record.get('seq', '')),
                int(record.get('frame', -1)))
            if replay_key not in gain_replayed_keys:
                gain_replayed_keys.add(replay_key)
                extra = int(args.s7_lane_gain_repeat) - 1
                if extra > 0:
                    ordered.extend([record] * extra)
                    gain_replay_extra_count += extra
        for name, value in loss_component_means(output).items():
            component_sums[name] = component_sums.get(name, 0.0) + value
        for name, value in output.items():
            if name in component_sums or 'loss' in str(name).lower():
                continue
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = float(value.detach().item())
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
        total = optimization_loss_total(output, args.train_components)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            heads.parameters(), args.max_grad_norm)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError('Non-finite detector-head gradient')
        optimizer.step()
        global_step += 1
        losses.append(float(total.item()))
        if (index + 1) % 25 == 0 or index + 1 == len(ordered):
            message = (
                '[source-train] epoch={} {}/{} loss={:.5f} cache={}/{}'
                .format(
                    epoch, index + 1, len(ordered),
                    float(np.mean(losses[-25:])), cache_hits, index + 1))
            if args.train_components == 'roi_cls_pairwise_v2':
                message += (
                    ' pair_count_total={} '
                    'failure_frames_total={}').format(
                    int(round(metric_sums.get('roi_pair_count', 0.0))),
                    int(round(metric_sums.get(
                        'roi_pairwise_failure_frame', 0.0))))
            elif args.train_components == 's7_merge':
                message += (
                    ' retain_pairs_total={} gain_pairs_total={} '
                    'retain_active_total={} gain_active_total={}').format(
                    int(round(metric_sums.get(
                        's7_merge_retain_pair_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_merge_gain_pair_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_merge_retention_active', 0.0))),
                    int(round(metric_sums.get(
                        's7_merge_gain_active', 0.0))))
            elif args.train_components == 's7_lane_arbitration':
                message += (
                    ' retain_pairs_total={} gain_pairs_total={} '
                    'retain_active_total={} gain_active_total={}').format(
                    int(round(metric_sums.get(
                        's7_lane_retain_pair_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_lane_gain_pair_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_lane_retention_active', 0.0))),
                    int(round(metric_sums.get(
                        's7_lane_gain_active', 0.0))))
            elif args.train_components == 's7_quality_suppression':
                message += (
                    ' risk_pairs_total={} preserve_pairs_total={} '
                    'retain_active_total={} mean_delta={:.5f}').format(
                    int(round(metric_sums.get(
                        's7_quality_risk_pair_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_quality_preserve_pair_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_quality_retention_active', 0.0))),
                    float(metric_sums.get('s7_quality_delta', 0.0))
                    / float(max(1, index + 1)))
            elif args.train_components == 's7_temporal_association':
                if bool(getattr(args, 's7_temporal_quality_head', False)):
                    message += (
                        ' quality_candidates={} usable={} mean_target={:.4f}').format(
                        int(round(metric_sums.get(
                            's7_candidate_quality_count', 0.0))),
                        int(round(metric_sums.get(
                            's7_candidate_quality_usable_count', 0.0))),
                        float(metric_sums.get(
                            's7_candidate_quality_mean_target', 0.0))
                        / float(max(1, index + 1)))
                    if bool(getattr(
                            args, 's7_temporal_relative_quality', False)):
                        message += (
                            ' relative_pairs={} active={} acc={:.4f}').format(
                            int(round(metric_sums.get(
                                's7_candidate_quality_relative_pair_count',
                                0.0))),
                            int(round(metric_sums.get(
                                's7_candidate_quality_relative_active_count',
                                0.0))),
                            float(metric_sums.get(
                                's7_candidate_quality_relative_accuracy', 0.0))
                            / float(max(1, index + 1)))
                else:
                    message += (
                        ' retain_pairs_total={} gain_pairs_total={} '
                        'weights={}').format(
                        int(round(metric_sums.get(
                            's7_temporal_retention_pair_count', 0.0))),
                        int(round(metric_sums.get(
                            's7_temporal_gain_pair_count', 0.0))),
                        heads.s7_temporal_scorer.state_summary())
            print(message)
        del feature, gt_boxes, gt_labels, total
    optimized_components = optimization_loss_component_names(
        args.train_components,
        quality_head=bool(getattr(args, 's7_temporal_quality_head', False)),
        relative_quality=bool(getattr(
            args, 's7_temporal_relative_quality', False)))
    summary = dict(
        epoch=int(epoch), count=len(ordered),
        global_step_end=int(global_step),
        lr_end=float(optimizer.param_groups[0]['lr']),
        mean_loss=float(np.mean(losses)),
        optimized_components=optimized_components,
        mean_loss_components={
            name: float(value / max(1, len(ordered)))
            for name, value in sorted(component_sums.items())},
        mean_training_metrics={
            name: float(value / max(1, len(ordered)))
            for name, value in sorted(metric_sums.items())},
        cache_hits=int(cache_hits),
        head_peak_memory_mb=(
            float(torch.cuda.max_memory_allocated(head_device) / (1024 ** 2))
            if head_device.type == 'cuda' else 0.0))
    if args.train_components == 's7_lane_arbitration':
        summary['s7_lane_gain_replay'] = dict(
            repeat=int(args.s7_lane_gain_repeat),
            unique_gain_frame_count=len(gain_replayed_keys),
            extra_record_count=int(gain_replay_extra_count),
            source_train_only=True)
    return summary


def rotated_box_corners(detections: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h, angle] OBBs to four xy corners."""
    boxes = np.asarray(detections, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 5:
        raise ValueError('Boxes must have shape [N,>=5]')
    if boxes.shape[0] == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    cx, cy = boxes[:, 0], boxes[:, 1]
    half_w, half_h = boxes[:, 2] * 0.5, boxes[:, 3] * 0.5
    angle = boxes[:, 4]
    local = np.stack([
        np.stack([-half_w, -half_h], axis=1),
        np.stack([half_w, -half_h], axis=1),
        np.stack([half_w, half_h], axis=1),
        np.stack([-half_w, half_h], axis=1),
    ], axis=1)
    cos_angle = np.cos(angle)[:, None]
    sin_angle = np.sin(angle)[:, None]
    rotated_x = local[:, :, 0] * cos_angle - local[:, :, 1] * sin_angle
    rotated_y = local[:, :, 0] * sin_angle + local[:, :, 1] * cos_angle
    return np.stack([
        rotated_x + cx[:, None], rotated_y + cy[:, None]], axis=2
    ).astype(np.float32, copy=False)


def valid_rotated_detection_mask(detections: np.ndarray, img_meta: Dict,
                                 tolerance: float = 1e-3) -> np.ndarray:
    """Return the label-free original-image content mask for aligned metadata."""
    array = np.asarray(detections, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError('Detections must have shape [N,6]')
    shape = img_meta.get('ori_shape')
    if shape is None or len(shape) < 2:
        raise ValueError('img_meta must contain ori_shape')
    height, width = int(shape[0]), int(shape[1])
    corners = rotated_box_corners(array[:, :5])
    if corners.shape[0] == 0:
        keep = np.zeros((0,), dtype=bool)
    else:
        x = corners[:, :, 0]
        y = corners[:, :, 1]
        keep = ((x >= -float(tolerance)).all(axis=1)
                & (x <= float(width) + float(tolerance)).all(axis=1)
                & (y >= -float(tolerance)).all(axis=1)
                & (y <= float(height) + float(tolerance)).all(axis=1))
    return keep


def filter_valid_rotated_detections(detections: np.ndarray,
                                    img_meta: Dict,
                                    tolerance: float = 1e-3):
    """Filter OBBs with corners outside the original image.

    The rule is label-free and is applied identically to source validation and
    target diagnosis.  Remaining detections keep their original score order.
    """
    array = np.asarray(detections, dtype=np.float32)
    keep = valid_rotated_detection_mask(array, img_meta, tolerance)
    filtered = array[keep]
    stats = dict(
        raw_detection_count=int(array.shape[0]),
        invalid_border_filtered_count=int((~keep).sum()),
        valid_detection_count=int(filtered.shape[0]))
    return filtered, stats


def gt_border_metrics(gt_original: np.ndarray, img_meta: Dict,
                      margin_ratio: float = 0.02,
                      tolerance: float = 1e-3) -> Dict:
    """Describe GT proximity to image borders without changing detections."""
    shape = img_meta.get('ori_shape')
    if shape is None or len(shape) < 2:
        raise ValueError('img_meta must contain ori_shape')
    height, width = int(shape[0]), int(shape[1])
    margin = float(min(height, width)) * float(margin_ratio)
    if gt_original.shape[0] == 0:
        return dict(
            gt_count=0, gt_all_corners_inside=None,
            gt_min_border_distance_px=None, gt_near_border=None,
            gt_border_margin_px=margin)
    corners = rotated_box_corners(gt_original)
    x, y = corners[:, :, 0], corners[:, :, 1]
    distances = np.stack([x, float(width) - x,
                          y, float(height) - y], axis=2)
    minimum = float(distances.min())
    return dict(
        gt_count=int(gt_original.shape[0]),
        gt_all_corners_inside=bool(minimum >= -float(tolerance)),
        gt_min_border_distance_px=minimum,
        gt_near_border=bool(minimum <= margin),
        gt_border_margin_px=margin)


def ranked_detection_metrics(detections: np.ndarray, gt_original: np.ndarray,
                             riou_thr: float,
                             deployment_score_thr: float = 0.05) -> Dict:
    from mmcv.ops import box_iou_rotated

    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError('Detections must have shape [N,6]')
    if detections.shape[0] == 0:
        return dict(
            detection_count=0, top1_riou=0.0,
            top1_hit=False, best_usable_rank=None, best_riou=0.0,
            top1_score=None, deployment_score_thr=float(
                deployment_score_thr), deployment_top1_hit=False,
            deployment_silence=True)
    if gt_original.shape[0] == 0:
        top1_score = float(detections[0, 5])
        return dict(
            detection_count=int(detections.shape[0]), top1_riou=0.0,
            top1_hit=False, best_usable_rank=None, best_riou=0.0,
            top1_score=top1_score,
            deployment_score_thr=float(deployment_score_thr),
            deployment_top1_hit=False,
            deployment_silence=bool(
                top1_score < float(deployment_score_thr)))
    boxes = torch.from_numpy(detections[:, :5]).float()
    gt = torch.from_numpy(gt_original).float()
    ious = box_iou_rotated(boxes, gt).max(dim=1).values
    best_rank = None
    for rank, value in enumerate(ious.tolist(), start=1):
        if float(value) >= float(riou_thr):
            best_rank = int(rank)
            break
    top1_score = float(detections[0, 5])
    top1_hit = bool(float(ious[0].item()) >= float(riou_thr))
    return dict(
        detection_count=int(detections.shape[0]),
        top1_riou=float(ious[0].item()),
        top1_hit=top1_hit,
        best_usable_rank=best_rank,
        best_riou=float(ious.max().item()),
        top1_score=top1_score,
        deployment_score_thr=float(deployment_score_thr),
        deployment_top1_hit=bool(
            top1_hit and top1_score >= float(deployment_score_thr)),
        deployment_silence=bool(
            top1_score < float(deployment_score_thr)))


def _temporal_candidate_evidence(
        detections: np.ndarray, source_ids: np.ndarray,
        index: Optional[int], original: np.ndarray, args) -> Optional[Dict]:
    """Return compact one-candidate source-GT evidence for a fixed index."""
    if index is None:
        return None
    index = int(index)
    if index < 0 or index >= int(detections.shape[0]):
        raise RuntimeError('Temporal attribution candidate index is invalid')
    metrics = ranked_detection_metrics(
        detections[index:index + 1], original, args.riou_thr,
        args.deployment_score_thr)
    return dict(
        index=index,
        source=('native_s14' if int(source_ids[index]) == 0
                else 'supplement_s7'),
        top1_hit=bool(metrics['top1_hit']),
        top1_riou=float(metrics['top1_riou']),
        top1_score=metrics['top1_score'])


def temporal_selection_attribution(
        pool_detections: torch.Tensor, source_ids: torch.Tensor,
        quality_logits: Optional[torch.Tensor], valid_mask: np.ndarray,
        selection: Dict, original: np.ndarray, args) -> Dict:
    """Audit selector stages without changing its output or temporal state."""
    detections = pool_detections.detach().cpu().numpy().astype(
        np.float32, copy=False)
    sources = source_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if (detections.shape[0] != sources.shape[0]
            or valid.shape[0] != sources.shape[0]):
        raise RuntimeError('Temporal attribution candidate metadata disagrees')
    bounded = np.arange(detections.shape[0]) < int(
        getattr(args, 's7_temporal_max_candidates', 100))
    eligible = valid & bounded
    eligible_indices = np.flatnonzero(eligible)
    native_indices = np.flatnonzero(eligible & (sources == 0))
    fallback_index = (
        int(native_indices[0]) if native_indices.size else
        int(eligible_indices[0]) if eligible_indices.size else None)
    selected_index = selection.get('selected_index')
    candidate_index = selection.get('candidate_index')
    margin_index = (
        candidate_index if selection.get('candidate_margin_ok', False)
        else fallback_index)
    preconfirmation_index = (
        candidate_index if selection.get('candidate_override_ok', False)
        else fallback_index)

    quality_ranked = None
    quality_top_index = None
    if quality_logits is not None and eligible_indices.size:
        quality = quality_logits.detach().cpu().numpy().reshape(-1)
        if quality.shape[0] != detections.shape[0]:
            raise RuntimeError('Temporal attribution quality logits disagree')
        quality_order = eligible_indices[np.argsort(
            -quality[eligible_indices], kind='stable')]
        quality_top_index = int(quality_order[0])
        quality_metrics = ranked_detection_metrics(
            detections[quality_order], original, args.riou_thr,
            args.deployment_score_thr)
        quality_ranked = dict(
            top1_hit=bool(quality_metrics['top1_hit']),
            best_usable_rank=quality_metrics['best_usable_rank'],
            recall_at_20=bool(
                quality_metrics['best_usable_rank'] is not None
                and quality_metrics['best_usable_rank'] <= 20),
            recall_at_100=bool(
                quality_metrics['best_usable_rank'] is not None
                and quality_metrics['best_usable_rank'] <= 100))

    return dict(
        read_only=True,
        eligible_candidate_count=int(eligible_indices.size),
        fallback=_temporal_candidate_evidence(
            detections, sources, fallback_index, original, args),
        quality_only=_temporal_candidate_evidence(
            detections, sources, quality_top_index, original, args),
        quality_ranked=quality_ranked,
        fused_candidate=_temporal_candidate_evidence(
            detections, sources, candidate_index, original, args),
        margin_counterfactual=_temporal_candidate_evidence(
            detections, sources, margin_index, original, args),
        preconfirmation_counterfactual=_temporal_candidate_evidence(
            detections, sources, preconfirmation_index, original, args),
        final_selected=_temporal_candidate_evidence(
            detections, sources, selected_index, original, args),
        candidate_margin_ok=bool(selection.get(
            'candidate_margin_ok', False)),
        candidate_continuity_ok=bool(selection.get(
            'candidate_continuity_ok', False)),
        candidate_override_ok=bool(selection.get(
            'candidate_override_ok', False)),
        pending_confirmation=bool(
            selection.get('reason') ==
            'native_fallback_pending_confirmation'))


def temporal_runtime_min_confirmations(args) -> int:
    """Return the inference confirmation count for the explicit audit mode."""
    if bool(getattr(
            args, 'source_temporal_immediate_override_audit', False)):
        return 1
    return int(getattr(args, 's7_temporal_min_confirmations', 2))


def evaluate_records(dino, heads, records: Sequence[Dict], args,
                     dino_device, head_device, role: str):
    heads.eval()
    rows = []
    temporal_selector = None
    temporal_scorer = getattr(heads, 's7_temporal_scorer', None)
    if (bool(getattr(heads, 's7_temporal_association', False))
            and temporal_scorer is not None
            and heads.s7_inference_enabled()):
        temporal_selector = temporal.CausalTemporalCandidateSelector(
            temporal_scorer,
            max_candidates=int(getattr(
                args, 's7_temporal_max_candidates', 100)),
            min_confirmations=temporal_runtime_min_confirmations(args),
            override_margin=float(getattr(
                args, 's7_temporal_override_margin', 0.25)),
            max_center_distance=float(getattr(
                args, 's7_temporal_max_center_distance', 3.0)),
            min_rotated_iou=float(getattr(
                args, 's7_temporal_min_riou', 0.05)),
            min_appearance_similarity=float(getattr(
                args, 's7_temporal_min_appearance', 0.20)))
    with torch.no_grad():
        for index, record in enumerate(records):
            feature, img_meta, _gt_boxes, _gt_labels, original, cached = (
                prepare_record(
                    dino, record, args, dino_device, head_device))
            raw_detections = heads.simple_test(feature, img_meta)
            candidate_merge = heads._last_candidate_merge
            temporal_selection = None
            temporal_attribution = None
            temporal_pool = getattr(heads, '_last_temporal_pool', None)
            if temporal_selector is not None:
                if temporal_pool is None:
                    raise RuntimeError(
                        'Temporal association is enabled but no candidate '
                        'pool was produced')
                pool_detections = temporal_pool['detections']
                if pool_detections.shape[0] != raw_detections.shape[0]:
                    raise RuntimeError(
                        'Temporal candidate pool and detections disagree')
                valid_mask_np = valid_rotated_detection_mask(
                    raw_detections, img_meta, args.valid_content_tolerance)
                temporal_selection = temporal_selector.select(
                    pool_detections, temporal_pool['embeddings'],
                    temporal_pool['source_ids'], str(record['seq']),
                    int(record['frame']),
                    valid_mask=torch.as_tensor(
                        valid_mask_np, dtype=torch.bool,
                        device=pool_detections.device),
                    quality_logits=temporal_pool.get('quality_logits'))
                if bool(getattr(
                        args, 'source_temporal_attribution_audit', False)
                        or getattr(
                            args, 'source_temporal_immediate_override_audit',
                            False)):
                    if role != 'source_validation':
                        raise RuntimeError(
                            'Temporal attribution is restricted to source '
                            'validation')
                    temporal_attribution = temporal_selection_attribution(
                        pool_detections, temporal_pool['source_ids'],
                        temporal_pool.get('quality_logits'), valid_mask_np,
                        temporal_selection, original, args)
                order = temporal_selection.pop('order').detach().cpu().numpy()
                raw_detections = raw_detections[order]
            if candidate_merge is not None:
                candidate_merge = dict(candidate_merge)
                source_metrics = {}
                for source, detection in candidate_merge.get(
                        'source_top1_detections', {}).items():
                    source_detection = np.asarray(
                        [] if detection is None else [detection],
                        dtype=np.float32).reshape((-1, 6))
                    lane_metrics = ranked_detection_metrics(
                        source_detection, original, args.riou_thr,
                        args.deployment_score_thr)
                    source_metrics[source] = dict(
                        top1_hit=bool(lane_metrics['top1_hit']),
                        top1_riou=float(lane_metrics['top1_riou']),
                        top1_score=lane_metrics['top1_score'])
                candidate_merge['source_top1_metrics'] = source_metrics
                candidate_merge['temporal_selection'] = temporal_selection
            raw_metrics = ranked_detection_metrics(
                raw_detections, original, args.riou_thr,
                args.deployment_score_thr)
            detections, filter_stats = filter_valid_rotated_detections(
                raw_detections, img_meta, args.valid_content_tolerance)
            metrics = ranked_detection_metrics(
                detections, original, args.riou_thr,
                args.deployment_score_thr)
            metrics.update(filter_stats)
            metrics['raw_unfiltered'] = raw_metrics
            metrics['filter_effect'] = dict(
                removed_usable_geometry=bool(
                    raw_metrics['best_usable_rank'] is not None
                    and metrics['best_usable_rank'] is None),
                promoted_to_top1=bool(
                    not raw_metrics['top1_hit'] and metrics['top1_hit']),
                demoted_from_top1=bool(
                    raw_metrics['top1_hit'] and not metrics['top1_hit']),
                raw_best_usable_rank=raw_metrics['best_usable_rank'],
                filtered_best_usable_rank=metrics['best_usable_rank'])
            metrics.update(gt_border_metrics(
                original, img_meta, args.border_margin_ratio,
                args.valid_content_tolerance))
            rows.append(dict(
                role=role, split=record['split'], seq=record['seq'],
                frame=int(record['frame']), feature_cache_hit=bool(cached),
                candidate_merge=candidate_merge,
                temporal_selection=temporal_selection,
                temporal_attribution=temporal_attribution,
                metrics=metrics,
                detections=[[float(value) for value in detection]
                            for detection in detections.tolist()]))
            if role == 'target_dev_diagnosis_only':
                print('[target-labeller] frame={} top1={} rank={} raw_rank={}'
                      .format(
                          record['frame'], metrics['top1_hit'],
                          metrics['best_usable_rank'],
                          raw_metrics['best_usable_rank']))
            elif (role == 'source_validation'
                  and ((index + 1) % 25 == 0
                       or index + 1 == len(records))):
                print('[source-val] {}/{} top1_hits={}'.format(
                    index + 1, len(records),
                    sum(row['metrics']['top1_hit'] for row in rows)))
            elif (role == 'target_holdout_readonly'
                  and ((index + 1) % 25 == 0
                       or index + 1 == len(records))):
                print('[target-holdout] seq={} {}/{} top1_hits={}'.format(
                    record['seq'], index + 1, len(records),
                    sum(row['metrics']['top1_hit'] for row in rows)))
            elif (role == 'target_full_test_readonly'
                  and ((index + 1) % 25 == 0
                       or index + 1 == len(records))):
                print('[full-test-dino] {}/{} seq={} frame={}'.format(
                    index + 1, len(records), record['seq'], record['frame']))
            del feature, _gt_boxes, _gt_labels
    return rows


def longest_miss(rows: Sequence[Dict], hit_key: str) -> int:
    longest = 0
    current = 0
    previous_frame = None
    previous_seq = None
    for row in rows:
        frame = int(row['frame'])
        seq = row.get('seq')
        if (previous_frame is None or frame != previous_frame + 1
                or seq != previous_seq):
            current = 0
        if row[hit_key]:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
        previous_frame = frame
        previous_seq = seq
    return int(longest)


def top1_temporal_geometry_metrics(
        rows: Sequence[Dict], angle_limit_deg: float = 35.0) -> Dict:
    """Match the project DFR/ACI definitions on consecutive selected boxes."""
    limit = math.radians(float(angle_limit_deg))
    if limit <= 0.0:
        raise ValueError('Temporal angle limit must be positive')
    dfr_values = []
    aci_values = []
    previous_box = None
    previous_seq = None
    previous_frame = None
    for row in rows:
        seq = row.get('seq')
        frame = int(row['frame'])
        detections = np.asarray(row.get('detections', []), dtype=np.float64)
        if detections.size == 0:
            previous_box = None
            previous_seq = seq
            previous_frame = frame
            continue
        detections = detections.reshape((-1, 6))
        current = detections[0, :5]
        continuous = bool(
            previous_box is not None and previous_seq == seq
            and previous_frame is not None
            and frame == int(previous_frame) + 1)
        if continuous:
            previous_diag = float(np.linalg.norm(previous_box[2:4]))
            current_diag = float(np.linalg.norm(current[2:4]))
            if previous_diag > 1e-6:
                dfr_values.append(abs(current_diag - previous_diag)
                                  / previous_diag)
            delta = float(current[4] - previous_box[4])
            periodic = abs(0.5 * math.atan2(
                math.sin(2.0 * delta), math.cos(2.0 * delta)))
            aci_values.append(float(np.clip(
                1.0 - periodic / (limit + 1e-9), 0.0, 1.0)))
        previous_box = current.copy()
        previous_seq = seq
        previous_frame = frame
    return dict(
        dfr_fraction_per_frame=(
            float(np.mean(dfr_values)) if dfr_values else None),
        dfr_percent_per_frame=(
            float(np.mean(dfr_values) * 100.0) if dfr_values else None),
        aci=(float(np.mean(aci_values)) if aci_values else None),
        transition_count=int(len(aci_values)),
        dfr_transition_count=int(len(dfr_values)))


def summarize_temporal_readonly_attribution(
        rows: Sequence[Dict]) -> Optional[Dict]:
    """Aggregate fixed-checkpoint selector-stage source correctness."""
    attributed = [row for row in rows
                  if row.get('temporal_attribution') is not None]
    if not attributed:
        return None

    def stage_summary(name: str) -> Dict:
        evidence = [row['temporal_attribution'].get(name)
                    for row in attributed]
        comparable = [
            (row['temporal_attribution'].get('fallback'), candidate)
            for row, candidate in zip(attributed, evidence)
            if (row['temporal_attribution'].get('fallback') is not None
                and candidate is not None)]
        return dict(
            evaluated_count=int(sum(item is not None for item in evidence)),
            top1_hits=int(sum(bool(item.get('top1_hit', False))
                              for item in evidence if item is not None)),
            gained_vs_fallback_count=int(sum(
                not bool(fallback['top1_hit'])
                and bool(candidate['top1_hit'])
                for fallback, candidate in comparable)),
            lost_vs_fallback_count=int(sum(
                bool(fallback['top1_hit'])
                and not bool(candidate['top1_hit'])
                for fallback, candidate in comparable)))

    stages = {
        name: stage_summary(name) for name in (
            'fallback', 'quality_only', 'fused_candidate',
            'margin_counterfactual', 'preconfirmation_counterfactual',
            'final_selected')}
    for stage in stages.values():
        stage['net_gain_vs_fallback'] = int(
            stage['gained_vs_fallback_count']
            - stage['lost_vs_fallback_count'])

    pending = [row for row in attributed
               if row['temporal_attribution']['pending_confirmation']]
    quality_ranked = [row['temporal_attribution']['quality_ranked']
                      for row in attributed
                      if row['temporal_attribution']['quality_ranked']
                      is not None]

    def condition_summary(name: str) -> Dict:
        selected = [row['temporal_attribution'] for row in attributed
                    if row['temporal_attribution'][name]]
        candidates = [item['fused_candidate'] for item in selected
                      if item['fused_candidate'] is not None]
        comparable = [
            (item['fallback'], item['fused_candidate']) for item in selected
            if (item['fallback'] is not None
                and item['fused_candidate'] is not None)]
        return dict(
            frame_count=int(len(selected)),
            candidate_top1_hits=int(sum(
                bool(item['top1_hit']) for item in candidates)),
            candidate_gain_count=int(sum(
                not fallback['top1_hit'] and candidate['top1_hit']
                for fallback, candidate in comparable)),
            candidate_loss_count=int(sum(
                fallback['top1_hit'] and not candidate['top1_hit']
                for fallback, candidate in comparable)))

    usable_quality_ranks = [
        int(item['best_usable_rank']) for item in quality_ranked
        if item['best_usable_rank'] is not None]

    return dict(
        read_only=True,
        frame_count=int(len(attributed)),
        stages=stages,
        conditions=dict(
            margin_ok=condition_summary('candidate_margin_ok'),
            continuity_ok=condition_summary('candidate_continuity_ok'),
            margin_and_continuity_ok=condition_summary(
                'candidate_override_ok')),
        pending_confirmation=dict(
            frame_count=int(len(pending)),
            candidate_top1_hits=int(sum(bool(
                row['temporal_attribution']['fused_candidate']
                and row['temporal_attribution']['fused_candidate'][
                    'top1_hit']) for row in pending)),
            candidate_gain_count=int(sum(bool(
                row['temporal_attribution']['fallback']
                and not row['temporal_attribution']['fallback']['top1_hit']
                and row['temporal_attribution']['fused_candidate']
                and row['temporal_attribution']['fused_candidate'][
                    'top1_hit']) for row in pending)),
            candidate_loss_count=int(sum(bool(
                row['temporal_attribution']['fallback']
                and row['temporal_attribution']['fallback']['top1_hit']
                and row['temporal_attribution']['fused_candidate']
                and not row['temporal_attribution']['fused_candidate'][
                    'top1_hit']) for row in pending))),
        quality_ranked=dict(
            evaluated_count=int(len(quality_ranked)),
            top1_hits=int(sum(item['top1_hit'] for item in quality_ranked)),
            usable_candidate_count=int(len(usable_quality_ranks)),
            mean_best_usable_rank=(
                float(np.mean(usable_quality_ranks))
                if usable_quality_ranks else None),
            median_best_usable_rank=(
                float(np.median(usable_quality_ranks))
                if usable_quality_ranks else None),
            recall_at_20=int(sum(item['recall_at_20']
                                 for item in quality_ranked)),
            recall_at_100=int(sum(item['recall_at_100']
                                  for item in quality_ranked))))


def summarize_temporal_association_audit(rows: Sequence[Dict]) -> Optional[Dict]:
    """Summarize causal association opportunities without changing outputs.

    This is a source/target-neutral diagnostic. It reports which explicit
    selector condition blocked a candidate; the existing source gate remains
    the only checkpoint-selection authority.
    """
    temporal_rows = [row for row in rows
                     if row.get('temporal_selection') is not None]
    if not temporal_rows:
        return None

    def native_top1_hit(row):
        source_metrics = (row.get('candidate_merge') or {}).get(
            'source_top1_metrics', {})
        native = source_metrics.get('native_s14')
        return None if native is None else bool(native.get('top1_hit', False))

    native_known = [row for row in temporal_rows
                    if native_top1_hit(row) is not None]
    native_wrong = [row for row in native_known
                    if not native_top1_hit(row)]
    usable = [row for row in temporal_rows if row.get('metrics', {}).get(
        'raw_unfiltered', row.get('metrics', {})).get(
            'best_usable_rank') is not None]
    usable_ids = {id(row) for row in usable}
    candidate_rows = [row for row in temporal_rows
                      if row['temporal_selection'].get('candidate_index')
                      is not None]
    non_fallback = [row for row in candidate_rows
                    if row['temporal_selection'].get('candidate_index')
                    != row['temporal_selection'].get(
                        'native_fallback_index')]
    non_fallback_ids = {id(row) for row in non_fallback}
    reason_counts = {}
    for row in temporal_rows:
        reason = str(row['temporal_selection'].get('reason'))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    def count(predicate):
        return int(sum(bool(predicate(row['temporal_selection']))
                       for row in temporal_rows))

    return dict(
        frame_count=int(len(temporal_rows)),
        native_top1_known_count=int(len(native_known)),
        native_top1_correct_count=int(len(native_known) - len(native_wrong)),
        native_top1_wrong_count=int(len(native_wrong)),
        usable_candidate_count=int(len(usable)),
        usable_candidate_when_native_wrong_count=int(sum(
            id(row) in usable_ids for row in native_wrong)),
        fused_non_fallback_candidate_count=int(len(non_fallback)),
        fused_non_fallback_when_native_wrong_count=int(sum(
            id(row) in non_fallback_ids for row in native_wrong)),
        candidate_margin_ok_count=count(
            lambda selection: selection.get('candidate_margin_ok', False)),
        candidate_continuity_ok_count=count(
            lambda selection: selection.get('candidate_continuity_ok', False)),
        candidate_override_ok_count=count(
            lambda selection: selection.get('candidate_override_ok', False)),
        pending_confirmation_count=int(sum(
            row['temporal_selection'].get('reason') ==
            'native_fallback_pending_confirmation'
            for row in temporal_rows)),
        override_selected_count=int(sum(
            bool(row['temporal_selection'].get('override', False))
            for row in temporal_rows)),
        override_selected_s7_count=int(sum(
            bool(row['temporal_selection'].get('override', False))
            and row['temporal_selection'].get('selected_source') ==
            'supplement_s7' for row in temporal_rows)),
        reset_count=int(sum(
            bool(row['temporal_selection'].get('reset', False))
            for row in temporal_rows)),
        reason_counts=dict(sorted(reason_counts.items())),
        readonly_attribution=summarize_temporal_readonly_attribution(rows))


def build_source_temporal_attribution_audit(
        full_summary: Dict, small_summary: Dict, args,
        checkpoint: str, checkpoint_epoch: int) -> Dict:
    """Make the bounded stop/go decision for one rejected checkpoint."""
    if full_summary is None or small_summary is None:
        raise RuntimeError(
            'Temporal attribution requires full and source-small summaries')

    def extract(summary: Dict, label: str) -> Dict:
        association = summary.get('temporal_association_audit') or {}
        attribution = association.get('readonly_attribution')
        if attribution is None:
            raise RuntimeError(
                'Temporal attribution evidence is absent for {}'.format(
                    label))
        stages = attribution['stages']
        fallback = stages['fallback']
        preconfirmation = stages['preconfirmation_counterfactual']
        final = stages['final_selected']
        if (fallback['evaluated_count'] != int(summary['frame_count'])
                or preconfirmation['evaluated_count']
                != int(summary['frame_count'])
                or final['evaluated_count'] != int(summary['frame_count'])):
            raise RuntimeError(
                'Temporal attribution has incomplete {} frame coverage'.format(
                    label))
        return dict(
            frame_count=int(summary['frame_count']),
            fallback_top1_hits=int(fallback['top1_hits']),
            final_top1_hits=int(final['top1_hits']),
            preconfirmation_top1_hits=int(preconfirmation['top1_hits']),
            preconfirmation_gained_vs_fallback_count=int(
                preconfirmation['gained_vs_fallback_count']),
            preconfirmation_lost_vs_fallback_count=int(
                preconfirmation['lost_vs_fallback_count']),
            pending_confirmation=attribution['pending_confirmation'],
            stages=stages,
            conditions=attribution['conditions'],
            quality_ranked=attribution['quality_ranked'])

    full = extract(full_summary, 'full source validation')
    small = extract(small_summary, 'small source validation')
    full_minimum = int(getattr(args, 's7_source_min_full_top1', 688))
    small_minimum = int(getattr(args, 's7_source_min_small_top1', 311))
    checks = dict(
        full_preconfirmation_absolute=(
            full['preconfirmation_top1_hits'] >= full_minimum),
        small_preconfirmation_absolute=(
            small['preconfirmation_top1_hits'] >= small_minimum),
        full_preconfirmation_exact_retention=(
            full['preconfirmation_lost_vs_fallback_count'] == 0),
        small_preconfirmation_exact_retention=(
            small['preconfirmation_lost_vs_fallback_count'] == 0))
    confirmation_revision_supported = bool(all(checks.values()))
    return dict(
        mode='fixed_rejected_checkpoint_source_val_readonly_attribution',
        checkpoint=os.path.abspath(checkpoint),
        checkpoint_epoch=int(checkpoint_epoch),
        checkpoint_selected_for_deployment=False,
        parameter_update=False,
        checkpoint_selection=False,
        best_epoch_selection=False,
        target_read=False,
        fused_candidate_definition='argmax_of_seven_cues',
        preconfirmation_definition=(
            'one_step_candidate_if_margin_and_continuity_pass_else_fallback'),
        counterfactual_updates_temporal_state=False,
        full=full,
        small=small,
        required_absolute=dict(full=full_minimum, small=small_minimum),
        final_shortfall=dict(
            full=max(0, full_minimum - full['final_top1_hits']),
            small=max(0, small_minimum - small['final_top1_hits'])),
        preconfirmation_shortfall=dict(
            full=max(
                0, full_minimum - full['preconfirmation_top1_hits']),
            small=max(
                0, small_minimum - small['preconfirmation_top1_hits'])),
        checks=checks,
        confirmation_rule_revision_supported=(
            confirmation_revision_supported),
        recommendation=(
            'ALLOW_ONE_BOUNDED_CONFIRMATION_RULE_REVISION'
            if confirmation_revision_supported else
            'CLOSE_CURRENT_QUALITY_TEMPORAL_MERGE_KEEP_NATIVE_BASELINE'))


def summarize_rows(rows: Sequence[Dict]) -> Dict:
    flat = []
    for row in rows:
        metrics = row['metrics']
        raw_metrics = metrics.get('raw_unfiltered', metrics)
        flat.append(dict(
            seq=row.get('seq'), frame=int(row['frame']),
            top1=bool(metrics['top1_hit']),
            deployment_top1=bool(metrics.get('deployment_top1_hit', False)),
            r20=bool(metrics['best_usable_rank'] is not None
                     and metrics['best_usable_rank'] <= 20),
            r100=bool(metrics['best_usable_rank'] is not None
                      and metrics['best_usable_rank'] <= 100),
            geometry=bool(metrics['best_usable_rank'] is not None),
            raw_top1=bool(raw_metrics['top1_hit']),
            raw_r20=bool(raw_metrics['best_usable_rank'] is not None
                         and raw_metrics['best_usable_rank'] <= 20),
            raw_r100=bool(raw_metrics['best_usable_rank'] is not None
                          and raw_metrics['best_usable_rank'] <= 100),
            raw_geometry=bool(raw_metrics['best_usable_rank'] is not None),
            removed_geometry=bool(metrics.get(
                'filter_effect', {}).get('removed_usable_geometry', False)),
            promoted_top1=bool(metrics.get(
                'filter_effect', {}).get('promoted_to_top1', False)),
            demoted_top1=bool(metrics.get(
                'filter_effect', {}).get('demoted_from_top1', False)),
            near_border=metrics.get('gt_near_border')))
    top1_rious = [float(row['metrics']['top1_riou']) for row in rows]
    top1_scores = [float(row['metrics']['top1_score']) for row in rows
                   if row['metrics']['top1_score'] is not None]
    raw_top1_rious = [float(row['metrics'].get(
        'raw_unfiltered', row['metrics'])['top1_riou']) for row in rows]
    raw_top1_scores = [
        float(row['metrics'].get('raw_unfiltered', row['metrics'])[
            'top1_score']) for row in rows
        if row['metrics'].get('raw_unfiltered', row['metrics'])[
            'top1_score'] is not None]
    raw_count = sum(int(row['metrics'].get('raw_detection_count', 0))
                    for row in rows)
    border_filtered = sum(int(row['metrics'].get(
        'invalid_border_filtered_count', 0)) for row in rows)
    valid_count = sum(int(row['metrics'].get('valid_detection_count', 0))
                      for row in rows)
    near_border = [row for row in flat if row['near_border'] is True]
    merge_rows = [row['candidate_merge'] for row in rows
                  if row.get('candidate_merge') is not None]
    merge_top1_sources = {
        source: int(sum(
            row.get('raw_top1_source') == source for row in merge_rows))
        for source in ('native_s14', 'supplement_s7')}
    lane_adjustment_maxima = [
        float(row.get('s7_lane_adjustment_max', 0.0))
        for row in merge_rows]
    lane_adjustment_means = [
        float(row.get('s7_lane_adjustment_mean', 0.0))
        for row in merge_rows]
    quality_deltas = [
        float(row.get('s7_quality_delta', 0.0)) for row in merge_rows]
    quality_risks = [
        float(row.get('s7_quality_risk_probability', 0.0))
        for row in merge_rows]
    temporal_rows = [row['temporal_selection'] for row in rows
                     if row.get('temporal_selection') is not None]
    temporal_geometry = top1_temporal_geometry_metrics(
        rows, SOURCE_TEMPORAL_ANGLE_LIMIT_DEG)
    temporal_audit = summarize_temporal_association_audit(rows)
    return dict(
        frame_count=len(rows),
        top1_hits=int(sum(row['top1'] for row in flat)),
        top1_mcml=longest_miss(flat, 'top1'),
        deployment_top1_hits=int(sum(
            row['deployment_top1'] for row in flat)),
        deployment_top1_mcml=longest_miss(flat, 'deployment_top1'),
        deployment_silence_count=int(sum(
            row['metrics'].get('deployment_silence', False)
            for row in rows)),
        recall_at_20=int(sum(row['r20'] for row in flat)),
        recall_at_100=int(sum(row['r100'] for row in flat)),
        geometry_eligible_count=int(sum(row['geometry'] for row in flat)),
        raw_unfiltered_top1_hits=int(sum(row['raw_top1'] for row in flat)),
        raw_unfiltered_top1_mcml=longest_miss(flat, 'raw_top1'),
        raw_unfiltered_recall_at_20=int(sum(row['raw_r20'] for row in flat)),
        raw_unfiltered_recall_at_100=int(sum(
            row['raw_r100'] for row in flat)),
        raw_unfiltered_geometry_eligible_count=int(sum(
            row['raw_geometry'] for row in flat)),
        filter_removed_usable_geometry_count=int(sum(
            row['removed_geometry'] for row in flat)),
        filter_promoted_to_top1_count=int(sum(
            row['promoted_top1'] for row in flat)),
        filter_demoted_from_top1_count=int(sum(
            row['demoted_top1'] for row in flat)),
        raw_detection_count=int(raw_count),
        invalid_border_filtered_count=int(border_filtered),
        valid_detection_count=int(valid_count),
        invalid_border_filtered_fraction=(
            float(border_filtered / raw_count) if raw_count else 0.0),
        near_border_frame_count=len(near_border),
        near_border_top1_hits=int(sum(row['top1'] for row in near_border)),
        candidate_merge_frame_count=len(merge_rows),
        temporal_selection_frame_count=len(temporal_rows),
        temporal_override_count=int(sum(
            bool(row.get('override', False)) for row in temporal_rows)),
        temporal_reset_count=int(sum(
            bool(row.get('reset', False)) for row in temporal_rows)),
        temporal_selected_source_counts={
            source: int(sum(row.get('selected_source') == source
                            for row in temporal_rows))
            for source in ('native_s14', 'supplement_s7')},
        temporal_reason_counts={
            reason: int(sum(row.get('reason') == reason
                            for row in temporal_rows))
            for reason in sorted(set(
                str(row.get('reason')) for row in temporal_rows))},
        top1_dfr_fraction_per_frame=temporal_geometry[
            'dfr_fraction_per_frame'],
        top1_dfr_percent_per_frame=temporal_geometry[
            'dfr_percent_per_frame'],
        top1_aci=temporal_geometry['aci'],
        top1_temporal_transition_count=temporal_geometry[
            'transition_count'],
        temporal_association_audit=temporal_audit,
        raw_top1_source_counts=merge_top1_sources,
        mean_native_s14_proposal_count=(
            float(np.mean([
                row['proposal_source_counts']['native_s14']
                for row in merge_rows])) if merge_rows else 0.0),
        mean_supplement_s7_proposal_count=(
            float(np.mean([
                row['proposal_source_counts']['supplement_s7']
                for row in merge_rows])) if merge_rows else 0.0),
        s7_affine_scale=(
            float(merge_rows[0]['s7_affine_scale']) if merge_rows else None),
        s7_affine_bias=(
            float(merge_rows[0]['s7_affine_bias']) if merge_rows else None),
        s7_lane_adjustment_abs_max=(
            max(lane_adjustment_maxima) if lane_adjustment_maxima else 0.0),
        mean_s7_lane_adjustment=(
            float(np.mean(lane_adjustment_means))
            if lane_adjustment_means else 0.0),
        min_s7_quality_delta=(
            min(quality_deltas) if quality_deltas else 0.0),
        mean_s7_quality_delta=(
            float(np.mean(quality_deltas)) if quality_deltas else 0.0),
        max_s7_quality_delta=(
            max(quality_deltas) if quality_deltas else 0.0),
        mean_s7_quality_risk_probability=(
            float(np.mean(quality_risks)) if quality_risks else 0.0),
        median_top1_score=(float(np.median(top1_scores))
                           if top1_scores else None),
        raw_unfiltered_median_top1_score=(
            float(np.median(raw_top1_scores)) if raw_top1_scores else None),
        raw_unfiltered_mean_top1_riou=(
            float(np.mean(raw_top1_rious)) if raw_top1_rious else 0.0),
        mean_top1_riou=(float(np.mean(top1_rious))
                        if top1_rious else 0.0))


def source_selection_key(summary: Dict) -> Tuple:
    return (
        int(summary['top1_hits']), int(summary['recall_at_20']),
        int(summary['recall_at_100']), float(summary['mean_top1_riou']))


def roi_cls_selection_key(full_summary: Dict, small_summary: Dict) -> Tuple:
    return source_selection_key(small_summary) + source_selection_key(
        full_summary)


def source_frame_key(row: Dict) -> str:
    return '{}|{}|{}'.format(
        row.get('split', ''), row.get('seq', ''), int(row['frame']))


def source_correct_frame_keys(rows: Sequence[Dict]) -> List[str]:
    return sorted(
        source_frame_key(row) for row in rows
        if bool(row['metrics']['top1_hit']))


def load_source_conflict_spec(path: str, epoch: int) -> Dict:
    with open(path, 'r') as handle:
        payload = json.load(handle)
    history = payload.get('source', {}).get('history', [])
    matches = [row for row in history if int(row.get('epoch', -1)) == int(epoch)]
    if len(matches) != 1:
        raise ValueError(
            'Expected one source history row for epoch {}, found {}'.format(
                epoch, len(matches)))
    retention = matches[0].get('source_exact_retention') or {}
    lost = [str(key) for key in retention.get('lost_frame_keys', [])]
    gained = [str(key) for key in retention.get('gained_frame_keys', [])]
    keys = sorted(set(lost + gained))
    if not keys:
        raise ValueError(
            'Source conflict epoch {} has no lost/gained frames'.format(epoch))
    return dict(
        result_json=os.path.abspath(path), epoch=int(epoch),
        lost_frame_keys=sorted(lost), gained_frame_keys=sorted(gained),
        frame_keys=keys)


def source_top1_retention_summary(
        baseline_correct_keys: Sequence[str], candidate_rows: Sequence[Dict]
        ) -> Dict:
    """Measure retention of the exact source frames that were correct."""
    baseline = set(str(key) for key in baseline_correct_keys)
    candidate_correct = {
        source_frame_key(row) for row in candidate_rows
        if bool(row['metrics']['top1_hit'])}
    retained = baseline & candidate_correct
    lost = baseline - candidate_correct
    gained = candidate_correct - baseline
    return dict(
        baseline_correct_count=len(baseline),
        retained_correct_count=len(retained),
        lost_correct_count=len(lost),
        gained_correct_count=len(gained),
        candidate_correct_count=len(candidate_correct),
        lost_frame_keys=sorted(lost),
        gained_frame_keys=sorted(gained))


def s7_calibration_state(heads) -> Optional[Dict]:
    calibrator = getattr(heads, 's7_score_calibrator', None)
    if calibrator is None:
        return None
    return dict(
        scale=float(calibrator.scale().detach().item()),
        bias=float(calibrator.bias.detach().item()),
        prior_loss=float(calibrator.prior_loss().detach().item()))


def source_merge_conflict_summary(
        baseline_correct_keys: Sequence[str], candidate_rows: Sequence[Dict]
        ) -> Dict:
    """Keep compact source-only evidence for frames changed by S7 merge."""
    retention = source_top1_retention_summary(
        baseline_correct_keys, candidate_rows)
    changed_keys = set(
        retention['lost_frame_keys'] + retention['gained_frame_keys'])

    rows_by_key = {
        source_frame_key(row): row for row in candidate_rows
        if source_frame_key(row) in changed_keys}
    return dict(
        lost=[source_merge_conflict_row(rows_by_key[key], 'lost')
              for key in retention['lost_frame_keys']],
        gained=[source_merge_conflict_row(rows_by_key[key], 'gained')
                for key in retention['gained_frame_keys']])


def source_merge_conflict_row(row: Dict, change: str) -> Dict:
    if change not in ('lost', 'gained'):
        raise ValueError('Conflict change must be lost or gained')
    metrics = row['metrics']
    merge = row.get('candidate_merge') or {}
    return dict(
        frame_key=source_frame_key(row), change=change,
        merged_top1=dict(
            source=merge.get('raw_top1_source'),
            hit=bool(metrics['top1_hit']),
            riou=float(metrics['top1_riou']),
            score=metrics['top1_score']),
        source_top1_metrics=merge.get('source_top1_metrics', {}),
        source_pre_nms_top_log_odds=merge.get(
            'source_pre_nms_top_log_odds', {}),
        s7_affine_scale=merge.get('s7_affine_scale'),
        s7_affine_bias=merge.get('s7_affine_bias'),
        s7_lane_adjustment_max=merge.get('s7_lane_adjustment_max'),
        s7_lane_adjustment_mean=merge.get('s7_lane_adjustment_mean'),
        s7_quality_delta=merge.get('s7_quality_delta'),
        s7_quality_risk_probability=merge.get(
            's7_quality_risk_probability'),
        temporal_selection=merge.get('temporal_selection'))


def pairwise_v2_source_selection_gate(
        baseline_full: Dict, baseline_small: Dict,
        candidate_full: Dict, candidate_small: Dict,
        retention: Dict) -> Dict:
    """Strict source-only gate for replacing the formal DINO classifier."""
    baseline_correct = int(retention['baseline_correct_count'])
    checks = dict(
        exact_old_correct_retention=(
            int(retention['retained_correct_count']) == baseline_correct
            and int(retention['lost_correct_count']) == 0),
        full_top1_nonregression=(
            int(candidate_full['top1_hits'])
            >= int(baseline_full['top1_hits'])),
        full_mcml_nonregression=(
            int(candidate_full['top1_mcml'])
            <= int(baseline_full['top1_mcml'])),
        small_top1_strict_improvement=(
            int(candidate_small['top1_hits'])
            > int(baseline_small['top1_hits'])),
        small_mcml_nonregression=(
            int(candidate_small['top1_mcml'])
            <= int(baseline_small['top1_mcml'])))
    return dict(checks=checks, passed=all(checks.values()))


def s7_source_selection_gate(
        baseline_full: Dict, baseline_small: Dict,
        candidate_full: Dict, candidate_small: Dict,
        retention: Dict, args) -> Dict:
    """Source-only absolute and relative gate for enabling S7 proposals."""
    baseline_correct = int(retention['baseline_correct_count'])
    checks = dict(
        exact_old_correct_retention=(
            int(retention['retained_correct_count']) == baseline_correct
            and int(retention['lost_correct_count']) == 0),
        full_top1_nonregression=(
            int(candidate_full['top1_hits'])
            >= int(baseline_full['top1_hits'])),
        full_top1_absolute=(
            int(candidate_full['top1_hits'])
            >= int(getattr(args, 's7_source_min_full_top1', 677))),
        small_top1_nonregression=(
            int(candidate_small['top1_hits'])
            >= int(baseline_small['top1_hits'])),
        small_top1_absolute=(
            int(candidate_small['top1_hits'])
            >= int(getattr(args, 's7_source_min_small_top1', 303))),
        full_mcml_absolute=(
            int(candidate_full['top1_mcml'])
            <= int(getattr(args, 's7_source_max_mcml', 3))),
        small_mcml_absolute=(
            int(candidate_small['top1_mcml'])
            <= int(getattr(args, 's7_source_max_mcml', 3))))
    if getattr(args, 'train_components', '') == 's7_temporal_association':
        baseline_dfr = baseline_full.get('top1_dfr_fraction_per_frame')
        candidate_dfr = candidate_full.get('top1_dfr_fraction_per_frame')
        baseline_aci = baseline_full.get('top1_aci')
        candidate_aci = candidate_full.get('top1_aci')
        checks.update(
            source_temporal_metrics_available=all(
                value is not None for value in (
                    baseline_dfr, candidate_dfr,
                    baseline_aci, candidate_aci)),
            source_dfr_nonregression=(
                baseline_dfr is not None and candidate_dfr is not None
                and float(candidate_dfr) <= float(baseline_dfr) + 1e-12),
            source_aci_nonregression=(
                baseline_aci is not None and candidate_aci is not None
                and float(candidate_aci) + 1e-12 >= float(baseline_aci)))
    return dict(checks=checks, passed=all(checks.values()))


def make_target_decision(summary: Dict, args,
                         source_summary: Dict = None) -> str:
    if summary['frame_count'] != args.target_end - args.target_start + 1:
        return 'AUDIT_INVALID_TARGET_FRAME_COUNT'
    if source_summary is not None:
        source_count = int(source_summary.get('frame_count', 0))
        source_hits = int(source_summary.get('top1_hits', 0))
        source_rate = (float(source_hits) / source_count
                       if source_count > 0 else 0.0)
        if source_rate < float(args.source_min_top1_rate):
            return 'AUDIT_INVALID_SOURCE_CONTROL'
    if (summary['top1_hits'] >= args.target_min_wins
            and summary['top1_mcml'] <= args.max_mcml):
        return 'FROZEN_DINO_ROTATED_LABELLER_RESTORES_ORDERING'
    if (summary['recall_at_100'] >= args.target_min_wins
            and summary['top1_mcml'] > args.max_mcml):
        return 'DINO_LABELLER_GEOMETRY_ONLY_RANKING_INSUFFICIENT'
    return 'FROZEN_DINO_ROTATED_LABELLER_INSUFFICIENT'


def s7_architecture(args) -> Dict:
    enabled = bool(getattr(args, 's7_residual', False))
    protected_merge = bool(getattr(
        args, 's7_protected_merge', False) or getattr(
        args, 'train_components', '') in (
                's7_merge', 's7_lane_arbitration',
                's7_quality_suppression', 's7_temporal_association'))
    lane_arbitration = bool(getattr(
        args, 's7_lane_arbitration', False) or getattr(
            args, 'train_components', '') == 's7_lane_arbitration')
    quality_suppression = bool(getattr(
        args, 's7_quality_suppression', False) or getattr(
            args, 'train_components', '') == 's7_quality_suppression')
    temporal_association = bool(getattr(
        args, 's7_temporal_association', False) or getattr(
            args, 'train_components', '') == 's7_temporal_association')
    temporal_quality_head = bool(getattr(
        args, 's7_temporal_quality_head', False))
    return dict(
        enabled=enabled,
        protected_merge=(protected_merge if enabled else False),
        stride=(int(args.patch_size) // 2 if enabled else None),
        channels=(int(getattr(args, 's7_channels', 128))
                  if enabled else None),
        rpn_feat_channels=(int(getattr(
            args, 's7_rpn_feat_channels', 128)) if enabled else None),
        proposal_count=(int(getattr(args, 's7_proposal_count', 500))
                        if enabled else None),
        nms_pre=(int(getattr(args, 's7_nms_pre', 2000))
                 if enabled else None),
        anchor_sizes=([float(value) for value in getattr(
            args, 's7_anchor_sizes', [])] if enabled else []),
        merge_initial_bias=(float(getattr(
            args, 's7_merge_init_bias', -2.0))
                            if enabled and protected_merge else None),
        lane_arbitration=(lane_arbitration if enabled and protected_merge
                          else False),
        lane_hidden=(int(getattr(args, 's7_lane_hidden', 32))
                     if lane_arbitration and enabled and protected_merge
                     else None),
        lane_max_adjustment=(float(getattr(
            args, 's7_lane_max_adjustment', 2.0))
                             if lane_arbitration and enabled and protected_merge
                             else None),
        quality_suppression=(
            quality_suppression if enabled and protected_merge else False),
        quality_hidden=(int(getattr(args, 's7_quality_hidden', 32))
                        if quality_suppression and enabled and protected_merge
                        else None),
        quality_max_suppression=(float(getattr(
            args, 's7_quality_max_suppression', 2.0))
            if quality_suppression and enabled and protected_merge else None),
        quality_initial_risk_bias=(float(getattr(
            args, 's7_quality_init_risk_bias', 0.0))
            if quality_suppression and enabled and protected_merge else None),
        temporal_association=(
            temporal_association if enabled and protected_merge else False),
        temporal_cues=(list(temporal.QUALITY_CUE_NAMES)
                       if temporal_quality_head and temporal_association
                       and enabled and protected_merge else
                       list(temporal.CUE_NAMES)
                       if temporal_association and enabled and protected_merge
                       else []),
        temporal_quality_head=(
            temporal_quality_head if temporal_association and enabled
            and protected_merge else False),
        temporal_quality_hidden=(int(getattr(
            args, 's7_temporal_quality_hidden', 128))
            if temporal_quality_head and temporal_association and enabled
            and protected_merge else None),
        temporal_max_candidates=(int(getattr(
            args, 's7_temporal_max_candidates', 100))
            if temporal_association and enabled and protected_merge else None),
        temporal_min_confirmations=(int(getattr(
            args, 's7_temporal_min_confirmations', 2))
            if temporal_association and enabled and protected_merge else None))



def load_heads_checkpoint_state(heads, payload: Dict,
                                allow_s7_base_initialization: bool = False,
                                allow_lane_arbitration_initialization: bool = False,
                                allow_quality_suppression_initialization: bool = False,
                                allow_temporal_association_initialization: bool = False):
    """Load a checkpoint while allowing only explicitly new branch keys."""
    if (allow_s7_base_initialization
            or allow_lane_arbitration_initialization
            or allow_quality_suppression_initialization
            or allow_temporal_association_initialization):
        incompatible = heads.load_state_dict(
            payload['heads_state_dict'], strict=False)
        allowed_prefixes = []
        if allow_s7_base_initialization:
            allowed_prefixes.extend((
                's7_readout.', 's7_rpn_head.', 's7_score_calibrator.'))
        if allow_lane_arbitration_initialization:
            allowed_prefixes.append('s7_lane_arbitrator.')
        if allow_quality_suppression_initialization:
            allowed_prefixes.append('s7_quality_suppressor.')
        if allow_temporal_association_initialization:
            allowed_prefixes.extend((
                's7_temporal_scorer.', 's7_candidate_quality_head.'))
        disallowed_missing = [
            name for name in incompatible.missing_keys
            if not any(name.startswith(prefix) for prefix in allowed_prefixes)]
        if disallowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                'S7 base initialization state mismatch: missing={} '
                'unexpected={}'.format(
                    disallowed_missing, incompatible.unexpected_keys))
        heads.set_s7_inference_enabled(False)
        return
    heads.load_state_dict(payload['heads_state_dict'], strict=True)
    enabled = bool(payload.get(
        's7_inference_enabled',
        payload.get('s7_architecture', {}).get('enabled', False)))
    heads.set_s7_inference_enabled(enabled)


def load_frozen_s7_component(heads, payload: Dict, in_channels: int, args):
    """Load only S7 readout/RPN tensors from a source-only S7 checkpoint."""
    if payload.get('source_only') is not True or payload.get(
            'frozen_dinov2') is not True:
        raise RuntimeError('S7 component checkpoint is not source-only')
    stored_mode = payload.get('training_protocol', {}).get('train_components')
    if stored_mode != 's7_rpn':
        raise RuntimeError(
            'S7 component checkpoint must come from s7_rpn training')
    if payload.get('s7_inference_enabled') is not True:
        raise RuntimeError(
            'S7 component checkpoint has S7 disabled; use a raw epoch '
            'checkpoint, not the epoch-0 fallback best checkpoint')
    if int(payload.get('in_channels', -1)) != int(in_channels):
        raise RuntimeError('S7 component checkpoint channel mismatch')
    stored_s7 = payload.get('s7_architecture', {})
    requested_s7 = s7_architecture(args)
    keys = ('stride', 'channels', 'rpn_feat_channels',
            'proposal_count', 'nms_pre', 'anchor_sizes')
    if (not bool(stored_s7.get('enabled', False))
            or any(stored_s7.get(key) != requested_s7.get(key)
                   for key in keys)):
        raise RuntimeError('S7 component checkpoint architecture mismatch')
    prefixes = ('s7_readout.', 's7_rpn_head.')
    source_state = {
        name: tensor for name, tensor in payload['heads_state_dict'].items()
        if name.startswith(prefixes)}
    expected = {
        name for name in heads.state_dict() if name.startswith(prefixes)}
    if set(source_state) != expected:
        raise RuntimeError(
            'S7 component state mismatch: missing={} unexpected={}'.format(
                sorted(expected - set(source_state)),
                sorted(set(source_state) - expected)))
    destination = heads.state_dict()
    destination.update(source_state)
    heads.load_state_dict(destination, strict=True)
    heads.set_s7_inference_enabled(True)
    return dict(
        epoch=int(payload.get('epoch', -1)),
        best_epoch=int(payload.get('best_epoch', -1)),
        inference_enabled_in_component=True,
        loaded_parameter_names=sorted(source_state))


def source_selected_checkpoint_gate(
        payload: Dict, min_full_top1: int = 688,
        min_small_top1: int = 311, max_mcml: int = 3) -> Dict:
    """Validate the metadata required before enabling a gated S7 runtime.

    A checkpoint with ``best_epoch=0`` is always the native fallback.  The
    explicit stored gate and retention evidence prevent a deployment config
    from accidentally turning a failed source-only experiment on.
    """
    best = payload.get('best_source_val_summary') or {}
    best_small = payload.get('best_source_small_val_summary') or {}
    retention = payload.get('source_exact_retention') or {}
    checks = dict(
        positive_best_epoch=int(payload.get('best_epoch', 0)) > 0,
        temporal_training_protocol=(
            (payload.get('training_protocol') or {}).get('train_components')
            == 's7_temporal_association'),
        stored_source_selection_gate=(
            payload.get('source_selection_gate_passed') is True),
        exact_old_correct_retention=(
            int(retention.get('lost_correct_count', -1)) == 0
            and int(retention.get('retained_correct_count', -1)) == int(
                retention.get('baseline_correct_count', -2))),
        full_top1_absolute=(
            int(best.get('top1_hits', -1)) >= int(min_full_top1)),
        small_top1_absolute=(
            int(best_small.get('top1_hits', -1)) >= int(min_small_top1)),
        full_mcml_absolute=(
            int(best.get('top1_mcml', max_mcml + 1)) <= int(max_mcml)),
        small_mcml_absolute=(
            int(best_small.get('top1_mcml', max_mcml + 1))
            <= int(max_mcml)))
    return dict(checks=checks, passed=all(checks.values()))


def checkpoint_payload(heads, optimizer, scheduler, epoch: int,
                       best_epoch: int, best_summary: Dict,
                       in_channels: int, args,
                       global_step: int = 0,
                       best_small_summary: Dict = None,
                       source_sampling: Dict = None,
                       source_baseline_summary: Dict = None,
                       source_baseline_small_summary: Dict = None,
                       source_baseline_correct_keys: Sequence[str] = None,
                       source_selection_gate_passed: Optional[bool] = None,
                       source_exact_retention: Optional[Dict] = None
                       ) -> Dict:
    return dict(
        labeller=LABELLER_NAME, protocol_version=PROTOCOL_VERSION,
        source_only=True, frozen_dinov2=True,
        epoch=int(epoch), best_epoch=int(best_epoch),
        global_step=int(global_step),
        best_source_val_summary=best_summary,
        best_source_small_val_summary=best_small_summary,
        source_selection_gate_passed=(
            None if source_selection_gate_passed is None
            else bool(source_selection_gate_passed)),
        source_exact_retention=source_exact_retention,
        source_baseline_val_summary=source_baseline_summary,
        source_baseline_small_val_summary=source_baseline_small_summary,
        source_baseline_correct_keys=(
            None if source_baseline_correct_keys is None else
            [str(key) for key in source_baseline_correct_keys]),
        source_sampling=source_sampling,
        s7_readout_reference=(
            dict(
                paper='LiDeRe', url=LIDERE_PAPER_URL,
                implementation_scope='inspired_lightweight_residual_readout')
            if bool(getattr(args, 's7_residual', False)) else None),
        s7_architecture=s7_architecture(args),
        s7_inference_enabled=bool(heads.s7_inference_enabled()),
        roi_cls_teacher_state=heads.roi_cls_teacher_state(),
        training_protocol=dict(
            train_components=str(args.train_components),
            optimization_loss_components=optimization_loss_component_names(
                args.train_components,
                quality_head=bool(getattr(
                    args, 's7_temporal_quality_head', False)),
                relative_quality=bool(getattr(
                    args, 's7_temporal_relative_quality', False))),
            trainable_parameter_names=[
                name for name, parameter in heads.named_parameters()
                if parameter.requires_grad],
            epochs=int(args.epochs), lr=float(args.lr),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
            max_grad_norm=float(args.max_grad_norm),
            warmup_iters=int(args.warmup_iters),
            warmup_ratio=float(args.warmup_ratio),
            lr_steps=[int(value) for value in args.lr_steps],
            lr_gamma=float(args.lr_gamma),
            checkpoint_interval=int(args.checkpoint_interval),
            selection_epochs=[int(value)
                              for value in args.selection_epochs],
            pairwise=(None if args.train_components not in (
                          'roi_cls_pairwise', 'roi_cls_pairwise_v2')
                      else dict(
                          version=(2 if args.train_components ==
                                   'roi_cls_pairwise_v2' else 1),
                          margin=float(args.pairwise_margin),
                          cls_loss_weight=float(
                              args.pairwise_cls_loss_weight),
                          pairwise_loss_weight=float(
                              args.pairwise_loss_weight),
                          retention_loss_weight=float(
                              args.retention_loss_weight),
                          retention_temperature=float(
                              args.retention_temperature),
                          negative_riou_thr=float(
                              args.pairwise_negative_riou_thr),
                          nms_iou_thr=float(args.pairwise_nms_iou_thr),
                          negatives_per_positive=int(
                              args.pairwise_negatives_per_positive))),
            s7_source_gate=(
                None if args.train_components not in (
                    's7_rpn', 's7_merge', 's7_lane_arbitration',
                    's7_quality_suppression', 's7_temporal_association')
                else dict(
                    min_full_top1=int(getattr(
                        args, 's7_source_min_full_top1', 677)),
                    min_small_top1=int(getattr(
                        args, 's7_source_min_small_top1', 303)),
                    max_mcml=int(getattr(
                        args, 's7_source_max_mcml', 3)))),
            s7_merge=(
                None if args.train_components != 's7_merge' else dict(
                    component_checkpoint=(
                        None if not getattr(
                            args, 's7_component_checkpoint', None)
                        else os.path.abspath(args.s7_component_checkpoint)),
                    margin=float(args.s7_merge_margin),
                    retention_weight=float(args.s7_merge_retention_weight),
                    gain_weight=float(args.s7_merge_gain_weight),
                    prior_weight=float(args.s7_merge_prior_weight),
                    initial_bias=float(args.s7_merge_init_bias),
                    native_nms_protected=True,
                    proposal_sources=['native_s14', 'supplement_s7'])),
            s7_lane_arbitration=(
                None if args.train_components != 's7_lane_arbitration' else dict(
                    base_checkpoint=(
                        None if not getattr(args, 'init_checkpoint', None)
                        else os.path.abspath(args.init_checkpoint)),
                    hidden=int(args.s7_lane_hidden),
                    max_adjustment=float(args.s7_lane_max_adjustment),
                    base_epoch=int(args.s7_lane_base_epoch),
                    hard_negatives=int(args.s7_lane_hard_negatives),
                    gain_repeat=int(args.s7_lane_gain_repeat),
                    hard_negative_ranking='current_adjusted_s7_log_odds',
                    gain_competitors=['native_s14', 'wrong_supplement_s7'],
                    native_nms_protected=True,
                    proposal_sources=['native_s14', 'supplement_s7'])),
            s7_quality_suppression=(
                None if args.train_components != 's7_quality_suppression'
                else dict(
                    base_checkpoint=(
                        None if not getattr(args, 'init_checkpoint', None)
                        else os.path.abspath(args.init_checkpoint)),
                    base_epoch=int(args.s7_quality_base_epoch),
                    lane_wide=True,
                    adjustment_range=[
                        -float(args.s7_quality_max_suppression), 0.0],
                    hidden=int(args.s7_quality_hidden),
                    initial_risk_bias=float(
                        args.s7_quality_init_risk_bias),
                    margin=float(args.s7_quality_margin),
                    risk_weight=float(args.s7_quality_risk_weight),
                    preserve_weight=float(
                        args.s7_quality_preserve_weight),
                    retention_weight=float(
                        args.s7_quality_retention_weight),
                    prior_weight=float(args.s7_quality_prior_weight),
                    preflight=dict(
                        exact_training_risk_miner=True,
                        minimum_risk_pairs=S7_QUALITY_MIN_RISK_PAIRS,
                        zero_risk_action=(
                            'skip_optimization_and_keep_epoch_0')),
                    positive_promotion=False, gain_replay=False,
                    source_train_only=True,
                    native_nms_protected=True,
                    proposal_sources=['native_s14', 'supplement_s7'])),
            s7_temporal_association=(
                None if args.train_components != 's7_temporal_association'
                else dict(
                    base_checkpoint=(
                        None if not getattr(args, 'init_checkpoint', None)
                        else os.path.abspath(args.init_checkpoint)),
                    base_epoch=int(
                        args.s7_temporal_relative_base_epoch
                        if bool(getattr(
                            args, 's7_temporal_relative_quality', False))
                        else args.s7_temporal_base_epoch),
                    cues=list(
                        temporal.QUALITY_CUE_NAMES
                        if bool(getattr(
                            args, 's7_temporal_quality_head', False))
                        else temporal.CUE_NAMES),
                    max_candidates=int(args.s7_temporal_max_candidates),
                    min_confirmations=int(
                        args.s7_temporal_min_confirmations),
                    override_margin=float(args.s7_temporal_override_margin),
                    max_center_distance=float(
                        args.s7_temporal_max_center_distance),
                    min_rotated_iou=float(args.s7_temporal_min_riou),
                    min_appearance_similarity=float(
                        args.s7_temporal_min_appearance),
                    dfr_aci_angle_limit_deg=(
                        SOURCE_TEMPORAL_ANGLE_LIMIT_DEG),
                    margin=float(args.s7_temporal_margin),
                    retention_weight=float(
                        args.s7_temporal_retention_weight),
                    gain_weight=float(args.s7_temporal_gain_weight),
                    prior_weight=float(args.s7_temporal_prior_weight),
                    quality_head_hidden=int(getattr(
                        args, 's7_temporal_quality_hidden', 128)),
                    quality_loss_weight=float(getattr(
                        args, 's7_temporal_quality_loss_weight', 1.0)),
                    relative_quality=bool(getattr(
                        args, 's7_temporal_relative_quality', False)),
                    relative_quality_weight=float(getattr(
                        args, 's7_temporal_relative_quality_weight', 0.5)),
                    relative_quality_margin=float(getattr(
                        args, 's7_temporal_relative_quality_margin', 0.25)),
                    relative_quality_min_gap=float(getattr(
                        args, 's7_temporal_relative_quality_min_gap', 0.10)),
                    relative_quality_max_pairs=int(getattr(
                        args, 's7_temporal_relative_quality_max_pairs', 128)),
                    source_train_only=True, target_read=False,
                    candidate_quality_head=bool(getattr(
                        args, 's7_temporal_quality_head', False)),
                    training_state=(
                        'dense_source_candidate_max_riou_plus_relative_rank'
                        if bool(getattr(
                            args, 's7_temporal_relative_quality', False))
                        else 'dense_source_candidate_max_riou'
                        if bool(getattr(
                            args, 's7_temporal_quality_head', False))
                        else 'previous_source_GT_usable_candidate'),
                    inference_state='strictly_previous_selected_candidate',
                    reset_on=['sequence_change', 'frame_gap'],
                    native_fallback=True))),
        in_channels=int(in_channels), patch_size=int(args.patch_size),
        rpn_feat_channels=int(args.rpn_feat_channels),
        roi_fc_channels=int(args.roi_fc_channels),
        roi_samples=int(args.roi_samples),
        proposal_count=int(args.proposal_count),
        max_detections=int(args.max_detections),
        heads_state_dict=heads.state_dict(),
        optimizer_state_dict=(None if optimizer is None
                              else optimizer.state_dict()),
        scheduler_state_dict=(None if scheduler is None
                              else scheduler.state_dict()))


def validate_checkpoint(payload: Dict, in_channels: int, args,
                        allow_training_mode_mismatch: bool = False,
                        allow_s7_base_initialization: bool = False,
                        allow_lane_arbitration_initialization: bool = False,
                        allow_quality_suppression_initialization: bool = False,
                        allow_temporal_association_initialization: bool = False,
                        allow_temporal_policy_mismatch: bool = False):
    required = (
        'source_only', 'frozen_dinov2', 'in_channels', 'patch_size',
        'rpn_feat_channels', 'roi_fc_channels', 'heads_state_dict')
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError('Labeller checkpoint lacks {}'.format(
            ', '.join(missing)))
    if (payload['source_only'] is not True
            or payload['frozen_dinov2'] is not True):
        raise RuntimeError('Checkpoint is not source-only/frozen-DINO')
    stored_mode = payload.get('training_protocol', {}).get(
        'train_components', 'all')
    requested_mode = getattr(args, 'train_components', stored_mode)
    if (not allow_training_mode_mismatch
            and stored_mode != requested_mode):
        raise RuntimeError(
            'Checkpoint training mode mismatch: stored={} requested={}'.format(
                stored_mode, requested_mode))
    expected = dict(
        in_channels=int(in_channels), patch_size=int(args.patch_size),
        rpn_feat_channels=int(args.rpn_feat_channels),
        roi_fc_channels=int(args.roi_fc_channels))
    mismatched = [key for key, value in expected.items()
                  if int(payload[key]) != int(value)]
    if mismatched:
        raise RuntimeError('Labeller architecture mismatch: {}'.format(
            ', '.join(mismatched)))
    requested_s7 = s7_architecture(args)
    stored_s7 = payload.get('s7_architecture', dict(enabled=False))
    if requested_s7['enabled']:
        if not bool(stored_s7.get('enabled', False)):
            if not allow_s7_base_initialization:
                raise RuntimeError(
                    'Checkpoint lacks the requested S7 residual branch')
        else:
            keys = ('stride', 'channels', 'rpn_feat_channels',
                    'proposal_count', 'nms_pre', 'anchor_sizes')
            architecture_mismatch = any(
                stored_s7.get(key) != requested_s7.get(key) for key in keys)
            stored_merge = bool(stored_s7.get('protected_merge', False))
            requested_merge = bool(requested_s7.get('protected_merge', False))
            if stored_merge != requested_merge:
                architecture_mismatch = True
            if (requested_merge and stored_s7.get('merge_initial_bias')
                    != requested_s7.get('merge_initial_bias')):
                architecture_mismatch = True
            if (requested_s7.get('lane_arbitration', False)
                    != bool(stored_s7.get('lane_arbitration', False))
                    and not allow_lane_arbitration_initialization):
                architecture_mismatch = True
            if (requested_s7.get('lane_arbitration', False)
                    and bool(stored_s7.get('lane_arbitration', False))
                    and (stored_s7.get('lane_hidden')
                         != requested_s7.get('lane_hidden')
                         or stored_s7.get('lane_max_adjustment')
                         != requested_s7.get('lane_max_adjustment'))):
                architecture_mismatch = True
            stored_quality = bool(stored_s7.get(
                'quality_suppression', False))
            requested_quality = bool(requested_s7.get(
                'quality_suppression', False))
            if (stored_quality != requested_quality
                    and not allow_quality_suppression_initialization):
                architecture_mismatch = True
            if (stored_quality and requested_quality
                    and (stored_s7.get('quality_hidden')
                         != requested_s7.get('quality_hidden')
                         or stored_s7.get('quality_max_suppression')
                         != requested_s7.get('quality_max_suppression')
                         or stored_s7.get('quality_initial_risk_bias')
                         != requested_s7.get('quality_initial_risk_bias'))):
                architecture_mismatch = True
            stored_temporal = bool(stored_s7.get(
                'temporal_association', False))
            requested_temporal = bool(requested_s7.get(
                'temporal_association', False))
            if (stored_temporal != requested_temporal
                    and not allow_temporal_association_initialization):
                architecture_mismatch = True
            if (stored_temporal and requested_temporal
                    and (stored_s7.get('temporal_cues')
                         != requested_s7.get('temporal_cues')
                         or stored_s7.get('temporal_max_candidates')
                         != requested_s7.get('temporal_max_candidates')
                         or (stored_s7.get('temporal_min_confirmations')
                             != requested_s7.get(
                                 'temporal_min_confirmations')
                             and not allow_temporal_policy_mismatch))):
                architecture_mismatch = True
            stored_temporal_quality = bool(stored_s7.get(
                'temporal_quality_head', False))
            requested_temporal_quality = bool(requested_s7.get(
                'temporal_quality_head', False))
            if (stored_temporal_quality != requested_temporal_quality
                    and not allow_temporal_association_initialization):
                architecture_mismatch = True
            if (stored_temporal_quality and requested_temporal_quality
                    and stored_s7.get('temporal_quality_hidden')
                    != requested_s7.get('temporal_quality_hidden')):
                architecture_mismatch = True
            if architecture_mismatch:
                raise RuntimeError('S7 checkpoint architecture mismatch')
    elif bool(stored_s7.get('enabled', False)):
        raise RuntimeError(
            'S7 checkpoint cannot load into the native single-scale head')


def train_source_only(dino, heads, train_records, val_records, args,
                      dino_device, head_device, in_channels: int):
    pairwise_modes = ('roi_cls_pairwise', 'roi_cls_pairwise_v2')
    roi_cls_mode = args.train_components in ('roi_cls',) + pairwise_modes
    s7_rpn_mode = args.train_components == 's7_rpn'
    s7_merge_mode = args.train_components == 's7_merge'
    s7_lane_mode = args.train_components == 's7_lane_arbitration'
    s7_quality_mode = args.train_components == 's7_quality_suppression'
    s7_temporal_mode = args.train_components == 's7_temporal_association'
    s7_temporal_relative_mode = bool(getattr(
        args, 's7_temporal_relative_quality', False))
    s7_mode = bool(
        s7_rpn_mode or s7_merge_mode or s7_lane_mode or s7_quality_mode
        or s7_temporal_mode)
    protected_source_mode = bool(roi_cls_mode or s7_mode)
    trainable_names = configure_trainable_components(
        heads, args.train_components)
    trainable_parameters = [
        parameter for parameter in heads.parameters()
        if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError('No trainable head parameters')
    print('[trainable] mode={} tensors={} parameters={}'.format(
        args.train_components, len(trainable_names),
        sum(int(parameter.numel()) for parameter in trainable_parameters)))
    optimizer = torch.optim.SGD(
        trainable_parameters, lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)
    scheduler = None
    start_epoch = 1
    global_step = 0
    best_epoch = 0
    best_summary = None
    best_small_summary = None
    best_key = None
    source_baseline_summary = None
    source_baseline_small_summary = None
    source_baseline_correct_keys = None
    source_initial_temporal_summary = None
    source_initial_temporal_small_summary = None
    s7_quality_support_audit = None
    history = []
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location='cpu')
        if s7_lane_mode and int(payload.get('epoch', -1)) != int(
                args.s7_lane_base_epoch):
            raise RuntimeError(
                'Lane arbitration base checkpoint must be audited epoch 1; '
                'found epoch {}'.format(payload.get('epoch')))
        if s7_quality_mode:
            if int(payload.get('epoch', -1)) != int(
                    args.s7_quality_base_epoch):
                raise RuntimeError(
                    'Quality suppression base checkpoint must be audited '
                    'affine epoch 1; found epoch {}'.format(
                        payload.get('epoch')))
            stored_mode = payload.get(
                'training_protocol', {}).get('train_components')
            if stored_mode != 's7_merge':
                raise RuntimeError(
                    'Quality suppression must initialize from the complete '
                    's7_merge epoch-1 checkpoint; found {}'.format(
                        stored_mode))
        if s7_temporal_mode and not s7_temporal_relative_mode:
            if int(payload.get('epoch', -1)) != int(
                    args.s7_temporal_base_epoch):
                raise RuntimeError(
                    'Temporal association base checkpoint must be audited '
                    'affine epoch 1; found epoch {}'.format(
                        payload.get('epoch')))
            stored_mode = payload.get(
                'training_protocol', {}).get('train_components')
            if stored_mode != 's7_merge':
                raise RuntimeError(
                    'Temporal association must initialize from the complete '
                    's7_merge epoch-1 checkpoint; found {}'.format(
                        stored_mode))
        if s7_temporal_relative_mode:
            if int(payload.get('epoch', -1)) != int(
                    args.s7_temporal_relative_base_epoch):
                raise RuntimeError(
                    'Relative candidate quality must initialize from fixed '
                    'pointwise quality epoch {}; found {}'.format(
                        args.s7_temporal_relative_base_epoch,
                        payload.get('epoch')))
            stored_mode = payload.get(
                'training_protocol', {}).get('train_components')
            if stored_mode != 's7_temporal_association':
                raise RuntimeError(
                    'Relative candidate quality must initialize from the '
                    'complete temporal quality checkpoint; found {}'.format(
                        stored_mode))
            stored_s7 = payload.get('s7_architecture', {})
            if not bool(stored_s7.get('temporal_quality_head', False)):
                raise RuntimeError(
                    'Relative candidate quality requires a pointwise '
                    'candidate quality head checkpoint')
        validate_checkpoint(
            payload, in_channels, args,
            allow_training_mode_mismatch=True,
            allow_s7_base_initialization=(
                s7_mode and not s7_lane_mode and not s7_quality_mode
                and not s7_temporal_mode),
            allow_lane_arbitration_initialization=s7_lane_mode,
            allow_quality_suppression_initialization=s7_quality_mode,
            allow_temporal_association_initialization=(
                s7_temporal_mode and not s7_temporal_relative_mode),
            allow_temporal_policy_mismatch=s7_temporal_relative_mode)
        load_heads_checkpoint_state(
            heads, payload,
            allow_s7_base_initialization=(
                s7_mode and not s7_lane_mode and not s7_quality_mode
                and not s7_temporal_mode),
            allow_lane_arbitration_initialization=s7_lane_mode,
            allow_quality_suppression_initialization=s7_quality_mode,
            allow_temporal_association_initialization=(
                s7_temporal_mode and not s7_temporal_relative_mode))
        if s7_merge_mode:
            component_payload = torch.load(
                args.s7_component_checkpoint, map_location='cpu')
            component_summary = load_frozen_s7_component(
                heads, component_payload, in_channels, args)
            print('[s7-component] epoch={} tensors={} checkpoint={}'.format(
                component_summary['epoch'],
                len(component_summary['loaded_parameter_names']),
                os.path.abspath(args.s7_component_checkpoint)))
        if args.train_components in pairwise_modes:
            heads.capture_roi_cls_teacher()
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location='cpu')
        validate_checkpoint(payload, in_channels, args)
        load_heads_checkpoint_state(heads, payload)
        if args.train_components in pairwise_modes:
            heads.load_roi_cls_teacher_state(
                payload.get('roi_cls_teacher_state'))
        if payload.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        start_epoch = int(payload.get('epoch', 0)) + 1
        global_step = int(payload.get(
            'global_step', (start_epoch - 1) * len(train_records)))
        best_epoch = int(payload.get('best_epoch', 0))
        best_summary = payload.get('best_source_val_summary')
        best_small_summary = payload.get('best_source_small_val_summary')
        source_baseline_summary = payload.get(
            'source_baseline_val_summary')
        source_baseline_small_summary = payload.get(
            'source_baseline_small_val_summary')
        source_baseline_correct_keys = payload.get(
            'source_baseline_correct_keys')

    frozen_parameter_versions = {
        name: int(parameter._version)
        for name, parameter in heads.named_parameters()
        if not parameter.requires_grad}

    best_path = os.path.join(args.work_dir, 'labeller_best_source_only.pth')
    latest_path = os.path.join(args.work_dir, 'labeller_latest.pth')
    epoch_train_records = list(train_records)
    source_sampling = None
    small_val_records = []
    if protected_source_mode:
        epoch_train_records, source_sampling = source_small_balanced_records(
            train_records, args)
        small_val_records = source_small_records(
            val_records, args, source_sampling['short_token_threshold'])
        if not small_val_records:
            raise RuntimeError(
                'Source validation has no objects within the source-train '
                'small-token threshold')
        print('[source-small] threshold={:.6f} train={}/{} balanced={} '
              'val={}/{}'.format(
                  source_sampling['short_token_threshold'],
                  source_sampling['small_frame_count'], len(train_records),
                  len(epoch_train_records), len(small_val_records),
                  len(val_records)))
        if args.resume_checkpoint:
            if (source_baseline_summary is None
                    or source_baseline_small_summary is None
                    or best_summary is None or best_small_summary is None):
                raise RuntimeError(
                    'ROI-cls resume checkpoint lacks source baseline/small '
                    'validation summaries')
            if ((args.train_components in pairwise_modes or s7_mode)
                    and source_baseline_correct_keys is None):
                raise RuntimeError(
                    'Protected resume checkpoint lacks exact source-retention '
                    'baseline keys')
            best_key = roi_cls_selection_key(
                best_summary, best_small_summary)
        else:
            if s7_mode:
                heads.set_s7_inference_enabled(False)
            baseline_rows = evaluate_records(
                dino, heads, val_records, args, dino_device, head_device,
                role='source_validation_baseline')
            source_baseline_summary = summarize_rows(baseline_rows)
            source_baseline_correct_keys = source_correct_frame_keys(
                baseline_rows)
            if s7_temporal_mode:
                small_keys = {
                    '{}|{}|{}'.format(
                        row.get('split', ''), row.get('seq', ''),
                        int(row.get('frame', -1)))
                    for row in small_val_records}
                baseline_small_rows = [
                    row for row in baseline_rows
                    if source_frame_key(row) in small_keys]
            else:
                baseline_small_rows = evaluate_records(
                    dino, heads, small_val_records, args,
                    dino_device, head_device,
                    role='source_small_validation_baseline')
            source_baseline_small_summary = summarize_rows(
                baseline_small_rows)
            best_summary = source_baseline_summary
            best_small_summary = source_baseline_small_summary
            best_key = roi_cls_selection_key(
                best_summary, best_small_summary)
            atomic_torch_save(checkpoint_payload(
                heads, None, None, 0, 0, best_summary, in_channels, args,
                global_step, best_small_summary, source_sampling,
                source_baseline_summary,
                source_baseline_small_summary,
                source_baseline_correct_keys,
                source_selection_gate_passed=False,
                source_exact_retention=None), best_path)
            if s7_mode:
                heads.set_s7_inference_enabled(True)
            print('[source-baseline] full_top1={}/{} small_top1={}/{} '
                  'fallback_epoch=0'.format(
                      best_summary['top1_hits'],
                      best_summary['frame_count'],
                      best_small_summary['top1_hits'],
                      best_small_summary['frame_count']))
            if s7_temporal_mode:
                initial_temporal_rows = evaluate_records(
                    dino, heads, val_records, args, dino_device, head_device,
                    role='source_validation_initial_temporal')
                source_initial_temporal_summary = summarize_rows(
                    initial_temporal_rows)
                initial_small_rows = [
                    row for row in initial_temporal_rows
                    if source_frame_key(row) in small_keys]
                source_initial_temporal_small_summary = summarize_rows(
                    initial_small_rows)
                print(
                    '[source-temporal-initial] full_top1={}/{} '
                    'small_top1={}/{} overrides={} resets={}'.format(
                        source_initial_temporal_summary['top1_hits'],
                        source_initial_temporal_summary['frame_count'],
                        source_initial_temporal_small_summary['top1_hits'],
                        source_initial_temporal_small_summary['frame_count'],
                        source_initial_temporal_summary[
                            'temporal_override_count'],
                        source_initial_temporal_summary[
                            'temporal_reset_count']))
    elif best_summary is not None:
        best_key = source_selection_key(best_summary)

    if s7_quality_mode:
        heads.set_s7_inference_enabled(True)
        s7_quality_support_audit = audit_s7_quality_training_support(
            dino, heads, train_records, args, dino_device, head_device)
        if not bool(s7_quality_support_audit['training_allowed']):
            progress_path, progress_replacements = (
                write_source_training_progress(
                    args, 0, best_epoch, best_path, best_path,
                    source_baseline_summary,
                    source_baseline_small_summary,
                    best_summary, best_small_summary, history,
                    status=(
                        'SOURCE_ONLY_TRAINING_SKIPPED_ZERO_'
                        'S7_QUALITY_RISK_SUPPORT'),
                    s7_quality_support_audit=s7_quality_support_audit))
            print('[s7-quality-preflight] training_skipped=True '
                  'fallback_epoch=0 nonfinite_replacements={} out={}'
                  .format(progress_replacements, progress_path))
            frozen_parameters_unchanged = bool(
                frozen_parameter_versions == {
                    name: int(parameter._version)
                    for name, parameter in heads.named_parameters()
                    if not parameter.requires_grad})
            if not frozen_parameters_unchanged:
                raise RuntimeError(
                    'A frozen detector-head parameter changed during '
                    'S7 quality preflight')
            return (
                best_path, best_epoch, best_summary, best_small_summary,
                source_sampling, source_baseline_summary,
                source_baseline_small_summary,
                frozen_parameters_unchanged, history,
                s7_quality_support_audit,
                source_initial_temporal_summary,
                source_initial_temporal_small_summary)

    selection_epochs = set(int(value) for value in args.selection_epochs)
    for epoch in range(start_epoch, args.epochs + 1):
        if s7_mode:
            heads.set_s7_inference_enabled(True)
        train_row = train_epoch(
            dino, heads, optimizer, epoch_train_records, epoch, global_step,
            args, dino_device, head_device)
        global_step = int(train_row['global_step_end'])
        evaluate_epoch = (
            epoch % int(args.checkpoint_interval) == 0
            or epoch == int(args.epochs))
        val_summary = None
        small_val_summary = None
        merge_conflicts = None
        if evaluate_epoch:
            val_rows = evaluate_records(
                dino, heads, val_records, args, dino_device, head_device,
                role='source_validation')
            val_summary = summarize_rows(val_rows)
            if protected_source_mode:
                if s7_temporal_mode:
                    small_keys = {
                        '{}|{}|{}'.format(
                            row.get('split', ''), row.get('seq', ''),
                            int(row.get('frame', -1)))
                        for row in small_val_records}
                    small_val_rows = [
                        row for row in val_rows
                        if source_frame_key(row) in small_keys]
                else:
                    small_val_rows = evaluate_records(
                        dino, heads, small_val_records, args,
                        dino_device, head_device,
                        role='source_small_validation')
                small_val_summary = summarize_rows(small_val_rows)
        selection_eligible = epoch in selection_epochs
        if selection_eligible and val_summary is None:
            raise RuntimeError(
                'A selection epoch must also be a validation epoch')
        if val_summary is None:
            key = None
        elif protected_source_mode:
            key = roi_cls_selection_key(val_summary, small_val_summary)
        else:
            key = source_selection_key(val_summary)
        source_gate_passed = True
        retention_summary = None
        protected_gate = None
        if val_summary is not None and protected_source_mode:
            if args.train_components in pairwise_modes or s7_mode:
                retention_summary = source_top1_retention_summary(
                    source_baseline_correct_keys, val_rows)
                if s7_mode:
                    if (s7_merge_mode or s7_lane_mode or s7_quality_mode
                            or s7_temporal_mode):
                        merge_conflicts = source_merge_conflict_summary(
                            source_baseline_correct_keys, val_rows)
                    protected_gate = s7_source_selection_gate(
                        source_baseline_summary,
                        source_baseline_small_summary,
                        val_summary, small_val_summary,
                        retention_summary, args)
                    source_gate_passed = bool(protected_gate['passed'])
                elif args.train_components == 'roi_cls_pairwise_v2':
                    protected_gate = pairwise_v2_source_selection_gate(
                        source_baseline_summary,
                        source_baseline_small_summary,
                        val_summary, small_val_summary,
                        retention_summary)
                    source_gate_passed = bool(protected_gate['passed'])
                else:
                    retention_floor = (
                        len(source_baseline_correct_keys)
                        - int(args.source_retain_max_top1_drop))
                    source_gate_passed = bool(
                        retention_summary['retained_correct_count']
                        >= retention_floor)
            else:
                retention_floor = (
                    int(source_baseline_summary['top1_hits'])
                    - int(args.source_retain_max_top1_drop))
                source_gate_passed = bool(
                    int(val_summary['top1_hits']) >= retention_floor)
        improved = bool(
            selection_eligible and source_gate_passed
            and (best_key is None or key > best_key))
        if improved:
            best_key = key
            best_epoch = int(epoch)
            best_summary = val_summary
            best_small_summary = small_val_summary
            atomic_torch_save(checkpoint_payload(
                heads, None, None, epoch, best_epoch, best_summary,
                in_channels, args, global_step, best_small_summary,
                source_sampling, source_baseline_summary,
                source_baseline_small_summary,
                source_baseline_correct_keys,
                source_selection_gate_passed=True,
                source_exact_retention=retention_summary), best_path)
        if evaluate_epoch:
            epoch_path = os.path.join(
                args.work_dir,
                'labeller_epoch_{:02d}_source_only.pth'.format(epoch))
            atomic_torch_save(checkpoint_payload(
                heads, optimizer, scheduler, epoch, best_epoch,
                best_summary, in_channels, args, global_step,
                best_small_summary, source_sampling,
                source_baseline_summary,
                source_baseline_small_summary,
                source_baseline_correct_keys,
                source_selection_gate_passed=source_gate_passed,
                source_exact_retention=retention_summary), epoch_path)
        exact_retention_passed = (
            bool(protected_gate['checks'][
                'exact_old_correct_retention'])
            if protected_gate is not None else bool(source_gate_passed))
        history.append(dict(
            epoch=int(epoch), train=train_row,
            source_val=val_summary,
            source_small_val=small_val_summary,
            source_selection_gate_passed=bool(source_gate_passed),
            source_retention_passed=exact_retention_passed,
            source_exact_retention=retention_summary,
            s7_merge_calibration=(
                s7_calibration_state(heads)
                if (s7_merge_mode or s7_lane_mode or s7_quality_mode
                    or s7_temporal_mode) else None),
            s7_temporal_cue_weights=(
                heads.s7_temporal_scorer.state_summary()
                if s7_temporal_mode else None),
            s7_merge_conflicts=merge_conflicts,
            source_selection_gate=protected_gate,
            pairwise_v2_source_gate=(
                protected_gate if args.train_components ==
                'roi_cls_pairwise_v2' else None),
            s7_source_gate=(protected_gate if s7_mode else None),
            selected_as_best=bool(improved),
            selection_eligible=bool(selection_eligible),
            checkpoint_saved=bool(evaluate_epoch),
            lr=float(optimizer.param_groups[0]['lr'])))
        atomic_torch_save(checkpoint_payload(
            heads, optimizer, scheduler, epoch, best_epoch, best_summary,
            in_channels, args, global_step, best_small_summary,
            source_sampling, source_baseline_summary,
            source_baseline_small_summary,
            source_baseline_correct_keys,
            source_selection_gate_passed=source_gate_passed,
            source_exact_retention=retention_summary), latest_path)
        progress_path = None
        progress_replacements = 0
        if args.train_components in (
                'roi_cls_pairwise_v2', 's7_rpn', 's7_merge',
                's7_lane_arbitration', 's7_quality_suppression',
                's7_temporal_association'):
            progress_path, progress_replacements = (
                write_source_training_progress(
                    args, epoch, best_epoch, best_path, latest_path,
                    source_baseline_summary,
                    source_baseline_small_summary,
                    best_summary, best_small_summary, history))
        if val_summary is None:
            print('[source-epoch] epoch={} validation=skipped best_epoch={}'
                  .format(epoch, best_epoch))
        else:
            small_text = (' n/a' if small_val_summary is None else
                          ' {}/{}'.format(
                              small_val_summary['top1_hits'],
                              small_val_summary['frame_count']))
            gate_label = ('source_gate' if protected_gate is not None
                          else 'retention')
            print('[source-epoch] epoch={} top1={}/{} small_top1={} r100={} '
                  '{}={} selection_eligible={} best_epoch={}'.format(
                      epoch, val_summary['top1_hits'],
                      val_summary['frame_count'], small_text,
                      val_summary['recall_at_100'], gate_label,
                      source_gate_passed,
                      selection_eligible, best_epoch))
            if protected_gate is not None:
                failed_checks = sorted(
                    name for name, passed
                    in protected_gate['checks'].items() if not passed)
                print(
                    '[source-gate] epoch={} passed={} retained={}/{} '
                    'lost={} gained={} full_mcml={}->{} small_mcml={}->{} '
                    'failed={}'.format(
                        epoch, protected_gate['passed'],
                        retention_summary['retained_correct_count'],
                        retention_summary['baseline_correct_count'],
                        retention_summary['lost_correct_count'],
                        retention_summary['gained_correct_count'],
                        source_baseline_summary['top1_mcml'],
                        val_summary['top1_mcml'],
                        source_baseline_small_summary['top1_mcml'],
                        small_val_summary['top1_mcml'],
                        ','.join(failed_checks) if failed_checks else 'none'))
                if s7_temporal_mode:
                    print(
                        '[source-temporal-gate] epoch={} dfr={}->{} '
                        'aci={}->{} transitions={}'.format(
                            epoch,
                            source_baseline_summary[
                                'top1_dfr_percent_per_frame'],
                            val_summary['top1_dfr_percent_per_frame'],
                            source_baseline_summary['top1_aci'],
                            val_summary['top1_aci'],
                            val_summary[
                                'top1_temporal_transition_count']))
        if progress_path is not None:
            print('[source-progress] epoch={} nonfinite_replacements={} '
                  'out={}'.format(
                      epoch, progress_replacements, progress_path))
    if not os.path.isfile(best_path):
        raise RuntimeError('No source-selected labeller checkpoint')
    frozen_parameters_unchanged = bool(
        frozen_parameter_versions == {
            name: int(parameter._version)
            for name, parameter in heads.named_parameters()
            if not parameter.requires_grad})
    if not frozen_parameters_unchanged:
        raise RuntimeError('A frozen detector-head parameter changed')
    return (best_path, best_epoch, best_summary, best_small_summary,
            source_sampling, source_baseline_summary,
            source_baseline_small_summary, frozen_parameters_unchanged,
            history, s7_quality_support_audit,
            source_initial_temporal_summary,
            source_initial_temporal_small_summary)


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    head_device = torch.device('cuda:{}'.format(args.head_gpu))
    dino_devices = [torch.device('cuda:{}'.format(gpu))
                    for gpu in args.dino_gpus]
    dino_device = dino_devices[0]

    formal_records = formal_source_records(args)
    if formal_records is None:
        source_records = [
            row for row in common.discover_labeled_records(
                args.data_root, args.source_split, 0)
            if row['seq'] == args.source_seq]
        source_train, source_val = split_source_records(
            source_records, args.source_val_modulus)
        source_protocol = dict(
            mode='legacy_single_sequence_modulus_split',
            source_split=args.source_split, source_seq=args.source_seq,
            source_val_modulus=int(args.source_val_modulus))
    else:
        source_train, source_val = formal_records
        source_protocol = dict(
            mode='official_source_train_and_val_splits',
            train_datasets=list(args.source_train_datasets),
            val_datasets=list(args.source_val_datasets))
    source_conflict_spec = None
    if getattr(args, 'source_conflict_result_json', None):
        source_conflict_spec = load_source_conflict_spec(
            args.source_conflict_result_json, args.source_conflict_epoch)
        records_by_key = {
            '{}|{}|{}'.format(
                record['split'], record['seq'], int(record['frame'])): record
            for record in source_val}
        missing = sorted(
            set(source_conflict_spec['frame_keys']) - set(records_by_key))
        if missing:
            raise RuntimeError(
                'Source conflict frames are absent from validation: {}'.format(
                    ', '.join(missing)))
        source_val = [records_by_key[key]
                      for key in source_conflict_spec['frame_keys']]
        source_protocol['bounded_conflict_audit'] = dict(
            result_json=source_conflict_spec['result_json'],
            epoch=source_conflict_spec['epoch'],
            frame_count=len(source_val))
    targets = [] if args.skip_target_eval else target_records(args)
    if targets:
        assert_training_target_isolation(
            list(source_train) + list(source_val), targets)

    dino, loaded_patch_size = common.load_frozen_dinov2(
        args.dinov2_repo, args.dinov2_checkpoint,
        args.dinov2_model, dino_devices,
        args.legacy_sdpa_query_chunk)
    if int(loaded_patch_size) != int(args.patch_size):
        raise RuntimeError('Unexpected DINO patch size')
    dino_versions = common.module_parameter_versions(dino)
    in_channels = int(getattr(dino, 'embed_dim', 0))
    if in_channels <= 0:
        sample_feature, _meta, _cached = extract_or_load_feature(
            dino, source_train[0], args, dino_device)
        in_channels = int(sample_feature.shape[1])
    heads = FrozenDinoRotatedHeads(in_channels, args).to(head_device)
    trainable_names = configure_trainable_components(
        heads, args.train_components)
    source_val_rows = None
    s7_quality_support_audit = None
    source_initial_temporal_summary = None
    source_initial_temporal_small_summary = None
    source_temporal_attribution_audit = None
    source_temporal_immediate_override_audit = None
    current_source_small_summary = None

    if args.eval_only_checkpoint:
        best_path = args.eval_only_checkpoint
        payload = torch.load(best_path, map_location='cpu')
        validate_checkpoint(payload, in_channels, args)
        if (source_conflict_spec is not None
                and int(payload.get('epoch', -1))
                != int(source_conflict_spec['epoch'])):
            raise RuntimeError(
                'Conflict checkpoint epoch {} does not match requested epoch {}'
                .format(payload.get('epoch'), source_conflict_spec['epoch']))
        if (bool(getattr(
                args, 'source_temporal_attribution_audit', False)
                or getattr(args, 'source_temporal_immediate_override_audit',
                            False))
                and int(payload.get('epoch', -1)) != int(getattr(
                    args, 'source_temporal_attribution_epoch', 4))):
            raise RuntimeError(
                'Temporal attribution checkpoint epoch {} does not match '
                'the fixed requested epoch {}'.format(
                    payload.get('epoch'),
                    args.source_temporal_attribution_epoch))
        load_heads_checkpoint_state(heads, payload)
        best_epoch = int(payload.get('best_epoch', payload.get('epoch', 0)))
        best_source_summary = payload.get('best_source_val_summary')
        best_source_small_summary = payload.get(
            'best_source_small_val_summary')
        source_sampling = payload.get('source_sampling')
        source_baseline_summary = payload.get(
            'source_baseline_val_summary')
        source_baseline_small_summary = payload.get(
            'source_baseline_small_val_summary')
        frozen_head_parameters_unchanged = True
        history = []
        # Recompute source validation with the current inference rule.  The
        # checkpoint's stored summary may predate the valid-content filter.
        source_val_rows = evaluate_records(
            dino, heads, source_val, args, dino_device, head_device,
            role='source_validation')
        current_source_summary = summarize_rows(source_val_rows)
        if bool(getattr(
                args, 'source_temporal_attribution_audit', False)
                or getattr(args, 'source_temporal_immediate_override_audit',
                            False)):
            if (not source_sampling
                    or source_sampling.get('short_token_threshold') is None):
                raise RuntimeError(
                    'Temporal attribution checkpoint has no source-small '
                    'sampling definition')
            small_keys = {
                (record['split'], record['seq'], int(record['frame']))
                for record in source_small_records(
                    source_val, args,
                    source_sampling['short_token_threshold'])}
            source_small_rows = [
                row for row in source_val_rows
                if (row['split'], row['seq'], int(row['frame']))
                in small_keys]
            current_source_small_summary = summarize_rows(source_small_rows)
            source_temporal_attribution_audit = (
                build_source_temporal_attribution_audit(
                    current_source_summary, current_source_small_summary,
                    args, best_path, int(payload.get('epoch', -1))))
            if bool(getattr(
                    args, 'source_temporal_immediate_override_audit', False)):
                source_temporal_immediate_override_audit = dict(
                    mode=(
                        'fixed_rejected_checkpoint_source_only_recursive_'
                        'immediate_override'),
                    checkpoint=os.path.abspath(best_path),
                    checkpoint_epoch=int(payload.get('epoch', -1)),
                    checkpoint_selected_for_deployment=False,
                    parameter_update=False, checkpoint_selection=False,
                    best_epoch_selection=False, target_read=False,
                    configured_min_confirmations=int(
                        args.s7_temporal_min_confirmations),
                    runtime_min_confirmations=temporal_runtime_min_confirmations(
                        args),
                    override_condition=(
                        'candidate_margin_ok_and_candidate_continuity_ok'),
                    native_fallback=True,
                    source_full_summary=current_source_summary,
                    source_small_summary=current_source_small_summary,
                    readonly_attribution=(
                        source_temporal_attribution_audit),
                    formal_gate_reference=dict(
                        full_min_top1=int(
                            getattr(args, 's7_source_min_full_top1', 688)),
                        small_min_top1=int(
                            getattr(args, 's7_source_min_small_top1', 311)),
                        max_mcml=int(
                            getattr(args, 's7_source_max_mcml', 3))),
                    next_stage=(
                        'run_stage3_source_only_student_training_after_'
                        'source_gate'))
    else:
        (best_path, best_epoch,
         best_source_summary, best_source_small_summary,
         source_sampling, source_baseline_summary,
         source_baseline_small_summary,
         frozen_head_parameters_unchanged, history,
         s7_quality_support_audit,
         source_initial_temporal_summary,
         source_initial_temporal_small_summary) = train_source_only(
            dino, heads, source_train, source_val, args,
            dino_device, head_device, in_channels)
        payload = torch.load(best_path, map_location='cpu')
        validate_checkpoint(payload, in_channels, args)
        load_heads_checkpoint_state(heads, payload)
        current_source_summary = best_source_summary

    source_val_results_path = None
    if args.source_val_results_out:
        if source_val_rows is None:
            source_val_rows = evaluate_records(
                dino, heads, source_val, args, dino_device, head_device,
                role='source_validation')
        write_detection_rows_pickle(
            source_val_rows, args.source_val_results_out)
        source_val_results_path = os.path.abspath(
            args.source_val_results_out)

    # The paper training stage does not read target data.  Legacy diagnostic
    # runs may still evaluate target only after the source checkpoint is fixed.
    if args.skip_target_eval:
        target_rows = None
        target_summary = None
        decision = (
            'SOURCE_ONLY_TEMPORAL_IMMEDIATE_OVERRIDE_AUDIT_COMPLETE_'
            'TARGET_NOT_READ'
            if source_temporal_immediate_override_audit is not None else
            'SOURCE_ONLY_TEMPORAL_ATTRIBUTION_AUDIT_COMPLETE_'
            'TARGET_NOT_READ'
            if source_temporal_attribution_audit is not None else
            'SOURCE_ONLY_CONFLICT_AUDIT_COMPLETE_TARGET_NOT_READ'
            if source_conflict_spec is not None else
            'SOURCE_ONLY_TRAINING_SKIPPED_ZERO_'
            'S7_QUALITY_RISK_SUPPORT_TARGET_NOT_READ'
            if (s7_quality_support_audit is not None
                and not s7_quality_support_audit['training_allowed']) else
            'SOURCE_ONLY_SOURCE_GATE_FAILED_NATIVE_FALLBACK_'
            'TARGET_NOT_READ'
            if (args.train_components == 's7_temporal_association'
                and int(best_epoch) == 0) else
            'SOURCE_ONLY_TRAINING_COMPLETE_TARGET_NOT_READ')
    else:
        target_rows = evaluate_records(
            dino, heads, targets, args, dino_device, head_device,
            role='target_dev_diagnosis_only')
        target_summary = summarize_rows(target_rows)
        decision = make_target_decision(
            target_summary, args, source_summary=current_source_summary)

    dino_unchanged = (
        dino_versions == common.module_parameter_versions(dino))
    if not dino_unchanged:
        raise RuntimeError('Frozen DINO parameter invariant failed')
    source_conflict_audit = None
    if source_conflict_spec is not None:
        rows_by_key = {
            source_frame_key(row): row for row in source_val_rows}
        source_conflict_audit = dict(source_conflict_spec)
        source_conflict_audit.update(
            checkpoint=os.path.abspath(args.eval_only_checkpoint),
            checkpoint_epoch=int(payload.get('epoch', -1)),
            target_read=False,
            rows=(
                [source_merge_conflict_row(rows_by_key[key], 'lost')
                 for key in source_conflict_spec['lost_frame_keys']]
                + [source_merge_conflict_row(rows_by_key[key], 'gained')
                   for key in source_conflict_spec['gained_frame_keys']]))
    payload = dict(
        labeller=LABELLER_NAME, protocol_version=PROTOCOL_VERSION,
        paper=PAPER_URL, paper_code=PAPER_CODE_URL,
        related_work=dict(
            lidere=dict(
                url=LIDERE_PAPER_URL,
                current_scope='inspired_lightweight_residual_readout',
                faithful_interpolation_attention_implemented=False)),
        dinov2_checkpoint=os.path.abspath(args.dinov2_checkpoint),
        source_selected_checkpoint=os.path.abspath(best_path),
        protocol=dict(
            architecture=(
                'frozen_DINOv2_native_S14_plus_protected_residual_S7_RPN_'
                'to_source_only_causal_multi_cue_candidate_quality_association'
                if (args.train_components == 's7_temporal_association'
                    and bool(getattr(
                        args, 's7_temporal_quality_head', False))) else
                'frozen_DINOv2_native_S14_plus_protected_residual_S7_RPN_'
                'to_source_only_causal_multi_cue_candidate_association'
                if args.train_components == 's7_temporal_association' else
                'frozen_DINOv2_native_S14_plus_protected_residual_S7_RPN_'
                'to_lane_wide_non_positive_source_quality_suppression'
                if args.train_components == 's7_quality_suppression' else
                'frozen_DINOv2_native_S14_plus_protected_residual_S7_RPN_'
                'to_source_aware_pre_NMS_lane_arbitration'
                if args.train_components == 's7_lane_arbitration' else
                'frozen_DINOv2_native_S14_plus_protected_residual_S7_RPN_'
                'to_source_aware_pre_NMS_affine_merge'
                if args.train_components == 's7_merge' else
                'frozen_DINOv2_native_S14_plus_protected_residual_S7_RPN_'
                'to_native_RotatedROIAlign7x7'
                if args.train_components == 's7_rpn' else
                'frozen_DINOv2_single_scale_to_OrientedRPN_'
                'to_RotatedROIAlign7x7_to_Shared2FC_cls_and_OBB_reg'),
            source_data=source_protocol,
            checkpoint_selection=(
                'source_validation_only_with_exact_retention_small_top1_'
                'nonregression_absolute_gates_and_proposal_sensitive_'
                'selection_key_improvement'
                if args.train_components in (
                    's7_rpn', 's7_merge', 's7_lane_arbitration',
                    's7_quality_suppression', 's7_temporal_association') else
                'source_validation_only_with_exact_retention_small_top1_'
                'strict_improvement_and_mcml_nonregression'
                if args.train_components == 'roi_cls_pairwise_v2' else
                'source_validation_only_with_exact_retention_and_'
                'roi_cls_small_control'
                if args.train_components == 'roi_cls_pairwise' else
                'source_validation_only_with_roi_cls_small_control'
                if args.train_components == 'roi_cls' else
                'source_validation_only_over_fixed_candidate_epochs'),
            selection_epochs=[int(value)
                              for value in args.selection_epochs],
            training_schedule=dict(
                train_components=str(args.train_components),
                optimization_loss_components=(
                    optimization_loss_component_names(
                        args.train_components,
                        quality_head=bool(getattr(
                            args, 's7_temporal_quality_head', False)),
                        relative_quality=bool(getattr(
                            args, 's7_temporal_relative_quality', False)))),
                epochs=int(args.epochs), optimizer='SGD',
                lr=float(args.lr), momentum=float(args.momentum),
                weight_decay=float(args.weight_decay),
                max_grad_norm=float(args.max_grad_norm),
                warmup='linear', warmup_iters=int(args.warmup_iters),
                warmup_ratio=float(args.warmup_ratio),
                lr_policy='step',
                lr_steps=[int(value) for value in args.lr_steps],
                lr_gamma=float(args.lr_gamma),
                checkpoint_interval=int(args.checkpoint_interval),
                pairwise=(
                    None if args.train_components not in (
                        'roi_cls_pairwise', 'roi_cls_pairwise_v2')
                    else dict(
                        version=(2 if args.train_components ==
                                 'roi_cls_pairwise_v2' else 1),
                        margin=float(args.pairwise_margin),
                        cls_loss_weight=float(
                            args.pairwise_cls_loss_weight),
                        pairwise_loss_weight=float(
                            args.pairwise_loss_weight),
                        retention_loss_weight=float(
                            args.retention_loss_weight),
                        retention_temperature=float(
                            args.retention_temperature),
                        negative_riou_thr=float(
                            args.pairwise_negative_riou_thr),
                        nms_iou_thr=float(args.pairwise_nms_iou_thr),
                        negatives_per_positive=int(
                            args.pairwise_negatives_per_positive),
                        source_exact_retention_max_drop=int(
                            args.source_retain_max_top1_drop)))),
                s7_merge=(
                    None if args.train_components != 's7_merge' else dict(
                        component_checkpoint=(
                            None if not args.s7_component_checkpoint else
                            os.path.abspath(args.s7_component_checkpoint)),
                        proposal_sources=['native_s14', 'supplement_s7'],
                        calibration_stage='ROI_foreground_log_odds_before_NMS',
                        native_nms_protected=True,
                        affine_scale_positive=True,
                        initial_bias=float(args.s7_merge_init_bias),
                        margin=float(args.s7_merge_margin),
                        retention_weight=float(
                            args.s7_merge_retention_weight),
                        gain_weight=float(args.s7_merge_gain_weight),
                        prior_weight=float(args.s7_merge_prior_weight))),
                s7_lane_arbitration=(
                    None if args.train_components != 's7_lane_arbitration'
                    else dict(
                        base_checkpoint=os.path.abspath(args.init_checkpoint),
                        base_epoch=int(args.s7_lane_base_epoch),
                        hard_negatives=int(args.s7_lane_hard_negatives),
                        gain_repeat=int(args.s7_lane_gain_repeat),
                        hard_negative_ranking=(
                            'current_adjusted_s7_log_odds'),
                        gain_competitors=[
                            'native_s14', 'wrong_supplement_s7'],
                        source_train_only=True)),
                s7_quality_suppression=(
                    None if args.train_components != 's7_quality_suppression'
                    else dict(
                        base_checkpoint=os.path.abspath(args.init_checkpoint),
                        base_epoch=int(args.s7_quality_base_epoch),
                        lane_wide=True,
                        adjustment_range=[
                            -float(args.s7_quality_max_suppression), 0.0],
                        initial_risk_bias=float(
                            args.s7_quality_init_risk_bias),
                        top_candidate_stage='fixed_affine_S7_log_odds',
                        risk_label=(
                            'native_top_correct_and_S7_top_wrong_within_'
                            'source_margin'),
                        preserve_label='S7_top_RIoU_at_least_threshold',
                        preflight=dict(
                            exact_training_risk_miner=True,
                            source_train_only=True,
                            minimum_risk_pairs=S7_QUALITY_MIN_RISK_PAIRS,
                            zero_risk_action=(
                                'skip_optimization_and_keep_epoch_0')),
                        positive_promotion=False,
                        gain_replay=False,
                        source_train_only=True)),
                s7_temporal_association=(
                    None if args.train_components != 's7_temporal_association'
                    else dict(
                        base_checkpoint=(
                            None if not args.init_checkpoint else
                            os.path.abspath(args.init_checkpoint)),
                        base_epoch=int(
                            args.s7_temporal_relative_base_epoch
                            if bool(getattr(
                                args, 's7_temporal_relative_quality', False))
                            else args.s7_temporal_base_epoch),
                        cues=list(
                            temporal.QUALITY_CUE_NAMES
                            if bool(getattr(
                                args, 's7_temporal_quality_head', False))
                            else temporal.CUE_NAMES),
                        learned_parameter_count=(
                            len(temporal.QUALITY_CUE_NAMES)
                            if bool(getattr(
                                args, 's7_temporal_quality_head', False))
                            else len(temporal.CUE_NAMES)),
                        candidate_quality_head=bool(getattr(
                            args, 's7_temporal_quality_head', False)),
                        quality_head_hidden=int(getattr(
                            args, 's7_temporal_quality_hidden', 128)),
                        quality_loss_weight=float(getattr(
                            args, 's7_temporal_quality_loss_weight', 1.0)),
                        relative_quality=bool(getattr(
                            args, 's7_temporal_relative_quality', False)),
                        relative_quality_weight=float(getattr(
                            args, 's7_temporal_relative_quality_weight', 0.5)),
                        relative_quality_margin=float(getattr(
                            args, 's7_temporal_relative_quality_margin', 0.25)),
                        relative_quality_min_gap=float(getattr(
                            args, 's7_temporal_relative_quality_min_gap', 0.10)),
                        relative_quality_max_pairs=int(getattr(
                            args, 's7_temporal_relative_quality_max_pairs', 128)),
                        max_candidates=int(args.s7_temporal_max_candidates),
                        min_confirmations=int(
                            args.s7_temporal_min_confirmations),
                        override_margin=float(
                            args.s7_temporal_override_margin),
                        max_center_distance=float(
                            args.s7_temporal_max_center_distance),
                        min_rotated_iou=float(args.s7_temporal_min_riou),
                        min_appearance_similarity=float(
                            args.s7_temporal_min_appearance),
                        dfr_aci_angle_limit_deg=(
                            SOURCE_TEMPORAL_ANGLE_LIMIT_DEG),
                        training_state=(
                            'dense_source_candidate_max_riou_plus_relative_rank'
                            if bool(getattr(
                                args, 's7_temporal_relative_quality', False))
                            else 'dense_source_candidate_max_riou'
                            if bool(getattr(
                                args, 's7_temporal_quality_head', False))
                            else 'previous_source_GT_usable_candidate'),
                        inference_state=(
                            'strictly_previous_selected_candidate'),
                        native_fallback=True,
                        target_read=False)),
            source_min_top1_rate=float(args.source_min_top1_rate),
            deployment_score_thr=float(args.deployment_score_thr),
            border_margin_ratio=float(args.border_margin_ratio),
            valid_content_filter=dict(
                rule='all_four_obb_corners_inside_original_image',
                tolerance=float(args.valid_content_tolerance),
                applied_to=['source_validation',
                            'target_dev_diagnosis_only'],
                uses_annotations=False),
            raw_filter_comparison=dict(
                reporting_only=True, changes_returned_detections=False,
                changes_checkpoint_selection=False,
                raw_candidate_stage='roi_output_before_valid_content_filter'),
            target_labels_first_used=(
                'not_read' if args.skip_target_eval
                else 'after_source_checkpoint_fixed'),
            target_dev_role=('not_read_during_source_only_training'
                             if args.skip_target_eval else 'diagnosis_only'),
            pseudo_label_student_training=False,
            reason=(
                'Only the paper labeller component is authorized; no separate '
                'unlabelled target-train split was supplied.')),
        isolation=dict(
            dino_frozen=True, dino_parameters_unchanged=dino_unchanged,
            read_only_evaluation=bool(
                source_temporal_attribution_audit is not None
                or source_temporal_immediate_override_audit is not None),
            parameter_updates_performed=bool(
                not args.eval_only_checkpoint),
            initialization_checkpoint=(
                None if not args.init_checkpoint
                else os.path.abspath(args.init_checkpoint)),
            s7_component_checkpoint=(
                None if not getattr(args, 's7_component_checkpoint', None)
                else os.path.abspath(args.s7_component_checkpoint)),
            train_components=str(args.train_components),
            trainable_parameter_names=trainable_names,
            trainable_parameter_count=int(sum(
                parameter.numel() for parameter in heads.parameters()
                if parameter.requires_grad)),
            frozen_head_parameters_unchanged=bool(
                frozen_head_parameters_unchanged),
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_labels_used_for_evaluation_only=bool(
                not args.skip_target_eval)),
        architecture=dict(
            in_channels=in_channels, patch_size=int(args.patch_size),
            rpn=rpn_config(in_channels, args),
            roi=roi_config(in_channels, args),
            s7=s7_architecture(args),
            s7_rpn=(s7_rpn_config(
                int(getattr(args, 's7_channels', 128)), args)
                if bool(getattr(args, 's7_residual', False)) else None)),
        source=dict(
            train_count=len(source_train), val_count=len(source_val),
            best_epoch=int(best_epoch),
            best_validation_summary=best_source_summary,
            best_small_validation_summary=best_source_small_summary,
            baseline_validation_summary=source_baseline_summary,
            baseline_small_validation_summary=source_baseline_small_summary,
            initial_temporal_validation_summary=(
                source_initial_temporal_summary),
            initial_temporal_small_validation_summary=(
                source_initial_temporal_small_summary),
            small_sampling=source_sampling,
            current_inference_validation_summary=current_source_summary,
            current_inference_small_validation_summary=(
                current_source_small_summary),
            current_inference_rule='valid_rotated_obb_corners',
            s7_quality_support_audit=s7_quality_support_audit,
            history=history,
            source_val_results_pickle=source_val_results_path),
        source_conflict_audit=source_conflict_audit,
        source_temporal_attribution_audit=(
            source_temporal_attribution_audit),
        source_temporal_immediate_override_audit=(
            source_temporal_immediate_override_audit),
        target_dev=(None if target_summary is None else dict(
            summary=target_summary, rows=target_rows)),
        decision=decision)
    replacements = common.write_json_atomic(args.out_json, payload)
    if target_summary is None:
        print('[dino-labeller] {}'.format(decision))
    else:
        print('[dino-labeller] {} top1={}/{} mcml={} r100={}'.format(
            decision, target_summary['top1_hits'],
            target_summary['frame_count'], target_summary['top1_mcml'],
            target_summary['recall_at_100']))
    if source_val_results_path:
        print('[source-val-results] {}'.format(source_val_results_path))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
