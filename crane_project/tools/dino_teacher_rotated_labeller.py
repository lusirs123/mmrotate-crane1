#!/usr/bin/env python3
"""Train a source-only rotated detector on a frozen DINOv2 feature map.

This is the bounded SymEOOD adaptation of the CVPR 2025 DINO Teacher
labeller.  A frozen single-scale DINOv2 feature map feeds an Oriented RPN and
an oriented two-FC ROI box head.  Only the RPN and ROI head are optimized.
Source validation selects the checkpoint; target-dev annotations are first
read after the source-selected checkpoint has been fixed.
"""

import argparse
import collections
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
from crane_project.utils import rotated_geometry_quality as geometry  # noqa: E402
from crane_project.utils import s7_temporal_association as temporal  # noqa: E402


LABELLER_NAME = 'Frozen DINOv2 Oriented RPN/ROI Source Labeller V1'
PROTOCOL_VERSION = 23
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
                 's7_temporal_association', 's7_temporal_student',
                 's7_static_domain_ranker', 's7_selective_promotion',
                 's7_highres_roi_ranker'],
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
              'source-only same-frame pairwise ranking term on that head. '
              's7_temporal_student freezes the source-gated phase-2 teacher '
              'and trains only a copied student quality head with source GT, '
              'relative ranking, and teacher distillation. '
              's7_static_domain_ranker freezes the proposal lanes and trains '
              'one non-temporal source-only quality residual with static '
              'feature-domain augmentation. s7_selective_promotion freezes '
              'the phase-2 quality teacher and trains a native-vs-S7 pair '
              'head with uncertainty-aware abstention. '
              's7_highres_roi_ranker adds one lightweight stride-7 ROI '
              'quality readout and trains same-frame listwise ranking with '
              'native protection. --s7-highres-unified-ranking changes this '
              'stage to whole-pool hard-pair ranking while retaining the '
              'same native-protected inference contract.'))
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
    parser.add_argument('--s7-temporal-student', action='store_true')
    parser.add_argument('--s7-student-base-epoch', type=int, default=4)
    parser.add_argument('--s7-student-hidden', type=int, default=128)
    parser.add_argument('--s7-student-quality-loss-weight', type=float,
                        default=1.0)
    parser.add_argument('--s7-student-relative-loss-weight', type=float,
                        default=0.5)
    parser.add_argument('--s7-student-distillation-weight', type=float,
                        default=1.0)
    parser.add_argument('--s7-student-distillation-temperature', type=float,
                        default=1.0)
    parser.add_argument('--s7-student-small-loss-weight', type=float,
                        default=2.0)
    parser.add_argument('--s7-student-small-token-thr', type=float,
                        default=4.0)
    parser.add_argument(
        '--s7-student-teacher-result-json', default=None,
        help=('Stage-2 source result used to verify that the copied stage-3 '
              'teacher reproduces its source summaries before optimization.'))
    parser.add_argument(
        '--s7-static-domain-ranker', action='store_true',
        help=('Enable one source-only static domain-generalized candidate '
              'ranker. It has no temporal state, target read, or slice '
              'routing and only applies a learned score-logit residual.'))
    parser.add_argument('--s7-static-base-epoch', type=int, default=4)
    parser.add_argument('--s7-static-hidden', type=int, default=128)
    parser.add_argument('--s7-static-quality-loss-weight', type=float,
                        default=1.0)
    parser.add_argument('--s7-static-relative-loss-weight', type=float,
                        default=0.5)
    parser.add_argument('--s7-static-relative-margin', type=float, default=0.25)
    parser.add_argument('--s7-static-relative-min-gap', type=float, default=0.10)
    parser.add_argument('--s7-static-relative-max-pairs', type=int, default=128)
    parser.add_argument('--s7-static-score-weight', type=float, default=1.0)
    parser.add_argument('--s7-static-rank-margin', type=float, default=0.25)
    parser.add_argument('--s7-static-retention-weight', type=float, default=2.0)
    parser.add_argument('--s7-static-gain-weight', type=float, default=1.0)
    parser.add_argument('--s7-static-prior-weight', type=float, default=0.01)
    parser.add_argument('--s7-static-max-candidates', type=int, default=100)
    parser.add_argument(
        '--s7-static-aug-prob', type=float, default=0.75,
        help='Probability of applying deterministic source feature-domain '
             'brightness/blur/scale augmentation per training frame.')
    parser.add_argument(
        '--s7-static-aug-strength', type=float, default=0.15,
        help='Strength of the source-only feature-domain augmentation.')
    parser.add_argument(
        '--s7-static-teacher-result-json', default=None,
        help=('Phase-2 source result JSON used only for provenance and '
              'checkpoint integrity; it is never used as target data.'))
    parser.add_argument('--s7-selective-promotion', action='store_true')
    parser.add_argument(
        '--s7-selective-two-frame', action='store_true',
        help=('Use the lightweight V2 scalar-only two-frame constant-velocity '
              'ranker. The legacy V1 selective head remains unchanged when '
              'this flag is absent.'))
    parser.add_argument('--s7-selective-base-epoch', type=int, default=4)
    parser.add_argument('--s7-selective-hidden', type=int, default=128)
    parser.add_argument('--s7-selective-initial-uncertainty', type=float,
                        default=0.5)
    parser.add_argument('--s7-selective-advantage-gap', type=float,
                        default=0.10)
    parser.add_argument('--s7-selective-promotion-margin', type=float,
                        default=0.10)
    parser.add_argument('--s7-selective-uncertainty-multiplier', type=float,
                        default=1.0)
    parser.add_argument('--s7-selective-quality-loss-weight', type=float,
                        default=1.0)
    parser.add_argument('--s7-selective-classification-loss-weight',
                        type=float, default=1.0)
    parser.add_argument('--s7-selective-retention-weight', type=float,
                        default=2.0)
    parser.add_argument('--s7-selective-gain-weight', type=float, default=1.0)
    parser.add_argument('--s7-selective-prior-weight', type=float,
                        default=0.01)
    parser.add_argument('--s7-selective-max-candidates', type=int, default=100)
    parser.add_argument('--s7-selective-min-gain-sequences', type=int,
                        default=2)
    parser.add_argument('--s7-selective-aug-prob', type=float, default=0.75)
    parser.add_argument('--s7-selective-aug-strength', type=float, default=0.15)
    parser.add_argument(
        '--s7-selective-teacher-result-json', default=None,
        help=('Phase-2 source result used only to verify the frozen quality '
              'teacher and initialization checkpoint.'))
    parser.add_argument(
        '--s7-highres-roi-ranker', action='store_true',
        help=('Train the first static high-resolution ROI quality readout. '
              'It uses the frozen S7 feature map, no extra DINO forward, and '
              'does not read target data.'))
    parser.add_argument('--s7-highres-base-epoch', type=int, default=4)
    parser.add_argument('--s7-highres-hidden', type=int, default=32)
    parser.add_argument('--s7-highres-channels', type=int, default=32)
    parser.add_argument('--s7-highres-max-candidates', type=int, default=32)
    parser.add_argument('--s7-highres-score-weight', type=float, default=1.0)
    parser.add_argument('--s7-highres-rank-margin', type=float, default=0.25)
    parser.add_argument('--s7-highres-promotion-margin', type=float, default=0.25)
    parser.add_argument('--s7-highres-quality-loss-weight', type=float, default=1.0)
    parser.add_argument('--s7-highres-relative-loss-weight', type=float, default=0.5)
    parser.add_argument('--s7-highres-relative-min-gap', type=float, default=0.10)
    parser.add_argument('--s7-highres-relative-max-pairs', type=int, default=128)
    parser.add_argument('--s7-highres-retention-weight', type=float, default=2.0)
    parser.add_argument('--s7-highres-gain-weight', type=float, default=1.0)
    parser.add_argument('--s7-highres-prior-weight', type=float, default=0.01)
    parser.add_argument(
        '--s7-highres-unified-ranking', action='store_true',
        help=('Use one source-only hard-pair objective over the complete '
              'native-top1 plus S7-top-k pool. The flag is inference-critical '
              'and must be repeated when loading the resulting checkpoint.'))
    parser.add_argument('--s7-highres-unified-hard-pairs', type=int, default=8)
    parser.add_argument('--s7-highres-unified-aug-prob', type=float,
                        default=0.75)
    parser.add_argument('--s7-highres-unified-aug-strength', type=float,
                        default=0.15)
    parser.add_argument(
        '--s7-highres-pairwise-takeover-v2', action='store_true',
        help=('Use the source-only relative delta-RIoU takeover head with '
              'uncertainty-aware native abstention.'))
    parser.add_argument('--s7-takeover-initial-uncertainty', type=float,
                        default=0.25)
    parser.add_argument('--s7-takeover-uncertainty-multiplier', type=float,
                        default=2.0)
    parser.add_argument('--s7-takeover-margin', type=float, default=0.05)
    parser.add_argument('--s7-takeover-retention-margin', type=float,
                        default=0.10)
    parser.add_argument('--s7-takeover-delta-weight', type=float, default=1.0)
    parser.add_argument('--s7-takeover-classification-weight', type=float,
                        default=1.0)
    parser.add_argument('--s7-takeover-ranking-weight', type=float, default=0.5)
    parser.add_argument('--s7-takeover-retention-weight', type=float,
                        default=4.0)
    parser.add_argument('--s7-takeover-gain-weight', type=float, default=2.0)
    parser.add_argument('--s7-takeover-consistency-weight', type=float,
                        default=0.5)
    parser.add_argument('--s7-takeover-prior-weight', type=float, default=0.01)
    parser.add_argument('--s7-takeover-ranking-min-gap', type=float,
                        default=0.05)
    parser.add_argument('--s7-takeover-max-ranking-pairs', type=int, default=64)
    parser.add_argument('--s7-takeover-group-dro-eta', type=float, default=0.01)
    parser.add_argument(
        '--s7-highres-teacher-result-json', default=None,
        help=('Phase-2 source result JSON used only for provenance and '
              'checkpoint integrity.'))
    parser.add_argument(
        '--source-highres-margin-audit', action='store_true',
        help=('Run one read-only source-validation pass and apply multiple '
              'promotion margins to the same frozen high-resolution logits.'))
    parser.add_argument(
        '--source-highres-margin-source-result-json', default=None,
        help=('Completed high-resolution training result used to lock the '
              'near-pass epoch checkpoint and source baselines.'))
    parser.add_argument(
        '--source-highres-margin-values', type=float, nargs='+', default=None)
    parser.add_argument('--source-highres-margin-epoch', type=int, default=3)
    parser.add_argument(
        '--source-smooth-geometry-rank-support-audit', action='store_true',
        help=('Run a read-only source audit of smooth Gaussian geometry '
              'surrogates on the frozen native-top1 plus S7-top-k pool. It '
              'does not train, select a checkpoint, read target data, or '
              'tune a deployment threshold.'))
    parser.add_argument(
        '--source-smooth-geometry-source-result-json', default=None,
        help=('Unified high-resolution source-only training result used to '
              'lock the epoch-3 checkpoint and source small sampling.'))
    parser.add_argument(
        '--source-smooth-geometry-min-gain-domains', type=int, default=2,
        help='Minimum source domains with native-wrong/S7-correct support.')
    parser.add_argument(
        '--source-smooth-geometry-min-gain-sequences', type=int, default=2,
        help='Minimum source sequences with native-wrong/S7-correct support.')
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
        's7_quality_suppression', 's7_temporal_association',
        's7_temporal_student', 's7_static_domain_ranker',
        's7_selective_promotion', 's7_highres_roi_ranker')
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
    student_mode = args.train_components == 's7_temporal_student'
    if bool(getattr(args, 's7_temporal_student', False)) != student_mode:
        raise ValueError(
            '--s7-temporal-student and its train-components mode must be '
            'enabled together during stage-3 training')
    temporal_enabled = bool(getattr(args, 's7_temporal_association', False))
    student_positive = (
        getattr(args, 's7_student_base_epoch', 4),
        getattr(args, 's7_student_hidden', 128),
        getattr(args, 's7_student_quality_loss_weight', 1.0),
        getattr(args, 's7_student_relative_loss_weight', 0.5),
        getattr(args, 's7_student_distillation_weight', 1.0),
        getattr(args, 's7_student_distillation_temperature', 1.0),
        getattr(args, 's7_student_small_loss_weight', 2.0),
        getattr(args, 's7_student_small_token_thr', 4.0))
    if student_mode and any(float(value) <= 0.0
                            for value in student_positive):
        raise ValueError('S7 temporal student settings must be positive')
    if student_mode and int(getattr(
            args, 's7_student_hidden', 128)) != int(getattr(
                args, 's7_temporal_quality_hidden', 128)):
        raise ValueError(
            'S7 temporal student hidden size must match the phase-2 teacher '
            'for exact-copy initialization')
    if student_mode and not (relative_quality and bool(getattr(
            args, 's7_temporal_quality_head', False))):
        raise ValueError(
            's7_temporal_student requires the phase-2 quality and relative '
            'quality architecture')
    static_mode = args.train_components == 's7_static_domain_ranker'
    if bool(getattr(args, 's7_static_domain_ranker', False)) != static_mode:
        raise ValueError(
            '--s7-static-domain-ranker and its train-components mode must be '
            'enabled together')
    static_positive = (
        getattr(args, 's7_static_base_epoch', 4),
        getattr(args, 's7_static_hidden', 128),
        getattr(args, 's7_static_quality_loss_weight', 1.0),
        getattr(args, 's7_static_relative_loss_weight', 0.5),
        getattr(args, 's7_static_relative_margin', 0.25),
        getattr(args, 's7_static_relative_min_gap', 0.10),
        getattr(args, 's7_static_relative_max_pairs', 128),
        getattr(args, 's7_static_score_weight', 1.0),
        getattr(args, 's7_static_rank_margin', 0.25),
        getattr(args, 's7_static_retention_weight', 2.0),
        getattr(args, 's7_static_gain_weight', 1.0),
        getattr(args, 's7_static_prior_weight', 0.01),
        getattr(args, 's7_static_max_candidates', 100))
    if static_mode and any(float(value) <= 0.0 for value in static_positive):
        raise ValueError('S7 static domain ranker settings must be positive')
    if not 0.0 <= float(getattr(args, 's7_static_aug_prob', 0.75)) <= 1.0:
        raise ValueError('--s7-static-aug-prob must be in [0, 1]')
    if not 0.0 <= float(getattr(args, 's7_static_aug_strength', 0.15)) <= 1.0:
        raise ValueError('--s7-static-aug-strength must be in [0, 1]')
    if static_mode and temporal_enabled:
        raise ValueError(
            's7_static_domain_ranker cannot use temporal association')
    if static_mode and (relative_quality or bool(getattr(
            args, 's7_temporal_quality_head', False)) or bool(getattr(
                args, 's7_temporal_student', False))):
        raise ValueError(
            's7_static_domain_ranker has its own static quality head; do not '
            'enable temporal quality/student flags')
    selective_mode = args.train_components == 's7_selective_promotion'
    if bool(getattr(args, 's7_selective_promotion', False)) != selective_mode:
        raise ValueError(
            '--s7-selective-promotion and its train-components mode must be '
            'enabled together')
    selective_two_frame = bool(getattr(
        args, 's7_selective_two_frame', False))
    if selective_two_frame and not selective_mode:
        raise ValueError(
            '--s7-selective-two-frame requires s7_selective_promotion mode')
    selective_positive = (
        getattr(args, 's7_selective_base_epoch', 4),
        getattr(args, 's7_selective_hidden', 128),
        getattr(args, 's7_selective_initial_uncertainty', 0.5),
        getattr(args, 's7_selective_advantage_gap', 0.10),
        getattr(args, 's7_selective_uncertainty_multiplier', 1.0),
        getattr(args, 's7_selective_quality_loss_weight', 1.0),
        getattr(args, 's7_selective_classification_loss_weight', 1.0),
        getattr(args, 's7_selective_retention_weight', 2.0),
        getattr(args, 's7_selective_gain_weight', 1.0),
        getattr(args, 's7_selective_prior_weight', 0.01),
        getattr(args, 's7_selective_max_candidates', 100),
        getattr(args, 's7_selective_min_gain_sequences', 2))
    if selective_mode and any(
            float(value) <= 0.0 for value in selective_positive):
        raise ValueError('S7 selective promotion settings must be positive')
    if float(getattr(args, 's7_selective_promotion_margin', 0.10)) < 0.0:
        raise ValueError('--s7-selective-promotion-margin must be non-negative')
    for name in ('s7_selective_aug_prob', 's7_selective_aug_strength'):
        value = float(getattr(args, name, 0.75 if name.endswith('prob') else 0.15))
        if not 0.0 <= value <= 1.0:
            raise ValueError('--{} must be in [0, 1]'.format(
                name.replace('_', '-')))
    if selective_mode and (temporal_enabled or static_mode or relative_quality
                           or bool(getattr(args, 's7_temporal_student', False))):
        raise ValueError(
            's7_selective_promotion has its own pair selector and cannot be '
            'combined with temporal/student/static ranker modes')
    if selective_two_frame:
        if int(getattr(args, 's7_selective_hidden', 128)) != 16:
            raise ValueError(
                'Two-frame selective promotion requires hidden_dim=16')
        if int(getattr(args, 's7_selective_max_candidates', 100)) != 20:
            raise ValueError(
                'Two-frame selective promotion requires S7 top-20')
    highres_mode = args.train_components == 's7_highres_roi_ranker'
    highres_unified = bool(getattr(
        args, 's7_highres_unified_ranking', False))
    highres_takeover_v2 = bool(getattr(
        args, 's7_highres_pairwise_takeover_v2', False))
    if highres_unified and not highres_mode:
        raise ValueError(
            '--s7-highres-unified-ranking requires '
            '--train-components s7_highres_roi_ranker')
    if highres_takeover_v2 and not highres_mode:
        raise ValueError(
            '--s7-highres-pairwise-takeover-v2 requires '
            '--train-components s7_highres_roi_ranker')
    if highres_takeover_v2 and highres_unified:
        raise ValueError('Pairwise Takeover V2 and unified V1 are exclusive')
    highres_margin_audit = bool(getattr(
        args, 'source_highres_margin_audit', False))
    smooth_geometry_audit = bool(getattr(
        args, 'source_smooth_geometry_rank_support_audit', False))
    if bool(getattr(args, 's7_highres_roi_ranker', False)) != highres_mode:
        raise ValueError(
            '--s7-highres-roi-ranker and its train-components mode must be '
            'enabled together')
    if highres_margin_audit and not highres_mode:
        raise ValueError(
            'High-resolution margin audit requires s7_highres_roi_ranker')
    if smooth_geometry_audit and not highres_mode:
        raise ValueError(
            'Smooth geometry audit requires s7_highres_roi_ranker')
    if smooth_geometry_audit and not highres_unified:
        raise ValueError(
            'Smooth geometry audit requires the unified high-resolution pool')
    if smooth_geometry_audit and highres_takeover_v2:
        raise ValueError(
            'Smooth geometry audit does not support Pairwise Takeover V2')
    if smooth_geometry_audit and highres_margin_audit:
        raise ValueError(
            'Smooth geometry audit cannot be combined with a margin audit')
    margin_values = getattr(args, 'source_highres_margin_values', None)
    if highres_margin_audit:
        if not margin_values:
            raise ValueError(
                'High-resolution margin audit requires margin values')
        margin_values = sorted(set(float(value) for value in margin_values))
        if any(value < 0.0 for value in margin_values):
            raise ValueError('High-resolution audit margins must be non-negative')
        if 0.25 not in margin_values:
            raise ValueError(
                'High-resolution audit must include the trained 0.25 margin')
        args.source_highres_margin_values = margin_values
    if smooth_geometry_audit:
        if (not args.eval_only_checkpoint or args.init_checkpoint
                or args.resume_checkpoint or not args.skip_target_eval):
            raise ValueError(
                'Smooth geometry audit requires one eval-only checkpoint, '
                'source-only evaluation, and no init/resume')
        if not os.path.isfile(args.eval_only_checkpoint):
            raise ValueError(
                'Smooth geometry audit checkpoint does not exist: {}'.format(
                    args.eval_only_checkpoint))
        result_json = getattr(
            args, 'source_smooth_geometry_source_result_json', None)
        if not result_json or not os.path.isfile(result_json):
            raise ValueError(
                'Smooth geometry audit requires the unified source-only '
                'training result JSON')
        if int(getattr(
                args, 'source_smooth_geometry_min_gain_domains', 2)) <= 0:
            raise ValueError(
                '--source-smooth-geometry-min-gain-domains must be positive')
        if int(getattr(
                args, 'source_smooth_geometry_min_gain_sequences', 2)) <= 0:
            raise ValueError(
                '--source-smooth-geometry-min-gain-sequences must be positive')
        args.source_smooth_geometry_audit_spec = (
            load_unified_highres_margin_audit_spec(
                result_json, args.eval_only_checkpoint, 3))
    highres_positive = (
        getattr(args, 's7_highres_base_epoch', 4),
        getattr(args, 's7_highres_hidden', 32),
        getattr(args, 's7_highres_channels', 32),
        getattr(args, 's7_highres_max_candidates', 32),
        getattr(args, 's7_highres_score_weight', 1.0),
        getattr(args, 's7_highres_rank_margin', 0.25),
        getattr(args, 's7_highres_promotion_margin', 0.25),
        getattr(args, 's7_highres_quality_loss_weight', 1.0),
        getattr(args, 's7_highres_relative_loss_weight', 0.5),
        getattr(args, 's7_highres_relative_min_gap', 0.10),
        getattr(args, 's7_highres_relative_max_pairs', 128),
        getattr(args, 's7_highres_retention_weight', 2.0),
        getattr(args, 's7_highres_gain_weight', 1.0),
        getattr(args, 's7_highres_prior_weight', 0.01),
        getattr(args, 's7_highres_unified_hard_pairs', 8),
        getattr(args, 's7_highres_unified_aug_prob', 0.75),
        getattr(args, 's7_highres_unified_aug_strength', 0.15))
    if highres_mode and any(float(value) <= 0.0 for value in highres_positive):
        raise ValueError('High-resolution ROI ranker settings must be positive')
    takeover_positive = (
        getattr(args, 's7_takeover_initial_uncertainty', 0.25),
        getattr(args, 's7_takeover_uncertainty_multiplier', 2.0),
        getattr(args, 's7_takeover_retention_margin', 0.10),
        getattr(args, 's7_takeover_delta_weight', 1.0),
        getattr(args, 's7_takeover_classification_weight', 1.0),
        getattr(args, 's7_takeover_ranking_weight', 0.5),
        getattr(args, 's7_takeover_retention_weight', 4.0),
        getattr(args, 's7_takeover_gain_weight', 2.0),
        getattr(args, 's7_takeover_consistency_weight', 0.5),
        getattr(args, 's7_takeover_prior_weight', 0.01),
        getattr(args, 's7_takeover_ranking_min_gap', 0.05),
        getattr(args, 's7_takeover_max_ranking_pairs', 64),
        getattr(args, 's7_takeover_group_dro_eta', 0.01))
    if highres_takeover_v2 and any(
            float(value) <= 0.0 for value in takeover_positive):
        raise ValueError('Pairwise Takeover V2 settings must be positive')
    if (highres_takeover_v2
            and float(getattr(args, 's7_takeover_margin', 0.05)) < 0.0):
        raise ValueError('Pairwise Takeover V2 margin must be non-negative')
    if (highres_takeover_v2
            and int(getattr(args, 's7_highres_max_candidates', 0)) != 64):
        raise ValueError('Pairwise Takeover V2 locks the S7 candidate pool to 64')
    if (highres_takeover_v2 and not math.isclose(
            float(args.deployment_score_thr), 0.05,
            rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError('Pairwise Takeover V2 locks deployment score to 0.05')
    if highres_mode and (temporal_enabled or static_mode or selective_mode
                         or relative_quality
                         or bool(getattr(args, 's7_temporal_student', False))):
        raise ValueError(
            'High-resolution ROI ranker is a standalone same-frame source '
            'stage and cannot be combined with temporal/static/selective modes')
    if (bool(getattr(args, 's7_lane_arbitration', False))
            and (args.train_components == 's7_quality_suppression'
                 or bool(getattr(args, 's7_quality_suppression', False)))):
        raise ValueError(
            'S7 positive lane arbitration and non-positive quality '
            'suppression are mutually exclusive')
    if temporal_enabled != (args.train_components in (
            's7_temporal_association', 's7_temporal_student')):
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
                's7_temporal_association', 's7_temporal_student',
                's7_static_domain_ranker', 's7_selective_promotion',
                's7_highres_roi_ranker')
            else [5, 7])
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
    if highres_margin_audit:
        if conflict_json or readonly_temporal_audit:
            raise ValueError(
                'High-resolution margin audit cannot be combined with other '
                'source audits')
        if (not args.eval_only_checkpoint or args.init_checkpoint
                or args.resume_checkpoint or not args.skip_target_eval):
            raise ValueError(
                'High-resolution margin audit requires one eval-only '
                'checkpoint, source-only evaluation, and no init/resume')
        if not os.path.isfile(args.eval_only_checkpoint):
            raise ValueError(
                'High-resolution audit checkpoint does not exist: {}'.format(
                    args.eval_only_checkpoint))
        result_json = getattr(
            args, 'source_highres_margin_source_result_json', None)
        if not result_json or not os.path.isfile(result_json):
            raise ValueError(
                'High-resolution margin audit requires its completed source '
                'training result JSON')
        args.source_highres_margin_audit_spec = (
            (load_unified_highres_margin_audit_spec
             if highres_unified else load_highres_margin_audit_spec)(
                 result_json, args.eval_only_checkpoint,
                 int(args.source_highres_margin_epoch)))
    if args.init_checkpoint and not os.path.isfile(args.init_checkpoint):
        raise ValueError('Init checkpoint does not exist: {}'.format(
            args.init_checkpoint))
    if student_mode:
        teacher_result_json = getattr(
            args, 's7_student_teacher_result_json', None)
        if not teacher_result_json or not os.path.isfile(teacher_result_json):
            raise ValueError(
                's7_temporal_student requires the phase-2 source result JSON')
        with open(teacher_result_json, 'r') as handle:
            teacher_result = json.load(handle)
        teacher_source = teacher_result.get('source') or {}
        teacher_isolation = teacher_result.get('isolation') or {}
        if teacher_result.get('target_dev') is not None:
            raise ValueError(
                'Stage-3 teacher result must not contain target-dev output')
        if int(teacher_source.get('best_epoch', -1)) != int(
                args.s7_student_base_epoch):
            raise ValueError(
                'Stage-3 teacher result must select phase-2 epoch {}; found '
                '{}'.format(args.s7_student_base_epoch,
                             teacher_source.get('best_epoch')))
        if teacher_isolation.get('train_components') != (
                's7_temporal_association'):
            raise ValueError(
                'Stage-3 teacher result must come from phase-2 '
                's7_temporal_association training')
        selected = teacher_result.get('source_selected_checkpoint')
        if (not selected or not args.init_checkpoint
                or os.path.realpath(selected) != os.path.realpath(
                    args.init_checkpoint)):
            raise ValueError(
                'Stage-3 teacher result must select --init-checkpoint')
        if not teacher_source.get('best_validation_summary') or not (
                teacher_source.get('best_small_validation_summary')):
            raise ValueError(
                'Stage-3 teacher result lacks phase-2 source summaries')
        args.s7_student_teacher_result = teacher_result
    if static_mode:
        if (not args.init_checkpoint or args.resume_checkpoint
                or args.eval_only_checkpoint):
            raise ValueError(
                's7_static_domain_ranker requires a fresh phase-2 init '
                'checkpoint and cannot resume or run eval-only')
        teacher_result_json = getattr(
            args, 's7_static_teacher_result_json', None)
        if not teacher_result_json or not os.path.isfile(teacher_result_json):
            raise ValueError(
                's7_static_domain_ranker requires the phase-2 source result '
                'JSON')
        with open(teacher_result_json, 'r') as handle:
            teacher_result = json.load(handle)
        teacher_source = teacher_result.get('source') or {}
        teacher_isolation = teacher_result.get('isolation') or {}
        if teacher_result.get('target_dev') is not None:
            raise ValueError(
                'Static ranker phase-2 result must not contain target-dev '
                'output')
        if int(teacher_source.get('best_epoch', -1)) != int(
                args.s7_static_base_epoch):
            raise ValueError(
                'Static ranker phase-2 result must select epoch {}; found {}'
                .format(args.s7_static_base_epoch,
                        teacher_source.get('best_epoch')))
        if teacher_isolation.get('train_components') != (
                's7_temporal_association'):
            raise ValueError(
                'Static ranker must initialize from phase-2 '
                's7_temporal_association training')
        selected = teacher_result.get('source_selected_checkpoint')
        if (not selected or os.path.realpath(selected) != os.path.realpath(
                args.init_checkpoint)):
            raise ValueError(
                'Static ranker phase-2 result must select --init-checkpoint')
        if not teacher_source.get('best_validation_summary') or not (
                teacher_source.get('best_small_validation_summary')):
            raise ValueError(
                'Static ranker phase-2 result lacks source summaries')
        args.s7_static_teacher_result = teacher_result
    if selective_mode:
        if (not args.init_checkpoint or args.resume_checkpoint
                or args.eval_only_checkpoint):
            raise ValueError(
                's7_selective_promotion requires a fresh phase-2 init '
                'checkpoint and cannot resume or run eval-only')
        teacher_result_json = getattr(
            args, 's7_selective_teacher_result_json', None)
        if not teacher_result_json or not os.path.isfile(teacher_result_json):
            raise ValueError(
                's7_selective_promotion requires the phase-2 source result '
                'JSON')
        with open(teacher_result_json, 'r') as handle:
            teacher_result = json.load(handle)
        teacher_source = teacher_result.get('source') or {}
        teacher_isolation = teacher_result.get('isolation') or {}
        if teacher_result.get('target_dev') is not None:
            raise ValueError(
                'Selective promotion phase-2 result must not contain '
                'target-dev output')
        if int(teacher_source.get('best_epoch', -1)) != int(
                args.s7_selective_base_epoch):
            raise ValueError(
                'Selective promotion phase-2 result must select epoch {}; '
                'found {}'.format(args.s7_selective_base_epoch,
                                  teacher_source.get('best_epoch')))
        if teacher_isolation.get('train_components') != (
                's7_temporal_association'):
            raise ValueError(
                'Selective promotion must initialize from phase-2 '
                's7_temporal_association training')
        provenance = phase2_selected_checkpoint_provenance_gate(
            teacher_result, args.init_checkpoint,
            expected_epoch=int(args.s7_selective_base_epoch),
            min_full_top1=int(args.s7_source_min_full_top1),
            min_small_top1=int(args.s7_source_min_small_top1),
            max_mcml=int(args.s7_source_max_mcml))
        if not provenance['passed']:
            failed = sorted(name for name, passed
                            in provenance['checks'].items() if not passed)
            raise ValueError(
                'Selective promotion phase-2 checkpoint provenance gate '
                'failed: ' + ', '.join(failed))
        if not teacher_source.get('best_validation_summary') or not (
                teacher_source.get('best_small_validation_summary')):
            raise ValueError(
                'Selective promotion phase-2 result lacks source summaries')
        args.s7_selective_teacher_result = teacher_result
    if highres_mode and not (highres_margin_audit or smooth_geometry_audit):
        teacher_result_json = getattr(
            args, 's7_highres_teacher_result_json', None)
        if not teacher_result_json or not os.path.isfile(teacher_result_json):
            raise ValueError(
                's7_highres_roi_ranker requires the phase-2 source result '
                'JSON')
        with open(teacher_result_json, 'r') as handle:
            teacher_result = json.load(handle)
        if highres_takeover_v2:
            audit = teacher_result.get('source_highres_margin_audit') or {}
            protocol = teacher_result.get('protocol') or {}
            isolation = teacher_result.get('isolation') or {}
            candidate = teacher_result.get(
                'source_research_candidate_checkpoint')
            checks = dict(
                protocol_version=int(teacher_result.get(
                    'protocol_version', -1)) == 26,
                locked_decision=(teacher_result.get('decision') ==
                    'SOURCE_ONLY_UNIFIED_HIGHRES_BOUNDED_RISK_'
                    'RESEARCH_GATE_PASSED_TARGET_NOT_READ'),
                checkpoint_epoch=int(audit.get('checkpoint_epoch', -1))
                    == int(args.s7_highres_base_epoch),
                bounded_risk=audit.get('audit_variant')
                    == 'unified_bounded_risk',
                research_margin=float(teacher_result.get(
                    'research_candidate_promotion_margin', -1.0)) == 0.3,
                source_only=protocol.get('source_only') is True,
                target_not_read=(protocol.get('target_read') is False
                                 and teacher_result.get('target_dev') is None),
                read_only=(isolation.get('read_only_evaluation') is True
                           and isolation.get('parameter_updates_performed')
                           is False),
                no_target_use=(isolation.get('target_used_for_training')
                               is False and isolation.get(
                                   'target_used_for_checkpoint_selection')
                               is False),
                research_only=(teacher_result.get('source_safe') is False
                               and teacher_result.get(
                                   'eligible_for_deployment') is False
                               and teacher_result.get(
                                   'eligible_for_full_test') is False),
                checkpoint_identity=bool(candidate) and os.path.realpath(
                    str(candidate)) == os.path.realpath(
                        args.init_checkpoint))
            if not all(checks.values()):
                failed = sorted(name for name, passed in checks.items()
                                if not passed)
                raise ValueError(
                    'Pairwise Takeover V2 protocol-26 gate failed: '
                    + ', '.join(failed))
            args.s7_highres_teacher_result = teacher_result
        else:
            if teacher_result.get('target_dev') is not None:
                raise ValueError(
                    'High-resolution phase-2 result must not contain '
                    'target-dev output')
            if int((teacher_result.get('source') or {}).get(
                    'best_epoch', -1)) != int(args.s7_highres_base_epoch):
                raise ValueError(
                    'High-resolution ranker phase-2 result must select epoch '
                    '{}; found {}'.format(
                        args.s7_highres_base_epoch,
                        (teacher_result.get('source') or {}).get('best_epoch')))
            if (teacher_result.get('isolation') or {}).get(
                    'train_components') != 's7_temporal_association':
                raise ValueError(
                    'High-resolution ranker must initialize from phase-2 '
                    's7_temporal_association training')
            provenance = phase2_selected_checkpoint_provenance_gate(
                teacher_result, args.init_checkpoint,
                expected_epoch=int(args.s7_highres_base_epoch),
                min_full_top1=int(args.s7_source_min_full_top1),
                min_small_top1=int(args.s7_source_min_small_top1),
                max_mcml=int(args.s7_source_max_mcml))
            if not provenance['passed']:
                failed = sorted(
                    name for name, passed in provenance['checks'].items()
                    if not passed)
                raise ValueError(
                    'High-resolution phase-2 checkpoint provenance gate '
                    'failed: ' + ', '.join(failed))
            source = teacher_result.get('source') or {}
            if not source.get('best_validation_summary') or not source.get(
                    'best_small_validation_summary'):
                raise ValueError(
                    'High-resolution ranker phase-2 result lacks source '
                    'summaries')
            args.s7_highres_teacher_result = teacher_result
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
            's7_quality_suppression', 's7_temporal_association',
            's7_temporal_student', 's7_static_domain_ranker',
            's7_selective_promotion', 's7_highres_roi_ranker')
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
    two_frame_selective = bool(
        args.train_components == 's7_selective_promotion'
        and getattr(args, 's7_selective_two_frame', False))
    if (args.train_components in (
            's7_temporal_association', 's7_temporal_student')
            or two_frame_selective):
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
        if student_mode and (not args.init_checkpoint
                             or args.resume_checkpoint
                             or args.eval_only_checkpoint):
            raise ValueError(
                's7_temporal_student requires a fresh phase-2 init '
                'checkpoint and cannot resume or run eval-only')
        if relative_quality and not student_mode:
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
        if student_mode:
            if int(args.source_small_repeat) != 1:
                raise ValueError(
                    's7_temporal_student uses loss weighting and forbids '
                    'source frame repetition')
    if static_mode:
        if args.source_retain_max_top1_drop != 0:
            raise ValueError(
                's7_static_domain_ranker requires exact source retention')
        if args.s7_component_checkpoint:
            raise ValueError(
                's7_static_domain_ranker initializes from the complete '
                'phase-2 checkpoint; do not pass --s7-component-checkpoint')
        if not getattr(args, 'skip_target_eval', False):
            raise ValueError(
                's7_static_domain_ranker is source-only; pass '
                '--skip-target-eval and run target diagnosis separately')
        if not args.source_train_datasets or not args.source_val_datasets:
            raise ValueError(
                's7_static_domain_ranker requires formal source train/val '
                'datasets')
        if int(getattr(args, 'source_small_repeat', 1)) != 1:
            raise ValueError(
                's7_static_domain_ranker forbids source frame repetition')
        if int(getattr(args, 's7_source_min_full_top1', 677)) < 688:
            raise ValueError(
                's7_static_domain_ranker requires '
                '--s7-source-min-full-top1 >= 688')
        if int(getattr(args, 's7_source_min_small_top1', 303)) < 311:
            raise ValueError(
                's7_static_domain_ranker requires '
                '--s7-source-min-small-top1 >= 311')
        if int(getattr(args, 's7_source_max_mcml', 3)) > 3:
            raise ValueError(
                's7_static_domain_ranker requires '
                '--s7-source-max-mcml <= 3')
    if highres_mode:
        if (not (highres_margin_audit or smooth_geometry_audit)
                and (not args.init_checkpoint or args.resume_checkpoint
                     or args.eval_only_checkpoint)):
            raise ValueError(
                's7_highres_roi_ranker requires a fresh phase-2 init '
                'checkpoint and cannot resume or run eval-only')
        if args.source_retain_max_top1_drop != 0:
            raise ValueError(
                's7_highres_roi_ranker requires exact source retention')
        if args.s7_component_checkpoint:
            raise ValueError(
                's7_highres_roi_ranker initializes from the complete '
                'phase-2 checkpoint; do not pass --s7-component-checkpoint')
        if not getattr(args, 'skip_target_eval', False):
            raise ValueError(
                's7_highres_roi_ranker is source-only; pass '
                '--skip-target-eval until the source gate passes')
        if not args.source_train_datasets or not args.source_val_datasets:
            raise ValueError(
                's7_highres_roi_ranker requires formal source train/val '
                'datasets')
        if int(args.source_small_repeat) != 1:
            raise ValueError(
                's7_highres_roi_ranker forbids source frame repetition')
        if int(getattr(args, 's7_source_min_full_top1', 677)) < 688:
            raise ValueError(
                's7_highres_roi_ranker requires '
                '--s7-source-min-full-top1 >= 688')
        if int(getattr(args, 's7_source_min_small_top1', 303)) < 311:
            raise ValueError(
                's7_highres_roi_ranker requires '
                '--s7-source-min-small-top1 >= 311')
        if int(getattr(args, 's7_source_max_mcml', 3)) > 3:
            raise ValueError(
                's7_highres_roi_ranker requires '
                '--s7-source-max-mcml <= 3')
    if selective_mode:
        if args.source_retain_max_top1_drop != 0:
            raise ValueError(
                's7_selective_promotion requires exact source retention')
        if args.s7_component_checkpoint:
            raise ValueError(
                's7_selective_promotion initializes from the complete '
                'phase-2 checkpoint; do not pass --s7-component-checkpoint')
        if not getattr(args, 'skip_target_eval', False):
            raise ValueError(
                's7_selective_promotion is source-only; pass '
                '--skip-target-eval until its formal gate passes')
        if not args.source_train_datasets or not args.source_val_datasets:
            raise ValueError(
                's7_selective_promotion requires formal source train/val '
                'datasets')
        if int(getattr(args, 'source_small_repeat', 1)) != 1:
            raise ValueError(
                's7_selective_promotion forbids source frame repetition')
        if int(getattr(args, 's7_source_min_full_top1', 677)) < 688:
            raise ValueError(
                's7_selective_promotion requires '
                '--s7-source-min-full-top1 >= 688')
        if int(getattr(args, 's7_source_min_small_top1', 303)) < 311:
            raise ValueError(
                's7_selective_promotion requires '
                '--s7-source-min-small-top1 >= 311')
        if int(getattr(args, 's7_source_max_mcml', 3)) > 3:
            raise ValueError(
                's7_selective_promotion requires '
                '--s7-source-max-mcml <= 3')
    if args.train_components in (
            's7_rpn', 's7_merge', 's7_lane_arbitration',
            's7_quality_suppression', 's7_temporal_association',
            's7_temporal_student', 's7_static_domain_ranker',
            's7_selective_promotion', 's7_highres_roi_ranker') and args.epochs > 4:
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
    elif train_components == 's7_temporal_student':
        if (not getattr(heads, 's7_protected_merge', False)
                or getattr(heads, 's7_candidate_quality_head', None) is None
                or getattr(heads, 's7_candidate_student_head', None) is None):
            raise RuntimeError(
                'S7 temporal student requires the fixed phase-2 teacher and '
                'a separate student head')
        for parameter in heads.s7_candidate_student_head.parameters():
            parameter.requires_grad_(True)
    elif train_components == 's7_static_domain_ranker':
        if (not getattr(heads, 's7_protected_merge', False)
                or getattr(heads, 's7_candidate_static_head', None) is None):
            raise RuntimeError(
                'S7 static domain ranker requires the protected affine '
                'candidate pool and static quality head')
        for parameter in heads.s7_candidate_static_head.parameters():
            parameter.requires_grad_(True)
    elif train_components == 's7_selective_promotion':
        if (not getattr(heads, 's7_protected_merge', False)
                or getattr(heads, 's7_candidate_quality_head', None) is None
                or getattr(heads, 's7_selective_promotion_head', None) is None):
            raise RuntimeError(
                'S7 selective promotion requires the frozen phase-2 quality '
                'teacher and a separate pairwise promotion head')
        for parameter in heads.s7_selective_promotion_head.parameters():
            parameter.requires_grad_(True)
    elif train_components == 's7_highres_roi_ranker':
        if (not getattr(heads, 's7_protected_merge', False)
                or getattr(heads, 's7_highres_spatial_projection', None)
                is None
                or getattr(heads, 's7_highres_candidate_quality_head', None)
                is None):
            raise RuntimeError(
                'High-resolution ranker requires the protected S7 pool and '
                'its spatial/quality readout')
        takeover_v2 = bool(getattr(
            heads, 's7_highres_pairwise_takeover_v2', False))
        takeover_head = getattr(
            heads, 's7_highres_pairwise_takeover_head', None)
        if takeover_v2 and takeover_head is None:
            raise RuntimeError('Pairwise Takeover V2 head is missing')
        modules = (heads.s7_highres_spatial_projection,
                   takeover_head if takeover_v2 else
                   heads.s7_highres_candidate_quality_head)
        for module in modules:
            for parameter in module.parameters():
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
    if train_components == 's7_temporal_student':
        required = (
            'loss_s7_student_quality', 'loss_s7_student_relative',
            'loss_s7_student_distillation')
        missing = [name for name in required if name not in losses]
        if missing:
            raise RuntimeError('S7 student losses missing: {}'.format(
                ', '.join(missing)))
        return loss_total({name: losses[name] for name in required})
    if train_components == 's7_static_domain_ranker':
        required = (
            'loss_s7_static_quality', 'loss_s7_static_relative',
            'loss_s7_static_retention', 'loss_s7_static_gain',
            'loss_s7_static_prior')
        missing = [name for name in required if name not in losses]
        if missing:
            raise RuntimeError('S7 static ranker losses missing: {}'.format(
                ', '.join(missing)))
        return loss_total({name: losses[name] for name in required})
    if train_components == 's7_selective_promotion':
        required = (
            'loss_s7_selective_quality',
            'loss_s7_selective_classification',
            'loss_s7_selective_retention', 'loss_s7_selective_gain',
            'loss_s7_selective_prior')
        missing = [name for name in required if name not in losses]
        if missing:
            raise RuntimeError(
                'S7 selective promotion losses missing: {}'.format(
                    ', '.join(missing)))
        return loss_total({name: losses[name] for name in required})
    if train_components == 's7_highres_roi_ranker':
        takeover_required = (
            'loss_s7_takeover_delta',
            'loss_s7_takeover_classification',
            'loss_s7_takeover_ranking',
            'loss_s7_takeover_retention', 'loss_s7_takeover_gain',
            'loss_s7_takeover_consistency', 'loss_s7_takeover_prior')
        if any(name in losses for name in takeover_required):
            missing = [name for name in takeover_required if name not in losses]
            if missing:
                raise RuntimeError(
                    'Pairwise Takeover V2 losses missing: {}'.format(
                        ', '.join(missing)))
            return loss_total({name: losses[name]
                               for name in takeover_required})
        required = (
            'loss_s7_highres_quality', 'loss_s7_highres_relative',
            'loss_s7_highres_retention', 'loss_s7_highres_gain',
            'loss_s7_highres_prior')
        missing = [name for name in required if name not in losses]
        if missing:
            raise RuntimeError(
                'High-resolution ranker losses missing: {}'.format(
                    ', '.join(missing)))
        selected = {name: losses[name] for name in required}
        if 'loss_s7_highres_unified' in losses:
            selected['loss_s7_highres_unified'] = (
                losses['loss_s7_highres_unified'])
        return loss_total(selected)
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
        relative_quality: bool = False,
        unified_highres: bool = False,
        pairwise_takeover_v2: bool = False) -> List[str]:
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
    if train_components == 's7_temporal_student':
        return [
            'loss_s7_student_quality', 'loss_s7_student_relative',
            'loss_s7_student_distillation']
    if train_components == 's7_static_domain_ranker':
        return [
            'loss_s7_static_quality', 'loss_s7_static_relative',
            'loss_s7_static_retention', 'loss_s7_static_gain',
            'loss_s7_static_prior']
    if train_components == 's7_selective_promotion':
        return [
            'loss_s7_selective_quality',
            'loss_s7_selective_classification',
            'loss_s7_selective_retention', 'loss_s7_selective_gain',
            'loss_s7_selective_prior']
    if train_components == 's7_highres_roi_ranker':
        if pairwise_takeover_v2:
            return [
                'loss_s7_takeover_delta',
                'loss_s7_takeover_classification',
                'loss_s7_takeover_ranking',
                'loss_s7_takeover_retention', 'loss_s7_takeover_gain',
                'loss_s7_takeover_consistency', 'loss_s7_takeover_prior']
        names = [
            'loss_s7_highres_quality', 'loss_s7_highres_relative',
            'loss_s7_highres_retention', 'loss_s7_highres_gain',
            'loss_s7_highres_prior']
        if unified_highres:
            names.insert(-1, 'loss_s7_highres_unified')
        return names
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
    if bool(getattr(args, 's7_selective_two_frame', False)):
        train_sequences = {
            str(row.get('seq', ''))
            for row in train}
        val_sequences = {
            str(row.get('seq', ''))
            for row in val}
        if train_sequences & val_sequences:
            raise RuntimeError(
                'Two-frame source validation must use held-out sequences')
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
        from mmrotate.models.builder import build_head, build_roi_extractor

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
                    's7_quality_suppression', 's7_temporal_association',
                    's7_temporal_student', 's7_static_domain_ranker',
                    's7_selective_promotion', 's7_highres_roi_ranker'))
        self.s7_lane_arbitration = bool(getattr(
            args, 's7_lane_arbitration', False) or getattr(
                args, 'train_components', '') == 's7_lane_arbitration')
        self.s7_quality_suppression = bool(getattr(
            args, 's7_quality_suppression', False) or getattr(
                args, 'train_components', '') == 's7_quality_suppression')
        self.s7_temporal_association = bool(getattr(
            args, 's7_temporal_association', False) or getattr(
            args, 'train_components', '') in (
                's7_temporal_association', 's7_temporal_student'))
        self.s7_temporal_student = bool(getattr(
            args, 's7_temporal_student', False) or getattr(
                args, 'train_components', '') == 's7_temporal_student')
        self.s7_static_domain_ranker = bool(getattr(
            args, 's7_static_domain_ranker', False) or getattr(
                args, 'train_components', '') == 's7_static_domain_ranker')
        self.s7_selective_promotion = bool(getattr(
            args, 's7_selective_promotion', False) or getattr(
                args, 'train_components', '') == 's7_selective_promotion')
        self.s7_highres_roi_ranker = bool(getattr(
            args, 's7_highres_roi_ranker', False) or getattr(
            args, 'train_components', '') == 's7_highres_roi_ranker')
        self.s7_highres_unified_ranking = bool(
            getattr(args, 's7_highres_unified_ranking', False))
        self.s7_highres_pairwise_takeover_v2 = bool(
            getattr(args, 's7_highres_pairwise_takeover_v2', False))
        if self.s7_highres_unified_ranking and not self.s7_highres_roi_ranker:
            raise ValueError(
                'Unified high-resolution ranking requires the high-resolution '
                'ROI ranker')
        if (self.s7_highres_pairwise_takeover_v2
                and not self.s7_highres_roi_ranker):
            raise ValueError(
                'Pairwise Takeover V2 requires the high-resolution ROI ranker')
        self.s7_temporal_quality_head_enabled = bool(getattr(
            args, 's7_temporal_quality_head', False)
            or self.s7_selective_promotion)
        if (self.s7_temporal_quality_head_enabled
                and not (self.s7_temporal_association
                         or self.s7_selective_promotion)):
            raise ValueError(
                'Candidate quality head requires temporal association or '
                'selective promotion')
        self._last_candidate_merge = None
        self._last_temporal_pool = None
        self._last_static_pool = None
        self._last_selective_pool = None
        self._last_highres_pool = None
        self._last_highres_pool = None
        self._last_s7_feature = None
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
            self.s7_candidate_student_head = (
                temporal.S7CandidateQualityHead(
                    int(getattr(args, 'roi_fc_channels', 1024)),
                    int(getattr(args, 's7_student_hidden', 128)))
                if self.s7_temporal_student else None)
            self.s7_candidate_static_head = (
                temporal.S7CandidateQualityHead(
                    int(getattr(args, 'roi_fc_channels', 1024)),
                    int(getattr(args, 's7_static_hidden', 128)))
                if self.s7_static_domain_ranker else None)
            self.s7_selective_promotion_head = (
                (temporal.S7SmallTemporalRankerHead(
                    int(getattr(args, 's7_selective_hidden', 16)),
                    float(getattr(
                        args, 's7_selective_initial_uncertainty', 0.5)))
                 if bool(getattr(args, 's7_selective_two_frame', False))
                 else temporal.S7SelectivePromotionHead(
                     int(getattr(args, 'roi_fc_channels', 1024)),
                     int(getattr(args, 's7_selective_hidden', 128)),
                     float(getattr(
                         args, 's7_selective_initial_uncertainty', 0.5))))
                if self.s7_selective_promotion else None)
            if self.s7_highres_roi_ranker:
                highres_channels = int(getattr(
                    args, 's7_highres_channels', 32))
                highres_stride = int(args.patch_size) // 2
                self.s7_highres_roi_extractor = build_roi_extractor(
                    ConfigDict(dict(
                        type='RotatedSingleRoIExtractor',
                        roi_layer=dict(
                            type='RoIAlignRotated', out_size=3,
                            sample_num=2, clockwise=True),
                        out_channels=highres_channels,
                        featmap_strides=[highres_stride])))
                groups = min(8, highres_channels)
                while highres_channels % groups != 0:
                    groups -= 1
                self.s7_highres_spatial_projection = nn.Sequential(
                    nn.Conv2d(s7_channels, highres_channels, 1, bias=False),
                    nn.GroupNorm(groups, highres_channels), nn.GELU())
                self.s7_highres_candidate_quality_head = (
                    temporal.S7HighResCandidateQualityHead(
                        int(getattr(args, 'roi_fc_channels', 1024)),
                        highres_channels,
                        int(getattr(args, 's7_highres_hidden', 32))))
                self.s7_highres_pairwise_takeover_head = (
                    temporal.S7HighResPairwiseTakeoverHead(
                        int(getattr(args, 'roi_fc_channels', 1024)),
                        highres_channels,
                        int(getattr(args, 's7_highres_hidden', 32)),
                        float(getattr(
                            args, 's7_takeover_initial_uncertainty', 0.25)))
                    if self.s7_highres_pairwise_takeover_v2 else None)
            else:
                self.s7_highres_roi_extractor = None
                self.s7_highres_spatial_projection = None
                self.s7_highres_candidate_quality_head = None
                self.s7_highres_pairwise_takeover_head = None
        else:
            self.s7_readout = None
            self.s7_rpn_head = None
            self.s7_proposal_cfg = None
            self.s7_score_calibrator = None
            self.s7_lane_arbitrator = None
            self.s7_quality_suppressor = None
            self.s7_temporal_scorer = None
            self.s7_candidate_quality_head = None
            self.s7_candidate_student_head = None
            self.s7_candidate_static_head = None
            self.s7_selective_promotion_head = None
            self.s7_highres_roi_extractor = None
            self.s7_highres_spatial_projection = None
            self.s7_highres_candidate_quality_head = None
            self.s7_highres_pairwise_takeover_head = None
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
        self._last_s7_feature = None
        native_features = self.feature_levels(feature)
        native = self.rpn_head.simple_test_rpn(
            native_features, [img_meta])[0]
        if not self.s7_inference_enabled():
            return native_features, dict(native_s14=native)
        s7 = self.s7_feature(feature)
        self._last_s7_feature = s7
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

    def _highres_roi_embeddings(self, feature: torch.Tensor,
                                img_meta: Dict, detections: torch.Tensor,
                                rescale: bool,
                                s7_feature_override: Optional[
                                    torch.Tensor] = None) -> torch.Tensor:
        """Read stride-7 spatial evidence for a bounded candidate subset."""
        from mmrotate.core import rbbox2roi

        extractor = getattr(self, 's7_highres_roi_extractor', None)
        projection = getattr(self, 's7_highres_spatial_projection', None)
        if extractor is None or projection is None:
            raise RuntimeError('High-resolution ROI readout is not configured')
        if detections.shape[0] == 0:
            channels = int(getattr(self._args, 's7_highres_channels', 32))
            return detections.new_zeros((0, channels))
        s7 = s7_feature_override
        if s7 is None:
            s7 = getattr(self, '_last_s7_feature', None)
        if s7 is None:
            s7 = self.s7_feature(feature)
        boxes = detections[:, :5].clone()
        if rescale:
            scale = torch.as_tensor(
                img_meta['scale_factor'][:4], device=boxes.device,
                dtype=boxes.dtype)
            boxes[:, :4] = boxes[:, :4] * scale.reshape(1, 4)
        rois = rbbox2roi([boxes])
        spatial = projection(s7)
        pooled = extractor([spatial], rois)
        return pooled.mean(dim=(2, 3))

    def _highres_active_indices(self, detections: torch.Tensor,
                                source_ids: torch.Tensor) -> torch.Tensor:
        native = torch.nonzero(source_ids == 0, as_tuple=False).flatten()
        s7 = torch.nonzero(source_ids == 1, as_tuple=False).flatten()
        if not native.numel():
            return s7[:int(getattr(
                self._args, 's7_highres_max_candidates', 32))]
        native_top = native[torch.argmax(detections[native, 5])]
        s7 = s7[:min(int(getattr(
            self._args, 's7_highres_max_candidates', 32)), int(s7.numel()))]
        return torch.cat((native_top.reshape(1), s7), dim=0)

    def _protected_merge_detections(self, feature: torch.Tensor,
                                    img_meta: Dict, rescale: bool = True,
                                    apply_static_ranker: Optional[bool] = None,
                                    apply_selective_promotion: Optional[
                                        bool] = None,
                                    apply_highres_ranker: Optional[
                                        bool] = None):
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
        base_detections = detections
        base_embeddings = embeddings
        base_source_ids = source_ids
        highres_selection = None
        self._last_highres_pool = None
        if bool(getattr(self, 's7_highres_roi_ranker', False)):
            highres_head = getattr(
                self, 's7_highres_candidate_quality_head', None)
            if highres_head is None:
                raise RuntimeError('High-resolution quality head is missing')
            active_indices = self._highres_active_indices(
                base_detections, base_source_ids)
            active_detections = base_detections[active_indices]
            active_embeddings = base_embeddings[active_indices]
            active_sources = base_source_ids[active_indices]
            highres_embeddings = self._highres_roi_embeddings(
                feature, img_meta, active_detections, rescale=rescale)
            pairwise_prediction = None
            if self.s7_highres_pairwise_takeover_v2:
                takeover_head = getattr(
                    self, 's7_highres_pairwise_takeover_head', None)
                if takeover_head is None:
                    raise RuntimeError('Pairwise Takeover V2 head is missing')
                pairwise_prediction = takeover_head(
                    active_embeddings, highres_embeddings,
                    active_detections, active_sources)
                highres_quality_logits = active_detections.new_zeros(
                    (active_detections.shape[0],))
            else:
                highres_quality_logits = highres_head(
                    active_embeddings, highres_embeddings, active_detections,
                    active_sources)
            self._last_highres_pool = dict(
                detections=active_detections.detach(),
                embeddings=active_embeddings.detach(),
                highres_embeddings=highres_embeddings.detach(),
                source_ids=active_sources.detach(),
                base_indices=active_indices.detach(),
                quality_logits=highres_quality_logits.detach(),
                pairwise_prediction=(
                    None if pairwise_prediction is None else dict(
                        native_index=pairwise_prediction['native_index'],
                        s7_indices=pairwise_prediction['s7_indices'].detach(),
                        mean=pairwise_prediction['mean'].detach(),
                        raw_mean=pairwise_prediction['raw_mean'].detach(),
                        uncertainty=(
                            pairwise_prediction['uncertainty'].detach()))),
                candidate_limit=int(getattr(
                    self._args, 's7_highres_max_candidates', 32)))
            if apply_highres_ranker is None:
                apply_highres_ranker = not self.training
            if bool(apply_highres_ranker) and active_detections.shape[0]:
                if self.s7_highres_pairwise_takeover_v2:
                    highres_selection = (
                        temporal.native_protected_pairwise_highres_takeover(
                            pairwise_prediction, active_detections,
                            active_sources,
                            max_candidates=int(getattr(
                                self._args, 's7_highres_max_candidates', 64)),
                            uncertainty_multiplier=float(getattr(
                                self._args,
                                's7_takeover_uncertainty_multiplier', 2.0)),
                            takeover_margin=float(getattr(
                                self._args, 's7_takeover_margin', 0.05)),
                            deployment_score_thr=float(getattr(
                                self._args, 'deployment_score_thr', 0.05))))
                elif self.s7_highres_unified_ranking:
                    highres_selection = (
                        temporal.native_protected_unified_highres_ranking_from_logits(
                            highres_quality_logits, active_detections,
                            active_sources,
                            max_candidates=int(getattr(
                                self._args, 's7_highres_max_candidates', 32)),
                            score_weight=float(getattr(
                                self._args, 's7_highres_score_weight', 1.0)),
                            promotion_margin=float(getattr(
                                self._args, 's7_highres_promotion_margin',
                                0.25))))
                else:
                    highres_selection = (
                        temporal.native_protected_highres_promotion_from_logits(
                            highres_quality_logits, active_detections,
                            active_sources,
                            max_candidates=int(getattr(
                                self._args, 's7_highres_max_candidates', 32)),
                            score_weight=float(getattr(
                                self._args, 's7_highres_score_weight', 1.0)),
                            promotion_margin=float(getattr(
                                self._args, 's7_highres_promotion_margin',
                                0.25))))
                if (highres_selection.get('selected_index') is not None
                        and (self.s7_highres_pairwise_takeover_v2
                             or highres_selection.get('reason') !=
                             'native_fallback_zero_residual')):
                    selected = active_indices[highres_selection['order']]
                    selected_first = selected[:1]
                    remaining = torch.arange(
                        base_detections.shape[0],
                        device=base_detections.device)
                    keep_remaining = torch.ones(
                        base_detections.shape[0], dtype=torch.bool,
                        device=base_detections.device)
                    keep_remaining[selected_first] = False
                    remaining = remaining[keep_remaining]
                    order = torch.cat((selected_first, remaining), dim=0)
                    detections = base_detections[order]
                    embeddings = base_embeddings[order]
                    source_ids = base_source_ids[order]
        static_quality_logits = None
        if bool(getattr(self, 's7_static_domain_ranker', False)):
            static_head = getattr(self, 's7_candidate_static_head', None)
            if static_head is None:
                raise RuntimeError('Static ranker head is missing')
            static_quality_logits = static_head(
                embeddings, detections, source_ids)
            if apply_static_ranker is None:
                apply_static_ranker = not self.training
            if apply_static_ranker and detections.shape[0]:
                static_limit = min(
                    int(getattr(self._args, 's7_static_max_candidates', 100)),
                    int(detections.shape[0]))
                active = detections[:static_limit]
                active_embeddings = embeddings[:static_limit]
                active_source_ids = source_ids[:static_limit]
                active_quality = static_quality_logits[:static_limit]
                scores = active[:, 5].clamp(1e-6, 1.0 - 1e-6)
                score_logits = torch.log(scores) - torch.log1p(-scores)
                adjusted = score_logits + float(getattr(
                    self._args, 's7_static_score_weight', 1.0)) * (
                        active_quality.detach())
                active = active.clone()
                active[:, 5] = torch.sigmoid(adjusted)
                order = torch.argsort(active[:, 5], descending=True)
                active = active[order]
                active_embeddings = active_embeddings[order]
                active_source_ids = active_source_ids[order]
                active_quality = active_quality[order]
                detections = torch.cat([active, detections[static_limit:]], 0)
                embeddings = torch.cat(
                    [active_embeddings, embeddings[static_limit:]], 0)
                source_ids = torch.cat(
                    [active_source_ids, source_ids[static_limit:]], 0)
                static_quality_logits = torch.cat(
                    [active_quality, static_quality_logits[static_limit:]], 0)
        selective_quality_logits = None
        selective_selection = None
        if bool(getattr(self, 's7_selective_promotion', False)):
            quality_head = getattr(self, 's7_candidate_quality_head', None)
            promotion_head = getattr(self, 's7_selective_promotion_head', None)
            if quality_head is None or promotion_head is None:
                raise RuntimeError(
                    'Selective promotion quality/pair head is missing')
            selective_quality_logits = quality_head(
                base_embeddings, base_detections, base_source_ids).detach()
            self._last_selective_pool = dict(
                detections=base_detections.detach(),
                embeddings=base_embeddings.detach(),
                source_ids=base_source_ids.detach(),
                quality_logits=selective_quality_logits,
                candidate_limit=int(getattr(
                    self._args, 's7_selective_max_candidates', 100)))
            if apply_selective_promotion is None:
                apply_selective_promotion = not self.training
            two_frame_selective = bool(getattr(
                self._args, 's7_selective_two_frame', False))
            if (bool(apply_selective_promotion) and not two_frame_selective
                    and detections.shape[0]):
                selective_selection = (
                    temporal.native_protected_selective_promotion(
                        promotion_head, base_embeddings, base_detections,
                        base_source_ids, selective_quality_logits,
                        max_candidates=int(getattr(
                            self._args, 's7_selective_max_candidates', 100)),
                        uncertainty_multiplier=float(getattr(
                            self._args,
                            's7_selective_uncertainty_multiplier', 1.0)),
                        promotion_margin=float(getattr(
                            self._args,
                            's7_selective_promotion_margin', 0.10))))
                selective_order = selective_selection['order']
                detections = base_detections[selective_order]
                embeddings = base_embeddings[selective_order]
                source_ids = base_source_ids[selective_order]
        quality_logits = None
        teacher_quality_logits = None
        if (bool(getattr(self, 's7_temporal_quality_head_enabled', False))
                and getattr(self, 's7_candidate_quality_head', None) is not None):
            teacher_quality_logits = self.s7_candidate_quality_head(
                embeddings, detections, source_ids).detach()
            quality_logits = teacher_quality_logits
            if bool(getattr(self, 's7_temporal_student', False)):
                student_head = getattr(
                    self, 's7_candidate_student_head', None)
                if student_head is None:
                    raise RuntimeError('Temporal student head is missing')
                quality_logits = student_head(
                    embeddings, detections, source_ids).detach()
        self._last_temporal_pool = dict(
            detections=detections.detach(),
            embeddings=embeddings.detach(),
            source_ids=source_ids.detach(),
            quality_logits=quality_logits,
            teacher_quality_logits=teacher_quality_logits,
            candidate_limit=int(getattr(
                self._args, 's7_temporal_max_candidates', 100)))
        self._last_static_pool = dict(
            detections=(base_detections.detach()
                        if not bool(apply_static_ranker) else detections.detach()),
            embeddings=(base_embeddings.detach()
                        if not bool(apply_static_ranker) else embeddings.detach()),
            source_ids=(base_source_ids.detach()
                        if not bool(apply_static_ranker) else source_ids.detach()),
            quality_logits=(static_quality_logits.detach()
                            if static_quality_logits is not None else None),
            candidate_limit=int(getattr(
                self._args, 's7_static_max_candidates', 100)))
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
                torch.sigmoid(quality_risk_logit.detach()).item()),
            s7_selective_promotion=(
                None if selective_selection is None else dict(
                    selected_index=selective_selection['selected_index'],
                    native_index=selective_selection['native_index'],
                    promoted=bool(selective_selection['promoted']),
                    reason=str(selective_selection['reason']),
                    best_lower_bound=selective_selection['best_lower_bound'],
                    best_advantage=selective_selection['best_advantage'],
                    best_uncertainty=selective_selection['best_uncertainty'],
                    candidate_count=int(
                        selective_selection['candidate_count']))),
            s7_highres_roi_ranker=(
                None if highres_selection is None else dict(
                    selected_index=highres_selection.get('selected_index'),
                    native_index=highres_selection.get('native_index'),
                    promoted=bool(highres_selection.get('promoted', False)),
                    reason=str(highres_selection.get('reason', '')),
                    candidate_count=int(highres_selection.get(
                        'candidate_count', 0)),
                    eligible_count=int(highres_selection.get(
                        'eligible_count', 0)),
                    best_advantage=highres_selection.get('best_advantage'),
                    best_uncertainty=highres_selection.get(
                        'best_uncertainty'),
                    best_lower_bound=highres_selection.get(
                        'best_lower_bound'),
                    deployment_score=highres_selection.get(
                        'deployment_score'))))
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

    def initialize_temporal_student_from_teacher(self):
        """Copy the selected phase-2 teacher into the stage-3 student."""
        teacher = getattr(self, 's7_candidate_quality_head', None)
        student = getattr(self, 's7_candidate_student_head', None)
        if teacher is None or student is None:
            raise RuntimeError(
                'Temporal teacher/student heads are not configured')
        student.load_state_dict(teacher.state_dict(), strict=True)

    def forward_s7_temporal_student_train(
            self, feature: torch.Tensor, img_meta: Dict,
            gt_boxes: torch.Tensor, riou_thr: float,
            quality_weight: float, relative_weight: float,
            relative_margin: float, relative_min_gap: float,
            relative_max_pairs: int, distillation_weight: float,
            distillation_temperature: float, small_loss_weight: float,
            small_token_threshold: float, max_candidates: int) -> Dict:
        """Fit only the stage-3 source student on the fixed phase-2 pool."""
        from mmcv.ops import box_iou_rotated

        teacher = getattr(self, 's7_candidate_quality_head', None)
        student = getattr(self, 's7_candidate_student_head', None)
        if (not self.s7_protected_merge
                or self.s7_score_calibrator is None
                or teacher is None or student is None):
            raise RuntimeError(
                'Temporal student training requires the fixed phase-2 '
                'teacher and protected affine candidate pool')
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
            if gt_boxes.shape[0]:
                short_tokens = float(
                    gt_boxes[:, 2:4].abs().min().item()) / float(
                        getattr(self._args, 'patch_size', 14))
            else:
                short_tokens = float('inf')
            frame_weight = (
                float(small_loss_weight)
                if short_tokens <= float(small_token_threshold) else 1.0)
        losses = temporal.candidate_student_losses(
            student, teacher, embeddings, detections, source_ids, overlap,
            riou_threshold=riou_thr,
            quality_weight=quality_weight,
            relative_weight=relative_weight,
            relative_margin=relative_margin,
            relative_min_gap=relative_min_gap,
            relative_max_pairs=relative_max_pairs,
            distillation_weight=distillation_weight,
            distillation_temperature=distillation_temperature,
            supervised_frame_weight=frame_weight)
        losses['s7_student_small_source_frame'] = int(frame_weight > 1.0)
        losses['s7_student_short_tokens'] = (
            0.0 if not math.isfinite(short_tokens) else short_tokens)
        return losses

    def forward_s7_static_domain_ranker_train(
            self, feature: torch.Tensor, img_meta: Dict,
            gt_boxes: torch.Tensor, riou_thr: float,
            quality_weight: float, relative_weight: float,
            relative_margin: float, relative_min_gap: float,
            relative_max_pairs: int, score_weight: float,
            rank_margin: float, retention_weight: float,
            gain_weight: float, prior_weight: float,
            max_candidates: int) -> Dict:
        """Fit one source-only static rank residual on the fixed S7 pool."""
        from mmcv.ops import box_iou_rotated

        quality_head = getattr(self, 's7_candidate_static_head', None)
        if (not self.s7_static_domain_ranker
                or not self.s7_protected_merge
                or self.s7_score_calibrator is None
                or quality_head is None):
            raise RuntimeError(
                'Static domain ranker requires the fixed affine pool and '
                'static quality head')
        with torch.no_grad():
            self._protected_merge_detections(
                feature, img_meta, rescale=False, apply_static_ranker=False)
            pool = self._last_static_pool
            if pool is None:
                raise RuntimeError('Static candidate pool was not produced')
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
        losses = temporal.static_candidate_rank_losses(
            quality_head, embeddings, detections, source_ids, overlap,
            riou_threshold=riou_thr,
            quality_weight=quality_weight,
            relative_weight=relative_weight,
            relative_margin=relative_margin,
            relative_min_gap=relative_min_gap,
            relative_max_pairs=relative_max_pairs,
            score_weight=score_weight,
            rank_margin=rank_margin,
            retention_weight=retention_weight,
            gain_weight=gain_weight,
            prior_weight=prior_weight)
        losses['s7_static_short_token_frame'] = 0
        return losses

    def forward_s7_selective_promotion_train(
            self, feature: torch.Tensor, img_meta: Dict,
            gt_boxes: torch.Tensor, riou_thr: float,
            advantage_gap: float, promotion_margin: float,
            uncertainty_multiplier: float, quality_weight: float,
            classification_weight: float, retention_weight: float,
            gain_weight: float, prior_weight: float,
            max_candidates: int,
            two_frame_state=None) -> Dict:
        """Fit the source-only native-vs-S7 conservative promotion head."""
        from mmcv.ops import box_iou_rotated

        promotion_head = getattr(self, 's7_selective_promotion_head', None)
        quality_head = getattr(self, 's7_candidate_quality_head', None)
        if (not self.s7_selective_promotion
                or not self.s7_protected_merge
                or self.s7_score_calibrator is None
                or promotion_head is None or quality_head is None):
            raise RuntimeError(
                'Selective promotion training requires the fixed phase-2 '
                'candidate pool, quality teacher, and promotion head')
        with torch.no_grad():
            self._protected_merge_detections(
                feature, img_meta, rescale=False,
                apply_selective_promotion=False)
            pool = self._last_selective_pool
            if pool is None:
                raise RuntimeError('Selective candidate pool was not produced')
            detections = pool['detections']
            embeddings = pool['embeddings']
            source_ids = pool['source_ids']
            quality_logits = pool['quality_logits']
            if gt_boxes.shape[0] and detections.shape[0]:
                overlap = box_iou_rotated(
                    detections[:, :5].float(), gt_boxes.float()).max(
                        dim=1).values
            else:
                overlap = detections.new_zeros((detections.shape[0],))
        if bool(getattr(self._args, 's7_selective_two_frame', False)):
            if two_frame_state is None:
                raise RuntimeError('Two-frame selective state is missing')
            losses = temporal.small_temporal_ranker_losses(
                promotion_head, embeddings, detections, source_ids,
                quality_logits, overlap, two_frame_state,
                riou_threshold=riou_thr, advantage_gap=advantage_gap,
                promotion_margin=promotion_margin,
                uncertainty_multiplier=uncertainty_multiplier,
                quality_weight=quality_weight,
                classification_weight=classification_weight,
                retention_weight=retention_weight, gain_weight=gain_weight,
                prior_weight=prior_weight, max_candidates=max_candidates)
            selected_index = losses.pop(
                '_s7_small_temporal_selected_index')
            losses['_s7_small_temporal_selected_box'] = (
                None if selected_index is None else
                detections[selected_index, :5].detach())
            losses['_s7_small_temporal_selected_embedding'] = (
                None if selected_index is None else
                embeddings[selected_index].detach())
            return losses
        return temporal.selective_promotion_losses(
            promotion_head, embeddings, detections, source_ids,
            quality_logits, overlap, riou_threshold=riou_thr,
            advantage_gap=advantage_gap,
            promotion_margin=promotion_margin,
            uncertainty_multiplier=uncertainty_multiplier,
            quality_weight=quality_weight,
            classification_weight=classification_weight,
            retention_weight=retention_weight, gain_weight=gain_weight,
            prior_weight=prior_weight, max_candidates=max_candidates)

    def forward_s7_highres_roi_ranker_train(
            self, feature: torch.Tensor, img_meta: Dict,
            gt_boxes: torch.Tensor, riou_thr: float,
            quality_weight: float, relative_weight: float,
            relative_margin: float, relative_min_gap: float,
            relative_max_pairs: int, score_weight: float,
            rank_margin: float, retention_weight: float,
            gain_weight: float, prior_weight: float,
            max_candidates: int,
            augmented_feature: Optional[torch.Tensor] = None) -> Dict:
        """Fit the lightweight stride-7 ROI quality readout on source GT."""
        from mmcv.ops import box_iou_rotated

        head = getattr(self, 's7_highres_candidate_quality_head', None)
        if (not self.s7_highres_roi_ranker
                or not self.s7_protected_merge
                or self.s7_score_calibrator is None or head is None):
            raise RuntimeError(
                'High-resolution ranker requires the protected S7 candidate '
                'pool and trainable quality readout')
        with torch.no_grad():
            self._protected_merge_detections(
                feature, img_meta, rescale=False,
                apply_highres_ranker=False)
            pool = self._last_highres_pool
            if pool is None:
                raise RuntimeError(
                    'High-resolution candidate pool was not produced')
            detections = pool['detections']
            embeddings = pool['embeddings']
            source_ids = pool['source_ids']
        highres_embeddings = self._highres_roi_embeddings(
            feature, img_meta, detections, rescale=False)
        if gt_boxes.shape[0] and detections.shape[0]:
            overlap = box_iou_rotated(
                detections[:, :5].float(), gt_boxes.float()).max(dim=1).values
        else:
            overlap = detections.new_zeros((detections.shape[0],))
        if self.s7_highres_pairwise_takeover_v2:
            takeover_head = getattr(
                self, 's7_highres_pairwise_takeover_head', None)
            if takeover_head is None:
                raise RuntimeError('Pairwise Takeover V2 head is missing')
            augmented_highres = None
            if augmented_feature is not None:
                with torch.no_grad():
                    augmented_s7 = self.s7_feature(augmented_feature)
                augmented_highres = self._highres_roi_embeddings(
                    augmented_feature, img_meta, detections, rescale=False,
                    s7_feature_override=augmented_s7)
            return temporal.pairwise_highres_takeover_losses(
                takeover_head, embeddings, highres_embeddings, detections,
                source_ids, overlap, riou_threshold=riou_thr,
                deployment_score_thr=float(getattr(
                    self._args, 'deployment_score_thr', 0.05)),
                uncertainty_multiplier=float(getattr(
                    self._args, 's7_takeover_uncertainty_multiplier', 2.0)),
                takeover_margin=float(getattr(
                    self._args, 's7_takeover_margin', 0.05)),
                retention_margin=float(getattr(
                    self._args, 's7_takeover_retention_margin', 0.10)),
                delta_weight=float(getattr(
                    self._args, 's7_takeover_delta_weight', 1.0)),
                classification_weight=float(getattr(
                    self._args, 's7_takeover_classification_weight', 1.0)),
                ranking_weight=float(getattr(
                    self._args, 's7_takeover_ranking_weight', 0.5)),
                retention_weight=float(getattr(
                    self._args, 's7_takeover_retention_weight', 4.0)),
                gain_weight=float(getattr(
                    self._args, 's7_takeover_gain_weight', 2.0)),
                consistency_weight=float(getattr(
                    self._args, 's7_takeover_consistency_weight', 0.5)),
                prior_weight=float(getattr(
                    self._args, 's7_takeover_prior_weight', 0.01)),
                ranking_min_gap=float(getattr(
                    self._args, 's7_takeover_ranking_min_gap', 0.05)),
                max_ranking_pairs=int(getattr(
                    self._args, 's7_takeover_max_ranking_pairs', 64)),
                augmented_highres_embedding=augmented_highres)
        loss_fn = (temporal.unified_highres_candidate_rank_losses
                   if self.s7_highres_unified_ranking
                   else temporal.highres_candidate_rank_losses)
        kwargs = dict(
            quality_weight=quality_weight, relative_weight=relative_weight,
            relative_margin=relative_margin, relative_min_gap=relative_min_gap,
            relative_max_pairs=relative_max_pairs, score_weight=score_weight,
            rank_margin=rank_margin, retention_weight=retention_weight,
            gain_weight=gain_weight, prior_weight=prior_weight)
        if self.s7_highres_unified_ranking:
            kwargs['hard_pair_count'] = int(getattr(
                self._args, 's7_highres_unified_hard_pairs', 8))
        return loss_fn(
            head, embeddings, highres_embeddings, detections, source_ids,
            overlap, riou_threshold=riou_thr, **kwargs)

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
        self._last_static_pool = None
        self._last_selective_pool = None
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


def static_source_feature_domain_augment(
        feature: torch.Tensor, args, seed: int):
    """Apply deterministic source-only brightness/blur/scale proxies.

    DINO features are cached and frozen, so the augmentation is deliberately
    feature-domain: it changes no labels, reads no target frame, and keeps
    the tensor shape and patch geometry required by the fixed detector heads.
    It is used by the static/selective rankers and the unified high-resolution
    source-view training path.
    """
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError('Static feature augmentation expects [1,C,H,W]')
    rng = random.Random(int(seed))
    selective = getattr(args, 'train_components', '') == (
        's7_selective_promotion')
    unified_highres = (
        getattr(args, 'train_components', '') == 's7_highres_roi_ranker'
        and (bool(getattr(args, 's7_highres_unified_ranking', False))
             or bool(getattr(
                 args, 's7_highres_pairwise_takeover_v2', False))))
    probability = float(getattr(
        args, ('s7_selective_aug_prob' if selective else
               's7_highres_unified_aug_prob' if unified_highres else
               's7_static_aug_prob'),
        0.75))
    strength = float(getattr(
        args, ('s7_selective_aug_strength' if selective else
               's7_highres_unified_aug_strength' if unified_highres else
               's7_static_aug_strength'), 0.15))
    if probability <= 0.0 or rng.random() >= probability:
        return feature, dict(applied=False, operations=[])
    import torch.nn.functional as functional

    augmented = feature
    operations = []
    # Contrast/offset is the frozen-feature analogue of brightness change.
    gain = 1.0 + rng.uniform(-strength, strength)
    mean = augmented.mean(dim=(2, 3), keepdim=True)
    std = augmented.std(
        dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
    offset = rng.uniform(-strength, strength) * std
    augmented = (augmented - mean) * gain + mean + offset
    operations.append('brightness')
    if rng.random() < 0.5:
        blur = functional.avg_pool2d(augmented, kernel_size=3,
                                     stride=1, padding=1)
        augmented = torch.lerp(augmented, blur, 0.25 + 0.5 * strength)
        operations.append('blur')
    if min(int(augmented.shape[-2]), int(augmented.shape[-1])) >= 2:
        ratio = 1.0 + rng.uniform(-strength, strength)
        height = max(1, int(round(float(augmented.shape[-2]) * ratio)))
        width = max(1, int(round(float(augmented.shape[-1]) * ratio)))
        resized = functional.interpolate(
            augmented, size=(height, width), mode='bilinear',
            align_corners=False)
        augmented = functional.interpolate(
            resized, size=augmented.shape[-2:], mode='bilinear',
            align_corners=False)
        operations.append('scale')
    return augmented.contiguous(), dict(applied=True, operations=operations)


def source_domain_label(record: Dict) -> str:
    explicit = record.get('domain')
    if explicit:
        value = str(explicit).lower()
        return 'sim' if 'sim' in value else 'real'
    identity = '{} {}'.format(
        record.get('split', ''), record.get('seq', '')).lower()
    return 'sim' if 'sim' in identity else 'real'


def ordered_source_training_records(
        records: Sequence[Dict], args, epoch: int) -> List[Dict]:
    """Preserve causal video order only for modes that consume history."""
    ordered = list(records)
    two_frame_selective = bool(
        args.train_components == 's7_selective_promotion'
        and getattr(args, 's7_selective_two_frame', False))
    if (args.train_components in (
            's7_temporal_association', 's7_temporal_student')
            or two_frame_selective):
        return sorted(
            ordered, key=lambda row: (
                str(row.get('split', '')), str(row.get('seq', '')),
                int(row.get('frame', -1))))
    if bool(getattr(args, 's7_highres_pairwise_takeover_v2', False)):
        by_domain = collections.defaultdict(list)
        for row in ordered:
            domain = source_domain_label(row)
            by_domain[domain].append(row)
        if len(by_domain) < 2:
            raise RuntimeError(
                'Pairwise Takeover V2 requires at least two source domains')
        largest = max(len(group) for group in by_domain.values())
        balanced = []
        for domain, group in sorted(by_domain.items()):
            domain_rng = random.Random(
                args.seed + epoch * 1009 + sum(ord(char) for char in domain))
            shuffled = list(group)
            domain_rng.shuffle(shuffled)
            balanced.extend(shuffled[index % len(shuffled)]
                            for index in range(largest))
        ordered = balanced
    random.Random(args.seed + epoch).shuffle(ordered)
    return ordered


def train_epoch(dino, heads, optimizer, records: Sequence[Dict], epoch: int,
                global_step: int, args, dino_device, head_device) -> Dict:
    two_frame_selective = bool(
        args.train_components == 's7_selective_promotion'
        and getattr(args, 's7_selective_two_frame', False))
    heads.train()
    if args.train_components in (
            's7_rpn', 's7_merge', 's7_lane_arbitration',
            's7_quality_suppression', 's7_temporal_association',
            's7_temporal_student', 's7_static_domain_ranker',
            's7_selective_promotion', 's7_highres_roi_ranker'):
        heads.rpn_head.eval()
        heads.roi_head.eval()
    if args.train_components in (
            's7_merge', 's7_lane_arbitration', 's7_quality_suppression',
            's7_temporal_association', 's7_temporal_student',
            's7_static_domain_ranker', 's7_selective_promotion',
            's7_highres_roi_ranker'):
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
    elif args.train_components == 's7_temporal_student':
        heads.s7_temporal_scorer.eval()
        heads.s7_candidate_quality_head.eval()
        heads.s7_candidate_student_head.train()
    elif args.train_components == 's7_static_domain_ranker':
        heads.s7_candidate_static_head.train()
    elif args.train_components == 's7_selective_promotion':
        heads.s7_candidate_quality_head.eval()
        heads.s7_selective_promotion_head.train()
    elif args.train_components == 's7_highres_roi_ranker':
        heads.s7_highres_spatial_projection.train()
        if bool(getattr(args, 's7_highres_pairwise_takeover_v2', False)):
            heads.s7_highres_candidate_quality_head.eval()
            heads.s7_highres_pairwise_takeover_head.train()
        else:
            heads.s7_highres_candidate_quality_head.train()
    if head_device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(head_device)
    ordered = ordered_source_training_records(records, args, epoch)
    losses = []
    component_sums = {}
    metric_sums = {}
    takeover_v2 = bool(getattr(
        args, 's7_highres_pairwise_takeover_v2', False))
    group_log_weights = ({
        source_domain_label(row): 0.0
        for row in ordered} if takeover_v2 else {})
    group_loss_sums = collections.defaultdict(float)
    group_counts = collections.defaultdict(int)
    cache_hits = 0
    gain_replayed_keys = set()
    gain_replay_extra_count = 0
    temporal_previous_box = None
    temporal_previous_embedding = None
    temporal_previous_key = None
    selective_two_frame_state = (
        temporal.TwoFrameMotionState() if two_frame_selective else None)
    for index, record in enumerate(ordered):
        current_lr = scheduled_lr(args, epoch, global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        feature, img_meta, gt_boxes, gt_labels, _original, cached = (
            prepare_record(
                dino, record, args, dino_device, head_device))
        cache_hits += int(cached)
        static_augmented = False
        paired_augmented_feature = None
        if (args.train_components in (
                's7_static_domain_ranker', 's7_selective_promotion')
                or (args.train_components == 's7_highres_roi_ranker'
                    and (bool(getattr(
                        args, 's7_highres_unified_ranking', False))
                         or bool(getattr(
                             args, 's7_highres_pairwise_takeover_v2',
                             False))))):
            augmented, augmentation = static_source_feature_domain_augment(
                feature, args, args.seed + epoch * 1000003 + index * 9176)
            static_augmented = bool(augmentation['applied'])
            if bool(getattr(
                    args, 's7_highres_pairwise_takeover_v2', False)):
                paired_augmented_feature = augmented
            else:
                feature = augmented
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
        if two_frame_selective:
            selective_current_key = (
                '{}|{}'.format(str(record.get('split', '')),
                               str(record.get('seq', ''))),
                int(record.get('frame', -1)))
            selective_two_frame_state.prepare(*selective_current_key)
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
        elif args.train_components == 's7_temporal_student':
            output = heads.forward_s7_temporal_student_train(
                feature, img_meta, gt_boxes,
                riou_thr=args.riou_thr,
                quality_weight=args.s7_student_quality_loss_weight,
                relative_weight=args.s7_student_relative_loss_weight,
                relative_margin=args.s7_temporal_relative_quality_margin,
                relative_min_gap=args.s7_temporal_relative_quality_min_gap,
                relative_max_pairs=(
                    args.s7_temporal_relative_quality_max_pairs),
                distillation_weight=args.s7_student_distillation_weight,
                distillation_temperature=(
                    args.s7_student_distillation_temperature),
                small_loss_weight=args.s7_student_small_loss_weight,
                small_token_threshold=args.s7_student_small_token_thr,
                max_candidates=args.s7_temporal_max_candidates)
        elif args.train_components == 's7_static_domain_ranker':
            output = heads.forward_s7_static_domain_ranker_train(
                feature, img_meta, gt_boxes,
                riou_thr=args.riou_thr,
                quality_weight=args.s7_static_quality_loss_weight,
                relative_weight=args.s7_static_relative_loss_weight,
                relative_margin=args.s7_static_relative_margin,
                relative_min_gap=args.s7_static_relative_min_gap,
                relative_max_pairs=args.s7_static_relative_max_pairs,
                score_weight=args.s7_static_score_weight,
                rank_margin=args.s7_static_rank_margin,
                retention_weight=args.s7_static_retention_weight,
                gain_weight=args.s7_static_gain_weight,
                prior_weight=args.s7_static_prior_weight,
                max_candidates=args.s7_static_max_candidates)
            output['s7_static_augmented'] = int(static_augmented)
        elif args.train_components == 's7_selective_promotion':
            output = heads.forward_s7_selective_promotion_train(
                feature, img_meta, gt_boxes,
                riou_thr=args.riou_thr,
                advantage_gap=args.s7_selective_advantage_gap,
                promotion_margin=args.s7_selective_promotion_margin,
                uncertainty_multiplier=(
                    args.s7_selective_uncertainty_multiplier),
                quality_weight=args.s7_selective_quality_loss_weight,
                classification_weight=(
                    args.s7_selective_classification_loss_weight),
                retention_weight=args.s7_selective_retention_weight,
                gain_weight=args.s7_selective_gain_weight,
                prior_weight=args.s7_selective_prior_weight,
                max_candidates=args.s7_selective_max_candidates,
                two_frame_state=selective_two_frame_state)
            output['s7_selective_augmented'] = int(static_augmented)
        elif args.train_components == 's7_highres_roi_ranker':
            output = heads.forward_s7_highres_roi_ranker_train(
                feature, img_meta, gt_boxes,
                riou_thr=args.riou_thr,
                quality_weight=args.s7_highres_quality_loss_weight,
                relative_weight=args.s7_highres_relative_loss_weight,
                relative_margin=args.s7_highres_rank_margin,
                relative_min_gap=args.s7_highres_relative_min_gap,
                relative_max_pairs=args.s7_highres_relative_max_pairs,
                score_weight=args.s7_highres_score_weight,
                rank_margin=args.s7_highres_rank_margin,
                retention_weight=args.s7_highres_retention_weight,
                gain_weight=args.s7_highres_gain_weight,
                prior_weight=args.s7_highres_prior_weight,
                max_candidates=args.s7_highres_max_candidates,
                augmented_feature=paired_augmented_feature)
            if (bool(getattr(args, 's7_highres_unified_ranking', False))
                    or bool(getattr(
                        args, 's7_highres_pairwise_takeover_v2', False))):
                output['s7_highres_augmented'] = int(static_augmented)
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
        if two_frame_selective:
            selected_box = output.pop('_s7_small_temporal_selected_box')
            selected_embedding = output.pop(
                '_s7_small_temporal_selected_embedding')
            if selected_box is not None and selected_embedding is not None:
                selective_two_frame_state.update(
                    selected_box, selected_embedding, *selective_current_key)
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
        optimized_total = total
        if takeover_v2:
            domain = source_domain_label(record)
            group_log_weights.setdefault(domain, 0.0)
            group_log_weights[domain] += (
                float(args.s7_takeover_group_dro_eta)
                * float(total.detach().item()))
            maximum = max(group_log_weights.values())
            unnormalized = {
                name: math.exp(value - maximum)
                for name, value in group_log_weights.items()}
            denominator = sum(unnormalized.values())
            weights = {
                name: value / denominator
                for name, value in unnormalized.items()}
            optimized_total = total * (
                len(weights) * weights[domain])
            group_loss_sums[domain] += float(total.detach().item())
            group_counts[domain] += 1
        optimized_total.backward()
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
            elif args.train_components == 's7_temporal_student':
                message += (
                    ' student_candidates={} small_frames={} '
                    'teacher_top1_agreement={:.4f}').format(
                    int(round(metric_sums.get(
                        's7_candidate_quality_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_student_small_source_frame', 0.0))),
                    float(metric_sums.get(
                        's7_student_teacher_top1_agreement', 0.0))
                    / float(max(1, index + 1)))
            elif args.train_components == 's7_static_domain_ranker':
                message += (
                    ' retain_pairs_total={} gain_pairs_total={} '
                    'hard_negatives_total={} augmented={:.3f}').format(
                    int(round(metric_sums.get(
                        's7_static_retention_pair_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_static_gain_pair_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_static_hard_negative_count', 0.0))),
                    float(metric_sums.get('s7_static_augmented', 0.0))
                    / float(max(1, index + 1)))
            elif args.train_components == 's7_selective_promotion':
                message += (
                    ' positives_total={} retain_pairs_total={} '
                    'gain_pairs_total={} uncertainty={:.4f} '
                    'augmented={:.3f}').format(
                    int(round(metric_sums.get(
                        's7_selective_positive_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_selective_retention_pair_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_selective_gain_pair_count', 0.0))),
                    float(metric_sums.get(
                        's7_selective_mean_uncertainty', 0.0))
                    / float(max(1, index + 1)),
                    float(metric_sums.get('s7_selective_augmented', 0.0))
                    / float(max(1, index + 1)))
            elif args.train_components == 's7_highres_roi_ranker':
                if takeover_v2:
                    message += (
                        ' takeover_candidates={} eligible={} rank_pairs={} '
                        'retain_pairs={} gain_pairs={} uncertainty={:.4f} '
                        'augmented={:.3f}').format(
                            int(round(metric_sums.get(
                                's7_takeover_candidate_count', 0.0))),
                            int(round(metric_sums.get(
                                's7_takeover_eligible_count', 0.0))),
                            int(round(metric_sums.get(
                                's7_takeover_ranking_pair_count', 0.0))),
                            int(round(metric_sums.get(
                                's7_takeover_retention_pair_count', 0.0))),
                            int(round(metric_sums.get(
                                's7_takeover_gain_pair_count', 0.0))),
                            float(metric_sums.get(
                                's7_takeover_mean_uncertainty', 0.0))
                            / float(max(1, index + 1)),
                            float(metric_sums.get(
                                's7_highres_augmented', 0.0))
                            / float(max(1, index + 1)))
                else:
                    message += (
                    'candidate_count={} usable_total={} '
                    'retain_pairs_total={} gain_pairs_total={} '
                    'relative_pairs_total={} unified_pairs_total={} '
                    'unified_active_total={} augmented={:.3f}').format(
                    int(round(metric_sums.get(
                        's7_highres_candidate_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_highres_usable_candidate_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_highres_retention_pair_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_highres_gain_pair_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_highres_relative_pair_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_highres_unified_pair_count', 0.0))),
                    int(round(metric_sums.get(
                        's7_highres_unified_active_count', 0.0))),
                    float(metric_sums.get('s7_highres_augmented', 0.0))
                    / float(max(1, index + 1)))
            print(message)
        del feature, gt_boxes, gt_labels, total
    optimized_components = optimization_loss_component_names(
        args.train_components,
        quality_head=bool(getattr(args, 's7_temporal_quality_head', False)),
        relative_quality=bool(getattr(
            args, 's7_temporal_relative_quality', False)),
        unified_highres=bool(getattr(
            args, 's7_highres_unified_ranking', False)),
        pairwise_takeover_v2=bool(getattr(
            args, 's7_highres_pairwise_takeover_v2', False)))
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
    if takeover_v2:
        maximum = max(group_log_weights.values())
        unnormalized = {
            name: math.exp(value - maximum)
            for name, value in group_log_weights.items()}
        denominator = sum(unnormalized.values())
        summary['source_group_dro'] = dict(
            eta=float(args.s7_takeover_group_dro_eta),
            balanced_sampling=True,
            source_labels_used_for_model_input=False,
            counts={name: int(group_counts[name])
                    for name in sorted(group_counts)},
            mean_losses={
                name: float(group_loss_sums[name] / group_counts[name])
                for name in sorted(group_counts)},
            final_weights={
                name: float(unnormalized[name] / denominator)
                for name in sorted(unnormalized)})
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
    small_temporal_selector = None
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
    if (bool(getattr(args, 's7_selective_two_frame', False))
            and bool(getattr(heads, 's7_selective_promotion', False))
            and heads.s7_inference_enabled()):
        ranker_head = getattr(heads, 's7_selective_promotion_head', None)
        if ranker_head is None:
            raise RuntimeError('Two-frame selective ranker head is missing')
        small_temporal_selector = temporal.CausalSmallTemporalRanker(
            ranker_head,
            max_candidates=int(getattr(
                args, 's7_selective_max_candidates', 20)),
            uncertainty_multiplier=float(getattr(
                args, 's7_selective_uncertainty_multiplier', 1.0)),
            promotion_margin=float(getattr(
                args, 's7_selective_promotion_margin', 0.10)))
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
            if small_temporal_selector is not None:
                selective_pool = getattr(heads, '_last_selective_pool', None)
                if selective_pool is None:
                    raise RuntimeError(
                        'Two-frame selective candidate pool was not produced')
                pool_detections = selective_pool['detections']
                if pool_detections.shape[0] != raw_detections.shape[0]:
                    raise RuntimeError(
                        'Two-frame selective pool and detections disagree')
                valid_mask_np = valid_rotated_detection_mask(
                    raw_detections, img_meta, args.valid_content_tolerance)
                sequence_key = '{}|{}'.format(
                    str(record.get('split', '')), str(record['seq']))
                temporal_selection = small_temporal_selector.select(
                    pool_detections, selective_pool['embeddings'],
                    selective_pool['source_ids'],
                    selective_pool['quality_logits'], sequence_key,
                    int(record['frame']), valid_mask=torch.as_tensor(
                        valid_mask_np, dtype=torch.bool,
                        device=pool_detections.device))
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
                if small_temporal_selector is not None:
                    candidate_merge['s7_selective_promotion'] = (
                        temporal_selection)
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
                domain=source_domain_label(record),
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


def _highres_margin_audit_row(
        record: Dict, original: Dict, img_meta: Dict,
        raw_detections: np.ndarray, candidate_merge: Optional[Dict],
        args, cached: bool, role: str = 'source_validation') -> Dict:
    """Build one read-only row from a shared-forward margin decision."""
    raw_detections = np.asarray(
        raw_detections, dtype=np.float32).reshape((-1, 6))
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
    return dict(
        role=role, split=record['split'], seq=record['seq'],
        domain=source_domain_label(record),
        frame=int(record['frame']), feature_cache_hit=bool(cached),
        candidate_merge=candidate_merge, temporal_selection=None,
        temporal_attribution=None, metrics=metrics,
        detections=[[float(value) for value in detection]
                    for detection in detections.tolist()])


def evaluate_highres_margin_grid_records(
        dino, heads, records: Sequence[Dict], args,
        dino_device, head_device, margins: Sequence[float],
        role: str = 'source_validation') -> Dict:
    """Evaluate several margins from one frozen forward per labeled frame."""
    margins = sorted(set(float(value) for value in margins))
    if not margins or any(value < 0.0 for value in margins):
        raise ValueError('Margin grid must be non-empty and non-negative')
    heads.eval()
    baseline_rows = []
    rows_by_margin = {value: [] for value in margins}
    with torch.no_grad():
        for index, record in enumerate(records):
            feature, img_meta, _gt_boxes, _gt_labels, original, cached = (
                prepare_record(
                    dino, record, args, dino_device, head_device))
            base = heads._protected_merge_detections(
                feature, img_meta, rescale=True,
                apply_highres_ranker=False)
            pool = getattr(heads, '_last_highres_pool', None)
            merge = getattr(heads, '_last_candidate_merge', None)
            if pool is None or merge is None:
                raise RuntimeError(
                    'High-resolution margin audit produced no shared pool')
            native_detection = (merge.get('source_top1_detections') or {}).get(
                'native_s14')
            native = np.asarray(
                [] if native_detection is None else [native_detection],
                dtype=np.float32).reshape((-1, 6))
            baseline_merge = dict(merge)
            baseline_merge['raw_top1_source'] = (
                None if native.shape[0] == 0 else 'native_s14')
            baseline_merge['s7_highres_roi_ranker'] = dict(
                promoted=False, reason='native_reference',
                candidate_count=int(pool['detections'].shape[0] - 1))
            baseline_rows.append(_highres_margin_audit_row(
                record, original, img_meta, native, baseline_merge,
                args, cached, role=role))
            for margin in margins:
                selector = (
                    temporal.native_protected_unified_highres_ranking_from_logits
                    if bool(getattr(
                        args, 's7_highres_unified_ranking', False)) else
                    temporal.native_protected_highres_promotion_from_logits)
                selection = selector(
                    pool['quality_logits'], pool['detections'],
                    pool['source_ids'], max_candidates=int(getattr(
                        args, 's7_highres_max_candidates', 32)),
                    score_weight=float(getattr(
                        args, 's7_highres_score_weight', 1.0)),
                    promotion_margin=margin)
                selected_local = selection.get('selected_index')
                selected_source = None
                candidate = base
                if selected_local is not None and base.shape[0]:
                    selected_local = int(selected_local)
                    selected_base = int(
                        pool['base_indices'][selected_local].item())
                    remaining = torch.arange(
                        base.shape[0], device=base.device)
                    remaining = remaining[remaining != selected_base]
                    order = torch.cat((
                        remaining.new_tensor([selected_base]), remaining), 0)
                    candidate = base[order]
                    selected_source = (
                        'native_s14'
                        if int(pool['source_ids'][selected_local].item()) == 0
                        else 'supplement_s7')
                margin_merge = dict(merge)
                margin_merge['raw_top1_source'] = selected_source
                margin_merge['s7_highres_roi_ranker'] = dict(
                    selected_index=selection.get('selected_index'),
                    native_index=selection.get('native_index'),
                    promoted=bool(selection.get('promoted', False)),
                    reason=str(selection.get('reason', '')),
                    candidate_count=int(selection.get('candidate_count', 0)),
                    best_margin=selection.get('best_margin'),
                    audited_promotion_margin=float(margin))
                rows_by_margin[margin].append(_highres_margin_audit_row(
                    record, original, img_meta,
                    candidate.detach().cpu().numpy(), margin_merge,
                    args, cached, role=role))
            if ((index + 1) % 25 == 0 or index + 1 == len(records)):
                counts = ','.join(
                    '{}:{}'.format(
                        margin,
                        sum(row['metrics']['top1_hit']
                            for row in rows_by_margin[margin]))
                    for margin in margins)
                print('[highres-margin:{}] {}/{} top1={}'.format(
                    role, index + 1, len(records), counts))
            del feature, _gt_boxes, _gt_labels
    return dict(
        baseline_rows=baseline_rows, rows_by_margin=rows_by_margin,
        shared_model_forward_count=len(records),
        margin_decision_count=len(records) * len(margins))


SMOOTH_GEOMETRY_METRICS = (
    'sym_kld', 'gwd', 'normalized_gwd')
SMOOTH_GEOMETRY_AUDIT_METRICS = (
    'roi_score', 'oracle_riou') + SMOOTH_GEOMETRY_METRICS


def _smooth_geometry_rank_order(values: np.ndarray, descending: bool):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise RuntimeError('Smooth geometry ranking received non-finite values')
    return np.argsort(-values if descending else values, kind='stable')


def _smooth_geometry_pair_agreement(
        values: np.ndarray, riou: np.ndarray, descending: bool):
    """Pairwise rank agreement with oracle RIoU, ignoring exact ties."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    riou = np.asarray(riou, dtype=np.float64).reshape(-1)
    if values.shape != riou.shape:
        raise ValueError('Pair agreement arrays must have the same shape')
    if values.size < 2:
        return None
    if not descending:
        values = -values
    value_delta = values[:, None] - values[None, :]
    riou_delta = riou[:, None] - riou[None, :]
    upper = np.triu(np.ones(value_delta.shape, dtype=bool), k=1)
    comparable = upper & (value_delta != 0.0) & (riou_delta != 0.0)
    if not np.any(comparable):
        return None
    agreement = np.sign(value_delta[comparable]) * np.sign(
        riou_delta[comparable])
    return float(np.mean(agreement > 0.0))


def _smooth_geometry_rank_metrics(
        order: np.ndarray, riou: np.ndarray, source_ids: np.ndarray,
        native_index: int, riou_thr: float) -> Dict:
    selected_index = int(order[0])
    native_correct = bool(float(riou[native_index]) >= float(riou_thr))
    selected_correct = bool(float(riou[selected_index]) >= float(riou_thr))
    usable = np.flatnonzero(riou[order] >= float(riou_thr))
    native_positions = np.flatnonzero(order == int(native_index))
    s7_correct = np.flatnonzero(
        (source_ids[order] == 1) & (riou[order] >= float(riou_thr)))
    best_s7_rank = None if s7_correct.size == 0 else int(s7_correct[0] + 1)
    native_rank = (None if native_positions.size == 0
                   else int(native_positions[0] + 1))
    return dict(
        selected_index=selected_index,
        selected_source=('native_s14' if int(source_ids[selected_index]) == 0
                         else 'supplement_s7'),
        selected_riou=float(riou[selected_index]),
        top1_hit=selected_correct,
        gain_vs_native=bool(not native_correct and selected_correct),
        loss_vs_native=bool(native_correct and not selected_correct),
        best_usable_rank=(None if usable.size == 0 else int(usable[0] + 1)),
        recall_at_20=bool(usable.size and int(usable[0]) < 20),
        recall_at_100=bool(usable.size and int(usable[0]) < 100),
        native_rank=native_rank,
        best_s7_correct_rank=best_s7_rank,
        s7_takeover_supported=bool(
            best_s7_rank is not None and native_rank is not None
            and best_s7_rank < native_rank))


def _smooth_geometry_rank_support_frame(
        record: Dict, original: np.ndarray, pool: Dict, args) -> Dict:
    detections = pool['detections'].detach().cpu().numpy().astype(
        np.float32, copy=False).reshape((-1, 6))
    source_ids = pool['source_ids'].detach().cpu().numpy().astype(
        np.int64, copy=False).reshape(-1)
    if detections.shape[0] != source_ids.shape[0]:
        raise RuntimeError('Smooth geometry pool metadata disagrees')
    row = dict(
        split=record['split'], seq=record['seq'],
        domain=source_domain_label(record), frame=int(record['frame']),
        candidate_count=int(detections.shape[0]), eligible=False,
        native_index=None, native_riou=0.0, native_correct=False,
        s7_correct_count=0, gain_pair_count=0, rankings={})
    if detections.shape[0] == 0 or original.shape[0] == 0:
        return row
    from mmcv.ops import box_iou_rotated

    boxes = torch.from_numpy(detections[:, :5]).float()
    gt = torch.from_numpy(np.asarray(original[:, :5], dtype=np.float32))
    overlaps = box_iou_rotated(boxes, gt).max(dim=1).values.cpu().numpy()
    best_gt = box_iou_rotated(boxes, gt).argmax(dim=1).cpu().numpy()
    targets = np.asarray(original, dtype=np.float32)[best_gt]
    candidate_boxes = torch.from_numpy(detections[:, :5]).float()
    target_boxes = torch.from_numpy(targets[:, :5]).float()
    sym_kld = geometry.symmetric_gaussian_kl(
        candidate_boxes, target_boxes).cpu().numpy()
    gwd = geometry.gaussian_wasserstein_distance(
        candidate_boxes, target_boxes).cpu().numpy()
    normalized_gwd = geometry.normalized_gaussian_wasserstein_distance(
        candidate_boxes, target_boxes).cpu().numpy()
    native_indices = np.flatnonzero(source_ids == 0)
    if native_indices.size == 0:
        raise RuntimeError('High-resolution pool has no native candidate')
    native_index = int(native_indices[np.argmax(
        detections[native_indices, 5])])
    native_riou = float(overlaps[native_index])
    native_correct = bool(native_riou >= float(args.riou_thr))
    s7_correct = (source_ids == 1) & (overlaps >= float(args.riou_thr))
    row.update(
        eligible=True, native_index=native_index,
        native_riou=native_riou, native_correct=native_correct,
        s7_correct_count=int(np.count_nonzero(s7_correct)),
        gain_pair_count=int(np.count_nonzero(
            s7_correct & (not native_correct))))
    metric_values = dict(
        roi_score=detections[:, 5],
        oracle_riou=overlaps,
        sym_kld=sym_kld, gwd=gwd, normalized_gwd=normalized_gwd)
    metric_descending = dict(
        roi_score=True, oracle_riou=True,
        sym_kld=False, gwd=False, normalized_gwd=False)
    for name, values in metric_values.items():
        order = _smooth_geometry_rank_order(
            values, metric_descending[name])
        metrics = _smooth_geometry_rank_metrics(
            order, overlaps, source_ids, native_index, args.riou_thr)
        metrics['pair_agreement_with_riou'] = (
            _smooth_geometry_pair_agreement(
                values, overlaps, metric_descending[name]))
        row['rankings'][name] = metrics
    row['geometry_values'] = dict(
        native_score=float(detections[native_index, 5]),
        native_sym_kld=float(sym_kld[native_index]),
        native_gwd=float(gwd[native_index]),
        native_normalized_gwd=float(normalized_gwd[native_index]))
    # Candidate-level arrays are intentionally omitted from the JSON.  The
    # frame-level ranks above are enough to decide whether a later quality
    # head has source support, while keeping the audit artifact compact.
    return row


def evaluate_smooth_geometry_rank_support_records(
        dino, heads, records: Sequence[Dict], args,
        dino_device, head_device) -> List[Dict]:
    """Collect frozen source candidate-pool geometry/rank diagnostics."""
    heads.eval()
    rows = []
    with torch.no_grad():
        for index, record in enumerate(records):
            feature, img_meta, _gt_boxes, _gt_labels, original, cached = (
                prepare_record(
                    dino, record, args, dino_device, head_device))
            heads._protected_merge_detections(
                feature, img_meta, rescale=True, apply_highres_ranker=False)
            pool = getattr(heads, '_last_highres_pool', None)
            if pool is None:
                raise RuntimeError(
                    'Smooth geometry audit produced no high-resolution pool')
            row = _smooth_geometry_rank_support_frame(
                record, original, pool, args)
            row['feature_cache_hit'] = bool(cached)
            rows.append(row)
            if ((index + 1) % 25 == 0 or index + 1 == len(records)):
                print('[smooth-geometry] {}/{} eligible={}'.format(
                    index + 1, len(records),
                    sum(bool(item['eligible']) for item in rows)))
            del feature, _gt_boxes, _gt_labels
    return rows


def summarize_smooth_geometry_rank_support(
        rows: Sequence[Dict], metric_names: Sequence[str] =
        SMOOTH_GEOMETRY_AUDIT_METRICS) -> Dict:
    """Summarize source candidate support without selecting a checkpoint."""
    eligible = [row for row in rows if bool(row.get('eligible', False))]
    native_hits = sum(bool(row['native_correct']) for row in eligible)
    gain_rows = [
        row for row in eligible
        if (not bool(row['native_correct'])
            and int(row.get('s7_correct_count', 0)) > 0)]
    domains = sorted(set(str(row['domain']) for row in gain_rows))
    sequences = sorted(set(str(row['seq']) for row in gain_rows))
    result = dict(
        frame_count=int(len(rows)), eligible_frame_count=int(len(eligible)),
        candidate_count_total=int(sum(
            int(row.get('candidate_count', 0)) for row in eligible)),
        candidate_count_mean=(
            0.0 if not eligible else float(np.mean([
                int(row.get('candidate_count', 0)) for row in eligible]))),
        native_top1_hits=int(native_hits),
        native_top1_misses=int(len(eligible) - native_hits),
        native_wrong_s7_correct_frame_count=int(len(gain_rows)),
        native_wrong_s7_correct_pair_count=int(sum(
            int(row.get('gain_pair_count', 0)) for row in gain_rows)),
        gain_domains=domains, gain_sequences=sequences,
        metrics={})
    for name in metric_names:
        metric_rows = [row for row in eligible
                       if name in (row.get('rankings') or {})]
        top1_hits = sum(bool(row['rankings'][name]['top1_hit'])
                        for row in metric_rows)
        gains = sum(bool(row['rankings'][name]['gain_vs_native'])
                    for row in metric_rows)
        losses = sum(bool(row['rankings'][name]['loss_vs_native'])
                     for row in metric_rows)
        recall20 = sum(bool(row['rankings'][name].get('recall_at_20', False))
                       for row in metric_rows)
        recall100 = sum(bool(row['rankings'][name].get(
            'recall_at_100', False))
                        for row in metric_rows)
        agreements = [
            float(row['rankings'][name]['pair_agreement_with_riou'])
            for row in metric_rows
            if row['rankings'][name]['pair_agreement_with_riou'] is not None]
        takeover_supported = sum(bool(
            row['rankings'][name]['s7_takeover_supported'])
            for row in metric_rows)
        result['metrics'][name] = dict(
            frame_count=int(len(metric_rows)), top1_hits=int(top1_hits),
            top1_gains=int(gains), top1_losses=int(losses),
            net_top1_gain=int(gains - losses),
            recall_at_20=int(recall20), recall_at_100=int(recall100),
            pair_agreement_mean=(
                None if not agreements else float(np.mean(agreements))),
            frames_with_s7_takeover_support=int(takeover_supported))
    return result


def _smooth_geometry_group_summaries(
        rows: Sequence[Dict], field: str) -> Dict:
    groups = collections.defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        label: summarize_smooth_geometry_rank_support(group)
        for label, group in sorted(groups.items())}


def _smooth_geometry_source_support_gate(
        summary: Dict, args) -> Dict:
    min_domains = int(getattr(
        args, 'source_smooth_geometry_min_gain_domains', 2))
    min_sequences = int(getattr(
        args, 'source_smooth_geometry_min_gain_sequences', 2))
    checks = dict(
        candidate_gain_pair_exists=(
            int(summary['native_wrong_s7_correct_pair_count']) > 0),
        minimum_gain_domains=(
            len(summary['gain_domains']) >= min_domains),
        minimum_gain_sequences=(
            len(summary['gain_sequences']) >= min_sequences))
    return dict(
        passed=bool(all(checks.values())), checks=checks,
        min_gain_domains=min_domains, min_gain_sequences=min_sequences,
        gain_domains=list(summary['gain_domains']),
        gain_sequences=list(summary['gain_sequences']))


def build_smooth_geometry_rank_support_audit(
        dino, heads, records: Sequence[Dict], args,
        dino_device, head_device, spec: Dict,
        source_protocol: Optional[Dict] = None) -> Dict:
    """Build the source-only feasibility artifact for smooth geometry ranking."""
    rows = evaluate_smooth_geometry_rank_support_records(
        dino, heads, records, args, dino_device, head_device)
    threshold = float(spec['small_sampling']['short_token_threshold'])
    small_keys = {
        (record['split'], record['seq'], int(record['frame']))
        for record in source_small_records(records, args, threshold)}
    small_rows = [
        row for row in rows
        if (row['split'], row['seq'], int(row['frame'])) in small_keys]
    full_summary = summarize_smooth_geometry_rank_support(rows)
    small_summary = summarize_smooth_geometry_rank_support(small_rows)
    expected_full = int(spec['baseline_summary']['top1_hits'])
    expected_small = int(spec['baseline_small_summary']['top1_hits'])
    if (int(full_summary['native_top1_hits']) != expected_full
            or int(small_summary['native_top1_hits']) != expected_small):
        raise RuntimeError(
            'Smooth geometry audit did not reproduce the locked native source '
            'baseline: expected {}/{} but found {}/{}'.format(
                expected_full, expected_small,
                full_summary['native_top1_hits'],
                small_summary['native_top1_hits']))
    full_support = _smooth_geometry_source_support_gate(full_summary, args)
    small_support = _smooth_geometry_source_support_gate(small_summary, args)
    quality_support = []
    for name in SMOOTH_GEOMETRY_METRICS:
        full_metric = full_summary['metrics'][name]
        small_metric = small_summary['metrics'][name]
        if (int(full_metric['net_top1_gain']) > 0
                and int(small_metric['net_top1_gain']) > 0
                and int(small_metric['top1_gains']) > 0):
            quality_support.append(name)
    support_gate_passed = bool(full_support['passed'] and small_support['passed'])
    quality_gate_passed = bool(quality_support)
    training_allowed = bool(support_gate_passed and quality_gate_passed)
    decision = (
        'SOURCE_ONLY_SMOOTH_GEOMETRY_RANK_SUPPORT_PASS_TARGET_NOT_READ'
        if training_allowed else
        'SOURCE_ONLY_SMOOTH_GEOMETRY_RANK_SUPPORT_INSUFFICIENT_TARGET_NOT_READ')
    return dict(
        protocol_version=27,
        audit_name='Source-only Smooth-Geometry Rank-Support Audit',
        protocol=dict(
            architecture='frozen_unified_highres_native_s7_candidate_pool',
            source_only=True, target_read=False, read_only_evaluation=True,
            parameter_update=False, shared_forward_per_frame=True,
            candidate_pool='native_s14_top1_plus_s7_top_k',
            geometry_quality_metrics=list(SMOOTH_GEOMETRY_METRICS),
            oracle_metric='oracle_riou',
            no_target_threshold_tuning=True,
            no_checkpoint_selection=True,
            feasibility_gate=(
                'source candidate support in full and small subsets, then '
                'positive net top1 gain from at least one smooth metric')),
        isolation=dict(
            dino_frozen=True, detector_parameters_unchanged=True,
            read_only_evaluation=True, parameter_updates_performed=False,
            trainable_parameter_count=0,
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_used_for_threshold_tuning=False),
        checkpoint_used=dict(
            path=spec['checkpoint'], epoch=int(spec['epoch']),
            source_result_json=spec['source_result_json']),
        source=dict(
            protocol=source_protocol,
            full=full_summary, small=small_summary,
            native_reproduction_gate=dict(
                passed=True, expected_full_top1=expected_full,
                expected_small_top1=expected_small,
                observed_full_top1=int(full_summary['native_top1_hits']),
                observed_small_top1=int(small_summary['native_top1_hits'])),
            by_domain=_smooth_geometry_group_summaries(rows, 'domain'),
            by_sequence=_smooth_geometry_group_summaries(rows, 'seq'),
            support_gate=dict(
                passed=support_gate_passed, full=full_support,
                small=small_support),
            quality_gate=dict(
                passed=quality_gate_passed,
                supported_metrics=quality_support,
                requirement=(
                    'positive net top1 gain in both full and source-small '
                    'source subsets')),
            training_feasibility=dict(
                allowed=training_allowed,
                reason=(
                    'A later source-only gain-balanced quality head may be '
                    'implemented.' if training_allowed else
                    'Do not start geometry-quality training from this audit; '
                    'the source support or ranking signal is insufficient.'))),
        frame_rows=rows,
        candidate_forward_count=int(len(records)),
        parameter_update_count=0,
        target_dev=None,
        eligible_for_training=training_allowed,
        eligible_for_deployment=False,
        eligible_for_full_test=False,
        decision=decision)


def build_highres_margin_source_audit(
        dino, heads, records: Sequence[Dict], args,
        dino_device, head_device, spec: Dict) -> Dict:
    """Run and gate the one frozen source-only margin calibration audit."""
    margins = list(args.source_highres_margin_values)
    evaluated = evaluate_highres_margin_grid_records(
        dino, heads, records, args, dino_device, head_device, margins)
    baseline_rows = evaluated['baseline_rows']
    native_reproduction_summary = summarize_rows(baseline_rows)
    threshold = float(spec['small_sampling']['short_token_threshold'])
    small_keys = {
        (record['split'], record['seq'], int(record['frame']))
        for record in source_small_records(records, args, threshold)}
    baseline_small_rows = [
        row for row in baseline_rows
        if (row['split'], row['seq'], int(row['frame'])) in small_keys]
    native_small_reproduction_summary = summarize_rows(baseline_small_rows)
    stored_baseline = spec['baseline_summary']
    stored_baseline_small = spec['baseline_small_summary']
    if (int(native_reproduction_summary['top1_hits'])
            != int(stored_baseline['top1_hits'])
            or int(native_small_reproduction_summary['top1_hits'])
            != int(stored_baseline_small['top1_hits'])
            or int(native_reproduction_summary['top1_mcml'])
            != int(stored_baseline['top1_mcml'])
            or int(native_small_reproduction_summary['top1_mcml'])
            != int(stored_baseline_small['top1_mcml'])):
        raise RuntimeError(
            'Frozen native source baseline did not reproduce 677/303')
    # The one-box native rows establish exact frame-level retention and
    # temporal metrics. Keep the original stored baseline summary so its
    # proposal-recall fields remain the formal native values rather than the
    # recall of the one-box audit representation.
    baseline_summary = dict(stored_baseline)
    baseline_small_summary = dict(stored_baseline_small)
    baseline_correct = source_correct_frame_keys(baseline_rows)
    margin_results = []
    reference = spec['history_row']
    for margin in margins:
        rows = evaluated['rows_by_margin'][margin]
        small_rows = [
            row for row in rows
            if (row['split'], row['seq'], int(row['frame'])) in small_keys]
        summary = summarize_rows(rows)
        small_summary = summarize_rows(small_rows)
        retention = source_top1_retention_summary(baseline_correct, rows)
        gate = s7_source_selection_gate(
            baseline_summary, baseline_small_summary,
            summary, small_summary, retention, args)
        sequence_gain = source_sequence_gain_summary(
            baseline_rows, rows, min_gain_sequences=2)
        bounded_risk_gate = (
            unified_highres_bounded_risk_source_gate(
                baseline_summary, baseline_small_summary,
                summary, small_summary, retention, sequence_gain, args)
            if spec.get('audit_variant') == 'unified_bounded_risk'
            else None)
        promotions = sum(bool(
            ((row.get('candidate_merge') or {}).get(
                's7_highres_roi_ranker') or {}).get('promoted', False))
                         for row in rows)
        reference_reproduced = None
        if abs(float(margin) - 0.25) <= 1e-12:
            reference_reproduced = bool(
                int(summary['top1_hits']) == int(
                    reference['source_val']['top1_hits'])
                and int(summary['top1_mcml']) == int(
                    reference['source_val']['top1_mcml'])
                and int(small_summary['top1_hits']) == int(
                    reference['source_small_val']['top1_hits'])
                and int(small_summary['top1_mcml']) == int(
                    reference['source_small_val']['top1_mcml'])
                and int(retention['lost_correct_count']) == int(
                    reference['source_exact_retention'][
                        'lost_correct_count'])
                and int(retention['gained_correct_count']) == int(
                    reference['source_exact_retention'][
                        'gained_correct_count'])
                and retention['lost_frame_keys'] == sorted(
                    reference['source_exact_retention'].get(
                        'lost_frame_keys', []))
                and retention['gained_frame_keys'] == sorted(
                    reference['source_exact_retention'].get(
                        'gained_frame_keys', []))
                and math.isclose(
                    float(summary['top1_dfr_fraction_per_frame']),
                    float(reference['source_val'][
                        'top1_dfr_fraction_per_frame']),
                    rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(
                    float(summary['top1_aci']),
                    float(reference['source_val']['top1_aci']),
                    rel_tol=0.0, abs_tol=1e-12))
            if not reference_reproduced:
                raise RuntimeError(
                    'Frozen 0.25 margin did not reproduce epoch 3')
        margin_results.append(dict(
            promotion_margin=float(margin), full_summary=summary,
            small_summary=small_summary, source_exact_retention=retention,
            source_gate=gate, gate_passed=bool(gate['passed']),
            source_sequence_gain=sequence_gain,
            bounded_risk_research_gate=bounded_risk_gate,
            bounded_risk_gate_passed=(
                None if bounded_risk_gate is None else
                bool(bounded_risk_gate['passed'])),
            promotion_count=int(promotions),
            epoch3_reference_reproduced=reference_reproduced))
    passed = [row for row in margin_results if row['gate_passed']]
    selected = (max(passed, key=lambda row: row['promotion_margin'])
                if passed else None)
    bounded_passed = [
        row for row in margin_results
        if row['bounded_risk_gate_passed'] is True]
    research_selected = (
        max(bounded_passed, key=lambda row: (
            int(row['full_summary']['top1_hits']),
            int(row['small_summary']['top1_hits']),
            -int(row['source_exact_retention']['lost_correct_count']),
            float(row['promotion_margin'])))
        if bounded_passed else None)
    return dict(
        mode=(
            'frozen_unified_epoch3_shared_forward_source_margin_audit'
            if spec.get('audit_variant') == 'unified_bounded_risk' else
            'frozen_epoch3_shared_forward_source_margin_audit'),
        audit_variant=spec.get('audit_variant'),
        source_result_json=spec['source_result_json'],
        checkpoint=spec['checkpoint'], checkpoint_epoch=int(spec['epoch']),
        margins=[float(value) for value in margins],
        selection_rule=(
            'formal_exact_gate_then_source_metric_bounded_risk_candidate'
            if spec.get('audit_variant') == 'unified_bounded_risk' else
            'highest_margin_among_formal_source_gate_passers'),
        shared_model_forward=True,
        shared_model_forward_count=int(
            evaluated['shared_model_forward_count']),
        margin_decision_count=int(evaluated['margin_decision_count']),
        baseline_summary=baseline_summary,
        baseline_small_summary=baseline_small_summary,
        native_frame_reproduction_summary=native_reproduction_summary,
        native_small_frame_reproduction_summary=(
            native_small_reproduction_summary),
        results=margin_results, formal_gate_passed=bool(selected is not None),
        bounded_risk_research_gate_passed=bool(research_selected is not None),
        bounded_risk_protocol=(
            None if spec.get('audit_variant') != 'unified_bounded_risk' else
            dict(
                status='post_source_result_research_continuation_gate',
                original_exact_retention_gate_unchanged=True,
                source_safe_claim_allowed=False,
                deployment_claim_allowed=False,
                target_threshold_tuning_allowed=False,
                full_test_allowed=False)),
        selected_margin=(None if selected is None else float(
            selected['promotion_margin'])),
        selected_full_summary=(None if selected is None else
                               selected['full_summary']),
        selected_small_summary=(None if selected is None else
                                selected['small_summary']),
        selected_exact_retention=(None if selected is None else
                                  selected['source_exact_retention']),
        research_candidate_margin=(
            None if research_selected is None else float(
                research_selected['promotion_margin'])),
        research_candidate_full_summary=(
            None if research_selected is None else
            research_selected['full_summary']),
        research_candidate_small_summary=(
            None if research_selected is None else
            research_selected['small_summary']),
        research_candidate_exact_retention=(
            None if research_selected is None else
            research_selected['source_exact_retention']),
        eligible_for_fixed_target_dev_diagnostic=bool(
            selected is not None or research_selected is not None),
        eligible_for_deployment=False, eligible_for_full_test=False,
        target_read=False)


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


def source_deployment_correct_frame_keys(rows: Sequence[Dict]) -> List[str]:
    return sorted(
        source_frame_key(row) for row in rows
        if bool(row['metrics'].get('deployment_top1_hit', False)))


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


def load_highres_margin_audit_spec(
        path: str, checkpoint: str, epoch: int) -> Dict:
    """Lock the one permitted margin audit to the source-safe near-pass."""
    with open(path, 'r') as handle:
        payload = json.load(handle)
    source = payload.get('source') or {}
    isolation = payload.get('isolation') or {}
    protocol = payload.get('protocol') or {}
    highres = protocol.get('s7_highres_roi_ranker') or {}
    if int(payload.get('protocol_version', -1)) != 23:
        raise ValueError('High-resolution margin audit requires protocol 23')
    if payload.get('target_dev') is not None or highres.get('target_read') is not False:
        raise ValueError('High-resolution source result must not read target')
    if (payload.get('decision') !=
            'SOURCE_ONLY_HIGHRES_ROI_RANKER_FALLBACK_TARGET_NOT_READ'):
        raise ValueError('High-resolution source result has an invalid decision')
    if (isolation.get('train_components') != 's7_highres_roi_ranker'
            or isolation.get('dino_parameters_unchanged') is not True
            or isolation.get('frozen_head_parameters_unchanged') is not True
            or isolation.get('target_used_for_training') is not False
            or isolation.get('target_used_for_checkpoint_selection') is not False):
        raise ValueError('High-resolution source isolation audit failed')
    if (highres.get('source_only') is not True
            or highres.get('exact_source_retention') is not True
            or float(highres.get('promotion_margin', -1.0)) != 0.25):
        raise ValueError('High-resolution source protocol mismatch')
    matches = [row for row in source.get('history', [])
               if int(row.get('epoch', -1)) == int(epoch)]
    if len(matches) != 1:
        raise ValueError(
            'Expected one high-resolution history row for epoch {}'.format(
                epoch))
    row = matches[0]
    full = row.get('source_val') or {}
    small = row.get('source_small_val') or {}
    retention = row.get('source_exact_retention') or {}
    checks = (row.get('source_selection_gate') or {}).get('checks') or {}
    failed_checks = sorted(name for name, passed in checks.items()
                           if passed is not True)
    if (int(epoch) != 3 or row.get('checkpoint_saved') is not True
            or row.get('selection_eligible') is not True
            or row.get('source_selection_gate_passed') is not False
            or int(full.get('top1_hits', -1)) != 687
            or int(small.get('top1_hits', -1)) != 310
            or int(full.get('top1_mcml', -1)) != 3
            or int(small.get('top1_mcml', -1)) != 3
            or int(retention.get('baseline_correct_count', -1)) != 677
            or int(retention.get('lost_correct_count', -1)) != 0
            or int(retention.get('gained_correct_count', -1)) != 10
            or failed_checks != [
                'full_top1_absolute', 'small_top1_absolute']):
        raise ValueError(
            'High-resolution epoch 3 is not the locked 687/310 near-pass')
    baseline = source.get('baseline_validation_summary') or {}
    baseline_small = source.get('baseline_small_validation_summary') or {}
    sampling = source.get('small_sampling') or {}
    if (int(baseline.get('top1_hits', -1)) != 677
            or int(baseline_small.get('top1_hits', -1)) != 303
            or sampling.get('short_token_threshold') is None):
        raise ValueError('High-resolution result lacks the locked baselines')
    selected = payload.get('source_selected_checkpoint')
    if not selected:
        raise ValueError('High-resolution result has no fallback checkpoint')
    expected_checkpoint = os.path.join(
        os.path.dirname(selected),
        'labeller_epoch_{:02d}_source_only.pth'.format(int(epoch)))
    if os.path.realpath(expected_checkpoint) != os.path.realpath(checkpoint):
        raise ValueError(
            'Margin audit checkpoint must be {}'.format(expected_checkpoint))
    return dict(
        audit_variant='lane_specific_exact_retention',
        source_result_json=os.path.abspath(path), epoch=int(epoch),
        checkpoint=os.path.abspath(checkpoint), training_result=payload,
        history_row=row, baseline_summary=baseline,
        baseline_small_summary=baseline_small,
        small_sampling=sampling)


def load_unified_highres_margin_audit_spec(
        path: str, checkpoint: str, epoch: int) -> Dict:
    """Lock the unified audit to the observed epoch-3 source-only result.

    This is intentionally a separate protocol from the earlier lane-specific
    audit.  It accepts the one-frame-loss source result only as input to a
    read-only margin audit; it does not reinterpret that training run as an
    exact-retention pass.
    """
    with open(path, 'r') as handle:
        payload = json.load(handle)
    source = payload.get('source') or {}
    isolation = payload.get('isolation') or {}
    protocol = payload.get('protocol') or {}
    highres = protocol.get('s7_highres_roi_ranker') or {}
    architecture = ((payload.get('architecture') or {}).get('s7') or {})
    if int(payload.get('protocol_version', -1)) != 23:
        raise ValueError('Unified high-resolution audit requires protocol 23')
    if (payload.get('target_dev') is not None
            or highres.get('target_read') is not False):
        raise ValueError('Unified high-resolution source result read target')
    if (payload.get('decision') !=
            'SOURCE_ONLY_HIGHRES_ROI_RANKER_FALLBACK_TARGET_NOT_READ'):
        raise ValueError('Unified high-resolution source decision is invalid')
    if (isolation.get('train_components') != 's7_highres_roi_ranker'
            or isolation.get('dino_parameters_unchanged') is not True
            or isolation.get('frozen_head_parameters_unchanged') is not True
            or isolation.get('target_used_for_training') is not False
            or isolation.get('target_used_for_checkpoint_selection') is not False):
        raise ValueError('Unified high-resolution source isolation failed')
    if (highres.get('source_only') is not True
            or highres.get('unified_ranking') is not True
            or float(highres.get('promotion_margin', -1.0)) != 0.25
            or architecture.get('highres_unified_ranking') is not True):
        raise ValueError('Unified high-resolution architecture mismatch')
    matches = [row for row in source.get('history', [])
               if int(row.get('epoch', -1)) == int(epoch)]
    if len(matches) != 1:
        raise ValueError(
            'Expected one unified high-resolution history row for epoch {}'
            .format(epoch))
    row = matches[0]
    full = row.get('source_val') or {}
    small = row.get('source_small_val') or {}
    retention = row.get('source_exact_retention') or {}
    checks = (row.get('source_selection_gate') or {}).get('checks') or {}
    failed_checks = sorted(name for name, passed in checks.items()
                           if passed is not True)
    if (int(epoch) != 3 or row.get('checkpoint_saved') is not True
            or row.get('selection_eligible') is not True
            or row.get('source_selection_gate_passed') is not False
            or int(full.get('top1_hits', -1)) != 696
            or int(small.get('top1_hits', -1)) != 320
            or int(full.get('top1_mcml', -1)) != 3
            or int(small.get('top1_mcml', -1)) != 3
            or int(retention.get('baseline_correct_count', -1)) != 677
            or int(retention.get('lost_correct_count', -1)) != 1
            or int(retention.get('gained_correct_count', -1)) != 20
            or retention.get('lost_frame_keys') != ['val|real_seq07|215']
            or failed_checks != ['exact_old_correct_retention']):
        raise ValueError(
            'Unified epoch 3 is not the locked 696/320 one-frame-loss result')
    baseline = source.get('baseline_validation_summary') or {}
    baseline_small = source.get('baseline_small_validation_summary') or {}
    sampling = source.get('small_sampling') or {}
    if (int(baseline.get('top1_hits', -1)) != 677
            or int(baseline_small.get('top1_hits', -1)) != 303
            or sampling.get('short_token_threshold') is None):
        raise ValueError('Unified high-resolution result lacks baselines')
    selected = payload.get('source_selected_checkpoint')
    if not selected:
        raise ValueError('Unified high-resolution result lacks fallback path')
    expected_checkpoint = os.path.join(
        os.path.dirname(selected),
        'labeller_epoch_{:02d}_source_only.pth'.format(int(epoch)))
    if os.path.realpath(expected_checkpoint) != os.path.realpath(checkpoint):
        raise ValueError(
            'Unified margin audit checkpoint must be {}'.format(
                expected_checkpoint))
    return dict(
        audit_variant='unified_bounded_risk',
        source_result_json=os.path.abspath(path), epoch=int(epoch),
        checkpoint=os.path.abspath(checkpoint), training_result=payload,
        history_row=row, baseline_summary=baseline,
        baseline_small_summary=baseline_small,
        small_sampling=sampling)


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


def source_deployment_retention_summary(
        baseline_correct_keys: Sequence[str], candidate_rows: Sequence[Dict]
        ) -> Dict:
    """Exact retention after applying the fixed deployment score threshold."""
    baseline = set(str(key) for key in baseline_correct_keys)
    candidate_correct = {
        source_frame_key(row) for row in candidate_rows
        if bool(row['metrics'].get('deployment_top1_hit', False))}
    retained = baseline & candidate_correct
    lost = baseline - candidate_correct
    gained = candidate_correct - baseline
    return dict(
        baseline_correct_count=len(baseline),
        retained_correct_count=len(retained),
        lost_correct_count=len(lost),
        gained_correct_count=len(gained),
        candidate_correct_count=len(candidate_correct),
        lost_frame_keys=sorted(lost), gained_frame_keys=sorted(gained))


def source_domain_nonregression_summary(
        baseline_rows: Sequence[Dict], candidate_rows: Sequence[Dict]) -> Dict:
    """Require raw and deployment Top-1 non-regression in every source domain."""
    baseline_by_domain = collections.defaultdict(list)
    candidate_by_domain = collections.defaultdict(list)
    for row in baseline_rows:
        baseline_by_domain[str(row.get('domain', 'unknown'))].append(row)
    for row in candidate_rows:
        candidate_by_domain[str(row.get('domain', 'unknown'))].append(row)
    same_domains = set(baseline_by_domain) == set(candidate_by_domain)
    rows = {}
    if same_domains:
        for domain in sorted(baseline_by_domain):
            baseline = baseline_by_domain[domain]
            candidate = candidate_by_domain[domain]
            baseline_raw = sum(bool(row['metrics']['top1_hit'])
                               for row in baseline)
            candidate_raw = sum(bool(row['metrics']['top1_hit'])
                                for row in candidate)
            baseline_deployment = sum(bool(row['metrics'].get(
                'deployment_top1_hit', False)) for row in baseline)
            candidate_deployment = sum(bool(row['metrics'].get(
                'deployment_top1_hit', False)) for row in candidate)
            rows[domain] = dict(
                baseline_raw_top1=int(baseline_raw),
                candidate_raw_top1=int(candidate_raw),
                raw_delta=int(candidate_raw - baseline_raw),
                baseline_deployment_top1=int(baseline_deployment),
                candidate_deployment_top1=int(candidate_deployment),
                deployment_delta=int(
                    candidate_deployment - baseline_deployment),
                passed=bool(candidate_raw >= baseline_raw
                            and candidate_deployment >= baseline_deployment))
    return dict(
        same_domains=bool(same_domains), domains=rows,
        passed=bool(same_domains and rows
                    and all(row['passed'] for row in rows.values())),
        any_positive_domain=bool(any(
            row['raw_delta'] > 0 or row['deployment_delta'] > 0
            for row in rows.values())))


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
    if getattr(args, 'train_components', '') in (
            's7_temporal_association', 's7_temporal_student',
            's7_selective_promotion', 's7_highres_roi_ranker'):
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


def unified_highres_bounded_risk_source_gate(
        baseline_full: Dict, baseline_small: Dict,
        candidate_full: Dict, candidate_small: Dict,
        retention: Dict, sequence_gain: Dict, args,
        max_lost_correct: int = 1,
        max_lost_fraction: float = 0.002,
        min_gain_loss_ratio: float = 10.0) -> Dict:
    """Research-continuation gate that does not replace exact retention.

    The original source-safe gate remains authoritative for deployment.  This
    second tier only decides whether a source-only candidate has enough net
    evidence to justify one fixed target-dev diagnosis under a new protocol.
    """
    formal = s7_source_selection_gate(
        baseline_full, baseline_small,
        candidate_full, candidate_small, retention, args)
    formal_without_exact = {
        name: passed for name, passed in formal['checks'].items()
        if name != 'exact_old_correct_retention'}
    baseline_correct = int(retention['baseline_correct_count'])
    lost = int(retention['lost_correct_count'])
    gained = int(retention['gained_correct_count'])
    loss_fraction = (
        float(lost) / float(baseline_correct)
        if baseline_correct > 0 else float('inf'))
    gain_loss_ratio = (
        float('inf') if lost == 0 else float(gained) / float(lost))
    checks = dict(
        standard_source_checks_except_exact=all(
            formal_without_exact.values()),
        bounded_lost_correct_count=lost <= int(max_lost_correct),
        bounded_lost_correct_fraction=(
            loss_fraction <= float(max_lost_fraction) + 1e-12),
        strong_gain_to_loss_ratio=(
            gain_loss_ratio + 1e-12 >= float(min_gain_loss_ratio)),
        positive_net_top1_gain=(
            int(candidate_full['top1_hits'])
            > int(baseline_full['top1_hits'])),
        gains_span_source_sequences=bool(sequence_gain.get('passed', False)))
    return dict(
        status='research_continuation_only_not_source_safe',
        checks=checks, passed=all(checks.values()),
        original_formal_gate_passed=bool(formal['passed']),
        original_formal_gate=formal,
        max_lost_correct=int(max_lost_correct),
        max_lost_fraction=float(max_lost_fraction),
        min_gain_loss_ratio=float(min_gain_loss_ratio),
        observed_lost_fraction=float(loss_fraction),
        observed_gain_loss_ratio=(
            None if math.isinf(gain_loss_ratio) else float(gain_loss_ratio)),
        source_safe_claim_allowed=False,
        deployment_claim_allowed=False,
        full_test_allowed=False)


def source_sequence_gain_summary(baseline_rows: Sequence[Dict],
                                 candidate_rows: Sequence[Dict],
                                 min_gain_sequences: int = 2) -> Dict:
    """Require selective-promotion gains to span source sequences."""
    baseline_by_key = {source_frame_key(row): row for row in baseline_rows}
    candidate_by_key = {source_frame_key(row): row for row in candidate_rows}
    same_frames = set(baseline_by_key) == set(candidate_by_key)
    sequence_counts = {}
    if same_frames:
        for key in sorted(baseline_by_key):
            baseline_row = baseline_by_key[key]
            candidate_row = candidate_by_key[key]
            sequence = '{}|{}'.format(
                baseline_row.get('split', ''), baseline_row.get('seq', ''))
            counts = sequence_counts.setdefault(
                sequence, dict(frame_count=0, baseline_top1_hits=0,
                               candidate_top1_hits=0, net_gain=0))
            counts['frame_count'] += 1
            counts['baseline_top1_hits'] += int(bool(
                baseline_row.get('metrics', {}).get('top1_hit', False)))
            counts['candidate_top1_hits'] += int(bool(
                candidate_row.get('metrics', {}).get('top1_hit', False)))
        for counts in sequence_counts.values():
            counts['net_gain'] = (
                counts['candidate_top1_hits'] - counts['baseline_top1_hits'])
    available = len(sequence_counts)
    required = int(min_gain_sequences)
    gain_sequences = sorted(
        sequence for sequence, counts in sequence_counts.items()
        if int(counts['net_gain']) > 0)
    return dict(
        passed=bool(same_frames and available >= required
                    and len(gain_sequences) >= required),
        same_frame_set=bool(same_frames),
        available_sequence_count=int(available),
        configured_min_gain_sequences=int(min_gain_sequences),
        required_gain_sequences=int(required),
        gained_sequence_count=len(gain_sequences),
        gained_sequences=gain_sequences, sequences=sequence_counts)


def selective_promotion_summary(rows: Sequence[Dict]) -> Dict:
    """Summarize actual V1/V2 S7 takeovers from evaluation rows."""
    selections = []
    for row in rows:
        merge = row.get('candidate_merge') or {}
        selection = merge.get('s7_selective_promotion')
        if selection is not None:
            selections.append(selection)
    promoted = [selection for selection in selections
                if bool(selection.get('promoted', False))]
    history_ready = [selection for selection in selections
                     if bool(selection.get('history_ready', False))]
    return dict(
        evaluated_frame_count=len(rows),
        selection_frame_count=len(selections),
        history_ready_frame_count=len(history_ready),
        promotion_count=len(promoted),
        nonzero_s7_promotion=bool(promoted),
        promotion_reasons=dict(collections.Counter(
            str(selection.get('reason', 'unknown'))
            for selection in selections)))


def selective_promotion_effect_summary(
        baseline_rows: Sequence[Dict], candidate_rows: Sequence[Dict],
        small_frame_keys: Sequence[str], min_gain_sequences: int = 2) -> Dict:
    """Measure gains caused on frames where the V1/V2 selector took over.

    The ordinary S7 source gate compares the complete candidate pipeline with
    native S14.  For selective-promotion experiments that is insufficient:
    the frozen affine/quality base may already provide the reported gain even
    when the newly trained selector contributes nothing.  This audit isolates
    frames with an actual S7 takeover and requires the selector itself to add
    source Top-1 evidence, including on the source-small subset.
    """
    baseline_by_key = {source_frame_key(row): row for row in baseline_rows}
    candidate_by_key = {source_frame_key(row): row for row in candidate_rows}
    same_frames = set(baseline_by_key) == set(candidate_by_key)
    small_keys = set(str(key) for key in small_frame_keys)
    sequence_counts = {}
    promotion_count = 0
    gained_correct_count = 0
    lost_correct_count = 0
    small_promotion_count = 0
    small_gained_correct_count = 0
    if same_frames:
        for key in sorted(candidate_by_key):
            candidate_row = candidate_by_key[key]
            selection = ((candidate_row.get('candidate_merge') or {}).get(
                's7_selective_promotion') or {})
            if not bool(selection.get('promoted', False)):
                continue
            promotion_count += 1
            baseline_hit = bool(
                baseline_by_key[key].get('metrics', {}).get(
                    'top1_hit', False))
            candidate_hit = bool(
                candidate_row.get('metrics', {}).get('top1_hit', False))
            gained = int(candidate_hit and not baseline_hit)
            lost = int(baseline_hit and not candidate_hit)
            gained_correct_count += gained
            lost_correct_count += lost
            sequence = '{}|{}'.format(
                candidate_row.get('split', ''), candidate_row.get('seq', ''))
            counts = sequence_counts.setdefault(
                sequence, dict(promotion_count=0, gained_correct_count=0,
                               lost_correct_count=0, net_gain=0))
            counts['promotion_count'] += 1
            counts['gained_correct_count'] += gained
            counts['lost_correct_count'] += lost
            counts['net_gain'] = (
                counts['gained_correct_count']
                - counts['lost_correct_count'])
            if key in small_keys:
                small_promotion_count += 1
                small_gained_correct_count += gained
    gain_sequences = sorted(
        sequence for sequence, counts in sequence_counts.items()
        if int(counts['net_gain']) > 0)
    required = int(min_gain_sequences)
    return dict(
        same_frame_set=bool(same_frames),
        promotion_count=int(promotion_count),
        gained_correct_count=int(gained_correct_count),
        lost_correct_count=int(lost_correct_count),
        net_gain=int(gained_correct_count - lost_correct_count),
        small_promotion_count=int(small_promotion_count),
        small_gained_correct_count=int(small_gained_correct_count),
        required_gain_sequences=required,
        gained_sequence_count=len(gain_sequences),
        gained_sequences=gain_sequences,
        sequences=sequence_counts,
        checks=dict(
            same_frame_set=bool(same_frames),
            selector_top1_loss_zero=bool(same_frames
                                         and lost_correct_count == 0),
            selector_top1_gain_nonzero=bool(same_frames
                                            and gained_correct_count > 0),
            selector_small_top1_gain_nonzero=bool(
                same_frames and small_gained_correct_count > 0),
            selector_gain_multi_sequence=bool(
                same_frames and len(gain_sequences) >= required)))


STAGE3_TEACHER_EXACT_FIELDS = (
    'frame_count', 'top1_hits', 'top1_mcml', 'recall_at_100',
    'top1_dfr_fraction_per_frame', 'top1_aci')
STAGE3_TEACHER_FLOAT_FIELDS = frozenset((
    'top1_dfr_fraction_per_frame', 'top1_aci'))


def stage3_teacher_reproduction_gate(
        expected_full: Dict, expected_small: Dict,
        actual_full: Dict, actual_small: Dict,
        tolerance: float = 1e-9) -> Dict:
    """Require the copied stage-2 teacher to reproduce source evidence.

    The ordinary S7 source gate is intentionally relative to native S14.  It
    is therefore insufficient for stage 3 initialization: a copied teacher
    could lose part of the phase-2 gain and still satisfy the native gate.
    This guard compares the actual first inference with the locked phase-2
    result before any student optimizer step is allowed.
    """
    checks = {}
    mismatches = {}
    for prefix, expected, actual in (
            ('full', expected_full or {}, actual_full or {}),
            ('small', expected_small or {}, actual_small or {})):
        for field in STAGE3_TEACHER_EXACT_FIELDS:
            name = '{}_{}'.format(prefix, field)
            expected_value = expected.get(field)
            actual_value = actual.get(field)
            if field in STAGE3_TEACHER_FLOAT_FIELDS:
                passed = (
                    expected_value is not None
                    and actual_value is not None
                    and math.isclose(
                        float(actual_value), float(expected_value),
                        rel_tol=0.0, abs_tol=float(tolerance)))
            else:
                passed = (expected_value is not None
                          and actual_value is not None
                          and int(actual_value) == int(expected_value))
            checks[name] = bool(passed)
            if not passed:
                mismatches[name] = dict(
                    expected=expected_value, actual=actual_value)
    return dict(
        passed=all(checks.values()), checks=checks, mismatches=mismatches,
        tolerance=float(tolerance),
        expected=dict(full={field: (expected_full or {}).get(field)
                             for field in STAGE3_TEACHER_EXACT_FIELDS},
                      small={field: (expected_small or {}).get(field)
                             for field in STAGE3_TEACHER_EXACT_FIELDS}),
        actual=dict(full={field: (actual_full or {}).get(field)
                           for field in STAGE3_TEACHER_EXACT_FIELDS},
                    small={field: (actual_small or {}).get(field)
                           for field in STAGE3_TEACHER_EXACT_FIELDS}))


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
                's7_quality_suppression', 's7_temporal_association',
                's7_temporal_student', 's7_static_domain_ranker',
                's7_selective_promotion', 's7_highres_roi_ranker'))
    lane_arbitration = bool(getattr(
        args, 's7_lane_arbitration', False) or getattr(
            args, 'train_components', '') == 's7_lane_arbitration')
    quality_suppression = bool(getattr(
        args, 's7_quality_suppression', False) or getattr(
            args, 'train_components', '') == 's7_quality_suppression')
    temporal_association = bool(getattr(
        args, 's7_temporal_association', False) or getattr(
            args, 'train_components', '') in (
                's7_temporal_association', 's7_temporal_student'))
    temporal_quality_head = bool(getattr(
        args, 's7_temporal_quality_head', False))
    temporal_student = bool(getattr(
        args, 's7_temporal_student', False) or getattr(
            args, 'train_components', '') == 's7_temporal_student')
    static_domain_ranker = bool(getattr(
        args, 's7_static_domain_ranker', False) or getattr(
            args, 'train_components', '') == 's7_static_domain_ranker')
    selective_promotion = bool(getattr(
        args, 's7_selective_promotion', False) or getattr(
            args, 'train_components', '') == 's7_selective_promotion')
    highres_roi_ranker = bool(getattr(
        args, 's7_highres_roi_ranker', False) or getattr(
            args, 'train_components', '') == 's7_highres_roi_ranker')
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
        temporal_student=(
            temporal_student if temporal_association and enabled
            and protected_merge else False),
        temporal_student_hidden=(int(getattr(
            args, 's7_student_hidden', 128))
            if temporal_student and temporal_association and enabled
            and protected_merge else None),
        temporal_max_candidates=(int(getattr(
            args, 's7_temporal_max_candidates', 100))
            if temporal_association and enabled and protected_merge else None),
        temporal_min_confirmations=(int(getattr(
            args, 's7_temporal_min_confirmations', 2))
            if temporal_association and enabled and protected_merge else None),
        static_domain_ranker=(static_domain_ranker if enabled and protected_merge
                              else False),
        static_hidden=(int(getattr(args, 's7_static_hidden', 128))
                       if static_domain_ranker and enabled and protected_merge
                       else None),
        static_score_weight=(float(getattr(args, 's7_static_score_weight', 1.0))
                             if static_domain_ranker and enabled
                             and protected_merge else None),
        static_max_candidates=(int(getattr(
            args, 's7_static_max_candidates', 100))
            if static_domain_ranker and enabled and protected_merge else None),
        selective_promotion=(
            selective_promotion if enabled and protected_merge else False),
        selective_two_frame=(bool(getattr(
            args, 's7_selective_two_frame', False))
            if selective_promotion and enabled and protected_merge else False),
        selective_scalar_channels=(
            temporal.S7SmallTemporalRankerHead.SCALAR_CHANNELS
            if bool(getattr(args, 's7_selective_two_frame', False))
            and selective_promotion and enabled and protected_merge else None),
        selective_hidden=(int(getattr(args, 's7_selective_hidden', 128))
                          if selective_promotion and enabled
                          and protected_merge else None),
        selective_initial_uncertainty=(float(getattr(
            args, 's7_selective_initial_uncertainty', 0.5))
            if selective_promotion and enabled and protected_merge else None),
        selective_promotion_margin=(float(getattr(
            args, 's7_selective_promotion_margin', 0.10))
            if selective_promotion and enabled and protected_merge else None),
        selective_uncertainty_multiplier=(float(getattr(
            args, 's7_selective_uncertainty_multiplier', 1.0))
            if selective_promotion and enabled and protected_merge else None),
        selective_max_candidates=(int(getattr(
            args, 's7_selective_max_candidates', 100))
            if selective_promotion and enabled and protected_merge else None),
        highres_roi_ranker=(highres_roi_ranker if enabled and protected_merge
                            else False),
        highres_channels=(int(getattr(args, 's7_highres_channels', 32))
                          if highres_roi_ranker and enabled and protected_merge
                          else None),
        highres_hidden=(int(getattr(args, 's7_highres_hidden', 32))
                        if highres_roi_ranker and enabled and protected_merge
                        else None),
        highres_max_candidates=(int(getattr(
            args, 's7_highres_max_candidates', 32))
            if highres_roi_ranker and enabled and protected_merge else None),
        highres_score_weight=(float(getattr(
            args, 's7_highres_score_weight', 1.0))
            if highres_roi_ranker and enabled and protected_merge else None),
        highres_promotion_margin=(float(getattr(
            args, 's7_highres_promotion_margin', 0.25))
            if highres_roi_ranker and enabled and protected_merge else None),
        highres_unified_ranking=(bool(getattr(
            args, 's7_highres_unified_ranking', False))
            if highres_roi_ranker and enabled and protected_merge else False),
        highres_pairwise_takeover_v2=(bool(getattr(
            args, 's7_highres_pairwise_takeover_v2', False))
            if highres_roi_ranker and enabled and protected_merge else False),
        takeover_initial_uncertainty=(float(getattr(
            args, 's7_takeover_initial_uncertainty', 0.25))
            if highres_roi_ranker and enabled and protected_merge else None),
        takeover_uncertainty_multiplier=(float(getattr(
            args, 's7_takeover_uncertainty_multiplier', 2.0))
            if highres_roi_ranker and enabled and protected_merge else None),
        takeover_margin=(float(getattr(args, 's7_takeover_margin', 0.05))
                         if highres_roi_ranker and enabled and protected_merge
                         else None),
        takeover_deployment_score_thr=(float(getattr(
            args, 'deployment_score_thr', 0.05))
            if highres_roi_ranker and enabled and protected_merge else None))



def load_heads_checkpoint_state(heads, payload: Dict,
                                allow_s7_base_initialization: bool = False,
                                allow_lane_arbitration_initialization: bool = False,
                                allow_quality_suppression_initialization: bool = False,
                                allow_temporal_association_initialization: bool = False,
                                allow_temporal_student_initialization: bool = False,
                                allow_static_domain_initialization: bool = False,
                                allow_selective_promotion_initialization: bool = False,
                                allow_highres_roi_initialization: bool = False):
    """Load a checkpoint while allowing only explicitly new branch keys."""
    if (allow_s7_base_initialization
            or allow_lane_arbitration_initialization
            or allow_quality_suppression_initialization
            or allow_temporal_association_initialization
            or allow_temporal_student_initialization
            or allow_static_domain_initialization
            or allow_selective_promotion_initialization
            or allow_highres_roi_initialization):
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
        if allow_temporal_student_initialization:
            allowed_prefixes.append('s7_candidate_student_head.')
        if allow_static_domain_initialization:
            allowed_prefixes.append('s7_candidate_static_head.')
        if allow_selective_promotion_initialization:
            allowed_prefixes.append('s7_selective_promotion_head.')
        if allow_highres_roi_initialization:
            allowed_prefixes.extend((
                's7_highres_spatial_projection.',
                's7_highres_candidate_quality_head.',
                's7_highres_pairwise_takeover_head.'))
        disallowed_missing = [
            name for name in incompatible.missing_keys
            if not any(name.startswith(prefix) for prefix in allowed_prefixes)]
        allowed_unexpected_prefixes = []
        if allow_static_domain_initialization:
            # The phase-2 checkpoint contains temporal-only teacher modules;
            # the static experiment intentionally does not instantiate them.
            allowed_unexpected_prefixes.extend((
                's7_temporal_scorer.', 's7_candidate_quality_head.'))
        if allow_selective_promotion_initialization:
            # Keep the phase-2 candidate-quality teacher but drop its temporal
            # scorer; selection is static, pairwise, and native protected.
            allowed_unexpected_prefixes.append('s7_temporal_scorer.')
        if allow_highres_roi_initialization:
            # The phase-2 relative-quality checkpoint contains its temporal
            # teacher; the high-resolution stage starts a separate readout.
            allowed_unexpected_prefixes.extend((
                's7_temporal_scorer.', 's7_candidate_quality_head.'))
        disallowed_unexpected = [
            name for name in incompatible.unexpected_keys
            if not any(name.startswith(prefix)
                       for prefix in allowed_unexpected_prefixes)]
        if disallowed_missing or disallowed_unexpected:
            raise RuntimeError(
                'S7 base initialization state mismatch: missing={} '
                'unexpected={}'.format(
                    disallowed_missing, disallowed_unexpected))
        heads.set_s7_inference_enabled(False)
        if allow_temporal_student_initialization:
            heads.initialize_temporal_student_from_teacher()
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
            in ('s7_temporal_association', 's7_temporal_student',
                's7_static_domain_ranker', 's7_selective_promotion',
                's7_highres_roi_ranker')),
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


def _checkpoint_summary_signature(summary: Dict) -> Dict:
    """Return stable source-evidence fields for checkpoint identity checks."""
    summary = summary or {}
    return {
        key: summary.get(key) for key in (
            'frame_count', 'top1_hits', 'top1_mcml', 'recall_at_20',
            'recall_at_100', 'top1_dfr_fraction_per_frame', 'top1_aci')}


def _checkpoint_state_equal(left: Dict, right: Dict) -> bool:
    """Compare serialized head tensors without relying on checkpoint paths."""
    left_state = left.get('heads_state_dict') or {}
    right_state = right.get('heads_state_dict') or {}
    if set(left_state) != set(right_state):
        return False
    for name in left_state:
        left_value = left_state[name]
        right_value = right_state[name]
        if not isinstance(left_value, torch.Tensor) or not isinstance(
                right_value, torch.Tensor):
            if left_value != right_value:
                return False
        elif not torch.equal(left_value.cpu(), right_value.cpu()):
            return False
    return True


def phase2_selected_checkpoint_provenance_gate(
        result: Dict, candidate_path: str, expected_epoch: int = 4,
        min_full_top1: int = 688, min_small_top1: int = 311,
        max_mcml: int = 3) -> Dict:
    """Allow only a safely verified alias of the phase-2 selected file.

    Phase-2 training writes both a stable ``labeller_best_source_only.pth``
    and an epoch-specific checkpoint.  They can be different paths while
    representing the same selected epoch.  A different path is accepted only
    after validating its complete phase-2 provenance and, when the recorded
    file is available, comparing all serialized head tensors.
    """
    source = result.get('source') or {}
    history = source.get('history') or []
    best_epoch = int(source.get('best_epoch', -1))
    selected = result.get('source_selected_checkpoint')
    candidate_path = os.path.abspath(candidate_path)
    selected_path = (None if not selected else os.path.abspath(selected))
    selected_exists = bool(selected_path and os.path.isfile(selected_path))
    candidate_exists = os.path.isfile(candidate_path)
    matching_rows = [
        row for row in history
        if int(row.get('epoch', -1)) == int(expected_epoch)]
    row = matching_rows[0] if len(matching_rows) == 1 else {}
    full = row.get('source_val') or source.get(
        'best_validation_summary') or {}
    small = row.get('source_small_val') or source.get(
        'best_small_validation_summary') or {}
    retention = row.get('source_exact_retention') or {}
    protocol = result.get('protocol') or {}
    temporal_protocol = protocol.get('s7_temporal_association') or {}
    isolation = result.get('isolation') or {}

    candidate = None
    candidate_load_error = None
    if candidate_exists:
        try:
            candidate = torch.load(candidate_path, map_location='cpu')
        except Exception as error:
            candidate_load_error = '{}: {}'.format(
                type(error).__name__, error)

    candidate_protocol = (candidate or {}).get('training_protocol') or {}
    candidate_temporal = candidate_protocol.get(
        's7_temporal_association') or {}
    candidate_architecture = (candidate or {}).get('s7_architecture') or {}
    candidate_retention = (candidate or {}).get(
        'source_exact_retention') or {}
    candidate_best = (candidate or {}).get(
        'best_source_val_summary') or {}
    candidate_small = (candidate or {}).get(
        'best_source_small_val_summary') or {}
    same_path = bool(selected_path and os.path.realpath(selected_path)
                     == os.path.realpath(candidate_path))
    state_match = same_path
    if not same_path and selected_exists and candidate is not None:
        try:
            selected_payload = torch.load(selected_path, map_location='cpu')
            state_match = _checkpoint_state_equal(selected_payload, candidate)
        except Exception as error:
            candidate_load_error = '{}: {}'.format(
                type(error).__name__, error)
            state_match = False

    checks = dict(
        phase2_selected_checkpoint_present=bool(selected),
        phase2_best_epoch=best_epoch == int(expected_epoch),
        phase2_history_row=len(matching_rows) == 1,
        phase2_row_selected=(row.get('selected_as_best') is True),
        phase2_row_checkpoint_saved=(row.get('checkpoint_saved') is True),
        phase2_source_gate=(
            row.get('source_selection_gate_passed') is True
            and (row.get('source_selection_gate') or {}).get('passed') is True
            and (row.get('s7_source_gate') or {}).get('passed') is True),
        phase2_exact_retention=(
            row.get('source_retention_passed') is True
            and int(retention.get('lost_correct_count', -1)) == 0
            and int(retention.get('retained_correct_count', -1)) == int(
                retention.get('baseline_correct_count', -2))),
        phase2_source_metrics=(
            int(full.get('top1_hits', -1)) >= int(min_full_top1)
            and int(small.get('top1_hits', -1)) >= int(min_small_top1)
            and int(full.get('top1_mcml', max_mcml + 1)) <= int(max_mcml)
            and int(small.get('top1_mcml', max_mcml + 1)) <= int(max_mcml)),
        phase2_target_not_read=(
            result.get('target_dev') is None
            and temporal_protocol.get('target_read') is False
            and isolation.get('target_used_for_training') is False
            and isolation.get('target_used_for_checkpoint_selection') is False
            and isolation.get('target_labels_used_for_evaluation_only') is False),
        phase2_training_protocol=(
            isolation.get('train_components') == 's7_temporal_association'),
        phase2_relative_quality_protocol=(
            temporal_protocol.get('candidate_quality_head') is True
            and temporal_protocol.get('relative_quality') is True
            and temporal_protocol.get('target_read') is False),
        candidate_exists=candidate_exists,
        candidate_loadable=candidate is not None and not candidate_load_error,
        candidate_source_only=(candidate or {}).get('source_only') is True,
        candidate_frozen_dinov2=(candidate or {}).get('frozen_dinov2') is True,
        candidate_s7_enabled=(candidate or {}).get(
            's7_inference_enabled') is True,
        candidate_epoch=int((candidate or {}).get('epoch', -1))
        == int(expected_epoch),
        candidate_best_epoch=int((candidate or {}).get('best_epoch', -1))
        == int(expected_epoch),
        candidate_training_protocol=(
            candidate_protocol.get('train_components')
            == 's7_temporal_association'),
        candidate_relative_quality=(
            candidate_architecture.get('temporal_association') is True
            and candidate_architecture.get('temporal_quality_head') is True
            and candidate_temporal.get('candidate_quality_head') is True
            and candidate_temporal.get('relative_quality') is True
            and candidate_temporal.get('target_read') is False),
        candidate_source_gate=(
            (candidate or {}).get('source_selection_gate_passed') is True),
        candidate_exact_retention=(
            int(candidate_retention.get('lost_correct_count', -1)) == 0
            and int(candidate_retention.get('retained_correct_count', -1))
            == int(candidate_retention.get('baseline_correct_count', -2))),
        candidate_source_metrics=(
            _checkpoint_summary_signature(candidate_best)
            == _checkpoint_summary_signature(full)
            and _checkpoint_summary_signature(candidate_small)
            == _checkpoint_summary_signature(small)),
        checkpoint_identity=(
            state_match or (not selected_exists and not same_path)),
    )
    return dict(
        checks=checks, passed=all(checks.values()),
        selected_checkpoint=selected, candidate_checkpoint=candidate_path,
        selected_checkpoint_exists=selected_exists,
        checkpoint_identity='same_path' if same_path else (
            'head_state_equal' if state_match else
            'selected_checkpoint_unavailable' if not selected_exists else
            'head_state_mismatch'),
        candidate_load_error=candidate_load_error)


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
                       source_exact_retention: Optional[Dict] = None,
                       source_deployment_exact_retention: Optional[Dict] = None
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
        source_deployment_exact_retention=(
            source_deployment_exact_retention),
        s7_student_teacher_reproduction=(
            getattr(args, 's7_student_teacher_reproduction_gate', None)
            if args.train_components == 's7_temporal_student' else None),
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
                    args, 's7_temporal_relative_quality', False)),
                unified_highres=bool(getattr(
                    args, 's7_highres_unified_ranking', False)),
                pairwise_takeover_v2=bool(getattr(
                    args, 's7_highres_pairwise_takeover_v2', False))),
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
                    's7_quality_suppression', 's7_temporal_association',
                    's7_temporal_student', 's7_static_domain_ranker',
                    's7_selective_promotion', 's7_highres_roi_ranker')
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
                None if args.train_components not in (
                    's7_temporal_association', 's7_temporal_student')
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
            s7_temporal_student=(
                None if args.train_components != 's7_temporal_student'
                else dict(
                    base_checkpoint=os.path.abspath(args.init_checkpoint),
                    base_epoch=int(args.s7_student_base_epoch),
                    teacher='frozen_phase2_relative_quality_epoch4',
                    student_head='candidate_quality_head_copy',
                    quality_loss_weight=float(
                        args.s7_student_quality_loss_weight),
                    relative_loss_weight=float(
                        args.s7_student_relative_loss_weight),
                    distillation_weight=float(
                        args.s7_student_distillation_weight),
                    distillation_temperature=float(
                        args.s7_student_distillation_temperature),
                    small_loss_weight=float(
                        args.s7_student_small_loss_weight),
                    small_token_threshold=float(
                        args.s7_student_small_token_thr),
                    small_weight_training_only=True,
                    inference_slice_routing=False,
                    source_only=True, target_read=False,
                    frozen_detector=True, frozen_teacher=True)),
            s7_static_domain_ranker=(
                None if args.train_components != 's7_static_domain_ranker'
                else dict(
                    base_checkpoint=os.path.abspath(args.init_checkpoint),
                    teacher_result_json=os.path.abspath(
                        args.s7_static_teacher_result_json),
                    base_epoch=int(args.s7_static_base_epoch),
                    quality_head='static_candidate_quality_head',
                    hidden=int(args.s7_static_hidden),
                    quality_loss_weight=float(
                        args.s7_static_quality_loss_weight),
                    relative_loss_weight=float(
                        args.s7_static_relative_loss_weight),
                    relative_margin=float(args.s7_static_relative_margin),
                    relative_min_gap=float(args.s7_static_relative_min_gap),
                    relative_max_pairs=int(args.s7_static_relative_max_pairs),
                    score_weight=float(args.s7_static_score_weight),
                    rank_margin=float(args.s7_static_rank_margin),
                    retention_weight=float(args.s7_static_retention_weight),
                    gain_weight=float(args.s7_static_gain_weight),
                    prior_weight=float(args.s7_static_prior_weight),
                    max_candidates=int(args.s7_static_max_candidates),
                    source_feature_domain_augmentation=dict(
                        enabled=True,
                        probability=float(args.s7_static_aug_prob),
                        strength=float(args.s7_static_aug_strength),
                        operations=['brightness', 'blur', 'scale']),
                    source_only=True, target_read=False,
                    temporal_association=False,
                    inference_slice_routing=False,
                    gain_replay=False, positive_promotion=True,
                    frozen_detector=True, exact_source_retention=True)),
            s7_selective_promotion=(
                None if args.train_components != 's7_selective_promotion'
                else dict(
                    base_checkpoint=os.path.abspath(args.init_checkpoint),
                    teacher_result_json=os.path.abspath(
                        args.s7_selective_teacher_result_json),
                    base_epoch=int(args.s7_selective_base_epoch),
                    frozen_teacher='phase2_candidate_quality_head',
                    trainable='native_vs_s7_advantage_uncertainty_head_only',
                    version=('v2_two_frame_constant_velocity'
                             if bool(getattr(
                                 args, 's7_selective_two_frame', False))
                             else 'v1_static_pair'),
                    two_frame_constant_velocity=bool(getattr(
                        args, 's7_selective_two_frame', False)),
                    scalar_channels=(
                        temporal.S7SmallTemporalRankerHead.SCALAR_CHANNELS
                        if bool(getattr(
                            args, 's7_selective_two_frame', False)) else None),
                    quality_prefilter=(
                        'one_s7_from_lane_top20'
                        if bool(getattr(
                            args, 's7_selective_two_frame', False))
                        else 'all_s7_lane_candidates'),
                    hidden=int(args.s7_selective_hidden),
                    initial_uncertainty=float(
                        args.s7_selective_initial_uncertainty),
                    advantage_gap=float(args.s7_selective_advantage_gap),
                    promotion_margin=float(
                        args.s7_selective_promotion_margin),
                    uncertainty_multiplier=float(
                        args.s7_selective_uncertainty_multiplier),
                    quality_loss_weight=float(
                        args.s7_selective_quality_loss_weight),
                    classification_loss_weight=float(
                        args.s7_selective_classification_loss_weight),
                    retention_weight=float(
                        args.s7_selective_retention_weight),
                    gain_weight=float(args.s7_selective_gain_weight),
                    prior_weight=float(args.s7_selective_prior_weight),
                    max_candidates=int(args.s7_selective_max_candidates),
                    min_gain_sequences=int(
                        args.s7_selective_min_gain_sequences),
                    source_feature_domain_augmentation=dict(
                        probability=float(args.s7_selective_aug_prob),
                        strength=float(args.s7_selective_aug_strength),
                        operations=['brightness', 'blur', 'scale']),
                    inference='lower_confidence_bound_selective_promotion',
                    native_fallback=True, positive_promotion=True,
                    temporal_association=bool(getattr(
                        args, 's7_selective_two_frame', False)),
                    causal_history_frames=(2 if bool(getattr(
                        args, 's7_selective_two_frame', False)) else 0),
                    additional_dino_forward=False,
                    dense_feature_history=False,
                    inference_slice_routing=False,
                    sequence_identity_feature=False,
                    source_only=True, target_read=False,
                    frozen_detector=True, exact_source_retention=True)),
            s7_highres_roi_ranker=(
                None if args.train_components != 's7_highres_roi_ranker'
                else dict(
                    base_checkpoint=os.path.abspath(args.init_checkpoint),
                    teacher_result_json=os.path.abspath(
                        args.s7_highres_teacher_result_json),
                    base_epoch=int(args.s7_highres_base_epoch),
                    trainable=[
                        's7_highres_spatial_projection',
                        ('s7_highres_pairwise_takeover_head'
                         if bool(getattr(
                             args, 's7_highres_pairwise_takeover_v2', False))
                         else 's7_highres_candidate_quality_head')],
                    frozen_detector=True,
                    highres_channels=int(args.s7_highres_channels),
                    hidden=int(args.s7_highres_hidden),
                    max_candidates=int(args.s7_highres_max_candidates),
                    score_weight=float(args.s7_highres_score_weight),
                    rank_margin=float(args.s7_highres_rank_margin),
                    promotion_margin=float(args.s7_highres_promotion_margin),
                    quality_loss_weight=float(
                        args.s7_highres_quality_loss_weight),
                    relative_loss_weight=float(
                        args.s7_highres_relative_loss_weight),
                    relative_min_gap=float(args.s7_highres_relative_min_gap),
                    relative_max_pairs=int(
                        args.s7_highres_relative_max_pairs),
                    retention_weight=float(args.s7_highres_retention_weight),
                    gain_weight=float(args.s7_highres_gain_weight),
                    prior_weight=float(args.s7_highres_prior_weight),
                    unified_ranking=bool(getattr(
                        args, 's7_highres_unified_ranking', False)),
                    pairwise_takeover_v2=bool(getattr(
                        args, 's7_highres_pairwise_takeover_v2', False)),
                    takeover_objective=(dict(
                        target='s7_RIoU_minus_native_top1_RIoU',
                        uncertainty='heteroscedastic_source_LCB',
                        multiplier=float(
                            args.s7_takeover_uncertainty_multiplier),
                        margin=float(args.s7_takeover_margin),
                        deployment_score_thr=float(
                            args.deployment_score_thr),
                        clean_aug_consistency=True,
                        source_domain_balancing=True,
                        group_dro_eta=float(args.s7_takeover_group_dro_eta),
                        raw_and_deployment_exact_retention=True)
                        if bool(getattr(
                            args, 's7_highres_pairwise_takeover_v2', False))
                        else None),
                    unified_hard_pairs=int(getattr(
                        args, 's7_highres_unified_hard_pairs', 8)),
                    source_feature_domain_augmentation=(
                        dict(
                            probability=float(getattr(
                                args, 's7_highres_unified_aug_prob', 0.75)),
                            strength=float(getattr(
                                args, 's7_highres_unified_aug_strength', 0.15)),
                            operations=['brightness', 'blur', 'scale'])
                        if (bool(getattr(
                            args, 's7_highres_unified_ranking', False))
                            or bool(getattr(
                                args, 's7_highres_pairwise_takeover_v2',
                                False)))
                        else None),
                    readout='frozen_s7_feature_stride7_roi_align',
                    candidate_pool='native_top1_plus_s7_lane_topk',
                    inference=(
                        'pairwise_delta_riou_LCB_native_abstention'
                        if bool(getattr(
                            args, 's7_highres_pairwise_takeover_v2', False))
                        else 'unified_native_protected_quality_margin'
                        if bool(getattr(
                            args, 's7_highres_unified_ranking', False))
                        else 'native_protected_quality_margin'),
                    additional_dino_forward=False,
                    dense_feature_history=False,
                    foreground_branch=False,
                    temporal_association=False,
                    inference_slice_routing=False,
                    sequence_identity_feature=False,
                    source_only=True, target_read=False,
                    exact_source_retention=True)),
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
                        allow_temporal_student_initialization: bool = False,
                        allow_static_domain_initialization: bool = False,
                        allow_selective_promotion_initialization: bool = False,
                        allow_highres_roi_initialization: bool = False,
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
                    and not (allow_temporal_association_initialization
                             or allow_static_domain_initialization
                             or allow_selective_promotion_initialization
                             or allow_highres_roi_initialization)):
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
                    and not (allow_temporal_association_initialization
                             or allow_static_domain_initialization
                             or allow_selective_promotion_initialization
                             or allow_highres_roi_initialization)):
                architecture_mismatch = True
            if (stored_temporal_quality and requested_temporal_quality
                    and stored_s7.get('temporal_quality_hidden')
                    != requested_s7.get('temporal_quality_hidden')):
                architecture_mismatch = True
            stored_student = bool(stored_s7.get(
                'temporal_student', False))
            requested_student = bool(requested_s7.get(
                'temporal_student', False))
            if (stored_student != requested_student
                    and not (allow_temporal_student_initialization
                             or allow_static_domain_initialization
                             or allow_selective_promotion_initialization
                             or allow_highres_roi_initialization)):
                architecture_mismatch = True
            if (stored_student and requested_student
                    and stored_s7.get('temporal_student_hidden')
                    != requested_s7.get('temporal_student_hidden')):
                architecture_mismatch = True
            stored_static = bool(stored_s7.get(
                'static_domain_ranker', False))
            requested_static = bool(requested_s7.get(
                'static_domain_ranker', False))
            if (stored_static != requested_static
                    and not (allow_static_domain_initialization
                             or allow_selective_promotion_initialization
                             or allow_highres_roi_initialization)):
                architecture_mismatch = True
            if (stored_static and requested_static
                    and (stored_s7.get('static_hidden')
                         != requested_s7.get('static_hidden')
                         or stored_s7.get('static_score_weight')
                         != requested_s7.get('static_score_weight')
                         or stored_s7.get('static_max_candidates')
                         != requested_s7.get('static_max_candidates'))):
                architecture_mismatch = True
            stored_selective = bool(stored_s7.get(
                'selective_promotion', False))
            requested_selective = bool(requested_s7.get(
                'selective_promotion', False))
            if (stored_selective != requested_selective
                    and not allow_selective_promotion_initialization):
                architecture_mismatch = True
            if (stored_selective and requested_selective
                    and (stored_s7.get('selective_hidden')
                         != requested_s7.get('selective_hidden')
                         or stored_s7.get('selective_initial_uncertainty')
                         != requested_s7.get(
                             'selective_initial_uncertainty')
                         or stored_s7.get('selective_promotion_margin')
                         != requested_s7.get('selective_promotion_margin')
                         or stored_s7.get(
                             'selective_uncertainty_multiplier')
                         != requested_s7.get(
                             'selective_uncertainty_multiplier')
                         or stored_s7.get('selective_max_candidates')
                         != requested_s7.get('selective_max_candidates')
                         or bool(stored_s7.get('selective_two_frame', False))
                         != bool(requested_s7.get(
                             'selective_two_frame', False))
                         or stored_s7.get('selective_scalar_channels')
                         != requested_s7.get(
                             'selective_scalar_channels'))):
                architecture_mismatch = True
            stored_highres = bool(stored_s7.get(
                'highres_roi_ranker', False))
            requested_highres = bool(requested_s7.get(
                'highres_roi_ranker', False))
            if (stored_highres != requested_highres
                    and not allow_highres_roi_initialization):
                architecture_mismatch = True
            pairwise_takeover_initialization = bool(
                allow_highres_roi_initialization
                and requested_s7.get('highres_pairwise_takeover_v2', False))
            if (stored_highres and requested_highres
                    and (stored_s7.get('highres_channels')
                         != requested_s7.get('highres_channels')
                         or stored_s7.get('highres_hidden')
                         != requested_s7.get('highres_hidden'))):
                # These dimensions determine checkpoint tensor shapes and can
                # never change while reusing the V1 spatial projection.
                architecture_mismatch = True
            if (stored_highres and requested_highres
                    and not pairwise_takeover_initialization
                    and (stored_s7.get('highres_max_candidates')
                         != requested_s7.get('highres_max_candidates')
                         or stored_s7.get('highres_score_weight')
                         != requested_s7.get('highres_score_weight')
                         or stored_s7.get('highres_promotion_margin')
                         != requested_s7.get('highres_promotion_margin'))):
                # Candidate count and selection margins are policy fields.
                # Only the explicitly gated V1 -> Pairwise V2 initialization
                # may replace them; ordinary resume/evaluation remains strict.
                architecture_mismatch = True
            if (bool(stored_s7.get('highres_unified_ranking', False))
                    != bool(requested_s7.get('highres_unified_ranking', False))
                    and not allow_highres_roi_initialization):
                architecture_mismatch = True
            if (bool(stored_s7.get(
                    'highres_pairwise_takeover_v2', False))
                    != bool(requested_s7.get(
                        'highres_pairwise_takeover_v2', False))
                    and not allow_highres_roi_initialization):
                architecture_mismatch = True
            if (bool(stored_s7.get(
                    'highres_pairwise_takeover_v2', False))
                    and bool(requested_s7.get(
                        'highres_pairwise_takeover_v2', False))
                    and (stored_s7.get('takeover_initial_uncertainty')
                         != requested_s7.get('takeover_initial_uncertainty')
                         or stored_s7.get(
                             'takeover_uncertainty_multiplier')
                         != requested_s7.get(
                             'takeover_uncertainty_multiplier')
                         or stored_s7.get('takeover_margin')
                         != requested_s7.get('takeover_margin')
                         or stored_s7.get(
                             'takeover_deployment_score_thr')
                         != requested_s7.get(
                             'takeover_deployment_score_thr'))):
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
    s7_student_mode = args.train_components == 's7_temporal_student'
    s7_static_mode = args.train_components == 's7_static_domain_ranker'
    s7_selective_mode = args.train_components == 's7_selective_promotion'
    s7_highres_mode = args.train_components == 's7_highres_roi_ranker'
    s7_temporal_relative_mode = bool(getattr(
        args, 's7_temporal_relative_quality', False))
    s7_mode = bool(
        s7_rpn_mode or s7_merge_mode or s7_lane_mode or s7_quality_mode
        or s7_temporal_mode or s7_student_mode or s7_static_mode
        or s7_selective_mode or s7_highres_mode)
    protected_source_mode = bool(roi_cls_mode or s7_mode)
    args.s7_student_teacher_reproduction_gate = None
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
    source_baseline_deployment_correct_keys = None
    source_baseline_rows = None
    source_initial_temporal_summary = None
    source_initial_temporal_small_summary = None
    s7_quality_support_audit = None
    history = []
    contextual_source_validation = bool(
        s7_temporal_mode or s7_student_mode
        or (s7_selective_mode and getattr(
            args, 's7_selective_two_frame', False)))
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
        if s7_temporal_relative_mode and not s7_student_mode:
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
        if s7_student_mode:
            if int(payload.get('epoch', -1)) != int(
                    args.s7_student_base_epoch):
                raise RuntimeError(
                    'Temporal student must initialize from selected phase-2 '
                    'epoch {}; found {}'.format(
                        args.s7_student_base_epoch, payload.get('epoch')))
            stored_mode = payload.get(
                'training_protocol', {}).get('train_components')
            stored_s7 = payload.get('s7_architecture', {})
            if (stored_mode != 's7_temporal_association'
                    or not bool(stored_s7.get(
                        'temporal_quality_head', False))
                    or not bool((payload.get('training_protocol') or {}).get(
                        's7_temporal_association', {}).get(
                            'relative_quality', False))):
                raise RuntimeError(
                    'Temporal student requires the complete phase-2 '
                    'relative-quality checkpoint')
            base_gate = source_selected_checkpoint_gate(payload)
            if not base_gate['passed']:
                failed = sorted(name for name, passed
                                in base_gate['checks'].items() if not passed)
                raise RuntimeError(
                    'Temporal student base checkpoint failed source gate: '
                    + ', '.join(failed))
        if s7_static_mode:
            if int(payload.get('epoch', -1)) != int(
                    args.s7_static_base_epoch):
                raise RuntimeError(
                    'Static ranker must initialize from phase-2 epoch {}; '
                    'found epoch {}'.format(
                        args.s7_static_base_epoch, payload.get('epoch')))
            stored_mode = payload.get(
                'training_protocol', {}).get('train_components')
            stored_s7 = payload.get('s7_architecture', {})
            stored_protocol = payload.get('training_protocol') or {}
            if (stored_mode != 's7_temporal_association'
                    or not bool(stored_s7.get('temporal_quality_head', False))
                    or not bool((stored_protocol.get(
                        's7_temporal_association') or {}).get(
                            'relative_quality', False))):
                raise RuntimeError(
                    'Static ranker requires the complete phase-2 '
                    'relative-quality checkpoint')
            base_gate = source_selected_checkpoint_gate(payload)
            if not base_gate['passed']:
                failed = sorted(name for name, passed
                                in base_gate['checks'].items() if not passed)
                raise RuntimeError(
                    'Static ranker base checkpoint failed source gate: '
                    + ', '.join(failed))
        if s7_selective_mode:
            if int(payload.get('epoch', -1)) != int(
                    args.s7_selective_base_epoch):
                raise RuntimeError(
                    'Selective promotion must initialize from phase-2 epoch '
                    '{}; found epoch {}'.format(
                        args.s7_selective_base_epoch, payload.get('epoch')))
            stored_protocol = payload.get('training_protocol') or {}
            stored_s7 = payload.get('s7_architecture', {})
            if (stored_protocol.get('train_components')
                    != 's7_temporal_association'
                    or not bool(stored_s7.get(
                        'temporal_quality_head', False))
                    or not bool((stored_protocol.get(
                        's7_temporal_association') or {}).get(
                            'relative_quality', False))):
                raise RuntimeError(
                    'Selective promotion requires the complete phase-2 '
                    'relative-quality checkpoint')
            base_gate = source_selected_checkpoint_gate(payload)
            if not base_gate['passed']:
                failed = sorted(name for name, passed
                                in base_gate['checks'].items() if not passed)
                raise RuntimeError(
                    'Selective promotion base checkpoint failed source gate: '
                    + ', '.join(failed))
        if s7_highres_mode:
            if int(payload.get('epoch', -1)) != int(
                    args.s7_highres_base_epoch):
                raise RuntimeError(
                    'High-resolution ranker must initialize from phase-2 '
                    'epoch {}; found epoch {}'.format(
                        args.s7_highres_base_epoch, payload.get('epoch')))
            stored_protocol = payload.get('training_protocol') or {}
            stored_s7 = payload.get('s7_architecture', {})
            if bool(getattr(
                    args, 's7_highres_pairwise_takeover_v2', False)):
                if (stored_protocol.get('train_components')
                        != 's7_highres_roi_ranker'
                        or not bool(stored_s7.get('highres_roi_ranker', False))
                        or not bool(stored_s7.get(
                            'highres_unified_ranking', False))):
                    raise RuntimeError(
                        'Pairwise Takeover V2 requires the locked unified V1 '
                        'high-resolution epoch checkpoint')
            else:
                if (stored_protocol.get('train_components')
                        != 's7_temporal_association'
                        or not bool(stored_s7.get(
                            'temporal_quality_head', False))
                        or not bool((stored_protocol.get(
                            's7_temporal_association') or {}).get(
                                'relative_quality', False))):
                    raise RuntimeError(
                        'High-resolution ranker requires the complete phase-2 '
                        'relative-quality checkpoint')
                base_gate = source_selected_checkpoint_gate(payload)
                if not base_gate['passed']:
                    failed = sorted(name for name, passed in
                                    base_gate['checks'].items() if not passed)
                    raise RuntimeError(
                        'High-resolution ranker base checkpoint failed source '
                        'gate: ' + ', '.join(failed))
        validate_checkpoint(
            payload, in_channels, args,
            allow_training_mode_mismatch=True,
            allow_s7_base_initialization=(
                s7_mode and not s7_lane_mode and not s7_quality_mode
                and not s7_temporal_mode and not s7_student_mode
                and not s7_static_mode and not s7_selective_mode),
            allow_lane_arbitration_initialization=s7_lane_mode,
            allow_quality_suppression_initialization=s7_quality_mode,
            allow_temporal_association_initialization=(
                s7_temporal_mode and not s7_temporal_relative_mode),
            allow_temporal_student_initialization=s7_student_mode,
            allow_static_domain_initialization=s7_static_mode,
            allow_selective_promotion_initialization=s7_selective_mode,
            allow_highres_roi_initialization=s7_highres_mode,
            allow_temporal_policy_mismatch=s7_temporal_relative_mode)
        load_heads_checkpoint_state(
            heads, payload,
            allow_s7_base_initialization=(
                s7_mode and not s7_lane_mode and not s7_quality_mode
                and not s7_temporal_mode and not s7_student_mode
                and not s7_static_mode and not s7_selective_mode),
            allow_lane_arbitration_initialization=s7_lane_mode,
            allow_quality_suppression_initialization=s7_quality_mode,
            allow_temporal_association_initialization=(
                s7_temporal_mode and not s7_temporal_relative_mode),
            allow_temporal_student_initialization=s7_student_mode,
            allow_static_domain_initialization=s7_static_mode,
            allow_selective_promotion_initialization=s7_selective_mode,
            allow_highres_roi_initialization=s7_highres_mode)
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
            source_baseline_rows = baseline_rows
            source_baseline_summary = summarize_rows(baseline_rows)
            source_baseline_correct_keys = source_correct_frame_keys(
                baseline_rows)
            source_baseline_deployment_correct_keys = (
                source_deployment_correct_frame_keys(baseline_rows))
            if contextual_source_validation:
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
            if s7_temporal_mode or s7_student_mode:
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
                if s7_student_mode:
                    teacher_result = getattr(
                        args, 's7_student_teacher_result', None)
                    if teacher_result is None:
                        raise RuntimeError(
                            'Stage-3 teacher result was not loaded before '
                            'the reproduction check')
                    teacher_source = teacher_result.get('source') or {}
                    reproduction_gate = stage3_teacher_reproduction_gate(
                        teacher_source.get('best_validation_summary'),
                        teacher_source.get('best_small_validation_summary'),
                        source_initial_temporal_summary,
                        source_initial_temporal_small_summary)
                    args.s7_student_teacher_reproduction_gate = (
                        reproduction_gate)
                    if not reproduction_gate['passed']:
                        failed = sorted(
                            name for name, passed in
                            reproduction_gate['checks'].items() if not passed)
                        raise RuntimeError(
                            'Copied stage-3 student does not exactly reproduce '
                            'the phase-2 teacher: ' + ', '.join(failed))
                    initial_retention = source_top1_retention_summary(
                        source_baseline_correct_keys, initial_temporal_rows)
                    initial_gate = s7_source_selection_gate(
                        source_baseline_summary,
                        source_baseline_small_summary,
                        source_initial_temporal_summary,
                        source_initial_temporal_small_summary,
                        initial_retention, args)
                    if not initial_gate['passed']:
                        failed = sorted(
                            name for name, passed
                            in initial_gate['checks'].items() if not passed)
                        raise RuntimeError(
                            'Copied stage-3 student does not reproduce the '
                            'source-gated teacher: ' + ', '.join(failed))
                    best_summary = source_initial_temporal_summary
                    best_small_summary = (
                        source_initial_temporal_small_summary)
                    best_key = roi_cls_selection_key(
                        best_summary, best_small_summary)
                    atomic_torch_save(checkpoint_payload(
                        heads, None, None, 0, 0, best_summary,
                        in_channels, args, global_step,
                        best_small_summary, source_sampling,
                        source_baseline_summary,
                        source_baseline_small_summary,
                        source_baseline_correct_keys,
                        source_selection_gate_passed=True,
                        source_exact_retention=initial_retention), best_path)
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
        sequence_gain = None
        if evaluate_epoch:
            val_rows = evaluate_records(
                dino, heads, val_records, args, dino_device, head_device,
                role='source_validation')
            val_summary = summarize_rows(val_rows)
            if protected_source_mode:
                if contextual_source_validation:
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
        deployment_retention_summary = None
        source_domain_summary = None
        protected_gate = None
        if val_summary is not None and protected_source_mode:
            if args.train_components in pairwise_modes or s7_mode:
                retention_summary = source_top1_retention_summary(
                    source_baseline_correct_keys, val_rows)
                if s7_highres_mode and bool(getattr(
                        args, 's7_highres_pairwise_takeover_v2', False)):
                    if (source_baseline_rows is None
                            or source_baseline_deployment_correct_keys is None):
                        raise RuntimeError(
                            'Pairwise Takeover V2 lost its source baselines')
                    deployment_retention_summary = (
                        source_deployment_retention_summary(
                            source_baseline_deployment_correct_keys,
                            val_rows))
                    source_domain_summary = (
                        source_domain_nonregression_summary(
                            source_baseline_rows, val_rows))
                if s7_mode:
                    if (s7_merge_mode or s7_lane_mode or s7_quality_mode
                            or s7_temporal_mode or s7_student_mode
                            or s7_static_mode or s7_selective_mode):
                        merge_conflicts = source_merge_conflict_summary(
                            source_baseline_correct_keys, val_rows)
                    protected_gate = s7_source_selection_gate(
                        source_baseline_summary,
                        source_baseline_small_summary,
                        val_summary, small_val_summary,
                        retention_summary, args)
                    if s7_highres_mode and bool(getattr(
                            args, 's7_highres_pairwise_takeover_v2', False)):
                        protected_gate['checks'].update(
                            exact_deployment_retention=bool(
                                deployment_retention_summary[
                                    'lost_correct_count'] == 0),
                            per_source_domain_nonregression=bool(
                                source_domain_summary['passed']),
                            positive_source_domain_gain=bool(
                                source_domain_summary[
                                    'any_positive_domain']))
                        protected_gate['deployment_exact_retention'] = (
                            deployment_retention_summary)
                        protected_gate['source_domain_nonregression'] = (
                            source_domain_summary)
                        protected_gate['passed'] = all(
                            protected_gate['checks'].values())
                    if s7_selective_mode:
                        if source_baseline_rows is None:
                            raise RuntimeError(
                                'Selective promotion lost its source baseline '
                                'rows before the multi-sequence gate')
                        sequence_gain = source_sequence_gain_summary(
                            source_baseline_rows, val_rows,
                            args.s7_selective_min_gain_sequences)
                        protected_gate['checks'][
                            'multi_sequence_net_gain'] = bool(
                                sequence_gain['passed'])
                        promotion_summary = selective_promotion_summary(
                            val_rows)
                        protected_gate['checks'][
                            'nonzero_s7_promotion'] = bool(
                                promotion_summary['nonzero_s7_promotion'])
                        protected_gate['selective_promotion'] = (
                            promotion_summary)
                        if bool(getattr(
                                args, 's7_selective_two_frame', False)):
                            small_keys = {
                                source_frame_key(row)
                                for row in small_val_records}
                            promotion_effect = (
                                selective_promotion_effect_summary(
                                    source_baseline_rows, val_rows,
                                    small_keys,
                                    args.s7_selective_min_gain_sequences))
                            protected_gate['checks'].update(
                                promotion_effect['checks'])
                            protected_gate[
                                'selective_promotion_effect'] = (
                                    promotion_effect)
                        protected_gate['passed'] = all(
                            protected_gate['checks'].values())
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
                source_exact_retention=retention_summary,
                source_deployment_exact_retention=(
                    deployment_retention_summary)), best_path)
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
                source_exact_retention=retention_summary,
                source_deployment_exact_retention=(
                    deployment_retention_summary)), epoch_path)
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
            source_deployment_exact_retention=(
                deployment_retention_summary),
            s7_merge_calibration=(
                s7_calibration_state(heads)
                if (s7_merge_mode or s7_lane_mode or s7_quality_mode
                    or s7_temporal_mode or s7_student_mode
                    or s7_static_mode or s7_selective_mode) else None),
            s7_temporal_cue_weights=(
                heads.s7_temporal_scorer.state_summary()
                if (s7_temporal_mode or s7_student_mode) else None),
            s7_merge_conflicts=merge_conflicts,
            source_selection_gate=protected_gate,
            pairwise_v2_source_gate=(
                protected_gate if args.train_components ==
                'roi_cls_pairwise_v2' else None),
            s7_source_gate=(protected_gate if s7_mode else None),
            source_sequence_generalization=sequence_gain,
            source_domain_generalization=source_domain_summary,
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
            source_exact_retention=retention_summary,
            source_deployment_exact_retention=(
                deployment_retention_summary)), latest_path)
        progress_path = None
        progress_replacements = 0
        if args.train_components in (
                'roi_cls_pairwise_v2', 's7_rpn', 's7_merge',
                's7_lane_arbitration', 's7_quality_suppression',
                's7_temporal_association', 's7_temporal_student',
                's7_static_domain_ranker', 's7_selective_promotion',
                's7_highres_roi_ranker'):
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
                if contextual_source_validation:
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
    if bool(getattr(args, 'source_highres_margin_audit', False)):
        spec = args.source_highres_margin_audit_spec
        checkpoint_payload = torch.load(
            args.eval_only_checkpoint, map_location='cpu')
        validate_checkpoint(checkpoint_payload, in_channels, args)
        if int(checkpoint_payload.get('epoch', -1)) != int(spec['epoch']):
            raise RuntimeError(
                'High-resolution margin checkpoint epoch mismatch')
        load_heads_checkpoint_state(heads, checkpoint_payload)
        for parameter in heads.parameters():
            parameter.requires_grad = False
        trainable_names = []
        head_versions = common.module_parameter_versions(heads)
        audit = build_highres_margin_source_audit(
            dino, heads, source_val, args,
            dino_device, head_device, spec)
        dino_unchanged = (
            dino_versions == common.module_parameter_versions(dino))
        heads_unchanged = (
            head_versions == common.module_parameter_versions(heads))
        if not dino_unchanged or not heads_unchanged:
            raise RuntimeError(
                'Read-only high-resolution margin audit changed parameters')
        unified_audit = spec.get('audit_variant') == 'unified_bounded_risk'
        if audit['formal_gate_passed']:
            decision = (
                'SOURCE_ONLY_UNIFIED_HIGHRES_MARGIN_AUDIT_FORMAL_GATE_'
                'PASSED_TARGET_NOT_READ' if unified_audit else
                'SOURCE_ONLY_HIGHRES_MARGIN_AUDIT_GATE_PASSED_TARGET_NOT_READ')
        elif (unified_audit
              and audit['bounded_risk_research_gate_passed']):
            decision = (
                'SOURCE_ONLY_UNIFIED_HIGHRES_BOUNDED_RISK_RESEARCH_GATE_'
                'PASSED_TARGET_NOT_READ')
        else:
            decision = (
                'SOURCE_ONLY_UNIFIED_HIGHRES_MARGIN_AUDIT_ALL_GATES_'
                'FAILED_TARGET_NOT_READ' if unified_audit else
                'SOURCE_ONLY_HIGHRES_MARGIN_AUDIT_GATE_FAILED_TARGET_NOT_READ')
        result = dict(
            protocol_version=(26 if unified_audit else 24),
            protocol=dict(
                architecture=(
                    'frozen_unified_highres_epoch3_shared_forward_margin_audit'
                    if unified_audit else
                    'frozen_highres_epoch3_shared_forward_margin_audit'),
                source_data=source_protocol,
                checkpoint_selection=(
                    'formal_exact_gate_then_bounded_risk_research_gate'
                    if unified_audit else
                    'highest_promotion_margin_among_formal_source_gate_passers'),
                fixed_margins=list(audit['margins']),
                shared_model_forward=True,
                parameter_update=False, source_only=True,
                target_read=False,
                source_gate=dict(
                    exact_retention=True, min_full_top1=int(
                        args.s7_source_min_full_top1),
                    min_small_top1=int(args.s7_source_min_small_top1),
                    max_mcml=int(args.s7_source_max_mcml),
                    dfr_aci_nonregression=True,
                    bounded_risk_research_continuation=(
                        None if not unified_audit else dict(
                            max_lost_correct=1,
                            max_lost_fraction=0.002,
                            min_gain_loss_ratio=10.0,
                            min_gain_sequences=2,
                            source_safe_claim_allowed=False,
                            deployment_claim_allowed=False,
                            full_test_allowed=False)))),
            isolation=dict(
                dino_frozen=True,
                dino_parameters_unchanged=bool(dino_unchanged),
                detector_parameters_unchanged=bool(heads_unchanged),
                read_only_evaluation=True,
                parameter_updates_performed=False,
                trainable_parameter_names=trainable_names,
                trainable_parameter_count=0,
                target_used_for_training=False,
                target_used_for_checkpoint_selection=False,
                target_labels_used_for_evaluation_only=False),
            architecture=dict(
                in_channels=in_channels, patch_size=int(args.patch_size),
                rpn=rpn_config(in_channels, args),
                roi=roi_config(in_channels, args),
                s7=s7_architecture(args),
                s7_rpn=s7_rpn_config(
                    int(getattr(args, 's7_channels', 128)), args)),
            source_highres_margin_audit=audit,
            source_selected_checkpoint=(
                audit['checkpoint'] if audit['formal_gate_passed'] else None),
            source_research_candidate_checkpoint=(
                audit['checkpoint']
                if (unified_audit and
                    audit['bounded_risk_research_gate_passed']) else None),
            selected_promotion_margin=audit['selected_margin'],
            research_candidate_promotion_margin=(
                audit['research_candidate_margin']
                if unified_audit else None),
            source_safe=bool(audit['formal_gate_passed']),
            eligible_for_fixed_target_dev_diagnostic=bool(
                audit['eligible_for_fixed_target_dev_diagnostic']
                if unified_audit else audit['formal_gate_passed']),
            eligible_for_deployment=False,
            eligible_for_full_test=False,
            target_dev=None, decision=decision)
        replacements = common.write_json_atomic(args.out_json, result)
        print('[dino-labeller] {}'.format(decision))
        print('[source-highres-margin] selected={}'.format(
            audit['selected_margin']))
        if unified_audit:
            print('[source-highres-margin] research_candidate={}'.format(
                audit['research_candidate_margin']))
        print('[json] nonfinite_replacements={}'.format(replacements))
        return
    if bool(getattr(
            args, 'source_smooth_geometry_rank_support_audit', False)):
        spec = args.source_smooth_geometry_audit_spec
        checkpoint_payload = torch.load(
            args.eval_only_checkpoint, map_location='cpu')
        validate_checkpoint(checkpoint_payload, in_channels, args)
        if int(checkpoint_payload.get('epoch', -1)) != int(spec['epoch']):
            raise RuntimeError(
                'Smooth geometry audit checkpoint epoch mismatch')
        load_heads_checkpoint_state(heads, checkpoint_payload)
        for parameter in heads.parameters():
            parameter.requires_grad = False
        trainable_names = []
        head_versions = common.module_parameter_versions(heads)
        audit = build_smooth_geometry_rank_support_audit(
            dino, heads, source_val, args,
            dino_device, head_device, spec, source_protocol)
        dino_unchanged = (
            dino_versions == common.module_parameter_versions(dino))
        heads_unchanged = (
            head_versions == common.module_parameter_versions(heads))
        if not dino_unchanged or not heads_unchanged:
            raise RuntimeError(
                'Read-only smooth geometry audit changed parameters')
        audit['isolation'].update(
            dino_parameters_unchanged=bool(dino_unchanged),
            detector_parameters_unchanged=bool(heads_unchanged),
            trainable_parameter_names=trainable_names)
        audit['source']['architecture'] = dict(
            in_channels=in_channels, patch_size=int(args.patch_size),
            rpn=rpn_config(in_channels, args),
            roi=roi_config(in_channels, args),
            s7=s7_architecture(args),
            s7_rpn=s7_rpn_config(
                int(getattr(args, 's7_channels', 128)), args))
        replacements = common.write_json_atomic(args.out_json, audit)
        print('[dino-labeller] {}'.format(audit['decision']))
        print('[smooth-geometry] training_allowed={}'.format(
            audit['eligible_for_training']))
        print('[json] nonfinite_replacements={}'.format(replacements))
        return
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
        current_source_small_summary = best_source_small_summary

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
            'SOURCE_ONLY_STAGE3_NO_STUDENT_IMPROVEMENT_'
            'PHASE2_FALLBACK_TARGET_NOT_READ'
            if (args.train_components == 's7_temporal_student'
                and int(best_epoch) == 0) else
            'SOURCE_ONLY_STATIC_DOMAIN_RANKER_FALLBACK_'
            'TARGET_NOT_READ'
            if (args.train_components == 's7_static_domain_ranker'
                and int(best_epoch) == 0) else
            'SOURCE_ONLY_SELECTIVE_PROMOTION_FALLBACK_'
            'TARGET_NOT_READ'
            if (args.train_components == 's7_selective_promotion'
                and int(best_epoch) == 0) else
            'SOURCE_ONLY_HIGHRES_ROI_RANKER_FALLBACK_'
            'TARGET_NOT_READ'
            if (args.train_components == 's7_highres_roi_ranker'
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
                'to_source_only_distilled_temporal_candidate_student'
                if args.train_components == 's7_temporal_student' else
                'frozen_DINOv2_native_S14_plus_protected_residual_S7_RPN_'
                'to_source_only_static_domain_generalized_candidate_ranker'
                if args.train_components == 's7_static_domain_ranker' else
                'frozen_DINOv2_native_S14_plus_protected_residual_S7_RPN_'
                'to_native_protected_uncertainty_aware_selective_promotion'
                if args.train_components == 's7_selective_promotion' else
                'frozen_DINOv2_native_S14_plus_protected_residual_S7_RPN_'
                'to_lightweight_stride7_ROI_quality_ranker'
                if args.train_components == 's7_highres_roi_ranker' else
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
                    's7_quality_suppression', 's7_temporal_association',
                    's7_temporal_student', 's7_static_domain_ranker',
                    's7_selective_promotion', 's7_highres_roi_ranker') else
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
                            args, 's7_temporal_relative_quality', False)),
                        unified_highres=bool(getattr(
                            args, 's7_highres_unified_ranking', False)),
                        pairwise_takeover_v2=bool(getattr(
                            args, 's7_highres_pairwise_takeover_v2',
                            False)))),
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
                    None if args.train_components not in (
                        's7_temporal_association', 's7_temporal_student')
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
                            sum(parameter.numel() for parameter in
                                heads.s7_candidate_student_head.parameters())
                            if args.train_components == 's7_temporal_student'
                            else len(temporal.QUALITY_CUE_NAMES)
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
                s7_temporal_student=(
                    None if args.train_components != 's7_temporal_student'
                    else dict(
                        base_checkpoint=os.path.abspath(
                            args.init_checkpoint),
                        base_epoch=int(args.s7_student_base_epoch),
                        trainable='student_candidate_quality_head_only',
                        teacher='frozen_phase2_relative_quality_head',
                        objectives=[
                            'source_continuous_max_riou',
                            'same_frame_relative_quality',
                            'teacher_bernoulli_distillation'],
                        quality_loss_weight=float(
                            args.s7_student_quality_loss_weight),
                        relative_loss_weight=float(
                            args.s7_student_relative_loss_weight),
                        distillation_weight=float(
                            args.s7_student_distillation_weight),
                        distillation_temperature=float(
                            args.s7_student_distillation_temperature),
                        small_loss_weight=float(
                            args.s7_student_small_loss_weight),
                        small_token_threshold=float(
                            args.s7_student_small_token_thr),
                        training_only_scale_weight=True,
                        inference_slice_routing=False,
                        source_only=True,
                        target_read=False,
                        teacher_reproduction_gate=getattr(
                            args, 's7_student_teacher_reproduction_gate',
                            None))),
                s7_static_domain_ranker=(
                    None if args.train_components != 's7_static_domain_ranker'
                    else dict(
                        base_checkpoint=os.path.abspath(
                            args.init_checkpoint),
                        teacher_result_json=os.path.abspath(
                            args.s7_static_teacher_result_json),
                        base_epoch=int(args.s7_static_base_epoch),
                        trainable='static_candidate_quality_head_only',
                        objectives=[
                            'source_continuous_max_riou',
                            'same_frame_relative_quality',
                            'native_retention_hard_negative',
                            'usable_s7_gain_hard_negative'],
                        quality_loss_weight=float(
                            args.s7_static_quality_loss_weight),
                        relative_loss_weight=float(
                            args.s7_static_relative_loss_weight),
                        relative_margin=float(args.s7_static_relative_margin),
                        relative_min_gap=float(args.s7_static_relative_min_gap),
                        relative_max_pairs=int(
                            args.s7_static_relative_max_pairs),
                        score_weight=float(args.s7_static_score_weight),
                        rank_margin=float(args.s7_static_rank_margin),
                        retention_weight=float(args.s7_static_retention_weight),
                        gain_weight=float(args.s7_static_gain_weight),
                        prior_weight=float(args.s7_static_prior_weight),
                        max_candidates=int(args.s7_static_max_candidates),
                        source_feature_domain_augmentation=dict(
                            probability=float(args.s7_static_aug_prob),
                            strength=float(args.s7_static_aug_strength),
                            operations=['brightness', 'blur', 'scale']),
                        inference='static_same_frame_score_logit_residual',
                        temporal_association=False,
                        inference_slice_routing=False,
                        source_only=True, target_read=False,
                        exact_source_retention=True)),
                s7_selective_promotion=(
                    None if args.train_components != 's7_selective_promotion'
                    else dict(
                        base_checkpoint=os.path.abspath(args.init_checkpoint),
                        teacher_result_json=os.path.abspath(
                            args.s7_selective_teacher_result_json),
                        base_epoch=int(args.s7_selective_base_epoch),
                        trainable=(
                            'native_vs_s7_advantage_uncertainty_head_only'),
                        frozen_teacher='phase2_candidate_quality_head',
                        version=('v2_two_frame_constant_velocity'
                                 if bool(getattr(
                                     args, 's7_selective_two_frame', False))
                                 else 'v1_static_pair'),
                        scalar_channels=(
                            temporal.S7SmallTemporalRankerHead.SCALAR_CHANNELS
                            if bool(getattr(
                                args, 's7_selective_two_frame', False))
                            else None),
                        quality_prefilter=(
                            'one_s7_from_lane_top20'
                            if bool(getattr(
                                args, 's7_selective_two_frame', False))
                            else 'all_s7_lane_candidates'),
                        objectives=[
                            'source_pair_advantage_regression',
                            'source_pair_promotion_classification',
                            'native_retention_hard_negative',
                            'usable_s7_gain_hard_negative'],
                        advantage_gap=float(
                            args.s7_selective_advantage_gap),
                        promotion_margin=float(
                            args.s7_selective_promotion_margin),
                        uncertainty_multiplier=float(
                            args.s7_selective_uncertainty_multiplier),
                        max_candidates=int(
                            args.s7_selective_max_candidates),
                        min_gain_sequences=int(
                            args.s7_selective_min_gain_sequences),
                        source_feature_domain_augmentation=dict(
                            probability=float(args.s7_selective_aug_prob),
                            strength=float(args.s7_selective_aug_strength),
                            operations=['brightness', 'blur', 'scale']),
                        inference=(
                            'lower_confidence_bound_selective_promotion'),
                        native_fallback=True,
                        temporal_association=bool(getattr(
                            args, 's7_selective_two_frame', False)),
                        causal_history_frames=(2 if bool(getattr(
                            args, 's7_selective_two_frame', False)) else 0),
                        additional_dino_forward=False,
                        dense_feature_history=False,
                        inference_slice_routing=False,
                        sequence_identity_feature=False,
                        source_only=True, target_read=False,
                        exact_source_retention=True)),
                s7_highres_roi_ranker=(
                    None if args.train_components != 's7_highres_roi_ranker'
                    else dict(
                        base_checkpoint=os.path.abspath(args.init_checkpoint),
                        teacher_result_json=os.path.abspath(
                            args.s7_highres_teacher_result_json),
                        base_epoch=int(args.s7_highres_base_epoch),
                        trainable=[
                            's7_highres_spatial_projection',
                            ('s7_highres_pairwise_takeover_head'
                             if bool(getattr(
                                 args, 's7_highres_pairwise_takeover_v2',
                                 False))
                             else 's7_highres_candidate_quality_head')],
                        highres_channels=int(args.s7_highres_channels),
                        hidden=int(args.s7_highres_hidden),
                        max_candidates=int(args.s7_highres_max_candidates),
                        score_weight=float(args.s7_highres_score_weight),
                        rank_margin=float(args.s7_highres_rank_margin),
                        promotion_margin=float(
                            args.s7_highres_promotion_margin),
                        quality_loss_weight=float(
                            args.s7_highres_quality_loss_weight),
                        relative_loss_weight=float(
                            args.s7_highres_relative_loss_weight),
                        relative_min_gap=float(
                            args.s7_highres_relative_min_gap),
                        relative_max_pairs=int(
                            args.s7_highres_relative_max_pairs),
                        retention_weight=float(
                            args.s7_highres_retention_weight),
                        gain_weight=float(args.s7_highres_gain_weight),
                        prior_weight=float(args.s7_highres_prior_weight),
                        unified_ranking=bool(getattr(
                            args, 's7_highres_unified_ranking', False)),
                        pairwise_takeover_v2=bool(getattr(
                            args, 's7_highres_pairwise_takeover_v2', False)),
                        takeover_objective=(dict(
                            target='s7_RIoU_minus_native_top1_RIoU',
                            uncertainty='heteroscedastic_source_LCB',
                            multiplier=float(
                                args.s7_takeover_uncertainty_multiplier),
                            margin=float(args.s7_takeover_margin),
                            deployment_score_thr=float(
                                args.deployment_score_thr),
                            clean_aug_consistency=True,
                            source_domain_balancing=True,
                            group_dro_eta=float(
                                args.s7_takeover_group_dro_eta),
                            raw_and_deployment_exact_retention=True)
                            if bool(getattr(
                                args, 's7_highres_pairwise_takeover_v2',
                                False)) else None),
                        unified_hard_pairs=int(getattr(
                            args, 's7_highres_unified_hard_pairs', 8)),
                        source_feature_domain_augmentation=(
                            dict(
                                probability=float(getattr(
                                    args, 's7_highres_unified_aug_prob',
                                    0.75)),
                                strength=float(getattr(
                                    args, 's7_highres_unified_aug_strength',
                                    0.15)),
                                operations=['brightness', 'blur', 'scale'])
                            if (bool(getattr(
                                args, 's7_highres_unified_ranking', False))
                                or bool(getattr(
                                    args, 's7_highres_pairwise_takeover_v2',
                                    False)))
                            else None),
                        readout='frozen_s7_feature_stride7_roi_align',
                        candidate_pool='native_top1_plus_s7_lane_topk',
                        inference=(
                            'pairwise_delta_riou_LCB_native_abstention'
                            if bool(getattr(
                                args, 's7_highres_pairwise_takeover_v2',
                                False))
                            else 'unified_native_protected_quality_margin'
                            if bool(getattr(
                                args, 's7_highres_unified_ranking', False))
                            else 'native_protected_quality_margin'),
                        additional_dino_forward=False,
                        dense_feature_history=False,
                        foreground_branch=False,
                        temporal_association=False,
                        inference_slice_routing=False,
                        sequence_identity_feature=False,
                        source_only=True, target_read=False,
                        exact_source_retention=True)),
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
            source_candidate_student_training=(
                args.train_components == 's7_temporal_student'),
            source_static_domain_ranker_training=(
                args.train_components == 's7_static_domain_ranker'),
            source_selective_promotion_training=(
                args.train_components == 's7_selective_promotion'),
            source_highres_roi_ranker_training=(
                args.train_components == 's7_highres_roi_ranker'),
            reason=(
                ('The unified high-resolution ranker uses only source GT, the '
                 'frozen S7 feature map, bounded stride-7 ROI evidence, '
                 'whole-pool hard-pair ranking, and native protection; its '
                 'feature-domain views add no extra DINO forward, target data, '
                 'foreground branch, or temporal state.'
                 if bool(getattr(
                     args, 's7_highres_unified_ranking', False)) else
                 'The high-resolution ROI ranker uses only source GT, the '
                 'frozen S7 feature map, a bounded stride-7 ROI readout, and '
                 'native-protected same-frame candidate ranking; no extra DINO '
                 'forward, target data, foreground branch, or temporal state '
                 'is used.')
                if args.train_components == 's7_highres_roi_ranker' else
                'Selective promotion uses only source GT, a frozen source-gated '
                'quality teacher, static feature perturbations, and optionally '
                'a causal two-frame scalar motion state; target and sequence '
                'identity are not learned features.'
                if args.train_components == 's7_selective_promotion' else
                'The static ranker uses only source GT, frozen detector heads, '
                'and deterministic feature-domain augmentation; target '
                'pseudo-label training remains disabled.'
                if args.train_components == 's7_static_domain_ranker' else
                'Stage-3 uses only source GT and a frozen source-gated teacher; '
                'target pseudo-label training remains disabled because no '
                'separate unlabelled target-train split was supplied.')),
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
            initial_teacher_reproduction_gate=(
                getattr(args, 's7_student_teacher_reproduction_gate', None)
                if args.train_components == 's7_temporal_student' else None),
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
