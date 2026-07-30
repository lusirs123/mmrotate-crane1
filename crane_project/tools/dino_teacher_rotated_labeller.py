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
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


PROJ_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import dino_teacher_common as common  # noqa: E402


LABELLER_NAME = 'Frozen DINOv2 Oriented RPN/ROI Source Labeller V1'
PROTOCOL_VERSION = 10
PAIRWISE_V2_MAX_EPOCHS = 4
PAPER_URL = (
    'https://openaccess.thecvf.com/content/CVPR2025/html/'
    'Lavoie_Large_Self-Supervised_Models_Bridge_the_Gap_in_Domain_Adaptive_'
    'Object_CVPR_2025_paper.html')
PAPER_CODE_URL = 'https://github.com/TRAILab/DINO_Teacher'

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
                 'roi_cls_pairwise_v2', 's7_rpn'], default='all',
        help=('Train all RPN/ROI heads, or only the final ROI classifier '
              'fc_cls while keeping RPN, shared ROI FCs, and bbox regression '
              'fixed. roi_cls_pairwise additionally uses source-only '
              'NMS-aware pairwise ranking and the initialized classifier as '
              'a frozen retention teacher. roi_cls_pairwise_v2 mines only '
              'false ROIs that actually outrank a usable ROI. s7_rpn trains '
              'only the residual stride-7 readout and proposal head.'))
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
    if s7_enabled != (args.train_components == 's7_rpn'):
        raise ValueError(
            '--s7-residual and --train-components s7_rpn must be enabled '
            'together during S7 training')
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
                'roi_cls_pairwise_v2', 's7_rpn') else [5, 7])
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
    if (args.train_components == 's7_rpn'
            and not (args.init_checkpoint or args.resume_checkpoint
                     or args.eval_only_checkpoint)):
        raise ValueError(
            'S7 RPN mode requires an init/resume/eval-only checkpoint; '
            'training must initialize from the retained native S14 heads')
    if args.train_components == 's7_rpn' and args.epochs > 4:
        raise ValueError(
            'The first causal S7 readout stage is limited to 4 epochs; '
            'extend only after source validation shows it is still improving')
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
    if train_components in ('roi_cls_pairwise', 'roi_cls_pairwise_v2'):
        required = ('loss_cls', 'loss_roi_pairwise', 'loss_roi_retention')
        missing = [name for name in required if name not in losses]
        if missing:
            raise RuntimeError('Pairwise ROI losses missing: {}'.format(
                ', '.join(missing)))
        return loss_total({name: losses[name] for name in required})
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
        best_small_summary: Dict, history: Sequence[Dict]) -> Tuple[str, int]:
    """Persist source-only selection evidence before target is ever read."""
    output_path = source_progress_path(args.out_json)
    payload = dict(
        labeller=LABELLER_NAME,
        protocol_version=PROTOCOL_VERSION,
        status='SOURCE_ONLY_TRAINING_IN_PROGRESS',
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
        self._s7_inference_enabled = self.s7_enabled
        if self.s7_enabled:
            s7_channels = int(getattr(args, 's7_channels', 128))
            self.s7_readout = ResidualS7Readout(
                int(in_channels), s7_channels)
            self.s7_rpn_head = build_head(ConfigDict(
                s7_rpn_config(s7_channels, args)))
            self.s7_rpn_head.init_weights()
            self.s7_proposal_cfg = ConfigDict(s7_rpn_proposal_config(args))
        else:
            self.s7_readout = None
            self.s7_rpn_head = None
            self.s7_proposal_cfg = None
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

    def simple_test_proposals(self, feature: torch.Tensor, img_meta: Dict):
        """Keep all native proposals and append a bounded S7 supplement."""
        native_features = self.feature_levels(feature)
        native = self.rpn_head.simple_test_rpn(
            native_features, [img_meta])[0]
        if not self.s7_inference_enabled():
            return native_features, [native]
        s7 = self.s7_feature(feature)
        supplement = self.s7_rpn_head.simple_test_rpn(
            [s7], [img_meta])[0]
        return native_features, [torch.cat([native, supplement], dim=0)]

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
    if args.train_components == 's7_rpn':
        heads.rpn_head.eval()
        heads.roi_head.eval()
    if head_device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(head_device)
    ordered = list(records)
    random.Random(args.seed + epoch).shuffle(ordered)
    losses = []
    component_sums = {}
    metric_sums = {}
    cache_hits = 0
    for index, record in enumerate(ordered):
        current_lr = scheduled_lr(args, epoch, global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        feature, img_meta, gt_boxes, gt_labels, _original, cached = (
            prepare_record(
                dino, record, args, dino_device, head_device))
        cache_hits += int(cached)
        optimizer.zero_grad()
        if args.train_components == 's7_rpn':
            output = heads.forward_s7_rpn_train(
                feature, img_meta, gt_boxes)
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
            print(message)
        del feature, gt_boxes, gt_labels, total
    return dict(
        epoch=int(epoch), count=len(ordered),
        global_step_end=int(global_step),
        lr_end=float(optimizer.param_groups[0]['lr']),
        mean_loss=float(np.mean(losses)),
        optimized_components=(
            ['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox']
            if args.train_components == 'all' else (
                ['loss_s7_rpn_cls', 'loss_s7_rpn_bbox']
                if args.train_components == 's7_rpn' else (
                ['loss_cls', 'loss_roi_pairwise', 'loss_roi_retention']
                if args.train_components in (
                    'roi_cls_pairwise', 'roi_cls_pairwise_v2')
                else ['loss_cls']))),
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


def filter_valid_rotated_detections(detections: np.ndarray,
                                    img_meta: Dict,
                                    tolerance: float = 1e-3):
    """Filter OBBs with corners outside the original image.

    The rule is label-free and is applied identically to source validation and
    target diagnosis.  Remaining detections keep their original score order.
    """
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


def evaluate_records(dino, heads, records: Sequence[Dict], args,
                     dino_device, head_device, role: str):
    heads.eval()
    rows = []
    with torch.no_grad():
        for index, record in enumerate(records):
            feature, img_meta, _gt_boxes, _gt_labels, original, cached = (
                prepare_record(
                    dino, record, args, dino_device, head_device))
            raw_detections = heads.simple_test(feature, img_meta)
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
    return dict(
        enabled=enabled,
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
            args, 's7_anchor_sizes', [])] if enabled else []))


def load_heads_checkpoint_state(heads, payload: Dict,
                                allow_s7_base_initialization: bool = False):
    """Load a native checkpoint into S7 only for explicit initialization."""
    if allow_s7_base_initialization:
        incompatible = heads.load_state_dict(
            payload['heads_state_dict'], strict=False)
        allowed_prefixes = ('s7_readout.', 's7_rpn_head.')
        disallowed_missing = [
            name for name in incompatible.missing_keys
            if not name.startswith(allowed_prefixes)]
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


def checkpoint_payload(heads, optimizer, scheduler, epoch: int,
                       best_epoch: int, best_summary: Dict,
                       in_channels: int, args,
                       global_step: int = 0,
                       best_small_summary: Dict = None,
                       source_sampling: Dict = None,
                       source_baseline_summary: Dict = None,
                       source_baseline_small_summary: Dict = None,
                       source_baseline_correct_keys: Sequence[str] = None
                       ) -> Dict:
    return dict(
        labeller=LABELLER_NAME, protocol_version=PROTOCOL_VERSION,
        source_only=True, frozen_dinov2=True,
        epoch=int(epoch), best_epoch=int(best_epoch),
        global_step=int(global_step),
        best_source_val_summary=best_summary,
        best_source_small_val_summary=best_small_summary,
        source_baseline_val_summary=source_baseline_summary,
        source_baseline_small_val_summary=source_baseline_small_summary,
        source_baseline_correct_keys=(
            None if source_baseline_correct_keys is None else
            [str(key) for key in source_baseline_correct_keys]),
        source_sampling=source_sampling,
        s7_architecture=s7_architecture(args),
        s7_inference_enabled=bool(heads.s7_inference_enabled()),
        roi_cls_teacher_state=heads.roi_cls_teacher_state(),
        training_protocol=dict(
            train_components=str(args.train_components),
            optimization_loss_components=(
                ['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox']
                if args.train_components == 'all' else (
                    ['loss_s7_rpn_cls', 'loss_s7_rpn_bbox']
                    if args.train_components == 's7_rpn' else (
                    ['loss_cls', 'loss_roi_pairwise', 'loss_roi_retention']
                    if args.train_components in (
                        'roi_cls_pairwise', 'roi_cls_pairwise_v2')
                    else ['loss_cls']))),
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
                              args.pairwise_negatives_per_positive)))),
            s7_source_gate=(
                None if args.train_components != 's7_rpn' else dict(
                    min_full_top1=int(getattr(
                        args, 's7_source_min_full_top1', 677)),
                    min_small_top1=int(getattr(
                        args, 's7_source_min_small_top1', 303)),
                    max_mcml=int(getattr(
                        args, 's7_source_max_mcml', 3)))),
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
                        allow_s7_base_initialization: bool = False):
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
            if any(stored_s7.get(key) != requested_s7.get(key)
                   for key in keys):
                raise RuntimeError('S7 checkpoint architecture mismatch')
    elif bool(stored_s7.get('enabled', False)):
        raise RuntimeError(
            'S7 checkpoint cannot load into the native single-scale head')


def train_source_only(dino, heads, train_records, val_records, args,
                      dino_device, head_device, in_channels: int):
    pairwise_modes = ('roi_cls_pairwise', 'roi_cls_pairwise_v2')
    roi_cls_mode = args.train_components in ('roi_cls',) + pairwise_modes
    s7_mode = args.train_components == 's7_rpn'
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
    history = []
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location='cpu')
        validate_checkpoint(
            payload, in_channels, args,
            allow_training_mode_mismatch=True,
            allow_s7_base_initialization=s7_mode)
        load_heads_checkpoint_state(
            heads, payload, allow_s7_base_initialization=s7_mode)
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
                source_baseline_correct_keys), best_path)
            if s7_mode:
                heads.set_s7_inference_enabled(True)
            print('[source-baseline] full_top1={}/{} small_top1={}/{} '
                  'fallback_epoch=0'.format(
                      best_summary['top1_hits'],
                      best_summary['frame_count'],
                      best_small_summary['top1_hits'],
                      best_small_summary['frame_count']))
    elif best_summary is not None:
        best_key = source_selection_key(best_summary)

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
        if evaluate_epoch:
            val_rows = evaluate_records(
                dino, heads, val_records, args, dino_device, head_device,
                role='source_validation')
            val_summary = summarize_rows(val_rows)
            if protected_source_mode:
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
        pairwise_v2_gate = None
        if val_summary is not None and protected_source_mode:
            if args.train_components in pairwise_modes or s7_mode:
                retention_summary = source_top1_retention_summary(
                    source_baseline_correct_keys, val_rows)
                if s7_mode:
                    pairwise_v2_gate = s7_source_selection_gate(
                        source_baseline_summary,
                        source_baseline_small_summary,
                        val_summary, small_val_summary,
                        retention_summary, args)
                    source_gate_passed = bool(pairwise_v2_gate['passed'])
                elif args.train_components == 'roi_cls_pairwise_v2':
                    pairwise_v2_gate = pairwise_v2_source_selection_gate(
                        source_baseline_summary,
                        source_baseline_small_summary,
                        val_summary, small_val_summary,
                        retention_summary)
                    source_gate_passed = bool(pairwise_v2_gate['passed'])
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
                source_baseline_correct_keys), best_path)
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
                source_baseline_correct_keys), epoch_path)
        exact_retention_passed = (
            bool(pairwise_v2_gate['checks'][
                'exact_old_correct_retention'])
            if pairwise_v2_gate is not None else bool(source_gate_passed))
        history.append(dict(
            epoch=int(epoch), train=train_row,
            source_val=val_summary,
            source_small_val=small_val_summary,
            source_selection_gate_passed=bool(source_gate_passed),
            source_retention_passed=exact_retention_passed,
            source_exact_retention=retention_summary,
            pairwise_v2_source_gate=pairwise_v2_gate,
            selected_as_best=bool(improved),
            selection_eligible=bool(selection_eligible),
            checkpoint_saved=bool(evaluate_epoch),
            lr=float(optimizer.param_groups[0]['lr'])))
        atomic_torch_save(checkpoint_payload(
            heads, optimizer, scheduler, epoch, best_epoch, best_summary,
            in_channels, args, global_step, best_small_summary,
            source_sampling, source_baseline_summary,
            source_baseline_small_summary,
            source_baseline_correct_keys), latest_path)
        progress_path = None
        progress_replacements = 0
        if args.train_components in ('roi_cls_pairwise_v2', 's7_rpn'):
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
            gate_label = ('source_gate' if pairwise_v2_gate is not None
                          else 'retention')
            print('[source-epoch] epoch={} top1={}/{} small_top1={} r100={} '
                  '{}={} selection_eligible={} best_epoch={}'.format(
                      epoch, val_summary['top1_hits'],
                      val_summary['frame_count'], small_text,
                      val_summary['recall_at_100'], gate_label,
                      source_gate_passed,
                      selection_eligible, best_epoch))
            if pairwise_v2_gate is not None:
                failed_checks = sorted(
                    name for name, passed
                    in pairwise_v2_gate['checks'].items() if not passed)
                print(
                    '[source-gate] epoch={} passed={} retained={}/{} '
                    'lost={} gained={} full_mcml={}->{} small_mcml={}->{} '
                    'failed={}'.format(
                        epoch, pairwise_v2_gate['passed'],
                        retention_summary['retained_correct_count'],
                        retention_summary['baseline_correct_count'],
                        retention_summary['lost_correct_count'],
                        retention_summary['gained_correct_count'],
                        source_baseline_summary['top1_mcml'],
                        val_summary['top1_mcml'],
                        source_baseline_small_summary['top1_mcml'],
                        small_val_summary['top1_mcml'],
                        ','.join(failed_checks) if failed_checks else 'none'))
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
            history)


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

    if args.eval_only_checkpoint:
        best_path = args.eval_only_checkpoint
        payload = torch.load(best_path, map_location='cpu')
        validate_checkpoint(payload, in_channels, args)
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
    else:
        (best_path, best_epoch,
         best_source_summary, best_source_small_summary,
         source_sampling, source_baseline_summary,
         source_baseline_small_summary,
         frozen_head_parameters_unchanged, history) = train_source_only(
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
        decision = 'SOURCE_ONLY_TRAINING_COMPLETE_TARGET_NOT_READ'
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
    payload = dict(
        labeller=LABELLER_NAME, protocol_version=PROTOCOL_VERSION,
        paper=PAPER_URL, paper_code=PAPER_CODE_URL,
        dinov2_checkpoint=os.path.abspath(args.dinov2_checkpoint),
        source_selected_checkpoint=os.path.abspath(best_path),
        protocol=dict(
            architecture=(
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
                if args.train_components == 's7_rpn' else
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
                    ['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox']
                    if args.train_components == 'all' else (
                        ['loss_s7_rpn_cls', 'loss_s7_rpn_bbox']
                        if args.train_components == 's7_rpn' else (
                        ['loss_cls', 'loss_roi_pairwise',
                         'loss_roi_retention']
                        if args.train_components in (
                            'roi_cls_pairwise', 'roi_cls_pairwise_v2')
                        else ['loss_cls']))),
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
            initialization_checkpoint=(
                None if not args.init_checkpoint
                else os.path.abspath(args.init_checkpoint)),
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
            small_sampling=source_sampling,
            current_inference_validation_summary=current_source_summary,
            current_inference_rule='valid_rotated_obb_corners',
            history=history,
            source_val_results_pickle=source_val_results_path),
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
