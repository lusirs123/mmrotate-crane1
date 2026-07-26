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
import random
import re
import sys
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import dino_teacher_frozen_region_audit as audit  # noqa: E402
from crane_project.tools import dino_teacher_source_roi_head_probe as roi_probe  # noqa: E402
from crane_project.tools import frozen_p3_feature_alignment_audit as alignment  # noqa: E402
from crane_project.tools import frozen_p3_objectness_transfer_probe as transfer  # noqa: E402
from crane_project.tools import p3_p4_neighborhood_rescue_audit as neighborhood  # noqa: E402


LABELLER_NAME = 'Frozen DINOv2 Oriented RPN/ROI Source Labeller V1'
PROTOCOL_VERSION = 6
PAPER_URL = (
    'https://openaccess.thecvf.com/content/CVPR2025/html/'
    'Lavoie_Large_Self-Supervised_Models_Bridge_the_Gap_in_Domain_Adaptive_'
    'Object_CVPR_2025_paper.html')
PAPER_CODE_URL = 'https://github.com/TRAILab/DINO_Teacher'


def parse_args():
    parser = argparse.ArgumentParser(description=LABELLER_NAME)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-split', default=neighborhood.SOURCE_SPLIT)
    parser.add_argument('--source-seq', default=neighborhood.SOURCE_SEQ)
    parser.add_argument('--source-val-modulus', type=int, default=5)
    parser.add_argument(
        '--source-train-datasets', nargs='+',
        help='Formal source train specs: annotation_split:image_split')
    parser.add_argument(
        '--source-val-datasets', nargs='+',
        help='Formal source validation specs: annotation_split:image_split')
    parser.add_argument('--target-split', default=neighborhood.TARGET_SPLIT)
    parser.add_argument('--target-seq', default=neighborhood.TARGET_SEQ)
    parser.add_argument('--target-start', type=int,
                        default=neighborhood.TARGET_START)
    parser.add_argument('--target-end', type=int,
                        default=neighborhood.TARGET_END)
    parser.add_argument('--dinov2-repo', required=True)
    parser.add_argument('--dinov2-checkpoint', required=True)
    parser.add_argument('--dinov2-model', default=audit.CANONICAL_MODEL)
    parser.add_argument('--dino-gpus', type=int, nargs='+', required=True)
    parser.add_argument('--head-gpu', type=int, default=0)
    parser.add_argument('--legacy-sdpa-query-chunk', type=int, default=512)
    parser.add_argument('--dino-height', type=int,
                        default=audit.CANONICAL_DINO_HEIGHT)
    parser.add_argument('--dino-max-long-side', type=int,
                        default=audit.CANONICAL_DINO_MAX_LONG_SIDE)
    parser.add_argument('--patch-size', type=int, default=14)
    parser.add_argument('--rpn-feat-channels', type=int, default=256)
    parser.add_argument('--roi-fc-channels', type=int, default=1024)
    parser.add_argument('--roi-samples', type=int, default=256)
    parser.add_argument('--proposal-count', type=int, default=2000)
    parser.add_argument('--max-detections', type=int, default=2000)
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
    parser.add_argument('--lr-steps', type=int, nargs='+', default=[5, 7])
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    parser.add_argument('--checkpoint-interval', type=int, default=1)
    parser.add_argument('--selection-epochs', type=int, nargs='+')
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--target-min-wins', type=int, default=26)
    parser.add_argument('--max-mcml', type=int, default=5)
    parser.add_argument('--source-min-top1-rate', type=float, default=0.8)
    parser.add_argument('--feature-cache-dir', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--resume-checkpoint')
    parser.add_argument('--eval-only-checkpoint')
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
            'Formal source train and validation specs must be supplied together')
    if args.source_train_datasets:
        parse_dataset_specs(args.source_train_datasets)
        parse_dataset_specs(args.source_val_datasets)
    if not args.dino_gpus:
        raise ValueError('At least one DINO GPU is required')
    if args.head_gpu in args.dino_gpus:
        raise ValueError(
            'Head GPU must be separate from sharded DINO GPUs on 8GB cards')
    positive = (
        args.patch_size, args.rpn_feat_channels, args.roi_fc_channels,
        args.roi_samples, args.proposal_count, args.max_detections,
        args.epochs, args.lr, args.max_grad_norm, args.lr_gamma,
        args.checkpoint_interval)
    if any(float(value) <= 0.0 for value in positive):
        raise ValueError('Head, optimizer, and count settings must be positive')
    if args.warmup_iters < 0:
        raise ValueError('--warmup-iters must be non-negative')
    if not 0.0 < args.warmup_ratio <= 1.0:
        raise ValueError('--warmup-ratio must be in (0, 1]')
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
    if args.valid_content_tolerance < 0.0:
        raise ValueError('--valid-content-tolerance must be non-negative')
    if args.deployment_score_thr < 0.0:
        raise ValueError('--deployment-score-thr must be non-negative')
    if not 0.0 <= args.border_margin_ratio <= 0.5:
        raise ValueError('--border-margin-ratio must be in [0, 0.5]')
    if not 0.0 <= args.source_min_top1_rate <= 1.0:
        raise ValueError('--source-min-top1-rate must be in [0, 1]')
    if args.resume_checkpoint and args.eval_only_checkpoint:
        raise ValueError('Resume and eval-only checkpoints are mutually exclusive')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    tensor, dino_meta = audit.resize_and_normalize_bgr(
        image, args.dino_height, args.patch_size,
        args.dino_max_long_side)
    tensor = tensor.to(dino_device)
    feature = audit.extract_patch_grid(dino, tensor, args.patch_size)
    if not bool(torch.isfinite(feature).all().item()):
        raise RuntimeError('Non-finite DINO feature')
    feature_cpu = feature.detach().cpu().half()
    atomic_torch_save(dict(
        signature=signature, feature=feature_cpu,
        dino_meta=dino_meta, frozen_dinov2=True), path)
    del tensor, feature
    return feature_cpu, dino_meta, False


def parse_original_gt(annotation: str) -> np.ndarray:
    diag = transfer.entry_probe.get_diag()
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
    """Read labels from one split and images from its configured image split."""
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
    diag = transfer.entry_probe.get_diag()
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


def rpn_config(in_channels: int, args) -> Dict:
    scales = [size / float(args.patch_size)
              for size in (32, 64, 128, 256, 512)]
    return dict(
        type='OrientedRPNHead', in_channels=int(in_channels),
        feat_channels=int(args.rpn_feat_channels), version='le90',
        anchor_generator=dict(
            type='AnchorGenerator', scales=scales,
            ratios=[0.5, 1.0, 2.0], strides=[int(args.patch_size)]),
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


def roi_config(in_channels: int, args) -> Dict:
    return dict(
        type='OrientedStandardRoIHead', version='le90',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated', out_size=7,
                sample_num=2, clockwise=True),
            out_channels=int(in_channels),
            featmap_strides=[int(args.patch_size)]),
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
            nms_pre=int(args.proposal_count), min_bbox_size=0,
            score_thr=0.0, nms=dict(iou_thr=0.1),
            max_per_img=int(args.max_detections)))


class FrozenDinoRotatedHeads(nn.Module):
    def __init__(self, in_channels: int, args):
        super().__init__()
        from mmcv import ConfigDict
        from mmrotate.models.builder import build_head

        self.in_channels = int(in_channels)
        self.rpn_head = build_head(ConfigDict(rpn_config(in_channels, args)))
        self.roi_head = build_head(ConfigDict(roi_config(in_channels, args)))
        self.rpn_head.init_weights()
        self.roi_head.init_weights()
        self.proposal_cfg = ConfigDict(rpn_proposal_config(args))

    def forward_train(self, feature: torch.Tensor, img_meta: Dict,
                      gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        features = [feature]
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

    def simple_test(self, feature: torch.Tensor, img_meta: Dict):
        features = [feature]
        proposals = self.rpn_head.simple_test_rpn(features, [img_meta])
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
    ordered = list(records)
    random.Random(args.seed + epoch).shuffle(ordered)
    losses = []
    component_sums = {}
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
        output = heads.forward_train(feature, img_meta, gt_boxes, gt_labels)
        for name, value in loss_component_means(output).items():
            component_sums[name] = component_sums.get(name, 0.0) + value
        total = loss_total(output)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            heads.parameters(), args.max_grad_norm)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError('Non-finite detector-head gradient')
        optimizer.step()
        global_step += 1
        losses.append(float(total.item()))
        if (index + 1) % 25 == 0 or index + 1 == len(ordered):
            print('[source-train] epoch={} {}/{} loss={:.5f} cache={}/{}'.format(
                epoch, index + 1, len(ordered),
                float(np.mean(losses[-25:])), cache_hits, index + 1))
        del feature, gt_boxes, gt_labels, total
    return dict(
        epoch=int(epoch), count=len(ordered),
        global_step_end=int(global_step),
        lr_end=float(optimizer.param_groups[0]['lr']),
        mean_loss=float(np.mean(losses)),
        mean_loss_components={
            name: float(value / max(1, len(ordered)))
            for name, value in sorted(component_sums.items())},
        cache_hits=int(cache_hits))


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


def checkpoint_payload(heads, optimizer, scheduler, epoch: int, best_epoch: int,
                       best_summary: Dict, in_channels: int, args,
                       global_step: int = 0) -> Dict:
    return dict(
        labeller=LABELLER_NAME, protocol_version=PROTOCOL_VERSION,
        source_only=True, frozen_dinov2=True,
        epoch=int(epoch), best_epoch=int(best_epoch),
        global_step=int(global_step),
        best_source_val_summary=best_summary,
        training_protocol=dict(
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
                              for value in args.selection_epochs]),
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


def validate_checkpoint(payload: Dict, in_channels: int, args):
    required = (
        'source_only', 'frozen_dinov2', 'in_channels', 'patch_size',
        'rpn_feat_channels', 'roi_fc_channels', 'heads_state_dict')
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError('Labeller checkpoint lacks {}'.format(
            ', '.join(missing)))
    if payload['source_only'] is not True or payload['frozen_dinov2'] is not True:
        raise RuntimeError('Checkpoint is not source-only/frozen-DINO')
    expected = dict(
        in_channels=int(in_channels), patch_size=int(args.patch_size),
        rpn_feat_channels=int(args.rpn_feat_channels),
        roi_fc_channels=int(args.roi_fc_channels))
    mismatched = [key for key, value in expected.items()
                  if int(payload[key]) != int(value)]
    if mismatched:
        raise RuntimeError('Labeller architecture mismatch: {}'.format(
            ', '.join(mismatched)))


def train_source_only(dino, heads, train_records, val_records, args,
                      dino_device, head_device, in_channels: int):
    optimizer = torch.optim.SGD(
        heads.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)
    scheduler = None
    start_epoch = 1
    global_step = 0
    best_epoch = 0
    best_summary = None
    best_key = None
    history = []
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location='cpu')
        validate_checkpoint(payload, in_channels, args)
        heads.load_state_dict(payload['heads_state_dict'], strict=True)
        if payload.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        start_epoch = int(payload.get('epoch', 0)) + 1
        global_step = int(payload.get(
            'global_step', (start_epoch - 1) * len(train_records)))
        best_epoch = int(payload.get('best_epoch', 0))
        best_summary = payload.get('best_source_val_summary')
        best_key = (None if best_summary is None
                    else source_selection_key(best_summary))

    best_path = os.path.join(args.work_dir, 'labeller_best_source_only.pth')
    latest_path = os.path.join(args.work_dir, 'labeller_latest.pth')
    selection_epochs = set(int(value) for value in args.selection_epochs)
    for epoch in range(start_epoch, args.epochs + 1):
        train_row = train_epoch(
            dino, heads, optimizer, train_records, epoch, global_step, args,
            dino_device, head_device)
        global_step = int(train_row['global_step_end'])
        evaluate_epoch = (
            epoch % int(args.checkpoint_interval) == 0
            or epoch == int(args.epochs))
        val_summary = None
        if evaluate_epoch:
            val_rows = evaluate_records(
                dino, heads, val_records, args, dino_device, head_device,
                role='source_validation')
            val_summary = summarize_rows(val_rows)
        selection_eligible = epoch in selection_epochs
        if selection_eligible and val_summary is None:
            raise RuntimeError(
                'A selection epoch must also be a validation epoch')
        key = (None if val_summary is None
               else source_selection_key(val_summary))
        improved = bool(
            selection_eligible
            and (best_key is None or key > best_key))
        if improved:
            best_key = key
            best_epoch = int(epoch)
            best_summary = val_summary
            atomic_torch_save(checkpoint_payload(
                heads, None, None, epoch, best_epoch, best_summary,
                in_channels, args, global_step), best_path)
        if evaluate_epoch:
            epoch_path = os.path.join(
                args.work_dir,
                'labeller_epoch_{:02d}_source_only.pth'.format(epoch))
            atomic_torch_save(checkpoint_payload(
                heads, optimizer, scheduler, epoch, best_epoch,
                best_summary, in_channels, args, global_step), epoch_path)
        history.append(dict(
            epoch=int(epoch), train=train_row,
            source_val=val_summary, selected_as_best=bool(improved),
            selection_eligible=bool(selection_eligible),
            checkpoint_saved=bool(evaluate_epoch),
            lr=float(optimizer.param_groups[0]['lr'])))
        atomic_torch_save(checkpoint_payload(
            heads, optimizer, scheduler, epoch, best_epoch, best_summary,
            in_channels, args, global_step), latest_path)
        if val_summary is None:
            print('[source-epoch] epoch={} validation=skipped best_epoch={}'
                  .format(epoch, best_epoch))
        else:
            print('[source-epoch] epoch={} top1={}/{} r100={} '
                  'selection_eligible={} best_epoch={}'.format(
                      epoch, val_summary['top1_hits'],
                      val_summary['frame_count'],
                      val_summary['recall_at_100'], selection_eligible,
                      best_epoch))
    if best_epoch <= 0 or not os.path.isfile(best_path):
        raise RuntimeError('No source-selected labeller checkpoint')
    return best_path, best_epoch, best_summary, history


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
            row for row in transfer.discover_labeled_records(
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

    dino, loaded_patch_size = audit.load_frozen_dinov2(
        args.dinov2_repo, args.dinov2_checkpoint,
        args.dinov2_model, dino_devices,
        args.legacy_sdpa_query_chunk)
    if int(loaded_patch_size) != int(args.patch_size):
        raise RuntimeError('Unexpected DINO patch size')
    dino_versions = alignment.module_parameter_versions(dino)
    in_channels = int(getattr(dino, 'embed_dim', 0))
    if in_channels <= 0:
        sample_feature, _meta, _cached = extract_or_load_feature(
            dino, source_train[0], args, dino_device)
        in_channels = int(sample_feature.shape[1])
    heads = FrozenDinoRotatedHeads(in_channels, args).to(head_device)

    if args.eval_only_checkpoint:
        best_path = args.eval_only_checkpoint
        payload = torch.load(best_path, map_location='cpu')
        validate_checkpoint(payload, in_channels, args)
        heads.load_state_dict(payload['heads_state_dict'], strict=True)
        best_epoch = int(payload.get('best_epoch', payload.get('epoch', 0)))
        best_source_summary = payload.get('best_source_val_summary')
        history = []
        # Recompute source validation with the current inference rule.  The
        # checkpoint's stored summary may predate the valid-content filter.
        source_val_rows = evaluate_records(
            dino, heads, source_val, args, dino_device, head_device,
            role='source_validation')
        current_source_summary = summarize_rows(source_val_rows)
    else:
        best_path, best_epoch, best_source_summary, history = train_source_only(
            dino, heads, source_train, source_val, args,
            dino_device, head_device, in_channels)
        payload = torch.load(best_path, map_location='cpu')
        validate_checkpoint(payload, in_channels, args)
        heads.load_state_dict(payload['heads_state_dict'], strict=True)
        current_source_summary = best_source_summary

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
        dino_versions == alignment.module_parameter_versions(dino))
    if not dino_unchanged:
        raise RuntimeError('Frozen DINO parameter invariant failed')
    payload = dict(
        labeller=LABELLER_NAME, protocol_version=PROTOCOL_VERSION,
        paper=PAPER_URL, paper_code=PAPER_CODE_URL,
        dinov2_checkpoint=os.path.abspath(args.dinov2_checkpoint),
        source_selected_checkpoint=os.path.abspath(best_path),
        protocol=dict(
            architecture=(
                'frozen_DINOv2_single_scale_to_OrientedRPN_'
                'to_RotatedROIAlign7x7_to_Shared2FC_cls_and_OBB_reg'),
            source_data=source_protocol,
            checkpoint_selection=(
                'source_validation_only_over_fixed_candidate_epochs'),
            selection_epochs=[int(value)
                              for value in args.selection_epochs],
            training_schedule=dict(
                epochs=int(args.epochs), optimizer='SGD',
                lr=float(args.lr), momentum=float(args.momentum),
                weight_decay=float(args.weight_decay),
                max_grad_norm=float(args.max_grad_norm),
                warmup='linear', warmup_iters=int(args.warmup_iters),
                warmup_ratio=float(args.warmup_ratio),
                lr_policy='step',
                lr_steps=[int(value) for value in args.lr_steps],
                lr_gamma=float(args.lr_gamma),
                checkpoint_interval=int(args.checkpoint_interval)),
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
            trainable_modules=['OrientedRPNHead',
                               'OrientedStandardRoIHead'],
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_labels_used_for_evaluation_only=bool(
                not args.skip_target_eval)),
        architecture=dict(
            in_channels=in_channels, patch_size=int(args.patch_size),
            rpn=rpn_config(in_channels, args), roi=roi_config(in_channels, args)),
        source=dict(
            train_count=len(source_train), val_count=len(source_val),
            best_epoch=int(best_epoch),
            best_validation_summary=best_source_summary,
            current_inference_validation_summary=current_source_summary,
            current_inference_rule='valid_rotated_obb_corners',
            history=history),
        target_dev=(None if target_summary is None else dict(
            summary=target_summary, rows=target_rows)),
        decision=decision)
    replacements = roi_probe.write_json_atomic(args.out_json, payload)
    if target_summary is None:
        print('[dino-labeller] {}'.format(decision))
    else:
        print('[dino-labeller] {} top1={}/{} mcml={} r100={}'.format(
            decision, target_summary['top1_hits'],
            target_summary['frame_count'], target_summary['top1_mcml'],
            target_summary['recall_at_100']))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
