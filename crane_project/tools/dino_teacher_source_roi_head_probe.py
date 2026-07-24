#!/usr/bin/env python3
"""Train a source-only frozen-DINO ROI objectness head and audit target rank.

This is the bounded experiment authorized by the frozen-region audit. It keeps
the BrightAug detector and DINOv2 backbone frozen, trains only a Faster R-CNN
style two-FC binary ROI classifier on labelled source proposals, freezes the
source-selected head, and then diagnoses detector-top-K level-0 target
candidates. Target labels never affect training or checkpoint selection.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import dino_teacher_frozen_region_audit as audit  # noqa: E402
from crane_project.tools import frozen_p3_feature_alignment_audit as alignment  # noqa: E402
from crane_project.tools import frozen_p3_objectness_transfer_probe as transfer  # noqa: E402
from crane_project.tools import p3_p4_neighborhood_rescue_audit as neighborhood  # noqa: E402


PROBE_NAME = 'DINO Teacher Source-Only Two-FC ROI Objectness Probe V1'
PROTOCOL_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(description=PROBE_NAME)
    parser.add_argument('--config', required=True)
    parser.add_argument('--detector-checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-seq', default=neighborhood.SOURCE_SEQ)
    parser.add_argument('--dinov2-repo', required=True)
    parser.add_argument('--dinov2-checkpoint', required=True)
    parser.add_argument('--dinov2-model', default=audit.CANONICAL_MODEL)
    parser.add_argument('--dino-gpus', type=int, nargs='+', required=True)
    parser.add_argument('--gpu', type=int, default=0,
                        help='Detector and trainable ROI-head GPU')
    parser.add_argument('--legacy-sdpa-query-chunk', type=int, default=512)
    parser.add_argument('--dino-height', type=int, default=600)
    parser.add_argument('--dino-max-long-side', type=int, default=1333)
    parser.add_argument('--patch-size', type=int, default=14)
    parser.add_argument('--pool-resolution', type=int, default=7)
    parser.add_argument('--min-roi-in-bounds', type=float, default=0.9)
    parser.add_argument('--source-folds', type=int, default=5)
    parser.add_argument('--source-negatives-per-image', type=int, default=3)
    parser.add_argument('--source-min-accuracy', type=float, default=0.8)
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--false-iou-thr', type=float, default=0.1)
    parser.add_argument('--hidden-dim', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--target-candidate-limit', type=int, default=10000)
    parser.add_argument('--target-min-wins', type=int, default=26)
    parser.add_argument('--target-start', type=int,
                        default=neighborhood.TARGET_START)
    parser.add_argument('--target-end', type=int,
                        default=neighborhood.TARGET_END)
    parser.add_argument('--roi-chunk-size', type=int, default=16)
    parser.add_argument('--max-source-samples', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def validate_args(args):
    if args.seed != 0:
        raise ValueError('The protocol requires --seed 0')
    if args.source_folds < 2:
        raise ValueError('--source-folds must be at least 2')
    if args.source_negatives_per_image < 1:
        raise ValueError('--source-negatives-per-image must be positive')
    if args.hidden_dim < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError('Head dimensions and training settings must be positive')
    if args.target_candidate_limit < 1 or args.roi_chunk_size < 1:
        raise ValueError('Target candidate and ROI chunk limits must be positive')
    if not 0.0 < args.min_roi_in_bounds <= 1.0:
        raise ValueError('--min-roi-in-bounds must be in (0, 1]')
    if not 0.0 < args.source_min_accuracy <= 1.0:
        raise ValueError('--source-min-accuracy must be in (0, 1]')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TwoFCObjectnessHead(nn.Module):
    """Faster R-CNN style two-FC box classifier with background/object logits."""

    def __init__(self, channels: int, pool_resolution: int,
                 hidden_dim: int = 1024):
        super().__init__()
        input_dim = int(channels) * int(pool_resolution) ** 2
        self.fc1 = nn.Linear(input_dim, int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.cls_score = nn.Linear(int(hidden_dim), 2)
        for layer in (self.fc1, self.fc2, self.cls_score):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.flatten(1).float()
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return self.cls_score(x)

    def objectness_logit(self, features: torch.Tensor) -> torch.Tensor:
        logits = self(features)
        return logits[:, 1] - logits[:, 0]


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_order(scores: torch.Tensor, ious: torch.Tensor,
                    layout: Sequence[Dict], level: Optional[int],
                    min_iou: Optional[float] = None,
                    max_iou: Optional[float] = None) -> List[int]:
    indices = []
    for index, location in enumerate(layout):
        if level is not None and int(location['level']) != int(level):
            continue
        iou = float(ious[index].item())
        if min_iou is not None and iou < float(min_iou):
            continue
        if max_iou is not None and iou >= float(max_iou):
            continue
        indices.append(index)
    indices.sort(key=lambda index: float(scores[index].item()), reverse=True)
    return indices


def valid_candidate_selections(
        ordered_indices: Sequence[int], boxes: torch.Tensor,
        detector_meta: Dict, dino_meta: Dict, dino_feature: torch.Tensor,
        patch_size: int, pool_resolution: int, min_in_bounds: float,
        limit: int) -> List[Dict]:
    selections = []
    for detector_rank, index in enumerate(ordered_indices, start=1):
        dino_box = audit.detector_box_to_dino(
            boxes[int(index), :5], detector_meta, dino_meta)
        fraction = audit.oriented_roi_in_bounds_fraction(
            dino_feature, dino_box.to(dino_feature.device),
            patch_size, pool_resolution)
        if fraction < float(min_in_bounds):
            continue
        selections.append(dict(
            index=int(index), detector_rank=int(detector_rank),
            dino_box=dino_box,
            in_bounds_fraction=float(fraction)))
        if len(selections) >= int(limit):
            break
    return selections


def sample_spatial_rois(dino_feature: torch.Tensor,
                        dino_boxes: Sequence[torch.Tensor],
                        patch_size: int, pool_resolution: int) -> torch.Tensor:
    if not dino_boxes:
        channels = int(dino_feature.shape[1])
        return torch.empty(0, channels, pool_resolution, pool_resolution)
    grids = [audit.oriented_roi_grid(
        dino_feature, box.to(dino_feature.device),
        patch_size, pool_resolution)[0] for box in dino_boxes]
    grid = torch.cat(grids, dim=0)
    expanded = dino_feature.expand(len(dino_boxes), -1, -1, -1)
    return F.grid_sample(
        expanded, grid, mode='bilinear', padding_mode='zeros',
        align_corners=True)


def _feature_sample(feature: torch.Tensor, label: int, record: Dict) -> Dict:
    return dict(feature=feature.detach().cpu().half(), label=int(label),
                row=record)


def collect_source_features(detector, dino, records: Sequence[Dict],
                            transforms, img_scale, flip, args,
                            dino_device: torch.device):
    from mmcv.ops import box_iou_rotated

    diag = transfer.entry_probe.get_diag()
    samples = []
    rows = []
    skipped_positive = 0
    skipped_negative_image = 0
    for record_index, record in enumerate(records):
        img_tensor, detector_meta, _image_stats = diag.preprocess_image(
            record['image'], transforms, img_scale, flip)
        if img_tensor is None:
            raise RuntimeError('Source preprocessing failed')
        img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))
        with torch.no_grad():
            detector_features = detector.extract_feat(img_tensor)
            _head, boxes, scores, layout, _decode = (
                transfer.forward_main_candidates(
                    detector, detector_features, detector_meta['img_shape']))
            gt_boxes = transfer.scaled_gt_tensors(
                record, detector_meta, boxes.device)
            if gt_boxes.numel() == 0:
                continue
            iou_matrix = box_iou_rotated(boxes.float(), gt_boxes.float())
            max_ious = iou_matrix.max(dim=1).values
            boxes_cpu = boxes.detach().cpu()
            scores_cpu = scores.detach().cpu()
            iou_matrix_cpu = iou_matrix.detach().cpu()
            max_ious_cpu = max_ious.detach().cpu()
        del img_tensor, detector_features, boxes, scores, gt_boxes, iou_matrix
        dino_feature, dino_meta = audit._prepare_image_features(
            dino, record['image'], args.dino_height, args.patch_size,
            args.dino_max_long_side, dino_device)

        positive_selections = []
        for gt_index in range(int(iou_matrix_cpu.shape[1])):
            order = candidate_order(
                scores_cpu, iou_matrix_cpu[:, gt_index], layout, None,
                min_iou=args.riou_thr)
            selected = valid_candidate_selections(
                order, boxes_cpu, detector_meta, dino_meta, dino_feature,
                args.patch_size, args.pool_resolution,
                args.min_roi_in_bounds, 1)
            if not selected:
                skipped_positive += 1
                continue
            selected[0]['gt_index'] = int(gt_index)
            positive_selections.extend(selected)

        negative_order = candidate_order(
            scores_cpu, max_ious_cpu, layout, 0,
            max_iou=args.false_iou_thr)
        negative_selections = valid_candidate_selections(
            negative_order, boxes_cpu, detector_meta, dino_meta, dino_feature,
            args.patch_size, args.pool_resolution, args.min_roi_in_bounds,
            args.source_negatives_per_image)
        if not negative_selections:
            skipped_negative_image += 1

        selections = positive_selections + negative_selections
        if selections:
            spatial = sample_spatial_rois(
                dino_feature, [item['dino_box'] for item in selections],
                args.patch_size, args.pool_resolution)
            for sample_index, selection in enumerate(selections):
                is_positive = sample_index < len(positive_selections)
                index = selection['index']
                row = dict(
                    split=neighborhood.SOURCE_SPLIT,
                    seq=record['seq'], frame=int(record['frame']),
                    label=int(is_positive), candidate_index=int(index),
                    detector_rank=int(selection['detector_rank']),
                    level=int(layout[index]['level']),
                    anchor_id=int(layout[index]['anchor_id']),
                    riou=float((iou_matrix_cpu[index, selection['gt_index']]
                                if is_positive else max_ious_cpu[index]).item()),
                    main_cls_score=float(scores_cpu[index].item()),
                    in_bounds_fraction=float(selection['in_bounds_fraction']))
                samples.append(_feature_sample(
                    spatial[sample_index], int(is_positive), row))
                rows.append(row)
        del dino_feature
        if (record_index + 1) % 25 == 0:
            print('[source-roi] {}/{} images samples={}'.format(
                record_index + 1, len(records), len(samples)))
    positives = int(sum(sample['label'] == 1 for sample in samples))
    negatives = int(sum(sample['label'] == 0 for sample in samples))
    if positives == 0 or negatives == 0:
        raise RuntimeError('Source ROI collection lacks a class')
    return samples, rows, dict(
        image_count=len(records), sample_count=len(samples),
        positive_count=positives, negative_count=negatives,
        skipped_positive=int(skipped_positive),
        skipped_negative_image=int(skipped_negative_image))


def grouped_fold_ids(samples: Sequence[Dict], folds: int) -> List[int]:
    groups = []
    group_index = {}
    for sample in samples:
        group = (str(sample['row']['seq']), int(sample['row']['frame']))
        if group not in group_index:
            group_index[group] = len(groups)
            groups.append(group)
    group_folds = neighborhood.contiguous_fold_ids(len(groups), int(folds))
    return [group_folds[group_index[(str(sample['row']['seq']),
                                     int(sample['row']['frame']))]]
            for sample in samples]


def _batches(indices: Sequence[int], batch_size: int, seed: int,
             shuffle: bool) -> List[List[int]]:
    ordered = list(indices)
    if shuffle:
        generator = random.Random(seed)
        generator.shuffle(ordered)
    return [ordered[start:start + int(batch_size)]
            for start in range(0, len(ordered), int(batch_size))]


def evaluate_head(head, samples: Sequence[Dict], indices: Sequence[int],
                  batch_size: int, device: torch.device) -> Dict:
    head.eval()
    predictions = []
    labels = []
    rows = []
    losses = []
    with torch.no_grad():
        for batch_indices in _batches(indices, batch_size, 0, False):
            features = torch.stack([
                samples[index]['feature'] for index in batch_indices]).to(
                    device=device, dtype=torch.float32)
            target = torch.tensor([
                samples[index]['label'] for index in batch_indices],
                dtype=torch.long, device=device)
            logits = head(features)
            losses.append(float(F.cross_entropy(logits, target).item()))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(target.cpu().tolist())
            rows.extend([samples[index]['row'] for index in batch_indices])
    positive_hits = [prediction == 1 for prediction, label in zip(
        predictions, labels) if label == 1]
    negative_hits = [prediction == 0 for prediction, label in zip(
        predictions, labels) if label == 0]
    frame_scores = {}
    with torch.no_grad():
        for batch_indices in _batches(indices, batch_size, 0, False):
            features = torch.stack([
                samples[index]['feature'] for index in batch_indices]).to(
                    device=device, dtype=torch.float32)
            logits = head.objectness_logit(features).cpu().tolist()
            for index, logit in zip(batch_indices, logits):
                row = samples[index]['row']
                key = (str(row['seq']), int(row['frame']))
                frame_scores.setdefault(key, {0: [], 1: []})[
                    int(samples[index]['label'])].append(float(logit))
    paired = [max(values[1]) > max(values[0])
              for values in frame_scores.values()
              if values[0] and values[1]]
    positive_accuracy = float(np.mean(positive_hits)) if positive_hits else 0.0
    negative_accuracy = float(np.mean(negative_hits)) if negative_hits else 0.0
    paired_accuracy = float(np.mean(paired)) if paired else 0.0
    return dict(
        count=len(indices), loss=float(np.mean(losses)) if losses else None,
        positive_accuracy=positive_accuracy,
        negative_accuracy=negative_accuracy,
        paired_accuracy=paired_accuracy,
        selection_score=min(
            positive_accuracy, negative_accuracy, paired_accuracy))


def train_one_head(samples: Sequence[Dict], train_indices: Sequence[int],
                   validation_indices: Optional[Sequence[int]], args,
                   device: torch.device, seed: int, epochs: int):
    channels = int(samples[0]['feature'].shape[0])
    head = TwoFCObjectnessHead(
        channels, args.pool_resolution, args.hidden_dim).to(device)
    optimizer = torch.optim.SGD(
        head.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)
    milestones = sorted(set([
        max(1, int(round(epochs * 2.0 / 3.0))),
        max(1, int(round(epochs * 5.0 / 6.0)))]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1)
    best_state = None
    best_metrics = None
    best_epoch = int(epochs)
    history = []
    for epoch in range(1, int(epochs) + 1):
        head.train()
        losses = []
        for batch_indices in _batches(
                train_indices, args.batch_size,
                seed + epoch * 1009, True):
            features = torch.stack([
                samples[index]['feature'] for index in batch_indices]).to(
                    device=device, dtype=torch.float32)
            labels = torch.tensor([
                samples[index]['label'] for index in batch_indices],
                dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = F.cross_entropy(head(features), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)))
        if validation_indices is not None:
            metrics = evaluate_head(
                head, samples, validation_indices,
                args.batch_size, device)
            record['validation'] = metrics
            key = (metrics['selection_score'], -metrics['loss'])
            previous = None if best_metrics is None else (
                best_metrics['selection_score'], -best_metrics['loss'])
            if previous is None or key > previous:
                best_metrics = metrics
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in head.state_dict().items()}
        history.append(record)
    if validation_indices is not None:
        head.load_state_dict(best_state)
    return head, dict(
        best_epoch=int(best_epoch), best_metrics=best_metrics,
        history=history)


def cross_validate(samples: Sequence[Dict], args, device: torch.device):
    fold_ids = grouped_fold_ids(samples, args.source_folds)
    folds = []
    for fold_id in range(int(args.source_folds)):
        train_indices = [index for index, value in enumerate(fold_ids)
                         if int(value) != fold_id]
        validation_indices = [index for index, value in enumerate(fold_ids)
                              if int(value) == fold_id]
        head, result = train_one_head(
            samples, train_indices, validation_indices,
            args, device, args.seed + fold_id * 100000, args.epochs)
        del head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result.update(
            fold_id=int(fold_id), train_count=len(train_indices),
            validation_count=len(validation_indices))
        folds.append(result)
        print('[source-cv] fold={} epoch={} pos={:.3f} neg={:.3f} pair={:.3f}'.format(
            fold_id, result['best_epoch'],
            result['best_metrics']['positive_accuracy'],
            result['best_metrics']['negative_accuracy'],
            result['best_metrics']['paired_accuracy']))
    best_epochs = [item['best_epoch'] for item in folds]
    final_epochs = int(round(float(np.median(best_epochs))))
    summary = dict(
        fold_count=len(folds), final_epochs=final_epochs,
        minimum_positive_accuracy=float(min(
            item['best_metrics']['positive_accuracy'] for item in folds)),
        minimum_negative_accuracy=float(min(
            item['best_metrics']['negative_accuracy'] for item in folds)),
        minimum_paired_accuracy=float(min(
            item['best_metrics']['paired_accuracy'] for item in folds)))
    summary['valid'] = bool(
        summary['minimum_positive_accuracy'] >= args.source_min_accuracy
        and summary['minimum_negative_accuracy'] >= args.source_min_accuracy
        and summary['minimum_paired_accuracy'] >= args.source_min_accuracy)
    return folds, summary


def score_target_candidates(head, dino_feature: torch.Tensor,
                            selections: Sequence[Dict], args,
                            head_device: torch.device) -> List[float]:
    logits = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(selections), args.roi_chunk_size):
            chunk = selections[start:start + args.roi_chunk_size]
            features = sample_spatial_rois(
                dino_feature, [item['dino_box'] for item in chunk],
                args.patch_size, args.pool_resolution)
            logits.extend(head.objectness_logit(
                features.to(device=head_device, dtype=torch.float32)
            ).cpu().tolist())
    return [float(value) for value in logits]


def collect_target_results(detector, dino, head, transforms,
                           img_scale, flip, args,
                           dino_device: torch.device,
                           head_device: torch.device):
    from mmcv.ops import box_iou_rotated

    diag = transfer.entry_probe.get_diag()
    rows = []
    for frame_id in range(args.target_start, args.target_end + 1):
        img_path, ann_path = diag.find_files(
            args.data_root, neighborhood.TARGET_SPLIT,
            neighborhood.TARGET_SEQ, frame_id)
        if img_path is None or ann_path is None:
            raise RuntimeError('Missing target-dev frame {}'.format(frame_id))
        record = dict(
            split=neighborhood.TARGET_SPLIT,
            seq=neighborhood.TARGET_SEQ, frame=frame_id,
            image=img_path, annotation=ann_path, domain='real')
        img_tensor, detector_meta, _stats = diag.preprocess_image(
            img_path, transforms, img_scale, flip)
        if img_tensor is None:
            raise RuntimeError('Target preprocessing failed')
        img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))
        with torch.no_grad():
            detector_features = detector.extract_feat(img_tensor)
            _main_head, boxes, scores, layout, _decode = (
                transfer.forward_main_candidates(
                    detector, detector_features, detector_meta['img_shape']))
            gt_boxes = transfer.scaled_gt_tensors(
                record, detector_meta, boxes.device)
            ious = box_iou_rotated(
                boxes.float(), gt_boxes.float()).max(dim=1).values
            boxes_cpu = boxes.detach().cpu()
            scores_cpu = scores.detach().cpu()
            ious_cpu = ious.detach().cpu()
        del img_tensor, detector_features, boxes, scores, gt_boxes, ious
        level0_order = candidate_order(scores_cpu, ious_cpu, layout, 0)
        limited_order = level0_order[:int(args.target_candidate_limit)]
        geometry_eligible = bool(any(
            float(ious_cpu[index].item()) >= args.riou_thr
            for index in level0_order))
        dino_feature, dino_meta = audit._prepare_image_features(
            dino, img_path, args.dino_height, args.patch_size,
            args.dino_max_long_side, dino_device)
        selections = valid_candidate_selections(
            limited_order, boxes_cpu, detector_meta, dino_meta, dino_feature,
            args.patch_size, args.pool_resolution, args.min_roi_in_bounds,
            args.target_candidate_limit)
        logits = score_target_candidates(
            head, dino_feature, selections, args, head_device)
        scored = []
        for selection, logit in zip(selections, logits):
            index = selection['index']
            scored.append(dict(
                candidate_index=int(index), detector_rank=int(
                    selection['detector_rank']),
                objectness_logit=float(logit),
                riou=float(ious_cpu[index].item()),
                main_cls_score=float(scores_cpu[index].item())))
        ranked = sorted(scored, key=lambda item: item['objectness_logit'],
                        reverse=True)
        usable_ranks = [rank for rank, item in enumerate(ranked, start=1)
                        if item['riou'] >= args.riou_thr]
        usable_logits = [item['objectness_logit'] for item in scored
                         if item['riou'] >= args.riou_thr]
        false_logits = [item['objectness_logit'] for item in scored
                        if item['riou'] < args.false_iou_thr]
        top1_hit = bool(ranked and ranked[0]['riou'] >= args.riou_thr)
        paired_margin = (None if not usable_logits or not false_logits
                         else float(max(usable_logits) - max(false_logits)))
        rows.append(dict(
            role='target_dev_diagnosis_only', frame=int(frame_id),
            geometry_eligible=geometry_eligible,
            level0_candidate_count=len(level0_order),
            detector_topk_count=len(limited_order),
            valid_dino_roi_count=len(scored),
            usable_in_topk=bool(usable_ranks), top1_hit=top1_hit,
            best_usable_rank=(None if not usable_ranks else min(usable_ranks)),
            paired_margin=paired_margin,
            top1=(None if not ranked else ranked[0]),
            best_usable=(None if not usable_ranks else ranked[min(usable_ranks) - 1])))
        print('[target-roi] frame={} geometry={} top1={} rank={}'.format(
            frame_id, geometry_eligible, top1_hit,
            None if not usable_ranks else min(usable_ranks)))
        del dino_feature
    return rows


def summarize_target(rows: Sequence[Dict], args) -> Dict:
    eligible = [row for row in rows if row['geometry_eligible']]
    evaluable = [row for row in eligible if row['usable_in_topk']]
    top1_wins = int(sum(row['top1_hit'] for row in evaluable))
    paired_wins = int(sum(
        row['paired_margin'] is not None and row['paired_margin'] > 0.0
        for row in evaluable))
    ranks = [row['best_usable_rank'] for row in evaluable]
    margins = [row['paired_margin'] for row in evaluable
               if row['paired_margin'] is not None]
    return dict(
        geometry_eligible_count=len(eligible), evaluable_count=len(evaluable),
        geometry_misses=[int(row['frame']) for row in rows
                         if not row['geometry_eligible']],
        topk_geometry_recall=(float(len(evaluable)) / len(eligible)
                              if eligible else 0.0),
        top1_wins=top1_wins, paired_wins=paired_wins,
        best_usable_rank=dict(
            minimum=None if not ranks else int(min(ranks)),
            median=None if not ranks else float(np.median(ranks)),
            maximum=None if not ranks else int(max(ranks))),
        paired_margin=dict(
            minimum=None if not margins else float(min(margins)),
            median=None if not margins else float(np.median(margins)),
            maximum=None if not margins else float(max(margins))))


def target_decision(source_cv: Dict, target: Dict, args) -> str:
    if not source_cv['valid']:
        return 'SOURCE_CONTROL_FAILED'
    if (target['geometry_eligible_count'] != neighborhood.EXPECTED_ELIGIBLE
            or target['geometry_misses']
            != neighborhood.EXPECTED_GEOMETRY_MISSES
            or target['evaluable_count'] != neighborhood.EXPECTED_ELIGIBLE):
        return 'AUDIT_INVALID'
    if (target['top1_wins'] >= args.target_min_wins
            and target['paired_margin']['median'] is not None
            and target['paired_margin']['median'] > 0.0):
        return 'SOURCE_ONLY_DINO_ROI_HEAD_RESTORES_ORDERING'
    if target['paired_wins'] >= args.target_min_wins:
        return 'PAIRWISE_SIGNAL_ONLY_GLOBAL_RANK_NOT_RESTORED'
    return 'SOURCE_ONLY_DINO_ROI_HEAD_INSUFFICIENT'


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    head_device = torch.device('cuda:{}'.format(args.gpu))
    dino_devices = [torch.device('cuda:{}'.format(gpu))
                    for gpu in args.dino_gpus]
    dino_device = dino_devices[0]

    detector, cfg = transfer.entry_probe.load_model(
        args.config, args.detector_checkpoint, args.gpu)
    transfer.freeze_detector(detector)
    detector_versions = alignment.module_parameter_versions(detector)
    dino, loaded_patch_size = audit.load_frozen_dinov2(
        args.dinov2_repo, args.dinov2_checkpoint,
        args.dinov2_model, dino_devices,
        args.legacy_sdpa_query_chunk)
    if int(loaded_patch_size) != int(args.patch_size):
        raise RuntimeError('Unexpected DINO patch size')
    dino_versions = alignment.module_parameter_versions(dino)

    diag = transfer.entry_probe.get_diag()
    transforms, img_scale, flip = diag.build_test_transforms(cfg)
    source_records = [
        record for record in transfer.discover_labeled_records(
            args.data_root, neighborhood.SOURCE_SPLIT, 0)
        if record['seq'] == args.source_seq]
    if args.max_source_samples > 0:
        source_records = source_records[:args.max_source_samples]
    source_samples, source_rows, source_collection = collect_source_features(
        detector, dino, source_records, transforms, img_scale, flip,
        args, dino_device)
    cv_folds, cv_summary = cross_validate(
        source_samples, args, head_device)
    if not cv_summary['valid']:
        target_rows = []
        target_summary = None
        final_checkpoint = None
        final_training = None
        head_unchanged_during_target = None
        decision = 'SOURCE_CONTROL_FAILED'
    else:
        all_indices = list(range(len(source_samples)))
        final_head, final_training = train_one_head(
            source_samples, all_indices, None, args, head_device,
            args.seed + 900000, cv_summary['final_epochs'])
        final_checkpoint = os.path.join(
            args.work_dir, 'source_only_dino_roi_head.pth')
        torch.save(dict(
            state_dict={name: value.detach().cpu()
                        for name, value in final_head.state_dict().items()},
            channels=int(source_samples[0]['feature'].shape[0]),
            pool_resolution=int(args.pool_resolution),
            hidden_dim=int(args.hidden_dim),
            epochs=int(cv_summary['final_epochs']),
            source_only=True), final_checkpoint)
        head_versions = alignment.module_parameter_versions(final_head)
        target_rows = collect_target_results(
            detector, dino, final_head, transforms, img_scale, flip,
            args, dino_device, head_device)
        head_unchanged_during_target = (
            head_versions == alignment.module_parameter_versions(final_head))
        if not head_unchanged_during_target:
            raise RuntimeError('ROI head changed during target diagnosis')
        target_summary = summarize_target(target_rows, args)
        decision = target_decision(cv_summary, target_summary, args)
        del final_head

    detector_unchanged = (
        detector_versions == alignment.module_parameter_versions(detector))
    dino_unchanged = (
        dino_versions == alignment.module_parameter_versions(dino))
    if not detector_unchanged or not dino_unchanged:
        raise RuntimeError('Frozen detector/DINO parameter invariant failed')
    payload = dict(
        probe=PROBE_NAME, protocol_version=PROTOCOL_VERSION,
        paper_mapping=dict(
            retained=[
                'frozen DINOv2 backbone',
                'single-scale 7x7 proposal ROI features',
                'Faster R-CNN style two-FC classification head',
                'source-labelled proposal supervision'],
            rotated_adaptation='orientation-aware grid_sample ROI features',
            omitted=[
                'trainable RPN', 'bbox regression update',
                'target pseudo-label training', 'student feature alignment']),
        isolation=dict(
            detector_frozen=True, detector_parameters_unchanged=detector_unchanged,
            dinov2_frozen=True, dinov2_parameters_unchanged=dino_unchanged,
            trains_roi_head=True, training_data='source_only',
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_used_for_diagnosis_only=True,
            roi_head_parameters_unchanged_during_target=(
                head_unchanged_during_target),
            modifies_detector_scores=False, modifies_bbox_regression=False,
            score_fusion=False),
        config=os.path.abspath(args.config),
        detector_checkpoint=os.path.abspath(args.detector_checkpoint),
        dinov2_checkpoint=os.path.abspath(args.dinov2_checkpoint),
        source=dict(collection=source_collection, rows=source_rows,
                    cross_validation=cv_summary, folds=cv_folds),
        final_head=(None if final_checkpoint is None else dict(
            checkpoint=os.path.abspath(final_checkpoint),
            sha256=sha256_file(final_checkpoint),
            selected_epochs=int(cv_summary['final_epochs']),
            selection='median_source_cv_best_epoch',
            source_training=final_training)),
        target_dev=(None if target_summary is None else dict(
            candidate_limit=int(args.target_candidate_limit),
            summary=target_summary, rows=target_rows)),
        decision=decision)
    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, 'w') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False,
                  allow_nan=False)
    print('[roi-head] {}'.format(decision))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
