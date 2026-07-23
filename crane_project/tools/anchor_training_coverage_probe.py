#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit source-only anchor assignment and classifier-gradient coverage.

This probe never creates an optimizer, never calls ``step()``, and never saves
a checkpoint.  Existing checkpoints are read to trace ``retina_cls`` filter
development.  Source-train samples are forwarded through the actual SymPOLA
target assignment; a small optional subset runs backward only to measure
gradient norms before gradients are immediately cleared.
"""

import argparse
import glob
import json
import math
import os
import random
import re
import sys
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import ctx_entry_probe as entry_probe  # noqa: E402
from crane_project.tools import retina_cls_contribution_probe as contribution  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Source-only SymPOLA anchor coverage and gradient audit.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--trajectory-checkpoints', nargs='*', default=[])
    parser.add_argument('--source-indexes', type=int, nargs='+', default=[0, 1],
                        help='Indexes in cfg.data.train, normally real/sim.')
    parser.add_argument('--assignment-phases', nargs='+',
                        choices=['warmup', 'steady'],
                        default=['warmup', 'steady'],
                        help='Explicit synthetic SymPOLA phases. Checkpoints '
                             'do not store _local_call_count.')
    parser.add_argument('--max-samples-per-source', type=int, default=100)
    parser.add_argument('--gradient-samples-per-source', type=int, default=8)
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def validate_args(args):
    indexes = [int(value) for value in args.source_indexes]
    phases = list(args.assignment_phases)
    if args.seed != 0:
        raise ValueError('The unified diagnostic protocol requires --seed 0')
    if not indexes or len(set(indexes)) != len(indexes):
        raise ValueError('--source-indexes requires unique values')
    if min(indexes) < 0:
        raise ValueError('--source-indexes must be non-negative')
    if not phases or len(set(phases)) != len(phases):
        raise ValueError('--assignment-phases requires unique values')
    if args.max_samples_per_source <= 0:
        raise ValueError('--max-samples-per-source must be positive')
    if not 0 <= args.gradient_samples_per_source <= args.max_samples_per_source:
        raise ValueError(
            '--gradient-samples-per-source must be in [0, max samples]')
    if not 0.0 < args.riou_thr <= 1.0:
        raise ValueError('--riou-thr must be in (0, 1]')
    return indexes


def configure_assignment_phase(assigner, phase: str) -> Dict:
    """Set a synthetic, explicit assigner phase for checkpoint diagnosis."""
    if phase not in ('warmup', 'steady'):
        raise ValueError('Unsupported assignment phase: {}'.format(phase))
    warmup_iters = int(getattr(assigner, 'o2m_warmup_iters', 0))
    o2m_topk = int(getattr(assigner, 'o2m_topk', 1))
    steady_topk = int(getattr(assigner, 'topk', 1))
    calls_per_iter = 2  # Matches SymPOLAAssigner._get_current_tau/current_iter.
    if phase == 'warmup':
        synthetic_iter = 0
        expected_topk = o2m_topk if warmup_iters > 0 else 1
    else:
        # The assigner ramps down for another warmup_iters after warmup.
        synthetic_iter = 2 * warmup_iters + 1
        expected_topk = steady_topk
    starting_call_count = synthetic_iter * calls_per_iter
    assigner._local_call_count = int(starting_call_count)
    return dict(
        phase=phase,
        counter_source='synthetic_explicit_not_checkpoint_state',
        starting_local_call_count=int(starting_call_count),
        synthetic_current_iter=int(synthetic_iter),
        o2m=bool(getattr(assigner, 'o2m', False)),
        o2m_warmup_iters=warmup_iters,
        o2m_topk=o2m_topk,
        steady_topk=steady_topk,
        expected_effective_topk=int(expected_topk))


def _number(value) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def _epoch_key(path: str):
    match = re.search(r'epoch_(\d+)\.pth$', path)
    return (int(match.group(1)) if match else 10 ** 9, path)


def expand_checkpoint_paths(current: str,
                            requested: Sequence[str]) -> List[str]:
    paths = [current]
    for value in requested:
        matches = glob.glob(value)
        paths.extend(matches if matches else [value])
    unique = []
    seen = set()
    for path in sorted(paths, key=_epoch_key):
        absolute = os.path.abspath(path)
        if absolute not in seen:
            unique.append(path)
            seen.add(absolute)
    return unique


def _find_state_tensor(state: Dict[str, torch.Tensor], suffix: str):
    matches = [(key, value) for key, value in state.items()
               if key.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            'Expected one checkpoint tensor ending {}, found {}'.format(
                suffix, [key for key, _ in matches]))
    return matches[0]


def checkpoint_filter_stats(path: str) -> Dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    weight_key, weight = _find_state_tensor(
        state, 'bbox_head.retina_cls.weight')
    bias_key, bias = _find_state_tensor(
        state, 'bbox_head.retina_cls.bias')
    weight = weight.detach().float()
    bias = bias.detach().float()
    if weight.ndim != 4 or weight.shape[0] != bias.numel():
        raise RuntimeError('Invalid retina_cls checkpoint tensors')
    rows = []
    for anchor_id in range(int(weight.shape[0])):
        item = weight[anchor_id]
        rows.append(dict(
            anchor_id=anchor_id,
            weight_norm=_number(item.norm().item()),
            weight_abs_mean=_number(item.abs().mean().item()),
            weight_std=_number(item.std(unbiased=False).item()),
            bias=_number(bias[anchor_id].item())))
    return dict(
        checkpoint=os.path.abspath(path),
        epoch=_epoch_key(path)[0] if _epoch_key(path)[0] < 10 ** 9 else None,
        weight_key=weight_key,
        bias_key=bias_key,
        filters=rows)


def positive_anchor_counts(labels_per_level: Sequence[torch.Tensor],
                           num_classes: int,
                           num_anchors: int) -> Dict:
    by_anchor = [0 for _ in range(num_anchors)]
    by_level_anchor = []
    total = 0
    for level, labels in enumerate(labels_per_level):
        flat = labels.reshape(-1)
        positive_indices = torch.nonzero(
            (flat >= 0) & (flat < int(num_classes)),
            as_tuple=False).reshape(-1)
        level_counts = [0 for _ in range(num_anchors)]
        for anchor_id in range(num_anchors):
            count = int((positive_indices % num_anchors == anchor_id).sum().item())
            level_counts[anchor_id] = count
            by_anchor[anchor_id] += count
            total += count
        by_level_anchor.append(dict(level=level, counts=level_counts))
    return dict(total=total, by_anchor=by_anchor,
                by_level_anchor=by_level_anchor)


def gt_geometry_stats(gt_bboxes: Sequence[torch.Tensor]) -> List[Dict]:
    rows = []
    for boxes in gt_bboxes:
        for box in boxes.detach().float().cpu():
            width = max(float(box[2].item()), 1e-6)
            height = max(float(box[3].item()), 1e-6)
            rows.append(dict(
                width=width, height=height,
                width_over_height=_number(width / height),
                symmetric_aspect=_number(max(width, height) / min(width, height)),
                area=_number(width * height)))
    return rows


def per_anchor_geometry(scores: torch.Tensor, ious: torch.Tensor,
                        layout: Sequence[Dict], num_anchors: int,
                        riou_thr: float) -> List[Dict]:
    selected = contribution.select_per_anchor_candidates(
        scores, ious, layout, num_anchors, riou_thr)
    rows = []
    for item in selected:
        rows.append(dict(
            anchor_id=int(item['anchor_id']),
            candidate_count=int(item['candidate_count']),
            highest_score=item['highest_score'],
            dense_best_geometry=item['dense_best_geometry'],
            best_usable_by_score=item['best_usable_by_score']))
    return rows


def _as_list(value):
    if isinstance(value, tuple):
        return list(value)
    return value if isinstance(value, list) else [value]


def normalize_scattered_batch(data: Dict) -> Tuple[
        torch.Tensor, List[Dict], List[torch.Tensor], List[torch.Tensor]]:
    img = data['img']
    if isinstance(img, (list, tuple)) and len(img) == 1:
        img = img[0]
    img_metas = _as_list(data['img_metas'])
    gt_bboxes = _as_list(data['gt_bboxes'])
    gt_labels = _as_list(data['gt_labels'])
    if not isinstance(img, torch.Tensor) or img.ndim != 4:
        raise RuntimeError('Expected scattered img tensor [B,C,H,W]')
    if not all(isinstance(item, dict) for item in img_metas):
        raise RuntimeError('Expected scattered img_metas list[dict]')
    if not all(isinstance(item, torch.Tensor) for item in gt_bboxes):
        raise RuntimeError('Expected scattered gt_bboxes list[tensor]')
    return img, img_metas, gt_bboxes, gt_labels


def build_flat_predictions(cls_scores, bbox_preds):
    flat_cls, flat_bbox = [], []
    batch_size = int(cls_scores[0].shape[0])
    for image_index in range(batch_size):
        image_cls, image_bbox = [], []
        for cls_level, bbox_level in zip(cls_scores, bbox_preds):
            image_cls.append(cls_level[image_index].permute(
                1, 2, 0).reshape(-1, cls_level.shape[1] // (
                    bbox_level.shape[1] // 5)))
            image_bbox.append(bbox_level[image_index].permute(
                1, 2, 0).reshape(-1, 5))
        flat_cls.append(torch.cat(image_cls))
        flat_bbox.append(torch.cat(image_bbox))
    return flat_cls, flat_bbox


def assignment_targets(head, cls_scores, bbox_preds, img_metas,
                       gt_bboxes, gt_labels):
    featmap_sizes = [item.shape[-2:] for item in cls_scores]
    anchor_list, valid_flags = head.get_anchors(
        featmap_sizes, img_metas, device=cls_scores[0].device)
    flat_cls, flat_bbox = build_flat_predictions(cls_scores, bbox_preds)
    targets = head.get_targets(
        anchor_list, valid_flags, flat_cls, flat_bbox,
        gt_bboxes, img_metas, gt_labels_list=gt_labels,
        label_channels=head.cls_out_channels)
    if targets is None:
        raise RuntimeError('SymPOLA returned no valid assignment targets')
    return targets


def classification_gradient_norms(head, cls_scores, bbox_preds,
                                  gt_bboxes, gt_labels, img_metas) -> Dict:
    head.zero_grad(set_to_none=True)
    losses = head.loss(
        cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas)
    if losses is None:
        raise RuntimeError('Main head returned no losses')
    loss_terms = losses.get('loss_cls', [])
    loss_terms = _as_list(loss_terms)
    cls_loss = sum(term for term in loss_terms if torch.is_tensor(term))
    cls_loss.backward()
    grad = head.retina_cls.weight.grad
    bias_grad = head.retina_cls.bias.grad
    if grad is None:
        raise RuntimeError('retina_cls received no classification gradient')
    result = dict(
        loss_cls=_number(cls_loss.detach().item()),
        weight_grad_norms=[
            _number(grad[index].detach().float().norm().item())
            for index in range(int(grad.shape[0]))],
        bias_grad_abs=[
            _number(bias_grad[index].detach().float().abs().item())
            for index in range(int(bias_grad.numel()))]
        if bias_grad is not None else None)
    head.zero_grad(set_to_none=True)
    return result


def analyze_batch(model, head, data: Dict, riou_thr: float,
                  measure_gradient: bool) -> Dict:
    from mmcv.ops import box_iou_rotated

    img, img_metas, gt_bboxes, gt_labels = normalize_scattered_batch(data)
    if int(img.shape[0]) != 1:
        raise RuntimeError('Coverage probe requires samples_per_gpu=1')
    context = nullcontext() if measure_gradient else torch.no_grad()
    with context:
        features = model.extract_feat(img)
        if measure_gradient:
            # Gradient attribution is restricted to the main head.  This
            # prevents diagnostic backward calls from accumulating gradients
            # in the backbone/FPN while preserving the exact head inputs.
            features = tuple(feature.detach() for feature in features)
        cls_scores, bbox_preds = head(features)
        targets = assignment_targets(
            head, cls_scores, bbox_preds, img_metas,
            gt_bboxes, gt_labels)
        labels_per_level = targets[0]
        positives = positive_anchor_counts(
            labels_per_level, head.num_classes, head.num_anchors)

        boxes, scores, _levels, _centers, _alignment = (
            entry_probe.flatten_decode_candidates(
                head, cls_scores, bbox_preds, img_metas[0]['img_shape']))
        if gt_bboxes[0].numel() > 0:
            ious = box_iou_rotated(
                boxes.float(), gt_bboxes[0].float()).max(dim=1).values
        else:
            ious = scores.new_zeros(scores.shape)
        layout = contribution.candidate_layout(
            cls_scores, head, img_metas[0]['img_shape'])
        geometry = per_anchor_geometry(
            scores, ious, layout, head.num_anchors, riou_thr)
        gradient = None
        if measure_gradient:
            gradient = classification_gradient_norms(
                head, cls_scores, bbox_preds,
                gt_bboxes, gt_labels, img_metas)

    return dict(
        filename=img_metas[0].get('filename'),
        gt_count=int(gt_bboxes[0].shape[0]),
        gt_geometry=gt_geometry_stats(gt_bboxes),
        positive_assignments=positives,
        per_anchor_geometry=geometry,
        classification_gradient=gradient)


def aggregate_source(rows: Sequence[Dict], num_anchors: int) -> Dict:
    positive_counts = [0 for _ in range(num_anchors)]
    usable_frames = [0 for _ in range(num_anchors)]
    dense_rious = [[] for _ in range(num_anchors)]
    gradient_norms = [[] for _ in range(num_anchors)]
    bias_gradients = [[] for _ in range(num_anchors)]
    aspects = []
    for row in rows:
        for anchor_id, count in enumerate(
                row['positive_assignments']['by_anchor']):
            positive_counts[anchor_id] += int(count)
        for item in row['per_anchor_geometry']:
            anchor_id = int(item['anchor_id'])
            dense = item.get('dense_best_geometry')
            if dense is not None:
                dense_rious[anchor_id].append(float(dense['riou']))
            if item.get('best_usable_by_score') is not None:
                usable_frames[anchor_id] += 1
        gradient = row.get('classification_gradient')
        if gradient:
            for anchor_id, value in enumerate(gradient['weight_grad_norms']):
                gradient_norms[anchor_id].append(float(value))
            for anchor_id, value in enumerate(gradient['bias_grad_abs'] or []):
                bias_gradients[anchor_id].append(float(value))
        aspects.extend(
            float(item['symmetric_aspect']) for item in row['gt_geometry'])

    total_positive = sum(positive_counts)
    return dict(
        samples=len(rows),
        gt_count=sum(int(row['gt_count']) for row in rows),
        positive_assignments_by_anchor=positive_counts,
        positive_assignment_fraction_by_anchor=[
            _number(count / max(total_positive, 1)) for count in positive_counts],
        usable_frames_by_anchor=usable_frames,
        dense_best_riou_median_by_anchor=[
            _number(np.median(values)) if values else None
            for values in dense_rious],
        weight_grad_norm_mean_by_anchor=[
            _number(np.mean(values)) if values else None
            for values in gradient_norms],
        bias_grad_abs_mean_by_anchor=[
            _number(np.mean(values)) if values else None
            for values in bias_gradients],
        gt_symmetric_aspect_median=(
            _number(np.median(aspects)) if aspects else None),
        gt_symmetric_aspect_p90=(
            _number(np.percentile(aspects, 90)) if aspects else None))


def source_name(train_cfg, index: int) -> str:
    ann_file = str(train_cfg.get('ann_file', 'source_{}'.format(index)))
    normalized = os.path.normpath(ann_file)
    parent = os.path.basename(os.path.dirname(normalized))
    return parent or os.path.basename(normalized) or 'source_{}'.format(index)


def main():
    args = parse_args()
    source_indexes = validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trajectory_paths = expand_checkpoint_paths(
        args.checkpoint, args.trajectory_checkpoints)
    trajectory = [checkpoint_filter_stats(path) for path in trajectory_paths]

    model, cfg = entry_probe.load_model(
        args.config, args.checkpoint, args.gpu)
    model.train()
    head = entry_probe.get_candidate_head(model, 'main')
    if not hasattr(head, 'retina_cls') or hasattr(head, 'cls_convs'):
        raise RuntimeError('Probe requires single-layer retina_cls main head')
    if int(head.cls_out_channels) != 1:
        raise RuntimeError('Probe currently requires one foreground class')
    # Keep all normalization buffers immutable even though the assigner needs
    # head.training=True to reproduce its training-time path.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    weight_before = head.retina_cls.weight.detach().clone()
    bias_before = (None if head.retina_cls.bias is None else
                   head.retina_cls.bias.detach().clone())
    loss_iter_before = None
    if hasattr(head.loss_cls, '_local_iter'):
        loss_iter_before = head.loss_cls._local_iter.detach().clone()

    from mmdet.datasets import build_dataloader, build_dataset
    from mmcv.parallel import scatter

    train_cfgs = cfg.data.train
    if not isinstance(train_cfgs, (list, tuple)):
        train_cfgs = [train_cfgs]
    if max(source_indexes) >= len(train_cfgs):
        raise ValueError(
            'Requested source index exceeds cfg.data.train length {}'.format(
                len(train_cfgs)))

    source_results = []
    phase_records = []
    for phase in args.assignment_phases:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        phase_record = configure_assignment_phase(head.assigner, phase)
        if loss_iter_before is not None:
            head.loss_cls._local_iter.copy_(loss_iter_before)
        print('\n[assignment phase {}] expected_topk={} start_calls={}'.format(
            phase, phase_record['expected_effective_topk'],
            phase_record['starting_local_call_count']))

        for source_index in source_indexes:
            source_cfg = train_cfgs[source_index]
            dataset = build_dataset(source_cfg)
            loader = build_dataloader(
                dataset, samples_per_gpu=1, workers_per_gpu=0,
                num_gpus=1, dist=False, shuffle=False, seed=args.seed)
            rows = []
            for sample_index, raw_data in enumerate(loader):
                if sample_index >= args.max_samples_per_source:
                    break
                # MMCV 1.x scatter expects CUDA device indices, not
                # torch.device objects.
                data = scatter(raw_data, [int(args.gpu)])[0]
                row = analyze_batch(
                    model, head, data, args.riou_thr,
                    measure_gradient=(
                        sample_index < args.gradient_samples_per_source))
                row['sample_index'] = int(sample_index)
                rows.append(row)
                if (sample_index + 1) % 10 == 0:
                    print('[{} source {}] {}/{} samples'.format(
                        phase, source_index, sample_index + 1,
                        args.max_samples_per_source))
            summary = aggregate_source(rows, head.num_anchors)
            result = dict(
                assignment_phase=phase,
                assigner_state=dict(phase_record),
                ending_local_call_count=int(head.assigner._local_call_count),
                source_index=int(source_index),
                source_name=source_name(source_cfg, source_index),
                data_role='source_train',
                config=dict(
                    ann_file=str(source_cfg.get('ann_file')),
                    img_prefix=str(source_cfg.get('img_prefix'))),
                summary=summary,
                rows=rows)
            source_results.append(result)
            print('[{} source {} {}] positives={} fractions={} usable={}'.format(
                phase, source_index, result['source_name'],
                summary['positive_assignments_by_anchor'],
                summary['positive_assignment_fraction_by_anchor'],
                summary['usable_frames_by_anchor']))
        phase_record['ending_local_call_count'] = int(
            head.assigner._local_call_count)
        phase_records.append(phase_record)

    model.zero_grad(set_to_none=True)
    parameters_unchanged = bool(torch.equal(
        weight_before, head.retina_cls.weight.detach()))
    if bias_before is not None:
        parameters_unchanged = bool(
            parameters_unchanged and torch.equal(
                bias_before, head.retina_cls.bias.detach()))
    if not parameters_unchanged:
        raise RuntimeError(
            'Read-only invariant violated: retina_cls parameters changed')
    current_filters = contribution.classifier_filter_stats(head.retina_cls)
    payload = dict(
        probe='anchor_training_coverage_probe',
        protocol_version=2,
        data_role='source_train_diagnostic',
        source_only=True,
        uses_target_domain=False,
        uses_target_labels=False,
        diagnosis_only=True,
        performs_optimizer_step=False,
        updates_model_parameters=False,
        parameters_verified_unchanged=parameters_unchanged,
        normalization_buffers_frozen=True,
        saves_checkpoint=False,
        eligible_for_training=False,
        eligible_for_checkpoint_selection=False,
        assigner_counter_checkpointed=False,
        assignment_phase_is_synthetic=True,
        config=args.config,
        checkpoint=args.checkpoint,
        parameters=dict(
            source_indexes=source_indexes,
            assignment_phases=list(args.assignment_phases),
            max_samples_per_source=int(args.max_samples_per_source),
            gradient_samples_per_source=int(
                args.gradient_samples_per_source),
            riou_thr=float(args.riou_thr), seed=int(args.seed)),
        current_classifier_filters=current_filters,
        checkpoint_trajectory=trajectory,
        assignment_phase_records=phase_records,
        sources=source_results)
    output_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.out_json, 'w') as handle:
        json.dump(payload, handle, indent=2)

    print('\nANCHOR TRAINING COVERAGE PROBE')
    print('optimizer steps: 0')
    print('checkpoints written: 0')
    print('parameters unchanged: {}'.format(parameters_unchanged))
    print('filter norms: {}'.format([
        item['weight_norm'] for item in current_filters]))
    print('filter biases: {}'.format([
        item['bias'] for item in current_filters]))
    for result in source_results:
        summary = result['summary']
        label = '{}:{}'.format(
            result['assignment_phase'], result['source_name'])
        print('{} positives: {} fractions: {}'.format(
            label,
            summary['positive_assignments_by_anchor'],
            summary['positive_assignment_fraction_by_anchor']))
        print('{} gradient means: {}'.format(
            label,
            summary['weight_grad_norm_mean_by_anchor']))
        print('{} usable frames: {}'.format(
            label, summary['usable_frames_by_anchor']))
    print('[out] wrote {}'.format(os.path.abspath(args.out_json)))
    print('[policy] SOURCE-ONLY READ/GRADIENT DIAGNOSIS; NO OPTIMIZER STEP')


if __name__ == '__main__':
    main()
