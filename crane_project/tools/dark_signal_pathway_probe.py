#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace target-dev signal attenuation from normalized input to cls logits.

This is a labelled target-dev diagnosis.  It compares the highest-scoring
false candidate with the highest-scoring usable (RIoU >= threshold) candidate
at their exact FPN locations.  Outputs are forbidden as training inputs or
checkpoint-selection metrics.
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import candidate_pool_oracle_probe as pool_probe  # noqa: E402
from crane_project.tools import ctx_entry_probe as entry_probe  # noqa: E402


CANONICAL_SPLIT = 'test'
CANONICAL_SEQ = 'real_seq02'
CANONICAL_CORE = set(range(137, 170))
DEFAULT_FRAMES = [150, 155, 164, 167]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Trace target-dev dark-signal attenuation by model stage.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default=CANONICAL_SPLIT)
    parser.add_argument('--seq', default=CANONICAL_SEQ)
    parser.add_argument('--frames', type=int, nargs='+', default=DEFAULT_FRAMES)
    parser.add_argument('--candidate-source', default='main', choices=['main'])
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--false-iou-thr', type=float, default=0.1)
    parser.add_argument('--signal-norm-ratio-thr', type=float, default=0.5,
                        help='Heuristic only: usable/false FPN norm below this '
                             'is tagged as signal attenuation.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def validate_args(args) -> List[int]:
    frames = [int(value) for value in args.frames]
    if args.seed != 0:
        raise ValueError('The unified diagnostic protocol requires --seed 0')
    if args.split != CANONICAL_SPLIT or args.seq != CANONICAL_SEQ:
        raise ValueError('This target-dev probe requires test/real_seq02')
    if len(frames) < 3 or len(frames) > 5 or len(set(frames)) != len(frames):
        raise ValueError('--frames requires 3-5 unique frame ids')
    if any(frame not in CANONICAL_CORE for frame in frames):
        raise ValueError('All frames must be inside real_seq02[137..169]')
    if args.candidate_source != 'main':
        raise ValueError('Only the deployable main candidate head is supported')
    if not 0.0 < args.riou_thr <= 1.0:
        raise ValueError('--riou-thr must be in (0, 1]')
    if not 0.0 <= args.false_iou_thr < args.riou_thr:
        raise ValueError('--false-iou-thr must be below --riou-thr')
    if args.signal_norm_ratio_thr <= 0.0:
        raise ValueError('--signal-norm-ratio-thr must be positive')
    return frames


def _json_number(value: float) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def tensor_stats(value: torch.Tensor, channelwise: bool = False) -> Dict:
    work = value.detach().float()
    result = dict(
        shape=[int(item) for item in work.shape],
        mean=_json_number(work.mean().item()),
        std=_json_number(work.std(unbiased=False).item()),
        rms=_json_number(work.square().mean().sqrt().item()),
        abs_mean=_json_number(work.abs().mean().item()),
        norm=_json_number(work.norm().item()))
    if channelwise and work.ndim == 4:
        channel_mean = work.mean(dim=(0, 2, 3))
        channel_std = work.std(dim=(0, 2, 3), unbiased=False)
        result['channel_mean_abs_mean'] = _json_number(
            channel_mean.abs().mean().item())
        result['channel_std_mean'] = _json_number(channel_std.mean().item())
    return result


def vector_stats(value: torch.Tensor) -> Dict:
    work = value.detach().float().reshape(-1)
    return dict(
        channels=int(work.numel()),
        mean=_json_number(work.mean().item()),
        std=_json_number(work.std(unbiased=False).item()),
        abs_mean=_json_number(work.abs().mean().item()),
        norm=_json_number(work.norm().item()))


def compare_vectors(usable: torch.Tensor, false: torch.Tensor) -> Dict:
    usable = usable.detach().float().reshape(-1)
    false = false.detach().float().reshape(-1)
    if usable.numel() != false.numel():
        return dict(comparable=False)
    cosine = torch.nn.functional.cosine_similarity(
        usable.unsqueeze(0), false.unsqueeze(0), dim=1).item()
    false_norm = float(false.norm().item())
    usable_norm = float(usable.norm().item())
    return dict(
        comparable=True,
        cosine=_json_number(cosine),
        l2_distance=_json_number((usable - false).norm().item()),
        usable_to_false_norm_ratio=_json_number(
            usable_norm / max(false_norm, 1e-12)))


def find_normalize_cfg(cfg) -> Tuple[List[float], List[float], bool]:
    found = []

    def visit(value):
        if isinstance(value, dict):
            if value.get('type') == 'Normalize':
                found.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(cfg.test_pipeline)
    if len(found) != 1:
        raise RuntimeError(
            'Expected exactly one Normalize in test_pipeline, found {}'.format(
                len(found)))
    item = found[0]
    return (list(item['mean']), list(item['std']),
            bool(item.get('to_rgb', True)))


def pre_normalization_stats(img_tensor: torch.Tensor, meta: Dict,
                            mean: Sequence[float], std: Sequence[float]) -> Dict:
    height, width = [int(value) for value in meta['img_shape'][:2]]
    valid = img_tensor[:, :, :height, :width].detach().float()
    mean_tensor = valid.new_tensor(mean).view(1, -1, 1, 1)
    std_tensor = valid.new_tensor(std).view(1, -1, 1, 1)
    restored = valid * std_tensor + mean_tensor
    return dict(
        normalized_valid=tensor_stats(valid, channelwise=True),
        pre_normalization_rgb=tensor_stats(restored, channelwise=True),
        pre_normalization_channel_mean=[
            _json_number(value) for value in
            restored.mean(dim=(0, 2, 3)).tolist()],
        pre_normalization_channel_std=[
            _json_number(value) for value in
            restored.std(dim=(0, 2, 3), unbiased=False).tolist()])


def norm_kind(module: nn.Module) -> Optional[str]:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return 'BatchNorm'
    if isinstance(module, nn.modules.instancenorm._InstanceNorm):
        return 'InstanceNorm'
    if isinstance(module, nn.GroupNorm):
        return 'GroupNorm'
    if isinstance(module, nn.LayerNorm):
        return 'LayerNorm'
    return None


def norm_input_record(name: str, module: nn.Module,
                      input_tensor: torch.Tensor) -> Dict:
    record = dict(
        module=name,
        type=norm_kind(module),
        input=tensor_stats(input_tensor, channelwise=True))
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        work = input_tensor.detach().float()
        channel_mean = work.mean(dim=(0, 2, 3))
        running_mean = module.running_mean.detach().float()
        running_std = (module.running_var.detach().float() + module.eps).sqrt()
        z = (channel_mean - running_mean) / running_std
        record['running_stat_mismatch'] = dict(
            mean_abs_z=_json_number(z.abs().mean().item()),
            max_abs_z=_json_number(z.abs().max().item()),
            batch_mean_l2=_json_number((channel_mean - running_mean).norm().item()))
    return record


class ActivationCapture:
    def __init__(self, model: nn.Module, candidate_head: nn.Module):
        self.norm_records = []
        self.cls_stage_outputs = defaultdict(list)
        self.handles = []
        self.norm_inventory = []
        for name, module in model.named_modules():
            kind = norm_kind(module)
            if kind is None:
                continue
            self.norm_inventory.append(dict(name=name, type=kind))
            self.handles.append(module.register_forward_pre_hook(
                self._make_norm_hook(name)))
        cls_convs = getattr(candidate_head, 'cls_convs', None)
        if cls_convs is not None:
            for index, module in enumerate(cls_convs):
                name = 'cls_convs.{}'.format(index)
                self.handles.append(module.register_forward_hook(
                    self._make_cls_hook(name)))

    def _make_norm_hook(self, name):
        def hook(module, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                self.norm_records.append(
                    norm_input_record(name, module, inputs[0]))
        return hook

    def _make_cls_hook(self, name):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                self.cls_stage_outputs[name].append(output.detach())
        return hook

    def reset(self):
        self.norm_records = []
        self.cls_stage_outputs = defaultdict(list)

    def close(self):
        for handle in self.handles:
            handle.remove()


def stride_xy(candidate_head, level: int) -> Tuple[float, float]:
    stride = candidate_head.anchor_generator.strides[level]
    if isinstance(stride, (list, tuple)):
        return float(stride[0]), float(stride[1])
    return float(stride), float(stride)


def candidate_location(candidate_head, levels: torch.Tensor,
                       anchor_centers: torch.Tensor,
                       candidate_index: int) -> Dict:
    level = int(levels[candidate_index].item())
    center = anchor_centers[candidate_index]
    sx, sy = stride_xy(candidate_head, level)
    col = int(math.floor(float(center[0].item()) / sx))
    row = int(math.floor(float(center[1].item()) / sy))
    return dict(level=level, row=row, col=col,
                anchor_center=[float(center[0].item()), float(center[1].item())],
                stride=[sx, sy])


def local_vector(feature_levels: Sequence[torch.Tensor], location: Dict):
    level = int(location['level'])
    feature = feature_levels[level]
    row = max(0, min(int(location['row']), feature.shape[-2] - 1))
    col = max(0, min(int(location['col']), feature.shape[-1] - 1))
    return feature[0, :, row, col]


def candidate_path(candidate: Dict, features: Sequence[torch.Tensor],
                   cls_stages: Dict[str, Sequence[torch.Tensor]]) -> Dict:
    location = candidate['location']
    fpn_vector = local_vector(features, location)
    stages = []
    for name, outputs in cls_stages.items():
        if int(location['level']) >= len(outputs):
            continue
        vector = local_vector(outputs, location)
        stages.append(dict(name=name, local=vector_stats(vector)))
    return dict(
        candidate_index=int(candidate['index']),
        level=int(location['level']), row=int(location['row']),
        col=int(location['col']), anchor_center=location['anchor_center'],
        score=float(candidate['score']), riou=float(candidate['riou']),
        rank=int(candidate['rank']),
        final_logit=_json_number(math.log(
            max(candidate['score'], 1e-12)
            / max(1.0 - candidate['score'], 1e-12))),
        fpn_local=vector_stats(fpn_vector),
        cls_stages=stages)


def select_candidates(scores: torch.Tensor, ious: torch.Tensor,
                      false_iou_thr: float, riou_thr: float) -> Tuple[Dict, Optional[Dict]]:
    order = torch.argsort(scores, descending=True)
    false_order = order[ious[order] < float(false_iou_thr)]
    if false_order.numel() == 0:
        raise RuntimeError('No hard false candidate in dense pool')
    false_index = int(false_order[0].item())
    usable_order = order[ious[order] >= float(riou_thr)]

    def record(index, rank):
        return dict(index=index, score=float(scores[index].item()),
                    riou=float(ious[index].item()), rank=int(rank))

    false_rank = int(torch.nonzero(order == false_index, as_tuple=False)[0].item()) + 1
    false = record(false_index, false_rank)
    if usable_order.numel() == 0:
        return false, None
    usable_index = int(usable_order[0].item())
    usable_rank = int(torch.nonzero(order == usable_index, as_tuple=False)[0].item()) + 1
    return false, record(usable_index, usable_rank)


def pathway_hint(usable: Optional[Dict], comparison: Optional[Dict],
                 ratio_thr: float) -> Dict:
    if usable is None:
        return dict(
            code='GEOMETRY_MISS',
            classification_attribution_valid=False,
            explanation='No dense RIoU-qualified candidate exists.')
    ratio = comparison.get('usable_to_false_norm_ratio') if comparison else None
    if ratio is not None and ratio < ratio_thr:
        return dict(
            code='SIGNAL_ATTENUATION_SUSPECT',
            classification_attribution_valid=True,
            explanation='Usable local FPN norm is strongly attenuated relative to false top1.')
    return dict(
        code='HEAD_RANKING_SUSPECT',
        classification_attribution_valid=True,
        explanation='Usable geometry reaches FPN without strong local-norm attenuation, but ranks below false top1.')


def analyze_frame(model, candidate_head, capture, transform_compose,
                  img_scale, flip, args, frame, normalization):
    from mmcv.ops import box_iou_rotated

    diag = entry_probe.get_diag()
    img_path, ann_path = diag.find_files(
        args.data_root, args.split, args.seq, frame)
    if img_path is None or ann_path is None:
        raise RuntimeError('Missing target-dev frame {}'.format(frame))
    gts = diag.parse_dota_ann(ann_path)
    if not gts:
        raise RuntimeError('Missing target-dev GT at frame {}'.format(frame))
    img_tensor, meta, image_stats = diag.preprocess_image(
        img_path, transform_compose, img_scale, flip)
    input_stats = pre_normalization_stats(
        img_tensor, meta, normalization[0], normalization[1])
    img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))

    capture.reset()
    with torch.no_grad():
        features = model.extract_feat(img_tensor)
        head, cls_scores, bbox_preds = entry_probe.forward_candidate_head(
            model, features, args.candidate_source)
        boxes, scores, levels, anchor_centers, alignment = (
            entry_probe.flatten_decode_candidates(
                head, cls_scores, bbox_preds, meta['img_shape']))
        scaled_gts = [pool_probe.scale_gt_to_img(gt, meta) for gt in gts]
        gt_boxes = torch.stack([
            entry_probe.gt_to_tensor(gt, boxes.device).reshape(5)
            for gt in scaled_gts])
        ious = box_iou_rotated(
            boxes.float(), gt_boxes.float()).max(dim=1).values
        false, usable = select_candidates(
            scores, ious, args.false_iou_thr, args.riou_thr)
        false['location'] = candidate_location(
            candidate_head, levels, anchor_centers, false['index'])
        false_path = candidate_path(
            false, features, capture.cls_stage_outputs)
        usable_path = None
        comparison = None
        if usable is not None:
            usable['location'] = candidate_location(
                candidate_head, levels, anchor_centers, usable['index'])
            usable_path = candidate_path(
                usable, features, capture.cls_stage_outputs)
            comparison = compare_vectors(
                local_vector(features, usable['location']),
                local_vector(features, false['location']))

    return dict(
        frame=int(frame),
        image=os.path.relpath(img_path, os.path.realpath(args.data_root)),
        raw_image_stats=image_stats,
        input_path=input_stats,
        fpn_levels=[tensor_stats(feature) for feature in features],
        normalization_inputs=capture.norm_records,
        false_top1=false_path,
        usable_candidate=usable_path,
        usable_vs_false_fpn=comparison,
        dense_best_riou=float(ious.max().item()),
        decode_alignment=alignment,
        pathway_hint=pathway_hint(
            usable_path, comparison, args.signal_norm_ratio_thr))


def build_summary(rows: Sequence[Dict], norm_inventory: Sequence[Dict],
                  cls_stage_names: Sequence[str]) -> Dict:
    hint_histogram = {}
    for row in rows:
        code = row['pathway_hint']['code']
        hint_histogram[code] = hint_histogram.get(code, 0) + 1
    norm_histogram = {}
    for item in norm_inventory:
        kind = item['type']
        norm_histogram[kind] = norm_histogram.get(kind, 0) + 1
    usable_rows = [row for row in rows if row['usable_candidate'] is not None]
    return dict(
        frames=len(rows),
        usable_frames=len(usable_rows),
        geometry_miss_frames=[
            row['frame'] for row in rows if row['usable_candidate'] is None],
        pathway_hint_histogram=hint_histogram,
        normalization_layer_histogram=norm_histogram,
        classification_tower_stages=list(cls_stage_names),
        classification_tower_present=bool(cls_stage_names),
        architecture_note=(
            'No cls_convs tower is present; the main classifier consumes FPN '
            'features directly through retina_cls.'
            if not cls_stage_names else
            'cls_convs stages were captured before the final classifier.'),
        verdict_policy=(
            'Hints are attribution diagnostics, not an authorization gate. '
            'Geometry-miss frames cannot support a classification-head verdict.'))


def main():
    args = parse_args()
    frames = validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, cfg = entry_probe.load_model(
        args.config, args.checkpoint, args.gpu)
    diag = entry_probe.get_diag()
    transform_compose, img_scale, flip = diag.build_test_transforms(cfg)
    candidate_head = entry_probe.get_candidate_head(
        model, args.candidate_source)
    normalization = find_normalize_cfg(cfg)
    capture = ActivationCapture(model, candidate_head)
    try:
        rows = []
        for frame in frames:
            row = analyze_frame(
                model, candidate_head, capture, transform_compose,
                img_scale, flip, args, frame, normalization)
            rows.append(row)
            usable = row['usable_candidate']
            print('[{}_{:05d}] dense_iou={:.3f} usable_rank={} hint={}'.format(
                args.seq, frame, row['dense_best_riou'],
                None if usable is None else usable['rank'],
                row['pathway_hint']['code']))
    finally:
        capture.close()

    cls_stage_names = sorted({
        stage['name'] for row in rows
        for candidate_key in ('false_top1', 'usable_candidate')
        if row.get(candidate_key) is not None
        for stage in row[candidate_key].get('cls_stages', [])})
    summary = build_summary(
        rows, capture.norm_inventory, cls_stage_names)
    payload = dict(
        probe='dark_signal_pathway_probe',
        protocol_version=1,
        data_role='target_dev',
        split=args.split,
        seq=args.seq,
        frames=frames,
        reference_only=True,
        diagnosis_only=True,
        uses_target_domain=True,
        uses_target_labels=True,
        eligible_for_training=False,
        eligible_for_checkpoint_selection=False,
        must_not_export_target_features_to_training=True,
        protocol_ready_for_p1_a=False,
        config=args.config,
        checkpoint=args.checkpoint,
        parameters=dict(
            candidate_source=args.candidate_source,
            riou_thr=float(args.riou_thr),
            false_iou_thr=float(args.false_iou_thr),
            signal_norm_ratio_thr=float(args.signal_norm_ratio_thr),
            heuristic_threshold_target_informed=True),
        normalization=dict(
            mean=normalization[0], std=normalization[1],
            to_rgb=normalization[2]),
        normalization_inventory=capture.norm_inventory,
        summary=summary,
        rows=rows)
    output_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.out_json, 'w') as handle:
        json.dump(payload, handle, indent=2)

    print('\nDARK SIGNAL PATHWAY PROBE')
    print('data:       target-dev {}/{} {}'.format(
        args.split, args.seq, frames))
    print('norms:      {}'.format(summary['normalization_layer_histogram']))
    print('cls tower:  {}'.format(summary['classification_tower_stages']))
    print('geometry misses: {}'.format(summary['geometry_miss_frames']))
    print('hints:      {}'.format(summary['pathway_hint_histogram']))
    print('[out] wrote {}'.format(os.path.abspath(args.out_json)))
    print('[policy] TARGET-DEV REFERENCE-ONLY DIAGNOSIS; OUTPUT MUST NOT ENTER TRAINING')


if __name__ == '__main__':
    main()
