#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exactly decompose the single-layer ``retina_cls`` candidate logits.

The target rows are labelled target-dev diagnostics.  Source-validation rows
are controls only.  No FPN patches are serialized, and no output from this
probe is eligible for training or checkpoint selection.
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import candidate_pool_oracle_probe as pool_probe  # noqa: E402
from crane_project.tools import ctx_entry_probe as entry_probe  # noqa: E402
from crane_project.tools import dark_signal_pathway_probe as pathway  # noqa: E402


TARGET_SPLIT = 'test'
TARGET_SEQ = 'real_seq02'
TARGET_ALLOWED = set(range(137, 170))
SOURCE_SPLIT = 'val'
SOURCE_SEQ = 'real_seq07'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Exact retina_cls channel and spatial contribution probe.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--target-frames', type=int, nargs='+', default=[150, 155])
    parser.add_argument('--source-frames', type=int, nargs='+', default=[104, 105])
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--false-iou-thr', type=float, default=0.1)
    parser.add_argument('--top-channels', type=int, default=20)
    parser.add_argument('--reconstruction-atol', type=float, default=1e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def validate_args(args) -> Tuple[List[int], List[int]]:
    target = [int(value) for value in args.target_frames]
    source = [int(value) for value in args.source_frames]
    if args.seed != 0:
        raise ValueError('The unified diagnostic protocol requires --seed 0')
    if not 1 <= len(target) <= 3 or len(set(target)) != len(target):
        raise ValueError('--target-frames requires 1-3 unique frames')
    if any(frame not in TARGET_ALLOWED for frame in target):
        raise ValueError('Target frames must be inside real_seq02[137..169]')
    if not 1 <= len(source) <= 3 or len(set(source)) != len(source):
        raise ValueError('--source-frames requires 1-3 unique frames')
    if not 0.0 <= args.false_iou_thr < args.riou_thr <= 1.0:
        raise ValueError('Require 0 <= false-iou-thr < riou-thr <= 1')
    if not 1 <= args.top_channels <= 256:
        raise ValueError('--top-channels must be in [1, 256]')
    if args.reconstruction_atol <= 0.0:
        raise ValueError('--reconstruction-atol must be positive')
    return target, source


def candidate_layout(cls_scores: Sequence[torch.Tensor],
                     candidate_head, img_shape) -> List[Dict]:
    """Return exact raw output coordinates in flattened candidate order."""
    rows = []
    for level, score in enumerate(cls_scores):
        if score.shape[0] != 1:
            raise RuntimeError('Contribution probe requires batch size 1')
        channels, height, width = [int(value) for value in score.shape[1:]]
        classes = int(candidate_head.cls_out_channels)
        if channels % classes != 0:
            raise RuntimeError('Classification channels do not divide classes')
        anchors = channels // classes
        expected = int(candidate_head.anchor_generator.num_base_anchors[level])
        if anchors != expected:
            raise RuntimeError(
                'Output/anchor mismatch at level {}: {} vs {}'.format(
                    level, anchors, expected))
        raw = torch.arange(
            height * width * anchors, device=score.device, dtype=torch.long)
        if bool(getattr(candidate_head, 'filter_padding_anchors', False)):
            priors = candidate_head.anchor_generator.grid_priors(
                [item.shape[-2:] for item in cls_scores],
                device=score.device)[level]
            img_h, img_w = img_shape[:2]
            keep = ((priors[:, 0] >= 0) & (priors[:, 1] >= 0)
                    & (priors[:, 0] < img_w) & (priors[:, 1] < img_h))
            raw = raw[keep]
        for raw_index in raw.tolist():
            location = raw_index // anchors
            anchor_id = raw_index % anchors
            row = location // width
            col = location % width
            rows.append(dict(
                level=int(level), row=int(row), col=int(col),
                anchor_id=int(anchor_id),
                output_channel=int(anchor_id * classes),
                raw_level_index=int(raw_index)))
    return rows


def _number(value) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def exact_contributions(feature: torch.Tensor, conv, location: Dict,
                        actual_logit: float, actual_score: float,
                        top_channels: int, atol: float) -> Dict:
    if conv.kernel_size != (3, 3) or conv.padding != (1, 1):
        raise RuntimeError(
            'Expected retina_cls Conv2d kernel=3 padding=1, got {} {}'.format(
                conv.kernel_size, conv.padding))
    level_row = int(location['row'])
    level_col = int(location['col'])
    output_channel = int(location['output_channel'])
    padded = F.pad(feature[0], (1, 1, 1, 1))
    patch = padded[:, level_row:level_row + 3,
                   level_col:level_col + 3].detach().float()
    weight = conv.weight[output_channel].detach().float()
    per_channel = (patch * weight).sum(dim=(1, 2))
    bias = (0.0 if conv.bias is None
            else float(conv.bias[output_channel].detach().item()))
    contribution_sum = float(per_channel.sum().item())
    reconstructed = bias + contribution_sum
    error = abs(reconstructed - float(actual_logit))
    score_reconstructed = float(torch.sigmoid(
        torch.tensor(reconstructed)).item())
    score_error = abs(score_reconstructed - float(actual_score))
    if error > float(atol):
        raise RuntimeError(
            'retina_cls reconstruction failed: error={:.8g} > atol={:.8g}; '
            'candidate channel mapping is invalid'.format(error, atol))

    positive = per_channel.clamp_min(0)
    negative = per_channel.clamp_max(0)
    flat_patch = patch.reshape(-1)
    flat_weight = weight.reshape(-1)
    cosine = F.cosine_similarity(
        flat_patch[None], flat_weight[None], dim=1).item()

    def top_records(values: torch.Tensor, largest: bool):
        count = min(int(top_channels), int(values.numel()))
        _, indices = torch.topk(values, k=count, largest=largest, sorted=True)
        return [dict(
            channel=int(index.item()),
            contribution=_number(per_channel[index].item()),
            patch_norm=_number(patch[index].norm().item()),
            weight_norm=_number(weight[index].norm().item()))
            for index in indices]

    return dict(
        reconstruction_passed=True,
        actual_logit=_number(actual_logit),
        reconstructed_logit=_number(reconstructed),
        reconstruction_abs_error=_number(error),
        actual_score=_number(actual_score),
        reconstructed_score=_number(score_reconstructed),
        score_abs_error=_number(score_error),
        bias=_number(bias),
        contribution_sum=_number(contribution_sum),
        positive_contribution_sum=_number(positive.sum().item()),
        negative_contribution_sum=_number(negative.sum().item()),
        positive_channel_count=int((per_channel > 0).sum().item()),
        negative_channel_count=int((per_channel < 0).sum().item()),
        patch_norm=_number(patch.norm().item()),
        weight_norm=_number(weight.norm().item()),
        patch_weight_cosine=_number(cosine),
        top_positive_channels=top_records(per_channel, True),
        top_negative_channels=top_records(per_channel, False),
        per_channel=[_number(value) for value in per_channel.tolist()])


def candidate_record(candidate: Dict, layout: Sequence[Dict],
                     features: Sequence[torch.Tensor],
                     cls_scores: Sequence[torch.Tensor], conv,
                     top_channels: int, atol: float) -> Dict:
    index = int(candidate['index'])
    location = dict(layout[index])
    actual_logit = float(cls_scores[location['level']][
        0, location['output_channel'], location['row'], location['col']].item())
    decomposition = exact_contributions(
        features[location['level']], conv, location,
        actual_logit, candidate['score'], top_channels, atol)
    return dict(
        candidate_index=index,
        score=float(candidate['score']),
        riou=float(candidate['riou']),
        rank=int(candidate['rank']),
        location=location,
        decomposition=decomposition)


def _rank_map(scores: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(
        1, int(order.numel()) + 1, device=order.device, dtype=order.dtype)
    return ranks


def select_per_anchor_candidates(scores: torch.Tensor, ious: torch.Tensor,
                                 layout: Sequence[Dict], num_anchors: int,
                                 riou_thr: float) -> List[Dict]:
    """Select score and geometry representatives independently per anchor."""
    if len(layout) != int(scores.numel()):
        raise RuntimeError('Per-anchor selection layout mismatch')
    anchor_ids = torch.tensor(
        [int(item['anchor_id']) for item in layout],
        device=scores.device, dtype=torch.long)
    ranks = _rank_map(scores)
    results = []
    for anchor_id in range(int(num_anchors)):
        indices = torch.nonzero(
            anchor_ids == anchor_id, as_tuple=False).reshape(-1)
        if indices.numel() == 0:
            results.append(dict(
                anchor_id=anchor_id, candidate_count=0,
                highest_score=None, dense_best_geometry=None,
                best_usable_by_score=None))
            continue

        anchor_scores = scores[indices]
        anchor_ious = ious[indices]
        highest_index = int(indices[torch.argmax(anchor_scores)].item())
        geometry_index = int(indices[torch.argmax(anchor_ious)].item())
        usable_indices = indices[anchor_ious >= float(riou_thr)]
        usable_index = None
        if usable_indices.numel() > 0:
            usable_index = int(usable_indices[
                torch.argmax(scores[usable_indices])].item())

        def item(index):
            if index is None:
                return None
            return dict(
                index=int(index), score=float(scores[index].item()),
                riou=float(ious[index].item()),
                rank=int(ranks[index].item()))

        results.append(dict(
            anchor_id=int(anchor_id),
            candidate_count=int(indices.numel()),
            highest_score=item(highest_index),
            dense_best_geometry=item(geometry_index),
            best_usable_by_score=item(usable_index)))
    return results


def decompose_per_anchor(selections: Sequence[Dict], layout: Sequence[Dict],
                         features: Sequence[torch.Tensor],
                         cls_scores: Sequence[torch.Tensor], conv,
                         top_channels: int, atol: float) -> List[Dict]:
    rows = []
    for selection in selections:
        row = dict(
            anchor_id=int(selection['anchor_id']),
            candidate_count=int(selection['candidate_count']))
        cache = {}
        for key in ('highest_score', 'dense_best_geometry',
                    'best_usable_by_score'):
            candidate = selection.get(key)
            if candidate is None:
                row[key] = None
                continue
            index = int(candidate['index'])
            if index not in cache:
                cache[index] = candidate_record(
                    candidate, layout, features, cls_scores, conv,
                    top_channels, atol)
            row[key] = cache[index]
        rows.append(row)
    return rows


def classifier_filter_stats(conv) -> List[Dict]:
    rows = []
    weights = conv.weight.detach().float()
    for anchor_id in range(int(weights.shape[0])):
        weight = weights[anchor_id]
        bias = (None if conv.bias is None
                else float(conv.bias[anchor_id].detach().item()))
        rows.append(dict(
            anchor_id=int(anchor_id),
            output_channel=int(anchor_id),
            weight_shape=[int(value) for value in weight.shape],
            weight_norm=_number(weight.norm().item()),
            weight_abs_mean=_number(weight.abs().mean().item()),
            weight_std=_number(weight.std(unbiased=False).item()),
            bias=_number(bias) if bias is not None else None))
    reference_norm = max(float(rows[0]['weight_norm']), 1e-12)
    for row in rows:
        row['weight_norm_relative_to_anchor0'] = _number(
            float(row['weight_norm']) / reference_norm)
    for row in rows:
        first = weights[row['anchor_id']].reshape(-1)
        row['weight_cosine_to_anchor0'] = _number(
            F.cosine_similarity(first[None], weights[0].reshape(1, -1),
                                dim=1).item())
    return rows


def compare_channel_contributions(first: Dict, second: Dict,
                                  top_channels: int) -> Dict:
    first_values = torch.tensor(first['decomposition']['per_channel'])
    second_values = torch.tensor(second['decomposition']['per_channel'])
    delta = first_values - second_values
    count = min(int(top_channels), int(delta.numel()))

    def records(largest):
        _, indices = torch.topk(delta, k=count, largest=largest, sorted=True)
        return [dict(
            channel=int(index.item()),
            delta=_number(delta[index].item()),
            first=_number(first_values[index].item()),
            second=_number(second_values[index].item()))
            for index in indices]

    same_filter = (
        first['location']['anchor_id'] == second['location']['anchor_id'])
    return dict(
        same_anchor_filter=bool(same_filter),
        interpretation_valid_without_anchor_confound=bool(same_filter),
        cosine=_number(F.cosine_similarity(
            first_values[None], second_values[None], dim=1).item()),
        top_first_minus_second=records(True),
        bottom_first_minus_second=records(False))


def analyze_frame(model, candidate_head, transforms, img_scale, flip,
                  args, split: str, seq: str, frame: int,
                  role: str) -> Dict:
    from mmcv.ops import box_iou_rotated

    diag = entry_probe.get_diag()
    img_path, ann_path = diag.find_files(
        args.data_root, split, seq, frame)
    if img_path is None or ann_path is None:
        raise RuntimeError('Missing {}/{} frame {}'.format(split, seq, frame))
    gts = diag.parse_dota_ann(ann_path)
    if not gts:
        raise RuntimeError('Missing labelled GT for {}/{} {}'.format(
            split, seq, frame))
    img_tensor, meta, image_stats = diag.preprocess_image(
        img_path, transforms, img_scale, flip)
    img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))

    with torch.no_grad():
        features = model.extract_feat(img_tensor)
        head, cls_scores, bbox_preds = entry_probe.forward_candidate_head(
            model, features, 'main')
        boxes, scores, _levels, _centers, alignment = (
            entry_probe.flatten_decode_candidates(
                head, cls_scores, bbox_preds, meta['img_shape']))
        layout = candidate_layout(cls_scores, head, meta['img_shape'])
        if len(layout) != int(scores.numel()):
            raise RuntimeError(
                'Candidate layout mismatch: {} vs {}'.format(
                    len(layout), scores.numel()))
        scaled_gts = [pool_probe.scale_gt_to_img(gt, meta) for gt in gts]
        gt_boxes = torch.stack([
            entry_probe.gt_to_tensor(gt, boxes.device).reshape(5)
            for gt in scaled_gts])
        ious = box_iou_rotated(
            boxes.float(), gt_boxes.float()).max(dim=1).values
        false, usable = pathway.select_candidates(
            scores, ious, args.false_iou_thr, args.riou_thr)
        false_record = candidate_record(
            false, layout, features, cls_scores, head.retina_cls,
            args.top_channels, args.reconstruction_atol)
        usable_record = None
        usable_vs_false = None
        if usable is not None:
            usable_record = candidate_record(
                usable, layout, features, cls_scores, head.retina_cls,
                args.top_channels, args.reconstruction_atol)
            usable_vs_false = compare_channel_contributions(
                usable_record, false_record, args.top_channels)
        per_anchor_selected = select_per_anchor_candidates(
            scores, ious, layout, candidate_head.num_anchors,
            args.riou_thr)
        per_anchor = decompose_per_anchor(
            per_anchor_selected, layout, features, cls_scores,
            head.retina_cls, args.top_channels, args.reconstruction_atol)

    return dict(
        role=role,
        split=split,
        seq=seq,
        frame=int(frame),
        image=os.path.relpath(img_path, os.path.realpath(args.data_root)),
        image_stats=image_stats,
        dense_best_riou=float(ious.max().item()),
        false_candidate=false_record,
        usable_candidate=usable_record,
        usable_vs_false=usable_vs_false,
        per_anchor=per_anchor,
        decode_alignment=alignment)


def control_comparisons(target_rows: Sequence[Dict],
                        source_rows: Sequence[Dict], top_channels: int) -> List[Dict]:
    comparisons = []
    for target in target_rows:
        if target['usable_candidate'] is None:
            continue
        for source in source_rows:
            if source['usable_candidate'] is None:
                continue
            detail = compare_channel_contributions(
                target['usable_candidate'], source['usable_candidate'],
                top_channels)
            comparisons.append(dict(
                target_frame=int(target['frame']),
                source_frame=int(source['frame']),
                target_anchor_id=int(
                    target['usable_candidate']['location']['anchor_id']),
                source_anchor_id=int(
                    source['usable_candidate']['location']['anchor_id']),
                comparison=detail))
    return comparisons


def false_source_control_comparisons(target_rows: Sequence[Dict],
                                     source_rows: Sequence[Dict],
                                     top_channels: int) -> List[Dict]:
    """Compare target false candidates with source usable candidates.

    When the target false and source usable share the same anchor_id, the
    comparison is free of anchor confound.  This is valuable because target
    false candidates may use a different anchor than target usable ones.
    """
    comparisons = []
    for target in target_rows:
        target_false = target['false_candidate']
        for source in source_rows:
            source_usable = source['usable_candidate']
            if source_usable is None:
                continue
            detail = compare_channel_contributions(
                target_false, source_usable, top_channels)
            comparisons.append(dict(
                target_frame=int(target['frame']),
                source_frame=int(source['frame']),
                target_anchor_id=int(
                    target_false['location']['anchor_id']),
                source_anchor_id=int(
                    source_usable['location']['anchor_id']),
                target_candidate_type='false',
                source_candidate_type='usable',
                comparison=detail))
    return comparisons


def cross_frame_consistency(target_rows: Sequence[Dict],
                            top_channels: int) -> List[Dict]:
    """Analyse consistency of top contributing channels across target frames.

    For each anchor_id, collect candidates of the same type across frames and
    report whether the same channels dominate the contribution pattern.
    """
    by_anchor: Dict[int, list] = {}
    for row in target_rows:
        for candidate_type in ('usable_candidate', 'false_candidate'):
            candidate = row.get(candidate_type)
            if candidate is None:
                continue
            anchor_id = int(candidate['location']['anchor_id'])
            by_anchor.setdefault(anchor_id, []).append(dict(
                frame=int(row['frame']),
                candidate_type=candidate_type,
                candidate=candidate))

    results = []
    for anchor_id, entries in sorted(by_anchor.items()):
        if len(entries) < 2:
            continue
        top_pos_sets = []
        top_neg_sets = []
        for entry in entries:
            decomp = entry['candidate']['decomposition']
            pos_channels = set(
                c['channel'] for c in
                decomp['top_positive_channels'][:top_channels])
            neg_channels = set(
                c['channel'] for c in
                decomp['top_negative_channels'][:top_channels])
            top_pos_sets.append(pos_channels)
            top_neg_sets.append(neg_channels)

        pos_intersection = (
            set.intersection(*top_pos_sets) if top_pos_sets else set())
        neg_intersection = (
            set.intersection(*top_neg_sets) if top_neg_sets else set())
        pos_union = (
            set.union(*top_pos_sets) if top_pos_sets else set())
        neg_union = (
            set.union(*top_neg_sets) if top_neg_sets else set())

        results.append(dict(
            anchor_id=int(anchor_id),
            frame_count=len(entries),
            frames=[e['frame'] for e in entries],
            candidate_types=[e['candidate_type'] for e in entries],
            top_positive_overlap_count=len(pos_intersection),
            top_positive_overlap_ratio=(
                len(pos_intersection) / len(pos_union)
                if pos_union else 0.0),
            top_negative_overlap_count=len(neg_intersection),
            top_negative_overlap_ratio=(
                len(neg_intersection) / len(neg_union)
                if neg_union else 0.0),
            consistent_positive_channels=sorted(pos_intersection),
            consistent_negative_channels=sorted(neg_intersection)))
    return results


def build_decomposition_table(target_rows: Sequence[Dict],
                              source_rows: Sequence[Dict]) -> List[Dict]:
    """Compact one-row-per-candidate summary for quick scanning."""
    rows = []
    for row in list(target_rows) + list(source_rows):
        role = row['role']
        frame = int(row['frame'])
        for candidate_type, candidate in [
                ('false', row['false_candidate']),
                ('usable', row['usable_candidate'])]:
            if candidate is None:
                continue
            d = candidate['decomposition']
            loc = candidate['location']
            rows.append(dict(
                role=role,
                frame=frame,
                candidate_type=candidate_type,
                rank=int(candidate['rank']),
                anchor_id=int(loc['anchor_id']),
                level=int(loc['level']),
                score=float(candidate['score']),
                logit=float(d['actual_logit']),
                bias=float(d['bias']),
                contribution_sum=float(d['contribution_sum']),
                positive_contribution_sum=float(
                    d['positive_contribution_sum']),
                negative_contribution_sum=float(
                    d['negative_contribution_sum']),
                positive_channel_count=int(d['positive_channel_count']),
                negative_channel_count=int(d['negative_channel_count']),
                patch_norm=float(d['patch_norm']),
                weight_norm=float(d['weight_norm']),
                patch_weight_cosine=float(d['patch_weight_cosine'])))
    return rows


def per_anchor_control_comparisons(target_rows: Sequence[Dict],
                                   source_rows: Sequence[Dict],
                                   top_channels: int) -> List[Dict]:
    comparisons = []
    for target in target_rows:
        target_by_anchor = {
            int(item['anchor_id']): item for item in target['per_anchor']}
        for source in source_rows:
            source_by_anchor = {
                int(item['anchor_id']): item for item in source['per_anchor']}
            for anchor_id in sorted(set(target_by_anchor) & set(source_by_anchor)):
                target_candidate = target_by_anchor[anchor_id].get(
                    'best_usable_by_score')
                source_candidate = source_by_anchor[anchor_id].get(
                    'best_usable_by_score')
                if target_candidate is None or source_candidate is None:
                    continue
                comparison = compare_channel_contributions(
                    target_candidate, source_candidate, top_channels)
                comparisons.append(dict(
                    anchor_id=int(anchor_id),
                    target_frame=int(target['frame']),
                    source_frame=int(source['frame']),
                    target_rank=int(target_candidate['rank']),
                    source_rank=int(source_candidate['rank']),
                    comparison=comparison))
    return comparisons


def build_summary(target_rows: Sequence[Dict], source_rows: Sequence[Dict],
                  comparisons: Sequence[Dict],
                  per_anchor_comparisons: Sequence[Dict],
                  false_source_comparisons: Sequence[Dict],
                  consistency: Sequence[Dict],
                  filter_stats: Sequence[Dict],
                  top_channels: int) -> Dict:
    all_rows = list(target_rows) + list(source_rows)
    reconstruction_errors = [
        candidate['decomposition']['reconstruction_abs_error']
        for row in all_rows
        for candidate in (row['false_candidate'], row['usable_candidate'])
        if candidate is not None]
    valid_controls = [
        item for item in comparisons
        if item['comparison']['same_anchor_filter']]
    valid_false_source = [
        item for item in false_source_comparisons
        if item['comparison']['same_anchor_filter']]

    # Collect target false anchor_ids for quick diagnostics
    target_false_anchors = [
        int(row['false_candidate']['location']['anchor_id'])
        for row in target_rows]
    target_usable_anchors = [
        int(row['usable_candidate']['location']['anchor_id'])
        for row in target_rows if row['usable_candidate'] is not None]
    source_usable_anchors = [
        int(row['usable_candidate']['location']['anchor_id'])
        for row in source_rows if row['usable_candidate'] is not None]

    # Extract consistent channels from cross-frame analysis
    consistent_neg = []
    consistent_pos = []
    for entry in consistency:
        if entry['candidate_types'].count('usable_candidate') >= 2:
            consistent_neg = entry['consistent_negative_channels']
            consistent_pos = entry['consistent_positive_channels']

    return dict(
        target_frames=[int(row['frame']) for row in target_rows],
        source_control_frames=[int(row['frame']) for row in source_rows],
        target_usable_ranks=[
            None if row['usable_candidate'] is None
            else int(row['usable_candidate']['rank'])
            for row in target_rows],
        source_usable_ranks=[
            None if row['usable_candidate'] is None
            else int(row['usable_candidate']['rank'])
            for row in source_rows],
        target_false_anchors=target_false_anchors,
        target_usable_anchors=target_usable_anchors,
        source_usable_anchors=source_usable_anchors,
        max_reconstruction_abs_error=(
            max(reconstruction_errors) if reconstruction_errors else None),
        exact_reconstruction_passed=bool(reconstruction_errors),
        source_target_control_pairs=len(comparisons),
        same_anchor_control_pairs=len(valid_controls),
        false_source_control_pairs=len(false_source_comparisons),
        false_source_same_anchor_pairs=len(valid_false_source),
        constrained_same_anchor_control_pairs=len(per_anchor_comparisons),
        cross_frame_consistency_groups=len(consistency),
        consistent_negative_channels_across_target_usable=consistent_neg,
        consistent_positive_channels_across_target_usable=consistent_pos,
        source_usable_frames_per_anchor={
            str(anchor_id): sum(
                row['per_anchor'][anchor_id]['best_usable_by_score'] is not None
                for row in source_rows)
            for anchor_id in range(len(filter_stats))},
        target_usable_frames_per_anchor={
            str(anchor_id): sum(
                row['per_anchor'][anchor_id]['best_usable_by_score'] is not None
                for row in target_rows)
            for anchor_id in range(len(filter_stats))},
        classifier_filter_weight_norms=[
            float(item['weight_norm']) for item in filter_stats],
        classifier_filter_biases=[item['bias'] for item in filter_stats],
        interpretation_policy=(
            'Use false-source same-anchor comparisons and constrained '
            'per-anchor source-target comparisons first. Different-anchor '
            'global pairs are descriptive because each anchor has distinct '
            'weights.'))


def main():
    args = parse_args()
    target_frames, source_frames = validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, cfg = entry_probe.load_model(
        args.config, args.checkpoint, args.gpu)
    diag = entry_probe.get_diag()
    transforms, img_scale, flip = diag.build_test_transforms(cfg)
    candidate_head = entry_probe.get_candidate_head(model, 'main')
    if not hasattr(candidate_head, 'retina_cls'):
        raise RuntimeError('Main candidate head has no retina_cls')
    if hasattr(candidate_head, 'cls_convs'):
        raise RuntimeError(
            'This exact single-layer probe is invalid when cls_convs exists')
    if int(candidate_head.cls_out_channels) != 1:
        raise RuntimeError('This probe currently requires one foreground class')
    filter_stats = classifier_filter_stats(candidate_head.retina_cls)

    target_rows = []
    for frame in target_frames:
        row = analyze_frame(
            model, candidate_head, transforms, img_scale, flip, args,
            TARGET_SPLIT, TARGET_SEQ, frame, 'target_dev')
        target_rows.append(row)
        print('[target {}] usable_rank={} anchor={} recon_err={:.3g}'.format(
            frame,
            None if row['usable_candidate'] is None
            else row['usable_candidate']['rank'],
            None if row['usable_candidate'] is None
            else row['usable_candidate']['location']['anchor_id'],
            row['false_candidate']['decomposition'][
                'reconstruction_abs_error']))

    source_rows = []
    for frame in source_frames:
        row = analyze_frame(
            model, candidate_head, transforms, img_scale, flip, args,
            SOURCE_SPLIT, SOURCE_SEQ, frame, 'source_val_control')
        source_rows.append(row)
        print('[source {}] usable_rank={} anchor={} recon_err={:.3g}'.format(
            frame,
            None if row['usable_candidate'] is None
            else row['usable_candidate']['rank'],
            None if row['usable_candidate'] is None
            else row['usable_candidate']['location']['anchor_id'],
            row['false_candidate']['decomposition'][
                'reconstruction_abs_error']))

    comparisons = control_comparisons(
        target_rows, source_rows, args.top_channels)
    constrained_comparisons = per_anchor_control_comparisons(
        target_rows, source_rows, args.top_channels)
    false_source_comps = false_source_control_comparisons(
        target_rows, source_rows, args.top_channels)
    consistency = cross_frame_consistency(target_rows, args.top_channels)
    decomp_table = build_decomposition_table(target_rows, source_rows)
    summary = build_summary(
        target_rows, source_rows, comparisons,
        constrained_comparisons, false_source_comps,
        consistency, filter_stats, args.top_channels)
    payload = dict(
        probe='retina_cls_contribution_probe',
        protocol_version=2,
        architecture='FPN -> retina_cls Conv2d(3x3) -> logits',
        target_data_role='target_dev',
        source_data_role='source_val_control',
        diagnosis_only=True,
        reference_only=True,
        uses_target_domain=True,
        uses_target_labels=True,
        eligible_for_training=False,
        eligible_for_checkpoint_selection=False,
        exports_raw_target_features=False,
        must_not_export_target_features_or_contributions_to_training=True,
        protocol_ready_for_p1_a=False,
        config=args.config,
        checkpoint=args.checkpoint,
        parameters=dict(
            riou_thr=float(args.riou_thr),
            false_iou_thr=float(args.false_iou_thr),
            top_channels=int(args.top_channels),
            reconstruction_atol=float(args.reconstruction_atol)),
        classifier_filters=filter_stats,
        summary=summary,
        decomposition_table=decomp_table,
        target_rows=target_rows,
        source_control_rows=source_rows,
        target_source_control_comparisons=comparisons,
        false_source_control_comparisons=false_source_comps,
        constrained_per_anchor_control_comparisons=constrained_comparisons,
        cross_frame_consistency=consistency)
    output_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.out_json, 'w') as handle:
        json.dump(payload, handle, indent=2)

    print('\nRETINA_CLS CONTRIBUTION PROBE v2')
    print('max reconstruction error: {}'.format(
        summary['max_reconstruction_abs_error']))
    print('target usable ranks: {}'.format(summary['target_usable_ranks']))
    print('source usable ranks: {}'.format(summary['source_usable_ranks']))
    print('target false anchors: {}'.format(summary['target_false_anchors']))
    print('target usable anchors: {}'.format(summary['target_usable_anchors']))
    print('source usable anchors: {}'.format(summary['source_usable_anchors']))
    print('same-anchor controls (usable-vs-usable): {}/{}'.format(
        summary['same_anchor_control_pairs'],
        summary['source_target_control_pairs']))
    print('false-source same-anchor controls: {}/{}'.format(
        summary['false_source_same_anchor_pairs'],
        summary['false_source_control_pairs']))
    print('constrained same-anchor controls: {}'.format(
        summary['constrained_same_anchor_control_pairs']))
    print('cross-frame consistency groups: {}'.format(
        summary['cross_frame_consistency_groups']))
    if summary['consistent_negative_channels_across_target_usable']:
        print('consistent neg channels (target usable): {}'.format(
            summary['consistent_negative_channels_across_target_usable']))
    if summary['consistent_positive_channels_across_target_usable']:
        print('consistent pos channels (target usable): {}'.format(
            summary['consistent_positive_channels_across_target_usable']))
    print('source usable frames/anchor: {}'.format(
        summary['source_usable_frames_per_anchor']))
    print('target usable frames/anchor: {}'.format(
        summary['target_usable_frames_per_anchor']))
    print('filter weight norms: {}'.format(
        summary['classifier_filter_weight_norms']))
    print('filter biases: {}'.format(
        summary['classifier_filter_biases']))

    # Print decomposition table
    print('\nDECOMPOSITION TABLE:')
    print('{:>5s} {:>5s} {:>6s} {:>5s} {:>3s} {:>5s} {:>8s} {:>8s} '
          '{:>8s} {:>8s} {:>8s} {:>5s} {:>5s} {:>8s} {:>8s} {:>8s}'.format(
              'role', 'frame', 'type', 'rank', 'anc', 'lvl', 'score',
              'logit', 'bias', 'contrib', 'pos_sum', 'pos_c', 'neg_c',
              'p_norm', 'w_norm', 'cos_pw'))
    for r in decomp_table:
        print('{:>5s} {:>5d} {:>6s} {:>5d} {:>3d} {:>5d} {:>8.6f} {:>8.4f} '
              '{:>8.4f} {:>8.4f} {:>8.4f} {:>5d} {:>5d} {:>8.4f} {:>8.4f} '
              '{:>8.4f}'.format(
                  r['role'][:5], r['frame'], r['candidate_type'][:6],
                  r['rank'], r['anchor_id'], r['level'], r['score'],
                  r['logit'], r['bias'], r['contribution_sum'],
                  r['positive_contribution_sum'], r['positive_channel_count'],
                  r['negative_channel_count'], r['patch_norm'],
                  r['weight_norm'], r['patch_weight_cosine']))

    # Print false-source same-anchor details
    valid_false_source = [
        item for item in false_source_comps
        if item['comparison']['same_anchor_filter']]
    if valid_false_source:
        print('\nFALSE-SOURCE SAME-ANCHOR COMPARISONS:')
        for item in valid_false_source:
            c = item['comparison']
            print('  t{}(false,a{}) vs s{}(usable,a{}): cos={:.4f}'.format(
                item['target_frame'], item['target_anchor_id'],
                item['source_frame'], item['source_anchor_id'],
                c['cosine']))
            print('    top3 delta(t-s): {}'.format(
                [(d['channel'], round(d['delta'], 4))
                 for d in c['top_first_minus_second'][:3]]))

    # Print cross-frame consistency
    if consistency:
        print('\nCROSS-FRAME CONSISTENCY:')
        for entry in consistency:
            print('  anchor={}: frames={} types={} pos_overlap={}/{} '
                  'neg_overlap={}/{}'.format(
                      entry['anchor_id'], entry['frames'],
                      entry['candidate_types'],
                      entry['top_positive_overlap_count'],
                      len(entry['consistent_positive_channels']) +
                      (entry['top_positive_overlap_count'] or 0),
                      entry['top_negative_overlap_count'],
                      len(entry['consistent_negative_channels']) +
                      (entry['top_negative_overlap_count'] or 0)))
            if entry['consistent_negative_channels']:
                print('    consistent neg: {}'.format(
                    entry['consistent_negative_channels']))
            if entry['consistent_positive_channels']:
                print('    consistent pos: {}'.format(
                    entry['consistent_positive_channels']))

    print('[out] wrote {}'.format(os.path.abspath(args.out_json)))
    print('[policy] TARGET-DEV ATTRIBUTION ONLY; OUTPUT MUST NOT ENTER TRAINING')


if __name__ == '__main__':
    main()
