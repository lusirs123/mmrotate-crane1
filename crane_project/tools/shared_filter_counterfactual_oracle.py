#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-training shared-filter counterfactual on target-dev.

The original bbox regression outputs are preserved exactly.  The
counterfactual copies the trained anchor-0 classification logit to every
anchor at the same FPN location.  Target labels are used only after scoring to
measure geometry and must never enter training or parameter selection.
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import candidate_pool_oracle_probe as pool_probe  # noqa: E402
from crane_project.tools import ctx_entry_probe as entry_probe  # noqa: E402
from crane_project.tools import retina_cls_contribution_probe as contribution  # noqa: E402


CANONICAL_SPLIT = 'test'
CANONICAL_SEQ = 'real_seq02'
CANONICAL_START = 137
CANONICAL_END = 169


def parse_args():
    parser = argparse.ArgumentParser(
        description='No-training anchor-0 shared classification oracle.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default=CANONICAL_SPLIT)
    parser.add_argument('--seq', default=CANONICAL_SEQ)
    parser.add_argument('--start', type=int, default=CANONICAL_START)
    parser.add_argument('--end', type=int, default=CANONICAL_END)
    parser.add_argument('--pool-size', type=int, default=10000)
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--tie-atol', type=float, default=1e-8)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--allow-noncanonical', action='store_true')
    return parser.parse_args()


def validate_args(args) -> bool:
    if args.seed != 0:
        raise ValueError('The unified diagnostic protocol requires --seed 0')
    if args.end < args.start:
        raise ValueError('--end must be >= --start')
    if args.pool_size <= 0:
        raise ValueError('--pool-size must be positive')
    if not 0.0 < args.riou_thr <= 1.0:
        raise ValueError('--riou-thr must be in (0, 1]')
    if args.score_thr < 0.0 or args.tie_atol < 0.0:
        raise ValueError('Score and tie thresholds must be non-negative')
    canonical = bool(
        args.split == CANONICAL_SPLIT
        and args.seq == CANONICAL_SEQ
        and args.start == CANONICAL_START
        and args.end == CANONICAL_END
        and args.pool_size == 10000
        and os.path.basename(args.checkpoint) == 'epoch_20.pth'
        and 'brightaug' in os.path.basename(args.config).lower())
    if not canonical and not args.allow_noncanonical:
        raise ValueError(
            'Canonical oracle requires BrightAug epoch_20 and '
            'test/real_seq02[137..169] with pool_size=10000')
    return canonical


def shared_anchor0_logits(cls_scores: Sequence[torch.Tensor], head):
    if int(head.cls_out_channels) != 1:
        raise RuntimeError('Shared-filter oracle requires one foreground class')
    if int(head.num_anchors) < 2:
        raise RuntimeError('Shared-filter oracle requires multiple anchors')
    shared = []
    for level in cls_scores:
        if int(level.shape[1]) != int(head.num_anchors):
            raise RuntimeError(
                'Unexpected classification channels: {}'.format(
                    tuple(level.shape)))
        anchor0 = level[:, 0:1, :, :]
        shared.append(anchor0.repeat(1, int(head.num_anchors), 1, 1))
    return shared


def candidate_metrics(scores: torch.Tensor, ious: torch.Tensor,
                      layout: Sequence[Dict], pool_size: int,
                      riou_thr: float, score_thr: float,
                      tie_atol: float) -> Dict:
    if scores.ndim != 1 or ious.shape != scores.shape:
        raise ValueError('scores and ious must be aligned vectors')
    order = torch.argsort(scores, descending=True)
    top1_index = int(order[0].item())
    top1_score = float(scores[top1_index].item())
    top1_riou = float(ious[top1_index].item())
    top1_tie = torch.nonzero(
        scores >= top1_score - float(tie_atol),
        as_tuple=False).reshape(-1)
    actual_pool = min(int(pool_size), int(scores.numel()))
    pool_indices = order[:actual_pool]
    pool_ious = ious[pool_indices]
    kth_score = float(scores[pool_indices[-1]].item())
    expanded_pool = torch.nonzero(
        scores >= kth_score - float(tie_atol),
        as_tuple=False).reshape(-1)

    usable = torch.nonzero(
        ious >= float(riou_thr), as_tuple=False).reshape(-1)
    usable_record = None
    if usable.numel() > 0:
        usable_scores = scores[usable]
        best_position = int(torch.argmax(usable_scores).item())
        usable_index = int(usable[best_position].item())
        usable_score = float(scores[usable_index].item())
        strict_position = torch.nonzero(
            order == usable_index, as_tuple=False).reshape(-1)
        strict_rank = int(strict_position[0].item()) + 1
        best_rank = int((scores > usable_score + float(tie_atol)).sum().item()) + 1
        worst_rank = int((scores >= usable_score - float(tie_atol)).sum().item())
        usable_record = dict(
            candidate_index=usable_index,
            score=usable_score,
            riou=float(ious[usable_index].item()),
            strict_rank=strict_rank,
            tie_best_rank=best_rank,
            tie_worst_rank=worst_rank,
            anchor_id=int(layout[usable_index]['anchor_id']),
            level=int(layout[usable_index]['level']))

    return dict(
        candidate_count=int(scores.numel()),
        top1=dict(
            candidate_index=top1_index,
            score=top1_score,
            riou=top1_riou,
            hit=bool(top1_riou >= riou_thr),
            anchor_id=int(layout[top1_index]['anchor_id']),
            level=int(layout[top1_index]['level'])),
        silent=bool(top1_score < score_thr),
        top1_tie=dict(
            count=int(top1_tie.numel()),
            best_riou=float(ious[top1_tie].max().item()),
            hit=bool(float(ious[top1_tie].max().item()) >= riou_thr),
            anchors=sorted(set(
                int(layout[index]['anchor_id'])
                for index in top1_tie.tolist()))),
        pool=dict(
            size=actual_pool,
            hit=bool(float(pool_ious.max().item()) >= riou_thr),
            best_riou=float(pool_ious.max().item()),
            boundary_score=kth_score,
            tie_expanded_size=int(expanded_pool.numel()),
            tie_expanded_hit=bool(
                float(ious[expanded_pool].max().item()) >= riou_thr)),
        dense=dict(
            hit=bool(float(ious.max().item()) >= riou_thr),
            best_riou=float(ious.max().item())),
        usable=usable_record)


def analyze_frame(model, transforms, img_scale, flip, args, frame: int):
    from mmcv.ops import box_iou_rotated

    diag = entry_probe.get_diag()
    img_path, ann_path = diag.find_files(
        args.data_root, args.split, args.seq, frame)
    if img_path is None or ann_path is None:
        raise RuntimeError('Missing target-dev frame {}'.format(frame))
    gts = diag.parse_dota_ann(ann_path)
    if not gts:
        raise RuntimeError('Missing target-dev GT at frame {}'.format(frame))
    img_tensor, meta, _stats = diag.preprocess_image(
        img_path, transforms, img_scale, flip)
    img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))

    with torch.no_grad():
        features = model.extract_feat(img_tensor)
        head, original_cls, bbox_preds = entry_probe.forward_candidate_head(
            model, features, 'main')
        shared_cls = shared_anchor0_logits(original_cls, head)
        original = entry_probe.flatten_decode_candidates(
            head, original_cls, bbox_preds, meta['img_shape'])
        shared = entry_probe.flatten_decode_candidates(
            head, shared_cls, bbox_preds, meta['img_shape'])
        original_boxes, original_scores, levels, centers, alignment = original
        shared_boxes, shared_scores, shared_levels, shared_centers, _ = shared
        geometry_identical = bool(
            torch.equal(original_boxes, shared_boxes)
            and torch.equal(levels, shared_levels)
            and torch.equal(centers, shared_centers))
        if not geometry_identical:
            raise RuntimeError(
                'Counterfactual changed decoded geometry or candidate order')
        scaled_gts = [pool_probe.scale_gt_to_img(gt, meta) for gt in gts]
        gt_boxes = torch.stack([
            entry_probe.gt_to_tensor(gt, original_boxes.device).reshape(5)
            for gt in scaled_gts])
        ious = box_iou_rotated(
            original_boxes.float(), gt_boxes.float()).max(dim=1).values
        layout = contribution.candidate_layout(
            original_cls, head, meta['img_shape'])
        if len(layout) != int(original_scores.numel()):
            raise RuntimeError('Candidate layout mismatch')
        modes = dict(
            original=candidate_metrics(
                original_scores, ious, layout, args.pool_size,
                args.riou_thr, args.score_thr, args.tie_atol),
            shared_anchor0=candidate_metrics(
                shared_scores, ious, layout, args.pool_size,
                args.riou_thr, args.score_thr, args.tie_atol))

    return dict(
        frame=int(frame),
        image=os.path.relpath(img_path, os.path.realpath(args.data_root)),
        geometry_identical=geometry_identical,
        decode_alignment=alignment,
        modes=modes)


def _rank_stats(rows: Sequence[Dict], mode: str, key: str) -> Dict:
    values = [row['modes'][mode]['usable'][key] for row in rows
              if row['modes'][mode]['usable'] is not None]
    return dict(
        count=len(values),
        median=float(np.median(values)) if values else None,
        p90=float(np.percentile(values, 90)) if values else None,
        max=max(values) if values else None)


def summarize(rows: Sequence[Dict], mode: str, riou_thr: float) -> Dict:
    projected = []
    for row in rows:
        item = row['modes'][mode]
        projected.append(dict(
            frame=int(row['frame']),
            top1_hit=bool(item['top1']['hit']),
            tie_hit=bool(item['top1_tie']['hit']),
            pool_hit=bool(item['pool']['hit']),
            expanded_pool_hit=bool(item['pool']['tie_expanded_hit']),
            dense_hit=bool(item['dense']['hit'])))
    total = len(rows)

    def count(key):
        return sum(bool(item[key]) for item in projected)

    return dict(
        frames=total,
        riou_thr=float(riou_thr),
        top1_hits=count('top1_hit'),
        top1_recall=count('top1_hit') / total if total else 0.0,
        top1_miss_run=pool_probe.longest_consecutive_miss(
            projected, 'top1_hit'),
        tie_aware_top1_hits=count('tie_hit'),
        tie_aware_top1_recall=count('tie_hit') / total if total else 0.0,
        tie_aware_top1_miss_run=pool_probe.longest_consecutive_miss(
            projected, 'tie_hit'),
        pool_hits=count('pool_hit'),
        pool_recall=count('pool_hit') / total if total else 0.0,
        pool_miss_run=pool_probe.longest_consecutive_miss(
            projected, 'pool_hit'),
        tie_expanded_pool_hits=count('expanded_pool_hit'),
        dense_hits=count('dense_hit'),
        dense_recall=count('dense_hit') / total if total else 0.0,
        dense_miss_run=pool_probe.longest_consecutive_miss(
            projected, 'dense_hit'),
        silent_frames=sum(
            bool(row['modes'][mode]['silent']) for row in rows),
        usable_rank_strict=_rank_stats(rows, mode, 'strict_rank'),
        usable_rank_tie_best=_rank_stats(rows, mode, 'tie_best_rank'),
        usable_rank_tie_worst=_rank_stats(rows, mode, 'tie_worst_rank'),
        top1_anchor_histogram={
            str(anchor): sum(
                int(row['modes'][mode]['top1']['anchor_id']) == anchor
                for row in rows)
            for anchor in range(3)})


def main():
    args = parse_args()
    canonical = validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, cfg = entry_probe.load_model(
        args.config, args.checkpoint, args.gpu)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    diag = entry_probe.get_diag()
    transforms, img_scale, flip = diag.build_test_transforms(cfg)
    head = entry_probe.get_candidate_head(model, 'main')
    if not hasattr(head, 'retina_cls') or hasattr(head, 'cls_convs'):
        raise RuntimeError('Oracle requires the single-layer retina_cls head')

    rows = []
    for frame in range(args.start, args.end + 1):
        row = analyze_frame(
            model, transforms, img_scale, flip, args, frame)
        rows.append(row)
        original = row['modes']['original']
        shared = row['modes']['shared_anchor0']
        print('[{}] original rank={} hit={} shared rank={}/{} hit={}/{}'.format(
            frame,
            None if original['usable'] is None
            else original['usable']['strict_rank'],
            original['top1']['hit'],
            None if shared['usable'] is None
            else shared['usable']['tie_best_rank'],
            None if shared['usable'] is None
            else shared['usable']['tie_worst_rank'],
            shared['top1']['hit'], shared['top1_tie']['hit']))

    summaries = {
        mode: summarize(rows, mode, args.riou_thr)
        for mode in ('original', 'shared_anchor0')}
    payload = dict(
        probe='shared_filter_counterfactual_oracle',
        protocol_version=1,
        canonical_protocol=bool(canonical),
        data_role='target_dev',
        split=args.split, seq=args.seq,
        start=int(args.start), end=int(args.end),
        diagnosis_only=True,
        counterfactual_oracle=True,
        deployable=False,
        performs_training=False,
        performs_optimizer_step=False,
        updates_model_parameters=False,
        uses_target_domain=True,
        uses_target_labels=True,
        eligible_for_training=False,
        eligible_for_checkpoint_selection=False,
        must_not_export_target_rows_to_training=True,
        protocol_ready_for_training=False,
        config=args.config,
        checkpoint=args.checkpoint,
        parameters=dict(
            pool_size=int(args.pool_size), riou_thr=float(args.riou_thr),
            score_thr=float(args.score_thr), tie_atol=float(args.tie_atol),
            source_filter_anchor=0,
            regression='original_unchanged'),
        geometry_identity_passed=all(
            row['geometry_identical'] for row in rows),
        summaries=summaries,
        rows=rows)
    output_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.out_json, 'w') as handle:
        json.dump(payload, handle, indent=2)

    print('\nSHARED-FILTER COUNTERFACTUAL ORACLE')
    print('geometry identity: {}'.format(payload['geometry_identity_passed']))
    for mode, summary in summaries.items():
        print('{} top1={}/{} miss_run={} tie_top1={}/{} tie_run={} '
              'R@{}={}/{} rank_med={}/{}/{} dense={}/{}'.format(
                  mode, summary['top1_hits'], summary['frames'],
                  summary['top1_miss_run'],
                  summary['tie_aware_top1_hits'], summary['frames'],
                  summary['tie_aware_top1_miss_run'], args.pool_size,
                  summary['pool_hits'], summary['frames'],
                  summary['usable_rank_strict']['median'],
                  summary['usable_rank_tie_best']['median'],
                  summary['usable_rank_tie_worst']['median'],
                  summary['dense_hits'], summary['frames']))
    print('[out] wrote {}'.format(os.path.abspath(args.out_json)))
    print('[policy] TARGET-DEV COUNTERFACTUAL ONLY; OUTPUT MUST NOT ENTER TRAINING')


if __name__ == '__main__':
    main()
