#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 1 gate for an independent PQA/reg-quality branch.

This probe does not train a model and does not modify the classification
target.  It replaces only PQA's predicted global position heatmap with the
perfect GT-derived heatmap, while every candidate is still encoded from its
own predicted OBB.  Consequently, candidate selection follows the deployable
PQA form instead of using GT IoU to choose a candidate::

    cls-topK pool -> oracle PQA quality -> cls * quality -> rank -> top-1

GT IoU is used only after ranking to evaluate whether the selected box is
control-usable.  The intended protocol is BrightAug epoch 20, K=10000, and
``real_seq02[137..169]``.  The gate passes when the resulting top-1 sequence
has MCML <= 5 at RIoU >= 0.5.  Dense-oracle and reranked Recall@K values are
reported as diagnostics and must not be described as deployable results.

Server example::

    PYTHONPATH=. python3 crane_project/tools/pqa_oracle_gate_probe.py \
      --config crane_project/configs/crane_symeood_k1_brightaug.py \
      --checkpoint work_dirs/crane_symeood_k1_brightaug/epoch_20.pth \
      --split test --seq real_seq02 --start 137 --end 169 \
      --pool-size 10000 --rerank-topks 1 10 100 500 2000 \
      --riou-thr 0.5 --gpu 0 \
      --out-json work_dirs/pqa_oracle_gate/brightaug_e20_seq02_137_169.json
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import candidate_pool_oracle_probe as pool_probe  # noqa: E402
from crane_project.tools import ctx_entry_probe as entry_probe  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Phase 1 oracle gate for pre-threshold PQA reranking.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default='test',
                        choices=['test', 'train', 'train_sim'])
    parser.add_argument('--seq', default='real_seq02')
    parser.add_argument('--start', type=int, default=137)
    parser.add_argument('--end', type=int, default=169)
    parser.add_argument('--sample', type=int, default=10)
    parser.add_argument('--candidate-source', default='main',
                        choices=['main', 'aux1'])
    parser.add_argument('--pool-size', type=int, default=10000,
                        help='Classification-topK pool retained before any '
                             'score threshold and reranked by PQA.')
    parser.add_argument('--rerank-topks', type=int, nargs='+',
                        default=[1, 10, 100, 500, 2000])
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--grid-size', type=int, default=25,
                        help='Uniform samples per OBB axis used to approximate '
                             'the pixel-wise Volume-IoU integral.')
    parser.add_argument('--quality-batch-size', type=int, default=512)
    parser.add_argument('--final-score-thr', type=float, default=0.0,
                        help='Threshold applied only after PQA reranking. Keep '
                             'at 0 for the Phase 1 pre-threshold gate.')
    parser.add_argument('--report-score-thrs', type=float, nargs='+',
                        default=[0.0, 0.0001, 0.001, 0.005, 0.05],
                        help='Report whether the selected top-1 would survive '
                             'each post-PQA threshold; does not alter ranking.')
    parser.add_argument('--max-mcml', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--allow-non-brightaug', action='store_true',
                        help='Allow a non-protocol checkpoint for smoke tests. '
                             'Such output is marked protocol_compliant=false.')
    return parser.parse_args()


def validate_args(args) -> bool:
    if args.pool_size <= 0:
        raise ValueError('--pool-size must be positive')
    if args.grid_size < 3:
        raise ValueError('--grid-size must be at least 3')
    if args.quality_batch_size <= 0:
        raise ValueError('--quality-batch-size must be positive')
    if not 0.0 <= args.riou_thr <= 1.0:
        raise ValueError('--riou-thr must be in [0, 1]')
    if args.final_score_thr < 0.0:
        raise ValueError('--final-score-thr must be non-negative')
    if args.seed != 0:
        raise ValueError('The unified experiment protocol requires --seed 0')

    config_ok = 'brightaug' in os.path.basename(args.config).lower()
    checkpoint_ok = (
        'brightaug' in os.path.abspath(args.checkpoint).lower()
        and os.path.basename(args.checkpoint) == 'epoch_20.pth')
    window_ok = (args.split == 'test' and args.seq == 'real_seq02'
                 and args.start == 137 and args.end == 169)
    source_ok = args.candidate_source == 'main'
    pool_ok = args.pool_size == 10000
    protocol_compliant = bool(
        config_ok and checkpoint_ok and window_ok and source_ok and pool_ok)
    if not protocol_compliant and not args.allow_non_brightaug:
        raise ValueError(
            'Phase 1 gate requires BrightAug epoch_20, test/real_seq02 '
            '[137..169], main candidates, and pool_size=10000. Use '
            '--allow-non-brightaug only for a clearly labelled smoke test.')
    return protocol_compliant


def _normalized_grid(grid_size: int, device, dtype) -> Tuple[torch.Tensor,
                                                               torch.Tensor]:
    """Return cell-centre samples in [-1, 1]^2."""
    axis = ((torch.arange(grid_size, device=device, dtype=dtype) + 0.5)
            * (2.0 / float(grid_size)) - 1.0)
    try:
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    except TypeError:  # PyTorch versions predating the ``indexing`` keyword.
        yy, xx = torch.meshgrid(axis, axis)
    return xx.reshape(1, -1), yy.reshape(1, -1)


def _points_inside_gaussian(points_x: torch.Tensor,
                            points_y: torch.Tensor,
                            boxes: torch.Tensor,
                            eps: float = 1e-6) -> torch.Tensor:
    """Evaluate paper Eq. (2), including the hard inside-OBB mask.

    Args:
        points_x/points_y: tensors shaped [N, P].
        boxes: one or N OBBs shaped [N, 5] in (cx, cy, w, h, theta).
    """
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError(f'boxes must have shape [N, 5], got {boxes.shape}')
    if boxes.shape[0] not in (1, points_x.shape[0]):
        raise ValueError('box batch must be 1 or match the point batch')

    cx, cy, width, height, angle = boxes.unbind(dim=1)
    width = width.clamp(min=eps)
    height = height.clamp(min=eps)
    dx = points_x - cx[:, None]
    dy = points_y - cy[:, None]
    cos_a = torch.cos(angle)[:, None]
    sin_a = torch.sin(angle)[:, None]
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    inside = ((local_x.abs() <= width[:, None] * 0.5)
              & (local_y.abs() <= height[:, None] * 0.5))

    # PQA Eq. (3): Sigma^(1/2) has eigenvalues w/4 and h/4,
    # therefore Sigma has eigenvalues w^2/16 and h^2/16.
    mahal = ((local_x / (width[:, None] * 0.25)) ** 2
             + (local_y / (height[:, None] * 0.25)) ** 2)
    return torch.exp(-0.5 * mahal) * inside.to(points_x.dtype)


def pqa_oracle_volume_iou(candidate_boxes: torch.Tensor,
                          gt_boxes: torch.Tensor,
                          grid_size: int = 25,
                          batch_size: int = 512,
                          image_shape: Optional[Sequence[int]] = None,
                          eps: float = 1e-12) -> torch.Tensor:
    """Compute GT-heatmap PQA quality without GT-based candidate selection.

    The local integration domain is each predicted OBB, as in paper Eqs.
    (5)-(9).  A fixed uniform grid approximates the sum over image pixels and
    makes K=10000 feasible.  For multiple GTs, H* is their point-wise maximum.
    """
    if candidate_boxes.ndim != 2 or candidate_boxes.shape[1] != 5:
        raise ValueError('candidate_boxes must have shape [N, 5]')
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 5:
        raise ValueError('gt_boxes must have shape [M, 5]')
    if gt_boxes.shape[0] == 0:
        return candidate_boxes.new_zeros(candidate_boxes.shape[0])

    results = []
    for start in range(0, candidate_boxes.shape[0], batch_size):
        boxes = candidate_boxes[start:start + batch_size].float()
        xx, yy = _normalized_grid(grid_size, boxes.device, boxes.dtype)
        cx, cy, width, height, angle = boxes.unbind(dim=1)
        half_w = width.clamp(min=1e-6)[:, None] * 0.5
        half_h = height.clamp(min=1e-6)[:, None] * 0.5
        local_x = xx * half_w
        local_y = yy * half_h
        cos_a = torch.cos(angle)[:, None]
        sin_a = torch.sin(angle)[:, None]
        points_x = cx[:, None] + cos_a * local_x - sin_a * local_y
        points_y = cy[:, None] + sin_a * local_x + cos_a * local_y

        # Fi is candidate-relative Eq. (2). Since samples are inside bi, the
        # closed form below is exactly the candidate Gaussian at those points.
        candidate_map = torch.exp(-2.0 * (xx.square() + yy.square()))
        candidate_map = candidate_map.expand(boxes.shape[0], -1)
        if image_shape is not None:
            image_h, image_w = int(image_shape[0]), int(image_shape[1])
            in_image = ((points_x >= 0.0) & (points_x < float(image_w))
                        & (points_y >= 0.0) & (points_y < float(image_h)))
            candidate_map = candidate_map * in_image.to(candidate_map.dtype)

        oracle_map = torch.zeros_like(candidate_map)
        for gt in gt_boxes.float():
            gt_map = _points_inside_gaussian(
                points_x, points_y, gt.reshape(1, 5))
            oracle_map = torch.maximum(oracle_map, gt_map)

        intersection = torch.minimum(oracle_map, candidate_map).sum(dim=1)
        union = torch.maximum(oracle_map, candidate_map).sum(dim=1)
        results.append(intersection / union.clamp(min=eps))
    return torch.cat(results, dim=0).to(candidate_boxes.dtype)


def _pearson(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    if x.numel() < 2:
        return None
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = torch.sqrt(x.square().sum() * y.square().sum())
    if float(denom.item()) <= 1e-12:
        return None
    return float((x * y).sum().div(denom).item())


def _rank_of_best_usable(values: torch.Tensor,
                         ious: torch.Tensor,
                         riou_thr: float) -> Optional[int]:
    usable = ious >= float(riou_thr)
    if not bool(usable.any()):
        return None
    best_value = values[usable].max()
    return int((values > best_value).sum().item()) + 1


def _stats(values: torch.Tensor) -> Dict:
    if values.numel() == 0:
        return dict(count=0, mean=None, min=None, max=None)
    return dict(
        count=int(values.numel()),
        mean=float(values.float().mean().item()),
        min=float(values.min().item()),
        max=float(values.max().item()),
    )


def discover_frame_ids(args) -> Tuple[str, List[int]]:
    return pool_probe.discover_frame_ids(args)


def analyze_frame(model, transform_compose, img_scale, flip, args,
                  seq: str, frame: int,
                  rerank_topks: Sequence[int]) -> Optional[Dict]:
    from mmcv.ops import box_iou_rotated

    diag = entry_probe.get_diag()
    img_path, ann_path = diag.find_files(
        args.data_root, args.split, seq, frame)
    if img_path is None:
        print(f'[skip] frame {frame:05d}: image not found')
        return None
    gts = diag.parse_dota_ann(ann_path)
    if not gts:
        print(f'[skip] frame {frame:05d}: GT not found')
        return None

    img_tensor, meta, img_stats = diag.preprocess_image(
        img_path, transform_compose, img_scale, flip)
    if img_tensor is None:
        print(f'[skip] frame {frame:05d}: preprocess failed')
        return None
    preprocess = pool_probe.build_preprocess_summary(
        img_tensor, meta, img_scale, flip)
    img_tensor = img_tensor.cuda(f'cuda:{args.gpu}')

    with torch.no_grad():
        feats = model.extract_feat(img_tensor)
        candidate_head, cls_scores, bbox_preds = (
            entry_probe.forward_candidate_head(
                model, feats, args.candidate_source))
        boxes, scores, levels, _, alignment = (
            entry_probe.flatten_decode_candidates(
                candidate_head, cls_scores, bbox_preds, meta['img_shape']))

        scaled_gts = [pool_probe.scale_gt_to_img(gt, meta) for gt in gts]
        gt_boxes = torch.stack([
            entry_probe.gt_to_tensor(gt, boxes.device).reshape(5)
            for gt in scaled_gts
        ])
        iou_matrix = box_iou_rotated(
            boxes.float(), gt_boxes.float())
        ious = iou_matrix.max(dim=1).values

        candidate_count = int(scores.numel())
        actual_pool_size = min(int(args.pool_size), candidate_count)
        pool_scores, pool_indices = torch.topk(
            scores, k=actual_pool_size, largest=True, sorted=True)
        pool_boxes = boxes[pool_indices]
        pool_ious = ious[pool_indices]
        pool_levels = levels[pool_indices]

        qualities = pqa_oracle_volume_iou(
            pool_boxes, gt_boxes, grid_size=args.grid_size,
            batch_size=args.quality_batch_size,
            image_shape=meta['img_shape'][:2])
        fused_scores = pool_scores * qualities
        rerank_order = torch.argsort(fused_scores, descending=True)
        selected_local = int(rerank_order[0].item())

        selected_score = float(fused_scores[selected_local].item())
        selected_riou = float(pool_ious[selected_local].item())
        selected_hit_geometry = bool(selected_riou >= args.riou_thr)
        selected_survives = bool(selected_score >= args.final_score_thr)
        selected_hit = bool(selected_hit_geometry and selected_survives)

        dense_best_pos = int(torch.argmax(ious).item())
        dense_best_riou = float(ious[dense_best_pos].item())
        quality_only_pos = int(torch.argmax(qualities).item())
        usable = pool_ious >= float(args.riou_thr)

        per_k = {}
        for topk in rerank_topks:
            actual_k = min(int(topk), actual_pool_size)
            local_indices = rerank_order[:actual_k]
            subset_ious = pool_ious[local_indices]
            best_riou = float(subset_ious.max().item())
            per_k[str(topk)] = dict(
                requested_k=int(topk),
                actual_k=actual_k,
                oracle_hit=bool(best_riou >= args.riou_thr),
                best_riou=best_riou,
            )

        threshold_survival = {
            str(float(thr)): bool(selected_score >= float(thr))
            for thr in sorted(set(args.report_score_thrs))
        }

    row = dict(
        frame=int(frame),
        fname=os.path.splitext(os.path.basename(img_path))[0],
        img_path=img_path,
        brightness=float(img_stats.get('raw_brightness', float('nan'))),
        candidate_source=args.candidate_source,
        candidate_head=entry_probe.candidate_head_name(candidate_head),
        candidate_count=candidate_count,
        pool_size=actual_pool_size,
        candidates_over_legacy_thr=int((scores >= 0.05).sum().item()),
        pool_candidates_over_legacy_thr=int(
            (pool_scores >= 0.05).sum().item()),
        dense_best_riou=dense_best_riou,
        dense_oracle_hit=bool(dense_best_riou >= args.riou_thr),
        baseline_top1=dict(
            cls_score=float(pool_scores[0].item()),
            riou=float(pool_ious[0].item()),
            hit=bool(pool_ious[0].item() >= args.riou_thr)),
        pqa_top1=dict(
            original_cls_rank=selected_local + 1,
            cls_score=float(pool_scores[selected_local].item()),
            quality=float(qualities[selected_local].item()),
            fused_score=selected_score,
            riou=selected_riou,
            level=int(pool_levels[selected_local].item()),
            survives_final_score_thr=selected_survives,
            geometry_hit=selected_hit_geometry,
            hit=selected_hit),
        quality_only_top1=dict(
            original_cls_rank=quality_only_pos + 1,
            quality=float(qualities[quality_only_pos].item()),
            riou=float(pool_ious[quality_only_pos].item()),
            hit=bool(pool_ious[quality_only_pos].item() >= args.riou_thr)),
        highest_cls_usable_rank=_rank_of_best_usable(
            pool_scores, pool_ious, args.riou_thr),
        highest_fused_usable_rank=_rank_of_best_usable(
            fused_scores, pool_ious, args.riou_thr),
        quality_iou_pearson=_pearson(qualities, pool_ious),
        fused_iou_pearson=_pearson(fused_scores, pool_ious),
        usable_quality_stats=_stats(qualities[usable]),
        unusable_quality_stats=_stats(qualities[~usable]),
        selected_threshold_survival=threshold_survival,
        per_rerank_k=per_k,
        gt_boxes=[dict(
            cx=float(gt['cx']), cy=float(gt['cy']),
            w=float(gt['w']), h=float(gt['h']),
            angle=float(gt['angle'])) for gt in scaled_gts],
        decode_alignment=alignment,
        preprocess=preprocess,
    )
    for topk in rerank_topks:
        row[f'rerank_hit_at_{topk}'] = bool(
            per_k[str(topk)]['oracle_hit'])

    print(
        f"[{row['fname']}] dense={dense_best_riou:.3f} "
        f"base={row['baseline_top1']['riou']:.3f} "
        f"PQA={selected_riou:.3f} hit={int(selected_hit)} "
        f"cls_rank={selected_local + 1} Q={qualities[selected_local]:.4f} "
        f"CQ={selected_score:.6f} usable_rerank="
        f"{row['highest_fused_usable_rank']}")
    return row


def _rank_stats(rows: Sequence[Dict], key: str) -> Dict:
    values = [int(row[key]) for row in rows if row[key] is not None]
    return dict(
        count=len(values),
        median=float(np.median(values)) if values else None,
        p90=float(np.percentile(values, 90)) if values else None,
        max=max(values) if values else None,
    )


def build_summary(rows: Sequence[Dict], rerank_topks: Sequence[int],
                  riou_thr: float, final_score_thr: float,
                  report_score_thrs: Sequence[float],
                  max_mcml: int) -> Dict:
    total = len(rows)
    dense_hits = sum(bool(row['dense_oracle_hit']) for row in rows)
    baseline_hits = sum(bool(row['baseline_top1']['hit']) for row in rows)
    pqa_hits = sum(bool(row['pqa_top1']['hit']) for row in rows)
    summary = dict(
        frames=total,
        riou_thr=float(riou_thr),
        final_score_thr=float(final_score_thr),
        dense_oracle_hits=dense_hits,
        dense_oracle_mcml=pool_probe.longest_consecutive_miss(
            rows, 'dense_oracle_hit') if rows else 0,
        baseline_top1_hits=baseline_hits,
        baseline_top1_mcml=pool_probe.longest_consecutive_miss(
            [dict(row, baseline_hit=row['baseline_top1']['hit'])
             for row in rows], 'baseline_hit') if rows else 0,
        pqa_top1_hits=pqa_hits,
        pqa_top1_mcml=pool_probe.longest_consecutive_miss(
            [dict(row, pqa_hit=row['pqa_top1']['hit'])
             for row in rows], 'pqa_hit') if rows else 0,
        pqa_dense_oracle_efficiency=(pqa_hits / dense_hits
                                     if dense_hits else 0.0),
        highest_cls_usable_rank=_rank_stats(
            rows, 'highest_cls_usable_rank'),
        highest_fused_usable_rank=_rank_stats(
            rows, 'highest_fused_usable_rank'),
        per_rerank_k={},
        per_score_threshold={},
    )
    for topk in rerank_topks:
        key = f'rerank_hit_at_{topk}'
        hits = sum(bool(row[key]) for row in rows)
        summary['per_rerank_k'][str(topk)] = dict(
            hits=hits,
            recall=hits / total if total else 0.0,
            oracle_mcml=pool_probe.longest_consecutive_miss(
                rows, key) if rows else 0,
        )
    for threshold in sorted(set(float(v) for v in report_score_thrs)):
        threshold_key = str(float(threshold))
        threshold_rows = []
        for row in rows:
            hit = bool(row['pqa_top1']['geometry_hit']
                       and row['selected_threshold_survival'][threshold_key])
            threshold_rows.append(dict(row, threshold_hit=hit))
        hits = sum(bool(row['threshold_hit']) for row in threshold_rows)
        summary['per_score_threshold'][threshold_key] = dict(
            hits=hits,
            recall=hits / total if total else 0.0,
            mcml=pool_probe.longest_consecutive_miss(
                threshold_rows, 'threshold_hit') if rows else 0,
        )

    feasible = bool(total and summary['dense_oracle_mcml'] <= max_mcml)
    passed = bool(total and summary['pqa_top1_mcml'] <= max_mcml)
    summary['gate'] = dict(
        max_mcml=int(max_mcml),
        dense_geometry_feasible=feasible,
        passed=passed,
        decision=('PASS' if passed else 'FAIL'),
        criterion='PQA top-1 MCML <= max_mcml at the configured post-PQA threshold',
    )
    return summary


def print_summary(summary: Dict, rerank_topks: Sequence[int]):
    print('\n' + '=' * 92)
    print('PHASE 1 PQA ORACLE GATE SUMMARY')
    print('=' * 92)
    print(
        f"frames={summary['frames']} RIoU_thr={summary['riou_thr']:.2f} "
        f"post_PQA_thr={summary['final_score_thr']:.6f}")
    print(
        f"dense oracle: {summary['dense_oracle_hits']}/{summary['frames']} "
        f"MCML={summary['dense_oracle_mcml']}")
    print(
        f"cls top-1:    {summary['baseline_top1_hits']}/{summary['frames']} "
        f"MCML={summary['baseline_top1_mcml']}")
    print(
        f"PQA top-1:    {summary['pqa_top1_hits']}/{summary['frames']} "
        f"MCML={summary['pqa_top1_mcml']} "
        f"dense_eff={summary['pqa_dense_oracle_efficiency']:.3f}")
    print('-' * 92)
    print(f"{'rerank K':>10} {'oracle hits':>14} {'recall':>10} {'oracle MCML':>14}")
    for topk in rerank_topks:
        item = summary['per_rerank_k'][str(topk)]
        print(
            f"{topk:>10d} {item['hits']:>6d}/{summary['frames']:<7d} "
            f"{item['recall']:>10.3f} {item['oracle_mcml']:>14d}")
    print('-' * 92)
    print('usable candidate rank before rerank: '
          f"{summary['highest_cls_usable_rank']}")
    print('usable candidate rank after rerank:  '
          f"{summary['highest_fused_usable_rank']}")
    print('post-PQA threshold sensitivity:')
    for threshold, item in summary['per_score_threshold'].items():
        print(
            f"  thr={float(threshold):.6f}: "
            f"hits={item['hits']}/{summary['frames']} MCML={item['mcml']}")
    gate = summary['gate']
    print('-' * 92)
    print(
        f"GATE={gate['decision']} (PQA top-1 MCML="
        f"{summary['pqa_top1_mcml']}, limit={gate['max_mcml']})")


def main():
    args = parse_args()
    protocol_compliant = validate_args(args)
    rerank_topks = pool_probe.normalize_topks(args.rerank_topks)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is required for model inference/mmcv rotated IoU. Run this '
            'probe on the experiment server; it never falls back to K1/CPU.')

    model, cfg = entry_probe.load_model(
        args.config, args.checkpoint, args.gpu)
    diag = entry_probe.get_diag()
    transform_compose, img_scale, flip = diag.build_test_transforms(cfg)
    seq, frame_ids = discover_frame_ids(args)

    print('=' * 92)
    print('PHASE 1: GT-HEATMAP PQA ORACLE GATE (NOT DEPLOYABLE)')
    print('=' * 92)
    print(f'config:              {args.config}')
    print(f'checkpoint:          {args.checkpoint}')
    print(f'protocol_compliant:  {protocol_compliant}')
    print(f'data:                {args.split}/{seq}')
    print(f'frames:              {frame_ids[0]}..{frame_ids[-1]} ({len(frame_ids)})')
    print(f'cls pre-pool K:      {args.pool_size}')
    print(f'rerank topks:        {rerank_topks}')
    print(f'PQA grid:            {args.grid_size}x{args.grid_size}')
    print(f'RIoU threshold:      {args.riou_thr}')
    print(f'post-PQA threshold:  {args.final_score_thr}')

    rows = []
    for frame in frame_ids:
        row = analyze_frame(
            model, transform_compose, img_scale, flip,
            args, seq, frame, rerank_topks)
        if row is not None:
            rows.append(row)
            if len(rows) == 1:
                prep = row['preprocess']
                print(
                    '[preprocess] '
                    f"ori={prep['ori_shape']} img={prep['img_shape']} "
                    f"pad={prep['pad_shape']} scale={prep['scale_factor']} "
                    f"tensor={prep['tensor_shape']} flip={prep['flip']}")
                print(f"[decode] {row['decode_alignment']}")

    summary = build_summary(
        rows, rerank_topks, args.riou_thr, args.final_score_thr,
        args.report_score_thrs, args.max_mcml)
    print_summary(summary, rerank_topks)

    output_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = dict(
        probe='pqa_gt_heatmap_oracle_gate',
        oracle=True,
        deployable=False,
        protocol_compliant=protocol_compliant,
        oracle_definition=(
            'GT boxes define the perfect global position heatmap H*. Each '
            'candidate keeps its own predicted OBB encoding Fi. Candidates '
            'are selected only by cls*VolumeIoU(H*,Fi); GT RIoU is evaluation-only.'),
        config=args.config,
        checkpoint=args.checkpoint,
        split=args.split,
        seq=seq,
        frame_ids=frame_ids,
        args=vars(args),
        summary=summary,
        rows=rows,
    )
    with open(output_path, 'w') as handle:
        json.dump(payload, handle, indent=2)
    print(f'\n[out] wrote {output_path}')


if __name__ == '__main__':
    main()
