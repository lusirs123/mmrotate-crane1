#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
temporal_reanchor_probe.py - 零训练时序重锚诊断工具。

目的:
  在不改模型、不训练 refiner 的前提下, 验证连续漏检段是否能通过
  "上一帧/速度预测位置 + raw decoded 低分候选局部搜索" 被切断。

默认口径:
  - final txt 作为原始检测基线;
  - final 检测命中 GT 时才作为可靠时序种子, 这是诊断上限, 不是部署实现;
  - 漏检帧不看 GT 选候选, 只用预测框附近的几何连续性排序;
  - GT 只用于事后评估 reanchor 是否真正命中。

示例:
PYTHONPATH=. python3 crane_project/tools/temporal_reanchor_probe.py \
  --config crane_project/configs/crane_eood_k1.py \
  --checkpoint work_dirs/crane_eood_k1/epoch_24.pth \
  --pred-dir work_dirs/crane_eood_k1/ckpt_sweep/final_test/epoch_24/preds/Task1_grab \
  --split test --seq real_seq02 --start 133 --end 166 \
  --head aux1 --pass-mode bidir --left-anchor 132 --right-anchor 167 \
  --gpu 0 --out-dir work_dirs/crane_eood_k1/temporal_reanchor_seq02_133_166_aux1
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools.analyze_crane_failures import parse_dota_file  # noqa: E402
from crane_project.tools.ctx_entry_probe import (  # noqa: E402
    flatten_decode_candidates,
    forward_candidate_head,
    load_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Zero-training temporal re-anchor probe')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--pred-dir', required=True,
                        help='Final-test DOTA txt directory, e.g. Task1_grab')
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default='test')
    parser.add_argument('--seq', required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--warmup-start', type=int, default=None,
                        help='Use frames [warmup_start, start-1] to build temporal state before probing')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--center-thr', type=float, default=25.0)
    parser.add_argument('--seed-mode', choices=['oracle-valid', 'all-final'],
                        default='oracle-valid',
                        help='oracle-valid only trusts final detections that hit GT')
    parser.add_argument('--bootstrap',
                        choices=['none', 'oracle-gt-start', 'oracle-gt-prev'],
                        default='none',
                        help='Diagnostic-only initial state when no trusted final seed exists')
    parser.add_argument('--head', choices=['main', 'aux1'], default='main',
                        help='Dense head used as the low-score candidate pool')
    parser.add_argument('--pass-mode', choices=['forward', 'bidir'],
                        default='forward',
                        help='forward uses only the left history; bidir also runs from a right anchor')
    parser.add_argument('--left-anchor', type=int, default=None,
                        help='Frame used as left temporal anchor; default=start-1')
    parser.add_argument('--right-anchor', type=int, default=None,
                        help='Required by --pass-mode bidir; usually the first true detection after the blind window')
    parser.add_argument(
        '--bidir-select',
        choices=['lower-cost', 'joint-rank', 'time-weighted-rank'],
        default='lower-cost',
        help='lower-cost preserves the legacy comparison between directional '
             'minima; joint-rank scores the same candidate pool against both '
             'temporal predictions and minimizes the worst normalized rank; '
             'time-weighted-rank weights the forward/backward normalized '
             'ranks by the frame position between the two anchors.')
    parser.add_argument('--motion', choices=['hold', 'linear'], default='linear')
    parser.add_argument('--update-mode',
                        choices=['none', 'selected', 'oracle-hit'],
                        default='selected',
                        help='How reanchored candidates update temporal state')
    parser.add_argument(
        '--teacher-force-gt', action='store_true',
        help='Diagnostic only: after evaluating each frame, append its GT as '
             'history for the next step. This measures one-step temporal '
             'trackability without chain lockout.')

    parser.add_argument('--radius-mul', type=float, default=1.5,
                        help='Search radius = radius_mul * previous box diag')
    parser.add_argument('--min-radius-px', type=float, default=32.0)
    parser.add_argument('--max-radius-px', type=float, default=220.0)
    parser.add_argument('--max-cands', type=int, default=0,
                        help='0 keeps all local candidates; positive caps by score for speed only')
    parser.add_argument('--score-floor', type=float, default=0.0)

    parser.add_argument('--w-center', type=float, default=1.0)
    parser.add_argument('--w-size', type=float, default=0.35)
    parser.add_argument('--w-angle', type=float, default=0.0,
                        help='Default 0: angle is reported but not used for selection')
    parser.add_argument('--w-score', type=float, default=0.0,
                        help='Small bonus for confidence; geometry remains dominant')
    parser.add_argument('--angle-norm-deg', type=float, default=45.0)
    parser.add_argument('--diverge-center-px', type=float, default=25.0)
    parser.add_argument('--diverge-size-log', type=float, default=0.5)
    return parser.parse_args()


def angle_diff_rad(a: float, b: float) -> float:
    d = abs(float(a) - float(b))
    while d >= math.pi:
        d -= math.pi
    return abs(min(d, math.pi - d))


def box_diag(box: Sequence[float]) -> float:
    return float(math.hypot(float(box[2]), float(box[3])))


def final_txt_path(pred_dir: str, seq: str, fid: int) -> str:
    return os.path.join(pred_dir, f'{seq}_{fid:05d}.txt')


def best_final_box(pred_dir: str, seq: str, fid: int) -> Optional[Dict]:
    from mmrotate.core import poly2obb_np

    path = final_txt_path(pred_dir, seq, fid)
    boxes = parse_dota_file(path, is_pred=True)
    if not boxes:
        return None
    best = max(boxes, key=lambda b: b.get('score') or 0.0)
    obb = poly2obb_np(
        np.asarray(best['poly'], dtype=np.float32), version='le90')
    if obb is None:
        return None
    return dict(
        cx=float(obb[0]),
        cy=float(obb[1]),
        w=float(obb[2]),
        h=float(obb[3]),
        angle=float(obb[4]),
        score=float(best.get('score') or 0.0),
        pred_count=len(boxes),
        path=path,
    )


def dict_box_to_array(box: Dict) -> np.ndarray:
    return np.array(
        [box['cx'], box['cy'], box['w'], box['h'], box['angle']],
        dtype=np.float32)


def gt_to_array(gt: Dict) -> np.ndarray:
    return np.array(
        [gt['cx'], gt['cy'], gt['w'], gt['h'], math.radians(gt['angle'])],
        dtype=np.float32)


def riou_one(box: Sequence[float], gt: Sequence[float]) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    b = torch.as_tensor(
        np.asarray(box, dtype=np.float32).reshape(1, 5), device=device)
    g = torch.as_tensor(
        np.asarray(gt, dtype=np.float32).reshape(1, 5), device=device)
    return float(rotated_ious(b, g)[0].detach().cpu().item())


def rotated_ious(boxes: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    from mmcv.ops import box_iou_rotated

    return box_iou_rotated(boxes.float(), gt.float()).reshape(-1)


def eval_hit(box: Optional[Sequence[float]], gt: Sequence[float],
             riou_thr: float, center_thr: float) -> Dict:
    if box is None:
        return dict(riou=0.0, center_dist=None, riou_hit=False,
                    center_hit=False, hit=False, gamma_error_deg=None)
    riou = riou_one(box, gt)
    center_dist = float(np.linalg.norm(
        np.asarray(box[:2], dtype=np.float32)
        - np.asarray(gt[:2], dtype=np.float32)))
    return dict(
        riou=riou,
        center_dist=center_dist,
        gamma_error_deg=float(math.degrees(
            angle_diff_rad(float(box[4]), float(gt[4])))),
        riou_hit=bool(riou >= riou_thr),
        center_hit=bool(center_dist <= center_thr),
        hit=bool(riou >= riou_thr),
    )


def size_log_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(
        abs(math.log(max(float(a[2]), 1e-3) / max(float(b[2]), 1e-3)))
        + abs(math.log(max(float(a[3]), 1e-3) / max(float(b[3]), 1e-3))))


def predict_box(history: List[Dict], fid: int, motion: str) -> Optional[np.ndarray]:
    if not history:
        return None
    last = history[-1]
    pred = np.array(last['box'], dtype=np.float32).copy()
    if motion == 'linear' and len(history) >= 2:
        prev = history[-2]
        dt_hist = int(last['fid']) - int(prev['fid'])
        if dt_hist == 0:
            return pred
        dt = int(fid) - int(last['fid'])
        v = (np.asarray(last['box'][:2]) - np.asarray(prev['box'][:2])) / dt_hist
        pred[:2] = np.asarray(last['box'][:2]) + v * dt
    return pred


def head_forward(model, feats, head_name: str):
    return forward_candidate_head(model, feats, head_name)


def decoded_boxes_to_ori(boxes: torch.Tensor, meta: Dict) -> torch.Tensor:
    """Map raw decoded boxes from resized test coordinates to original pixels.

    Final DOTA predictions, annotations, and all temporal radius thresholds in
    this probe use original-image coordinates.  Keeping decoded candidates in
    resized coordinates would make temporal distances and RIoU comparisons
    invalid after the shared test-pipeline fix.
    """
    scale_factor = meta.get('scale_factor')
    if isinstance(scale_factor, torch.Tensor):
        scale_factor = scale_factor.detach().cpu().numpy()
    scale = np.asarray(scale_factor, dtype=np.float64).reshape(-1)
    if scale.size == 0 or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise RuntimeError(
            f'Invalid scale_factor for temporal decode: {scale_factor!r}')
    sx = float(scale[0])
    sy = float(scale[1]) if scale.size >= 2 else sx
    size_scale = float(math.sqrt(sx * sy))
    mapped = boxes.clone()
    mapped[:, 0] /= sx
    mapped[:, 1] /= sy
    mapped[:, 2] /= size_scale
    mapped[:, 3] /= size_scale
    return mapped


def decode_frame_candidates(model, transform_compose, img_scale, flip,
                            data_root: str, split: str, seq: str, fid: int,
                            gpu: int, head_name: str):
    from crane_project.tools import mcml_diag as diag

    img_path, ann_path = diag.find_files(data_root, split, seq, fid)
    if img_path is None:
        return None, None, None
    gts = diag.parse_dota_ann(ann_path)
    if not gts:
        return None, None, None
    img_tensor, meta, img_stats = diag.preprocess_image(
        img_path, transform_compose, img_scale, flip)
    if img_tensor is None:
        return None, None, None
    img_tensor = img_tensor.cuda(f'cuda:{gpu}')
    with torch.no_grad():
        feat = model.extract_feat(img_tensor)
        head, cls_scores, bbox_preds = head_forward(model, feat, head_name)
        boxes, scores, levels, _, decode_alignment = flatten_decode_candidates(
            head, cls_scores, bbox_preds, meta['img_shape'])
        boxes = decoded_boxes_to_ori(boxes, meta)
    img_stats = dict(img_stats)
    img_stats['candidate_coordinates'] = 'original_image'
    img_stats['decode_scale_factor'] = np.asarray(
        meta['scale_factor'], dtype=np.float64).reshape(-1).tolist()
    img_stats['decode_alignment'] = decode_alignment
    return (boxes.detach(), scores.detach(), levels.detach()), gts[0], img_stats


def build_temporal_ranking(cands, pred_box: np.ndarray, args) -> Dict:
    boxes, scores, levels = cands
    radius = min(
        float(args.max_radius_px),
        max(float(args.min_radius_px), float(args.radius_mul) * box_diag(pred_box)))
    d = torch.norm(boxes[:, :2] - boxes.new_tensor(pred_box[:2])[None, :],
                   dim=1)
    mask = (d <= radius) & (scores >= float(args.score_floor))
    raw_inds = torch.nonzero(mask, as_tuple=False).reshape(-1)
    raw_count = int(raw_inds.numel())
    if raw_count == 0:
        return dict(
            found=False, radius=radius, raw_count=0,
            raw_inds=raw_inds, inds=raw_inds, cost=None, distances=d)
    inds = raw_inds
    if raw_count > int(args.max_cands) > 0:
        _, order = torch.topk(scores[inds], k=int(args.max_cands), largest=True)
        inds = inds[order]

    cost = temporal_cost_for_indices(
        cands, pred_box, inds, radius, args)
    return dict(
        found=True,
        radius=radius,
        raw_count=raw_count,
        used_count=int(inds.numel()),
        raw_inds=raw_inds,
        inds=inds,
        cost=cost,
        distances=d,
    )


def temporal_cost_for_indices(cands, pred_box: np.ndarray,
                              inds: torch.Tensor, radius: float,
                              args) -> torch.Tensor:
    boxes, scores, _ = cands
    sub_boxes = boxes[inds]
    sub_scores = scores[inds]
    pred_t = boxes.new_tensor(pred_box)
    center_cost = torch.norm(
        sub_boxes[:, :2] - pred_t[:2][None, :], dim=1) / max(radius, 1e-6)
    size_cost = (
        torch.abs(torch.log(
            sub_boxes[:, 2].clamp(min=1e-3)
            / pred_t[2].clamp(min=1e-3)))
        + torch.abs(torch.log(
            sub_boxes[:, 3].clamp(min=1e-3)
            / pred_t[3].clamp(min=1e-3))))
    angle_diffs = [
        angle_diff_rad(angle, float(pred_box[4]))
        for angle in sub_boxes[:, 4].detach().cpu().numpy().tolist()
    ]
    angle_cost = boxes.new_tensor(angle_diffs) / math.radians(
        float(args.angle_norm_deg))
    score_bonus = sub_scores.clamp(min=0.0, max=1.0)
    return (
        float(args.w_center) * center_cost
        + float(args.w_size) * size_cost
        + float(args.w_angle) * angle_cost
        - float(args.w_score) * score_bonus)


def normalized_ordinal_ranks(cost: torch.Tensor) -> torch.Tensor:
    """Return zero-to-one ordinal ranks; lower cost receives lower rank."""
    if cost.numel() <= 1:
        return torch.zeros_like(cost)
    order = torch.argsort(cost, stable=True)
    ranks = torch.empty_like(cost)
    ranks[order] = torch.arange(
        cost.numel(), dtype=cost.dtype, device=cost.device)
    return ranks / float(cost.numel() - 1)


def bidir_time_weights(fid: int, left_anchor: int,
                       right_anchor: int) -> Tuple[float, float, float]:
    """Return forward/backward weights at one frame between two anchors."""
    span = int(right_anchor) - int(left_anchor)
    if span <= 0:
        raise ValueError(
            'right_anchor must be greater than left_anchor for '
            'time-weighted bidirectional selection')
    alpha = float(np.clip(
        (int(fid) - int(left_anchor)) / float(span), 0.0, 1.0))
    return 1.0 - alpha, alpha, alpha


def select_joint_rank_candidate(cands, fwd_pred: np.ndarray,
                                bwd_pred: np.ndarray,
                                fwd_ranking: Dict,
                                bwd_ranking: Dict, args,
                                directional_weights: Optional[
                                    Tuple[float, float]] = None) -> Dict:
    """Select one existing candidate using comparable directional ranks."""
    boxes, scores, levels = cands
    if not fwd_ranking['found'] and not bwd_ranking['found']:
        return dict(found=False, candidate_count=0)
    candidate_inds = torch.unique(torch.cat([
        fwd_ranking['inds'], bwd_ranking['inds']
    ]), sorted=True)
    if candidate_inds.numel() == 0:
        return dict(found=False, candidate_count=0)

    fwd_cost = temporal_cost_for_indices(
        cands, fwd_pred, candidate_inds, fwd_ranking['radius'], args)
    bwd_cost = temporal_cost_for_indices(
        cands, bwd_pred, candidate_inds, bwd_ranking['radius'], args)
    fwd_rank = normalized_ordinal_ranks(fwd_cost)
    bwd_rank = normalized_ordinal_ranks(bwd_cost)
    worst_rank = torch.maximum(fwd_rank, bwd_rank)
    if directional_weights is None:
        # The mean rank is only a deterministic tie-breaker. The minimax term
        # prevents a candidate favored by one direction and rejected by the
        # other from winning purely through incomparable raw costs.
        fwd_weight = bwd_weight = 0.5
        joint_metric = worst_rank + 1e-3 * (fwd_rank + bwd_rank)
    else:
        fwd_weight, bwd_weight = map(float, directional_weights)
        if (fwd_weight < 0.0 or bwd_weight < 0.0
                or not math.isclose(
                    fwd_weight + bwd_weight, 1.0,
                    rel_tol=1e-6, abs_tol=1e-6)):
            raise ValueError(
                'directional_weights must be non-negative and sum to one')
        # Near the left anchor, the forward state is more informative; near
        # the right anchor, the backward state is more informative. The worst
        # rank remains a small tie-breaker instead of dominating every frame.
        joint_metric = (
            fwd_weight * fwd_rank + bwd_weight * bwd_rank
            + 1e-3 * worst_rank)
    best_pos = int(torch.argmin(joint_metric).item())
    best_idx = candidate_inds[best_pos]
    best_box = boxes[best_idx].detach().cpu().numpy().astype(float)
    return dict(
        found=True,
        index=int(best_idx.item()),
        candidate_count=int(candidate_inds.numel()),
        box=best_box.tolist(),
        score=float(scores[best_idx].item()),
        level=int(levels[best_idx].item()),
        joint_metric=float(joint_metric[best_pos].item()),
        fwd_weight=fwd_weight,
        bwd_weight=bwd_weight,
        fwd_rank=float(fwd_rank[best_pos].item()),
        bwd_rank=float(bwd_rank[best_pos].item()),
        fwd_cost=float(fwd_cost[best_pos].item()),
        bwd_cost=float(bwd_cost[best_pos].item()),
    )


def select_reanchor_candidate(cands, pred_box: np.ndarray, args,
                              ranking: Optional[Dict] = None) -> Optional[Dict]:
    boxes, scores, levels = cands
    ranking = ranking or build_temporal_ranking(cands, pred_box, args)
    if not ranking['found']:
        return dict(
            found=False, radius=ranking['radius'],
            raw_count=ranking['raw_count'])
    inds = ranking['inds']
    cost = ranking['cost']
    best_pos = int(torch.argmin(cost).item())
    best_idx = inds[best_pos]
    best_box = boxes[best_idx].detach().cpu().numpy().astype(float)
    return dict(
        found=True,
        index=int(best_idx.item()),
        radius=ranking['radius'],
        raw_count=ranking['raw_count'],
        used_count=ranking['used_count'],
        box=best_box.tolist(),
        score=float(scores[best_idx].detach().cpu().item()),
        level=int(levels[best_idx].detach().cpu().item()),
        cost=float(cost[best_pos].detach().cpu().item()),
        dist_to_pred=float(
            ranking['distances'][best_idx].detach().cpu().item()),
    )


def candidate_oracle_diagnostics(cands, gt: np.ndarray, ranking: Dict,
                                 riou_thr: float) -> Dict:
    """Measure dense/local candidate availability independently of selection."""
    boxes, scores, levels = cands
    gt_tensor = boxes.new_tensor(gt).reshape(1, 5)
    ious = rotated_ious(boxes, gt_tensor)
    dense_idx = int(torch.argmax(ious).item())
    dense_best_riou = float(ious[dense_idx].item())
    dense_score = scores[dense_idx]
    result = dict(
        dense_best_riou=dense_best_riou,
        dense_oracle_hit=bool(dense_best_riou >= float(riou_thr)),
        dense_best_score=float(dense_score.item()),
        dense_best_score_rank=int((scores > dense_score).sum().item()) + 1,
        dense_best_level=int(levels[dense_idx].item()),
        local_best_riou=0.0,
        local_oracle_hit=False,
        local_usable_count=0,
        local_best_score=None,
        local_best_level=None,
        local_best_dist_to_pred=None,
        local_best_temporal_rank=None,
    )

    raw_inds = ranking['raw_inds']
    if raw_inds.numel() == 0:
        return result
    local_ious = ious[raw_inds]
    local_pos = int(torch.argmax(local_ious).item())
    local_idx = raw_inds[local_pos]
    local_best_riou = float(local_ious[local_pos].item())
    result.update(
        local_best_riou=local_best_riou,
        local_oracle_hit=bool(local_best_riou >= float(riou_thr)),
        local_usable_count=int(
            (local_ious >= float(riou_thr)).sum().item()),
        local_best_score=float(scores[local_idx].item()),
        local_best_level=int(levels[local_idx].item()),
        local_best_dist_to_pred=float(
            ranking['distances'][local_idx].item()),
    )

    used_matches = torch.nonzero(
        ranking['inds'] == local_idx, as_tuple=False).reshape(-1)
    if used_matches.numel():
        local_used_pos = int(used_matches[0].item())
        local_cost = ranking['cost'][local_used_pos]
        result['local_best_temporal_rank'] = int(
            (ranking['cost'] < local_cost).sum().item()) + 1
    return result


def longest_false_run(flags: Sequence[bool]) -> int:
    best = cur = 0
    for ok in flags:
        if ok:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def longest_false_span(flags: Sequence[bool], frames: Sequence[int]) -> Dict:
    best_len = cur_len = 0
    best_start = best_end = None
    cur_start = None
    for ok, fid in zip(flags, frames):
        if ok:
            cur_len = 0
            cur_start = None
            continue
        if cur_len == 0:
            cur_start = int(fid)
        cur_len += 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start
            best_end = int(fid)
    return dict(length=int(best_len), start=best_start, end=best_end)


def build_summary_metrics(rows: List[Dict]) -> Dict:
    frames = [int(row['frame']) for row in rows]

    def metric(hit_key: str) -> Dict:
        flags = [bool(row.get(hit_key)) for row in rows]
        return dict(
            hits=int(sum(flags)), total=len(flags),
            longest_miss=longest_false_span(flags, frames))

    metrics = dict(
        final=metric('final_hit'),
        after=metric('after_hit'),
        dense_oracle=metric('dense_oracle_hit'),
        local_oracle=metric('local_oracle_hit'),
    )
    if any('bidir_hit' in row for row in rows):
        metrics.update(
            heuristic_or=metric('heuristic_or_hit'),
            bidir_strict=metric('bidir_hit'),
            joint_selected=metric('joint_hit'),
            bidir_consistent=metric('bidir_consistent'))
    else:
        metrics['heuristic_selected'] = metric('reanchor_hit')
    return metrics


def write_csv(path: str, rows: List[Dict]):
    preferred = [
        'frame', 'zone', 'final_has_box', 'final_score', 'final_riou',
        'final_center_dist', 'final_gamma_error_deg', 'final_hit',
        'seed_used', 'pred_cx', 'pred_cy', 'reanchor_found',
        'reanchor_score', 'reanchor_level', 'reanchor_cost',
        'reanchor_dist_to_pred', 'reanchor_raw_count', 'reanchor_riou',
        'reanchor_center_dist', 'reanchor_gamma_error_deg', 'reanchor_hit',
        'dense_best_riou', 'dense_oracle_hit', 'dense_best_score_rank',
        'local_best_riou', 'local_oracle_hit', 'local_usable_count',
        'local_best_dist_to_pred', 'local_best_temporal_rank',
        'after_hit', 'bidir_hit', 'heuristic_or_hit', 'bidir_consistent',
        'fb_center_dist', 'fb_size_log_dist', 'fb_angle_diff_deg',
        'fwd_found', 'bwd_found', 'fwd_riou', 'bwd_riou',
        'fwd_local_best_riou', 'bwd_local_best_riou',
        'heuristic_best_pass', 'heuristic_best_riou',
        'heuristic_best_gamma_error_deg',
        'selection_mode', 'joint_found', 'joint_candidate_count',
        'joint_riou', 'joint_hit', 'joint_time_alpha',
        'joint_fwd_weight', 'joint_bwd_weight',
        'joint_fwd_rank', 'joint_bwd_rank',
        'selected_riou', 'selected_gamma_error_deg',
        'teacher_forced', 'history_updated', 'update_source', 'brightness',
    ]
    seen = set()
    fieldnames = []
    for key in preferred + [k for row in rows for k in row.keys()]:
        if key not in seen:
            fieldnames.append(key)
            seen.add(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def zone_for_frame(fid: int, start: int, end: int) -> str:
    n = max(1, int(end) - int(start) + 1)
    rel = int(fid) - int(start)
    if rel < n / 3.0:
        return 'left'
    if rel >= 2.0 * n / 3.0:
        return 'right'
    return 'mid'


def load_anchor_history(model, transform_compose, img_scale, flip,
                        args, fid: int, source: str) -> List[Dict]:
    _, gt_dict, _ = decode_frame_candidates(
        model, transform_compose, img_scale, flip,
        args.data_root, args.split, args.seq, fid, args.gpu, args.head)
    if gt_dict is None:
        raise RuntimeError(f'No GT/image for anchor {args.seq}_{fid:05d}')
    gt = gt_to_array(gt_dict)
    final_path = final_txt_path(args.pred_dir, args.seq, fid)
    final = best_final_box(args.pred_dir, args.seq, fid)
    final_box = dict_box_to_array(final) if final is not None else None
    final_eval = eval_hit(final_box, gt, args.riou_thr, args.center_thr)
    final_hit = bool(final_eval['hit'])
    seed_allowed = (
        final is not None and (
            args.seed_mode == 'all-final'
            or (args.seed_mode == 'oracle-valid' and final_hit)))
    if seed_allowed:
        print(f'[anchor] {args.seq}_{fid:05d} source={source} '
              f'riou={final_eval["riou"]:.3f}')
        return [dict(fid=fid, box=final_box.tolist(), source=source)]
    if args.bootstrap != 'none':
        print(f'[anchor] {args.seq}_{fid:05d} source={source}_gt '
              f'final_riou={final_eval["riou"]:.3f}')
        return [dict(fid=fid, box=gt.tolist(), source=f'{source}_{args.bootstrap}')]
    raise RuntimeError(
        f'Anchor {args.seq}_{fid:05d} is not trusted: '
        f'has_final={final is not None} final_hit={final_hit} '
        f'riou={final_eval["riou"]:.3f} final_path={final_path}')


def run_temporal_pass(model, transform_compose, img_scale, flip, args,
                      frame_ids: Sequence[int], init_history: List[Dict],
                      pass_name: str) -> Dict[int, Dict]:
    history = [dict(item) for item in init_history]
    rows = {}
    ordered = list(frame_ids)
    if pass_name == 'bwd':
        ordered = list(reversed(ordered))

    for fid in ordered:
        cands, gt_dict, img_stats = decode_frame_candidates(
            model, transform_compose, img_scale, flip,
            args.data_root, args.split, args.seq, fid, args.gpu, args.head)
        if gt_dict is None:
            rows[int(fid)] = dict(frame=int(fid), missing=True)
            continue
        gt = gt_to_array(gt_dict)
        final = best_final_box(args.pred_dir, args.seq, fid)
        final_box = dict_box_to_array(final) if final is not None else None
        final_eval = eval_hit(final_box, gt, args.riou_thr, args.center_thr)
        final_hit = bool(final_eval['hit'])

        seed_allowed = (
            final is not None and (
                args.seed_mode == 'all-final'
                or (args.seed_mode == 'oracle-valid' and final_hit)))
        if seed_allowed:
            history.append(dict(fid=fid, box=final_box.tolist(), source='final'))

        pred = predict_box(history, fid, args.motion)
        selected = None
        ranking = None
        selected_eval = dict(riou=0.0, center_dist=None,
                             gamma_error_deg=None, hit=False)
        oracle_diag = dict(
            dense_best_riou=None, dense_oracle_hit=False,
            dense_best_score=None, dense_best_score_rank=None,
            dense_best_level=None, local_best_riou=None,
            local_oracle_hit=False, local_usable_count=0,
            local_best_score=None, local_best_level=None,
            local_best_dist_to_pred=None,
            local_best_temporal_rank=None)
        seed_used = bool((not final_hit) and pred is not None and cands is not None)
        if pred is not None and cands is not None:
            ranking = build_temporal_ranking(cands, pred, args)
            oracle_diag = candidate_oracle_diagnostics(
                cands, gt, ranking, args.riou_thr)
        if seed_used:
            selected = select_reanchor_candidate(
                cands, pred, args, ranking=ranking)
            if selected and selected.get('found'):
                selected_eval = eval_hit(
                    selected['box'], gt, args.riou_thr, args.center_thr)

        history_updated = False
        update_source = ''
        if args.teacher_force_gt:
            teacher_item = dict(
                fid=fid, box=gt.tolist(), source=f'{pass_name}_teacher_gt')
            if history and int(history[-1]['fid']) == int(fid):
                history[-1] = teacher_item
            else:
                history.append(teacher_item)
            history_updated = True
            update_source = 'teacher_gt'
        elif selected and selected.get('found'):
            do_update = (
                args.update_mode == 'selected'
                or (args.update_mode == 'oracle-hit' and selected_eval['hit']))
            if do_update:
                history.append(dict(fid=fid, box=selected['box'], source=pass_name))
                history_updated = True
                update_source = pass_name

        rows[int(fid)] = dict(
            frame=int(fid),
            pass_name=pass_name,
            head=args.head,
            final_has_box=final is not None,
            final_score=None if final is None else final['score'],
            final_riou=final_eval['riou'],
            final_center_dist=final_eval['center_dist'],
            final_gamma_error_deg=final_eval['gamma_error_deg'],
            final_hit=final_hit,
            seed_used=seed_used,
            pred_cx=None if pred is None else float(pred[0]),
            pred_cy=None if pred is None else float(pred[1]),
            reanchor_found=bool(selected and selected.get('found')),
            reanchor_score=None if not selected or not selected.get('found') else selected['score'],
            reanchor_level=None if not selected or not selected.get('found') else selected['level'],
            reanchor_cost=None if not selected or not selected.get('found') else selected['cost'],
            reanchor_dist_to_pred=None if not selected or not selected.get('found') else selected['dist_to_pred'],
            reanchor_raw_count=0 if not selected else selected.get('raw_count', 0),
            reanchor_box=None if not selected or not selected.get('found') else selected['box'],
            reanchor_riou=selected_eval['riou'],
            reanchor_center_dist=selected_eval['center_dist'],
            reanchor_gamma_error_deg=selected_eval['gamma_error_deg'],
            reanchor_hit=bool(selected_eval['hit']),
            after_hit=bool(final_hit or selected_eval['hit']),
            teacher_forced=bool(args.teacher_force_gt),
            history_updated=history_updated,
            update_source=update_source,
            brightness=None if img_stats is None else img_stats.get('raw_brightness'),
            # Internal-only fields consumed by merge_bidir_rows. They are
            # removed before serialization by constructing a fresh public row.
            _pred_box=None if pred is None else pred.copy(),
            _gt_box=gt.copy(),
            _cands=cands,
            _ranking=ranking,
            **oracle_diag,
        )
    return rows


def merge_bidir_rows(frame_ids: Sequence[int], fwd_rows: Dict[int, Dict],
                     bwd_rows: Dict[int, Dict], args) -> List[Dict]:
    rows = []
    rank_selection = args.bidir_select in (
        'joint-rank', 'time-weighted-rank')
    left_anchor = (
        int(args.left_anchor) if args.left_anchor is not None
        else int(args.start) - 1)
    right_anchor = (
        None if args.right_anchor is None else int(args.right_anchor))
    for fid in frame_ids:
        fwd = fwd_rows.get(int(fid), dict(frame=int(fid), reanchor_found=False))
        bwd = bwd_rows.get(int(fid), dict(frame=int(fid), reanchor_found=False))
        final_hit = bool(fwd.get('final_hit', False) or bwd.get('final_hit', False))
        f_box = fwd.get('reanchor_box')
        b_box = bwd.get('reanchor_box')
        both_found = f_box is not None and b_box is not None
        fb_center = None
        fb_size = None
        fb_angle = None
        if both_found:
            fb_center = float(np.linalg.norm(
                np.asarray(f_box[:2], dtype=np.float32)
                - np.asarray(b_box[:2], dtype=np.float32)))
            fb_size = size_log_dist(f_box, b_box)
            fb_angle = float(math.degrees(angle_diff_rad(f_box[4], b_box[4])))
        consistent = bool(
            both_found
            and fb_center <= float(args.diverge_center_px)
            and fb_size <= float(args.diverge_size_log))

        chosen = None
        joint = dict(found=False, candidate_count=0)
        joint_time_alpha = None
        directional_weights = None
        if args.bidir_select == 'time-weighted-rank':
            if right_anchor is None:
                raise ValueError(
                    'time-weighted-rank requires --right-anchor')
            fwd_weight, bwd_weight, joint_time_alpha = bidir_time_weights(
                int(fid), left_anchor, right_anchor)
            directional_weights = (fwd_weight, bwd_weight)
        joint_eval = dict(
            riou=0.0, center_dist=None, gamma_error_deg=None, hit=False)
        if rank_selection:
            fwd_cands = fwd.get('_cands')
            bwd_cands = bwd.get('_cands')
            fwd_pred = fwd.get('_pred_box')
            bwd_pred = bwd.get('_pred_box')
            fwd_ranking = fwd.get('_ranking')
            bwd_ranking = bwd.get('_ranking')
            if (fwd_cands is not None and bwd_cands is not None
                    and fwd_pred is not None and bwd_pred is not None
                    and fwd_ranking is not None and bwd_ranking is not None):
                # The two passes decode the same frame independently. Reuse
                # the forward pool after checking that their shapes agree.
                if (fwd_cands[0].shape != bwd_cands[0].shape
                        or fwd_cands[1].shape != bwd_cands[1].shape):
                    raise RuntimeError(
                        'Bidirectional candidate pools have different shapes')
                joint = select_joint_rank_candidate(
                    fwd_cands, np.asarray(fwd_pred), np.asarray(bwd_pred),
                    fwd_ranking, bwd_ranking, args,
                    directional_weights=directional_weights)
                if joint.get('found'):
                    joint_eval = eval_hit(
                        joint['box'], fwd.get('_gt_box'),
                        args.riou_thr, args.center_thr)
        elif consistent:
            f_cost = float(fwd.get('reanchor_cost') or 1e9)
            b_cost = float(bwd.get('reanchor_cost') or 1e9)
            chosen = fwd if f_cost <= b_cost else bwd
        selected_hit = bool(
            joint_eval['hit'] if rank_selection
            else chosen and chosen.get('reanchor_hit'))
        heuristic_or_hit = bool(
            fwd.get('reanchor_hit', False) or bwd.get('reanchor_hit', False))
        fwd_riou = float(fwd.get('reanchor_riou', 0.0) or 0.0)
        bwd_riou = float(bwd.get('reanchor_riou', 0.0) or 0.0)
        fwd_local_riou = float(fwd.get('local_best_riou', 0.0) or 0.0)
        bwd_local_riou = float(bwd.get('local_best_riou', 0.0) or 0.0)
        local_oracle_hit = bool(
            fwd.get('local_oracle_hit', False)
            or bwd.get('local_oracle_hit', False))
        # This is only the better of two heuristic selections. It is not an
        # oracle over the local candidate pool.
        if fwd_riou >= bwd_riou:
            heuristic_best = fwd
            heuristic_best_pass = 'fwd'
            heuristic_best_riou = fwd_riou
        else:
            heuristic_best = bwd
            heuristic_best_pass = 'bwd'
            heuristic_best_riou = bwd_riou
        row = dict(
            frame=int(fid),
            zone=zone_for_frame(fid, args.start, args.end),
            head=args.head,
            final_hit=final_hit,
            final_riou=fwd.get('final_riou', bwd.get('final_riou', 0.0)),
            final_center_dist=fwd.get('final_center_dist', bwd.get('final_center_dist')),
            final_gamma_error_deg=fwd.get('final_gamma_error_deg', bwd.get('final_gamma_error_deg')),
            fwd_found=bool(fwd.get('reanchor_found', False)),
            bwd_found=bool(bwd.get('reanchor_found', False)),
            fwd_riou=fwd_riou,
            bwd_riou=bwd_riou,
            fwd_gamma_error_deg=fwd.get('reanchor_gamma_error_deg'),
            bwd_gamma_error_deg=bwd.get('reanchor_gamma_error_deg'),
            fwd_cost=fwd.get('reanchor_cost'),
            bwd_cost=bwd.get('reanchor_cost'),
            fwd_raw_count=fwd.get('reanchor_raw_count', 0),
            bwd_raw_count=bwd.get('reanchor_raw_count', 0),
            fb_center_dist=fb_center,
            fb_size_log_dist=fb_size,
            fb_angle_diff_deg=fb_angle,
            bidir_consistent=consistent,
            dense_best_riou=fwd.get('dense_best_riou'),
            dense_oracle_hit=bool(fwd.get('dense_oracle_hit', False)),
            dense_best_score_rank=fwd.get('dense_best_score_rank'),
            fwd_local_best_riou=fwd_local_riou,
            bwd_local_best_riou=bwd_local_riou,
            local_best_riou=max(fwd_local_riou, bwd_local_riou),
            local_oracle_hit=local_oracle_hit,
            local_usable_count=max(
                int(fwd.get('local_usable_count', 0) or 0),
                int(bwd.get('local_usable_count', 0) or 0)),
            fwd_local_temporal_rank=fwd.get('local_best_temporal_rank'),
            bwd_local_temporal_rank=bwd.get('local_best_temporal_rank'),
            heuristic_or_hit=heuristic_or_hit,
            bidir_hit=bool(final_hit or selected_hit),
            heuristic_best_pass=heuristic_best_pass,
            heuristic_best_riou=heuristic_best_riou,
            heuristic_best_gamma_error_deg=heuristic_best.get(
                'reanchor_gamma_error_deg'),
            selection_mode=args.bidir_select,
            joint_found=bool(joint.get('found', False)),
            joint_candidate_count=int(joint.get('candidate_count', 0)),
            joint_riou=float(joint_eval['riou']),
            joint_hit=bool(joint_eval['hit']),
            joint_time_alpha=joint_time_alpha,
            joint_fwd_weight=(
                directional_weights[0] if directional_weights is not None
                else joint.get('fwd_weight')),
            joint_bwd_weight=(
                directional_weights[1] if directional_weights is not None
                else joint.get('bwd_weight')),
            joint_fwd_rank=joint.get('fwd_rank'),
            joint_bwd_rank=joint.get('bwd_rank'),
            selected_riou=(
                float(joint_eval['riou'])
                if rank_selection
                else (0.0 if chosen is None else float(
                    chosen.get('reanchor_riou', 0.0) or 0.0))),
            selected_gamma_error_deg=(
                joint_eval['gamma_error_deg']
                if rank_selection
                else (None if chosen is None else
                      chosen.get('reanchor_gamma_error_deg'))),
            after_hit=bool(final_hit or selected_hit),
            teacher_forced=bool(args.teacher_force_gt),
            brightness=fwd.get('brightness', bwd.get('brightness')),
        )
        rows.append(row)
    return rows


def run_bidir_probe(model, transform_compose, img_scale, flip, args) -> List[Dict]:
    left_anchor = int(args.left_anchor) if args.left_anchor is not None else int(args.start) - 1
    if args.right_anchor is None:
        raise ValueError('--pass-mode bidir requires --right-anchor')
    right_anchor = int(args.right_anchor)
    frame_ids = list(range(int(args.start), int(args.end) + 1))
    left_history = load_anchor_history(
        model, transform_compose, img_scale, flip, args, left_anchor, 'left_anchor')
    right_history = load_anchor_history(
        model, transform_compose, img_scale, flip, args, right_anchor, 'right_anchor')
    print(f'[bidir] left_anchor={left_anchor} right_anchor={right_anchor}')
    fwd_rows = run_temporal_pass(
        model, transform_compose, img_scale, flip, args, frame_ids,
        left_history, 'fwd')
    bwd_rows = run_temporal_pass(
        model, transform_compose, img_scale, flip, args, frame_ids,
        right_history, 'bwd')
    rows = merge_bidir_rows(frame_ids, fwd_rows, bwd_rows, args)
    for row in rows:
        time_info = ''
        if row.get('joint_time_alpha') is not None:
            time_info = (
                f" alpha={row['joint_time_alpha']:.3f}"
                f" weights={row['joint_fwd_weight']:.3f}/"
                f"{row['joint_bwd_weight']:.3f}")
        print(
            f"[{args.seq}_{row['frame']:05d}] zone={row['zone']} "
            f"strict={'OK' if row['bidir_hit'] else 'MISS'} "
            f"dense={row['dense_best_riou']:.3f} "
            f"local={row['local_best_riou']:.3f} "
            f"heuristic={row['heuristic_best_riou']:.3f} "
            f"selected={row['selected_riou']:.3f} "
            f"f/b={row['fwd_riou']:.3f}/{row['bwd_riou']:.3f} "
            f"found={int(row['fwd_found'])}/{int(row['bwd_found'])} "
            f"mode={row['selection_mode']} "
            f"h_best={row['heuristic_best_pass']} "
            f"agree={int(row['bidir_consistent'])} "
            f"fb_d={row['fb_center_dist']}"
            f"{time_info}")
    return rows


def print_summary(rows: List[Dict]):
    total = len(rows)
    frames = [int(r['frame']) for r in rows]
    final_hits = sum(bool(r.get('final_hit')) for r in rows)
    after_flags = [bool(r.get('after_hit')) for r in rows]
    after_hits = sum(after_flags)
    attempted = sum(bool(r.get('seed_used')) and not bool(r.get('final_hit')) for r in rows)
    found = sum(bool(r.get('reanchor_found')) for r in rows)
    revived = sum((not bool(r.get('final_hit'))) and bool(r.get('reanchor_hit')) for r in rows)
    false_selected = sum(
        bool(r.get('reanchor_found')) and not bool(r.get('reanchor_hit'))
        for r in rows if not bool(r.get('final_hit')))
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'  frames:              {total}')
    print(f'  final_hits:          {final_hits}/{total}')
    print(f'  after_hits:          {after_hits}/{total}')
    if any('bidir_hit' in r for r in rows):
        strict_flags = [bool(r.get('bidir_hit')) for r in rows]
        dense_flags = [bool(r.get('dense_oracle_hit')) for r in rows]
        local_flags = [bool(r.get('local_oracle_hit')) for r in rows]
        heuristic_flags = [bool(r.get('heuristic_or_hit')) for r in rows]
        strict_span = longest_false_span(strict_flags, frames)
        dense_span = longest_false_span(dense_flags, frames)
        local_span = longest_false_span(local_flags, frames)
        heuristic_span = longest_false_span(heuristic_flags, frames)
        final_span = longest_false_span([bool(r.get('final_hit')) for r in rows], frames)
        print(f'  dense_oracle_hits:   {sum(dense_flags)}/{total}')
        print(f'  local_oracle_hits:   {sum(local_flags)}/{total}')
        print(f'  heuristic_or_hits:   {sum(heuristic_flags)}/{total}')
        print(f'  bidir_strict_hits:   {sum(strict_flags)}/{total}')
        if any(r.get('selection_mode') in (
                'joint-rank', 'time-weighted-rank') for r in rows):
            joint_flags = [bool(r.get('joint_hit')) for r in rows]
            joint_span = longest_false_span(joint_flags, frames)
            print(f'  joint_selected_hits: {sum(joint_flags)}/{total}')
            print(f'  miss_run joint:      {joint_span["length"]} '
                  f'({joint_span["start"]}..{joint_span["end"]})')
        print(f'  consistent:          {sum(bool(r.get("bidir_consistent")) for r in rows)}/{total}')
        print(f'  fwd_found:           {sum(bool(r.get("fwd_found")) for r in rows)}/{total}')
        print(f'  bwd_found:           {sum(bool(r.get("bwd_found")) for r in rows)}/{total}')
        print(f'  miss_run dense:      {dense_span["length"]} '
              f'({dense_span["start"]}..{dense_span["end"]})')
        print(f'  miss_run local:      {local_span["length"]} '
              f'({local_span["start"]}..{local_span["end"]})')
        print(f'  miss_run heuristic:  {heuristic_span["length"]} '
              f'({heuristic_span["start"]}..{heuristic_span["end"]})')
        print(f'  miss_run strict:     {final_span["length"]} -> {strict_span["length"]} '
              f'({strict_span["start"]}..{strict_span["end"]})')
        for zone in ['left', 'mid', 'right']:
            zrows = [r for r in rows if r.get('zone') == zone]
            if not zrows:
                continue
            strict_gamma_vals = [
                float(r['selected_gamma_error_deg']) for r in zrows
                if r.get('selected_gamma_error_deg') is not None]
            heuristic_gamma_vals = [
                float(r['heuristic_best_gamma_error_deg']) for r in zrows
                if r.get('heuristic_best_gamma_error_deg') is not None]
            div_vals = [
                float(r['fb_center_dist']) for r in zrows
                if r.get('fb_center_dist') is not None]
            riou_vals = [float(r.get('selected_riou', 0.0) or 0.0) for r in zrows]
            local_riou_vals = [
                float(r.get('local_best_riou', 0.0) or 0.0)
                for r in zrows]
            heuristic_riou_vals = [
                float(r.get('heuristic_best_riou', 0.0) or 0.0)
                for r in zrows]
            print(
                f'  zone {zone}: strict={sum(bool(r.get("bidir_hit")) for r in zrows)}/{len(zrows)} '
                f'riou_mean={np.mean(riou_vals):.3f} '
                f'local_mean={np.mean(local_riou_vals):.3f} '
                f'heuristic_mean={np.mean(heuristic_riou_vals):.3f} '
                f'fb_center_mean={(np.mean(div_vals) if div_vals else float("nan")):.1f} '
                f'gamma_mean={(np.mean(strict_gamma_vals) if strict_gamma_vals else float("nan")):.1f} '
                f'heuristic_gamma_mean={(np.mean(heuristic_gamma_vals) if heuristic_gamma_vals else float("nan")):.1f}')
    else:
        dense_flags = [bool(r.get('dense_oracle_hit')) for r in rows]
        local_flags = [bool(r.get('local_oracle_hit')) for r in rows]
        if any(r.get('dense_best_riou') is not None for r in rows):
            dense_span = longest_false_span(dense_flags, frames)
            local_span = longest_false_span(local_flags, frames)
            print(f'  dense_oracle_hits:   {sum(dense_flags)}/{total}')
            print(f'  local_oracle_hits:   {sum(local_flags)}/{total}')
            print(f'  miss_run dense:      {dense_span["length"]} '
                  f'({dense_span["start"]}..{dense_span["end"]})')
            print(f'  miss_run local:      {local_span["length"]} '
                  f'({local_span["start"]}..{local_span["end"]})')
        print(f'  reanchor_attempted:  {attempted}')
        print(f'  reanchor_found:      {found}')
        print(f'  valid_revive:        {revived}')
        print(f'  selected_but_miss:   {false_selected}')
        print(f'  miss_run:            {longest_false_run([r["final_hit"] for r in rows])}'
              f' -> {longest_false_run([r["after_hit"] for r in rows])}')


def main():
    args = parse_args()
    if args.bidir_select == 'time-weighted-rank':
        if args.pass_mode != 'bidir':
            raise ValueError(
                '--bidir-select time-weighted-rank requires '
                '--pass-mode bidir')
        left_anchor = (
            int(args.left_anchor) if args.left_anchor is not None
            else int(args.start) - 1)
        if args.right_anchor is None:
            raise ValueError(
                '--bidir-select time-weighted-rank requires '
                '--right-anchor')
        bidir_time_weights(
            int(args.start), left_anchor, int(args.right_anchor))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    model, cfg = load_model(args.config, args.checkpoint, args.gpu)
    from crane_project.tools import mcml_diag as diag
    transform_compose, img_scale, flip = diag.build_test_transforms(cfg)
    print('[coordinates] decoded candidates, GT, final predictions, and '
          'temporal thresholds all use original-image pixels')
    if args.teacher_force_gt:
        print('[teacher-force] GT is appended only after each frame is '
              'evaluated; update-mode is ignored for temporal history')
    if args.bidir_select == 'time-weighted-rank':
        print('[time-weighted-rank] directional ranks are weighted by each '
              'frame position between the left and right anchors')

    if args.pass_mode == 'bidir':
        rows = run_bidir_probe(model, transform_compose, img_scale, flip, args)
        csv_path = os.path.join(args.out_dir, 'per_frame.csv')
        json_path = os.path.join(args.out_dir, 'summary.json')
        write_csv(csv_path, rows)
        payload = dict(
            args=vars(args), metrics=build_summary_metrics(rows), rows=rows)
        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print_summary(rows)
        print(f'\n[out] wrote {csv_path}')
        print(f'[out] wrote {json_path}')
        return

    history: List[Dict] = []
    rows: List[Dict] = []

    warmup_rows = []
    if args.warmup_start is not None and int(args.warmup_start) < int(args.start):
        for fid in range(int(args.warmup_start), int(args.start)):
            _, gt_dict, _ = decode_frame_candidates(
                model, transform_compose, img_scale, flip,
                args.data_root, args.split, args.seq, fid, args.gpu,
                args.head)
            if gt_dict is None:
                continue
            gt = gt_to_array(gt_dict)
            final = best_final_box(args.pred_dir, args.seq, fid)
            final_box = dict_box_to_array(final) if final is not None else None
            final_eval = eval_hit(final_box, gt, args.riou_thr, args.center_thr)
            final_hit = bool(final_eval['hit'])
            seed_allowed = (
                final is not None and (
                    args.seed_mode == 'all-final'
                    or (args.seed_mode == 'oracle-valid' and final_hit)))
            if seed_allowed:
                history.append(dict(fid=fid, box=final_box.tolist(), source='warmup_final'))
            warmup_rows.append(dict(frame=fid, final_hit=final_hit, seeded=seed_allowed))
        print(f'[warmup] frames={len(warmup_rows)} seeded={sum(r["seeded"] for r in warmup_rows)}')

    if not history and args.bootstrap != 'none':
        bootstrap_fid = int(args.start)
        if args.bootstrap == 'oracle-gt-prev':
            bootstrap_fid = int(args.start) - 1
        _, gt_dict, _ = decode_frame_candidates(
            model, transform_compose, img_scale, flip,
            args.data_root, args.split, args.seq, bootstrap_fid, args.gpu,
            args.head)
        if gt_dict is not None:
            gt_box = gt_to_array(gt_dict)
            history.append(dict(
                fid=bootstrap_fid,
                box=gt_box.tolist(),
                source=args.bootstrap))
            print(f'[bootstrap] {args.bootstrap} at {args.seq}_{bootstrap_fid:05d}')
        else:
            print(f'[bootstrap] failed: no GT at {args.seq}_{bootstrap_fid:05d}')

    for fid in range(int(args.start), int(args.end) + 1):
        cands, gt_dict, img_stats = decode_frame_candidates(
            model, transform_compose, img_scale, flip,
            args.data_root, args.split, args.seq, fid, args.gpu,
            args.head)
        if gt_dict is None:
            print(f'[skip] {args.seq}_{fid:05d}: missing image or GT')
            continue
        gt = gt_to_array(gt_dict)
        final = best_final_box(args.pred_dir, args.seq, fid)
        final_box = dict_box_to_array(final) if final is not None else None
        final_eval = eval_hit(final_box, gt, args.riou_thr, args.center_thr)
        final_hit = bool(final_eval['hit'])

        seed_allowed = False
        if final is not None:
            seed_allowed = (
                args.seed_mode == 'all-final'
                or (args.seed_mode == 'oracle-valid' and final_hit))
        if seed_allowed:
            history.append(dict(fid=fid, box=final_box.tolist(), source='final'))

        pred = predict_box(history, fid, args.motion)
        selected = None
        ranking = None
        selected_eval = dict(
            riou=0.0, center_dist=None, gamma_error_deg=None, hit=False)
        oracle_diag = dict(
            dense_best_riou=None, dense_oracle_hit=False,
            dense_best_score=None, dense_best_score_rank=None,
            dense_best_level=None, local_best_riou=None,
            local_oracle_hit=False, local_usable_count=0,
            local_best_score=None, local_best_level=None,
            local_best_dist_to_pred=None,
            local_best_temporal_rank=None)
        seed_used = bool((not final_hit) and pred is not None and cands is not None)
        if pred is not None and cands is not None:
            ranking = build_temporal_ranking(cands, pred, args)
            oracle_diag = candidate_oracle_diagnostics(
                cands, gt, ranking, args.riou_thr)
        if seed_used:
            selected = select_reanchor_candidate(
                cands, pred, args, ranking=ranking)
            if selected and selected.get('found'):
                selected_eval = eval_hit(
                    selected['box'], gt, args.riou_thr, args.center_thr)

        history_updated = False
        update_source = ''
        if args.teacher_force_gt:
            teacher_item = dict(
                fid=fid, box=gt.tolist(), source='forward_teacher_gt')
            if history and int(history[-1]['fid']) == int(fid):
                history[-1] = teacher_item
            else:
                history.append(teacher_item)
            history_updated = True
            update_source = 'teacher_gt'
        elif selected and selected.get('found'):
            do_update = (
                args.update_mode == 'selected'
                or (args.update_mode == 'oracle-hit' and selected_eval['hit']))
            if do_update:
                history.append(dict(fid=fid, box=selected['box'], source='reanchor'))
                history_updated = True
                update_source = 'reanchor'

        after_hit = bool(final_hit or selected_eval['hit'])
        row = dict(
            frame=fid,
            head=args.head,
            final_has_box=final is not None,
            final_score=None if final is None else final['score'],
            final_riou=final_eval['riou'],
            final_center_dist=final_eval['center_dist'],
            final_hit=final_hit,
            seed_used=seed_used,
            pred_cx=None if pred is None else float(pred[0]),
            pred_cy=None if pred is None else float(pred[1]),
            reanchor_found=bool(selected and selected.get('found')),
            reanchor_score=None if not selected or not selected.get('found') else selected['score'],
            reanchor_level=None if not selected or not selected.get('found') else selected['level'],
            reanchor_cost=None if not selected or not selected.get('found') else selected['cost'],
            reanchor_dist_to_pred=None if not selected or not selected.get('found') else selected['dist_to_pred'],
            reanchor_raw_count=0 if not selected else selected.get('raw_count', 0),
            reanchor_riou=selected_eval['riou'],
            reanchor_center_dist=selected_eval['center_dist'],
            reanchor_gamma_error_deg=selected_eval['gamma_error_deg'],
            reanchor_hit=bool(selected_eval['hit']),
            after_hit=after_hit,
            teacher_forced=bool(args.teacher_force_gt),
            history_updated=history_updated,
            update_source=update_source,
            brightness=None if img_stats is None else img_stats.get('raw_brightness'),
            **oracle_diag,
        )
        rows.append(row)
        print(
            f"[{args.seq}_{fid:05d}] final={'OK' if final_hit else 'MISS'} "
            f"seed={int(seed_used)} reanchor="
            f"{'OK' if row['reanchor_hit'] else ('cand' if row['reanchor_found'] else '-')}"
            f" dense={float(row['dense_best_riou'] or 0.0):.3f}"
            f" local={float(row['local_best_riou'] or 0.0):.3f}"
            f" selected={row['reanchor_riou']:.3f}"
            f" score={row['reanchor_score']}")

    csv_path = os.path.join(args.out_dir, 'per_frame.csv')
    json_path = os.path.join(args.out_dir, 'summary.json')
    write_csv(csv_path, rows)
    payload = dict(
        args=vars(args), metrics=build_summary_metrics(rows),
        warmup_rows=warmup_rows, rows=rows)
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print_summary(rows)
    print(f'\n[out] wrote {csv_path}')
    print(f'[out] wrote {json_path}')


if __name__ == '__main__':
    main()
