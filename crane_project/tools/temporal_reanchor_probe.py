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
    parser.add_argument('--motion', choices=['hold', 'linear'], default='linear')
    parser.add_argument('--update-mode',
                        choices=['none', 'selected', 'oracle-hit'],
                        default='selected',
                        help='How reanchored candidates update temporal state')

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
    path = final_txt_path(pred_dir, seq, fid)
    boxes = parse_dota_file(path, is_pred=True)
    if not boxes:
        return None
    best = max(boxes, key=lambda b: b.get('score') or 0.0)
    return dict(
        cx=float(best['cx']),
        cy=float(best['cy']),
        w=float(best['w']),
        h=float(best['h']),
        angle=float(math.radians(best['angle_deg'])),
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
    from mmcv.ops import box_iou_rotated

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    b = torch.tensor([box], dtype=torch.float32, device=device)
    g = torch.tensor([gt], dtype=torch.float32, device=device)
    return float(box_iou_rotated(b, g).reshape(-1)[0].detach().cpu().item())


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


def _repeat_scores_for_anchors(scores, bbox_flat, anchors, lvl):
    if anchors.shape[0] == scores.shape[0] == bbox_flat.shape[0]:
        return scores
    if anchors.shape[0] != bbox_flat.shape[0]:
        raise RuntimeError(
            f'Anchor/bbox mismatch at level {lvl}: '
            f'anchors={anchors.shape}, bbox={bbox_flat.shape}')
    if anchors.shape[0] % scores.shape[0] != 0:
        raise RuntimeError(
            f'Anchor/order mismatch at level {lvl}: '
            f'anchors={anchors.shape}, scores={scores.shape}, bbox={bbox_flat.shape}')
    repeat_factor = anchors.shape[0] // scores.shape[0]
    return scores[:, None].expand(-1, repeat_factor).reshape(-1)


def flatten_decode_candidates_for_head(head, cls_scores, bbox_preds, img_shape):
    device = cls_scores[0].device
    featmap_sizes = [score.shape[-2:] for score in cls_scores]
    anchors_per_level = head.anchor_generator.grid_priors(
        featmap_sizes, device=device)

    all_boxes, all_scores, all_levels = [], [], []
    for lvl, (cls_lvl, bbox_lvl, anchors) in enumerate(
            zip(cls_scores, bbox_preds, anchors_per_level)):
        cls_feat = cls_lvl[0]
        bbox_feat = bbox_lvl[0]
        scores = cls_feat.permute(1, 2, 0).reshape(-1, 1).sigmoid().reshape(-1)
        bbox_flat = bbox_feat.permute(1, 2, 0).reshape(-1, 5)
        scores = _repeat_scores_for_anchors(scores, bbox_flat, anchors, lvl)
        decoded = head.bbox_coder.decode(anchors, bbox_flat, max_shape=img_shape)
        all_boxes.append(decoded)
        all_scores.append(scores)
        all_levels.append(torch.full(
            (scores.numel(),), lvl, dtype=torch.long, device=device))

    return (
        torch.cat(all_boxes, dim=0),
        torch.cat(all_scores, dim=0),
        torch.cat(all_levels, dim=0),
    )


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
        boxes, scores, levels = flatten_decode_candidates_for_head(
            head, cls_scores, bbox_preds, meta['img_shape'])
    return (boxes.detach(), scores.detach(), levels.detach()), gts[0], img_stats


def select_reanchor_candidate(cands, pred_box: np.ndarray, args) -> Optional[Dict]:
    boxes, scores, levels = cands
    radius = min(
        float(args.max_radius_px),
        max(float(args.min_radius_px), float(args.radius_mul) * box_diag(pred_box)))
    d = torch.norm(boxes[:, :2] - boxes.new_tensor(pred_box[:2])[None, :],
                   dim=1)
    mask = (d <= radius) & (scores >= float(args.score_floor))
    inds = torch.nonzero(mask, as_tuple=False).reshape(-1)
    raw_count = int(inds.numel())
    if raw_count == 0:
        return dict(found=False, radius=radius, raw_count=0)
    if raw_count > int(args.max_cands) > 0:
        _, order = torch.topk(scores[inds], k=int(args.max_cands), largest=True)
        inds = inds[order]

    sub_boxes = boxes[inds]
    sub_scores = scores[inds]
    sub_levels = levels[inds]
    pred_t = boxes.new_tensor(pred_box)

    center_cost = torch.norm(sub_boxes[:, :2] - pred_t[:2][None, :], dim=1) / max(radius, 1e-6)
    size_cost = (
        torch.abs(torch.log((sub_boxes[:, 2].clamp(min=1e-3) / pred_t[2].clamp(min=1e-3))))
        + torch.abs(torch.log((sub_boxes[:, 3].clamp(min=1e-3) / pred_t[3].clamp(min=1e-3)))))
    angle_diffs = []
    for a in sub_boxes[:, 4].detach().cpu().numpy().tolist():
        angle_diffs.append(angle_diff_rad(a, float(pred_box[4])))
    angle_cost = boxes.new_tensor(angle_diffs) / math.radians(float(args.angle_norm_deg))
    score_bonus = sub_scores.clamp(min=0.0, max=1.0)

    cost = (
        float(args.w_center) * center_cost
        + float(args.w_size) * size_cost
        + float(args.w_angle) * angle_cost
        - float(args.w_score) * score_bonus
    )
    best_pos = int(torch.argmin(cost).item())
    best_idx = inds[best_pos]
    best_box = boxes[best_idx].detach().cpu().numpy().astype(float)
    return dict(
        found=True,
        radius=radius,
        raw_count=raw_count,
        used_count=int(inds.numel()),
        box=best_box.tolist(),
        score=float(scores[best_idx].detach().cpu().item()),
        level=int(levels[best_idx].detach().cpu().item()),
        cost=float(cost[best_pos].detach().cpu().item()),
        dist_to_pred=float(d[best_idx].detach().cpu().item()),
    )


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


def write_csv(path: str, rows: List[Dict]):
    preferred = [
        'frame', 'zone', 'final_has_box', 'final_score', 'final_riou',
        'final_center_dist', 'final_gamma_error_deg', 'final_hit',
        'seed_used', 'pred_cx', 'pred_cy', 'reanchor_found',
        'reanchor_score', 'reanchor_level', 'reanchor_cost',
        'reanchor_dist_to_pred', 'reanchor_raw_count', 'reanchor_riou',
        'reanchor_center_dist', 'reanchor_gamma_error_deg', 'reanchor_hit',
        'after_hit', 'bidir_hit', 'bidir_or_hit', 'bidir_consistent',
        'fb_center_dist', 'fb_size_log_dist', 'fb_angle_diff_deg',
        'fwd_found', 'bwd_found', 'fwd_riou', 'bwd_riou',
        'or_best_pass', 'or_best_riou', 'or_best_gamma_error_deg',
        'selected_riou', 'selected_gamma_error_deg',
        'history_updated', 'update_source', 'brightness',
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
        return [dict(fid=fid, box=final_box.tolist(), source=source)]
    if args.bootstrap != 'none':
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
        selected_eval = dict(riou=0.0, center_dist=None,
                             gamma_error_deg=None, hit=False)
        seed_used = bool((not final_hit) and pred is not None and cands is not None)
        if seed_used:
            selected = select_reanchor_candidate(cands, pred, args)
            if selected and selected.get('found'):
                selected_eval = eval_hit(
                    selected['box'], gt, args.riou_thr, args.center_thr)

        history_updated = False
        update_source = ''
        if selected and selected.get('found'):
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
            history_updated=history_updated,
            update_source=update_source,
            brightness=None if img_stats is None else img_stats.get('raw_brightness'),
        )
    return rows


def merge_bidir_rows(frame_ids: Sequence[int], fwd_rows: Dict[int, Dict],
                     bwd_rows: Dict[int, Dict], args) -> List[Dict]:
    rows = []
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
        if consistent:
            f_cost = float(fwd.get('reanchor_cost') or 1e9)
            b_cost = float(bwd.get('reanchor_cost') or 1e9)
            chosen = fwd if f_cost <= b_cost else bwd
        selected_hit = bool(chosen and chosen.get('reanchor_hit'))
        or_hit = bool(fwd.get('reanchor_hit', False) or bwd.get('reanchor_hit', False))
        fwd_riou = float(fwd.get('reanchor_riou', 0.0) or 0.0)
        bwd_riou = float(bwd.get('reanchor_riou', 0.0) or 0.0)
        # 诊断上界: 用 GT 事后挑前/后向中 RIoU 更高的那个。
        # 这不是可部署选择规则, 只用于避免 strict=0 时把候选信息全部吞掉。
        if fwd_riou >= bwd_riou:
            or_best = fwd
            or_best_pass = 'fwd'
            or_best_riou = fwd_riou
        else:
            or_best = bwd
            or_best_pass = 'bwd'
            or_best_riou = bwd_riou
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
            bidir_or_hit=or_hit,
            bidir_hit=bool(final_hit or selected_hit),
            or_best_pass=or_best_pass,
            or_best_riou=or_best_riou,
            or_best_gamma_error_deg=or_best.get('reanchor_gamma_error_deg'),
            selected_riou=0.0 if chosen is None else float(chosen.get('reanchor_riou', 0.0) or 0.0),
            selected_gamma_error_deg=None if chosen is None else chosen.get('reanchor_gamma_error_deg'),
            after_hit=bool(final_hit or selected_hit),
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
        print(
            f"[{args.seq}_{row['frame']:05d}] zone={row['zone']} "
            f"strict={'OK' if row['bidir_hit'] else 'MISS'} "
            f"or={'OK' if row['bidir_or_hit'] else 'MISS'} "
            f"f/b={row['fwd_riou']:.3f}/{row['bwd_riou']:.3f} "
            f"found={int(row['fwd_found'])}/{int(row['bwd_found'])} "
            f"or_best={row['or_best_riou']:.3f}/{row['or_best_pass']} "
            f"agree={int(row['bidir_consistent'])} "
            f"fb_d={row['fb_center_dist']}")
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
        or_flags = [bool(r.get('bidir_or_hit')) for r in rows]
        strict_span = longest_false_span(strict_flags, frames)
        or_span = longest_false_span(or_flags, frames)
        final_span = longest_false_span([bool(r.get('final_hit')) for r in rows], frames)
        print(f'  bidir_strict_hits:   {sum(strict_flags)}/{total}')
        print(f'  bidir_or_hits:       {sum(or_flags)}/{total}')
        print(f'  consistent:          {sum(bool(r.get("bidir_consistent")) for r in rows)}/{total}')
        print(f'  fwd_found:           {sum(bool(r.get("fwd_found")) for r in rows)}/{total}')
        print(f'  bwd_found:           {sum(bool(r.get("bwd_found")) for r in rows)}/{total}')
        print(f'  miss_run strict:     {final_span["length"]} -> {strict_span["length"]} '
              f'({strict_span["start"]}..{strict_span["end"]})')
        print(f'  miss_run or-bound:   {or_span["length"]} '
              f'({or_span["start"]}..{or_span["end"]})')
        for zone in ['left', 'mid', 'right']:
            zrows = [r for r in rows if r.get('zone') == zone]
            if not zrows:
                continue
            strict_gamma_vals = [
                float(r['selected_gamma_error_deg']) for r in zrows
                if r.get('selected_gamma_error_deg') is not None]
            or_gamma_vals = [
                float(r['or_best_gamma_error_deg']) for r in zrows
                if r.get('or_best_gamma_error_deg') is not None]
            div_vals = [
                float(r['fb_center_dist']) for r in zrows
                if r.get('fb_center_dist') is not None]
            riou_vals = [float(r.get('selected_riou', 0.0) or 0.0) for r in zrows]
            or_riou_vals = [float(r.get('or_best_riou', 0.0) or 0.0) for r in zrows]
            print(
                f'  zone {zone}: strict={sum(bool(r.get("bidir_hit")) for r in zrows)}/{len(zrows)} '
                f'riou_mean={np.mean(riou_vals):.3f} '
                f'or_riou_mean={np.mean(or_riou_vals):.3f} '
                f'fb_center_mean={(np.mean(div_vals) if div_vals else float("nan")):.1f} '
                f'gamma_mean={(np.mean(strict_gamma_vals) if strict_gamma_vals else float("nan")):.1f} '
                f'or_gamma_mean={(np.mean(or_gamma_vals) if or_gamma_vals else float("nan")):.1f}')
    else:
        print(f'  reanchor_attempted:  {attempted}')
        print(f'  reanchor_found:      {found}')
        print(f'  valid_revive:        {revived}')
        print(f'  selected_but_miss:   {false_selected}')
        print(f'  miss_run:            {longest_false_run([r["final_hit"] for r in rows])}'
              f' -> {longest_false_run([r["after_hit"] for r in rows])}')


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    model, cfg = load_model(args.config, args.checkpoint, args.gpu)
    from crane_project.tools import mcml_diag as diag
    transform_compose, img_scale, flip = diag.build_test_transforms(cfg)

    if args.pass_mode == 'bidir':
        rows = run_bidir_probe(model, transform_compose, img_scale, flip, args)
        csv_path = os.path.join(args.out_dir, 'per_frame.csv')
        json_path = os.path.join(args.out_dir, 'summary.json')
        write_csv(csv_path, rows)
        payload = dict(args=vars(args), rows=rows)
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
        selected_eval = dict(riou=0.0, center_dist=None, hit=False)
        seed_used = bool((not final_hit) and pred is not None and cands is not None)
        if seed_used:
            selected = select_reanchor_candidate(cands, pred, args)
            if selected and selected.get('found'):
                selected_eval = eval_hit(
                    selected['box'], gt, args.riou_thr, args.center_thr)

        history_updated = False
        update_source = ''
        if selected and selected.get('found'):
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
            reanchor_hit=bool(selected_eval['hit']),
            after_hit=after_hit,
            history_updated=history_updated,
            update_source=update_source,
            brightness=None if img_stats is None else img_stats.get('raw_brightness'),
        )
        rows.append(row)
        print(
            f"[{args.seq}_{fid:05d}] final={'OK' if final_hit else 'MISS'} "
            f"seed={int(seed_used)} reanchor="
            f"{'OK' if row['reanchor_hit'] else ('cand' if row['reanchor_found'] else '-')}"
            f" riou={row['reanchor_riou']:.3f} score={row['reanchor_score']}")

    csv_path = os.path.join(args.out_dir, 'per_frame.csv')
    json_path = os.path.join(args.out_dir, 'summary.json')
    write_csv(csv_path, rows)
    payload = dict(args=vars(args), warmup_rows=warmup_rows, rows=rows)
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print_summary(rows)
    print(f'\n[out] wrote {csv_path}')
    print(f'[out] wrote {json_path}')


if __name__ == '__main__':
    main()
