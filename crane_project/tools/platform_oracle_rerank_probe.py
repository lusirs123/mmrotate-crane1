#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe 2: K-transform oracle platform rerank.

Given Probe 1's conclusion that ``real_seq02[133..171]`` has dense-oracle 39/39
(geometry exists but ranking is terrible), this script answers a single question:

    If we had *perfect* platform geometry information (derived from GT beam
    via K-transform), can platform-consistency scoring select the correct
    beam candidate from a top-K pool?

The K-transform maps beam OBB → platform OBB (anisotropic scaling + optional
centre offset).  It is fitted from 6 manually annotated platform polygons
(frames 137/144/150/156/162/169).

Three oracle reranking methods are evaluated:

    A. platform-only:  rank_score = S_platform
    B. multiplicative: rank_score = S_cls × S_platform
    C. log-space:      rank_score = log(S_cls+eps) + λ·log(S_platform+eps)

The probe does NOT train anything — it only inspects frozen K1 epoch_24 output.

Server example::

    PYTHONPATH=. python3 crane_project/tools/platform_oracle_rerank_probe.py \\
        --config crane_project/configs/crane_symeood_k1.py \\
        --checkpoint work_dirs/crane_symeood_k1/epoch_24.pth \\
        --split test --seq real_seq02 --start 133 --end 171 \\
        --topks 200 500 1000 --riou-thr 0.5 --gpu 0 \\
        --out-json work_dirs/candidate_pool_probe/k1_plat_oracle_133_171.json
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

from crane_project.tools import ctx_entry_probe as entry_probe  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Probe 2: K-transform oracle platform rerank.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default='test',
                        choices=['test', 'train', 'train_sim'])
    parser.add_argument('--seq', default='real_seq02')
    parser.add_argument('--start', type=int, default=133)
    parser.add_argument('--end', type=int, default=171)
    parser.add_argument('--topks', type=int, nargs='+',
                        default=[200, 500, 1000])
    parser.add_argument('--riou-thr', type=float, default=0.5,
                        help='RIoU threshold for oracle hit')
    parser.add_argument('--manual-platform-json',
                        default=os.path.join(
                            'work_dirs', 'crane_symeood_k1',
                            'manual_platform_polygons_real_seq02.json'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# OBB helpers
# ---------------------------------------------------------------------------

def corners_to_obb_le90(corners: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Convert 4-point polygon (N×2) to OBB via minAreaRect, le90 convention.

    Returns (cx, cy, w, h, angle_rad) where angle is in [-pi/2, pi/2).
    """
    from mmrotate.core import poly2obb_np
    obb = poly2obb_np(corners.reshape(-1).astype(np.float32), version='le90')
    return (float(obb[0]), float(obb[1]), float(obb[2]),
            float(obb[3]), float(obb[4]))


def obb_to_polygon(cx, cy, w, h, theta):
    """OBB (cx,cy,w,h,theta_rad) → 4×2 corner array."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    half_w = w / 2.0
    half_h = h / 2.0
    offsets = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h],
    ], dtype=np.float64)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    corners = offsets @ rot.T  # (4,2)
    corners[:, 0] += cx
    corners[:, 1] += cy
    return corners


def obb_riou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """Rotated IoU between two single-OBB tensors."""
    from mmcv.ops import box_iou_rotated
    iou = box_iou_rotated(
        box_a.float().unsqueeze(0), box_b.float().unsqueeze(0))
    return float(iou.item())


def obb_to_tensor(cx, cy, w, h, theta_rad, device='cpu'):
    """Build (1,5) tensor in (cx,cy,w,h,theta_rad) format."""
    return torch.tensor(
        [[cx, cy, w, h, theta_rad]], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# K-transform fit
# ---------------------------------------------------------------------------

def load_manual_platforms(json_path: str) -> Dict[int, np.ndarray]:
    """Return {frame_number: 4x2 corner_array_in_original_pixels}."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    result = {}
    for key, val in data.get('frames', {}).items():
        frame_id = int(val['frame'])
        corners = np.array(val['platform_corners'], dtype=np.float64)
        result[frame_id] = corners
    return result


def fit_k_transform(manual_platforms: Dict[int, np.ndarray],
                    data_root: str, split: str, seq: str,
                    annotated_frames: List[int]) -> Dict:
    """Fit anisotropic K-transform and centre offset from manual annotations.

    Returns dict with keys k_w, k_h, dx, dy and per-frame diagnostics.
    """
    diag = entry_probe.get_diag()

    k_w_vals, k_h_vals, dx_vals, dy_vals = [], [], [], []
    per_frame = []

    for frame in annotated_frames:
        if frame not in manual_platforms:
            print(f'  [warn] frame {frame} not in manual annotations; skip')
            continue

        img_path, ann_path = diag.find_files(data_root, split, seq, frame)
        if ann_path is None:
            print(f'  [warn] frame {frame}: DOTA annotation not found; skip')
            continue

        gts = diag.parse_dota_ann(ann_path)
        if not gts:
            print(f'  [warn] frame {frame}: no GT; skip')
            continue
        gt = gts[0]  # first (and only) beam GT

        plat_corners = manual_platforms[frame]

        # Convert manual platform polygon to OBB
        plat_cx, plat_cy, plat_w, plat_h, plat_theta = corners_to_obb_le90(
            plat_corners)

        beam_w = gt['w']
        beam_h = gt['h']
        beam_cx = gt['cx']
        beam_cy = gt['cy']

        k_w = plat_w / beam_w if beam_w > 0 else float('nan')
        k_h = plat_h / beam_h if beam_h > 0 else float('nan')
        dx = plat_cx - beam_cx
        dy = plat_cy - beam_cy

        k_w_vals.append(k_w)
        k_h_vals.append(k_h)
        dx_vals.append(dx)
        dy_vals.append(dy)

        per_frame.append(dict(
            frame=frame,
            beam=(beam_cx, beam_cy, beam_w, beam_h, gt['angle']),
            platform=(plat_cx, plat_cy, plat_w, plat_h,
                       float(np.degrees(plat_theta))),
            k_w=k_w, k_h=k_h, dx=dx, dy=dy,
        ))

    k_w_arr = np.array(k_w_vals)
    k_h_arr = np.array(k_h_vals)
    dx_arr = np.array(dx_vals)
    dy_arr = np.array(dy_vals)

    return dict(
        k_w_median=float(np.median(k_w_arr)),
        k_h_median=float(np.median(k_h_arr)),
        dx_median=float(np.median(dx_arr)),
        dy_median=float(np.median(dy_arr)),
        k_w_mean=float(np.mean(k_w_arr)),
        k_h_mean=float(np.mean(k_h_arr)),
        k_w_std=float(np.std(k_w_arr)),
        k_h_std=float(np.std(k_h_arr)),
        dx_std=float(np.std(dx_arr)),
        dy_std=float(np.std(dy_arr)),
        n_frames=len(k_w_vals),
        per_frame=per_frame,
    )


def apply_k_transform(obb_tensor: torch.Tensor,
                      k_params: Dict) -> torch.Tensor:
    """Apply anisotropic K-transform to a batch of OBBs.

    obb_tensor: (N, 5) in (cx, cy, w, h, theta_rad)
    Returns:    (N, 5) platform OBB in same format.

    The transform applies:
        cx' = cx + dx
        cy' = cy + dy
        w'  = w * k_w
        h'  = h * k_h
        theta' = theta   (orientation preserved)
    """
    k_w = float(k_params['k_w_median'])
    k_h = float(k_params['k_h_median'])
    dx = float(k_params['dx_median'])
    dy = float(k_params['dy_median'])

    plat = obb_tensor.clone()
    plat[:, 0] += dx       # cx
    plat[:, 1] += dy       # cy
    plat[:, 2] *= k_w      # w
    plat[:, 3] *= k_h      # h
    # theta unchanged
    return plat


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------

def scale_gt_to_img(gt: Dict, meta: Dict) -> Dict:
    """Scale GT from original image space to model image space."""
    scale_factor = meta.get('scale_factor', 1.0)
    if isinstance(scale_factor, torch.Tensor):
        scale_factor = scale_factor.detach().cpu().numpy()
    if isinstance(scale_factor, (list, tuple, np.ndarray)):
        flat = np.asarray(scale_factor, dtype=np.float64).reshape(-1)
        sx = float(flat[0]) if flat.size >= 1 else 1.0
        sy = float(flat[1]) if flat.size >= 2 else sx
    else:
        sx = sy = float(scale_factor)

    scaled = dict(gt)
    scaled['cx'] = float(gt['cx']) * sx
    scaled['cy'] = float(gt['cy']) * sy
    scaled['w'] = float(gt['w']) * sx
    scaled['h'] = float(gt['h']) * sy
    return scaled


def gt_to_tensor(gt: Dict, device) -> torch.Tensor:
    return torch.tensor(
        [[gt['cx'], gt['cy'], gt['w'], gt['h'], np.radians(gt['angle'])]],
        dtype=torch.float32, device=device)


def longest_consecutive_miss(rows, hit_key):
    longest = cur = 0
    prev = None
    for row in sorted(rows, key=lambda r: int(r['frame'])):
        f = int(row['frame'])
        if prev is not None and f != prev + 1:
            cur = 0
        if bool(row[hit_key]):
            cur = 0
        else:
            cur += 1
            longest = max(longest, cur)
        prev = f
    return int(longest)


def analyze_frame(model, transform_compose, img_scale, flip, args,
                  seq, frame_id, topks, k_params,
                  manual_platforms) -> Optional[Dict]:
    """Run K1 on one frame, extract top-K, compute platform oracle scores."""
    from mmcv.ops import box_iou_rotated

    diag = entry_probe.get_diag()
    img_path, ann_path = diag.find_files(
        args.data_root, args.split, seq, frame_id)
    if img_path is None or ann_path is None:
        return None

    gts = diag.parse_dota_ann(ann_path)
    if not gts:
        return None

    img_tensor, meta, img_stats = diag.preprocess_image(
        img_path, transform_compose, img_scale, flip)
    if img_tensor is None:
        return None
    img_tensor = img_tensor.cuda(f'cuda:{args.gpu}')

    with torch.no_grad():
        feats = model.extract_feat(img_tensor)
        candidate_head, cls_scores, bbox_preds = (
            entry_probe.forward_candidate_head(
                model, feats, 'main'))
        boxes, scores, levels, _, alignment = (
            entry_probe.flatten_decode_candidates(
                candidate_head, cls_scores, bbox_preds, meta['img_shape']))

        # GT in model-image space
        gt = scale_gt_to_img(gts[0], meta)
        gt_beam_tensor = gt_to_tensor(gt, boxes.device)

        # Reference platform: K_transform(GT_beam)
        reference_platform = apply_k_transform(gt_beam_tensor, k_params)

        # For frames with manual annotation: also compute actual platform OBB
        frame_has_manual = frame_id in manual_platforms
        manual_plat_tensor = None
        if frame_has_manual:
            # Manual platform corners are in ORIGINAL image pixels.
            # Scale to model-image space.
            sf = meta.get('scale_factor', 1.0)
            if isinstance(sf, torch.Tensor):
                sf = sf.detach().cpu().numpy()
            if isinstance(sf, (list, tuple, np.ndarray)):
                flat = np.asarray(sf, dtype=np.float64).reshape(-1)
                sx = float(flat[0]) if flat.size >= 1 else 1.0
                sy = float(flat[1]) if flat.size >= 2 else sx
            else:
                sx = sy = float(sf)

            plat_corners_orig = manual_platforms[frame_id].copy()
            plat_corners_orig[:, 0] *= sx
            plat_corners_orig[:, 1] *= sy
            plat_cx, plat_cy, plat_w, plat_h, plat_theta = corners_to_obb_le90(
                plat_corners_orig)
            manual_plat_tensor = obb_to_tensor(
                plat_cx, plat_cy, plat_w, plat_h, plat_theta,
                device=boxes.device)

        # Beam-gt RIoU for all candidates
        ious = box_iou_rotated(
            boxes.float(), gt_beam_tensor.float()).reshape(-1)

        # Dense oracle
        dense_best_pos = int(torch.argmax(ious).item())
        dense_best_riou = float(ious[dense_best_pos].item())

        max_k = min(max(topks), int(scores.numel()))
        top_scores, top_indices = torch.topk(
            scores, k=max_k, largest=True, sorted=True)
        top_boxes = boxes[top_indices]
        top_ious = ious[top_indices]

        # Platform scores: transform each top-K beam → platform, compare
        plat_preds = apply_k_transform(top_boxes, k_params)
        plat_ious_ref = box_iou_rotated(
            plat_preds.float(),
            reference_platform.expand(plat_preds.shape[0], -1).float()
        ).reshape(-1)

        plat_ious_manual = None
        if manual_plat_tensor is not None:
            plat_ious_manual = box_iou_rotated(
                plat_preds.float(),
                manual_plat_tensor.expand(plat_preds.shape[0], -1).float()
            ).reshape(-1)

        # Three reranking methods per K
        per_k = {}
        for topk in topks:
            actual_k = min(int(topk), max_k)
            sub_boxes = top_boxes[:actual_k]
            sub_scores = top_scores[:actual_k]
            sub_ious = top_ious[:actual_k]
            sub_plat = plat_ious_ref[:actual_k]
            sub_plat_man = (plat_ious_manual[:actual_k]
                            if plat_ious_manual is not None else None)

            methods = {}

            # A: platform-only
            rank_a = torch.argsort(sub_plat, descending=True)
            methods['A_plat_only'] = _method_result(
                sub_boxes, sub_ious, rank_a)

            # B: multiplicative S_cls × S_platform
            s_mul = sub_scores * sub_plat
            rank_b = torch.argsort(s_mul, descending=True)
            methods['B_multiplicative'] = _method_result(
                sub_boxes, sub_ious, rank_b)

            # C: log-space  log(S_cls+eps) + λ·log(S_plat+eps)
            eps = 1e-8
            lam = 1.0
            s_log = (torch.log(sub_scores + eps) +
                     lam * torch.log(sub_plat + eps))
            rank_c = torch.argsort(s_log, descending=True)
            methods['C_logspace'] = _method_result(
                sub_boxes, sub_ious, rank_c)

            # Also report if manual annotation exists
            if sub_plat_man is not None:
                # A-manual: platform-only using manual annotation ground truth
                rank_am = torch.argsort(sub_plat_man, descending=True)
                best_riou_am = float(sub_ious[rank_am[0]].item())
                methods['A_plat_manual_gt'] = dict(
                    best_riou=best_riou_am)

                # Compute K-transform error on manual frame
                ktl_err = float(torch.abs(
                    sub_plat - sub_plat_man).mean().item())
            else:
                ktl_err = None

            per_k[str(topk)] = dict(
                requested_k=int(topk),
                actual_k=actual_k,
                methods=methods,
                ktl_error=ktl_err,
            )

    row = dict(
        frame=int(frame_id),
        fname=os.path.splitext(os.path.basename(img_path))[0],
        global_max=float(scores.max().item()),
        top1_riou=float(top_ious[0].item()),
        dense_best_riou=dense_best_riou,
        dense_oracle_hit=bool(dense_best_riou >= args.riou_thr),
        candidate_count=int(scores.numel()),
        per_k=per_k,
        has_manual_platform=frame_has_manual,
    )

    # Print per-frame line: show top-1 RIoU from main head vs platform oracle methods
    riou_thr = args.riou_thr
    k200_methods = per_k.get('200', {}).get('methods', {})
    a_riou_200 = k200_methods.get('A_plat_only', {}).get('best_riou', 0.0)
    b_riou_200 = k200_methods.get('B_multiplicative', {}).get('best_riou', 0.0)
    c_riou_200 = k200_methods.get('C_logspace', {}).get('best_riou', 0.0)
    a_man_riou = None
    if k200_methods.get('A_plat_manual_gt'):
        a_man_riou = k200_methods['A_plat_manual_gt'].get('best_riou', 0.0)
    top1 = row['top1_riou']
    man_str = f" platGT_RIoU={a_man_riou:.3f}" if a_man_riou is not None else ""
    print(
        f"[{row['fname']}] top1_RIoU={top1:.3f} "
        f"K200_A(plat)={a_riou_200:.3f} "
        f"K200_B(mul)={b_riou_200:.3f} "
        f"K200_C(log)={c_riou_200:.3f}"
        f"{man_str} "
        f"dense_best={dense_best_riou:.3f}")
    return row


def _method_result(boxes, ious, rank_indices):
    """Given sorted rank indices, return best RIoU of top-ranked candidate.

    hit is computed later by build_summary using the configurable riou_thr.
    """
    best_idx = int(rank_indices[0].item())
    best_riou = float(ious[best_idx].item())
    return dict(best_riou=best_riou)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(rows, topks, riou_thr):
    """Build summary statistics across all frames."""
    total = len(rows)
    methods = ['A_plat_only', 'B_multiplicative', 'C_logspace']

    summary = dict(
        frames=total,
        riou_thr=float(riou_thr),
        dense_oracle_hits=sum(r['dense_oracle_hit'] for r in rows),
        dense_oracle_recall=(
            sum(r['dense_oracle_hit'] for r in rows) / total
            if total else 0.0),
        dense_oracle_mcml=longest_consecutive_miss(rows, 'dense_oracle_hit'),
        per_k={},
    )

    # Frame-level: compute selected RIoU for each method/K combo
    for topk in topks:
        k_str = str(topk)
        k_summary = dict(requested_k=int(topk))
        for method in methods:
            key = f'{method}'
            k_summary[method] = dict(
                hits=0, misses=0, recall=0.0, oracle_mcml=0,
                best_riou_vals=[],
            )

        for row in rows:
            pk = row['per_k'].get(k_str, {})
            for method in methods:
                m = pk.get('methods', {}).get(method, {})
                riou = m.get('best_riou', 0.0)
                hit = bool(riou >= riou_thr)
                row[f'{method}_hit_{topk}'] = hit
                k_summary[method]['best_riou_vals'].append(riou)
                if hit:
                    k_summary[method]['hits'] += 1
                else:
                    k_summary[method]['misses'] += 1

        for method in methods:
            hits = k_summary[method]['hits']
            k_summary[method]['recall'] = hits / total if total else 0.0
            hit_key = f'{method}_hit_{topk}'
            k_summary[method]['oracle_mcml'] = (
                longest_consecutive_miss(rows, hit_key))
            vals = k_summary[method]['best_riou_vals']
            k_summary[method]['best_riou_mean'] = (
                float(np.mean(vals)) if vals else 0.0)
            k_summary[method]['best_riou_min'] = (
                float(np.min(vals)) if vals else 0.0)
            k_summary[method]['best_riou_max'] = (
                float(np.max(vals)) if vals else 0.0)

        summary['per_k'][k_str] = k_summary

    # K-transform errors (only on manual frames)
    ktl_errors = []
    for row in rows:
        if row.get('has_manual_platform'):
            for topk in topks:
                err = row['per_k'].get(str(topk), {}).get('ktl_error')
                if err is not None:
                    ktl_errors.append(dict(frame=row['frame'], K=topk, error=err))
    summary['ktl_errors'] = ktl_errors

    return summary


def print_summary(summary, topks, k_params):
    riou_thr = summary['riou_thr']
    methods = ['A_plat_only', 'B_multiplicative', 'C_logspace']
    method_labels = {
        'A_plat_only': 'A (S_plat)',
        'B_multiplicative': 'B (S_cls×S_plat)',
        'C_logspace': 'C (log-space)',
    }

    print('\n' + '=' * 88)
    print('PROBE 2: PLATFORM ORACLE RERANK')
    print('=' * 88)
    print(f"frames={summary['frames']}  RIoU_thr={riou_thr:.2f}")
    print()
    print(f"K-transform fit (n={k_params['n_frames']} manual frames):")
    print(f"  k_w  = {k_params['k_w_median']:.4f} "
          f"± {k_params['k_w_std']:.4f}")
    print(f"  k_h  = {k_params['k_h_median']:.4f} "
          f"± {k_params['k_h_std']:.4f}")
    print(f"  dx   = {k_params['dx_median']:.1f} "
          f"± {k_params['dx_std']:.1f} px")
    print(f"  dy   = {k_params['dy_median']:.1f} "
          f"± {k_params['dy_std']:.1f} px")
    print()

    # Beam-only baselines
    print(f"Beam-only dense oracle: "
          f"hits={summary['dense_oracle_hits']}/{summary['frames']} "
          f"recall={summary['dense_oracle_recall']:.3f} "
          f"MCML={summary['dense_oracle_mcml']}")
    print()

    for topk in topks:
        k_str = str(topk)
        print(f"{'─' * 88}")
        print(f"  K = {topk}")
        print(f"  {'Method':<24} {'hits':>8} {'recall':>8} "
              f"{'MCML':>6} {'RIoU_mean':>10} {'RIoU_min':>8} {'RIoU_max':>8}")
        for method in methods:
            m = summary['per_k'][k_str][method]
            label = method_labels.get(method, method)
            print(f"  {label:<24} "
                  f"{m['hits']:>4d}/{summary['frames']:<5d} "
                  f"{m['recall']:>8.3f} "
                  f"{m['oracle_mcml']:>6d} "
                  f"{m['best_riou_mean']:>10.3f} "
                  f"{m['best_riou_min']:>8.3f} "
                  f"{m['best_riou_max']:>8.3f}")

    # K-transform error on manual frames
    if summary['ktl_errors']:
        ktl = summary['ktl_errors']
        errs = [e['error'] for e in ktl]
        print(f"\nK-transform error on manual frames "
              f"(mean |S_plat_ref - S_plat_manual|):")
        print(f"  mean={np.mean(errs):.6f}  median={np.median(errs):.6f}  "
              f"max={np.max(errs):.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def normalize_topks(values):
    topks = sorted({int(v) for v in values if int(v) > 0})
    if not topks:
        raise ValueError('--topks must contain at least one positive integer')
    return topks


def main():
    args = parse_args()
    topks = normalize_topks(args.topks)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Fit K-transform from manual platform annotations
    manual_json = os.path.join(PROJ_ROOT, args.manual_platform_json)
    print(f'[load] manual platforms: {manual_json}')
    manual_platforms = load_manual_platforms(manual_json)
    print(f'[load] {len(manual_platforms)} frames with manual platform annotations')

    annotated_frames_in_slice = [
        137, 144, 150, 156, 162, 169]

    print('\n' + '=' * 88)
    print('K-TRANSFORM FIT')
    print('=' * 88)
    k_params = fit_k_transform(
        manual_platforms, args.data_root, args.split, args.seq,
        annotated_frames_in_slice)
    print(f"  k_w  = {k_params['k_w_median']:.4f} "
          f"(mean={k_params['k_w_mean']:.4f} ± {k_params['k_w_std']:.4f})")
    print(f"  k_h  = {k_params['k_h_median']:.4f} "
          f"(mean={k_params['k_h_mean']:.4f} ± {k_params['k_h_std']:.4f})")
    print(f"  dx   = {k_params['dx_median']:.1f} px "
          f"(± {k_params['dx_std']:.1f})")
    print(f"  dy   = {k_params['dy_median']:.1f} px "
          f"(± {k_params['dy_std']:.1f})")
    for diag in k_params['per_frame']:
        print(f"  frame {diag['frame']:5d}: "
              f"k_w={diag['k_w']:.4f}  k_h={diag['k_h']:.4f}  "
              f"dx={diag['dx']:7.1f}  dy={diag['dy']:7.1f}")

    # Load model
    model, cfg = entry_probe.load_model(args.config, args.checkpoint, args.gpu)
    diag = entry_probe.get_diag()
    transform_compose, img_scale, flip = diag.build_test_transforms(cfg)
    seq = args.seq
    frame_ids = list(range(args.start, args.end + 1))

    print('\n' + '=' * 88)
    print('PROBE 2: PLATFORM ORACLE RERANK')
    print('=' * 88)
    print(f'config:       {args.config}')
    print(f'checkpoint:   {args.checkpoint}')
    print(f'data:         {args.split}/{seq}')
    print(f'frames:       {frame_ids[0]}..{frame_ids[-1]} ({len(frame_ids)})')
    print(f'topks:        {topks}')
    print(f'RIoU_thr:     {args.riou_thr}')

    rows = []
    for frame_id in frame_ids:
        row = analyze_frame(
            model, transform_compose, img_scale, flip,
            args, seq, frame_id, topks, k_params, manual_platforms)
        if row is not None:
            rows.append(row)

    summary = build_summary(rows, topks, args.riou_thr)
    print_summary(summary, topks, k_params)

    # Write output
    output_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = dict(
        probe='platform_oracle_rerank',
        config=args.config,
        checkpoint=args.checkpoint,
        split=args.split,
        seq=seq,
        frame_ids=frame_ids,
        topks=topks,
        riou_thr=args.riou_thr,
        k_transform_fit=k_params,
        summary=summary,
        rows=rows,
    )
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\n[out] wrote {output_path}')


if __name__ == '__main__':
    main()
