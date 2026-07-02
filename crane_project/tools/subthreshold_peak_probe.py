#!/usr/bin/env python3
"""
subthreshold_peak_probe.py

死段响应证伪协议的最小可跑实现:
  1) 导出 main / aux1 的 pre-thr, pre-NMS dense cls field;
  2) 对 GT-ROI 的 within-window max 做同尺寸背景窗口极值校准;
  3) 加定位约束, 可选角度自洽约束;
  4) 用同帧随机非抓斗 ROI 估计经验假阳率;
  5) 输出逐帧/逐 level CSV 和段级 JSON 汇总.

示例:
PYTHONPATH=. python3 crane_project/tools/subthreshold_peak_probe.py \
  --config crane_project/configs/crane_eood_k1.py \
  --checkpoint work_dirs/crane_eood_k1/epoch_24.pth \
  --seq real_seq02 --start 133 --end 171 --ok-radius 10 \
  --heads main aux1 --out-dir work_dirs/subthreshold_peak_probe/eood_k1_ep24
"""

import argparse
import csv
import json
import math
import os
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mcml_diag import (  # noqa: E402
    build_test_transforms,
    find_files,
    image_stats_bgr,
    parse_dota_ann,
    preprocess_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='CFAR-style GT-ROI subthreshold peak probe')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default='test')
    parser.add_argument('--seq', default='real_seq02')
    parser.add_argument('--start', type=int, default=133)
    parser.add_argument('--end', type=int, default=171)
    parser.add_argument('--ok-radius', type=int, default=10,
                        help='Use [start-r,start-1] and [end+1,end+r] as OK controls')
    parser.add_argument('--extra-ok-frames', default='',
                        help='Comma-separated extra OK frame ids')
    parser.add_argument('--heads', nargs='+', default=['main', 'aux1'],
                        choices=['main', 'aux1'])
    parser.add_argument('--preproc', default='none',
                        choices=[
                            'none', 'linear-brighten', 'clahe', 'gray-world',
                            'linear-clahe', 'linear-clahe-gray',
                        ])
    parser.add_argument('--linear-gain', type=float, default=2.0)
    parser.add_argument('--clahe-clip-limit', type=float, default=2.0)
    parser.add_argument('--clahe-tile-grid', type=int, default=8)
    parser.add_argument('--roi-scale', type=float, default=1.75,
                        help='GT axis-aligned center window = box size * roi_scale')
    parser.add_argument('--min-roi-cells', type=int, default=3)
    parser.add_argument('--guard-cells', type=int, default=2)
    parser.add_argument('--bg-samples', type=int, default=4096)
    parser.add_argument('--neg-samples', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--dist-scale', type=float, default=0.75,
                        help='Location pass if dist <= dist_scale * max(gt_w, gt_h)')
    parser.add_argument('--min-dist-px', type=float, default=12.0)
    parser.add_argument('--angle-thr-deg', type=float, default=30.0)
    parser.add_argument('--require-angle', action='store_true',
                        help='Make angle consistency a hard pass gate')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def scale_gt_to_img(gt: Dict, meta: Dict) -> Dict:
    """把 DOTA 标注尺度映射到当前 test pipeline 后的 img_shape 尺度。"""
    sf = meta.get('scale_factor', 1.0)
    if isinstance(sf, (list, tuple, np.ndarray)):
        sx = float(sf[0])
        sy = float(sf[1])
    else:
        sx = sy = float(sf)
    out = dict(gt)
    out['cx'] = float(gt['cx']) * sx
    out['cy'] = float(gt['cy']) * sy
    out['w'] = float(gt['w']) * sx
    out['h'] = float(gt['h']) * sy
    return out


def sigmoid_score_field(cls_feat: torch.Tensor) -> torch.Tensor:
    """Return [C,H,W] sigmoid probabilities on CPU."""
    if cls_feat.dim() == 4:
        cls_feat = cls_feat[0]
    return cls_feat.detach().float().cpu().sigmoid()


def level_roi(gt: Dict, stride: float, shape_hw: Tuple[int, int],
              roi_scale: float, min_cells: int) -> Tuple[int, int, int, int]:
    h_feat, w_feat = shape_hw
    cx_cell = float(gt['cx']) / stride
    cy_cell = float(gt['cy']) / stride
    half_w = max((float(gt['w']) * roi_scale * 0.5) / stride,
                 (min_cells - 1) * 0.5)
    half_h = max((float(gt['h']) * roi_scale * 0.5) / stride,
                 (min_cells - 1) * 0.5)
    c0 = max(0, int(math.floor(cx_cell - half_w)))
    c1 = min(w_feat - 1, int(math.ceil(cx_cell + half_w)))
    r0 = max(0, int(math.floor(cy_cell - half_h)))
    r1 = min(h_feat - 1, int(math.ceil(cy_cell + half_h)))
    return r0, r1, c0, c1


def window_shape(roi: Tuple[int, int, int, int]) -> Tuple[int, int]:
    r0, r1, c0, c1 = roi
    return r1 - r0 + 1, c1 - c0 + 1


def intersects(a: Tuple[int, int, int, int],
               b: Tuple[int, int, int, int]) -> bool:
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def valid_background_windows(shape_hw: Tuple[int, int],
                             win_hw: Tuple[int, int],
                             roi: Tuple[int, int, int, int],
                             guard_cells: int) -> List[Tuple[int, int, int, int]]:
    h_feat, w_feat = shape_hw
    win_h, win_w = win_hw
    if win_h > h_feat or win_w > w_feat:
        return []
    r0, r1, c0, c1 = roi
    guard = (
        max(0, r0 - guard_cells),
        min(h_feat - 1, r1 + guard_cells),
        max(0, c0 - guard_cells),
        min(w_feat - 1, c1 + guard_cells),
    )
    windows = []
    for top in range(0, h_feat - win_h + 1):
        bottom = top + win_h - 1
        for left in range(0, w_feat - win_w + 1):
            right = left + win_w - 1
            win = (top, bottom, left, right)
            if not intersects(win, guard):
                windows.append(win)
    return windows


def sample_windows(windows: Sequence[Tuple[int, int, int, int]],
                   n: int,
                   rng: random.Random) -> List[Tuple[int, int, int, int]]:
    if n <= 0 or not windows:
        return []
    if len(windows) <= n:
        return list(windows)
    return rng.sample(list(windows), n)


def max_in_window(score: torch.Tensor,
                  win: Tuple[int, int, int, int]) -> Tuple[float, int, int, int]:
    r0, r1, c0, c1 = win
    patch = score[:, r0:r1 + 1, c0:c1 + 1]
    flat_idx = int(torch.argmax(patch).item())
    c_count, _, w_count = patch.shape
    ch = flat_idx // ((r1 - r0 + 1) * (c1 - c0 + 1))
    rem = flat_idx % ((r1 - r0 + 1) * (c1 - c0 + 1))
    rr = rem // w_count
    cc = rem % w_count
    val = float(patch.reshape(-1)[flat_idx].item())
    return val, ch, r0 + rr, c0 + cc


def decode_at(score_ch: int, row: int, col: int, lvl: int,
              bbox_pred: torch.Tensor, anchors: torch.Tensor,
              bbox_coder, img_shape) -> Optional[Dict]:
    """Decode the anchor/bbox at the peak location for angle sanity checks."""
    if bbox_pred is None or anchors is None or bbox_coder is None:
        return None
    if bbox_pred.dim() == 4:
        bbox_pred = bbox_pred[0]
    bbox_pred = bbox_pred.detach().float().cpu()
    _, h_feat, w_feat = bbox_pred.shape
    num_anchors = bbox_pred.shape[0] // 5
    anchor_idx = int(score_ch)
    if anchor_idx >= num_anchors or row >= h_feat or col >= w_feat:
        return None
    flat_idx = (row * w_feat + col) * num_anchors + anchor_idx
    bbox_flat = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
    box_delta = bbox_flat[flat_idx:flat_idx + 1]
    anchor = anchors[flat_idx:flat_idx + 1].cpu()
    with torch.no_grad():
        decoded = bbox_coder.decode(anchor, box_delta, max_shape=img_shape)
    box = decoded[0].cpu().numpy()
    return dict(
        level=lvl, anchor_idx=anchor_idx,
        cx=float(box[0]), cy=float(box[1]),
        w=float(box[2]), h=float(box[3]),
        angle_rad=float(box[4]), angle_deg=float(np.degrees(box[4])),
    )


def angle_diff_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def binom_sf(k: int, n: int, p: float) -> float:
    if n <= 0:
        return 1.0
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return min(total, 1.0)


def head_forward(model, feats, head_name: str):
    if head_name == 'main':
        cls_scores, bbox_preds = model.bbox_head(feats)
        return model.bbox_head, cls_scores, bbox_preds
    if head_name == 'aux1':
        if getattr(model, 'aux_heads', None) is None or len(model.aux_heads) == 0:
            return None, None, None
        aux_head = model.aux_heads[0]
        if hasattr(model, '_build_aux_feats'):
            aux_feats = model._build_aux_feats(feats, aux_head)
        else:
            aux_feats = [(feat, feat) for feat in feats]
        cls_scores, bbox_preds = aux_head(aux_feats)
        return aux_head, cls_scores, bbox_preds
    raise ValueError(head_name)


def get_strides(head) -> List[float]:
    strides = head.anchor_generator.strides
    return [float(s[0] if isinstance(s, (tuple, list)) else s) for s in strides]


def analyze_level(score: torch.Tensor,
                  bbox_pred: torch.Tensor,
                  anchors: torch.Tensor,
                  bbox_coder,
                  gt: Dict,
                  img_shape,
                  lvl: int,
                  stride: float,
                  args,
                  rng: random.Random) -> Dict:
    _, h_feat, w_feat = score.shape
    roi = level_roi(gt, stride, (h_feat, w_feat),
                    args.roi_scale, args.min_roi_cells)
    win_hw = window_shape(roi)
    bg_windows_all = valid_background_windows(
        (h_feat, w_feat), win_hw, roi, args.guard_cells)
    bg_windows = sample_windows(bg_windows_all, args.bg_samples, rng)
    neg_windows = sample_windows(bg_windows_all, args.neg_samples, rng)

    roi_max, peak_ch, peak_r, peak_c = max_in_window(score, roi)
    bg_maxes = np.array(
        [max_in_window(score, win)[0] for win in bg_windows],
        dtype=np.float64)
    if bg_maxes.size:
        p_frame = float((np.sum(bg_maxes >= roi_max) + 1) / (bg_maxes.size + 1))
        bg_mean = float(bg_maxes.mean())
        bg_std = float(bg_maxes.std())
        bg_q95 = float(np.quantile(bg_maxes, 0.95))
        bg_q99 = float(np.quantile(bg_maxes, 0.99))
    else:
        p_frame = 1.0
        bg_mean = bg_std = bg_q95 = bg_q99 = float('nan')

    neg_ps = []
    neg_pass = 0
    for win in neg_windows:
        neg_max, _, _, _ = max_in_window(score, win)
        if bg_maxes.size:
            p_neg = float((np.sum(bg_maxes >= neg_max) + 1) /
                          (bg_maxes.size + 1))
        else:
            p_neg = 1.0
        neg_ps.append(p_neg)
        if p_neg < args.alpha:
            neg_pass += 1

    peak_x = (peak_c + 0.5) * stride
    peak_y = (peak_r + 0.5) * stride
    loc_dist = float(math.hypot(peak_x - gt['cx'], peak_y - gt['cy']))
    dist_thr = max(args.min_dist_px, args.dist_scale * max(gt['w'], gt['h']))
    loc_pass = loc_dist <= dist_thr
    sig_pass = p_frame < args.alpha

    decoded = decode_at(peak_ch, peak_r, peak_c, lvl, bbox_pred, anchors,
                        bbox_coder, img_shape)
    angle_diff = float('nan')
    angle_pass = True
    pred_angle = float('nan')
    if decoded is not None:
        pred_angle = decoded['angle_deg']
        angle_diff = angle_diff_deg(pred_angle, gt['angle'])
        angle_pass = angle_diff <= args.angle_thr_deg

    pass_all = sig_pass and loc_pass and (angle_pass or not args.require_angle)

    return dict(
        level=lvl, stride=stride, feat_h=h_feat, feat_w=w_feat,
        roi_r0=roi[0], roi_r1=roi[1], roi_c0=roi[2], roi_c1=roi[3],
        roi_h=win_hw[0], roi_w=win_hw[1], roi_cells=win_hw[0] * win_hw[1],
        bg_windows=len(bg_windows), bg_windows_total=len(bg_windows_all),
        roi_max=roi_max, bg_max_mean=bg_mean, bg_max_std=bg_std,
        bg_max_q95=bg_q95, bg_max_q99=bg_q99, p_frame=p_frame,
        peak_ch=peak_ch, peak_r=peak_r, peak_c=peak_c,
        peak_x=peak_x, peak_y=peak_y, loc_dist=loc_dist,
        loc_dist_thr=dist_thr, pred_angle=pred_angle,
        angle_diff_to_gt=angle_diff, sig_pass=sig_pass,
        loc_pass=loc_pass, angle_pass=angle_pass,
        pass_all=pass_all, neg_samples=len(neg_windows),
        neg_pass=neg_pass,
        neg_pass_rate=(neg_pass / len(neg_windows)) if neg_windows else float('nan'),
    )


def frame_ids_from_args(args) -> List[Tuple[int, str]]:
    ids = []
    for fid in range(args.start, args.end + 1):
        ids.append((fid, 'dead'))
    for fid in range(args.start - args.ok_radius, args.start):
        if fid >= 0:
            ids.append((fid, 'ok_pre'))
    for fid in range(args.end + 1, args.end + args.ok_radius + 1):
        ids.append((fid, 'ok_post'))
    if args.extra_ok_frames.strip():
        for part in args.extra_ok_frames.split(','):
            part = part.strip()
            if part:
                ids.append((int(part), 'ok_extra'))
    seen = set()
    out = []
    for fid, role in ids:
        key = (fid, role)
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows_frame: List[Dict], args) -> Dict:
    summary = dict(alpha=args.alpha, require_angle=args.require_angle, groups={})
    keys = sorted(set((r['head'], r['role']) for r in rows_frame))
    for head, role in keys:
        subset = [r for r in rows_frame if r['head'] == head and r['role'] == role]
        total = len(subset)
        passed = sum(1 for r in subset if r['pass_all'])
        neg_total = sum(int(r.get('neg_samples_total', 0)) for r in subset)
        neg_pass = sum(int(r.get('neg_pass_total', 0)) for r in subset)
        empirical_p0 = (
            neg_pass / neg_total if neg_total > 0 else args.alpha)
        empirical_p0_floor = (
            max(empirical_p0, 1.0 / (neg_total + 1)) if neg_total > 0
            else args.alpha)
        summary['groups'][f'{head}/{role}'] = dict(
            n=total, k=passed,
            pass_rate=(passed / total) if total else 0.0,
            neg_samples=neg_total, neg_pass=neg_pass,
            neg_pass_rate=empirical_p0 if neg_total else None,
            binom_p_nominal_alpha=binom_sf(passed, total, args.alpha),
            binom_p_empirical_neg=binom_sf(passed, total, empirical_p0_floor),
            mean_best_roi_max=float(np.mean([r['best_roi_max'] for r in subset])) if subset else None,
            mean_best_p=float(np.mean([r['best_p_frame'] for r in subset])) if subset else None,
            mean_best_loc_dist=float(np.mean([r['best_loc_dist'] for r in subset])) if subset else None,
        )
    return summary


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    from mmcv import Config
    from mmrotate.models import build_detector

    cfg = Config.fromfile(args.config)
    cfg.model.test_cfg.score_thr = 0.0
    cfg.model.test_cfg.max_per_img = 100
    model = build_detector(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model = model.to(device)
    model.eval()

    transform_compose, img_scale, flip = build_test_transforms(cfg)
    preproc_cfg = dict(
        mode=args.preproc,
        linear_gain=args.linear_gain,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=args.clahe_tile_grid,
    )

    rows_level = []
    rows_frame = []
    frame_items = frame_ids_from_args(args)
    print(f'[probe] frames={len(frame_items)} seq={args.seq} '
          f'dead=[{args.start},{args.end}] heads={args.heads} '
          f'preproc={args.preproc} out={args.out_dir}')

    with torch.no_grad():
        for fid, role in frame_items:
            img_path, ann_path = find_files(args.data_root, args.split, args.seq, fid)
            if img_path is None:
                print(f'[skip] frame={fid:05d} role={role}: image not found')
                continue
            gts = parse_dota_ann(ann_path)
            if not gts:
                print(f'[skip] frame={fid:05d} role={role}: GT not found')
                continue
            img_tensor, meta, img_stats = preprocess_image(
                img_path, transform_compose, img_scale, flip,
                preproc_cfg=preproc_cfg)
            if img_tensor is None:
                print(f'[skip] frame={fid:05d} role={role}: preprocess failed')
                continue
            gt = scale_gt_to_img(gts[0], meta)
            img_tensor = img_tensor.to(device)
            feats = model.extract_feat(img_tensor)

            for head_name in args.heads:
                head, cls_scores, bbox_preds = head_forward(model, feats, head_name)
                if head is None:
                    print(f'[skip] frame={fid:05d}: head {head_name} unavailable')
                    continue
                strides = get_strides(head)
                anchors_per_level = head.anchor_generator.grid_priors(
                    [s.shape[-2:] for s in cls_scores], device='cpu')

                level_results = []
                for lvl, cls_feat in enumerate(cls_scores):
                    score = sigmoid_score_field(cls_feat)
                    result = analyze_level(
                        score=score,
                        bbox_pred=bbox_preds[lvl],
                        anchors=anchors_per_level[lvl],
                        bbox_coder=head.bbox_coder,
                        gt=gt,
                        img_shape=meta['img_shape'],
                        lvl=lvl,
                        stride=strides[lvl],
                        args=args,
                        rng=rng)
                    row = dict(
                        frame=fid, role=role, split=args.split, seq=args.seq,
                        head=head_name, img_path=img_path,
                        raw_brightness=img_stats['raw_brightness'],
                        raw_contrast=img_stats['raw_contrast'],
                        raw_ud_delta=img_stats['raw_ud_delta'],
                        proc_brightness=img_stats['proc_brightness'],
                        proc_contrast=img_stats['proc_contrast'],
                        proc_ud_delta=img_stats['proc_ud_delta'],
                        gt_cx=gt['cx'], gt_cy=gt['cy'],
                        gt_w=gt['w'], gt_h=gt['h'], gt_angle=gt['angle'],
                        **result)
                    rows_level.append(row)
                    level_results.append(row)

                # 选该 head/frame 最有利的 level: pass 优先, 其次 p 最小, 再 roi_max 最大。
                best = sorted(
                    level_results,
                    key=lambda r: (
                        not bool(r['pass_all']),
                        float(r['p_frame']),
                        -float(r['roi_max'])),
                )[0]
                rows_frame.append(dict(
                    frame=fid, role=role, split=args.split, seq=args.seq,
                    head=head_name, pass_all=bool(best['pass_all']),
                    best_level=best['level'], best_stride=best['stride'],
                    best_roi_max=best['roi_max'],
                    best_p_frame=best['p_frame'],
                    best_loc_dist=best['loc_dist'],
                    best_loc_dist_thr=best['loc_dist_thr'],
                    best_pred_angle=best['pred_angle'],
                    best_angle_diff_to_gt=best['angle_diff_to_gt'],
                    best_sig_pass=bool(best['sig_pass']),
                    best_loc_pass=bool(best['loc_pass']),
                    best_angle_pass=bool(best['angle_pass']),
                    neg_samples_total=sum(r['neg_samples'] for r in level_results),
                    neg_pass_total=sum(r['neg_pass'] for r in level_results),
                    raw_brightness=img_stats['raw_brightness'],
                    raw_contrast=img_stats['raw_contrast'],
                    raw_ud_delta=img_stats['raw_ud_delta'],
                    proc_brightness=img_stats['proc_brightness'],
                    proc_contrast=img_stats['proc_contrast'],
                    proc_ud_delta=img_stats['proc_ud_delta'],
                    gt_cx=gt['cx'], gt_cy=gt['cy'],
                    gt_w=gt['w'], gt_h=gt['h'], gt_angle=gt['angle'],
                ))
                print(
                    f'[frame] {head_name:4s} {role:7s} {args.seq}_{fid:05d} '
                    f'pass={int(best["pass_all"])} P{best["level"]} '
                    f'roi={best["roi_max"]:.5f} p={best["p_frame"]:.4g} '
                    f'd={best["loc_dist"]:.1f}/{best["loc_dist_thr"]:.1f} '
                    f'neg={sum(r["neg_pass"] for r in level_results)}/'
                    f'{sum(r["neg_samples"] for r in level_results)}')

    level_fields = [
        'frame', 'role', 'split', 'seq', 'head', 'level', 'stride',
        'roi_max', 'p_frame', 'loc_dist', 'loc_dist_thr', 'pass_all',
        'sig_pass', 'loc_pass', 'angle_pass', 'pred_angle',
        'angle_diff_to_gt', 'bg_max_mean', 'bg_max_std', 'bg_max_q95',
        'bg_max_q99', 'bg_windows', 'bg_windows_total', 'neg_samples',
        'neg_pass', 'neg_pass_rate', 'peak_ch', 'peak_r', 'peak_c',
        'peak_x', 'peak_y', 'roi_r0', 'roi_r1', 'roi_c0', 'roi_c1',
        'roi_h', 'roi_w', 'roi_cells', 'feat_h', 'feat_w',
        'raw_brightness', 'raw_contrast', 'raw_ud_delta',
        'proc_brightness', 'proc_contrast', 'proc_ud_delta',
        'gt_cx', 'gt_cy', 'gt_w', 'gt_h', 'gt_angle', 'img_path',
    ]
    frame_fields = [
        'frame', 'role', 'split', 'seq', 'head', 'pass_all',
        'best_level', 'best_stride', 'best_roi_max', 'best_p_frame',
        'best_loc_dist', 'best_loc_dist_thr', 'best_pred_angle',
        'best_angle_diff_to_gt', 'best_sig_pass', 'best_loc_pass',
        'best_angle_pass', 'neg_samples_total', 'neg_pass_total',
        'raw_brightness', 'raw_contrast', 'raw_ud_delta',
        'proc_brightness', 'proc_contrast', 'proc_ud_delta',
        'gt_cx', 'gt_cy', 'gt_w', 'gt_h', 'gt_angle',
    ]
    write_csv(os.path.join(args.out_dir, 'per_level.csv'),
              rows_level, level_fields)
    write_csv(os.path.join(args.out_dir, 'per_frame.csv'),
              rows_frame, frame_fields)
    summary = summarize(rows_frame, args)
    summary['config'] = args.config
    summary['checkpoint'] = args.checkpoint
    summary['preproc'] = args.preproc
    summary['seq'] = args.seq
    summary['dead_range'] = [args.start, args.end]
    summary['note'] = (
        'Positive dead GT-ROI passes can unlock field re-anchoring. '
        'Negative dead results are non-detections only until sim power '
        'calibration proves the test can detect weak GT peaks.')
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('\n[summary]')
    for key, value in summary['groups'].items():
        print(f'  {key}: k={value["k"]}/{value["n"]} '
              f'neg={value["neg_pass"]}/{value["neg_samples"]} '
              f'p_nom={value["binom_p_nominal_alpha"]:.4g} '
              f'p_emp={value["binom_p_empirical_neg"]:.4g}')
    print(f'[done] wrote {args.out_dir}/per_level.csv, '
          f'{args.out_dir}/per_frame.csv, {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
