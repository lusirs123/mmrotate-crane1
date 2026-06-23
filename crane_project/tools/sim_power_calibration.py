#!/usr/bin/env python3
"""
sim_power_calibration.py

§14.4 功效标定:
  1) 从真实 seq02 死段拟合光度签名终点 TGT;
  2) 对 sim 帧做确定性 δ 阶梯退化, 几何不动;
  3) 每个 δ 复用 subthreshold_peak_probe.py 的同一套 ROI 极值检验;
  4) 输出 power(delta), global_max(delta), 以及真实中段的 delta 投影.

示例:
PYTHONPATH=. python3 crane_project/tools/sim_power_calibration.py \
  --config crane_project/configs/crane_symeood_m2_equi_degraded_cls.py \
  --checkpoint work_dirs/crane_symeood_m2_equi_degraded_cls/epoch_22.pth \
  --real-seq real_seq02 --real-start 138 --real-end 171 \
  --sim-seq sim_seq09 --sim-sample 120 \
  --head aux1 --device cuda:0 \
  --out-dir work_dirs/subthreshold_peak_probe/sim_power_degraded_cls_aux1
"""

import argparse
import csv
import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from mcml_diag import (  # noqa: E402
    build_test_transforms,
    find_files,
    parse_dota_ann,
)
from subthreshold_peak_probe import (  # noqa: E402
    analyze_level,
    get_strides,
    head_forward,
    scale_gt_to_img,
    sigmoid_score_field,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Deterministic sim degradation power calibration')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--real-split', default='test')
    parser.add_argument('--real-seq', default='real_seq02')
    parser.add_argument('--real-start', type=int, default=138)
    parser.add_argument('--real-end', type=int, default=171)
    parser.add_argument('--project-start', type=int, default=138,
                        help='Real frames projected onto the fitted severity axis')
    parser.add_argument('--project-end', type=int, default=161)
    parser.add_argument('--sim-split', default='test')
    parser.add_argument('--sim-seq', default='sim_seq09')
    parser.add_argument('--sim-start', type=int, default=None)
    parser.add_argument('--sim-end', type=int, default=None)
    parser.add_argument('--sim-sample', type=int, default=120,
                        help='Deterministic sample count when start/end is omitted')
    parser.add_argument('--deltas', default='0:1.3:0.1',
                        help='Either start:end:step or comma-separated values')
    parser.add_argument('--head', default='aux1', choices=['main', 'aux1'])
    parser.add_argument('--roi-scale', type=float, default=1.75)
    parser.add_argument('--min-roi-cells', type=int, default=3)
    parser.add_argument('--guard-cells', type=int, default=2)
    parser.add_argument('--bg-samples', type=int, default=2048)
    parser.add_argument('--neg-samples', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--dist-scale', type=float, default=0.75)
    parser.add_argument('--min-dist-px', type=float, default=12.0)
    parser.add_argument('--angle-thr-deg', type=float, default=30.0)
    parser.add_argument('--require-angle', action='store_true')
    parser.add_argument('--power-thr', type=float, default=0.8)
    parser.add_argument('--global-thr', type=float, default=0.05,
                        help='Mean global_max threshold marking subthreshold regime')
    parser.add_argument('--knee', type=float, default=0.85)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-degraded-sample', action='store_true')
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def parse_deltas(spec: str) -> List[float]:
    if ':' in spec:
        start, end, step = [float(x) for x in spec.split(':')]
        vals = []
        x = start
        # Include the endpoint with a small tolerance.
        while x <= end + step * 0.5:
            vals.append(round(x, 6))
            x += step
        return vals
    return [float(x.strip()) for x in spec.split(',') if x.strip()]


def frame_ids_from_ann(data_root: str, split: str, seq: str,
                       start: Optional[int], end: Optional[int],
                       sample: int, seed: int) -> List[int]:
    if start is not None and end is not None:
        return list(range(start, end + 1))
    ann_dir = os.path.join(data_root, split, 'annfiles')
    files = []
    if os.path.isdir(ann_dir):
        files = [f for f in os.listdir(ann_dir)
                 if f.startswith(seq + '_') and f.endswith('.txt')]
    ids = []
    for name in files:
        m = re.search(r'_(\d{5})\.txt$', name)
        if m:
            ids.append(int(m.group(1)))
    ids = sorted(ids)
    if sample and len(ids) > sample:
        rng = random.Random(seed)
        ids = sorted(rng.sample(ids, sample))
    return ids


def signature_rgb(img_rgb: np.ndarray) -> Dict[str, float]:
    img = img_rgb.astype(np.float32)
    light = img.mean(axis=2) / 255.0
    h = light.shape[0]
    top = float(light[:max(1, h // 3)].mean())
    bot = float(light[max(0, h - h // 3):].mean())
    r_mean = float(img[..., 0].mean())
    g_mean = float(img[..., 1].mean())
    return dict(
        mean_L=float(light.mean()),
        UD=top / max(bot, 1e-6),
        RG=r_mean / max(g_mean, 1e-6),
        contrast=float(light.std()),
    )


def load_rgb(path: str) -> Optional[np.ndarray]:
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fit_real_target(args) -> Tuple[Dict[str, float], List[Dict]]:
    sigs = []
    rows = []
    for fid in range(args.real_start, args.real_end + 1):
        img_path, _ = find_files(args.data_root, args.real_split,
                                 args.real_seq, fid)
        if img_path is None:
            continue
        img = load_rgb(img_path)
        if img is None:
            continue
        sig = signature_rgb(img)
        sigs.append(sig)
        rows.append(dict(frame=fid, split=args.real_split, seq=args.real_seq,
                         img_path=img_path, **sig))
    if not sigs:
        raise RuntimeError('No real frames available for target signature fit')
    target = {
        key: float(np.median([sig[key] for sig in sigs]))
        for key in ('mean_L', 'UD', 'RG', 'contrast')
    }
    return target, rows


def degrade_rgb(img_rgb: np.ndarray, delta: float, clean_sig: Dict[str, float],
                target: Dict[str, float], knee: float = 0.85) -> np.ndarray:
    """确定性整帧光度退化, 只改像素不改几何。"""
    x = img_rgb.astype(np.float32) / 255.0
    h = x.shape[0]

    k1 = target['contrast'] / max(clean_sig['contrast'], 1e-6)
    k = 1.0 + (k1 - 1.0) * delta
    mu = x.mean(axis=(0, 1), keepdims=True)
    x = mu + (x - mu) * k

    ud1 = target['UD']
    ramp01 = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
    mult = ud1 * ramp01 + 1.0 * (1.0 - ramp01)
    mult = mult / max(float(mult.mean()), 1e-6)
    mult = 1.0 + (mult - 1.0) * delta
    x = x * mult[..., None]

    a1 = target['mean_L'] / max(clean_sig['mean_L'], 1e-6)
    x = x * (1.0 + (a1 - 1.0) * delta)

    rg1 = target['RG'] / max(clean_sig['RG'], 1e-6)
    c_r = 1.0 + (rg1 - 1.0) * delta
    x[..., 0] *= c_r

    hi = x > knee
    x[hi] = knee + (1.0 - knee) * np.tanh((x[hi] - knee) / (1.0 - knee))
    return np.clip(x, 0.0, 1.0)


def preprocess_rgb_array(img_rgb: np.ndarray, transform_compose, flip=False):
    """与 mcml_diag.preprocess_image 同结构, 但输入是内存 RGB 图。"""
    ori_h, ori_w = img_rgb.shape[:2]
    results = dict(
        img=img_rgb,
        filename='<memory>',
        ori_filename='<memory>',
        img_shape=img_rgb.shape,
        ori_shape=img_rgb.shape,
        pad_shape=img_rgb.shape,
        scale_factor=1.0,
        flip=flip,
        flip_direction='horizontal' if flip else None,
        img_norm_cfg=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        img_fields=['img'],
    )
    results = transform_compose(results)
    img_tensor = results['img']
    if hasattr(img_tensor, 'data'):
        img_tensor = img_tensor.data
    if not isinstance(img_tensor, torch.Tensor):
        img_tensor = torch.from_numpy(img_tensor)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_metas = dict(
        filename='<memory>',
        ori_shape=results.get('ori_shape', (ori_h, ori_w, 3)),
        img_shape=results.get('img_shape', img_tensor.shape[2:]),
        pad_shape=results.get('pad_shape', img_tensor.shape[2:]),
        scale_factor=results.get('scale_factor', 1.0),
        flip=flip,
        flip_direction=None,
    )
    return img_tensor, img_metas


def severity_projection(sig: Dict[str, float],
                        clean_sig: Dict[str, float],
                        target: Dict[str, float]) -> float:
    """把签名投影到 clean->target 轴上; 只作为严重度指数。"""
    vals = []
    for key in ('mean_L', 'UD', 'RG', 'contrast'):
        denom = target[key] - clean_sig[key]
        if abs(denom) > 1e-6:
            vals.append((sig[key] - clean_sig[key]) / denom)
    if not vals:
        return float('nan')
    return float(np.median(vals))


def analyze_one_image(img_rgb: np.ndarray, gt_raw: Dict, model,
                      transform_compose, flip, head_name: str, args,
                      rng: random.Random, device) -> Tuple[List[Dict], Dict]:
    img_tensor, meta = preprocess_rgb_array(img_rgb, transform_compose, flip)
    gt = scale_gt_to_img(gt_raw, meta)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        feats = model.extract_feat(img_tensor)
        head, cls_scores, bbox_preds = head_forward(model, feats, head_name)
        if head is None:
            raise RuntimeError(f'Head {head_name} unavailable')
        strides = get_strides(head)
        anchors_per_level = head.anchor_generator.grid_priors(
            [s.shape[-2:] for s in cls_scores], device='cpu')

        level_rows = []
        global_max = 0.0
        for lvl, cls_feat in enumerate(cls_scores):
            score = sigmoid_score_field(cls_feat)
            global_max = max(global_max, float(score.max().item()))
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
            level_rows.append(result)
    for row in level_rows:
        row['global_max'] = global_max
    best = sorted(
        level_rows,
        key=lambda r: (
            not bool(r['pass_all']),
            float(r['p_frame']),
            -float(r['roi_max'])),
    )[0]
    best = dict(best)
    best['global_max'] = global_max
    return level_rows, best


def write_rows(path: str, rows: List[Dict], fields: Sequence[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float('nan'), float('nan')
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    target, real_fit_rows = fit_real_target(args)
    write_rows(
        os.path.join(args.out_dir, 'real_fit_signature.csv'),
        real_fit_rows,
        ['frame', 'split', 'seq', 'mean_L', 'UD', 'RG', 'contrast', 'img_path'])

    from mmcv import Config
    from mmrotate.models import build_detector

    cfg = Config.fromfile(args.config)
    cfg.model.test_cfg.score_thr = 0.0
    cfg.model.test_cfg.max_per_img = 100
    model = build_detector(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == 'cpu'
        else 'cpu')
    model = model.to(device)
    model.eval()
    transform_compose, _, flip = build_test_transforms(cfg)

    sim_ids = frame_ids_from_ann(
        args.data_root, args.sim_split, args.sim_seq,
        args.sim_start, args.sim_end, args.sim_sample, args.seed)
    deltas = parse_deltas(args.deltas)
    print(f'[power] sim_frames={len(sim_ids)} deltas={deltas} '
          f'head={args.head} target={target}')

    per_frame_rows = []
    per_level_rows = []
    power_rows = []

    sample_dir = os.path.join(args.out_dir, 'degraded_samples')
    if args.save_degraded_sample:
        os.makedirs(sample_dir, exist_ok=True)

    for delta in deltas:
        pass_count = 0
        global_vals = []
        neg_pass = 0
        neg_total = 0
        used = 0
        for fid in sim_ids:
            img_path, ann_path = find_files(
                args.data_root, args.sim_split, args.sim_seq, fid)
            if img_path is None:
                continue
            gts = parse_dota_ann(ann_path)
            if not gts:
                continue
            img_rgb = load_rgb(img_path)
            if img_rgb is None:
                continue
            clean_sig = signature_rgb(img_rgb)
            degraded = degrade_rgb(img_rgb, delta, clean_sig, target, args.knee)
            degraded_u8 = np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
            deg_sig = signature_rgb(degraded_u8)
            if args.save_degraded_sample and used < 3:
                out_img = cv2.cvtColor(degraded_u8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(sample_dir, f'd{delta:.2f}_{args.sim_seq}_{fid:05d}.jpg'),
                    out_img)

            level_rows, best = analyze_one_image(
                degraded_u8, gts[0], model, transform_compose, flip,
                args.head, args, rng, device)
            used += 1
            pass_count += int(best['pass_all'])
            global_vals.append(best['global_max'])
            neg_pass += sum(int(r['neg_pass']) for r in level_rows)
            neg_total += sum(int(r['neg_samples']) for r in level_rows)

            base = dict(
                delta=delta, frame=fid, split=args.sim_split, seq=args.sim_seq,
                head=args.head, img_path=img_path,
                clean_mean_L=clean_sig['mean_L'],
                clean_UD=clean_sig['UD'],
                clean_RG=clean_sig['RG'],
                clean_contrast=clean_sig['contrast'],
                degraded_mean_L=deg_sig['mean_L'],
                degraded_UD=deg_sig['UD'],
                degraded_RG=deg_sig['RG'],
                degraded_contrast=deg_sig['contrast'],
            )
            per_frame_rows.append(dict(
                **base,
                pass_all=bool(best['pass_all']),
                best_level=best['level'],
                best_roi_max=best['roi_max'],
                best_p_frame=best['p_frame'],
                best_loc_dist=best['loc_dist'],
                best_loc_dist_thr=best['loc_dist_thr'],
                best_pred_angle=best['pred_angle'],
                best_angle_diff_to_gt=best['angle_diff_to_gt'],
                global_max=best['global_max'],
                neg_samples_total=sum(int(r['neg_samples']) for r in level_rows),
                neg_pass_total=sum(int(r['neg_pass']) for r in level_rows),
            ))
            for row in level_rows:
                per_level_rows.append(dict(**base, **row))

        power = pass_count / used if used else 0.0
        ci_lo, ci_hi = wilson_ci(pass_count, used)
        power_rows.append(dict(
            delta=delta, n=used, k=pass_count, power=power,
            power_ci_low=ci_lo, power_ci_high=ci_hi,
            mean_global_max=float(np.mean(global_vals)) if global_vals else None,
            median_global_max=float(np.median(global_vals)) if global_vals else None,
            neg_samples=neg_total, neg_pass=neg_pass,
            neg_pass_rate=(neg_pass / neg_total) if neg_total else None,
        ))
        print(f'[delta] {delta:.2f}: power={pass_count}/{used}={power:.3f} '
              f'gmax={np.mean(global_vals) if global_vals else float("nan"):.4f} '
              f'neg={neg_pass}/{neg_total}')

    project_rows = []
    # 用 sim clean 签名中位数作为 clean anchor, 只用于严重度投影。
    clean_sigs = []
    for fid in sim_ids[:max(1, min(len(sim_ids), 64))]:
        img_path, _ = find_files(args.data_root, args.sim_split, args.sim_seq, fid)
        img_rgb = load_rgb(img_path) if img_path else None
        if img_rgb is not None:
            clean_sigs.append(signature_rgb(img_rgb))
    clean_anchor = {
        key: float(np.median([sig[key] for sig in clean_sigs]))
        for key in ('mean_L', 'UD', 'RG', 'contrast')
    } if clean_sigs else None

    if clean_anchor is not None:
        for fid in range(args.project_start, args.project_end + 1):
            img_path, _ = find_files(args.data_root, args.real_split,
                                     args.real_seq, fid)
            img_rgb = load_rgb(img_path) if img_path else None
            if img_rgb is None:
                continue
            sig = signature_rgb(img_rgb)
            project_rows.append(dict(
                frame=fid, split=args.real_split, seq=args.real_seq,
                delta_proj=severity_projection(sig, clean_anchor, target),
                img_path=img_path, **sig))

    write_rows(
        os.path.join(args.out_dir, 'power_curve.csv'),
        power_rows,
        ['delta', 'n', 'k', 'power', 'power_ci_low', 'power_ci_high',
         'mean_global_max', 'median_global_max',
         'neg_samples', 'neg_pass', 'neg_pass_rate'])
    write_rows(
        os.path.join(args.out_dir, 'per_frame.csv'),
        per_frame_rows,
        ['delta', 'frame', 'split', 'seq', 'head', 'pass_all',
         'best_level', 'best_roi_max', 'best_p_frame',
         'best_loc_dist', 'best_loc_dist_thr', 'best_pred_angle',
         'best_angle_diff_to_gt', 'global_max',
         'neg_samples_total', 'neg_pass_total',
         'clean_mean_L', 'clean_UD', 'clean_RG', 'clean_contrast',
         'degraded_mean_L', 'degraded_UD', 'degraded_RG',
         'degraded_contrast', 'img_path'])
    write_rows(
        os.path.join(args.out_dir, 'per_level.csv'),
        per_level_rows,
        ['delta', 'frame', 'split', 'seq', 'head', 'level', 'stride',
         'roi_max', 'p_frame', 'loc_dist', 'loc_dist_thr', 'pass_all',
         'sig_pass', 'loc_pass', 'angle_pass', 'pred_angle',
         'angle_diff_to_gt', 'global_max',
         'bg_max_mean', 'bg_max_std', 'bg_max_q95', 'bg_max_q99',
         'bg_windows', 'neg_samples', 'neg_pass',
         'peak_ch', 'peak_r', 'peak_c', 'peak_x', 'peak_y',
         'clean_mean_L', 'clean_UD', 'clean_RG', 'clean_contrast',
         'degraded_mean_L', 'degraded_UD', 'degraded_RG',
         'degraded_contrast', 'img_path'])
    write_rows(
        os.path.join(args.out_dir, 'real_delta_projection.csv'),
        project_rows,
        ['frame', 'split', 'seq', 'delta_proj',
         'mean_L', 'UD', 'RG', 'contrast', 'img_path'])

    delta_star = None
    for row in power_rows:
        if row['power'] < args.power_thr:
            delta_star = row['delta']
            break
    subthr_delta = None
    for row in power_rows:
        mg = row['mean_global_max']
        if mg is not None and mg < args.global_thr:
            subthr_delta = row['delta']
            break
    summary = dict(
        config=args.config,
        checkpoint=args.checkpoint,
        head=args.head,
        target_signature=target,
        clean_anchor_signature=clean_anchor,
        deltas=deltas,
        power_threshold=args.power_thr,
        delta_star_power_below_threshold=delta_star,
        global_threshold=args.global_thr,
        delta_global_max_below_threshold=subthr_delta,
        real_fit_range=[args.real_start, args.real_end],
        real_projection_range=[args.project_start, args.project_end],
        sim_seq=args.sim_seq,
        sim_frames=len(sim_ids),
        note=(
            'delta_proj is a severity index on the sim-clean to fitted-real '
            'signature axis, not an exact physical inverse. Negative real '
            'findings should be treated as true blind only when delta_proj is '
            'below the calibrated loss-of-power boundary.')
    )
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[done] wrote {args.out_dir}/power_curve.csv, per_frame.csv, '
          f'per_level.csv, real_delta_projection.csv, summary.json')


if __name__ == '__main__':
    main()
