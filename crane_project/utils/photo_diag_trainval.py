#!/usr/bin/env python3
"""
photo_diag_trainval.py — 基于 train + val 的 photometric 分布统计。

用途: 为 L_invar 的 T_photo 参数提供数据依据。
数据源: 仅 train + train_sim + val (不含 test，避免泄露)。
test 的诊断结果在 photo_diag.py 中保留，仅用于 error analysis。

Run:
    cd /Users/mac/Documents/paper/symEOOD
    python3 crane_project/utils/photo_diag_trainval.py
"""

import os
import re
import glob
from collections import defaultdict

import cv2
import numpy as np


# =====================================================================
# 配置: 只扫描 train / val
# =====================================================================
IMG_DIRS = [
    'crane_project/data/crane_grab/train/images/',
    'crane_project/data/crane_grab/val/images/',
]


# =====================================================================
# 工具函数 (与 photo_diag.py 共用)
# =====================================================================

def parse_seq_frame(basename):
    m = re.match(r'^(real|sim)_(.+?)_(\d+)$', basename)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, None


def compute_photo_features(img_bgr):
    H, W = img_bgr.shape[:2]
    B, G, R = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    eps = 1e-6

    gray = 0.114 * B.astype(np.float64) + 0.587 * G.astype(np.float64) + 0.299 * R.astype(np.float64)
    mean_brightness = float(gray.mean())
    contrast = float(gray.std())

    R_mean = float(R.astype(np.float64).mean()) + eps
    G_mean = float(G.astype(np.float64).mean()) + eps
    B_mean = float(B.astype(np.float64).mean()) + eps

    R_G_ratio = R_mean / G_mean
    B_G_ratio = B_mean / G_mean

    left = img_bgr[:, :W // 2].astype(np.float64)
    right = img_bgr[:, W // 2:].astype(np.float64)
    left_gray = (0.114 * left[:, :, 0] + 0.587 * left[:, :, 1] + 0.299 * left[:, :, 2]).mean()
    right_gray = (0.114 * right[:, :, 0] + 0.587 * right[:, :, 1] + 0.299 * right[:, :, 2]).mean()
    lr_ratio = float(left_gray / (right_gray + eps))

    top = img_bgr[:H // 2].astype(np.float64)
    bottom = img_bgr[H // 2:].astype(np.float64)
    top_gray = (0.114 * top[:, :, 0] + 0.587 * top[:, :, 1] + 0.299 * top[:, :, 2]).mean()
    bot_gray = (0.114 * bottom[:, :, 0] + 0.587 * bottom[:, :, 1] + 0.299 * bottom[:, :, 2]).mean()
    ud_ratio = float(top_gray / (bot_gray + eps))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue_mean = float(hsv[:, :, 0].mean())
    sat_mean = float(hsv[:, :, 1].mean())

    return dict(
        mean_brightness=mean_brightness,
        contrast=contrast,
        R_G_ratio=R_G_ratio,
        B_G_ratio=B_G_ratio,
        lr_ratio=lr_ratio,
        ud_ratio=ud_ratio,
        hue_mean=hue_mean,
        sat_mean=sat_mean,
    )


def group_stats(features_list):
    if not features_list:
        return {}
    keys = features_list[0].keys()
    stats = {}
    for k in keys:
        vals = np.array([f[k] for f in features_list])
        stats[k] = dict(
            mean=float(vals.mean()),
            std=float(vals.std()),
            min=float(vals.min()),
            max=float(vals.max()),
            p5=float(np.percentile(vals, 5)),
            p25=float(np.percentile(vals, 25)),
            p50=float(np.percentile(vals, 50)),
            p75=float(np.percentile(vals, 75)),
            p95=float(np.percentile(vals, 95)),
        )
    return stats


# =====================================================================
# 主逻辑
# =====================================================================

def main():
    groups = defaultdict(list)

    for img_dir in IMG_DIRS:
        all_imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        for img_path in all_imgs:
            bn = os.path.splitext(os.path.basename(img_path))[0]
            domain, seq_id, fid = parse_seq_frame(bn)
            if domain is None:
                continue

            seq_key = f'{domain}_{seq_id}'
            subset = 'train' if 'train' in img_dir else 'val'
            group_key = f'{subset}_{seq_key}'

            img = cv2.imread(img_path)
            if img is None:
                continue
            features = compute_photo_features(img)
            groups[group_key].append(features)

    # 汇总
    print('=' * 100)
    print('PHOTOMETRIC DISTRIBUTION: TRAIN + VAL ONLY (no test data)')
    print('=' * 100)

    key_features = ['mean_brightness', 'contrast', 'R_G_ratio', 'B_G_ratio',
                    'lr_ratio', 'ud_ratio', 'hue_mean', 'sat_mean']

    feature_labels = {
        'mean_brightness': '亮度均值 (0-255)',
        'contrast': '亮度标准差',
        'R_G_ratio': 'R/G 通道比',
        'B_G_ratio': 'B/G 通道比',
        'lr_ratio': '左/右 亮度比',
        'ud_ratio': '上/下 亮度比',
        'hue_mean': 'HSV色相均值',
        'sat_mean': 'HSV饱和度均值',
    }

    # 逐序列统计
    for feat in key_features:
        print(f'\n--- {feature_labels.get(feat, feat)} ---')
        print(f'{"Group":<28} {"N":>4} {"Mean":>8} {"Std":>7} {"Min":>8} {"P5":>8} {"P50":>8} {"P95":>8} {"Max":>8}')
        for group in sorted(groups.keys()):
            if not groups[group]:
                continue
            stats = group_stats(groups[group])
            s = stats[feat]
            n = len(groups[group])
            print(f'{group:<28} {n:>4} {s["mean"]:>8.3f} {s["std"]:>7.3f} '
                  f'{s["min"]:>8.3f} {s["p5"]:>8.3f} {s["p50"]:>8.3f} '
                  f'{s["p95"]:>8.3f} {s["max"]:>8.3f}')

    # 合并 real vs sim 统计
    print('\n' + '=' * 100)
    print('AGGREGATED: real (train+val) vs sim (train+val)')
    print('=' * 100)

    real_features = []
    sim_features = []
    for group, features_list in groups.items():
        if 'real_' in group:
            real_features.extend(features_list)
        elif 'sim_' in group:
            sim_features.extend(features_list)

    for label, feat_list in [('real (train+val)', real_features), ('sim (train+val)', sim_features)]:
        if not feat_list:
            continue
        stats = group_stats(feat_list)
        print(f'\n{label}: {len(feat_list)} frames')
        print(f'{"Feature":<22} {"P5":>8} {"P25":>8} {"P50":>8} {"P75":>8} {"P95":>8} {"Range":>14}')
        for feat in key_features:
            s = stats[feat]
            print(f'{feature_labels.get(feat, feat):<22} '
                  f'{s["p5"]:>8.3f} {s["p25"]:>8.3f} {s["p50"]:>8.3f} '
                  f'{s["p75"]:>8.3f} {s["p95"]:>8.3f} '
                  f'[{s["min"]:.1f}, {s["max"]:.1f}]')

    # T_photo 参数建议 (仅基于 train/val)
    if real_features:
        rs = group_stats(real_features)
        ss = group_stats(sim_features) if sim_features else rs

        print('\n' + '=' * 100)
        print('T_PHOTO PARAMETER RANGE (based on train+val only, NO test data)')
        print('=' * 100)

        # gamma: 覆盖 real 和 sim 的亮度范围
        real_p5 = rs['mean_brightness']['p5']
        real_p50 = rs['mean_brightness']['p50']
        real_p95 = rs['mean_brightness']['p95']
        sim_p50 = ss['mean_brightness']['p50']
        # gamma < 1 使图像变亮, gamma > 1 使图像变暗 (linear space)
        # 覆盖 train+val 的亮度 P5-P95 范围，再留 20% margin
        brightness_ratio_lo = real_p5 / real_p50
        brightness_ratio_hi = real_p95 / real_p50
        gamma_lo = max(0.4, brightness_ratio_lo * 0.8)    # 暗端 (放大 gamma 范围)
        gamma_hi = min(2.5, brightness_ratio_hi * 1.2)    # 亮端

        # 通道增益: 覆盖 R/G 和 B/G 的 P5-P95
        rg_lo = min(rs['R_G_ratio']['p5'], ss['R_G_ratio']['p5'])
        rg_hi = max(rs['R_G_ratio']['p95'], ss['R_G_ratio']['p95'])
        bg_lo = min(rs['B_G_ratio']['p5'], ss['B_G_ratio']['p5'])
        bg_hi = max(rs['B_G_ratio']['p95'], ss['B_G_ratio']['p95'])

        # 空间渐变: 覆盖 L/R 和 U/D 的 P5-P95
        lr_lo = min(rs['lr_ratio']['p5'], ss['lr_ratio']['p5'])
        lr_hi = max(rs['lr_ratio']['p95'], ss['lr_ratio']['p95'])
        ud_lo = min(rs['ud_ratio']['p5'], ss['ud_ratio']['p5'])
        ud_hi = max(rs['ud_ratio']['p95'], ss['ud_ratio']['p95'])

        # 对比度
        ct_lo = rs['contrast']['p5'] / rs['contrast']['p95']
        ct_hi = rs['contrast']['p95'] / rs['contrast']['p5']

        print(f"""
基于 train+val {len(real_features)} real + {len(sim_features)} sim 帧的 photometric 分布:

1. gamma (曝光):
   real 亮度 P5-P95: [{rs['mean_brightness']['p5']:.1f}, {rs['mean_brightness']['p95']:.1f}]
   sim  亮度 P5-P95: [{ss['mean_brightness']['p5']:.1f}, {ss['mean_brightness']['p95']:.1f}]
   → gamma 范围建议: [{gamma_lo:.2f}, {gamma_hi:.2f}]

2. ch_gain (白平衡 R 通道):
   real R/G P5-P95: [{rs['R_G_ratio']['p5']:.3f}, {rs['R_G_ratio']['p95']:.3f}]
   sim  R/G P5-P95: [{ss['R_G_ratio']['p5']:.3f}, {ss['R_G_ratio']['p95']:.3f}]
   → ch_gain R 范围建议: [{rg_lo:.3f}, {rg_hi:.3f}]

3. ch_gain (白平衡 B 通道):
   real B/G P5-P95: [{rs['B_G_ratio']['p5']:.3f}, {rs['B_G_ratio']['p95']:.3f}]
   sim  B/G P5-P95: [{ss['B_G_ratio']['p5']:.3f}, {ss['B_G_ratio']['p95']:.3f}]
   → ch_gain B 范围建议: [{bg_lo:.3f}, {bg_hi:.3f}]

4. 空间渐变 (左右):
   real L/R P5-P95: [{rs['lr_ratio']['p5']:.3f}, {rs['lr_ratio']['p95']:.3f}]
   → grad_lr 范围建议: [{lr_lo:.3f}, {lr_hi:.3f}]

5. 空间渐变 (上下):
   real U/D P5-P95: [{rs['ud_ratio']['p5']:.3f}, {rs['ud_ratio']['p95']:.3f}]
   → grad_ud 范围建议: [{ud_lo:.3f}, {ud_hi:.3f}]

6. 对比度:
   real contrast P5/P95: [{rs['contrast']['p5']:.1f}, {rs['contrast']['p95']:.1f}]
   → contrast 范围建议: [{ct_lo:.2f}, {ct_hi:.2f}]
""")

        # 与 test 诊断的对照 (仅标注 gap，不用于定参数)
        print('=' * 100)
        print('REFERENCE: train+val range vs test miss segments (for paper error analysis only)')
        print('=' * 100)
        print(f'  train+val real 亮度 range: [{rs["mean_brightness"]["min"]:.0f}, {rs["mean_brightness"]["max"]:.0f}]')
        print(f'  train+val real R/G range:  [{rs["R_G_ratio"]["min"]:.3f}, {rs["R_G_ratio"]["max"]:.3f}]')
        print(f'  test seq02_miss2 亮度:     [44, 112]  (gap: 下方 {rs["mean_brightness"]["min"] - 44:.0f})')
        print(f'  test seq03_miss 亮度:      [219, 225] (gap: 上方 {225 - rs["mean_brightness"]["max"]:.0f})')
        print(f'  test seq02_miss2 R/G:      [0.72, 0.96] (gap: {rs["R_G_ratio"]["min"] - 0.72:.3f} 下方)')


if __name__ == '__main__':
    main()