#!/usr/bin/env python3
"""
photo_diag.py — 量化 real 域测试集各段的 photometric 分布，
为 L_invar 的 T_photo 扰动幅度提供数据依据。

分析维度:
  1. 全局亮度 (mean brightness, 0-255)
  2. RGB 通道比 (R/G, B/G) — 色温/白平衡偏移
  3. 空间光照梯度 (左半 vs 右半, 上半 vs 下半 均值比)
  4. 对比度 (std of brightness)
  5. 检测头在该段的漏检率 (对照用)

输出:
  - 按段分组的统计表 (miss 段 vs detected 段 vs sim)
  - T_photo 各分量的建议扰动范围

Run:
    cd /Users/mac/Documents/paper/symEOOD
    python3 crane_project/utils/photo_diag.py
"""

import os
import re
import sys
import glob
import math
from collections import defaultdict
import cv2
import numpy as np


# =====================================================================
# 配置
# =====================================================================
IMG_DIR = 'crane_project/data/crane_grab/test/images/'
GT_DIR = 'crane_project/data/crane_grab/test/annfiles/'
EQUI_PRED_DIR = 'work_dirs/crane_symeood_m2_equi/ckpt_sweep/final_test/epoch_24/preds/Task1_grab/'
SA_PRED_DIR = 'work_dirs/crane_symeood_m2_simpleaug/ckpt_sweep/final_test/epoch_22/preds/Task1_grab/'

# 定义 miss 段 (从先前诊断得到)
MISS_SEGMENTS = {
    'real_seq02': [
        (2, 36, 'seq02_miss1'),       # 35 帧
        (133, 171, 'seq02_miss2'),    # 39 帧 (MCML=39)
    ],
    'real_seq03': [
        (123, 131, 'seq03_miss1'),    # 9 帧
        (155, 192, 'seq03_miss2'),    # 38 帧
    ],
}


# =====================================================================
# 工具函数
# =====================================================================

def parse_seq_frame(basename):
    """从文件名解析 (domain, seq_id, frame_id)"""
    m = re.match(r'^(real|sim)_(.+?)_(\d+)$', basename)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, None


def is_pred_nonempty(pred_path):
    """检查预测文件是否非空"""
    if not os.path.exists(pred_path):
        return False
    if os.path.getsize(pred_path) < 10:
        return False
    with open(pred_path) as f:
        parts = f.read().strip().split()
    return len(parts) >= 9


def compute_photo_features(img_bgr):
    """从 BGR 图像提取 photometric 特征.

    Returns:
        dict with keys:
            mean_brightness: 全局亮度均值 (0-255)
            contrast: 亮度标准差
            R_G_ratio: R/G 通道比
            B_G_ratio: B/G 通道比
            lr_ratio: 左半/右半 亮度比
            ud_ratio: 上半/下半 亮度比
            lr_R_ratio: 左半/右半 R 通道比
            lr_G_ratio: 左半/右半 G 通道比
            lr_B_ratio: 左半/右半 B 通道比
            hue_mean: HSV 色相均值 (0-180)
            sat_mean: HSV 饱和度均值 (0-255)
    """
    H, W = img_bgr.shape[:2]

    # BGR 通道分离
    B, G, R = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]

    # 避免除零
    eps = 1e-6

    # 全局亮度 (加权灰度)
    gray = 0.114 * B.astype(np.float64) + 0.587 * G.astype(np.float64) + 0.299 * R.astype(np.float64)
    mean_brightness = float(gray.mean())
    contrast = float(gray.std())

    # 通道比 (取全图均值后比)
    R_mean = float(R.astype(np.float64).mean()) + eps
    G_mean = float(G.astype(np.float64).mean()) + eps
    B_mean = float(B.astype(np.float64).mean()) + eps

    R_G_ratio = R_mean / G_mean
    B_G_ratio = B_mean / G_mean

    # 空间梯度: 左半 vs 右半
    left = img_bgr[:, :W // 2].astype(np.float64)
    right = img_bgr[:, W // 2:].astype(np.float64)

    left_gray = (0.114 * left[:, :, 0] + 0.587 * left[:, :, 1] + 0.299 * left[:, :, 2]).mean()
    right_gray = (0.114 * right[:, :, 0] + 0.587 * right[:, :, 1] + 0.299 * right[:, :, 2]).mean()
    lr_ratio = float(left_gray / (right_gray + eps))

    left_rgb = left.reshape(-1, 3).mean(0)   # B, G, R
    right_rgb = right.reshape(-1, 3).mean(0)
    lr_R_ratio = float(left_rgb[2] / (right_rgb[2] + eps))
    lr_G_ratio = float(left_rgb[1] / (right_rgb[1] + eps))
    lr_B_ratio = float(left_rgb[0] / (right_rgb[0] + eps))

    # 上半 vs 下半
    top = img_bgr[:H // 2].astype(np.float64)
    bottom = img_bgr[H // 2:].astype(np.float64)
    top_gray = (0.114 * top[:, :, 0] + 0.587 * top[:, :, 1] + 0.299 * top[:, :, 2]).mean()
    bot_gray = (0.114 * bottom[:, :, 0] + 0.587 * bottom[:, :, 1] + 0.299 * bottom[:, :, 2]).mean()
    ud_ratio = float(top_gray / (bot_gray + eps))

    # HSV
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
        lr_R_ratio=lr_R_ratio,
        lr_G_ratio=lr_G_ratio,
        lr_B_ratio=lr_B_ratio,
        hue_mean=hue_mean,
        sat_mean=sat_mean,
    )


def group_stats(features_list):
    """对一组 features dict 计算统计量"""
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
    # 1. 收集所有 test 帧的 photometric 特征，按分组
    groups = defaultdict(list)  # group_name -> [features_dict, ...]

    # 建立 miss 段索引: (domain_seq, frame_id) -> group_name
    miss_lookup = {}
    for seq_key, segments in MISS_SEGMENTS.items():
        for start, end, name in segments:
            for fid in range(start, end + 1):
                miss_lookup[(seq_key, fid)] = name

    # 遍历所有 test 图像
    all_imgs = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))
    print(f'Found {len(all_imgs)} test images')

    for img_path in all_imgs:
        bn = os.path.splitext(os.path.basename(img_path))[0]
        domain, seq_id, fid = parse_seq_frame(bn)
        if domain is None:
            continue

        seq_key = f'{domain}_{seq_id}'

        # 判断分组
        if domain == 'sim':
            group = 'sim_all'
        elif (seq_key, fid) in miss_lookup:
            group = miss_lookup[(seq_key, fid)]
        else:
            group = f'{seq_key}_detected'

        # 读图计算特征
        img = cv2.imread(img_path)
        if img is None:
            continue
        features = compute_photo_features(img)
        groups[group].append(features)

    # 2. 计算统计量
    print('\n' + '=' * 100)
    print('PHOTOMETRIC DISTRIBUTION BY GROUP')
    print('=' * 100)

    # 定义输出顺序
    group_order = []
    for seq_key in sorted(MISS_SEGMENTS.keys()):
        for _, _, name in MISS_SEGMENTS[seq_key]:
            group_order.append(name)
        group_order.append(f'{seq_key}_detected')
    group_order.append('sim_all')

    # 关键特征
    key_features = ['mean_brightness', 'contrast', 'R_G_ratio', 'B_G_ratio',
                    'lr_ratio', 'ud_ratio', 'lr_R_ratio', 'hue_mean', 'sat_mean']

    feature_labels = {
        'mean_brightness': '亮度均值 (0-255)',
        'contrast': '亮度标准差',
        'R_G_ratio': 'R/G 通道比',
        'B_G_ratio': 'B/G 通道比',
        'lr_ratio': '左/右 亮度比',
        'ud_ratio': '上/下 亮度比',
        'lr_R_ratio': '左/右 R通道比',
        'hue_mean': 'HSV色相均值',
        'sat_mean': 'HSV饱和度均值',
    }

    for feat in key_features:
        print(f'\n--- {feature_labels.get(feat, feat)} ---')
        print(f'{"Group":<22} {"N":>4} {"Mean":>8} {"Std":>7} {"Min":>8} {"P5":>8} {"P50":>8} {"P95":>8} {"Max":>8}')
        for group in group_order:
            if group not in groups or not groups[group]:
                continue
            stats = group_stats(groups[group])
            s = stats[feat]
            n = len(groups[group])
            print(f'{group:<22} {n:>4} {s["mean"]:>8.3f} {s["std"]:>7.3f} '
                  f'{s["min"]:>8.3f} {s["p5"]:>8.3f} {s["p50"]:>8.3f} '
                  f'{s["p95"]:>8.3f} {s["max"]:>8.3f}')

    # 3. 诊断 miss 段 vs detected 段的差异
    print('\n' + '=' * 100)
    print('MISS vs DETECTED COMPARISON (real domain only)')
    print('=' * 100)

    # 合并所有 miss 帧和 detected 帧
    all_miss_features = []
    all_detected_features = []
    for group, features_list in groups.items():
        if 'miss' in group:
            all_miss_features.extend(features_list)
        elif 'detected' in group:
            all_detected_features.extend(features_list)

    if all_miss_features and all_detected_features:
        miss_stats = group_stats(all_miss_features)
        det_stats = group_stats(all_detected_features)

        print(f'\n{"Feature":<22} {"Miss Mean":>10} {"Det Mean":>10} {"Ratio":>8} {"方向":>10}')
        for feat in key_features:
            mm = miss_stats[feat]['mean']
            dm = det_stats[feat]['mean']
            ratio = mm / (dm + 1e-6)
            direction = '← miss更暗' if (feat == 'mean_brightness' and mm < dm) else \
                        '← miss更亮' if (feat == 'mean_brightness' and mm > dm) else \
                        '← miss偏色' if (feat in ('R_G_ratio', 'B_G_ratio') and abs(ratio - 1) > 0.05) else ''
            print(f'{feature_labels.get(feat, feat):<22} {mm:>10.3f} {dm:>10.3f} {ratio:>8.3f} {direction}')

    # 4. T_photo 建议范围
    print('\n' + '=' * 100)
    print('T_PHOTO PARAMETER RANGE RECOMMENDATIONS')
    print('=' * 100)

    if all_miss_features:
        ms = group_stats(all_miss_features)
        ds = group_stats(all_detected_features) if all_detected_features else ms

        print(f"""
基于 miss 段 {len(all_miss_features)} 帧的 photometric 分布 (P5-P95):

1. gamma (曝光):
   miss 段亮度 P5-P95: [{ms['mean_brightness']['p5']:.1f}, {ms['mean_brightness']['p95']:.1f}]
   detected 段亮度 P5-P95: [{ds['mean_brightness']['p5']:.1f}, {ds['mean_brightness']['p95']:.1f}]
   → gamma 范围建议: [0.4, 2.0] (覆盖两端)

2. 分通道增益 (白平衡):
   miss 段 R/G P5-P95: [{ms['R_G_ratio']['p5']:.3f}, {ms['R_G_ratio']['p95']:.3f}]
   detected 段 R/G P5-P95: [{ds['R_G_ratio']['p5']:.3f}, {ds['R_G_ratio']['p95']:.3f}]
   miss 段 B/G P5-P95: [{ms['B_G_ratio']['p5']:.3f}, {ms['B_G_ratio']['p95']:.3f}]
   → ch_gain 范围建议: [0.7, 1.4]^3 (独立 R/G/B)

3. 空间渐变 (光照不均匀):
   miss 段 L/R 亮度比 P5-P95: [{ms['lr_ratio']['p5']:.3f}, {ms['lr_ratio']['p95']:.3f}]
   → grad 范围建议: [0.6, 1.5] (左右方向线性渐变)

4. 对比度:
   miss 段对比度 P5-P95: [{ms['contrast']['p5']:.1f}, {ms['contrast']['p95']:.1f}]
   → contrast 范围建议: [0.6, 1.5]
""")

    # 5. 逐帧明细 (miss 段)
    print('\n' + '=' * 100)
    print('PER-FRAME DETAIL: MISS SEGMENTS')
    print('=' * 100)

    for group in group_order:
        if 'miss' not in group:
            continue
        if group not in groups:
            continue
        features_list = groups[group]
        print(f'\n--- {group} ({len(features_list)} frames) ---')
        print(f'{"Idx":>4} {"Bright":>7} {"Contr":>6} {"R/G":>6} {"B/G":>6} {"L/R":>6} {"U/D":>6} {"Sat":>6}')
        for i, f in enumerate(features_list):
            print(f'{i:>4} {f["mean_brightness"]:>7.1f} {f["contrast"]:>6.1f} '
                  f'{f["R_G_ratio"]:>6.3f} {f["B_G_ratio"]:>6.3f} '
                  f'{f["lr_ratio"]:>6.3f} {f["ud_ratio"]:>6.3f} '
                  f'{f["sat_mean"]:>6.1f}')


if __name__ == '__main__':
    main()