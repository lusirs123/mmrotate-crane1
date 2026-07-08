#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_path_a_probes.py - 路径 A 前置诊断结果比较工具。

读取 run_path_a_probes.sh 输出的 JSON 文件，打印横向对比表，
并给出路径 A 可信性的初步判据。

用法:
    python3 crane_project/tools/compare_path_a_probes.py \
        --out-dir work_dirs/path_a_probes
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

PROBE_FILES = {
    'P1': 'P1_baseline.json',
    'P2': 'P2_ordinary_no_injection.json',
    'P3': 'P3_ordinary_with_injection.json',
    'P4': 'P4_strong_no_injection.json',
    'P5': 'P5_strong_with_injection.json',
}

LABELS = {
    'P1': 'baseline (无injector)',
    'P2': 'ordinary, 不inject',
    'P3': 'ordinary, with inject',
    'P4': 'strong, 不inject',
    'P5': 'strong, with inject',
}


def load_probe(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(probe: Dict) -> Dict:
    rows = probe.get('rows', [])
    if not rows:
        return dict(n=0)

    n = len(rows)
    decisions = {}
    for r in rows:
        d = r.get('decision', '')
        decisions[d] = decisions.get(d, 0) + 1

    topk_vals = [r['score_topk'].get('best_riou', 0.0) for r in rows]
    dec_vals = [r['decoded_center_neighborhood'].get('best_riou', 0.0)
                for r in rows]
    global_vals = [r.get('global_max', 0.0) for r in rows]
    brightness_vals = [r.get('brightness', 0.0) for r in rows]

    usable_thr = probe.get('args', {}).get('usable_iou_thr', 0.50)
    topk_usable = sum(1 for v in topk_vals if v >= usable_thr)
    dec_usable = sum(1 for v in dec_vals if v >= usable_thr)

    return dict(
        n=n,
        decisions=decisions,
        score_topk_mean=float(np.mean(topk_vals)),
        score_topk_min=float(np.min(topk_vals)),
        score_topk_max=float(np.max(topk_vals)),
        decoded_neigh_mean=float(np.mean(dec_vals)),
        decoded_neigh_min=float(np.min(dec_vals)),
        decoded_neigh_max=float(np.max(dec_vals)),
        usable_topk=f'{topk_usable}/{n}',
        usable_decoded=f'{dec_usable}/{n}',
        global_max_mean=float(np.mean(global_vals)),
        global_max_min=float(np.min(global_vals)),
        global_max_max=float(np.max(global_vals)),
        brightness_mean=float(np.mean(brightness_vals)),
    )


def print_table(probes: Dict[str, Optional[Dict]]):
    print('\n' + '=' * 100)
    print('路径 A 前置诊断：横向对比')
    print('=' * 100)
    print()

    # 表头
    cols = ['指标'] + [f'{k}: {LABELS[k]}' for k in PROBE_FILES
                       if probes.get(k) is not None]
    col_widths = [28] + [20] * (len(cols) - 1)
    header = ' | '.join(c.ljust(w) for c, w in zip(cols, col_widths))
    print(header)
    print('-' * len(header))

    available = [k for k in PROBE_FILES if probes.get(k) is not None]
    metrics = {k: extract_metrics(probes[k]) for k in available}

    if not available:
        print('  [无数据] 没有找到任何 probe JSON。')
        return

    # 行：decision 计数
    for decision in ['SCORE_ENTRY_EXISTS', 'ENTRY_EXISTS_NOT_SCORE_RANKED',
                     'NO_GEOM_ENTRY']:
        row_vals = []
        for k in available:
            d = metrics[k].get('decisions', {})
            row_vals.append(f'{d.get(decision, 0)}/{metrics[k]["n"]}')
        print_row(f'  {decision}', row_vals, col_widths)

    print('-' * len(header))

    # decoded-neighborhood
    for label, key in [
        ('decoded-neigh mean', 'decoded_neigh_mean'),
        ('decoded-neigh min', 'decoded_neigh_min'),
        ('decoded-neigh max', 'decoded_neigh_max'),
    ]:
        row_vals = [f'{metrics[k].get(key, 0):.3f}' for k in available]
        print_row(f'  {label}', row_vals, col_widths)

    print('-' * len(header))

    # score-topK
    for label, key in [
        ('score-topK mean', 'score_topk_mean'),
        ('score-topK max', 'score_topk_max'),
    ]:
        row_vals = [f'{metrics[k].get(key, 0):.3f}' for k in available]
        print_row(f'  {label}', row_vals, col_widths)

    print('-' * len(header))

    # usable
    for label, key in [
        ('usable@0.50 (topK)', 'usable_topk'),
        ('usable@0.50 (decoded)', 'usable_decoded'),
    ]:
        row_vals = [metrics[k].get(key, '-') for k in available]
        print_row(f'  {label}', row_vals, col_widths)

    print('-' * len(header))

    # global_max
    for label, key in [
        ('global_max mean', 'global_max_mean'),
        ('global_max min', 'global_max_min'),
        ('global_max max', 'global_max_max'),
    ]:
        row_vals = [f'{metrics[k].get(key, 0):.4f}' for k in available]
        print_row(f'  {label}', row_vals, col_widths)

    print('-' * len(header))

    # brightness
    row_vals = [f'{metrics[k].get("brightness_mean", 0):.1f}' for k in available]
    print_row('  brightness mean', row_vals, col_widths)

    print()


def print_row(label: str, vals: List[str], col_widths: List[int]):
    parts = [label.ljust(col_widths[0])]
    for v, w in zip(vals, col_widths[1:]):
        parts.append(v.ljust(w))
    print(' | '.join(parts))


def print_verdict(probes: Dict[str, Optional[Dict]]):
    """根据 P1/P2/P3 对比给出路径 A 初步判据。"""
    print('=' * 100)
    print('路径 A 判据分析')
    print('=' * 100)
    print()

    m = {k: extract_metrics(probes[k]) for k in probes if probes[k] is not None}

    p1 = m.get('P1')
    p2 = m.get('P2')
    p3 = m.get('P3')

    if p2 is not None and p3 is not None:
        d2 = p2.get('decoded_neigh_mean', 0)
        d3 = p3.get('decoded_neigh_mean', 0)
        print(f'  P2 (ordinary, 不inject) decoded-neigh mean = {d2:.3f}')
        print(f'  P3 (ordinary, with inject) decoded-neigh mean = {d3:.3f}')
        print(f'  injection 效果: {d3:.3f} vs {d2:.3f} (Δ={d3-d2:+.3f}, '
              f'{d3/d2:.2f}x)' if d2 > 0 else '')
        print()

        if d3 > d2 * 1.2:
            print('  ✓ injection 改善了几何 (P3 > P2 × 1.2)')
            if p1 is not None:
                d1 = p1.get('decoded_neigh_mean', 0)
                print(f'  P1 (baseline) decoded-neigh mean = {d1:.3f}')
                if d3 > d1 * 1.2:
                    print('  ✓ injection 也优于 baseline → 路径 A 值得做')
                elif d3 >= d1 * 0.9:
                    print('  ~ injection 接近 baseline → 路径 A 可能有用'
                          '（门控保护 easy frames）')
                else:
                    print('  ✗ injection 仍不如 baseline → 路径 A 价值存疑')
        elif d3 < d2 * 0.8:
            print('  ✗ injection 恶化了几何 (P3 < P2 × 0.8)')
            print('  → 路径 A 死：调制本身有害，门控救不了')
        else:
            print('  ~ injection 效果不显著 (P3 ≈ P2)')
            print('  → 路径 A 价值低：调制没改变 hard-slice 几何')
    else:
        if p2 is None and p3 is None:
            print('  [缺数据] P2/P3 均未找到（需要 ordinary injector checkpoint）')
        elif p1 is not None:
            print(f'  P1 (baseline) decoded-neigh mean = '
                  f'{p1.get("decoded_neigh_mean", 0):.3f}')
            print('  [提示] 只有 P1，无法判断 injection 效果')

    print()
    print('  注意：以上判据仅基于 decoded-neighborhood（几何指标）。')
    print('  即使几何改善，分数 gap (global_max 0.006→0.05, 8x) 仍可能')
    print('  是路径 A 的瓶颈。需结合 score-topK 和 usable@0.50 综合判断。')


def main():
    parser = argparse.ArgumentParser(
        description='比较路径 A 前置诊断 probe 结果')
    parser.add_argument('--out-dir', default='work_dirs/path_a_probes',
                        help='probe JSON 输出目录')
    args = parser.parse_args()

    probes = {}
    for key, fname in PROBE_FILES.items():
        path = os.path.join(args.out_dir, fname)
        probes[key] = load_probe(path)

    found = sum(1 for v in probes.values() if v is not None)
    if found == 0:
        print(f'[错误] 在 {args.out_dir} 中未找到任何 probe JSON。')
        print('  请先运行: bash crane_project/tools/run_path_a_probes.sh')
        sys.exit(1)

    print(f'找到 {found}/{len(PROBE_FILES)} 个 probe 结果')
    print_table(probes)
    print_verdict(probes)


if __name__ == '__main__':
    main()
