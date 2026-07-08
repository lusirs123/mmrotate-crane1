#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_context_map_activation.py - 检查平台 context map 在死段上是否激活。

核心问题：
  P3≈P2 表明 inference injection 对几何零效果。这有两个可能：
  A. scale ≈ 0（alpha 没学动）→ 调制本身就是恒等
  B. gate ≈ 0 在死段上（平台 context head 看不到平台）→ 调制是空操作

  本脚本同时检查 A 和 B：打印学到的 scale、每帧每层的 context map logit
  统计、gate 统计、有效调制范围。并对比死段帧 vs 健康帧。

  如果死段上 logit ≈ 0 / active_frac ≈ 0，则设计核心假设
  （平台比顶梁更可读）被推翻，重新训练也救不了。

示例:
  PYTHONPATH=. python3 crane_project/tools/probe_context_map_activation.py \
      --config crane_project/configs/crane_symeood_k1_platform_injector.py \
      --checkpoint work_dirs/crane_symeood_k1_platform_injector/epoch_24.pth \
      --gpu 0 \
      --out-json work_dirs/path_a_probes/context_map_ordinary.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import mcml_diag as diag
from crane_project.tools.ctx_entry_probe import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Probe platform context map activation on hard-slice frames.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default='test')
    parser.add_argument('--seq', default='real_seq02')
    parser.add_argument('--hard-start', type=int, default=137)
    parser.add_argument('--hard-end', type=int, default=169)
    parser.add_argument('--healthy-frames', type=int, nargs='*',
                        default=[50, 80, 200, 500],
                        help='Healthy frames for contrast (outside dead segments)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out-json', default=None)
    return parser.parse_args()


def analyze_frame(model, transform_compose, img_scale, flip,
                  args, seq: str, fid: int) -> Optional[Dict]:
    img_path, ann_path = diag.find_files(args.data_root, args.split, seq, fid)
    if img_path is None:
        return None
    gts = diag.parse_dota_ann(ann_path)
    if not gts:
        return None
    gt = gts[0]

    img_tensor, meta, img_stats = diag.preprocess_image(
        img_path, transform_compose, img_scale, flip)
    if img_tensor is None:
        return None
    img_tensor = img_tensor.cuda(f'cuda:{args.gpu}')

    injector = model.platform_context_injector

    with torch.no_grad():
        feat = model.extract_feat(img_tensor)

        # context map predictions (before modulation)
        preds = injector(feat)

        # global_max from main head (unmodulated)
        outs = model.bbox_head(feat)
        cls_scores = outs[0] if isinstance(outs, tuple) else outs
        global_max = max(cs.sigmoid().max().item() for cs in cls_scores)

        # learned modulation strength
        gate_scale = float(injector.gate_scale)
        gate_alpha = float(injector.gate_alpha.item())
        scale_eff = gate_scale * float(torch.tanh(injector.gate_alpha).item())

        level_stats = []
        for lvl_idx, pred in enumerate(preds):
            logit = pred[0, 0]
            gate = logit.tanh()
            eff_mod = 1.0 + scale_eff * gate

            level_stats.append(dict(
                level=lvl_idx,
                logit_mean=float(logit.mean()),
                logit_max=float(logit.max()),
                logit_min=float(logit.min()),
                logit_std=float(logit.std()),
                active_frac=float((logit > 0).float().mean()),
                strong_frac=float((logit > 1.0).float().mean()),
                gate_mean=float(gate.mean()),
                gate_max=float(gate.max()),
                gate_min=float(gate.min()),
                gate_abs_mean=float(gate.abs().mean()),
                eff_mod_mean=float(eff_mod.mean()),
                eff_mod_max=float(eff_mod.max()),
                eff_mod_min=float(eff_mod.min()),
                eff_mod_std=float(eff_mod.std()),
            ))

        return dict(
            frame=int(fid),
            seq=seq,
            global_max=global_max,
            brightness=float(img_stats['raw_brightness']),
            gate_scale=gate_scale,
            gate_alpha=gate_alpha,
            scale_eff=scale_eff,
            is_hard=global_max < 0.05,
            levels=level_stats,
        )


def print_frame(result: Dict):
    tag = 'DEAD' if result['is_hard'] else 'OK  '
    print(f"  F{result['frame']:5d} [{tag}] "
          f"gmax={result['global_max']:.4f} "
          f"bright={result['brightness']:.1f}  "
          f"scale={result['scale_eff']:.6f}")
    for ls in result['levels']:
        print(f"       L{ls['level']}: "
              f"logit[{ls['logit_mean']:+.3f},{ls['logit_max']:+.3f}] "
              f"act={ls['active_frac']:.3f} "
              f"strong={ls['strong_frac']:.3f} "
              f"gate[{ls['gate_mean']:+.4f},{ls['gate_max']:+.4f}] "
              f"abs={ls['gate_abs_mean']:.4f} "
              f"mod[{ls['eff_mod_min']:.4f},{ls['eff_mod_max']:.4f}]")


def print_summary(results: List[Dict]):
    print('\n' + '=' * 90)
    print('SUMMARY')
    print('=' * 90)

    if not results:
        print('  No valid frames.')
        return

    r0 = results[0]
    print(f"  gate_scale (config)  = {r0['gate_scale']}")
    print(f"  gate_alpha (learned) = {r0['gate_alpha']:.6f}")
    print(f"  scale_eff            = {r0['scale_eff']:.6f}")
    print(f"  (scale_eff ≈ 0 means modulation is identity regardless of gate)")
    print()

    hard = [r for r in results if r['is_hard']]
    healthy = [r for r in results if not r['is_hard']]

    for label, group in [('HARD (global_max<0.05)', hard), ('HEALTHY', healthy)]:
        if not group:
            print(f"  {label}: no frames")
            continue
        n = len(group)
        print(f"  {label} ({n} frames):")
        for lvl_idx in range(len(group[0]['levels'])):
            vals = {
                'logit_mean': [r['levels'][lvl_idx]['logit_mean'] for r in group],
                'logit_max': [r['levels'][lvl_idx]['logit_max'] for r in group],
                'active': [r['levels'][lvl_idx]['active_frac'] for r in group],
                'strong': [r['levels'][lvl_idx]['strong_frac'] for r in group],
                'gate_mean': [r['levels'][lvl_idx]['gate_mean'] for r in group],
                'gate_abs': [r['levels'][lvl_idx]['gate_abs_mean'] for r in group],
                'mod_mean': [r['levels'][lvl_idx]['eff_mod_mean'] for r in group],
                'mod_max': [r['levels'][lvl_idx]['eff_mod_max'] for r in group],
                'mod_min': [r['levels'][lvl_idx]['eff_mod_min'] for r in group],
            }
            print(f"    L{lvl_idx}: "
                  f"logit_mean={np.mean(vals['logit_mean']):+.4f} "
                  f"logit_max={np.mean(vals['logit_max']):+.4f} "
                  f"active={np.mean(vals['active']):.4f} "
                  f"strong={np.mean(vals['strong']):.4f}")
            print(f"          "
                  f"gate_mean={np.mean(vals['gate_mean']):+.4f} "
                  f"gate_abs={np.mean(vals['gate_abs']):.4f} "
                  f"mod=[{np.mean(vals['mod_min']):.4f},"
                  f"{np.mean(vals['mod_mean']):.4f},"
                  f"{np.mean(vals['mod_max']):.4f}]")
        print()

    # Verdict
    print("  " + "-" * 70)
    print("  VERDICT:")
    print()

    # Check A: is scale near 0?
    if abs(r0['scale_eff']) < 0.001:
        print("    [A] scale_eff ≈ 0  (alpha 没学动)")
        print("        → 调制本身就是恒等，gate 无论激活与否都没用")
        print("        → 原因：L_plat 梯度不经过 gate_alpha，没有信号推 alpha 移动")
        print("        → 这是 ordinary (init_alpha=0.0) 的预期行为")
        print("        → strong (init_alpha=0.50) 的 scale_eff 应该 > 0，需单独检查")
    else:
        print(f"    [A] scale_eff = {r0['scale_eff']:.6f}  (非零，调制有强度)")
        print("        → scale 不是问题，需检查 gate 是否激活")

    # Check B: is gate near 0 on hard-slice?
    if hard:
        hard_active = np.mean([r['levels'][0]['active_frac'] for r in hard])
        hard_logit = np.mean([r['levels'][0]['logit_mean'] for r in hard])
        hard_gate_abs = np.mean([r['levels'][0]['gate_abs_mean'] for r in hard])

        if healthy:
            healthy_active = np.mean([r['levels'][0]['active_frac'] for r in healthy])
            healthy_logit = np.mean([r['levels'][0]['logit_mean'] for r in healthy])
            healthy_gate_abs = np.mean([r['levels'][0]['gate_abs_mean'] for r in healthy])
        else:
            healthy_active = healthy_logit = healthy_gate_abs = float('nan')

        print()
        print(f"    [B] 死段 context map (level 0):")
        print(f"        active_frac:  hard={hard_active:.4f}  healthy={healthy_active:.4f}")
        print(f"        logit_mean:   hard={hard_logit:+.4f}  healthy={healthy_logit:+.4f}")
        print(f"        gate_abs:     hard={hard_gate_abs:.4f}  healthy={healthy_gate_abs:.4f}")
        print()

        # Interpretation thresholds:
        # active_frac: fraction of pixels where logit > 0 (i.e. gate positive)
        # If logit_mean is massively negative and active_frac is < 0.05, the
        # context map has collapsed to an all-negative prediction. This is the
        # expected BCE optimum under extreme class imbalance (<1% platform
        # pixels), and means the platform is NOT being highlighted.
        saturated_negative = (hard_active < 0.05) and (hard_logit < -5.0)
        weak_on_hard = healthy_active > 0 and (hard_active < healthy_active * 0.3)

        if saturated_negative:
            print("    ✗ 平台 context map 在死段上几乎不激活")
            print("      (active_frac < 0.05, logit_mean ≪ 0，context map 已坍缩为全负预测)")
            print("      → 说明 context head 没有真正学会定位平台")
            print("      → 训练目标 BCE 在极端正负样本不平衡下最优解就是全负")
            print("      → 设计核心假设（平台可作为可读上下文）被推翻")
            print("      → 此时 injection 无论是否门控、scale 大小，都是空间常数调制")
        elif weak_on_hard:
            print("    ⚠ 平台 context map 在死段上激活显著弱于健康帧")
            print(f"      (hard active={hard_active:.3f} < healthy {healthy_active:.3f} × 0.3)")
            print("      → injection 在死段上效果有限")
        else:
            print("    ✓ 平台 context map 在死段上存在可检测激活")
            print(f"      (hard active={hard_active:.3f}, healthy={healthy_active:.4f})")
            if abs(r0['scale_eff']) < 0.001:
                print("      → 但 scale_eff ≈ 0，调制仍是恒等")
                print("      → 需要重新训练让 alpha 学动，或用 strong 配置")
            else:
                print("      → gate 和 scale 都非零，injection 理论上有效")
                print("      → P3≈P2 的原因需进一步排查")
    else:
        print("    [B] 无死段帧数据，无法判断")

    print()


def main():
    args = parse_args()
    model, cfg = load_model(args.config, args.checkpoint, args.gpu)

    if not hasattr(model, 'platform_context_injector') or \
            model.platform_context_injector is None:
        print('[error] model has no platform_context_injector')
        sys.exit(1)

    transform_compose, img_scale, flip = diag.build_test_transforms(cfg)

    hard_frames = list(range(args.hard_start, args.hard_end + 1))
    healthy_frames = args.healthy_frames

    print('=' * 90)
    print('CONTEXT MAP ACTIVATION PROBE')
    print('=' * 90)
    print(f'config:     {args.config}')
    print(f'checkpoint: {args.checkpoint}')
    print(f'seq:        {args.seq}')
    print(f'hard:       [{args.hard_start}..{args.hard_end}] ({len(hard_frames)} frames)')
    print(f'healthy:    {healthy_frames}')
    print()

    results = []

    print('--- Hard-slice frames ---')
    for fid in hard_frames:
        r = analyze_frame(model, transform_compose, img_scale, flip,
                          args, args.seq, fid)
        if r is not None:
            results.append(r)
            print_frame(r)

    print('\n--- Healthy frames (contrast) ---')
    for fid in healthy_frames:
        r = analyze_frame(model, transform_compose, img_scale, flip,
                          args, args.seq, fid)
        if r is not None:
            results.append(r)
            print_frame(r)

    print_summary(results)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'[out] wrote {args.out_json}')


if __name__ == '__main__':
    main()
