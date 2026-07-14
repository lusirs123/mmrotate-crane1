#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe 2: K-transform 平台几何 oracle 候选重排。

先由少量人工平台 polygon 拟合序列级 beam->platform K，再把主头 top-K
候选映射成平台 OBB。用 K(GT beam) 作为“完美平台观测”上界，对候选进行
重排，判断平台位置证据是否足以从低分候选池中选回正确顶梁框。

注意：该 probe 使用 GT 间接构造期望平台位置，只是可行性上界，不是可部署
推理算法，也不能证明视觉平台头一定可学。

服务器示例：
  PYTHONPATH=. python3 crane_project/tools/platform_k_oracle_rerank_probe.py \
      --config crane_project/configs/crane_symeood_k1.py \
      --checkpoint work_dirs/crane_symeood_k1/epoch_24.pth \
      --manual-platform-json \
        work_dirs/crane_symeood_k1/manual_platform_polygons_real_seq02.json \
      --split test --seq real_seq02 --start 133 --end 171 \
      --topks 200 500 1000 --log-lambdas 0.25 0.5 1.0 2.0 4.0 \
      --riou-thr 0.5 --gpu 0 \
      --out-json \
        work_dirs/platform_k_oracle/k1_real_seq02_133_171.json
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import candidate_pool_oracle_probe as pool_probe  # noqa: E402
from crane_project.tools import ctx_entry_probe as entry_probe  # noqa: E402
from crane_project.tools import platform_context_probe as platform_probe  # noqa: E402


K_KEYS = (
    'width_k', 'height_k', 'offset_long_k', 'offset_short_k', 'dtheta')


def parse_args():
    parser = argparse.ArgumentParser(
        description='K-transform platform oracle reranking probe.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manual-platform-json', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default='test',
                        choices=['test', 'train', 'train_sim'])
    parser.add_argument('--seq', default='real_seq02')
    parser.add_argument('--start', type=int, default=133)
    parser.add_argument('--end', type=int, default=171)
    parser.add_argument('--candidate-source', default='main',
                        choices=['main', 'aux1'])
    parser.add_argument('--topks', type=int, nargs='+',
                        default=[200, 500, 1000])
    parser.add_argument('--log-lambdas', type=float, nargs='+',
                        default=[0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--seq-platform-angle-mode', default='zero',
                        choices=['zero', 'median'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def polygon_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Convex quadrilateral IoU in image coordinates."""
    a = platform_probe.order_corners(
        np.asarray(poly_a, dtype=np.float32).reshape(4, 2))
    b = platform_probe.order_corners(
        np.asarray(poly_b, dtype=np.float32).reshape(4, 2))
    area_a = abs(float(cv2.contourArea(a)))
    area_b = abs(float(cv2.contourArea(b)))
    if area_a <= 1e-6 or area_b <= 1e-6:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(a, b)
    union = area_a + area_b - float(inter_area)
    return float(inter_area) / max(union, 1e-6)


def obbs_to_polygons(boxes: np.ndarray) -> np.ndarray:
    from mmrotate.core import obb2poly_np

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    # This mmrotate version's NumPy converter expects an optional score column
    # and returns it as the ninth value. Append a dummy score explicitly.
    boxes_with_score = np.concatenate([
        boxes, np.zeros((boxes.shape[0], 1), dtype=np.float32)
    ], axis=1)
    polys = obb2poly_np(boxes_with_score, version='le90')
    return np.asarray(polys[:, :8], dtype=np.float32).reshape(-1, 4, 2)


def median_k(samples: Sequence[Dict], angle_mode: str) -> Optional[Dict]:
    if not samples:
        return None
    result = dict(
        source='manual_polygon_median',
        sample_count=len(samples),
        sample_frames=[int(item['frame']) for item in samples],
    )
    for key in K_KEYS:
        values = [float(item[key]) for item in samples]
        result[key] = float(np.median(values))
        result[f'{key}_std'] = float(np.std(values))
        result[f'{key}_min'] = float(np.min(values))
        result[f'{key}_max'] = float(np.max(values))
    result['observed_dtheta'] = result['dtheta']
    result['dtheta'] = result['dtheta'] if angle_mode == 'median' else 0.0
    result['angle_mode'] = angle_mode
    result['samples'] = list(samples)
    return result


def fit_k_and_validate(manual_platforms: Dict[int, Dict], args):
    """拟合全量 K，并在人工帧上计算 in-sample/leave-one-out IoU。"""
    diag = entry_probe.get_diag()
    samples = []
    records = []
    for frame, item in sorted(manual_platforms.items()):
        manual_poly = platform_probe.manual_polygon(item)
        if manual_poly is None:
            continue
        _, ann_path = diag.find_files(
            args.data_root, args.split, args.seq, int(frame))
        beam_poly = platform_probe.ann_to_poly(ann_path) if ann_path else None
        if beam_poly is None:
            continue
        sample = platform_probe.frame_platform_k(
            beam_poly, manual_poly, int(frame))
        samples.append(sample)
        records.append((int(frame), beam_poly, manual_poly))

    seq_k = median_k(samples, args.seq_platform_angle_mode)
    if seq_k is None:
        raise RuntimeError('No valid manual platform polygons for K fitting')

    validation_rows = []
    for frame, beam_poly, manual_poly in records:
        pred_full = platform_probe.platform_poly_from_seq_k(beam_poly, seq_k)
        full_iou = polygon_iou(pred_full, manual_poly)
        loo_samples = [sample for sample in samples
                       if int(sample['frame']) != int(frame)]
        loo_k = median_k(loo_samples, args.seq_platform_angle_mode)
        loo_iou = None
        if loo_k is not None:
            pred_loo = platform_probe.platform_poly_from_seq_k(beam_poly, loo_k)
            loo_iou = polygon_iou(pred_loo, manual_poly)
        validation_rows.append(dict(
            frame=int(frame),
            in_sample_iou=float(full_iou),
            leave_one_out_iou=None if loo_iou is None else float(loo_iou),
        ))

    def stats(key):
        values = [float(row[key]) for row in validation_rows
                  if row.get(key) is not None]
        return dict(
            count=len(values),
            mean=float(np.mean(values)) if values else None,
            median=float(np.median(values)) if values else None,
            min=float(np.min(values)) if values else None,
            max=float(np.max(values)) if values else None,
        )

    validation = dict(
        rows=validation_rows,
        in_sample=stats('in_sample_iou'),
        leave_one_out=stats('leave_one_out_iou'),
    )
    return seq_k, validation


def mode_names(log_lambdas: Sequence[float]) -> List[str]:
    names = ['cls', 'beam_oracle', 'platform_only', 'cls_x_platform']
    names.extend(f'log_lambda_{value:g}' for value in log_lambdas)
    return names


def select_index(mode: str, cls_scores: np.ndarray,
                 beam_ious: np.ndarray, platform_ious: np.ndarray,
                 log_lambdas: Sequence[float]) -> int:
    eps = 1e-12
    if mode == 'cls':
        metric = cls_scores
    elif mode == 'beam_oracle':
        metric = beam_ious
    elif mode == 'platform_only':
        metric = platform_ious
    elif mode == 'cls_x_platform':
        metric = cls_scores * platform_ious
    elif mode.startswith('log_lambda_'):
        value_text = mode[len('log_lambda_'):]
        lam = float(value_text)
        metric = np.log(np.clip(cls_scores, eps, None)) + lam * np.log(
            np.clip(platform_ious, eps, None))
    else:
        raise ValueError(f'Unknown rerank mode: {mode}')
    return int(np.argmax(metric))


def analyze_frame(model, transform_compose, img_scale, flip, seq_k,
                  args, frame: int, topks: Sequence[int],
                  modes: Sequence[str]) -> Optional[Dict]:
    from mmcv.ops import box_iou_rotated

    diag = entry_probe.get_diag()
    img_path, ann_path = diag.find_files(
        args.data_root, args.split, args.seq, frame)
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
    img_tensor = img_tensor.cuda(f'cuda:{args.gpu}')

    with torch.no_grad():
        feats = model.extract_feat(img_tensor)
        candidate_head, cls_scores, bbox_preds = (
            entry_probe.forward_candidate_head(
                model, feats, args.candidate_source))
        boxes, scores, levels, _, _ = entry_probe.flatten_decode_candidates(
            candidate_head, cls_scores, bbox_preds, meta['img_shape'])

        gt = pool_probe.scale_gt_to_img(gts[0], meta)
        gt_box = entry_probe.gt_to_tensor(gt, boxes.device)
        beam_ious_all = box_iou_rotated(
            boxes.float(), gt_box.float()).reshape(-1)

        max_k = min(max(topks), int(scores.numel()))
        top_scores_t, top_indices = torch.topk(
            scores, k=max_k, largest=True, sorted=True)
        top_boxes_t = boxes[top_indices]
        top_levels_t = levels[top_indices]
        top_beam_ious_t = beam_ious_all[top_indices]

    top_scores = top_scores_t.detach().cpu().numpy().astype(np.float64)
    top_boxes = top_boxes_t.detach().cpu().numpy().astype(np.float32)
    top_levels = top_levels_t.detach().cpu().numpy().astype(np.int64)
    top_beam_ious = top_beam_ious_t.detach().cpu().numpy().astype(np.float64)

    gt_poly = obbs_to_polygons(
        np.asarray([[gt['cx'], gt['cy'], gt['w'], gt['h'],
                     math.radians(gt['angle'])]], dtype=np.float32))[0]
    expected_platform = platform_probe.platform_poly_from_seq_k(gt_poly, seq_k)
    candidate_polys = obbs_to_polygons(top_boxes)
    platform_ious = np.asarray([
        polygon_iou(
            platform_probe.platform_poly_from_seq_k(poly, seq_k),
            expected_platform)
        for poly in candidate_polys
    ], dtype=np.float64)

    per_k = {}
    for topk in topks:
        actual_k = min(int(topk), len(top_scores))
        per_mode = {}
        for mode in modes:
            selected = select_index(
                mode,
                top_scores[:actual_k],
                top_beam_ious[:actual_k],
                platform_ious[:actual_k],
                args.log_lambdas)
            selected_riou = float(top_beam_ious[selected])
            per_mode[mode] = dict(
                selected_rank=selected + 1,
                selected_cls_score=float(top_scores[selected]),
                selected_platform_iou=float(platform_ious[selected]),
                selected_beam_riou=selected_riou,
                selected_level=int(top_levels[selected]),
                hit=bool(selected_riou >= args.riou_thr),
            )
        per_k[str(topk)] = dict(actual_k=actual_k, modes=per_mode)

    row = dict(
        frame=int(frame),
        fname=os.path.splitext(os.path.basename(img_path))[0],
        brightness=float(img_stats.get('raw_brightness', float('nan'))),
        global_max=float(top_scores[0]),
        expected_platform=expected_platform.astype(float).tolist(),
        per_k=per_k,
    )

    focus_k = 500 if 500 in topks else topks[-1]
    focus = per_k[str(focus_k)]['modes']
    print(
        f"[{row['fname']}] K={focus_k} "
        f"cls={focus['cls']['selected_beam_riou']:.3f}/"
        f"{int(focus['cls']['hit'])} "
        f"plat={focus['platform_only']['selected_beam_riou']:.3f}/"
        f"{int(focus['platform_only']['hit'])} "
        f"mul={focus['cls_x_platform']['selected_beam_riou']:.3f}/"
        f"{int(focus['cls_x_platform']['hit'])} "
        f"ceiling={focus['beam_oracle']['selected_beam_riou']:.3f}/"
        f"{int(focus['beam_oracle']['hit'])}")
    return row


def build_summary(rows: Sequence[Dict], topks: Sequence[int],
                  modes: Sequence[str]) -> Dict:
    summary = dict(frames=len(rows), per_k={})
    for topk in topks:
        mode_summary = {}
        for mode in modes:
            hit_key = f'hit_{topk}_{mode}'
            proxy_rows = []
            selected_rious = []
            for row in rows:
                item = row['per_k'][str(topk)]['modes'][mode]
                proxy_rows.append(dict(
                    frame=int(row['frame']), **{hit_key: bool(item['hit'])}))
                selected_rious.append(float(item['selected_beam_riou']))
            hits = sum(bool(row[hit_key]) for row in proxy_rows)
            mode_summary[mode] = dict(
                hits=hits,
                misses=len(rows) - hits,
                recall=hits / len(rows) if rows else 0.0,
                mcml=pool_probe.longest_consecutive_miss(
                    proxy_rows, hit_key) if rows else 0,
                selected_riou_mean=(float(np.mean(selected_rious))
                                    if selected_rious else 0.0),
                selected_riou_min=(float(np.min(selected_rious))
                                   if selected_rious else 0.0),
                selected_riou_max=(float(np.max(selected_rious))
                                   if selected_rious else 0.0),
            )
        summary['per_k'][str(topk)] = mode_summary
    return summary


def print_summary(summary: Dict, topks: Sequence[int], modes: Sequence[str]):
    print('\n' + '=' * 104)
    print('PROBE 2 SUMMARY: K-TRANSFORM PLATFORM ORACLE RERANK')
    print('=' * 104)
    for topk in topks:
        print(f'K={topk}')
        print(f"  {'mode':<24} {'hits':>10} {'recall':>10} "
              f"{'MCML':>8} {'mean_RIoU':>12} {'min':>8} {'max':>8}")
        for mode in modes:
            item = summary['per_k'][str(topk)][mode]
            print(
                f"  {mode:<24} "
                f"{item['hits']:>4d}/{summary['frames']:<5d} "
                f"{item['recall']:>10.3f} "
                f"{item['mcml']:>8d} "
                f"{item['selected_riou_mean']:>12.3f} "
                f"{item['selected_riou_min']:>8.3f} "
                f"{item['selected_riou_max']:>8.3f}")
        print('-' * 104)


def main():
    args = parse_args()
    topks = pool_probe.normalize_topks(args.topks)
    if args.end < args.start:
        raise ValueError('--end must be greater than or equal to --start')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manual_platforms_all = platform_probe.load_manual_platforms(
        args.manual_platform_json, args.split, args.seq)
    in_range_manual = {
        frame: item for frame, item in manual_platforms_all.items()
        if int(args.start) <= int(frame) <= int(args.end)
    }
    # Prefer manual anchors from the probed operating window. For 133..171
    # this selects 137/144/150/156/162/169 and avoids mixing the small-target
    # geometry at 2/12/20/26 into the dark-large-target K upper bound.
    manual_platforms = (
        in_range_manual if len(in_range_manual) >= 2 else manual_platforms_all)
    seq_k, k_validation = fit_k_and_validate(manual_platforms, args)
    print('[K fit] '
          f"frames={seq_k['sample_frames']} "
          f"width={seq_k['width_k']:.4f} "
          f"height={seq_k['height_k']:.4f} "
          f"offset_long={seq_k['offset_long_k']:.4f} "
          f"offset_short={seq_k['offset_short_k']:.4f} "
          f"dtheta={seq_k['dtheta']:.4f}")
    print('[K validation] '
          f"in_sample={k_validation['in_sample']} "
          f"leave_one_out={k_validation['leave_one_out']}")

    model, cfg = entry_probe.load_model(
        args.config, args.checkpoint, args.gpu)
    diag = entry_probe.get_diag()
    transform_compose, img_scale, flip = diag.build_test_transforms(cfg)
    modes = mode_names(args.log_lambdas)

    print('=' * 104)
    print('PROBE 2: K-TRANSFORM PLATFORM ORACLE RERANK')
    print('=' * 104)
    print(f'config:       {args.config}')
    print(f'checkpoint:   {args.checkpoint}')
    print(f'data:         {args.split}/{args.seq} {args.start}..{args.end}')
    print(f'topks:        {topks}')
    print(f'modes:        {modes}')

    rows = []
    for frame in range(int(args.start), int(args.end) + 1):
        row = analyze_frame(
            model, transform_compose, img_scale, flip, seq_k,
            args, frame, topks, modes)
        if row is not None:
            rows.append(row)

    summary = build_summary(rows, topks, modes)
    print_summary(summary, topks, modes)

    output_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = dict(
        probe='platform_k_oracle_rerank',
        config=args.config,
        checkpoint=args.checkpoint,
        split=args.split,
        seq=args.seq,
        frame_ids=list(range(int(args.start), int(args.end) + 1)),
        args=vars(args),
        seq_platform_k=seq_k,
        k_validation=k_validation,
        summary=summary,
        rows=rows,
    )
    with open(output_path, 'w') as handle:
        json.dump(payload, handle, indent=2)
    print(f'\n[out] wrote {output_path}')


if __name__ == '__main__':
    main()
