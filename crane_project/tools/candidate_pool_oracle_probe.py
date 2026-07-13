#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe 1: score-threshold 前的候选池几何上界。

该脚本回答一个单一问题：主头按分类分保留的 top-K 稠密候选中，是否至少
存在一个与 GT 顶梁框 RIoU 达标的候选。它不修改模型、不重新训练，也不使用
平台标注。

输出重点：
  1. Recall@K：top-K 中存在 RIoU >= 阈值候选的帧占比；
  2. Oracle MCML@K：假设 oracle 能在 top-K 中挑中最佳候选时的连续失败长度；
  3. usable_best_rank：最高分类分可用候选在全量候选中的排名；
  4. dense oracle：不考虑分类排名时，稠密候选本身的几何上界。

服务器示例：
  PYTHONPATH=. python3 crane_project/tools/candidate_pool_oracle_probe.py \
      --config crane_project/configs/crane_symeood_k1.py \
      --checkpoint work_dirs/crane_symeood_k1/epoch_24.pth \
      --split test --seq real_seq02 --start 133 --end 171 \
      --candidate-source main --topks 1 10 50 100 200 \
      --riou-thr 0.5 --gpu 0 \
      --out-json work_dirs/candidate_pool_probe/k1_seq02_133_171_main.json
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Probe Recall@K and oracle MCML before score thresholding.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default='test',
                        choices=['test', 'train', 'train_sim'])
    parser.add_argument('--seq', default='real_seq02')
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--sample', type=int, default=10)
    parser.add_argument('--candidate-source', default='main',
                        choices=['main', 'aux1'])
    parser.add_argument('--topks', type=int, nargs='+',
                        default=[1, 10, 50, 100, 200])
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--inference-score-thr', type=float, default=0.05,
                        help='Only reports how many raw candidates pass this '
                             'threshold; it never filters the oracle pool.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def normalize_topks(values: Sequence[int]) -> List[int]:
    topks = sorted({int(v) for v in values if int(v) > 0})
    if not topks:
        raise ValueError('--topks must contain at least one positive integer')
    return topks


def scale_gt_to_img(gt: Dict, meta: Dict) -> Dict:
    """将原图 GT 映射到 test pipeline 处理后的图像尺度。"""
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


def longest_consecutive_miss(rows: Sequence[Dict], hit_key: str) -> int:
    """按帧号计算最长连续失败；帧号断裂时重新计数。"""
    longest = 0
    current = 0
    previous_frame = None
    for row in sorted(rows, key=lambda item: int(item['frame'])):
        frame = int(row['frame'])
        if previous_frame is None or frame != previous_frame + 1:
            current = 0
        if bool(row[hit_key]):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
        previous_frame = frame
    return int(longest)


def discover_frame_ids(args) -> Tuple[str, List[int]]:
    diag = entry_probe.get_diag()
    if args.start is not None or args.end is not None:
        if args.start is None or args.end is None:
            raise ValueError('--start and --end must be provided together')
        if args.end < args.start:
            raise ValueError('--end must be greater than or equal to --start')
        return args.seq, list(range(args.start, args.end + 1))
    return args.seq, diag.discover_frames(
        args.data_root, args.split, args.seq, args.sample)


def analyze_frame(model, transform_compose, img_scale, flip, args,
                  seq: str, frame: int, topks: Sequence[int]) -> Optional[Dict]:
    from mmcv.ops import box_iou_rotated

    diag = entry_probe.get_diag()
    img_path, ann_path = diag.find_files(
        args.data_root, args.split, seq, frame)
    if img_path is None:
        print(f'[skip] frame {frame:05d}: image not found')
        return None

    gts = diag.parse_dota_ann(ann_path)
    if not gts:
        print(f'[skip] frame {frame:05d}: GT not found')
        return None
    if len(gts) > 1:
        print(f'[warn] frame {frame:05d}: {len(gts)} GTs found; using first')

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
        boxes, scores, levels, _, alignment = (
            entry_probe.flatten_decode_candidates(
                candidate_head, cls_scores, bbox_preds, meta['img_shape']))

        gt = scale_gt_to_img(gts[0], meta)
        gt_box = entry_probe.gt_to_tensor(gt, boxes.device)
        ious = box_iou_rotated(
            boxes.float(), gt_box.float()).reshape(-1)

        candidate_count = int(scores.numel())
        max_k = min(max(topks), candidate_count)
        top_scores, top_indices = torch.topk(
            scores, k=max_k, largest=True, sorted=True)
        top_ious = ious[top_indices]

        dense_best_pos = int(torch.argmax(ious).item())
        dense_best_riou = float(ious[dense_best_pos].item())
        dense_best_score = float(scores[dense_best_pos].item())
        dense_best_rank = int((scores > scores[dense_best_pos]).sum().item()) + 1

        usable_mask = ious >= float(args.riou_thr)
        usable_count = int(usable_mask.sum().item())
        if usable_count:
            usable_indices = torch.nonzero(
                usable_mask, as_tuple=False).reshape(-1)
            usable_scores = scores[usable_indices]
            usable_local_pos = int(torch.argmax(usable_scores).item())
            usable_index = int(usable_indices[usable_local_pos].item())
            usable_best_score = float(scores[usable_index].item())
            usable_best_rank = int(
                (scores > scores[usable_index]).sum().item()) + 1
            usable_best_riou = float(ious[usable_index].item())
            usable_best_level = int(levels[usable_index].item())
        else:
            usable_best_score = None
            usable_best_rank = None
            usable_best_riou = 0.0
            usable_best_level = None

        per_k = {}
        for topk in topks:
            actual_k = min(int(topk), candidate_count)
            subset_ious = top_ious[:actual_k]
            best_pos = int(torch.argmax(subset_ious).item())
            best_index = int(top_indices[best_pos].item())
            best_riou = float(subset_ious[best_pos].item())
            per_k[str(topk)] = dict(
                requested_k=int(topk),
                actual_k=actual_k,
                oracle_hit=bool(best_riou >= args.riou_thr),
                best_riou=best_riou,
                best_score=float(scores[best_index].item()),
                best_level=int(levels[best_index].item()),
                best_rank=best_pos + 1,
            )

    row = dict(
        frame=int(frame),
        fname=os.path.splitext(os.path.basename(img_path))[0],
        img_path=img_path,
        brightness=float(img_stats.get('raw_brightness', float('nan'))),
        candidate_source=args.candidate_source,
        candidate_head=entry_probe.candidate_head_name(candidate_head),
        candidate_count=candidate_count,
        candidates_over_inference_thr=int(
            (scores > float(args.inference_score_thr)).sum().item()),
        global_max=float(scores.max().item()),
        top1_score=float(top_scores[0].item()),
        top1_riou=float(top_ious[0].item()),
        dense_best_riou=dense_best_riou,
        dense_best_score=dense_best_score,
        dense_best_rank=dense_best_rank,
        dense_oracle_hit=bool(dense_best_riou >= args.riou_thr),
        usable_count=usable_count,
        usable_best_score=usable_best_score,
        usable_best_rank=usable_best_rank,
        usable_best_riou=usable_best_riou,
        usable_best_level=usable_best_level,
        gt=dict(
            cx=float(gt['cx']), cy=float(gt['cy']),
            w=float(gt['w']), h=float(gt['h']),
            angle=float(gt['angle'])),
        decode_alignment=alignment,
        per_k=per_k,
    )
    for topk in topks:
        row[f'hit_at_{topk}'] = bool(per_k[str(topk)]['oracle_hit'])

    rank_text = 'none' if usable_best_rank is None else str(usable_best_rank)
    recalls = ' '.join(
        f'K{topk}={int(per_k[str(topk)]["oracle_hit"])}'
        for topk in topks)
    print(
        f"[{row['fname']}] global={row['global_max']:.6f} "
        f"top1_RIoU={row['top1_riou']:.3f} "
        f"dense_best={row['dense_best_riou']:.3f} "
        f"usable_rank={rank_text} {recalls}")
    return row


def build_summary(rows: Sequence[Dict], topks: Sequence[int],
                  riou_thr: float) -> Dict:
    total = len(rows)
    summary = dict(
        frames=total,
        riou_thr=float(riou_thr),
        top1_hits=sum(bool(row['hit_at_1']) for row in rows)
        if 1 in topks else None,
        dense_oracle_hits=sum(bool(row['dense_oracle_hit']) for row in rows),
        dense_oracle_recall=(
            sum(bool(row['dense_oracle_hit']) for row in rows) / total
            if total else 0.0),
        dense_oracle_mcml=longest_consecutive_miss(
            rows, 'dense_oracle_hit') if rows else 0,
        per_k={},
    )

    usable_ranks = [
        int(row['usable_best_rank']) for row in rows
        if row['usable_best_rank'] is not None
    ]
    summary['usable_rank_stats'] = dict(
        count=len(usable_ranks),
        median=float(np.median(usable_ranks)) if usable_ranks else None,
        p90=float(np.percentile(usable_ranks, 90)) if usable_ranks else None,
        max=max(usable_ranks) if usable_ranks else None,
    )

    for topk in topks:
        key = f'hit_at_{topk}'
        hits = sum(bool(row[key]) for row in rows)
        riou_values = [float(row['per_k'][str(topk)]['best_riou'])
                       for row in rows]
        summary['per_k'][str(topk)] = dict(
            hits=hits,
            misses=total - hits,
            recall=hits / total if total else 0.0,
            oracle_mcml=longest_consecutive_miss(rows, key) if rows else 0,
            best_riou_mean=float(np.mean(riou_values)) if riou_values else 0.0,
            best_riou_min=float(np.min(riou_values)) if riou_values else 0.0,
            best_riou_max=float(np.max(riou_values)) if riou_values else 0.0,
        )
    return summary


def print_summary(summary: Dict, topks: Sequence[int]):
    print('\n' + '=' * 88)
    print('PROBE 1 SUMMARY: PRE-THRESHOLD CANDIDATE POOL ORACLE')
    print('=' * 88)
    print(f"frames={summary['frames']}  RIoU_thr={summary['riou_thr']:.2f}")
    print(
        f"dense oracle: hits={summary['dense_oracle_hits']}/"
        f"{summary['frames']} recall={summary['dense_oracle_recall']:.3f} "
        f"MCML={summary['dense_oracle_mcml']}")
    print('-' * 88)
    print(f"{'K':>6} {'hits':>10} {'recall':>10} {'oracle_MCML':>14} "
          f"{'mean_best_RIoU':>16} {'min':>8} {'max':>8}")
    for topk in topks:
        item = summary['per_k'][str(topk)]
        print(
            f"{topk:>6d} "
            f"{item['hits']:>4d}/{summary['frames']:<5d} "
            f"{item['recall']:>10.3f} "
            f"{item['oracle_mcml']:>14d} "
            f"{item['best_riou_mean']:>16.3f} "
            f"{item['best_riou_min']:>8.3f} "
            f"{item['best_riou_max']:>8.3f}")
    rank = summary['usable_rank_stats']
    print('-' * 88)
    print('highest-score usable candidate rank: '
          f"count={rank['count']} median={rank['median']} "
          f"p90={rank['p90']} max={rank['max']}")


def main():
    args = parse_args()
    topks = normalize_topks(args.topks)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, cfg = entry_probe.load_model(
        args.config, args.checkpoint, args.gpu)
    diag = entry_probe.get_diag()
    transform_compose, img_scale, flip = diag.build_test_transforms(cfg)
    seq, frame_ids = discover_frame_ids(args)

    print('=' * 88)
    print('PROBE 1: PRE-THRESHOLD CANDIDATE POOL ORACLE')
    print('=' * 88)
    print(f'config:       {args.config}')
    print(f'checkpoint:   {args.checkpoint}')
    print(f'data:         {args.split}/{seq}')
    print(f'frames:       {frame_ids[0] if frame_ids else "-"}..'
          f'{frame_ids[-1] if frame_ids else "-"} ({len(frame_ids)})')
    print(f'head:         {args.candidate_source}')
    print(f'topks:        {topks}')
    print(f'RIoU_thr:     {args.riou_thr}')
    print(f'report_thr:   {args.inference_score_thr}')

    rows = []
    for frame in frame_ids:
        row = analyze_frame(
            model, transform_compose, img_scale, flip,
            args, seq, frame, topks)
        if row is not None:
            rows.append(row)

    summary = build_summary(rows, topks, args.riou_thr)
    print_summary(summary, topks)

    output_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = dict(
        probe='pre_threshold_candidate_pool_oracle',
        config=args.config,
        checkpoint=args.checkpoint,
        split=args.split,
        seq=seq,
        frame_ids=frame_ids,
        args=vars(args),
        summary=summary,
        rows=rows,
    )
    with open(output_path, 'w') as handle:
        json.dump(payload, handle, indent=2)
    print(f'\n[out] wrote {output_path}')


if __name__ == '__main__':
    main()
