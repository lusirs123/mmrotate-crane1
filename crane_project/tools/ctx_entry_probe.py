#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctx_entry_probe.py - context ROI 外扩头入口诊断工具。

目的:
  在实现 ContextRoIRefineHead 前, 先回答一个更硬的问题:
  当前 dense detector 的 raw decoded candidates 里, 是否存在可供外扩头
  refine 的几何入口。

核心区别:
  1. score-topK: 按分类分数排序的候选, 模拟常规推理入口。
  2. GT 邻域 best: 不按分数筛选, 只看 GT 附近 decoded candidates 的最佳 RIoU。

解释:
  dense detector 每个 anchor/位置都会 decode 出框, 所以“有候选”几乎恒真。
  真正要判断的是 GT 空间邻域内有没有几何可修的候选。如果邻域最佳
  RIoU 也很低, 外扩头没有可靠 crop 入口, 应先改候选播种策略。

示例:
  PYTHONPATH=. python3 crane_project/tools/ctx_entry_probe.py \\
      --config crane_project/configs/crane_symeood_m2_equi.py \\
      --checkpoint work_dirs/crane_symeood_m2_equi/epoch_24.pth \\
      --split test --seq real_seq02 --start 133 --end 171 \\
      --gpu 0 --topk 50 --out-json work_dirs/ctx_entry_seq02_133_171.json
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def get_diag():
    from crane_project.tools import mcml_diag as diag
    return diag


def parse_args():
    parser = argparse.ArgumentParser(
        description='Probe whether context ROI refinement has a candidate entry.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default='test',
                        choices=['test', 'train', 'train_sim'])
    parser.add_argument('--seq', default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--sample', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--topk', type=int, default=50,
                        help='score-topK decoded candidates to inspect')
    parser.add_argument('--entry-iou-thr', type=float, default=0.10,
                        help='RIoU threshold for saying a geometric entry exists')
    parser.add_argument('--neighbor-radius-mul', type=float, default=1.5,
                        help='GT diagonal multiplier for local decoded-center neighborhood')
    parser.add_argument('--neighbor-radius-px', type=float, default=None,
                        help='Override local neighborhood radius in pixels')
    parser.add_argument('--min-radius-px', type=float, default=16.0,
                        help='Lower bound for local neighborhood radius')
    parser.add_argument('--max-neighbor-cands', type=int, default=3000,
                        help='Cap local candidates by score if the neighborhood is huge')
    parser.add_argument('--out-json', default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_model(config_path: str, checkpoint_path: str, gpu: int):
    from mmcv import Config
    from mmrotate.models import build_detector

    cfg = Config.fromfile(config_path)
    # 诊断看 raw candidates, 不让 score_thr/max_per_img 影响内部 forward。
    if hasattr(cfg.model, 'test_cfg') and cfg.model.test_cfg is not None:
        cfg.model.test_cfg.score_thr = 0.0
        cfg.model.test_cfg.max_per_img = max(
            int(getattr(cfg.model.test_cfg, 'max_per_img', 1)), 100)

    model = build_detector(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'[load] missing keys: {len(missing)}')
    if unexpected:
        print(f'[load] unexpected keys: {len(unexpected)}')
    model = model.cuda(f'cuda:{gpu}')
    model.eval()
    return model, cfg


def discover_frame_ids(args) -> Tuple[str, List[int]]:
    diag = get_diag()
    seq = args.seq
    if seq is None:
        seqs = diag.discover_sequences(args.data_root, args.split)
        if not seqs:
            raise RuntimeError(f'No sequences found in {args.data_root}/{args.split}')
        seq = seqs[0]
        print(f'[data] auto-selected seq: {seq}')

    if args.start is not None and args.end is not None:
        frame_ids = list(range(args.start, args.end + 1))
    else:
        frame_ids = diag.discover_frames(
            args.data_root, args.split, seq, args.sample)
    return seq, frame_ids


def flatten_decode_candidates(model, cls_scores, bbox_preds, img_shape):
    """按 SymEOODHead 推理顺序展平并 decode 全量候选。"""
    device = cls_scores[0].device
    featmap_sizes = [score.shape[-2:] for score in cls_scores]
    anchors_per_level = model.bbox_head.anchor_generator.grid_priors(
        featmap_sizes, device=device)

    all_boxes, all_scores, all_levels, all_anchor_centers = [], [], [], []
    for lvl, (cls_lvl, bbox_lvl, anchors) in enumerate(
            zip(cls_scores, bbox_preds, anchors_per_level)):
        cls_feat = cls_lvl[0]
        bbox_feat = bbox_lvl[0]
        scores = cls_feat.permute(1, 2, 0).reshape(-1, 1).sigmoid().reshape(-1)
        bbox_flat = bbox_feat.permute(1, 2, 0).reshape(-1, 5)
        if anchors.shape[0] != scores.shape[0]:
            raise RuntimeError(
                f'Anchor/order mismatch at level {lvl}: '
                f'anchors={anchors.shape}, scores={scores.shape}')

        decoded = model.bbox_head.bbox_coder.decode(
            anchors, bbox_flat, max_shape=img_shape)
        all_boxes.append(decoded)
        all_scores.append(scores)
        all_levels.append(torch.full(
            (scores.numel(),), lvl, dtype=torch.long, device=device))
        all_anchor_centers.append(anchors[:, :2])

    return (
        torch.cat(all_boxes, dim=0),
        torch.cat(all_scores, dim=0),
        torch.cat(all_levels, dim=0),
        torch.cat(all_anchor_centers, dim=0),
    )


def gt_to_tensor(gt: Dict, device) -> torch.Tensor:
    return torch.tensor(
        [[gt['cx'], gt['cy'], gt['w'], gt['h'], np.radians(gt['angle'])]],
        dtype=torch.float32,
        device=device)


def summarize_subset(boxes, scores, levels, gt_box, inds) -> Optional[Dict]:
    """返回指定候选集合里的最佳 RIoU 候选。"""
    from mmcv.ops import box_iou_rotated

    if inds.numel() == 0:
        return None
    sub_boxes = boxes[inds]
    sub_scores = scores[inds]
    sub_levels = levels[inds]
    ious = box_iou_rotated(sub_boxes.float(), gt_box.float()).reshape(-1)
    best_pos = int(torch.argmax(ious).item())
    best_idx = inds[best_pos]
    best_box = boxes[best_idx].detach().cpu().numpy()
    center_dist = float(torch.norm(boxes[best_idx, :2] - gt_box[0, :2]).item())
    return dict(
        count=int(inds.numel()),
        best_riou=float(ious[best_pos].item()),
        best_score=float(scores[best_idx].item()),
        best_level=int(levels[best_idx].item()),
        best_center_dist=center_dist,
        best_box=[
            float(best_box[0]), float(best_box[1]), float(best_box[2]),
            float(best_box[3]), float(best_box[4])
        ],
        score_mean=float(sub_scores.mean().item()),
        score_max=float(sub_scores.max().item()),
        levels_hist={
            str(int(lvl.item())): int((sub_levels == lvl).sum().item())
            for lvl in torch.unique(sub_levels)
        },
    )


def topk_summary(boxes, scores, levels, gt_box, topk: int) -> Dict:
    from mmcv.ops import box_iou_rotated

    k = min(int(topk), int(scores.numel()))
    top_scores, top_inds = torch.topk(scores, k=k, largest=True)
    sub = summarize_subset(boxes, scores, levels, gt_box, top_inds)
    if sub is None:
        return dict(count=0, best_riou=0.0, top1_score=0.0, top1_riou=0.0)
    top1_box = boxes[top_inds[0:1]]
    top1_riou = box_iou_rotated(top1_box.float(), gt_box.float()).reshape(-1)[0]
    sub['top1_score'] = float(top_scores[0].item())
    sub['top1_riou'] = float(top1_riou.item())
    return sub


def neighborhood_summary(boxes, scores, levels, anchor_centers, gt_box,
                         radius: float, max_cands: int) -> Tuple[Dict, Dict]:
    gt_center = gt_box[0, :2]
    decoded_dist = torch.norm(boxes[:, :2] - gt_center[None, :], dim=1)
    anchor_dist = torch.norm(anchor_centers - gt_center[None, :], dim=1)

    def capped(mask):
        inds = torch.nonzero(mask, as_tuple=False).reshape(-1)
        raw_count = int(inds.numel())
        if raw_count > max_cands > 0:
            _, order = torch.topk(scores[inds], k=max_cands, largest=True)
            inds = inds[order]
        return inds, raw_count

    decoded_inds, decoded_raw = capped(decoded_dist <= radius)
    anchor_inds, anchor_raw = capped(anchor_dist <= radius)

    decoded = summarize_subset(boxes, scores, levels, gt_box, decoded_inds)
    anchor = summarize_subset(boxes, scores, levels, gt_box, anchor_inds)
    if decoded is None:
        decoded = dict(count=0, best_riou=0.0, raw_count=decoded_raw)
    else:
        decoded['raw_count'] = decoded_raw
    if anchor is None:
        anchor = dict(count=0, best_riou=0.0, raw_count=anchor_raw)
    else:
        anchor['raw_count'] = anchor_raw
    return decoded, anchor


def classify_entry(score_topk, decoded_neigh, entry_iou_thr: float) -> str:
    score_best = float(score_topk.get('best_riou', 0.0))
    neigh_best = float(decoded_neigh.get('best_riou', 0.0))
    if neigh_best < entry_iou_thr:
        return 'NO_GEOM_ENTRY'
    if score_best < entry_iou_thr:
        return 'ENTRY_EXISTS_NOT_SCORE_RANKED'
    return 'SCORE_ENTRY_EXISTS'


def analyze_frame(model, transform_compose, img_scale, flip, args,
                  seq: str, fid: int) -> Optional[Dict]:
    diag = get_diag()
    img_path, ann_path = diag.find_files(args.data_root, args.split, seq, fid)
    if img_path is None:
        print(f'[skip] frame {fid:05d}: image not found')
        return None
    gts = diag.parse_dota_ann(ann_path)
    if not gts:
        print(f'[skip] frame {fid:05d}: GT not found')
        return None
    gt = gts[0]

    img_tensor, meta, img_stats = diag.preprocess_image(
        img_path, transform_compose, img_scale, flip)
    if img_tensor is None:
        print(f'[skip] frame {fid:05d}: preprocess failed')
        return None
    img_tensor = img_tensor.cuda(f'cuda:{args.gpu}')

    with torch.no_grad():
        feat = model.extract_feat(img_tensor)
        cls_scores, bbox_preds = model.bbox_head(feat)
        boxes, scores, levels, anchor_centers = flatten_decode_candidates(
            model, cls_scores, bbox_preds, meta['img_shape'])

    gt_box = gt_to_tensor(gt, boxes.device)
    gt_diag = float(np.sqrt(gt['w'] ** 2 + gt['h'] ** 2))
    if args.neighbor_radius_px is None:
        radius = max(float(args.min_radius_px),
                     float(args.neighbor_radius_mul) * gt_diag)
    else:
        radius = float(args.neighbor_radius_px)

    score_topk = topk_summary(boxes, scores, levels, gt_box, args.topk)
    decoded_neigh, anchor_neigh = neighborhood_summary(
        boxes, scores, levels, anchor_centers, gt_box, radius,
        args.max_neighbor_cands)
    decision = classify_entry(score_topk, decoded_neigh, args.entry_iou_thr)

    base = seq if 'seq' in seq else f'real_{seq}'
    fname = f'{base}_{fid:05d}'
    row = dict(
        frame=int(fid),
        fname=fname,
        split=args.split,
        seq=seq,
        img_path=img_path,
        gt=dict(cx=gt['cx'], cy=gt['cy'], w=gt['w'], h=gt['h'],
                angle=gt['angle'], diag=gt_diag),
        radius=radius,
        global_max=float(scores.max().item()),
        brightness=float(img_stats['raw_brightness']),
        score_topk=score_topk,
        decoded_center_neighborhood=decoded_neigh,
        anchor_center_neighborhood=anchor_neigh,
        decision=decision,
    )
    return row


def print_frame(row: Dict, entry_iou_thr: float):
    score = row['score_topk']
    dec = row['decoded_center_neighborhood']
    anc = row['anchor_center_neighborhood']
    marker = {
        'NO_GEOM_ENTRY': 'X',
        'ENTRY_EXISTS_NOT_SCORE_RANKED': '!',
        'SCORE_ENTRY_EXISTS': '+',
    }[row['decision']]
    print(
        f"  [{row['fname']}] {marker} {row['decision']} "
        f"global={row['global_max']:.4f} bright={row['brightness']:.1f} "
        f"topK_best={score.get('best_riou', 0.0):.3f} "
        f"top1={score.get('top1_riou', 0.0):.3f}/s{score.get('top1_score', 0.0):.4f} "
        f"dec_neigh_best={dec.get('best_riou', 0.0):.3f}"
        f"/n{dec.get('raw_count', dec.get('count', 0))} "
        f"anchor_neigh_best={anc.get('best_riou', 0.0):.3f}"
        f"/n{anc.get('raw_count', anc.get('count', 0))} "
        f"thr={entry_iou_thr:.2f}")


def print_summary(rows: List[Dict], entry_iou_thr: float):
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    if not rows:
        print('No valid frames analyzed.')
        return

    decisions = {}
    for row in rows:
        decisions[row['decision']] = decisions.get(row['decision'], 0) + 1
    total = len(rows)
    for key in ['SCORE_ENTRY_EXISTS', 'ENTRY_EXISTS_NOT_SCORE_RANKED',
                'NO_GEOM_ENTRY']:
        print(f'  {key}: {decisions.get(key, 0)}/{total}')

    topk_vals = [r['score_topk'].get('best_riou', 0.0) for r in rows]
    dec_vals = [
        r['decoded_center_neighborhood'].get('best_riou', 0.0) for r in rows
    ]
    anc_vals = [
        r['anchor_center_neighborhood'].get('best_riou', 0.0) for r in rows
    ]
    global_vals = [r['global_max'] for r in rows]
    print(f'  score-topK best RIoU: mean={np.mean(topk_vals):.3f} '
          f'min={np.min(topk_vals):.3f} max={np.max(topk_vals):.3f}')
    print(f'  decoded-neighborhood best RIoU: mean={np.mean(dec_vals):.3f} '
          f'min={np.min(dec_vals):.3f} max={np.max(dec_vals):.3f}')
    print(f'  anchor-neighborhood best RIoU: mean={np.mean(anc_vals):.3f} '
          f'min={np.min(anc_vals):.3f} max={np.max(anc_vals):.3f}')
    print(f'  global_max: mean={np.mean(global_vals):.4f} '
          f'min={np.min(global_vals):.4f} max={np.max(global_vals):.4f}')

    no_geom = decisions.get('NO_GEOM_ENTRY', 0)
    not_ranked = decisions.get('ENTRY_EXISTS_NOT_SCORE_RANKED', 0)
    if no_geom > total * 0.5:
        print('  verdict: GT 邻域几何入口大多不存在; 先改候选播种策略, '
              '不要直接进入 ROI refine MVP。')
    elif not_ranked > total * 0.5:
        print('  verdict: 邻域有几何入口, 但按分数 topK 抓不到; '
              'MVP 需要非分数候选播种或低分邻域保留。')
    else:
        print('  verdict: score-topK 已有足够几何入口; 可以进入 R1 单区 refine MVP。')
    print(f'  entry_iou_thr={entry_iou_thr:.2f}')


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, cfg = load_model(args.config, args.checkpoint, args.gpu)
    diag = get_diag()
    transform_compose, img_scale, flip = diag.build_test_transforms(cfg)
    seq, frame_ids = discover_frame_ids(args)

    print('=' * 80)
    print('CTX ENTRY PROBE')
    print('=' * 80)
    print(f'config:     {args.config}')
    print(f'checkpoint: {args.checkpoint}')
    print(f'source:     {args.split}/{seq} frames={len(frame_ids)}')
    print(f'topK:       {args.topk}')
    print(f'entry_thr:  {args.entry_iou_thr}')

    rows = []
    for fid in frame_ids:
        row = analyze_frame(
            model, transform_compose, img_scale, flip, args, seq, fid)
        if row is None:
            continue
        rows.append(row)
        print_frame(row, args.entry_iou_thr)

    print_summary(rows, args.entry_iou_thr)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        payload = dict(
            config=args.config,
            checkpoint=args.checkpoint,
            split=args.split,
            seq=seq,
            frame_ids=frame_ids,
            args=vars(args),
            rows=rows,
        )
        with open(args.out_json, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'\n[out] wrote {args.out_json}')


if __name__ == '__main__':
    main()
