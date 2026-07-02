#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctx_entry_stratify.py - stratify ctx_entry_probe JSON by usable geometry.

This is a post-processing tool for ctx_entry_probe outputs.  It separates the
old loose "entry exists" question from the control-facing question:

  - loose_entry: decoded-neighborhood best RIoU >= --entry-iou-thr
  - usable_geom: decoded-neighborhood best RIoU >= --usable-riou-thr
  - usable_score: score-topK best RIoU >= --usable-riou-thr

The resulting labels are meant to decide which frames are score disease
(geometry usable, ranking bad) and which are geometry disease (local oracle
geometry itself is not usable).

Example:
  PYTHONPATH=. python3 crane_project/tools/ctx_entry_stratify.py \
    --input work_dirs/crane_symeood_k1/ctx_entry_seq02_129_172_aux1.json \
    --out-dir work_dirs/crane_symeood_k1/ctx_entry_stratified \
    --name seq02_129_172_aux1
"""

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Stratify ctx_entry_probe outputs by disease type.')
    parser.add_argument('--input', nargs='+', required=True,
                        help='One or more ctx_entry_probe JSON files.')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--name', default=None,
                        help='Output basename when exactly one input is used.')
    parser.add_argument('--entry-iou-thr', type=float, default=0.10,
                        help='Loose decoded-entry threshold used by the old probe.')
    parser.add_argument('--usable-riou-thr', type=float, default=0.50,
                        help='Control-facing usable geometry threshold.')
    parser.add_argument('--near-riou-thr', type=float, default=0.45,
                        help='Borderline geometry threshold below usable.')
    parser.add_argument('--score-thr', type=float, default=0.05,
                        help='Score threshold for reporting global score state.')
    return parser.parse_args()


def longest_false_span(flags: Sequence[bool], frames: Sequence[int]) -> Dict:
    best_len = cur_len = 0
    best_start = best_end = None
    cur_start = None
    for ok, fid in zip(flags, frames):
        if ok:
            cur_len = 0
            cur_start = None
            continue
        if cur_len == 0:
            cur_start = int(fid)
        cur_len += 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start
            best_end = int(fid)
    return dict(length=int(best_len), start=best_start, end=best_end)


def contiguous_spans(rows: Sequence[Dict], key: str) -> List[Dict]:
    spans = []
    cur_val = None
    cur_rows: List[Dict] = []
    for row in rows:
        val = row.get(key)
        if cur_rows and val != cur_val:
            spans.append(summarize_span(cur_rows, key, cur_val))
            cur_rows = []
        cur_val = val
        cur_rows.append(row)
    if cur_rows:
        spans.append(summarize_span(cur_rows, key, cur_val))
    return spans


def summarize_span(rows: Sequence[Dict], key: str, value) -> Dict:
    dec = [float(r['decoded_riou']) for r in rows]
    topk = [float(r['topk_riou']) for r in rows]
    bright = [
        float(r['brightness']) for r in rows
        if r.get('brightness') is not None
    ]
    return dict(
        key=key,
        value=value,
        start=int(rows[0]['frame']),
        end=int(rows[-1]['frame']),
        length=len(rows),
        decoded_riou_mean=float(np.mean(dec)) if dec else 0.0,
        decoded_riou_min=float(np.min(dec)) if dec else 0.0,
        topk_riou_mean=float(np.mean(topk)) if topk else 0.0,
        brightness_mean=float(np.mean(bright)) if bright else None,
    )


def disease_label(decoded_riou: float, topk_riou: float, global_max: float,
                  entry_iou_thr: float, usable_riou_thr: float,
                  near_riou_thr: float, score_thr: float) -> str:
    loose_entry = decoded_riou >= entry_iou_thr
    usable_geom = decoded_riou >= usable_riou_thr
    usable_score = topk_riou >= usable_riou_thr
    if not loose_entry:
        return 'NO_LOOSE_GEOM_ENTRY'
    if usable_score:
        return 'SCORE_USABLE'
    if usable_geom:
        return 'SCORE_DISEASE_USABLE_GEOM'
    if decoded_riou >= near_riou_thr:
        return 'BORDERLINE_GEOM_SCORE_WEAK'
    if global_max < score_thr:
        return 'GEOM_DISEASE_LOW_SCORE'
    return 'GEOM_DISEASE_SCORE_PRESENT'


def normalize_rows(payload: Dict, args) -> List[Dict]:
    out = []
    rows = sorted(payload.get('rows', []), key=lambda r: int(r['frame']))
    for row in rows:
        frame = int(row['frame'])
        topk = row.get('score_topk', {}) or {}
        decoded = row.get('decoded_center_neighborhood', {}) or {}
        anchor = row.get('anchor_center_neighborhood', {}) or {}
        topk_riou = float(topk.get('best_riou', 0.0) or 0.0)
        decoded_riou = float(decoded.get('best_riou', 0.0) or 0.0)
        anchor_riou = float(anchor.get('best_riou', 0.0) or 0.0)
        global_max = float(row.get('global_max', 0.0) or 0.0)
        label = disease_label(
            decoded_riou, topk_riou, global_max,
            args.entry_iou_thr, args.usable_riou_thr,
            args.near_riou_thr, args.score_thr)
        out.append(dict(
            frame=frame,
            fname=row.get('fname'),
            split=row.get('split'),
            seq=row.get('seq'),
            candidate_source=row.get('candidate_source'),
            candidate_head=row.get('candidate_head'),
            brightness=row.get('brightness'),
            global_max=global_max,
            decoded_riou=decoded_riou,
            anchor_riou=anchor_riou,
            topk_riou=topk_riou,
            top1_riou=float(topk.get('top1_riou', 0.0) or 0.0),
            top1_score=float(topk.get('top1_score', 0.0) or 0.0),
            decoded_raw_count=int(decoded.get('raw_count', decoded.get('count', 0)) or 0),
            loose_entry=decoded_riou >= args.entry_iou_thr,
            usable_geom=decoded_riou >= args.usable_riou_thr,
            usable_score=topk_riou >= args.usable_riou_thr,
            global_low_score=global_max < args.score_thr,
            old_decision=row.get('decision'),
            disease_type=label,
        ))
    return out


def write_csv(path: str, rows: Sequence[Dict]):
    fieldnames = [
        'frame', 'fname', 'split', 'seq', 'candidate_source', 'candidate_head',
        'brightness', 'global_max', 'decoded_riou', 'anchor_riou',
        'topk_riou', 'top1_riou', 'top1_score', 'decoded_raw_count',
        'loose_entry', 'usable_geom', 'usable_score', 'global_low_score',
        'old_decision', 'disease_type',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: Sequence[Dict], args) -> Dict:
    frames = [int(r['frame']) for r in rows]
    decoded_vals = [float(r['decoded_riou']) for r in rows]
    topk_vals = [float(r['topk_riou']) for r in rows]
    global_vals = [float(r['global_max']) for r in rows]
    usable_geom_flags = [bool(r['usable_geom']) for r in rows]
    usable_score_flags = [bool(r['usable_score']) for r in rows]
    score_disease_flags = [
        r['disease_type'] == 'SCORE_DISEASE_USABLE_GEOM' for r in rows
    ]
    hard_geom_disease_flags = [
        str(r['disease_type']).startswith('GEOM_DISEASE') for r in rows
    ]
    borderline_flags = [
        r['disease_type'] == 'BORDERLINE_GEOM_SCORE_WEAK' for r in rows
    ]
    counts = Counter(r['disease_type'] for r in rows)
    usable_geom_count = sum(usable_geom_flags)
    usable_score_count = sum(usable_score_flags)
    score_disease_count = sum(score_disease_flags)
    hard_geom_disease_count = sum(hard_geom_disease_flags)
    borderline_count = sum(borderline_flags)
    unusable_geom_count = len(rows) - usable_geom_count
    return dict(
        frames=len(rows),
        frame_start=min(frames) if frames else None,
        frame_end=max(frames) if frames else None,
        thresholds=dict(
            entry_iou_thr=args.entry_iou_thr,
            usable_riou_thr=args.usable_riou_thr,
            near_riou_thr=args.near_riou_thr,
            score_thr=args.score_thr,
        ),
        counts=dict(counts),
        decoded_riou=dict(
            mean=float(np.mean(decoded_vals)) if decoded_vals else 0.0,
            min=float(np.min(decoded_vals)) if decoded_vals else 0.0,
            max=float(np.max(decoded_vals)) if decoded_vals else 0.0,
        ),
        topk_riou=dict(
            mean=float(np.mean(topk_vals)) if topk_vals else 0.0,
            min=float(np.min(topk_vals)) if topk_vals else 0.0,
            max=float(np.max(topk_vals)) if topk_vals else 0.0,
        ),
        global_max=dict(
            mean=float(np.mean(global_vals)) if global_vals else 0.0,
            min=float(np.min(global_vals)) if global_vals else 0.0,
            max=float(np.max(global_vals)) if global_vals else 0.0,
        ),
        usable_geom_count=usable_geom_count,
        unusable_geom_count=unusable_geom_count,
        usable_score_count=usable_score_count,
        score_disease_count=score_disease_count,
        borderline_geom_count=borderline_count,
        hard_geom_disease_count=hard_geom_disease_count,
        # Backward-compatible alias. Prefer hard_geom_disease_count in reports:
        # this excludes borderline frames, while unusable_geom_count includes them.
        geom_disease_count=hard_geom_disease_count,
        longest_unusable_geom_span=longest_false_span(usable_geom_flags, frames),
        longest_unusable_score_span=longest_false_span(usable_score_flags, frames),
        disease_spans=contiguous_spans(rows, 'disease_type'),
        usable_geom_spans=contiguous_spans(rows, 'usable_geom'),
    )


def output_base(input_path: str, args, index: int) -> str:
    if args.name and len(args.input) == 1:
        return args.name
    stem = os.path.splitext(os.path.basename(input_path))[0]
    if len(args.input) > 1:
        return f'{index:02d}_{stem}'
    return stem


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    combined_summary = []
    for index, input_path in enumerate(args.input, start=1):
        with open(input_path, 'r') as f:
            payload = json.load(f)
        rows = normalize_rows(payload, args)
        summary = summarize(rows, args)
        base = output_base(input_path, args, index)
        csv_path = os.path.join(args.out_dir, f'{base}_per_frame.csv')
        json_path = os.path.join(args.out_dir, f'{base}_summary.json')
        write_csv(csv_path, rows)
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        combined_summary.append(dict(
            input=input_path,
            csv=csv_path,
            summary_json=json_path,
            summary=summary,
        ))
        print('=' * 80)
        print(f'CTX ENTRY STRATIFY: {input_path}')
        print(f'frames: {summary["frames"]} '
              f'({summary["frame_start"]}..{summary["frame_end"]})')
        print(f'usable_geom: {summary["usable_geom_count"]}/{summary["frames"]}')
        print(f'unusable_geom: {summary["unusable_geom_count"]}/{summary["frames"]}')
        print(f'usable_score: {summary["usable_score_count"]}/{summary["frames"]}')
        print(f'score_disease: {summary["score_disease_count"]}/{summary["frames"]}')
        print(f'borderline_geom: {summary["borderline_geom_count"]}/{summary["frames"]}')
        print(f'hard_geom_disease: '
              f'{summary["hard_geom_disease_count"]}/{summary["frames"]}')
        span = summary['longest_unusable_geom_span']
        print(f'longest_unusable_geom_span: {span["length"]} '
              f'({span["start"]}..{span["end"]})')
        print('counts:')
        for key, val in sorted(summary['counts'].items()):
            print(f'  {key}: {val}')
        print(f'[out] wrote {csv_path}')
        print(f'[out] wrote {json_path}')

    if len(combined_summary) > 1:
        path = os.path.join(args.out_dir, 'combined_summary.json')
        with open(path, 'w') as f:
            json.dump(combined_summary, f, indent=2)
        print(f'[out] wrote {path}')


if __name__ == '__main__':
    main()
