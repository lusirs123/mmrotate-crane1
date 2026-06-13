#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze hard Crane test frames from DOTA-style prediction txt files.

This is a lightweight failure-audit helper for the CraneDataset experiments.
It reports missing/empty prediction files, continuous empty/miss segments,
GT geometry for the hardest frames, and prediction score distributions.
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict

import cv2
import numpy as np


def parse_seq_frame(path):
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r'^(real|sim)_(.+)_(\d+)$', name)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    m = re.match(r'^(.+)_(\d+)$', name)
    if m:
        return 'unknown', m.group(1), int(m.group(2))
    return 'unknown', 'default', abs(hash(name)) % (10 ** 8)


def normalize_angle_deg(angle):
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def poly_to_obb(poly):
    pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, 8, 2)]
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    edges = []
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        edges.append((length, angle))
    e0, e1 = edges[0], edges[1]
    if e0[0] >= e1[0]:
        w, h, angle = e0[0], e1[0], e0[1]
    else:
        w, h, angle = e1[0], e0[0], e1[1]
    return {
        'cx': cx,
        'cy': cy,
        'w': w,
        'h': h,
        'angle_deg': normalize_angle_deg(angle),
        'diag': math.hypot(w, h),
        'poly': [float(x) for x in poly],
    }


def parse_dota_file(path, is_pred=False):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                poly = [float(x) for x in parts[:8]]
            except ValueError:
                continue
            box = poly_to_obb(poly)
            score = None
            if is_pred and len(parts) >= 9:
                try:
                    score = float(parts[8])
                except ValueError:
                    score = None
            box['score'] = score
            boxes.append(box)
    return boxes


def dist(a, b):
    return math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy'])


def best_prediction(preds):
    if not preds:
        return None
    with_score = [p for p in preds if p.get('score') is not None]
    if with_score:
        return max(with_score, key=lambda p: p['score'])
    return preds[0]


def percentile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def score_summary(scores):
    if not scores:
        return {}
    return {
        'count': len(scores),
        'min': round(min(scores), 6),
        'p10': round(percentile(scores, 0.10), 6),
        'median': round(percentile(scores, 0.50), 6),
        'p90': round(percentile(scores, 0.90), 6),
        'max': round(max(scores), 6),
        'mean': round(sum(scores) / len(scores), 6),
    }


def value_summary(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {}
    return {
        'count': len(values),
        'min': round(min(values), 6),
        'p10': round(percentile(values, 0.10), 6),
        'median': round(percentile(values, 0.50), 6),
        'p90': round(percentile(values, 0.90), 6),
        'max': round(max(values), 6),
        'mean': round(sum(values) / len(values), 6),
    }


def summarize_metrics(rows):
    if not rows:
        return {}
    out = {}
    keys = sorted({k for r in rows for k in r.keys()})
    for key in keys:
        vals = [r.get(key) for r in rows]
        out[key] = value_summary(vals)
    return out


def image_name_from_ann(filename):
    return os.path.splitext(filename)[0] + '.jpg'


def gt_crop(img, gt, pad=32):
    if img is None or not gt or 'poly' not in gt:
        return None
    pts = np.array(gt['poly'], dtype=np.float32).reshape(4, 2)
    h, w = img.shape[:2]
    x1 = max(0, int(np.floor(pts[:, 0].min() - pad)))
    y1 = max(0, int(np.floor(pts[:, 1].min() - pad)))
    x2 = min(w, int(np.ceil(pts[:, 0].max() + pad)))
    y2 = min(h, int(np.ceil(pts[:, 1].max() + pad)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def degradation_metrics(img):
    if img is None or img.size == 0:
        return None
    imgf = img.astype(np.float32)
    b, g, r = cv2.split(imgf)
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        'luma_mean': float(y.mean()),
        'luma_p10': float(np.percentile(y, 10)),
        'luma_p90': float(np.percentile(y, 90)),
        'luma_std': float(y.std()),
        'dark_ratio': float((y < 40).mean()),
        'very_dark_ratio': float((y < 25).mean()),
        'bright_ratio': float((y > 220).mean()),
        'sat_mean': float(sat.mean()),
        'b_mean': float(b.mean()),
        'g_mean': float(g.mean()),
        'r_mean': float(r.mean()),
        'green_cast': float((g - (r + b) / 2.0).mean()),
        'cyan_cast': float(((g + b) / 2.0 - r).mean()),
        'lap_var': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def attach_image_stats(records, img_dir, crop_pad=32):
    if not img_dir:
        return {}

    groups = {
        'all': records,
        'hit': [r for r in records if r['hit']],
        'miss': [r for r in records if r['miss']],
        'empty': [r for r in records if r['empty']],
        'nonempty_miss': [
            r for r in records if r['miss'] and not r['empty']],
    }
    top_miss = continuous_segments(records, 'miss')[:5]
    top_files = set()
    for seg in top_miss:
        for frame_id in range(seg['start_frame'], seg['end_frame'] + 1):
            top_files.add(
                f"{seg['domain']}_{seg['seq_id']}_{frame_id:05d}.txt")
    groups['top_miss_segments'] = [
        r for r in records if r['filename'] in top_files]
    worst_records = [r for r in records if r['miss']]
    worst_records.sort(key=lambda r: (
        -1 if r['center_dist'] is None else -r['center_dist'],
        r['domain'], r['seq_id'], r['frame_id']))
    groups['worst_miss_frames'] = worst_records[:min(20, len(worst_records))]

    out = {}
    cache = {}
    for name, items in groups.items():
        full_rows = []
        crop_rows = []
        missing = 0
        for r in items:
            img_name = image_name_from_ann(r['filename'])
            img_path = os.path.join(img_dir, img_name)
            if img_path not in cache:
                cache[img_path] = cv2.imread(img_path)
            img = cache[img_path]
            if img is None:
                missing += 1
                continue
            full = degradation_metrics(img)
            if full:
                full_rows.append(full)
            crop = gt_crop(img, r.get('gt'), crop_pad)
            crop_stat = degradation_metrics(crop)
            if crop_stat:
                crop_rows.append(crop_stat)
        out[name] = {
            'num_frames': len(items),
            'num_images_found': len(items) - missing,
            'num_images_missing': missing,
            'full_image': summarize_metrics(full_rows),
            'gt_crop': summarize_metrics(crop_rows),
        }
    return out


def continuous_segments(records, flag_name):
    segments = []
    grouped = defaultdict(list)
    for r in records:
        grouped[(r['domain'], r['seq_id'])].append(r)
    for (domain, seq_id), frames in grouped.items():
        frames = sorted(frames, key=lambda x: x['frame_id'])
        start = None
        prev = None
        items = []
        for r in frames:
            is_bad = bool(r[flag_name])
            consecutive = prev is not None and r['frame_id'] == prev['frame_id'] + 1
            if is_bad:
                if start is None or not consecutive:
                    if start is not None:
                        segments.append(make_segment(domain, seq_id, start, prev, items))
                    start = r
                    items = []
                items.append(r)
            elif start is not None:
                segments.append(make_segment(domain, seq_id, start, prev, items))
                start = None
                items = []
            prev = r
        if start is not None:
            segments.append(make_segment(domain, seq_id, start, prev, items))
    segments.sort(key=lambda s: (-s['length'], s['domain'], s['seq_id'], s['start_frame']))
    return segments


def make_segment(domain, seq_id, start, end, items):
    gt_diags = [x['gt']['diag'] for x in items if x.get('gt')]
    gt_ws = [x['gt']['w'] for x in items if x.get('gt')]
    gt_hs = [x['gt']['h'] for x in items if x.get('gt')]
    pred_scores = [x['pred_score'] for x in items if x.get('pred_score') is not None]
    return {
        'domain': domain,
        'seq_id': seq_id,
        'start_frame': start['frame_id'],
        'end_frame': end['frame_id'],
        'length': len(items),
        'first_file': start['filename'],
        'last_file': end['filename'],
        'gt_diag_mean': round(sum(gt_diags) / len(gt_diags), 3) if gt_diags else None,
        'gt_w_mean': round(sum(gt_ws) / len(gt_ws), 3) if gt_ws else None,
        'gt_h_mean': round(sum(gt_hs) / len(gt_hs), 3) if gt_hs else None,
        'pred_score_summary': score_summary(pred_scores),
    }


def analyze(args):
    gt_files = sorted(
        f for f in os.listdir(args.gt_dir)
        if f.endswith('.txt') and os.path.isfile(os.path.join(args.gt_dir, f))
    )
    records = []
    for filename in gt_files:
        gt_path = os.path.join(args.gt_dir, filename)
        pred_path = os.path.join(args.pred_dir, filename)
        domain, seq_id, frame_id = parse_seq_frame(filename)
        if args.domain != 'all' and domain != args.domain:
            continue

        gt_boxes = parse_dota_file(gt_path, is_pred=False)
        pred_boxes = parse_dota_file(pred_path, is_pred=True)
        gt = gt_boxes[0] if gt_boxes else None
        pred = best_prediction(pred_boxes)
        center_dist = dist(gt, pred) if gt and pred else None
        hit = bool(gt and pred and center_dist <= args.center_thresh)
        empty = len(pred_boxes) == 0
        missing_file = not os.path.exists(pred_path)

        records.append({
            'filename': filename,
            'domain': domain,
            'seq_id': seq_id,
            'frame_id': frame_id,
            'gt': gt,
            'pred': pred,
            'pred_count': len(pred_boxes),
            'pred_score': pred.get('score') if pred else None,
            'center_dist': center_dist,
            'hit': hit,
            'miss': bool(gt and not hit),
            'empty': empty,
            'missing_file': missing_file,
        })

    scores = [r['pred_score'] for r in records if r['pred_score'] is not None]
    empty_segments = continuous_segments(records, 'empty')
    miss_segments = continuous_segments(records, 'miss')
    missing_segments = continuous_segments(records, 'missing_file')

    summary = {
        'gt_dir': os.path.abspath(args.gt_dir),
        'pred_dir': os.path.abspath(args.pred_dir),
        'domain': args.domain,
        'center_thresh': args.center_thresh,
        'num_frames': len(records),
        'num_gt_frames': sum(1 for r in records if r['gt'] is not None),
        'num_pred_nonempty': sum(1 for r in records if not r['empty']),
        'num_empty_pred': sum(1 for r in records if r['empty']),
        'num_missing_pred_file': sum(1 for r in records if r['missing_file']),
        'num_center_hits': sum(1 for r in records if r['hit']),
        'num_center_misses': sum(1 for r in records if r['miss']),
        'hit_rate': round(
            sum(1 for r in records if r['hit']) /
            max(1, sum(1 for r in records if r['gt'] is not None)),
            6),
        'score_summary': score_summary(scores),
        'top_empty_segments': empty_segments[:args.topk],
        'top_missing_file_segments': missing_segments[:args.topk],
        'top_miss_segments': miss_segments[:args.topk],
        'worst_miss_frames': worst_miss_frames(records, args.topk),
    }
    if getattr(args, 'img_dir', None):
        summary['image_degradation'] = attach_image_stats(
            records, args.img_dir, args.crop_pad)
    return summary


def compact_box(box):
    if not box:
        return None
    keys = ['cx', 'cy', 'w', 'h', 'angle_deg', 'diag', 'score']
    out = {}
    for k in keys:
        if k in box and box[k] is not None:
            out[k] = round(float(box[k]), 4)
    return out


def worst_miss_frames(records, topk):
    misses = [r for r in records if r['miss']]
    misses.sort(key=lambda r: (
        -1 if r['center_dist'] is None else -r['center_dist'],
        r['domain'], r['seq_id'], r['frame_id']))
    out = []
    for r in misses[:topk]:
        out.append({
            'filename': r['filename'],
            'domain': r['domain'],
            'seq_id': r['seq_id'],
            'frame_id': r['frame_id'],
            'empty': r['empty'],
            'missing_file': r['missing_file'],
            'pred_count': r['pred_count'],
            'pred_score': round(float(r['pred_score']), 6)
            if r['pred_score'] is not None else None,
            'center_dist': round(float(r['center_dist']), 4)
            if r['center_dist'] is not None else None,
            'gt': compact_box(r['gt']),
            'pred': compact_box(r['pred']),
        })
    return out


def print_summary(summary):
    print('\n[Crane Failure Audit]')
    print(f"  GT dir:   {summary['gt_dir']}")
    print(f"  Pred dir: {summary['pred_dir']}")
    print(f"  Domain:   {summary['domain']}")
    print(f"  Center threshold: {summary['center_thresh']}")
    print('\n[Frame Summary]')
    for k in [
            'num_frames', 'num_gt_frames', 'num_pred_nonempty',
            'num_empty_pred', 'num_missing_pred_file',
            'num_center_hits', 'num_center_misses', 'hit_rate']:
        print(f'  {k}: {summary[k]}')
    print('\n[Score Summary]')
    if summary['score_summary']:
        for k, v in summary['score_summary'].items():
            print(f'  {k}: {v}')
    else:
        print('  no prediction scores found')

    def show_segments(title, segments):
        print(f'\n[{title}]')
        if not segments:
            print('  none')
            return
        for i, s in enumerate(segments, 1):
            print(
                f"  {i:02d}. {s['domain']}_{s['seq_id']} "
                f"{s['start_frame']}->{s['end_frame']} "
                f"len={s['length']} "
                f"gt_w/h={s['gt_w_mean']}/{s['gt_h_mean']} "
                f"diag={s['gt_diag_mean']} "
                f"files={s['first_file']}..{s['last_file']}")

    show_segments('Top Empty Prediction Segments', summary['top_empty_segments'])
    show_segments('Top Missing Prediction-File Segments',
                  summary['top_missing_file_segments'])
    show_segments('Top Center-Miss Segments', summary['top_miss_segments'])

    print('\n[Worst Miss Frames]')
    if not summary['worst_miss_frames']:
        print('  none')
    for i, r in enumerate(summary['worst_miss_frames'], 1):
        print(
            f"  {i:02d}. {r['filename']} empty={r['empty']} "
            f"pred_count={r['pred_count']} score={r['pred_score']} "
            f"center_dist={r['center_dist']} gt={r['gt']} pred={r['pred']}")

    if summary.get('image_degradation'):
        print('\n[Image Degradation Summary]')
        for group in ['all', 'miss', 'hit', 'empty', 'nonempty_miss',
                      'top_miss_segments', 'worst_miss_frames']:
            item = summary['image_degradation'].get(group)
            if not item:
                continue
            print(
                f"  {group}: frames={item['num_frames']} "
                f"found={item['num_images_found']} "
                f"missing={item['num_images_missing']}")
            crop = item.get('gt_crop', {})
            full = item.get('full_image', {})
            for region_name, region in [('full', full), ('gt', crop)]:
                brief = []
                for key in [
                        'luma_mean', 'luma_std', 'dark_ratio',
                        'bright_ratio', 'green_cast', 'cyan_cast', 'lap_var']:
                    stat = region.get(key)
                    if stat:
                        brief.append(f"{key}={stat['mean']:.3f}")
                if brief:
                    print(f"    {region_name}: " + ', '.join(brief))


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Crane DOTA prediction failures by sequence.')
    parser.add_argument('--gt-dir', required=True, help='GT annfiles directory')
    parser.add_argument('--pred-dir', required=True,
                        help='Prediction txt directory, e.g. Task1_grab')
    parser.add_argument('--domain', default='real',
                        choices=['real', 'sim', 'unknown', 'all'],
                        help='Domain to analyze, default: real')
    parser.add_argument('--center-thresh', type=float, default=15.0,
                        help='Center hit threshold in pixels')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of top segments/frames to print')
    parser.add_argument('--out', default=None,
                        help='Optional JSON output path')
    parser.add_argument('--img-dir', default=None,
                        help='Optional image directory for degradation stats')
    parser.add_argument('--crop-pad', type=int, default=32,
                        help='Padding pixels around GT crop for image stats')
    args = parser.parse_args()

    summary = analyze(args)
    print_summary(summary)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f'\nSaved JSON: {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
