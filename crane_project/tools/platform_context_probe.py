#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
platform_context_probe.py - zero-training platform-context feasibility probe.

This script tests the precondition for a platform-aware context supervision
branch: in hard miss frames, is the platform region more visible / more salient
than the beam region, and is the platform geometry inferred from calibration
stable enough to serve as an auxiliary training target?

It does not validate the trained mechanism.  It only answers whether the
information source exists before spending a training run.

Example:
  PYTHONPATH=. python3 crane_project/tools/platform_context_probe.py \
    --config crane_project/configs/crane_symeood_k1.py \
    --checkpoint work_dirs/crane_symeood_k1/epoch_24.pth \
    --split test --seq real_seq02 --start 137 --end 169 \
    --calibration calibration_results.json --gpu 2 \
    --out-dir work_dirs/crane_symeood_k1/platform_probe_seq02_137_169
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import mcml_diag as diag  # noqa: E402
from crane_project.tools.ctx_entry_probe import load_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Probe platform visibility and feature saliency.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default='test')
    parser.add_argument('--seq', required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--calibration', default='calibration_results.json',
                        help='calibrate_k.py output with beam/platform pairs.')
    parser.add_argument('--manual-platform-json', default='',
                        help=('Optional manual platform annotations. Accepts a '
                              'JSON object/list with frame ids mapped to polygon '
                              'corners or center/size fields.'))
    parser.add_argument('--manual-platform-mode',
                        choices=['auto', 'polygon', 'center_expand'],
                        default='auto',
                        help=('polygon: use manual polygon directly; '
                              'center_expand: use manual center and expand from '
                              'that center; auto: prefer polygon and fall back '
                              'to center_expand.'))
    parser.add_argument('--manual-platform-source',
                        choices=['center', 'polygon_center'], default='center',
                        help=('For center_expand, prefer explicit manual center '
                              'or the center of manual polygon corners.'))
    parser.add_argument('--platform-width-scale', type=float, default=1.8,
                        help='Width multiplier for center-expanded platform box.')
    parser.add_argument('--platform-height-scale', type=float, default=2.2,
                        help='Height multiplier for center-expanded platform box.')
    parser.add_argument('--platform-min-width', type=float, default=0.0,
                        help='Minimum center-expanded platform box width in px.')
    parser.add_argument('--platform-min-height', type=float, default=0.0,
                        help='Minimum center-expanded platform box height in px.')
    parser.add_argument('--platform-angle-deg', type=float, default=None,
                        help=('Override center-expanded platform box angle in '
                              'degrees. Default follows beam long-axis angle.'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--feature-levels', default='0,1,2',
                        help='Comma-separated FPN levels used for saliency.')
    parser.add_argument('--vis-sample', type=int, default=12,
                        help='Number of overlay frames to save evenly.')
    parser.add_argument('--vis-frames', default='',
                        help='Comma-separated absolute frame ids to save overlays.')
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_center(poly: np.ndarray) -> np.ndarray:
    return poly.astype(np.float32).mean(axis=0)


def edge_lengths(poly: np.ndarray) -> np.ndarray:
    return np.array([
        np.linalg.norm(poly[(i + 1) % 4] - poly[i])
        for i in range(4)
    ], dtype=np.float32)


def order_corners(poly: np.ndarray) -> np.ndarray:
    pts = poly.astype(np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def fit_affine(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    src = order_corners(src)
    dst = order_corners(dst)
    ones = np.ones((src.shape[0], 1), dtype=np.float32)
    design = np.concatenate([src, ones], axis=1)
    try:
        matrix, _, _, _ = np.linalg.lstsq(design, dst, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return matrix.T.astype(np.float32)


def apply_affine(poly: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((poly.shape[0], 1), dtype=np.float32)
    design = np.concatenate([poly.astype(np.float32), ones], axis=1)
    return design @ matrix.T


def load_calibration(calib_path: str, seq: str) -> Dict:
    with open(calib_path, 'r') as f:
        data = json.load(f)
    samples = data.get('samples', [])
    real_samples = [s for s in samples if str(s.get('domain')) == 'real']
    seq_samples = [s for s in real_samples if s.get('sequence') == seq]
    pool = seq_samples or real_samples or samples
    if not pool:
        raise RuntimeError(f'No calibration samples in {calib_path}')

    matrices = []
    k_values = []
    offsets = []
    for sample in pool:
        beam = np.asarray(sample['beam_corners'], dtype=np.float32)
        plat = np.asarray(sample['platform_corners'], dtype=np.float32)
        matrix = fit_affine(beam, plat)
        if matrix is not None:
            matrices.append(matrix)
        kv = sample.get('k_values', {})
        if kv:
            k_values.append(kv)
        offsets.append((polygon_center(plat) - polygon_center(beam)).tolist())
    if not matrices:
        raise RuntimeError(f'No valid affine calibration in {calib_path}')

    return dict(
        source='sequence' if seq_samples else 'real_global',
        sequence=seq,
        sample_count=len(pool),
        affine=np.median(np.stack(matrices, axis=0), axis=0).astype(np.float32),
        offsets=np.asarray(offsets, dtype=np.float32),
        k_values=k_values,
    )


def ann_to_poly(ann_path: str) -> Optional[np.ndarray]:
    if ann_path is None or not os.path.exists(ann_path):
        return None
    with open(ann_path, 'r') as f:
        line = f.readline().strip()
    parts = line.split()
    if len(parts) < 8:
        return None
    return np.asarray([float(x) for x in parts[:8]], dtype=np.float32).reshape(4, 2)


def normalize_frame_key(value) -> Optional[int]:
    if isinstance(value, int):
        return int(value)
    text = str(value)
    stem = os.path.splitext(os.path.basename(text))[0]
    digits = ''.join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits[-5:])
    try:
        return int(text)
    except ValueError:
        return None


def load_manual_platforms(path: str) -> Dict[int, Dict]:
    if not path:
        return {}
    with open(path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = data.get('frames', data.get('annotations', data))
        if isinstance(items, dict):
            iterable = []
            for key, value in items.items():
                if isinstance(value, dict):
                    row = dict(value)
                else:
                    row = {'polygon': value}
                row.setdefault('frame', key)
                iterable.append(row)
        else:
            iterable = items
    else:
        iterable = data

    rows: Dict[int, Dict] = {}
    for item in iterable:
        if not isinstance(item, dict):
            continue
        fid = normalize_frame_key(
            item.get('frame', item.get('frame_id', item.get('image', item.get('img_path')))))
        if fid is None:
            continue
        rows[int(fid)] = item
    return rows


def manual_polygon(item: Dict) -> Optional[np.ndarray]:
    corners = (
        item.get('platform_corners')
        or item.get('corners')
        or item.get('polygon')
        or item.get('poly'))
    if corners is None:
        return None
    arr = np.asarray(corners, dtype=np.float32)
    if arr.size == 8:
        return arr.reshape(4, 2)
    if arr.shape == (4, 2):
        return arr
    return None


def manual_center(item: Dict, poly: Optional[np.ndarray]) -> Optional[np.ndarray]:
    center = item.get('center', item.get('platform_center'))
    if center is not None:
        arr = np.asarray(center, dtype=np.float32).reshape(-1)
        if arr.size >= 2:
            return arr[:2]
    if 'cx' in item and 'cy' in item:
        return np.asarray([float(item['cx']), float(item['cy'])], dtype=np.float32)
    if poly is not None:
        return polygon_center(poly)
    return None


def beam_oriented_box_from_center(
        center: np.ndarray,
        beam: np.ndarray,
        width_scale: float,
        height_scale: float,
        min_width: float,
        min_height: float,
        angle_deg: Optional[float]) -> np.ndarray:
    lengths = edge_lengths(order_corners(beam))
    long_len = float(np.max(lengths))
    short_len = float(np.min(lengths))
    width = max(long_len * float(width_scale), float(min_width))
    height = max(short_len * float(height_scale), float(min_height))

    ordered = order_corners(beam)
    edges = np.roll(ordered, -1, axis=0) - ordered
    edge_norms = np.linalg.norm(edges, axis=1)
    long_edge = edges[int(np.argmax(edge_norms))]
    if angle_deg is None:
        theta = math.atan2(float(long_edge[1]), float(long_edge[0]))
    else:
        theta = math.radians(float(angle_deg))
    ux = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32)
    uy = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float32)

    c = center.astype(np.float32)
    return np.stack([
        c - ux * width / 2 - uy * height / 2,
        c + ux * width / 2 - uy * height / 2,
        c + ux * width / 2 + uy * height / 2,
        c - ux * width / 2 + uy * height / 2,
    ], axis=0).astype(np.float32)


def resolve_platform(
        beam: np.ndarray,
        fid: int,
        calibration: Dict,
        manual_platforms: Dict[int, Dict],
        args) -> Tuple[np.ndarray, str]:
    item = manual_platforms.get(int(fid))
    if item:
        poly = manual_polygon(item)
        if args.manual_platform_mode == 'polygon':
            if poly is None:
                raise RuntimeError(
                    f'Manual platform annotation for frame {fid} has no polygon.')
            return poly.astype(np.float32), 'manual_polygon'
        if args.manual_platform_mode == 'auto' and poly is not None:
            return poly.astype(np.float32), 'manual_polygon'

        center = manual_center(
            item, poly if args.manual_platform_source == 'polygon_center' else None)
        if center is None:
            raise RuntimeError(
                f'Manual platform annotation for frame {fid} has no center.')
        return beam_oriented_box_from_center(
            center=center,
            beam=beam,
            width_scale=args.platform_width_scale,
            height_scale=args.platform_height_scale,
            min_width=args.platform_min_width,
            min_height=args.platform_min_height,
            angle_deg=args.platform_angle_deg,
        ), 'manual_center_expand'

    return apply_affine(beam, calibration['affine']), 'calibration_affine'


def polygon_mask(shape_hw: Tuple[int, int], poly: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    return mask.astype(bool)


def ring_mask(shape_hw: Tuple[int, int], poly: np.ndarray, pad: int = 12) -> np.ndarray:
    h, w = shape_hw
    mask = polygon_mask(shape_hw, poly).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    dil = cv2.dilate(mask, kernel)
    ring = (dil.astype(bool) & (~mask.astype(bool)))
    return ring[:h, :w]


def region_image_stats(gray: np.ndarray, poly: np.ndarray) -> Dict:
    mask = polygon_mask(gray.shape, poly)
    values = gray[mask]
    if values.size == 0:
        return dict(mean=None, std=None, edge_mean=None, ring_contrast=None)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    ring = ring_mask(gray.shape, poly)
    ring_values = gray[ring]
    mean = float(values.mean())
    ring_mean = float(ring_values.mean()) if ring_values.size else mean
    return dict(
        mean=mean,
        std=float(values.std()),
        edge_mean=float(edges[mask].mean()),
        ring_contrast=float(abs(mean - ring_mean)),
        area=float(mask.sum()),
    )


def resize_mask(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    out = cv2.resize(mask.astype(np.uint8), (shape_hw[1], shape_hw[0]),
                     interpolation=cv2.INTER_NEAREST)
    return out.astype(bool)


def feature_map_saliency(feat: torch.Tensor) -> np.ndarray:
    fmap = feat.detach().float()[0]
    sal = fmap.abs().mean(dim=0).cpu().numpy()
    if sal.size == 0:
        return sal
    lo, hi = float(np.percentile(sal, 1)), float(np.percentile(sal, 99))
    if hi > lo:
        sal = np.clip((sal - lo) / (hi - lo), 0.0, 1.0)
    return sal.astype(np.float32)


def region_feature_stats(feats: Sequence[torch.Tensor],
                         beam_mask: np.ndarray,
                         platform_mask: np.ndarray,
                         levels: Sequence[int]) -> Dict:
    rows = {}
    beam_vals = []
    platform_vals = []
    for lvl in levels:
        if lvl < 0 or lvl >= len(feats):
            continue
        sal = feature_map_saliency(feats[lvl])
        if sal.size == 0:
            continue
        bmask = resize_mask(beam_mask, sal.shape)
        pmask = resize_mask(platform_mask, sal.shape)
        bvals = sal[bmask]
        pvals = sal[pmask]
        if bvals.size:
            beam_vals.extend(bvals.tolist())
        if pvals.size:
            platform_vals.extend(pvals.tolist())
        rows[f'feat_l{lvl}_beam_mean'] = float(bvals.mean()) if bvals.size else None
        rows[f'feat_l{lvl}_platform_mean'] = float(pvals.mean()) if pvals.size else None
        rows[f'feat_l{lvl}_platform_over_beam'] = (
            float(pvals.mean() / max(float(bvals.mean()), 1e-6))
            if bvals.size and pvals.size else None)

    bmean = float(np.mean(beam_vals)) if beam_vals else None
    pmean = float(np.mean(platform_vals)) if platform_vals else None
    rows.update(dict(
        feat_beam_mean=bmean,
        feat_platform_mean=pmean,
        feat_platform_over_beam=(
            pmean / max(bmean, 1e-6)
            if bmean is not None and pmean is not None else None),
    ))
    return rows


def ratio_ge(value: Optional[float], threshold: float) -> bool:
    return value is not None and not math.isnan(float(value)) and float(value) >= threshold


def summarize_bool(rows: Sequence[Dict], key: str) -> Dict:
    flags = [bool(r.get(key)) for r in rows]
    return dict(
        count=int(sum(flags)),
        rate=(sum(flags) / len(flags)) if flags else None,
    )


def draw_overlay(img_bgr: np.ndarray, beam: np.ndarray, platform: np.ndarray,
                 out_path: str):
    vis = img_bgr.copy()
    cv2.polylines(vis, [np.round(beam).astype(np.int32)], True, (255, 0, 0), 2)
    cv2.polylines(vis, [np.round(platform).astype(np.int32)], True, (0, 255, 0), 2)
    cv2.putText(vis, 'beam', tuple(np.round(polygon_center(beam)).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, 'platform', tuple(np.round(polygon_center(platform)).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


def write_csv(path: str, rows: Sequence[Dict]):
    fieldnames = []
    seen = set()
    preferred = [
        'frame', 'img_path', 'platform_source', 'brightness',
        'beam_area', 'platform_area', 'platform_over_beam_area',
        'center_offset_x', 'center_offset_y', 'center_offset_norm',
        'beam_edge_mean', 'platform_edge_mean', 'platform_over_beam_edge',
        'beam_ring_contrast', 'platform_ring_contrast',
        'platform_over_beam_ring_contrast',
        'feat_beam_mean', 'feat_platform_mean', 'feat_platform_over_beam',
        'platform_image_visible_proxy',
        'platform_feature_salient_proxy',
        'platform_visible_proxy',
    ]
    for key in preferred + [k for row in rows for k in row.keys()]:
        if key not in seen:
            fieldnames.append(key)
            seen.add(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: Sequence[Dict], calibration: Dict) -> Dict:
    def vals(key):
        return [
            float(r[key]) for r in rows
            if r.get(key) is not None and not math.isnan(float(r[key]))
        ]

    def stats(key):
        v = vals(key)
        return dict(
            mean=float(np.mean(v)) if v else None,
            median=float(np.median(v)) if v else None,
            min=float(np.min(v)) if v else None,
            max=float(np.max(v)) if v else None,
        )

    offsets = calibration.get('offsets')
    offset_norms = (
        np.linalg.norm(offsets, axis=1).astype(float).tolist()
        if offsets is not None and len(offsets) else [])
    return dict(
        frames=len(rows),
        frame_start=int(rows[0]['frame']) if rows else None,
        frame_end=int(rows[-1]['frame']) if rows else None,
        calibration=dict(
            source=calibration['source'],
            sequence=calibration['sequence'],
            sample_count=calibration['sample_count'],
            offset_norm_mean=float(np.mean(offset_norms)) if offset_norms else None,
            offset_norm_std=float(np.std(offset_norms)) if offset_norms else None,
        ),
        platform_image_visible_proxy=summarize_bool(
            rows, 'platform_image_visible_proxy'),
        platform_feature_salient_proxy=summarize_bool(
            rows, 'platform_feature_salient_proxy'),
        platform_visible_proxy=summarize_bool(rows, 'platform_visible_proxy'),
        platform_over_beam_area=stats('platform_over_beam_area'),
        platform_over_beam_edge=stats('platform_over_beam_edge'),
        platform_over_beam_ring_contrast=stats('platform_over_beam_ring_contrast'),
        feat_platform_over_beam=stats('feat_platform_over_beam'),
        brightness=stats('brightness'),
    )


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, 'vis')
    levels = [int(x) for x in args.feature_levels.split(',') if x.strip()]
    calibration = load_calibration(args.calibration, args.seq)
    manual_platforms = load_manual_platforms(args.manual_platform_json)

    model, cfg = load_model(args.config, args.checkpoint, args.gpu)
    transform_compose, _, flip = diag.build_test_transforms(cfg)

    rows: List[Dict] = []
    frame_ids = list(range(int(args.start), int(args.end) + 1))
    vis_ids = set()
    if args.vis_sample and len(frame_ids) > 0:
        take = min(int(args.vis_sample), len(frame_ids))
        vis_ids = set(np.linspace(0, len(frame_ids) - 1, take, dtype=int).tolist())
    explicit_vis_frames = {
        int(x) for x in str(args.vis_frames).split(',') if x.strip()
    }

    for pos, fid in enumerate(frame_ids):
        img_path, ann_path = diag.find_files(args.data_root, args.split, args.seq, fid)
        if img_path is None or ann_path is None:
            print(f'[skip] {args.seq}_{fid:05d}: missing image or ann')
            continue
        beam = ann_to_poly(ann_path)
        if beam is None:
            print(f'[skip] {args.seq}_{fid:05d}: missing beam OBB')
            continue
        platform, platform_source = resolve_platform(
            beam, fid, calibration, manual_platforms, args)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f'[skip] {args.seq}_{fid:05d}: unreadable image')
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        beam_stats = region_image_stats(gray, beam)
        platform_stats = region_image_stats(gray, platform)
        beam_mask = polygon_mask(gray.shape, beam)
        platform_mask = polygon_mask(gray.shape, platform)

        img_tensor, meta, img_stats = diag.preprocess_image(
            img_path, transform_compose, img_scale=None, flip=flip)
        img_tensor = img_tensor.cuda(f'cuda:{args.gpu}')
        with torch.no_grad():
            feats = model.extract_feat(img_tensor)
        feat_stats = region_feature_stats(feats, beam_mask, platform_mask, levels)

        area_ratio = platform_stats['area'] / max(float(beam_stats['area']), 1e-6)
        edge_ratio = (
            platform_stats['edge_mean'] / max(float(beam_stats['edge_mean']), 1e-6)
            if beam_stats.get('edge_mean') is not None
            and platform_stats.get('edge_mean') is not None else None)
        ring_ratio = (
            platform_stats['ring_contrast']
            / max(float(beam_stats['ring_contrast']), 1e-6)
            if beam_stats.get('ring_contrast') is not None
            and platform_stats.get('ring_contrast') is not None else None)
        feat_ratio = feat_stats.get('feat_platform_over_beam')
        platform_image_visible_proxy = bool(
            area_ratio >= 1.05
            and (ratio_ge(edge_ratio, 1.05) or ratio_ge(ring_ratio, 1.05)))
        platform_feature_salient_proxy = bool(
            area_ratio >= 1.05 and ratio_ge(feat_ratio, 1.05))
        # Visibility is an image-domain precondition.  Feature saliency is
        # reported separately and must not rescue an image-invisible frame.
        visible_proxy = platform_image_visible_proxy

        offset = polygon_center(platform) - polygon_center(beam)
        row = dict(
            frame=int(fid),
            img_path=img_path,
            platform_source=platform_source,
            brightness=float(img_stats['raw_brightness']),
            beam_area=float(beam_stats['area']),
            platform_area=float(platform_stats['area']),
            platform_over_beam_area=float(area_ratio),
            center_offset_x=float(offset[0]),
            center_offset_y=float(offset[1]),
            center_offset_norm=float(np.linalg.norm(offset)),
            beam_edge_mean=beam_stats.get('edge_mean'),
            platform_edge_mean=platform_stats.get('edge_mean'),
            platform_over_beam_edge=edge_ratio,
            beam_ring_contrast=beam_stats.get('ring_contrast'),
            platform_ring_contrast=platform_stats.get('ring_contrast'),
            platform_over_beam_ring_contrast=ring_ratio,
            platform_image_visible_proxy=platform_image_visible_proxy,
            platform_feature_salient_proxy=platform_feature_salient_proxy,
            platform_visible_proxy=visible_proxy,
            **feat_stats,
        )
        rows.append(row)
        print(
            f"[{args.seq}_{fid:05d}] "
            f"image_visible={int(platform_image_visible_proxy)} "
            f"feature_salient={int(platform_feature_salient_proxy)} "
            f"visible={int(visible_proxy)} "
            f"source={platform_source} "
            f"area={area_ratio:.2f} edge={edge_ratio} "
            f"contrast={ring_ratio} feat={feat_ratio}")

        if pos in vis_ids or fid in explicit_vis_frames:
            draw_overlay(
                img_bgr, beam, platform,
                os.path.join(vis_dir, f'{args.seq}_{fid:05d}.jpg'))

    csv_path = os.path.join(args.out_dir, 'per_frame.csv')
    json_path = os.path.join(args.out_dir, 'summary.json')
    write_csv(csv_path, rows)
    summary = summarize(rows, calibration)
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('=' * 80)
    print('PLATFORM CONTEXT PROBE')
    print(f'frames: {summary["frames"]} '
          f'({summary["frame_start"]}..{summary["frame_end"]})')
    print(f'calibration: {summary["calibration"]}')
    image_visible = summary['platform_image_visible_proxy']
    feature_salient = summary['platform_feature_salient_proxy']
    visible = summary['platform_visible_proxy']
    print(f'image_visible_proxy: {image_visible["count"]}/'
          f'{summary["frames"]} rate={image_visible["rate"]}')
    print(f'feature_salient_proxy: {feature_salient["count"]}/'
          f'{summary["frames"]} rate={feature_salient["rate"]}')
    print(f'visible_proxy: {visible["count"]}/'
          f'{summary["frames"]} rate={visible["rate"]}')
    print(f'platform/beam area: {summary["platform_over_beam_area"]}')
    print(f'platform/beam edge: {summary["platform_over_beam_edge"]}')
    print(f'platform/beam contrast: '
          f'{summary["platform_over_beam_ring_contrast"]}')
    print(f'platform/beam feature: {summary["feat_platform_over_beam"]}')
    print(f'[out] wrote {csv_path}')
    print(f'[out] wrote {json_path}')
    if vis_ids:
        print(f'[out] wrote overlays to {vis_dir}')


if __name__ == '__main__':
    main()
