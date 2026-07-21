#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a diagnosis-only atlas of target-domain classification false peaks.

This probe deliberately does *not* repeat the candidate-pool oracle gate.  The
oracle result already established that usable geometry exists in 31/33 core
frames but is ranked very deeply by classification.  This tool answers the
remaining question: where do the winning false peaks occur, what do they look
like, and do the same structures recur across consecutive frames?

Target labels are used only to separate false peaks from usable candidates.
Images, crops, boxes, and statistics produced by this script are forbidden as
training inputs.  They are target-dev diagnostics for choosing the next
source-domain intervention.
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import candidate_pool_oracle_probe as pool_probe  # noqa: E402
from crane_project.tools import ctx_entry_probe as entry_probe  # noqa: E402


CANONICAL_SPLIT = 'test'
CANONICAL_SEQ = 'real_seq02'
CANONICAL_START = 137
CANONICAL_END = 169


def parse_args():
    parser = argparse.ArgumentParser(
        description='Map recurrent target-dev classification false peaks.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default=CANONICAL_SPLIT)
    parser.add_argument('--seq', default=CANONICAL_SEQ)
    parser.add_argument('--start', type=int, default=CANONICAL_START)
    parser.add_argument('--end', type=int, default=CANONICAL_END)
    parser.add_argument('--candidate-source', default='main',
                        choices=['main'])
    parser.add_argument('--pool-size', type=int, default=10000)
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--false-iou-thr', type=float, default=0.1)
    parser.add_argument('--false-peaks-per-frame', type=int, default=5)
    parser.add_argument('--false-diversity-iou', type=float, default=0.3)
    parser.add_argument('--false-search-limit', type=int, default=500)
    parser.add_argument('--cluster-center-thr', type=float, default=0.06,
                        help='Normalized image-diagonal distance.')
    parser.add_argument('--cluster-log-area-thr', type=float, default=0.8)
    parser.add_argument('--cluster-angle-thr', type=float, default=35.0)
    parser.add_argument('--cluster-max-frame-gap', type=int, default=2)
    parser.add_argument('--min-recurrent-frames', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--preview-dir', default=None)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--allow-noncanonical', action='store_true',
                        help='Only for interface smoke tests; output cannot '
                             'authorize a method decision.')
    return parser.parse_args()


def validate_args(args) -> bool:
    if args.seed != 0:
        raise ValueError('The unified protocol requires --seed 0')
    if args.end < args.start:
        raise ValueError('--end must be >= --start')
    if args.pool_size <= 0:
        raise ValueError('--pool-size must be positive')
    if args.false_peaks_per_frame <= 0:
        raise ValueError('--false-peaks-per-frame must be positive')
    if args.false_search_limit < args.false_peaks_per_frame:
        raise ValueError('--false-search-limit must cover requested peaks')
    if not 0.0 <= args.false_iou_thr < args.riou_thr <= 1.0:
        raise ValueError(
            'Require 0 <= false-iou-thr < riou-thr <= 1')
    if not 0.0 <= args.false_diversity_iou <= 1.0:
        raise ValueError('--false-diversity-iou must be in [0, 1]')
    if args.cluster_center_thr <= 0.0:
        raise ValueError('--cluster-center-thr must be positive')
    if args.cluster_log_area_thr < 0.0:
        raise ValueError('--cluster-log-area-thr must be non-negative')
    if not 0.0 <= args.cluster_angle_thr <= 90.0:
        raise ValueError('--cluster-angle-thr must be in [0, 90]')
    if args.cluster_max_frame_gap < 1:
        raise ValueError('--cluster-max-frame-gap must be >= 1')
    if args.min_recurrent_frames < 2:
        raise ValueError('--min-recurrent-frames must be >= 2')

    canonical = bool(
        args.split == CANONICAL_SPLIT
        and args.seq == CANONICAL_SEQ
        and args.start == CANONICAL_START
        and args.end == CANONICAL_END
        and args.candidate_source == 'main'
        and args.pool_size == 10000)
    if not canonical and not args.allow_noncanonical:
        raise ValueError(
            'Canonical atlas requires test/real_seq02[137..169], main '
            'candidates, and pool_size=10000. Use --allow-noncanonical only '
            'for an interface smoke test.')
    return canonical


def _scale_factor_xy(meta: Dict) -> Tuple[float, float]:
    scale = meta.get('scale_factor', 1.0)
    if isinstance(scale, torch.Tensor):
        scale = scale.detach().cpu().numpy()
    flat = np.asarray(scale, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError('Empty scale_factor')
    sx = float(flat[0])
    sy = float(flat[1]) if flat.size >= 2 else sx
    if sx <= 0.0 or sy <= 0.0:
        raise ValueError('scale_factor must be positive')
    return sx, sy


def decoded_box_to_original(box: Sequence[float], meta: Dict) -> List[float]:
    """Map an OBB from resized test coordinates back to original pixels."""
    if bool(meta.get('flip', False)):
        raise ValueError('Atlas does not support flipped test preprocessing')
    sx, sy = _scale_factor_xy(meta)
    if abs(sx - sy) > 1e-3:
        raise ValueError(
            'Anisotropic resize requires polygon-level OBB conversion')
    values = [float(value) for value in box[:5]]
    values[0] /= sx
    values[1] /= sy
    values[2] /= sx
    values[3] /= sy
    return values


def _angle_distance_deg(first: float, second: float) -> float:
    """Smallest undirected OBB angle difference in degrees."""
    difference = abs(float(first) - float(second)) % 180.0
    return float(min(difference, 180.0 - difference))


def _normalized_peak(box_ori: Sequence[float], image_shape: Sequence[int]) -> Dict:
    image_h, image_w = float(image_shape[0]), float(image_shape[1])
    cx, cy, width, height, angle = [float(value) for value in box_ori[:5]]
    area_ratio = max(width * height / max(image_h * image_w, 1.0), 1e-12)
    return dict(
        center_x=cx / max(image_w, 1.0),
        center_y=cy / max(image_h, 1.0),
        area_ratio=area_ratio,
        log_area=float(math.log(area_ratio)),
        aspect_ratio=max(width, height) / max(min(width, height), 1e-6),
        angle_deg=float(math.degrees(angle)),
    )


def _cluster_distance(first: Dict, second: Dict) -> Tuple[float, float, float]:
    center = math.hypot(
        float(first['center_x']) - float(second['center_x']),
        float(first['center_y']) - float(second['center_y']))
    log_area = abs(float(first['log_area']) - float(second['log_area']))
    angle = _angle_distance_deg(
        float(first['angle_deg']), float(second['angle_deg']))
    return center, log_area, angle


def cluster_false_peaks(peaks: Sequence[Dict], center_thr: float = 0.06,
                        log_area_thr: float = 0.8,
                        angle_thr: float = 35.0,
                        max_frame_gap: int = 2) -> List[Dict]:
    """Greedily link spatially stable peaks across nearby video frames."""
    clusters = []
    ordered = sorted(peaks, key=lambda item: (
        int(item['frame']), int(item.get('peak_order', 0))))
    for peak in ordered:
        frame = int(peak['frame'])
        best_cluster = None
        best_distance = None
        for cluster in clusters:
            gap = frame - int(cluster['last_frame'])
            if gap < 1 or gap > int(max_frame_gap):
                continue
            center, log_area, angle = _cluster_distance(
                peak['normalized'], cluster['last_normalized'])
            if (center > center_thr or log_area > log_area_thr
                    or angle > angle_thr):
                continue
            distance = center + 0.02 * log_area + 0.001 * angle
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_cluster = cluster
        if best_cluster is None:
            clusters.append(dict(
                cluster_id=len(clusters),
                first_frame=frame,
                last_frame=frame,
                frames=[frame],
                peak_orders=[int(peak.get('peak_order', 0))],
                occurrences=[dict(
                    frame=frame,
                    peak_order=int(peak.get('peak_order', 0)))],
                scores=[float(peak['score'])],
                rious=[float(peak['riou'])],
                centers=[[float(peak['normalized']['center_x']),
                          float(peak['normalized']['center_y'])]],
                last_normalized=dict(peak['normalized'])))
        else:
            best_cluster['last_frame'] = frame
            if frame not in best_cluster['frames']:
                best_cluster['frames'].append(frame)
            best_cluster['peak_orders'].append(
                int(peak.get('peak_order', 0)))
            best_cluster['occurrences'].append(dict(
                frame=frame,
                peak_order=int(peak.get('peak_order', 0))))
            best_cluster['scores'].append(float(peak['score']))
            best_cluster['rious'].append(float(peak['riou']))
            best_cluster['centers'].append([
                float(peak['normalized']['center_x']),
                float(peak['normalized']['center_y'])])
            best_cluster['last_normalized'] = dict(peak['normalized'])

    for cluster in clusters:
        centers = np.asarray(cluster.pop('centers'), dtype=np.float64)
        cluster.pop('last_normalized', None)
        frames = sorted(set(int(value) for value in cluster['frames']))
        longest = 0
        current = 0
        previous = None
        for frame in frames:
            current = current + 1 if previous is not None and frame == previous + 1 else 1
            longest = max(longest, current)
            previous = frame
        cluster['frames'] = frames
        cluster['frame_count'] = len(frames)
        cluster['longest_consecutive'] = int(longest)
        cluster['center_mean'] = centers.mean(axis=0).tolist()
        cluster['center_std'] = centers.std(axis=0).tolist()
        cluster['score_mean'] = float(np.mean(cluster.pop('scores')))
        cluster['riou_mean'] = float(np.mean(cluster.pop('rious')))
    return sorted(
        clusters,
        key=lambda item: (item['frame_count'], item['longest_consecutive']),
        reverse=True)


def _box_polygon(box_ori: Sequence[float]) -> np.ndarray:
    import cv2

    cx, cy, width, height, angle = [float(value) for value in box_ori[:5]]
    rect = ((cx, cy), (max(width, 1.0), max(height, 1.0)),
            float(math.degrees(angle)))
    return cv2.boxPoints(rect).astype(np.float32)


def _region_stats(image_bgr: np.ndarray, box_ori: Sequence[float]) -> Dict:
    import cv2

    image_h, image_w = image_bgr.shape[:2]
    polygon = _box_polygon(box_ori)
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 255)
    kernel_size = max(3, int(round(
        min(float(box_ori[2]), float(box_ori[3])) * 0.25)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, 101)
    dilated = cv2.dilate(
        mask, np.ones((kernel_size, kernel_size), dtype=np.uint8))
    ring = (dilated > 0) & (mask == 0)
    inside = mask > 0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    edges = cv2.Canny(image_bgr, 50, 150) > 0

    def stats(region):
        values = gray[region]
        if values.size == 0:
            return dict(count=0)
        return dict(
            count=int(values.size),
            brightness_mean=float(values.mean()),
            brightness_std=float(values.std()),
            brightness_p10=float(np.percentile(values, 10)),
            brightness_p90=float(np.percentile(values, 90)),
            dark_ratio=float((values < 32.0).mean()),
            bright_ratio=float((values > 224.0).mean()),
            edge_density=float(edges[region].mean()),
            saturation_mean=float(hsv[..., 1][region].mean()),
        )

    return dict(inside=stats(inside), context_ring=stats(ring))


def _candidate_record(frame: int, peak_order: int, index: int,
                      boxes: torch.Tensor, scores: torch.Tensor,
                      levels: torch.Tensor, anchor_centers: torch.Tensor,
                      ious: torch.Tensor,
                      meta: Dict, image_bgr: np.ndarray) -> Dict:
    box_img = boxes[index].detach().cpu().float().tolist()
    anchor_center = anchor_centers[index].detach().cpu().float().tolist()
    box_ori = decoded_box_to_original(box_img, meta)
    riou = float(ious[index].item())
    return dict(
        frame=int(frame),
        peak_order=int(peak_order),
        candidate_index=int(index),
        score=float(scores[index].item()),
        riou=riou,
        fpn_level=int(levels[index].item()),
        box_img=[float(value) for value in box_img],
        box_ori=[float(value) for value in box_ori],
        normalized=_normalized_peak(box_ori, image_bgr.shape),
        origin=candidate_origin_geometry(
            box_img, anchor_center, meta['img_shape']),
        appearance=_region_stats(image_bgr, box_ori),
    )


def candidate_origin_geometry(box_img: Sequence[float],
                              anchor_center_img: Sequence[float],
                              img_shape: Sequence[int],
                              border_ratio: float = 0.02) -> Dict:
    """Describe where a decoded candidate originated, without changing it."""
    image_h, image_w = float(img_shape[0]), float(img_shape[1])
    anchor_x, anchor_y = [float(value) for value in anchor_center_img[:2]]
    decoded_x, decoded_y = [float(value) for value in box_img[:2]]

    anchor_margins = dict(
        left=anchor_x / max(image_w, 1.0),
        right=(image_w - anchor_x) / max(image_w, 1.0),
        top=anchor_y / max(image_h, 1.0),
        bottom=(image_h - anchor_y) / max(image_h, 1.0))
    decoded_margins = dict(
        left=decoded_x / max(image_w, 1.0),
        right=(image_w - decoded_x) / max(image_w, 1.0),
        top=decoded_y / max(image_h, 1.0),
        bottom=(image_h - decoded_y) / max(image_h, 1.0))
    anchor_nearest_edge = min(anchor_margins, key=anchor_margins.get)
    decoded_nearest_edge = min(decoded_margins, key=decoded_margins.get)
    center_shift = math.hypot(decoded_x - anchor_x, decoded_y - anchor_y)
    image_diag = math.hypot(image_w, image_h)
    boundary_eps = 1e-3

    return dict(
        anchor_center_img=[anchor_x, anchor_y],
        anchor_nearest_edge=anchor_nearest_edge,
        anchor_edge_distance_ratio=float(
            anchor_margins[anchor_nearest_edge]),
        anchor_near_border=bool(
            anchor_margins[anchor_nearest_edge] <= border_ratio),
        decoded_nearest_edge=decoded_nearest_edge,
        decoded_edge_distance_ratio=float(
            decoded_margins[decoded_nearest_edge]),
        decoded_near_border=bool(
            decoded_margins[decoded_nearest_edge] <= border_ratio),
        decoded_on_boundary=bool(
            decoded_x <= boundary_eps or decoded_y <= boundary_eps
            or decoded_x >= image_w - boundary_eps
            or decoded_y >= image_h - boundary_eps),
        anchor_to_decoded_shift_px=float(center_shift),
        anchor_to_decoded_shift_ratio=float(
            center_shift / max(image_diag, 1.0)),
    )


def _select_diverse_false_indices(pool_boxes: torch.Tensor,
                                  pool_ious: torch.Tensor,
                                  unusable_iou_thr: float,
                                  max_peaks: int,
                                  search_limit: int,
                                  diversity_iou: float) -> List[int]:
    from mmcv.ops import box_iou_rotated

    false_positions = torch.nonzero(
        pool_ious < float(unusable_iou_thr), as_tuple=False).reshape(-1)
    false_positions = false_positions[:int(search_limit)]
    selected = []
    for position in false_positions.tolist():
        if selected:
            overlaps = box_iou_rotated(
                pool_boxes[position:position + 1].float(),
                pool_boxes[selected].float()).reshape(-1)
            if bool((overlaps > float(diversity_iou)).any()):
                continue
        selected.append(int(position))
        if len(selected) >= int(max_peaks):
            break
    return selected


def _draw_preview(image_bgr: np.ndarray, gt_box: Sequence[float],
                  usable: Optional[Dict], false_peaks: Sequence[Dict],
                  output_path: str):
    import cv2

    canvas = image_bgr.copy()

    def draw(box, color, text):
        polygon = np.round(_box_polygon(box)).astype(np.int32)
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
        anchor = tuple(polygon[0].tolist())
        cv2.putText(canvas, text, anchor, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)

    draw(gt_box, (0, 220, 0), 'GT')
    if usable is not None:
        draw(usable['box_ori'], (255, 180, 0),
             'U rank={} s={:.4f}'.format(
                 usable['cls_rank'], usable['score']))
    for peak in false_peaks:
        draw(peak['box_ori'], (0, 0, 255),
             'F{} s={:.4f}'.format(peak['peak_order'], peak['score']))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)


def analyze_frame(model, transform_compose, img_scale, flip, args,
                  frame: int) -> Dict:
    import cv2
    from mmcv.ops import box_iou_rotated

    diag = entry_probe.get_diag()
    img_path, ann_path = diag.find_files(
        args.data_root, args.split, args.seq, frame)
    if img_path is None or ann_path is None:
        raise RuntimeError('Missing target-dev frame {}'.format(frame))
    gts = diag.parse_dota_ann(ann_path)
    if not gts:
        raise RuntimeError('Missing target-dev GT at frame {}'.format(frame))
    image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError('Failed to read {}'.format(img_path))
    img_tensor, meta, _ = diag.preprocess_image(
        img_path, transform_compose, img_scale, flip)
    preprocess = pool_probe.build_preprocess_summary(
        img_tensor, meta, img_scale, flip)
    img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))

    with torch.no_grad():
        features = model.extract_feat(img_tensor)
        candidate_head, cls_scores, bbox_preds = (
            entry_probe.forward_candidate_head(
                model, features, args.candidate_source))
        boxes, scores, levels, anchor_centers, alignment = (
            entry_probe.flatten_decode_candidates(
                candidate_head, cls_scores, bbox_preds, meta['img_shape']))
        scaled_gts = [pool_probe.scale_gt_to_img(gt, meta) for gt in gts]
        gt_boxes = torch.stack([
            entry_probe.gt_to_tensor(gt, boxes.device).reshape(5)
            for gt in scaled_gts])
        ious = box_iou_rotated(boxes.float(), gt_boxes.float()).max(dim=1).values
        actual_pool = min(int(args.pool_size), int(scores.numel()))
        pool_scores, pool_indices = torch.topk(
            scores, k=actual_pool, largest=True, sorted=True)
        pool_boxes = boxes[pool_indices]
        pool_levels = levels[pool_indices]
        pool_ious = ious[pool_indices]

        false_positions = _select_diverse_false_indices(
            pool_boxes, pool_ious, args.riou_thr,
            args.false_peaks_per_frame, args.false_search_limit,
            args.false_diversity_iou)
        false_peaks = [
            _candidate_record(
                frame, order + 1, int(pool_indices[position].item()),
                boxes, scores, levels, anchor_centers, ious,
                meta, image_bgr)
            for order, position in enumerate(false_positions)
        ]
        for peak in false_peaks:
            peak['candidate_kind'] = (
                'hard_false_background'
                if peak['riou'] < args.false_iou_thr
                else 'near_target_unusable')

        usable_positions = torch.nonzero(
            pool_ious >= float(args.riou_thr), as_tuple=False).reshape(-1)
        usable = None
        if usable_positions.numel() > 0:
            usable_position = int(usable_positions[0].item())
            usable_index = int(pool_indices[usable_position].item())
            usable = _candidate_record(
                frame, 0, usable_index, boxes, scores, levels,
                anchor_centers, ious,
                meta, image_bgr)
            usable['cls_rank'] = usable_position + 1
            usable['candidate_kind'] = 'usable_correct'

    gt_ori = gts[0]
    gt_box_ori = [
        float(gt_ori['cx']), float(gt_ori['cy']), float(gt_ori['w']),
        float(gt_ori['h']), float(math.radians(gt_ori['angle']))]
    if args.preview_dir:
        preview_path = os.path.join(
            args.preview_dir, '{}_{:05d}.jpg'.format(args.seq, frame))
        _draw_preview(image_bgr, gt_box_ori, usable, false_peaks, preview_path)
    else:
        preview_path = None

    return dict(
        frame=int(frame),
        image=os.path.relpath(img_path, os.path.realpath(args.data_root)),
        preview=preview_path,
        brightness=float(cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2GRAY).mean()),
        gt_box_ori=gt_box_ori,
        candidate_count=int(scores.numel()),
        pool_size=int(actual_pool),
        top1_is_false=bool(float(pool_ious[0].item()) < args.riou_thr),
        top1_score=float(pool_scores[0].item()),
        top1_riou=float(pool_ious[0].item()),
        false_peaks=false_peaks,
        usable_candidate=usable,
        dense_best_riou=float(ious.max().item()),
        decode_alignment=alignment,
        preprocess=preprocess,
    )


def build_summary(rows: Sequence[Dict], clusters: Sequence[Dict],
                  min_recurrent_frames: int) -> Dict:
    total = len(rows)
    usable_rows = [row for row in rows if row['usable_candidate'] is not None]
    ranks = [row['usable_candidate']['cls_rank'] for row in usable_rows]
    score_ratios = [
        float(row['top1_score'])
        / max(float(row['usable_candidate']['score']), 1e-12)
        for row in usable_rows]
    false_peaks = [peak for row in rows for peak in row.get('false_peaks', [])]
    top1_false_peaks = [
        peak for peak in false_peaks if int(peak.get('peak_order', 0)) == 1]
    removed_per_frame = [
        sum(int(level.get('padding_anchors_removed', 0))
            for level in row.get('decode_alignment', []))
        for row in rows]
    anchors_before_per_frame = [
        sum(int(level.get('anchors_before_content_filter',
                          level.get('anchors', 0)))
            for level in row.get('decode_alignment', []))
        for row in rows]
    total_removed = sum(removed_per_frame)
    total_before = sum(anchors_before_per_frame)

    def histogram(items, key):
        values = {}
        for item in items:
            value = str(item[key])
            values[value] = values.get(value, 0) + 1
        return values
    recurrent = [
        cluster for cluster in clusters
        if int(cluster['frame_count']) >= int(min_recurrent_frames)]
    top1_cluster_frames = set()
    for cluster in recurrent:
        top1_cluster_frames.update(
            int(item['frame']) for item in cluster['occurrences']
            if int(item['peak_order']) == 1)
    return dict(
        frames=total,
        false_top1_frames=sum(bool(row['top1_is_false']) for row in rows),
        usable_frames=len(usable_rows),
        geometry_miss_frames=[
            int(row['frame']) for row in rows
            if row['usable_candidate'] is None],
        usable_rank_median=(float(np.median(ranks)) if ranks else None),
        usable_rank_p90=(float(np.percentile(ranks, 90)) if ranks else None),
        top1_to_usable_score_ratio_median=(
            float(np.median(score_ratios)) if score_ratios else None),
        padding_anchors_removed_total=int(total_removed),
        padding_anchors_removed_median_per_frame=(
            float(np.median(removed_per_frame))
            if removed_per_frame else 0.0),
        padding_anchor_removed_ratio=(
            float(total_removed / total_before) if total_before else 0.0),
        top1_source_anchor_near_border=sum(
            bool(peak.get('origin', {}).get('anchor_near_border', False))
            for peak in top1_false_peaks),
        top1_decoded_near_border=sum(
            bool(peak.get('origin', {}).get('decoded_near_border', False))
            for peak in top1_false_peaks),
        top1_decoded_on_boundary=sum(
            bool(peak.get('origin', {}).get('decoded_on_boundary', False))
            for peak in top1_false_peaks),
        top1_anchor_to_decoded_shift_ratio_median=(
            float(np.median([
                peak['origin']['anchor_to_decoded_shift_ratio']
                for peak in top1_false_peaks if 'origin' in peak]))
            if any('origin' in peak for peak in top1_false_peaks) else None),
        false_candidate_kind_histogram=histogram(
            false_peaks, 'candidate_kind'),
        false_fpn_level_histogram=histogram(false_peaks, 'fpn_level'),
        usable_fpn_level_histogram=histogram(
            [row['usable_candidate'] for row in usable_rows], 'fpn_level'),
        false_peak_clusters=len(clusters),
        recurrent_clusters=len(recurrent),
        recurrent_top1_coverage=(
            float(len(top1_cluster_frames) / total) if total else 0.0),
        recurrent_cluster_ids=[
            int(cluster['cluster_id']) for cluster in recurrent],
    )


def main():
    args = parse_args()
    canonical = validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, cfg = entry_probe.load_model(
        args.config, args.checkpoint, args.gpu)
    diag = entry_probe.get_diag()
    transform_compose, img_scale, flip = diag.build_test_transforms(cfg)
    if flip:
        raise RuntimeError('Canonical target atlas requires flip=False')

    rows = []
    for frame in range(args.start, args.end + 1):
        row = analyze_frame(
            model, transform_compose, img_scale, flip, args, frame)
        rows.append(row)
        usable_rank = (None if row['usable_candidate'] is None
                       else row['usable_candidate']['cls_rank'])
        print('[{}_{:05d}] top1_iou={:.3f} usable_rank={} false_peaks={}'.format(
            args.seq, frame, row['top1_riou'], usable_rank,
            len(row['false_peaks'])))

    all_false_peaks = [
        peak for row in rows for peak in row['false_peaks']]
    clusters = cluster_false_peaks(
        all_false_peaks,
        center_thr=args.cluster_center_thr,
        log_area_thr=args.cluster_log_area_thr,
        angle_thr=args.cluster_angle_thr,
        max_frame_gap=args.cluster_max_frame_gap)
    summary = build_summary(rows, clusters, args.min_recurrent_frames)
    payload = dict(
        probe='target_hard_negative_atlas',
        canonical_protocol=bool(canonical),
        data_role='target_dev',
        split=args.split,
        seq=args.seq,
        start=int(args.start),
        end=int(args.end),
        uses_target_labels=True,
        diagnosis_only=True,
        eligible_for_training=False,
        eligible_for_checkpoint_selection=False,
        must_not_export_target_crops_to_training=True,
        config=args.config,
        checkpoint=args.checkpoint,
        parameters=dict(
            pool_size=int(args.pool_size),
            riou_thr=float(args.riou_thr),
            false_iou_thr=float(args.false_iou_thr),
            false_peaks_per_frame=int(args.false_peaks_per_frame),
            false_diversity_iou=float(args.false_diversity_iou),
            cluster_center_thr=float(args.cluster_center_thr),
            cluster_log_area_thr=float(args.cluster_log_area_thr),
            cluster_angle_thr=float(args.cluster_angle_thr),
            cluster_max_frame_gap=int(args.cluster_max_frame_gap),
            min_recurrent_frames=int(args.min_recurrent_frames)),
        summary=summary,
        clusters=clusters,
        rows=rows)
    output_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.out_json, 'w') as handle:
        json.dump(payload, handle, indent=2)

    print('\nTARGET HARD-NEGATIVE ATLAS')
    print('frames:                 {}'.format(summary['frames']))
    print('false top-1 frames:     {}'.format(summary['false_top1_frames']))
    print('usable frames:          {}'.format(summary['usable_frames']))
    print('geometry misses:        {}'.format(summary['geometry_miss_frames']))
    print('usable rank median/p90: {} / {}'.format(
        summary['usable_rank_median'], summary['usable_rank_p90']))
    print('padding anchors removed: {} ({:.1%}, median {:.0f}/frame)'.format(
        summary['padding_anchors_removed_total'],
        summary['padding_anchor_removed_ratio'],
        summary['padding_anchors_removed_median_per_frame']))
    print('top1 source/decoded edge: {} / {} (boundary={})'.format(
        summary['top1_source_anchor_near_border'],
        summary['top1_decoded_near_border'],
        summary['top1_decoded_on_boundary']))
    print('top1 anchor->box shift:  {}'.format(
        summary['top1_anchor_to_decoded_shift_ratio_median']))
    print('recurrent clusters:     {}'.format(summary['recurrent_clusters']))
    print('recurrent top1 cover:   {:.1%}'.format(
        summary['recurrent_top1_coverage']))
    print('[out] wrote {}'.format(os.path.abspath(args.out_json)))
    print('[policy] TARGET-DEV DIAGNOSIS ONLY; DO NOT TRAIN ON OUTPUT CROPS')


if __name__ == '__main__':
    main()
