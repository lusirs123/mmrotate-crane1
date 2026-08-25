"""Inventory whether an additional source dataset supports DINO takeover.

This is a read-only CPU audit.  It reports micro and sequence-macro accuracy
separately so a large easy dataset cannot hide weak sequences.  It also
counts the only samples that can supervise a learned takeover rule:
SymEOOD-present-but-wrong and frozen-DINO-correct frames.
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from crane_project.tools import symeood_dino_distillation_support_audit as base
from crane_project.tools.eval_crane_offline import (
    METRIC_PROTOCOL_VERSION, parse_dota_txt, parse_seq_frame)


PROTOCOL = 'source_only_additional_dataset_support_inventory_v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-audit-json', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument('--source-split', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--hit-iou', type=float, default=0.5)
    parser.add_argument('--teacher-gain-delta', type=float, default=0.10)
    parser.add_argument('--min-router-real-sequences', type=int, default=2)
    parser.add_argument('--min-present-wrong-per-sequence', type=int, default=8)
    return parser.parse_args()


def _safe_source_split(value):
    split = str(value).strip().strip('/')
    parts = [part.lower() for part in split.split('/') if part]
    if not split or any(
            part == 'test' or part.startswith('val') for part in parts):
        raise RuntimeError('Inventory split must be source-only, not test/val')
    return split


def _paths(data_root, split, filename):
    name = Path(filename).name
    image = data_root / split / 'images' / name
    annotation = data_root / split / 'annfiles' / (Path(name).stem + '.txt')
    return image, annotation


def _image_stats(image_path, gt):
    image = cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('Cannot read source image: ' + os.fspath(image_path))
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((height, width), dtype=np.uint8)
    box = ((float(gt[0]), float(gt[1])),
           (float(gt[2]), float(gt[3])),
           float(np.degrees(gt[4])))
    polygon = cv2.boxPoints(box).round().astype(np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    target = gray[mask > 0]
    if not target.size:
        target = gray.reshape(-1)
    diag_ratio = float(
        np.hypot(float(gt[2]), float(gt[3])) / np.hypot(width, height))
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    target_laplacian = laplacian[mask > 0]
    if not target_laplacian.size:
        target_laplacian = laplacian.reshape(-1)
    return dict(
        image_width=int(width),
        image_height=int(height),
        target_diag_ratio=diag_ratio,
        target_area_ratio=float(float(gt[2]) * float(gt[3]) / (width * height)),
        global_luma_mean=float(gray.mean()),
        target_luma_mean=float(target.mean()),
        target_luma_std=float(target.std()),
        target_laplacian_variance=float(target_laplacian.var()),
        target_dark_fraction=float((target < 32).mean()),
        target_bright_clip_fraction=float((target > 245).mean()))


def _quantiles(values):
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        key: float(np.quantile(array, q))
        for key, q in [('min', 0.0), ('p10', 0.1), ('median', 0.5),
                       ('p90', 0.9), ('max', 1.0)]
    }


def inventory(payload, data_root, split, hit_iou, gain_delta,
              min_router_real_sequences, min_present_wrong_per_sequence):
    if payload.get('protocol') != 'source_owned_geometry_union_v2':
        raise RuntimeError('Expected an unrouted all-lane fusion audit')
    metadata = dict(payload.get('metadata') or {})
    if metadata.get('fusion_policy') not in (
            None, 'sym_eood_proposal_dino_roi_union'):
        raise RuntimeError('Unexpected fusion policy in source inventory')
    if bool(metadata.get('conditional_dino_enabled', False)):
        raise RuntimeError('Collection must disable conditional DINO routing')
    records = list(payload.get('records') or [])
    if not records:
        raise RuntimeError('Source audit contains no records')

    rows = []
    buckets = defaultdict(list)
    seen = set()
    for record in records:
        normalized = os.fspath(record.get('filename', '')).replace('\\', '/')
        if '/test/' in normalized or '/val/' in normalized:
            raise RuntimeError('Target test/val record found in source inventory')
        domain, sequence_name, frame = parse_seq_frame(record['filename'])
        sequence = '{}_{}'.format(domain, sequence_name)
        key = (sequence, int(frame))
        if key in seen:
            raise RuntimeError('Duplicate source frame: {}'.format(key))
        seen.add(key)
        image_path, annotation_path = _paths(
            data_root, split, record['filename'])
        if not annotation_path.is_file():
            raise RuntimeError(
                'Source annotation does not exist: ' + os.fspath(annotation_path))
        annotations = parse_dota_txt(os.fspath(annotation_path))
        if not annotations:
            raise RuntimeError('Source annotation has no OBB: ' + os.fspath(annotation_path))
        gt = np.asarray(annotations[0], dtype=np.float64)
        result = base._classify_frame(
            base._box(record, 'sym_eood_original_box'),
            base._box(record, 'dino_native_box'), gt, hit_iou, gain_delta)
        row = dict(
            frame_key='{}|{}'.format(sequence, int(frame)),
            domain=domain,
            sequence=sequence,
            frame=int(frame),
            **result,
            **_image_stats(image_path, gt))
        rows.append(row)
        buckets[sequence].append(row)

    sequence_rows = []
    for sequence in sorted(buckets):
        group = buckets[sequence]
        sequence_rows.append(dict(
            sequence=sequence,
            domain=group[0]['domain'],
            frame_count=len(group),
            sym_hits=sum(int(row['sym_hit']) for row in group),
            dino_hits=sum(int(row['dino_hit']) for row in group),
            present_wrong_teacher_gain_count=sum(
                int(row['present_wrong_teacher_gain']) for row in group),
            hard_teacher_gain_count=sum(
                int(row['hard_teacher_gain']) for row in group),
            student_preserve_count=sum(
                int(row['student_preserve']) for row in group),
            category_counts=dict(Counter(row['category'] for row in group)),
            mean_sym_riou=float(np.mean([row['sym_riou'] for row in group])),
            mean_dino_riou=float(np.mean([row['dino_riou'] for row in group])),
            target_diag_ratio_quantiles=_quantiles(
                [row['target_diag_ratio'] for row in group]),
            target_luma_quantiles=_quantiles(
                [row['target_luma_mean'] for row in group])))

    real_router_sequences = [
        row['sequence'] for row in sequence_rows
        if row['domain'] == 'real'
        and row['present_wrong_teacher_gain_count']
        >= min_present_wrong_per_sequence]
    usable = len(real_router_sequences) >= min_router_real_sequences
    sym_rates = [row['sym_hits'] / row['frame_count'] for row in sequence_rows]
    dino_rates = [row['dino_hits'] / row['frame_count'] for row in sequence_rows]
    return dict(
        protocol=PROTOCOL,
        metric_protocol_version=METRIC_PROTOCOL_VERSION,
        source_split=split,
        target_data_read=False,
        parameter_update_count=0,
        frame_count=len(rows),
        micro_metrics=dict(
            sym_hit_rate=float(np.mean([row['sym_hit'] for row in rows])),
            dino_hit_rate=float(np.mean([row['dino_hit'] for row in rows]))),
        complementarity_counts=dict(
            categories=dict(Counter(row['category'] for row in rows)),
            hard_teacher_gain=sum(
                int(row['hard_teacher_gain']) for row in rows),
            present_wrong_teacher_gain=sum(
                int(row['present_wrong_teacher_gain']) for row in rows),
            student_preserve=sum(
                int(row['student_preserve']) for row in rows)),
        sequence_macro_metrics=dict(
            sym_hit_rate=float(np.mean(sym_rates)),
            dino_hit_rate=float(np.mean(dino_rates)),
            worst_sym_hit_rate=float(min(sym_rates)),
            worst_dino_hit_rate=float(min(dino_rates))),
        difficulty_quantiles=dict(
            target_diag_ratio=_quantiles(
                [row['target_diag_ratio'] for row in rows]),
            target_luma_mean=_quantiles(
                [row['target_luma_mean'] for row in rows]),
            target_laplacian_variance=_quantiles(
                [row['target_laplacian_variance'] for row in rows])),
        router_support=dict(
            min_real_sequences=int(min_router_real_sequences),
            min_present_wrong_per_sequence=int(
                min_present_wrong_per_sequence),
            qualifying_real_sequences=real_router_sequences,
            eligible=bool(usable)),
        recommended_use=(
            'PREREGISTER_SOURCE_ONLY_ROUTER_GATE'
            if usable else 'RETENTION_AND_NONREGRESSION_ONLY'),
        sequence_summaries=sequence_rows,
        frame_rows=rows)


def main():
    args = parse_args()
    source_split = _safe_source_split(args.source_split)
    source_path = Path(args.source_audit_json)
    payload = json.loads(source_path.read_text())
    output = inventory(
        payload, Path(args.data_root), source_split, args.hit_iou,
        args.teacher_gain_delta, args.min_router_real_sequences,
        args.min_present_wrong_per_sequence)
    output['source_audit_json'] = os.fspath(source_path)
    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    temporary.replace(output_path)
    print('[source-inventory] frames={}'.format(output['frame_count']))
    print('[source-inventory] micro={}'.format(output['micro_metrics']))
    print('[source-inventory] sequence_macro={}'.format(
        output['sequence_macro_metrics']))
    print('[source-inventory] router_support={}'.format(
        output['router_support']))
    print('[source-inventory] use={}'.format(output['recommended_use']))


if __name__ == '__main__':
    main()
