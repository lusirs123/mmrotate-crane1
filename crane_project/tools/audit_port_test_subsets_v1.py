#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only per-sequence audit for the port day/night TEST result.

This tool reuses the project's DOTA parser and offline metric implementation.
It does not alter predictions, checkpoints, thresholds, or dataset files.
The requested sequences are evaluated separately so a long failure interval
cannot be hidden by aggregation with another real or simulated sequence.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from crane_project.tools.eval_crane_offline import (
    CraneOfflineEvaluator,
    parse_dota_txt,
    parse_seq_frame,
    compute_riou,
)


TARGETS = (('real', 'seq04'), ('real', 'seq03'), ('sim', 'seq09'))


def _records(gt_dir: str, pred_dir: str) -> List[dict]:
    rows = []
    for name in sorted(os.listdir(gt_dir)):
        if not name.endswith('.txt'):
            continue
        gt_path = os.path.join(gt_dir, name)
        pred_path = os.path.join(pred_dir, name)
        domain, seq_id, frame_id = parse_seq_frame(name)
        gt_boxes = parse_dota_txt(gt_path)
        pred_boxes = parse_dota_txt(pred_path)
        if len(gt_boxes) != 1 or len(pred_boxes) > 1:
            raise ValueError('Expected one GT and at most one prediction: ' + name)
        for box in gt_boxes + pred_boxes:
            if not np.isfinite(box).all() or np.any(box[2:4] <= 0):
                raise ValueError('Invalid box: ' + name)
        rows.append({
            'domain': domain,
            'seq_id': seq_id,
            'frame_id': int(frame_id),
            'gt_box': gt_boxes[0] if gt_boxes else None,
            'pred_box': pred_boxes[0] if pred_boxes else None,
            'score': 1.0 if pred_boxes else 0.0,
            'plc_rope': None,
            'image': name,
        })
    return rows


def _subset_summary(rows: List[dict]) -> Dict[str, object]:
    rows = sorted(rows, key=lambda r: r['frame_id'])
    if not rows:
        return {'frames': 0, 'error': 'no matching GT frames'}

    valid = [r for r in rows if r['gt_box'] is not None]
    outputs = [r for r in valid if r['pred_box'] is not None]
    rios = [compute_riou(r['pred_box'], r['gt_box']) for r in outputs]
    center_hits = []
    for r in outputs:
        center_hits.append(float(np.linalg.norm(
            r['pred_box'][:2] - r['gt_box'][:2]) < 15.0))
    iou_hits = [float(v >= 0.5) for v in rios]
    all_iou_hits = []
    for r in valid:
        all_iou_hits.append(
            float(r['pred_box'] is not None and
                  compute_riou(r['pred_box'], r['gt_box']) >= 0.5))

    evaluator = CraneOfflineEvaluator(mode='test')
    metrics = evaluator.evaluate_records(rows)
    segments = evaluator._split_contiguous_frames(rows)
    no_output_runs = []
    failure_runs = []
    for segment in segments:
        for kind, runs in [('no_output', no_output_runs), ('riou_failure', failure_runs)]:
            start = None
            for r in segment:
                failed = (r['pred_box'] is None or (kind == 'riou_failure' and
                          compute_riou(r['pred_box'], r['gt_box']) < 0.5))
                if failed and start is None:
                    start = r['frame_id']
                if not failed and start is not None:
                    runs.append(dict(start=start, end=r['frame_id'] - 1, length=r['frame_id'] - start))
                    start = None
            if start is not None:
                end = segment[-1]['frame_id']
                runs.append(dict(start=start, end=end, length=end - start + 1))
    return {
        'frames': len(rows),
        'valid_gt_frames': len(valid),
        'output_frames': len(outputs),
        'no_output_frames': len(valid) - len(outputs),
        'output_coverage_pct': round(100.0 * len(outputs) / len(valid), 4)
        if valid else None,
        # Explicitly output-conditional, matching the requested center-rate
        # convention; evaluator R_center is retained separately for audit.
        'output_conditional_center_hit_pct': round(
            100.0 * float(np.mean(center_hits)), 4) if center_hits else None,
        'all_gt_frame_center_hit_pct': round(
            100.0 * float(np.sum(center_hits)) / len(valid), 4)
        if valid else None,
        'output_conditional_mean_RIoU': round(float(np.mean(rios)), 6)
        if rios else None,
        'all_gt_frame_mean_RIoU': round(
            float(np.sum(rios)) / len(valid), 6) if valid else None,
        'output_conditional_RIoU_hit_pct': round(
            100.0 * float(np.mean(iou_hits)), 4) if iou_hits else None,
        'all_gt_frame_RIoU_hit_pct': round(
            100.0 * float(np.sum(all_iou_hits)) / len(valid), 4)
        if valid else None,
        'longest_no_output_run_frames': max([r['length'] for r in no_output_runs], default=0),
        'longest_RIoU_failure_run_frames': max([r['length'] for r in failure_runs], default=0),
        'no_output_intervals': no_output_runs,
        'RIoU_failure_intervals': failure_runs,
        'metric_protocol_v2': metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--pred-dir', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.gt_dir):
        raise FileNotFoundError(args.gt_dir)
    if not os.path.isdir(args.pred_dir):
        raise FileNotFoundError(args.pred_dir)

    rows = _records(args.gt_dir, args.pred_dir)
    expected = {('real', 'seq04'): 668, ('real', 'seq03'): 200, ('sim', 'seq09'): 572}
    actual = {}
    for row in rows:
        key = (row['domain'], row['seq_id'])
        actual[key] = actual.get(key, 0) + 1
    if actual != expected:
        raise ValueError('Unexpected TEST sequence counts: ' + str(actual))
    gt_names = {r['image'] for r in rows}
    extra = set(n for n in os.listdir(args.pred_dir) if n.endswith('.txt')) - gt_names
    if extra:
        raise ValueError('Predictions without matching GT: ' + str(sorted(extra)[:10]))
    grouped = {(d, s): [r for r in rows if r['domain'] == d and r['seq_id'] == s]
               for d, s in TARGETS}
    result = {
        'protocol': 'port_test_subset_audit_v1',
        'evidence_role': 'fixed_test_read_only_diagnosis_after_source_val_selection',
        'gt_dir': os.path.abspath(args.gt_dir),
        'pred_dir': os.path.abspath(args.pred_dir),
        'requested_subsets': [f'{d}_{s}' for d, s in TARGETS],
        'subsets': {
            f'{d}_{s}': _subset_summary(grouped[(d, s)])
            for d, s in TARGETS
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
