"""JSON audit for source-val Dual-Tower V2 component equivalence."""

import argparse
import glob
import json
import math
import os
import pickle

import numpy as np

from crane_project.tools.eval_crane_offline import (
    CraneOfflineEvaluator, parse_dota_txt, parse_seq_frame)


EXPECTED_FRAME_COUNT = 738


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size-results', required=True)
    parser.add_argument('--full-results', required=True)
    parser.add_argument('--dual-results')
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--atol', type=float, default=1e-4)
    return parser.parse_args()


def _load_results(path):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or len(payload) != EXPECTED_FRAME_COUNT:
        raise RuntimeError(
            'Result PKL must contain exactly {} frames'.format(
                EXPECTED_FRAME_COUNT))
    boxes = []
    for index, result in enumerate(payload):
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise RuntimeError(
                'Frame {} does not contain exactly one class'.format(index))
        detections = np.asarray(result[0], dtype=np.float64)
        if detections.size == 0:
            boxes.append(None)
            continue
        detections = detections.reshape((-1, 6))
        if detections.shape[0] != 1:
            raise RuntimeError(
                'Frame {} does not contain exactly one OBB'.format(index))
        box = detections[0, :5].copy()
        if (not np.isfinite(box).all()
                or np.any(box[2:4] <= 0.0)):
            raise RuntimeError('Frame {} contains an invalid OBB'.format(
                index))
        boxes.append(box)
    return absolute, boxes


def _annotations(ann_dir):
    absolute = os.path.abspath(os.fspath(ann_dir))
    paths = sorted(glob.glob(os.path.join(absolute, '*.txt')))
    if len(paths) != EXPECTED_FRAME_COUNT:
        raise RuntimeError(
            'Annotation directory must contain exactly {} frames'.format(
                EXPECTED_FRAME_COUNT))
    records = []
    for path in paths:
        boxes = parse_dota_txt(path)
        if len(boxes) != 1:
            raise RuntimeError(
                'Source-val frame must contain exactly one GT: ' + path)
        domain, sequence, frame = parse_seq_frame(path)
        records.append(dict(
            domain=domain, seq_id=sequence, frame_id=frame,
            gt_box=boxes[0], score=1.0, plc_rope=None))
    counts = dict(
        real=sum(item['domain'] == 'real' for item in records),
        sim=sum(item['domain'] == 'sim' for item in records))
    if counts != {'real': 226, 'sim': 512}:
        raise RuntimeError('Unexpected source-val domain counts')
    return absolute, records, counts


def _hybrid(size_boxes, full_boxes):
    boxes = []
    for size, full in zip(size_boxes, full_boxes):
        if size is None or full is None:
            boxes.append(None)
            continue
        boxes.append(np.asarray([
            full[0], full[1], size[2], size[3], full[4]],
            dtype=np.float64))
    return boxes


def _metrics(metadata, boxes):
    records = []
    for meta, box in zip(metadata, boxes):
        record = dict(meta)
        record['pred_box'] = box
        records.append(record)
    evaluator = CraneOfflineEvaluator(
        mode='test', center_thresh_px=15.0,
        sim_angle_center_thresh_px=10.0,
        ekf_window=10, mcml_limit=5, iou_thresh=0.5)
    return evaluator.evaluate_records(records)


def _metric_close(first, second, key, atol):
    return math.isclose(
        float(first[key]), float(second[key]),
        rel_tol=0.0, abs_tol=atol)


def _dual_equivalence(expected, observed, atol):
    center_errors = []
    size_errors = []
    angle_errors = []
    presence_equal = True
    for wanted, actual in zip(expected, observed):
        if (wanted is None) != (actual is None):
            presence_equal = False
            continue
        if wanted is None:
            continue
        center_errors.append(float(np.max(np.abs(actual[:2] - wanted[:2]))))
        size_errors.append(float(np.max(np.abs(actual[2:4] - wanted[2:4]))))
        delta = float(actual[4] - wanted[4])
        periodic = abs(0.5 * math.atan2(
            math.sin(2.0 * delta), math.cos(2.0 * delta)))
        angle_errors.append(periodic)
    maxima = dict(
        center_abs_px=max(center_errors, default=0.0),
        size_abs_px=max(size_errors, default=0.0),
        angle_abs_rad=max(angle_errors, default=0.0))
    checks = dict(
        presence_equal=presence_equal,
        center_equal=maxima['center_abs_px'] <= atol,
        size_equal=maxima['size_abs_px'] <= atol,
        angle_equal=maxima['angle_abs_rad'] <= atol)
    return dict(max_abs_error=maxima, checks=checks,
                passed=all(checks.values()))


def main():
    args = parse_args()
    size_path, size_boxes = _load_results(args.size_results)
    full_path, full_boxes = _load_results(args.full_results)
    ann_dir, metadata, domain_counts = _annotations(args.ann_dir)
    hybrid_boxes = _hybrid(size_boxes, full_boxes)
    size_metrics = _metrics(metadata, size_boxes)
    full_metrics = _metrics(metadata, full_boxes)
    hybrid_metrics = _metrics(metadata, hybrid_boxes)
    invariants = dict(
        real_dfr_exact_size=_metric_close(
            hybrid_metrics, size_metrics, 'real/DFR(%/frame)', args.atol),
        sim_dfr_exact_size=_metric_close(
            hybrid_metrics, size_metrics, 'sim/DFR(%/frame)', args.atol),
        real_center_exact_full=_metric_close(
            hybrid_metrics, full_metrics, 'real/R_center(%)', args.atol),
        sim_center_exact_full=_metric_close(
            hybrid_metrics, full_metrics, 'sim/R_center(%)', args.atol),
        real_aci_exact_full=_metric_close(
            hybrid_metrics, full_metrics, 'real/ACI', args.atol),
        sim_aci_exact_full=_metric_close(
            hybrid_metrics, full_metrics, 'sim/ACI', args.atol),
        sim_a_rmse_exact_full=_metric_close(
            hybrid_metrics, full_metrics, 'sim/A-RMSE(deg)', args.atol))
    dual = None
    if args.dual_results:
        dual_path, dual_boxes = _load_results(args.dual_results)
        equivalence = _dual_equivalence(
            hybrid_boxes, dual_boxes, args.atol)
        dual = dict(
            path=dual_path,
            metrics=_metrics(metadata, dual_boxes),
            expected_hybrid_equivalence=equivalence)
    passed = all(invariants.values()) and (
        dual is None or dual['expected_hybrid_equivalence']['passed'])
    report = dict(
        protocol='source_val_dual_tower_component_equivalence_v2',
        metric_protocol_version=2,
        evidence_boundary='source_val_only',
        target_data_read=False,
        fixed_test_read=False,
        input=dict(
            size_results=size_path, full_results=full_path,
            dual_results=None if dual is None else dual['path'],
            ann_dir=ann_dir, frame_count=EXPECTED_FRAME_COUNT,
            domain_counts=domain_counts),
        metrics=dict(
            size_epoch10=size_metrics,
            full_epoch11=full_metrics,
            expected_dual_tower=hybrid_metrics,
            dual_tower=None if dual is None else dual['metrics']),
        counterfactual_invariants=invariants,
        dual_equivalence=(
            None if dual is None else
            dual['expected_hybrid_equivalence']),
        passed=passed,
        eligible_for_fixed_test=False,
        decision=(
            'DUAL_TOWER_V2_SOURCE_VAL_EQUIVALENCE_CONFIRMED'
            if passed and dual is not None else
            'ALLOW_DUAL_TOWER_V2_PACKAGING_SOURCE_ONLY'
            if passed else 'STOP_DUAL_TOWER_V2_AUDIT_FAILED'))
    output = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
