#!/usr/bin/env python3
"""Source-selected causal box stabilization probe for frozen DINO outputs.

The probe never changes detection presence, ranking, scores, or box centers.
It smooths only width/height and the pi-periodic angle of the top-1 box.  The
EMA coefficient is selected on official source validation, then applied once
to target-dev/full-test outputs for diagnosis.  Target labels are never used
to choose the coefficient.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
from typing import Dict, List, Sequence, Tuple

import numpy as np

from crane_project.tools import (
    dino_teacher_baseline_first_rescue_audit as rescue,
    dino_teacher_box_stability_audit as stability,
    dino_teacher_rotated_labeller as labeller,
    dino_teacher_scoped_full_test as full_test,
)


AUDIT_NAME = 'Source-Selected Causal DINO Box Stabilization Probe V1'
DEFAULT_ALPHAS = (0.25, 0.5, 0.75, 1.0)
RIOU_THR = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument(
        '--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-val-datasets', nargs='+',
                        default=['val:val'])
    parser.add_argument('--test-split', default='test')
    parser.add_argument('--source-val-results', required=True)
    parser.add_argument('--baseline-results', required=True)
    parser.add_argument('--scoped-dino-results', required=True)
    parser.add_argument('--scope-manifest', required=True)
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=list(DEFAULT_ALPHAS))
    parser.add_argument('--max-source-riou-drop', type=float, default=0.005)
    parser.add_argument('--out-pkl', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_args(args):
    if args.test_split != 'test':
        raise ValueError('The stabilization probe is fixed to test')
    if not args.source_val_datasets:
        raise ValueError('At least one source-val dataset is required')
    if any(float(alpha) <= 0.0 or float(alpha) > 1.0
           for alpha in args.alphas):
        raise ValueError('All alphas must be in (0, 1]')
    if 1.0 not in [float(alpha) for alpha in args.alphas]:
        raise ValueError('Alpha candidates must include 1.0 as raw baseline')
    if args.max_source_riou_drop < 0.0:
        raise ValueError('--max-source-riou-drop must be non-negative')
    for path in (args.source_val_results, args.baseline_results,
                 args.scoped_dino_results, args.scope_manifest):
        if not os.path.isfile(path):
            raise ValueError('Required file does not exist: {}'.format(path))
    for path in (args.out_pkl, args.out_json):
        if os.path.exists(path):
            raise ValueError('Refusing to overwrite {}'.format(path))


def load_results(path: str) -> List[np.ndarray]:
    return stability.load_result_stream(path)


def source_val_records(args) -> List[Dict]:
    records = []
    for spec in args.source_val_datasets:
        annotation_split, image_split = labeller.parse_dataset_specs([spec])[0]
        records.extend(labeller.discover_labeled_records_with_image_split(
            args.data_root, annotation_split, image_split))
    records.sort(key=lambda row: (row['split'], row['seq'], int(row['frame'])))
    if not records:
        raise RuntimeError('No source-val records found')
    return records


def smooth_box(previous: np.ndarray, current: np.ndarray,
               alpha: float) -> np.ndarray:
    output = np.asarray(current, dtype=np.float32).copy()
    weight = float(alpha)
    if weight == 1.0:
        return output
    previous = np.asarray(previous, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    current = align_box_representation(previous, current)
    output[:2] = current[:2]
    previous_log_size = np.log(np.maximum(previous[2:4], 1e-6))
    current_log_size = np.log(np.maximum(current[2:4], 1e-6))
    output[2:4] = np.exp(
        (1.0 - weight) * previous_log_size + weight * current_log_size)
    previous_vector = np.asarray(
        [math.cos(2.0 * float(previous[4])),
         math.sin(2.0 * float(previous[4]))], dtype=np.float64)
    current_vector = np.asarray(
        [math.cos(2.0 * float(current[4])),
         math.sin(2.0 * float(current[4]))], dtype=np.float64)
    vector = (1.0 - weight) * previous_vector + weight * current_vector
    output[4] = 0.5 * math.atan2(float(vector[1]), float(vector[0]))
    return output


def pi_angle_delta(first: float, second: float) -> float:
    """Shortest OBB angle difference, where theta and theta+pi agree."""
    return 0.5 * math.atan2(
        math.sin(2.0 * (first - second)),
        math.cos(2.0 * (first - second)))


def align_box_representation(previous: np.ndarray,
                             current: np.ndarray) -> np.ndarray:
    """Choose the equivalent (w,h,theta) form closest to the previous box.

    Rotated rectangles have an edge-swap ambiguity:
    ``(w,h,theta) == (h,w,theta+pi/2)``.  Resolving it before temporal
    interpolation avoids manufacturing a square box at a representation flip.
    """
    previous = np.asarray(previous, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    direct = current.copy()
    swapped = current.copy()
    swapped[2:4] = current[[3, 2]]
    swapped[4] = current[4] + math.pi / 2.0

    def cost(candidate: np.ndarray) -> float:
        size_delta = np.log(np.maximum(candidate[2:4], 1e-6)) - np.log(
            np.maximum(previous[2:4], 1e-6))
        angle_delta = pi_angle_delta(float(candidate[4]),
                                     float(previous[4]))
        return float(np.dot(size_delta, size_delta) + angle_delta ** 2)

    return direct if cost(direct) <= cost(swapped) else swapped


def apply_causal_smoothing(records: Sequence[Dict],
                           detections: Sequence[np.ndarray], alpha: float,
                           scope_values: Dict = None) -> List[np.ndarray]:
    if len(records) != len(detections):
        raise ValueError('Record/result lengths differ')
    output = []
    previous_smoothed = None
    previous_seq = None
    for record, frame_detections in zip(records, detections):
        enabled = (True if scope_values is None else bool(scope_values[(
            record['split'], record['seq'], int(record['frame']))]))
        current = np.asarray(frame_detections, dtype=np.float32).copy()
        if not enabled:
            previous_smoothed = None
            previous_seq = record['seq']
            output.append(current)
            continue
        if current.shape[0] == 0:
            previous_smoothed = None
            previous_seq = record['seq']
            output.append(current)
            continue
        if previous_seq != record['seq']:
            previous_smoothed = None
        if previous_smoothed is not None:
            current[0, :5] = smooth_box(
                previous_smoothed, current[0, :5], alpha)
        previous_smoothed = current[0, :5].copy()
        previous_seq = record['seq']
        output.append(current)
    return output


def one_class_pickle(detections: Sequence[np.ndarray], path: str):
    payload = [[np.asarray(item, dtype=np.float32).reshape((-1, 6))]
               for item in detections]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def exact_frame_metrics(records: Sequence[Dict],
                        detections: Sequence[np.ndarray]) -> Dict:
    if len(records) != len(detections):
        raise ValueError('Record/result lengths differ')
    rows = []
    riou_values = []
    for record, frame_detections in zip(records, detections):
        gt = labeller.parse_original_gt(record['annotation'])
        metrics = labeller.ranked_detection_metrics(
            frame_detections[:1], gt, RIOU_THR, 0.05)
        rows.append(dict(
            seq=record['seq'], frame=int(record['frame']),
            hit=bool(metrics['top1_hit'])))
        riou_values.append(float(metrics['top1_riou']))
    return dict(
        frame_count=len(records),
        top1_hits=int(sum(row['hit'] for row in rows)),
        top1_mcml=labeller.longest_miss(rows, 'hit'),
        mean_top1_riou=(float(np.mean(riou_values))
                        if riou_values else 0.0))


def transition_summary(records: Sequence[Dict],
                       detections: Sequence[np.ndarray]) -> Dict:
    grouped = {}
    for record, frame_detections in zip(records, detections):
        grouped.setdefault(record['seq'], []).append(
            (record, frame_detections))
    rows = []
    for seq, items in sorted(grouped.items()):
        items.sort(key=lambda item: int(item[0]['frame']))
        transitions = zip(items[:-1], items[1:])
        for ((previous_record, previous_det),
             (current_record, current_det)) in transitions:
            if previous_det.shape[0] == 0 or current_det.shape[0] == 0:
                continue
            gap = int(current_record['frame']) - int(previous_record['frame'])
            rows.append(stability.transition_metrics(
                previous_det[0, :5], current_det[0, :5], gap))
    return stability.summarize_transitions(rows)


def select_alpha(
        records: Sequence[Dict], detections: Sequence[np.ndarray],
        alphas: Sequence[float],
        max_riou_drop: float) -> Tuple[float, Dict]:
    candidates = []
    raw_metrics = exact_frame_metrics(records, detections)
    raw_transitions = transition_summary(records, detections)
    if raw_transitions['dfr_percent']['count'] == 0:
        raise RuntimeError(
            'Source validation has no consecutive non-silence transitions')
    for alpha in sorted(set(float(value) for value in alphas)):
        smoothed = apply_causal_smoothing(records, detections, alpha)
        metrics = exact_frame_metrics(records, smoothed)
        transitions = transition_summary(records, smoothed)
        eligible = bool(
            metrics['top1_hits'] >= raw_metrics['top1_hits']
            and metrics['top1_mcml'] <= raw_metrics['top1_mcml']
            and metrics['mean_top1_riou'] >= (
                raw_metrics['mean_top1_riou'] - max_riou_drop)
            and transitions['aci']['mean'] >=
            raw_transitions['aci']['mean'])
        candidates.append(dict(
            alpha=alpha, eligible=eligible, frame_metrics=metrics,
            transition_metrics=transitions))
    eligible = [item for item in candidates if item['eligible']]
    selected = min(
        eligible or [item for item in candidates if item['alpha'] == 1.0],
        key=lambda item: (
            item['transition_metrics']['dfr_percent']['mean'],
            -item['transition_metrics']['aci']['mean'],
            -item['frame_metrics']['mean_top1_riou'], item['alpha']))
    return float(selected['alpha']), dict(
        raw_frame_metrics=raw_metrics,
        raw_transition_metrics=raw_transitions,
        candidates=candidates,
        selected=selected)


def target_scope_metrics(
        records: Sequence[Dict], detections: Sequence[np.ndarray],
        scope_values: Dict) -> Dict:
    selected_records = []
    selected_detections = []
    for record, frame_detections in zip(records, detections):
        key = (record['split'], record['seq'], int(record['frame']))
        if scope_values[key]:
            selected_records.append(record)
            selected_detections.append(frame_detections)
    return exact_frame_metrics(selected_records, selected_detections)


def main():
    args = parse_args()
    validate_args(args)
    source_records = source_val_records(args)
    source_detections = load_results(args.source_val_results)
    if len(source_records) != len(source_detections):
        raise RuntimeError('Source-val result ordering/count mismatch')

    test_records = full_test.all_test_records(args)
    scope = rescue.load_scope_manifest(args.scope_manifest, test_records)
    baseline = load_results(args.baseline_results)
    scoped = load_results(args.scoped_dino_results)
    if len(test_records) != len(baseline) or len(baseline) != len(scoped):
        raise RuntimeError('Test result ordering/count mismatch')

    alpha, source_selection = select_alpha(
        source_records, source_detections, args.alphas,
        args.max_source_riou_drop)
    stabilized = apply_causal_smoothing(
        test_records, scoped, alpha, scope['values'])
    for record, baseline_det, scoped_det in zip(
            test_records, baseline, scoped):
        enabled = scope['values'][(
            record['split'], record['seq'], int(record['frame']))]
        if not enabled and not np.array_equal(baseline_det, scoped_det):
            raise RuntimeError(
                'Existing scoped result changed outside scope at {} {}'.format(
                    record['seq'], record['frame']))
    if any(not np.array_equal(raw, new)
           for record, raw, new in zip(test_records, scoped, stabilized)
           if not scope['values'][(
               record['split'], record['seq'], int(record['frame']))]):
        raise RuntimeError('Stabilizer changed a scope-disabled frame')
    one_class_pickle(stabilized, args.out_pkl)

    raw_scope = target_scope_metrics(test_records, scoped, scope['values'])
    stabilized_scope = target_scope_metrics(
        test_records, stabilized, scope['values'])
    raw_stability = stability.build_report(
        test_records, baseline, scoped, scope, top_transitions=10)
    stabilized_stability = stability.build_report(
        test_records, baseline, stabilized, scope, top_transitions=10)
    payload = dict(
        audit=AUDIT_NAME,
        protocol=dict(
            read_only=True, model_loaded=False, optimizer_steps=0,
            checkpoint_writes=0, target_used_for_alpha_selection=False,
            smoothing='causal_ema_log_wh_sin2_cos2_angle',
            centers_scores_and_detection_presence_unchanged=True,
            source_selection_datasets=list(args.source_val_datasets),
            target_scope_is_diagnosis_only=True),
        inputs=dict(
            source_val_results=os.path.abspath(args.source_val_results),
            source_val_results_sha256=file_sha256(args.source_val_results),
            baseline_results=os.path.abspath(args.baseline_results),
            scoped_dino_results=os.path.abspath(args.scoped_dino_results),
            scope_manifest=os.path.abspath(args.scope_manifest)),
        source_selection=source_selection,
        stability_before=raw_stability['by_sequence'],
        stability_after=stabilized_stability['by_sequence'],
        target_dev_scope=dict(raw=raw_scope, stabilized=stabilized_scope),
        output_stabilized_results=os.path.abspath(args.out_pkl),
        decision='SOURCE_SELECTED_CAUSAL_STABILIZER_TARGET_DIAGNOSIS')
    temporary = args.out_json + '.tmp'
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False,
                  allow_nan=False)
    os.replace(temporary, args.out_json)
    print('[source-selection] alpha={} eligible_candidates={}'.format(
        alpha, sum(item['eligible']
                   for item in source_selection['candidates'])))
    print('[target-dev] raw_top1={}/{} raw_mcml={} stabilized_top1={}/{} '
          'stabilized_mcml={}'.format(
              raw_scope['top1_hits'], raw_scope['frame_count'],
              raw_scope['top1_mcml'], stabilized_scope['top1_hits'],
              stabilized_scope['frame_count'], stabilized_scope['top1_mcml']))
    raw_seq02 = raw_stability['by_sequence']['real_seq02']['scoped_dino']
    new_seq02 = stabilized_stability['by_sequence'][
        'real_seq02']['scoped_dino']
    print('[stability] real_seq02 DFR={}->{} ACI={}->{}'.format(
        raw_seq02['dfr_percent']['mean'],
        new_seq02['dfr_percent']['mean'],
        raw_seq02['aci']['mean'], new_seq02['aci']['mean']))
    print('[out] pkl={} json={}'.format(args.out_pkl, args.out_json))


if __name__ == '__main__':
    main()
