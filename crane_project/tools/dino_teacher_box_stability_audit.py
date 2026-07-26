#!/usr/bin/env python3
"""Read-only temporal stability audit for scoped frozen-DINO detections.

The audit consumes exported result pickles.  It does not load a model, read
annotations, select a checkpoint, or modify detections.  DFR is decomposed into
the original output-stream statistic, paired transitions available to both
methods, and transitions newly observable after DINO fills baseline silence.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import warnings
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from crane_project.tools import (
    dino_teacher_baseline_first_rescue_audit as rescue,
    dino_teacher_scoped_full_test as full_test,
)


AUDIT_NAME = 'Frozen DINO Box Stability Read-Only Audit V1'


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument(
        '--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--test-split', default='test')
    parser.add_argument('--baseline-results', required=True)
    parser.add_argument('--scoped-dino-results', required=True)
    parser.add_argument('--scope-manifest', required=True)
    parser.add_argument('--top-transitions', type=int, default=10)
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
        raise ValueError('The stability audit is fixed to the test split')
    if int(args.top_transitions) <= 0:
        raise ValueError('--top-transitions must be positive')
    for path in (args.baseline_results, args.scoped_dino_results,
                 args.scope_manifest):
        if not os.path.isfile(path):
            raise ValueError('Required file does not exist: {}'.format(path))
    if os.path.exists(args.out_json):
        raise ValueError('Refusing to overwrite {}'.format(args.out_json))


def load_result_stream(path: str) -> List[np.ndarray]:
    with open(path, 'rb') as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='numpy.core.numeric is deprecated',
                category=DeprecationWarning)
            payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)):
        raise ValueError('Result pickle must contain a sequence')
    stream = []
    for index, result in enumerate(payload):
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise ValueError(
                'Expected one-class result at index {}'.format(index))
        detections = np.asarray(result[0], dtype=np.float32)
        if detections.size == 0:
            detections = np.zeros((0, 6), dtype=np.float32)
        detections = detections.reshape((-1, 6))
        if not bool(np.isfinite(detections).all()):
            raise ValueError(
                'Non-finite detection at index {}'.format(index))
        stream.append(detections)
    return stream


def top1(detections: np.ndarray):
    return None if detections.shape[0] == 0 else detections[0, :5]


def angle_delta_deg(current: np.ndarray, previous: np.ndarray) -> float:
    delta = float(current[4]) - float(previous[4])
    delta = math.atan2(math.sin(delta), math.cos(delta))
    if delta > math.pi / 2:
        delta -= math.pi
    if delta < -math.pi / 2:
        delta += math.pi
    return abs(math.degrees(delta))


def transition_metrics(previous: np.ndarray, current: np.ndarray,
                       frame_gap: int) -> Dict:
    previous_diag = float(math.hypot(previous[2], previous[3]))
    current_diag = float(math.hypot(current[2], current[3]))
    previous_area = max(float(previous[2] * previous[3]), 1e-12)
    current_area = max(float(current[2] * current[3]), 1e-12)
    gap = max(int(frame_gap), 1)
    total_angle_change = float(angle_delta_deg(current, previous))
    angle_change = float(total_angle_change / gap)
    return dict(
        dfr=float(abs(current_diag - previous_diag) /
                  max(previous_diag * gap, 1e-12)),
        diagonal_previous=previous_diag,
        diagonal_current=current_diag,
        log_area_change=float(abs(math.log(current_area / previous_area)) /
                              gap),
        angle_change_deg=angle_change,
        # Match CraneOfflineEvaluator: ACI uses the total angle difference
        # between valid outputs, whereas DFR is normalized by frame gap.
        aci=float(np.clip(
            1.0 - total_angle_change / 35.0, 0.0, 1.0)),
        center_step_px=float(np.linalg.norm(
            current[:2] - previous[:2]) / gap))


def numeric_summary(values: Sequence[float], percent: bool = False) -> Dict:
    array = np.asarray(list(values), dtype=np.float64)
    scale = 100.0 if percent else 1.0
    if array.size == 0:
        return dict(count=0, mean=None, median=None, p90=None, maximum=None)
    return dict(
        count=int(array.size),
        mean=float(array.mean() * scale),
        median=float(np.median(array) * scale),
        p90=float(np.percentile(array, 90) * scale),
        maximum=float(array.max() * scale))


def summarize_transitions(rows: Sequence[Dict]) -> Dict:
    return dict(
        count=len(rows),
        dfr_percent=numeric_summary(
            [row['dfr'] for row in rows], percent=True),
        log_area_change=numeric_summary(
            [row['log_area_change'] for row in rows]),
        angle_change_deg=numeric_summary(
            [row['angle_change_deg'] for row in rows]),
        aci=numeric_summary([row['aci'] for row in rows]),
        center_step_px=numeric_summary(
            [row['center_step_px'] for row in rows]))


def collect_transition_rows(records: Sequence[Dict],
                            baseline: Sequence[np.ndarray],
                            scoped: Sequence[np.ndarray],
                            scope_values: Dict) -> Dict:
    grouped = defaultdict(list)
    for index, (record, baseline_det, scoped_det) in enumerate(zip(
            records, baseline, scoped)):
        grouped[record['seq']].append(dict(
            index=index, record=record,
            baseline=top1(baseline_det), scoped=top1(scoped_det),
            scope_enabled=bool(scope_values[(
                record['split'], record['seq'], int(record['frame']))])))

    output = {}
    for seq, frames in sorted(grouped.items()):
        frames.sort(key=lambda row: int(row['record']['frame']))
        baseline_rows = []
        scoped_rows = []
        paired_baseline_rows = []
        paired_scoped_rows = []
        newly_observed_rows = []
        scope_internal_rows = []
        for previous_row, current_row in zip(frames[:-1], frames[1:]):
            previous_frame = int(previous_row['record']['frame'])
            current_frame = int(current_row['record']['frame'])
            gap = current_frame - previous_frame
            if gap <= 0:
                raise RuntimeError(
                    'Non-increasing frame ids in {}'.format(seq))
            baseline_pair = (previous_row['baseline'] is not None
                             and current_row['baseline'] is not None)
            scoped_pair = (previous_row['scoped'] is not None
                           and current_row['scoped'] is not None)

            def make_row(method: str, metrics: Dict) -> Dict:
                return dict(
                    seq=seq, previous_frame=previous_frame,
                    current_frame=current_frame, frame_gap=gap,
                    method=method, **metrics)

            baseline_transition = None
            scoped_transition = None
            if baseline_pair:
                baseline_transition = make_row(
                    'baseline', transition_metrics(
                        previous_row['baseline'], current_row['baseline'],
                        gap))
                baseline_rows.append(baseline_transition)
            if scoped_pair:
                scoped_transition = make_row(
                    'scoped_dino', transition_metrics(
                        previous_row['scoped'], current_row['scoped'], gap))
                scoped_rows.append(scoped_transition)
            if baseline_pair and scoped_pair:
                paired_baseline_rows.append(baseline_transition)
                paired_scoped_rows.append(scoped_transition)
            elif scoped_pair:
                newly_observed_rows.append(scoped_transition)
            if (scoped_pair and previous_row['scope_enabled']
                    and current_row['scope_enabled']):
                scope_internal_rows.append(scoped_transition)

        output[seq] = dict(
            baseline=baseline_rows,
            scoped_dino=scoped_rows,
            paired_baseline=paired_baseline_rows,
            paired_scoped_dino=paired_scoped_rows,
            newly_observed_scoped_dino=newly_observed_rows,
            scope_internal_scoped_dino=scope_internal_rows)
    return output


def changed_frame_audit(records: Sequence[Dict],
                        baseline: Sequence[np.ndarray],
                        scoped: Sequence[np.ndarray],
                        scope_values: Dict) -> Dict:
    changed = []
    changed_outside = []
    enabled = []
    unchanged_enabled = []
    for record, baseline_det, scoped_det in zip(records, baseline, scoped):
        key = (record['split'], record['seq'], int(record['frame']))
        is_enabled = bool(scope_values[key])
        is_changed = not np.array_equal(baseline_det, scoped_det)
        item = dict(seq=record['seq'], frame=int(record['frame']))
        if is_enabled:
            enabled.append(item)
        if is_changed:
            changed.append(item)
            if not is_enabled:
                changed_outside.append(item)
        elif is_enabled:
            unchanged_enabled.append(item)
    return dict(
        enabled_scope_count=len(enabled), changed_frame_count=len(changed),
        changed_outside_scope_count=len(changed_outside),
        unchanged_enabled_count=len(unchanged_enabled),
        changed_frames=changed,
        changed_outside_scope=changed_outside,
        unchanged_enabled_frames=unchanged_enabled)


def build_report(records: Sequence[Dict], baseline: Sequence[np.ndarray],
                 scoped: Sequence[np.ndarray], scope: Dict,
                 top_transitions: int) -> Dict:
    if not (len(records) == len(baseline) == len(scoped)):
        raise RuntimeError('Record/result lengths differ')
    changes = changed_frame_audit(
        records, baseline, scoped, scope['values'])
    if changes['changed_outside_scope_count']:
        raise RuntimeError('Scoped results changed outside the declared scope')
    transitions = collect_transition_rows(
        records, baseline, scoped, scope['values'])
    by_sequence = {}
    all_scope_internal = []
    for seq, rows in sorted(transitions.items()):
        all_scope_internal.extend(rows['scope_internal_scoped_dino'])
        by_sequence[seq] = dict(
            baseline=summarize_transitions(rows['baseline']),
            scoped_dino=summarize_transitions(rows['scoped_dino']),
            common_output_baseline=summarize_transitions(
                rows['paired_baseline']),
            common_output_scoped_dino=summarize_transitions(
                rows['paired_scoped_dino']),
            newly_observed_scoped_dino=summarize_transitions(
                rows['newly_observed_scoped_dino']),
            scope_internal_scoped_dino=summarize_transitions(
                rows['scope_internal_scoped_dino']))
    unstable = sorted(
        all_scope_internal, key=lambda row: float(row['dfr']), reverse=True)
    return dict(
        changed_frames=changes,
        by_sequence=by_sequence,
        scope_internal_top_unstable_transitions=unstable[:top_transitions],
        interpretation=dict(
            coverage_accounting_present=any(
                report['newly_observed_scoped_dino']['count'] > 0
                for report in by_sequence.values()),
            scope_internal_jitter_observed=bool(all_scope_internal)))


def write_json_atomic(path: str, payload: Dict):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False,
                  allow_nan=False)
    os.replace(temporary, path)


def main():
    args = parse_args()
    validate_args(args)
    records = full_test.all_test_records(args)
    scope = rescue.load_scope_manifest(args.scope_manifest, records)
    baseline = load_result_stream(args.baseline_results)
    scoped = load_result_stream(args.scoped_dino_results)
    report = build_report(
        records, baseline, scoped, scope, args.top_transitions)
    payload = dict(
        audit=AUDIT_NAME,
        protocol=dict(
            read_only=True, model_loaded=False,
            annotation_contents_read=False,
            optimizer_steps=0, checkpoint_writes=0,
            dfr_definition=(
                'abs(diagonal_t-diagonal_prev)/(diagonal_prev*frame_gap)'),
            common_output_definition=(
                'same consecutive transition has output in both methods'),
            newly_observed_definition=(
                'scoped DINO has consecutive outputs but baseline does not')),
        inputs=dict(
            baseline_results=os.path.abspath(args.baseline_results),
            baseline_sha256=file_sha256(args.baseline_results),
            scoped_dino_results=os.path.abspath(args.scoped_dino_results),
            scoped_dino_sha256=file_sha256(args.scoped_dino_results),
            scope_manifest=os.path.abspath(args.scope_manifest),
            scope_manifest_sha256=file_sha256(args.scope_manifest)),
        result=report,
        decision='READ_ONLY_DINO_BOX_STABILITY_AUDIT_COMPLETE')
    write_json_atomic(args.out_json, payload)
    for seq, summary in sorted(report['by_sequence'].items()):
        base = summary['baseline']['dfr_percent']
        dino = summary['scoped_dino']['dfr_percent']
        paired_base = summary['common_output_baseline']['dfr_percent']
        paired_dino = summary['common_output_scoped_dino']['dfr_percent']
        print('[stability] seq={} stream={}->{} common={}->{} '
              'new_transitions={}'.format(
                  seq, base['mean'], dino['mean'], paired_base['mean'],
                  paired_dino['mean'],
                  summary['newly_observed_scoped_dino']['count']))
    changes = report['changed_frames']
    print('[isolation] enabled={} changed={} outside_scope={}'.format(
        changes['enabled_scope_count'], changes['changed_frame_count'],
        changes['changed_outside_scope_count']))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
