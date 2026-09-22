#!/usr/bin/env python3
"""Report raw and measurement-valid metrics for a fixed online report.

The excluded intervals are supplied explicitly and are never inferred from
predictions.  The input report is not modified and this audit cannot select a
checkpoint or tune a threshold.
"""

import argparse
import hashlib
import json
from pathlib import Path


PROTOCOL_V1 = 'base_v3_obb_measurement_validity_audit_v1'
PROTOCOL_V2 = 'base_v3_obb_measurement_validity_audit_v2'


def _identity(path):
    p = Path(path).resolve()
    if not p.is_file():
        raise RuntimeError('Missing input: ' + str(p))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    return {'path': str(p), 'sha256': h, 'size_bytes': p.stat().st_size}


def _in_intervals(sequence, frame, intervals):
    for start, end in intervals.get(sequence, []):
        if int(start) <= int(frame) <= int(end):
            return True
    return False


def _validate_intervals(intervals):
    if not isinstance(intervals, dict):
        raise RuntimeError('Intervals must be a JSON object')
    normalized = {}
    for sequence, ranges in intervals.items():
        if not isinstance(sequence, str) or not isinstance(ranges, list):
            raise RuntimeError('Intervals must map sequence names to lists')
        validated = []
        for index, pair in enumerate(ranges):
            if not isinstance(pair, list) or len(pair) != 2:
                raise RuntimeError(
                    'Invalid interval structure: sequence={} index={} value={!r}'
                    .format(sequence, index, pair))
            try:
                start, end = int(pair[0]), int(pair[1])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    'Invalid interval endpoint: sequence={} index={} value={!r}'
                    .format(sequence, index, pair)) from exc
            if start > end:
                raise RuntimeError(
                    'Invalid interval order: sequence={} index={} start={} end={}'
                    .format(sequence, index, start, end))
            validated.append([start, end])
        validated.sort(key=lambda pair: (pair[0], pair[1]))
        previous_end = None
        for start, end in validated:
            if previous_end is not None and start <= previous_end:
                raise RuntimeError(
                    'Overlapping intervals: sequence={} previous_end={} start={}'
                    .format(sequence, previous_end, start))
            previous_end = end
        normalized[sequence] = validated
    return normalized


def _frame_number(frame_key):
    return int(str(frame_key).rsplit('_', 1)[1])


def _metric(rows, method):
    errors = [r.get('offline_errors', {}).get(method, {}) for r in rows]
    riou = [r.get('offline_riou', {}).get(method) for r in rows]
    present = [e for e in errors if e.get('center') is not None]
    riou_values = [float(v) for v in riou if v is not None]
    return {
        'frame_count': len(rows),
        'center_valid_count': len(present),
        'center_valid_rate': len(present) / len(rows) if rows else 0.0,
        'riou_count': len(riou_values),
        'mean_riou': (sum(riou_values) / len(riou_values)
                      if riou_values else None),
    }


def _state_metric(rows, online_by_key, method, component, threshold):
    states = []
    errors = []
    for row in rows:
        frame_key = row['frame_key']
        online_row = online_by_key.get(frame_key)
        observations = (online_row.get('observations')
                        if isinstance(online_row, dict) else None)
        method_observations = (observations.get(method)
                               if isinstance(observations, dict) else None)
        observation = (method_observations.get(component)
                       if isinstance(method_observations, dict) else None)
        state = (observation.get('state')
                 if isinstance(observation, dict) else None)
        if state not in {'measurement', 'prediction', 'unavailable'}:
            raise RuntimeError(
                'Invalid online observation: frame={} method={} component={} '
                'state={!r}'.format(frame_key, method, component, state))
        states.append(state)
        value = row.get('offline_errors', {}).get(method, {}).get(component)
        if value is not None:
            errors.append(float(value))
    longest = current = 0
    for state in states:
        current = current + 1 if state == 'unavailable' else 0
        longest = max(longest, current)
    bad = sum(value > threshold for value in errors)
    return {
        'frame_count': len(rows),
        'measurement_count': states.count('measurement'),
        'prediction_count': states.count('prediction'),
        'unavailable_count': states.count('unavailable'),
        'output_coverage': ((len(rows) - states.count('unavailable')) / len(rows)
                            if rows else 0.0),
        'longest_unavailable_run': longest,
        'bad_available_output_rate': bad / len(errors) if errors else None,
        'correct_output_coverage': ((len(errors) - bad) / len(rows)
                                    if rows else 0.0),
    }


def build_audit(report, intervals, online_report=None):
    if report.get('fixed_test_read') is not True:
        raise RuntimeError('Input must be a fixed-test finalization report')
    intervals = _validate_intervals(intervals)
    records = report.get('records')
    if not isinstance(records, list) or not records:
        raise RuntimeError('Input report must contain non-empty records')
    methods = sorted({m for row in records for m in
                      row.get('offline_errors', {})})
    if not methods:
        raise RuntimeError('Input report contains no offline method records')
    online_by_key = None
    if online_report is not None:
        online_records = online_report.get('records') or []
        if not isinstance(online_records, list):
            raise RuntimeError('Online report records must be a list')
        online_by_key = {row.get('frame_key'): row for row in online_records}
        if len(online_by_key) != len(online_records):
            raise RuntimeError('Online report contains duplicate frame keys')
        if set(online_by_key) != {row.get('frame_key') for row in records}:
            raise RuntimeError('Online and finalization frame keys differ')
    excluded = []
    valid = []
    for row in records:
        key = str(row.get('frame_key', ''))
        if '_' not in key:
            raise RuntimeError('Invalid frame_key: ' + key)
        sequence = key.rsplit('_', 1)[0]
        frame = _frame_number(key)
        item = dict(frame_key=key, sequence=sequence, frame=frame,
                    exclusion_reason=(
                        'material_contact_grabbing_or_not_fully_detached'
                        if _in_intervals(sequence, frame, intervals)
                        else None))
        (excluded if item['exclusion_reason'] else valid).append(item)
    by_method = {}
    thresholds = report.get('evaluation_thresholds') or {}
    component_thresholds = {
        'center': float(thresholds.get('center_error_threshold_px', 5.0)),
        'scale': float(thresholds.get('scale_relative_error_threshold', 0.1)),
        'angle': float(thresholds.get('angle_error_threshold_deg', 3.0)),
    }
    for method in methods:
        all_rows = records
        valid_keys = {r['frame_key'] for r in valid}
        valid_rows = [r for r in records if r['frame_key'] in valid_keys]
        excluded_keys = {r['frame_key'] for r in excluded}
        excluded_rows = [r for r in records if r['frame_key'] in excluded_keys]
        by_method[method] = {
            'raw': _metric(all_rows, method),
            'measurement_valid': _metric(valid_rows, method),
            'excluded_interval': _metric(excluded_rows, method),
        }
        if online_by_key is not None:
            by_method[method]['component_raw'] = {
                component: _state_metric(
                    all_rows, online_by_key, method, component,
                    component_thresholds[component])
                for component in component_thresholds}
            by_method[method]['component_measurement_valid'] = {
                component: _state_metric(
                    valid_rows, online_by_key, method, component,
                    component_thresholds[component])
                for component in component_thresholds}
    return {
        'protocol': PROTOCOL_V2 if online_by_key is not None else PROTOCOL_V1,
        'input_protocol': report.get('protocol'),
        'fixed_test_read': bool(report.get('fixed_test_read', False)),
        'prediction_or_threshold_selection_performed': False,
        'intervals_are_prediction_independent': True,
        'metrics_scope': [
            'center_valid_count', 'center_valid_rate', 'riou_count',
            'mean_riou'],
        'full_temporal_metrics_remain_in_source_report': [
            'DFR', 'ACI', 'MCML'],
        'online_component_states_included': online_by_key is not None,
        'intervals': intervals,
        'excluded_interval_count': len(excluded),
        'excluded_frame_keys': [r['frame_key'] for r in excluded],
        'methods': by_method,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--intervals-json', required=True,
                        help='JSON object: sequence -> [[first_frame,last_frame]]')
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--online-report')
    args = parser.parse_args()
    identity = _identity(args.report)
    report = json.loads(Path(args.report).read_text(encoding='utf-8'))
    online_report = None
    online_identity = None
    if args.online_report:
        online_identity = _identity(args.online_report)
        online_report = json.loads(
            Path(online_identity['path']).read_text(encoding='utf-8'))
    intervals_identity = _identity(args.intervals_json)
    intervals = json.loads(
        Path(intervals_identity['path']).read_text(encoding='utf-8'))
    audit = build_audit(report, intervals, online_report=online_report)
    audit['input_identity'] = identity
    audit['intervals_input_identity'] = intervals_identity
    if online_identity is not None:
        audit['online_input_identity'] = online_identity
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n',
                   encoding='utf-8')
    print('MEASUREMENT_VALIDITY_AUDIT_OK')
    print(json.dumps(audit['methods'], ensure_ascii=False))
    print('excluded_interval_count:', audit['excluded_interval_count'])
    print('out:', out)


if __name__ == '__main__':
    main()
