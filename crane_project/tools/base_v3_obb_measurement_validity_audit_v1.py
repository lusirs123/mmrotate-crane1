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


PROTOCOL = 'base_v3_obb_measurement_validity_audit_v1'


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
    for sequence, ranges in intervals.items():
        previous_end = None
        if not isinstance(sequence, str) or not isinstance(ranges, list):
            raise RuntimeError('Intervals must map sequence names to lists')
        for pair in sorted(ranges):
            if (not isinstance(pair, list) or len(pair) != 2
                    or int(pair[0]) > int(pair[1])):
                raise RuntimeError('Invalid interval for ' + sequence)
            start, end = int(pair[0]), int(pair[1])
            if previous_end is not None and start <= previous_end:
                raise RuntimeError('Overlapping intervals for ' + sequence)
            previous_end = end


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


def build_audit(report, intervals):
    if report.get('fixed_test_read') is not True:
        raise RuntimeError('Input must be a fixed-test finalization report')
    _validate_intervals(intervals)
    records = report.get('records')
    if not isinstance(records, list) or not records:
        raise RuntimeError('Input report must contain non-empty records')
    methods = sorted({m for row in records for m in
                      row.get('offline_errors', {})})
    if not methods:
        raise RuntimeError('Input report contains no offline method records')
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
    return {
        'protocol': PROTOCOL,
        'input_protocol': report.get('protocol'),
        'fixed_test_read': bool(report.get('fixed_test_read', False)),
        'prediction_or_threshold_selection_performed': False,
        'intervals_are_prediction_independent': True,
        'metrics_scope': [
            'center_valid_count', 'center_valid_rate', 'riou_count',
            'mean_riou'],
        'full_temporal_metrics_remain_in_source_report': [
            'DFR', 'ACI', 'MCML'],
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
    args = parser.parse_args()
    identity = _identity(args.report)
    report = json.loads(Path(args.report).read_text(encoding='utf-8'))
    intervals = json.loads(Path(args.intervals_json).read_text(encoding='utf-8'))
    if not isinstance(intervals, dict):
        raise RuntimeError('Intervals must be a JSON object')
    audit = build_audit(report, intervals)
    audit['input_identity'] = identity
    audit['intervals'] = intervals
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
