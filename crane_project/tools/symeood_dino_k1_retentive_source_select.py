#!/usr/bin/env python3
"""Select one V3 checkpoint from source-val gate reports only."""

import argparse
import hashlib
import json
import os
import re


SOURCE_GATE_PROTOCOL = (
    'k1_retentive_causal_phase_refiner_source_gate_v3_geometry_v1')
SELECTION_PROTOCOL = (
    'k1_retentive_causal_phase_refiner_source_selection_v3')
SELECTION_POLICY = 'passing_only_min_mcml_max_riou_min_dfr_earliest_v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-gates', nargs='+', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _epoch(path):
    match = re.search(r'(?:^|[/\\])epoch_(\d+)\.pth$', os.fspath(path))
    if match is None:
        raise RuntimeError('Gate checkpoint path has no epoch_<N>.pth')
    return int(match.group(1))


def _read_gate(path):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        raw = handle.read()
    gate = json.loads(raw.decode('utf-8'))
    required = dict(
        protocol=SOURCE_GATE_PROTOCOL,
        evidence_boundary='source_val_only',
        target_data_read=False,
        fixed_test_read=False,
        eligible_for_fixed_test=False,
        eligible_for_unknown_sequence_claim=False)
    failures = [
        '{}={!r} expected {!r}'.format(key, gate.get(key), expected)
        for key, expected in required.items()
        if gate.get(key) != expected]
    inputs = dict(gate.get('input') or {})
    checkpoint = inputs.get('candidate_checkpoint', '')
    results = inputs.get('candidate_results', '')
    if not inputs.get('candidate_checkpoint_sha256'):
        failures.append('candidate checkpoint SHA256 is absent')
    if not inputs.get('candidate_results_sha256'):
        failures.append('candidate results SHA256 is absent')
    depth_gate = dict(gate.get('depth_interface_geometry_gate') or {})
    if gate.get('passed') is True and depth_gate.get('passed') is not True:
        failures.append('passing gate lacks a passing depth-interface gate')
    if failures:
        raise RuntimeError(
            'Invalid V3 source gate {}: {}'.format(
                absolute, '; '.join(failures)))
    metrics = dict(gate.get('candidate_metrics') or {})
    required_metrics = (
        'real/MCML_max(frames)', 'sim/MCML_max(frames)',
        'real/mean_RIoU', 'sim/mean_RIoU',
        'real/DFR(%/frame)', 'sim/DFR(%/frame)')
    missing = [key for key in required_metrics if key not in metrics]
    if missing:
        raise RuntimeError('Source gate lacks selection metrics: '
                           + ', '.join(missing))
    return dict(
        path=absolute, sha256=_sha256(raw), epoch=_epoch(checkpoint),
        checkpoint=checkpoint, results=results,
        checkpoint_sha256=inputs['candidate_checkpoint_sha256'],
        results_sha256=inputs['candidate_results_sha256'],
        passed=gate.get('passed') is True,
        decision=gate.get('decision'), metrics=metrics)


def _rank(row):
    metrics = row['metrics']
    return (
        max(int(metrics['real/MCML_max(frames)']),
            int(metrics['sim/MCML_max(frames)'])),
        -(float(metrics['real/mean_RIoU'])
          + float(metrics['sim/mean_RIoU'])),
        float(metrics['real/DFR(%/frame)'])
        + float(metrics['sim/DFR(%/frame)']),
        int(row['epoch']))


def select(source_gates):
    if len(source_gates) != 10:
        raise RuntimeError('V3 selection requires exactly ten source gates')
    rows = [_read_gate(path) for path in source_gates]
    if sorted(row['epoch'] for row in rows) != list(range(1, 11)):
        raise RuntimeError('Source gates must cover epochs 1..10 exactly')
    eligible = [row for row in rows if row['passed']]
    if not eligible:
        raise RuntimeError('No V3 epoch passed the complete source gate')
    selected = min(eligible, key=_rank)
    return dict(
        protocol=SELECTION_PROTOCOL,
        evidence_boundary='source_val_only',
        target_data_read=False,
        fixed_test_read=False,
        selection_policy=SELECTION_POLICY,
        evaluated_epochs=list(range(1, 11)),
        passing_epochs=sorted(row['epoch'] for row in eligible),
        ranking=[
            dict(epoch=row['epoch'], rank=list(_rank(row)),
                 gate=row['path'], gate_sha256=row['sha256'])
            for row in sorted(eligible, key=_rank)],
        selected=dict(
            epoch=selected['epoch'],
            checkpoint=selected['checkpoint'],
            checkpoint_sha256=selected['checkpoint_sha256'],
            results=selected['results'],
            results_sha256=selected['results_sha256'],
            source_gate=selected['path'],
            source_gate_sha256=selected['sha256']),
        passed=True,
        eligible_for_checkpoint_promotion=True,
        eligible_for_fixed_test=False,
        eligible_for_unknown_sequence_claim=False,
        decision='SELECT_V3_SOURCE_CHECKPOINT_FOR_PROMOTION_ONLY')


def main():
    args = parse_args()
    report = select(args.source_gates)
    output = os.path.abspath(os.fspath(args.out_json))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
