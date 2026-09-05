#!/usr/bin/env python3
"""Audit actual seq11-v2 causal-history dependencies before fold creation."""

import argparse
import hashlib
import json
import os
from pathlib import Path


PROTOCOL = 'seq11_v2_history_dependency_audit_v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-json', required=True)
    parser.add_argument('--frame-order-json', required=True)
    parser.add_argument('--metric-protocol-contract', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument(
        '--source-split',
        default='extra_source_real_seq11_pilot_k1p9_v2')
    parser.add_argument('--history-horizon', type=int, default=4)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    absolute = Path(path).resolve()
    if not absolute.is_file():
        raise RuntimeError('Missing dependency-audit input: '
                           + os.fspath(absolute))
    with open(absolute, 'r', encoding='utf-8') as handle:
        return absolute, json.load(handle)


def _write_exact(path, payload):
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, ensure_ascii=False) + '\n').encode(
        'utf-8')
    if output.exists() and output.read_bytes() != raw:
        raise RuntimeError('Refusing to overwrite different history audit: '
                           + os.fspath(output))
    if not output.exists():
        output.write_bytes(raw)
    return os.fspath(output)


def audit(args):
    if args.history_horizon != 4:
        raise RuntimeError('V2 contract fixes history_horizon=4')
    audit_path, source_audit = _read(args.audit_json)
    order_path, order = _read(args.frame_order_json)
    metric_path, metric = _read(args.metric_protocol_contract)
    if (source_audit.get('protocol') != 'source_owned_geometry_union_v2'
            or order.get('protocol') !=
            'seq11_v2_full251_frame_order_manifest_v2'
            or metric.get('protocol') !=
            'real_seq11_k1p9_251_event_block_cv_v2'):
        raise RuntimeError('Unexpected V2 dependency-audit protocol input')
    records = {}
    for row in source_audit.get('records') or []:
        stem = Path(row.get('filename', '')).stem
        if not stem or stem in records:
            raise RuntimeError('Invalid or duplicate DINO audit record')
        records[stem] = row
    ordered_stems = [row.get('frame_key') for row in order.get('frames') or []]
    if len(ordered_stems) != 251 or set(ordered_stems) != set(records):
        raise RuntimeError('Frame order and DINO audit sets differ')
    image_root = Path(args.data_root).resolve() / args.source_split / 'images'
    history_edges = []
    unavailable_attempts = []
    metric_pair_candidates = []
    for stem in ordered_stems:
        prefix, raw_frame = stem.rsplit('_', 1)
        frame = int(raw_frame)
        for age in range(1, args.history_horizon + 1):
            parent = '{}_{:06d}'.format(prefix, frame - age)
            record = records.get(parent)
            if record is None:
                unavailable_attempts.append(dict(
                    current=stem, history=parent, age=age,
                    reason='frame_absent_from_audit'))
                continue
            recorded = Path(record.get('filename', ''))
            sibling = image_root / recorded.name
            readable = sibling.is_file() or recorded.is_file()
            box = record.get('dino_native_box')
            usable = (readable and isinstance(box, list) and len(box) >= 5
                      and float(box[2]) > 0 and float(box[3]) > 0)
            if usable:
                history_edges.append(dict(
                    current=stem, history=parent, age=age))
            else:
                unavailable_attempts.append(dict(
                    current=stem, history=parent, age=age,
                    reason=('image_unreadable' if not readable else
                            'dino_proposal_missing_or_invalid')))
        previous = '{}_{:06d}'.format(prefix, frame - 1)
        if previous in records:
            metric_pair_candidates.append(dict(previous=previous, current=stem))
    checks = dict(
        exact_251_frames=len(ordered_stems) == 251,
        frame_sets_equal=set(ordered_stems) == set(records),
        history_horizon_4=args.history_horizon == 4,
        history_edges_use_exact_numeric_offsets=all(
            int(row['current'].rsplit('_', 1)[1])
            - int(row['history'].rsplit('_', 1)[1]) == row['age']
            for row in history_edges),
        metric_candidates_use_frame_difference_1=all(
            int(row['current'].rsplit('_', 1)[1])
            - int(row['previous'].rsplit('_', 1)[1]) == 1
            for row in metric_pair_candidates),
        sorted_sample_adjacency_not_used=True,
        fixed_test_not_read=True)
    passed = all(checks.values())
    return dict(
        protocol=PROTOCOL,
        evidence_boundary='source_only_pre_fold_dependency_audit',
        inputs=dict(
            audit_json=os.fspath(audit_path), audit_sha256=_sha256(audit_path),
            frame_order_json=os.fspath(order_path),
            frame_order_sha256=_sha256(order_path),
            metric_protocol_contract=os.fspath(metric_path),
            metric_protocol_contract_sha256=_sha256(metric_path)),
        source_split=args.source_split, history_horizon=args.history_horizon,
        history_dependency_edges=history_edges,
        unavailable_history_attempts=unavailable_attempts,
        metric_pair_candidates=metric_pair_candidates,
        note=('Metric candidates become metric_pair_edges only after the two '
              'frames are assigned to the same validation block and fold.'),
        counts=dict(
            history_dependency_edge_count=len(history_edges),
            unavailable_history_attempt_count=len(unavailable_attempts),
            metric_pair_candidate_count=len(metric_pair_candidates)),
        checks=checks, target_data_read=False, fixed_test_read=False,
        passed=passed,
        eligible_for_fold_construction=passed,
        eligible_for_fold_training=False,
        decision=('ALLOW_SEQ11_V2_DEPENDENCIES_FOR_FOLD_CONSTRUCTION'
                  if passed else 'STOP_SEQ11_V2_HISTORY_DEPENDENCY_FAILED'))


def main():
    args = parse_args()
    report = audit(args)
    output = _write_exact(args.out_json, report)
    print('[seq11-history] output={}'.format(output))
    print('[seq11-history] counts={}'.format(report['counts']))
    print('[seq11-history] decision={}'.format(report['decision']))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
