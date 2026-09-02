#!/usr/bin/env python3
"""Validate an auxiliary labelled source split without detector inference.

This audit is deliberately narrower than a source gate.  It proves that the
unique image/annotation pairs and the cached all-lane DINO proposals are safe
to consume during source-only refiner training.  It does not identify the
cached SymEOOD lane as the formal K1 baseline, select a checkpoint, tune a
threshold, or read fixed TEST.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from crane_project.tools.eval_crane_offline import compute_riou, parse_dota_txt


PROTOCOL = 'auxiliary_source_training_contract_audit_v1'
ALL_LANE_PROTOCOL = 'source_owned_geometry_union_v2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-json', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument('--source-split', required=True)
    parser.add_argument('--expected-frame-count', type=int, default=59)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _visible_files(folder, suffixes):
    output = []
    sidecars = []
    for path in Path(folder).iterdir():
        if path.name.startswith('._'):
            if path.suffix.lower() in suffixes:
                sidecars.append(path)
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            output.append(path)
    return sorted(output), sorted(sidecars)


def _dino_computed(record):
    if record.get('raw_selected_source') == 'not_computed':
        return False
    if 'dino_invoked' not in record:
        return 'dino_native_box' in record
    marker = record['dino_invoked']
    return marker is True or (
        isinstance(marker, int) and not isinstance(marker, bool)
        and marker == 1)


def _box(record, key):
    value = record.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if (array.size < 5 or not np.isfinite(array[:5]).all()
            or np.any(array[2:4] <= 0.0)):
        raise RuntimeError('Invalid {} for {}'.format(
            key, record.get('filename')))
    return array[:5]


def _sym_key(records):
    for key in ('sym_eood_box', 'sym_eood_original_box', 'symeood_box'):
        if any(key in record for record in records):
            return key
    return None


def audit_payload(audit_json, data_root, source_split,
                  expected_frame_count=59):
    split_root = Path(data_root) / source_split
    images, image_sidecars = _visible_files(
        split_root / 'images', {'.jpg', '.jpeg', '.png'})
    annotations, annotation_sidecars = _visible_files(
        split_root / 'annfiles', {'.txt'})
    image_index = {path.stem: path for path in images}
    annotation_index = {path.stem: path for path in annotations}
    if len(image_index) != len(images) or len(annotation_index) != len(
            annotations):
        raise RuntimeError('Duplicate visible image or annotation stem')

    audit_path = os.path.abspath(os.fspath(audit_json))
    with open(audit_path, 'rb') as handle:
        raw = handle.read()
    payload = json.loads(raw.decode('utf-8'))
    if payload.get('protocol') != ALL_LANE_PROTOCOL:
        raise RuntimeError('Auxiliary source requires a complete all-lane audit')
    records = list(payload.get('records') or [])
    record_index = {}
    for record in records:
        stem = Path(record.get('filename', '')).stem
        if not stem or stem.startswith('._') or stem in record_index:
            raise RuntimeError('Invalid/duplicate all-lane frame key: ' + stem)
        record_index[stem] = record

    visible_keys = set(image_index)
    checks = dict(
        expected_frame_count=(len(visible_keys) == int(expected_frame_count)),
        image_annotation_exact_match=(visible_keys == set(annotation_index)),
        audit_exact_match=(visible_keys == set(record_index)),
        dino_computed_every_frame=all(
            _dino_computed(record) for record in records),
        one_valid_gt_per_frame=True,
        real_sequence_only=all(key.startswith('real_seq')
                               for key in visible_keys))

    gt_index = {}
    for key, path in annotation_index.items():
        boxes = parse_dota_txt(os.fspath(path))
        if len(boxes) != 1:
            checks['one_valid_gt_per_frame'] = False
            continue
        gt_index[key] = np.asarray(boxes[0], dtype=np.float64)[:5]

    dino_counts = Counter()
    sym_counts = Counter()
    selected_sym_key = _sym_key(records)
    rows = []
    for key in sorted(visible_keys & set(record_index) & set(gt_index)):
        record = record_index[key]
        gt = gt_index[key]
        dino = _box(record, 'dino_native_box')
        dino_riou = 0.0 if dino is None else float(compute_riou(dino, gt))
        dino_counts['present' if dino is not None else 'missing'] += 1
        dino_counts['hit' if dino_riou >= 0.5 else 'miss'] += 1
        sym = None if selected_sym_key is None else _box(
            record, selected_sym_key)
        sym_riou = 0.0 if sym is None else float(compute_riou(sym, gt))
        if selected_sym_key is not None:
            sym_counts['present' if sym is not None else 'missing'] += 1
            sym_counts['hit' if sym_riou >= 0.5 else 'miss'] += 1
        frame = int(key.rsplit('_', 1)[1])
        rows.append(dict(
            frame_key=key, frame=frame,
            dino_present=dino is not None, dino_riou=dino_riou,
            dino_hit=dino_riou >= 0.5,
            cached_sym_present=sym is not None,
            cached_sym_riou=sym_riou,
            cached_sym_hit=sym_riou >= 0.5))

    frames = sorted(row['frame'] for row in rows)
    frame_set = set(frames)
    history_support = {
        str(age): sum((frame - age) in frame_set for frame in frames)
        for age in range(1, 5)}
    checks['row_count_complete'] = len(rows) == int(expected_frame_count)
    passed = all(checks.values())
    return dict(
        protocol=PROTOCOL,
        evidence_boundary='auxiliary_source_only_no_fixed_test',
        source_split=str(source_split),
        expected_frame_count=int(expected_frame_count),
        visible_image_count=len(images),
        visible_annotation_count=len(annotations),
        ignored_appledouble_image_count=len(image_sidecars),
        ignored_appledouble_annotation_count=len(annotation_sidecars),
        all_lane_record_count=len(records),
        audit_json=audit_path,
        audit_sha256=_sha256(raw),
        checks=checks,
        dino_summary=dict(dino_counts),
        cached_sym_summary=dict(sym_counts),
        cached_sym_key=selected_sym_key,
        cached_sym_identity='unverified_not_formal_k1_evidence',
        causal_history_available_count_by_age=history_support,
        adjacent_pair_count=history_support['1'],
        rows=rows,
        eligible_for_auxiliary_source_training=passed,
        eligible_for_router_claim=False,
        eligible_for_independent_sequence_claim=False,
        decision=('ALLOW_AUXILIARY_SOURCE_TRAINING_INPUT'
                  if passed else 'STOP_AUXILIARY_SOURCE_CONTRACT_FAILED'))


def main():
    args = parse_args()
    report = audit_payload(
        args.audit_json, args.data_root, args.source_split,
        expected_frame_count=args.expected_frame_count)
    out = os.path.abspath(os.fspath(args.out_json))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print('[aux-source] frames={}'.format(report['visible_image_count']))
    print('[aux-source] dino={}'.format(report['dino_summary']))
    print('[aux-source] adjacent_pairs={}'.format(
        report['adjacent_pair_count']))
    print('[aux-source] decision={}'.format(report['decision']))


if __name__ == '__main__':
    main()
