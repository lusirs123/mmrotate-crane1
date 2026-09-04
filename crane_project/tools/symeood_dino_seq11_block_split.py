#!/usr/bin/env python3
"""Materialize a leakage-safe real_seq11 auxiliary train/val split.

The original 59-frame pilot remains immutable.  This tool creates two small
views using hard links (or copies), filters the complete all-lane audit for
each view, and refuses overlaps, missing files, sidecars, or stale outputs.
It never invokes a detector and never reads fixed TEST.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path


PROTOCOL = 'real_seq11_auxiliary_temporal_block_split_v1'
PROTOCOL_V2 = 'real_seq11_source_k1p9_block_manifest_v2'
REPORT_PROTOCOL = 'real_seq11_auxiliary_temporal_block_materialization_v1'
ALL_LANE_PROTOCOL = 'source_owned_geometry_union_v2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument(
        '--source-split', default='extra_source_real_seq11_pilot_k1p9')
    parser.add_argument(
        '--train-split',
        default='extra_source_real_seq11_pilot_k1p9_train_v1')
    parser.add_argument(
        '--val-split', default='extra_source_real_seq11_pilot_k1p9_val_v1')
    parser.add_argument(
        '--audit-json', default=None,
        help=('Optional complete all-lane audit.  Omit it for data-only '
              'materialization; data-only output never authorizes training.'))
    parser.add_argument(
        '--split-manifest',
        default=('crane_project/data_contracts/'
                 'real_seq11_pilot_k1p9_blocksplit_v1.json'))
    parser.add_argument('--train-audit-json', default=None)
    parser.add_argument('--val-audit-json', default=None)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--mode', choices=('hardlink', 'copy'),
                        default='hardlink')
    return parser.parse_args()


def _sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        raw = handle.read()
    return absolute, raw, json.loads(raw.decode('utf-8'))


def _stem(sequence, frame):
    # The pilot preserves original video frame ids with six decimal digits,
    # for example real_seq11_000125.jpg.  This differs from some legacy crane
    # splits that use five digits, so the width is part of this data contract.
    return '{}_{:06d}'.format(sequence, int(frame))


def _manifest(path):
    absolute, raw, payload = _read_json(path)
    protocol = payload.get('protocol')
    if protocol not in (PROTOCOL, PROTOCOL_V2):
        raise RuntimeError('Unexpected seq11 split-manifest protocol')
    sequence = str(payload.get('sequence'))
    if protocol == PROTOCOL:
        train = [int(value) for value in
                 payload.get('aux_train_frames') or []]
        val = [int(value) for value in payload.get('aux_val_frames') or []]
        train_stems = {_stem(sequence, frame) for frame in train}
        val_stems = {_stem(sequence, frame) for frame in val}
    else:
        train_list = [str(value) for value in
                      payload.get('train_stems') or []]
        val_list = [str(value) for value in
                    payload.get('aux_val_stems') or []]
        pattern = re.compile(r'^{}_[0-9]{{6}}$'.format(
            re.escape(sequence)))
        if not all(pattern.fullmatch(stem)
                   for stem in train_list + val_list):
            raise RuntimeError('Invalid V2 seq11 stem')
        train_stems, val_stems = set(train_list), set(val_list)
        train = [int(stem.rsplit('_', 1)[1]) for stem in train_list]
        val = [int(stem.rsplit('_', 1)[1]) for stem in val_list]
    train_set, val_set = set(train), set(val)
    if protocol == PROTOCOL:
        checks = dict(
            train_unique=len(train) == len(train_set),
            val_unique=len(val) == len(val_set),
            train_val_disjoint=not bool(train_set & val_set),
            train_count=(len(train) == int(
                payload.get('aux_train_frame_count', -1))),
            val_count=(len(val) == int(
                payload.get('aux_val_frame_count', -1))),
            all_count=(len(train_set | val_set)
                       == int(payload.get('all_frame_count', -1))),
            temporal_metrics_prohibited=(
                payload.get('temporal_metrics_authorized') is False),
            independent_sequence_claim_prohibited=(
                payload.get('independent_sequence_claim_authorized')
                is False),
            fixed_test_prohibited=(
                payload.get('fixed_test_use_authorized') is False))
        holdout = dict(payload.get('holdout_rule') or {})
        start, end = int(holdout.get('start_frame', -1)), int(
            holdout.get('end_frame', -1))
        checks['closed_holdout_interval_exact'] = val_set == {
            frame for frame in train_set | val_set if start <= frame <= end}
    else:
        boundary = dict(payload.get('evidence_boundary') or {})
        checks = dict(
            train_unique=len(train) == len(train_stems),
            val_unique=len(val) == len(val_stems),
            train_val_disjoint=not bool(train_stems & val_stems),
            train_count=(len(train_stems) == int(
                payload.get('train_frame_count', -1))),
            val_count=(len(val_stems) == int(
                payload.get('aux_val_frame_count', -1))),
            all_count=(len(train_stems | val_stems) == int(
                payload.get('all_frame_count', -1))),
            whole_window_split=(payload.get('split_policy') ==
                                'whole_contiguous_windows_no_frame_randomization'),
            six_digit_file_naming=(payload.get('file_naming') ==
                                   'real_seq11_XXXXXX'),
            independent_sequence_claim_prohibited=(
                boundary.get('not_independent_sequence') is True),
            official_source_val_kept_separate=(
                boundary.get('official_source_val_738_remains_separate')
                is True),
            fixed_test_prohibited=(boundary.get('fixed_test_role') is False),
            positive_k0=float(payload.get('k0', 0.0)) > 0.0)
    if not all(checks.values()):
        raise RuntimeError('Invalid seq11 split manifest: {}'.format(
            [key for key, value in checks.items() if not value]))
    return dict(
        path=absolute, raw=raw, sha256=_sha256_bytes(raw), payload=payload,
        protocol=protocol,
        sequence=sequence, train_frames=train, val_frames=val,
        train_stems=train_stems, val_stems=val_stems, checks=checks)


def _visible_index(folder, suffixes):
    folder = Path(folder)
    if not folder.is_dir():
        raise RuntimeError('Missing input directory: ' + os.fspath(folder))
    index = {}
    sidecars = []
    for path in sorted(folder.iterdir()):
        if path.name.startswith('._'):
            sidecars.append(path.name)
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem in index:
                raise RuntimeError('Duplicate visible stem: ' + path.stem)
            index[path.stem] = path
    return index, sidecars


def _record_index(payload):
    if payload.get('protocol') != ALL_LANE_PROTOCOL:
        raise RuntimeError(
            'Input must be a complete source-owned all-lane audit')
    records = list(payload.get('records') or [])
    index = {}
    for record in records:
        stem = Path(record.get('filename', '')).stem
        if not stem or stem.startswith('._') or stem in index:
            raise RuntimeError('Invalid or duplicate audit record: ' + stem)
        index[stem] = record
    return index


def _same_file(first, second):
    return (os.path.getsize(first) == os.path.getsize(second)
            and _sha256_file(first) == _sha256_file(second))


def _materialize_file(source, target, mode):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not _same_file(source, target):
            raise RuntimeError('Refusing to overwrite a different file: '
                               + os.fspath(target))
        return 'reused'
    if mode == 'hardlink':
        try:
            os.link(source, target)
            return 'hardlinked'
        except OSError:
            shutil.copy2(source, target)
            return 'copied_after_hardlink_fallback'
    shutil.copy2(source, target)
    return 'copied'


def _write_exact_json(path, payload):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, ensure_ascii=False) + '\n').encode(
        'utf-8')
    if target.exists():
        if target.read_bytes() != raw:
            raise RuntimeError('Refusing to overwrite a different JSON: '
                               + os.fspath(target))
    else:
        target.write_bytes(raw)
    return os.fspath(target), _sha256_bytes(raw)


def _filtered_audit(payload, stems, output_image_dir, role, manifest_sha256):
    index = _record_index(payload)
    records = []
    for stem in sorted(stems):
        record = dict(index[stem])
        record['source_filename'] = record.get('filename')
        suffix = Path(record['filename']).suffix or '.jpg'
        record['filename'] = os.fspath(
            (Path(output_image_dir) / (stem + suffix)).resolve())
        record['auxiliary_split_role'] = role
        record['auxiliary_split_manifest_sha256'] = manifest_sha256
        records.append(record)
    result = dict(payload)
    result['records'] = records
    result['frame_count'] = len(records)
    result['auxiliary_split_role'] = role
    result['auxiliary_split_manifest_sha256'] = manifest_sha256
    result['fixed_test_read'] = False
    return result


def materialize(args):
    contract = _manifest(args.split_manifest)
    data_root = Path(args.data_root).resolve()
    source_root = data_root / args.source_split
    train_root = data_root / args.train_split
    val_root = data_root / args.val_split
    if len({source_root, train_root, val_root}) != 3:
        raise RuntimeError('Source/train/val split paths must be distinct')

    images, image_sidecars = _visible_index(
        source_root / 'images', {'.jpg', '.jpeg', '.png'})
    annotations, annotation_sidecars = _visible_index(
        source_root / 'annfiles', {'.txt'})
    wanted = contract['train_stems'] | contract['val_stems']
    if set(images) != wanted or set(annotations) != wanted:
        mismatch = dict(
            expected_count=len(wanted),
            image_count=len(images),
            annotation_count=len(annotations),
            missing_images=sorted(wanted - set(images))[:20],
            extra_images=sorted(set(images) - wanted)[:20],
            missing_annotations=sorted(wanted - set(annotations))[:20],
            extra_annotations=sorted(set(annotations) - wanted)[:20])
        raise RuntimeError(
            'Original seq11 files do not exactly match manifest: {}'.format(
                mismatch))

    audit_args = [getattr(args, name, None) for name in (
        'audit_json', 'train_audit_json', 'val_audit_json')]
    if any(audit_args) and not all(audit_args):
        raise RuntimeError(
            'audit-json, train-audit-json, and val-audit-json must be '
            'provided together')
    audit_enabled = all(audit_args)
    audit_path = audit_raw = audit_payload = None
    if audit_enabled:
        audit_path, audit_raw, audit_payload = _read_json(args.audit_json)
        audit_index = _record_index(audit_payload)
        if set(audit_index) != wanted:
            raise RuntimeError(
                'Original seq11 audit does not exactly match manifest')

    action_counts = {}
    for role, stems, target_root in (
            ('aux-train', contract['train_stems'], train_root),
            ('aux-val', contract['val_stems'], val_root)):
        for stem in sorted(stems):
            for source, target in (
                    (images[stem], target_root / 'images' / images[stem].name),
                    (annotations[stem],
                     target_root / 'annfiles' / annotations[stem].name)):
                action = _materialize_file(source, target, args.mode)
                action_counts[action] = action_counts.get(action, 0) + 1
        observed_images, _ = _visible_index(
            target_root / 'images', {'.jpg', '.jpeg', '.png'})
        observed_annotations, _ = _visible_index(
            target_root / 'annfiles', {'.txt'})
        if set(observed_images) != stems or set(observed_annotations) != stems:
            raise RuntimeError(
                'Materialized {} split has stale/extra files'.format(role))

    train_audit = train_audit_sha = None
    val_audit = val_audit_sha = None
    if audit_enabled:
        train_audit, train_audit_sha = _write_exact_json(
            args.train_audit_json,
            _filtered_audit(
                audit_payload, contract['train_stems'],
                train_root / 'images', 'aux-train', contract['sha256']))
        val_audit, val_audit_sha = _write_exact_json(
            args.val_audit_json,
            _filtered_audit(
                audit_payload, contract['val_stems'],
                val_root / 'images', 'aux-val', contract['sha256']))

    report = dict(
        protocol=REPORT_PROTOCOL,
        evidence_boundary='auxiliary_source_only_no_fixed_test',
        target_data_read=False,
        fixed_test_read=False,
        source_split=args.source_split,
        train_split=args.train_split,
        val_split=args.val_split,
        input_manifest_protocol=contract['protocol'],
        split_manifest=contract['path'],
        split_manifest_sha256=contract['sha256'],
        input_audit=audit_path,
        input_audit_sha256=(
            _sha256_bytes(audit_raw) if audit_raw is not None else None),
        train_audit=train_audit,
        train_audit_sha256=train_audit_sha,
        val_audit=val_audit,
        val_audit_sha256=val_audit_sha,
        all_frame_count=len(wanted),
        aux_train_frame_count=len(contract['train_stems']),
        aux_val_frame_count=len(contract['val_stems']),
        train_val_overlap_count=len(
            contract['train_stems'] & contract['val_stems']),
        ignored_appledouble_image_count=len(image_sidecars),
        ignored_appledouble_annotation_count=len(annotation_sidecars),
        materialization_actions=action_counts,
        checks=contract['checks'],
        filtered_audits_written=audit_enabled,
        detector_forward_run=False,
        temporal_metrics_authorized=False,
        eligible_for_auxiliary_blocksplit_training=audit_enabled,
        eligible_for_independent_sequence_claim=False,
        decision=('ALLOW_SEQ11_BLOCKSPLIT_SOURCE_TRAINING'
                  if audit_enabled else
                  'SEQ11_BLOCKSPLIT_DATA_MATERIALIZED_AUDIT_REQUIRED'))
    output, _ = _write_exact_json(args.out_json, report)
    report['out_json'] = output
    return report


def main():
    args = parse_args()
    report = materialize(args)
    print('[seq11-split] train={}'.format(report['aux_train_frame_count']))
    print('[seq11-split] val={}'.format(report['aux_val_frame_count']))
    print('[seq11-split] overlap={}'.format(
        report['train_val_overlap_count']))
    print('[seq11-split] decision={}'.format(report['decision']))


if __name__ == '__main__':
    main()
