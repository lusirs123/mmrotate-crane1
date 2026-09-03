#!/usr/bin/env python3
"""Materialize all three pre-registered seq11 block-CV folds."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from crane_project.tools.symeood_dino_seq11_block_split import (
    ALL_LANE_PROTOCOL, _materialize_file, _record_index, _visible_index,
    _write_exact_json)


PROTOCOL = 'real_seq11_auxiliary_three_window_block_cv_v1'
REPORT_PROTOCOL = 'real_seq11_three_window_block_cv_materialization_v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument(
        '--source-split', default='extra_source_real_seq11_pilot_k1p9')
    parser.add_argument('--audit-json', required=True)
    parser.add_argument(
        '--cv-manifest',
        default=('crane_project/data_contracts/'
                 'real_seq11_pilot_k1p9_three_window_block_cv_v1.json'))
    parser.add_argument(
        '--out-root',
        default=('work_dirs/crane_symeood_dino_source_inventory_v1/'
                 'real_seq11_pilot_k1p9/three_window_block_cv_v1'))
    parser.add_argument('--mode', choices=('hardlink', 'copy'),
                        default='hardlink')
    return parser.parse_args()


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _read_json(path):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        raw = handle.read()
    return absolute, raw, json.loads(raw.decode('utf-8'))


def _stem(frame):
    return 'real_seq11_{:06d}'.format(int(frame))


def load_manifest(path):
    absolute, raw, payload = _read_json(path)
    if payload.get('protocol') != PROTOCOL:
        raise RuntimeError('Unexpected seq11 block-CV manifest protocol')
    if int(payload.get('all_frame_count', -1)) != 59:
        raise RuntimeError('Block-CV manifest must describe all 59 frames')
    folds = []
    validation_union = set()
    for expected_id, item in enumerate(payload.get('folds') or [], 1):
        fold = dict(item)
        fold_id = int(fold.get('fold_id', -1))
        if fold_id != expected_id:
            raise RuntimeError('Block-CV folds must be ordered 1..3')
        val_frames = [int(value) for value in
                      fold.get('validation_frames') or []]
        val_stems = {_stem(frame) for frame in val_frames}
        if len(val_stems) != len(val_frames):
            raise RuntimeError('Duplicate validation frame in fold {}'.format(
                fold_id))
        if validation_union & val_stems:
            raise RuntimeError('Validation folds overlap')
        validation_union.update(val_stems)
        if len(val_stems) != int(fold.get('validation_frame_count', -1)):
            raise RuntimeError('Fold validation count mismatch')
        folds.append(dict(
            fold_id=fold_id, validation_frames=val_frames,
            validation_stems=val_stems,
            expected_train_count=int(fold.get('training_frame_count', -1)),
            validation_rule=fold.get('validation_rule')))
    checks = dict(
        fold_count=len(folds) == 3,
        oof_count=(len(validation_union)
                   == int(payload.get('oof_frame_count', -1))),
        validation_overlap_zero=(
            int(payload.get('fold_validation_overlap_count', -1)) == 0),
        sparse_temporal_metrics_prohibited=(
            payload.get('temporal_metrics_authorized_on_sparse_oof') is False),
        fixed_test_prohibited=(
            payload.get('fixed_test_use_authorized') is False),
        performance_fold_selection_prohibited=(
            payload.get(
                'fold_selection_from_observed_performance_authorized')
            is False))
    if not all(checks.values()):
        raise RuntimeError('Invalid block-CV manifest: {}'.format(
            [key for key, value in checks.items() if not value]))
    return dict(
        path=absolute, sha256=_sha256(raw), payload=payload, folds=folds,
        validation_union=validation_union, checks=checks)


def _filtered_audit(payload, stems, image_dir, role, fold_id, manifest_sha):
    source_index = _record_index(payload)
    records = []
    for stem in sorted(stems):
        record = dict(source_index[stem])
        record['source_filename'] = record.get('filename')
        suffix = Path(record['filename']).suffix or '.jpg'
        record['filename'] = os.fspath(
            (Path(image_dir) / (stem + suffix)).resolve())
        record['auxiliary_cv_role'] = role
        record['auxiliary_cv_fold'] = int(fold_id)
        record['auxiliary_cv_manifest_sha256'] = manifest_sha
        records.append(record)
    result = dict(payload)
    result.update(dict(
        records=records, frame_count=len(records),
        auxiliary_cv_role=role, auxiliary_cv_fold=int(fold_id),
        auxiliary_cv_manifest_sha256=manifest_sha,
        fixed_test_read=False))
    return result


def _mismatch(observed, expected):
    return dict(
        expected_count=len(expected), observed_count=len(observed),
        missing=sorted(expected - set(observed))[:20],
        extra=sorted(set(observed) - expected)[:20])


def materialize(args):
    manifest = load_manifest(args.cv_manifest)
    data_root = Path(args.data_root).resolve()
    source_root = data_root / args.source_split
    images, image_sidecars = _visible_index(
        source_root / 'images', {'.jpg', '.jpeg', '.png'})
    annotations, annotation_sidecars = _visible_index(
        source_root / 'annfiles', {'.txt'})
    audit_path, audit_raw, audit_payload = _read_json(args.audit_json)
    if audit_payload.get('protocol') != ALL_LANE_PROTOCOL:
        raise RuntimeError('Input audit is not a complete all-lane audit')
    audit_index = _record_index(audit_payload)
    all_stems = set(audit_index)
    if len(all_stems) != 59:
        raise RuntimeError('Original seq11 audit must contain 59 frames')
    if set(images) != all_stems or set(annotations) != all_stems:
        raise RuntimeError('Original seq11 files/audit mismatch: {}'.format(
            dict(images=_mismatch(images, all_stems),
                 annotations=_mismatch(annotations, all_stems))))
    if not manifest['validation_union'] <= all_stems:
        raise RuntimeError('CV manifest contains frames absent from source')

    out_root = Path(args.out_root).resolve()
    full_split_name = 'extra_source_real_seq11_pilot_k1p9_cv_full59_v1'
    full_root = data_root / full_split_name
    action_counts = {}
    for stem in sorted(all_stems):
        for source, target in (
                (images[stem], full_root / 'images' / images[stem].name),
                (annotations[stem],
                 full_root / 'annfiles' / annotations[stem].name)):
            action = _materialize_file(source, target, args.mode)
            action_counts[action] = action_counts.get(action, 0) + 1
    full_images, _ = _visible_index(
        full_root / 'images', {'.jpg', '.jpeg', '.png'})
    full_annotations, _ = _visible_index(
        full_root / 'annfiles', {'.txt'})
    if (set(full_images) != all_stems
            or set(full_annotations) != all_stems):
        raise RuntimeError('Clean full59 view contains stale/extra files')
    full_audit, full_audit_sha = _write_exact_json(
        out_root / 'full59_all_lane_audit.json',
        _filtered_audit(
            audit_payload, all_stems, full_root / 'images', 'full59', 0,
            manifest['sha256']))

    fold_reports = []
    for fold in manifest['folds']:
        fold_id = fold['fold_id']
        val_stems = fold['validation_stems']
        train_stems = all_stems - val_stems
        if len(train_stems) != fold['expected_train_count']:
            raise RuntimeError('Fold {} train count mismatch'.format(fold_id))
        split_reports = {}
        for role, stems in (('train', train_stems), ('val', val_stems)):
            split_name = (
                'extra_source_real_seq11_pilot_k1p9_cv_fold{}_{}_v1'.format(
                    fold_id, role))
            target_root = data_root / split_name
            for stem in sorted(stems):
                for source, target in (
                        (images[stem],
                         target_root / 'images' / images[stem].name),
                        (annotations[stem],
                         target_root / 'annfiles' / annotations[stem].name)):
                    action = _materialize_file(source, target, args.mode)
                    action_counts[action] = action_counts.get(action, 0) + 1
            observed_images, _ = _visible_index(
                target_root / 'images', {'.jpg', '.jpeg', '.png'})
            observed_annotations, _ = _visible_index(
                target_root / 'annfiles', {'.txt'})
            if (set(observed_images) != stems
                    or set(observed_annotations) != stems):
                raise RuntimeError('Fold view contains stale/extra files')
            audit_output, audit_sha = _write_exact_json(
                out_root / 'fold{}'.format(fold_id)
                / '{}_all_lane_audit.json'.format(role),
                _filtered_audit(
                    audit_payload, stems, target_root / 'images', role,
                    fold_id, manifest['sha256']))
            split_reports[role] = dict(
                split_name=split_name, frame_count=len(stems),
                audit_json=audit_output, audit_sha256=audit_sha)
        fold_reports.append(dict(
            fold_id=fold_id,
            validation_rule=fold['validation_rule'],
            train=split_reports['train'], val=split_reports['val'],
            train_val_overlap_count=len(train_stems & val_stems)))

    report = dict(
        protocol=REPORT_PROTOCOL,
        evidence_boundary='auxiliary_source_only_no_fixed_test',
        target_data_read=False, fixed_test_read=False,
        input=dict(
            source_root=os.fspath(source_root), audit_json=audit_path,
            audit_sha256=_sha256(audit_raw),
            cv_manifest=manifest['path'],
            cv_manifest_sha256=manifest['sha256'],
            image_sidecar_count=len(image_sidecars),
            annotation_sidecar_count=len(annotation_sidecars)),
        frame_count=len(all_stems), oof_frame_count=len(
            manifest['validation_union']),
        clean_full59=dict(
            split_name=full_split_name, frame_count=len(all_stems),
            audit_json=full_audit, audit_sha256=full_audit_sha),
        fold_reports=fold_reports, action_counts=action_counts,
        checks=dict(
            source_frame_count_59=len(all_stems) == 59,
            clean_full59_frame_count_59=(
                len(full_images) == len(full_annotations) == 59),
            fold_count_3=len(fold_reports) == 3,
            every_fold_disjoint=all(
                row['train_val_overlap_count'] == 0
                for row in fold_reports),
            temporal_metrics_not_authorized=True,
            fixed_test_not_read=True),
        passed=True,
        decision='ALLOW_SEQ11_THREE_WINDOW_BLOCK_CV_SUPPORT_AUDIT')
    report_path, _ = _write_exact_json(
        out_root / 'cv_materialization.json', report)
    report['out_json'] = report_path
    return report


def main():
    args = parse_args()
    report = materialize(args)
    print('[seq11-cv] folds={}'.format(len(report['fold_reports'])))
    print('[seq11-cv] oof={}'.format(report['oof_frame_count']))
    print('[seq11-cv] decision={}'.format(report['decision']))


if __name__ == '__main__':
    main()
