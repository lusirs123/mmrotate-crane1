#!/usr/bin/env python3
"""Audit source TRAIN/VAL small-target coverage without model inference."""

import argparse
import collections
import hashlib
import json
import math
import os
from types import SimpleNamespace
from typing import Dict, Sequence

import cv2
import numpy as np

from crane_project.tools import dino_teacher_common as common
from crane_project.tools import dino_teacher_rotated_labeller as labeller


PROTOCOL = 'frozen_dino_source_small_coverage_audit_v1'
MIN_SMALL_FRAMES_PER_SEQUENCE = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--source-train-datasets', nargs='+', required=True)
    parser.add_argument('--source-val-datasets', nargs='+', required=True)
    parser.add_argument('--source-quality-report', required=True)
    parser.add_argument('--dino-height', type=int, default=600)
    parser.add_argument('--dino-max-long-side', type=int, default=1333)
    parser.add_argument('--patch-size', type=int, default=14)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def load_frozen_definition(path: str, args) -> Dict:
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if payload.get('audit') != (
            'Frozen DINO Native-S14 Source Quality Feasibility Audit V2'):
        raise RuntimeError('Source quality report is not the audited V2')
    if int(payload.get('protocol_version', -1)) != 2:
        raise RuntimeError('Source quality report protocol version is not 2')
    if payload.get('decision') not in {
            'SOURCE_NATIVE_QUALITY_SUPPORT_SUFFICIENT_FOR_NEW_SOURCE_ONLY_'
            'DEVELOPMENT_TARGET_NOT_READ',
            'SOURCE_NATIVE_QUALITY_SUPPORT_INSUFFICIENT_COLLECT_NEW_LABELED_'
            'SEQUENCES_TARGET_NOT_READ'}:
        raise RuntimeError('Unexpected source quality report decision')
    if bool((payload.get('protocol') or {}).get('target_read')):
        raise RuntimeError('Source quality report read target data')
    source_data = (payload.get('protocol') or {}).get('source_data') or {}
    if list(source_data.get('train_datasets') or []) != list(
            args.source_train_datasets):
        raise RuntimeError('Source train dataset identity disagrees with V2')
    if list(source_data.get('val_datasets') or []) != list(
            args.source_val_datasets):
        raise RuntimeError('Source validation identity disagrees with V2')
    definition = (payload.get('protocol') or {}).get(
        'source_small_definition') or {}
    if definition.get('definition') != (
            'source_train_short_token_lower_tertile'):
        raise RuntimeError('Unexpected frozen source-small definition')
    threshold = float(definition.get('short_token_threshold', float('nan')))
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise RuntimeError('Invalid frozen source-small threshold')
    return dict(
        path=os.path.abspath(path), sha256=common.file_sha256(path),
        definition=definition, threshold=threshold,
        expected=dict(
            train_frame_count=int(definition['original_count']),
            train_small_frame_count=int(definition['small_frame_count']),
            val_frame_count=int(payload['summary']['frame_count']),
            val_small_frame_count=int(
                payload['summary']['source_small_frame_count'])))


def _record_identity(record: Dict) -> str:
    return '{}|{}|{}|{}'.format(
        record['split'], record['image_split'], record['seq'],
        int(record['frame']))


def read_dota_grab_short_sides(annotation: str):
    """Match MMRotate le90 polygon sizing without importing MMRotate."""
    short_sides = []
    with open(annotation, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 9 or parts[8] != 'grab':
                continue
            polygon = np.asarray(
                [float(value) for value in parts[:8]],
                dtype=np.float32).reshape(4, 2)
            (_, _), (width, height), _ = cv2.minAreaRect(polygon)
            if width >= 2.0 and height >= 2.0:
                short_sides.append(float(min(width, height)))
    return short_sides


def inspect_record(record: Dict, scale_args, threshold: float) -> Dict:
    image = cv2.imread(record['image'], cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('Cannot read {}'.format(record['image']))
    height, width = image.shape[:2]
    scale = min(
        float(scale_args.dino_height) / float(min(height, width)),
        float(scale_args.dino_max_long_side) / float(max(height, width)))
    short_sides = read_dota_grab_short_sides(record['annotation'])
    token = (
        min(short_sides) * scale / float(scale_args.patch_size)
        if short_sides else float('inf'))
    return dict(
        role=record['role'], annotation_split=record['split'],
        image_split=record['image_split'], domain=record['domain'],
        seq=record['seq'], frame=int(record['frame']),
        object_count=len(short_sides),
        short_token=(None if not math.isfinite(token) else float(token)),
        source_small=bool(math.isfinite(token) and token <= threshold),
        image_sha256=common.file_sha256(record['image']),
        annotation_sha256=common.file_sha256(record['annotation']))


def summarize_group(rows: Sequence[Dict]) -> Dict:
    tokens = [row['short_token'] for row in rows
              if row['short_token'] is not None]
    small = sum(row['source_small'] for row in rows)
    return dict(
        frame_count=len(rows),
        labeled_frame_count=sum(row['object_count'] > 0 for row in rows),
        empty_gt_frame_count=sum(row['object_count'] == 0 for row in rows),
        object_count=sum(row['object_count'] for row in rows),
        small_frame_count=int(small),
        small_frame_fraction=(float(small / len(rows)) if rows else None),
        short_token_min=(float(min(tokens)) if tokens else None),
        short_token_median=(float(np.median(tokens)) if tokens else None),
        short_token_max=(float(max(tokens)) if tokens else None))


def summarize_coverage(rows: Sequence[Dict]) -> Dict:
    roles = ('source_train', 'source_validation')
    by_role = {}
    by_domain = {}
    by_sequence = []
    for role in roles:
        role_rows = [row for row in rows if row['role'] == role]
        by_role[role] = summarize_group(role_rows)
        for domain in sorted({row['domain'] for row in role_rows}):
            by_domain['{}|{}'.format(role, domain)] = summarize_group([
                row for row in role_rows if row['domain'] == domain])
        for (split, domain, seq), grouped in sorted(
                _group_rows(role_rows).items()):
            summary = summarize_group(grouped)
            summary.update(
                role=role, annotation_split=split,
                domain=domain, sequence=seq,
                qualifies=bool(
                    summary['small_frame_count'] >=
                    MIN_SMALL_FRAMES_PER_SEQUENCE))
            by_sequence.append(summary)

    def role_gate(role):
        qualifying = [row for row in by_sequence
                      if row['role'] == role and row['qualifies']]
        real = [row for row in qualifying if row['domain'] == 'real']
        return dict(
            minimum_small_frames_per_sequence=(
                MIN_SMALL_FRAMES_PER_SEQUENCE),
            qualifying_sequence_count=len(qualifying),
            qualifying_real_sequence_count=len(real),
            checks=dict(
                at_least_two_independent_sequences=len(qualifying) >= 2,
                at_least_one_real_sequence=len(real) >= 1),
            passed=bool(len(qualifying) >= 2 and len(real) >= 1))

    train_gate = role_gate('source_train')
    val_gate = role_gate('source_validation')
    if not train_gate['passed']:
        decision = 'SOURCE_TRAIN_SMALL_COVERAGE_INSUFFICIENT'
    elif not val_gate['passed']:
        decision = 'SOURCE_TRAIN_AVAILABLE_VALIDATION_COVERAGE_INSUFFICIENT'
    else:
        decision = 'SOURCE_TRAIN_AND_VALIDATION_SMALL_COVERAGE_AVAILABLE'
    return dict(
        by_role=by_role, by_role_and_domain=by_domain,
        by_sequence=by_sequence,
        development_coverage=dict(
            source_train=train_gate, source_validation=val_gate,
            passed=bool(train_gate['passed'] and val_gate['passed'])),
        decision=decision)


def _group_rows(rows: Sequence[Dict]):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row['annotation_split'], row['domain'], row['seq'])].append(
            row)
    return grouped


def build_audit(args) -> Dict:
    frozen = load_frozen_definition(args.source_quality_report, args)
    records = []
    for role, specs in (
            ('source_train', args.source_train_datasets),
            ('source_validation', args.source_val_datasets)):
        for annotation_split, image_split in labeller.parse_dataset_specs(
                specs):
            discovered = labeller.discover_labeled_records_with_image_split(
                args.data_root, annotation_split, image_split)
            for record in discovered:
                record = dict(record)
                record['role'] = role
                records.append(record)
    identities = [_record_identity(record) for record in records]
    if len(identities) != len(set(identities)):
        raise RuntimeError('Duplicate source record identity')
    scale_args = SimpleNamespace(
        dino_height=args.dino_height,
        dino_max_long_side=args.dino_max_long_side,
        patch_size=args.patch_size)
    rows = [inspect_record(record, scale_args, frozen['threshold'])
            for record in records]
    manifest = hashlib.sha256()
    for row in rows:
        manifest.update((
            '{}|{}|{}|{}|{}|{}|{}\n'.format(
                row['role'], row['annotation_split'], row['seq'],
                row['frame'], row['image_sha256'],
                row['annotation_sha256'], row['source_small'])
        ).encode('utf-8'))
    summary = summarize_coverage(rows)
    actual = dict(
        train_frame_count=summary['by_role']['source_train']['frame_count'],
        train_small_frame_count=summary['by_role']['source_train'][
            'small_frame_count'],
        val_frame_count=summary['by_role']['source_validation']['frame_count'],
        val_small_frame_count=summary['by_role']['source_validation'][
            'small_frame_count'])
    if actual != frozen['expected']:
        raise RuntimeError(
            'Metadata-only coverage does not reproduce frozen TRAIN/VAL '
            'small counts: expected {} found {}'.format(
                frozen['expected'], actual))
    return dict(
        audit='Frozen DINO Source TRAIN/VAL Small Coverage Audit V1',
        protocol=PROTOCOL, protocol_version=1,
        evidence_boundary=dict(
            metadata_and_ground_truth_only=True, model_inference=False,
            parameter_update=False, target_discovered=False,
            target_read=False, target_used_for_training=False,
            target_used_for_selection=False,
            coverage_gate_is_data_audit_not_performance_claim=True),
        inputs=dict(
            source_quality_report=dict(
                path=frozen['path'], sha256=frozen['sha256']),
            source_small_definition=frozen['definition'],
            scale=dict(
                dino_height=args.dino_height,
                dino_max_long_side=args.dino_max_long_side,
                patch_size=args.patch_size),
            source_data=dict(
                train_datasets=list(args.source_train_datasets),
                val_datasets=list(args.source_val_datasets)),
            source_manifest=dict(
                definition=('sha256_of_ordered_role_split_seq_frame_'
                            'image_sha_annotation_sha_small_lines'),
                frame_count=len(rows), sha256=manifest.hexdigest())),
        summary=summary, rows=rows, decision=summary['decision'])


def main():
    args = parse_args()
    if os.path.exists(args.out_json):
        raise RuntimeError('Refusing to overwrite {}'.format(args.out_json))
    output_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(output_dir, exist_ok=True)
    result = build_audit(args)
    replacements = common.write_json_atomic(args.out_json, result)
    print('[dino-source-small-coverage] {}'.format(result['decision']))
    print('[dino-source-small-coverage] train_small={} val_small={} '
          'train_gate={} val_gate={}'.format(
              result['summary']['by_role']['source_train'][
                  'small_frame_count'],
              result['summary']['by_role']['source_validation'][
                  'small_frame_count'],
              result['summary']['development_coverage']['source_train'][
                  'passed'],
              result['summary']['development_coverage'][
                  'source_validation']['passed']))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(os.path.abspath(args.out_json)))


if __name__ == '__main__':
    main()
