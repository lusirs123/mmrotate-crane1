#!/usr/bin/env python3
"""Inventory server inputs required before seq11-v2 CV can be designed."""

import argparse
import hashlib
import json
import os
from pathlib import Path


PROTOCOL = 'symeood_dino_seq11_v2_server_input_inventory_v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument(
        '--source-split',
        default='extra_source_real_seq11_pilot_k1p9_v2')
    parser.add_argument('--split-manifest', required=True)
    parser.add_argument('--dino-all-lane-audit', required=True)
    parser.add_argument('--original-source-audit', required=True)
    parser.add_argument('--official-source-val-audit', required=True)
    parser.add_argument('--base-v3-promotion', required=True)
    parser.add_argument('--base-v3-checkpoint', required=True)
    parser.add_argument('--k1-checkpoint', required=True)
    parser.add_argument('--k1-config', required=True)
    parser.add_argument('--v4-replay-config', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    absolute = Path(path).resolve()
    if not absolute.is_file():
        raise RuntimeError('Missing required JSON: ' + os.fspath(absolute))
    with open(absolute, 'r', encoding='utf-8') as handle:
        return absolute, json.load(handle)


def _file_identity(path):
    absolute = Path(path).resolve()
    if not absolute.is_file():
        raise RuntimeError('Missing required file: ' + os.fspath(absolute))
    return dict(
        path=os.fspath(absolute), size_bytes=absolute.stat().st_size,
        sha256=_sha256(absolute))


def _visible(folder, suffixes):
    root = Path(folder).resolve()
    if not root.is_dir():
        raise RuntimeError('Missing required directory: ' + os.fspath(root))
    return {
        path.stem for path in root.iterdir()
        if path.is_file() and not path.name.startswith('._')
        and path.suffix.lower() in suffixes}


def _audit_identity(path, expected_count):
    absolute, payload = _read_json(path)
    records = payload.get('records')
    observed = len(records) if isinstance(records, list) else int(
        payload.get('frame_count', -1))
    if observed != int(expected_count):
        raise RuntimeError(
            '{} must describe exactly {} frames'.format(
                absolute, expected_count))
    if payload.get('fixed_test_read') is True:
        raise RuntimeError('Pre-CV input declares fixed TEST use: '
                           + os.fspath(absolute))
    result = _file_identity(absolute)
    result.update(dict(
        protocol=payload.get('protocol'), frame_count=observed,
        target_data_read=payload.get('target_data_read'),
        fixed_test_read=payload.get('fixed_test_read')))
    return result


def inventory(args):
    source_root = Path(args.data_root).resolve() / args.source_split
    images = _visible(source_root / 'images', {'.jpg', '.jpeg', '.png'})
    annotations = _visible(source_root / 'annfiles', {'.txt'})
    split_manifest_path, split_manifest = _read_json(args.split_manifest)
    dino_audit = _audit_identity(args.dino_all_lane_audit, 251)
    original_audit = _audit_identity(args.original_source_audit, 2781)
    source_val_audit = _audit_identity(args.official_source_val_audit, 738)
    promotion_path, promotion = _read_json(args.base_v3_promotion)
    promotion_output = dict(promotion.get('output') or {})
    base_checkpoint = _file_identity(args.base_v3_checkpoint)
    checks = dict(
        visible_image_count_251=len(images) == 251,
        visible_annotation_count_251=len(annotations) == 251,
        image_annotation_sets_equal=images == annotations,
        split_manifest_frame_count_251=(
            int(split_manifest.get('all_frame_count', -1)) == 251),
        dino_all_lane_count_251=dino_audit['frame_count'] == 251,
        original_source_count_2781=original_audit['frame_count'] == 2781,
        official_source_val_count_738=source_val_audit['frame_count'] == 738,
        base_v3_promotion_passed=(
            promotion.get('decision') ==
            'ALLOW_K1_RETENTIVE_CAUSAL_PHASE_FIXED_BENCHMARK_TEST'),
        base_v3_promotion_target_unread=(
            promotion.get('target_data_read') is False),
        base_v3_promotion_fixed_test_unread=(
            promotion.get('fixed_test_read') is False),
        base_v3_checkpoint_hash_matches_promotion=(
            promotion_output.get('checkpoint_sha256') ==
            base_checkpoint['sha256']))
    passed = all(checks.values())
    return dict(
        protocol=PROTOCOL,
        evidence_boundary='source_only_pre_cv_inventory',
        source_split=args.source_split,
        source_root=os.fspath(source_root),
        source_frame_count=len(images),
        inputs=dict(
            split_manifest=dict(
                **_file_identity(split_manifest_path),
                protocol=split_manifest.get('protocol')),
            dino_all_lane_audit=dino_audit,
            original_source_audit=original_audit,
            official_source_val_audit=source_val_audit,
            base_v3_promotion=dict(
                **_file_identity(promotion_path),
                protocol=promotion.get('protocol'),
                decision=promotion.get('decision')),
            base_v3_checkpoint=base_checkpoint,
            k1_checkpoint=_file_identity(args.k1_checkpoint),
            k1_config=_file_identity(args.k1_config),
            v4_replay_config=_file_identity(args.v4_replay_config)),
        checks=checks,
        target_data_read=False,
        fixed_test_read=False,
        passed=passed,
        decision=(
            'ALLOW_SEQ11_V2_PREFLIGHT_IDENTITY_STAGES' if passed else
            'STOP_SEQ11_V2_SERVER_INPUT_INVENTORY_FAILED'))


def main():
    args = parse_args()
    report = inventory(args)
    output = Path(args.out_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(report, indent=2, ensure_ascii=False) + '\n').encode(
        'utf-8')
    if output.exists() and output.read_bytes() != raw:
        raise RuntimeError('Refusing to overwrite different inventory: '
                           + os.fspath(output))
    if not output.exists():
        output.write_bytes(raw)
    print('[seq11-v2-inventory] output={}'.format(output))
    print('[seq11-v2-inventory] decision={}'.format(report['decision']))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
