#!/usr/bin/env python3
"""Bind the seq11-v2 cached native-S14 DINO lane to its source inputs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


PROTOCOL = 'seq11_v2_dino_cache_identity_v1'
AUDIT_PROTOCOL = 'source_owned_geometry_union_v2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-json', required=True)
    parser.add_argument('--full-source-contract', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--collection-config', required=True)
    parser.add_argument('--unified-config', required=True)
    parser.add_argument('--dino-head-checkpoint', required=True)
    parser.add_argument('--dinov2-checkpoint', required=True)
    parser.add_argument('--k1-checkpoint', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path):
    absolute = Path(path).resolve()
    if not absolute.is_file():
        raise RuntimeError('Missing DINO identity input: ' + os.fspath(absolute))
    return dict(path=os.fspath(absolute), size_bytes=absolute.stat().st_size,
                sha256=_sha256(absolute))


def _json(path):
    identity = _identity(path)
    with open(identity['path'], 'r', encoding='utf-8') as handle:
        return identity, json.load(handle)


def _write_exact(path, payload):
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, ensure_ascii=False) + '\n').encode(
        'utf-8')
    if output.exists() and output.read_bytes() != raw:
        raise RuntimeError('Refusing to overwrite different DINO identity: '
                           + os.fspath(output))
    if not output.exists():
        output.write_bytes(raw)
    return os.fspath(output)


def audit(args):
    audit_id, audit = _json(args.audit_json)
    contract_id, contract = _json(args.full_source_contract)
    manifest_id, manifest = _json(args.source_manifest)
    collection = _identity(args.collection_config)
    unified = _identity(args.unified_config)
    head = _identity(args.dino_head_checkpoint)
    backbone = _identity(args.dinov2_checkpoint)
    k1 = _identity(args.k1_checkpoint)
    if Path(head['path']).name != 'source_safe_interpolated_head.pth':
        raise RuntimeError('Unexpected formal DINO head checkpoint name')
    if Path(backbone['path']).name != 'dinov2_vitl14_pretrain.pth':
        raise RuntimeError('Unexpected DINOv2 backbone checkpoint name')
    if Path(k1['path']).name != 'epoch_24.pth':
        raise RuntimeError('Unexpected ordinary K1 checkpoint name')

    records = audit.get('records') or []
    stems = [Path(row.get('filename', '')).stem for row in records]
    declared = set(manifest.get('train_stems') or [])
    declared.update(manifest.get('aux_val_stems') or [])
    invocation_ok = all(
        row.get('raw_selected_source') != 'not_computed'
        and row.get('dino_invoked', True) in (True, 1)
        and 'dino_native_box' in row for row in records)
    boxes_ok = True
    for row in records:
        box = row.get('dino_native_box')
        if box is not None:
            boxes_ok = boxes_ok and isinstance(box, list) and len(box) >= 5
            boxes_ok = boxes_ok and all(
                math.isfinite(float(value)) for value in box[:5])
            boxes_ok = boxes_ok and float(box[2]) > 0 and float(box[3]) > 0
    metadata = audit.get('metadata') or {}
    forwards = metadata.get('runtime_forward_counts') or {}
    collection_text = Path(collection['path']).read_text(encoding='utf-8')
    unified_text = Path(unified['path']).read_text(encoding='utf-8')
    checks = dict(
        audit_protocol_valid=audit.get('protocol') == AUDIT_PROTOCOL,
        exact_251_records=len(records) == 251,
        unique_frame_keys=len(set(stems)) == len(stems),
        source_manifest_exact_set=set(stems) == declared,
        source_manifest_protocol=(
            manifest.get('protocol') ==
            'real_seq11_source_k1p9_block_manifest_v2'),
        source_manifest_geometry=(
            manifest.get('target_geometry') == 'top_beam_only'
            and float(manifest.get('k0', -1)) == 1.9),
        dino_computed_every_frame=invocation_ok,
        dino_boxes_valid=boxes_ok,
        dino_forward_count_251=forwards.get('dino') == 251,
        geometry_refiner_forward_count_zero=(
            forwards.get('geometry_refiner') == 0),
        contract_binds_audit=(
            contract.get('audit_sha256') == audit_id['sha256']),
        contract_source_only=(
            contract.get('evidence_boundary') ==
            'auxiliary_source_only_no_fixed_test'
            and contract.get('eligible_for_auxiliary_source_training') is True),
        collection_declares_native_s14=(
            "frozen_dino_variant='native_s14_source_safe_interpolated_head'"
            in collection_text),
        collection_declares_both_lanes=(
            'both_lanes_required_every_frame=True' in collection_text),
        collection_declares_unread_flags=(
            'target_data_read=False' in collection_text
            and 'fixed_test_read=False' in collection_text),
        unified_declares_s14_only=(
            'feature_strides=[14]' in unified_text
            and 's7_residual=False' in unified_text),
        unified_declares_alpha_0p5=('alpha=0.5' in unified_text))
    passed = all(checks.values())
    return dict(
        protocol=PROTOCOL,
        evidence_boundary='source_only_cached_dino_input_identity',
        identity_grade=(
            'reconstructed_from_config_checkpoint_and_runtime_audit'),
        inputs=dict(
            audit_json=audit_id, full_source_contract=contract_id,
            source_manifest=manifest_id, collection_config=collection,
            unified_config=unified, dino_head_checkpoint=head,
            dinov2_checkpoint=backbone, k1_checkpoint=k1),
        dino_contract=dict(
            variant='native_s14_source_safe_interpolated_head',
            feature_stride=14, classifier_interpolation_alpha=0.5,
            s7_enabled=False, target_geometry='top_beam_only',
            prediction_coordinate_system='original_image_pixels',
            obb_convention='le90'),
        frame_count=len(records), runtime_forward_counts=forwards,
        checks=checks, target_data_read=False, fixed_test_read=False,
        passed=passed,
        eligible_for_cv_cached_input=passed,
        decision=('ALLOW_SEQ11_V2_DINO_CACHE_AS_CV_INPUT' if passed else
                  'STOP_SEQ11_V2_DINO_CACHE_IDENTITY_FAILED'))


def main():
    args = parse_args()
    report = audit(args)
    output = _write_exact(args.out_json, report)
    print('[dino-cache-identity] output={}'.format(output))
    print('[dino-cache-identity] decision={}'.format(report['decision']))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
