#!/usr/bin/env python3
"""Bind formal ordinary-K1 results to the exact seq11-v2 frame order."""

import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path


PROTOCOL = 'formal_k1_seq11_v2_full251_identity_v1'
FRAME_ORDER_PROTOCOL = 'seq11_v2_full251_frame_order_manifest_v1'
EXPECTED_CONFIG = 'crane_symeood_k1_seq11_v2_full251_eval.py'
EXPECTED_CHECKPOINT = 'epoch_24.pth'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--base-k1-config', required=True)
    parser.add_argument('--evaluation-summary', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument(
        '--source-split',
        default='extra_source_real_seq11_pilot_k1p9_v2')
    parser.add_argument('--frame-order-json', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exact_json(path, payload):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, ensure_ascii=False) + '\n').encode(
        'utf-8')
    if target.exists() and target.read_bytes() != raw:
        raise RuntimeError('Refusing to overwrite different identity: '
                           + os.fspath(target))
    if not target.exists():
        target.write_bytes(raw)
    return os.fspath(target), hashlib.sha256(raw).hexdigest()


def _visible_index(folder, suffixes):
    root = Path(folder).resolve()
    if not root.is_dir():
        raise RuntimeError('Missing seq11-v2 directory: ' + os.fspath(root))
    result = {}
    for path in sorted(root.iterdir()):
        if path.name.startswith('._'):
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem in result:
                raise RuntimeError('Duplicate visible stem: ' + path.stem)
            result[path.stem] = path
    return result


def _result_shape(detections):
    shape = getattr(detections, 'shape', None)
    if shape is not None:
        return tuple(int(value) for value in shape)
    rows = list(detections)
    if not rows:
        return (0, 6)
    widths = {len(row) for row in rows}
    return (len(rows), next(iter(widths))) if len(widths) == 1 else (-1, -1)


def _validate_results(path, expected_count):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise RuntimeError('Formal K1 results must contain exactly 251 frames')
    missing = 0
    for index, result in enumerate(payload):
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise RuntimeError(
                'Formal K1 result {} must contain exactly one class'.format(
                    index))
        shape = _result_shape(result[0])
        if len(shape) != 2 or shape[1] != 6 or shape[0] not in (0, 1):
            raise RuntimeError(
                'Formal K1 frame {} must contain at most one OBB'.format(
                    index))
        missing += int(shape[0] == 0)
    return absolute, missing


def audit(args):
    config = Path(args.config).resolve()
    base_k1_config = Path(args.base_k1_config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    source_manifest = Path(args.source_manifest).resolve()
    evaluation_summary = Path(args.evaluation_summary).resolve()
    if config.name != EXPECTED_CONFIG:
        raise RuntimeError('Unexpected formal K1 config: ' + config.name)
    if checkpoint.name != EXPECTED_CHECKPOINT:
        raise RuntimeError('Formal K1 checkpoint must be epoch_24.pth')
    if base_k1_config.name != 'crane_symeood_k1.py':
        raise RuntimeError('Unexpected base K1 config: ' + base_k1_config.name)
    for required in (
            config, base_k1_config, checkpoint, evaluation_summary,
            source_manifest):
        if not required.is_file():
            raise RuntimeError('Missing identity input: ' + os.fspath(required))
    config_text = config.read_text(encoding='utf-8')
    required_config_tokens = (
        "expected_checkpoint = 'work_dirs/crane_symeood_k1/epoch_24.pth'",
        "expected_frame_count=251",
        "prediction_coordinate_system='original_image_pixels'",
        "obb_convention='le90'",
        'target_data_read=False',
        'fixed_test_read=False')
    if not all(token in config_text for token in required_config_tokens):
        raise RuntimeError('Formal K1 config contract is incomplete')
    if "ann_file='test/" in config_text or "expected_split='test'" in config_text:
        raise RuntimeError('Formal K1 config references fixed TEST')

    source_root = Path(args.data_root).resolve() / args.source_split
    images = _visible_index(source_root / 'images', {'.jpg', '.jpeg', '.png'})
    annotations = _visible_index(source_root / 'annfiles', {'.txt'})
    if set(images) != set(annotations) or len(images) != 251:
        raise RuntimeError('Formal K1 frame set must contain 251 image/label pairs')
    stems = sorted(images)
    pattern = re.compile(r'^real_seq11_(\d{6})$')
    if not all(pattern.match(stem) for stem in stems):
        raise RuntimeError('Unexpected seq11-v2 frame naming')
    results_path, missing_count = _validate_results(args.results, len(stems))
    with open(evaluation_summary, 'r', encoding='utf-8') as handle:
        evaluation_rows = json.load(handle)
    if not isinstance(evaluation_rows, list) or not evaluation_rows:
        raise RuntimeError('Formal K1 evaluation summary is empty')
    checkpoint_sha = _sha256_file(checkpoint)
    matching_evaluations = [
        row for row in evaluation_rows
        if row.get('checkpoint_sha256') == checkpoint_sha
        and Path(row.get('config', '')).name == EXPECTED_CONFIG]
    if len(matching_evaluations) != 1:
        raise RuntimeError(
            'Evaluation summary must bind exactly one formal K1 run')

    frame_rows = []
    for index, stem in enumerate(stems):
        frame_rows.append(dict(
            result_index=index,
            frame_key=stem,
            frame=int(pattern.match(stem).group(1)),
            image_filename=images[stem].name,
            image_sha256=_sha256_file(images[stem]),
            annotation_filename=annotations[stem].name,
            annotation_sha256=_sha256_file(annotations[stem])))
    frame_manifest = dict(
        protocol=FRAME_ORDER_PROTOCOL,
        source_split=args.source_split,
        frame_count=len(frame_rows),
        ordering='lexicographic_visible_annotation_stem',
        appledouble_sidecars_are_samples=False,
        frames=frame_rows,
        target_data_read=False,
        fixed_test_read=False)
    frame_path, frame_sha = _write_exact_json(
        args.frame_order_json, frame_manifest)
    checks = dict(
        frame_count_251=len(frame_rows) == 251,
        image_annotation_sets_equal=set(images) == set(annotations),
        result_count_matches_frame_order=True,
        evaluation_summary_matches_checkpoint=True,
        ordinary_k1_epoch24=True,
        original_image_coordinate_system=True,
        le90_obb_convention=True,
        fixed_test_not_read=True)
    report = dict(
        protocol=PROTOCOL,
        evidence_boundary='seq11_v2_source_only_formal_k1_identity',
        inputs=dict(
            results=results_path,
            results_sha256=_sha256_file(results_path),
            checkpoint=os.fspath(checkpoint),
            checkpoint_sha256=checkpoint_sha,
            config=os.fspath(config),
            config_sha256=_sha256_file(config),
            base_k1_config=os.fspath(base_k1_config),
            base_k1_config_sha256=_sha256_file(base_k1_config),
            evaluation_summary=os.fspath(evaluation_summary),
            evaluation_summary_sha256=_sha256_file(evaluation_summary),
            source_manifest=os.fspath(source_manifest),
            source_manifest_sha256=_sha256_file(source_manifest),
            frame_order_manifest=frame_path,
            frame_order_manifest_sha256=frame_sha),
        preprocessing=dict(
            resize=(1024, 1024), flip=False,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375], to_rgb=True,
            pad_size=(1024, 1024),
            pad_value=[114.0, 114.0, 114.0]),
        coordinate_contract=dict(
            prediction_coordinate_system='original_image_pixels',
            obb_convention='le90',
            target_geometry='top_beam_only', annotation_k0=1.9),
        frame_count=len(frame_rows),
        missing_prediction_count=missing_count,
        evaluation_metric=matching_evaluations[0].get('metric'),
        checks=checks,
        target_data_read=False,
        fixed_test_read=False,
        passed=all(checks.values()),
        decision='ALLOW_FORMAL_K1_SEQ11_V2_SUPPORT_CLASSIFICATION')
    return report


def main():
    args = parse_args()
    report = audit(args)
    output, _ = _write_exact_json(args.out_json, report)
    print('[formal-k1-identity] output={}'.format(output))
    print('[formal-k1-identity] frames={}'.format(report['frame_count']))
    print('[formal-k1-identity] decision={}'.format(report['decision']))


if __name__ == '__main__':
    main()
