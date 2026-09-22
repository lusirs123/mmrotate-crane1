"""CPU-only completeness/identity check for DINO feature distillation cache."""

import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import torch


PROTOCOL = 'dino_feature_cache_preflight_v1'
FRAME_RE = re.compile(r'^(?:real|sim)_seq\d+_\d{5}$')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def parse_dataset_specs(values):
    specs = []
    for value in values:
        parts = str(value).split(':')
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                'datasets must use annotation_split:image_split')
        specs.append((parts[0], parts[1]))
    return specs


def image_files(data_root, datasets):
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    rows = []
    for annotation_split, image_split in parse_dataset_specs(datasets):
        annotations = sorted(glob.glob(os.path.join(
            data_root, annotation_split, 'annfiles', '*.txt')))
        for annotation in annotations:
            stem = Path(annotation).stem
            if not FRAME_RE.match(stem):
                continue
            matches = [os.path.join(
                data_root, image_split, 'images', stem + extension)
                for extension in extensions]
            matches = [path for path in matches if os.path.isfile(path)]
            if len(matches) == 1:
                rows.append((annotation_split, os.path.abspath(matches[0])))
    return rows


def inspect_cache(data_root, cache_dir, datasets, expected_channels=1024,
                  expected_model='dinov2_vitl14'):
    records = []
    valid = 0
    annotation_splits = [item[0] for item in parse_dataset_specs(datasets)]
    split_counts = {str(split): 0 for split in annotation_splits}
    for split, image in image_files(data_root, datasets):
        split_counts[split] += 1
        stem = Path(image).stem
        candidates = sorted(glob.glob(os.path.join(
            cache_dir, split, stem + '_*.pth')))
        matched = []
        stat = os.stat(image)
        for path in candidates:
            payload = torch_load(path)
            signature = payload.get('signature') or {}
            identity = signature.get('image') or {}
            feature = payload.get('feature')
            okay = (
                payload.get('frozen_dinov2') is True
                and 'paired_view' not in signature
                and signature.get('dinov2_model') == expected_model
                and int(identity.get('size', -1)) == int(stat.st_size)
                and int(identity.get('mtime_ns', -1)) == int(
                    getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9)))
                and isinstance(feature, torch.Tensor)
                and feature.ndim == 4 and feature.size(0) == 1
                and feature.size(1) == int(expected_channels)
                and bool(torch.isfinite(feature.float()).all().item()))
            if okay:
                matched.append(path)
        status = 'valid' if len(matched) == 1 else (
            'missing' if not matched else 'ambiguous')
        valid += int(status == 'valid')
        records.append(dict(split=split, image=image, status=status,
                            matching_cache_files=matched))
    missing_splits = sorted(
        split for split, count in split_counts.items() if count == 0)
    return dict(
        protocol=PROTOCOL,
        data_root=os.path.abspath(data_root),
        cache_dir=os.path.abspath(cache_dir),
        datasets=list(datasets),
        expected_channels=int(expected_channels),
        expected_model=str(expected_model),
        images_by_split=split_counts, missing_splits=missing_splits,
        image_count=len(records), valid_count=valid,
        missing_or_ambiguous_count=len(records) - valid,
        complete=(bool(records) and not missing_splits
                  and valid == len(records)), records=records)


def write_exact(path, payload):
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding='utf-8') != encoded:
        raise RuntimeError('Refusing to overwrite different output: ' + str(path))
    path.write_text(encoded, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--datasets', nargs='+',
                        default=['train:train', 'train_sim:train'])
    parser.add_argument('--expected-channels', type=int, default=1024)
    parser.add_argument('--expected-model', default='dinov2_vitl14')
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    report = inspect_cache(
        args.data_root, args.cache_dir, args.datasets,
        args.expected_channels, args.expected_model)
    write_exact(args.out_json, report)
    print(json.dumps(dict(
        decision=('CACHE_COMPLETE' if report['complete'] else
                  'CACHE_INCOMPLETE_DO_NOT_TRAIN'),
        image_count=report['image_count'], valid_count=report['valid_count'],
        out_json=os.path.abspath(args.out_json),
        out_json_sha256=sha256(args.out_json)), indent=2))
    if not report['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
