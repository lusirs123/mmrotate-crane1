#!/usr/bin/env python3
"""Snapshot only current TRAIN legacy axes, without changing labels/images.

The original folder has other sequence roles. Never glob/load those JSONs.
Current TRAIN image identity is recorded; original source JPGs are absent in
the recovered JSON folder, so independent original-image equality is unknown.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.check_port_reliability_readiness_v1 import (
    DATA, LEGACY_TRAIN, NATIVE_K, TRAIN_COUNTS, axis_consistency,
    files, polygon_box, read_axis, sha, write_new)


def snapshot(source, data_root):
    destination = data_root/'provenance/axis_legacy_train_v1'
    if destination.exists():
        raise FileExistsError('Refuse to overwrite legacy snapshot: ' + str(destination))
    bindings, sources = {}, {}
    # Finish every binding check before creating an output directory.
    for sequence in LEGACY_TRAIN:
        annotations = [p for p in files(data_root/'train/annfiles', '.txt')
                       if p.stem.rsplit('_',1)[0] == sequence]
        if len(annotations) != TRAIN_COUNTS[sequence]:
            raise ValueError('Unexpected current TRAIN count: ' + sequence)
        for annotation in annotations:
            name = annotation.stem
            original = source/(name+'.json')
            image = data_root/'train/images'/(name+'.jpg')
            with Image.open(image) as im:
                size = im.size
            axis, flags = read_axis(original, name, size, legacy=True)
            _, box = polygon_box(annotation)
            consistency = axis_consistency(axis, box, NATIVE_K[sequence])
            # Mismatches are review evidence, not an excuse to relabel the OBB.
            bindings[name] = dict(sequence=sequence, split='train', original_json=str(original.resolve()),
                axis_json_sha256=sha(original), image_sha256=sha(image), annotation_sha256=sha(annotation),
                original_source_image_equality='UNVERIFIED_ORIGINAL_JPG_UNAVAILABLE',
                image_binding='allowlisted current TRAIN stem, legacy imagePath alias, exact dimensions, numerical axis/OBB comparison',
                flags=flags, conversion_k=NATIVE_K[sequence], consistency=consistency)
            sources[name] = original
    manifest = dict(protocol='port_legacy_train_axis_snapshot_v1', sequences=list(LEGACY_TRAIN),
                    evidence_role='recovered_native_train_axis_labels_not_independent_gt', rows=bindings,
                    note='Byte-preserved source JSON; no TEST/VAL JSON opened; no annotations or images modified.')
    # Validate serializability before making the snapshot.
    json.dumps(manifest, allow_nan=False)
    destination.mkdir(parents=True, exist_ok=False)
    for name, original in sources.items():
        target = destination/bindings[name]['sequence']/'axis_json'/original.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as dst, original.open('rb') as src:
            shutil.copyfileobj(src, dst)
        if sha(target) != bindings[name]['axis_json_sha256']:
            raise ValueError('Snapshot byte copy mismatch: ' + name)
    write_new(destination/'manifest.json', manifest)
    print('Saved', destination, 'TRAIN axes', len(bindings), 'conversion consistent',
          sum(r['consistency']['conversion_consistent'] for r in bindings.values()))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-axis-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=DATA)
    args = parser.parse_args(argv)
    snapshot(args.source_axis_dir.resolve(), args.data_root.resolve())


if __name__ == '__main__':
    main()
