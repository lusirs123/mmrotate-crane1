#!/usr/bin/env python3
"""Pre-training support check for the held-out seq11 mechanism block."""

import argparse
import json
import os

from crane_project.tools.symeood_dino_seq11_aux_mechanism_gate import (
    _annotations, _box, _load_results, _read_aux_audit, _sha256)
from crane_project.tools.symeood_dino_seq11_block_split import _manifest
from crane_project.tools.eval_crane_offline import compute_riou


PROTOCOL = 'k1_dino_seq11_aux_mechanism_support_audit_v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k1-reference-results', required=True)
    parser.add_argument('--aux-val-audit', required=True)
    parser.add_argument(
        '--split-manifest',
        default=('crane_project/data_contracts/'
                 'real_seq11_pilot_k1p9_blocksplit_v1.json'))
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument(
        '--aux-val-split',
        default='extra_source_real_seq11_pilot_k1p9_val_v1')
    parser.add_argument('--min-hard-support', type=int, default=3)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def audit(args):
    manifest = _manifest(args.split_manifest)
    k1_path, k1 = _load_results(args.k1_reference_results)
    ann_dir, stems, gt = _annotations(
        args.data_root, args.aux_val_split, manifest['val_stems'])
    audit_path, audit_sha, audit_index = _read_aux_audit(
        args.aux_val_audit, manifest['val_stems'], manifest['sha256'])
    rows = []
    for stem, gt_box, k1_box in zip(stems, gt, k1):
        dino_box = _box(audit_index[stem].get('dino_native_box'), 'DINO')
        k1_riou = (0.0 if k1_box is None else
                   float(compute_riou(k1_box, gt_box)))
        dino_riou = (0.0 if dino_box is None else
                     float(compute_riou(dino_box, gt_box)))
        rows.append(dict(
            frame_key=stem,
            k1_present=k1_box is not None, k1_riou=k1_riou,
            k1_hit=k1_riou >= 0.5,
            dino_present=dino_box is not None, dino_riou=dino_riou,
            dino_hit=dino_riou >= 0.5,
            is_target_support=(
                k1_box is not None and k1_riou < 0.5
                and dino_riou >= 0.5)))
    support_count = sum(row['is_target_support'] for row in rows)
    passed = support_count >= int(args.min_hard_support)
    return dict(
        protocol=PROTOCOL,
        evidence_boundary='same_video_heldout_auxiliary_support_only',
        target_data_read=False, fixed_test_read=False,
        input=dict(
            k1_reference_results=k1_path,
            k1_reference_results_sha256=_sha256(k1_path),
            aux_val_audit=audit_path, aux_val_audit_sha256=audit_sha,
            split_manifest=manifest['path'],
            split_manifest_sha256=manifest['sha256'], ann_dir=ann_dir,
            frame_count=len(rows)),
        minimum_hard_support=int(args.min_hard_support),
        k1_present_wrong_dino_hit_count=support_count,
        checks=dict(
            heldout_frame_count_11=len(rows) == 11,
            target_support_sufficient=passed,
            temporal_metrics_not_computed=True),
        rows=rows,
        passed=passed,
        eligible_for_blocksplit_training=passed,
        eligible_for_router_claim=False,
        eligible_for_independent_sequence_claim=False,
        decision=(
            'ALLOW_E1_SEQ11_BLOCKSPLIT_TRAINING'
            if passed else 'STOP_SEQ11_AUX_SUPPORT_INSUFFICIENT'))


def main():
    args = parse_args()
    report = audit(args)
    output = os.path.abspath(os.fspath(args.out_json))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print('[seq11-support] target_support={}'.format(
        report['k1_present_wrong_dino_hit_count']))
    print('[seq11-support] decision={}'.format(report['decision']))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
