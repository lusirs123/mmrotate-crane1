#!/usr/bin/env python3
"""Single-frame mechanism gate on the held-out 11-frame seq11 block.

This gate verifies whether an E1 checkpoint improves the pre-declared case
"formal K1 is present but geometrically wrong while cached DINO is a hit".
The samples are sparse, so DFR, ACI, MCML, and any independent-sequence claim
are explicitly prohibited here.
"""

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from crane_project.tools.eval_crane_offline import compute_riou, parse_dota_txt
from crane_project.tools.symeood_dino_causal_history_source_gate import (
    _checkpoint_contract)
from crane_project.tools.symeood_dino_seq11_block_split import (
    ALL_LANE_PROTOCOL, _manifest)


PROTOCOL = 'k1_retentive_seq11_aux_mechanism_gate_v2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-results', required=True)
    parser.add_argument('--candidate-checkpoint', required=True)
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
    parser.add_argument('--expected-candidate-sha256')
    parser.add_argument('--min-hard-support', type=int, default=3)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_results(path, expected_count=11):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or len(payload) != int(expected_count):
        raise RuntimeError(
            'Aux result PKL must contain exactly {} frames'.format(
                expected_count))
    boxes = []
    for index, result in enumerate(payload):
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise RuntimeError('Aux frame {} must contain one class'.format(
                index))
        detections = np.asarray(result[0], dtype=np.float64)
        if detections.size == 0:
            boxes.append(None)
            continue
        detections = detections.reshape((-1, 6))
        if detections.shape[0] != 1:
            raise RuntimeError(
                'Aux frame {} must contain at most one OBB'.format(index))
        box = detections[0, :5].copy()
        if not np.isfinite(box).all() or np.any(box[2:4] <= 0.0):
            raise RuntimeError(
                'Invalid auxiliary prediction at index {}'.format(index))
        boxes.append(box)
    return absolute, boxes


def _box(value, name):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if (array.size < 5 or not np.isfinite(array[:5]).all()
            or np.any(array[2:4] <= 0.0)):
        raise RuntimeError('Invalid {} OBB'.format(name))
    return array[:5].copy()


def _read_aux_audit(path, wanted_stems, manifest_sha256):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        raw = handle.read()
    payload = json.loads(raw.decode('utf-8'))
    if payload.get('protocol') != ALL_LANE_PROTOCOL:
        raise RuntimeError('Aux-val audit is not an all-lane audit')
    if payload.get('auxiliary_split_role') != 'aux-val':
        raise RuntimeError('Aux-val audit has the wrong split role')
    if payload.get('auxiliary_split_manifest_sha256') != manifest_sha256:
        raise RuntimeError('Aux-val audit/manifest hash mismatch')
    index = {}
    for record in payload.get('records') or []:
        stem = Path(record.get('filename', '')).stem
        if not stem or stem in index:
            raise RuntimeError('Invalid/duplicate aux-val audit frame')
        if record.get('raw_selected_source') == 'not_computed':
            raise RuntimeError('DINO was not computed for ' + stem)
        if ('dino_invoked' in record
                and record.get('dino_invoked') not in (True, 1)):
            raise RuntimeError('DINO was not computed for ' + stem)
        if 'dino_native_box' not in record:
            raise RuntimeError('Aux-val audit lacks DINO box key for ' + stem)
        index[stem] = record
    if set(index) != set(wanted_stems):
        raise RuntimeError('Aux-val audit does not match held-out manifest')
    return absolute, hashlib.sha256(raw).hexdigest(), index


def _annotations(data_root, split, wanted_stems):
    ann_dir = Path(data_root).resolve() / split / 'annfiles'
    paths = sorted(
        path for path in ann_dir.glob('*.txt')
        if not path.name.startswith('._'))
    stems = [path.stem for path in paths]
    if stems != sorted(wanted_stems):
        raise RuntimeError('Aux-val annotations do not match manifest/order')
    boxes = []
    for path in paths:
        parsed = parse_dota_txt(os.fspath(path))
        if len(parsed) != 1:
            raise RuntimeError('Aux-val frame must have exactly one GT: '
                               + os.fspath(path))
        boxes.append(np.asarray(parsed[0], dtype=np.float64)[:5])
    return os.fspath(ann_dir), stems, boxes


def _mean(values):
    return float(np.mean(values)) if values else None


def audit(args):
    manifest = _manifest(args.split_manifest)
    wanted_stems = manifest['val_stems']
    candidate_path, candidate = _load_results(args.candidate_results)
    k1_path, k1 = _load_results(args.k1_reference_results)
    ann_dir, stems, gt = _annotations(
        args.data_root, args.aux_val_split, wanted_stems)
    audit_path, audit_sha, audit_index = _read_aux_audit(
        args.aux_val_audit, wanted_stems, manifest['sha256'])
    (checkpoint_path, checkpoint_sha, checkpoint_contract,
     _v2, v3, e2, _seq11, blocksplit) = _checkpoint_contract(
        args.candidate_checkpoint, args.expected_candidate_sha256)
    if not v3 or e2 or not blocksplit:
        raise RuntimeError('Aux mechanism gate requires E1 48/11 checkpoint')
    if checkpoint_contract.get(
            'auxiliary_split_manifest_sha256') != manifest['sha256']:
        raise RuntimeError('Checkpoint/manifest hash mismatch')

    rows = []
    for stem, gt_box, candidate_box, k1_box in zip(
            stems, gt, candidate, k1):
        dino_box = _box(audit_index[stem].get('dino_native_box'), 'DINO')
        candidate_riou = (0.0 if candidate_box is None else
                          float(compute_riou(candidate_box, gt_box)))
        k1_riou = (0.0 if k1_box is None else
                   float(compute_riou(k1_box, gt_box)))
        dino_riou = (0.0 if dino_box is None else
                     float(compute_riou(dino_box, gt_box)))
        rows.append(dict(
            frame_key=stem,
            candidate_present=candidate_box is not None,
            candidate_riou=candidate_riou,
            candidate_hit=candidate_riou >= 0.5,
            k1_present=k1_box is not None,
            k1_riou=k1_riou,
            k1_hit=k1_riou >= 0.5,
            dino_present=dino_box is not None,
            dino_riou=dino_riou,
            dino_hit=dino_riou >= 0.5))

    hard = [row for row in rows if (
        row['k1_present'] and not row['k1_hit'] and row['dino_hit'])]
    k1_good = [row for row in rows if row['k1_hit']]
    hard_rescued = sum(row['candidate_hit'] for row in hard)
    hard_lost = sum(not row['candidate_hit'] for row in hard)
    k1_good_lost = sum(not row['candidate_hit'] for row in k1_good)
    all_candidate = [row['candidate_riou'] for row in rows]
    all_k1 = [row['k1_riou'] for row in rows]
    hard_candidate = [row['candidate_riou'] for row in hard]
    hard_k1 = [row['k1_riou'] for row in hard]
    support_sufficient = len(hard) >= int(args.min_hard_support)
    mechanism_checks = dict(
        hard_support_sufficient=support_sufficient,
        hard_net_hit_gain_positive=(hard_rescued > 0),
        hard_mean_riou_improved=(
            bool(hard) and _mean(hard_candidate) > _mean(hard_k1)),
        no_k1_good_hit_lost=(k1_good_lost == 0),
        all_mean_riou_within_0p01=(
            _mean(all_candidate) >= _mean(all_k1) - 0.01),
        candidate_missing_not_worse=(
            sum(not row['candidate_present'] for row in rows)
            <= sum(not row['k1_present'] for row in rows)))
    passed = all(mechanism_checks.values())
    return dict(
        protocol=PROTOCOL,
        metric_protocol_version=2,
        evidence_boundary='same_video_heldout_auxiliary_block_only',
        target_data_read=False,
        fixed_test_read=False,
        temporal_metrics_computed=False,
        temporal_metrics_authorized=False,
        input=dict(
            candidate_results=candidate_path,
            candidate_results_sha256=_sha256(candidate_path),
            k1_reference_results=k1_path,
            k1_reference_results_sha256=_sha256(k1_path),
            candidate_checkpoint=checkpoint_path,
            candidate_checkpoint_sha256=checkpoint_sha,
            candidate_checkpoint_contract=checkpoint_contract,
            aux_val_audit=audit_path,
            aux_val_audit_sha256=audit_sha,
            split_manifest=manifest['path'],
            split_manifest_sha256=manifest['sha256'],
            ann_dir=ann_dir,
            frame_count=len(rows)),
        support=dict(
            minimum_hard_support=int(args.min_hard_support),
            k1_present_wrong_dino_hit_count=len(hard),
            k1_good_count=len(k1_good)),
        metrics=dict(
            all_candidate_mean_riou=_mean(all_candidate),
            all_k1_mean_riou=_mean(all_k1),
            all_mean_riou_gain=_mean(all_candidate) - _mean(all_k1),
            hard_candidate_mean_riou=_mean(hard_candidate),
            hard_k1_mean_riou=_mean(hard_k1),
            hard_mean_riou_gain=(
                None if not hard else
                _mean(hard_candidate) - _mean(hard_k1)),
            hard_rescued_hit_count=hard_rescued,
            hard_unrescued_count=hard_lost,
            k1_good_lost_count=k1_good_lost,
            candidate_missing_count=sum(
                not row['candidate_present'] for row in rows),
            k1_missing_count=sum(not row['k1_present'] for row in rows)),
        checks=mechanism_checks,
        rows=rows,
        passed=passed,
        eligible_for_dual_source_gate=passed,
        eligible_for_checkpoint_promotion=False,
        eligible_for_fixed_test=False,
        eligible_for_unknown_sequence_claim=False,
        decision=(
            'ALLOW_SEQ11_AUX_MECHANISM_HALF' if passed else
            'STOP_SEQ11_AUX_SUPPORT_INSUFFICIENT'
            if not support_sufficient else
            'STOP_SEQ11_AUX_MECHANISM_FAILED'))


def main():
    args = parse_args()
    report = audit(args)
    output = os.path.abspath(os.fspath(args.out_json))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print('[seq11-aux-gate] support={}'.format(
        report['support']['k1_present_wrong_dino_hit_count']))
    print('[seq11-aux-gate] metrics={}'.format(report['metrics']))
    print('[seq11-aux-gate] decision={}'.format(report['decision']))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
