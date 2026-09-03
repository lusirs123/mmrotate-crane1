#!/usr/bin/env python3
"""Select one E1 checkpoint only after both source evidence halves pass."""

import argparse
import hashlib
import json
import os
import re


LEGACY_PROTOCOL = 'k1_retentive_seq11_blocksplit_legacy_source_gate_v2'
AUX_PROTOCOL = 'k1_retentive_seq11_aux_mechanism_gate_v2'
PROTOCOL = 'k1_retentive_seq11_dual_source_selection_v2'
SELECTION_POLICY = (
    'both_pass_then_aux_rescue_aux_riou_legacy_mcml_riou_dfr_earliest_v1')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-gates', nargs='+', required=True)
    parser.add_argument('--aux-gates', nargs='+', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _epoch(checkpoint):
    match = re.search(r'(?:^|[/\\])epoch_(\d+)\.pth$', checkpoint)
    if match is None:
        raise RuntimeError('Checkpoint path has no epoch_<N>.pth')
    return int(match.group(1))


def _read(path, protocol, boundary):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        raw = handle.read()
    payload = json.loads(raw.decode('utf-8'))
    failures = []
    required = dict(
        protocol=protocol, evidence_boundary=boundary,
        target_data_read=False, fixed_test_read=False,
        eligible_for_checkpoint_promotion=False,
        eligible_for_fixed_test=False,
        eligible_for_unknown_sequence_claim=False)
    for key, expected in required.items():
        if payload.get(key) != expected:
            failures.append('{}={!r}'.format(key, payload.get(key)))
    inputs = dict(payload.get('input') or {})
    checkpoint = os.fspath(inputs.get('candidate_checkpoint', ''))
    checkpoint_sha = inputs.get('candidate_checkpoint_sha256')
    if not checkpoint_sha:
        failures.append('candidate checkpoint SHA256 absent')
    if failures:
        raise RuntimeError('Invalid dual-gate input {}: {}'.format(
            absolute, '; '.join(failures)))
    return dict(
        path=absolute, sha256=_sha256(raw), epoch=_epoch(checkpoint),
        checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha,
        passed=payload.get('passed') is True, payload=payload)


def _rank(row):
    legacy_metrics = row['legacy']['payload']['candidate_metrics']
    aux_metrics = row['aux']['payload']['metrics']
    return (
        -int(aux_metrics['hard_rescued_hit_count']),
        -float(aux_metrics['hard_mean_riou_gain']),
        max(int(legacy_metrics['real/MCML_max(frames)']),
            int(legacy_metrics['sim/MCML_max(frames)'])),
        -(float(legacy_metrics['real/mean_RIoU'])
          + float(legacy_metrics['sim/mean_RIoU'])),
        float(legacy_metrics['real/DFR(%/frame)'])
        + float(legacy_metrics['sim/DFR(%/frame)']),
        int(row['epoch']))


def select(legacy_paths, aux_paths):
    if len(legacy_paths) != 10 or len(aux_paths) != 10:
        raise RuntimeError(
            'Dual selection requires ten gates per evidence half')
    legacy = [_read(
        path, LEGACY_PROTOCOL, 'legacy_source_val_738_only')
        for path in legacy_paths]
    aux = [_read(
        path, AUX_PROTOCOL, 'same_video_heldout_auxiliary_block_only')
        for path in aux_paths]
    legacy_by_epoch = {row['epoch']: row for row in legacy}
    aux_by_epoch = {row['epoch']: row for row in aux}
    expected = set(range(1, 11))
    if set(legacy_by_epoch) != expected or set(aux_by_epoch) != expected:
        raise RuntimeError('Dual gates must each cover epochs 1..10 exactly')
    rows = []
    for epoch in range(1, 11):
        left, right = legacy_by_epoch[epoch], aux_by_epoch[epoch]
        if (left['checkpoint'] != right['checkpoint']
                or left['checkpoint_sha256'] != right['checkpoint_sha256']):
            raise RuntimeError('Gate checkpoint mismatch at epoch {}'.format(
                epoch))
        rows.append(dict(epoch=epoch, legacy=left, aux=right,
                         passed=left['passed'] and right['passed']))
    eligible = [row for row in rows if row['passed']]
    diagnostic_order = sorted(
        rows,
        key=lambda row: (
            -int(row['legacy']['passed']) - int(row['aux']['passed']),
            _rank(row)))
    selected = None if not eligible else min(eligible, key=_rank)
    report = dict(
        protocol=PROTOCOL,
        evidence_boundary=(
            'legacy_source_val_plus_same_video_heldout_auxiliary_block'),
        target_data_read=False,
        fixed_test_read=False,
        selection_policy=SELECTION_POLICY,
        evaluated_epochs=list(range(1, 11)),
        passing_epochs=[row['epoch'] for row in eligible],
        diagnostic_epoch_order_not_promotion=[
            dict(
                epoch=row['epoch'],
                legacy_passed=row['legacy']['passed'],
                aux_passed=row['aux']['passed'])
            for row in diagnostic_order],
        ranking=[dict(
            epoch=row['epoch'], rank=list(_rank(row)),
            legacy_gate=row['legacy']['path'],
            legacy_gate_sha256=row['legacy']['sha256'],
            aux_gate=row['aux']['path'],
            aux_gate_sha256=row['aux']['sha256'])
            for row in sorted(eligible, key=_rank)],
        selected=(None if selected is None else dict(
            epoch=selected['epoch'],
            checkpoint=selected['legacy']['checkpoint'],
            checkpoint_sha256=selected['legacy']['checkpoint_sha256'],
            legacy_gate=selected['legacy']['path'],
            legacy_gate_sha256=selected['legacy']['sha256'],
            aux_gate=selected['aux']['path'],
            aux_gate_sha256=selected['aux']['sha256'])),
        passed=selected is not None,
        eligible_for_checkpoint_promotion=selected is not None,
        eligible_for_fixed_test=False,
        eligible_for_unknown_sequence_claim=False,
        decision=(
            'SELECT_E1_DUAL_SOURCE_CHECKPOINT_FOR_PROMOTION_ONLY'
            if selected is not None else
            'STOP_E1_NO_EPOCH_PASSED_BOTH_SOURCE_HALVES'))
    return report


def main():
    args = parse_args()
    report = select(args.legacy_gates, args.aux_gates)
    output = os.path.abspath(os.fspath(args.out_json))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
