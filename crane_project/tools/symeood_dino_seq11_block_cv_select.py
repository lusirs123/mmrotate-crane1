#!/usr/bin/env python3
"""Select an epoch only after the complete three-fold source CV gate."""

import argparse
import hashlib
import json
import os


GATE_PROTOCOL = 'k1_retentive_seq11_three_window_block_cv_gate_v1'
PROTOCOL = 'k1_retentive_seq11_three_window_block_cv_selection_v1'
SELECTION_POLICY = (
    'passed_then_present_wrong_rescue_missing_rescue_oof_riou_legacy_riou_'
    'legacy_dfr_earliest_v1')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gates', nargs='+', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _read(path):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        raw = handle.read()
    payload = json.loads(raw.decode('utf-8'))
    required = dict(
        protocol=GATE_PROTOCOL,
        evidence_boundary=(
            'three_legacy_source_val_streams_plus_pooled_sparse_oof_33'),
        target_data_read=False, fixed_test_read=False,
        temporal_metrics_computed_on_oof=False,
        eligible_for_checkpoint_promotion=False,
        eligible_for_fixed_test=False,
        eligible_for_unknown_sequence_claim=False)
    failures = [
        '{}={!r}'.format(key, payload.get(key))
        for key, expected in required.items()
        if payload.get(key) != expected]
    if failures:
        raise RuntimeError('Invalid CV gate {}: {}'.format(
            absolute, '; '.join(failures)))
    return dict(
        path=absolute, sha256=hashlib.sha256(raw).hexdigest(),
        epoch=int(payload['epoch']), passed=payload.get('passed') is True,
        payload=payload)


def _rank(row):
    payload = row['payload']
    oof = payload['pooled_oof']
    folds = payload['folds']
    legacy_riou = sum(
        float(fold['source_val_metrics']['real/mean_RIoU'])
        + float(fold['source_val_metrics']['sim/mean_RIoU'])
        for fold in folds) / len(folds)
    legacy_dfr = sum(
        float(fold['source_val_metrics']['real/DFR(%/frame)'])
        + float(fold['source_val_metrics']['sim/DFR(%/frame)'])
        for fold in folds) / len(folds)
    return (
        -float(oof['present_wrong_rescue_rate']),
        -float(oof['missing_rescue_rate']),
        -float(oof['candidate_mean_riou']),
        -legacy_riou,
        legacy_dfr,
        int(row['epoch']))


def select(paths):
    if len(paths) != 10:
        raise RuntimeError('CV selection requires exactly ten epoch gates')
    rows = [_read(path) for path in paths]
    by_epoch = {row['epoch']: row for row in rows}
    if set(by_epoch) != set(range(1, 11)) or len(by_epoch) != len(rows):
        raise RuntimeError('CV gates must cover epochs 1..10 exactly')
    manifest_hashes = {
        row['payload']['input']['cv_manifest_sha256'] for row in rows}
    support_hashes = {
        row['payload']['input']['support_audit_sha256'] for row in rows}
    if len(manifest_hashes) != 1 or len(support_hashes) != 1:
        raise RuntimeError('All CV gates must use identical evidence inputs')
    eligible = [row for row in rows if row['passed']]
    selected = None if not eligible else min(eligible, key=_rank)
    diagnostic = sorted(
        rows, key=lambda row: (-int(row['passed']), _rank(row)))
    return dict(
        protocol=PROTOCOL,
        evidence_boundary='three_window_block_cv_source_only',
        target_data_read=False, fixed_test_read=False,
        selection_policy=SELECTION_POLICY,
        evaluated_epochs=list(range(1, 11)),
        passing_epochs=[row['epoch'] for row in eligible],
        diagnostic_epoch_order_not_promotion=[
            dict(epoch=row['epoch'], passed=row['passed'], rank=list(
                _rank(row))) for row in diagnostic],
        selected=(None if selected is None else dict(
            epoch=selected['epoch'], gate=selected['path'],
            gate_sha256=selected['sha256'],
            fold_checkpoints=[dict(
                fold_id=fold['fold_id'],
                checkpoint=fold['input']['checkpoint'],
                checkpoint_sha256=fold['input']['checkpoint_sha256'])
                for fold in selected['payload']['folds']])),
        passed=selected is not None,
        eligible_for_final_all59_source_training=selected is not None,
        eligible_for_checkpoint_promotion=False,
        eligible_for_fixed_test=False,
        eligible_for_unknown_sequence_claim=False,
        decision=(
            'SELECT_SEQ11_BLOCK_CV_EPOCH_FOR_FINAL_SOURCE_TRAINING'
            if selected is not None else
            'STOP_SEQ11_NO_EPOCH_PASSED_THREE_WINDOW_BLOCK_CV'))


def main():
    args = parse_args()
    report = select(args.gates)
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
