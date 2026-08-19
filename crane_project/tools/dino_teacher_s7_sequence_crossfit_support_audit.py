#!/usr/bin/env python3
"""Run protocol-32: source-only sequence cross-fit support feasibility audit.

This is a JSON-only follow-up to protocol-31.  It never opens images, DINO,
checkpoints, target data, or annotations.  Its purpose is to determine whether
the source sequences already measured by protocol-31 can support a future
sequence-isolated quality-ranker experiment without moving validation frames
into training under the old split.
"""

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from typing import Dict, Iterable, List


AUDIT_NAME = 'DINO S7 Source Sequence Cross-fit Support Audit'
PROTOCOL_VERSION = 32


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument('--paired-view-result-json', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--min-train-gain-frames', type=int, default=16)
    parser.add_argument('--min-train-gain-sequences', type=int, default=2)
    parser.add_argument('--min-heldout-gain-frames', type=int, default=7)
    parser.add_argument('--min-valid-folds', type=int, default=2)
    parser.add_argument('--min-heldout-real-hard-sequences', type=int,
                        default=1)
    parser.add_argument('--min-heldout-sim-hard-sequences', type=int,
                        default=1)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def validate_args(args):
    if args.seed != 0:
        raise ValueError('Sequence cross-fit audit requires --seed 0')
    if not os.path.isfile(args.paired_view_result_json):
        raise ValueError('paired_view_result_json does not exist: {}'.format(
            args.paired_view_result_json))
    if os.path.exists(args.out_json):
        raise ValueError('Refusing to overwrite result: {}'.format(args.out_json))
    for name in (
            'min_train_gain_frames', 'min_train_gain_sequences',
            'min_heldout_gain_frames', 'min_valid_folds',
            'min_heldout_real_hard_sequences',
            'min_heldout_sim_hard_sequences'):
        if int(getattr(args, name)) <= 0:
            raise ValueError('--{} must be positive'.format(name.replace('_', '-')))


def _atomic_write_json(path: str, payload: Dict):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix='.sequence_crossfit.', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2,
                      sort_keys=True)
            stream.write('\n')
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _input_identity(path: str) -> Dict:
    raw = open(path, 'rb').read()
    stat = os.stat(path)
    return dict(
        path=os.path.abspath(path), size=int(stat.st_size),
        sha256=hashlib.sha256(raw).hexdigest())


def load_protocol31(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as stream:
        payload = json.load(stream)
    protocol = payload.get('protocol') or {}
    isolation = payload.get('isolation') or {}
    clean_gate = (payload.get('source') or {}).get(
        'clean_native_reproduction_gate') or {}
    checks = dict(
        protocol_version=int(payload.get('protocol_version', -1)) == 31,
        target_not_read=(payload.get('target_dev') is None
                         and protocol.get('target_read') is False),
        read_only=(payload.get('parameter_update_count') == 0
                   and isolation.get('parameter_updates_performed') is False),
        clean_reproduction=clean_gate.get('passed') is True,
        full_image_views=(protocol.get('full_dino_rpn_roi_forward_per_view')
                          is True
                          and protocol.get('no_feature_tensor_augmentation')
                          is True),
        has_train_rows=isinstance(payload.get('train_frame_rows'), list),
        has_validation_rows=isinstance(
            payload.get('validation_clean_frame_rows'), list))
    if not all(checks.values()):
        raise ValueError('Protocol-31 input failed: {}'.format(', '.join(
            sorted(name for name, passed in checks.items() if not passed))) )
    return payload


def _natural_gain(row: Dict) -> bool:
    clean = (row.get('views') or {}).get('clean') or {}
    return bool(clean.get('eligible')) and not bool(
        clean.get('native_correct')) and int(clean.get('s7_correct_count', 0)) > 0


def _native_correct(row: Dict) -> bool:
    clean = (row.get('views') or {}).get('clean') or {}
    return bool(clean.get('eligible')) and bool(clean.get('native_correct'))


def _sequence_rows(payload: Dict) -> Dict[str, List[Dict]]:
    source = payload.get('source') or {}
    combined = []
    for partition, rows in (
            ('protocol31_train', payload.get('train_frame_rows') or []),
            ('protocol31_validation_clean',
             payload.get('validation_clean_frame_rows') or [])):
        for row in rows:
            required = ('split', 'seq', 'domain', 'frame', 'views')
            if any(name not in row for name in required):
                raise ValueError('Protocol-31 frame row lacks required fields')
            clone = dict(row)
            clone['protocol31_partition'] = partition
            combined.append(clone)
    by_sequence = {}
    for row in combined:
        key = str(row['seq'])
        by_sequence.setdefault(key, []).append(row)
    if len(by_sequence) < 3:
        raise ValueError('Cross-fit audit requires at least three sequences')
    for sequence, rows in by_sequence.items():
        domains = {str(row['domain']) for row in rows}
        if len(domains) != 1:
            raise ValueError('Sequence {} has mixed domain labels'.format(sequence))
        frames = [(str(row['split']), int(row['frame'])) for row in rows]
        if len(frames) != len(set(frames)):
            raise ValueError('Sequence {} has duplicate frame keys'.format(sequence))
    return by_sequence


def summarize_sequences(rows_by_sequence: Dict[str, List[Dict]]) -> Dict:
    result = {}
    for sequence, rows in sorted(rows_by_sequence.items()):
        gain_rows = [row for row in rows if _natural_gain(row)]
        result[sequence] = dict(
            domain=str(rows[0]['domain']),
            frame_count=int(len(rows)),
            protocol31_partitions=sorted(set(
                str(row['protocol31_partition']) for row in rows)),
            native_top1_hits=int(sum(_native_correct(row) for row in rows)),
            native_wrong_s7_correct_frame_count=int(len(gain_rows)),
            native_wrong_s7_correct_frame_keys=[
                str(row.get('frame_key', '{}|{}|{}'.format(
                    row['split'], row['seq'], int(row['frame']))))
                for row in gain_rows])
    return result


def _aggregate_gain(rows: Iterable[Dict]) -> Dict:
    gain_rows = [row for row in rows if _natural_gain(row)]
    sequences = Counter(str(row['seq']) for row in gain_rows)
    domains = Counter(str(row['domain']) for row in gain_rows)
    return dict(
        frame_count=int(len(gain_rows)),
        domains=sorted(domains), sequences=sorted(sequences),
        by_domain=dict(sorted((key, int(value)) for key, value in domains.items())),
        by_sequence=dict(sorted((key, int(value)) for key, value in sequences.items())))


def build_crossfit_folds(rows_by_sequence: Dict[str, List[Dict]], args) -> List[Dict]:
    sequences = sorted(rows_by_sequence)
    folds = []
    for heldout in sequences:
        train_sequences = [sequence for sequence in sequences
                           if sequence != heldout]
        train_rows = [row for sequence in train_sequences
                      for row in rows_by_sequence[sequence]]
        heldout_rows = list(rows_by_sequence[heldout])
        train_gain = _aggregate_gain(train_rows)
        heldout_gain = _aggregate_gain(heldout_rows)
        checks = dict(
            train_minimum_gain_frames=(
                int(train_gain['frame_count']) >= int(
                    args.min_train_gain_frames)),
            train_real_and_sim_gain=(set(train_gain['domains']) == {'real', 'sim'}),
            train_minimum_gain_sequences=(len(train_gain['sequences']) >= int(
                args.min_train_gain_sequences)),
            heldout_is_hard=(int(heldout_gain['frame_count']) >= int(
                args.min_heldout_gain_frames)))
        folds.append(dict(
            heldout_sequence=heldout,
            heldout_domain=str(heldout_rows[0]['domain']),
            train_sequences=train_sequences,
            train_natural_gain=train_gain,
            heldout_natural_gain=heldout_gain,
            viable_for_future_crossfit_training=bool(all(checks.values())),
            checks=checks))
    return folds


def build_audit(payload: Dict, args, input_identity: Dict) -> Dict:
    rows_by_sequence = _sequence_rows(payload)
    inventory = summarize_sequences(rows_by_sequence)
    folds = build_crossfit_folds(rows_by_sequence, args)
    viable = [fold for fold in folds if fold['viable_for_future_crossfit_training']]
    viable_real = sorted(
        fold['heldout_sequence'] for fold in viable
        if fold['heldout_domain'] == 'real')
    viable_sim = sorted(
        fold['heldout_sequence'] for fold in viable
        if fold['heldout_domain'] == 'sim')
    checks = dict(
        minimum_valid_folds=len(viable) >= int(args.min_valid_folds),
        heldout_real_hard_sequence_coverage=(len(viable_real) >= int(
            args.min_heldout_real_hard_sequences)),
        heldout_sim_hard_sequence_coverage=(len(viable_sim) >= int(
            args.min_heldout_sim_hard_sequences)))
    passed = bool(all(checks.values()))
    decision = (
        'SOURCE_ONLY_SEQUENCE_CROSSFIT_SUPPORT_PASS_TARGET_NOT_READ'
        if passed else
        'SOURCE_ONLY_SEQUENCE_CROSSFIT_SUPPORT_INSUFFICIENT_TARGET_NOT_READ')
    return dict(
        protocol_version=PROTOCOL_VERSION,
        audit_name='Source-only Sequence Cross-fit Candidate Support Audit',
        protocol=dict(
            source_only=True, target_read=False, read_only_evaluation=True,
            parameter_update=False, dino_forward_count=0,
            input='protocol-31 frame-level clean-view evidence only',
            sequence_isolation='leave-one-complete-sequence-out',
            no_frame_random_split=True, no_target_threshold_tuning=True,
            no_checkpoint_selection=True,
            training_authorization=(
                'A later source-only candidate-quality training design may be '
                'implemented only if a real and a sim hard held-out sequence '
                'both satisfy this cross-fit support gate.')),
        isolation=dict(
            input_target_read=False, parameter_updates_performed=False,
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_used_for_threshold_tuning=False),
        input_protocol31=input_identity,
        source=dict(
            sequence_inventory=inventory,
            crossfit_folds=folds,
            support_gate=dict(
                passed=passed, checks=checks,
                min_train_gain_frames=int(args.min_train_gain_frames),
                min_train_gain_sequences=int(args.min_train_gain_sequences),
                min_heldout_gain_frames=int(args.min_heldout_gain_frames),
                min_valid_folds=int(args.min_valid_folds),
                min_heldout_real_hard_sequences=int(
                    args.min_heldout_real_hard_sequences),
                min_heldout_sim_hard_sequences=int(
                    args.min_heldout_sim_hard_sequences),
                viable_fold_count=int(len(viable)),
                viable_heldout_real_sequences=viable_real,
                viable_heldout_sim_sequences=viable_sim)),
        candidate_forward_count=0,
        parameter_update_count=0,
        target_dev=None,
        eligible_for_training=passed,
        eligible_for_deployment=False,
        eligible_for_full_test=False,
        decision=decision)


def main():
    args = parse_args()
    validate_args(args)
    payload = load_protocol31(args.paired_view_result_json)
    result = build_audit(
        payload, args, _input_identity(args.paired_view_result_json))
    _atomic_write_json(args.out_json, result)
    print('[sequence-crossfit] {}'.format(result['decision']))
    print('[sequence-crossfit] training_allowed={}'.format(
        result['eligible_for_training']))


if __name__ == '__main__':
    main()
