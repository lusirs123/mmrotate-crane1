"""JSON-only causal-history support audit for SymEOOD--DINO OBB streams.

The audit answers one bounded question before any temporal model is built:
when a current-frame prediction is unusable, do the immediately preceding
one to four frames contain useful causal evidence, or are they already bad as
well?

No detector is invoked and no parameter or threshold is fitted.  Ground truth
is used only to measure support and to construct explicitly non-deployable
capacity oracles.  Sequence identifiers are used only to preserve chronology;
they never select a detector, history horizon, or output policy.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from crane_project.tools import (
    symeood_dino_application_domain_v4_audit as base)
from crane_project.tools.eval_crane_offline import (
    METRIC_PROTOCOL_VERSION, angle_diff, compute_riou, parse_seq_frame)


PROTOCOL = 'multi_stream_causal_history_support_audit_v1'
INPUT_PROTOCOL = base.INPUT_PROTOCOL
HISTORY_HORIZONS = (1, 2, 3, 4)
HIT_RIOU = 0.5
STREAM_NAMES = ('sym_eood', 'dino_native', 'v4_diagnostic')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Audit causal one-to-four-frame history support from an existing '
            'unrouted all-lane JSON'))
    parser.add_argument('--audit-json', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument('--split', required=True, choices=['val', 'test'])
    parser.add_argument(
        '--evidence-role', required=True,
        choices=sorted(base.ROLE_FRAME_COUNTS))
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--require-frame-count', type=int)
    return parser.parse_args()


def _copy_box(box):
    if box is None:
        return None
    return np.asarray(box, dtype=np.float64).reshape(-1)[:6].copy()


def _riou(box, gt):
    if box is None:
        return 0.0
    return float(compute_riou(box[:5], gt[:5]))


def _hit(box, gt):
    return bool(_riou(box, gt) >= HIT_RIOU)


def _normalize_le90(angle):
    return float((float(angle) + math.pi / 2.0) % math.pi - math.pi / 2.0)


def _constant_velocity(previous_previous, previous):
    """One-step causal OBB extrapolation in centre/log-size/periodic angle."""
    if previous_previous is None or previous is None:
        return None
    first = _copy_box(previous_previous)
    second = _copy_box(previous)
    result = second.copy()
    result[:2] = 2.0 * second[:2] - first[:2]
    result[2:4] = np.exp(
        2.0 * np.log(second[2:4]) - np.log(first[2:4]))
    delta_angle = float(angle_diff(
        np.asarray([second[4]], dtype=np.float64),
        np.asarray([first[4]], dtype=np.float64))[0])
    result[4] = _normalize_le90(second[4] + delta_angle)
    result[5] = float(second[5])
    if not np.isfinite(result).all() or np.any(result[2:4] <= 0.0):
        return None
    return result


def _best_gt_oracle(candidates, gt):
    """Non-deployable GT oracle over already available causal candidates."""
    valid = [box for box in candidates if box is not None]
    if not valid:
        return None
    return _copy_box(max(valid, key=lambda box: _riou(box, gt)))


def _selected_box(record, sym_key, stream_name):
    sym = base._box(record, sym_key)
    dino = base._box(record, 'dino_native_box')
    if stream_name == 'sym_eood':
        return sym
    if stream_name == 'dino_native':
        return dino
    if stream_name != 'v4_diagnostic':
        raise ValueError('Unknown stream: {}'.format(stream_name))
    domain, _, _ = parse_seq_frame(record['filename'])
    if domain == 'real':
        return dino if dino is not None else sym
    if domain == 'sim':
        return sym
    raise RuntimeError('V4 diagnostic requires real/sim input')


def _rows(records, annotation_dir, sym_key, stream_name):
    rows = []
    for record in records:
        domain, seq_id, frame = parse_seq_frame(record['filename'])
        gt = base._annotation(record, annotation_dir)
        box = _selected_box(record, sym_key, stream_name)
        rows.append(dict(
            record=record,
            domain=domain,
            seq_id=seq_id,
            sequence=str(record['sequence']),
            frame=int(frame),
            frame_key='{}|{}'.format(record['sequence'], int(frame)),
            gt=gt,
            box=box,
            riou=_riou(box, gt),
            hit=_hit(box, gt)))
    return sorted(
        rows, key=lambda row: (
            row['domain'], row['seq_id'], int(row['frame'])))


def _history_index(rows):
    return {
        (row['sequence'], int(row['frame'])): row
        for row in rows
    }


def _history_rows(row, index, horizon):
    history = []
    for offset in range(1, int(horizon) + 1):
        previous = index.get((row['sequence'], int(row['frame']) - offset))
        if previous is None:
            break
        history.append(previous)
    return history


def _temporal_metrics(rows, selected_by_key):
    temporal = []
    for row in rows:
        selected = selected_by_key.get(row['frame_key'])
        temporal.append(base._temporal_record(
            row['record'], selected, row['gt']))
    return base._offline_metrics(temporal)


def _quantiles(values):
    values = list(values)
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        name: float(np.quantile(array, quantile))
        for name, quantile in (
            ('min', 0.0), ('median', 0.5), ('p90', 0.9), ('max', 1.0))
    }


def _run_summary(rows, predicate):
    runs = []
    current = []
    previous_key = None
    previous_frame = None

    def flush():
        if current:
            runs.append(dict(
                sequence=current[0]['sequence'],
                domain=current[0]['domain'],
                start_frame=int(current[0]['frame']),
                end_frame=int(current[-1]['frame']),
                length=len(current),
                frame_keys=[row['frame_key'] for row in current]))
            current.clear()

    for row in rows:
        key = (row['domain'], row['seq_id'])
        frame = int(row['frame'])
        continuous = (
            previous_key == key and previous_frame is not None
            and frame == previous_frame + 1)
        if previous_key is not None and not continuous:
            flush()
        if predicate(row):
            current.append(row)
        else:
            flush()
        previous_key = key
        previous_frame = frame
    flush()
    runs.sort(key=lambda item: (
        -int(item['length']), item['sequence'], int(item['start_frame'])))
    return dict(
        run_count=len(runs),
        max_run_length=max((item['length'] for item in runs), default=0),
        longest_runs=[
            item for item in runs
            if item['length'] == max(
                (candidate['length'] for candidate in runs), default=0)
        ])


def _slice_summary(rows, all_rows):
    keys = {row['frame_key'] for row in rows}
    ordered = [row for row in all_rows if row['frame_key'] in keys]
    index = _history_index(all_rows)
    misses = [row for row in ordered if not row['hit']]
    baseline_boxes = {
        row['frame_key']: row['box'] for row in ordered}
    output = dict(
        frame_count=len(ordered),
        hit_count=sum(row['hit'] for row in ordered),
        miss_count=len(misses),
        baseline_metrics=_temporal_metrics(ordered, baseline_boxes),
        baseline_miss_runs=_run_summary(ordered, lambda row: not row['hit']),
        history_horizons={})

    last_hit_age = []
    no_previous_hit_count = 0
    for row in misses:
        age = None
        offset = 1
        while True:
            candidate = index.get(
                (row['sequence'], int(row['frame']) - offset))
            if candidate is None:
                break
            if candidate['hit']:
                age = offset
                break
            offset += 1
        if age is None:
            no_previous_hit_count += 1
        else:
            last_hit_age.append(age)
    output['last_previous_hit_age'] = dict(
        miss_count=len(misses),
        no_previous_hit_count=no_previous_hit_count,
        age_quantiles=_quantiles(last_hit_age))

    for horizon in HISTORY_HORIZONS:
        own_support_count = 0
        transfer_support_count = 0
        full_window_miss_count = 0
        full_window_all_own_bad_count = 0
        oracle_boxes = {}
        enriched = []
        for row in ordered:
            history = _history_rows(row, index, horizon)
            full_window = len(history) == horizon
            history_own_hit = any(candidate['hit'] for candidate in history)
            transfer_hit = any(
                _hit(candidate['box'], row['gt']) for candidate in history)
            candidates = [row['box']] + [
                candidate['box'] for candidate in history]
            oracle_boxes[row['frame_key']] = _best_gt_oracle(
                candidates, row['gt'])
            enriched_row = dict(row)
            enriched_row.update(
                history_own_hit=history_own_hit,
                history_transfer_hit=transfer_hit,
                full_history_window=full_window)
            enriched.append(enriched_row)
            if not row['hit']:
                own_support_count += int(history_own_hit)
                transfer_support_count += int(transfer_hit)
                if full_window:
                    full_window_miss_count += 1
                    full_window_all_own_bad_count += int(
                        not history_own_hit)

        miss_count = len(misses)
        output['history_horizons'][str(horizon)] = dict(
            current_miss_count=miss_count,
            history_own_hit_support_count=own_support_count,
            history_own_hit_support_rate=(
                float(own_support_count) / miss_count
                if miss_count else None),
            history_hold_current_gt_support_count=transfer_support_count,
            history_hold_current_gt_support_rate=(
                float(transfer_support_count) / miss_count
                if miss_count else None),
            full_history_window_miss_count=full_window_miss_count,
            full_window_all_own_bad_count=full_window_all_own_bad_count,
            full_window_all_own_bad_rate=(
                float(full_window_all_own_bad_count) /
                full_window_miss_count
                if full_window_miss_count else None),
            current_and_history_all_own_bad_runs=_run_summary(
                enriched,
                lambda item: (
                    not item['hit'] and not item['history_own_hit'])),
            current_and_history_hold_all_bad_runs=_run_summary(
                enriched,
                lambda item: (
                    not item['hit'] and not item['history_transfer_hit'])),
            current_plus_history_gt_oracle_metrics=_temporal_metrics(
                ordered, oracle_boxes))

    velocity_boxes = {}
    velocity_support_count = 0
    velocity_available_miss_count = 0
    diagnostics = []
    for row in ordered:
        previous = index.get((row['sequence'], int(row['frame']) - 1))
        previous_previous = index.get(
            (row['sequence'], int(row['frame']) - 2))
        velocity = _constant_velocity(
            None if previous_previous is None else previous_previous['box'],
            None if previous is None else previous['box'])
        velocity_boxes[row['frame_key']] = _best_gt_oracle(
            [row['box'], velocity], row['gt'])
        if not row['hit']:
            velocity_hit = _hit(velocity, row['gt'])
            velocity_available_miss_count += int(velocity is not None)
            velocity_support_count += int(velocity_hit)
            history = _history_rows(row, index, max(HISTORY_HORIZONS))
            diagnostics.append(dict(
                frame_key=row['frame_key'],
                baseline_riou=float(row['riou']),
                available_history_count=len(history),
                history_own_hit=[bool(item['hit']) for item in history],
                history_hold_riou=[
                    _riou(item['box'], row['gt']) for item in history],
                constant_velocity_available=velocity is not None,
                constant_velocity_riou=_riou(velocity, row['gt'])))
    output['constant_velocity'] = dict(
        current_miss_count=len(misses),
        available_miss_count=velocity_available_miss_count,
        support_count=velocity_support_count,
        support_rate=(
            float(velocity_support_count) / len(misses)
            if misses else None),
        current_plus_velocity_gt_oracle_metrics=_temporal_metrics(
            ordered, velocity_boxes))
    output['miss_diagnostics'] = diagnostics
    return output


def _stream_summary(rows):
    result = {}
    groups = (
        ('all', rows),
        ('real', [row for row in rows if row['domain'] == 'real']),
        ('sim', [row for row in rows if row['domain'] == 'sim']))
    for name, group in groups:
        result[name] = _slice_summary(group, rows)
    result['per_sequence'] = {
        sequence: _slice_summary(
            [row for row in rows if row['sequence'] == sequence], rows)
        for sequence in sorted({row['sequence'] for row in rows})}
    return result


def audit_payload(payload, audit_bytes, split_root, evidence_role,
                  required_frame_count=None):
    records = list(payload.get('records') or [])
    if not records:
        raise RuntimeError('All-lane audit has no records')
    expected = (base.ROLE_FRAME_COUNTS[evidence_role]
                if required_frame_count is None else
                int(required_frame_count))
    split_root = Path(split_root)
    sym_key, domains = base._validate(
        payload, records, split_root, evidence_role, expected)
    annotation_dir = split_root / 'annfiles'
    stream_rows = {
        name: _rows(records, annotation_dir, sym_key, name)
        for name in STREAM_NAMES}
    summaries = {
        name: _stream_summary(rows)
        for name, rows in stream_rows.items()}
    return dict(
        protocol=PROTOCOL,
        metric_protocol_version=METRIC_PROTOCOL_VERSION,
        evidence_role=evidence_role,
        evidence_boundary=(
            'source_only_temporal_capacity_audit'
            if evidence_role == 'source-val' else
            'fixed_target_posthoc_temporal_diagnostic_not_model_selection'),
        input=dict(
            protocol=payload.get('protocol'),
            sha256=hashlib.sha256(audit_bytes).hexdigest(),
            frame_count=len(records),
            sequence_counts=dict(Counter(
                str(record['sequence']) for record in records)),
            domain_counts=domains,
            sym_eood_box_key=sym_key,
            dino_computed_on_every_frame=True),
        audit_contract=dict(
            detector_forward=False,
            parameter_update=False,
            threshold_search=False,
            history_horizons=list(HISTORY_HORIZONS),
            history_is_strictly_causal=True,
            history_never_crosses_sequence_or_frame_gap=True,
            domain_labels_used_to_reproduce_v4_diagnostic_stream=True,
            native_stream_history_analysis_is_domain_agnostic=True,
            v4_diagnostic_is_not_a_future_temporal_policy=True,
            sequence_identity_used_for_chronology_only=True,
            sequence_frame_slice_routing=False,
            gt_history_selection_is_non_deployable_oracle=True,
            gt_velocity_selection_is_non_deployable_oracle=True,
            eligible_for_runtime_policy=False),
        streams=summaries,
        interpretation_contract=dict(
            history_own_hit_support=(
                'a past prediction was correct at its own timestamp'),
            history_hold_current_gt_support=(
                'an unchanged past OBB overlaps the current GT at RIoU>=0.5'),
            constant_velocity_support=(
                'a fixed two-frame extrapolation overlaps current GT'),
            oracle_warning=(
                'GT-selected history and velocity streams prove capacity only')),
        next_stage=dict(
            temporal_model_implemented=False,
            temporal_model_training_authorized=False,
            fixed_target_allowed_for_model_selection=False,
            eligible_for_unknown_sequence_claim=False,
            required_review=(
                'review source-val history support and all-bad runs before '
                'designing a history-rejecting temporal OBB refiner')),
        decision=(
            'SOURCE_VAL_CAUSAL_HISTORY_SUPPORT_AUDIT_COMPLETE'
            if evidence_role == 'source-val' else
            'FIXED_TARGET_CAUSAL_HISTORY_DIAGNOSTIC_ONLY'))


def main():
    args = parse_args()
    audit_path = Path(args.audit_json)
    audit_bytes = audit_path.read_bytes()
    payload = json.loads(audit_bytes)
    result = audit_payload(
        payload, audit_bytes, Path(args.data_root) / args.split,
        args.evidence_role, args.require_frame_count)
    base._write_json_atomic(args.out_json, result)
    v4 = result['streams']['v4_diagnostic']
    print('[history-audit] role={}'.format(result['evidence_role']))
    print('[history-audit] frames={}'.format(result['input']['frame_count']))
    print('[history-audit] v4_baseline={}'.format(
        v4['all']['baseline_metrics']))
    for domain in ('real', 'sim'):
        compact = {}
        for horizon, row in v4[domain]['history_horizons'].items():
            compact[horizon] = dict(
                miss_count=row['current_miss_count'],
                own_hit_support_rate=row['history_own_hit_support_rate'],
                hold_support_rate=row[
                    'history_hold_current_gt_support_rate'],
                all_bad_max_run=row[
                    'current_and_history_all_own_bad_runs'][
                        'max_run_length'],
                oracle_mcml=row[
                    'current_plus_history_gt_oracle_metrics'].get(
                        domain + '/MCML_max(frames)'))
        print('[history-audit] v4_{}_history={}'.format(domain, compact))
        print('[history-audit] v4_{}_velocity={}'.format(
            domain, v4[domain]['constant_velocity']))
    print('[history-audit] decision={}'.format(result['decision']))


if __name__ == '__main__':
    main()
