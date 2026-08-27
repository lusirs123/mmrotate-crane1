"""JSON-only audit for Application-Domain Asymmetric Primary Fusion V4.

This tool never opens a detector checkpoint or image and never runs either
model.  It reconstructs one fixed output stream from an existing all-lane
audit:

* real: native DINO primary, SymEOOD only when native DINO is missing;
* sim: SymEOOD primary, with no DINO fallback.

Sequence and frame identifiers are retained only to reconstruct temporal
metrics.  They are never used to choose a lane.

An optional operation-phase manifest can add a separate post-hoc
``measurement-valid`` diagnostic for fixed-target evidence.  The raw metrics,
documented gate, detector outputs, and OBB annotations remain unchanged.
"""

import argparse
import functools
import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path

import numpy as np

from crane_project.tools.eval_crane_offline import (
    METRIC_PROTOCOL_VERSION, CraneOfflineEvaluator, compute_riou,
    parse_dota_txt, parse_seq_frame)


PROTOCOL = 'application_domain_asymmetric_primary_fusion_v4_json_audit'
INPUT_PROTOCOL = 'source_owned_geometry_union_v2'
MEASUREMENT_VALIDITY_PROTOCOL = 'crane_measurement_validity_v1'
MEASUREMENT_VALIDITY_STATUS = (
    'POST_HOC_OPERATIONAL_VALIDITY_DIAGNOSTIC')
MEASUREMENT_VALIDITY_SELECTION_BASIS = (
    'manual_video_operation_phase_review')
MCML_LIMIT = 5
FIXED_TEST_SIM_A_RMSE_REFERENCE_DEG = 1.5487
METRIC_TOLERANCE = 1e-4
ROLE_FRAME_COUNTS = {
    'source-val': 738,
    'fixed-target': 992,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the JSON-only asymmetric-primary V4 audit')
    parser.add_argument('--audit-json', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument('--split', required=True, choices=['val', 'test'])
    parser.add_argument(
        '--evidence-role', required=True,
        choices=sorted(ROLE_FRAME_COUNTS))
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--require-frame-count', type=int)
    parser.add_argument(
        '--measurement-validity-json',
        help=(
            'Optional fixed-target-only operation-phase manifest.  It adds '
            'a separate measurement-valid diagnostic and never overrides '
            'the documented all-frame gate.'))
    return parser.parse_args()


def _box(record, key):
    value = record.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if (array.size < 6 or not np.isfinite(array[:6]).all()
            or float(array[2]) <= 0.0 or float(array[3]) <= 0.0):
        raise RuntimeError('Invalid {} in audit record'.format(key))
    return array[:6].copy()


def _resolve_sym_key(records):
    for key in ('sym_eood_original_box', 'sym_eood_box'):
        if all(key in record for record in records):
            return key
    raise RuntimeError(
        'All-lane audit must contain one complete SymEOOD field: '
        'sym_eood_original_box or legacy sym_eood_box')


@functools.lru_cache(maxsize=None)
def _ground_truth(path):
    boxes = parse_dota_txt(path)
    if not boxes:
        raise RuntimeError('Annotation has no OBB: {}'.format(path))
    return np.asarray(boxes[0], dtype=np.float64)


def _annotation(record, annotation_dir):
    path = annotation_dir / (Path(record['filename']).stem + '.txt')
    if not path.is_file():
        raise RuntimeError('Annotation does not exist: {}'.format(path))
    return _ground_truth(os.fspath(path))


def _temporal_record(record, selected, gt):
    domain, seq_id, frame = parse_seq_frame(record['filename'])
    return dict(
        domain=domain,
        seq_id=seq_id,
        frame_id=frame,
        pred_box=None if selected is None else selected[:5],
        gt_box=gt[:5],
        score=0.0 if selected is None else float(selected[5]),
        plc_rope=None)


def _offline_metrics(records):
    logging.disable(logging.CRITICAL)
    try:
        # V4 gates require MCML on both source-val and fixed target.  Test mode
        # here means the full metric set, not access to the target test split.
        return CraneOfflineEvaluator(mode='test').evaluate_records(records)
    finally:
        logging.disable(logging.NOTSET)


def _hit(box, gt):
    return bool(
        box is not None
        and compute_riou(box[:5], gt[:5]) >= 0.5)


def _riou(box, gt):
    if box is None:
        return 0.0
    return float(compute_riou(box[:5], gt[:5]))


def _summarize(records, annotation_dir, sym_key, policy):
    temporal = []
    hit_keys = set()
    selected_source_counts = Counter()
    selected_source_counts_by_domain = {
        'real': Counter(),
        'sim': Counter(),
    }
    decisions = []

    for record in records:
        domain, seq_id, frame = parse_seq_frame(record['filename'])
        sym = _box(record, sym_key)
        dino = _box(record, 'dino_native_box')
        gt = _annotation(record, annotation_dir)

        if policy == 'sym_eood':
            selected = sym
            source = 'sym_eood' if sym is not None else 'missing'
            reason = ('sym_eood_baseline' if sym is not None
                      else 'sym_eood_missing')
        elif policy == 'dino_native':
            selected = dino
            source = 'dino_native' if dino is not None else 'missing'
            reason = ('dino_native_baseline' if dino is not None
                      else 'dino_native_missing')
        elif policy == 'v4':
            if domain == 'real':
                if dino is not None:
                    selected = dino
                    source = 'dino_native'
                    reason = 'real_dino_primary'
                elif sym is not None:
                    selected = sym
                    source = 'sym_eood_fallback'
                    reason = 'real_dino_missing_sym_eood_fallback'
                else:
                    selected = None
                    source = 'missing'
                    reason = 'real_both_lanes_missing'
            elif domain == 'sim':
                selected = sym
                source = 'sym_eood' if sym is not None else 'missing'
                reason = ('sim_sym_eood_primary' if sym is not None
                          else 'sim_sym_eood_missing_no_dino_fallback')
            else:
                raise RuntimeError(
                    'V4 requires an explicit real/sim application domain')
        else:
            raise ValueError('Unknown policy: {}'.format(policy))

        frame_key = '{}|{}'.format(record['sequence'], int(record['frame']))
        selected_riou = _riou(selected, gt)
        sym_riou = _riou(sym, gt)
        dino_riou = _riou(dino, gt)
        selected_hit = selected_riou >= 0.5
        if selected_hit:
            hit_keys.add(frame_key)
        selected_source_counts[source] += 1
        selected_source_counts_by_domain[domain][source] += 1
        temporal.append(_temporal_record(record, selected, gt))
        decisions.append(dict(
            filename=str(record['filename']),
            sequence=str(record['sequence']),
            frame=int(record['frame']),
            domain=domain,
            seq_id=seq_id,
            selected_source=source,
            selection_reason=reason,
            selected_present=selected is not None,
            selected_riou=selected_riou,
            selected_hit=selected_hit,
            sym_eood_present=sym is not None,
            sym_eood_riou=sym_riou,
            sym_eood_hit=sym_riou >= 0.5,
            dino_native_present=dino is not None,
            dino_native_riou=dino_riou,
            dino_native_hit=dino_riou >= 0.5,
            selected_box=(None if selected is None else
                          [float(value) for value in selected[:6]])))

    return dict(
        hit_count=len(hit_keys),
        hit_frame_keys=sorted(hit_keys),
        selected_source_counts=dict(selected_source_counts),
        selected_source_counts_by_domain={
            domain: dict(counts)
            for domain, counts in selected_source_counts_by_domain.items()},
        metrics=_offline_metrics(temporal),
        decisions=decisions)


def _summarize_miss_run(rows):
    return dict(
        domain=rows[0]['domain'],
        sequence=rows[0]['sequence'],
        seq_id=rows[0]['seq_id'],
        start_frame=int(rows[0]['frame']),
        end_frame=int(rows[-1]['frame']),
        length=len(rows),
        frame_keys=[
            '{}|{}'.format(row['sequence'], int(row['frame']))
            for row in rows],
        selected_missing_count=sum(
            not row['selected_present'] for row in rows),
        dino_missing_count=sum(
            not row['dino_native_present'] for row in rows),
        dino_present_wrong_count=sum(
            row['dino_native_present'] and not row['dino_native_hit']
            for row in rows),
        dino_hit_count=sum(row['dino_native_hit'] for row in rows),
        sym_eood_missing_count=sum(
            not row['sym_eood_present'] for row in rows),
        sym_eood_present_wrong_count=sum(
            row['sym_eood_present'] and not row['sym_eood_hit']
            for row in rows),
        sym_eood_hit_count=sum(row['sym_eood_hit'] for row in rows),
        selected_source_counts=dict(Counter(
            row['selected_source'] for row in rows)),
        frames=[dict(
            frame=int(row['frame']),
            selected_source=row['selected_source'],
            selected_present=bool(row['selected_present']),
            selected_riou=float(row['selected_riou']),
            dino_native_present=bool(row['dino_native_present']),
            dino_native_riou=float(row['dino_native_riou']),
            sym_eood_present=bool(row['sym_eood_present']),
            sym_eood_riou=float(row['sym_eood_riou']))
            for row in rows])


def _failure_attribution(summary):
    ordered = sorted(
        summary['decisions'],
        key=lambda row: (row['domain'], row['seq_id'], int(row['frame'])))
    runs = []
    current = []
    previous_key = None
    previous_frame = None

    def flush():
        if current:
            runs.append(_summarize_miss_run(list(current)))
            current.clear()

    for row in ordered:
        key = (row['domain'], row['seq_id'])
        frame = int(row['frame'])
        continuous = (
            previous_key == key
            and previous_frame is not None
            and frame == previous_frame + 1)
        if previous_key is not None and not continuous:
            flush()
        if row['selected_hit']:
            flush()
        else:
            current.append(row)
        previous_key = key
        previous_frame = frame
    flush()

    runs.sort(key=lambda row: (
        -int(row['length']), row['domain'], row['seq_id'],
        int(row['start_frame'])))
    over_limit = [row for row in runs if row['length'] > MCML_LIMIT]
    if not over_limit:
        holder = dict(
            eligible_for_followup_counterfactual=False,
            reason='no_over_limit_failure_run',
            necessary_condition='not_applicable')
    elif all(row['selected_missing_count'] > 0 for row in over_limit):
        holder = dict(
            eligible_for_followup_counterfactual=True,
            reason=(
                'every_over_limit_run_contains_a_selected_missing_frame'),
            necessary_condition=(
                'passed_but_not_sufficient_previous_observation_must_still_hit'))
    else:
        holder = dict(
            eligible_for_followup_counterfactual=False,
            reason='over_limit_run_has_no_selected_missing_frame',
            necessary_condition='failed')
    return dict(
        run_count=len(runs),
        max_run_length=max(
            (int(row['length']) for row in runs), default=0),
        runs=runs,
        over_limit_runs=over_limit,
        missing_only_observation_holder=holder)


def _canonical_json_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False).encode('utf-8')


def _validate_measurement_validity(payload, records, evidence_role):
    if evidence_role != 'fixed-target':
        raise RuntimeError(
            'Measurement validity is a fixed-target diagnostic only')
    if payload.get('protocol') != MEASUREMENT_VALIDITY_PROTOCOL:
        raise RuntimeError(
            'Unknown measurement-validity protocol: {}'.format(
                payload.get('protocol')))
    status = payload.get('status', MEASUREMENT_VALIDITY_STATUS)
    if status != MEASUREMENT_VALIDITY_STATUS:
        raise RuntimeError(
            'Measurement validity status must remain {}'.format(
                MEASUREMENT_VALIDITY_STATUS))
    if payload.get('selection_basis') != (
            MEASUREMENT_VALIDITY_SELECTION_BASIS):
        raise RuntimeError(
            'Measurement validity requires selection_basis={}'.format(
                MEASUREMENT_VALIDITY_SELECTION_BASIS))

    sequences = payload.get('sequences')
    if not isinstance(sequences, dict):
        raise RuntimeError(
            'Measurement validity requires a sequences object')

    observed_real_frames = {}
    for record in records:
        domain, _seq_id, frame = parse_seq_frame(record['filename'])
        if domain == 'real':
            sequence = str(record['sequence'])
            observed_real_frames.setdefault(sequence, set()).add(int(frame))

    observed_sequences = set(observed_real_frames)
    manifest_sequences = set(sequences)
    if manifest_sequences != observed_sequences:
        raise RuntimeError(
            'Measurement-validity sequences must exactly match real input; '
            'missing={} extra={}'.format(
                sorted(observed_sequences - manifest_sequences),
                sorted(manifest_sequences - observed_sequences)))

    invalid_reasons = {}
    normalized_sequences = {}
    for sequence in sorted(observed_sequences):
        spec = sequences[sequence]
        if not isinstance(spec, dict):
            raise RuntimeError(
                'Measurement-validity sequence spec must be an object')
        if spec.get('default_valid') is not True:
            raise RuntimeError(
                '{} must use default_valid=true'.format(sequence))
        intervals = spec.get('invalid_intervals', [])
        if not isinstance(intervals, list):
            raise RuntimeError(
                '{} invalid_intervals must be a list'.format(sequence))

        normalized_intervals = []
        previous_end = None
        observed_frames = observed_real_frames[sequence]
        for interval in intervals:
            if not isinstance(interval, dict):
                raise RuntimeError('Invalid interval must be an object')
            try:
                start = int(interval['start_frame'])
                end = int(interval['end_frame'])
            except (KeyError, TypeError, ValueError):
                raise RuntimeError(
                    '{} interval requires integer start/end frames'.format(
                        sequence))
            reason = str(interval.get('reason', '')).strip()
            if not reason:
                raise RuntimeError(
                    '{} interval requires a non-empty reason'.format(
                        sequence))
            if start > end:
                raise RuntimeError(
                    '{} interval start exceeds end'.format(sequence))
            if previous_end is not None and start <= previous_end:
                raise RuntimeError(
                    '{} invalid intervals overlap'.format(sequence))
            selected_frames = sorted(
                frame for frame in observed_frames if start <= frame <= end)
            if not selected_frames:
                raise RuntimeError(
                    '{} interval {}..{} selects no input frames'.format(
                        sequence, start, end))
            for frame in selected_frames:
                invalid_reasons[(sequence, frame)] = reason
            normalized_intervals.append(dict(
                start_frame=start,
                end_frame=end,
                reason=reason,
                selected_frame_count=len(selected_frames)))
            previous_end = end

        normalized_sequences[sequence] = dict(
            default_valid=True,
            invalid_intervals=normalized_intervals)

    real_frame_count = sum(
        len(frames) for frames in observed_real_frames.values())
    if len(invalid_reasons) >= real_frame_count:
        raise RuntimeError(
            'Measurement validity excludes every real frame')
    return invalid_reasons, normalized_sequences


def _measurement_validity_audit(
        payload, payload_bytes, records, annotation_dir, v4,
        evidence_role):
    invalid_reasons, normalized_sequences = (
        _validate_measurement_validity(payload, records, evidence_role))
    if payload_bytes is None:
        payload_bytes = _canonical_json_bytes(payload)

    decisions_by_key = {
        (str(row['sequence']), int(row['frame'])): row
        for row in v4['decisions']}
    valid_temporal = []
    valid_decisions = []
    excluded_frames = []
    per_sequence = {}

    for record in records:
        sequence = str(record['sequence'])
        frame = int(record['frame'])
        key = (sequence, frame)
        decision = decisions_by_key[key]
        reason = invalid_reasons.get(key)
        sequence_summary = None
        if decision['domain'] == 'real':
            sequence_summary = per_sequence.setdefault(sequence, dict(
                raw_frame_count=0,
                measurement_valid_frame_count=0,
                excluded_frame_count=0,
                excluded_hit_count=0,
                excluded_miss_count=0,
                invalid_reason_counts=Counter()))
            sequence_summary['raw_frame_count'] += 1

        if reason is not None:
            if sequence_summary is None:
                raise RuntimeError(
                    'Measurement validity may not exclude sim frames')
            sequence_summary['excluded_frame_count'] += 1
            sequence_summary['invalid_reason_counts'][reason] += 1
            if decision['selected_hit']:
                sequence_summary['excluded_hit_count'] += 1
            else:
                sequence_summary['excluded_miss_count'] += 1
            excluded_frames.append(dict(
                frame_key='{}|{}'.format(sequence, frame),
                sequence=sequence,
                frame=frame,
                reason=reason,
                selected_hit=bool(decision['selected_hit']),
                selected_riou=float(decision['selected_riou'])))
            continue

        if sequence_summary is not None:
            sequence_summary['measurement_valid_frame_count'] += 1
        selected = decision['selected_box']
        selected = (None if selected is None
                    else np.asarray(selected, dtype=np.float64))
        gt = _annotation(record, annotation_dir)
        valid_temporal.append(_temporal_record(record, selected, gt))
        valid_decisions.append(decision)

    if not valid_temporal:
        raise RuntimeError('Measurement validity leaves no evaluable frames')

    for summary in per_sequence.values():
        summary['invalid_reason_counts'] = dict(
            summary['invalid_reason_counts'])
        raw_count = summary['raw_frame_count']
        summary['measurement_valid_fraction'] = (
            float(summary['measurement_valid_frame_count']) / raw_count)

    valid_metrics = _offline_metrics(valid_temporal)
    valid_failure = _failure_attribution(dict(decisions=valid_decisions))
    excluded_hits = sum(row['selected_hit'] for row in excluded_frames)
    excluded_count = len(excluded_frames)
    all_frame_count = len(records)
    raw_real_count = sum(
        1 for row in v4['decisions'] if row['domain'] == 'real')
    sim_count = all_frame_count - raw_real_count
    valid_real_count = raw_real_count - excluded_count
    valid_real_mcml = float(valid_metrics.get(
        'real/MCML_max(frames)', float('inf')))
    operational_checks = dict(
        measurement_valid_real_mcml_max_le_5=(
            valid_real_mcml <= MCML_LIMIT),
        original_documented_gate_not_overridden=True,
        detector_forward_not_run=True,
        parameter_update_not_run=True)

    return dict(
        protocol=MEASUREMENT_VALIDITY_PROTOCOL,
        status=MEASUREMENT_VALIDITY_STATUS,
        selection_basis=MEASUREMENT_VALIDITY_SELECTION_BASIS,
        evidence_boundary=(
            'separate_operational_diagnostic_not_original_gate_override'),
        input=dict(
            sha256=hashlib.sha256(payload_bytes).hexdigest(),
            normalized_sequences=normalized_sequences),
        scope=dict(
            all_frame_count=all_frame_count,
            sim_frame_count=sim_count,
            raw_real_frame_count=raw_real_count,
            measurement_valid_real_frame_count=valid_real_count,
            measurement_invalid_real_frame_count=excluded_count,
            measurement_valid_real_fraction=(
                float(valid_real_count) / raw_real_count),
            excluded_hit_count=int(excluded_hits),
            excluded_miss_count=int(excluded_count - excluded_hits),
            per_sequence=per_sequence),
        metrics=valid_metrics,
        failure_attribution=valid_failure,
        excluded_frames=excluded_frames,
        operational_gate=dict(
            passed=all(operational_checks.values()),
            checks=operational_checks,
            mcml_limit=MCML_LIMIT),
        original_documented_gate_overridden=False,
        eligible_for_original_gate_override=False,
        eligible_for_unknown_sequence_claim=False)


def _metric(summary, name, default):
    return float(summary['metrics'].get(name, default))


def _comparison(candidate, baseline):
    candidate_hits = set(candidate['hit_frame_keys'])
    baseline_hits = set(baseline['hit_frame_keys'])
    common_metrics = sorted(
        set(candidate['metrics']) & set(baseline['metrics']))
    return dict(
        lost_hit_frame_keys=sorted(baseline_hits - candidate_hits),
        gained_hit_frame_keys=sorted(candidate_hits - baseline_hits),
        metric_delta={
            name: round(
                float(candidate['metrics'][name])
                - float(baseline['metrics'][name]), 6)
            for name in common_metrics})


def _gate(v4, sym_baseline, evidence_role):
    v4_sim_angle = _metric(v4, 'sim/A-RMSE(deg)', float('inf'))
    sym_sim_angle = _metric(
        sym_baseline, 'sim/A-RMSE(deg)', float('inf'))
    sim_sources = v4['selected_source_counts_by_domain']['sim']
    checks = dict(
        real_mcml_max_le_5=(
            _metric(v4, 'real/MCML_max(frames)', float('inf'))
            <= MCML_LIMIT),
        sim_mcml_max_le_5=(
            _metric(v4, 'sim/MCML_max(frames)', float('inf'))
            <= MCML_LIMIT),
        sim_a_rmse_no_worse_than_sym_eood=(
            v4_sim_angle <= sym_sim_angle + METRIC_TOLERANCE),
        sim_stream_is_exact_symeood_primary=(
            sim_sources.get('dino_native', 0) == 0
            and sim_sources.get(
                'sym_eood_fallback', 0) == 0))
    if evidence_role == 'fixed-target':
        checks['sim_a_rmse_le_1_5487_deg'] = (
            v4_sim_angle
            <= FIXED_TEST_SIM_A_RMSE_REFERENCE_DEG + METRIC_TOLERANCE)
    return dict(
        passed=all(checks.values()),
        checks=checks,
        mcml_limit=MCML_LIMIT,
        fixed_test_sim_a_rmse_reference_deg=(
            FIXED_TEST_SIM_A_RMSE_REFERENCE_DEG
            if evidence_role == 'fixed-target' else None),
        sim_a_rmse_sym_eood_deg=sym_sim_angle,
        sim_a_rmse_v4_deg=v4_sim_angle)


def _validate(payload, records, split_root, evidence_role,
              required_frame_count):
    if payload.get('protocol') != INPUT_PROTOCOL:
        raise RuntimeError('V4 requires an unrouted all-lane audit')
    if len(records) != required_frame_count:
        raise RuntimeError(
            'Expected {} records for {}, got {}'.format(
                required_frame_count, evidence_role, len(records)))
    if evidence_role == 'source-val' and split_root.name != 'val':
        raise RuntimeError('source-val evidence must use the val split')
    if evidence_role == 'fixed-target' and split_root.name != 'test':
        raise RuntimeError('fixed-target evidence must use the test split')

    annotation_dir = split_root / 'annfiles'
    if not annotation_dir.is_dir():
        raise RuntimeError('Annotation directory does not exist')

    sym_key = _resolve_sym_key(records)
    seen = set()
    observed_stems = set()
    domains = Counter()
    for record in records:
        for key in ('filename', 'sequence', 'frame', 'dino_native_box'):
            if key not in record:
                raise RuntimeError('Audit record is missing ' + key)
        if (record.get('dino_invoked') is False
                or record.get('raw_selected_source') == 'not_computed'):
            raise RuntimeError(
                'V4 requires DINO to have been computed on every input frame')
        domain, seq_id, parsed_frame = parse_seq_frame(record['filename'])
        if domain not in ('real', 'sim'):
            raise RuntimeError('Audit contains an unknown application domain')
        parsed_sequence = '{}_{}'.format(domain, seq_id)
        if str(record['sequence']) != parsed_sequence:
            raise RuntimeError(
                'Audit sequence metadata disagrees with filename')
        if int(record['frame']) != int(parsed_frame):
            raise RuntimeError('Audit frame metadata disagrees with filename')
        domains[domain] += 1
        frame_key = (str(record['sequence']), int(record['frame']))
        if frame_key in seen:
            raise RuntimeError('Duplicate audit frame: {}'.format(frame_key))
        seen.add(frame_key)
        observed_stems.add(Path(record['filename']).stem)
        _box(record, sym_key)
        _box(record, 'dino_native_box')

    if not domains['real'] or not domains['sim']:
        raise RuntimeError('V4 audit must contain both real and sim frames')
    expected_stems = {
        path.stem for path in annotation_dir.iterdir()
        if path.is_file() and path.suffix.lower() == '.txt'}
    if observed_stems != expected_stems:
        missing = sorted(expected_stems - observed_stems)[:5]
        extra = sorted(observed_stems - expected_stems)[:5]
        raise RuntimeError(
            'Audit does not exactly match annotations; missing={} extra={}'
            .format(missing, extra))
    return sym_key, dict(domains)


def audit_payload(payload, audit_bytes, split_root, evidence_role,
                  required_frame_count=None, measurement_validity=None,
                  measurement_validity_bytes=None):
    records = list(payload.get('records') or [])
    if not records:
        raise RuntimeError('All-lane audit has no records')
    expected = (ROLE_FRAME_COUNTS[evidence_role]
                if required_frame_count is None else
                int(required_frame_count))
    split_root = Path(split_root)
    sym_key, domains = _validate(
        payload, records, split_root, evidence_role, expected)
    annotation_dir = split_root / 'annfiles'

    sym_baseline = _summarize(
        records, annotation_dir, sym_key, 'sym_eood')
    dino_baseline = _summarize(
        records, annotation_dir, sym_key, 'dino_native')
    v4 = _summarize(records, annotation_dir, sym_key, 'v4')
    failure_attribution = _failure_attribution(v4)
    gate = _gate(v4, sym_baseline, evidence_role)
    measurement_validity_result = None
    if measurement_validity is not None:
        measurement_validity_result = _measurement_validity_audit(
            measurement_validity, measurement_validity_bytes, records,
            annotation_dir, v4, evidence_role)
    if gate['passed']:
        decision = (
            'PASS_SOURCE_VAL_DOCUMENTED_GATE'
            if evidence_role == 'source-val'
            else 'PASS_FIXED_TARGET_DIAGNOSTIC_GATE')
    else:
        decision = 'STOP_V4_GATE_FAILED'

    return dict(
        protocol=PROTOCOL,
        metric_protocol_version=METRIC_PROTOCOL_VERSION,
        evidence_role=evidence_role,
        evidence_boundary=(
            'source_only_gate' if evidence_role == 'source-val'
            else 'fixed_target_diagnostic_not_unknown_sequence'),
        input=dict(
            protocol=payload.get('protocol'),
            sha256=hashlib.sha256(audit_bytes).hexdigest(),
            frame_count=len(records),
            sequence_counts=dict(Counter(
                str(record['sequence']) for record in records)),
            domain_counts=domains,
            sym_eood_box_key=sym_key,
            dino_computed_on_every_frame=True),
        policy=dict(
            real='dino_native_primary_then_sym_eood_missing_fallback',
            sim='sym_eood_primary_no_dino_fallback',
            parameter_update=False,
            detector_forward=False,
            threshold_search=False,
            sequence_frame_slice_routing=False,
            sequence_frame_used_for_temporal_metrics_only=True),
        sym_eood_baseline=sym_baseline,
        native_dino_baseline=dino_baseline,
        v4=v4,
        failure_attribution=failure_attribution,
        v4_vs_sym_eood=_comparison(v4, sym_baseline),
        v4_vs_native_dino=_comparison(v4, dino_baseline),
        documented_gate=gate,
        measurement_validity=measurement_validity_result,
        eligible_for_formal_config_from_this_report_alone=False,
        eligible_for_unknown_sequence_claim=False,
        decision=decision)


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def main():
    args = parse_args()
    audit_path = Path(args.audit_json)
    audit_bytes = audit_path.read_bytes()
    payload = json.loads(audit_bytes)
    measurement_validity = None
    measurement_validity_bytes = None
    if args.measurement_validity_json:
        measurement_validity_path = Path(args.measurement_validity_json)
        measurement_validity_bytes = measurement_validity_path.read_bytes()
        measurement_validity = json.loads(measurement_validity_bytes)
    result = audit_payload(
        payload, audit_bytes, Path(args.data_root) / args.split,
        args.evidence_role, args.require_frame_count,
        measurement_validity=measurement_validity,
        measurement_validity_bytes=measurement_validity_bytes)
    _write_json_atomic(args.out_json, result)
    print('[v4-json] role={}'.format(result['evidence_role']))
    print('[v4-json] frames={}'.format(result['input']['frame_count']))
    print('[v4-json] sym_key={}'.format(
        result['input']['sym_eood_box_key']))
    print('[v4-json] selected_sources={}'.format(
        result['v4']['selected_source_counts']))
    print('[v4-json] metrics={}'.format(result['v4']['metrics']))
    print('[v4-json] vs_sym_lost={} vs_sym_gained={}'.format(
        len(result['v4_vs_sym_eood']['lost_hit_frame_keys']),
        len(result['v4_vs_sym_eood']['gained_hit_frame_keys'])))
    print('[v4-json] failure_max_run={} holder={}'.format(
        result['failure_attribution']['max_run_length'],
        result['failure_attribution'][
            'missing_only_observation_holder']))
    print('[v4-json] gate={}'.format(result['documented_gate']))
    if result['measurement_validity'] is not None:
        validity = result['measurement_validity']
        print('[v4-json] measurement_validity_scope={}'.format(
            validity['scope']))
        print('[v4-json] measurement_validity_metrics={}'.format(
            validity['metrics']))
        print('[v4-json] measurement_validity_gate={}'.format(
            validity['operational_gate']))
    print('[v4-json] decision={}'.format(result['decision']))


if __name__ == '__main__':
    main()
