"""JSON-only geometry refinability audit for the unified SymEOOD--DINO model.

This audit never runs either detector and never updates a parameter.  It uses
an existing unrouted all-lane audit plus DOTA ground truth to answer a bounded
question before a geometry refiner is implemented:

* is native-DINO mainly wrong in centre, size, or angle;
* can the existing SymEOOD/native-DINO candidate pool jointly retain recall
  and geometry; and
* is native-DINO centre support strong enough for a SymEOOD-feature geometry
  refiner to be a plausible next source-only experiment.

Ground-truth component replacement and per-frame best-candidate streams are
explicitly non-deployable diagnostic oracles.  They are never converted into
a routing rule.  Real and simulated records follow the same audit policy;
domain labels are used only to report evaluation slices.
"""

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from crane_project.tools import (
    symeood_dino_application_domain_v4_audit as base)
from crane_project.tools.eval_crane_offline import (
    METRIC_PROTOCOL_VERSION, angle_diff, compute_riou, obb_diag,
    parse_seq_frame)


PROTOCOL = 'domain_agnostic_dual_candidate_geometry_refinability_audit_v1'
INPUT_PROTOCOL = 'source_owned_geometry_union_v2'
ROLE_FRAME_COUNTS = dict(base.ROLE_FRAME_COUNTS)
ANGLE_LIMIT_DEG = 35.0

STREAM_NAMES = (
    'sym_eood',
    'dino_native',
    'candidate_riou_oracle',
    'dino_gt_center_oracle',
    'dino_gt_size_oracle',
    'dino_gt_angle_oracle',
    'dino_center_gt_geometry_oracle',
)

HYBRID_NAMES = (
    'sym_eood',
    'dino_native',
    'dino_center_sym_geometry',
    'sym_center_dino_geometry',
    'dino_with_sym_size',
    'dino_with_sym_angle',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Audit domain-agnostic SymEOOD--DINO geometry refinability '
            'from an existing all-lane JSON'))
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
            'Optional fixed-target-only operation-phase manifest.  It '
            'creates a separate evaluation slice and is never supplied to '
            'a detector, refiner, or candidate selector.'))
    return parser.parse_args()


def _quantiles(values):
    values = list(values)
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        name: float(np.quantile(array, q))
        for name, q in (
            ('min', 0.0), ('p10', 0.1), ('median', 0.5),
            ('p90', 0.9), ('max', 1.0))
    }


def _copy_box(box):
    if box is None:
        return None
    return np.asarray(box, dtype=np.float64).reshape(-1)[:6].copy()


def _hybrid(center_source, geometry_source, score_source):
    if center_source is None or geometry_source is None:
        return None
    result = _copy_box(score_source)
    if result is None:
        return None
    result[:2] = center_source[:2]
    result[2:5] = geometry_source[2:5]
    return result


def _replace_size(base_box, size_source):
    if base_box is None or size_source is None:
        return None
    result = _copy_box(base_box)
    result[2:4] = size_source[2:4]
    return result


def _replace_angle(base_box, angle_source):
    if base_box is None or angle_source is None:
        return None
    result = _copy_box(base_box)
    result[4] = angle_source[4]
    return result


def _riou(box, gt):
    if box is None:
        return 0.0
    return float(compute_riou(box[:5], gt[:5]))


def _center_error(box, gt):
    if box is None:
        return None
    return float(np.linalg.norm(box[:2] - gt[:2]))


def _angle_error_deg(box, gt):
    if box is None:
        return None
    error = angle_diff(
        np.asarray([box[4]], dtype=np.float64),
        np.asarray([gt[4]], dtype=np.float64))[0]
    return float(abs(math.degrees(float(error))))


def _local_center_residual(box, gt):
    """Return GT-centre residual in the proposal's local OBB coordinates."""
    if box is None:
        return None
    dx = float(gt[0] - box[0])
    dy = float(gt[1] - box[1])
    cosine = math.cos(float(box[4]))
    sine = math.sin(float(box[4]))
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    return local_x, local_y


def _dino_residuals(dino, gt):
    if dino is None:
        return None
    local_x, local_y = _local_center_residual(dino, gt)
    angle_signed = float(angle_diff(
        np.asarray([gt[4]], dtype=np.float64),
        np.asarray([dino[4]], dtype=np.float64))[0])
    return dict(
        center_error_px=_center_error(dino, gt),
        local_dx_over_w=float(local_x / dino[2]),
        local_dy_over_h=float(local_y / dino[3]),
        abs_local_dx_over_w=float(abs(local_x / dino[2])),
        abs_local_dy_over_h=float(abs(local_y / dino[3])),
        log_width_ratio=float(math.log(float(gt[2] / dino[2]))),
        log_height_ratio=float(math.log(float(gt[3] / dino[3]))),
        abs_log_width_ratio=float(abs(math.log(float(gt[2] / dino[2])))),
        abs_log_height_ratio=float(abs(math.log(float(gt[3] / dino[3])))),
        angle_residual_deg=float(math.degrees(angle_signed)),
        abs_angle_residual_deg=float(abs(math.degrees(angle_signed))),
        gt_center_inside_dino=bool(
            abs(local_x) <= float(dino[2]) / 2.0
            and abs(local_y) <= float(dino[3]) / 2.0),
        normalized_center_error_by_gt_diag=float(
            _center_error(dino, gt) / max(obb_diag(gt), 1e-9)))


def _candidate_category(sym, dino):
    if sym is None and dino is None:
        return 'both_missing'
    if sym is None:
        return 'dino_only'
    if dino is None:
        return 'sym_only'
    return 'both_present'


def _best_candidate(sym, dino, sym_riou, dino_riou):
    """Non-deployable GT oracle; prefer SymEOOD on an exact RIoU tie."""
    if sym is None:
        return _copy_box(dino), ('dino_native' if dino is not None
                                 else 'missing')
    if dino is None or sym_riou >= dino_riou:
        return _copy_box(sym), 'sym_eood'
    return _copy_box(dino), 'dino_native'


def _frame_row(record, annotation_dir, sym_key):
    domain, seq_id, frame = parse_seq_frame(record['filename'])
    sym = base._box(record, sym_key)
    dino = base._box(record, 'dino_native_box')
    gt = base._annotation(record, annotation_dir)
    sym_riou = _riou(sym, gt)
    dino_riou = _riou(dino, gt)
    best, best_source = _best_candidate(
        sym, dino, sym_riou, dino_riou)

    streams = {
        'sym_eood': _copy_box(sym),
        'dino_native': _copy_box(dino),
        'candidate_riou_oracle': best,
        'dino_gt_center_oracle': _hybrid(gt, dino, dino),
        'dino_gt_size_oracle': _replace_size(dino, gt),
        'dino_gt_angle_oracle': _replace_angle(dino, gt),
        'dino_center_gt_geometry_oracle': _hybrid(dino, gt, dino),
    }
    hybrids = {
        'sym_eood': _copy_box(sym),
        'dino_native': _copy_box(dino),
        'dino_center_sym_geometry': _hybrid(dino, sym, dino),
        'sym_center_dino_geometry': _hybrid(sym, dino, dino),
        'dino_with_sym_size': _replace_size(dino, sym),
        'dino_with_sym_angle': _replace_angle(dino, sym),
    }
    return dict(
        record=record,
        frame_key='{}|{}'.format(record['sequence'], int(frame)),
        domain=domain,
        seq_id=seq_id,
        sequence=str(record['sequence']),
        frame=int(frame),
        gt=gt,
        streams=streams,
        hybrids=hybrids,
        candidate_category=_candidate_category(sym, dino),
        candidate_oracle_source=best_source,
        dino_residuals=_dino_residuals(dino, gt),
        report=dict(
            frame_key='{}|{}'.format(record['sequence'], int(frame)),
            domain=domain,
            sequence=str(record['sequence']),
            frame=int(frame),
            candidate_category=_candidate_category(sym, dino),
            sym_present=bool(sym is not None),
            dino_present=bool(dino is not None),
            candidate_oracle_source=best_source,
            riou={
                name: _riou(box, gt)
                for name, box in {**streams, **hybrids}.items()},
            center_error_px=dict(
                sym_eood=_center_error(sym, gt),
                dino_native=_center_error(dino, gt)),
            angle_error_deg=dict(
                sym_eood=_angle_error_deg(sym, gt),
                dino_native=_angle_error_deg(dino, gt)),
            dino_residuals=_dino_residuals(dino, gt)))


def _stream_metrics(rows, stream_name):
    temporal = []
    for row in rows:
        temporal.append(base._temporal_record(
            row['record'], row['streams'][stream_name], row['gt']))
    return base._offline_metrics(temporal)


def _all_stream_metrics(rows):
    return {
        name: _stream_metrics(rows, name)
        for name in STREAM_NAMES
    }


def _hybrid_summary(rows):
    both = [row for row in rows
            if row['candidate_category'] == 'both_present']
    output = {}
    for slice_name, group in (
            ('all', both),
            ('real', [row for row in both if row['domain'] == 'real']),
            ('sim', [row for row in both if row['domain'] == 'sim'])):
        win_counts = Counter()
        for row in group:
            sym_riou = _riou(row['hybrids']['sym_eood'], row['gt'])
            dino_riou = _riou(row['hybrids']['dino_native'], row['gt'])
            if abs(sym_riou - dino_riou) <= 1e-12:
                win_counts['tie'] += 1
            elif sym_riou > dino_riou:
                win_counts['sym_eood_better'] += 1
            else:
                win_counts['dino_native_better'] += 1
        output[slice_name] = dict(
            frame_count=len(group),
            mean_riou={
                name: (None if not group else float(np.mean([
                    _riou(row['hybrids'][name], row['gt'])
                    for row in group])))
                for name in HYBRID_NAMES},
            candidate_win_counts=dict(win_counts))
    return output


def _residual_summary(rows):
    output = {}
    fields = (
        'center_error_px',
        'normalized_center_error_by_gt_diag',
        'local_dx_over_w', 'local_dy_over_h',
        'abs_local_dx_over_w', 'abs_local_dy_over_h',
        'log_width_ratio', 'log_height_ratio',
        'abs_log_width_ratio', 'abs_log_height_ratio',
        'angle_residual_deg', 'abs_angle_residual_deg',
    )
    for slice_name, group in (
            ('all', rows),
            ('real', [row for row in rows if row['domain'] == 'real']),
            ('sim', [row for row in rows if row['domain'] == 'sim'])):
        present = [row for row in group if row['dino_residuals'] is not None]
        inside = sum(
            row['dino_residuals']['gt_center_inside_dino']
            for row in present)
        output[slice_name] = dict(
            frame_count=len(group),
            dino_present_count=len(present),
            dino_presence_rate=(
                float(len(present)) / len(group) if group else None),
            gt_center_inside_dino_count=int(inside),
            gt_center_inside_dino_rate=(
                float(inside) / len(present) if present else None),
            residual_quantiles={
                field: _quantiles(
                    row['dino_residuals'][field] for row in present)
                for field in fields})
    return output


def _transition_summary(rows):
    ordered = sorted(
        rows, key=lambda row: (
            row['domain'], row['seq_id'], int(row['frame'])))
    buckets = defaultdict(lambda: defaultdict(list))
    previous = None
    for row in ordered:
        current_box = row['streams']['candidate_riou_oracle']
        if previous is not None:
            continuous = (
                previous['domain'] == row['domain']
                and previous['seq_id'] == row['seq_id']
                and int(row['frame']) == int(previous['frame']) + 1)
            previous_box = previous['streams']['candidate_riou_oracle']
            if continuous and previous_box is not None and current_box is not None:
                kind = (
                    'same_source'
                    if previous['candidate_oracle_source']
                    == row['candidate_oracle_source'] else 'source_switch')
                dfr = abs(
                    obb_diag(current_box) - obb_diag(previous_box)) / max(
                        obb_diag(previous_box), 1e-9)
                delta_angle = abs(float(angle_diff(
                    np.asarray([current_box[4]]),
                    np.asarray([previous_box[4]]))[0]))
                aci = float(np.clip(
                    1.0 - math.degrees(delta_angle) / ANGLE_LIMIT_DEG,
                    0.0, 1.0))
                buckets[row['domain']][kind].append((dfr, aci))
        previous = row

    output = {}
    for domain in ('all', 'real', 'sim'):
        domain_buckets = defaultdict(list)
        source = buckets if domain == 'all' else {domain: buckets[domain]}
        for rows_by_kind in source.values():
            for kind, values in rows_by_kind.items():
                domain_buckets[kind].extend(values)
        output[domain] = {}
        for kind in ('same_source', 'source_switch'):
            values = domain_buckets[kind]
            output[domain][kind] = dict(
                transition_count=len(values),
                mean_dfr_fraction=(
                    None if not values else float(np.mean(
                        [value[0] for value in values]))),
                mean_aci=(
                    None if not values else float(np.mean(
                        [value[1] for value in values]))))
    return output


def _candidate_counts(rows):
    by_domain = {}
    for domain, group in (
            ('all', rows),
            ('real', [row for row in rows if row['domain'] == 'real']),
            ('sim', [row for row in rows if row['domain'] == 'sim'])):
        by_domain[domain] = dict(
            frame_count=len(group),
            category_counts=dict(Counter(
                row['candidate_category'] for row in group)),
            candidate_oracle_source_counts=dict(Counter(
                row['candidate_oracle_source'] for row in group)))
    return by_domain


def _metric(metrics, name):
    value = metrics.get(name)
    return None if value is None else float(value)


def _capacity_summary(stream_metrics):
    """Report diagnostic deltas without turning TEST GT into model choice."""
    dino = stream_metrics['dino_native']
    geometry = stream_metrics['dino_center_gt_geometry_oracle']
    candidate = stream_metrics['candidate_riou_oracle']
    result = {}
    for domain in ('real', 'sim'):
        prefix = domain + '/'
        dino_riou = _metric(dino, prefix + 'mean_RIoU')
        geometry_riou = _metric(geometry, prefix + 'mean_RIoU')
        result[domain] = dict(
            dino_mean_riou=dino_riou,
            dino_center_gt_geometry_mean_riou=geometry_riou,
            geometry_oracle_mean_riou_gain=(
                None if dino_riou is None or geometry_riou is None else
                geometry_riou - dino_riou),
            candidate_oracle_mean_riou=_metric(
                candidate, prefix + 'mean_RIoU'),
            candidate_oracle_dfr=_metric(
                candidate, prefix + 'DFR(%/frame)'),
            candidate_oracle_aci=_metric(
                candidate, prefix + 'ACI'),
            candidate_oracle_mcml=_metric(
                candidate, prefix + 'MCML_max(frames)'))
    return result


def _compact_refinability_summary(summary):
    fields = (
        'center_error_px',
        'abs_local_dx_over_w', 'abs_local_dy_over_h',
        'abs_log_width_ratio', 'abs_log_height_ratio',
        'abs_angle_residual_deg')
    output = {}
    for domain, row in summary.items():
        output[domain] = dict(
            dino_presence_rate=row['dino_presence_rate'],
            gt_center_inside_dino_rate=row[
                'gt_center_inside_dino_rate'],
            median_and_p90={
                field: dict(
                    median=row['residual_quantiles'][field].get('median'),
                    p90=row['residual_quantiles'][field].get('p90'))
                for field in fields})
    return output


def _measurement_validity_slice(
        manifest, manifest_bytes, records, rows, evidence_role):
    invalid_reasons, normalized = base._validate_measurement_validity(
        manifest, records, evidence_role)
    if manifest_bytes is None:
        manifest_bytes = base._canonical_json_bytes(manifest)
    kept = []
    excluded = []
    for row in rows:
        reason = invalid_reasons.get((row['sequence'], row['frame']))
        if reason is None:
            kept.append(row)
        else:
            excluded.append(dict(
                frame_key=row['frame_key'],
                sequence=row['sequence'],
                frame=row['frame'],
                reason=reason))
    return dict(
        protocol=base.MEASUREMENT_VALIDITY_PROTOCOL,
        status=base.MEASUREMENT_VALIDITY_STATUS,
        selection_basis=base.MEASUREMENT_VALIDITY_SELECTION_BASIS,
        use='evaluation_scope_only_never_model_input',
        input=dict(
            sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            normalized_sequences=normalized),
        scope=dict(
            kept_frame_count=len(kept),
            excluded_real_frame_count=len(excluded),
            excluded_frames=excluded),
        stream_metrics=_all_stream_metrics(kept),
        candidate_counts=_candidate_counts(kept),
        refinability=_residual_summary(kept),
        candidate_oracle_transition_analysis=_transition_summary(kept),
        eligible_for_original_gate_override=False)


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
    sym_key, domains = base._validate(
        payload, records, split_root, evidence_role, expected)
    annotation_dir = split_root / 'annfiles'
    rows = [_frame_row(record, annotation_dir, sym_key)
            for record in records]
    stream_metrics = _all_stream_metrics(rows)
    measurement = None
    if measurement_validity is not None:
        measurement = _measurement_validity_slice(
            measurement_validity, measurement_validity_bytes,
            records, rows, evidence_role)

    return dict(
        protocol=PROTOCOL,
        metric_protocol_version=METRIC_PROTOCOL_VERSION,
        evidence_role=evidence_role,
        evidence_boundary=(
            'source_only_refiner_hypothesis_audit'
            if evidence_role == 'source-val' else
            'fixed_target_posthoc_capacity_diagnostic_not_model_selection'),
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
            domain_specific_routing=False,
            sequence_frame_slice_routing=False,
            operation_phase_used_as_model_input=False,
            measurement_validity_used_for_selection=False,
            gt_component_replacement_is_non_deployable_oracle=True,
            candidate_riou_selection_is_non_deployable_oracle=True,
            eligible_for_runtime_policy=False),
        candidate_counts=_candidate_counts(rows),
        stream_metrics=stream_metrics,
        both_present_component_decomposition=_hybrid_summary(rows),
        dino_refinability=_residual_summary(rows),
        candidate_oracle_transition_analysis=_transition_summary(rows),
        capacity_summary=_capacity_summary(stream_metrics),
        measurement_validity=measurement,
        frame_rows=[row['report'] for row in rows],
        next_stage=dict(
            recommended_action=(
                'REVIEW_SOURCE_VAL_CAPACITY_THEN_PREREGISTER_BOUNDED_REFINER'
                if evidence_role == 'source-val' else
                'DIAGNOSTIC_ONLY_DO_NOT_SELECT_REFINER_FROM_FIXED_TARGET'),
            geometry_refiner_trained=False,
            localization_quality_head_trained=False,
            temporal_loss_trained=False,
            eligible_for_unknown_sequence_claim=False),
        decision=(
            'SOURCE_VAL_GEOMETRY_CAPACITY_AUDIT_COMPLETE'
            if evidence_role == 'source-val' else
            'FIXED_TARGET_GEOMETRY_CAPACITY_DIAGNOSTIC_ONLY'))


def main():
    args = parse_args()
    audit_path = Path(args.audit_json)
    audit_bytes = audit_path.read_bytes()
    payload = json.loads(audit_bytes)
    manifest = None
    manifest_bytes = None
    if args.measurement_validity_json:
        manifest_path = Path(args.measurement_validity_json)
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    result = audit_payload(
        payload, audit_bytes, Path(args.data_root) / args.split,
        args.evidence_role, args.require_frame_count,
        measurement_validity=manifest,
        measurement_validity_bytes=manifest_bytes)
    base._write_json_atomic(args.out_json, result)
    print('[geometry-audit] role={}'.format(result['evidence_role']))
    print('[geometry-audit] frames={}'.format(result['input']['frame_count']))
    print('[geometry-audit] candidate_counts={}'.format(
        result['candidate_counts']))
    print('[geometry-audit] capacity={}'.format(
        result['capacity_summary']))
    print('[geometry-audit] both_present_decomposition={}'.format(
        result['both_present_component_decomposition']))
    print('[geometry-audit] dino_refinability_summary={}'.format(
        _compact_refinability_summary(result['dino_refinability'])))
    if result['measurement_validity'] is not None:
        print('[geometry-audit] measurement_validity_stream_metrics={}'.format(
            result['measurement_validity']['stream_metrics']))
    print('[geometry-audit] decision={}'.format(result['decision']))


if __name__ == '__main__':
    main()
