"""Source-only support audit for lane-isolated conditional DINO rescue V3."""

import argparse
import functools
import itertools
import json
import logging
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from crane_project.tools.eval_crane_offline import (
    METRIC_PROTOCOL_VERSION, CraneOfflineEvaluator, compute_riou,
    parse_dota_txt, parse_seq_frame)
from crane_project.utils.conservative_takeover import geometry_change
from crane_project.utils.lane_isolated_conditional_dino import (
    LaneIsolatedConditionalDinoSelector, normalized_diagonal)


PROTOCOL = 'source_calibrated_lane_isolated_conditional_dino_v3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-audit-json', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument('--split', default='val', choices=['val'])
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--require-frame-count', type=int, default=738)
    return parser.parse_args()


def _record_box(record, key, required=False):
    if required and key not in record:
        raise RuntimeError(
            'Source audit is missing {}; recollect it with the V3 '
            'source-val config'.format(key))
    value = record.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < 6 or not np.isfinite(array[:6]).all():
        if required:
            raise RuntimeError('Invalid {} in source audit'.format(key))
        return None
    if float(array[2]) <= 0.0 or float(array[3]) <= 0.0:
        if required:
            raise RuntimeError('Invalid {} geometry'.format(key))
        return None
    return array[:6].copy()


@functools.lru_cache(maxsize=None)
def _ground_truth_path(path):
    boxes = parse_dota_txt(path)
    if not boxes:
        return None
    return np.asarray(boxes[0], dtype=np.float32)


def _ground_truth(record, annotation_dir):
    filename = Path(record['filename']).stem + '.txt'
    return _ground_truth_path(str(annotation_dir / filename))


@functools.lru_cache(maxsize=None)
def _image_shape(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('Cannot read source image: {}'.format(path))
    return int(image.shape[0]), int(image.shape[1])


def _resolved_image_path(record, image_dir):
    candidate = image_dir / Path(record['filename']).name
    if not candidate.is_file():
        raise RuntimeError('Source image does not exist: {}'.format(candidate))
    return candidate


def _hit(box, gt):
    return bool(
        box is not None and gt is not None
        and compute_riou(box[:5], gt[:5]) >= 0.5)


def _temporal_record(record, selected, gt):
    domain, seq_id, frame = parse_seq_frame(record['filename'])
    return dict(
        domain=domain,
        seq_id=seq_id,
        frame_id=frame,
        pred_box=None if selected is None else selected[:5],
        gt_box=None if gt is None else gt[:5],
        score=0.0 if selected is None else float(selected[5]),
        plc_rope=None)


def _offline_metrics(records):
    logging.disable(logging.CRITICAL)
    try:
        return CraneOfflineEvaluator(mode='test').evaluate_records(records)
    finally:
        logging.disable(logging.NOTSET)


def _baseline(records, annotation_dir, lane):
    temporal = []
    hit_keys = set()
    for record in records:
        key = ('sym_eood_original_box' if lane == 'sym_eood' else
               'dino_native_box')
        box = _record_box(record, key)
        gt = _ground_truth(record, annotation_dir)
        frame_key = '{}|{}'.format(record['sequence'], int(record['frame']))
        if _hit(box, gt):
            hit_keys.add(frame_key)
        temporal.append(_temporal_record(record, box, gt))
    return dict(
        lane=lane,
        hit_count=len(hit_keys),
        hit_frame_keys=sorted(hit_keys),
        metrics=_offline_metrics(temporal))


def _evaluate(records, annotation_dir, image_dir, parameters):
    selector = LaneIsolatedConditionalDinoSelector(**parameters)
    temporal = []
    hit_keys = set()
    invoked = 0
    valid_measurements = 0
    sources = Counter()
    triggers = Counter()
    risks = Counter()
    decisions = []
    for record in records:
        sym = _record_box(record, 'sym_eood_original_box')
        dino = _record_box(record, 'dino_native_box')
        gt = _ground_truth(record, annotation_dir)
        image_path = _resolved_image_path(record, image_dir)
        shape = _image_shape(str(image_path))
        decision = selector.select(
            sym, dino, shape, record['sequence'], int(record['frame']))
        selected = decision['selected']
        frame_key = '{}|{}'.format(record['sequence'], int(record['frame']))
        if _hit(selected, gt):
            hit_keys.add(frame_key)
        invoked += int(decision['invoke_dino'])
        valid_measurements += int(decision['measurement_valid'])
        sources[decision['selected_source']] += 1
        triggers.update(decision['trigger_reasons'])
        risks.update(decision['risk_reasons'])
        decisions.append(dict(
            frame_key=frame_key,
            invoke_dino=bool(decision['invoke_dino']),
            trigger_reasons=list(decision['trigger_reasons']),
            selected_source=decision['selected_source'],
            measurement_valid=bool(decision['measurement_valid'])))
        temporal.append(_temporal_record(record, selected, gt))
    return dict(
        hit_count=len(hit_keys),
        hit_frame_keys=sorted(hit_keys),
        metrics=_offline_metrics(temporal),
        dino_invocation_count=int(invoked),
        dino_invocation_rate=float(invoked / max(len(records), 1)),
        measurement_valid_count=int(valid_measurements),
        selected_source_counts=dict(sources),
        trigger_reason_counts=dict(triggers),
        risk_reason_counts=dict(risks),
        decisions=decisions)


def _metric(summary, name, default):
    return float(summary['metrics'].get(name, default))


def _source_gate(candidate, sym_baseline, dino_baseline, frame_count):
    candidate_hits = set(candidate['hit_frame_keys'])
    sym_hits = set(sym_baseline['hit_frame_keys'])
    dino_hits = set(dino_baseline['hit_frame_keys'])
    lost_vs_sym = sorted(sym_hits - candidate_hits)
    lost_vs_dino = sorted(dino_hits - candidate_hits)
    gained_vs_sym = sorted(candidate_hits - sym_hits)
    checks = dict(
        exact_sym_correct_retention=(not lost_vs_sym),
        exact_native_dino_correct_retention=(not lost_vs_dino),
        has_source_gain_vs_sym=bool(gained_vs_sym),
        dino_is_conditionally_active=(
            0 < candidate['dino_invocation_count'] < frame_count),
        sim_angle_safe=(
            _metric(candidate, 'sim/A-RMSE(deg)', 90.0)
            <= _metric(sym_baseline, 'sim/A-RMSE(deg)', 90.0) + 0.25),
        sim_riou_safe=(
            _metric(candidate, 'sim/mean_RIoU', 0.0)
            >= _metric(sym_baseline, 'sim/mean_RIoU', 0.0) - 0.01),
        real_dfr_safe=(
            _metric(candidate, 'real/DFR(%/frame)', 100.0)
            <= min(_metric(sym_baseline, 'real/DFR(%/frame)', 100.0),
                   _metric(dino_baseline, 'real/DFR(%/frame)', 100.0)) + 0.5),
        sim_dfr_safe=(
            _metric(candidate, 'sim/DFR(%/frame)', 100.0)
            <= _metric(sym_baseline, 'sim/DFR(%/frame)', 100.0) + 0.5),
        real_mcml_non_amplification=(
            _metric(candidate, 'real/MCML_max(frames)', 10**9)
            <= min(_metric(sym_baseline, 'real/MCML_max(frames)', 10**9),
                   _metric(dino_baseline, 'real/MCML_max(frames)', 10**9))),
        sim_mcml_non_amplification=(
            _metric(candidate, 'sim/MCML_max(frames)', 10**9)
            <= _metric(sym_baseline, 'sim/MCML_max(frames)', 10**9)))
    return dict(
        passed=all(checks.values()),
        checks=checks,
        lost_vs_sym_frame_keys=lost_vs_sym,
        lost_vs_native_dino_frame_keys=lost_vs_dino,
        gained_vs_sym_frame_keys=gained_vs_sym)


def _selection_key(row):
    summary = row['summary']
    return (
        int(_metric(summary, 'real/MCML_max(frames)', 10**9)),
        int(_metric(summary, 'sim/MCML_max(frames)', 10**9)),
        -int(summary['hit_count']),
        float(summary['dino_invocation_rate']),
        float(_metric(summary, 'real/DFR(%/frame)', 100.0)))


def _quantile_thresholds(records, image_dir):
    ratios = []
    dino_diag_changes = []
    dino_angle_changes = []
    previous = {}
    for record in records:
        sym = _record_box(record, 'sym_eood_original_box')
        image_path = _resolved_image_path(record, image_dir)
        if sym is not None:
            ratios.append(normalized_diagonal(
                sym, _image_shape(str(image_path))))
        sequence = str(record['sequence'])
        frame = int(record['frame'])
        dino = _record_box(record, 'dino_native_box')
        old = previous.get(sequence)
        if (dino is not None and old is not None
                and frame == old[0] + 1 and old[1] is not None):
            change = geometry_change(old[1], dino)
            dino_diag_changes.append(change['diag_change'])
            dino_angle_changes.append(change['angle_change_deg'])
        previous[sequence] = (frame, dino)
    if not ratios:
        raise RuntimeError('Source audit contains no valid raw SymEOOD boxes')
    quantiles = np.quantile(
        np.asarray(ratios, dtype=np.float64),
        np.asarray([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]))
    small_thresholds = sorted(set(
        [0.0] + [round(float(value), 8) for value in quantiles]))
    dino_diag_limit = max(
        0.10, float(np.quantile(dino_diag_changes, 0.95))
        if dino_diag_changes else 0.10)
    dino_angle_limit = max(
        10.0, float(np.quantile(dino_angle_changes, 0.95))
        if dino_angle_changes else 10.0)
    return small_thresholds, dino_diag_limit, dino_angle_limit


def _validate_source_records(payload, records, image_dir,
                             required_frame_count):
    if payload.get('protocol') != 'source_owned_geometry_union_v2':
        raise RuntimeError(
            'V3 calibration requires an all-lane source collection, not a '
            'routed or target audit')
    metadata = dict(payload.get('metadata') or {})
    if metadata.get('fusion_policy') != 'sym_eood_proposal_dino_roi_union':
        raise RuntimeError('Source audit did not use the unified lane policy')
    if bool(metadata.get('conditional_dino_enabled', False)):
        raise RuntimeError('Source audit already used conditional DINO')
    if int(payload.get('frame_count', len(records))) != len(records):
        raise RuntimeError('Source audit frame_count does not match records')
    if len(records) != int(required_frame_count):
        raise RuntimeError(
            'Expected {} source-val records, got {}'.format(
                required_frame_count, len(records)))
    seen = set()
    observed_stems = set()
    closed_sequences = set()
    active_sequence = None
    previous_frame = None
    for record in records:
        for key in ('filename', 'sequence', 'frame'):
            if key not in record:
                raise RuntimeError('Source audit record is missing ' + key)
        _record_box(record, 'sym_eood_original_box', required=True)
        image_path = _resolved_image_path(record, image_dir)
        frame_key = (str(record['sequence']), int(record['frame']))
        if frame_key in seen:
            raise RuntimeError('Duplicate source frame: {}'.format(frame_key))
        seen.add(frame_key)
        observed_stems.add(image_path.stem)
        if image_path.parent.name != 'images':
            raise RuntimeError('Unexpected source image path')
        sequence, frame = frame_key
        if sequence != active_sequence:
            if sequence in closed_sequences:
                raise RuntimeError(
                    'Source audit sequence order is not contiguous: '
                    + sequence)
            if active_sequence is not None:
                closed_sequences.add(active_sequence)
            active_sequence = sequence
            previous_frame = None
        if previous_frame is not None and frame <= previous_frame:
            raise RuntimeError(
                'Source audit frames are not strictly increasing in '
                + sequence)
        previous_frame = frame
    image_suffixes = {'.jpg', '.jpeg', '.png', '.bmp'}
    expected_stems = {
        path.stem for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_suffixes}
    if observed_stems != expected_stems:
        missing = sorted(expected_stems - observed_stems)[:5]
        extra = sorted(observed_stems - expected_stems)[:5]
        raise RuntimeError(
            'Source audit does not exactly match val/images; missing={} '
            'extra={}'.format(missing, extra))


def _diagnostic_key(row):
    gate = row['source_gate']
    passed_checks = sum(bool(value) for value in gate['checks'].values())
    losses = (len(gate['lost_vs_sym_frame_keys'])
              + len(gate['lost_vs_native_dino_frame_keys']))
    return (
        -passed_checks,
        losses,
        -len(gate['gained_vs_sym_frame_keys']),
        float(row['summary']['dino_invocation_rate']))


def _compact_row(row):
    summary = dict(row['summary'])
    summary.pop('decisions', None)
    summary.pop('hit_frame_keys', None)
    return dict(
        parameters=dict(row['parameters']),
        source_gate=dict(row['source_gate']),
        summary=summary)


def main():
    args = parse_args()
    source_path = Path(args.source_audit_json)
    payload = json.loads(source_path.read_text())
    records = list(payload.get('records') or [])
    if not records:
        raise RuntimeError('Source fusion audit has no records')
    split_root = Path(args.data_root) / args.split
    annotation_dir = split_root / 'annfiles'
    image_dir = split_root / 'images'
    if not annotation_dir.is_dir() or not image_dir.is_dir():
        raise RuntimeError('Source val data directories do not exist')
    _validate_source_records(
        payload, records, image_dir, args.require_frame_count)

    sym_baseline = _baseline(records, annotation_dir, 'sym_eood')
    dino_baseline = _baseline(records, annotation_dir, 'dino_native')
    small_thresholds, dino_diag_limit, dino_angle_limit = (
        _quantile_thresholds(records, image_dir))
    candidates = []
    for small_ratio, sym_diag, sym_angle in itertools.product(
            small_thresholds, (0.10, 0.20, 0.30, 1e6),
            (10.0, 15.0, 25.0, 180.0)):
        parameters = dict(
            small_diag_ratio=float(small_ratio),
            max_sym_diag_change=float(sym_diag),
            max_sym_angle_change_deg=float(sym_angle),
            max_dino_diag_change=float(dino_diag_limit),
            max_dino_angle_change_deg=float(dino_angle_limit))
        summary = _evaluate(
            records, annotation_dir, image_dir, parameters)
        gate = _source_gate(
            summary, sym_baseline, dino_baseline, len(records))
        candidates.append(dict(
            parameters=parameters, source_gate=gate, summary=summary))

    feasible = [row for row in candidates if row['source_gate']['passed']]
    feasible.sort(key=_selection_key)
    diagnostic = sorted(candidates, key=_diagnostic_key)
    selected = feasible[0] if feasible else None
    output = dict(
        protocol=PROTOCOL,
        metric_protocol_version=METRIC_PROTOCOL_VERSION,
        selection_split='val',
        target_data_read=False,
        source_audit_json=os.fspath(source_path),
        source_frame_count=len(records),
        sym_eood_baseline=sym_baseline,
        native_dino_baseline=dino_baseline,
        candidate_count=len(candidates),
        feasible_candidate_count=len(feasible),
        eligible_for_fixed_test=bool(selected is not None),
        selected_parameters=(None if selected is None else
                             selected['parameters']),
        source_gate=(None if selected is None else selected['source_gate']),
        selected_summary=(None if selected is None else selected['summary']),
        top_feasible=[_compact_row(row) for row in feasible[:20]],
        top_candidates=[_compact_row(row) for row in diagnostic[:20]])
    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    temporary.replace(output_path)
    print('[lane-v3] frames={}'.format(len(records)))
    print('[lane-v3] candidates={} feasible={}'.format(
        len(candidates), len(feasible)))
    print('[lane-v3] eligible_for_fixed_test={}'.format(
        output['eligible_for_fixed_test']))
    print('[lane-v3] selected_parameters={}'.format(
        output['selected_parameters']))
    if selected is not None:
        print('[lane-v3] dino_invocation_rate={:.6f}'.format(
            selected['summary']['dino_invocation_rate']))
    else:
        best = diagnostic[0]
        failed = sorted(
            name for name, passed in best['source_gate']['checks'].items()
            if not passed)
        print('[lane-v3] best_candidate_failed={}'.format(','.join(failed)))
    print('[lane-v3] out={}'.format(output_path))


if __name__ == '__main__':
    main()
