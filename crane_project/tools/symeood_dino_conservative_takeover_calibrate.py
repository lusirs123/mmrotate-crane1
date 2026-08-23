"""Select conservative takeover parameters on source validation only."""

import argparse
import functools
import itertools
import json
import logging
import os
from pathlib import Path

import numpy as np
from crane_project.tools.eval_crane_offline import (
    METRIC_PROTOCOL_VERSION, CraneOfflineEvaluator, compute_riou,
    parse_dota_txt, parse_seq_frame)
from crane_project.utils.conservative_takeover import (
    ConservativeTakeoverSelector)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-audit-json', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument('--split', default='val', choices=['val'])
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _record_box(record, key):
    value = record.get(key)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).reshape(-1)[:6]


@functools.lru_cache(maxsize=None)
def _ground_truth_path(path):
    boxes = parse_dota_txt(path)
    if not boxes:
        return None
    return np.asarray(boxes[0], dtype=np.float32)


def _ground_truth(record, annotation_dir):
    filename = Path(record['filename']).stem + '.txt'
    return _ground_truth_path(str(annotation_dir / filename))


def _rotated_iou(first, second):
    return compute_riou(first, second)


def _evaluate(records, annotation_dir, parameters=None):
    selector = (None if parameters is None else
                ConservativeTakeoverSelector(**parameters))
    temporal_records = []
    selected_sources = []
    lost = []
    gained = []
    previous_source = {}
    switches = 0

    for record in records:
        sym = _record_box(record, 'sym_eood_box')
        dino = _record_box(record, 'dino_native_box')
        sequence = str(record['sequence'])
        frame = int(record['frame'])
        if selector is None:
            selected = sym if sym is not None else dino
            source = 'sym_eood' if sym is not None else 'dino_native'
        else:
            decision = selector.select(sym, dino, sequence, frame)
            selected = decision['selected']
            source = decision['selected_source']
        selected_sources.append(source)
        if sequence in previous_source and previous_source[sequence] != source:
            switches += 1
        previous_source[sequence] = source

        gt = _ground_truth(record, annotation_dir)
        baseline_hit = bool(
            sym is not None and gt is not None
            and _rotated_iou(sym[:5], gt[:5]) >= 0.5)
        selected_hit = bool(
            selected is not None and gt is not None
            and _rotated_iou(selected[:5], gt[:5]) >= 0.5)
        frame_key = '{}|{}'.format(sequence, frame)
        if baseline_hit and not selected_hit:
            lost.append(frame_key)
        if selected_hit and not baseline_hit:
            gained.append(frame_key)

        domain, seq_id, parsed_frame = parse_seq_frame(record['filename'])
        temporal_records.append(dict(
            domain=domain, seq_id=seq_id, frame_id=parsed_frame,
            pred_box=None if selected is None else selected[:5],
            gt_box=None if gt is None else gt[:5],
            score=0.0 if selected is None else float(selected[5]),
            plc_rope=None))

    logging.disable(logging.CRITICAL)
    try:
        metrics = CraneOfflineEvaluator(mode='test').evaluate_records(
            temporal_records)
    finally:
        logging.disable(logging.NOTSET)
    return dict(
        metrics=metrics,
        lost_frame_keys=lost,
        gained_frame_keys=gained,
        switches=int(switches),
        dino_selected_frames=int(sum(
            source == 'dino_native' for source in selected_sources)))


def _metric(summary, name, default):
    return float(summary['metrics'].get(name, default))


def _source_gate(candidate, baseline):
    checks = dict(
        exact_old_correct_retention=(
            len(candidate['lost_frame_keys']) == 0),
        dino_is_active=(candidate['dino_selected_frames'] > 0),
        has_source_gain=(len(candidate['gained_frame_keys']) > 0),
        sim_angle_safe=(
            _metric(candidate, 'sim/A-RMSE(deg)', 90.0)
            <= _metric(baseline, 'sim/A-RMSE(deg)', 90.0) + 0.25),
        real_dfr_safe=(
            _metric(candidate, 'real/DFR(%/frame)', 100.0)
            <= _metric(baseline, 'real/DFR(%/frame)', 100.0) + 0.5),
        sim_dfr_safe=(
            _metric(candidate, 'sim/DFR(%/frame)', 100.0)
            <= _metric(baseline, 'sim/DFR(%/frame)', 100.0) + 0.5),
        real_aci_safe=(
            _metric(candidate, 'real/ACI', 0.0)
            >= _metric(baseline, 'real/ACI', 0.0) - 0.01),
        sim_aci_safe=(
            _metric(candidate, 'sim/ACI', 0.0)
            >= _metric(baseline, 'sim/ACI', 0.0) - 0.01),
        real_mcml_safe=(
            _metric(candidate, 'real/MCML_max(frames)', 10**9)
            <= _metric(baseline, 'real/MCML_max(frames)', 10**9)),
        sim_mcml_safe=(
            _metric(candidate, 'sim/MCML_max(frames)', 10**9)
            <= _metric(baseline, 'sim/MCML_max(frames)', 10**9)))
    return dict(passed=all(checks.values()), checks=checks)


def _selection_key(row):
    summary = row['summary']
    metrics = summary['metrics']
    return (
        int(metrics.get('real/MCML_max(frames)', 10**9)),
        int(metrics.get('sim/MCML_max(frames)', 10**9)),
        -len(summary['gained_frame_keys']),
        int(summary['switches']),
        float(metrics.get('real/DFR(%/frame)', 100.0)),
        float(metrics.get('sim/A-RMSE(deg)', 90.0)))


def main():
    args = parse_args()
    source_path = Path(args.source_audit_json)
    payload = json.loads(source_path.read_text())
    records = list(payload.get('records') or [])
    if not records:
        raise RuntimeError('Source fusion audit has no records')
    annotation_dir = Path(args.data_root) / args.split / 'annfiles'
    if not annotation_dir.is_dir():
        raise RuntimeError('Annotation directory does not exist: {}'.format(
            annotation_dir))

    baseline = _evaluate(records, annotation_dir, parameters=None)
    candidates = []
    for values in itertools.product(
            (0.05, 0.10, 0.15, 0.20),
            (0.00, 0.025, 0.05),
            (1, 2), (0.10, 0.20), (10.0, 15.0)):
        enter, exit_margin, confirmations, diag, angle = values
        if exit_margin > enter:
            continue
        parameters = dict(
            enter_margin=enter, exit_margin=exit_margin,
            min_confirmations=confirmations,
            max_diag_change=diag, max_angle_change_deg=angle)
        summary = _evaluate(records, annotation_dir, parameters)
        gate = _source_gate(summary, baseline)
        candidates.append(dict(
            parameters=parameters, source_gate=gate, summary=summary))

    feasible = [row for row in candidates if row['source_gate']['passed']]
    feasible.sort(key=_selection_key)
    selected = feasible[0] if feasible else None
    output = dict(
        protocol='source_calibrated_conservative_takeover_v2',
        metric_protocol_version=METRIC_PROTOCOL_VERSION,
        selection_split='val',
        target_data_read=False,
        source_audit_json=os.fspath(source_path),
        source_frame_count=len(records),
        baseline=baseline,
        candidate_count=len(candidates),
        feasible_candidate_count=len(feasible),
        eligible_for_test=bool(selected is not None),
        selected_parameters=(None if selected is None else
                             selected['parameters']),
        source_gate=(None if selected is None else selected['source_gate']),
        selected_summary=(None if selected is None else selected['summary']),
        top_feasible=feasible[:20])
    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    temporary.replace(output_path)
    print('[calibration] frames={}'.format(len(records)))
    print('[calibration] candidates={} feasible={}'.format(
        len(candidates), len(feasible)))
    print('[calibration] eligible_for_test={}'.format(
        output['eligible_for_test']))
    print('[calibration] selected_parameters={}'.format(
        output['selected_parameters']))
    print('[calibration] out={}'.format(output_path))


if __name__ == '__main__':
    main()
