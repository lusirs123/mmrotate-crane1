#!/usr/bin/env python3
"""Explain the failed Base-V3 current-only source-val gate from saved data."""

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from crane_project.tools.eval_crane_offline import (
    angle_diff, compute_riou, parse_dota_txt, parse_seq_frame)
from crane_project.tools.symeood_dino_dual_tower_v2_audit import (
    _load_results)


PROTOCOL = 'base_v3_current_only_source_val_failure_attribution_v1'
CONTRACT_PROTOCOL = (
    'base_v3_current_only_source_val_failure_attribution_contract_v1')
GATE_PROTOCOL = 'base_v3_epoch9_current_only_official_source_val_gate_v1'
HISTORY_PROTOCOL = 'symeood_dino_seq11_v2_history_contribution_audit_v2'
RECEIPT_PROTOCOL = 'mmdet_runtime_result_order_identity_v1'
IOU_THRESHOLD = 0.5
ANGLE_CENTER_THRESHOLD = 10.0
ANGLE_PENALTY_DEG = 90.0
TOP_ANGLE_FRAMES = 12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate-report', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument(
        '--full-results',
        help='defaults to the immutable full Base V3 source-gate input')
    parser.add_argument(
        '--k1-results',
        help='defaults to the immutable K1 source-gate reference input')
    parser.add_argument('--source-val-audit', required=True)
    parser.add_argument('--seq11-history-report', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument('--overlay-dir', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _identity(path):
    absolute = Path(path).resolve()
    if not absolute.is_file():
        raise RuntimeError('Missing required input: ' + os.fspath(absolute))
    digest = hashlib.sha256()
    with absolute.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=os.fspath(absolute), sha256=digest.hexdigest(),
                size_bytes=absolute.stat().st_size)


def _json(path, protocol=None):
    identity = _identity(path)
    with open(identity['path'], 'r', encoding='utf-8') as stream:
        payload = json.load(stream)
    if protocol is not None and payload.get('protocol') != protocol:
        raise RuntimeError(
            'Unexpected protocol in {}: {!r}'.format(
                identity['path'], payload.get('protocol')))
    return identity, payload


def _verify_identity(observed_path, recorded, role):
    observed = _identity(observed_path)
    if (not isinstance(recorded, dict)
            or observed['sha256'] != recorded.get('sha256')):
        raise RuntimeError('Identity mismatch for ' + role)
    return observed


def _angle_geometry(box, gt):
    if box is None:
        return dict(
            prediction_present=False, center_error_px=None, riou=0.0,
            riou_hit=False, raw_angle_error_deg=None,
            angle_metric_state='missing_prediction_penalty',
            angle_metric_error_deg=ANGLE_PENALTY_DEG,
            angle_metric_squared_error_deg2=ANGLE_PENALTY_DEG ** 2)
    center = float(np.linalg.norm(box[:2] - gt[:2]))
    riou = float(compute_riou(box, gt))
    raw = abs(math.degrees(float(angle_diff(
        np.asarray([box[4]]), np.asarray([gt[4]]))[0])))
    if center < ANGLE_CENTER_THRESHOLD:
        metric_error = raw
        state = 'direct_periodic_angle_error'
    else:
        metric_error = ANGLE_PENALTY_DEG
        state = 'center_error_penalty'
    return dict(
        prediction_present=True, center_error_px=center, riou=riou,
        riou_hit=riou >= IOU_THRESHOLD, raw_angle_error_deg=raw,
        angle_metric_state=state,
        angle_metric_error_deg=metric_error,
        angle_metric_squared_error_deg2=metric_error ** 2)


def _split_segments(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['domain'], row['sequence'])].append(row)
    segments = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda item: item['frame'])
        current = []
        for row in ordered:
            if current and row['frame'] != current[-1]['frame'] + 1:
                segments.append(current)
                current = []
            current.append(row)
        if current:
            segments.append(current)
    return segments


def _failure_runs(rows, method):
    runs = []
    for segment in _split_segments(rows):
        current = []
        for row in segment:
            if not row[method]['riou_hit']:
                current.append(row)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
    return [dict(
        domain=run[0]['domain'], sequence=run[0]['sequence'],
        start_frame=run[0]['frame'], end_frame=run[-1]['frame'],
        length=len(run), frame_keys=[row['frame_key'] for row in run],
        terminal_within_contiguous_block=(
            not any(other['frame'] == run[-1]['frame'] + 1
                    and other['domain'] == run[-1]['domain']
                    and other['sequence'] == run[-1]['sequence']
                    for other in rows))) for run in runs]


def _audit_dino_boxes(path, frame_keys):
    identity, payload = _json(path, 'source_owned_geometry_union_v2')
    records = payload.get('records') or []
    by_key = {}
    for record in records:
        key = Path(record.get('filename', '')).stem
        value = record.get('dino_native_box')
        if not key or key in by_key:
            raise RuntimeError('Invalid DINO audit frame identity')
        if value is None:
            by_key[key] = None
        else:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            if array.size < 5 or not np.isfinite(array[:5]).all():
                raise RuntimeError('Invalid DINO OBB for ' + key)
            by_key[key] = array[:5]
    if set(by_key) != set(frame_keys):
        raise RuntimeError('DINO audit frame set differs from source-val')
    return identity, [by_key[key] for key in frame_keys]


def _polygon(box):
    if box is None:
        return None
    rect = ((float(box[0]), float(box[1])),
            (float(box[2]), float(box[3])), math.degrees(float(box[4])))
    return np.rint(cv2.boxPoints(rect)).astype(np.int32)


def _draw(image, boxes):
    colors = dict(
        GT=(0, 220, 0), K1=(255, 255, 0), DINO=(0, 165, 255),
        current_only=(0, 255, 255), full=(0, 0, 255))
    canvas = image.copy()
    y = 24
    for label in ('GT', 'K1', 'DINO', 'current_only', 'full'):
        polygon = _polygon(boxes[label])
        if polygon is not None:
            cv2.polylines(canvas, [polygon], True, colors[label], 2,
                          lineType=cv2.LINE_AA)
        cv2.putText(
            canvas, '{}: {}'.format(
                label, 'present' if polygon is not None else 'missing'),
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, colors[label], 2,
            cv2.LINE_AA)
        y += 21
    return canvas


def _write_exact(path, raw):
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != raw:
        raise RuntimeError('Refusing to overwrite different output: '
                           + os.fspath(output))
    if not output.exists():
        output.write_bytes(raw)
    return _identity(output)


def _angle_method_summary(rows, method):
    direct = [
        row[method]['raw_angle_error_deg'] for row in rows
        if row[method]['angle_metric_state'] ==
        'direct_periodic_angle_error']
    penalty = [
        row for row in rows if row[method]['angle_metric_state'] !=
        'direct_periodic_angle_error']
    total_squared = sum(
        row[method]['angle_metric_squared_error_deg2'] for row in rows)
    return dict(
        frame_count=len(rows),
        direct_angle_frame_count=len(direct),
        center_penalty_frame_count=sum(
            row[method]['angle_metric_state'] == 'center_error_penalty'
            for row in rows),
        missing_penalty_frame_count=sum(
            row[method]['angle_metric_state'] ==
            'missing_prediction_penalty' for row in rows),
        direct_angle_rmse_deg=(
            None if not direct else
            math.sqrt(sum(value ** 2 for value in direct) / len(direct))),
        total_squared_error_deg2=total_squared,
        penalty_squared_error_deg2=len(penalty) * ANGLE_PENALTY_DEG ** 2,
        penalty_squared_error_fraction=(
            0.0 if total_squared <= 0.0 else
            len(penalty) * ANGLE_PENALTY_DEG ** 2 / total_squared))


def audit(args):
    contract_id, contract = _json(args.contract, CONTRACT_PROTOCOL)
    expected_status = 'post_gate_explanatory_audit_frozen_before_attribution_output'
    if (contract.get('status') != expected_status
            or contract.get('training') is not False
            or contract.get('target_data_read') is not False
            or contract.get('fixed_test_read') is not False):
        raise RuntimeError('Invalid explanatory attribution contract')
    expected_prohibited = dict(
        threshold_tuning=True, epoch_selection=True,
        checkpoint_promotion=True, training_authorization=True,
        fixed_test_authorization=True)
    if dict(contract.get('prohibited_uses') or {}) != expected_prohibited:
        raise RuntimeError('Attribution evidence-use boundary changed')
    expected_selection = dict(
        include_every_full_hit_current_only_miss=True,
        include_top_positive_sim_angle_squared_error_contributors=
        TOP_ANGLE_FRAMES,
        include_seq11_full_vs_current_only_hit_changes=True)
    if dict(contract.get('selection') or {}) != expected_selection:
        raise RuntimeError('Attribution frame-selection policy changed')
    metric = dict(contract.get('metric_decomposition') or {})
    expected_metric = dict(
        riou_hit_threshold=IOU_THRESHOLD,
        source_center_threshold_px=15.0,
        sim_angle_center_threshold_px=ANGLE_CENTER_THRESHOLD,
        sim_angle_penalty_deg=ANGLE_PENALTY_DEG,
        angle_periodicity='le90_pi_periodic',
        mcml_boundary=(
            'same domain, same sequence, original frame difference exactly one'),
        missing_prediction_breaks_temporal_continuity=True)
    if metric != expected_metric:
        raise RuntimeError('Attribution metric contract changed')

    gate_id, gate = _json(args.gate_report, GATE_PROTOCOL)
    if (gate.get('decision') != 'STOP_CURRENT_ONLY_SOURCE_NON_REGRESSION_FAILED'
            or gate.get('passed') is not False
            or gate.get('fixed_test_read') is not False):
        raise RuntimeError('Attribution requires the failed source-val gate')
    gate_inputs = dict(gate.get('inputs') or {})
    current_id = _verify_identity(
        gate_inputs['candidate_results']['path'],
        gate_inputs['candidate_results'], 'current-only results')
    receipt_id, receipt = _json(
        gate_inputs['candidate_result_receipt']['path'], RECEIPT_PROTOCOL)
    if receipt_id['sha256'] != gate_inputs['candidate_result_receipt']['sha256']:
        raise RuntimeError('Current-only receipt changed after gate')
    full_gate_id, full_gate = _json(
        gate_inputs['full_base_v3_source_gate']['path'])
    if full_gate_id['sha256'] != gate_inputs['full_base_v3_source_gate']['sha256']:
        raise RuntimeError('Full Base V3 source gate changed')
    full_input = dict(full_gate.get('input') or {})
    full_id = _verify_identity(
        args.full_results or full_input.get('candidate_results'),
        dict(path=full_input.get('candidate_results'),
             sha256=full_input.get('candidate_results_sha256')),
        'full Base V3 results')
    k1_id = _verify_identity(
        args.k1_results or full_input.get('sym_reference_results'),
        dict(path=full_input.get('sym_reference_results'),
             sha256=full_input.get('sym_reference_results_sha256')),
        'ordinary K1 results')
    source_audit_id = _verify_identity(
        args.source_val_audit, gate_inputs['source_val_audit'],
        'source-val DINO audit')
    history_id, history = _json(args.seq11_history_report, HISTORY_PROTOCOL)
    if (history.get('passed') is not True
            or history.get('fixed_test_read') is not False):
        raise RuntimeError('seq11 strict history report is invalid')

    _, current_boxes = _load_results(current_id['path'])
    _, full_boxes = _load_results(full_id['path'])
    _, k1_boxes = _load_results(k1_id['path'])
    ann_root = Path(args.data_root).resolve() / 'val' / 'annfiles'
    ann_paths = sorted(ann_root.glob('*.txt'), key=lambda path: path.name)
    if len(ann_paths) != 738:
        raise RuntimeError('Official source-val must contain 738 annotations')
    frame_keys = [path.stem for path in ann_paths]
    receipt_keys = [
        row.get('frame_key') for row in receipt.get('runtime_dataset_order') or []]
    if receipt_keys != frame_keys:
        raise RuntimeError('Runtime result order differs from annotations')
    dino_id, dino_boxes = _audit_dino_boxes(
        source_audit_id['path'], frame_keys)

    rows = []
    for index, path in enumerate(ann_paths):
        gt = parse_dota_txt(os.fspath(path))
        if len(gt) != 1:
            raise RuntimeError('Expected one source-val GT: ' + os.fspath(path))
        domain, sequence, frame = parse_seq_frame(path.name)
        full_geometry = _angle_geometry(full_boxes[index], gt[0])
        current_geometry = _angle_geometry(current_boxes[index], gt[0])
        rows.append(dict(
            result_index=index, frame_key=path.stem, domain=domain,
            sequence=sequence, frame=frame,
            full=full_geometry, current_only=current_geometry,
            angle_squared_error_delta_deg2=(
                current_geometry['angle_metric_squared_error_deg2']
                - full_geometry['angle_metric_squared_error_deg2'])))

    sim_rows = [row for row in rows if row['domain'] == 'sim']
    full_angle_sq = sum(
        row['full']['angle_metric_squared_error_deg2'] for row in sim_rows)
    current_angle_sq = sum(
        row['current_only']['angle_metric_squared_error_deg2'] for row in sim_rows)
    transition_rows = defaultdict(lambda: dict(count=0, squared_error_delta_deg2=0.0))
    for row in sim_rows:
        transition = '{} -> {}'.format(
            row['full']['angle_metric_state'],
            row['current_only']['angle_metric_state'])
        transition_rows[transition]['count'] += 1
        transition_rows[transition]['squared_error_delta_deg2'] += row[
            'angle_squared_error_delta_deg2']
    angle_summary = dict(
        sim_frame_count=len(sim_rows),
        full_rmse_deg=math.sqrt(full_angle_sq / len(sim_rows)),
        current_only_rmse_deg=math.sqrt(current_angle_sq / len(sim_rows)),
        rmse_delta_deg=(math.sqrt(current_angle_sq / len(sim_rows))
                        - math.sqrt(full_angle_sq / len(sim_rows))),
        squared_error_delta_deg2=current_angle_sq - full_angle_sq,
        full=_angle_method_summary(sim_rows, 'full'),
        current_only=_angle_method_summary(sim_rows, 'current_only'),
        state_transition_contributions=dict(sorted(transition_rows.items())),
        full_state_counts={state: sum(
            row['full']['angle_metric_state'] == state for row in sim_rows)
            for state in ('direct_periodic_angle_error',
                          'center_error_penalty',
                          'missing_prediction_penalty')},
        current_only_state_counts={state: sum(
            row['current_only']['angle_metric_state'] == state
            for row in sim_rows)
            for state in ('direct_periodic_angle_error',
                          'center_error_penalty',
                          'missing_prediction_penalty')})
    if (not math.isclose(angle_summary['full_rmse_deg'],
                         gate['full_base_v3_reference_metrics'][
                             'sim/A-RMSE(deg)'], abs_tol=5e-4)
            or not math.isclose(angle_summary['current_only_rmse_deg'],
                                gate['candidate_metrics']['sim/A-RMSE(deg)'],
                                abs_tol=5e-4)):
        raise RuntimeError('Per-frame angle decomposition does not reproduce gate')

    full_runs = _failure_runs(rows, 'full')
    current_runs = _failure_runs(rows, 'current_only')
    mcml_by_domain = {domain: dict(
        full=max([run['length'] for run in full_runs
                  if run['domain'] == domain] or [0]),
        current_only=max([run['length'] for run in current_runs
                          if run['domain'] == domain] or [0]))
        for domain in ('real', 'sim')}
    reproduced_riou = {}
    for domain in ('real', 'sim'):
        domain_rows = [row for row in rows if row['domain'] == domain]
        for method, gate_key in (
                ('full', 'full_base_v3_reference_metrics'),
                ('current_only', 'candidate_metrics')):
            observed = float(np.mean(
                [row[method]['riou'] for row in domain_rows]))
            expected = float(gate[gate_key][domain + '/mean_RIoU'])
            if not math.isclose(observed, expected, abs_tol=5e-4):
                raise RuntimeError(
                    '{} {} RIoU does not reproduce gate'.format(
                        domain, method))
            reproduced_riou[domain + '/' + method] = observed
        for method, gate_key in (
                ('full', 'full_base_v3_reference_metrics'),
                ('current_only', 'candidate_metrics')):
            expected = int(gate[gate_key][domain + '/MCML_max(frames)'])
            if mcml_by_domain[domain][method] != expected:
                raise RuntimeError(
                    '{} {} MCML does not reproduce gate'.format(
                        domain, method))
    full_hit_current_miss = [
        row for row in rows
        if row['full']['riou_hit'] and not row['current_only']['riou_hit']]
    current_hit_full_miss = [
        row for row in rows
        if row['current_only']['riou_hit'] and not row['full']['riou_hit']]
    ranked_angle = sorted(
        sim_rows, key=lambda row: row['angle_squared_error_delta_deg2'],
        reverse=True)
    selected_keys = {
        row['frame_key'] for row in full_hit_current_miss}
    full_hit_current_miss_keys = set(selected_keys)
    selected_keys.update(
        row['frame_key'] for row in ranked_angle[:TOP_ANGLE_FRAMES]
        if row['angle_squared_error_delta_deg2'] > 0.0)

    image_root = Path(args.data_root).resolve() / 'val' / 'images'
    overlay_root = Path(args.overlay_dir).resolve()
    overlay_root.mkdir(parents=True, exist_ok=True)
    overlay_rows = []
    for row in rows:
        if row['frame_key'] not in selected_keys:
            continue
        index = row['result_index']
        filename = receipt['runtime_dataset_order'][index]['dataset_filename']
        image_path = image_root / Path(filename).name
        image = cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Cannot read source-val image: ' + str(image_path))
        gt = parse_dota_txt(os.fspath(ann_paths[index]))[0]
        output = overlay_root / (row['frame_key'] + '_attribution.png')
        if output.exists():
            raise RuntimeError('Refusing to overwrite overlay: ' + str(output))
        canvas = _draw(image, dict(
            GT=gt, K1=k1_boxes[index], DINO=dino_boxes[index],
            current_only=current_boxes[index], full=full_boxes[index]))
        if not cv2.imwrite(os.fspath(output), canvas):
            raise RuntimeError('Failed to write overlay: ' + str(output))
        overlay_rows.append(dict(
            frame_key=row['frame_key'], path=os.fspath(output),
            sha256=_identity(output)['sha256'],
            full_hit_current_only_miss=(
                row['frame_key'] in full_hit_current_miss_keys),
            angle_squared_error_delta_deg2=
            row['angle_squared_error_delta_deg2']))

    seq11_changed = list((history.get('key_frame_summary') or {}).get(
        'history_changed_hit_frames') or [])
    seq11_overlays = []
    for item in (history.get('key_frame_summary') or {}).get('rows') or []:
        if item.get('frame_key') in seq11_changed and item.get('overlay'):
            recorded_overlay = dict(item['overlay'])
            observed_overlay = _verify_identity(
                recorded_overlay['path'], recorded_overlay,
                'seq11 overlay ' + item['frame_key'])
            seq11_overlays.append(dict(
                frame_key=item['frame_key'], overlay=observed_overlay,
                full_minus_current_only_riou=item.get(
                    'full_minus_current_only_riou')))

    return dict(
        protocol=PROTOCOL,
        evidence_boundary=(
            'post_gate_source_val_and_seq11_explanation_only'),
        inputs=dict(
            contract=contract_id, failed_gate=gate_id,
            current_only_results=current_id, full_results=full_id,
            k1_results=k1_id, dino_source_val_audit=dino_id,
            current_only_result_receipt=receipt_id,
            seq11_history_report=history_id,
            ann_dir=os.fspath(ann_root), frame_count=len(rows)),
        angle_metric_decomposition=angle_summary,
        metric_reproduction=dict(
            mean_RIoU=reproduced_riou,
            MCML_max=mcml_by_domain,
            tolerance=5e-4,
            passed=True),
        riou_failure_attribution=dict(
            full_hit_current_only_miss_count=len(full_hit_current_miss),
            full_hit_current_only_miss_frames=[
                row['frame_key'] for row in full_hit_current_miss],
            current_only_hit_full_miss_count=len(current_hit_full_miss),
            current_only_hit_full_miss_frames=[
                row['frame_key'] for row in current_hit_full_miss]),
        failure_runs=dict(
            full=full_runs, current_only=current_runs,
            full_mcml_max=max([run['length'] for run in full_runs] or [0]),
            current_only_mcml_max=max(
                [run['length'] for run in current_runs] or [0]),
            by_domain=mcml_by_domain),
        top_positive_sim_angle_contributors=[dict(
            frame_key=row['frame_key'], frame=row['frame'],
            sequence=row['sequence'],
            squared_error_delta_deg2=row['angle_squared_error_delta_deg2'],
            full=row['full'], current_only=row['current_only'])
            for row in ranked_angle[:TOP_ANGLE_FRAMES]
            if row['angle_squared_error_delta_deg2'] > 0.0],
        source_val_overlays=overlay_rows,
        seq11_cross_evidence=dict(
            full_vs_current_only_hit_change_count=len(seq11_changed),
            frames=seq11_changed, overlays=seq11_overlays,
            interpretation_role=(
                'same-video evidence of local history harm; not an '
                'independent generalization estimate')),
        per_frame=rows,
        interpretation_policy=dict(
            failed_gate_decision_unchanged=True,
            seq11_dataset_has_diagnostic_value=True,
            current_only_promotion_authorized=False,
            threshold_or_epoch_reselection_authorized=False,
            training_authorized=False),
        training_run=False, optimizer_steps=0,
        target_data_read=False, fixed_test_read=False,
        decision=(
            'CURRENT_ONLY_FAILURE_ATTRIBUTION_READY_'
            'NO_MODEL_CHANGE_AUTHORIZED'))


def main():
    args = parse_args()
    report = audit(args)
    raw = (json.dumps(report, indent=2, ensure_ascii=False) + '\n').encode(
        'utf-8')
    output = _write_exact(args.out_json, raw)
    angle = report['angle_metric_decomposition']
    riou = report['riou_failure_attribution']
    print('[current-only-attribution] output={}'.format(output['path']))
    print('[current-only-attribution] decision={}'.format(report['decision']))
    print('[current-only-attribution] sim A-RMSE full={:.4f} '
          'current_only={:.4f} delta={:.4f}'.format(
              angle['full_rmse_deg'], angle['current_only_rmse_deg'],
              angle['rmse_delta_deg']))
    print('[current-only-attribution] full-hit/current-miss={} '
          'current-hit/full-miss={}'.format(
              riou['full_hit_current_only_miss_count'],
              riou['current_only_hit_full_miss_count']))
    print('[current-only-attribution] overlays={}'.format(
        len(report['source_val_overlays'])))


if __name__ == '__main__':
    main()
