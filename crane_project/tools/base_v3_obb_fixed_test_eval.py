#!/usr/bin/env python3
"""Evaluate frozen Base V3 OBB reliability on the existing 992-frame TEST.

The source-val calibration is immutable.  The fixed-target attribution report
binds the formal K1, Base V3 full stream, and all-lane DINO audit used to
reconstruct the online inputs.  TEST ground truth is attached only after each
causal observation has been emitted.
"""

import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path

from crane_project.tools.base_v3_obb_external_sequence_eval import (
    matched_coverage, replay, state_report)
from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    COMPONENTS, _error_from_value, _measurement_value)
from crane_project.tools.base_v3_obb_observation_interface_v3 import (
    CALIBRATION_PROTOCOL)
from crane_project.tools.base_v3_obb_reliability_baseline import (
    AUDIT_PROTOCOL, _identity, _json_box, _load_annotations,
    _load_dino_audit, _load_result_boxes, _write_exact)


PROTOCOL = 'base_v3_obb_fixed_test_eval_v1'
CONTRACT_PROTOCOL = 'base_v3_obb_fixed_test_eval_contract_v1'
ATTRIBUTION_PROTOCOL = 'k1_anchor_fallback_attribution_audit_v1'


def _require_identity(identity, expected, role):
    if identity['sha256'] != expected:
        raise RuntimeError('{} identity mismatch: expected {}, got {}'.format(
            role, expected, identity['sha256']))


def _read_json(path, role):
    identity = _identity(path)
    with open(identity['path'], 'r', encoding='utf-8') as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(role + ' must be a JSON object')
    return identity, payload


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected fixed-TEST evaluation contract')
    if contract.get('fixed_test_read') is not True:
        raise ValueError('Contract must explicitly authorize fixed TEST')
    if contract.get('target_data_read') is not True:
        raise ValueError('Contract must explicitly declare target-data read')
    expected = contract.get('expected_inputs', {})
    if expected.get('runtime_calibration_protocol') != CALIBRATION_PROTOCOL:
        raise ValueError('Contract must bind the V3 calibration protocol')
    if expected.get('attribution_protocol') != ATTRIBUTION_PROTOCOL:
        raise ValueError('Contract must bind the attribution protocol')
    if expected.get('attribution_evidence_role') != 'fixed-target':
        raise ValueError('Contract must bind fixed-target attribution')
    if expected.get('all_lane_audit_protocol') != AUDIT_PROTOCOL:
        raise ValueError('Contract must bind the all-lane audit protocol')
    scope = contract.get('fixed_test_scope', {})
    if int(scope.get('frame_count', 0)) <= 0:
        raise ValueError('Contract must bind a positive fixed-TEST frame count')
    if sum(scope.get('domain_counts', {}).values()) != scope['frame_count']:
        raise ValueError('Fixed-TEST domain counts must sum to frame count')
    if scope.get('annotation_status') != 'complete':
        raise ValueError('Fixed-TEST evaluation requires complete annotations')


def _bound_path(override, recorded, expected_sha256, role):
    path = override or recorded
    if not path:
        raise RuntimeError('No path recorded for ' + role)
    identity = _identity(path)
    _require_identity(identity, expected_sha256, role)
    return identity


def _attribution_inputs(attribution, args, frame_count):
    if attribution.get('protocol') != ATTRIBUTION_PROTOCOL:
        raise ValueError('Unexpected fixed-target attribution protocol')
    if attribution.get('evidence_role') != 'fixed-target':
        raise ValueError('Attribution report is not for fixed-target')
    inputs = attribution.get('input', {})
    if int(inputs.get('frame_count', -1)) != frame_count:
        raise ValueError('Attribution frame count differs from contract')
    streams = inputs.get('result_streams', {})
    for name in ('k1_reference', 'full'):
        if name not in streams:
            raise ValueError('Attribution is missing result stream ' + name)
    k1 = _bound_path(
        args.k1_results, streams['k1_reference'].get('path'),
        streams['k1_reference'].get('sha256'), 'formal K1 TEST results')
    base = _bound_path(
        args.base_v3_results, streams['full'].get('path'),
        streams['full'].get('sha256'), 'Base V3 TEST results')
    audit = _bound_path(
        args.all_lane_audit, inputs.get('audit_json_path'),
        inputs.get('audit_json_sha256'), 'fixed-TEST all-lane audit')
    return dict(k1=k1, base_v3=base, all_lane_audit=audit)


def build_records(attribution, args, contract):
    scope = contract['fixed_test_scope']
    frame_count = int(scope['frame_count'])
    inputs = _attribution_inputs(attribution, args, frame_count)
    _, base_boxes = _load_result_boxes(inputs['base_v3']['path'], frame_count)
    _, k1_boxes = _load_result_boxes(inputs['k1']['path'], frame_count)
    ann_dir = args.ann_dir or 'crane_project/data/crane_grab/test/annfiles'
    annotations_id, annotations = _load_annotations(
        ann_dir, frame_count, scope['domain_counts'])
    frame_keys = [row['frame_key'] for row in annotations]
    _, audit, audit_by_key = _load_dino_audit(
        inputs['all_lane_audit']['path'], frame_keys)
    if audit.get('protocol') != AUDIT_PROTOCOL:
        raise ValueError('Unexpected fixed-TEST all-lane audit protocol')

    rows = []
    for index, meta in enumerate(annotations):
        key = meta['frame_key']
        base = base_boxes[index]
        k1 = k1_boxes[index]
        dino = audit_by_key[key]['dino']
        if k1 is not None:
            anchor_source, anchor = 'k1', k1
        elif dino is not None:
            anchor_source, anchor = 'dino_fallback', dino
        else:
            anchor_source, anchor = None, None
        if base is not None and anchor is None:
            raise RuntimeError('Base V3 output has no valid anchor for ' + key)
        rows.append(dict(
            frame_key=key, domain=meta['domain'], sequence=meta['sequence'],
            frame=int(meta['frame']), anchor_source=anchor_source,
            anchor_score=None if anchor is None else float(anchor[5]),
            base_v3_box=_json_box(None if base is None else base[:5]),
            k1_box=_json_box(None if k1 is None else k1[:5]),
            dino_box=_json_box(None if dino is None else dino[:5]),
            gt_box=_json_box(meta['gt'])))
    counts = dict(Counter(row['domain'] for row in rows))
    if counts != scope['domain_counts']:
        raise RuntimeError('Fixed-TEST domain counts changed: {!r}'.format(counts))
    return rows, dict(
        attribution_bound=True,
        annotations=annotations_id,
        base_v3_results=inputs['base_v3'],
        k1_results=inputs['k1'],
        all_lane_audit=inputs['all_lane_audit'],
        frame_count=len(rows), domain_counts=counts,
        sequence_counts=dict(Counter(row['sequence'] for row in rows)),
        base_v3_present=sum(row['base_v3_box'] is not None for row in rows),
        anchor_source_counts=dict(Counter(row['anchor_source'] for row in rows)))


def run(rows, calibration, contract, input_audit):
    frozen_before = json.dumps(calibration, sort_keys=True)
    outputs = {mode: replay(rows, calibration, mode)
               for mode in ('raw', 'score', 'component')}
    if json.dumps(calibration, sort_keys=True) != frozen_before:
        raise RuntimeError('Runtime calibration was mutated during TEST')

    groups = [('all', rows)]
    groups.extend((domain, [row for row in rows if row['domain'] == domain])
                  for domain in sorted({row['domain'] for row in rows}))
    groups.extend(('sequence:' + sequence,
                   [row for row in rows if row['sequence'] == sequence])
                  for sequence in sorted({row['sequence'] for row in rows}))
    reports = {mode: {name: state_report(
        selected, mode_outputs, contract, annotated=True)
        for name, selected in groups}
        for mode, mode_outputs in outputs.items()}

    records = []
    for row in rows:
        item = {key: copy.deepcopy(row.get(key)) for key in (
            'frame_key', 'domain', 'sequence', 'frame', 'anchor_source',
            'anchor_score')}
        item['detector_measurement_available'] = row['base_v3_box'] is not None
        item['online_observations'] = {
            mode: mode_outputs[row['frame_key']]['components']
            for mode, mode_outputs in outputs.items()}
        item['offline_errors'] = {mode: {
            component: (None if observation['value'] is None else
                _error_from_value(observation['value'], row, component))
            for component, observation in mode_outputs[
                row['frame_key']]['components'].items()}
            for mode, mode_outputs in outputs.items()}
        records.append(item)

    coverage = matched_coverage(
        rows, outputs['component'], calibration, contract)
    by_key = {row['frame_key']: row for row in rows}
    error_thresholds = dict(
        center=contract['evaluation']['center_error_threshold_px'],
        scale=contract['evaluation']['scale_relative_error_threshold'],
        angle=contract['evaluation']['angle_error_threshold_deg'])
    present_count = sum(row['base_v3_box'] is not None for row in rows)
    for component in COMPONENTS:
        for curves in coverage[component].values():
            for point in curves:
                errors = [_error_from_value(
                    _measurement_value(by_key[key], component),
                    by_key[key], component)
                    for key in point['retained_frame_keys']]
                point['realized_detector_present_coverage'] = (
                    len(errors)/present_count)
                point['failure_rate'] = (
                    sum(error > error_thresholds[component]
                        for error in errors)/len(errors))

    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        claim_status='FROZEN_CALIBRATION_FIXED_TEST_EVALUATION',
        fixed_test_read=True, target_data_read=True,
        frozen_calibration=True,
        feature_selection_performed=False,
        threshold_fitting_performed=False,
        epoch_selection_performed=False,
        checkpoint_selection_performed=False,
        input_audit=input_audit,
        leakage_controls=dict(
            gt_used_in_online_interface=False,
            gt_used_in_calibration=False,
            gt_attached_after_online_output=True,
            domain_identity_used_in_decisions=False,
            future_frames_used=False,
            state_reset_on_sequence_or_frame_gap=True),
        method_reports=reports,
        matched_coverage=coverage,
        records=records,
        eligible_for_unknown_sequence_claim=False,
        eligible_for_parameter_tuning_from_this_report=False,
        claim_limit=contract['claim_limit'])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-test-attribution', required=True)
    parser.add_argument('--runtime-calibration', required=True)
    parser.add_argument('--development-report', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--base-v3-results')
    parser.add_argument('--k1-results')
    parser.add_argument('--all-lane-audit')
    parser.add_argument('--ann-dir')
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    contract_id, contract = _read_json(args.contract, 'contract')
    validate_contract(contract)
    attribution_id, attribution = _read_json(
        args.fixed_test_attribution, 'fixed-test attribution')
    calibration_id, calibration = _read_json(
        args.runtime_calibration, 'runtime calibration')
    development_id, development = _read_json(
        args.development_report, 'development report')
    expected = contract['expected_inputs']
    _require_identity(calibration_id, expected['runtime_calibration_sha256'],
                      'runtime calibration')
    _require_identity(development_id, expected['development_report_sha256'],
                      'development report')
    if calibration.get('protocol') != expected['runtime_calibration_protocol']:
        raise ValueError('Unexpected runtime calibration protocol')
    if development.get('protocol') != expected['development_report_protocol']:
        raise ValueError('Unexpected development report protocol')
    rows, input_audit = build_records(attribution, args, contract)
    input_audit.update(dict(
        contract=contract_id, attribution=attribution_id,
        runtime_calibration=calibration_id,
        development_report=development_id))
    payload = run(rows, calibration, contract, input_audit)
    output = _write_exact(args.out_json, payload)
    print(json.dumps(dict(
        output=output,
        frame_count=input_audit['frame_count'],
        domain_counts=input_audit['domain_counts'],
        sequence_counts=input_audit['sequence_counts'],
        anchor_source_counts=input_audit['anchor_source_counts'],
        claim_status=payload['claim_status'],
        frozen_calibration=payload['frozen_calibration'],
        threshold_fitting_performed=payload['threshold_fitting_performed'],
        eligible_for_parameter_tuning_from_this_report=False), indent=2))


if __name__ == '__main__':
    main()
