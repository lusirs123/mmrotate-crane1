#!/usr/bin/env python3
"""Evaluate the frozen Base V3 OBB observation interface on a new sequence.

This entrypoint loads an already-frozen runtime calibration artifact. It does
not fit thresholds, select features, or use ground truth in online decisions.
"""

import argparse
import copy
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    COMPONENTS, _error_from_value, _measurement_value)
from crane_project.tools.base_v3_obb_observation_interface_v3 import (
    CALIBRATION_PROTOCOL, ComponentObservationManager, _component_risk,
    _online_row)
from crane_project.tools.base_v3_obb_reliability_baseline import (
    _identity, _write_exact)


PROTOCOL = 'base_v3_obb_external_sequence_eval_v1'
CONTRACT_PROTOCOL = 'base_v3_obb_external_sequence_eval_contract_v1'
RECORDS_PROTOCOL = 'base_v3_obb_external_sequence_records_v1'
HEX64 = re.compile(r'^[0-9a-f]{64}$')


def _require_identity(identity, expected, role):
    if identity['sha256'] != expected:
        raise RuntimeError('{} identity mismatch: expected {}, got {}'.format(
            role, expected, identity['sha256']))


def _validate_box(box, role, nullable=True):
    if box is None:
        if nullable:
            return
        raise ValueError(role + ' must not be null')
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if (values.size != 5 or not np.isfinite(values).all()
            or values[2] <= 0.0 or values[3] <= 0.0):
        raise ValueError(role + ' must be a finite five-value OBB')


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected external-evaluation contract')
    if contract.get('fixed_test_read') is not False:
        raise ValueError('Contract must not authorize fixed TEST')
    if contract.get('target_data_read') is not False:
        raise ValueError('Contract must not authorize target data')
    expected = contract.get('expected_inputs', {})
    if expected.get('runtime_calibration_protocol') != CALIBRATION_PROTOCOL:
        raise ValueError('Contract must bind the V3 calibration protocol')
    if expected.get('external_records_protocol') != RECORDS_PROTOCOL:
        raise ValueError('Contract must bind the external-record protocol')
    for name in ('runtime_calibration_sha256',
                 'development_report_sha256'):
        if not HEX64.fullmatch(str(expected.get(name, ''))):
            raise ValueError(name + ' must be a lowercase SHA256')


def validate_external_records(payload, development_report, contract):
    if payload.get('protocol') != RECORDS_PROTOCOL:
        raise ValueError('Unexpected external sequence records protocol')
    for field in contract['required_provenance']:
        if field not in payload:
            raise ValueError('Missing provenance field: ' + field)
    if not payload.get('dataset_id') or not payload.get('capture_session_id'):
        raise ValueError('dataset_id and capture_session_id must be non-empty')
    if not HEX64.fullmatch(str(payload.get('frame_manifest_sha256', ''))):
        raise ValueError('frame_manifest_sha256 must be a lowercase SHA256')
    if payload.get('not_used_for_calibration_or_feature_selection') is not True:
        raise ValueError('External sequence must be unused for development')
    annotation_status = payload.get('annotation_status')
    if annotation_status not in ('complete', 'none'):
        raise ValueError('annotation_status must be complete or none')
    records = payload.get('records')
    if not isinstance(records, list) or not records:
        raise ValueError('External records must be a non-empty list')

    required = contract['input_schema']['required_record_fields']
    keys = set()
    sequence_pairs = set()
    annotation_count = 0
    for index, row in enumerate(records):
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError('Record {} missing {}'.format(index, missing))
        key = row['frame_key']
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError('External frame keys must be unique strings')
        keys.add(key)
        if not isinstance(row['domain'], str) or not row['domain']:
            raise ValueError('domain must be a non-empty string')
        if not isinstance(row['sequence'], str) or not row['sequence']:
            raise ValueError('sequence must be a non-empty string')
        sequence_pairs.add((row['domain'], row['sequence']))
        if not isinstance(row['frame'], int) or row['frame'] < 0:
            raise ValueError('frame must be a non-negative integer')
        if row['anchor_source'] not in ('k1', 'dino_fallback', None):
            raise ValueError('Unexpected anchor_source for ' + key)
        _validate_box(row['base_v3_box'], key + '.base_v3_box')
        _validate_box(row['k1_box'], key + '.k1_box')
        _validate_box(row['dino_box'], key + '.dino_box')
        if row['base_v3_box'] is not None:
            score = row['anchor_score']
            if score is None or not math.isfinite(float(score)):
                raise ValueError('Detected frames need a finite anchor_score')
            if not 0.0 <= float(score) <= 1.0:
                raise ValueError('anchor_score must be within [0, 1]')
            if row['anchor_source'] == 'k1' and row['k1_box'] is None:
                raise ValueError('K1 anchor requires k1_box')
            if (row['anchor_source'] == 'dino_fallback'
                    and row['dino_box'] is None):
                raise ValueError('DINO fallback requires dino_box')
            if row['anchor_source'] is None:
                raise ValueError('Detected frames need an anchor_source')
        if row.get('gt_box') is not None:
            _validate_box(row['gt_box'], key + '.gt_box', nullable=False)
            annotation_count += 1
    if annotation_status == 'complete' and annotation_count != len(records):
        raise ValueError('Complete annotation requires gt_box on every frame')
    if annotation_status == 'none' and annotation_count != 0:
        raise ValueError('Unannotated input must not contain gt_box')

    development_records = development_report.get('records') or []
    development_keys = {row['frame_key'] for row in development_records}
    development_pairs = {(row['domain'], row['sequence'])
                         for row in development_records}
    if keys & development_keys:
        raise ValueError('External frame keys overlap development replay')
    if sequence_pairs & development_pairs:
        raise ValueError(
            'External domain/sequence identities overlap development replay')
    return dict(
        frame_count=len(records),
        annotation_status=annotation_status,
        domain_counts=dict(Counter(row['domain'] for row in records)),
        sequence_count=len(sequence_pairs),
        frame_key_overlap_count=0,
        domain_sequence_overlap_count=0,
        provenance_attested=True)


def _ordered(rows):
    return sorted(rows, key=lambda row: (
        row['domain'], row['sequence'], row['frame'], row['frame_key']))


def replay(rows, calibration, gate_mode):
    manager = ComponentObservationManager(calibration, gate_mode)
    return {row['frame_key']: manager.update(_online_row(row))
            for row in _ordered(rows)}


def _quantiles(values, requested):
    return ({str(q): float(np.quantile(values, q)) for q in requested}
            if values else {})


def _longest_unavailable(rows, outputs, component):
    longest = current = 0
    previous = None
    for row in _ordered(rows):
        identity = (row['domain'], row['sequence'])
        contiguous = (previous is not None and previous[0] == identity
                      and row['frame'] == previous[1]+1)
        if not contiguous:
            current = 0
        state = outputs[row['frame_key']]['components'][component]['state']
        current = current+1 if state == 'unavailable' else 0
        longest = max(longest, current)
        previous = (identity, row['frame'])
    return longest


def state_report(rows, outputs, contract, annotated):
    report = {}
    thresholds = dict(
        center=contract['evaluation']['center_error_threshold_px'],
        scale=contract['evaluation']['scale_relative_error_threshold'],
        angle=contract['evaluation']['angle_error_threshold_deg'])
    requested = contract['evaluation']['tail_quantiles']
    for component in COMPONENTS:
        states = Counter(outputs[row['frame_key']]['components'][component][
            'state'] for row in rows)
        missing_rows = [row for row in rows if row['base_v3_box'] is None]
        missing_states = Counter(outputs[row['frame_key']]['components'][
            component]['state'] for row in missing_rows)
        item = dict(
            frame_count=len(rows),
            measurement_count=states['measurement'],
            prediction_count=states['prediction'],
            unavailable_count=states['unavailable'],
            measurement_coverage=states['measurement']/len(rows),
            output_coverage=(states['measurement']+states['prediction'])/len(rows),
            longest_unavailable_run=_longest_unavailable(
                rows, outputs, component),
            detector_missing_frame_count=len(missing_rows),
            detector_missing_state_counts=dict(missing_states))
        if annotated:
            available_errors = []
            measurement_errors = []
            prediction_errors = []
            prediction_deltas = []
            missing_available_errors = []
            for row in rows:
                observation = outputs[row['frame_key']]['components'][component]
                if observation['value'] is None:
                    continue
                error = _error_from_value(observation['value'], row, component)
                available_errors.append(error)
                if observation['state'] == 'measurement':
                    measurement_errors.append(error)
                else:
                    prediction_errors.append(error)
                    if row['base_v3_box'] is not None:
                        raw_value = _measurement_value(row, component)
                        prediction_deltas.append(
                            error-_error_from_value(raw_value, row, component))
                if row['base_v3_box'] is None:
                    missing_available_errors.append(error)
            bad = sum(error > thresholds[component]
                      for error in available_errors)
            item.update(
                mean_available_output_error=(None if not available_errors else
                    float(np.mean(available_errors))),
                available_error_quantiles=_quantiles(
                    available_errors, requested),
                bad_available_output_rate=(None if not available_errors else
                    bad/len(available_errors)),
                correct_output_coverage=(len(available_errors)-bad)/len(rows),
                mean_measurement_error=(None if not measurement_errors else
                    float(np.mean(measurement_errors))),
                mean_prediction_error=(None if not prediction_errors else
                    float(np.mean(prediction_errors))),
                prediction_improved_count=sum(
                    delta < 0 for delta in prediction_deltas),
                prediction_degraded_count=sum(
                    delta > 0 for delta in prediction_deltas),
                mean_prediction_error_delta_vs_raw=(
                    None if not prediction_deltas else
                    float(np.mean(prediction_deltas))),
                detector_missing_available_count=len(missing_available_errors),
                detector_missing_mean_error=(
                    None if not missing_available_errors else
                    float(np.mean(missing_available_errors))))
        report[component] = item
    return report


def matched_coverage(rows, component_outputs, calibration, contract):
    present = [row for row in rows if row['base_v3_box'] is not None]
    if not present:
        return {}
    result = {}
    for component in COMPONENTS:
        orderings = dict(
            raw_anchor_score=sorted(
                present, key=lambda row: (-float(row['anchor_score']),
                                          row['frame_key'])),
            component_risk=sorted(present, key=lambda row: (
                _component_risk(component_outputs[row['frame_key']][
                    'online_features'], component, calibration),
                row['frame_key'])))
        result[component] = {}
        for method, ordered_rows in orderings.items():
            curve = []
            for coverage in contract['risk_coverage_targets']:
                kept = ordered_rows[:max(1, round(len(ordered_rows)*coverage))]
                errors = [_error_from_value(
                    _measurement_value(row, component), row, component)
                    for row in kept]
                curve.append(dict(
                    target_coverage=float(coverage),
                    retained_count=len(kept),
                    mean_error=float(np.mean(errors)),
                    retained_frame_keys=[row['frame_key'] for row in kept]))
            result[component][method] = curve
    return result


def run(external, calibration, development_report, contract):
    validate_contract(contract)
    if calibration.get('protocol') != CALIBRATION_PROTOCOL:
        raise ValueError('Unexpected runtime calibration protocol')
    independence = validate_external_records(
        external, development_report, contract)
    rows = external['records']
    annotated = external['annotation_status'] == 'complete'
    frozen_before = json.dumps(calibration, sort_keys=True)
    outputs = {mode: replay(rows, calibration, mode)
               for mode in ('raw', 'score', 'component')}
    if json.dumps(calibration, sort_keys=True) != frozen_before:
        raise RuntimeError('Runtime calibration was mutated during evaluation')
    groups = [('all', rows)]
    groups.extend((domain, [row for row in rows if row['domain'] == domain])
                  for domain in sorted({row['domain'] for row in rows}))
    reports = {mode: {name: state_report(
        selected, mode_outputs, contract, annotated)
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
        if annotated:
            item['offline_errors'] = {mode: {
                component: (None if observation['value'] is None else
                    _error_from_value(observation['value'], row, component))
                for component, observation in mode_outputs[
                    row['frame_key']]['components'].items()}
                for mode, mode_outputs in outputs.items()}
        records.append(item)
    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        claim_status=('INDEPENDENT_SEQUENCE_EVALUATION_CANDIDATE'
                      if annotated else 'INTERFACE_ONLY_NO_ACCURACY_CLAIM'),
        annotation_status=external['annotation_status'],
        frozen_calibration=True,
        feature_selection_performed=False,
        threshold_fitting_performed=False,
        fixed_test_read=False,
        target_data_read=False,
        independence_checks=independence,
        leakage_controls=dict(
            gt_used_in_online_interface=False,
            gt_used_in_calibration=False,
            gt_attached_after_online_output=annotated,
            domain_identity_used_in_decisions=False,
            future_frames_used=False,
            state_reset_on_sequence_or_frame_gap=True),
        method_reports=reports,
        matched_coverage=(matched_coverage(
            rows, outputs['component'], calibration, contract)
            if annotated else None),
        records=records,
        claim_limit=contract['claim_limit'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-records', required=True)
    parser.add_argument('--runtime-calibration', required=True)
    parser.add_argument('--development-report', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    contract = json.loads(Path(args.contract).read_text())
    validate_contract(contract)
    identities = dict(
        sequence_records=_identity(args.sequence_records),
        runtime_calibration=_identity(args.runtime_calibration),
        development_report=_identity(args.development_report),
        contract=_identity(args.contract))
    expected = contract['expected_inputs']
    _require_identity(identities['runtime_calibration'],
                      expected['runtime_calibration_sha256'],
                      'runtime calibration')
    _require_identity(identities['development_report'],
                      expected['development_report_sha256'],
                      'development report')
    external = json.loads(Path(args.sequence_records).read_text())
    calibration = json.loads(Path(args.runtime_calibration).read_text())
    development_report = json.loads(Path(args.development_report).read_text())
    if development_report.get('protocol') != expected[
            'development_report_protocol']:
        raise ValueError('Unexpected development report protocol')
    payload = run(external, calibration, development_report, contract)
    payload['inputs'] = identities
    output = _write_exact(args.out_json, payload)
    print(json.dumps(dict(
        output=output,
        claim_status=payload['claim_status'],
        frozen_calibration=payload['frozen_calibration'],
        threshold_fitting_performed=payload['threshold_fitting_performed'],
        independence_checks=payload['independence_checks'],
        leakage_controls=payload['leakage_controls']), indent=2))


if __name__ == '__main__':
    main()
