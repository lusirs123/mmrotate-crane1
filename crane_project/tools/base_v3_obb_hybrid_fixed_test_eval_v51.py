#!/usr/bin/env python3
"""Re-evaluate frozen V5.1 OBB observations on the existing fixed TEST.

The fixed TEST was inspected before V5/V5.1 was designed, so this entrypoint
records the run as a post-exposure re-evaluation rather than fresh validation.
V5.1 calibration is immutable and TEST labels are attached only after causal
raw, score-only, and hybrid observations have been emitted.
"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    COMPONENTS, _error_from_value, _measurement_value)
from crane_project.tools.base_v3_obb_external_sequence_eval import state_report
from crane_project.tools.base_v3_obb_fixed_test_eval import (
    ATTRIBUTION_PROTOCOL, build_records, _read_json, _require_identity)
from crane_project.tools.base_v3_obb_hybrid_policy_freeze_v51 import (
    PROTOCOL as FREEZE_PROTOCOL, RUNTIME_PROTOCOL)
from crane_project.tools.base_v3_obb_hybrid_runtime_v51 import (
    HybridObservationManagerV51, validate_runtime_calibration)
from crane_project.tools.base_v3_obb_reliability_baseline import (
    AUDIT_PROTOCOL, _identity, _write_exact)


PROTOCOL = 'base_v3_obb_hybrid_fixed_test_eval_v51'
CONTRACT_PROTOCOL = 'base_v3_obb_hybrid_fixed_test_eval_contract_v51'
PREVIOUS_TEST_PROTOCOL = 'base_v3_obb_fixed_test_eval_v1'


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected V5.1 fixed-TEST evaluation contract')
    if contract.get('fixed_test_read') is not True:
        raise ValueError('Contract must explicitly authorize fixed TEST')
    if contract.get('target_data_read') is not True:
        raise ValueError('Contract must explicitly declare target-data read')
    if contract.get('fixed_test_previously_exposed') is not True:
        raise ValueError('Contract must preserve prior TEST exposure')
    if contract.get('post_fixed_test_hypothesis_generation') is not True:
        raise ValueError('Contract must preserve post-TEST method development')
    if contract.get('fixed_test_reuse_role') != (
            'post_exposure_frozen_policy_reevaluation'):
        raise ValueError('Unexpected fixed TEST reuse role')
    expected = contract.get('expected_inputs', {})
    if expected.get('runtime_calibration_protocol') != RUNTIME_PROTOCOL:
        raise ValueError('Contract must bind V5.1 runtime calibration')
    if expected.get('policy_freeze_report_protocol') != FREEZE_PROTOCOL:
        raise ValueError('Contract must bind the V5.1 freeze report')
    if expected.get('previous_fixed_test_report_protocol') != (
            PREVIOUS_TEST_PROTOCOL):
        raise ValueError('Contract must bind the previous TEST report')
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
        raise ValueError('V5.1 fixed-TEST evaluation requires annotations')


def _observation(row, component, accepted, gate):
    available = row.get('base_v3_box') is not None
    value = _measurement_value(row, component) if accepted else None
    return dict(
        value=value, valid=bool(accepted),
        state='measurement' if accepted else 'unavailable',
        source='base_v3_measurement' if accepted else None,
        age_since_measurement_frames=0 if accepted else None,
        measurement_available=available,
        measurement_accepted=bool(accepted), risk=None, gate=gate,
        reason=('accepted_measurement' if accepted else
                'detector_missing' if not available else
                'rejected_measurement'))


def _replay_simple(rows, calibration, score_only):
    outputs = {}
    threshold = float(calibration['score_gate']['score_threshold'])
    for row in rows:
        available = row.get('base_v3_box') is not None
        accepted = available and (not score_only or (
            row.get('anchor_score') is not None and
            float(row['anchor_score']) >= threshold))
        components = {
            component: _observation(
                row, component, accepted,
                'anchor_score' if score_only else 'detector_availability')
            for component in COMPONENTS}
        outputs[row['frame_key']] = dict(
            frame_key=row['frame_key'], domain=row['domain'],
            sequence=row['sequence'], frame=int(row['frame']),
            online_features=None, components=components)
    return outputs


def _replay_v51(rows, calibration):
    manager = HybridObservationManagerV51(calibration)
    return {row['frame_key']: manager.update(row) for row in rows}


def _groups(rows):
    groups = [('all', rows)]
    groups.extend((domain, [row for row in rows if row['domain'] == domain])
                  for domain in sorted({row['domain'] for row in rows}))
    groups.extend(('sequence:' + sequence,
                   [row for row in rows if row['sequence'] == sequence])
                  for sequence in sorted({row['sequence'] for row in rows}))
    groups.extend(('anchor:' + source,
                   [row for row in rows if row['anchor_source'] == source])
                  for source in ('k1', 'dino_fallback')
                  if any(row['anchor_source'] == source for row in rows))
    return groups


def _compact_metrics(reports):
    return {method: {component: {
        key: reports[method]['all'][component][key]
        for key in ('measurement_coverage', 'output_coverage',
                    'mean_available_output_error',
                    'bad_available_output_rate',
                    'correct_output_coverage', 'longest_unavailable_run')}
        for component in COMPONENTS} for method in reports}


def run(rows, calibration, contract, input_audit):
    validate_contract(contract)
    validate_runtime_calibration(calibration)
    frozen_before = json.dumps(calibration, sort_keys=True)
    outputs = {
        'raw': _replay_simple(rows, calibration, score_only=False),
        'score_rejection_only': _replay_simple(
            rows, calibration, score_only=True),
        'v51_hybrid': _replay_v51(rows, calibration)}
    if json.dumps(calibration, sort_keys=True) != frozen_before:
        raise RuntimeError('V5.1 runtime calibration mutated during TEST')

    reports = {method: {name: state_report(
        selected, method_outputs, contract, annotated=True)
        for name, selected in _groups(rows)}
        for method, method_outputs in outputs.items()}
    records = []
    for row in rows:
        item = {key: copy.deepcopy(row.get(key)) for key in (
            'frame_key', 'domain', 'sequence', 'frame', 'anchor_source',
            'anchor_score')}
        item['detector_measurement_available'] = (
            row.get('base_v3_box') is not None)
        item['online_observations'] = {
            method: outputs[method][row['frame_key']]['components']
            for method in outputs}
        item['v51_online_features'] = outputs['v51_hybrid'][
            row['frame_key']]['online_features']
        item['offline_errors'] = {method: {
            component: (None if observation['value'] is None else
                        _error_from_value(observation['value'], row, component))
            for component, observation in outputs[method][
                row['frame_key']]['components'].items()}
            for method in outputs}
        records.append(item)

    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        claim_status='FROZEN_V51_PREVIOUSLY_EXPOSED_FIXED_TEST_REEVALUATION',
        fixed_test_read=True, target_data_read=True,
        fixed_test_previously_exposed=True,
        post_fixed_test_hypothesis_generation=True,
        fresh_sequence_evaluation=False,
        frozen_calibration=True, feature_selection_performed=False,
        threshold_fitting_performed=False, policy_selection_performed=False,
        epoch_selection_performed=False, checkpoint_selection_performed=False,
        input_audit=input_audit,
        leakage_controls=dict(
            gt_used_in_online_interface=False,
            gt_used_in_calibration=False,
            gt_attached_after_online_output=True,
            domain_identity_used_in_decisions=False,
            future_frames_used=False,
            state_reset_on_sequence_or_frame_gap=True),
        method_inventory=list(outputs), method_reports=reports,
        compact_all_metrics=_compact_metrics(reports), records=records,
        eligible_for_parameter_tuning_from_this_report=False,
        eligible_for_final_independent_validation_claim=False,
        eligible_for_unknown_sequence_claim=False,
        claim_limit=contract['claim_limit'])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-test-attribution', required=True)
    parser.add_argument('--runtime-calibration-v51', required=True)
    parser.add_argument('--policy-freeze-report-v51', required=True)
    parser.add_argument('--previous-fixed-test-report', required=True)
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
    runtime_id, calibration = _read_json(
        args.runtime_calibration_v51, 'V5.1 runtime calibration')
    freeze_id, freeze_report = _read_json(
        args.policy_freeze_report_v51, 'V5.1 policy freeze report')
    previous_id, previous_report = _read_json(
        args.previous_fixed_test_report, 'previous fixed-TEST report')
    expected = contract['expected_inputs']
    for identity, field, role in (
            (runtime_id, 'runtime_calibration_sha256', 'V5.1 calibration'),
            (freeze_id, 'policy_freeze_report_sha256', 'V5.1 freeze report'),
            (previous_id, 'previous_fixed_test_report_sha256',
             'previous fixed-TEST report')):
        _require_identity(identity, expected[field], role)
    if calibration.get('protocol') != RUNTIME_PROTOCOL:
        raise ValueError('Unexpected V5.1 runtime calibration protocol')
    if freeze_report.get('protocol') != FREEZE_PROTOCOL:
        raise ValueError('Unexpected V5.1 policy freeze protocol')
    if freeze_report.get('decision') != (
            'FREEZE_V51_FOR_FRESH_SEQUENCE_EVALUATION'):
        raise ValueError('V5.1 policy was not successfully frozen')
    if freeze_report.get('frozen_runtime_calibration', {}).get(
            'sha256') != runtime_id['sha256']:
        raise RuntimeError('Freeze report does not bind this runtime artifact')
    if previous_report.get('protocol') != PREVIOUS_TEST_PROTOCOL or (
            previous_report.get('fixed_test_read') is not True):
        raise ValueError('Previous report does not prove fixed TEST exposure')

    rows, input_audit = build_records(attribution, args, contract)
    input_audit.update(dict(
        contract=contract_id, attribution=attribution_id,
        runtime_calibration_v51=runtime_id,
        policy_freeze_report_v51=freeze_id,
        previous_fixed_test_report=previous_id))
    payload = run(rows, calibration, contract, input_audit)
    output = _write_exact(args.out_json, payload)
    print(json.dumps(dict(
        output=output, frame_count=input_audit['frame_count'],
        domain_counts=input_audit['domain_counts'],
        sequence_counts=input_audit['sequence_counts'],
        anchor_source_counts=input_audit['anchor_source_counts'],
        method_count=len(payload['method_inventory']),
        compact_all_metrics=payload['compact_all_metrics'],
        claim_status=payload['claim_status'],
        fixed_test_previously_exposed=True,
        frozen_calibration=True, threshold_fitting_performed=False,
        eligible_for_parameter_tuning_from_this_report=False), indent=2))


if __name__ == '__main__':
    main()
