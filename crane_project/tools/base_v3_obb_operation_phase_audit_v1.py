#!/usr/bin/env python3
"""Audit frozen V5.1 OBB reliability by manually reviewed operation group.

Operation labels are attached after inference and are never consumed by the
online policy. The audit compares transport, the confirmed grabbing interval,
the remaining real seq03 frames, and the simulation reference separately.
"""

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from crane_project.tools import (
    base_v3_obb_hybrid_fixed_test_diagnostic_v51 as diagnostic)
from crane_project.tools.base_v3_obb_reliability_baseline import (
    _identity, _write_exact)


PROTOCOL = 'base_v3_obb_operation_phase_audit_v1'
CONTRACT_PROTOCOL = 'base_v3_obb_operation_phase_audit_contract_v1'
MANIFEST_PROTOCOL = 'fixed_test_operation_phase_manifest_v1'
INPUT_PROTOCOL = diagnostic.INPUT_PROTOCOL
METHODS = ('raw', 'score_rejection_only', 'v51_hybrid')
COMPARISON_TOLERANCE = 1e-12


def _valid_sha256(value):
    text = str(value)
    return len(text) == 64 and all(c in '0123456789abcdef' for c in text)


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected operation-phase audit contract')
    if contract.get('fixed_test_read') is not True:
        raise ValueError('Operation-phase audit must declare fixed TEST read')
    if contract.get('phase_labels_used_online') is not False:
        raise ValueError('Operation phase labels must remain offline-only')
    if contract.get('parameter_tuning_authorized') is not False:
        raise ValueError('Operation-phase audit cannot authorize tuning')
    expected = contract.get('expected_inputs', {})
    if expected.get('fixed_test_report_protocol') != INPUT_PROTOCOL:
        raise ValueError('Contract must bind the frozen V5.1 TEST report')
    if expected.get('phase_manifest_protocol') != MANIFEST_PROTOCOL:
        raise ValueError('Contract must bind the phase manifest')
    for field in ('fixed_test_report_sha256', 'phase_manifest_sha256'):
        if not _valid_sha256(expected.get(field)):
            raise ValueError(field + ' must be a lowercase SHA256')


def validate_manifest(manifest):
    if manifest.get('protocol') != MANIFEST_PROTOCOL:
        raise ValueError('Unexpected operation-phase manifest')
    if manifest.get('target_definition') != 'grab_top_beam_obb':
        raise ValueError('Operation-phase audit requires top-beam OBB semantics')
    if manifest.get('online_input_authorized') is not False:
        raise ValueError('Phase labels must not become online inputs')
    rules = manifest.get('rules')
    if not isinstance(rules, list) or not rules:
        raise ValueError('Phase manifest needs at least one rule')
    for rule in rules:
        for field in ('domain', 'sequence', 'operation_group'):
            if not rule.get(field):
                raise ValueError('Incomplete phase rule: ' + field)
        start = rule.get('frame_start_inclusive')
        end = rule.get('frame_end_inclusive')
        if start is not None and end is not None and int(start) > int(end):
            raise ValueError('Phase rule starts after it ends')


def _rule_matches(row, rule):
    if (row['domain'], row['sequence']) != (
            rule['domain'], rule['sequence']):
        return False
    frame = int(row['frame'])
    start = rule.get('frame_start_inclusive')
    end = rule.get('frame_end_inclusive')
    return ((start is None or frame >= int(start)) and
            (end is None or frame <= int(end)))


def assign_operation_groups(records, manifest):
    grouped = {}
    annotated = []
    for row in records:
        matches = [rule for rule in manifest['rules']
                   if _rule_matches(row, rule)]
        if len(matches) != 1:
            raise ValueError('{} matches {} phase rules'.format(
                row['frame_key'], len(matches)))
        group = matches[0]['operation_group']
        grouped.setdefault(group, []).append(row)
        annotated.append(dict(
            frame_key=row['frame_key'], domain=row['domain'],
            sequence=row['sequence'], frame=int(row['frame']),
            operation_group=group,
            operation_interpretation=matches[0].get('interpretation')))
    return grouped, annotated


def _component_report(rows, method, component, threshold):
    observations = [row['online_observations'][method][component]
                    for row in rows]
    errors = [float(row['offline_errors'][method][component])
              for row, observation in zip(rows, observations)
              if observation['valid']]
    bad = sum(error > threshold for error in errors)
    return dict(
        frame_count=len(rows), output_count=len(errors),
        output_coverage=len(errors)/len(rows),
        mean_available_output_error=(
            None if not errors else float(np.mean(errors))),
        bad_available_output_count=bad,
        bad_available_output_rate=(None if not errors else bad/len(errors)),
        correct_output_count=len(errors)-bad,
        correct_output_coverage=(len(errors)-bad)/len(rows))


def component_performance(grouped, contract):
    evaluation = contract['evaluation']
    thresholds = dict(
        center=float(evaluation['center_error_threshold_px']),
        scale=float(evaluation['scale_relative_error_threshold']),
        angle=float(evaluation['angle_error_threshold_deg']))
    return {group: {method: {
        component: _component_report(rows, method, component,
                                     thresholds[component])
        for component in diagnostic.COMPONENTS} for method in METHODS}
        for group, rows in grouped.items()}


def _comparison_label(delta_mean, delta_failure):
    mean_tie = abs(delta_mean) <= COMPARISON_TOLERANCE
    failure_tie = abs(delta_failure) <= COMPARISON_TOLERANCE
    if mean_tie and failure_tie:
        return 'tie'
    risk_no_worse = (delta_mean <= COMPARISON_TOLERANCE and
                     delta_failure <= COMPARISON_TOLERANCE)
    score_no_worse = (delta_mean >= -COMPARISON_TOLERANCE and
                      delta_failure >= -COMPARISON_TOLERANCE)
    if risk_no_worse:
        return 'learned_risk_better'
    if score_no_worse:
        return 'anchor_score_better'
    return 'mixed'


def scale_matched_coverage_by_operation(grouped, contract):
    result = {}
    for group, rows in grouped.items():
        audit = diagnostic.scale_matched_coverage(rows, contract)['all']
        counts = Counter()
        for point in audit['points']:
            delta = point['learned_minus_score']
            label = _comparison_label(
                float(delta['mean_error']), float(delta['failure_rate']))
            delta['comparison'] = label
            counts[label] += 1
        audit.pop('risk_wins_both_count', None)
        audit['comparison_counts'] = dict(counts)
        result[group] = audit
    return result


def angle_hold_by_operation(records, grouped, contract):
    threshold = float(contract['evaluation']['angle_error_threshold_deg'])
    run_lengths = diagnostic.angle_rejection_run_lengths(records)
    return {group: diagnostic._paired_angle_report(
        rows, run_lengths, threshold) for group, rows in grouped.items()}


def joint_availability_by_operation(grouped, contract):
    evaluation = contract['evaluation']
    thresholds = dict(
        center=float(evaluation['center_error_threshold_px']),
        scale=float(evaluation['scale_relative_error_threshold']),
        angle=float(evaluation['angle_error_threshold_deg']))
    return {group: {
        method: diagnostic._joint_report(rows, method, thresholds)
        for method in METHODS} for group, rows in grouped.items()}


def run(report, manifest, contract, identities=None):
    validate_contract(contract)
    diagnostic.validate_report(report, {
        'fixed_test_scope': contract['fixed_test_scope']})
    validate_manifest(manifest)
    grouped, annotated = assign_operation_groups(report['records'], manifest)
    expected = contract['expected_operation_group_counts']
    actual = {name: len(rows) for name, rows in grouped.items()}
    if actual != expected:
        raise RuntimeError('Operation group counts changed: {!r}'.format(actual))
    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        claim_status='POST_EXPOSURE_OPERATION_PHASE_DIAGNOSTIC_ONLY',
        target_definition=manifest['target_definition'],
        target_geometry_note=manifest['target_geometry_note'],
        fixed_test_read=True, fixed_test_previously_exposed=True,
        operation_labels_attached_after_online_output=True,
        operation_labels_used_in_online_decisions=False,
        descriptive_analysis_only=True,
        threshold_fitting_performed=False,
        feature_selection_performed=False,
        policy_selection_performed=False,
        parameter_tuning_authorized=False,
        inputs=identities,
        operation_group_counts=actual,
        operation_group_assignments=annotated,
        component_performance=component_performance(grouped, contract),
        scale_matched_coverage=scale_matched_coverage_by_operation(
            grouped, contract),
        angle_hold_pair_audit=angle_hold_by_operation(
            report['records'], grouped, contract),
        joint_component_availability=joint_availability_by_operation(
            grouped, contract),
        eligible_for_phase_conditioned_online_policy_claim=False,
        eligible_for_parameter_tuning_from_this_report=False,
        eligible_for_final_independent_validation_claim=False,
        claim_limit=contract['claim_limit'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-test-v51-report', required=True)
    parser.add_argument('--phase-manifest', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    identities = dict(
        fixed_test_v51_report=_identity(args.fixed_test_v51_report),
        phase_manifest=_identity(args.phase_manifest),
        contract=_identity(args.contract))
    contract = json.loads(Path(args.contract).read_text())
    validate_contract(contract)
    expected = contract['expected_inputs']
    if identities['fixed_test_v51_report']['sha256'] != expected[
            'fixed_test_report_sha256']:
        raise RuntimeError('Frozen V5.1 TEST report identity mismatch')
    if identities['phase_manifest']['sha256'] != expected[
            'phase_manifest_sha256']:
        raise RuntimeError('Operation-phase manifest identity mismatch')
    report = json.loads(Path(args.fixed_test_v51_report).read_text())
    manifest = json.loads(Path(args.phase_manifest).read_text())
    payload = run(report, manifest, contract, identities)
    output = _write_exact(args.out_json, payload)
    summary = {}
    for group in contract['reported_operation_groups']:
        summary[group] = dict(
            frame_count=payload['operation_group_counts'][group],
            scale_comparison_counts=payload['scale_matched_coverage'][group][
                'comparison_counts'],
            angle_hold={key: payload['angle_hold_pair_audit'][group][key]
                        for key in ('prediction_count', 'improved_count',
                                    'degraded_count',
                                    'mean_error_delta_vs_raw')},
            v51_joint={key: payload['joint_component_availability'][group][
                'v51_hybrid'][key] for key in (
                    'all_components_valid_coverage',
                    'jointly_correct_coverage',
                    'false_all_components_valid_rate',
                    'longest_incomplete_obb_run')})
    print(json.dumps(dict(
        output=output, operation_group_summary=summary,
        claim_status=payload['claim_status'],
        operation_labels_used_in_online_decisions=False,
        parameter_tuning_authorized=False), indent=2))


if __name__ == '__main__':
    main()
