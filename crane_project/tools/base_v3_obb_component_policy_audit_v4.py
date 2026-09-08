#!/usr/bin/env python3
"""Audit component gates and bounded output policies on source-val only.

This entrypoint never reads fixed TEST.  It loads the already-frozen V3
calibration and decomposes each gate into rejection-only and bounded-hold
variants.  Simple feature-availability and K1-anchor controls reveal whether
an apparent component-risk gain needs the numeric risk score at all.
"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    COMPONENTS, _measurement_value,
    assign_split, build_feature_rows)
from crane_project.tools.base_v3_obb_observation_interface_v3 import (
    CALIBRATION_PROTOCOL, _component_risk, _direct_risk_curves)
from crane_project.tools.base_v3_obb_external_sequence_eval import state_report
from crane_project.tools.base_v3_obb_reliability_baseline import (
    PROTOCOL as BASELINE_PROTOCOL, _identity, _write_exact)
from crane_project.tools.base_v3_reliability_failure_audit import (
    portable_digest, validate)


PROTOCOL = 'base_v3_obb_component_policy_audit_v4'
CONTRACT_PROTOCOL = 'base_v3_obb_component_policy_audit_contract_v4'
POLICIES = ('rejection_only', 'bounded_hold')


def _require_identity(identity, expected, role):
    if identity['sha256'] != expected:
        raise RuntimeError('{} identity mismatch: expected {}, got {}'.format(
            role, expected, identity['sha256']))


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected V4 component-policy audit contract')
    if contract.get('fixed_test_read') is not False:
        raise ValueError('V4 development audit must not authorize fixed TEST')
    if contract.get('target_data_read') is not False:
        raise ValueError('V4 development audit must remain source-only')
    expected = contract.get('expected_inputs', {})
    if expected.get('baseline_protocol') != BASELINE_PROTOCOL:
        raise ValueError('Contract must bind the Base V3 baseline protocol')
    if expected.get('runtime_calibration_protocol') != CALIBRATION_PROTOCOL:
        raise ValueError('Contract must bind the V3 calibration protocol')
    if set(contract.get('methods', ())) != {
            'raw_rejection_only',
            'score_rejection_only', 'score_bounded_hold',
            'component_rejection_only', 'component_bounded_hold',
            'feature_availability_rejection_only',
            'feature_availability_bounded_hold',
            'k1_anchor_rejection_only', 'k1_anchor_bounded_hold'}:
        raise ValueError('Contract method inventory changed')


def _ordered(rows):
    return sorted(rows, key=lambda row: (
        row['domain'], row['sequence'], row['frame'], row['frame_key']))


def _features(rows):
    """Create the same causal online features used by interface V3."""
    enriched = build_feature_rows(rows)
    return {row['frame_key']: row['online_features'] for row in enriched}


def gate_decisions(rows, features, calibration, gate):
    decisions = {}
    for row in rows:
        available = row.get('base_v3_box') is not None
        current = features[row['frame_key']]
        if gate == 'raw':
            accepted = {component: available for component in COMPONENTS}
        elif gate == 'score':
            value = (available and row.get('anchor_score') is not None
                     and float(row['anchor_score']) >=
                     float(calibration['score_threshold']))
            accepted = {component: value for component in COMPONENTS}
        elif gate == 'component':
            accepted = {component: bool(
                available and _component_risk(
                    current, component, calibration) <= float(
                        calibration['component_risk_thresholds'][component]))
                for component in COMPONENTS}
        elif gate == 'feature_availability':
            accepted = {component: bool(
                available and all(current.get(name) is not None
                                  for name in calibration[
                                      'component_features'][component]))
                for component in COMPONENTS}
        elif gate == 'k1_anchor':
            value = available and row.get('anchor_source') == 'k1'
            accepted = {component: value for component in COMPONENTS}
        else:
            raise ValueError('Unknown gate: ' + gate)
        decisions[row['frame_key']] = accepted
    return decisions


def replay_policy(rows, features, decisions, calibration, gate, policy):
    if policy not in POLICIES:
        raise ValueError('Unknown output policy: ' + policy)
    outputs = {}
    last_measurement = {component: None for component in COMPONENTS}
    last_identity = None
    last_frame = None
    for row in _ordered(rows):
        identity = (row['domain'], row['sequence'])
        if (identity != last_identity or last_frame is None
                or int(row['frame']) != last_frame+1):
            last_measurement = {component: None for component in COMPONENTS}
        current_features = features[row['frame_key']]
        observations = {}
        for component in COMPONENTS:
            accepted = decisions[row['frame_key']][component]
            if accepted:
                state = 'measurement'
                value = _measurement_value(row, component)
                source = 'base_v3_measurement'
                age = 0
                last_measurement[component] = dict(
                    frame=int(row['frame']), value=copy.deepcopy(value))
            else:
                previous = last_measurement[component]
                age = (None if previous is None else
                       int(row['frame'])-previous['frame'])
                frozen = calibration['component_output_policy'][component]
                may_hold = (policy == 'bounded_hold'
                            and frozen['rejected_or_missing'] ==
                            'bounded_last_measurement_hold')
                if (may_hold and previous is not None and age > 0
                        and age <= int(frozen['max_hold_frames'])):
                    state = 'prediction'
                    value = copy.deepcopy(previous['value'])
                    source = 'bounded_last_measurement_hold'
                else:
                    state, value, source = 'unavailable', None, None
            observations[component] = dict(
                value=value, valid=state != 'unavailable', state=state,
                source=source, age_since_measurement_frames=age,
                measurement_available=row.get('base_v3_box') is not None,
                measurement_accepted=accepted,
                risk=(None if row.get('base_v3_box') is None else
                      _component_risk(current_features, component,
                                      calibration)),
                gate_threshold=(
                    float(calibration['score_threshold'])
                    if gate == 'score' else
                    float(calibration['component_risk_thresholds'][
                        component]) if gate == 'component' else None),
                reason=(('accepted_' + gate + '_gate')
                        if accepted else
                        ('detector_missing'
                         if row.get('base_v3_box') is None else
                         ('rejected_' + gate + '_gate'))))
        outputs[row['frame_key']] = dict(
            frame_key=row['frame_key'], domain=row['domain'],
            sequence=row['sequence'], frame=int(row['frame']),
            online_features=current_features, components=observations)
        last_identity, last_frame = identity, int(row['frame'])
    return outputs


def _groups(rows):
    groups = [('all', rows)]
    groups.extend((domain, [row for row in rows if row['domain'] == domain])
                  for domain in sorted({row['domain'] for row in rows}))
    groups.extend(('sequence:' + sequence,
                   [row for row in rows if row['sequence'] == sequence])
                  for sequence in sorted({row['sequence'] for row in rows}))
    return groups


def _source_slice(rows, outputs, contract):
    result = {}
    minimum = int(contract['minimum_source_slice_count'])
    for source in ('k1', 'dino_fallback'):
        selected = [row for row in rows if row['anchor_source'] == source]
        if not selected:
            result[source] = dict(
                frame_count=0, sufficient_for_comparative_claim=False,
                temporal_run_metrics_valid=False, components={})
            continue
        report = state_report(selected, outputs, contract, annotated=True)
        for component in COMPONENTS:
            report[component].pop('longest_unavailable_run', None)
        result[source] = dict(
            frame_count=len(selected),
            sufficient_for_comparative_claim=len(selected) >= minimum,
            temporal_run_metrics_valid=False,
            components=report)
    return result


def _gate_overlap(rows, decisions, first, second):
    output = {}
    for component in COMPONENTS:
        counts = Counter()
        for row in rows:
            a = decisions[first][row['frame_key']][component]
            b = decisions[second][row['frame_key']][component]
            counts[('accept' if a else 'reject') + '_' +
                   ('accept' if b else 'reject')] += 1
        output[component] = dict(counts)
    return output


def run(baseline, baseline_contract, calibration, contract):
    validate_contract(contract)
    validate(baseline, baseline_contract)
    if calibration.get('protocol') != CALIBRATION_PROTOCOL:
        raise ValueError('Unexpected runtime calibration protocol')
    before = json.dumps(calibration, sort_keys=True)
    rows = build_feature_rows(baseline['records'])
    split_counts = assign_split(
        rows, float(contract['split']['calibration_prefix_fraction']),
        int(contract['split']['minimum_calibration_frames_per_segment']))
    expected_counts = dict(
        calibration=int(contract['split']['expected_calibration_frames']),
        evaluation=int(contract['split']['expected_evaluation_frames']))
    if split_counts != expected_counts:
        raise RuntimeError('Source split counts changed: {!r}'.format(
            split_counts))
    evaluation_rows = [row for row in rows if row['split'] == 'evaluation']
    features = {row['frame_key']: row['online_features']
                for row in evaluation_rows}
    gates = ('raw', 'score', 'component',
             'feature_availability', 'k1_anchor')
    decisions = {gate: gate_decisions(
        evaluation_rows, features, calibration, gate) for gate in gates}
    specs = [('raw', 'rejection_only')]
    specs.extend((gate, policy)
                 for gate in gates[1:] for policy in POLICIES)
    reports = {}
    outputs_by_method = {}
    for gate, policy in specs:
        name = gate + '_' + policy
        outputs = replay_policy(
            evaluation_rows, features, decisions[gate], calibration, gate, policy)
        outputs_by_method[name] = outputs
        reports[name] = {
            group: state_report(selected, outputs, contract, annotated=True)
            for group, selected in _groups(evaluation_rows)}
        reports[name]['anchor_source_slices'] = _source_slice(
            evaluation_rows, outputs, contract)
    if json.dumps(calibration, sort_keys=True) != before:
        raise RuntimeError('Frozen runtime calibration was mutated')

    risk_curves = {group: _direct_risk_curves(
        selected, calibration, contract)
        for group, selected in _groups(evaluation_rows)}
    risk_curves['anchor_source:k1'] = _direct_risk_curves(
        [row for row in evaluation_rows if row['anchor_source'] == 'k1'],
        calibration, contract)
    fallback_rows = [row for row in evaluation_rows
                     if row['anchor_source'] == 'dino_fallback']
    risk_curves['anchor_source:dino_fallback'] = (
        _direct_risk_curves(fallback_rows, calibration, contract)
        if fallback_rows else {})

    output_rows = []
    for row in rows:
        item = {key: copy.deepcopy(row.get(key)) for key in (
            'frame_key', 'domain', 'sequence', 'frame', 'split',
            'anchor_source', 'anchor_score')}
        item['online_features'] = row['online_features']
        item['offline_errors'] = row['errors']
        if row['split'] == 'evaluation':
            item['gate_decisions'] = {
                gate: decisions[gate][row['frame_key']] for gate in gates}
            item['method_states'] = {
                method: output[row['frame_key']]['components']
                for method, output in outputs_by_method.items()}
        output_rows.append(item)

    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        claim_status='SOURCE_VAL_DEVELOPMENT_POLICY_AUDIT_ONLY',
        fixed_test_read=False, target_data_read=False,
        post_fixed_test_hypothesis_generation=True,
        frozen_calibration=True,
        threshold_fitting_performed=False,
        feature_selection_performed=False,
        leakage_controls=dict(
            gt_used_in_online_features=False,
            gt_used_in_gate_decisions=False,
            gt_used_in_state_output=False,
            gt_attached_after_online_output=True,
            domain_identity_used_in_decisions=False,
            future_frames_used=False,
            state_reset_on_sequence_or_frame_gap=True),
        split_counts=split_counts,
        method_inventory=[gate + '_' + policy for gate, policy in specs],
        method_reports=reports,
        matched_coverage=risk_curves,
        gate_overlap=dict(
            component_vs_feature_availability=_gate_overlap(
                evaluation_rows, decisions,
                'component', 'feature_availability'),
            component_vs_k1_anchor=_gate_overlap(
                evaluation_rows, decisions, 'component', 'k1_anchor')),
        records=output_rows,
        eligible_for_fixed_test_reuse_as_final_validation=False,
        eligible_for_unknown_sequence_claim=False,
        claim_limit=contract['claim_limit'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--baseline-contract', required=True)
    parser.add_argument('--runtime-calibration', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    identities = dict(
        baseline=_identity(args.baseline),
        baseline_contract=_identity(args.baseline_contract),
        runtime_calibration=_identity(args.runtime_calibration),
        contract=_identity(args.contract))
    baseline = json.loads(Path(args.baseline).read_text())
    baseline_contract = json.loads(Path(args.baseline_contract).read_text())
    calibration = json.loads(Path(args.runtime_calibration).read_text())
    contract = json.loads(Path(args.contract).read_text())
    validate_contract(contract)
    expected = contract['expected_inputs']
    _require_identity(identities['baseline_contract'],
                      expected['baseline_contract_sha256'],
                      'baseline contract')
    _require_identity(identities['runtime_calibration'],
                      expected['runtime_calibration_sha256'],
                      'runtime calibration')
    portable = portable_digest(baseline)
    if portable != expected['baseline_portable_sha256']:
        raise RuntimeError('Portable source baseline identity mismatch')
    payload = run(baseline, baseline_contract, calibration, contract)
    payload['inputs'] = identities
    payload['inputs']['baseline_portable_sha256'] = portable
    output = _write_exact(args.out_json, payload)
    fallback = sum(row['anchor_source'] == 'dino_fallback'
                   and row['split'] == 'evaluation'
                   for row in payload['records'])
    print(json.dumps(dict(
        output=output,
        split_counts=payload['split_counts'],
        method_count=len(payload['method_inventory']),
        evaluation_dino_fallback_frames=fallback,
        claim_status=payload['claim_status'],
        frozen_calibration=payload['frozen_calibration'],
        threshold_fitting_performed=False,
        fixed_test_read=False), indent=2))


if __name__ == '__main__':
    main()
