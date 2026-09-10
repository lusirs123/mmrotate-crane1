#!/usr/bin/env python3
"""Consolidate frozen OBB observation evidence for the focused paper."""

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from crane_project.tools import (
    base_v3_obb_hybrid_fixed_test_diagnostic_v51 as diagnostic)
from crane_project.tools import (
    base_v3_obb_image_quality_block_oof_development_v53 as v53)
from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    COMPONENTS)
from crane_project.tools.base_v3_obb_hybrid_fixed_test_eval_v51 import (
    CONTRACT_PROTOCOL as V51_EVAL_CONTRACT_PROTOCOL,
    PROTOCOL as V51_REPORT_PROTOCOL,
    build_records,
    validate_contract as validate_v51_eval_contract)
from crane_project.tools.base_v3_obb_hybrid_policy_freeze_v51 import (
    RUNTIME_PROTOCOL as V51_RUNTIME_CALIBRATION_PROTOCOL)
from crane_project.tools.base_v3_obb_hybrid_runtime_v51 import (
    validate_runtime_calibration)
from crane_project.tools.base_v3_obb_paper_runtime_v1 import (
    METHODS, PROTOCOL as RUNTIME_PROTOCOL, PaperObservationManagerV1)
from crane_project.tools.base_v3_obb_reliability_baseline import (
    _identity, _write_exact)
from crane_project.tools.eval_crane_offline import compute_riou


PROTOCOL = 'base_v3_obb_paper_final_report_v1'
CONTRACT_PROTOCOL = 'base_v3_obb_paper_final_report_contract_v1'


def _require_identity(identity, expected, role):
    if identity['sha256'] != expected:
        raise RuntimeError('{} identity mismatch: expected {}, got {}'.format(
            role, expected, identity['sha256']))


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected paper-finalization contract')
    if contract.get('fixed_test_read') is not True:
        raise ValueError('Final report must declare fixed TEST use')
    if contract.get('fixed_test_previously_exposed') is not True:
        raise ValueError('Final report must preserve prior TEST exposure')
    if contract.get('parameter_tuning_authorized') is not False:
        raise ValueError('Final report must not authorize tuning')
    if contract.get('methods') != list(METHODS):
        raise ValueError('Final method inventory changed')
    if contract.get('components') != list(COMPONENTS):
        raise ValueError('Final component inventory changed')
    expected = contract.get('expected_inputs', {})
    required_protocols = {
        'v51_report_protocol': V51_REPORT_PROTOCOL,
        'v51_diagnostic_protocol': diagnostic.PROTOCOL,
        'v53_report_protocol': v53.PROTOCOL,
        'v51_eval_contract_protocol': V51_EVAL_CONTRACT_PROTOCOL,
        'runtime_calibration_protocol':
            V51_RUNTIME_CALIBRATION_PROTOCOL}
    for field, value in required_protocols.items():
        if expected.get(field) != value:
            raise ValueError('Final input protocol changed: ' + field)
    evaluation = contract.get('evaluation', {})
    riou_threshold = float(evaluation.get('riou_hit_threshold', -1.0))
    if not 0.0 <= riou_threshold <= 1.0:
        raise ValueError('RIoU hit threshold must be within [0, 1]')
    if int(evaluation.get('runtime_benchmark_repeats', 0)) <= 0:
        raise ValueError('Runtime benchmark repeats must be positive')
    if contract['runtime_interface'].get('protocol') != RUNTIME_PROTOCOL:
        raise ValueError('Final report must bind the paper runtime')
    closure = contract['selection_closure']
    if closure.get('overall_winner_claimed') is not False:
        raise ValueError('Final report cannot declare an overall winner')
    if closure.get('future_parameter_selection_from_fixed_test') is not False:
        raise ValueError('Final report cannot tune from fixed TEST')


def validate_inputs(v51_report, v51_diagnostic, v53_report, contract):
    scope = contract['fixed_test_scope']
    if v51_report.get('protocol') != V51_REPORT_PROTOCOL:
        raise ValueError('Unexpected V5.1 fixed-TEST report')
    flags = {
        'fixed_test_read': True,
        'fixed_test_previously_exposed': True,
        'frozen_calibration': True,
        'threshold_fitting_performed': False,
        'policy_selection_performed': False,
        'eligible_for_parameter_tuning_from_this_report': False}
    for field, expected in flags.items():
        if v51_report.get(field) is not expected:
            raise ValueError('Unexpected V5.1 report flag: ' + field)
    records = v51_report.get('records') or []
    if len(records) != int(scope['frame_count']):
        raise ValueError('Final report frame count changed')
    if len({row['frame_key'] for row in records}) != len(records):
        raise ValueError('Final report contains duplicate frame keys')
    if dict(Counter(row['domain'] for row in records)) != scope[
            'domain_counts']:
        raise ValueError('Final report domain counts changed')
    if dict(Counter(row['sequence'] for row in records)) != scope[
            'sequence_counts']:
        raise ValueError('Final report sequence counts changed')
    if v51_report.get('method_inventory') != list(METHODS):
        raise ValueError('V5.1 report method inventory changed')
    required_fields = set(contract['runtime_interface'][
        'required_component_fields'])
    for row in records:
        observations = row.get('online_observations', {})
        if set(observations) != set(METHODS):
            raise ValueError('Incomplete final method observations')
        for method in METHODS:
            if set(observations[method]) != set(COMPONENTS):
                raise ValueError('Incomplete final component observations')
            for component in COMPONENTS:
                if not required_fields.issubset(observations[method][component]):
                    raise ValueError('Final observation schema changed')
    if v51_diagnostic.get('protocol') != diagnostic.PROTOCOL:
        raise ValueError('Unexpected V5.1 diagnostic report')
    if v51_diagnostic.get('parameter_tuning_authorized') is not False:
        raise ValueError('V5.1 diagnostic permits tuning')
    diagnostic_input = v51_diagnostic.get('input') or {}
    if diagnostic_input.get('sha256') != contract['expected_inputs'][
            'v51_report_sha256']:
        raise ValueError('V5.1 diagnostic does not bind the final report')
    expected_evaluation = contract['evaluation']
    diagnostic_evaluation = v51_diagnostic.get('evaluation_thresholds', {})
    for field in ('center_error_threshold_px',
                  'scale_relative_error_threshold',
                  'angle_error_threshold_deg'):
        if float(diagnostic_evaluation.get(field, -1)) != float(
                expected_evaluation[field]):
            raise ValueError('Final evaluation threshold changed: ' + field)
    if v53_report.get('protocol') != v53.PROTOCOL:
        raise ValueError('Unexpected V5.3 report')
    if v53_report.get('decision') != contract['selection_closure'][
            'required_v53_decision']:
        raise ValueError('V5.3 development is not closed')
    if v53_report.get('fixed_test_read') is not False:
        raise ValueError('V5.3 unexpectedly read fixed TEST')
    return records


def _ordered(records):
    return sorted(records, key=lambda row: (
        row['domain'], row['sequence'], int(row['frame']), row['frame_key']))


def _groups(records):
    output = [('all', records)]
    output.extend((domain, [row for row in records
                            if row['domain'] == domain])
                  for domain in sorted({row['domain'] for row in records}))
    output.extend(('sequence:' + sequence, [row for row in records
                                             if row['sequence'] == sequence])
                  for sequence in sorted({row['sequence'] for row in records}))
    output.extend(('anchor:' + source, [row for row in records
                                        if row['anchor_source'] == source])
                  for source in sorted({row['anchor_source']
                                        for row in records}))
    return output


def _recovery_report(records, method, component):
    runs = []
    current = []
    previous = None
    for row in _ordered(records):
        identity = (row['domain'], row['sequence'])
        contiguous = (previous is not None and previous[0] == identity and
                      int(row['frame']) == previous[1]+1)
        accepted = bool(row['online_observations'][method][component][
            'measurement_accepted'])
        if not contiguous:
            if current:
                runs.append((len(current), False))
            current = []
        if accepted:
            if current:
                runs.append((len(current), True))
                current = []
        else:
            current.append(row['frame_key'])
        previous = (identity, int(row['frame']))
    if current:
        runs.append((len(current), False))
    recovered = [length for length, status in runs if status]
    return dict(
        rejection_run_count=len(runs),
        recovered_run_count=len(recovered),
        unrecovered_run_count=len(runs)-len(recovered),
        mean_recovery_delay_frames=(None if not recovered else
                                    float(np.mean(recovered))),
        maximum_recovery_delay_frames=(None if not recovered else
                                       max(recovered)),
        longest_rejection_run=(0 if not runs else
                               max(length for length, _ in runs)))


def _metric_table(v51_report):
    keys = (
        'measurement_coverage', 'output_coverage',
        'mean_available_output_error', 'bad_available_output_rate',
        'correct_output_coverage', 'longest_unavailable_run')
    rows = []
    for method in METHODS:
        for group, group_report in v51_report['method_reports'][method].items():
            for component in COMPONENTS:
                metrics = group_report[component]
                row = dict(method=method, group=group, component=component)
                row.update({key: metrics.get(key) for key in keys})
                rows.append(row)
    return rows


def _runtime_replay(rows, calibration):
    manager = PaperObservationManagerV1(calibration)
    return {row['frame_key']: manager.update(row) for row in _ordered(rows)}


def _assert_runtime_reproduction(rows, records, outputs):
    """Prove that the reusable runtime reproduces the formal V5.1 report."""
    report_by_key = {row['frame_key']: row for row in records}
    if set(report_by_key) != set(outputs):
        raise RuntimeError('Runtime replay frame keys differ from V5.1 report')
    for row in rows:
        key = row['frame_key']
        formal = report_by_key[key]
        if bool(row.get('base_v3_box') is not None) != bool(
                formal['detector_measurement_available']):
            raise RuntimeError('Detector availability changed for ' + key)
        if row.get('anchor_source') != formal.get('anchor_source'):
            raise RuntimeError('Anchor source changed for ' + key)
        if row.get('anchor_score') != formal.get('anchor_score'):
            raise RuntimeError('Anchor score changed for ' + key)
        replayed = outputs[key]
        if replayed['observations'] != formal['online_observations']:
            raise RuntimeError('Runtime observations changed for ' + key)
        if replayed['v51_online_features'] != formal.get(
                'v51_online_features'):
            raise RuntimeError('V5.1 online features changed for ' + key)
    return dict(
        verified_frame_count=len(rows),
        observations_exact_match=True,
        online_features_exact_match=True)


def _reconstruct_obb(row, observations):
    """Reassemble one valid OBB while preserving the raw w/h angle axis."""
    raw = row.get('base_v3_box')
    if raw is None or any(not observations[name]['valid']
                          for name in COMPONENTS):
        return None
    center = observations['center']['value']
    scale = observations['scale']['value']
    angle = observations['angle']['value']
    if center is None or scale is None or angle is None:
        raise RuntimeError('Valid component has no value for ' + row['frame_key'])
    if observations['scale']['state'] != 'measurement':
        raise RuntimeError('Final policy unexpectedly predicts scale')
    box = np.asarray(raw[:5], dtype=np.float64).copy()
    box[:2] = np.asarray(center, dtype=np.float64)
    long_side, short_side = map(float, scale)
    if float(raw[2]) >= float(raw[3]):
        box[2], box[3] = long_side, short_side
    else:
        box[2], box[3] = short_side, long_side
    box[4] = float(angle)
    if (not np.isfinite(box).all() or
            float(box[2]) <= 0.0 or float(box[3]) <= 0.0):
        raise RuntimeError('Invalid reconstructed OBB for ' + row['frame_key'])
    return box.tolist()


def _riou_metrics(rows, outputs, threshold):
    per_frame = {}
    for row in rows:
        key = row['frame_key']
        per_frame[key] = {}
        for method in METHODS:
            box = _reconstruct_obb(row, outputs[key]['observations'][method])
            per_frame[key][method] = (
                None if box is None else float(compute_riou(
                    np.asarray(box, dtype=np.float64),
                    np.asarray(row['gt_box'], dtype=np.float64))))
    reports = {method: {} for method in METHODS}
    for group, selected in _groups(rows):
        for method in METHODS:
            values = [per_frame[row['frame_key']][method] for row in selected]
            available = [value for value in values if value is not None]
            hits = [value for value in available if value >= threshold]
            reports[method][group] = dict(
                frame_count=len(values), available_obb_count=len(available),
                available_obb_coverage=len(available)/len(values),
                mean_available_riou=(None if not available else
                                     float(np.mean(available))),
                riou_hit_count=len(hits),
                riou_hit_coverage=len(hits)/len(values),
                riou_hit_rate_given_available=(
                    None if not available else len(hits)/len(available)),
                bad_available_obb_rate=(
                    None if not available else
                    (len(available)-len(hits))/len(available)))
    return reports, per_frame


def _benchmark_runtime(rows, calibration, repeats):
    if repeats <= 0:
        raise ValueError('Runtime benchmark repeats must be positive')
    ordered = _ordered(rows)
    elapsed = []
    for _ in range(repeats):
        manager = PaperObservationManagerV1(calibration)
        started = time.perf_counter()
        for row in ordered:
            manager.update(row)
        elapsed.append(time.perf_counter()-started)
    median = float(np.median(elapsed))
    return dict(
        scope='observation_layer_only_excludes_detector_and_image_io',
        frame_count=len(ordered), repeats=repeats,
        median_total_seconds=median,
        minimum_total_seconds=float(min(elapsed)),
        median_milliseconds_per_frame=1000.0*median/len(ordered))


def _paper_records(records, riou_by_key, runtime_outputs):
    output = []
    for row in _ordered(records):
        runtime = runtime_outputs[row['frame_key']]
        output.append(dict(
            frame_key=row['frame_key'], domain=row['domain'],
            sequence=row['sequence'], frame=int(row['frame']),
            timestamp_seconds=runtime.get('timestamp_seconds'),
            image_size=runtime.get('image_size'),
            detector=dict(
                raw_obb=copy.deepcopy(runtime['detector']['raw_obb']),
                measurement_available=bool(
                    row['detector_measurement_available']),
                anchor_source=row['anchor_source'],
                anchor_score=row['anchor_score']),
            observations=copy.deepcopy(row['online_observations']),
            v51_online_features=copy.deepcopy(
                row.get('v51_online_features')),
            offline_errors=copy.deepcopy(row['offline_errors']),
            offline_riou=copy.deepcopy(riou_by_key[row['frame_key']])))
    return output


def run(v51_report, v51_diagnostic, v53_report, contract, rows,
        calibration, inputs=None):
    validate_contract(contract)
    records = validate_inputs(
        v51_report, v51_diagnostic, v53_report, contract)
    validate_runtime_calibration(calibration)
    frozen_before = json.dumps(calibration, sort_keys=True)
    runtime_outputs = _runtime_replay(rows, calibration)
    reproduction = _assert_runtime_reproduction(
        rows, records, runtime_outputs)
    riou_reports, riou_by_key = _riou_metrics(
        rows, runtime_outputs,
        float(contract['evaluation']['riou_hit_threshold']))
    if json.dumps(calibration, sort_keys=True) != frozen_before:
        raise RuntimeError('Runtime calibration mutated during finalization')
    recovery = {method: {name: {component: _recovery_report(
        selected, method, component) for component in COMPONENTS}
        for name, selected in _groups(records)} for method in METHODS}
    return dict(
        protocol=PROTOCOL,
        evidence_boundary=contract['evidence_boundary'],
        claim_status='POST_EXPOSURE_FIXED_TEST_PAPER_FINALIZATION_V1',
        fixed_test_read=True, fixed_test_previously_exposed=True,
        frozen_runtime=True, threshold_fitting_performed=False,
        feature_selection_performed=False, policy_selection_performed=False,
        parameter_tuning_authorized=False, inputs=inputs,
        leakage_controls=dict(
            gt_used_in_online_runtime=False,
            gt_attached_after_online_output=True,
            gt_used_only_for_offline_metrics=True,
            domain_identity_used_in_decisions=False,
            future_frames_used=False,
            state_reset_on_sequence_or_frame_gap=True),
        dataset_summary=copy.deepcopy(contract['fixed_test_scope']),
        evaluation_thresholds=copy.deepcopy(contract['evaluation']),
        method_inventory=list(METHODS),
        component_inventory=list(COMPONENTS),
        runtime_interface=copy.deepcopy(contract['runtime_interface']),
        final_policy_reference=contract['selection_closure'][
            'final_runtime_policy'],
        overall_winner_claimed=False,
        component_metric_table=_metric_table(v51_report),
        recovery_metrics=recovery,
        runtime_reproduction_audit=reproduction,
        obb_riou_metrics=riou_reports,
        runtime_benchmark_spec=dict(
            scope='observation_layer_only_excludes_detector_and_image_io',
            repeats=int(contract['evaluation'][
                'runtime_benchmark_repeats']),
            measured_values_stored_in_formal_report=False,
            reporting_channel='terminal_output'),
        joint_component_metrics=copy.deepcopy(
            v51_diagnostic['joint_component_availability']),
        scale_matched_coverage=copy.deepcopy(
            v51_diagnostic['scale_matched_coverage']),
        angle_hold_pair_audit=copy.deepcopy(
            v51_diagnostic['angle_hold_pair_audit']),
        stopped_development_candidates=dict(
            v53_image_quality=dict(
                decision=v53_report['decision'],
                gate_checks=copy.deepcopy(v53_report[
                    'image_quality_development_verdict']['checks']),
                claim_status=v53_report['claim_status'])),
        records=_paper_records(records, riou_by_key, runtime_outputs),
        eligible_for_parameter_tuning_from_this_report=False,
        eligible_for_fresh_unknown_sequence_claim=False,
        eligible_for_physical_state_claim=False,
        claim_limit=contract['claim_limit'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v51-report', required=True)
    parser.add_argument('--v51-diagnostic', required=True)
    parser.add_argument('--v53-report', required=True)
    parser.add_argument('--fixed-test-attribution', required=True)
    parser.add_argument('--runtime-calibration-v51', required=True)
    parser.add_argument('--v51-eval-contract', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--base-v3-results')
    parser.add_argument('--k1-results')
    parser.add_argument('--all-lane-audit')
    parser.add_argument('--ann-dir')
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    paths = dict(
        v51_report=args.v51_report,
        v51_diagnostic=args.v51_diagnostic,
        v53_report=args.v53_report,
        fixed_test_attribution=args.fixed_test_attribution,
        runtime_calibration=args.runtime_calibration_v51,
        v51_eval_contract=args.v51_eval_contract,
        contract=args.contract)
    identities = {name: _identity(path) for name, path in paths.items()}
    contract = json.loads(Path(args.contract).read_text(encoding='utf-8'))
    v51_eval_contract = json.loads(Path(args.v51_eval_contract).read_text(
        encoding='utf-8'))
    calibration = json.loads(Path(args.runtime_calibration_v51).read_text(
        encoding='utf-8'))
    attribution = json.loads(Path(args.fixed_test_attribution).read_text(
        encoding='utf-8'))
    v51_report = json.loads(Path(args.v51_report).read_text(encoding='utf-8'))
    validate_contract(contract)
    validate_v51_eval_contract(v51_eval_contract)
    validate_runtime_calibration(calibration)
    expected = contract['expected_inputs']
    for role, field in (
            ('v51_report', 'v51_report_sha256'),
            ('v51_diagnostic', 'v51_diagnostic_sha256'),
            ('v53_report', 'v53_report_sha256'),
            ('v51_eval_contract', 'v51_eval_contract_sha256'),
            ('runtime_calibration', 'runtime_calibration_sha256')):
        _require_identity(identities[role], expected[field], role)
    formal_audit = v51_report.get('input_audit', {})
    for role, field in (
            ('fixed_test_attribution', 'attribution'),
            ('v51_eval_contract', 'contract'),
            ('runtime_calibration', 'runtime_calibration_v51')):
        if identities[role]['sha256'] != formal_audit.get(field, {}).get(
                'sha256'):
            raise RuntimeError(
                '{} is not the input bound by V5.1 TEST'.format(role))
    rows, reconstruction_audit = build_records(
        attribution, args, v51_eval_contract)
    reconstruction_audit.update(dict(
        attribution=identities['fixed_test_attribution'],
        v51_eval_contract=identities['v51_eval_contract'],
        runtime_calibration=identities['runtime_calibration']))
    final_inputs = copy.deepcopy(identities)
    final_inputs['fixed_test_reconstruction'] = reconstruction_audit
    payload = run(
        v51_report,
        json.loads(Path(args.v51_diagnostic).read_text(encoding='utf-8')),
        json.loads(Path(args.v53_report).read_text(encoding='utf-8')),
        contract, rows, calibration, final_inputs)
    output = _write_exact(args.out_json, payload)
    runtime_benchmark = _benchmark_runtime(
        rows, calibration,
        int(contract['evaluation']['runtime_benchmark_repeats']))
    all_joint = {method: payload['joint_component_metrics'][method]['all']
                 for method in METHODS}
    all_riou = {method: payload['obb_riou_metrics'][method]['all']
                for method in METHODS}
    print(json.dumps(dict(
        output=output, dataset_summary=payload['dataset_summary'],
        method_count=len(payload['method_inventory']),
        component_count=len(payload['component_inventory']),
        v53_decision=payload['stopped_development_candidates'][
            'v53_image_quality']['decision'],
        joint_all_summary={method: {key: values[key] for key in (
            'all_components_valid_coverage', 'jointly_correct_coverage',
            'false_all_components_valid_rate',
            'longest_incomplete_obb_run')}
            for method, values in all_joint.items()},
        riou_all_summary={method: {key: values[key] for key in (
            'available_obb_coverage', 'mean_available_riou',
            'riou_hit_coverage', 'bad_available_obb_rate')}
            for method, values in all_riou.items()},
        runtime_reproduction_audit=payload['runtime_reproduction_audit'],
        runtime_benchmark=runtime_benchmark,
        claim_status=payload['claim_status'],
        parameter_tuning_authorized=False), indent=2))


if __name__ == '__main__':
    main()
