#!/usr/bin/env python3
"""One-shot official source-val gate for frozen Base-V3 current-only mode."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from crane_project.tools.symeood_dino_application_domain_v4_audit import (
    audit_payload)
from crane_project.tools.symeood_dino_causal_history_source_gate import (
    _sym_geometry_preservation)
from crane_project.tools.symeood_dino_dual_tower_v2_audit import (
    _annotations, _load_results, _metrics)
from crane_project.utils.depth_interface_geometry_gate import (
    depth_interface_geometry_gate)
from crane_project.utils.geometry_refiner_source_gate import (
    relaxed_composite_gate)


PROTOCOL = 'base_v3_epoch9_current_only_official_source_val_gate_v1'
CONTRACT_PROTOCOL = 'base_v3_epoch9_current_only_source_val_gate_contract_v1'
PROMOTION_PROTOCOL = 'source_gated_k1_retentive_causal_phase_refiner_v3'
RECEIPT_PROTOCOL = 'mmdet_runtime_result_order_identity_v1'
RUNTIME_PROTOCOL = 'mmdet_runtime_inference_resource_audit_v2'
RESULT_PROTOCOL = 'base_v3_epoch9_current_only_official_source_val_v1'
EXPECTED_ABSOLUTE_GATE = dict(
    native_dino_min_composite_gain=0.005,
    reuse_v3_sym_geometry_preservation=True,
    reuse_depth_interface_geometry_gate=True)
EXPECTED_RELATIVE_LIMITS = dict(
    real_center_drop_pp_max=0.0,
    sim_center_drop_pp_max=0.0,
    real_mean_riou_drop_max=0.01,
    sim_mean_riou_drop_max=0.01,
    real_dfr_increase_pp_per_frame_max=0.25,
    sim_dfr_increase_pp_per_frame_max=0.25,
    real_aci_drop_max=0.003,
    sim_aci_drop_max=0.003,
    sim_a_rmse_increase_deg_max=0.0,
    real_tdr_drop_pp_max=1.0,
    sim_tdr_drop_pp_max=1.0,
    real_mcml_policy='not_worse',
    sim_mcml_policy='not_worse')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-results', required=True)
    parser.add_argument('--candidate-result-receipt', required=True)
    parser.add_argument('--candidate-runtime-audit', required=True)
    parser.add_argument('--candidate-config', required=True)
    parser.add_argument('--candidate-checkpoint', required=True)
    parser.add_argument('--base-v3-promotion', required=True)
    parser.add_argument('--source-val-audit', required=True)
    parser.add_argument('--sym-reference-results', required=True)
    parser.add_argument('--gate-contract', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
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


def _relative_gate(candidate, reference, limits):
    checks = dict(
        real_center_no_drop=(
            float(candidate['real/R_center(%)']) >=
            float(reference['real/R_center(%)']) -
            float(limits['real_center_drop_pp_max'])),
        sim_center_no_drop=(
            float(candidate['sim/R_center(%)']) >=
            float(reference['sim/R_center(%)']) -
            float(limits['sim_center_drop_pp_max'])),
        real_riou_within_limit=(
            float(candidate['real/mean_RIoU']) >=
            float(reference['real/mean_RIoU']) -
            float(limits['real_mean_riou_drop_max'])),
        sim_riou_within_limit=(
            float(candidate['sim/mean_RIoU']) >=
            float(reference['sim/mean_RIoU']) -
            float(limits['sim_mean_riou_drop_max'])),
        real_dfr_within_limit=(
            float(candidate['real/DFR(%/frame)']) <=
            float(reference['real/DFR(%/frame)']) +
            float(limits['real_dfr_increase_pp_per_frame_max'])),
        sim_dfr_within_limit=(
            float(candidate['sim/DFR(%/frame)']) <=
            float(reference['sim/DFR(%/frame)']) +
            float(limits['sim_dfr_increase_pp_per_frame_max'])),
        real_aci_within_limit=(
            float(candidate['real/ACI']) >= float(reference['real/ACI']) -
            float(limits['real_aci_drop_max'])),
        sim_aci_within_limit=(
            float(candidate['sim/ACI']) >= float(reference['sim/ACI']) -
            float(limits['sim_aci_drop_max'])),
        sim_angle_rmse_within_limit=(
            float(candidate['sim/A-RMSE(deg)']) <=
            float(reference['sim/A-RMSE(deg)']) +
            float(limits['sim_a_rmse_increase_deg_max'])),
        real_tdr_within_limit=(
            float(candidate['real/TDR_w10(%)']) >=
            float(reference['real/TDR_w10(%)']) -
            float(limits['real_tdr_drop_pp_max'])),
        sim_tdr_within_limit=(
            float(candidate['sim/TDR_w10(%)']) >=
            float(reference['sim/TDR_w10(%)']) -
            float(limits['sim_tdr_drop_pp_max'])),
        real_mcml_not_worse=(
            int(candidate['real/MCML_max(frames)']) <=
            int(reference['real/MCML_max(frames)'])),
        sim_mcml_not_worse=(
            int(candidate['sim/MCML_max(frames)']) <=
            int(reference['sim/MCML_max(frames)'])))
    return dict(limits=dict(limits), checks=checks, passed=all(checks.values()))


def gate(args):
    contract_id, contract = _json(args.gate_contract, CONTRACT_PROTOCOL)
    if (contract.get('status') !=
            'preregistered_before_current_only_official_source_val_inference'
            or contract.get('target_data_read') is not False
            or contract.get('fixed_test_read') is not False):
        raise RuntimeError('Invalid preregistered current-only gate contract')
    if dict(contract.get('absolute_formal_gate') or {}) != EXPECTED_ABSOLUTE_GATE:
        raise RuntimeError('Absolute source-gate thresholds changed')
    if (dict(contract.get('relative_non_regression') or {}) !=
            EXPECTED_RELATIVE_LIMITS):
        raise RuntimeError('Relative non-regression thresholds changed')
    policy = dict(contract.get('decision_policy') or {})
    required_policy = dict(
        all_absolute_and_relative_checks_required=True,
        failure_allows_alternate_epoch_selection=False,
        failure_allows_threshold_change=False,
        official_source_val_used_for_ranking=False,
        automatic_fixed_test_authorization=False)
    if policy != required_policy:
        raise RuntimeError('Current-only source-gate decision policy changed')

    promotion_id, promotion = _json(args.base_v3_promotion, PROMOTION_PROTOCOL)
    if (promotion.get('decision') !=
            'ALLOW_K1_RETENTIVE_CAUSAL_PHASE_FIXED_BENCHMARK_TEST'
            or promotion.get('fixed_test_read') is not False):
        raise RuntimeError('Base V3 promotion evidence is invalid')
    selected = dict((promotion.get('selection') or {}).get('selected') or {})
    if selected.get('epoch') != 9:
        raise RuntimeError('Promotion did not bind Base V3 epoch9')
    reference_gate_id, reference_gate = _json(selected.get('source_gate'))
    if reference_gate_id['sha256'] != selected.get('source_gate_sha256'):
        raise RuntimeError('Promotion/source-gate SHA256 mismatch')
    if (reference_gate.get('protocol') !=
            'k1_retentive_causal_phase_refiner_source_gate_v3_geometry_v1'
            or reference_gate.get('evidence_boundary') != 'source_val_only'
            or reference_gate.get('target_data_read') is not False
            or reference_gate.get('fixed_test_read') is not False):
        raise RuntimeError('Selected Base V3 source gate has invalid provenance')
    reference_metrics = dict(reference_gate.get('candidate_metrics') or {})
    if not reference_gate.get('passed') or not reference_metrics:
        raise RuntimeError('Selected Base V3 source gate did not pass')

    checkpoint_id = _identity(args.candidate_checkpoint)
    promoted_output = dict(promotion.get('output') or {})
    if checkpoint_id['sha256'] != promoted_output.get('checkpoint_sha256'):
        raise RuntimeError('Candidate is not the promoted Base V3 checkpoint')
    config_id = _identity(args.candidate_config)
    candidate_id, candidate_boxes = _load_results(args.candidate_results)
    candidate_id = _identity(candidate_id)
    sym_id, sym_boxes = _load_results(args.sym_reference_results)
    sym_id = _identity(sym_id)
    ann_dir, metadata, domain_counts = _annotations(
        os.path.join(os.path.abspath(args.data_root), 'val', 'annfiles'))
    candidate_metrics = _metrics(metadata, candidate_boxes)
    sym_metrics = _metrics(metadata, sym_boxes)

    receipt_id, receipt = _json(args.candidate_result_receipt,
                                RECEIPT_PROTOCOL)
    expected_order = [
        Path(path).stem for path in sorted(
            Path(ann_dir).glob('*.txt'), key=lambda item: item.name)]
    receipt_order = [
        row.get('frame_key') for row in receipt.get('runtime_dataset_order') or []]
    receipt_contract = dict(receipt.get('evidence_contract') or {})
    if (receipt.get('result_count') != 738
            or receipt_order != expected_order
            or receipt.get('results_sha256') != candidate_id['sha256']
            or receipt.get('checkpoint_sha256') != checkpoint_id['sha256']
            or receipt.get('config_sha256') != config_id['sha256']
            or receipt_contract.get('protocol') != RESULT_PROTOCOL
            or receipt.get('target_data_read') is not False
            or receipt.get('fixed_test_read') is not False):
        raise RuntimeError('Candidate result receipt mismatch')

    runtime_id, runtime = _json(args.candidate_runtime_audit,
                                RUNTIME_PROTOCOL)
    runtime_model = dict(runtime.get('runtime_model_contract') or {})
    effective_model = dict(
        dict(runtime.get('effective_config') or {}).get('model') or {})
    effective_refiner = dict(effective_model.get('geometry_refiner') or {})
    forward = dict(runtime.get('forward_counts') or {})
    cuda = dict(runtime.get('cuda') or {})
    runtime_source_audit = dict(runtime.get('runtime_input_files') or {}).get(
        'dino_source_val_audit') or {}
    source_audit_id = _identity(args.source_val_audit)
    if (runtime.get('result_count') != 738
            or runtime.get('checkpoint_sha256') != checkpoint_id['sha256']
            or runtime.get('config_sha256') != config_id['sha256']
            or runtime_source_audit.get('sha256') != source_audit_id['sha256']
            or forward.get('inference_forward') != 738
            or forward.get('dino_detector_forward') != 0
            or cuda.get('cuda_visible_devices') != '0'
            or int(cuda.get('peak_allocated_bytes', 0)) <= 0
            or int(cuda.get('peak_reserved_bytes', 0)) <= 0
            or runtime.get('cli_cfg_options') is not None
            or not runtime.get('effective_config_sha256')
            or runtime_model.get('evaluation_only') is not True
            or runtime_model.get('inference_component_mode') != 'current_only'
            or runtime_model.get('causal_history_features_computed') is not True
            or runtime_model.get('history_output_contribution') is not False
            or effective_refiner.get('inference_component_mode') !=
            'current_only'):
        raise RuntimeError('Candidate runtime audit mismatch')

    with open(source_audit_id['path'], 'rb') as stream:
        source_audit_raw = stream.read()
    audited = audit_payload(
        json.loads(source_audit_raw.decode('utf-8')), source_audit_raw,
        os.path.join(os.path.abspath(args.data_root), 'val'), 'source-val')
    dino_metrics = audited['native_dino_baseline']['metrics']
    absolute_config = dict(EXPECTED_ABSOLUTE_GATE)
    dino_gain = relaxed_composite_gate(
        candidate_metrics, dino_metrics,
        min_composite_gain=float(
            absolute_config['native_dino_min_composite_gain']),
        reference_policy='native_dino_source_val')
    sym_preservation = _sym_geometry_preservation(
        candidate_metrics, sym_metrics)
    depth_gate = depth_interface_geometry_gate(
        metadata, candidate_boxes, sym_boxes)
    relative = _relative_gate(
        candidate_metrics, reference_metrics,
        EXPECTED_RELATIVE_LIMITS)
    checks = dict(
        absolute_gain_over_native_dino=dino_gain['passed'],
        absolute_sym_geometry_preservation=sym_preservation['passed'],
        absolute_depth_interface_geometry=depth_gate['passed'],
        relative_to_full_base_v3=relative['passed'],
        runtime_identity=True,
        fixed_test_not_read=True)
    passed = all(checks.values())
    return dict(
        protocol=PROTOCOL,
        evidence_boundary='official_source_val_738_pass_fail_only',
        inputs=dict(
            gate_contract=contract_id, base_v3_promotion=promotion_id,
            full_base_v3_source_gate=reference_gate_id,
            candidate_config=config_id, candidate_checkpoint=checkpoint_id,
            candidate_results=candidate_id,
            candidate_result_receipt=receipt_id,
            candidate_runtime_audit=runtime_id,
            source_val_audit=source_audit_id,
            sym_reference_results=sym_id, ann_dir=ann_dir,
            frame_count=738, domain_counts=domain_counts),
        candidate_metrics=candidate_metrics,
        full_base_v3_reference_metrics=reference_metrics,
        sym_reference_metrics=sym_metrics,
        native_dino_reference_metrics=dino_metrics,
        absolute_gain_over_native_dino=dino_gain,
        absolute_sym_geometry_preservation=sym_preservation,
        absolute_depth_interface_geometry=depth_gate,
        relative_non_regression=relative,
        checks=checks, passed=passed,
        official_source_val_used_for_ranking=False,
        training_run=False, optimizer_steps=0,
        target_data_read=False, fixed_test_read=False,
        eligible_for_checkpoint_promotion=False,
        eligible_for_fixed_test=False,
        decision=(
            'ALLOW_CURRENT_ONLY_SOURCE_VALIDATED_ABLATION_NO_TEST_AUTHORIZATION'
            if passed else 'STOP_CURRENT_ONLY_SOURCE_NON_REGRESSION_FAILED'))


def main():
    args = parse_args()
    report = gate(args)
    output = Path(args.out_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(report, indent=2, ensure_ascii=False) + '\n').encode(
        'utf-8')
    if output.exists() and output.read_bytes() != raw:
        raise RuntimeError('Refusing to overwrite different gate: ' + str(output))
    if not output.exists():
        output.write_bytes(raw)
    print('[current-only-source-gate] output={}'.format(output))
    print('[current-only-source-gate] decision={}'.format(report['decision']))
    for name, value in report['checks'].items():
        print('[current-only-source-gate] {:40s} {}'.format(name, value))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
