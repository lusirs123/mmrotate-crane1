#!/usr/bin/env python3
"""Pre-registered source-val gate for causal-history refiner checkpoints."""

import argparse
import hashlib
import json
import os

import torch

from crane_project.tools.symeood_dino_application_domain_v4_audit import (
    audit_payload)
from crane_project.tools.symeood_dino_dual_tower_v2_audit import (
    _annotations, _load_results, _metrics)
from crane_project.utils.geometry_refiner_source_gate import (
    relaxed_composite_gate)


PROTOCOL = 'causal_history_refiner_source_gate_v1'
V2_PROTOCOL = 'k1_anchored_causal_phase_refiner_source_gate_v2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-results', required=True)
    parser.add_argument('--candidate-checkpoint', required=True)
    parser.add_argument('--source-val-audit', required=True)
    parser.add_argument('--sym-reference-results', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--expected-candidate-sha256')
    parser.add_argument('--min-composite-gain', type=float, default=0.005)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_contract(path, expected_sha256=None):
    absolute = os.path.abspath(os.fspath(path))
    observed = _sha256(absolute)
    if (expected_sha256 is not None
            and observed.lower() != expected_sha256.lower()):
        raise RuntimeError('Candidate checkpoint SHA256 mismatch')
    payload = torch.load(absolute, map_location='cpu')
    contract = dict(payload.get('meta') or {}).get(
        'geometry_refiner_checkpoint_contract')
    if not isinstance(contract, dict):
        raise RuntimeError('Candidate checkpoint has no refiner contract')
    architecture = contract.get('architecture')
    v2 = architecture == 'k1_anchored_causal_phase_refiner_v2'
    required = dict(
        protocol=(
            'source_only_k1_anchored_causal_phase_refiner_v2' if v2 else
            'source_only_causal_history_refiner_v1'),
        architecture=(
            'k1_anchored_causal_phase_refiner_v2' if v2 else
            'current_anchored_causal_history_refiner_v1'),
        frozen_baseline_variant='symeood_k1_epoch24',
        frozen_baseline_config='crane_project/configs/crane_symeood_k1.py',
        frozen_baseline_checkpoint=(
            'work_dirs/crane_symeood_k1/epoch_24.pth'),
        source_train_frames=2781,
        source_val_frames=738,
        target_data_read=False,
        fixed_test_read=False,
        source_gate_passed=False,
        detector_forward_during_training=False,
        dino_detector_forward_during_training=False,
        frozen_symeood_feature_forward=True,
        cached_dino_proposals_only=True,
        domain_routing=False,
        sequence_frame_routing=False,
        temporal_state=False,
        causal_history_input=True,
        history_horizon=4,
        history_identity_model_input=False,
        current_frame_anchored=True,
        bounded_history_residual=True,
        rejectable_history_gate=True,
        exact_current_only_when_no_history=True,
        source_only_proposal_corruption=True,
        fixed_target_parameter_selection=False)
    if v2:
        required.update(dict(
            detector_forward_during_training=True,
            frozen_symeood_detection_head_forward=True,
            frozen_symeood_detection_from_shared_features=True,
            current_k1_geometry_anchor=True,
            native_dino_anchor_fallback=True,
            native_dino_current_conditioning=True,
            same_forward_all_domains=True,
            bounded_current_residual=True,
            continuous_double_angle_phase=True,
            zero_phase_is_exact_identity=True,
            representation='six_delta_xywh_sin2a_cos2a_residual'))
    failures = [
        '{}={!r} expected {!r}'.format(key, contract.get(key), expected)
        for key, expected in required.items()
        if contract.get(key) != expected]
    if failures:
        raise RuntimeError(
            'Candidate checkpoint contract failed: ' + '; '.join(failures))
    return absolute, observed, contract, v2


def _sym_geometry_preservation(candidate, sym):
    checks = dict(
        real_center_within_1pp=(
            float(candidate['real/R_center(%)']) >=
            float(sym['real/R_center(%)']) - 1.0),
        sim_center_within_1pp=(
            float(candidate['sim/R_center(%)']) >=
            float(sym['sim/R_center(%)']) - 1.0),
        real_riou_within_0p03=(
            float(candidate['real/mean_RIoU']) >=
            float(sym['real/mean_RIoU']) - 0.03),
        sim_riou_within_0p03=(
            float(candidate['sim/mean_RIoU']) >=
            float(sym['sim/mean_RIoU']) - 0.03),
        real_dfr_within_0p75pp=(
            float(candidate['real/DFR(%/frame)']) <=
            float(sym['real/DFR(%/frame)']) + 0.75),
        sim_dfr_within_0p75pp=(
            float(candidate['sim/DFR(%/frame)']) <=
            float(sym['sim/DFR(%/frame)']) + 0.75),
        real_aci_within_0p02=(
            float(candidate['real/ACI']) >=
            float(sym['real/ACI']) - 0.02),
        sim_aci_within_0p02=(
            float(candidate['sim/ACI']) >=
            float(sym['sim/ACI']) - 0.02),
        sim_a_rmse_within_2deg=(
            float(candidate['sim/A-RMSE(deg)']) <=
            float(sym['sim/A-RMSE(deg)']) + 2.0))
    return dict(
        tolerances=dict(
            center_drop_pp=1.0,
            mean_riou_drop=0.03,
            dfr_increase_pp_per_frame=0.75,
            aci_drop=0.02,
            sim_a_rmse_increase_deg=2.0),
        checks=checks,
        passed=all(checks.values()))


def main():
    args = parse_args()
    candidate_path, candidate_boxes = _load_results(args.candidate_results)
    sym_reference_path, sym_reference_boxes = _load_results(
        args.sym_reference_results)
    ann_dir = os.path.join(
        os.path.abspath(os.fspath(args.data_root)), 'val', 'annfiles')
    ann_dir, metadata, domain_counts = _annotations(ann_dir)
    candidate_metrics = _metrics(metadata, candidate_boxes)
    sym_metrics = _metrics(metadata, sym_reference_boxes)

    audit_path = os.path.abspath(os.fspath(args.source_val_audit))
    with open(audit_path, 'rb') as handle:
        audit_bytes = handle.read()
    audit = audit_payload(
        json.loads(audit_bytes.decode('utf-8')), audit_bytes,
        os.path.join(os.path.abspath(args.data_root), 'val'),
        'source-val')
    dino_metrics = audit['native_dino_baseline']['metrics']
    average_gain_gate = relaxed_composite_gate(
        candidate_metrics, dino_metrics,
        min_composite_gain=args.min_composite_gain,
        reference_policy='native_dino_source_val')
    geometry_preservation = _sym_geometry_preservation(
        candidate_metrics, sym_metrics)
    checkpoint_path, checkpoint_hash, checkpoint_contract, v2 = (
        _checkpoint_contract(
            args.candidate_checkpoint,
            args.expected_candidate_sha256))
    passed = bool(
        average_gain_gate['passed'] and geometry_preservation['passed'])
    report = dict(
        protocol=V2_PROTOCOL if v2 else PROTOCOL,
        metric_protocol_version=2,
        evidence_boundary='source_val_only',
        target_data_read=False,
        fixed_test_read=False,
        input=dict(
            candidate_results=candidate_path,
            candidate_results_sha256=_sha256(candidate_path),
            candidate_checkpoint=checkpoint_path,
            candidate_checkpoint_sha256=checkpoint_hash,
            candidate_checkpoint_contract=checkpoint_contract,
            source_val_audit=audit_path,
            source_val_audit_sha256=_sha256(audit_path),
            sym_reference_results=sym_reference_path,
            sym_reference_results_sha256=_sha256(sym_reference_path),
            ann_dir=ann_dir,
            frame_count=738,
            domain_counts=domain_counts),
        native_dino_reference_metrics=dino_metrics,
        sym_eood_reference_metrics=sym_metrics,
        candidate_metrics=candidate_metrics,
        average_gain_over_native_dino=average_gain_gate,
        sym_eood_geometry_preservation=geometry_preservation,
        passed=passed,
        eligible_for_checkpoint_promotion=passed,
        eligible_for_fixed_test=False,
        eligible_for_unknown_sequence_claim=False,
        decision=(
            ('ALLOW_K1_ANCHORED_CAUSAL_PHASE_CHECKPOINT_PROMOTION'
             if v2 else 'ALLOW_CAUSAL_HISTORY_CHECKPOINT_PROMOTION')
            if passed else 'STOP_CAUSAL_HISTORY_SOURCE_GATE_FAILED'))
    output = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
