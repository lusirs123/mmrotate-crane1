#!/usr/bin/env python3
"""Detection/control non-regression half of the seq11 dual source gate."""

import argparse
import json
import os

from crane_project.tools.symeood_dino_application_domain_v4_audit import (
    audit_payload)
from crane_project.tools.symeood_dino_causal_history_source_gate import (
    SEQ11_BLOCKSPLIT_LEGACY_PROTOCOL, _checkpoint_contract, _sha256,
    _seq11_blocksplit_legacy_preservation)
from crane_project.tools.symeood_dino_dual_tower_v2_audit import (
    _annotations, _load_results, _metrics)
from crane_project.utils.depth_interface_geometry_gate import (
    depth_interface_geometry_gate)
from crane_project.utils.geometry_refiner_source_gate import (
    relaxed_composite_gate)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-results', required=True)
    parser.add_argument('--candidate-checkpoint', required=True)
    parser.add_argument('--source-val-audit', required=True)
    parser.add_argument('--sym-reference-results', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument('--expected-candidate-sha256')
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def audit(args):
    candidate_path, candidate_boxes = _load_results(args.candidate_results)
    sym_path, sym_boxes = _load_results(args.sym_reference_results)
    ann_dir, metadata, domain_counts = _annotations(os.path.join(
        os.path.abspath(os.fspath(args.data_root)), 'val', 'annfiles'))
    candidate_metrics = _metrics(metadata, candidate_boxes)
    sym_metrics = _metrics(metadata, sym_boxes)
    audit_path = os.path.abspath(os.fspath(args.source_val_audit))
    with open(audit_path, 'rb') as handle:
        audit_bytes = handle.read()
    dino = audit_payload(
        json.loads(audit_bytes.decode('utf-8')), audit_bytes,
        os.path.join(os.path.abspath(args.data_root), 'val'),
        'source-val')['native_dino_baseline']['metrics']
    (checkpoint_path, checkpoint_sha, checkpoint_contract,
     _v2, v3, e2, _seq11, blocksplit) = _checkpoint_contract(
        args.candidate_checkpoint, args.expected_candidate_sha256)
    if not v3 or e2 or not blocksplit:
        raise RuntimeError('Legacy half requires an E1 48/11 checkpoint')
    preservation = _seq11_blocksplit_legacy_preservation(
        candidate_metrics, sym_metrics)
    average_k1 = relaxed_composite_gate(
        candidate_metrics, sym_metrics, min_composite_gain=0.0,
        reference_policy='formal_k1_source_val_738')
    depth = depth_interface_geometry_gate(
        metadata, candidate_boxes, sym_boxes)
    passed = bool(preservation['passed'] and average_k1['passed'])
    return dict(
        protocol=SEQ11_BLOCKSPLIT_LEGACY_PROTOCOL,
        metric_protocol_version=2,
        evidence_boundary='legacy_source_val_738_only',
        target_data_read=False, fixed_test_read=False,
        input=dict(
            candidate_results=candidate_path,
            candidate_results_sha256=_sha256(candidate_path),
            candidate_checkpoint=checkpoint_path,
            candidate_checkpoint_sha256=checkpoint_sha,
            candidate_checkpoint_contract=checkpoint_contract,
            source_val_audit=audit_path,
            source_val_audit_sha256=_sha256(audit_path),
            sym_reference_results=sym_path,
            sym_reference_results_sha256=_sha256(sym_path),
            ann_dir=ann_dir, frame_count=738, domain_counts=domain_counts),
        native_dino_reference_metrics=dino,
        sym_eood_reference_metrics=sym_metrics,
        candidate_metrics=candidate_metrics,
        average_gain_over_formal_k1=average_k1,
        sym_eood_detection_control_preservation=preservation,
        depth_interface_geometry_gate=dict(
            depth, hard_gate=False, role='diagnostic_depth_postponed'),
        passed=passed,
        eligible_for_dual_source_gate=passed,
        eligible_for_checkpoint_promotion=False,
        eligible_for_fixed_test=False,
        eligible_for_unknown_sequence_claim=False,
        decision=(
            'ALLOW_SEQ11_LEGACY_NONREGRESSION_HALF' if passed else
            'STOP_SEQ11_LEGACY_NONREGRESSION_FAILED'))


def main():
    args = parse_args()
    report = audit(args)
    output = os.path.abspath(os.fspath(args.out_json))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print('[seq11-legacy-gate] metrics={}'.format(
        report['candidate_metrics']))
    print('[seq11-legacy-gate] decision={}'.format(report['decision']))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
