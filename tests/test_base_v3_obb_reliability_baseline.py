import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from crane_project.tools import base_v3_obb_reliability_baseline as baseline


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding='utf-8')


def _dota_line(cx, cy, width, height):
    x0, x1 = cx - width / 2, cx + width / 2
    y0, y1 = cy - height / 2, cy + height / 2
    return '{} {} {} {} {} {} {} {} grab 0\n'.format(
        x0, y0, x1, y0, x1, y1, x0, y1)


def _make_case(tmp_path):
    promoted = tmp_path / 'promoted.pth'
    promoted.write_bytes(b'promoted-refiner')
    k1_checkpoint = tmp_path / 'k1.pth'
    k1_checkpoint.write_bytes(b'k1-checkpoint')

    base_results = tmp_path / 'base.pkl'
    with base_results.open('wb') as stream:
        pickle.dump([
            [np.asarray([[10, 10, 8, 4, 0, 1.0]])],
            [np.asarray([[21, 20, 8, 4, 0, 1.0]])],
        ], stream)
    k1_results = tmp_path / 'k1.pkl'
    with k1_results.open('wb') as stream:
        pickle.dump([
            [np.asarray([[10, 10, 8, 4, 0, 0.4]])],
            [np.empty((0, 6))],
        ], stream)

    audit = tmp_path / 'audit.json'
    audit_payload = dict(
        protocol=baseline.AUDIT_PROTOCOL,
        metadata=dict(fusion_policy='sym_eood_proposal_dino_roi_union'),
        records=[
            dict(filename='real_seq07_0001.jpg',
                 selected_source='dino_native',
                 sym_eood_box=[100, 100, 8, 4, 0, 0.99],
                 dino_native_box=[11, 10, 8, 4, 0, 0.8],
                 dino_native_common_score=0.8),
            dict(filename='sim_seq10_0001.jpg',
                 selected_source='sym_eood',
                 sym_eood_box=[200, 200, 8, 4, 0, 0.98],
                 dino_native_box=[20, 20, 8, 4, 0, 0.9],
                 dino_native_common_score=0.9),
        ])
    _write_json(audit, audit_payload)

    promotion = tmp_path / 'promotion.json'
    promotion_payload = dict(
        evidence_boundary='source_gate_only', fixed_test_read=False,
        target_data_read=False,
        input=dict(candidate_results_sha256=_sha(base_results)),
        output=dict(checkpoint_sha256=_sha(promoted)))
    _write_json(promotion, promotion_payload)

    gate = tmp_path / 'gate.json'
    gate_payload = dict(
        evidence_boundary='source_val_only', fixed_test_read=False,
        target_data_read=False,
        input=dict(
            candidate_results_sha256=_sha(base_results),
            sym_reference_results_sha256=_sha(k1_results),
            source_val_audit_sha256=_sha(audit)))
    _write_json(gate, gate_payload)

    ann_dir = tmp_path / 'annfiles'
    ann_dir.mkdir()
    (ann_dir / 'real_seq07_0001.txt').write_text(
        _dota_line(10, 10, 8, 4), encoding='utf-8')
    (ann_dir / 'sim_seq10_0001.txt').write_text(
        _dota_line(20, 20, 8, 4), encoding='utf-8')

    contract = tmp_path / 'contract.json'
    contract_payload = dict(
        protocol=baseline.CONTRACT_PROTOCOL,
        evidence_boundary='official_source_val_exploratory_only',
        expected_inputs=dict(
            promoted_refiner_sha256=_sha(promoted),
            promotion_report_sha256=_sha(promotion),
            source_gate_sha256=_sha(gate),
            base_v3_results_sha256=_sha(base_results),
            k1_checkpoint_sha256=_sha(k1_checkpoint),
            k1_results_sha256=_sha(k1_results),
            source_val_audit_sha256=_sha(audit)),
        source_val_scope=dict(
            frame_count=2, domain_counts=dict(real=1, sim=1),
            expected_k1_present=1, expected_dino_fallback=1,
            expected_base_v3_present=2),
        score_baseline=dict(
            coverage_targets=[1.0, 0.5],
            component_error_thresholds=dict(
                center_error_px=[5.0], long_relative_error=[0.1],
                short_relative_error=[0.1], angle_error_deg=[3.0]),
            minimum_fallback_samples_for_cross_source_calibration_claim=2))
    _write_json(contract, contract_payload)

    args = argparse.Namespace(
        contract=str(contract), promoted_refiner=str(promoted),
        promotion_report=str(promotion), source_gate=str(gate),
        base_v3_results=str(base_results),
        k1_checkpoint=str(k1_checkpoint), k1_results=str(k1_results),
        source_val_audit=str(audit), ann_dir=str(ann_dir),
        out_json=str(tmp_path / 'output.json'))
    return args


def test_build_uses_actual_anchor_branch_and_not_legacy_union(tmp_path):
    args = _make_case(tmp_path)
    payload = baseline.build(args)

    assert [row['anchor_source'] for row in payload['records']] == [
        'k1', 'dino_fallback']
    assert [row['anchor_score'] for row in payload['records']] == [0.4, 0.9]
    assert payload['provenance_checks'][
        'legacy_union_selected_source_used'] is False
    assert payload['provenance_checks'][
        'legacy_union_sym_eood_box_exact_k1_count'] == 0
    assert payload['final_score_audit']['role'] == 'prohibited_as_confidence'
    assert payload['risk_coverage']['global_rank'][1][
        'minimum_retained_score'] == 0.9


def test_build_fails_closed_on_identity_mismatch(tmp_path):
    args = _make_case(tmp_path)
    Path(args.promoted_refiner).write_bytes(b'tampered')

    with pytest.raises(RuntimeError, match='promoted refiner identity mismatch'):
        baseline.build(args)


def test_write_exact_refuses_different_existing_output(tmp_path):
    output = tmp_path / 'result.json'
    baseline._write_exact(output, {'value': 1})
    baseline._write_exact(output, {'value': 1})

    with pytest.raises(RuntimeError, match='Refusing to overwrite'):
        baseline._write_exact(output, {'value': 2})
