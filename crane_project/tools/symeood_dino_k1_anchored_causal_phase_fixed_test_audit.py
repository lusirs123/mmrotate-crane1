"""Audit the one authorized causal-phase V2 fixed TEST result.

The promoted epoch is immutable.  TEST output may only confirm or reject the
pre-registered system; it cannot select another epoch or update parameters.
Raw metrics are the primary result.  An optional operational-validity manifest
adds a clearly secondary diagnostic and never changes the primary decision.
"""

import argparse
import hashlib
import json
import os

from crane_project.tools.symeood_dino_application_domain_v4_audit import (
    _validate_measurement_validity)
from crane_project.tools.symeood_dino_dual_tower_v21_fixed_test_audit import (
    EXPECTED_FRAME_COUNT, _all_lane_boxes, _annotations, _load_results,
    _metrics, _sha256)
from crane_project.utils.geometry_refiner_source_gate import (
    relaxed_composite_gate)


PROMOTION_PROTOCOL = 'source_gated_k1_anchored_causal_phase_refiner_v2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-results', required=True)
    parser.add_argument('--candidate-checkpoint', required=True)
    parser.add_argument('--promotion-json', required=True)
    parser.add_argument('--k1-reference-results', required=True)
    parser.add_argument('--all-lane-audit', required=True)
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--measurement-validity-json')
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _promotion(path, checkpoint_path):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        raw = handle.read()
    report = json.loads(raw.decode('utf-8'))
    required = dict(
        protocol=PROMOTION_PROTOCOL,
        evidence_boundary='source_gate_only',
        target_data_read=False, fixed_test_read=False, passed=True,
        eligible_for_one_fixed_test=True,
        decision='ALLOW_ONE_K1_ANCHORED_CAUSAL_PHASE_FIXED_TEST')
    failures = ['{}={!r}'.format(key, report.get(key))
                for key, expected in required.items()
                if report.get(key) != expected]
    output = dict(report.get('output') or {})
    contract = dict(output.get('contract') or {})
    checkpoint_hash = _sha256(checkpoint_path)
    if output.get('checkpoint_sha256') != checkpoint_hash:
        failures.append('promoted checkpoint SHA256 mismatch')
    required_contract = dict(
        selected_source_epoch=10,
        selection_policy='min_worst_mcml_then_max_combined_riou_v1',
        source_gate_passed=True, promotion_before_fixed_test=True,
        one_fixed_test_only=True, fixed_test_consumed=False,
        domain_routing=False, sequence_frame_routing=False,
        temporal_state=False)
    failures.extend(
        ['contract {}={!r}'.format(key, contract.get(key))
         for key, expected in required_contract.items()
         if contract.get(key) != expected])
    if failures:
        raise RuntimeError('Promotion validation failed: ' + '; '.join(
            failures))
    return absolute, hashlib.sha256(raw).hexdigest(), checkpoint_hash, report


def _geometry_guard(candidate, k1):
    checks = dict(
        real_center_within_1pp=(candidate['real/R_center(%)'] >=
                                k1['real/R_center(%)'] - 1.0),
        sim_center_within_1pp=(candidate['sim/R_center(%)'] >=
                               k1['sim/R_center(%)'] - 1.0),
        real_riou_within_0p03=(candidate['real/mean_RIoU'] >=
                               k1['real/mean_RIoU'] - 0.03),
        sim_riou_within_0p03=(candidate['sim/mean_RIoU'] >=
                              k1['sim/mean_RIoU'] - 0.03),
        real_dfr_within_0p75pp=(candidate['real/DFR(%/frame)'] <=
                                k1['real/DFR(%/frame)'] + 0.75),
        sim_dfr_within_0p75pp=(candidate['sim/DFR(%/frame)'] <=
                               k1['sim/DFR(%/frame)'] + 0.75),
        real_aci_within_0p02=(candidate['real/ACI'] >=
                              k1['real/ACI'] - 0.02),
        sim_aci_within_0p02=(candidate['sim/ACI'] >=
                             k1['sim/ACI'] - 0.02),
        sim_a_rmse_within_2deg=(candidate['sim/A-RMSE(deg)'] <=
                                k1['sim/A-RMSE(deg)'] + 2.0))
    return dict(
        tolerances=dict(center_drop_pp=1.0, mean_riou_drop=0.03,
                        dfr_increase_pp_per_frame=0.75, aci_drop=0.02,
                        sim_a_rmse_increase_deg=2.0),
        checks=checks, passed=all(checks.values()))


def _primary_gate(candidate, k1):
    composite = relaxed_composite_gate(
        candidate, k1, min_composite_gain=0.005,
        reference_policy='symeood_k1_fixed_test')
    geometry = _geometry_guard(candidate, k1)
    control = dict(
        real_tdr_ge_99=candidate['real/TDR_w10(%)'] >= 99.0,
        sim_tdr_ge_99=candidate['sim/TDR_w10(%)'] >= 99.0,
        real_mcml_max_le_6=candidate['real/MCML_max(frames)'] <= 6,
        sim_mcml_max_le_5=candidate['sim/MCML_max(frames)'] <= 5)
    composite_gain_passed = bool(
        composite['checks']['composite_mean_relative_gain'])
    return dict(
        preregistered_before_fixed_test=True,
        reference='symeood_k1_epoch24_fixed_test',
        # The shared helper also contains its older <=5 MCML guardrail.  Only
        # its scalar composite utility is reused here; the explicitly
        # preregistered control block below implements the accepted raw-real
        # <=6 engineering limit.
        composite_utility=composite,
        composite_gain_passed=composite_gain_passed,
        geometry_nonregression=geometry,
        control_checks=control,
        passed=(composite_gain_passed and geometry['passed']
                and all(control.values())))


def _measurement_slice(path, metadata, keys, candidate_boxes):
    if path is None:
        return None
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        raw = handle.read()
    payload = json.loads(raw.decode('utf-8'))
    validity_records = []
    for meta, key in zip(metadata, keys):
        sequence = '{}_{}'.format(meta['domain'], meta['seq_id'])
        validity_records.append(dict(
            filename=key + '.jpg', sequence=sequence,
            frame=int(meta['frame_id'])))
    invalid, normalized = _validate_measurement_validity(
        payload, validity_records, 'fixed-target')
    keep = []
    excluded = []
    for index, (meta, box) in enumerate(zip(metadata, candidate_boxes)):
        sequence = '{}_{}'.format(meta['domain'], meta['seq_id'])
        key = (sequence, int(meta['frame_id']))
        if meta['domain'] == 'real' and key in invalid:
            excluded.append(dict(sequence=sequence, frame=key[1],
                                 reason=invalid[key]))
        else:
            keep.append((meta, box))
    return dict(
        protocol=payload.get('protocol'), path=absolute,
        sha256=hashlib.sha256(raw).hexdigest(),
        used_for_primary_decision=False,
        excluded_real_frame_count=len(excluded),
        remaining_frame_count=len(keep), sequences=normalized,
        metrics=_metrics([row[0] for row in keep],
                         [row[1] for row in keep]))


def main():
    args = parse_args()
    candidate_path, candidate_boxes = _load_results(args.candidate_results)
    k1_path, k1_boxes = _load_results(args.k1_reference_results)
    ann_dir, metadata, keys, counts = _annotations(args.ann_dir)
    audit_path, dino_boxes, _unused_sym = _all_lane_boxes(
        args.all_lane_audit, keys)
    checkpoint_path = os.path.abspath(args.candidate_checkpoint)
    promotion_path, promotion_hash, checkpoint_hash, promotion = _promotion(
        args.promotion_json, checkpoint_path)
    candidate_metrics = _metrics(metadata, candidate_boxes)
    k1_metrics = _metrics(metadata, k1_boxes)
    dino_metrics = _metrics(metadata, dino_boxes)
    gate = _primary_gate(candidate_metrics, k1_metrics)
    measurement = _measurement_slice(
        args.measurement_validity_json, metadata, keys, candidate_boxes)
    report = dict(
        protocol='k1_anchored_causal_phase_one_time_fixed_test_v1',
        metric_protocol_version=2,
        evidence_boundary='fixed_test_once_after_source_promotion',
        source_selected_epoch=10,
        parameter_update_after_test=False,
        epoch_reselection_after_test=False,
        domain_routing=False, sequence_frame_routing=False,
        temporal_inference_state=False, dino_detector_rerun=False,
        input=dict(
            candidate_results=candidate_path,
            candidate_results_sha256=_sha256(candidate_path),
            k1_reference_results=k1_path,
            k1_reference_results_sha256=_sha256(k1_path),
            candidate_checkpoint=checkpoint_path,
            candidate_checkpoint_sha256=checkpoint_hash,
            promotion_json=promotion_path,
            promotion_json_sha256=promotion_hash,
            promotion=promotion, all_lane_audit=audit_path,
            all_lane_audit_sha256=_sha256(audit_path), ann_dir=ann_dir,
            frame_count=EXPECTED_FRAME_COUNT, domain_counts=counts),
        raw_metrics=dict(candidate=candidate_metrics,
                         symeood_k1=k1_metrics, native_dino=dino_metrics),
        primary_fixed_test_gate=gate,
        measurement_validity_diagnostic=measurement,
        passed=gate['passed'], eligible_for_unknown_sequence_claim=False,
        eligible_for_parameter_tuning_from_this_report=False,
        eligible_for_epoch_reselection_from_this_report=False,
        decision=('PASS_K1_ANCHORED_CAUSAL_PHASE_ONE_TIME_FIXED_TEST'
                  if gate['passed'] else
                  'STOP_K1_ANCHORED_CAUSAL_PHASE_FIXED_TEST_FAILED'))
    output = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not gate['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
