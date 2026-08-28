"""Promote the uniquely selected Dual-Tower V2.1 source checkpoint.

Promotion is deliberately separate from source validation.  It verifies the
source-gate report and every referenced artifact, extracts only the geometry
refiner state, and emits an immutable source-gated runtime checkpoint.  No
fixed TEST data are read here.
"""

import argparse
import hashlib
import json
import os
import re

import torch


CONTRACT_KEY = 'geometry_refiner_checkpoint_contract'
SOURCE_GATE_PROTOCOL = 'dual_tower_v21_relaxed_composite_source_gate_v1'
PROMOTION_PROTOCOL = 'source_gated_dual_tower_v21_promotion_v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-checkpoint', required=True)
    parser.add_argument('--candidate-results', required=True)
    parser.add_argument('--source-gate', required=True)
    parser.add_argument('--expected-candidate-sha256', required=True)
    parser.add_argument('--expected-source-gate-sha256', required=True)
    parser.add_argument('--expected-epoch', type=int, default=7)
    parser.add_argument('--out-checkpoint', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        raw = handle.read()
    return absolute, hashlib.sha256(raw).hexdigest(), json.loads(
        raw.decode('utf-8'))


def _selected_epoch(path):
    match = re.search(r'(?:^|[/\\])epoch_(\d+)\.pth$', os.fspath(path))
    if match is None:
        raise RuntimeError('Candidate checkpoint filename has no epoch')
    return int(match.group(1))


def _validate_gate(gate, checkpoint_hash, results_hash, expected_epoch):
    required = dict(
        protocol=SOURCE_GATE_PROTOCOL,
        evidence_boundary='source_val_only',
        target_data_read=False,
        fixed_test_read=False,
        passed=True,
        eligible_for_checkpoint_promotion=True,
        eligible_for_fixed_test=False,
        decision='ALLOW_DUAL_TOWER_V21_CHECKPOINT_PROMOTION')
    failures = [
        '{}={!r} expected {!r}'.format(key, gate.get(key), expected)
        for key, expected in required.items()
        if gate.get(key) != expected]
    inputs = dict(gate.get('input') or {})
    if inputs.get('candidate_checkpoint_sha256') != checkpoint_hash:
        failures.append('candidate checkpoint SHA256 disagrees with gate')
    if inputs.get('candidate_results_sha256') != results_hash:
        failures.append('candidate results SHA256 disagrees with gate')
    if (_selected_epoch(inputs.get('candidate_checkpoint', '')) !=
            expected_epoch):
        failures.append('source gate selected a different epoch')
    relaxed = dict(gate.get('relaxed_composite_gate') or {})
    if relaxed.get('passed') is not True:
        failures.append('nested relaxed composite gate did not pass')
    if failures:
        raise RuntimeError('Source-gate promotion failed: ' + '; '.join(
            failures))


def _extract_refiner_state(payload):
    state = dict(payload.get('state_dict') or payload)
    selected = {}
    for key, value in state.items():
        for prefix in ('module.geometry_refiner.', 'geometry_refiner.'):
            if key.startswith(prefix):
                selected['geometry_refiner.' + key[len(prefix):]] = value
                break
    if not selected or any(not torch.is_tensor(value)
                           for value in selected.values()):
        raise RuntimeError(
            'Candidate contains no valid geometry-refiner state')
    return selected


def _validate_candidate_contract(contract):
    required = dict(
        protocol='source_only_dual_tower_size_refinement_v21',
        architecture='dual_tower_size_pose_v2',
        source_train_frames=2781,
        source_val_frames=738,
        target_data_read=False,
        fixed_test_read=False,
        source_gate_passed=False,
        domain_routing=False,
        sequence_frame_routing=False,
        temporal_state=False,
        source_adjacent_pair_supervision=True,
        inference_sequence_input=False,
        train_size_tower=True,
        train_pose_tower=False,
        train_roi_extractor=False)
    failures = [
        '{}={!r} expected {!r}'.format(key, contract.get(key), expected)
        for key, expected in required.items()
        if contract.get(key) != expected]
    if failures:
        raise RuntimeError('Candidate contract failed: ' + '; '.join(
            failures))


def promote(candidate_checkpoint, candidate_results, source_gate,
            expected_candidate_sha256, expected_source_gate_sha256,
            expected_epoch,
            out_checkpoint, out_json):
    candidate_path = os.path.abspath(os.fspath(candidate_checkpoint))
    results_path = os.path.abspath(os.fspath(candidate_results))
    candidate_hash = _sha256(candidate_path)
    results_hash = _sha256(results_path)
    if candidate_hash.lower() != str(expected_candidate_sha256).lower():
        raise RuntimeError('Candidate checkpoint SHA256 mismatch')
    if _selected_epoch(candidate_path) != int(expected_epoch):
        raise RuntimeError('Candidate checkpoint is not the locked epoch')
    gate_path, gate_hash, gate = _read_json(source_gate)
    if gate_hash.lower() != str(expected_source_gate_sha256).lower():
        raise RuntimeError('Source-gate report SHA256 mismatch')
    _validate_gate(gate, candidate_hash, results_hash, int(expected_epoch))

    payload = torch.load(candidate_path, map_location='cpu')
    candidate_contract = dict(payload.get('meta') or {}).get(CONTRACT_KEY)
    if not isinstance(candidate_contract, dict):
        raise RuntimeError('Candidate checkpoint has no refiner contract')
    _validate_candidate_contract(candidate_contract)
    state = _extract_refiner_state(payload)
    promoted_contract = dict(candidate_contract)
    promoted_contract.update(dict(
        protocol=PROMOTION_PROTOCOL,
        source_gate_passed=True,
        selected_source_epoch=int(expected_epoch),
        selected_source_checkpoint_sha256=candidate_hash,
        selected_source_results_sha256=results_hash,
        source_gate_report_sha256=gate_hash,
        promotion_before_fixed_test=True))
    output_path = os.path.abspath(os.fspath(out_checkpoint))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dict(
        state_dict=state,
        meta={CONTRACT_KEY: promoted_contract}), output_path)
    output_hash = _sha256(output_path)
    report = dict(
        protocol=PROMOTION_PROTOCOL,
        evidence_boundary='source_gate_only',
        target_data_read=False,
        fixed_test_read=False,
        input=dict(
            candidate_checkpoint=candidate_path,
            candidate_checkpoint_sha256=candidate_hash,
            candidate_results=results_path,
            candidate_results_sha256=results_hash,
            source_gate=gate_path,
            source_gate_sha256=gate_hash,
            selected_epoch=int(expected_epoch)),
        output=dict(
            checkpoint=output_path,
            checkpoint_sha256=output_hash,
            tensor_count=len(state),
            contract=promoted_contract),
        passed=True,
        eligible_for_one_fixed_test=True,
        eligible_for_unknown_sequence_claim=False,
        decision='ALLOW_ONE_DUAL_TOWER_V21_FIXED_TEST')
    report_path = os.path.abspath(os.fspath(out_json))
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    return report


def main():
    args = parse_args()
    report = promote(
        args.candidate_checkpoint, args.candidate_results,
        args.source_gate, args.expected_candidate_sha256,
        args.expected_source_gate_sha256, args.expected_epoch,
        args.out_checkpoint, args.out_json)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
