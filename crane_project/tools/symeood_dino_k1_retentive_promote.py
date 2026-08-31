#!/usr/bin/env python3
"""Promote the source-selected K1-retentive causal-phase V3 checkpoint.

The promotion is derived exclusively from the ten immutable source-val gates.
It never reads TEST data and locks the only fully passing candidate, epoch 9.
"""

import argparse
import hashlib
import json
import os

import torch

from crane_project.tools.symeood_dino_k1_retentive_source_select import (
    SELECTION_POLICY, select)


CONTRACT_KEY = 'geometry_refiner_checkpoint_contract'
PROMOTION_PROTOCOL = 'source_gated_k1_retentive_causal_phase_refiner_v3'
LOCKED_EPOCH = 9
LOCKED_CHECKPOINT_SHA256 = (
    '0183cb51413741149f7624801f73e95b0863ce675a407f3675a5aa6816284f67')
LOCKED_RESULTS_SHA256 = (
    'bc6ed73b633df451ff52500f545b7f5618ece6bd6c567307f3a32f3f31b421f7')
LOCKED_GATE_SHA256 = {
    1: 'cb3d6be8593ba15d7c2fdf61c1251eeea8bfdfc6c461a39456b551152d997e22',
    2: 'efb4cb5966ee182a9398c23fe9df1a0c23e0f026bb25862ba1e0fa90233fa64d',
    3: '25882ff690697814d5ec7a9b6ba0e87dc834a18e33b9dab0a0050c3f9703cc7d',
    4: '52338df233cf106920492e20bd34678da9769d76fbf5a4aea26a9fd06de9290f',
    5: '8f06627de14715258bb04aef9137ba1b715df43ac2303a3aa66c5157eb3ef07d',
    6: '1301e123818ed03a3f72d06b91179ec2998bb132fa8d32c9647feb3697c28919',
    7: '61c1833f6f184d41f1cd41648c39678e7d3e3c17ebc60ffe23dafeade3b03de5',
    8: '4282c1f9ad0cb8f8e4cb226e5fcab8fb15466d93eea3900441a4f715761d99d1',
    9: 'c279024dd7447260d5edaeeaa91bdb63aed768c14db1d8384365ed2a8ff7368f',
    10: '0630baa32d7708827d54a98a33812b78912560c8eaecc79d994ef784e2ce6058',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-checkpoint', required=True)
    parser.add_argument('--candidate-results', required=True)
    parser.add_argument('--source-gates', nargs='+', required=True)
    parser.add_argument('--out-checkpoint', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _gate_hashes(paths):
    hashes = {}
    for path in paths:
        with open(path, 'rb') as handle:
            raw = handle.read()
        gate = json.loads(raw.decode('utf-8'))
        checkpoint = os.fspath(
            dict(gate.get('input') or {}).get('candidate_checkpoint', ''))
        stem = os.path.basename(checkpoint)
        if not (stem.startswith('epoch_') and stem.endswith('.pth')):
            raise RuntimeError('Gate checkpoint has no epoch_<N>.pth name')
        epoch = int(stem[len('epoch_'):-len('.pth')])
        if epoch in hashes:
            raise RuntimeError('Duplicate source gate for epoch {}'.format(epoch))
        hashes[epoch] = hashlib.sha256(raw).hexdigest()
    return hashes


def _validate_candidate_contract(contract):
    required = dict(
        protocol='source_only_k1_retentive_causal_phase_refiner_v3',
        architecture='k1_retentive_causal_phase_refiner_v3',
        frozen_baseline_variant='symeood_k1_epoch24',
        frozen_baseline_config='crane_project/configs/crane_symeood_k1.py',
        frozen_baseline_checkpoint='work_dirs/crane_symeood_k1/epoch_24.pth',
        source_train_frames=2781, source_val_frames=738,
        target_data_read=False, fixed_test_read=False,
        source_gate_passed=False,
        dino_detector_forward_during_training=False,
        cached_dino_proposals_only=True,
        domain_routing=False, sequence_frame_routing=False,
        temporal_state=False, causal_history_input=True,
        history_horizon=4, history_identity_model_input=False,
        current_k1_geometry_anchor=True,
        native_dino_anchor_fallback=True,
        native_dino_current_conditioning=True,
        same_forward_all_domains=True,
        continuous_double_angle_phase=True,
        continuous_k1_retention=True,
        retention_loss_weight=0.25,
        source_adjacent_pair_supervision=True,
        adjacent_pair_identity_model_input=False,
        temporal_size_error_consistency=True,
        temporal_size_loss_weight=0.20,
        inference_sequence_input=False,
        single_gpu_adjacent_pair_training=True,
        fixed_target_parameter_selection=False)
    failures = [
        '{}={!r} expected {!r}'.format(key, contract.get(key), expected)
        for key, expected in required.items()
        if contract.get(key) != expected]
    if failures:
        raise RuntimeError('Candidate contract failed: ' + '; '.join(failures))


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
        raise RuntimeError('Candidate contains no valid refiner tensors')
    return selected


def promote(candidate_checkpoint, candidate_results, source_gates,
            out_checkpoint, out_json):
    if len(source_gates) != 10:
        raise RuntimeError('Promotion requires exactly ten source gates')
    observed_gate_hashes = _gate_hashes(source_gates)
    if observed_gate_hashes != LOCKED_GATE_SHA256:
        raise RuntimeError('Locked source-gate SHA256 set mismatch')
    selection = select(source_gates)
    selected = dict(selection['selected'])
    if selection.get('passing_epochs') != [LOCKED_EPOCH]:
        raise RuntimeError('Locked V3 run must have only epoch 9 passing')
    if selected.get('epoch') != LOCKED_EPOCH:
        raise RuntimeError('Source-only selection did not choose epoch 9')

    checkpoint_path = os.path.abspath(os.fspath(candidate_checkpoint))
    results_path = os.path.abspath(os.fspath(candidate_results))
    checkpoint_hash = _sha256(checkpoint_path)
    results_hash = _sha256(results_path)
    if checkpoint_hash != LOCKED_CHECKPOINT_SHA256:
        raise RuntimeError('Locked epoch-9 checkpoint SHA256 mismatch')
    if results_hash != LOCKED_RESULTS_SHA256:
        raise RuntimeError('Locked epoch-9 results SHA256 mismatch')
    if selected.get('checkpoint_sha256') != checkpoint_hash:
        raise RuntimeError('Selection/checkpoint SHA256 disagreement')
    if selected.get('results_sha256') != results_hash:
        raise RuntimeError('Selection/results SHA256 disagreement')

    payload = torch.load(checkpoint_path, map_location='cpu')
    contract = dict(payload.get('meta') or {}).get(CONTRACT_KEY)
    if not isinstance(contract, dict):
        raise RuntimeError('Candidate checkpoint has no refiner contract')
    _validate_candidate_contract(contract)
    state = _extract_refiner_state(payload)
    promoted_contract = dict(contract)
    promoted_contract.update(dict(
        protocol=PROMOTION_PROTOCOL,
        source_gate_passed=True,
        selected_source_epoch=LOCKED_EPOCH,
        selection_policy=SELECTION_POLICY,
        selected_source_checkpoint_sha256=checkpoint_hash,
        selected_source_results_sha256=results_hash,
        selected_source_gate_sha256=selected['source_gate_sha256'],
        promotion_before_fixed_test=True,
        fixed_benchmark_test=True,
        test_used_for_model_selection=False,
        parameter_update_after_test=False))

    output_path = os.path.abspath(os.fspath(out_checkpoint))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dict(
        state_dict=state,
        meta={CONTRACT_KEY: promoted_contract}), output_path)
    report = dict(
        protocol=PROMOTION_PROTOCOL,
        evidence_boundary='source_gate_only',
        target_data_read=False,
        fixed_test_read=False,
        selection=selection,
        input=dict(
            candidate_checkpoint=checkpoint_path,
            candidate_checkpoint_sha256=checkpoint_hash,
            candidate_results=results_path,
            candidate_results_sha256=results_hash),
        output=dict(
            checkpoint=output_path,
            checkpoint_sha256=_sha256(output_path),
            tensor_count=len(state),
            contract=promoted_contract),
        passed=True,
        eligible_for_fixed_benchmark_test=True,
        eligible_for_unknown_sequence_claim=False,
        decision='ALLOW_K1_RETENTIVE_CAUSAL_PHASE_FIXED_BENCHMARK_TEST')
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
        args.source_gates, args.out_checkpoint, args.out_json)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
