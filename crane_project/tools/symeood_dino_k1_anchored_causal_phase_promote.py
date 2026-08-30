"""Promote the uniquely selected K1-anchored causal-phase V2 checkpoint.

The selection is recomputed from all ten source-val gate reports.  This tool
does not read fixed TEST data and refuses every checkpoint except the locked
epoch-10 source artifact produced by the formal seed-3407 run.
"""

import argparse
import hashlib
import json
import os
import re

import torch


CONTRACT_KEY = 'geometry_refiner_checkpoint_contract'
SOURCE_GATE_PROTOCOL = 'k1_anchored_causal_phase_refiner_source_gate_v2'
PROMOTION_PROTOCOL = 'source_gated_k1_anchored_causal_phase_refiner_v2'
SELECTION_POLICY = 'min_worst_mcml_then_max_combined_riou_v1'
LOCKED_EPOCH = 10
LOCKED_CHECKPOINT_SHA256 = (
    '758a576dc586b334fdd03257cf3691cd5cde7bdf0759d60d685d8d988d10e5ae')
LOCKED_RESULTS_SHA256 = (
    '8ce7b2680970a456ecc9669d696d7941cce030c24cc098051670e0c9dde07cdb')
LOCKED_GATE_SHA256 = {
    1: '122d8718df27070346207cf452a3030dac9058ee96d63a6c0d7891f7ab002177',
    2: '23fca9797de85fcb1a6e259041d9f459e31f9406d0849ce86277bf97fd4deb45',
    3: 'fd293a605e0fda9d5cf050317f8e37a44ca1ff002141207171b953a23e3238d8',
    4: '567e37220e51431bb01dfc5f5659d8a9689a533eafe84855566458fe034cf01c',
    5: 'b617cefa02c096cbe5134e76be387847ec0495b1b782474f22b3c7211a72482c',
    6: 'cb923d5f279f54ce966ce94e8d368a061db52110ff856a2edb192264768fec2b',
    7: 'daf0c1d015c1248ee6d4fbc2448203c8d3630a8fccbe6064c7955b0aa6e7abe8',
    8: 'c9822b98e179ba5d869088c932ececa9fd9f8f68129d8f8eb1be34a41d21fef0',
    9: '5ea0a395c8b65b866c7285aaee0b5c6e5eced147f9a03f700c0695c07b4d974f',
    10: '23b54eff1c66b3c61efbe9d01c171b5081af115d2145a2e0f83522f004c1d40c',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-checkpoint', required=True)
    parser.add_argument('--candidate-results', required=True)
    parser.add_argument('--source-gates', nargs='+', required=True)
    parser.add_argument('--expected-selected-gate-sha256', required=True)
    parser.add_argument('--out-checkpoint', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _epoch(path):
    match = re.search(r'(?:^|[/\\])epoch_(\d+)\.pth$', os.fspath(path))
    if match is None:
        raise RuntimeError('Checkpoint filename has no epoch_<N>.pth suffix')
    return int(match.group(1))


def _read_gate(path):
    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'rb') as handle:
        raw = handle.read()
    gate = json.loads(raw.decode('utf-8'))
    required = dict(
        protocol=SOURCE_GATE_PROTOCOL,
        evidence_boundary='source_val_only',
        target_data_read=False,
        fixed_test_read=False,
        passed=True,
        eligible_for_checkpoint_promotion=True,
        eligible_for_fixed_test=False,
        decision='ALLOW_K1_ANCHORED_CAUSAL_PHASE_CHECKPOINT_PROMOTION')
    failures = ['{}={!r}'.format(key, gate.get(key))
                for key, expected in required.items()
                if gate.get(key) != expected]
    inputs = dict(gate.get('input') or {})
    epoch = _epoch(inputs.get('candidate_checkpoint', ''))
    if inputs.get('candidate_checkpoint_sha256') is None:
        failures.append('gate lacks candidate checkpoint SHA256')
    if inputs.get('candidate_results_sha256') is None:
        failures.append('gate lacks candidate results SHA256')
    if failures:
        raise RuntimeError('Invalid source gate: ' + '; '.join(failures))
    metrics = dict(gate.get('candidate_metrics') or {})
    required_metrics = (
        'real/MCML_max(frames)', 'sim/MCML_max(frames)',
        'real/mean_RIoU', 'sim/mean_RIoU',
        'real/DFR(%/frame)', 'sim/DFR(%/frame)')
    if any(key not in metrics for key in required_metrics):
        raise RuntimeError('Source gate lacks selection metrics')
    return dict(path=absolute, sha256=hashlib.sha256(raw).hexdigest(),
                epoch=epoch, gate=gate, metrics=metrics)


def _rank(row):
    metrics = row['metrics']
    return (
        max(int(metrics['real/MCML_max(frames)']),
            int(metrics['sim/MCML_max(frames)'])),
        -(float(metrics['real/mean_RIoU']) +
          float(metrics['sim/mean_RIoU'])),
        float(metrics['real/DFR(%/frame)']) +
        float(metrics['sim/DFR(%/frame)']),
        int(row['epoch']))


def _validate_candidate_contract(contract):
    required = dict(
        protocol='source_only_k1_anchored_causal_phase_refiner_v2',
        architecture='k1_anchored_causal_phase_refiner_v2',
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
        fixed_target_parameter_selection=False)
    failures = ['{}={!r} expected {!r}'.format(key, contract.get(key), value)
                for key, value in required.items()
                if contract.get(key) != value]
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
            expected_selected_gate_sha256, out_checkpoint, out_json):
    if len(source_gates) != 10:
        raise RuntimeError('Promotion requires exactly ten source gate reports')
    gates = [_read_gate(path) for path in source_gates]
    if sorted(row['epoch'] for row in gates) != list(range(1, 11)):
        raise RuntimeError('Source gates must cover each epoch 1..10 exactly')
    for row in gates:
        if row['sha256'] != LOCKED_GATE_SHA256[row['epoch']]:
            raise RuntimeError(
                'Locked source-gate SHA256 mismatch at epoch {}'.format(
                    row['epoch']))
    selected = min(gates, key=_rank)
    if selected['epoch'] != LOCKED_EPOCH:
        raise RuntimeError('Pre-registered selection did not choose epoch 10')
    if selected['sha256'].lower() != expected_selected_gate_sha256.lower():
        raise RuntimeError('Selected source-gate SHA256 mismatch')

    checkpoint_path = os.path.abspath(os.fspath(candidate_checkpoint))
    results_path = os.path.abspath(os.fspath(candidate_results))
    checkpoint_hash = _sha256(checkpoint_path)
    results_hash = _sha256(results_path)
    if _epoch(checkpoint_path) != LOCKED_EPOCH:
        raise RuntimeError('Candidate checkpoint is not locked epoch 10')
    if checkpoint_hash != LOCKED_CHECKPOINT_SHA256:
        raise RuntimeError('Locked epoch-10 checkpoint SHA256 mismatch')
    if results_hash != LOCKED_RESULTS_SHA256:
        raise RuntimeError('Locked epoch-10 results SHA256 mismatch')
    selected_input = dict(selected['gate'].get('input') or {})
    if selected_input.get('candidate_checkpoint_sha256') != checkpoint_hash:
        raise RuntimeError('Selected gate/checkpoint SHA256 disagreement')
    if selected_input.get('candidate_results_sha256') != results_hash:
        raise RuntimeError('Selected gate/results SHA256 disagreement')

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
        selected_source_gate_sha256=selected['sha256'],
        promotion_before_fixed_test=True,
        one_fixed_test_only=True,
        fixed_test_consumed=False))
    output_path = os.path.abspath(os.fspath(out_checkpoint))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dict(state_dict=state,
                    meta={CONTRACT_KEY: promoted_contract}), output_path)
    report = dict(
        protocol=PROMOTION_PROTOCOL,
        evidence_boundary='source_gate_only',
        target_data_read=False, fixed_test_read=False,
        selection=dict(policy=SELECTION_POLICY,
                       evaluated_epochs=list(range(1, 11)),
                       selected_epoch=LOCKED_EPOCH,
                       ranking=[dict(epoch=row['epoch'], rank=list(_rank(row)))
                                for row in sorted(gates, key=_rank)]),
        input=dict(candidate_checkpoint=checkpoint_path,
                   candidate_checkpoint_sha256=checkpoint_hash,
                   candidate_results=results_path,
                   candidate_results_sha256=results_hash,
                   selected_source_gate=selected['path'],
                   selected_source_gate_sha256=selected['sha256']),
        output=dict(checkpoint=output_path,
                    checkpoint_sha256=_sha256(output_path),
                    tensor_count=len(state), contract=promoted_contract),
        passed=True, eligible_for_one_fixed_test=True,
        eligible_for_unknown_sequence_claim=False,
        decision='ALLOW_ONE_K1_ANCHORED_CAUSAL_PHASE_FIXED_TEST')
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
        args.source_gates, args.expected_selected_gate_sha256,
        args.out_checkpoint, args.out_json)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
