"""Relaxed composite source-val gate for Dual-Tower V2.1.

This tool reads only official source-val predictions and annotations.  It does
not run a detector, inspect fixed TEST, change a checkpoint, or select by any
domain/sequence/frame router.  A pass authorizes checkpoint promotion as a
separate auditable action; it does not silently rewrite checkpoint metadata.
"""

import argparse
import hashlib
import json
import os

import torch

from crane_project.tools.symeood_dino_dual_tower_v2_audit import (
    _annotations, _load_results, _metrics)
from crane_project.utils.geometry_refiner_source_gate import (
    relaxed_composite_gate)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-results', required=True)
    parser.add_argument('--reference-results', required=True)
    parser.add_argument('--candidate-checkpoint', required=True)
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--min-composite-gain', type=float, default=0.005)
    parser.add_argument('--expected-candidate-sha256')
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
    if (expected_sha256 is not None and
            observed.lower() != expected_sha256.lower()):
        raise RuntimeError('Candidate checkpoint SHA256 mismatch')
    payload = torch.load(absolute, map_location='cpu')
    contract = dict(payload.get('meta') or {}).get(
        'geometry_refiner_checkpoint_contract')
    if not isinstance(contract, dict):
        raise RuntimeError('Candidate checkpoint has no refiner contract')
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
        '{}={!r} expected {!r}'.format(
            key, contract.get(key), expected)
        for key, expected in required.items()
        if contract.get(key) != expected]
    if failures:
        raise RuntimeError(
            'Candidate checkpoint contract failed: ' + '; '.join(failures))
    return absolute, observed, contract


def main():
    args = parse_args()
    candidate_path, candidate_boxes = _load_results(
        args.candidate_results)
    reference_path, reference_boxes = _load_results(
        args.reference_results)
    ann_dir, metadata, domain_counts = _annotations(args.ann_dir)
    candidate_metrics = _metrics(metadata, candidate_boxes)
    reference_metrics = _metrics(metadata, reference_boxes)
    gate = relaxed_composite_gate(
        candidate_metrics, reference_metrics,
        min_composite_gain=args.min_composite_gain)
    checkpoint_path, checkpoint_hash, checkpoint_contract = (
        _checkpoint_contract(
            args.candidate_checkpoint,
            args.expected_candidate_sha256))
    report = dict(
        protocol='dual_tower_v21_relaxed_composite_source_gate_v1',
        metric_protocol_version=2,
        evidence_boundary='source_val_only',
        target_data_read=False,
        fixed_test_read=False,
        input=dict(
            candidate_results=candidate_path,
            candidate_results_sha256=_sha256(candidate_path),
            reference_results=reference_path,
            reference_results_sha256=_sha256(reference_path),
            candidate_checkpoint=checkpoint_path,
            candidate_checkpoint_sha256=checkpoint_hash,
            candidate_checkpoint_contract=checkpoint_contract,
            ann_dir=ann_dir,
            frame_count=738,
            domain_counts=domain_counts),
        reference_metrics=reference_metrics,
        candidate_metrics=candidate_metrics,
        relaxed_composite_gate=gate,
        passed=gate['passed'],
        eligible_for_checkpoint_promotion=gate['passed'],
        eligible_for_fixed_test=False,
        decision=(
            'ALLOW_DUAL_TOWER_V21_CHECKPOINT_PROMOTION'
            if gate['passed'] else
            'STOP_DUAL_TOWER_V21_SOURCE_GATE_FAILED'))
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
