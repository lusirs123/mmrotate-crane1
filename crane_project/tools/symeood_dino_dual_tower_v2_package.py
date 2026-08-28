"""Package Size-e10 and Full-e11 source artifacts as one dual-tower head."""

import argparse
import json
import os

import torch

from crane_project.utils.dual_tower_geometry_refiner_checkpoint import (
    CONTRACT_KEY, compose_dual_tower_state, file_sha256,
    load_parent_checkpoint)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size-checkpoint', required=True)
    parser.add_argument('--full-checkpoint', required=True)
    parser.add_argument('--out-checkpoint', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--expected-size-sha256')
    parser.add_argument('--expected-full-sha256')
    return parser.parse_args()


def _check_hash(name, observed, expected):
    if expected is not None and observed.lower() != expected.lower():
        raise RuntimeError(
            '{} checkpoint SHA256 mismatch: expected {} got {}'.format(
                name, expected, observed))


def main():
    args = parse_args()
    size_path = os.path.abspath(args.size_checkpoint)
    full_path = os.path.abspath(args.full_checkpoint)
    output_path = os.path.abspath(args.out_checkpoint)
    report_path = os.path.abspath(args.out_json)
    size_hash = file_sha256(size_path)
    full_hash = file_sha256(full_path)
    _check_hash('size', size_hash, args.expected_size_sha256)
    _check_hash('full', full_hash, args.expected_full_sha256)
    dual, size_contract, full_contract = compose_dual_tower_state(
        load_parent_checkpoint(size_path),
        load_parent_checkpoint(full_path))
    contract = dict(
        protocol='source_only_dual_tower_component_recomposition_v2',
        architecture='dual_tower_size_pose_v2',
        source_train_frames=2781,
        source_val_frames=738,
        target_data_read=False,
        fixed_test_read=False,
        source_gate_passed=False,
        domain_routing=False,
        sequence_frame_routing=False,
        temporal_state=False,
        representation='five_delta_xywha',
        angle_range='le90', edge_swap=True, proj_xy=True,
        refine_center=True, refine_size=True, refine_angle=True,
        size_source='size_only_epoch10',
        pose_source='full_single_frame_epoch11',
        size_parent_sha256=size_hash,
        full_parent_sha256=full_hash,
        trained_after_recomposition=False)
    state = {
        'geometry_refiner.' + key: value for key, value in dual.items()}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dict(
        state_dict=state,
        meta={CONTRACT_KEY: contract}), output_path)
    output_hash = file_sha256(output_path)
    report = dict(
        protocol=contract['protocol'],
        evidence_boundary='source_checkpoints_only',
        target_data_read=False,
        fixed_test_read=False,
        inputs=dict(
            size=dict(path=size_path, sha256=size_hash,
                      contract=size_contract),
            full=dict(path=full_path, sha256=full_hash,
                      contract=full_contract)),
        output=dict(path=output_path, sha256=output_hash,
                    tensor_count=len(state), contract=contract),
        component_mapping=dict(
            size=['size.shared_fcs', 'size.delta_head[2:4]'],
            pose=['full.shared_fcs', 'full.delta_head[0,1,4]']),
        passed=True,
        eligible_for_fixed_test=False,
        decision='RUN_SOURCE_VAL_DUAL_TOWER_EQUIVALENCE_AUDIT')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
