#!/usr/bin/env python3
"""Write an exact, framework-free audit of fixed-ratio replay scheduling."""

import argparse
import hashlib
import json
import os

from crane_project.utils.fixed_ratio_replay_schedule import (
    enumerate_replay_schedule, replay_schedule_contract)


PROTOCOL = 'symeood_dino_replay_schedule_audit_v2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer-steps-per-epoch', type=int, default=1391)
    parser.add_argument(
        '--original-batches-per-auxiliary-batch', type=int, default=14)
    parser.add_argument('--samples-per-batch', type=int, default=2)
    parser.add_argument('--training-epochs', type=int, default=10)
    parser.add_argument('--original-sample-count', type=int, default=2781)
    parser.add_argument('--auxiliary-sample-count', type=int, required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def audit(args):
    epochs = int(args.training_epochs)
    samples_per_batch = int(args.samples_per_batch)
    original_count = int(args.original_sample_count)
    auxiliary_count = int(args.auxiliary_sample_count)
    if epochs <= 0 or samples_per_batch <= 0:
        raise ValueError('training epochs and samples per batch must be positive')
    if original_count <= 0 or auxiliary_count <= 0:
        raise ValueError('child sample counts must be positive')
    routes = enumerate_replay_schedule(
        args.optimizer_steps_per_epoch,
        args.original_batches_per_auxiliary_batch)
    schedule = replay_schedule_contract(
        args.optimizer_steps_per_epoch,
        args.original_batches_per_auxiliary_batch)
    per_epoch = []
    original_seen, auxiliary_seen = set(), set()
    serialized = []
    for epoch in range(epochs):
        original_indices = []
        auxiliary_indices = []
        for auxiliary, lane_batch in routes:
            lane_steps = schedule[
                'scheduled_auxiliary_steps' if auxiliary else
                'scheduled_original_steps']
            lane_count = auxiliary_count if auxiliary else original_count
            offset = epoch * lane_steps * samples_per_batch
            indices = [
                (offset + lane_batch * samples_per_batch + within) % lane_count
                for within in range(samples_per_batch)]
            (auxiliary_indices if auxiliary else original_indices).extend(
                indices)
            serialized.extend(
                '{}:{}:{}'.format(
                    epoch, 'auxiliary' if auxiliary else 'original', value)
                for value in indices)
        original_seen.update(original_indices)
        auxiliary_seen.update(auxiliary_indices)
        per_epoch.append(dict(
            epoch=epoch,
            original_step_count=schedule['scheduled_original_steps'],
            auxiliary_step_count=schedule['scheduled_auxiliary_steps'],
            original_sample_offset=(
                epoch * schedule['scheduled_original_steps']
                * samples_per_batch),
            auxiliary_sample_offset=(
                epoch * schedule['scheduled_auxiliary_steps']
                * samples_per_batch),
            original_first_indices=original_indices[:4],
            auxiliary_first_indices=auxiliary_indices[:4]))
    checks = dict(
        optimizer_steps_fixed=(
            schedule['enumerated_total_steps']
            == int(args.optimizer_steps_per_epoch)),
        exact_1391_route=(
            int(args.optimizer_steps_per_epoch) != 1391
            or (schedule['scheduled_original_steps'] == 1299
                and schedule['scheduled_auxiliary_steps'] == 92)),
        per_epoch_route_counts_stable=all(
            row['original_step_count'] ==
            schedule['scheduled_original_steps']
            and row['auxiliary_step_count'] ==
            schedule['scheduled_auxiliary_steps']
            for row in per_epoch),
        original_full_coverage=len(original_seen) == original_count,
        auxiliary_full_coverage=len(auxiliary_seen) == auxiliary_count,
        deterministic_reenumeration=(
            routes == enumerate_replay_schedule(
                args.optimizer_steps_per_epoch,
                args.original_batches_per_auxiliary_batch)))
    passed = all(checks.values())
    return dict(
        protocol=PROTOCOL,
        replay_schedule=schedule,
        samples_per_batch=samples_per_batch,
        training_epochs=epochs,
        original_sample_count=original_count,
        auxiliary_sample_count=auxiliary_count,
        per_epoch=per_epoch,
        actual_sample_schedule_sha256=hashlib.sha256(
            '\n'.join(serialized).encode('ascii')).hexdigest(),
        coverage=dict(
            original_unique_count=len(original_seen),
            auxiliary_unique_count=len(auxiliary_seen)),
        checks=checks,
        target_data_read=False,
        fixed_test_read=False,
        passed=passed,
        decision=(
            'ALLOW_REPLAY_SCHEDULE_V2' if passed else
            'STOP_REPLAY_SCHEDULE_V2_AUDIT_FAILED'))


def main():
    args = parse_args()
    report = audit(args)
    output = os.path.abspath(os.fspath(args.out_json))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
