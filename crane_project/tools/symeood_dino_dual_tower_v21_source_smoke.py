#!/usr/bin/env python3
"""One-step real-stack preflight for source-only Dual-Tower V2.1."""

import argparse
import copy
import json
import os
import re
import traceback

import numpy as np

from crane_project.tools.symeood_dino_geometry_refiner_source_smoke import (
    EXPECTED_COUNTS, _dataset_contract, _first_sample, _optimizer_ids)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=(
            'crane_project/configs/'
            'crane_symeood_dino_geometry_refiner_'
            'dual_tower_size_source_v21.py'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _identity(info):
    stem = os.path.splitext(os.path.basename(info['filename']))[0]
    match = re.match(r'^(real|sim)_(.+)_(\d+)$', stem)
    if match is None:
        return None
    return match.group(1), match.group(2), int(match.group(3))


def _adjacent_samples(dataset, entries):
    for (first_index, first_info), (second_index, second_info) in zip(
            entries, entries[1:]):
        first = _identity(first_info)
        second = _identity(second_info)
        if (first is None or second is None or first[:2] != second[:2]
                or second[2] != first[2] + 1):
            continue
        samples = [dataset[first_index], dataset[second_index]]
        if all(item is not None for item in samples):
            return samples, [first, second]
    raise RuntimeError('No adjacent source-train pair is available')


def _config_contract(cfg):
    refiner = cfg.model.geometry_refiner
    checkpoint = cfg.checkpoint_config.meta[
        'geometry_refiner_checkpoint_contract']
    checks = dict(
        dual_tower_type=(
            refiner.type == 'DinoConditionedDualTowerGeometryRefiner'),
        only_size_tower_trainable=(
            refiner.train_size_tower is True and
            refiner.train_pose_tower is False and
            refiner.train_roi_extractor is False),
        decoded_geometry_enabled=(
            refiner.decoded_geometry_loss_weight > 0.0),
        temporal_error_consistency_enabled=(
            refiner.temporal_size_loss_weight > 0.0),
        sequential_batch_pairing=(
            cfg.data.train_dataloader.shuffle is False and
            cfg.data.train_dataloader.samples_per_gpu == 2),
        initialization_hash_locked=(
            len(cfg.model.geometry_refiner_checkpoint_sha256) == 64),
        source_only=(
            checkpoint.target_data_read is False and
            checkpoint.fixed_test_read is False and
            checkpoint.source_gate_passed is False),
        no_routing_or_inference_state=(
            checkpoint.domain_routing is False and
            checkpoint.sequence_frame_routing is False and
            checkpoint.temporal_state is False and
            checkpoint.inference_sequence_input is False),
        adjacent_metadata_is_supervision_only=(
            checkpoint.source_adjacent_pair_supervision is True),
        manual_source_gate=(
            'save_best' not in cfg.evaluation and
            'rule' not in cfg.evaluation))
    return dict(checks=checks, passed=all(checks.values()))


def _run(args):
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    from mmcv.runner import build_optimizer
    from mmcv.utils import import_modules_from_strings
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from mmrotate.utils.compat_config import compat_cfg

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the V2.1 source smoke')
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    cfg = Config.fromfile(args.config)
    imports = cfg.get('custom_imports')
    if imports:
        import_modules_from_strings(**imports)
    cfg = compat_cfg(cfg)
    config_contract = _config_contract(cfg)
    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)
    train_report, train_entries = _dataset_contract(
        train_dataset, cfg.source_train_audit,
        EXPECTED_COUNTS['train'], EXPECTED_COUNTS['train_real'],
        EXPECTED_COUNTS['train_sim'])
    val_report, val_entries = _dataset_contract(
        val_dataset, cfg.source_val_audit,
        EXPECTED_COUNTS['val'], EXPECTED_COUNTS['val_real'],
        EXPECTED_COUNTS['val_sim'])
    if not all(item['passed'] for item in (
            config_contract, train_report, val_report)):
        return dict(
            protocol='dual_tower_v21_source_real_stack_smoke_v1',
            evidence_boundary='source_train_and_source_val_only',
            target_data_read=False, fixed_test_read=False,
            checkpoint_written=False, optimizer_steps_in_memory=0,
            config_contract=config_contract,
            datasets=dict(train=train_report, val=val_report),
            passed=False, decision='STOP_DUAL_TOWER_V21_PREFLIGHT_FAILED')

    raw_model = build_detector(copy.deepcopy(cfg.model))
    raw_model.init_weights()
    raw_model = raw_model.cuda(args.gpu)
    raw_model.train()
    optimizer = build_optimizer(raw_model, copy.deepcopy(cfg.optimizer))
    trainable = [parameter for parameter in
                 raw_model.geometry_refiner.parameters()
                 if parameter.requires_grad]
    optimizer_ids = _optimizer_ids(optimizer)
    trainable_ids = [id(parameter) for parameter in trainable]
    before_baseline = raw_model.frozen_parameter_hash()
    before_frozen_refiner = raw_model.frozen_refiner_hash()
    samples, identities = _adjacent_samples(train_dataset, train_entries)
    batch = collate(samples, samples_per_gpu=2)
    parallel = MMDataParallel(raw_model, device_ids=[args.gpu])
    optimizer.zero_grad()
    losses = parallel(return_loss=True, **batch)
    loss, log_vars = raw_model._parse_losses(losses)
    loss.backward()

    size_parameters = list(raw_model.geometry_refiner.size_fcs.parameters())
    size_parameters += list(raw_model.geometry_refiner.size_head.parameters())
    pose_parameters = list(raw_model.geometry_refiner.pose_fcs.parameters())
    pose_parameters += list(raw_model.geometry_refiner.pose_head.parameters())
    size_gradients = [parameter.grad for parameter in size_parameters]
    checks = dict(
        finite_loss=bool(torch.isfinite(loss).item()),
        optimizer_exactly_trainable_refiner=(
            len(optimizer_ids) == len(set(optimizer_ids)) and
            set(optimizer_ids) == set(trainable_ids)),
        size_gradients_finite_and_nonzero=all(
            gradient is not None and torch.isfinite(gradient).all().item()
            and torch.count_nonzero(gradient).item() > 0
            for gradient in size_gradients),
        pose_requires_grad_false=all(
            not parameter.requires_grad for parameter in pose_parameters),
        pose_gradients_none=all(
            parameter.grad is None for parameter in pose_parameters),
        decoded_geometry_loss_reported=(
            'refiner_decoded_geometry_objective' in log_vars),
        temporal_size_loss_reported=(
            'refiner_temporal_size_objective' in log_vars),
        one_adjacent_pair_reported=(
            int(round(log_vars.get('refiner_temporal_pair_count', -1))) == 1),
        source_pair_is_consecutive=(
            identities[0][:2] == identities[1][:2] and
            identities[1][2] == identities[0][2] + 1))
    optimizer.step()
    after_baseline = raw_model.frozen_parameter_hash()
    after_frozen_refiner = raw_model.frozen_refiner_hash()
    checks.update(dict(
        baseline_hash_unchanged=(before_baseline == after_baseline),
        frozen_pose_hash_unchanged=(
            before_frozen_refiner == after_frozen_refiner)))

    raw_model.eval()
    validation_batch = collate(
        [_first_sample(val_dataset, val_entries)], samples_per_gpu=1)
    with torch.no_grad():
        output = parallel(
            return_loss=False, rescale=True, **validation_batch)
    values = np.asarray(output[0][0])
    checks.update(dict(
        source_val_output_shape=(
            isinstance(output, list) and len(output) == 1 and
            isinstance(output[0], list) and len(output[0]) == 1),
        source_val_output_finite=bool(np.isfinite(values).all())))
    passed = all(checks.values())
    return dict(
        protocol='dual_tower_v21_source_real_stack_smoke_v1',
        evidence_boundary='source_train_and_source_val_only',
        target_data_read=False, fixed_test_read=False,
        checkpoint_written=False, optimizer_steps_in_memory=1,
        seed=args.seed, config=os.path.abspath(args.config),
        config_contract=config_contract,
        datasets=dict(train=train_report, val=val_report),
        adjacent_pair=dict(first=identities[0], second=identities[1]),
        loss=float(loss.detach().cpu()),
        log_vars={key: float(value) for key, value in log_vars.items()},
        checks=checks, passed=passed,
        decision=(
            'ALLOW_DUAL_TOWER_V21_SOURCE_TRAINING' if passed else
            'STOP_DUAL_TOWER_V21_SOURCE_SMOKE_FAILED'))


def main():
    args = parse_args()
    try:
        report = _run(args)
    except Exception as error:
        report = dict(
            protocol='dual_tower_v21_source_real_stack_smoke_v1',
            evidence_boundary='source_train_and_source_val_only',
            target_data_read=False, fixed_test_read=False,
            checkpoint_written=False, passed=False,
            decision='STOP_DUAL_TOWER_V21_SOURCE_SMOKE_ERROR',
            error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc())
    output = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
