#!/usr/bin/env python3
"""Real-stack, source-only smoke test for the geometry refiner.

This tool builds both locked source configs and their train/val datasets, uses
the real rotated resize/flip implementations, and performs one in-memory GPU
optimizer step for Size-only and Full.  It never builds ``data.test``, never
reads a target audit, and never writes a checkpoint.
"""

import argparse
import copy
import json
import math
import os
import random
import traceback

import numpy as np


EXPECTED_COUNTS = dict(
    train=2781, val=738,
    train_real=2033, train_sim=748,
    val_real=226, val_sim=512)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--full-config',
        default=('crane_project/configs/'
                 'crane_symeood_dino_geometry_refiner_full_source_v1.py'))
    parser.add_argument(
        '--size-config',
        default=('crane_project/configs/'
                 'crane_symeood_dino_geometry_refiner_size_source_v1.py'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _records(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get('records'), list):
        return payload['records']
    raise RuntimeError('Audit JSON has no records list')


def _frame_key(filename):
    return os.path.splitext(os.path.basename(os.fspath(filename)))[0]


def _load_audit(path):
    from mmrotate.datasets.pipelines.loading import (
        dino_invocation_encoding, dino_record_was_computed)

    absolute = os.path.abspath(os.fspath(path))
    with open(absolute, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    records = _records(payload)
    keys = [_frame_key(item['filename']) for item in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError('Audit contains duplicate frame keys: ' + absolute)
    if any(not dino_record_was_computed(item) for item in records):
        raise RuntimeError('Audit contains a frame where DINO was not invoked')
    if any('dino_native_box' not in item for item in records):
        raise RuntimeError('Audit contains no dino_native_box key')
    encoding_counts = {}
    for item in records:
        encoding = dino_invocation_encoding(item)
        encoding_counts[encoding] = encoding_counts.get(encoding, 0) + 1
    return absolute, set(keys), encoding_counts


def _leaf_entries(dataset, base_index=0):
    children = getattr(dataset, 'datasets', None)
    if children is not None:
        offset = base_index
        entries = []
        for child in children:
            entries.extend(_leaf_entries(child, offset))
            offset += len(child)
        return entries
    return [(base_index + index, info) for index, info in
            enumerate(dataset.data_infos)]


def _dataset_contract(dataset, audit_path, expected_count,
                      expected_real, expected_sim):
    absolute_audit, audit_keys, invocation_encodings = _load_audit(audit_path)
    entries = _leaf_entries(dataset)
    dataset_keys = [_frame_key(info['filename']) for _, info in entries]
    domains = [info.get('domain', _frame_key(info['filename']).split('_')[0])
               for _, info in entries]
    checks = dict(
        dataset_length=(len(dataset) == expected_count),
        audit_length=(len(audit_keys) == expected_count),
        dataset_keys_unique=(len(dataset_keys) == len(set(dataset_keys))),
        exact_audit_coverage=(set(dataset_keys) == audit_keys),
        real_count=(domains.count('real') == expected_real),
        sim_count=(domains.count('sim') == expected_sim))
    return dict(
        audit_json=absolute_audit,
        frame_count=len(dataset),
        domain_counts=dict(
            real=domains.count('real'), sim=domains.count('sim')),
        dino_invocation_encoding_counts=invocation_encodings,
        checks=checks,
        passed=all(checks.values())), entries


def _component_contract(cfg, expected):
    refiner = cfg.model.geometry_refiner
    checkpoint = cfg.checkpoint_config.meta[
        'geometry_refiner_checkpoint_contract']
    component = dict(
        refine_center=bool(refiner.refine_center),
        refine_size=bool(refiner.refine_size),
        refine_angle=bool(refiner.refine_angle))
    checks = dict(
        model_components=(component == expected),
        checkpoint_components=(
            {key: bool(checkpoint[key]) for key in expected} == expected),
        source_train_frames=(checkpoint.source_train_frames == 2781),
        source_val_frames=(checkpoint.source_val_frames == 738),
        target_data_read_is_false=(checkpoint.target_data_read is False),
        source_gate_not_predeclared=(checkpoint.source_gate_passed is False),
        no_domain_routing=(checkpoint.domain_routing is False),
        no_sequence_frame_routing=(
            checkpoint.sequence_frame_routing is False),
        no_temporal_state=(checkpoint.temporal_state is False),
        dedicated_optimizer=(
            cfg.optimizer.constructor ==
            'GeometryRefinerOptimizerConstructor'),
        optimizer_keys_exact=(
            set(cfg.optimizer.keys()) ==
            {'type', 'constructor', 'lr', 'weight_decay'}),
        optimizer_has_no_inherited_momentum=(
            'momentum' not in cfg.optimizer),
        no_legacy_top_level_loader_args=(
            'samples_per_gpu' not in cfg.data and
            'workers_per_gpu' not in cfg.data),
        train_batch_size_is_two=(
            cfg.data.train_dataloader.samples_per_gpu == 2),
        validation_batch_size_is_one=(
            cfg.data.val_dataloader.samples_per_gpu == 1),
        manual_source_gate_no_auto_best=(
            'save_best' not in cfg.evaluation and
            'rule' not in cfg.evaluation),
        evaluation_thresholds_preserved=(
            cfg.evaluation.thresh_sim == 10.0 and
            cfg.evaluation.thresh_real == 25.0 and
            cfg.evaluation.weight_sim == 0.7 and
            cfg.evaluation.weight_real == 0.3))
    return dict(components=component, checks=checks,
                passed=all(checks.values()))


def _pipeline_coordinate_contract():
    from mmrotate.datasets.pipelines.transforms import RRandomFlip, RResize

    original = np.array(
        [[61.0, 47.0, 34.0, 13.0, 0.31]], dtype=np.float32)
    resize = RResize(img_scale=(160, 120))
    resized = dict(
        gt_bboxes=original.copy(), dino_proposals=original.copy(),
        bbox_fields=['gt_bboxes', 'dino_proposals'],
        scale_factor=np.array([1.7, 0.65, 1.7, 0.65], dtype=np.float32))
    resize._resize_bboxes(resized)
    size_scale = math.sqrt(1.7 * 0.65)
    expected_resized = np.array(
        [[61.0 * 1.7, 47.0 * 0.65,
          34.0 * size_scale, 13.0 * size_scale, 0.31]],
        dtype=np.float32)
    checks = dict(
        nonuniform_scale_exercised=(
            float(resized['scale_factor'][0]) !=
            float(resized['scale_factor'][1])),
        resized_fields_equal=np.array_equal(
            resized['gt_bboxes'], resized['dino_proposals']),
        resized_matches_manual_expected=np.allclose(
            resized['gt_bboxes'], expected_resized,
            rtol=1e-6, atol=1e-6),
        resize_changed_from_input=(
            not np.array_equal(resized['gt_bboxes'], original)))
    directions = [None, 'horizontal', 'vertical', 'diagonal']
    for direction in directions:
        name = 'no_flip' if direction is None else direction
        flip = RRandomFlip(
            flip_ratio=0.0 if direction is None else 1.0,
            direction='horizontal' if direction is None else direction,
            version='le90')
        transformed = dict(
            img=np.zeros((120, 160, 3), dtype=np.uint8),
            img_shape=(120, 160, 3),
            img_fields=['img'],
            gt_bboxes=resized['gt_bboxes'].copy(),
            dino_proposals=resized['dino_proposals'].copy(),
            bbox_fields=['gt_bboxes', 'dino_proposals'],
            flip=(direction is not None),
            flip_direction=direction)
        transformed = flip(transformed)
        checks[name + '_fields_equal'] = np.array_equal(
            transformed['gt_bboxes'], transformed['dino_proposals'])
        expected = expected_resized.copy()
        if direction in ('horizontal', 'diagonal'):
            expected[:, 0] = 160.0 - expected[:, 0] - 1.0
        if direction in ('vertical', 'diagonal'):
            expected[:, 1] = 120.0 - expected[:, 1] - 1.0
        if direction in ('horizontal', 'vertical'):
            angle = math.pi - expected[:, 4]
            expected[:, 4] = (
                (angle + math.pi / 2.0) % math.pi - math.pi / 2.0)
        checks[name + '_matches_manual_expected'] = np.allclose(
            transformed['gt_bboxes'], expected, rtol=1e-6, atol=1e-6)
        if direction is not None:
            checks[name + '_changed_from_resized'] = not np.array_equal(
                transformed['gt_bboxes'], resized['gt_bboxes'])
    return dict(
        implementation='real_RResize_and_RRandomFlip_bbox_operations',
        checks=checks, passed=all(checks.values()))


def _sample_pair(dataset, entries):
    chosen = {}
    for index, info in entries:
        domain = info.get('domain', _frame_key(info['filename']).split('_')[0])
        if domain in ('real', 'sim') and domain not in chosen:
            item = dataset[index]
            if item is not None:
                chosen[domain] = item
        if len(chosen) == 2:
            break
    if set(chosen) != {'real', 'sim'}:
        raise RuntimeError(
            'Could not obtain one real and one sim train sample')
    return [chosen['real'], chosen['sim']]


def _first_sample(dataset, entries):
    for index, _info in entries:
        item = dataset[index]
        if item is not None:
            return item
    raise RuntimeError('Could not obtain a source validation sample')


def _optimizer_ids(optimizer):
    return [id(parameter) for group in optimizer.param_groups
            for parameter in group['params']]


def _gpu_step(cfg, dataset, entries, val_dataset, val_entries, gpu,
              expected_active):
    import torch
    from mmcv.parallel import MMDataParallel, collate
    from mmcv.runner import build_optimizer
    from mmrotate.models import build_detector

    torch.cuda.set_device(gpu)
    raw_model = build_detector(copy.deepcopy(cfg.model))
    constructor_hash = raw_model.frozen_parameter_hash()
    raw_model.init_weights()
    public_init_hash = raw_model.frozen_parameter_hash()
    raw_model = raw_model.cuda(gpu)
    raw_model.train()
    optimizer = build_optimizer(raw_model, copy.deepcopy(cfg.optimizer))
    refiner_parameters = [parameter for parameter in
                          raw_model.geometry_refiner.parameters()
                          if parameter.requires_grad]
    optimizer_ids = _optimizer_ids(optimizer)
    refiner_ids = [id(parameter) for parameter in refiner_parameters]
    before_hash = raw_model.frozen_parameter_hash()
    before_head = raw_model.geometry_refiner.delta_head.weight.detach().clone()
    zero_initialized_before_step = bool(
        torch.count_nonzero(before_head).item() == 0)
    samples = _sample_pair(dataset, entries)
    batch = collate(samples, samples_per_gpu=len(samples))
    data_container_keys = [key for key, value in batch.items()
                           if value.__class__.__name__ == 'DataContainer']
    parallel = MMDataParallel(raw_model, device_ids=[gpu])
    optimizer.zero_grad()
    losses = parallel(return_loss=True, **batch)
    loss, log_vars = raw_model._parse_losses(losses)
    loss.backward()

    head = raw_model.geometry_refiner.delta_head
    row_norms = []
    for index in range(5):
        value = head.weight.grad[index].abs().sum()
        if head.bias.grad is not None:
            value = value + head.bias.grad[index].abs()
        row_norms.append(float(value.detach().cpu()))
    baseline_grad_none = all(parameter.grad is None for parameter in
                             raw_model.baseline.parameters())
    refiner_grads = [parameter.grad for parameter in refiner_parameters]
    refiner_grad_finite = all(
        gradient is not None and torch.isfinite(gradient).all().item()
        for gradient in refiner_grads)
    active_rows_nonzero = all(
        row_norms[index] > 0.0 for index, active in
        enumerate(expected_active) if active)
    inactive_rows_zero = all(
        row_norms[index] == 0.0 for index, active in
        enumerate(expected_active) if not active)
    optimizer.step()
    after_step_hash = raw_model.frozen_parameter_hash()
    after_head = raw_model.geometry_refiner.delta_head.weight.detach()

    raw_model.eval()
    validation_batch = collate(
        [_first_sample(val_dataset, val_entries)], samples_per_gpu=1)
    with torch.no_grad():
        validation_output = parallel(
            return_loss=False, rescale=True, **validation_batch)
    validation_output_valid = (
        isinstance(validation_output, list)
        and len(validation_output) == 1
        and isinstance(validation_output[0], list)
        and len(validation_output[0]) == 1)
    validation_values = (
        np.asarray(validation_output[0][0])
        if validation_output_valid else np.asarray([np.nan]))
    after_validation_hash = raw_model.frozen_parameter_hash()
    checks = dict(
        real_datacontainer_batch=(
            {'img', 'img_metas', 'gt_bboxes', 'gt_labels',
             'dino_proposals'}.issubset(data_container_keys)),
        finite_loss=bool(torch.isfinite(loss).item()),
        optimizer_exactly_refiner=(
            len(optimizer_ids) == len(set(optimizer_ids)) and
            set(optimizer_ids) == set(refiner_ids)),
        baseline_gradients_none=baseline_grad_none,
        refiner_gradients_finite=refiner_grad_finite,
        active_delta_rows_nonzero=active_rows_nonzero,
        inactive_delta_rows_zero=inactive_rows_zero,
        public_init_preserved_frozen_checkpoint=(
            constructor_hash == public_init_hash),
        zero_initialized_before_step=zero_initialized_before_step,
        frozen_parameter_and_buffer_hash_unchanged=(
            before_hash == after_step_hash == after_validation_hash),
        zero_initialized_head_changed=bool(
            torch.any(after_head != before_head).item()),
        source_val_forward_output_valid=validation_output_valid,
        source_val_forward_output_finite=bool(
            np.isfinite(validation_values).all()))
    report = dict(
        batch_domains=['real', 'sim'],
        data_container_keys=sorted(data_container_keys),
        loss=float(loss.detach().cpu()),
        log_vars={key: float(value) for key, value in log_vars.items()},
        delta_head_row_gradient_l1=row_norms,
        frozen_hash_at_construction=constructor_hash,
        frozen_hash_after_public_init=public_init_hash,
        frozen_hash_before=before_hash,
        frozen_hash_after_step=after_step_hash,
        frozen_hash_after_validation=after_validation_hash,
        checks=checks,
        passed=all(checks.values()))
    del parallel, raw_model, optimizer, loss, losses, batch
    del validation_batch, validation_output
    torch.cuda.empty_cache()
    return report


def _run(args):
    import torch
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmrotate.datasets import build_dataset
    from mmrotate.utils.compat_config import compat_cfg

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the real-stack smoke test')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    full_cfg = Config.fromfile(args.full_config)
    size_cfg = Config.fromfile(args.size_config)
    for cfg in (full_cfg, size_cfg):
        imports = cfg.get('custom_imports')
        if imports:
            import_modules_from_strings(**imports)
    # Exercise the same legacy-loader compatibility pass as tools/train.py.
    full_cfg = compat_cfg(full_cfg)
    size_cfg = compat_cfg(size_cfg)

    full_contract = _component_contract(
        full_cfg, dict(refine_center=True, refine_size=True,
                       refine_angle=True))
    size_contract = _component_contract(
        size_cfg, dict(refine_center=False, refine_size=True,
                       refine_angle=False))

    full_train = build_dataset(full_cfg.data.train)
    full_val = build_dataset(full_cfg.data.val)
    size_train = build_dataset(size_cfg.data.train)
    size_val = build_dataset(size_cfg.data.val)
    full_train_report, full_entries = _dataset_contract(
        full_train, full_cfg.source_train_audit,
        EXPECTED_COUNTS['train'], EXPECTED_COUNTS['train_real'],
        EXPECTED_COUNTS['train_sim'])
    full_val_report, full_val_entries = _dataset_contract(
        full_val, full_cfg.source_val_audit,
        EXPECTED_COUNTS['val'], EXPECTED_COUNTS['val_real'],
        EXPECTED_COUNTS['val_sim'])
    size_train_report, size_entries = _dataset_contract(
        size_train, size_cfg.source_train_audit,
        EXPECTED_COUNTS['train'], EXPECTED_COUNTS['train_real'],
        EXPECTED_COUNTS['train_sim'])
    size_val_report, size_val_entries = _dataset_contract(
        size_val, size_cfg.source_val_audit,
        EXPECTED_COUNTS['val'], EXPECTED_COUNTS['val_real'],
        EXPECTED_COUNTS['val_sim'])
    pipeline = _pipeline_coordinate_contract()

    preflight_sections = [
        full_contract, size_contract, full_train_report, full_val_report,
        size_train_report, size_val_report, pipeline]
    if not all(section['passed'] for section in preflight_sections):
        return dict(
            protocol='source_only_geometry_refiner_real_stack_smoke_v1',
            evidence_boundary='source_train_and_source_val_only',
            target_data_read=False,
            checkpoint_written=False,
            optimizer_steps_in_memory=0,
            seed=args.seed,
            configs=dict(full=os.path.abspath(args.full_config),
                         size=os.path.abspath(args.size_config)),
            config_contracts=dict(full=full_contract, size=size_contract),
            datasets=dict(full_train=full_train_report,
                          full_val=full_val_report,
                          size_train=size_train_report,
                          size_val=size_val_report),
            pipeline_coordinate_contract=pipeline,
            gpu_steps=dict(skipped='source_preflight_failed'),
            passed=False,
            decision='STOP_SOURCE_PREFLIGHT_FAILED')

    # Independent builds preserve the identical zero-init starting contract.
    size_gpu = _gpu_step(
        size_cfg, size_train, size_entries,
        size_val, size_val_entries, args.gpu,
        [False, False, True, True, False])
    full_gpu = _gpu_step(
        full_cfg, full_train, full_entries,
        full_val, full_val_entries, args.gpu,
        [True, True, True, True, True])
    sections = [full_contract, size_contract, full_train_report,
                full_val_report, size_train_report, size_val_report,
                pipeline, size_gpu, full_gpu]
    return dict(
        protocol='source_only_geometry_refiner_real_stack_smoke_v1',
        evidence_boundary='source_train_and_source_val_only',
        target_data_read=False,
        checkpoint_written=False,
        optimizer_steps_in_memory=2,
        seed=args.seed,
        configs=dict(full=os.path.abspath(args.full_config),
                     size=os.path.abspath(args.size_config)),
        config_contracts=dict(full=full_contract, size=size_contract),
        datasets=dict(full_train=full_train_report,
                      full_val=full_val_report,
                      size_train=size_train_report,
                      size_val=size_val_report),
        pipeline_coordinate_contract=pipeline,
        gpu_steps=dict(size_only=size_gpu, full=full_gpu),
        passed=all(section['passed'] for section in sections),
        decision=('ALLOW_SIZE_ONLY_SOURCE_TRAINING'
                  if all(section['passed'] for section in sections)
                  else 'STOP_SOURCE_SMOKE_FAILED'))


def main():
    args = parse_args()
    try:
        report = _run(args)
    except Exception as error:
        report = dict(
            protocol='source_only_geometry_refiner_real_stack_smoke_v1',
            evidence_boundary='source_train_and_source_val_only',
            target_data_read=False,
            checkpoint_written=False,
            passed=False,
            decision='STOP_SOURCE_SMOKE_ERROR',
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc())
    output_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
