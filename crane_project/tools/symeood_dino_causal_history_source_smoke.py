#!/usr/bin/env python3
"""One-step real-stack smoke for the source-only causal history refiner.

The smoke builds only source-train/source-val, performs one in-memory optimizer
step, and writes no checkpoint.  It also reports CUDA peak allocated/reserved
memory so host GPU mapping remains auditable when CUDA_VISIBLE_DEVICES is used.
"""

import argparse
import copy
import json
import os
import random
import traceback

import numpy as np


PROTOCOL = 'source_only_causal_history_refiner_smoke_v1'
V2_PROTOCOL = 'source_only_k1_anchored_causal_phase_refiner_smoke_v2'
V3_PROTOCOL = 'source_only_k1_retentive_causal_phase_refiner_smoke_v3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=(
            'crane_project/configs/'
            'crane_symeood_dino_causal_history_refiner_source_v1.py'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _container_tensor(sample, key):
    value = sample[key]
    return value.data if value.__class__.__name__ == 'DataContainer' else value


def _sample_with_history(dataset, limit=256):
    for index in range(min(len(dataset), int(limit))):
        sample = dataset[index]
        if sample is None:
            continue
        mask = _container_tensor(sample, 'causal_history_valid_mask')
        if bool(mask.any()):
            return sample, index
    raise RuntimeError('No source-train sample has valid causal history')


def _identity(sample):
    meta = _container_tensor(sample, 'img_metas')
    filename = os.path.splitext(os.path.basename(
        meta.get('ori_filename', meta.get('filename', ''))))[0]
    parts = filename.rsplit('_', 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return parts[0], int(parts[1])


def _adjacent_pair_with_history(dataset, limit=512):
    previous = None
    previous_index = None
    for index in range(min(len(dataset), int(limit))):
        sample = dataset[index]
        if sample is None:
            previous = None
            previous_index = None
            continue
        identity = _identity(sample)
        mask = _container_tensor(sample, 'causal_history_valid_mask')
        if (previous is not None and identity is not None
                and previous[0] == identity[0]
                and identity[1] == previous[1] + 1
                and bool(mask.any())):
            return previous[2], sample, previous_index, index
        previous = (identity[0], identity[1], sample) if identity else None
        previous_index = index
    raise RuntimeError('No adjacent source pair has valid causal history')


def _first_sample(dataset):
    for index in range(min(len(dataset), 256)):
        sample = dataset[index]
        if sample is not None:
            return sample, index
    raise RuntimeError('No source-val sample is available')


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
        raise RuntimeError('CUDA is required for the causal-history smoke')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = Config.fromfile(args.config)
    imports = cfg.get('custom_imports')
    if imports:
        import_modules_from_strings(**imports)
    cfg = compat_cfg(cfg)
    checkpoint_contract = dict(
        cfg.checkpoint_config.meta.geometry_refiner_checkpoint_contract)
    architecture = checkpoint_contract.get('architecture')
    v2 = architecture == 'k1_anchored_causal_phase_refiner_v2'
    v3 = architecture == 'k1_retentive_causal_phase_refiner_v3'
    required_contract = dict(
        architecture=(
            ('k1_retentive_causal_phase_refiner_v3' if v3 else
             'k1_anchored_causal_phase_refiner_v2') if (v2 or v3) else
            'current_anchored_causal_history_refiner_v1'),
        frozen_baseline_variant='symeood_k1_epoch24',
        frozen_baseline_config='crane_project/configs/crane_symeood_k1.py',
        frozen_baseline_checkpoint=(
            'work_dirs/crane_symeood_k1/epoch_24.pth'),
        source_train_frames=2781,
        source_val_frames=738,
        target_data_read=False,
        fixed_test_read=False,
        source_gate_passed=False,
        domain_routing=False,
        sequence_frame_routing=False,
        temporal_state=False,
        history_identity_model_input=False,
        fixed_target_parameter_selection=False)
    if v2 or v3:
        required_contract.update(dict(
            detector_forward_during_training=True,
            frozen_symeood_detection_head_forward=True,
            frozen_symeood_detection_from_shared_features=True,
            current_k1_geometry_anchor=True,
            native_dino_anchor_fallback=True,
            native_dino_current_conditioning=True,
            same_forward_all_domains=True,
            bounded_current_residual=True,
            continuous_double_angle_phase=True,
            zero_phase_is_exact_identity=True,
            representation='six_delta_xywh_sin2a_cos2a_residual'))
    if v3:
        required_contract.update(dict(
            continuous_k1_retention=True,
            retention_loss_weight=0.25,
            source_adjacent_pair_supervision=True,
            adjacent_pair_identity_model_input=False,
            temporal_size_error_consistency=True,
            temporal_size_loss_weight=0.20,
            inference_sequence_input=False,
            single_gpu_adjacent_pair_training=True,
            train_samples_per_gpu=2,
            train_shuffle=False))
    contract_checks = {
        key: checkpoint_contract.get(key) == expected
        for key, expected in required_contract.items()}
    config_text = open(args.config, 'r', encoding='utf-8').read()
    contract_checks.update(dict(
        no_test_annotation=("ann_file='test/" not in config_text),
        no_test_audit=("expected_split='test'" not in config_text),
        single_source_forward=(
            cfg.model.geometry_refiner.type ==
            ('K1RetentiveCausalPhaseGeometryRefiner' if v3 else
             'K1AnchoredCausalPhaseGeometryRefiner' if v2 else
             'DinoConditionedCausalHistoryRefiner')),
        train_batch_contract=(
            cfg.data.train_dataloader.samples_per_gpu == (2 if v3 else 1)),
        train_shuffle_contract=(
            bool(cfg.data.train_dataloader.shuffle) is (False if v3 else True))))
    if not all(contract_checks.values()):
        return dict(
            protocol=V3_PROTOCOL if v3 else V2_PROTOCOL if v2 else PROTOCOL,
            evidence_boundary='source_train_and_source_val_only',
            target_data_read=False,
            fixed_test_read=False,
            checkpoint_written=False,
            contract_checks=contract_checks,
            passed=False,
            decision='STOP_CAUSAL_HISTORY_SOURCE_PREFLIGHT_FAILED')

    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)
    if len(train_dataset) != 2781 or len(val_dataset) != 738:
        raise RuntimeError('Source dataset frame-count contract failed')
    if v3:
        first_train_sample, train_sample, first_train_index, train_index = (
            _adjacent_pair_with_history(train_dataset))
        train_samples = [first_train_sample, train_sample]
    else:
        train_sample, train_index = _sample_with_history(train_dataset)
        first_train_index = None
        train_samples = [train_sample]
    val_sample, val_index = _first_sample(val_dataset)
    history_images = _container_tensor(
        train_sample, 'causal_history_images')
    history_proposals = _container_tensor(
        train_sample, 'causal_history_proposals')
    history_mask = _container_tensor(
        train_sample, 'causal_history_valid_mask')
    train_meta = _container_tensor(train_sample, 'img_metas')
    horizon = int(cfg.model.geometry_refiner.history_horizon)
    input_checks = dict(
        history_images_shape=(
            tuple(history_images.shape[:2]) == (horizon, 3)),
        history_proposals_shape=(
            tuple(history_proposals.shape) == (horizon, 5)),
        history_mask_shape=(tuple(history_mask.shape) == (horizon,)),
        at_least_one_valid_history=bool(history_mask.any()),
        explicit_no_flip_metadata=(
            train_meta.get('flip') is False
            and train_meta.get('flip_direction') is None))
    if not all(input_checks.values()):
        raise RuntimeError('Causal history source tensor contract failed')

    torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.gpu)
    raw_model = build_detector(copy.deepcopy(cfg.model))
    raw_model.init_weights()
    frozen_before = raw_model.frozen_parameter_hash()
    raw_model = raw_model.cuda(args.gpu)
    raw_model.train()
    optimizer = build_optimizer(raw_model, copy.deepcopy(cfg.optimizer))
    trainable = [parameter for parameter in
                 raw_model.geometry_refiner.parameters()
                 if parameter.requires_grad]
    parallel = MMDataParallel(raw_model, device_ids=[args.gpu])
    batch = collate(train_samples, samples_per_gpu=(2 if v3 else 1))
    optimizer.zero_grad()
    losses = parallel(return_loss=True, **batch)
    loss, log_vars = raw_model._parse_losses(losses)
    loss.backward()
    baseline_grad_none = all(
        parameter.grad is None for parameter in raw_model.baseline.parameters())
    gradients = [parameter.grad for parameter in trainable]
    trainable_gradients_finite = all(
        gradient is not None and torch.isfinite(gradient).all().item()
        for gradient in gradients)
    history_head_gradient_nonzero = bool(
        raw_model.geometry_refiner.history_delta_head.weight.grad is not None
        and torch.count_nonzero(
            raw_model.geometry_refiner.history_delta_head.weight.grad).item()
        > 0)
    phase_head_gradient_nonzero = bool(
        not (v2 or v3) or (
            raw_model.geometry_refiner.delta_head.weight.grad is not None
            and torch.count_nonzero(
                raw_model.geometry_refiner.delta_head.weight.grad).item()
            > 0))
    conditioning_head_gradients_finite = bool(
        not (v2 or v3) or any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all().item()
            for parameter in
            raw_model.geometry_refiner.conditioning_fusion.parameters()))
    optimizer.step()
    frozen_after_step = raw_model.frozen_parameter_hash()

    raw_model.eval()
    val_batch = collate([val_sample], samples_per_gpu=1)
    with torch.no_grad():
        output = parallel(return_loss=False, rescale=True, **val_batch)
    output_valid = (
        isinstance(output, list) and len(output) == 1
        and isinstance(output[0], list) and len(output[0]) == 1
        and np.isfinite(np.asarray(output[0][0])).all())
    frozen_after_val = raw_model.frozen_parameter_hash()
    allocated = int(torch.cuda.max_memory_allocated(args.gpu))
    reserved = int(torch.cuda.max_memory_reserved(args.gpu))
    checks = dict(
        finite_loss=bool(torch.isfinite(loss).item()),
        baseline_gradients_none=baseline_grad_none,
        trainable_gradients_finite=trainable_gradients_finite,
        history_head_gradient_nonzero=history_head_gradient_nonzero,
        phase_head_gradient_nonzero=phase_head_gradient_nonzero,
        conditioning_head_gradients_finite=conditioning_head_gradients_finite,
        adjacent_pair_objective_active=(
            not v3 or (
                float(log_vars.get('refiner_temporal_pair_count', 0.0)) == 1.0
                and 'refiner_temporal_size_objective' in log_vars)),
        retention_objective_active=(
            not v3 or
            'refiner_continuous_retention_objective' in log_vars),
        frozen_baseline_unchanged=(
            frozen_before == frozen_after_step == frozen_after_val),
        source_val_forward_valid=bool(output_valid))
    return dict(
        protocol=V3_PROTOCOL if v3 else V2_PROTOCOL if v2 else PROTOCOL,
        evidence_boundary='source_train_and_source_val_only',
        target_data_read=False,
        fixed_test_read=False,
        checkpoint_written=False,
        optimizer_steps_in_memory=1,
        seed=int(args.seed),
        train_frame_count=len(train_dataset),
        source_val_frame_count=len(val_dataset),
        selected_train_index=int(train_index),
        selected_previous_train_index=(
            None if first_train_index is None else int(first_train_index)),
        selected_val_index=int(val_index),
        contract_checks=contract_checks,
        input_checks=input_checks,
        gpu=dict(
            requested_visible_index=int(args.gpu),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            device_name=torch.cuda.get_device_name(args.gpu),
            peak_allocated_bytes=allocated,
            peak_reserved_bytes=reserved),
        loss=float(loss.detach().cpu()),
        log_vars={key: float(value) for key, value in log_vars.items()},
        checks=checks,
        passed=all(checks.values()),
        decision=(
            ('ALLOW_K1_RETENTIVE_CAUSAL_PHASE_SOURCE_TRAINING'
             if v3 else
             'ALLOW_K1_ANCHORED_CAUSAL_PHASE_SOURCE_TRAINING'
             if v2 else 'ALLOW_CAUSAL_HISTORY_SOURCE_TRAINING')
            if all(checks.values()) else
            ('STOP_K1_RETENTIVE_CAUSAL_PHASE_SOURCE_SMOKE_FAILED'
             if v3 else
             'STOP_K1_ANCHORED_CAUSAL_PHASE_SOURCE_SMOKE_FAILED'
             if v2 else 'STOP_CAUSAL_HISTORY_SOURCE_SMOKE_FAILED')))


def main():
    args = parse_args()
    requested_v2 = 'k1_anchored_causal_phase' in os.path.basename(args.config)
    requested_v3 = 'k1_retentive_causal_phase' in os.path.basename(args.config)
    try:
        report = _run(args)
    except Exception as error:
        report = dict(
            protocol=(V3_PROTOCOL if requested_v3 else
                      V2_PROTOCOL if requested_v2 else PROTOCOL),
            evidence_boundary='source_train_and_source_val_only',
            target_data_read=False,
            fixed_test_read=False,
            checkpoint_written=False,
            passed=False,
            decision=(
                'STOP_K1_RETENTIVE_CAUSAL_PHASE_SOURCE_SMOKE_ERROR'
                if requested_v3 else
                'STOP_K1_ANCHORED_CAUSAL_PHASE_SOURCE_SMOKE_ERROR'
                if requested_v2 else
                'STOP_CAUSAL_HISTORY_SOURCE_SMOKE_ERROR'),
            error_type=type(error).__name__,
            error=str(error),
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
