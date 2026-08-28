#!/usr/bin/env python3
"""Source-only numerical and resource audit for SymEOOD FPN reuse.

The tool rejects the fixed TEST split.  It compares the historical public
``simple_test(img)`` result with ``extract_feat`` followed by
``simple_test_from_features`` and can additionally run one complete unified
source frame to record SymEOOD/DINO/refiner forward counts.
"""

import argparse
import json
import os
import random
import statistics
import time

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab')
    parser.add_argument('--split', required=True,
                        choices=['train', 'train_sim', 'val'])
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--frame', required=True, type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--run-unified-runtime', action='store_true')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--out-json')
    return parser.parse_args()


def _array(result):
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise RuntimeError('Expected one-image output')
    classes = result[0]
    if not isinstance(classes, (list, tuple)) or len(classes) != 1:
        raise RuntimeError('Expected one-class output')
    values = np.asarray(classes[0], dtype=np.float32)
    return values.reshape((-1, 6)) if values.size else np.zeros(
        (0, 6), dtype=np.float32)


def _measure_once(call, device):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = call()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    mib = float(1024 ** 2)
    return output, dict(
        latency_ms=float(elapsed_ms),
        peak_allocated_mib=float(
            torch.cuda.max_memory_allocated(device)) / mib,
        peak_reserved_mib=float(
            torch.cuda.max_memory_reserved(device)) / mib,
        allocated_after_mib=float(torch.cuda.memory_allocated(device)) / mib,
        reserved_after_mib=float(torch.cuda.memory_reserved(device)) / mib)


def _benchmark(call, device, warmup, repeats):
    if warmup < 0 or repeats <= 0:
        raise ValueError('warmup must be >= 0 and repeats must be > 0')
    torch.cuda.empty_cache()
    with torch.no_grad():
        for _ in range(warmup):
            call()
        torch.cuda.synchronize(device)
        samples = []
        peak_allocated = []
        peak_reserved = []
        output = None
        for _ in range(repeats):
            output, resource = _measure_once(call, device)
            samples.append(resource['latency_ms'])
            peak_allocated.append(resource['peak_allocated_mib'])
            peak_reserved.append(resource['peak_reserved_mib'])
    ordered = sorted(samples)
    p90_index = max(0, int(np.ceil(0.9 * len(ordered))) - 1)
    return output, dict(
        warmup=int(warmup),
        repeats=int(repeats),
        latency_median_ms=float(statistics.median(samples)),
        latency_p90_ms=float(ordered[p90_index]),
        latency_samples_ms=[float(value) for value in samples],
        peak_allocated_max_mib=float(max(peak_allocated)),
        peak_reserved_max_mib=float(max(peak_reserved)))


def _finalize_passed(report):
    checks = list(report['checks'].values())
    if 'unified_runtime' in report:
        checks.extend(report['unified_runtime']['checks'].values())
    report['passed'] = all(checks)
    return report['passed']


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmrotate.models import build_detector
    from crane_project.tools import mcml_diag as diag

    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model)
    load_checkpoint(
        model, args.checkpoint, map_location='cpu', strict=False)
    device = torch.device('cuda:{}'.format(args.gpu))
    model = model.to(device).eval()
    baseline = getattr(model, 'baseline', model)
    if not hasattr(baseline, 'simple_test_from_features'):
        raise RuntimeError('Baseline has no simple_test_from_features')

    img_path, _ann_path = diag.find_files(
        args.data_root, args.split, args.sequence, args.frame)
    if img_path is None:
        raise RuntimeError('Source image was not found')
    transforms, img_scale, flip = diag.build_test_transforms(cfg)
    img, meta, _stats = diag.preprocess_image(
        img_path, transforms, img_scale, flip)
    if img is None:
        raise RuntimeError('Source image preprocessing failed')
    img = img.to(device)
    metas = [meta]

    counter = dict(value=0)

    def count_backbone(_module, _inputs):
        counter['value'] += 1

    hook = baseline.backbone.register_forward_pre_hook(count_backbone)
    try:
        with torch.no_grad():
            counter['value'] = 0
            legacy, legacy_resource = _benchmark(
                lambda: baseline.simple_test(img, metas, rescale=True),
                device, args.warmup, args.repeats)
            legacy_count = counter['value']

            counter['value'] = 0

            def feature_path():
                features = baseline.extract_feat(img)
                return baseline.simple_test_from_features(
                    features, metas, rescale=True)

            reused, reused_resource = _benchmark(
                feature_path, device, args.warmup, args.repeats)
            reused_count = counter['value']
            counter['value'] = 0
            repeated, repeated_resource = _benchmark(
                feature_path, device, 0, 1)
            repeated_count = counter['value']
    finally:
        hook.remove()

    legacy_array = _array(legacy)
    reused_array = _array(reused)
    repeated_array = _array(repeated)
    checks = dict(
        box_count_equal=(legacy_array.shape == reused_array.shape),
        scores_elementwise_equal=(
            legacy_array.shape == reused_array.shape
            and np.array_equal(legacy_array[:, 5], reused_array[:, 5])),
        obb_elementwise_equal=(
            legacy_array.shape == reused_array.shape
            and np.array_equal(legacy_array[:, :5], reused_array[:, :5])),
        repeated_output_deterministic=np.array_equal(
            reused_array, repeated_array),
        legacy_backbone_forward_count_is_one_per_call=(
            legacy_count == args.warmup + args.repeats),
        reused_backbone_forward_count_is_one_per_call=(
            reused_count == args.warmup + args.repeats),
        repeated_backbone_forward_count_is_one=(repeated_count == 1))
    report = dict(
        protocol='symeood_feature_reuse_source_equivalence_v1',
        evidence_boundary='source_only_no_fixed_test',
        input=dict(
            split=args.split, sequence=args.sequence, frame=args.frame,
            filename=os.path.abspath(img_path), seed=args.seed),
        checks=checks,
        passed=False,
        resources=dict(
            legacy_simple_test=legacy_resource,
            feature_reuse=reused_resource,
            feature_reuse_repeat=repeated_resource),
        forward_counts=dict(
            legacy_backbone=legacy_count,
            feature_reuse_backbone=reused_count,
            feature_reuse_repeat_backbone=repeated_count,
            dino=0,
            geometry_refiner=0))

    if args.run_unified_runtime:
        if model is baseline:
            raise RuntimeError(
                '--run-unified-runtime requires a unified detector config')
        before = dict(getattr(model, '_runtime_forward_counts', {}))
        with torch.no_grad():
            _runtime_output, runtime_resource = _benchmark(
                lambda: model.simple_test(img, metas, rescale=True), device,
                args.warmup, args.repeats)
        after = dict(getattr(model, '_runtime_forward_counts', {}))
        delta = {key: int(after.get(key, 0) - before.get(key, 0))
                 for key in set(before) | set(after)}
        runtime_call_count = args.warmup + args.repeats
        report['unified_runtime'] = dict(
            resources=runtime_resource,
            forward_count_delta=delta,
            checks=dict(
                symeood_backbone_fpn_once=(
                    delta.get('symeood_backbone_fpn') == runtime_call_count),
                dino_once=(delta.get('dino') == runtime_call_count),
                refiner_at_most_once=(
                    delta.get('geometry_refiner', 0)
                    in (0, runtime_call_count))))
        report['forward_counts'].update(
            dino=delta.get('dino', 0),
            geometry_refiner=delta.get('geometry_refiner', 0))

    _finalize_passed(report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out_json:
        out = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
