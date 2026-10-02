#!/usr/bin/env python3
"""Frozen B + isolated structure/quality branch, bounded TRAIN integration check.

Twelve predeclared TRAIN images, plus two half-scale/horizontal-flip views.
Each view starts the branch at the SAME seed and performs ONE discarded SGD
step. No fitted checkpoint, detector updates, VAL/TEST inference or threshold
selection. --check-only validates source/data bindings without reading weights.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import check_port_reliability_readiness_v1 as ready
from crane_project.utils.port_structure_reliability_v1 import (
    SETTINGS, StructureComponentReliability, assert_detector_frozen,
    balanced_response_loss, canonical_boxes, component_probes_original,
    freeze_detector, map_boxes, map_points, quality_loss,
    quality_targets_original, response_targets)

MANIFEST = ROOT/'crane_project/tools/port_structure_reliability_v1_sources.json'
CHECKPOINT = ROOT/'work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth'
STRUCTURE_SHA = '86e3126b9dda0301c22b83a8d196e2e6ef86267b855ee322d86de1b89c691e8f'
SAMPLES = (
    'real_seq01_00000', 'real_seq01_00170', 'real_seq05_00000', 'real_seq05_00281',
    'real_seq06_00000', 'real_seq06_00234', 'real_seq12_00000', 'real_seq12_00070',
    'real_seq13_00000', 'real_seq13_00152', 'sim_seq08_00000', 'sim_seq08_00374')
VIEWS = [(name, 1., False) for name in SAMPLES] + [
    ('real_seq05_00281', .5, True), ('sim_seq08_00374', .5, True)]


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def checked_sources():
    fixed = json.loads(MANIFEST.read_text())
    if fixed.get('protocol') != SETTINGS['version'] or fixed.get('settings') != SETTINGS:
        raise ValueError('Predeclared settings differ from the reviewed manifest')
    current = {p: ready.sha(ROOT/p) for p in fixed['sources']}
    if current != fixed['sources']:
        bad = [p for p in current if current[p] != fixed['sources'][p]]
        raise ValueError('Reviewed source SHA differs: '+', '.join(bad))
    if fixed.get('views') != [dict(image=n, scale=s, horizontal_flip=f) for n, s, f in VIEWS]:
        raise ValueError('Predeclared TRAIN view list differs')
    return dict(manifest_sha256=ready.sha(MANIFEST), sources=current,
                frozen_f_s_sources_preserved=fixed['f_s_source_count'])


def prerequisites(structure_report):
    """Reuse reviewed readiness; byte-bind current TRAIN sets, no full reaudit."""
    if ready.sha(structure_report) != STRUCTURE_SHA:
        raise ValueError('Requires the reviewed complete 1810-axis server report')
    report = json.loads(structure_report.read_text())
    if (report['train_frames'] != 2558 or report['native_axis_frames'] != 1810
            or report['native_conversion_consistent'] != 1810
            or report['obb_derived_weak_axis_frames'] != 748
            or report['legacy_snapshot_available'] is not True):
        raise ValueError('Unexpected structure readiness counts')
    current = {}
    for split in ('train', 'train_sim'):
        paths = ready.files(ready.DATA/split/'annfiles', '.txt')
        expected = {k: v for k, v in ready.TRAIN_COUNTS.items() if ready.TRAIN_SPLITS[k] == split}
        if Counter(p.stem.rsplit('_', 1)[0] for p in paths) != Counter(expected):
            raise ValueError('TRAIN role/count differs')
        current[split+'_annotation_sha256'] = ready.set_sha(paths)
    for sequence in ready.NATIVE_K:
        folder = ('axis_legacy_train_v1' if sequence in ready.LEGACY_TRAIN else 'axis_k2p1')
        current[sequence+'_axis_json_sha256'] = ready.set_sha(
            ready.files(ready.DATA/'provenance'/folder/sequence/'axis_json', '.json'))
    current['legacy_snapshot_manifest_sha256'] = ready.sha(
        ready.DATA/'provenance/axis_legacy_train_v1/manifest.json')
    if current != report['source_identities']:
        raise ValueError('Current TRAIN bytes differ from the complete reviewed readiness report')
    indexed = {r['image']: r for r in report['review_samples']}
    if set(indexed) != set(SAMPLES):
        raise ValueError('Readiness fixed image list differs')
    samples = []
    for name in SAMPLES:
        old = indexed[name]; split = ready.TRAIN_SPLITS[name.rsplit('_', 1)[0]]
        image = ready.DATA/split/'images'/(name+'.jpg')
        ann = ready.DATA/split/'annfiles'/(name+'.txt')
        with Image.open(image) as im:
            size = im.size
        if (ready.sha(image) != old['image_sha256'] or ready.sha(ann) != old['annotation_sha256']
                or list(size) != old['image_size']):
            raise ValueError('Fixed TRAIN image/annotation identity differs: '+name)
        samples.append(dict(old, image_path=str(image.resolve())))
    return dict(reviewed_structure_report=str(structure_report.resolve()),
                reviewed_structure_sha256=STRUCTURE_SHA, source_identities=current,
                finite_train_sample_count=len(samples), samples=samples)


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str((str(value.dtype), list(value.shape))).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def state_digest(model):
    parameters = {n: tensor_sha(t) for n, t in model.named_parameters()}
    buffers = {n: tensor_sha(t) for n, t in model.named_buffers()}
    def merged(values):
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    return dict(parameters=merged(parameters), buffers=merged(buffers),
                parameter_count=sum(t.numel() for t in model.parameters()),
                buffer_count=len(buffers))


def fixed_specs(cfg, scale, flip):
    specs = deepcopy(cfg.data.train)
    for entry in specs:
        for step in entry.pipeline:
            if step['type'] == 'PortIsotropicShrink':
                step['prob'] = 0. if scale == 1. else 1.
                step['scale_range'] = (scale, scale)
            if step['type'] == 'RRandomFlip':
                step['flip_ratio'] = [1., 0., 0.] if flip else [0., 0., 0.]
    return specs


def check_cfg():
    from mmcv import Config
    cfg = Config.fromfile(str(ready.B_CONFIG))
    if ready.sha(ready.B_CONFIG) != ready.FROZEN_B['config_sha256']:
        raise ValueError('Retained B config identity differs')
    if (cfg.model.type != 'SymEOOD' or cfg.model.neck.out_channels != SETTINGS['in_channels']
            or cfg.model.bbox_head.anchor_generator.strides[0] != SETTINGS['stride']
            or cfg.model.test_cfg.max_per_img != 1 or cfg.model.test_cfg.score_thr != .05
            or cfg.model.bbox_head.get('center_size_compensation') is not None
            or cfg.model.bbox_head.get('shape_compensation') is not None):
        raise ValueError('B inference/feature interface changed')
    return cfg


def flatten_prediction(result):
    if len(result) != 1 or len(result[0]) != 1:
        raise ValueError('Expected one image / one grab class')
    a = np.asarray(result[0][0]).copy()
    if a.ndim != 2 or a.shape[1] != 6 or len(a) > 1 or not np.isfinite(a).all():
        raise ValueError('Expected zero or one finite original B output')
    return a


def seed_all():
    import torch
    random.seed(SETTINGS['seed']); np.random.seed(SETTINGS['seed'])
    torch.manual_seed(SETTINGS['seed']); torch.cuda.manual_seed_all(SETTINGS['seed'])


def measured(fn, gpu):
    import torch
    torch.cuda.synchronize(gpu)
    start_alloc = torch.cuda.memory_allocated(gpu)
    start_reserved = torch.cuda.memory_reserved(gpu)
    torch.cuda.reset_peak_memory_stats(gpu)
    begin = time.perf_counter()
    value = fn()
    torch.cuda.synchronize(gpu)
    return value, dict(elapsed_ms=(time.perf_counter()-begin)*1000,
        allocated_at_start_mib=start_alloc/2**20, reserved_at_start_mib=start_reserved/2**20,
        peak_allocated_mib=torch.cuda.max_memory_allocated(gpu)/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved(gpu)/2**20,
        peak_above_stage_start_mib=(torch.cuda.max_memory_allocated(gpu)-start_alloc)/2**20)


def norms(model):
    groups = {'stem': model.stem, 'response': model.response, 'quality': model.quality}
    result = {}
    for name, module in groups.items():
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if (len(grads) != len(list(module.parameters()))
                or not all(bool(g.isfinite().all()) for g in grads)):
            raise ValueError('Missing/nonfinite branch gradients: '+name)
        result[name] = sum(float(g.double().square().sum()) for g in grads)**.5
        if result[name] <= 0:
            raise ValueError('No effective branch gradient: '+name)
    return result


def probe_view(detector, frozen_state, batch, sample, scale, flip, gpu):
    import torch
    image, metas = batch['img'], batch['img_metas']
    meta = metas[0]
    if (image.shape[0] != 1 or Path(meta['filename']).stem != sample['image']
            or bool(meta.get('flip', False)) != flip
            or (flip and meta.get('flip_direction') != 'horizontal')):
        raise ValueError('Fixed TRAIN loader substituted an image/view')
    assert_detector_frozen(detector)
    gt = image.new_tensor(sample['gt']).reshape(1, 5)
    expected_gt = canonical_boxes(map_boxes(gt, meta, size_mode='annotation'))
    actual_gt = canonical_boxes(batch['gt_bboxes'][0])
    if actual_gt.shape != (1, 5) or not torch.allclose(expected_gt, actual_gt, atol=1e-3, rtol=1e-5):
        raise ValueError('Actual TRAIN GT transform disagrees with RResize/shrink/reflection')
    axis = image.new_tensor(sample.get('native_axis', sample['obb_derived_axis']))
    axis_model = map_points(axis, meta)
    if not torch.allclose(map_points(axis_model, meta, inverse=True), axis, atol=1e-3, rtol=1e-5):
        raise ValueError('Axis roundtrip to original coordinates failed')

    def original_forward():
        with torch.no_grad():
            feats = detector.extract_feat(image)
            raw = flatten_prediction(detector.simple_test_from_features(feats, metas, rescale=False))
        return feats, raw
    (features, raw), detector_cost = measured(original_forward, gpu)
    if any(f.requires_grad or f.grad_fn is not None for f in features):
        raise ValueError('Frozen FPN retained a detector computation graph')
    p3 = features[SETTINGS['feature_level']]
    if p3.shape != (1, 256, 128, 128) or tuple(image.shape[-2:]) != (1024, 1024):
        raise ValueError('Fixed B pad/P3 resolution differs')
    b_model = image.new_tensor(raw[:, :5])
    b_original = map_boxes(b_model, meta, inverse=True)
    if len(raw):
        if not torch.allclose(map_boxes(b_original, meta), b_model, atol=1e-3, rtol=1e-5):
            raise ValueError('Original B box restoration roundtrip failed')
        if not flip:
            with torch.no_grad():
                native_original = flatten_prediction(detector.simple_test_from_features(features, metas, rescale=True))
            if not np.allclose(b_original.cpu().numpy(), native_original[:, :5], atol=1e-4, rtol=1e-6):
                raise ValueError('Reliability wrapper disagrees with B native rescale=True')
    targets_map, valid = response_targets(axis_model, p3.shape[-2:], meta)
    probes_original, probe_names = component_probes_original(gt)
    boxes_original = torch.cat((b_original, probes_original))
    boxes_model = torch.cat((b_model, map_boxes(probes_original, meta)))
    # Same observed B score for every probe; missing output uses one fixed value.
    # No different sentinel, GT quality, probe type, sequence/domain input.
    score = float(raw[0, 5]) if len(raw) else .5
    scores = image.new_full((len(boxes_model),), score)
    quality_targets, quality_mask, errors = quality_targets_original(boxes_original, gt)
    seed_all()
    branch = StructureComponentReliability().cuda(gpu).train()
    initial_branch = state_digest(branch)
    optimizer = torch.optim.SGD(branch.parameters(), lr=SETTINGS['lr'], momentum=0., weight_decay=0.)

    def branch_update():
        optimizer.zero_grad()
        online = branch(p3, boxes_model, scores, meta)
        structure, components = balanced_response_loss(online['response_logits'], targets_map, valid)
        quality, parts = quality_loss(online['qualities'], quality_targets, quality_mask, len(raw))
        loss = SETTINGS['structure_loss_weight']*structure+SETTINGS['quality_loss_weight']*quality
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite branch loss')
        before_q = online['qualities'].detach().cpu().tolist()
        # Narrow checks of each NEW task's own final-layer supervision. These
        # diagnostic backwards are included in measured preflight overhead.
        gs = torch.autograd.grad(structure, branch.response.weight, retain_graph=True)[0]
        gq = torch.autograd.grad(quality, branch.quality[-1].weight, retain_graph=True)[0]
        task_norms = dict(
            structure_center=float(gs[0].double().square().sum().sqrt()),
            structure_axis=float(gs[1].double().square().sum().sqrt()),
            quality_center=float(gq[0].double().square().sum().sqrt()),
            quality_size=float(gq[1].double().square().sum().sqrt()),
            quality_angle=float(gq[2].double().square().sum().sqrt()))
        required = list(task_norms) if bool(quality_mask[:, 2].sum() > 0) else list(task_norms)[:-1]
        if not all(np.isfinite(v) for v in task_norms.values()) or any(task_norms[k] <= 0 for k in required):
            raise ValueError('A supervised structure/quality task has no finite effective final-layer gradient')
        loss.backward()
        gradients = norms(branch)
        norm = torch.nn.utils.clip_grad_norm_(branch.parameters(), SETTINGS['clip_norm'])
        if not bool(torch.isfinite(norm)):
            raise ValueError('Nonfinite branch clipping norm')
        optimizer.step()
        return dict(loss_total=float(loss.detach()), loss_structure=float(structure.detach()),
            loss_center_response=float(components[0].detach()), loss_axis_response=float(components[1].detach()),
            loss_quality=float(quality.detach()), loss_quality_genuine=float(parts['genuine'].detach()),
            loss_quality_probes=float(parts['probe'].detach()), group_gradient_norms=gradients,
            supervised_task_final_layer_gradient_norms=task_norms,
            global_preclip_norm=float(norm),
            clip_multiplier=min(1., SETTINGS['clip_norm']/(float(norm)+1e-6)),
            qualities_before_update=before_q, roi_support=online['roi_support'].detach().cpu().tolist())
    update, branch_cost = measured(branch_update, gpu)
    changed_branch = state_digest(branch)
    if changed_branch == initial_branch:
        raise ValueError('Branch-only optimizer failed to change parameters')
    assert_detector_frozen(detector)
    if state_digest(detector) != frozen_state:
        raise ValueError('B parameters or buffers changed after branch update')
    (after_features, after_raw), repeat_cost = measured(original_forward, gpu)
    if not np.array_equal(raw, after_raw):
        raise ValueError('Original B predictions changed after branch update')
    with torch.no_grad():
        # This checks the ONLINE genuine-output API, without passing probes/GT.
        genuine = branch(after_features[0], image.new_tensor(after_raw[:, :5]),
                         image.new_tensor(after_raw[:, 5]), meta)
    if not bool(genuine['qualities'].isfinite().all()):
        raise ValueError('Nonfinite genuine online qualities')
    return dict(image=sample['image'], source=sample['source'], split=sample['split'],
        view=dict(scale=scale, horizontal_flip=flip), seed=SETTINGS['seed'],
        input_sha256=tensor_sha(image), image_sha256=sample['image_sha256'],
        annotation_sha256=sample['annotation_sha256'], axis_json_sha256=sample.get('axis_json_sha256'),
        img_shape=list(meta['img_shape']), pad_shape=list(meta['pad_shape']),
        scale_factor=np.asarray(meta['scale_factor']).tolist(), p3_shape=list(p3.shape),
        original_gt=gt.cpu().tolist(), model_gt=actual_gt.cpu().tolist(),
        axis_original=axis.cpu().tolist(), axis_model=axis_model.cpu().tolist(),
        axis_roundtrip_max_px=float((map_points(axis_model, meta, inverse=True)-axis).abs().max()),
        model_gt_transform_max=float((actual_gt-expected_gt).abs().max()),
        gt_long_cells=float(expected_gt[0, 2]/8), gt_short_cells=float(expected_gt[0, 3]/8),
        response_positive_mass=(targets_map*valid).sum(dim=(0, 2, 3)).cpu().tolist(),
        valid_feature_cells=int(valid.sum()), genuine_count=len(raw), probes=len(probe_names),
        offline_probe_names=probe_names, offline_errors_original=errors.cpu().tolist(),
        offline_quality_targets=quality_targets.cpu().tolist(), offline_quality_mask=quality_mask.cpu().tolist(),
        original_b_output_model=raw.tolist(), original_b_output_original=b_original.cpu().tolist(),
        genuine_online_qualities_after_one_step=genuine['qualities'].cpu().tolist(),
        native_rescale_checked=not flip, b_raw_prediction_exactly_unchanged=True,
        b_parameters_buffers_unchanged=True, b_eval_no_grad=True,
        branch_state_before=initial_branch, branch_state_after=changed_branch,
        branch_optimizer_steps=1, branch_checkpoint_saved=False,
        measurements=dict(detector_initial=detector_cost, branch_forward_backward_step=branch_cost,
                          detector_repeat=repeat_cost), **update)


def runtime(cfg, prerequisite, checkpoint, gpu, progress, rows):
    import torch
    import mmcv
    import cv2
    import mmdet
    import mmrotate
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for actual frozen-B integration preflight')
    if ready.sha(checkpoint) != ready.FROZEN_B['checkpoint_sha256']:
        raise ValueError('Requires frozen B epoch_24 checkpoint SHA')
    import_modules_from_strings(**cfg.custom_imports)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_all()
    # build+strict load only; do not init_weights/download ImageNet or load optimizer.
    detector = build_detector(deepcopy(cfg.model))
    loaded = load_checkpoint(detector, str(checkpoint), map_location='cpu', strict=True)
    if loaded.get('meta', {}).get('epoch') != 24:
        raise ValueError('B checkpoint metadata epoch differs')
    del loaded
    freeze_detector(detector.cuda(gpu))
    before = state_digest(detector)
    indexed = {r['image']: r for r in prerequisite['samples']}
    initial_branch_states = set()
    # Two deterministic pipeline instances; no random TRAIN sample reselection.
    datasets = {}
    indices = {}
    for scale, flip in sorted({(s, f) for _, s, f in VIEWS}):
        ds = build_dataset(fixed_specs(cfg, scale, flip))
        if [len(d) for d in ds.datasets] != [1810, 748]:
            raise ValueError('Actual TRAIN datasets differ')
        names = [Path(info['filename']).stem for part in ds.datasets for info in part.data_infos]
        if len(set(names)) != len(names) or not set(SAMPLES).issubset(names):
            raise ValueError('TRAIN image index differs')
        datasets[(scale, flip)] = ds
        indices[(scale, flip)] = {n: i for i, n in enumerate(names)}
    for name, scale, flip in VIEWS:
        seed_all()
        ds = datasets[(scale, flip)]
        item = ds[indices[(scale, flip)][name]]
        batch = scatter(collate([item], samples_per_gpu=1), [gpu])[0]
        row = probe_view(detector, before, batch, indexed[name], scale, flip, gpu)
        rows.append(row)
        initial_branch_states.add(row['branch_state_before']['parameters'])
        with progress.open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        print('TRAIN', name, 'scale', scale, 'flip', flip, 'B outputs', row['genuine_count'],
              'loss', round(row['loss_total'], 6), 'grad', round(row['global_preclip_norm'], 6),
              'B unchanged', row['b_raw_prediction_exactly_unchanged'], flush=True)
        del batch, item, row
    if len(initial_branch_states) != 1 or state_digest(detector) != before:
        raise ValueError('Branch reset or final frozen B state differs')
    base_rows = [r for r in rows if r['view'] == dict(scale=1., horizontal_flip=False)]
    outputs = sum(r['genuine_count'] for r in base_rows)
    hits = sum(r['genuine_count'] == 1 and r['offline_errors_original'][0][0] < 15 for r in base_rows)
    return dict(b_state_before=before, b_state_after=state_digest(detector),
        branch_initialization_equal_across_views=True, detector_optimizer_steps=0,
        discarded_independent_branch_steps=len(rows), checkpoint_sha256=ready.sha(checkpoint),
        fixed_train_sample_raw_b=dict(frames=len(base_rows), outputs=outputs, center_hits=hits,
            output_coverage=outputs/len(base_rows), center_hit_rate_on_outputs=hits/outputs if outputs else None,
            all_frame_center_correct_coverage=hits/len(base_rows),
            role='Only twelve fixed TRAIN integration examples; not dataset performance/continuity evidence.'),
        torch_version=torch.__version__, mmcv_version=mmcv.__version__,
        mmdet_version=mmdet.__version__, mmrotate_version=mmrotate.__version__,
        numpy_version=np.__version__, opencv_version=cv2.__version__, python_version=sys.version,
        cuda_device=torch.cuda.get_device_name(gpu), cuda_logical_index=gpu,
        memory_note='Allocated and reserved peaks per synchronized stage; cached reservations carry over. '
                    'Stage delta is diagnostic, not a matched formal-training incremental-memory benchmark. '
                    'Branch stage includes two task-gradient diagnostic calls. '
                    'First view includes CUDA/operator warmup; timings are not deployment latency.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--structure-report', type=Path, default=ROOT/'work_dirs/port_reliability_readiness_v1_structure_complete/train_structure_check.json')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--out-json', type=Path, default=ROOT/'work_dirs/port_structure_reliability_v1_train_preflight.json')
    args = parser.parse_args()
    out = args.out_json.resolve()
    progress = out.with_suffix('.progress.jsonl')
    if out.exists() or progress.exists():
        raise FileExistsError('Preserve existing evidence; use a new --out-json path')
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    report = dict(protocol=SETTINGS['version'], evidence_role='finite_train_integration_not_fitted_reliability',
        settings=SETTINGS, views=[dict(image=n, scale=s, horizontal_flip=f) for n, s, f in VIEWS],
        test_repeatedly_exposed=True, val_or_test_used=False, detector_optimizer_steps=0,
        branch_checkpoint_saved=False,
        limitations=['No trained/calibrated reliability model, policy or performance result.',
                     'B outputs and depth inputs remain unchanged; depth accuracy is not assessed.',
                     'Real short edges are L/k annotation constructions, not independently visible boundaries.',
                     'Sim axis targets are OBB-derived weak labels; endpoint visibility is not supervised.',
                     'TRAIN images were previously used to fit B; this is not independent error calibration.',
                     'Quality target references are preflight design scales, not selected acceptance thresholds.'])
    try:
        report['source_contract'] = checked_sources()
        report['prerequisite'] = prerequisites(args.structure_report)
        cfg = check_cfg()
        if args.check_only:
            report['status'] = 'STATIC_TRAIN_CONTRACT_PASS_NO_GPU'
            report['branch_optimizer_steps'] = 0
        else:
            progress.touch(exist_ok=False)
            report['runtime'] = runtime(cfg, report['prerequisite'], args.checkpoint, args.gpu, progress, rows)
            report['branch_optimizer_steps'] = len(rows)
            report['status'] = 'TRAIN_PREFLIGHT_COMPLETE_REVIEW_REQUIRED'
        if checked_sources() != report['source_contract']:
            raise ValueError('Source contract changed during the run')
    except Exception as exc:
        report.update(status='FAILED_REVIEW_REQUIRED', error_type=type(exc).__name__, error=str(exc), rows=rows,
                      completed_discarded_branch_steps=len(rows))
        write_new(out, report)
        raise
    report['rows'] = rows
    write_new(out, report)
    print('Saved', out, 'status', report['status'], flush=True)


if __name__ == '__main__':
    main()
