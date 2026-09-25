"""Short source-only probe of corrected-mask DINO feature supervision.

Build one K1-initialized arm C student, use one real and one sim training
sample, and measure isolated FPN gradients and a temporary distillation-only
raw-gradient step at the configured LR. No training checkpoint, VAL, or TEST
is read or written.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crane_project.tools.preflight_k1_dino_fpn_gradient_v3 import (
    _check_cached_feature_transform, _loss_sum, _trainable_contract,
    validate_configs)
from crane_project.tools.preflight_k1_dino_warmstart_v2 import sha256


PROTOCOL = 'k1_dino_corrected_mask_source_probe_v4'
CONFIG_DIR = Path('crane_project/configs')
CONFIG_NAME = 'crane_symeood_k1_dino_fpn_gradient_{}_v3.py'


def gradient_comparison(detection_grads, distillation_grads):
    """Return global FPN norms and cosine, treating unused grads as zero."""
    if len(detection_grads) != len(distillation_grads):
        raise ValueError('Gradient groups have different lengths')
    det_sq = dist_sq = dot = 0.0
    for det, dist in zip(detection_grads, distillation_grads):
        if det is not None:
            det = det.detach().float()
            if not torch.isfinite(det).all().item():
                raise ValueError('Nonfinite detection gradient')
            det_sq += float((det * det).sum().item())
        if dist is not None:
            dist = dist.detach().float()
            if not torch.isfinite(dist).all().item():
                raise ValueError('Nonfinite distillation gradient')
            dist_sq += float((dist * dist).sum().item())
        if det is not None and dist is not None:
            dot += float((det * dist).sum().item())
    det_norm, dist_norm = math.sqrt(det_sq), math.sqrt(dist_sq)
    if det_norm <= 0 or dist_norm <= 0:
        raise ValueError('Detection or distillation FPN gradient is zero')
    return dict(detection_l2=det_norm, distillation_l2=dist_norm,
                distillation_to_detection_ratio=dist_norm / det_norm,
                cosine=max(-1.0, min(1.0, dot / (det_norm * dist_norm))))


def _old_unrotated_mask(feature, img_metas, gt_bboxes):
    """Historical V3 box envelope, for measurement only."""
    batch, _, feat_h, feat_w = feature.shape
    mask = feature.new_zeros((batch, 1, feat_h, feat_w))
    for index, (meta, boxes) in enumerate(zip(img_metas, gt_bboxes)):
        pad_h, pad_w = meta.get('pad_shape', meta['img_shape'])[:2]
        sx, sy = feat_w / float(pad_w), feat_h / float(pad_h)
        for box in boxes.detach():
            cx, cy, width, height = [float(v) for v in box[:4]]
            x0 = max(0, min(feat_w - 1, int((cx - width / 2) * sx)))
            x1 = max(x0 + 1, min(feat_w,
                                 int((cx + width / 2) * sx) + 1))
            y0 = max(0, min(feat_h - 1, int((cy - height / 2) * sy)))
            y1 = max(y0 + 1, min(feat_h,
                                 int((cy + height / 2) * sy) + 1))
            mask[index, 0, y0:y1, x0:x1] = 1
    return mask


def _foreground_teacher_tokens(mask, teacher_size):
    return F.adaptive_max_pool2d(mask.float(), teacher_size)[:, 0] > 0


def _verify_corrected_runtime(model, gpu):
    """Fail if the server still runs either historical foreground operation."""
    with torch.no_grad():
        fake_feature = torch.zeros((1, 1, 16, 16), device='cuda:' + str(gpu))
        meta = [dict(img_shape=(16, 16, 3), pad_shape=(16, 16, 3))]
        rotated_box = [torch.tensor(
            [[8., 8., 10., 2., math.pi / 2]], device=fake_feature.device)]
        mask = model._build_distillation_foreground_mask(
            fake_feature, meta, rotated_box)
        if (mask[0, 0, 4, 8].item() != 1
                or mask[0, 0, 8, 4].item() != 0):
            raise RuntimeError('Server is using the old angle-blind mask')
        student = torch.randn((1, 256, 4, 4), device=fake_feature.device)
        teacher = torch.randn((1, 1024, 2, 2), device=fake_feature.device)
        single_cell = torch.zeros((1, 1, 4, 4),
                                  device=fake_feature.device)
        single_cell[0, 0, 1, 1] = 1
        loss = model.semantic_distillation((student,), teacher, single_cell)
        if not torch.isfinite(loss).item() or loss.item() <= 0:
            raise RuntimeError('Server drops a one-cell foreground token')


def _snapshot(model, images):
    with torch.no_grad():
        features = model.extract_feat(images)
        logits = model.bbox_head(features)[0]
        return (features[0].detach().clone(),
                [value.detach().clone() for value in logits])


def _response_change(before, after):
    before_fpn, before_logits = before
    after_fpn, after_logits = after
    if len(before_logits) != len(after_logits):
        raise ValueError('Classification head level count changed')
    result = []
    for index in range(before_fpn.size(0)):
        logit_deltas = [(a[index] - b[index]).abs()
                        for a, b in zip(after_logits, before_logits)]
        score_deltas = [(a[index].sigmoid() - b[index].sigmoid()).abs()
                        for a, b in zip(after_logits, before_logits)]
        fpn_delta = (after_fpn[index] - before_fpn[index]).abs()
        if (not torch.isfinite(fpn_delta).all().item()
                or any(not torch.isfinite(value).all().item()
                       for value in logit_deltas + score_deltas)):
            raise ValueError('Nonfinite temporary-step student response')
        result.append(dict(
            fpn_p3_mean_abs_change=float(fpn_delta.mean().item()),
            fpn_p3_max_abs_change=float(fpn_delta.max().item()),
            classification_logit_max_abs_change=max(
                float(value.max().item()) for value in logit_deltas),
            classification_score_max_abs_change=max(
                float(value.max().item()) for value in score_deltas)))
    return result


def probe(project_root, gpu):
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector

    root = Path(project_root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError('Run the source probe from the project root')
    cfg_paths = [root / CONFIG_DIR / CONFIG_NAME.format(arm)
                 for arm in 'abc']
    configs = [Config.fromfile(str(path)) for path in cfg_paths]
    checkpoint = validate_configs(configs)
    cfg = configs[2]
    if cfg.model.semantic_distillation.protect_geometry:
        raise ValueError('Arm C must allow distillation into FPN')

    random.seed(1701)
    np.random.seed(1701)
    torch.manual_seed(1701)
    torch.cuda.manual_seed_all(1701)
    torch.cuda.set_device(gpu)
    dataset = build_dataset(cfg.data.train)
    if not hasattr(dataset, 'datasets') or len(dataset.datasets) != 2:
        raise ValueError('Expected train and train_sim source datasets')
    source_indices = [0, len(dataset.datasets[0])]
    samples = [dataset[index] for index in source_indices]
    cache_checks = []
    for sample in samples:
        single = scatter(collate([sample], samples_per_gpu=1), [gpu])[0]
        cache_checks.append(_check_cached_feature_transform(cfg, single))
    batch = scatter(collate(samples, samples_per_gpu=len(samples)), [gpu])[0]
    if (batch['teacher_features'].shape[1:] != (1024, 64, 64)
            or not torch.isfinite(batch['teacher_features']).all().item()):
        raise ValueError('Invalid source teacher cache tensor')

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    load_checkpoint(model, str(checkpoint), map_location='cpu', strict=False)
    if torch.count_nonzero(model.bbox_head.semantic_cls_adapter.weight).item():
        raise ValueError('K1 initialization did not leave adapter at zero')
    model = model.cuda(gpu)
    model.train()
    _trainable_contract(model, 'C')
    _verify_corrected_runtime(model, gpu)
    torch.cuda.reset_peak_memory_stats(gpu)

    model.eval()
    before = _snapshot(model, batch['img'])
    with torch.no_grad():
        corrected_mask = model._build_distillation_foreground_mask(
            before[0], batch['img_metas'], batch['gt_bboxes'])
        old_mask = _old_unrotated_mask(
            before[0], batch['img_metas'], batch['gt_bboxes'])
        corrected_tokens = _foreground_teacher_tokens(
            corrected_mask, batch['teacher_features'].shape[-2:])
        old_nearest_tokens = F.interpolate(
            old_mask.float(), size=batch['teacher_features'].shape[-2:],
            mode='nearest')[:, 0] > 0
    if not corrected_mask.any().item() or not corrected_tokens.any().item():
        raise ValueError('Corrected mask has no supervised source tokens')

    model.train()
    losses = model(return_loss=True, **batch)
    distillation = losses.get('loss_semantic_distill')
    if distillation is None or not torch.isfinite(distillation).item():
        raise ValueError('Missing or nonfinite distillation loss')
    detection = _loss_sum({key: value for key, value in losses.items()
                           if key != 'loss_semantic_distill'})
    if not torch.isfinite(detection).item():
        raise ValueError('Nonfinite detection loss')
    fpn_params = list(model.neck.parameters())
    adapter_params = list(model.bbox_head.semantic_cls_adapter.parameters())
    dist_grads = torch.autograd.grad(
        distillation, fpn_params + adapter_params,
        retain_graph=True, allow_unused=True)
    det_grads = torch.autograd.grad(
        detection, fpn_params, allow_unused=True)
    fpn_stats = gradient_comparison(
        det_grads, dist_grads[:len(fpn_params)])
    if any(not torch.isfinite(grad).all().item()
           for grad in dist_grads[len(fpn_params):] if grad is not None):
        raise ValueError('Nonfinite adapter distillation gradient')
    adapter_norm = math.sqrt(sum(
        float((grad.detach().float() ** 2).sum().item())
        for grad in dist_grads[len(fpn_params):] if grad is not None))
    if adapter_norm <= 0:
        raise ValueError('Distillation does not update classification adapter')

    params = fpn_params + adapter_params
    saved = [param.detach().clone() for param in params]
    lr = float(cfg.optimizer.lr)
    try:
        with torch.no_grad():
            for param, grad in zip(params, dist_grads):
                if grad is not None:
                    param.add_(grad, alpha=-lr)
        model.eval()
        after = _snapshot(model, batch['img'])
        changes = _response_change(before, after)
    finally:
        with torch.no_grad():
            for param, value in zip(params, saved):
                param.copy_(value)

    sample_reports = []
    for index, meta in enumerate(batch['img_metas']):
        if len(batch['gt_bboxes'][index]) == 0:
            raise ValueError('Source probe sample has no GT box')
        new = corrected_mask[index, 0] > 0
        old = old_mask[index, 0] > 0
        sample_reports.append(dict(
            source_dataset_index=source_indices[index],
            image=meta.get('filename'),
            cache_transform=cache_checks[index],
            gt_count=int(len(batch['gt_bboxes'][index])),
            gt_short_edge_px=min(float(box[2:4].min().item())
                                 for box in batch['gt_bboxes'][index]),
            corrected_fpn_mask_pixels=int(new.sum().item()),
            historical_fpn_mask_pixels=int(old.sum().item()),
            corrected_only_fpn_pixels=int((new & ~old).sum().item()),
            historical_only_fpn_pixels=int((old & ~new).sum().item()),
            corrected_teacher_tokens=int(
                corrected_tokens[index].sum().item()),
            historical_nearest_teacher_tokens=int(
                old_nearest_tokens[index].sum().item()),
            temporary_distillation_step_response=changes[index]))
    return dict(protocol=PROTOCOL, evidence_role='source_only_design_probe',
                checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                config_sha256={arm.upper(): sha256(path)
                               for arm, path in zip('abc', cfg_paths)},
                mask_implementation_sha256=sha256(
                    root / 'mmrotate/models/detectors/sym_eood_detector.py'),
                loss_implementation_sha256=sha256(
                    root / 'mmrotate/models/losses/semantic_feature_distill.py'),
                corrected_mask_runtime_verified=True,
                corrected_downsample_runtime_verified=True,
                fixed_seed=1701, source_sample_count=len(samples),
                training_epochs=0, temporary_gradient_steps=1,
                temporary_update_scope='FPN and classification adapter',
                temporary_update_uses_detection_loss=False,
                temporary_update_uses_momentum_weight_decay_or_clip=False,
                temporary_step_lr=lr,
                detection_loss=float(detection.detach().item()),
                distillation_loss=float(distillation.detach().item()),
                fpn_gradients=fpn_stats,
                adapter_distillation_gradient_l2=adapter_norm,
                samples=sample_reports,
                peak_allocated_mib=(torch.cuda.max_memory_allocated(gpu)
                                    / 2 ** 20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('One CUDA GPU is required for the source probe')
    output = Path(args.out_json)
    if output.exists():
        raise FileExistsError('Refusing to overwrite ' + str(output))
    report = probe(args.project_root, args.gpu)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False,
                  allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, indent=2, ensure_ascii=False,
                     allow_nan=False))


if __name__ == '__main__':
    main()
