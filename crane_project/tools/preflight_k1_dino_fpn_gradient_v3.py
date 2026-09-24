"""Check the A/B/C contract and one real source batch before training.

This is deliberately a one-batch gradient probe, not a training run or a
checkpoint selector.  Run it on the server with the existing DINO cache.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Direct invocation by path starts with crane_project/tools on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crane_project.tools.preflight_k1_dino_warmstart_v2 import (
    EXPECTED_BASELINE_SHA256, sha256)


def validate_configs(configs):
    """Enforce every row of the three-arm comparison before model creation."""
    if len(configs) != 3:
        raise ValueError('Exactly A, B, C configurations are required')
    shared = ('data', 'data_root', 'train_pipeline', 'test_pipeline',
              'optimizer', 'optimizer_config', 'lr_config', 'runner',
              'checkpoint_config', 'evaluation', 'load_from', 'resume_from')
    first = configs[0]
    for arm, cfg in zip('ABC', configs):
        for key in shared:
            if cfg.get(key) != first.get(key):
                raise ValueError('{} differs in {}'.format(arm, key))
        if cfg.model.backbone.frozen_stages != 4:
            raise ValueError('{} does not freeze all ResNet stages'.format(arm))
        if not cfg.model.bbox_head.use_semantic_cls_adapter:
            raise ValueError('{} lacks the classification adapter'.format(arm))
        if cfg.runner.max_epochs != 4 or cfg.optimizer.lr != 0.00025:
            raise ValueError('{} changes the warmstart budget'.format(arm))
        if cfg.data.samples_per_gpu != 2:
            raise ValueError('{} changes the training batch size'.format(arm))
        if sum(step['type'] == 'LoadDinoFeatureFromCache'
               for step in cfg.train_pipeline) != 1:
            raise ValueError('{} must use the existing DINO cache'.format(arm))
        if [(item.ann_file, item.img_prefix) for item in cfg.data.train] != [
                ('train/annfiles/', 'train/images/'),
                ('train_sim/annfiles/', 'train/images/')]:
            raise ValueError('{} changes source training data'.format(arm))
        if cfg.resume_from is not None:
            raise ValueError('{} unexpectedly resumes training'.format(arm))
        if arm == 'A':
            if cfg.model.get('semantic_distillation') is not None:
                raise ValueError('A must have no distillation loss')
        else:
            loss = cfg.model.get('semantic_distillation')
            if (not loss or not loss.get('enabled')
                    or loss.get('mode') != 'feature'
                    or loss.get('scope') != 'foreground'
                    or loss.get('feature_level') != 0
                    or loss.get('loss_weight') != 0.05
                    or loss.get('max_tokens') != 4096
                    or loss.get('teacher_channels') != 1024
                    or loss.get('protect_geometry') != (arm == 'B')):
                raise ValueError('{} has the wrong distillation contract'.format(
                    arm))
        model_without_loss = dict(cfg.model)
        model_without_loss.pop('semantic_distillation', None)
        reference = dict(first.model)
        reference.pop('semantic_distillation', None)
        if model_without_loss != reference:
            raise ValueError('{} changes model architecture'.format(arm))
    b_loss = dict(configs[1].model.semantic_distillation)
    c_loss = dict(configs[2].model.semantic_distillation)
    b_loss.pop('protect_geometry')
    c_loss.pop('protect_geometry')
    if b_loss != c_loss:
        raise ValueError('B and C differ beyond FPN gradient routing')
    checkpoint = Path(first.load_from).resolve()
    if sha256(checkpoint) != EXPECTED_BASELINE_SHA256:
        raise ValueError('K1 epoch_20 checkpoint SHA256 mismatch')
    return checkpoint


def _nonzero(grads):
    return any(g is not None and torch.isfinite(g).all().item()
               and torch.count_nonzero(g).item() > 0 for g in grads)


def _loss_sum(losses):
    terms = []
    for name, value in losses.items():
        if 'loss' not in name:
            continue
        if isinstance(value, torch.Tensor):
            terms.append(value.mean())
        elif isinstance(value, (list, tuple)):
            terms.extend(item.mean() for item in value)
        else:
            raise TypeError('Unsupported loss type: ' + name)
    if not terms:
        raise ValueError('Model returned no training losses')
    return sum(terms)


def _check_cached_feature_transform(cfg, batch):
    """Recreate the cache flip/pad/resize for this source image and metadata."""
    meta = batch['img_metas'][0]
    filename = meta.get('filename')
    if not filename:
        raise ValueError('Training metadata has no source image filename')
    cache_step = next(step for step in cfg.train_pipeline
                      if step['type'] == 'LoadDinoFeatureFromCache')
    candidates = list(Path(cache_step['cache_dir']).glob(
        '*/' + Path(filename).stem + '_*.pth'))
    matches = []
    stat = Path(filename).stat()
    for path in candidates:
        try:
            candidate = torch.load(str(path), map_location='cpu',
                                   weights_only=False)
        except TypeError:
            candidate = torch.load(str(path), map_location='cpu')
        signature = candidate.get('signature') or {}
        identity = signature.get('image') or {}
        if (signature.get('dinov2_model') == 'dinov2_vitl14'
                and 'paired_view' not in signature
                and Path(str(identity.get('path', ''))).name == Path(filename).name
                and int(identity.get('size', -1)) == stat.st_size
                and int(identity.get('mtime_ns', -1)) == stat.st_mtime_ns):
            matches.append((path, candidate))
    if len(matches) != 1:
        raise ValueError('Source sample has no unique identity-bound DINO cache')
    cache_path, payload = matches[0]
    feature = payload['feature'][0].float()
    direction = meta.get('flip_direction') if meta.get('flip') else None
    if direction in ('horizontal', 'diagonal'):
        feature = torch.flip(feature, dims=(-1,))
    if direction in ('vertical', 'diagonal'):
        feature = torch.flip(feature, dims=(-2,))
    if direction not in (None, 'horizontal', 'vertical', 'diagonal'):
        raise ValueError('Unexpected source image flip direction')
    h, w = feature.shape[-2:]
    side = max(h, w)
    feature = F.pad(feature, (0, side - w, 0, side - h))
    expected = F.interpolate(feature.unsqueeze(0), size=(64, 64),
                             mode='bilinear', align_corners=False)[0].half()
    if not torch.equal(expected.to(batch['teacher_features'].device),
                       batch['teacher_features'][0]):
        raise ValueError('Cached teacher feature does not match image flip')
    return dict(cache_path=str(cache_path),
                flip_direction=direction,
                cache_image_identity_checked_by_loader=True)


def _trainable_contract(model, arm):
    groups = dict(
        backbone=list(model.backbone.named_parameters()),
        fpn=list(model.neck.named_parameters()),
        detection_head=list(model.bbox_head.named_parameters()),
        adapter=list(model.bbox_head.semantic_cls_adapter.named_parameters()))
    if not groups['backbone'] or any(p.requires_grad for _, p in groups['backbone']):
        raise ValueError('{} backbone is not fully frozen'.format(arm))
    for name in ('fpn', 'detection_head', 'adapter'):
        if not groups[name] or not all(p.requires_grad for _, p in groups[name]):
            raise ValueError('{} {} is not fully trainable'.format(arm, name))
    if model.aux_heads is None or not all(
            p.requires_grad for p in model.aux_heads.parameters()):
        raise ValueError('{} auxiliary detection head is not trainable'.format(
            arm))
    if arm == 'A' and model.semantic_distillation is not None:
        raise ValueError('A unexpectedly built the DINO projection')
    if arm != 'A' and (model.semantic_distillation is None or not all(
            p.requires_grad for p in model.semantic_distillation.parameters())):
        raise ValueError('{} projection is not trainable'.format(arm))


def probe_arm(cfg, arm, checkpoint, gpu):
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector

    random.seed(1701)
    np.random.seed(1701)
    torch.manual_seed(1701)
    torch.cuda.manual_seed_all(1701)
    dataset = build_dataset(cfg.data.train)
    sample = dataset[0]
    batch = scatter(collate([sample], samples_per_gpu=1), [gpu])[0]
    if (batch['teacher_features'].shape[1:] != (1024, 64, 64)
            or not torch.isfinite(batch['teacher_features']).all().item()):
        raise ValueError('{} invalid cached teacher feature'.format(arm))
    cache_check = _check_cached_feature_transform(cfg, batch)
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    load_checkpoint(model, str(checkpoint), map_location='cpu', strict=False)
    if torch.count_nonzero(model.bbox_head.semantic_cls_adapter.weight).item():
        raise ValueError('{} adapter is not zero after K1 load'.format(arm))
    model = model.cuda(gpu)
    model.train()
    _trainable_contract(model, arm)
    torch.cuda.reset_peak_memory_stats(gpu)

    with torch.no_grad():
        features = model.extract_feat(batch['img'])
        mask = model._build_distillation_foreground_mask(
            features[0], batch['img_metas'], batch['gt_bboxes'])
        if not mask.any().item() or mask.all().item():
            raise ValueError('{} foreground mask misses source GT'.format(arm))
        before = model.bbox_head.forward_single(features[0])[0].detach().clone()
    mask_pixels = int(mask.sum().item())
    del features, mask

    losses = model(return_loss=True, **batch)
    distill = losses.get('loss_semantic_distill')
    fpn_params = list(model.neck.parameters())
    adapter_params = list(model.bbox_head.semantic_cls_adapter.parameters())
    projection_params = (list(model.semantic_distillation.parameters())
                         if arm != 'A' else [])
    if arm == 'A':
        if distill is not None:
            raise ValueError('A returned a distillation loss')
        distill_fpn = False
        distill_adapter = False
        distill_projection = False
    else:
        if distill is None or not torch.isfinite(distill).item() or distill.item() <= 0:
            raise ValueError('{} invalid DINO loss'.format(arm))
        isolated = torch.autograd.grad(
            distill, fpn_params + adapter_params + projection_params,
            retain_graph=True, allow_unused=True)
        n_fpn = len(fpn_params)
        n_adapter = len(adapter_params)
        distill_fpn = _nonzero(isolated[:n_fpn])
        distill_adapter = _nonzero(isolated[n_fpn:n_fpn + n_adapter])
        distill_projection = _nonzero(isolated[n_fpn + n_adapter:])
        if (distill_fpn != (arm == 'C') or not distill_adapter
                or not distill_projection):
            raise ValueError('{} DINO gradient reaches wrong parameters'.format(
                arm))

    total = _loss_sum(losses)
    if not torch.isfinite(total).item():
        raise ValueError('{} nonfinite total training loss'.format(arm))
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay)
    optimizer.zero_grad()
    total.backward()
    if not _nonzero([p.grad for p in fpn_params]):
        raise ValueError('{} FPN gets no total training gradient'.format(arm))
    if any(p.grad is not None for p in model.backbone.parameters()):
        raise ValueError('{} frozen backbone received gradients'.format(arm))
    optimizer.step()
    with torch.no_grad():
        after_features = model.extract_feat(batch['img'])
        after = model.bbox_head.forward_single(after_features[0])[0]
        cls_change = float((after - before).abs().max().item())
    if cls_change == 0:
        raise ValueError('{} classification output did not change'.format(arm))
    return dict(arm=arm, source_image=batch['img_metas'][0].get('filename'),
                cache_transform=cache_check,
                foreground_fpn_pixels=mask_pixels,
                distillation_loss=(float(distill.item()) if distill is not None
                                   else None),
                distillation_gradient_to_fpn=distill_fpn,
                distillation_gradient_to_adapter=distill_adapter,
                distillation_gradient_to_projection=distill_projection,
                classification_max_abs_change=cls_change,
                peak_allocated_mib=torch.cuda.max_memory_allocated(gpu) / 2**20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in 'abc':
        parser.add_argument('--config-' + arm, required=True)
    parser.add_argument('--gpu', type=int, default=0,
                        help='Logical CUDA device after CUDA_VISIBLE_DEVICES')
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('One CUDA GPU is required for the real-batch probe')
    from mmcv import Config
    cfgs = [Config.fromfile(getattr(args, 'config_' + arm))
            for arm in 'abc']
    checkpoint = validate_configs(cfgs)
    torch.cuda.set_device(args.gpu)
    results = [probe_arm(cfg, arm, checkpoint, args.gpu)
               for cfg, arm in zip(cfgs, 'ABC')]
    if len({r['source_image'] for r in results}) != 1:
        raise ValueError('A/B/C source samples are not aligned')
    if len({r['cache_transform']['flip_direction'] for r in results}) != 1:
        raise ValueError('A/B/C source augmentations are not aligned')
    report = dict(protocol='k1_dino_fpn_gradient_preflight_v3',
                  decision='MATCHED_FPN_GRADIENT_READY',
                  source_only=True, full_training_performed=False,
                  temporary_probe_optimizer_steps_per_arm=1,
                  k1_checkpoint=str(checkpoint),
                  k1_checkpoint_sha256=sha256(checkpoint),
                  config_sha256={arm: sha256(getattr(args, 'config_' + arm.lower()))
                                 for arm in 'ABC'},
                  arms=results)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError('Refusing to overwrite ' + str(out))
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('MATCHED_FPN_GRADIENT_READY')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
