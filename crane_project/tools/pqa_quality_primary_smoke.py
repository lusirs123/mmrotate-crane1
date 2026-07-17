#!/usr/bin/env python3
"""One-image CUDA smoke test for full PQA quality-primary.

The test loads the K1 checkpoint with ``strict=False``, verifies PQA inference,
checks finite clean/dark losses, and proves that PQA loss gradients do not
leak into backbone, FPN, classification, or bbox regression parameters.
"""

import argparse
import os
import sys

import numpy as np
import torch


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='crane_project/configs/'
                'crane_symeood_k1_pqa_quality_primary.py')
    parser.add_argument(
        '--checkpoint',
        default='work_dirs/crane_symeood_k1/epoch_24.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def module_has_grad(module):
    return any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum().item()) > 0.0
        for parameter in module.parameters())


def main():
    args = parse_args()
    if args.seed != 0:
        raise ValueError('Unified protocol requires seed=0')
    if args.image_size < 128:
        raise ValueError('--image-size must be at least 128')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the MMRotate smoke test')

    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmrotate.models import build_detector

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = Config.fromfile(args.config)
    if cfg.model.get('backbone', None) is not None:
        cfg.model.backbone.init_cfg = None
    model = build_detector(cfg.model)
    if model.pqa_head is None:
        raise RuntimeError('Config did not construct pqa_head')
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model = model.cuda(args.gpu)

    size = int(args.image_size)
    device = f'cuda:{args.gpu}'
    image = torch.zeros(1, 3, size, size, device=device)
    meta = dict(
        filename='synthetic_pqa_smoke.png',
        ori_filename='synthetic_pqa_smoke.png',
        ori_shape=(size, size, 3),
        img_shape=(size, size, 3),
        pad_shape=(size, size, 3),
        img_norm_cfg=dict(
            mean=np.asarray([123.675, 116.28, 103.53], dtype=np.float32),
            std=np.asarray([58.395, 57.12, 57.375], dtype=np.float32),
            to_rgb=True),
        scale_factor=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        flip=False,
        flip_direction=None)

    model.eval()
    with torch.no_grad():
        result = model.simple_test(image, [meta], rescale=False)
    if len(result) != 1 or len(result[0]) != model.bbox_head.num_classes:
        raise RuntimeError('Unexpected PQA simple_test result structure')
    detections = result[0][0]
    if detections.shape != (1, 6):
        raise RuntimeError(
            f'Expected one [cx,cy,w,h,a,Q] detection, got {detections.shape}')
    if not np.isfinite(detections).all():
        raise RuntimeError('Non-finite PQA inference output')

    model.train()
    model.zero_grad()
    features = model.extract_feat(image)
    logits = model.pqa_head([feature.detach() for feature in features])
    gt_box = torch.tensor(
        [[size * 0.5, size * 0.5, size * 0.35, size * 0.10, 0.0]],
        device=device)
    targets, valid = model.pqa_head.build_targets(
        logits, [meta], [gt_box], model.bbox_head.anchor_generator.strides)
    clean_loss, stats = model.pqa_head.ld_loss(
        logits, targets, valid, gamma=model.pqa_ld_gamma,
        loss_weight=model.pqa_ld_loss_weight)
    dark_image = model._build_pqa_dark_view(image, [meta])
    with torch.no_grad():
        dark_features = model.extract_feat(dark_image)
    dark_logits = model.pqa_head([feature.detach() for feature in dark_features])
    dark_loss, _ = model.pqa_head.ld_loss(
        dark_logits, targets, valid, gamma=model.pqa_ld_gamma,
        loss_weight=(model.pqa_ld_loss_weight
                     * model.pqa_dark_supervision_weight))
    consistency = model.pqa_head.consistency_loss(
        logits, dark_logits, targets, valid,
        loss_weight=model.pqa_dark_consistency_weight)
    total = clean_loss + dark_loss + consistency
    if not bool(torch.isfinite(total)):
        raise RuntimeError('Non-finite PQA smoke loss')
    total.backward()

    grad_state = dict(
        pqa=module_has_grad(model.pqa_head),
        backbone=module_has_grad(model.backbone),
        neck=module_has_grad(model.neck),
        bbox_head=module_has_grad(model.bbox_head),
    )
    if not grad_state['pqa']:
        raise RuntimeError('PQA losses did not update PQAHeatmapHead')
    leaked = [name for name in ('backbone', 'neck', 'bbox_head')
              if grad_state[name]]
    if leaked:
        raise RuntimeError('PQA gradient leaked into: ' + ', '.join(leaked))

    print('[inference] detection_shape={} pqa_score={:.6f}'.format(
        detections.shape, float(detections[0, 5])))
    print('[pqa_loss] clean={:.6f} dark={:.6f} consistency={:.6f} '
          'positive={} target_mean={:.6f} pred_positive_mean={:.6f}'.format(
              float(clean_loss.detach()), float(dark_loss.detach()),
              float(consistency.detach()), int(stats['pqa_positive'].item()),
              float(stats['pqa_target_mean']),
              float(stats['pqa_pred_positive_mean'])))
    print(f'[gradients] {grad_state}')
    print('PQA QUALITY-PRIMARY SMOKE PASS')


if __name__ == '__main__':
    main()
