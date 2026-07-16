#!/usr/bin/env python3
"""One-image interface/gradient smoke test for reg-quality-primary.

This is not an experiment and does not write checkpoints or evaluation
caches.  It verifies that a K1 checkpoint can load with the new quality head,
quality-primary inference returns one OBB, and the isolated quality loss sends
gradients only to RegQualityHead.
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
                'crane_symeood_k1_regquality_primary.py')
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
    # The K1 checkpoint supplies backbone weights; avoid a torchvision fetch.
    if cfg.model.get('backbone', None) is not None:
        cfg.model.backbone.init_cfg = None
    model = build_detector(cfg.model)
    if model.reg_quality_head is None:
        raise RuntimeError('Config did not construct reg_quality_head')
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model = model.cuda(args.gpu)

    size = int(args.image_size)
    image = torch.zeros(1, 3, size, size, device=f'cuda:{args.gpu}')
    meta = dict(
        filename='synthetic_smoke.png',
        ori_filename='synthetic_smoke.png',
        ori_shape=(size, size, 3),
        img_shape=(size, size, 3),
        pad_shape=(size, size, 3),
        scale_factor=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        flip=False,
        flip_direction=None)

    model.eval()
    with torch.no_grad():
        result = model.simple_test(image, [meta], rescale=False)
    if len(result) != 1 or len(result[0]) != model.bbox_head.num_classes:
        raise RuntimeError('Unexpected simple_test result structure')
    detections = result[0][0]
    if detections.shape != (1, 6):
        raise RuntimeError(
            f'Expected one [cx,cy,w,h,a,Q] detection, got {detections.shape}')
    if not np.isfinite(detections).all():
        raise RuntimeError('Non-finite quality-primary inference output')

    model.train()
    model.zero_grad()
    features = model.extract_feat(image)
    main_outs = model.bbox_head(features)
    quality_logits = model.reg_quality_head(
        [feature.detach() for feature in features])
    gt_box = torch.tensor(
        [[size * 0.5, size * 0.5, size * 0.35, size * 0.10, 0.0]],
        device=image.device)
    quality_loss, stats = model._compute_reg_quality_loss(
        quality_logits, main_outs[0], main_outs[1], [meta], [gt_box])
    if not bool(torch.isfinite(quality_loss)):
        raise RuntimeError('Non-finite reg-quality loss')
    quality_loss.backward()

    grad_state = dict(
        quality=module_has_grad(model.reg_quality_head),
        backbone=module_has_grad(model.backbone),
        neck=module_has_grad(model.neck),
        bbox_head=module_has_grad(model.bbox_head),
    )
    if not grad_state['quality']:
        raise RuntimeError('Quality loss did not update RegQualityHead')
    leaked = [name for name in ('backbone', 'neck', 'bbox_head')
              if grad_state[name]]
    if leaked:
        raise RuntimeError(
            'Quality gradient leaked into main model: ' + ', '.join(leaked))

    print('[inference] detection_shape={} quality={:.6f}'.format(
        detections.shape, float(detections[0, 5])))
    print('[quality_loss] value={:.6f} positive={} target_mean={:.6f} '
          'pred_positive_mean={:.6f}'.format(
              float(quality_loss.detach().item()),
              int(stats['reg_quality_positive'].item()),
              float(stats['reg_quality_target_mean'].item()),
              float(stats['reg_quality_pred_positive_mean'].item())))
    print(f'[gradients] {grad_state}')
    print('REG-QUALITY PRIMARY SMOKE PASS')


if __name__ == '__main__':
    main()
