#!/usr/bin/env python3
"""
gradient_match.py — 1-step 梯度幅度匹配, 确定 L_invar 的 λ₂ 初值.

按 λ₂ × ‖grad L_invar‖ ≈ 0.3 × λ₁ × ‖grad L_equi‖ 定 λ₂.

Run:
    cd /Users/mac/Documents/paper/symEOOD
    PYTHONPATH=. /opt/anaconda3/envs/mmrot/bin/python3 \
        crane_project/tools/gradient_match.py
"""

import sys
import torch

from mmcv import Config
from mmrotate.models import build_detector


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else \
        'crane_project/configs/crane_symeood_m2_equi_invar.py'

    cfg = Config.fromfile(cfg_path)
    model = build_detector(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    device = next(model.parameters()).device
    print(f'Model device: {device}')

    # 构造假输入 (单 batch, 不走 dataloader)
    B = 2
    img = torch.randn(B, 3, 1024, 1024, device=device)
    img_metas = [
        dict(img_shape=(1024, 1024, 3), pad_shape=(1024, 1024, 3),
             img_norm_cfg=dict(mean=(123.675, 116.28, 103.53),
                               std=(58.395, 57.12, 57.375), to_rgb=True))
        for _ in range(B)
    ]
    gt_bboxes = [
        torch.tensor([[200., 200., 100., 50., 0.3]], device=device),
        torch.tensor([[300., 300., 80., 40., -0.2]], device=device),
    ]
    gt_labels = [
        torch.tensor([0], device=device),
        torch.tensor([0], device=device),
    ]

    # Forward + loss (不 backward, 只看 loss 值)
    losses = model.forward_train(img, img_metas, gt_bboxes, gt_labels)

    def _sum_loss(key):
        v = losses.get(key, [])
        items = v if isinstance(v, list) else [v]
        return sum(x.item() for x in items if isinstance(x, torch.Tensor) and x.numel() > 0)

    # losses['loss_equi/invar'] 已经乘过 λ，需要除回去得到裸值
    lambda1 = cfg.model.bbox_head.get('equi_loss_weight', 0.2)
    lambda2_current = cfg.model.bbox_head.get('invar_loss_weight', 0.05)

    equi_weighted = _sum_loss('loss_equi')    # = L_equi_raw × λ₁
    invar_weighted = _sum_loss('loss_invar')  # = L_invar_raw × λ₂_current

    equi_raw = equi_weighted / lambda1 if lambda1 > 0 else 0.0
    invar_raw = invar_weighted / lambda2_current if lambda2_current > 0 else 0.0

    l_det = _sum_loss('loss_cls') + _sum_loss('loss_bbox')

    print('=' * 60)
    print('Gradient Match — 1-step λ₂ estimation')
    print('=' * 60)
    print(f'  L_det (cls+bbox):              {l_det:.4f}')
    print(f'  L_equi  raw (÷λ₁={lambda1}):  {equi_raw:.6f}  (weighted={equi_weighted:.6f})')
    print(f'  L_invar raw (÷λ₂={lambda2_current}): {invar_raw:.6f}  (weighted={invar_weighted:.6f})')

    if invar_raw > 0:
        lambda2 = 0.3 * lambda1 * equi_raw / invar_raw
        lambda2 = round(lambda2, 4)
        print(f'\n  Suggested λ₂ = {lambda2}')
        print(f'  (target: λ₂ × L_invar_raw ≈ 0.3 × λ₁ × L_equi_raw)')
        print(f'  Verify:  {lambda2:.4f} × {invar_raw:.4f} = {lambda2 * invar_raw:.6f}')
        print(f'           0.3 × {lambda1} × {equi_raw:.4f} = {0.3 * lambda1 * equi_raw:.6f}')
    else:
        print('\n  L_invar = 0 — check wiring or increase on_prob / perturbation ranges')
        print(f'  Default λ₂ = {lambda2_current}')

    print('\n' + '=' * 60)


if __name__ == '__main__':
    main()