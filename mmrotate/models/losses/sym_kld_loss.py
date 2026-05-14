# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet.models.losses.utils import weight_reduce_loss
from mmrotate.models.builder import ROTATED_LOSSES

# 显式引入底层度量算子
from .sym_kld_calculator import sym_kld

@ROTATED_LOSSES.register_module(force=True)
class SymKLDLoss(nn.Module):
    """Symmetric Kullback-Leibler Divergence Loss for OBB Regression.
    
    This loss strictly penalizes angular deviations and aspect ratio mismatches 
    by mapping oriented bounding boxes to 2D Gaussian distributions.
    """

    def __init__(self, 
                 eps: float = 1e-6, 
                 reduction: str = 'mean', 
                 loss_weight: float = 1.0):
        super(SymKLDLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # 屏蔽无效的前向传播以节约显存，避免触发分布式训练的 SyncBN 假死
        if pred.size(0) == 0:
            return pred.sum() * 0.0

        # 1. 调用底层度量算子，得到原始对称 KLD 距离
        raw_kld = sym_kld(pred, target, eps=self.eps)

        # 2. 对原始距离做有界化压缩，避免极端错位样本把梯度拉爆
        raw_kld = torch.nan_to_num(raw_kld, nan=1e4, posinf=1e4, neginf=0.0)
        # 使用 sqrt(1 + x) - 1 替代 log1p：梯度衰减更平滑，大偏差时梯度 ~ 1/(2*sqrt(x))
        # 相比 log1p 的 1/(1+x)，对中等偏差（10~100）保留更多梯度信号，
        # 同时对极端偏差（>1000）压制更强
        loss = torch.sqrt(1.0 + raw_kld) - 1.0
        # 上界保护
        loss = loss.clamp(max=10.0)

        # 3. 权重张量降维对齐
        # 回归权重的原始 shape 通常为 [N, 5]。取第一列作为代表权重，避免均值平滑掉正样本的极值
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight[:, 0]

        # 4. 执行梯度聚合与平滑降维
        loss = weight_reduce_loss(
            loss, weight, reduction, avg_factor)

        return loss * self.loss_weight