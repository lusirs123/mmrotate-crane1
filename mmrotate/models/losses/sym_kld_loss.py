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

        # 1. 调用底层度量算子，计算纯代数张量图
        loss = sym_kld(pred, target, eps=self.eps)

        # 2. 权重张量降维对齐
        # 回归权重的原始 shape 通常为 [N, 5]。取第一列作为代表权重，避免均值平滑掉正样本的极值
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight[:, 0]

        # 3. 执行梯度聚合与平滑降维
        loss = weight_reduce_loss(
            loss, weight, reduction, avg_factor)

        return loss * self.loss_weight