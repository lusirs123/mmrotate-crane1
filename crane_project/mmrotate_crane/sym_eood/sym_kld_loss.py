
# 自定义对称 KLD 损失函数


# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet.models.losses.utils import weight_reduce_loss
from mmrotate.models.builder import ROTATED_LOSSES

# 显式引入底层度量算子，确保分配与优化共享同一套数学标尺
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
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted OBBs, shape (N, 5) [x, y, w, h, theta].
            target (torch.Tensor): Ground truth OBBs, shape (N, 5).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # 屏蔽无效的前向传播以节约显存
        if pred.size(0) == 0:
            return pred.sum() * 0.0

        # 1. 调用底层度量算子，计算纯代数张量图
        # 注意：这里的 pred 带有 requires_grad=True，sym_kld 必须使用纯 PyTorch 算子
        loss = sym_kld(pred, target, eps=self.eps)

        # 2. 权重张量降维对齐
        # 回归权重的原始 shape 通常为 [N, 5]，而 sym_kld 导出的距离流形为一维 [N]
        # 必须提取多维权重的均值或单轴分量，防止 weight_reduce_loss 触发维度广播错误
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(dim=-1)

        # 3. 执行梯度聚合与平滑降维
        loss = weight_reduce_loss(
            loss, weight, reduction, avg_factor)

        return loss * self.loss_weight
