

# 动态 NFL Loss 负样本重新加权损失函数

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models.builder import ROTATED_LOSSES
from mmdet.models.losses.utils import weight_reduce_loss
from mmengine.logging import MessageHub

# 严格复用底层的纯算子，确保分配器与惩罚器使用绝对一致的数学度量
from .sym_kld_calculator import sym_kld

@ROTATED_LOSSES.register_module(force=True)
class SymNFLLoss(nn.Module):
    """Symmetric KLD Normalized Focal Loss.
    
    Dynamically re-weights the focal loss based on the Symmetric KLD 
    between predicted OBBs and ground truth to suppress geometrically flawed negative samples.
    """

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 tau_init=10.0,    
                 tau_min=1.0,      
                 warmup_iters=2000,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(SymNFLLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss is supported.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.warmup_iters = warmup_iters
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def _get_current_tau(self):
        """同步时间流形感知：动态计算温度系数"""
        try:
            message_hub = MessageHub.get_current_instance()
            current_iter = message_hub.get_info('iter')
            
            if current_iter >= self.warmup_iters:
                return self.tau_min
            
            decay_ratio = current_iter / max(1, self.warmup_iters)
            return self.tau_init - decay_ratio * (self.tau_init - self.tau_min)
        except Exception:
            return self.tau_min

    def forward(self,
                pred_logits,
                labels,
                pred_bboxes,       # 强制依赖：必须传入解码后的预测框
                gt_bboxes,         # 强制依赖：必须传入真实框
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """
        Args:
            pred_logits (Tensor): 分类预测 logits, shape [N, num_classes].
            labels (Tensor): 真实标签, shape [N]. 背景类标签通常为 num_classes.
            pred_bboxes (Tensor): 预测旋转框, shape [N, 5].
            gt_bboxes (Tensor): 真实旋转框, shape [M, 5].
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        if pred_logits.numel() == 0:
            return pred_logits.sum() * 0.0

        num_classes = pred_logits.size(1)
        
        # 1. 提取度量同构重加权因子 (mu_sym)
        if gt_bboxes is not None and gt_bboxes.size(0) > 0:
            # 获取全连接距离矩阵 [N, M]
            raw_kld = sym_kld(pred_bboxes[:, None, :], gt_bboxes[None, :, :], eps=self.eps)
            # 获取每个预测框到最近 GT 的距离极小值 [N]
            min_kld, _ = raw_kld.min(dim=1)
            
            current_tau = self._get_current_tau()
            # 拓扑压制乘子，值域 (1, 2]
            mu_sym = 1.0 + torch.exp(-min_kld / current_tau) 
        else:
            mu_sym = torch.ones_like(pred_logits[:, 0])

        # 2. 基础 Focal Loss
        pred_sigmoid = pred_logits.sigmoid()
        target = F.one_hot(labels, num_classes=num_classes + 1)[:, :-1].float()

        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        
        loss = F.binary_cross_entropy_with_logits(
            pred_logits, target, reduction='none') * focal_weight

        # 3. 注入 Sym-NFL 拓扑压制
        loss = loss * mu_sym.unsqueeze(-1)

        # 4. 降维对齐
        if weight is not None:
            if weight.shape != loss.shape:
                if weight.size(0) == loss.size(0):
                    weight = weight.view(-1, 1)
                else:
                    assert weight.numel() == loss.numel()
                    weight = weight.view(loss.size(0), -1)

        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        
        return loss * self.loss_weight
