# mmrotate/models/losses/sym_nfl_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models.builder import ROTATED_LOSSES
from mmdet.models.losses.utils import weight_reduce_loss

# 相对路径直接调用同目录下的算子
from .sym_kld_calculator import sym_kld

@ROTATED_LOSSES.register_module(force=True)
class SymNFLLoss(nn.Module):
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
        
        # [降维改造] 注册不参与梯度的长整型 Buffer，使其随权重一起保存
        self.register_buffer('_local_iter', torch.tensor(0, dtype=torch.long))

    def _get_current_tau(self):
        """同步时间流形感知"""
        current_iter = self._local_iter.item()
        if current_iter >= self.warmup_iters:
            return self.tau_min
        
        decay_ratio = current_iter / max(1, self.warmup_iters)
        return self.tau_init - decay_ratio * (self.tau_init - self.tau_min)

    def forward(self,
                pred_logits,
                labels,
                pred_bboxes,
                gt_bboxes,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        
        # 仅在训练模式下累加迭代步数
        if self.training:
            self._local_iter += 1

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        if pred_logits.numel() == 0:
            return pred_logits.sum() * 0.0

        num_classes = pred_logits.size(1)
        
        if gt_bboxes is not None and gt_bboxes.size(0) > 0:
            raw_kld = sym_kld(pred_bboxes[:, None, :], gt_bboxes[None, :, :], eps=self.eps)
            min_kld, _ = raw_kld.min(dim=1)
            
            current_tau = self._get_current_tau()
            mu_sym = 1.0 + torch.exp(-min_kld / current_tau) 
        else:
            mu_sym = torch.ones_like(pred_logits[:, 0])

        pred_sigmoid = pred_logits.sigmoid()
        target = F.one_hot(labels, num_classes=num_classes + 1)[:, :-1].float()

        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        
        loss = F.binary_cross_entropy_with_logits(
            pred_logits, target, reduction='none') * focal_weight

        loss = loss * mu_sym.unsqueeze(-1)

        if weight is not None:
            if weight.shape != loss.shape:
                if weight.size(0) == loss.size(0):
                    weight = weight.view(-1, 1)
                else:
                    assert weight.numel() == loss.numel()
                    weight = weight.view(loss.size(0), -1)

        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        
        return loss * self.loss_weight