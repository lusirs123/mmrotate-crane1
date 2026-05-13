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
    """
    Symmetric Normalized Focal Loss with Spatial Exclusion Penalty.
    
    【工程优化记录】
    1. O(K) FLOPs 降维：引入 L2 中心点掩码，截断 99% 的远端冗余背景计算。
    2. Detached 阻断：寻址阶段彻底切断隐式计算图，防显存泄漏。
    3. Chunking 保底：对过滤后的 Top-K 候选依然执行分块，护航极限显存。
    4. 原生指数核回归：适配 1:2 刚体顶梁曲率，提供锐利排他惩罚。
    """
    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 tau_init=10.0,    
                 tau_min=1.0,      
                 warmup_iters=2000,
                 spatial_topk=300,  # [新增] 空间掩码物理爆炸半径
                 loss_call_factor=5,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 kld_chunk_size=1024):
        super(SymNFLLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss is supported.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.warmup_iters = warmup_iters
        self.spatial_topk = spatial_topk
        self.loss_call_factor = max(1, int(loss_call_factor))
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.kld_chunk_size = kld_chunk_size
        
        # [降维改造] 注册不参与梯度的长整型 Buffer，使其随权重一起保存
        self.register_buffer('_local_iter', torch.tensor(0, dtype=torch.long))

    def _get_current_tau(self):
        """同步时间流形感知"""
        # multi_apply 会让同一个 optimizer step 下的 loss.forward 被重复调用
        # 这里按特征层数折算为“真实迭代步”，避免 warmup 提前 5 倍结束。
        current_iter = self._local_iter.item() // self.loss_call_factor
        if current_iter >= self.warmup_iters:
            return self.tau_min
        
        decay_ratio = current_iter / max(1, self.warmup_iters)
        return self.tau_init - decay_ratio * (self.tau_init - self.tau_min)

    def _min_sym_kld(self, pred_bboxes, gt_bboxes):
        """Chunked minimum symmetric KLD over all GT boxes.
        
        即便在 Spatial Mask 之后，保留此机制可确保极端 Batch/TopK 下的绝对安全。
        """
        if gt_bboxes is None or gt_bboxes.size(0) == 0:
            return None

        chunk_size = self.kld_chunk_size
        if chunk_size is None or chunk_size <= 0 or pred_bboxes.size(0) <= chunk_size:
            raw_kld = sym_kld(
                pred_bboxes[:, None, :],
                gt_bboxes[None, :, :],
                eps=self.eps)
            min_kld, _ = raw_kld.min(dim=1)
            return min_kld

        min_kld_chunks = []
        for start in range(0, pred_bboxes.size(0), chunk_size):
            end = min(start + chunk_size, pred_bboxes.size(0))
            raw_kld = sym_kld(
                pred_bboxes[start:end, None, :],
                gt_bboxes[None, :, :],
                eps=self.eps)
            chunk_min_kld, _ = raw_kld.min(dim=1)
            min_kld_chunks.append(chunk_min_kld)

        return torch.cat(min_kld_chunks, dim=0)

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

        num_samples = pred_logits.size(0)
        num_classes = pred_logits.size(1)
        
        # 1. 默认底盘初始化：所有 Anchor 的乘子为 1.0 (等价于普通 Focal Loss)
        mu_sym = pred_logits.new_ones(num_samples)
        
        if gt_bboxes is not None and gt_bboxes.size(0) > 0:
            current_tau = self._get_current_tau()

            # ==============================================================
            # 【核心架构优化】: Detached 空间掩码，斩断 O(N^2) 算力黑洞
            # ==============================================================
            # 剥离隐式计算图，仅做纯粹的数学寻址
            pred_centers = pred_bboxes[:, :2].detach()  # [N, 2]
            gt_centers = gt_bboxes[:, :2].detach()      # [M, 2]

            # 计算欧氏距离的平方 [N, M]
            center_dist_sq = ((pred_centers[:, None, :] -
                               gt_centers[None, :, :]) ** 2).sum(-1)
            # 取每个 Anchor 到任意 GT 的最小距离 [N]
            min_dist_sq, _ = center_dist_sq.min(dim=1)

            # 选出距离最近的 Top-K 索引
            k = min(self.spatial_topk, num_samples)
            _, topk_inds = min_dist_sq.topk(k, largest=False)

            # ==============================================================
            # 【有梯度微雕】: 仅对 Top-K 执行昂贵的协方差张量计算
            # ==============================================================
            # 重新切片带有 grad_fn 的原始张量，确保 Sym-KLD 梯度回传
            topk_pred = pred_bboxes[topk_inds]  # [K, 5]
            
            # 走 Chunking 安全通道计算 KLD
            min_kld = self._min_sym_kld(topk_pred, gt_bboxes)  # [K]

            # 应用指数核 (适配 1:2 顶梁场景)
            topk_mu = 1.0 + torch.exp(-min_kld / current_tau)
            
            # 原位写回
            mu_sym[topk_inds] = topk_mu

        # ==============================================================
        # 【基础惩罚层】: 标准 Focal Loss 与重加权融合
        # ==============================================================
        pred_sigmoid = pred_logits.sigmoid()
        target = F.one_hot(labels, num_classes=num_classes + 1)[:, :-1].float()

        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        
        loss = F.binary_cross_entropy_with_logits(
            pred_logits, target, reduction='none') * focal_weight

        # 空间排他性压制注入
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