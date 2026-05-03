

# 自定义分配器，定义训练阶段的样本分配规则，
# 负责决定哪些预测框、候选点或 proposal 是正样本，哪些是负样本，哪些要忽略，并生成 AssignResult
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmengine.logging import MessageHub
from mmrotate.core.bbox.builder import ROTATED_BBOX_ASSIGNERS
from .sym_kld_calculator import sym_kld


@ROTATED_BBOX_ASSIGNERS.register_module(force=True)
class SymPOLAAssigner(BaseAssigner):
    """POLA assigner with symmetric KLD cost and dynamic temperature constraints."""

    def __init__(self,
                 cost_class: float = 1.0,
                 cost_reg: float = 1.0,
                 o2m: bool = False,
                 tau_init: float = 10.0,   # [新增] 预热期高温系数，强力压缩几何代价
                 tau_min: float = 1.0,     # [新增] 衰减后的基准温度
                 warmup_iters: int = 2000, # [新增] 线性衰减的迭代步数阈值
                 eps: float = 1e-6,
                 topk: int = 6):
        super().__init__()
        self.cost_class = cost_class
        self.cost_reg = cost_reg
        self.o2m = o2m
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.warmup_iters = warmup_iters
        self.eps = eps
        self.topk = topk
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        assert cost_class != 0 or cost_reg != 0, 'all costs cant be 0'

    def _get_current_tau(self):
        """动态感知训练进度并调度温度曲率"""
        try:
            # 挂载 MMEngine 的全局通讯枢纽获取实时迭代步数
            message_hub = MessageHub.get_current_instance()
            current_iter = message_hub.get_info('iter')
            
            if current_iter >= self.warmup_iters:
                return self.tau_min
            
            # 执行线性退火
            decay_ratio = current_iter / max(1, self.warmup_iters)
            return self.tau_init - decay_ratio * (self.tau_init - self.tau_min)
        except Exception:
            # 安全回退：模型评估（Eval）阶段或离线验证时，直接使用最小基准温度
            return self.tau_min

    @torch.no_grad()
    def assign(self, pred_logits, pred_bboxes, gt_labels, gt_bboxes, img_metas=None):
        INF = 100000000
        num_gt, num_bboxes = gt_bboxes.shape[0], pred_bboxes.shape[0]

        assigned_gt_inds = pred_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        assigned_labels = pred_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            min_cost = pred_bboxes.new_ones((num_bboxes,)) * INF
            if gt_labels is None:
                assigned_labels = None
            return AssignResult(num_gt, assigned_gt_inds, min_cost, labels=assigned_labels)

        # 1. 计算分类代价矩阵
        out_prob = pred_logits.sigmoid()
        pos_cost_class = self.focal_loss_alpha * ((1 - out_prob) ** self.focal_loss_gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, gt_labels]  # [num_bboxes, num_gt]

        # 2. 计算各向异性 KLD 代价矩阵
        raw_cost_reg = sym_kld(pred_bboxes[:, None, :].expand(-1, num_gt, -1),
                       gt_bboxes[None, :, :].expand(num_bboxes, -1, -1),
                                eps=self.eps)

        # 3. 动态温度平滑压缩
        current_tau = self._get_current_tau()
        cost_reg = 1.0 - torch.exp(-raw_cost_reg / current_tau)

        # 4. 联合代价组装
        C = self.cost_class * cost_class + self.cost_reg * cost_reg

        # 5. 极值匹配拓扑
        if not self.o2m:
            mincost, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(len(gt_labels), device=src_ind.device)

            assigned_gt_inds[src_ind] = tgt_ind + 1

            if gt_labels is not None:
                assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
                pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
                if pos_inds.numel() > 0:
                    assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
            else:
                assigned_labels = None

            return AssignResult(num_gt, assigned_gt_inds, mincost, labels=assigned_labels)

        # 辅助头的 Top-K 回退匹配保留原逻辑
        mincost, src_ind = torch.topk(C, k=min(self.topk, C.shape[0]), dim=0, largest=False)
        for i, ind in enumerate(src_ind.transpose(0, 1)):
            assigned_gt_inds[ind] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)
    


    
