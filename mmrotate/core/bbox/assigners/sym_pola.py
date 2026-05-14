# mmrotate/core/bbox/assigners/sym_pola.py
import torch
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmrotate.core.bbox.builder import ROTATED_BBOX_ASSIGNERS

# 导入路径修正：从 core 跨目录调用 models/losses 中的算子

@ROTATED_BBOX_ASSIGNERS.register_module(force=True)
class SymPOLAAssigner(BaseAssigner):
    def __init__(self,
                 cost_class: float = 1.0,
                 cost_reg: float = 1.0,
                 o2m: bool = False,
                 tau_init: float = 10.0,
                 tau_min: float = 1.0,
                 warmup_iters: int = 2000,
                 eps: float = 1e-6,
                 topk: int = 6,
                 o2m_warmup_iters: int = 0,
                 o2m_topk: int = 9):
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
        
        # [O2M 冷启动] 训练前 o2m_warmup_iters 步使用 O2M 分配，提供充足正样本
        self.o2m_warmup_iters = o2m_warmup_iters
        self.o2m_topk = o2m_topk
        
        # [降维改造] 0.x 无法获取全局 iter，使用内部计数器近似模拟
        # 注意：assign() 是逐图像调用的，若 batch_size=2，则每 2 次调用等于 1 个 iter
        self._local_call_count = 0 

    def _get_current_tau(self, is_training):
        """动态感知训练进度并调度温度曲率"""
        if not is_training:
            return self.tau_min
            
        # 假设单卡 batch_size=2，将调用次数折算为迭代步数
        current_iter = self._local_call_count // 2 
        
        if current_iter >= self.warmup_iters:
            return self.tau_min
            
        decay_ratio = current_iter / max(1, self.warmup_iters)
        return self.tau_init - decay_ratio * (self.tau_init - self.tau_min)

    @torch.no_grad()
    def assign(self, pred_logits, pred_bboxes, gt_labels, gt_bboxes, img_metas=None, is_training=True):
        # 每次分配时计数器累加
        self._local_call_count += 1
        
        INF = 100000000
        num_gt, num_bboxes = gt_bboxes.shape[0], pred_bboxes.shape[0]

        assigned_gt_inds = pred_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        assigned_labels = pred_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            min_cost = pred_bboxes.new_ones((num_bboxes,)) * INF
            if gt_labels is None:
                assigned_labels = None
            return AssignResult(num_gt, assigned_gt_inds, min_cost, labels=assigned_labels)

        out_prob = pred_logits.sigmoid()
        pos_cost_class = self.focal_loss_alpha * ((1 - out_prob) ** self.focal_loss_gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, gt_labels]

        from mmrotate.models.losses.sym_kld_calculator import sym_kld
        
        raw_cost_reg = sym_kld(pred_bboxes[:, None, :].expand(-1, num_gt, -1),
                               gt_bboxes[None, :, :].expand(num_bboxes, -1, -1),
                               eps=self.eps)
        # NaN/Inf 保护：用大值填充异常，不影响 argmin 排序
        raw_cost_reg = torch.nan_to_num(
            raw_cost_reg, nan=1e6, posinf=1e6, neginf=0.0)
        current_tau = self._get_current_tau(is_training)
        cost_reg = raw_cost_reg / (current_tau + self.eps)

        C = self.cost_class * cost_class + self.cost_reg * cost_reg

        # [O2M 冷启动切换] 训练前期使用 O2M 提供充足正样本，避免分类分支退化
        current_iter = self._local_call_count // 2
        use_o2m = self.o2m or (is_training and self.o2m_warmup_iters > 0 
                               and current_iter < self.o2m_warmup_iters)

        if not use_o2m:
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

        effective_topk = self.o2m_topk if (is_training and self.o2m_warmup_iters > 0 
                                           and current_iter < self.o2m_warmup_iters) else self.topk
        mincost, src_ind = torch.topk(C, k=min(effective_topk, C.shape[0]), dim=0, largest=False)
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