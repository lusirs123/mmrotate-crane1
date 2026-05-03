

# 自定义分配器，定义训练阶段的样本分配规则，
# 负责决定哪些预测框、候选点或 proposal 是正样本，哪些是负样本，哪些要忽略，并生成 AssignResult
# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

from ..builder import ROTATED_BBOX_ASSIGNERS
from ..transforms import obb2xyxy


def _xywha_to_gaussian(boxes):
    cx, cy, w, h, a = boxes.unbind(-1)
    cos_a = torch.cos(a)
    sin_a = torch.sin(a)

    w2 = (w * 0.5) ** 2
    h2 = (h * 0.5) ** 2

    sxx = cos_a * cos_a * w2 + sin_a * sin_a * h2
    sxy = cos_a * sin_a * (w2 - h2)
    syy = sin_a * sin_a * w2 + cos_a * cos_a * h2

    mu = torch.stack([cx, cy], dim=-1)
    sigma = torch.stack([
        torch.stack([sxx, sxy], dim=-1),
        torch.stack([sxy, syy], dim=-1)
    ], dim=-2)
    return mu, sigma


def _inv2x2_safe(sigma, eps=1e-6):
    a = sigma[..., 0, 0]
    b = sigma[..., 0, 1]
    c = sigma[..., 1, 0]
    d = sigma[..., 1, 1]
    det = (a * d - b * c).clamp(min=eps)
    inv = torch.stack([
        torch.stack([d, -b], dim=-1),
        torch.stack([-c, a], dim=-1)
    ], dim=-2) / det[..., None, None]
    return inv


def _sym_kld(boxes_p, boxes_q, eps=1e-6):
    mu_p, sigma_p = _xywha_to_gaussian(boxes_p)
    mu_q, sigma_q = _xywha_to_gaussian(boxes_q)

    inv_p = _inv2x2_safe(sigma_p, eps=eps)
    inv_q = _inv2x2_safe(sigma_q, eps=eps)

    trace_qp = torch.einsum('...ij,...ji->...', inv_q, sigma_p)
    trace_pq = torch.einsum('...ij,...ji->...', inv_p, sigma_q)

    delta = (mu_p - mu_q).unsqueeze(-1)
    maha = (delta.transpose(-1, -2) @ (inv_p + inv_q) @ delta).squeeze(-1).squeeze(-1)

    return 0.5 * (trace_qp + trace_pq - 4.0 + maha)


@ROTATED_BBOX_ASSIGNERS.register_module()
class SymPOLAAssigner(BaseAssigner):
    """POLA assigner with symmetric KLD cost."""

    def __init__(self,
                 cost_class: float = 1.0,
                 cost_reg: float = 1.0,
                 o2m: bool = False,
                 tau: float = 1.0,
                 eps: float = 1e-6,
                 topk: int = 6):
        super().__init__()
        self.cost_class = cost_class
        self.cost_reg = cost_reg
        self.o2m = o2m
        self.tau = tau
        self.eps = eps
        self.topk = topk
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        assert cost_class != 0 or cost_reg != 0, 'all costs cant be 0'

    @torch.no_grad()
    def assign(self, pred_logits, pred_bboxes, gt_labels, gt_bboxes, img_metas):
        INF = 100000000
        num_gt, num_bboxes = gt_bboxes.shape[0], pred_bboxes.shape[0]

        assigned_gt_inds = pred_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        assigned_labels = pred_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            min_cost = pred_bboxes.new_ones((num_bboxes,)) * INF
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            return AssignResult(num_gt, assigned_gt_inds, min_cost, labels=assigned_labels)

        batch_out_prob = pred_logits.sigmoid()
        tgt_ids = gt_labels
        out_prob = batch_out_prob
        out_bbox = pred_bboxes

        alpha = self.focal_loss_alpha
        gamma = self.focal_loss_gamma
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids]

        cost_reg = _sym_kld(out_bbox[:, None, :].expand(-1, num_gt, -1),
                             gt_bboxes[None, :, :].expand(num_bboxes, -1, -1),
                             eps=self.eps)

        cost_reg = 1.0 - torch.exp(-cost_reg / self.tau)

        C = self.cost_class * cost_class + self.cost_reg * cost_reg

        if not self.o2m:
            mincost, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(len(tgt_ids), device=src_ind.device)

            assigned_gt_inds[:] = 0
            assigned_gt_inds[src_ind] = tgt_ind + 1

            if gt_labels is not None:
                assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
                pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
                if pos_inds.numel() > 0:
                    assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
            else:
                assigned_labels = None

            return AssignResult(num_gt, assigned_gt_inds, mincost, labels=assigned_labels)

        mincost, src_ind = torch.topk(C, k=min(self.topk, C.shape[0]), dim=0, largest=False)
        assigned_gt_inds[:] = 0
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





