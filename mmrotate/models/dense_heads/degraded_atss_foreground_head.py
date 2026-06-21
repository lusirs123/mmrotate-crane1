# Copyright (c) OpenMMLab. All rights reserved.
# degraded_atss_foreground_head.py
#
# 退化视图前景鲁棒辅助头。
# 设计目标:
#   - 使用 ATSS 一对多分配提供密集前景/objectness 监督;
#   - 只计算分类损失, 不计算 OBB/角度回归;
#   - 由 detector 在训练期把 degraded view 的 FPN 特征送入该头;
#   - 推理期不参与, 零额外开销。

import torch

from mmrotate.models.builder import ROTATED_HEADS
from .rotated_atss_head import RotatedATSSHead


@ROTATED_HEADS.register_module(force=True)
class DegradedATSSForegroundHead(RotatedATSSHead):
    """ATSS cls-only head for degraded-view foreground robustness.

    该头继承 RotatedATSSHead 的 anchor、ATSS assigner 和 forward_train 流程,
    但覆写 loss_single, 只保留分类/objectness 损失。bbox_pred 仍由父类
    forward 产生以保持接口兼容, 但回归 loss 恒为 0, 不向角度/OBB 回归传梯度。
    """

    def __init__(self,
                 *args,
                 use_degraded_view=True,
                 degraded_aux2_amp_loss_weight=0.0,
                 degraded_aux2_amp_levels=(0, 1, 2),
                 **kwargs):
        super(DegradedATSSForegroundHead, self).__init__(*args, **kwargs)
        # detector 根据该标记决定把 clean feats 还是 degraded feats 送入辅助头。
        self.use_degraded_view = use_degraded_view
        # 幅度一致性由 detector 计算, 这里仅保存配置。
        self.degraded_aux2_amp_loss_weight = degraded_aux2_amp_loss_weight
        self.degraded_aux2_amp_levels = degraded_aux2_amp_levels

    def loss_single(self, cls_score, bbox_pred, anchors, labels,
                    label_weights, bbox_targets, bbox_weights,
                    num_total_samples):
        """只计算 ATSS 分类/objectness 损失.

        bbox/angle 不作为 aux2 监督目标, 防止低质量退化视图污染主几何分支。
        """
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)

        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        if torch.isnan(loss_cls) or torch.isinf(loss_cls):
            loss_cls = cls_score.sum() * 0.0

        # 保持父类 loss dict 键名兼容, 但不产生回归/角度梯度。
        loss_bbox = bbox_pred.sum() * 0.0
        return loss_cls, loss_bbox
