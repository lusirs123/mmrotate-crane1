
import torch
from mmrotate.models.builder import ROTATED_HEADS
from mmrotate.models.dense_heads.rotated_retina_head import RotatedRetinaHead

@ROTATED_HEADS.register_module(force=True)
class SymEOODHead(RotatedRetinaHead):
    """
    Symmetric EOOD Head.
    拦截单层前向传播，将解码后的物理包围框强行注入 SymNFLLoss。
    """
    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        # 1. 维度展平对齐
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        anchors = anchors.reshape(-1, 5)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)

        # 2. 物理框解码
        decoded_pred_bboxes = self.bbox_coder.decode(anchors, bbox_pred)
        decoded_gt_bboxes = self.bbox_coder.decode(anchors, bbox_targets)

        # 3. 提取有效的 Ground Truth 框 (去除背景产生的占位符)
        pos_inds = (labels >= 0) & (labels < self.num_classes)
        pos_gt_bboxes = decoded_gt_bboxes[pos_inds]

        # 4. 计算 Sym-NFL 分类损失 (劫持并注入几何参数)
        loss_cls = self.loss_cls(
            cls_score, 
            labels, 
            pred_bboxes=decoded_pred_bboxes, # <-- 注入！
            gt_bboxes=pos_gt_bboxes,         # <-- 注入！
            weight=label_weights, 
            avg_factor=num_total_samples
        )

        # 5. 计算 Sym-KLD 回归损失
        if bbox_weights.sum() > 0:
            loss_bbox = self.loss_bbox(
                decoded_pred_bboxes,
                decoded_gt_bboxes,
                weight=bbox_weights,
                avg_factor=num_total_samples
            )
        else:
            loss_bbox = bbox_pred.sum() * 0

        return loss_cls, loss_bbox
