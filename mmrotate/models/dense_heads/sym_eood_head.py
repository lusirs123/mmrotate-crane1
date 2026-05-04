# mmrotate/models/dense_heads/sym_eood_head.py
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

        # 2. 物理框解码 (核心：SymNFLLoss 需要物理流形而非 Delta 偏差)
        decoded_pred_bboxes = self.bbox_coder.decode(anchors, bbox_pred)
        decoded_gt_bboxes = self.bbox_coder.decode(anchors, bbox_targets)

        # 3. 提取有效的 Ground Truth 框
        pos_inds = (labels >= 0) & (labels < self.num_classes)
        pos_gt_bboxes = decoded_gt_bboxes[pos_inds]
        
        # 安全截断：如果当前层没有任何正样本，传入 dummy 以防止 NaN 传播
        if pos_gt_bboxes.numel() == 0:
            pos_gt_bboxes = decoded_gt_bboxes.new_zeros((0, 5))

        # 4. 计算 Sym-NFL 分类损失 (劫持并注入几何参数)
        loss_cls = self.loss_cls(
            cls_score, 
            labels, 
            pred_bboxes=decoded_pred_bboxes,
            gt_bboxes=pos_gt_bboxes,         
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
    

    
    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=False): # 强制阻断外部 NMS 信号
        """
        重写底层后处理算子，物理切断 NMS 调用链。
        基于 Sym-NFL 的单峰特性，直接执行端到端 Top-K 提取。
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        
        mlvl_bboxes = []
        mlvl_scores = []

        # 1. 遍历特征金字塔 (FPN) 各层进行特征解算
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
                
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            
            # 2. 过滤极低置信度噪声（极大地缩减排序开销）
            if cfg.get('score_thr', 0.0) > 0:
                max_scores, _ = scores.max(dim=1)
                valid_mask = max_scores > cfg.score_thr
                scores = scores[valid_mask]
                bbox_pred = bbox_pred[valid_mask]
                anchors = anchors[valid_mask]

            if scores.numel() == 0:
                continue
                
            # 3. 物理框解码
            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        # 边缘情况防御：全图无有效目标
        if not mlvl_bboxes:
            return torch.zeros((0, 6), device=cls_score_list[0].device), \
                   torch.zeros((0,), dtype=torch.long, device=cls_score_list[0].device)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)

        # 4. 展平多类别分数并锁定最高置信度节点
        scores, labels = mlvl_scores.max(dim=1) 
        
        # =========================================================
        # 5. [核心架构变动] 绝对拓扑隔离，禁止调用 multiclass_nms_rotated
        # =========================================================
        max_per_img = cfg.get('max_per_img', 1) # 抓斗场景单一目标，通常限制为 1
        
        if scores.numel() > max_per_img:
            # 仅执行数学上的 argsort/topk，复杂度 O(N log K)，远低于 NMS 的 O(N^2)
            _, topk_inds = scores.topk(max_per_img)
            bboxes = mlvl_bboxes[topk_inds]
            scores = scores[topk_inds]
            labels = labels[topk_inds]
        else:
            bboxes = mlvl_bboxes

        # 6. 还原尺度映射
        if rescale and bboxes.size(0) > 0:
            bboxes[:, :4] /= scale_factor

        # 7. 组装 0.x 标准输出张量 [N, 6] (x, y, w, h, theta, score)
        det_bboxes = torch.cat([bboxes, scores[:, None]], dim=1)
        
        return det_bboxes, labels