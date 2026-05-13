# mmrotate/models/detectors/sym_eood_detector.py
import copy
import torch.nn as nn
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmrotate.models.builder import ROTATED_DETECTORS, build_head
from mmrotate.models.dense_heads.rotated_atss_head import RotatedATSSHead
from mmdet.core import bbox2result


@ROTATED_DETECTORS.register_module(force=True)
class SymEOOD(SingleStageDetector):
    """
    Symmetric EOOD Detector（MMRotate 0.x 兼容版）
    支持两种辅助头模式（互斥）：
      mode A: aux_bbox_head — Anchor-based 辅助头（如 RotatedATSS）
      mode B: gaussian_head — Anchor-free 高斯热图辅助头
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 aux_bbox_head=None,
                 gaussian_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
            # 如果 detector 级别没有 train_cfg/test_cfg，从 bbox_head 内部提取
        if train_cfg is None and bbox_head is not None:
            train_cfg = bbox_head.get('train_cfg', None)
        if test_cfg is None and bbox_head is not None:
            test_cfg = bbox_head.get('test_cfg', None)
            
        super(SymEOOD, self).__init__(
            backbone, neck, bbox_head,
            train_cfg, test_cfg, pretrained, init_cfg
        )

        # Mode A: Anchor-based 辅助头
        if aux_bbox_head is not None:
            self.aux_heads = nn.ModuleList()
            for head_cfg in aux_bbox_head:
                head_cfg = copy.deepcopy(head_cfg)
                head_cfg.setdefault('train_cfg', train_cfg)
                head_cfg.setdefault('test_cfg', test_cfg)
                self.aux_heads.append(build_head(head_cfg))
        else:
            self.aux_heads = None

        # Mode B: Anchor-free 高斯热图辅助头
        if gaussian_head is not None:
            self.gaussian_head = build_head(gaussian_head)
        else:
            self.gaussian_head = None

    def _build_aux_feats(self, feats, aux_head):
        if isinstance(aux_head, RotatedATSSHead):
            return [(feat, feat) for feat in feats]
        return feats

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """0.x 标准联合训练入口"""
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = dict()

        # 主头损失
        main_losses = self.bbox_head.forward_train(
            x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        losses.update(main_losses)

        # Mode A: Anchor-based 辅助头损失
        if self.aux_heads is not None:
            for i, aux_head in enumerate(self.aux_heads):
                aux_feats = self._build_aux_feats(x, aux_head)
                aux_losses = aux_head.forward_train(
                    aux_feats, img_metas, gt_bboxes, gt_labels,
                    gt_bboxes_ignore)
                for k, v in aux_losses.items():
                    losses['aux{:d}_{:s}'.format(i, k)] = v

        # Mode B: 高斯热图辅助头损失
        if self.gaussian_head is not None:
            heatmap = self.gaussian_head(x[0])  # 仅用 P3
            losses['loss_heatmap'] = self.gaussian_head.loss(
                heatmap, gt_bboxes, img_metas)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """
        0.x 标准推理入口。
        辅助头天然不参与推理，零额外开销。
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results