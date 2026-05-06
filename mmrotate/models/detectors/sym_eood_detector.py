import copy
import torch.nn as nn
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmrotate.models.builder import ROTATED_DETECTORS, build_head
from mmdet.core import bbox2result

@ROTATED_DETECTORS.register_module(force=True)
class SymEOOD(SingleStageDetector):
    """
    Symmetric EOOD Detector with Topological Isolation. (MMRotate 0.x Compatible)
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 aux_bbox_head=None,  # 辅头挂载点
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        # 彻底移除 1.x 的 data_preprocessor
        super().__init__(backbone, neck, bbox_head, train_cfg, 
                         test_cfg, pretrained, init_cfg)

        if aux_bbox_head is not None:
            self.aux_heads = nn.ModuleList()
            for head_cfg in aux_bbox_head:
                # 允许辅头保留自己的 train/test 配置；未显式提供时才回退到全局配置。
                head_cfg = copy.deepcopy(head_cfg)
                head_cfg.setdefault('train_cfg', train_cfg)
                head_cfg.setdefault('test_cfg', test_cfg)
                self.aux_heads.append(build_head(head_cfg))
        else:
            self.aux_heads = None

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """0.x 标准的联合训练流入口"""
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = dict()

        # [主头] 前向与 Loss 计算
        main_losses = self.bbox_head.forward_train(
            x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        losses.update(main_losses)

        # [辅头] 密集监督计算
        if self.aux_heads is not None:
            for i, aux_head in enumerate(self.aux_heads):
                aux_losses = aux_head.forward_train(
                    x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
                
                # 前缀隔离，防止 Loss 键值冲突引发字典覆盖
                for k, v in aux_losses.items():
                    losses[f'aux{i}_{k}'] = v

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """
        0.x 标准的推理流入口。
        物理切断辅头：此方法天然不调用 self.aux_heads，实现零开销推理。
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        
        # 0.x 要求按类别转换为 list 格式的 DOTA 标准输出
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results