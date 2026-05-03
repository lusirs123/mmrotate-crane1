
# SymEOOD 拓扑隔离检测器，实现推理流形（predict 方法）和训练流形（loss 方法）分离
import torch.nn as nn
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmrotate.models.builder import ROTATED_DETECTORS, build_head

@ROTATED_DETECTORS.register_module(force=True)
class SymEOOD(SingleStageDetector):
    """
    Symmetric EOOD Detector with Topological Isolation.
    
    Training Phase: Jointly optimizes the Main Head (for high-precision, NMS-free alignment) 
                    and Auxiliary Heads (for dense feature stabilization).
    Inference Phase: Physically severs the Auxiliary Heads from the computational graph,
                     guaranteeing pure execution with zero computational overhead.
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 aux_bbox_head=None,  # 开放辅头挂载接口
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        # 初始化标准的单阶段检测器基座
        super().__init__(backbone, neck, bbox_head, train_cfg, 
                         test_cfg, data_preprocessor, init_cfg)

        # 训练期：动态挂载辅助训练路径
        if aux_bbox_head is not None:
            self.aux_heads = nn.ModuleList()
            for head_cfg in aux_bbox_head:
                self.aux_heads.append(build_head(head_cfg))
        else:
            self.aux_heads = None

    def loss(self, batch_inputs, batch_data_samples):
        """联合训练损失计算流 (前向传播 + 梯度生成)"""
        # 提取 FPN 共享特征
        x = self.extract_feat(batch_inputs)
        losses = dict()

        # [Main Head] 主头计算 (SymEOODHead 内部已封装好 SymKLD 与 SymNFL 的截获)
        main_losses = self.bbox_head.loss(x, batch_data_samples)
        losses.update(main_losses)

        # [Aux Head] 辅头密集监督计算 (如 ATSS)
        if self.aux_heads is not None:
            for i, aux_head in enumerate(self.aux_heads):
                aux_losses = aux_head.loss(x, batch_data_samples)
                
                # 为辅头 Loss 加上前缀以防字典键冲突，例如 'aux0_loss_cls'
                for k, v in aux_losses.items():
                    losses[f'aux{i}_{k}'] = v

        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        """
        核心拓扑隔离：部署推理时，计算图完全无视 self.aux_heads。
        这不仅免除了 NMS 的耗时，还彻底免除了辅头推理的算力浪费。
        """
        x = self.extract_feat(batch_inputs)
        
        # 仅通过度量同构主头生成唯一的单峰检测结果
        results_list = self.bbox_head.predict(x, batch_data_samples, rescale=rescale)
        
        return results_list
