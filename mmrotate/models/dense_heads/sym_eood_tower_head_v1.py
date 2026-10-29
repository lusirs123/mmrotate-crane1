"""Single structural control: add EOOD-style main towers to frozen S design."""
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmrotate.models.builder import ROTATED_HEADS
from .sym_eood_head import SymEOODHead


@ROTATED_HEADS.register_module()
class SymEOODTowerHeadV1(SymEOODHead):
    def _init_layers(self):
        super()._init_layers()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            channels = self.in_channels if i == 0 else self.feat_channels
            for layers in (self.cls_convs, self.reg_convs):
                layers.append(ConvModule(channels, self.feat_channels, 3,
                    stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))

    def forward_single(self, x):
        if self.use_semantic_cls_adapter:
            raise ValueError('T4 control requires the original disabled semantic adapter')
        cls_feat = reg_feat = x
        for layer in self.cls_convs:
            cls_feat = layer(cls_feat)
        for layer in self.reg_convs:
            reg_feat = layer(reg_feat)
        return self.retina_cls(cls_feat), self.retina_reg(reg_feat)
