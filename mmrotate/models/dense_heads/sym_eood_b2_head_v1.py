"""Controlled B2 head: direct classification, zero or four regression layers."""
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmrotate.models.builder import ROTATED_HEADS
from .sym_eood_head import SymEOODHead


@ROTATED_HEADS.register_module()
class SymEOODB2HeadV1(SymEOODHead):
    def __init__(self, reg_tower_layers=0, memory_limit_mib=6144, **kwargs):
        if reg_tower_layers not in (0, 4):
            raise ValueError('B2 controls only zero versus four regression layers')
        if not 0 < memory_limit_mib <= 6144:
            raise ValueError('Never increase the authorized allocator ceiling')
        self.reg_tower_layers = reg_tower_layers
        self.memory_limit_mib = memory_limit_mib
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            limit = memory_limit_mib * 2**20
            total = torch.cuda.get_device_properties(device).total_memory
            if limit >= total:
                raise RuntimeError('GPU capacity is below the B2 resource contract')
            torch.cuda.set_per_process_memory_fraction(limit / total, device)
        super().__init__(**kwargs)

    def _init_layers(self):
        super()._init_layers()
        self.reg_convs = nn.ModuleList()
        for i in range(self.reg_tower_layers):
            channels = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(ConvModule(channels, self.feat_channels, 3,
                stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))

    def forward_single(self, x):
        if self.use_semantic_cls_adapter:
            raise ValueError('B2 preserves the original disabled semantic adapter')
        reg_feat = x
        for layer in self.reg_convs:
            reg_feat = layer(reg_feat)
        return self.retina_cls(x), self.retina_reg(reg_feat)
