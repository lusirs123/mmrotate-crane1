"""Gradient-isolated platform OBB detection head for Probe 4."""

import torch.nn as nn
from mmcv.cnn import bias_init_with_prob
from mmdet.core import multi_apply

from mmrotate.models.builder import ROTATED_HEADS
from .rotated_atss_head import RotatedATSSHead


@ROTATED_HEADS.register_module(force=True)
class PlatformOBBATSSHead(RotatedATSSHead):
    """ATSS assignment with independent classification/regression towers.

    The stock ``RotatedATSSHead`` in this checkout stores ``stacked_convs`` but
    its forward path consists only of the final 3x3 prediction convolutions.
    Platform detection needs enough capacity to interpret frozen FPN features,
    so this probe head adds explicit towers while retaining the established
    rotated ATSS target assignment, bbox coder, losses and post-processing.

    Gradient isolation is enforced by the caller: all input FPN tensors must be
    detached and the optimizer must own only this module's parameters.
    """

    def __init__(self, stacked_convs=4, **kwargs):
        self.stacked_convs = int(stacked_convs)
        super().__init__(stacked_convs=stacked_convs, **kwargs)
        self.init_weights()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for index in range(self.stacked_convs):
            in_channels = (
                self.in_channels if index == 0 else self.feat_channels)
            self.cls_convs.extend([
                nn.Conv2d(in_channels, self.feat_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            ])
            self.reg_convs.extend([
                nn.Conv2d(in_channels, self.feat_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            ])
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels, 3, padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)

    def init_weights(self):
        for module in list(self.cls_convs) + list(self.reg_convs):
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        nn.init.normal_(self.retina_cls.weight, std=0.01)
        nn.init.constant_(self.retina_cls.bias, bias_init_with_prob(0.01))
        nn.init.normal_(self.retina_reg.weight, std=0.01)
        nn.init.constant_(self.retina_reg.bias, 0.0)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        cls_feat = feat
        reg_feat = feat
        for layer in self.cls_convs:
            cls_feat = layer(cls_feat)
        for layer in self.reg_convs:
            reg_feat = layer(reg_feat)
        return self.retina_cls(cls_feat), self.retina_reg(reg_feat)
