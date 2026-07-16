"""Gradient-isolated localization-quality prediction head.

The head predicts one scalar quality logit for every main-head anchor.  It is
fed detached FPN features during training, so its loss cannot alter the main
classification target, bbox regression, backbone, or FPN.  At inference its
sigmoid outputs are used as the primary ranking key inside a classification
top-K candidate pool.
"""

import torch.nn as nn
from mmcv.cnn import bias_init_with_prob

from mmrotate.models.builder import ROTATED_HEADS


@ROTATED_HEADS.register_module(force=True)
class RegQualityHead(nn.Module):
    """Small per-anchor quality head shared across FPN levels.

    Args:
        in_channels (int): FPN input channels.
        feat_channels (int): Hidden channels.
        stacked_convs (int): Number of Conv-ReLU blocks.
        num_anchors (int): Main-head anchors per spatial position.
        prior_prob (float): Initial sigmoid probability of the final logits.
    """

    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 stacked_convs=2,
                 num_anchors=3,
                 prior_prob=0.01,
                 init_cfg=None):
        super().__init__()
        if stacked_convs < 1:
            raise ValueError('stacked_convs must be at least 1')
        if num_anchors < 1:
            raise ValueError('num_anchors must be at least 1')
        if not 0.0 < prior_prob < 1.0:
            raise ValueError('prior_prob must be in (0, 1)')

        self.in_channels = int(in_channels)
        self.feat_channels = int(feat_channels)
        self.stacked_convs = int(stacked_convs)
        self.num_anchors = int(num_anchors)
        self.prior_prob = float(prior_prob)
        self.init_cfg = init_cfg

        layers = []
        for index in range(self.stacked_convs):
            in_ch = self.in_channels if index == 0 else self.feat_channels
            layers.append(nn.Conv2d(
                in_ch, self.feat_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.quality_convs = nn.Sequential(*layers)
        self.quality_pred = nn.Conv2d(
            self.feat_channels, self.num_anchors, kernel_size=3, padding=1)

        # Explicit initialization is required in this MMRotate 0.x project:
        # build_head does not reliably call init_weights when init_cfg is None.
        self.init_weights()

    def init_weights(self):
        for module in self.quality_convs.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        nn.init.normal_(self.quality_pred.weight, std=0.01)
        nn.init.constant_(
            self.quality_pred.bias, bias_init_with_prob(self.prior_prob))

    def forward(self, feats):
        quality_logits = []
        for feat in feats:
            hidden = self.quality_convs(feat)
            quality_logits.append(self.quality_pred(hidden))
        return tuple(quality_logits)
