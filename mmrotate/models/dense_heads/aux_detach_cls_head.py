# mmrotate/models/dense_heads/aux_detach_cls_head.py
"""Gradient-isolated auxiliary classification head.

This head receives DETACHED FPN features during training, ensuring that
gradients from this head NEVER flow back to backbone/FPN. Combined with
strong augmentations on the input image, it learns to output high
classification confidence on frames where the main head's cls scores
collapse (e.g., dark/blurry dead segments).

At inference, the head runs on the SAME clean image as the main head
(no augmentation, no detach), and its cls scores are fused with the
main head via element-wise max in logit space:

    cls_fused = max(main_cls_logits, aux_cls_logits)

Bbox predictions ALWAYS come from the main head — this head only
provides a second opinion on classification.
"""

import torch.nn as nn
from mmcv.cnn import bias_init_with_prob
from mmrotate.models.builder import ROTATED_HEADS


@ROTATED_HEADS.register_module(force=True)
class AuxDetachClsHead(nn.Module):
    """Detachable auxiliary classification head.

    Structure mirrors the main RetinaNet cls tower but with stacked convs
    for extra capacity (the main head uses a single conv via
    RotatedRetinaHead._init_layers override):

        stacked_convs × (Conv3x3 + ReLU) + final Conv3x3

    Args:
        in_channels (int): Input channels from FPN. Default 256.
        feat_channels (int): Internal feature channels. Default 256.
        stacked_convs (int): Number of stacked conv layers. Default 4.
        num_classes (int): Number of classes. Default 1.
        num_anchors (int): Number of anchors per spatial position.
            Default 3 (ratios [0.5, 1.0, 2.0] × 1 scale).
        init_cfg (dict | None): Initialization config.
    """

    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 stacked_convs=4,
                 num_classes=1,
                 num_anchors=3,
                 init_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.cls_out_channels = num_classes

        self._init_layers()
        if init_cfg is not None:
            self.init_weights()

    def _init_layers(self):
        """Build stacked conv layers + final classification conv."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            ch_in = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                nn.Conv2d(ch_in, self.feat_channels, 3, padding=1))
            self.cls_convs.append(nn.ReLU(inplace=True))

        self.aux_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)

    def init_weights(self):
        """Initialize conv layers with Normal(0, 0.01) and cls bias
        with prior probability 0.01."""
        for m in self.cls_convs:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        bias_cls = bias_init_with_prob(0.01)
        nn.init.normal_(self.aux_cls.weight, std=0.01)
        nn.init.constant_(self.aux_cls.bias, bias_cls)

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (tuple[Tensor]): FPN feature maps at multiple scales,
                each of shape [B, in_channels, H, W].

        Returns:
            tuple[Tensor]: Classification logits per level, each of shape
                [B, num_anchors * cls_out_channels, H, W].
        """
        cls_scores = []
        for feat in feats:
            x = feat
            for layer in self.cls_convs:
                x = layer(x)
            cls_score = self.aux_cls(x)
            cls_scores.append(cls_score)
        return tuple(cls_scores)
