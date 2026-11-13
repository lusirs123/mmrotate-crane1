"""Partial visual-only spatial normalization in the original 2480-param head.

Per branch: ReLU(cat(N_W(V[:2]), V[2:]) + G + original_bias).
No affine normalization parameters; geometry11 is never normalized.
Inspired by the partial-normalization principle of HINet (CVPRW2021), not
its complete block/network. Original local-vector decoding is inherited.
"""
import torch
from torch import nn
from torch.nn import functional as F
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead

NORMALIZED_CHANNELS = 2
VARIANCE_EPSILON = 1e-5
SUPPORT_EPSILON = 1e-8


def normalize_supported(v, support):
    """Per ROI/channel population moments, weighted by detached common W.

    Moments remain differentiable in V. Invalid positions and effectively empty
    supports stay zero; support mapping and final aggregation are unchanged.
    """
    w = support.detach().clamp(0, 1)
    mass = w.sum(dim=(2, 3), keepdim=True)
    denominator = mass.clamp_min(SUPPORT_EPSILON)
    mean = (v*w).sum(dim=(2, 3), keepdim=True)/denominator
    variance = ((v-mean).square()*w).sum(dim=(2, 3), keepdim=True)/denominator
    normalized = (v-mean)/(variance+VARIANCE_EPSILON).sqrt()
    return torch.where((w>0)&(mass>SUPPORT_EPSILON), normalized, torch.zeros_like(normalized))


class PartialVisualProjection(nn.Module):
    normalization_enabled = True

    def __init__(self, original):
        super().__init__()
        if original.in_channels!=267 or original.out_channels!=4 or original.kernel_size!=(1, 1):
            raise ValueError('Expected original267to4 projection')
        self.weight = original.weight
        self.bias = original.bias

    def forward(self, x):
        if not self.normalization_enabled:
            return F.conv2d(x, self.weight, self.bias)
        visual = F.conv2d(x[:, :256], self.weight[:, :256], None)
        geometry = F.conv2d(x[:, 256:], self.weight[:, 256:], None)
        partial = torch.cat((normalize_supported(visual[:, :NORMALIZED_CHANNELS], x[:, 256:257]),
                             visual[:, NORMALIZED_CHANNELS:]), dim=1)
        return partial + geometry + self.bias[None, :, None, None]


class PairPartialVisualNormSizeHead(PairSpecificSizeHead):
    def __init__(self):
        super().__init__()
        # Preserve state keys, original tensor objects and RNG consumption.
        for branch in self.stem.branches:
            branch[0] = PartialVisualProjection(branch[0])
        assert sum(p.numel() for p in self.parameters()) == 2480

    def set_normalization(self, enabled):
        for branch in self.stem.branches:
            branch[0].normalization_enabled = enabled
