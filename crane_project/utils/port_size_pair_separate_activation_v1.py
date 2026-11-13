"""2480-parameter LVSR: separate visual/geometry activation before addition.

Original initialization tensors, state keys, decoder and delivery are retained.
The original single bias belongs to the visual projection. No second bias.
"""
import torch
from torch import nn
from torch.nn import functional as F

from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead


class SeparateActivationProjection(nn.Module):
    def __init__(self, original):
        super().__init__()
        if original.in_channels != 267 or original.out_channels != 4:
            raise ValueError('Expected original267to4 projection')
        self.weight = original.weight
        self.bias = original.bias
        self.joint_control = False

    def forward(self, x):
        visual = F.conv2d(x[:, :256], self.weight[:, :256], self.bias)
        geometry = F.conv2d(x[:, 256:], self.weight[:, 256:], None)
        if self.joint_control:
            return torch.relu(visual+geometry)
        return torch.relu(visual)+torch.relu(geometry)


class PairSeparateActivationSizeHead(PairSpecificSizeHead):
    def __init__(self):
        super().__init__()
        # Wrapping existing parameters consumes no extra RNG or parameters.
        for branch in self.stem.branches:
            branch[0] = SeparateActivationProjection(branch[0])
            branch[1] = nn.Identity()
        if sum(p.numel() for p in self.parameters()) != 2480:
            raise ValueError('Separate activation must retain2480 parameters')
