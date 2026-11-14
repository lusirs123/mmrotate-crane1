"""Direct canonical log-size readout control for the unchanged pair LVSR.

Two existing pair stems are copied, then each has a scalar terminal. Their
outputs mean canonical long/short, unlike the vector head's opposing pairs.
This tests parameterization/readout together, not any single decoder operator.
"""
from copy import deepcopy
import torch
from torch import nn
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
from crane_project.utils.port_size_structure_v1 import common_input
from crane_project.utils import port_size_qalign_v1 as vector

PARAMETER_COUNT = 2450


class DirectTerminal(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv2d(4, 1, 1) for _ in range(2)])
        for layer in self.branches:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        return torch.cat([layer(features[:, 4*i:4*i+4])
                          for i, layer in enumerate(self.branches)], 1)


class PairDirectSizeControlHead(nn.Module):
    def __init__(self, reference=None):
        super().__init__()
        reference = PairSpecificSizeHead() if reference is None else reference
        self.stem = deepcopy(reference.stem)
        self.terminal = DirectTerminal()
        if sum(p.numel() for p in self.parameters()) != PARAMETER_COUNT:
            raise ValueError('Expected2450 active parameters; no padding parameters')

    def forward(self, roi, support, b, bm, xy, m):
        vector.residual.checked_inputs(roi, support, b, bm, xy, m)
        if not len(b):
            delta = b.new_empty(0, 2)
            return dict(vector.residual.deliver_sizes(delta, b, bm, xy, m), delta_roi=delta,
                        vector_valid=b.new_empty(0, dtype=torch.bool), local_size_variance=delta)
        features, w = common_input(roi, support, b, bm, xy, m)
        local = self.terminal(self.stem(features)).permute(0, 2, 3, 1).reshape(len(b), 81, 2)
        weights = w.detach().reshape(len(b), 81).clamp(0, 1)
        valid = ((torch.isfinite(local).all(-1) | (weights == 0)).all(-1)) & (weights.sum(-1) > 0)
        safe = torch.where(torch.isfinite(local), local, torch.zeros_like(local))
        denominator = weights.sum(-1).clamp(min=1e-12)[:, None]
        canonical = (safe*weights[:, :, None]).sum(1)/denominator
        variance = ((safe-canonical[:, None]).square()*weights[:, :, None]).sum(1)/denominator
        raw = torch.where((m[:, 2] < m[:, 3])[:, None], canonical.flip(-1), canonical)
        delta = vector.residual.raw_roi_pair(raw, bm)
        delivery = torch.where(valid[:, None], delta, torch.full_like(delta, float('nan')))
        # Historical trainer name; denotes valid regression/support, not vectors.
        return dict(vector.residual.deliver_sizes(delivery, b, bm, xy, m), delta_roi=delta,
                    vector_valid=valid, local_size_variance=variance, common_support=w)
