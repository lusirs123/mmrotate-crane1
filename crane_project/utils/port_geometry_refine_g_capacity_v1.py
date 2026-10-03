"""One fixed G capacity intervention: 64 to 16 hidden FC units.

Inherit the reviewed forward/coordinates/loss unchanged. No trainable sampling,
new feature source, center/score output or GT path is introduced.
"""
import torch
from torch import nn

from crane_project.utils.port_geometry_refine_g_v1 import (
    LocalGeometryRefiner, SETTINGS)

HIDDEN_FC = 16
PARAMETER_COUNT = 59107


class CompactLocalGeometryRefiner(LocalGeometryRefiner):
    def __init__(self):
        nn.Module.__init__(self)
        c, h, r = SETTINGS['in_channels'], SETTINGS['hidden_channels'], SETTINGS['roi_size']
        self.stem = nn.Sequential(nn.Conv2d(c+1, h, 1), nn.ReLU(inplace=False),
                                  nn.Conv2d(h, h, 3, padding=1), nn.ReLU(inplace=False))
        self.hidden = nn.Sequential(nn.Linear(h*r*r+4, HIDDEN_FC), nn.ReLU(inplace=False))
        self.output = nn.Linear(HIDDEN_FC, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


def compact_initial_state(reference):
    """Copy the predetermined first16 units of the seed1703 UNTRAINED wide head.

This is initialization matching, not pruning a learned head or selecting units
using data. Stem/first16 hidden weights and biases are identical; output is zero.
    """
    expected = CompactLocalGeometryRefiner().state_dict()
    if set(reference) != set(expected):
        raise ValueError('Wide initialization keys differ')
    if (reference['hidden.0.weight'].shape != (64, expected['hidden.0.weight'].shape[1]) or
            reference['hidden.0.bias'].shape != (64,) or
            reference['output.weight'].shape != (3, 64) or
            not bool((reference['output.weight'] == 0).all()) or
            not bool((reference['output.bias'] == 0).all())):
        raise ValueError('Requires the untrained zero-output FC64 initialization')
    state = {}
    for key, target in expected.items():
        value = reference[key]
        if key in ('hidden.0.weight', 'hidden.0.bias'):
            value = value[:HIDDEN_FC]
        elif key == 'output.weight':
            value = value[:, :HIDDEN_FC]
        if value.shape != target.shape or value.dtype != target.dtype or value.device.type != 'cpu':
            raise ValueError('Expected unchanged finite CPU initialization: '+key)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite initialization: '+key)
        state[key] = value.detach().clone()
    return state
