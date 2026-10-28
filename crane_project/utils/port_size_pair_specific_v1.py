"""Pair-specific spatial transforms; sealed LVSR forward/decoder inherited.

Pairs are canonical M points 0/2 and 1/3, not permanently decoded long/short.
Per-location sorting and whole-frame fallback deliberately remain unchanged.
"""
import torch
from torch import nn
from crane_project.utils.port_size_qalign_v1 import AlignedSizeHead

PARAMETER_COUNT = 2480


class PairStem(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class PairTerminal(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        a, b = [branch(x[:, 4*i:4*i+4]) for i, branch in enumerate(self.branches)]
        # Each branch emits (+point xy, -point xy); restore (+u,+v,-u,-v).
        return torch.cat((a[:, :2], b[:, :2], a[:, 2:], b[:, 2:]), dim=1)


class PairSpecificSizeHead(AlignedSizeHead):
    def __init__(self):
        # Avoid constructing/discarding the historical 2800-parameter head.
        nn.Module.__init__(self)
        self.arm = 'fixed_b'
        stems, terminals = [], []
        for _ in range(2):
            stems.append(nn.Sequential(nn.Conv2d(267, 4, 1), nn.ReLU(),
                nn.Conv2d(4, 4, 3, padding=1), nn.ReLU()))
            terminal = nn.Conv2d(4, 4, 1)
            nn.init.zeros_(terminal.weight)
            nn.init.zeros_(terminal.bias)
            terminals.append(terminal)
        self.stem, self.terminal = PairStem(stems), PairTerminal(terminals)
        assert sum(p.numel() for p in self.parameters()) == PARAMETER_COUNT

    # Inherited AlignedSizeHead.forward -> LocalVectorSizeHead.forward calls
    # these two adapters; encode/decode/W/mapping/delivery remain the originals.
