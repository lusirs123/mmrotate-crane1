"""Small trainable refinement for frozen native DINO feature maps.

The adapter is intentionally residual with a zero output projection.  At
initialization it is an exact identity, so a checkpoint loaded into the new
route reproduces the audited native S14 detector before source-only
optimization, while the projection still has a usable gradient.
"""

import torch
import torch.nn as nn


class NativeSpatialAdapter(nn.Module):
    """Lightweight same-stride spatial refinement for native RPN/ROI features."""

    def __init__(self, channels: int, hidden_channels: int = 128):
        super().__init__()
        channels = int(channels)
        hidden_channels = int(hidden_channels)
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError('Adapter channel counts must be positive')
        self.projection = nn.Conv2d(channels, hidden_channels, 1, bias=False)
        self.refinement = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                      groups=hidden_channels, bias=False),
            nn.GroupNorm(max(1, min(32, hidden_channels)), hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
        )
        # Keep the initial mapping exactly identity because the output
        # projection is zero, while leaving a gradient path open through the
        # residual branch.  A zero gate together with a zero output
        # projection would multiply both gradients by zero and make the
        # adapter permanently untrainable.
        self.residual_gate = nn.Parameter(torch.ones(()))
        nn.init.zeros_(self.refinement[-1].weight)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError('Native spatial adapter expects [B,C,H,W]')
        delta = self.refinement(self.projection(feature))
        return feature + torch.tanh(self.residual_gate) * delta
