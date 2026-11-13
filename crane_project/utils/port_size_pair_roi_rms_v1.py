"""Visual-only support-weighted ROI RMS input for the original pair LVSR.

One scalar per ROI over 256 visual channels and common-support positions.
No centering, learned affine, geometry normalization or decoder changes.
This is an ROI input adaptation, not standard RMSNorm or a full FRN layer.
"""
import torch
from crane_project.utils.port_size_pair_specific_v1 import (
    PARAMETER_COUNT, PairSpecificSizeHead, PairStem,
)

VISUAL_CHANNELS = 256
RMS_EPSILON = 1e-8


def normalize_visual_input(x):
    """Preserve geometry11; W is the inherited common-support channel256.

    Upstream read_pair already zeros visual features outside W>0. Empty
    support therefore remains finite/zero; original decoding owns validity.
    """
    visual = x[:, :VISUAL_CHANNELS]
    support = x[:, VISUAL_CHANNELS:VISUAL_CHANNELS+1]
    mass = support.sum(dim=(1, 2, 3), keepdim=True)
    energy = (visual.square() * support).sum(dim=(1, 2, 3), keepdim=True)
    rms = (energy / (VISUAL_CHANNELS * mass.clamp_min(RMS_EPSILON))
           + RMS_EPSILON).sqrt()
    return torch.cat((visual / rms, x[:, VISUAL_CHANNELS:]), dim=1)


class ROIRMSPairStem(PairStem):
    normalization_enabled = True

    def forward(self, x):
        if self.normalization_enabled:
            x = normalize_visual_input(x)
        return super().forward(x)


class PairROIRMSSizeHead(PairSpecificSizeHead):
    def __init__(self):
        super().__init__()
        # Reuse exact original tensors/state keys and consume no additional RNG.
        self.stem = ROIRMSPairStem(self.stem.branches)
        assert sum(p.numel() for p in self.parameters()) == PARAMETER_COUNT
