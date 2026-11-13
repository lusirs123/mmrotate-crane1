"""Review-only shared-weight local/dilated spatial operator.

No production trainer/config imports this module. Preserve both original
parameter objects and state keys; support/decoding remain inherited.
"""
import torch
from torch import nn
from torch.nn import functional as F
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead


class SharedSpatialProjection(nn.Module):
    enabled=True

    def __init__(self,original):
        super().__init__()
        if original.in_channels!=4 or original.out_channels!=4 or original.kernel_size!=(3,3) or original.padding!=(1,1):
            raise ValueError('Expected original4to4 3x3 padding1 operator')
        self.weight=original.weight
        self.bias=original.bias

    def forward(self,x):
        if not self.enabled:return F.conv2d(x,self.weight,self.bias,padding=1)
        local=F.conv2d(x,self.weight,None,padding=1)
        distant=F.conv2d(x,self.weight,None,padding=2,dilation=2)
        return (local+distant)*.5+self.bias[None,:,None,None]


class SharedSpatialReviewHead(PairSpecificSizeHead):
    def __init__(self):
        super().__init__()
        for branch in self.stem.branches:branch[2]=SharedSpatialProjection(branch[2])
        assert sum(p.numel() for p in self.parameters())==2480

    def set_spatial_review(self,enabled):
        for branch in self.stem.branches:branch[2].enabled=bool(enabled)


def tap_geometry(w):
    """Geometry-only fractions; taps are paths, not new image information."""
    w=w.detach().clamp(0,1)
    if w.ndim!=4 or w.shape[1:]!=(1,9,9):raise ValueError('Expected Nx1x9x9 common W')
    active=(w>0).to(w);inside=torch.ones_like(w);kernel=w.new_ones(1,1,3,3)
    mass=w.sum((1,2,3)).clamp_min(1e-12);result={}
    for dilation in (1,2):
        all_in=F.conv2d(inside,kernel,padding=dilation,dilation=dilation)
        valid=F.conv2d(active,kernel,padding=dilation,dilation=dilation)
        def weighted(v):return (v*w).sum((1,2,3))/mass
        result['d'+str(dilation)]=dict(
            valid_fraction=weighted(valid/9),
            outside_ROI_fraction=weighted((9-all_in)/9),
            inside_zero_support_fraction=weighted((all_in-valid)/9))
        if dilation==2:
            result['new_far_valid_taps_mean']=weighted(valid-active)
            result['supported_outputs_with_any_new_valid_neighbor_fraction']=weighted((valid-active>0).to(w))
    return result
