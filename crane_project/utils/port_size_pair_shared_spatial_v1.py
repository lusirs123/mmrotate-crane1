"""Single shared-weight d1/d2 spatial change in the original2480 LVSR head.

Per branch: ReLU(0.5*K_d1(h)+0.5*K_d2(h)+b), with the same K.
Reference, common support, per-location decoding and delivery are inherited.
"""
from torch import nn
from torch.nn import functional as F
from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead

LOCAL_DILATION=1
DISTANT_DILATION=2
MIX_WEIGHT=.5


class SharedSpatialConv2d(nn.Module):
    mixing_enabled=True

    def __init__(self,original):
        super().__init__()
        if original.in_channels!=4 or original.out_channels!=4 or original.kernel_size!=(3,3) or original.padding!=(1,1):
            raise ValueError('Expected original4to4 3x3 padding1 operator')
        self.weight=original.weight
        self.bias=original.bias

    def forward(self,x):
        if not self.mixing_enabled:return F.conv2d(x,self.weight,self.bias,padding=1)
        local=F.conv2d(x,self.weight,None,padding=LOCAL_DILATION,dilation=LOCAL_DILATION)
        distant=F.conv2d(x,self.weight,None,padding=DISTANT_DILATION,dilation=DISTANT_DILATION)
        return (local+distant)*MIX_WEIGHT+self.bias[None,:,None,None]


class PairSharedSpatialSizeHead(PairSpecificSizeHead):
    def __init__(self):
        super().__init__()
        for branch in self.stem.branches:branch[2]=SharedSpatialConv2d(branch[2])
        assert sum(p.numel() for p in self.parameters())==2480

    def set_spatial_mixing(self,enabled):
        for branch in self.stem.branches:branch[2].mixing_enabled=bool(enabled)
