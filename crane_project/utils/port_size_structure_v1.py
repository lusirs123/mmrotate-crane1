"""S045: same fixed-B267-channel input/stem, vector vs direct-log decoding.

No GT in either forward. V reuses sealed S038 vector head byte-for-byte.
D has no vector-validity surrogate. Both share original size delivery.
"""
import torch
from torch import nn
from crane_project.utils import port_size_qalign_v1 as prior
from crane_project.utils import port_size_local_vectors_v1 as vector

ARMS=('V_structure','D_direct')
PARAMETERS=dict(V_structure=2800,D_direct=2746)
residual=prior.residual
read_pair=prior.read_pair
cross_edge=prior.cross_edge


def common_input(roi,support,b,bm,xy,m):
    pair=read_pair(roi,support,b,bm,xy,m)
    n=len(b);w=pair['support']
    base=vector.encode(m,b,bm,xy).detach()
    coords=vector.grid(b).expand(n,-1,-1).transpose(1,2).reshape(n,2,9,9)
    channels=base.reshape(n,81,8).transpose(1,2).reshape(n,8,9,9)
    values=torch.cat((pair['features']['fixed_b'].detach(),w.detach(),coords,channels),1)
    return values,w


class DirectSizeHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Constructing the same sealed vector net first makes common stem RNG
        # initialization exactly equal to V. No dead parameters are registered.
        base=prior.AlignedSizeHead('fixed_b')
        self.stem=base.stem
        self.terminal=nn.Conv2d(8,2,1)
        nn.init.zeros_(self.terminal.weight);nn.init.zeros_(self.terminal.bias)
        assert sum(p.numel() for p in self.parameters())==PARAMETERS['D_direct']

    def forward(self,roi,support,b,bm,xy,m):
        residual.checked_inputs(roi,support,b,bm,xy,m)
        n=len(b)
        if not n:
            delta=b.new_empty(0,2)
            return dict(residual.deliver_sizes(delta,b,bm,xy,m),delta_roi=delta,
                regression_valid=b.new_empty(0,dtype=torch.bool),local_log_variance=delta)
        features,w=common_input(roi,support,b,bm,xy,m)
        local=self.terminal(self.stem(features)).permute(0,2,3,1).reshape(n,81,2)
        weights=w.detach().reshape(n,81).clamp(0,1)
        finite=torch.isfinite(local).all(-1)
        valid=((finite | (weights==0)).all(-1)) & (weights.sum(-1)>0)
        safe=torch.where(torch.isfinite(local),local,torch.zeros_like(local))
        canonical=(safe*weights[:,:,None]).sum(1)/weights.sum(-1).clamp(min=1e-12)[:,None]
        variance=((safe-canonical[:,None]).square()*weights[:,:,None]).sum(1)/weights.sum(-1).clamp(min=1e-12)[:,None]
        raw=torch.where((m[:,2]<m[:,3])[:,None],canonical.flip(-1),canonical)
        delta=residual.raw_roi_pair(raw,bm)
        final_delta=torch.where(valid[:,None],delta,torch.full_like(delta,float('nan')))
        return dict(residual.deliver_sizes(final_delta,b,bm,xy,m),delta_roi=delta,
            regression_valid=valid,local_log_variance=variance,common_support=w)


def head(arm):
    if arm not in ARMS:raise ValueError('Unknown structural arm')
    return prior.AlignedSizeHead('fixed_b') if arm==ARMS[0] else DirectSizeHead()


def valid(output,arm):
    return output['vector_valid' if arm==ARMS[0] else 'regression_valid']


def size_loss(output,gt,m,bm,arm):
    if not bool(valid(output,arm).all()) or not bool(output['candidate_valid'].all()):
        raise ValueError('Invalid representation/candidate; stop training, do not filter')
    return residual.size_loss(output['delta_roi'],gt,m,bm)
