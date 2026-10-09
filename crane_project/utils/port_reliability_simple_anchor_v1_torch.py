"""FP64 bounded residual; exact neutral risk without blocking its gradient."""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from crane_project.utils import port_reliability_simple_anchor_v1 as core


class BoundedRisk(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(258,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,1))
        nn.init.zeros_(self.network[4].weight)
        nn.init.zeros_(self.network[4].bias)
        self.beta = nn.Parameter(torch.zeros(1))

    def delta(self, normalized):
        return .5*torch.tanh(self.network(normalized[:,1:]).squeeze(-1)+self.beta[0])

    def forward(self, normalized, anchor):
        return torch.log(anchor)-torch.log1p(-anchor)+self.delta(normalized)

    def risk(self, normalized, anchor):
        e = torch.expm1(self.delta(normalized))
        return anchor*(1+e)/(1+anchor*e)


def make_models(device):
    torch.manual_seed(1701)
    first = BoundedRisk().to(device=device,dtype=torch.float64)
    return {core.ARMS[0]:first, core.ARMS[1]:deepcopy(first)}


def exported(model):
    return {k:v.detach().cpu().tolist() for k,v in model.state_dict().items()}


def optimizer(model):
    decay, no_decay = [], []
    for n,p in model.named_parameters():
        (decay if n.endswith('weight') else no_decay).append(p)
    return torch.optim.AdamW([dict(params=decay,weight_decay=.0001),dict(params=no_decay,weight_decay=0.)],lr=.001)


def update(model, optim, x, anchor, y, weights):
    optim.zero_grad()
    logits = model(x.detach(),anchor.detach())
    loss = (F.binary_cross_entropy_with_logits(logits,y.detach(),reduction='none')*weights.detach()).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite balanced objective')
    loss.backward()
    grad = {n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
    before = float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
    after = math.sqrt(sum(float(p.grad.norm())**2 for p in model.parameters() if p.grad is not None))
    if not grad or not all(math.isfinite(v) for v in [before,after,*grad.values()]):
        raise ValueError('Nonfinite/disconnected quality gradient')
    optim.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Nonfinite parameters')
    with torch.no_grad():
        d = model.delta(x)
    return dict(loss=float(loss.detach()),gradient_by_parameter=grad,
        gradient_norm_before=before,gradient_norm_after=after,
        delta_min=float(d.min()),delta_max=float(d.max()),
        saturation_fraction=float((d.abs()>.49).double().mean()))
