"""FP64 bounded residual; exact neutral risk without blocking its gradient."""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from crane_project.utils import port_reliability_within_video_rank_v1 as core


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


def tensor_pairs(plan, device):
    return {name:dict(bad=torch.tensor(v['bad'],dtype=torch.long,device=device),
                      good=torch.tensor(v['good'],dtype=torch.long,device=device))
            for name,v in plan.items()}


def rank_loss(logits, plan):
    terms = {name:F.softplus(logits[v['good']][None,:]-logits[v['bad']][:,None]).mean()
             for name,v in plan.items()}
    return torch.stack(list(terms.values())).mean(),terms


def component_diagnostics(model, bce, weighted_rank):
    params = list(model.parameters())
    parts = [torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
             for loss in (bce,weighted_rank)]
    norms = [math.sqrt(sum(float(g.detach().norm())**2 for g in part if g is not None))
             for part in parts]
    dot = sum(float((a.detach()*b.detach()).sum()) for a,b in zip(*parts)
              if a is not None and b is not None)
    cosine = dot/(norms[0]*norms[1]) if min(norms)>0 else None
    by_parameter = {name:dict(bce=float(a.detach().norm()) if a is not None else 0.,
                             weighted_rank=float(b.detach().norm()) if b is not None else 0.)
                    for (name,_),a,b in zip(model.named_parameters(),*parts)}
    if not all(math.isfinite(v) for v in norms+[dot]):
        raise ValueError('Nonfinite component gradient')
    return dict(bce_norm=norms[0],weighted_rank_norm=norms[1],
                relative_strength=norms[1]/norms[0] if norms[0]>0 else None,
                cosine=cosine,by_parameter=by_parameter)


def update(model, optim, x, anchor, y, weights, indices, plan, arm, diagnose=False):
    if arm not in core.ARMS:
        raise ValueError('Unknown ranking arm')
    optim.zero_grad()
    logits = model(x.detach(),anchor.detach())
    bce = (F.binary_cross_entropy_with_logits(logits[indices],y[indices].detach(),reduction='none')
           *weights[indices].detach()).mean()
    rank,terms = rank_loss(logits,plan)
    weighted_rank = core.SETTINGS['rank_weight']*rank
    loss = bce if arm=='bce_control' else bce+weighted_rank
    if not all(bool(torch.isfinite(v)) for v in (bce,rank,loss)):
        raise ValueError('Nonfinite BCE/ranking objective')
    diagnostics = component_diagnostics(model,bce,weighted_rank) if diagnose else None
    loss.backward()
    grad = {n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
    before = float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
    after = math.sqrt(sum(float(p.grad.norm())**2 for p in model.parameters() if p.grad is not None))
    if not grad or not all(math.isfinite(v) for v in [before,after,*grad.values()]):
        raise ValueError('Nonfinite/disconnected quality gradient')
    retention = after/before if before>0 else 1.
    if diagnostics is not None:
        diagnostics['shared_clip_retention'] = retention
        diagnostics['bce_norm_after_shared_clip'] = diagnostics['bce_norm']*retention
        diagnostics['weighted_rank_norm_after_shared_clip'] = diagnostics['weighted_rank_norm']*retention
    optim.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Nonfinite parameters')
    with torch.no_grad():
        d = model.delta(x)
    return dict(loss=float(loss.detach()),bce_loss=float(bce.detach()),rank_loss=float(rank.detach()),
        rank_by_video={k:float(v.detach()) for k,v in terms.items()},
        weighted_rank_loss=float(weighted_rank.detach()),rank_applied=arm=='within_video_rank',
        component_gradients=diagnostics,full_forward_rows=len(x),
        shared_clip_retention=retention,
        gradient_by_parameter=grad,
        gradient_norm_before=before,gradient_norm_after=after,
        delta_min=float(d.min()),delta_max=float(d.max()),
        saturation_fraction=float((d.abs()>.49).double().mean()))
