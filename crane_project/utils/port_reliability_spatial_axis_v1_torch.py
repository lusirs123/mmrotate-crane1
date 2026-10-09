"""Only quality head trainable. GT enters offline losses, never forward."""
from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from crane_project.utils import port_reliability_spatial_axis_v1 as core

class SpatialRisk(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(257,8,1),nn.ReLU(),nn.Conv2d(8,8,3,padding=1),nn.ReLU())
        self.hidden=nn.Linear(650,16)
        self.risk_head=nn.Linear(16,1);self.axis_head=nn.Linear(16,2)
        nn.init.zeros_(self.risk_head.weight);nn.init.zeros_(self.risk_head.bias)
        nn.init.normal_(self.axis_head.weight,std=.001);nn.init.zeros_(self.axis_head.bias)
    def forward(self,x,support,descriptors,anchor,arm):
        if arm not in core.ARMS:raise ValueError('Unknown arm')
        if x.shape[1:]!=(256,9,9) or support.shape!=(len(x),1,9,9) or descriptors.shape!=(len(x),2):raise ValueError('Spatial shape differs')
        if arm=='coarse_bce':x=F.avg_pool2d(x.detach(),3,3).repeat_interleave(3,2).repeat_interleave(3,3)
        h=self.stem(torch.cat((x.detach(),support.detach()),1))
        h=F.relu(self.hidden(torch.cat((h.flatten(1),descriptors.detach()),1)))
        delta=.5*torch.tanh(self.risk_head(h).squeeze(1).double())
        a=anchor.detach().double();e=torch.expm1(delta)
        return dict(logits=torch.log(a)-torch.log1p(-a)+delta,risk=a*(1+e)/(1+a*e),delta=delta,axis=self.axis_head(h).double())

def make_models(device):
    torch.manual_seed(1701);first=SpatialRisk().to(device)
    return {a:deepcopy(first) for a in core.ARMS}

def exported(model):return {k:v.detach().cpu().tolist() for k,v in model.state_dict().items()}

def optimizer(model):
    return torch.optim.AdamW([dict(params=[p for n,p in model.named_parameters() if n.endswith('weight')],weight_decay=.0001),
      dict(params=[p for n,p in model.named_parameters() if not n.endswith('weight')],weight_decay=0.)],lr=.001)

def component(model,bce,aux):
    ps=list(model.parameters());parts=[]
    for l in (bce,aux):
        gs=torch.autograd.grad(l,ps,retain_graph=True,allow_unused=True)
        parts.append([torch.zeros_like(p) if g is None else g.detach() for p,g in zip(ps,gs)])
    norms=[math.sqrt(sum(float(g.norm())**2 for g in q)) for q in parts]
    dot=sum(float((g*h).sum()) for g,h in zip(*parts))
    return dict(bce_norm=norms[0],weighted_aux_norm=norms[1],relative_strength=norms[1]/norms[0] if norms[0] else None,
      cosine=dot/(norms[0]*norms[1]) if min(norms)>0 else None,
      by_parameter={n:dict(bce=float(g.norm()),weighted_aux=float(h.norm())) for (n,_),g,h in zip(model.named_parameters(),*parts)})

def update(model,opt,x,support,ds,anchor,y,w,t,arm,diagnose=False):
    opt.zero_grad();v=model(x,support,ds,anchor,arm)
    bce=(F.binary_cross_entropy_with_logits(v['logits'],y.detach(),reduction='none')*w.detach()).mean()
    aux=F.smooth_l1_loss(v['axis'],t.detach(),beta=1.)*.25
    loss=bce+aux if arm=='spatial_axis' else bce
    comp=component(model,bce,aux) if diagnose else None
    loss.backward();grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
    before=float(nn.utils.clip_grad_norm_(model.parameters(),5.));after=math.sqrt(sum(float(p.grad.norm())**2 for p in model.parameters() if p.grad is not None))
    if not all(math.isfinite(v) for v in [float(loss),before,after,*grads.values()]):raise ValueError('Nonfinite update')
    if comp:
        comp.update(shared_clip_retention=min(1.,5./(before+1e-6)),bce_norm_after_shared_clip=comp['bce_norm']*min(1.,5./(before+1e-6)),weighted_aux_norm_after_shared_clip=comp['weighted_aux_norm']*min(1.,5./(before+1e-6)))
    opt.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError('Nonfinite parameters')
    return dict(loss=float(loss.detach()),bce_loss=float(bce.detach()),weighted_auxiliary_loss=float(aux.detach()),
      auxiliary_applied=arm=='spatial_axis',component_gradients=comp,gradient_by_parameter=grads,
      gradient_norm_before=before,gradient_norm_after=after,actual_clipping=before>5.,
      delta_min=float(v['delta'].min()),delta_max=float(v['delta'].max()),saturation_fraction=float((v['delta'].abs()>.49).double().mean()))
