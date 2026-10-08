"""Small quality heads only. Backbone and midpoint NEVER receive gradients."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from crane_project.utils import port_reliability_redc_size_v1 as core


class LinearRisk(nn.Module):
    def __init__(self, normalizer):
        super().__init__()
        self.output = nn.Linear(291, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, -normalizer['mean'][0])
        with torch.no_grad():
            self.output.weight[0,0] = -normalizer['scale'][0]

    def forward(self, normalized, raw_logit):
        return self.output(normalized).squeeze(-1)


class ReDCRisk(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(290,16),nn.ReLU(),
                                     nn.Linear(16,8),nn.ReLU(),nn.Linear(8,1))
        nn.init.zeros_(self.network[4].weight)
        nn.init.constant_(self.network[4].bias, math.log(math.expm1(.75)))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, normalized, raw_logit):
        temperature = .25+F.softplus(self.network(normalized[:,1:]).squeeze(-1))
        return -(raw_logit/temperature+self.beta[0])


def exported(model):
    return {k:v.detach().cpu().tolist() for k,v in model.state_dict().items()}


def balanced_loss(logits, labels, weights):
    if (logits.shape!=labels.shape or labels.shape!=weights.shape
            or not bool(torch.isfinite(logits).all())):
        raise ValueError('Invalid balanced objective')
    return (F.binary_cross_entropy_with_logits(logits,labels,reduction='none')*weights).mean()


def optimizer(model):
    decay, no_decay = [], []
    for name,p in model.named_parameters():
        (decay if name.endswith('weight') else no_decay).append(p)
    return torch.optim.AdamW([dict(params=decay,weight_decay=.0001),
                              dict(params=no_decay,weight_decay=0.)],lr=.001)


def update(model, optim, x, z, y, w):
    optim.zero_grad()
    logits = model(x.detach(),z.detach())
    loss = balanced_loss(logits,y.detach(),w.detach())
    loss.backward()
    gradients = {n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
    if not all(math.isfinite(v) for v in gradients.values()) or not gradients:
        raise ValueError('Nonfinite or disconnected quality gradient')
    before = float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
    after = math.sqrt(sum(float(p.grad.norm())**2 for p in model.parameters() if p.grad is not None))
    if not math.isfinite(before) or not math.isfinite(after):
        raise ValueError('Nonfinite clipping')
    optim.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Nonfinite head update')
    return dict(loss=float(loss.detach()),gradient_by_parameter=gradients,
                gradient_norm_before=before,gradient_norm_after=after)


def make_models(normalizer, device):
    torch.manual_seed(1701)
    return dict(linear=LinearRisk(normalizer).to(device),redc=ReDCRisk().to(device))


def freeze_features(head, records, device):
    """Observe actual frozen stem once; never alter its forward/decoder."""
    captured = []
    hook = head.stem.register_forward_hook(lambda module,inputs,output: captured.append(output.detach()))
    results = []
    try:
        with torch.no_grad():
            for record in records:
                captured.clear()
                inputs = [record[k].to(device) for k in ('roi','support','boxes_original','boxes_model','scale_xy')]
                decoded = head(*inputs)
                if len(record['boxes_original']):
                    if len(captured)!=1:
                        raise ValueError('Expected one actual geometry stem call')
                    pooled = F.avg_pool2d(captured[0],3,stride=3).reshape(288).cpu().numpy()
                    independent = core.pooled_feature(captured[0][0].cpu().numpy())
                    if not np.allclose(pooled,independent,rtol=1e-5,atol=1e-5):
                        raise ValueError('Spatial pooling differs from scalar reference')
                    results.append((pooled,decoded))
                else:
                    if captured:
                        raise ValueError('Missing box cannot produce stem features')
                    results.append((None,decoded))
    finally:
        hook.remove()
    return results
