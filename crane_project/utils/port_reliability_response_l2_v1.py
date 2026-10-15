"""Fixed seed1701 L2 removal control; retain all other warm-start settings."""
import copy
import math
import time
import numpy as np
import torch
from crane_project.utils import port_reliability_response_warm_v1 as warm
from crane_project.utils import port_reliability_response_tail_v1 as old
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils.port_reliability_response_stats_v1_torch import Risk
SETTINGS = dict(warm.SETTINGS, l2=0.)
ARMS = ('bce', 'soft')
VERSION = 'port_reliability_response_l2_v1'

def objective(model,x,y,kind,temperature):
    return warm.data_loss(model(x),y,kind,temperature)

def train_points(z,y):
    goods=np.sort(z[y==0]);out={}
    for rate in (.95,.97,.9849):
        target=math.ceil(rate*len(goods));cut=goods[target-1];accepted=z<=cut
        out[str(rate)]=dict(requested_CR=target,CR=int((accepted & (y==0)).sum()),
            FA=int((accepted & (y==1)).sum()),logit_cut=float(cut))
    return out

def snapshot(model,x,y,temperature):
    result=warm.distribution(model,x,y,temperature)
    with torch.no_grad():
        z=model(x).detach().cpu().numpy();labels=y.cpu().numpy()
    result.update(train_points=train_points(z,labels),train_logits=z.tolist(),
        weights={k:v.detach().cpu().tolist() for k,v in model.state_dict().items()})
    return result

def fit(raw, labels, parent, kind, temperature, device, steps=5000):
    if kind not in ARMS or parent['dimensions'] != 19 or parent['objective_kind'] != 'bce':
        raise ValueError('Fixed19 BCE parent required')
    norm = copy.deepcopy(parent['normalizer'])
    fitted = response.normalize_fit(raw, 'TRAIN')
    if norm != fitted:
        raise ValueError('Parent TRAIN normalization differs')
    x = torch.tensor(response.normalize(raw, norm), dtype=torch.float64, device=device)
    y = torch.tensor(labels, dtype=torch.float64, device=device)
    model = Risk(parent['seed'], 19)
    model.load_state_dict({k:torch.tensor(v,dtype=torch.float64) for k,v in parent['weights'].items()},strict=True)
    initial = old.state_digest(model)
    if initial != parent['final_state_sha256']:
        raise ValueError('Warm start differs from frozen parent')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS['learning_rate'],
        betas=tuple(SETTINGS['betas']),eps=SETTINGS['eps'],weight_decay=0)
    records = []; started = time.monotonic()
    for step in range(steps):
        optimizer.zero_grad(); value=objective(model,x,y,kind,temperature); value.backward()
        if not bool(torch.isfinite(value)) or not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError('Nonfinite GPU fit')
        if step % 250 == 0:
            records.append(dict(step=step,objective=value.item(),**snapshot(model,x,y,temperature)))
            print('GPU',kind,parent['seed'],'step',step,'objective',value.item(),flush=True)
        optimizer.step()
    optimizer.zero_grad(); value=objective(model,x,y,kind,temperature); value.backward()
    maximum=max(p.grad.abs().max().item() for p in model.parameters())
    if not math.isfinite(value.item()) or not math.isfinite(maximum):
        raise ValueError('Nonfinite final GPU state')
    records.append(dict(step=steps,objective=value.item(),**snapshot(model,x,y,temperature)))
    exported=dict(seed=parent['seed'],dimensions=19,objective_kind=kind,temperature=temperature,
        normalizer=norm,weights={k:v.detach().cpu().tolist() for k,v in model.state_dict().items()},
        parameters=sum(p.numel() for p in model.parameters()),steps=steps,l2=0.,
        parent_final_sha256=parent['final_state_sha256'],initial_state_sha256=initial,
        final_state_sha256=old.state_digest(model),objective_initial=records[0]['objective'],
        objective_final=value.item(),max_abs_gradient=maximum,gradient_is_diagnostic_only=True,
        optimizer_reset=True,training_device=device,elapsed_seconds=time.monotonic()-started,
        final_distribution=records[-1])
    with torch.no_grad():
        z=model(x); exported['torch_numpy_logit_max_abs']=float(np.max(np.abs(z.cpu().numpy()-response.numpy_logits(exported,raw))))
        if kind=='hard':
            h=F.softplus(z[y==0][:,None]-z[y==1][None,:]).mean(1)
            exported['final_tail_indices_within_goods']=h.topk(math.ceil(.02*len(h)),sorted=False).indices.cpu().tolist()
    if exported['torch_numpy_logit_max_abs']>1e-10 or initial==exported['final_state_sha256']:
        raise ValueError('Export differs or no training update')
    return model.eval(),exported,records

