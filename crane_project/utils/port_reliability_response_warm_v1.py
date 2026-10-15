"""BCE warm start: matched BCE / hard tail / paper KL-DRO continuation."""
import copy
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from crane_project.utils import port_reliability_response_tail_v1 as old
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils.port_reliability_response_stats_v1_torch import Risk

VERSION = 'port_reliability_response_warm_v1'
SEEDS = (1701, 1702, 1703)
ARMS = ('bce', 'hard', 'soft')
SETTINGS = dict(steps=5000, learning_rate=.001, l2=.1, tail_fraction=.02,
    temperature_log_bounds=[-18., 10.], temperature_bisections=64,
    primary_CR=653, minimum_Real_CR=189, minimum_FA_gain=3,
    hidden=8, optimizer='Adam', betas=[.9, .999], eps=1e-8,
    dtype='float64', final_step_only=True, training_device='cuda:0')


def soft_loss(z, y, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive frozen temperature required')
    good, wrong = z[y == 0], z[y == 1]
    pair = F.softplus(good[None, :]-wrong[:, None])
    return temperature*(torch.logsumexp(pair/temperature, 1)-math.log(len(good))).mean()


def data_loss(z, y, kind, temperature):
    if kind == 'soft':
        return soft_loss(z, y, temperature)
    return old.data_loss(z, y, 'tail' if kind == 'hard' else kind)


def objective(model, x, y, kind, temperature):
    penalty = sum(p.square().sum() for n, p in model.named_parameters() if n.endswith('weight'))
    return data_loss(model(x), y, kind, temperature)+.5*SETTINGS['l2']*penalty


def effective_support(pair, temperature):
    scaled = pair/temperature
    w = torch.softmax(scaled, 1)
    return (1/w.square().sum(1)).mean().item()


def temperature_fit(raw, labels, parents):
    """One common TRAIN-only temperature; mean ESS equals ceil(2% goods)."""
    pairs = []
    y = torch.tensor(labels, dtype=torch.float64)
    for parent in parents:
        z = torch.tensor(response.numpy_logits(parent, raw), dtype=torch.float64)
        pairs.append(F.softplus(z[y == 0][None, :]-z[y == 1][:, None]))
    pair = torch.cat(pairs, 0)
    target = math.ceil(SETTINGS['tail_fraction']*int((y == 0).sum()))
    lo, hi = SETTINGS['temperature_log_bounds']
    bounds = [effective_support(pair, math.exp(v)) for v in (lo, hi)]
    if not bounds[0] < target < bounds[1]:
        raise ValueError('TRAIN ESS target not bracketed; no automatic fallback')
    for _ in range(SETTINGS['temperature_bisections']):
        mid = (lo+hi)/2
        if effective_support(pair, math.exp(mid)) < target:
            lo = mid
        else:
            hi = mid
    temperature = math.exp((lo+hi)/2)
    return dict(temperature=temperature, target_mean_ESS=target,
        initial_mean_ESS=effective_support(pair, temperature), TRAIN_only=True,
        method='bisection of one common temperature on initial three BCE models; not VAL selection',
        per_seed_initial_ESS=[effective_support(p, temperature) for p in pairs],
        bounds_ESS=bounds, no_exact_FPR_equivalence=True)


def distribution(model, x, y, temperature):
    with torch.no_grad():
        z = model(x); good, wrong = z[y == 0], z[y == 1]
        pair = F.softplus(good[None, :]-wrong[:, None])
        w = torch.softmax(pair/temperature, 1)
        h = pair.mean(0); k = math.ceil(SETTINGS['tail_fraction']*len(good))
        return dict(logit_std=z.std(unbiased=False).item(), logit_range=(z.max()-z.min()).item(),
            unique_risks=int(torch.unique(torch.sigmoid(z)).numel()),
            soft_mean_ESS=(1/w.square().sum(1)).mean().item(),
            soft_mean_max_weight=w.max(1).values.mean().item(),
            top2_weight_mass=w.topk(k, dim=1).values.sum(1).mean().item(),
            data_BCE=old.data_loss(z, y, 'bce').item(),
            data_hard=h.topk(k).values.mean().item(), data_soft=soft_loss(z,y,temperature).item())


def fit(raw, labels, parent, kind, temperature, device):
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
    for step in range(SETTINGS['steps']):
        optimizer.zero_grad(); value=objective(model,x,y,kind,temperature); value.backward()
        if not bool(torch.isfinite(value)) or not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError('Nonfinite GPU fit')
        if step % 250 == 0:
            records.append(dict(step=step,objective=value.item(),**distribution(model,x,y,temperature)))
            print('GPU',kind,parent['seed'],'step',step,'objective',value.item(),flush=True)
        optimizer.step()
    optimizer.zero_grad(); value=objective(model,x,y,kind,temperature); value.backward()
    maximum=max(p.grad.abs().max().item() for p in model.parameters())
    if not math.isfinite(value.item()) or not math.isfinite(maximum):
        raise ValueError('Nonfinite final GPU state')
    records.append(dict(step=SETTINGS['steps'],objective=value.item(),**distribution(model,x,y,temperature)))
    exported=dict(seed=parent['seed'],dimensions=19,objective_kind=kind,temperature=temperature,
        normalizer=norm,weights={k:v.detach().cpu().tolist() for k,v in model.state_dict().items()},
        parameters=sum(p.numel() for p in model.parameters()),steps=SETTINGS['steps'],
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


global_cut = old.global_cut


def gate(rows,cuts):
    from crane_project.utils import port_simple_pilot_increment_v1 as base
    from crane_project.utils.port_reliability_paired_ranking_audit_v1 import Ranking
    formal=base.at_cut(rows,'formal',cuts['formal'])['all']['states']; checks=[]
    for seed in SEEDS:
        n='soft_%d'%seed; b='bce_%d'%seed
        d=base.at_cut(rows,n,cuts[n]); control=base.at_cut(rows,b,cuts[b])
        checks.append(dict(seed=seed,protects_CR=d['all']['states']['CR']>=653,
            protects_real=d['domain:real']['states']['CR']>=189,
            formal_FA_gain=formal['FA']-d['all']['states']['FA']>=3,
            paired_BCE_FA_gain=control['all']['states']['FA']-d['all']['states']['FA']>=3,
            same_count_gain=Ranking(rows,n).same_count(formal['FA']+formal['CR'])['tie_bounds']['FA_max']<formal['FA']))
    passed=all(all(v for k,v in c.items() if k!='seed') for c in checks)
    return dict(passed=passed,checks=checks,representative_seed=1701,gradient_gate=False,
        latency_gate=False,automatic_promotion=False,
        TEST_rule='Only fixed1701 three branches and parent after soft passes; no TEST selection')
