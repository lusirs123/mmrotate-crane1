"""Small scorer fitting and on-device response statistics, frozen M untouched."""
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from crane_project.utils import port_reliability_response_stats_v1 as core


def statistics(probability):
    if probability.ndim != 4 or tuple(probability.shape[1:]) != (4, 9, 9):
        raise ValueError('Invalid four response maps')
    p = probability.double()
    p = p/p.sum((2, 3), keepdim=True)
    axis = torch.linspace(-.5, .5, 9, device=p.device, dtype=p.dtype)
    yy, xx = torch.meshgrid(axis, axis)
    mx = (p*xx).sum((2, 3)); my = (p*yy).sum((2, 3))
    entropy = -(p*p.clamp_min(np.finfo(float).tiny).log()).sum((2, 3))/np.log(81.)
    second = (p*((xx-mx[..., None, None]).square()+(yy-my[..., None, None]).square())).sum((2, 3))
    ring = torch.ones((9, 9), device=p.device, dtype=p.dtype); ring[1:-1, 1:-1] = 0
    return torch.stack((entropy, p.flatten(2).max(2)[0], second, (p*ring).sum((2, 3))), -1)


def response_features(probability, prior_probability):
    return (statistics(probability)-statistics(prior_probability)).reshape(-1, 16)


class Risk(nn.Module):
    def __init__(self, seed, dimensions):
        super().__init__()
        if dimensions not in (3, 19):
            raise ValueError('Only fixed3/19 input arms')
        # Pair common descriptor/output initialization exactly across arms.
        torch.manual_seed(seed)
        first, last = nn.Linear(3, 8).double(), nn.Linear(8, 1).double()
        if dimensions == 19:
            enlarged = nn.Linear(19, 8).double()
            with torch.no_grad():
                enlarged.weight.zero_(); enlarged.weight[:, :3].copy_(first.weight)
                enlarged.bias.copy_(first.bias)
            first = enlarged
        self.network = nn.Sequential(first, nn.ReLU(), last)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def objective(model, x, y, weights):
    data = (F.binary_cross_entropy_with_logits(model(x), y, reduction='none')*weights).sum()
    penalty = sum(p.square().sum() for name, p in model.named_parameters() if name.endswith('weight'))
    return data+.5*core.SETTINGS['l2']*penalty


def fit(raw, labels, seed):
    norm = core.normalize_fit(raw, 'TRAIN')
    x = torch.tensor(core.normalize(raw, norm), dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    if y.shape != (len(x),) or not bool(((y == 0) | (y == 1)).all()) or not 0 < y.sum() < len(y):
        raise ValueError('Invalid TRAIN binary labels')
    weights = torch.where(y == 1, .5/y.sum(), .5/(len(y)-y.sum()))
    model = Risk(seed, x.shape[1]); settings = core.SETTINGS
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=settings['max_iterations'],
        max_eval=settings['max_evaluations'], tolerance_grad=settings['gradient_tolerance'],
        tolerance_change=settings['change_tolerance'], history_size=20, line_search_fn='strong_wolfe')
    records = []; started = time.monotonic()
    def closure():
        optimizer.zero_grad(); loss = objective(model, x, y, weights)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        loss.backward()
        maximum = max(float(p.grad.abs().max()) for p in model.parameters())
        if not np.isfinite(maximum):
            raise ValueError('Nonfinite gradient')
        records.append(dict(evaluation=len(records)+1, objective=float(loss.detach()), max_abs_gradient=maximum))
        return loss
    optimizer.step(closure)
    optimizer.zero_grad(); final = objective(model, x, y, weights); final.backward()
    maximum = max(float(p.grad.abs().max()) for p in model.parameters())
    exported = dict(seed=seed, dimensions=x.shape[1], normalizer=norm,
        weights={k: v.detach().tolist() for k, v in model.state_dict().items()},
        parameters=sum(p.numel() for p in model.parameters()),
        optimizer_iterations=int(optimizer.state[next(iter(model.parameters()))]['n_iter']),
        function_evaluations=len(records), max_abs_gradient=maximum,
        objective_initial=records[0]['objective'], objective_final=float(final.detach()),
        converged=bool(np.isfinite(maximum) and maximum <= settings['stationarity_limit']),
        elapsed_seconds=time.monotonic()-started)
    with torch.no_grad():
        expected = model(x).numpy()
    exported['torch_numpy_logit_max_abs'] = float(np.max(np.abs(expected-core.numpy_logits(exported, raw))))
    if exported['torch_numpy_logit_max_abs'] > 1e-10:
        raise ValueError('Model export replay differs')
    return model.eval(), exported, records
