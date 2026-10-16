"""Video-held-out five-score stacking; GT is an offline fitting argument only."""
import copy
import math
import time
import numpy as np
import torch
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils import port_reliability_response_tail_v1 as tail
from crane_project.utils import port_reliability_response_warm_v1 as warm
from crane_project.utils.port_reliability_response_stats_v1_torch import Risk

VERSION = 'port_response_fusion_crossfit_v1'
NAMES = ('formal', 'bce_1701', 'soft_1701', 'bce0_1701', 'soft0_1701')
SEQUENCES = ('real_seq01', 'real_seq05', 'real_seq06', 'real_seq12', 'real_seq13', 'sim_seq08')
LINEAR = dict(l2=.1, max_iterations=100, gradient_tolerance=1e-8)
SETTINGS = dict(seed=1701, temperature_parent_seeds=[1701, 1702, 1703],
    parent_steps=5000, continuation_steps=5000, continuation_lr=.001,
    linear=LINEAR, primary_CR=653, minimum_Real_CR=189,
    score_mapping='per-base-model fitting-TRAIN logit mean/std; then meta TRAIN normalization',
    report_seed=1701, final_steps_only=True, no_hyperparameter_search=True)


def folds(sequences):
    sequences = np.asarray(sequences)
    if set(sequences) != set(SEQUENCES):
        raise ValueError('Exact six original TRAIN videos required')
    return [(s, np.flatnonzero(sequences != s), np.flatnonzero(sequences == s)) for s in SEQUENCES]


def simple_logits(model, raw):
    return model['weights'][0] + ((raw[:, :3]-model['mean'])/model['scale']) @ np.asarray(model['weights'][1:])


def logits(models, raw):
    return np.column_stack([simple_logits(models[n], raw) if n == 'formal'
                            else response.numpy_logits(models[n], raw) for n in NAMES])


def mapping_fit(z):
    return response.normalize_fit(z, 'TRAIN')


def continuation(raw, labels, parent, kind, l2, temperature, device, steps=5000):
    if kind not in ('bce', 'soft') or l2 not in (0., .1):
        raise ValueError('Only four preset branches')
    if parent['normalizer'] != response.normalize_fit(raw, 'TRAIN'):
        raise ValueError('Parent must be fitted in this fold')
    model = Risk(1701, 19)
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in parent['weights'].items()})
    initial = tail.state_digest(model)
    if initial != parent['final_state_sha256']:
        raise ValueError('Parent digest mismatch')
    model.to(device)
    x = torch.tensor(response.normalize(raw, parent['normalizer']), dtype=torch.float64, device=device)
    y = torch.tensor(labels, dtype=torch.float64, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8)
    history = []; started = time.monotonic()
    for step in range(steps+1):
        opt.zero_grad()
        data = warm.data_loss(model(x), y, kind, temperature)
        penalty = sum(p.square().sum() for n, p in model.named_parameters() if n.endswith('weight'))
        loss = data + .5*l2*penalty
        loss.backward()
        gradient = max(p.grad.abs().max().item() for p in model.parameters())
        if not math.isfinite(loss.item()) or not math.isfinite(gradient):
            raise ValueError('Nonfinite fit')
        if step % 500 == 0 or step == steps:
            with torch.no_grad():
                z = model(x).cpu().numpy()
            history.append(dict(step=step, objective=loss.item(), max_abs_gradient=gradient,
                logit_std=float(z.std()), TRAIN_FA_at_CR9849=int(sum(z[labels == 1] <= np.sort(z[labels == 0])[math.ceil(.9849*sum(labels == 0))-1]))))
            if step % 1000 == 0:
                print('UPDATE', kind, 'L2', l2, step, loss.item(), flush=True)
        if step == steps:
            break
        opt.step()
    exported = dict(seed=1701, dimensions=19, objective_kind=kind, l2=l2, temperature=temperature,
        normalizer=copy.deepcopy(parent['normalizer']), steps=steps, parameters=169,
        initial_state_sha256=initial, parent_final_sha256=parent['final_state_sha256'],
        final_state_sha256=tail.state_digest(model), optimizer_reset=True,
        weights={k: v.detach().cpu().tolist() for k, v in model.state_dict().items()},
        objective_final=history[-1]['objective'], max_abs_gradient=gradient,
        elapsed_seconds=time.monotonic()-started, history=history, training_device=device)
    with torch.no_grad():
        np.testing.assert_allclose(model(x).cpu().numpy(), response.numpy_logits(exported, raw), rtol=0, atol=1e-10)
    return exported


def fit_linear(x, y):
    """Six-parameter convex balanced logistic fit; same objective for both arms."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[1] != 5 or y.shape != (len(x),) or not np.isin(y, [0, 1]).all() or not 0 < y.sum() < len(y):
        raise ValueError('Five finite TRAIN scores and both labels required')
    norm = response.normalize_fit(x, 'TRAIN')
    design = np.column_stack([np.ones(len(x)), response.normalize(x, norm)])
    weights = np.where(y == 1, .5/y.sum(), .5/(len(y)-y.sum()))
    theta = np.zeros(6)
    for step in range(LINEAR['max_iterations']):
        loss, gradient, hessian = simple.logistic_objective(theta, design, y, weights, LINEAR['l2'])
        if abs(gradient).max() <= LINEAR['gradient_tolerance']:
            break
        direction = np.linalg.solve(hessian, gradient)
        for backtrack in range(30):
            candidate = theta-(.5**backtrack)*direction
            if simple.logistic_objective(candidate, design, y, weights, LINEAR['l2'])[0] <= loss-1e-4*(.5**backtrack)*gradient.dot(direction):
                theta = candidate
                break
        else:
            raise ValueError('Fixed Newton line search failed')
    loss, gradient, _ = simple.logistic_objective(theta, design, y, weights, LINEAR['l2'])
    if abs(gradient).max() > LINEAR['gradient_tolerance']:
        raise ValueError('Convex meta fit did not converge')
    return dict(weights=theta.tolist(), normalizer=norm, objective=loss,
        max_abs_gradient=float(abs(gradient).max()), iterations=step+1,
        samples=len(y), wrong=int(y.sum()), correct=int(sum(y == 0)), parameters=6, settings=LINEAR)


def fusion_risk(model, mapped_scores):
    x = response.normalize(mapped_scores, model['normalizer'])
    return simple.sigmoid(model['weights'][0]+x@np.asarray(model['weights'][1:]))


def conditional_summary(z, labels, sequences):
    """No sign flip or rule selection; within-video AUC plus pooled AUC."""
    def auc(a, y):
        good = np.sort(a[y == 0]); wrong = a[y == 1]
        if not len(good) or not len(wrong):
            return None
        lo = np.searchsorted(good, wrong, side='left'); hi = np.searchsorted(good, wrong, side='right')
        return float((lo+.5*(hi-lo)).sum()/(len(good)*len(wrong)))
    return {g: dict(correct=int(sum(labels[indices] == 0)), wrong=int(sum(labels[indices] == 1)),
                   AUROC={n: auc(z[indices, j], labels[indices]) for j, n in enumerate(NAMES)})
            for g, indices in [('all', np.arange(len(z)))]+[(s, np.flatnonzero(np.asarray(sequences) == s)) for s in sorted(set(sequences))]}
