"""Finite data x scorer-form experiment; cached three-feature inputs only."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from crane_project.utils import port_simple_component_reliability_v1 as linear
from crane_project.utils import port_simple_pilot_increment_v1 as base
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import Ranking, bad

SEEDS = (1701, 1702, 1703)
ARMS = ('old', 'augmented')


class SmallRisk(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.network = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
        self.double()

    def forward(self, normalized):
        return self.network(normalized).squeeze(-1)


def prepare(rows, normalizer):
    if len({r['image'] for r in rows}) != len(rows):
        raise ValueError('Duplicate TRAIN identity')
    if any(r['reliability_role'] != 'train' for r in rows):
        raise ValueError('TRAIN only')
    outputs = [r for r in rows if r['pred'] is not None]
    x = np.asarray([linear.descriptor(r['pred'], r['image_size']) for r in outputs], dtype=np.float64)
    y = np.asarray([bad(r) for r in outputs], dtype=np.float64)
    if not 0 < y.sum() < len(y):
        raise ValueError('Both classes required')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-8, 1., scale)
    if not (np.array_equal(mean, np.asarray(normalizer['mean'])) and
            np.array_equal(scale, np.asarray(normalizer['scale']))):
        raise ValueError('Linear/MLP normalization must match exactly per material')
    normalized = (x-mean)/scale
    weights = np.where(y == 1, .5/y.sum(), .5/(len(y)-y.sum()))
    return normalized, y, weights, [r['image'] for r in outputs]


def objective(model, x, y, weights, l2):
    data = (F.binary_cross_entropy_with_logits(model(x), y, reduction='none')*weights).sum()
    penalty = sum((p*p).sum() for name, p in model.named_parameters() if name.endswith('weight'))
    return data+.5*l2*penalty


def fit(rows, normalizer, seed, settings):
    x, y, w, images = prepare(rows, normalizer)
    tx, ty, tw = (torch.tensor(v, dtype=torch.float64) for v in (x, y, w))
    model = SmallRisk(seed)
    if sum(p.numel() for p in model.parameters()) != 41:
        raise ValueError('Unexpected architecture')
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Same deterministic full-batch optimizer/budget for both material arms.
    # Different from the archived linear Newton solver: NOT a pure capacity ablation.
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=settings['max_iterations'],
        max_eval=settings['max_evaluations'], tolerance_grad=settings['gradient_tolerance'],
        tolerance_change=settings['change_tolerance'], history_size=20, line_search_fn='strong_wolfe')
    records = []
    def closure():
        optimizer.zero_grad()
        loss = objective(model, tx, ty, tw, settings['l2'])
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite TRAIN objective')
        loss.backward()
        gradient = max(float(p.grad.abs().max()) for p in model.parameters())
        if not np.isfinite(gradient):
            raise ValueError('Nonfinite TRAIN gradient')
        records.append(dict(evaluation=len(records)+1, objective=float(loss.detach()), max_abs_gradient=gradient))
        return loss
    optimizer.step(closure)
    optimizer.zero_grad(); final = objective(model, tx, ty, tw, settings['l2']); final.backward()
    gradient = max(float(p.grad.abs().max()) for p in model.parameters())
    state = optimizer.state[next(iter(model.parameters()))]
    changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    if not changed or not np.isfinite(float(final.detach())):
        raise ValueError('Invalid or missing model update')
    exported = dict(kind='three_feature_relu8', seed=seed,
        mean=list(normalizer['mean']), scale=list(normalizer['scale']),
        weights={k: v.detach().tolist() for k, v in model.state_dict().items()},
        train_rows=len(y), bad_rows=int(y.sum()), good_rows=int(len(y)-y.sum()), parameter_count=41,
        optimizer_iterations=int(state['n_iter']), function_evaluations=int(state['func_evals']),
        objective_initial=records[0]['objective'], objective_final=float(final.detach()),
        max_abs_gradient=gradient, converged=gradient <= settings['stationarity_limit'],
        convergence_definition='Approximate TRAIN stationarity only, not global optimum or generalization',
        train_identity_sha256=hashlib.sha256(('\n'.join(images)+'\n').encode()).hexdigest())
    model.eval()
    with torch.no_grad():
        expected = torch.sigmoid(model(tx)).numpy()
    replay = numpy_risk(exported, np.asarray([linear.descriptor(r['pred'], r['image_size'])
        for r in rows if r['pred'] is not None]))
    exported['torch_numpy_max_abs'] = float(np.max(np.abs(expected-replay)))
    if exported['torch_numpy_max_abs'] > 1e-12:
        raise ValueError('Saved model replay failed')
    return model, exported, records


def numpy_risk(model, x):
    z = (np.asarray(x, dtype=np.float64)-model['mean'])/model['scale']
    w = model['weights']
    hidden = np.maximum(0., z.dot(np.asarray(w['network.0.weight']).T)+w['network.0.bias'])
    logits = hidden.dot(np.asarray(w['network.2.weight']).T).reshape(-1)+w['network.2.bias'][0]
    return linear.sigmoid(logits)


def add_scores(rows, models):
    outputs = [r for r in rows if r['pred'] is not None]
    x = np.asarray([linear.descriptor(r['pred'], r['image_size']) for r in outputs])
    for name, model in models.items():
        if model.get('kind') != 'three_feature_relu8':
            # Preserve archived scalar dot order/cutoff boundary exactly.
            base.add_scores(rows, {name: model})
            continue
        values = (numpy_risk(model, x) if model.get('kind') == 'three_feature_relu8'
                  else linear.linear_risk(model, x))
        by_id = dict(zip((r['image'] for r in outputs), map(float, values)))
        for r in rows:
            r['audit_risks'][name] = by_id.get(r['image'])


def gate(results, models):
    checks = []
    for seed in SEEDS:
        old, aug = ('mlp_'+a+'_'+str(seed) for a in ARMS)
        o = results[old]['at_frozen_VAL_cut']; a = results[aug]['at_frozen_VAL_cut']
        checks.append(dict(seed=seed,
            beats_old=a['all']['states']['FA'] < o['all']['states']['FA'],
            beats_formal=a['all']['states']['FA'] < results['formal']['at_frozen_VAL_cut']['all']['states']['FA'],
            protects_CR=a['all']['states']['CR'] >= 653,
            protects_real=a['domain:real']['states']['CR'] >= 189,
            TRAIN_stationary=bool(models[old]['converged'] and models[aug]['converged'])))
    passed = all(all(v for k, v in c.items() if k != 'seed') for c in checks)
    return dict(passed=passed, checks=checks, representative_seed=1701 if passed else None,
        selected_method='mlp_augmented_1701' if passed else None,
        selection='All3 seeds; fixed1701 representative, no best seed/epoch',
        TEST_rule='Frozen linear supplement always reported; new MLP TEST only if this gate passes',
        test_repeatedly_exposed=True, automatic_promotion=False)
