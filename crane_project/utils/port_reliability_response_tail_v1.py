"""Fixed input x objective control on saved frozen-M response features."""
import hashlib
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils.port_reliability_response_stats_v1_torch import Risk

VERSION = 'port_reliability_response_tail_v1'
SEEDS = (1701, 1702, 1703)
ARMS = {'A': (3, 'bce'), 'B': (19, 'bce'), 'C': (3, 'tail'), 'D': (19, 'tail')}
SETTINGS = dict(steps=5000, learning_rate=.001, l2=.1, tail_fraction=.02,
                primary_CR=653, minimum_Real_CR=189, minimum_FA_gain=3,
                hidden=8, optimizer='Adam', betas=[.9, .999], eps=1e-8,
                dtype='float64', final_step_only=True)


def data_loss(logits, labels, objective):
    good, wrong = logits[labels == 0], logits[labels == 1]
    if not len(good) or not len(wrong):
        raise ValueError('Both TRAIN classes required')
    if objective == 'bce':
        return .5*(F.softplus(good).mean()+F.softplus(-wrong).mean())
    if objective != 'tail':
        raise ValueError('Unknown fixed objective')
    # Error risk is high. Penalize good risk >= wrong risk; top-k over GOODS.
    h = F.softplus(good[:, None]-wrong[None, :]).mean(1)
    k = math.ceil(SETTINGS['tail_fraction']*len(good))
    return h.topk(k, sorted=False).values.mean()


def objective(model, x, y, kind):
    loss = data_loss(model(x), y, kind)
    penalty = sum(p.square().sum() for n, p in model.named_parameters() if n.endswith('weight'))
    return loss+.5*SETTINGS['l2']*penalty


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit(raw, labels, seed, kind):
    normalizer = response.normalize_fit(raw, 'TRAIN')
    x = torch.tensor(response.normalize(raw, normalizer), dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    if y.shape != (len(x),) or not bool(((y == 0) | (y == 1)).all()):
        raise ValueError('Invalid complete TRAIN labels')
    model = Risk(seed, x.shape[1]); initial = state_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS['learning_rate'],
        betas=tuple(SETTINGS['betas']), eps=SETTINGS['eps'], weight_decay=0)
    records = []; started = time.monotonic()
    for step in range(SETTINGS['steps']):
        optimizer.zero_grad(); value = objective(model, x, y, kind)
        if not bool(torch.isfinite(value)):
            raise ValueError('Nonfinite TRAIN loss')
        value.backward()
        maximum = max(float(p.grad.abs().max()) for p in model.parameters())
        if not np.isfinite(maximum):
            raise ValueError('Nonfinite TRAIN gradient')
        if step % 100 == 0:
            records.append(dict(step=step, objective=float(value.detach()), max_abs_gradient=maximum))
        optimizer.step()
    optimizer.zero_grad(); value = objective(model, x, y, kind); value.backward()
    maximum = max(float(p.grad.abs().max()) for p in model.parameters())
    if not np.isfinite(float(value.detach())) or not np.isfinite(maximum):
        raise ValueError('Nonfinite final TRAIN state')
    records.append(dict(step=SETTINGS['steps'], objective=float(value.detach()), max_abs_gradient=maximum))
    exported = dict(seed=seed, dimensions=x.shape[1], objective_kind=kind, normalizer=normalizer,
        weights={k: v.detach().tolist() for k, v in model.state_dict().items()},
        parameters=sum(p.numel() for p in model.parameters()), steps=SETTINGS['steps'],
        initial_state_sha256=initial, final_state_sha256=state_digest(model),
        objective_initial=records[0]['objective'], objective_final=float(value.detach()),
        max_abs_gradient=maximum, gradient_is_diagnostic_only=True,
        tail_correct_count=math.ceil(SETTINGS['tail_fraction']*int((y == 0).sum())),
        elapsed_seconds=time.monotonic()-started)
    with torch.no_grad():
        expected = model(x).numpy()
        if kind == 'tail':
            good, wrong = model(x)[y == 0], model(x)[y == 1]
            h = F.softplus(good[:, None]-wrong[None, :]).mean(1)
            exported['final_tail_indices_within_goods'] = h.topk(exported['tail_correct_count'], sorted=False).indices.tolist()
    exported['torch_numpy_logit_max_abs'] = float(np.max(np.abs(expected-response.numpy_logits(exported, raw))))
    if exported['torch_numpy_logit_max_abs'] > 1e-10 or initial == exported['final_state_sha256']:
        raise ValueError('Export replay differs or model never updated')
    return model.eval(), exported, records


def global_cut(rows, method):
    """Strictest ONE global whole-tie cutoff meeting both correct protections."""
    from crane_project.utils.port_reliability_paired_ranking_audit_v1 import Ranking
    real = [r for r in rows if r['domain'] == 'real']
    return max(Ranking(rows, method).same_cr(SETTINGS['primary_CR'])['risk_le'],
               Ranking(real, method).same_cr(SETTINGS['minimum_Real_CR'])['risk_le'])


def gate(rows, cuts):
    from crane_project.utils import port_simple_pilot_increment_v1 as base
    from crane_project.utils.port_reliability_paired_ranking_audit_v1 import Ranking
    formal = base.at_cut(rows, 'formal', cuts['formal'])['all']['states']
    checks = []
    for seed in SEEDS:
        name, control = 'D_%d' % seed, 'B_%d' % seed
        d = base.at_cut(rows, name, cuts[name]); b = base.at_cut(rows, control, cuts[control])
        count = Ranking(rows, name).same_count(formal['FA']+formal['CR'])
        checks.append(dict(seed=seed, protects_CR=d['all']['states']['CR'] >= SETTINGS['primary_CR'],
            protects_real=d['domain:real']['states']['CR'] >= SETTINGS['minimum_Real_CR'],
            formal_FA_gain=formal['FA']-d['all']['states']['FA'] >= SETTINGS['minimum_FA_gain'],
            paired_BCE_FA_gain=b['all']['states']['FA']-d['all']['states']['FA'] >= SETTINGS['minimum_FA_gain'],
            same_count_gain=count['tie_bounds']['FA_max'] < formal['FA']))
    passed = all(all(v for k, v in c.items() if k != 'seed') for c in checks)
    return dict(passed=passed, checks=checks, representative_seed=1701,
                gradient_gate=False, latency_gate=False, automatic_promotion=False,
                TEST_rule='If D passes all fixed checks, report all four frozen1701 arms; never tune on TEST')
