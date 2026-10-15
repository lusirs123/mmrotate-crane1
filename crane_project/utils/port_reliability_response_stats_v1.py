"""Frozen midpoint response evidence; no GT or video identity in online APIs."""
import math
import numpy as np
from crane_project.utils import port_simple_component_reliability_v1 as simple

VERSION = 'port_reliability_response_stats_v1'
SEEDS = (1701, 1702, 1703)
STAT_NAMES = ('entropy_normalized', 'peak', 'central_second_moment', 'outer_ring_mass')
SETTINGS = dict(l2=.1, max_iterations=2000, max_evaluations=2500,
                gradient_tolerance=1e-8, change_tolerance=1e-12,
                stationarity_limit=1e-5, primary_CR=653, minimum_Real_CR=189,
                minimum_FA_gain=3, hidden=8)


def statistics(probability):
    """N x 4 x 9 x 9 -> N x 4 x 4. Fixed ROI units, no confidence claim."""
    p = np.asarray(probability, dtype=np.float64)
    if (p.ndim != 4 or p.shape[1:] != (4, 9, 9) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum((2, 3)), 1., atol=2e-6, rtol=0)):
        raise ValueError('Four normalized finite 9x9 responses required')
    # Normalize FP32 summation drift before computing float64 statistics.
    p = p/p.sum((2, 3), keepdims=True)
    axis = np.linspace(-.5, .5, 9)
    xx, yy = np.meshgrid(axis, axis)
    mx = (p*xx).sum((2, 3)); my = (p*yy).sum((2, 3))
    entropy = -(p*np.log(np.maximum(p, np.finfo(float).tiny))).sum((2, 3))/math.log(81)
    second = (p*((xx-mx[..., None, None])**2+(yy-my[..., None, None])**2)).sum((2, 3))
    ring = np.ones((9, 9)); ring[1:-1, 1:-1] = 0
    return np.stack((entropy, p.max((2, 3)), second, (p*ring).sum((2, 3))), -1)


def response_features(probability, prior_probability):
    if np.shape(probability) != np.shape(prior_probability):
        raise ValueError('Prior and response pairing mismatch')
    return (statistics(probability)-statistics(prior_probability)).reshape(-1, 16)


def normalize_fit(x, role):
    x = np.asarray(x, dtype=np.float64)
    if role != 'TRAIN' or x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Complete finite TRAIN features only')
    scale = x.std(0)
    return dict(mean=x.mean(0).tolist(), scale=np.where(scale < 1e-8, 1., scale).tolist())


def normalize(x, normalizer):
    x = np.asarray(x, dtype=np.float64)
    mean, scale = np.asarray(normalizer['mean']), np.asarray(normalizer['scale'])
    if (x.ndim != 2 or x.shape[1:] != mean.shape or mean.shape != scale.shape
            or not np.isfinite(x).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError('Invalid features/normalizer')
    return (x-mean)/scale


def numpy_logits(model, raw):
    x = normalize(raw, model['normalizer']); w = model['weights']
    h = np.maximum(0, x@np.asarray(w['network.0.weight']).T+w['network.0.bias'])
    return (h@np.asarray(w['network.2.weight']).T).ravel()+w['network.2.bias'][0]


def numpy_risk(model, raw):
    return simple.sigmoid(numpy_logits(model, raw))


def size_decision(original_decision, risk, cutoff):
    """Only replace the size-use flag. Preserve every other baseline field."""
    from copy import deepcopy
    value = deepcopy(original_decision)
    if value['final_box_original'] is None:
        if risk is not None:
            raise ValueError('MISSING has no risk')
        value['size_accepted'] = False
    else:
        if risk is None or not math.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError('Invalid size risk')
        value['size_accepted'] = bool(cutoff is not None and risk <= cutoff)
    return value


def gate(val, models, cuts):
    from crane_project.utils.port_reliability_paired_ranking_audit_v1 import Ranking
    from crane_project.utils import port_simple_pilot_increment_v1 as base
    formal = base.at_cut(val, 'formal', cuts['formal'])['all']['states']
    checks = []
    for seed in SEEDS:
        control, enhanced = ('%s_%d' % (a, seed) for a in ('control', 'response'))
        c = base.at_cut(val, control, cuts[control]); a = base.at_cut(val, enhanced, cuts[enhanced])
        state = a['all']['states']
        same_count = Ranking(val, enhanced).same_count(formal['FA']+formal['CR'])
        checks.append(dict(seed=seed,
            protects_CR=state['CR'] >= SETTINGS['primary_CR'],
            protects_real=a['domain:real']['states']['CR'] >= SETTINGS['minimum_Real_CR'],
            meaningful_formal_gain=formal['FA']-state['FA'] >= SETTINGS['minimum_FA_gain'],
            meaningful_control_gain=c['all']['states']['FA']-state['FA'] >= SETTINGS['minimum_FA_gain'],
            same_count_gain=same_count['tie_bounds']['FA_max'] < formal['FA'],
            stationary=models[control]['converged'] and models[enhanced]['converged']))
    passed = all(all(v for k, v in c.items() if k != 'seed') for c in checks)
    return dict(passed=passed, checks=checks, representative_seed=1701 if passed else None,
                automatic_promotion=False, TEST_rule='Only fixed1701 pair if VAL passes; no TEST tuning')
