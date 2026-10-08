"""ReDC-inspired SIZE RANKING, not a reproduction or calibrated probability.

Online descriptors/decisions accept no GT. Only offline labels/evaluation do.
No geometry, center, direction, score, missing-frame or old-policy updates.
"""
from copy import deepcopy
import math

import numpy as np

from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.utils import port_reliability_separability_v1 as sep
from crane_project.utils import port_reliability_state_continuity_v1 as states
from crane_project.utils.port_reliability_tradeoff_v1 import required_good

VERSION = 'port_reliability_redc_size_v1'
ARMS = ('linear', 'redc')
CONTROLS = ('full_simple', 'score_only', 'linear')
SETTINGS = dict(seed=1701, epochs=100, batch_size=256, lr=.001,
                weight_decay=.0001, clip_norm=5., temperature_floor=.25,
                feature_channels=32, roi_size=9, pool_size=3,
                hidden=[16, 8], correct_retention=.95)


def pooled_feature(value):
    """Channel-major 3x3 nonoverlapping means of frozen 32x9x9 stem."""
    x = np.asarray(value)
    if x.shape != (32, 9, 9) or not np.isfinite(x).all():
        raise ValueError('Expected finite frozen 32x9x9 stem')
    return x.astype(np.float64).reshape(32, 3, 3, 3, 3).mean((2, 4)).reshape(288)


def descriptor(final_box_original, image_size, pooled):
    if final_box_original is None:
        if pooled is not None:
            raise ValueError('Missing output cannot have quality features')
        return None
    f = np.asarray(pooled, dtype=np.float64)
    if f.shape != (288,) or not np.isfinite(f).all():
        raise ValueError('Invalid frozen geometry feature')
    return np.r_[simple.descriptor(final_box_original, image_size), f]


def normalization(train_features):
    x = np.asarray(train_features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 291 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Standardization requires complete finite TRAIN descriptors')
    return dict(mean=x.mean(0).tolist(), scale=np.where(x.std(0)<1e-8, 1., x.std(0)).tolist())


def normalize(features, normalizer):
    x = np.asarray(features, dtype=np.float64)
    mean, scale = (np.asarray(normalizer[k], dtype=np.float64) for k in ('mean', 'scale'))
    if (x.shape[-1:] != (291,) or mean.shape != (291,) or scale.shape != (291,)
            or not np.isfinite(x).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale<=0)):
        raise ValueError('Invalid fixed normalization')
    return (x-mean)/scale


def class_weights(labels):
    y = np.asarray(labels, dtype=np.float64)
    if y.ndim != 1 or not np.isin(y, [0., 1.]).all() or not 0<y.sum()<len(y):
        raise ValueError('Both genuine TRAIN classes are required')
    # Mean minibatch loss is an unbiased estimate of the FULL class-balanced
    # objective. Never re-normalize weights by the current minibatch's classes.
    return np.where(y==1, len(y)/(2*y.sum()), len(y)/(2*(len(y)-y.sum())))


def numpy_logits(model, features, normalizer, arm):
    """Independent CPU replay of serialized final Torch head."""
    x = np.asarray(features, dtype=np.float64)
    f = normalize(x, normalizer)
    if arm == 'linear':
        return (f @ np.asarray(model['output.weight']).T + model['output.bias'])[..., 0]
    if arm != 'redc':
        raise ValueError('Unknown frozen arm')
    h = f[..., 1:]
    for n in (0, 2, 4):
        h = h @ np.asarray(model['network.%d.weight'%n]).T + model['network.%d.bias'%n]
        if n != 4:
            h = np.maximum(h, 0.)
    temperature = SETTINGS['temperature_floor'] + np.logaddexp(0., h[..., 0])
    return -(x[..., 0]/temperature + float(model['beta'][0]))


def decide(original_decision, risk, cutoff):
    """GT/sequence/domain-free replacement of only size risk/flag."""
    d = deepcopy(original_decision)
    present = d['final_box_original'] is not None
    if d['center_accepted'] != present:
        raise ValueError('Center identity changed')
    if not present:
        if risk is not None or d['size_accepted'] or d['angle_accepted']:
            raise ValueError('No box means no quality decision')
        return d
    if (risk is None or not math.isfinite(risk) or not 0<=risk<=1
            or not math.isfinite(cutoff) or not 0<=cutoff<=1):
        raise ValueError('Invalid frozen size risk/threshold')
    d['risks']['size'] = float(risk)
    d['size_accepted'] = bool(risk<=cutoff)
    return d


def describe(rows, scores, arm, cutoff):
    result = {}
    controls = tuple(k for k in scores if k != arm)
    for name, group in ab.groups(rows).items():
        out = [r for r in group if r['pred'] is not None]
        ids = [r['image'] for r in out]
        bad = [ab.size_bad(r) for r in out]
        accepted = {i for i in ids if scores[arm][i]<=cutoff}
        value = states.compare(group, accepted, scores, controls)
        value['rank'] = {m:sep.rank_metrics([scores[m][i] for i in ids],bad) for m in scores}
        value['control_tie_bounds'] = {}
        value['matched_CR_controls'] = {}
        for m in controls:
            k = len(accepted)
            value['control_tie_bounds'][m] = sep.same_count_reference(
                [scores[m][i] for i in ids],bad,ids,[k])[k]
            target = value['actual']['states']['CR']
            ranked = sorted(out,key=lambda r:(scores[m][r['image']],r['image']))
            good = 0; bound = None
            if target:
                for r in ranked:
                    good += int(not ab.size_bad(r))
                    if good >= target:
                        bound = scores[m][r['image']]; break
            kept = {i for i in ids if bound is not None and scores[m][i]<=bound}
            matched = states.summarize(group,kept)
            matched.update(offline_GT_reference_only=True, whole_boundary_tie=True,
                           requested_CR=target, exact_CR=matched['states']['CR']==target)
            value['matched_CR_controls'][m] = matched
        result[name] = value
    return result


def gate(statistics):
    """Only predeclared ReDC candidate can pass. Never select linear post hoc."""
    failures = []
    for group, value in statistics.items():
        a = value['actual']; c = a['states']
        if c['CR'] < required_good(c['CR']+c['FR'],.95):
            failures.append(dict(group=group,check='correct_retention'))
        for name in CONTROLS:
            bounds = value['control_tie_bounds'][name]
            matched = value['matched_CR_controls'][name]
            if c['FA'] > bounds['bad_min_over_tie']:
                failures.append(dict(group=group,control=name,check='same_count_FA_nonincrease'))
            if group=='all' and c['FA']>=bounds['bad_min_over_tie']:
                failures.append(dict(group=group,control=name,check='strict_overall_same_count_FA_gain'))
            if c['FA'] > matched['states']['FA']:
                failures.append(dict(group=group,control=name,check='same_CR_FA_nonincrease'))
            if group=='all' and c['FA']>=matched['states']['FA']:
                failures.append(dict(group=group,control=name,check='strict_overall_same_CR_FA_gain'))
            if not matched['exact_CR']:
                # Cannot establish identical retention with an unsplittable
                # tie. Report conservatively, never split its members by GT.
                failures.append(dict(group=group,control=name,check='same_CR_tie_inconclusive'))
            if a['runs']['correct_rejection']['longest']>matched['runs']['correct_rejection']['longest']:
                failures.append(dict(group=group,control=name,check='longest_FR_same_CR_nonincrease'))
    return dict(passed=not failures,failures=failures,gate_role='VAL_development_not_independent_test',
                selected_arm='redc' if not failures else None,TEST_used=False)
