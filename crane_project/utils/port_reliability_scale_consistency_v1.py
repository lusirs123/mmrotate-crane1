"""Bounded size-only reliability evidence. Online functions never receive GT."""
from copy import deepcopy
import math

import numpy as np

from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.utils import port_reliability_separability_v1 as sep
from crane_project.utils import port_reliability_state_continuity_v1 as states
from crane_project.utils.port_reliability_tradeoff_v1 import required_good

VERSION = 'port_reliability_scale_consistency_v1'
FEATURES = simple.FEATURES+('max_abs_log_edge_disagreement', 'auxiliary_missing')
FIT = dict(l2=.1, max_iterations=100, gradient_tolerance=1e-8)
CONTROLS = ('refit_simple', 'full_simple', 'score_only')


def descriptor(main, auxiliary, image_size):
    """Both boxes already restored by native sx/sy; do not scale them again.

    Canonical edge sorting handles w/h swaps; pi-periodic angle is irrelevant
    to size comparison. A missing auxiliary is an explicit feature, not a veto.
    """
    if main is None:
        if auxiliary is not None:
            simple.prediction(auxiliary)
        return None
    base = simple.descriptor(main, image_size)
    if auxiliary is None:
        return np.r_[base, 0., 1.]
    a, b = simple.prediction(main), simple.prediction(auxiliary)
    ae, be = sorted(a[2:4]), sorted(b[2:4])
    # Difference of logs avoids overflowing edge ratios.
    difference = max(abs(math.log(x)-math.log(y)) for x, y in zip(ae, be))
    return np.r_[base, difference, 0.]


def fit(features, labels):
    """Fixed zero-init/class-balanced Newton; only dimensions 3 or 5 allowed."""
    x, y = np.asarray(features, dtype=float), np.asarray(labels, dtype=float)
    if (x.ndim != 2 or x.shape[1] not in (3, 5) or y.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isin(y, [0., 1.]).all()):
        raise ValueError('Invalid fixed fitting inputs')
    bad, good = int(y.sum()), int(len(y)-y.sum())
    if not bad or not good:
        raise ValueError('Both TRAIN error classes required')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-8, 1., scale)
    design = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    sw = np.where(y == 1, .5/bad, .5/good)
    w = np.zeros(x.shape[1]+1)
    initial = simple.logistic_objective(w, design, y, sw, FIT['l2'])[0]
    for step in range(FIT['max_iterations']):
        loss, grad, h = simple.logistic_objective(w, design, y, sw, FIT['l2'])
        if np.max(np.abs(grad)) <= FIT['gradient_tolerance']:
            break
        direction = np.linalg.solve(h, grad)
        for k in range(30):
            candidate = w-.5**k*direction
            value = simple.logistic_objective(candidate, design, y, sw, FIT['l2'])[0]
            if value <= loss-1e-4*.5**k*grad.dot(direction):
                w = candidate
                break
        else:
            raise RuntimeError('Fixed Newton line search failed')
    loss, grad, _ = simple.logistic_objective(w, design, y, sw, FIT['l2'])
    if not np.isfinite(w).all() or np.max(np.abs(grad)) > FIT['gradient_tolerance']:
        raise RuntimeError('Fixed fit did not converge; no retry/search')
    return dict(feature_names=list(FEATURES[:x.shape[1]]), mean=mean.tolist(),
                scale=scale.tolist(), weights=w.tolist(), settings=deepcopy(FIT),
                train_rows=len(y), bad_rows=bad, good_rows=good, iterations=step+1,
                initial_objective=initial, final_objective=loss,
                max_abs_gradient=float(np.max(np.abs(grad))), initialization='all_zero')


def risk(model, features):
    names = model['feature_names']
    if names not in (list(FEATURES), list(simple.FEATURES)) or model['settings'] != FIT:
        raise ValueError('Invalid fixed model contract')
    mean, scale, w = [np.asarray(model[k], dtype=float) for k in ('mean', 'scale', 'weights')]
    x = np.asarray(features, dtype=float)
    n = len(names)
    if (mean.shape != (n,) or scale.shape != (n,) or w.shape != (n+1,)
            or x.ndim not in (1, 2) or x.shape[-1] != n
            or not np.isfinite(np.r_[mean, scale, w]).all() or not np.isfinite(x).all()
            or np.any(scale <= 0)):
        raise ValueError('Invalid model or features')
    return simple.sigmoid(w[0]+((x-mean)/scale).dot(w[1:]))


def decide(model, cutoff, original_decision, main, auxiliary, image_size):
    """Copy the frozen three-flag decision; modify size alone, no GT input."""
    if (not math.isfinite(cutoff) or not 0 <= cutoff <= 1
            or original_decision['final_box_original'] != main
            or original_decision['center_accepted'] != (main is not None)):
        raise ValueError('Invalid original decision or cutoff')
    result = deepcopy(original_decision)
    if main is None:
        if (result['size_accepted'] or result['angle_accepted']
                or any(v is not None for v in result['risks'].values())):
            raise ValueError('Missing output cannot have a quality decision')
        return result
    value = float(risk(model, descriptor(main, auxiliary, image_size)))
    result['risks']['size'] = value
    result['size_accepted'] = value <= cutoff
    return result


def cutoff(rows, risks):
    """TRAIN probe diagnostic, NOT deployable or independent calibration."""
    if not rows or any(r['reliability_role'] != 'train' for r in rows):
        raise ValueError('TRAIN-only capability check')
    requirements = {}
    for name, group in ab.groups(rows).items():
        good = sorted(risks[r['image']] for r in group
                      if r['pred'] is not None and not ab.size_bad(r))
        needed = required_good(len(good), .95)
        requirements[name] = dict(good=len(good), required=needed,
                                  risk_le=good[needed-1] if needed else None)
    points = [x['risk_le'] for x in requirements.values() if x['risk_le'] is not None]
    if not points:
        raise ValueError('No correct observations')
    return dict(risk_le=max(points), requirements=requirements, one_global_cutoff=True,
                GT_diagnostic_only=True, deployable=False, independent_validation=False)


def describe(rows, scores, threshold):
    result = {}
    for name, group in ab.groups(rows).items():
        out = [r for r in group if r['pred'] is not None]
        ids = [r['image'] for r in out]
        bad = [ab.size_bad(r) for r in out]
        accepted = {i for i in ids if scores['scale_consistency'][i] <= threshold}
        value = states.compare(group, accepted, scores, CONTROLS)
        value['rank'] = {m: sep.rank_metrics([scores[m][i] for i in ids], bad) for m in scores}
        value['control_tie_bounds'] = {}
        value['matched_CR_controls'] = {}
        for m in CONTROLS:
            k = len(accepted)
            value['control_tie_bounds'][m] = sep.same_count_reference(
                [scores[m][i] for i in ids], bad, ids, [k])[k]
            # First whole-tie prefix attaining candidate CR. GT chooses offline
            # comparison point only, never tie membership or an online rule.
            target = value['actual']['states']['CR']
            ranked = sorted(out, key=lambda r: (scores[m][r['image']], r['image']))
            good = 0; bound = None
            if target:
                for r in ranked:
                    good += int(not ab.size_bad(r))
                    if good >= target:
                        bound = scores[m][r['image']]; break
            kept = {i for i in ids if bound is not None and scores[m][i] <= bound}
            value['matched_CR_controls'][m] = states.summarize(group, kept)
        result[name] = value
    return result


def gate(stats):
    """Finite TRAIN capability gate; failure ends this version before VAL."""
    failures = []
    for name, value in stats.items():
        a = value['actual']; count = a['states']; good = count['CR']+count['FR']
        if count['CR'] < required_good(good, .95):
            failures.append(dict(group=name, check='correct_retention'))
        for m in CONTROLS:
            b = value['same_count_controls'][m]
            bound = value['control_tie_bounds'][m]['bad_min_over_tie']
            strict = name in ('all', 'domain:real')
            insufficient = count['FA'] >= bound if strict else count['FA'] > bound
            if insufficient:
                failures.append(dict(group=name, control=m, check='same_count_FA',
                                     candidate=count['FA'], control_best_tie_FA=bound,
                                     strict=strict))
            if a['runs']['correct_rejection']['longest'] > b['runs']['correct_rejection']['longest']:
                failures.append(dict(group=name, control=m, check='longest_FR'))
    return dict(passed=not failures, failures=failures, selected=False,
                TEST_used=False, gate_scope='TRAIN_capability_not_VAL_adoption')
