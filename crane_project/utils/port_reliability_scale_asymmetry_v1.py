"""Size-only main/auxiliary role evidence; GT is confined to offline reports."""
from copy import deepcopy
import math

import numpy as np

from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.utils import port_reliability_separability_v1 as sep
from crane_project.utils import port_reliability_state_continuity_v1 as states
from crane_project.utils import port_reliability_scale_consistency_v1 as old
from crane_project.utils.port_reliability_tradeoff_v1 import required_good

VERSION = 'port_reliability_scale_asymmetry_v1'
ARM = 'scale_asymmetry'
FEATURES = old.FEATURES+('signed_log_long_aux_minus_main',
                       'signed_log_short_aux_minus_main', 'score_logit_aux_minus_main')
FIT = dict(l2=.1, max_iterations=100, gradient_tolerance=1e-8)
CONTROLS = ('scale_consistency',)+old.CONTROLS
COMBINATIONS = ('main_good_aux_good', 'main_good_aux_bad', 'main_bad_aux_good',
                'main_bad_aux_bad', 'main_good_aux_missing', 'main_bad_aux_missing')


def descriptor(main, auxiliary, image_size):
    """Predictions are already in original coordinates after actual sx/sy.

    The first five values exactly preserve the old descriptor. Canonical
    long/short sorting handles swaps; no angle or nominal scale correction.
    """
    base = old.descriptor(main, auxiliary, image_size)
    if base is None:
        return None
    if auxiliary is None:
        return np.r_[base, 0., 0., 0.]
    a, b = simple.prediction(main), simple.prediction(auxiliary)
    short_m, long_m = sorted(a[2:4]); short_a, long_a = sorted(b[2:4])
    qm, qa = (float(np.clip(p[5], 1e-6, 1-1e-6)) for p in (a, b))
    dq = (math.log(qa)-math.log1p(-qa))-(math.log(qm)-math.log1p(-qm))
    return np.r_[base, math.log(long_a)-math.log(long_m),
                 math.log(short_a)-math.log(short_m), dq]


def fit(features, labels):
    """Fixed zero-init/class-balanced Newton; exactly eight dimensions allowed."""
    x, y = np.asarray(features, dtype=float), np.asarray(labels, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 8 or y.shape != (len(x),)
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
    if names != list(FEATURES) or model['settings'] != FIT:
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
        descriptor(main, auxiliary, image_size)
        return result
    value = float(risk(model, descriptor(main, auxiliary, image_size)))
    result['risks']['size'] = value
    result['size_accepted'] = value <= cutoff
    return result


# Whole-tie cutoff uses the same predeclared TRAIN-only retention contract.
cutoff = old.cutoff


def combination(row):
    """GT is used AFTER whole-group selection, never in online scoring."""
    if row['pred'] is None:
        return None
    main = 'bad' if ab.size_bad(row) else 'good'
    auxiliary = row['auxiliary_pred_original']
    aux = ('missing' if auxiliary is None else
           'bad' if simple.geometry_errors(row['gt'], auxiliary)['size_max_relative'] > .1 else 'good')
    return 'main_'+main+'_aux_'+aux


def combination_summary(rows, accepted):
    ids = {r['image'] for r in rows if r['pred'] is not None}
    if not set(accepted) <= ids:
        raise ValueError('Combination selection contains missing/unknown main output')
    result = {}
    for key in COMBINATIONS:
        group = [r for r in rows if combination(r) == key]
        counts = {s: sum(states.size_state(r, r['image'] in accepted) == s for r in group)
                  for s in ('FA', 'FR', 'ED', 'CR')}
        result[key] = dict(support=len(group), states=counts,
            false_acceptance_rate=sep.ratio(counts['FA'], counts['FA']+counts['ED']),
            false_rejection_rate=sep.ratio(counts['FR'], counts['FR']+counts['CR']))
    return result


def describe(rows, scores, threshold):
    result = {}
    for name, group in ab.groups(rows).items():
        out = [r for r in group if r['pred'] is not None]
        ids = [r['image'] for r in out]; bad = [ab.size_bad(r) for r in out]
        accepted = {i for i in ids if scores[ARM][i] <= threshold}
        value = states.compare(group, accepted, scores, CONTROLS)
        value['rank'] = {m: sep.rank_metrics([scores[m][i] for i in ids], bad) for m in scores}
        value['control_tie_bounds'] = {}; value['matched_CR_controls'] = {}
        value['combinations'] = dict(actual=combination_summary(group, accepted),
                                    same_count_controls={})
        for m in CONTROLS:
            k = len(accepted)
            value['control_tie_bounds'][m] = sep.same_count_reference(
                [scores[m][i] for i in ids], bad, ids, [k])[k]
            # Select on the ENTIRE group first, with GT-free image tie order.
            ranked = sorted(out, key=lambda r: (scores[m][r['image']], r['image']))
            kept = {r['image'] for r in ranked[:k]}
            value['combinations']['same_count_controls'][m] = combination_summary(group, kept)
            target = value['actual']['states']['CR']; good = 0; bound = None
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
    """Predeclared exposed-TRAIN diagnostic; passing never adopts a policy."""
    failures = []
    for name, value in stats.items():
        a = value['actual']; count = a['states']; good = count['CR']+count['FR']
        if count['CR'] < required_good(good, .95):
            failures.append(dict(group=name, check='correct_retention'))
        for m in CONTROLS:
            b = value['same_count_controls'][m]
            bound = value['control_tie_bounds'][m]['bad_min_over_tie']
            strict = name in ('all', 'domain:real')
            if (count['FA'] >= bound if strict else count['FA'] > bound):
                failures.append(dict(group=name, control=m, check='same_count_FA',
                    candidate=count['FA'], control_best_tie_FA=bound, strict=strict))
            if a['runs']['correct_rejection']['longest'] > b['runs']['correct_rejection']['longest']:
                failures.append(dict(group=name, control=m, check='longest_FR'))
        combos = value['combinations']
        for key, state in (('main_good_aux_bad', 'FR'), ('main_bad_aux_good', 'FA'),
                           ('main_bad_aux_bad', 'FA')):
            candidate = combos['actual'][key]['states'][state]
            control = combos['same_count_controls']['scale_consistency'][key]['states'][state]
            if candidate > control:
                failures.append(dict(group=name, control='scale_consistency',
                    check='combination_'+state, combination=key, candidate=candidate, control_count=control))
    return dict(passed=not failures, failures=failures, selected=False,
        TEST_used=False, gate_scope='exposed_TRAIN_capability_not_VAL_adoption')
