"""One fixed TRAIN-only hard-error weighting contrast over existing four inputs."""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_reliability_partial_auc_v1 as parent

VERSION = 'port_reliability_hard_error_pauc_v1'
METHOD = 'hard_error_pauc4'
CONTROL = 'uniform_pauc4'
SETTINGS = dict(parent.SETTINGS, hard_fraction=.25, hard_multiplier=2.)
FALLBACK_CUTOFF = .4094748402105834
FUSION_CUTOFF = .5317310050227115
level = parent.base.level


def data(rows, base_models):
    if not rows or any(r['reliability_role'] != 'train' or r['split'] not in ('train', 'train_sim') for r in rows):
        raise ValueError('Original TRAIN only')
    part = [r for r in rows if r['pred'] is not None and r['temporal']['available']]
    m = base_models['fusion4']
    if [r['image'] for r in part] != m['train_images']:
        raise ValueError('Paired TRAIN membership changed')
    z = (np.array([parent.base.features(r['pred'], r['image_size'], r['temporal'], 4) for r in part]) - m['mean']) / m['scale']
    y = np.array([bool(level.metrics.bad(r)) for r in part])
    if not y.any() or y.all(): raise ValueError('Both genuine classes required')
    return part, z, y


def hard_set(rows):
    if any(r['reliability_role'] != 'train' or r['split'] not in ('train', 'train_sim') for r in rows):
        raise ValueError('TRAIN-only hard mining')
    wrong = [r for r in rows if r['temporal']['available'] and level.metrics.bad(r) is True]
    if not wrong: raise ValueError('No true TRAIN errors')
    risks = [r['risks']['partial_auc4'] for r in wrong]
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in risks): raise ValueError('Invalid frozen TRAIN score')
    n = math.ceil(SETTINGS['hard_fraction'] * len(wrong))
    boundary = sorted(risks)[n - 1]
    selected = [r['image'] for r in wrong if r['risks']['partial_auc4'] <= boundary]
    return dict(images=selected, wrong_images=[r['image'] for r in wrong], wrong=len(wrong),
        nominal_count=n, actual_count=len(selected), TRAIN_risk_boundary=boundary,
        tied_boundary_all_included=True, no_VAL_threshold_used=True,
        hard_fraction=SETTINGS['hard_fraction'], multiplier=SETTINGS['hard_multiplier'])


def objective(v, good, wrong, q):
    v, good, wrong, q = map(lambda x: np.asarray(x, dtype=float), (v, good, wrong, q))
    if v.shape != (5,) or good.ndim != 2 or wrong.ndim != 2 or good.shape[1] != 4 or wrong.shape[1] != 4 or not len(good) or not len(wrong) or q.shape != (len(wrong),):
        raise ValueError('Fixed four slopes, tail threshold and two classes required')
    if not all(np.isfinite(x).all() for x in (v, good, wrong, q)) or (q <= 0).any() or abs(q.sum() - 1) > 1e-12:
        raise ValueError('Finite normalized positive error weights required')
    d = good[:, None, :] - wrong[None, :, :]
    u = d @ v[:4]; prob = parent.base.simple.sigmoid(u)
    h = np.logaddexp(0., u) @ q
    jac = np.einsum('gb,gbi,b->gi', prob, d, q)
    hes = np.einsum('gb,gbi,gbj,b->gij', prob * (1 - prob), d, d, q)
    T, beta, lam = (SETTINGS[k] for k in ('temperature', 'beta', 'l2'))
    p = parent.base.simple.sigmoid((h - v[4]) / T)
    loss = float(v[4] + T * np.logaddexp(0., (h - v[4]) / T).mean() / beta + .5 * lam * v[:4].dot(v[:4]))
    grad = np.r_[np.mean(p[:, None] * jac, axis=0) / beta + lam * v[:4], 1 - p.mean() / beta]
    outer = np.column_stack((jac, -np.ones(len(good))))
    H = np.einsum('g,gi,gj->ij', p * (1 - p) / (beta * T * len(good)), outer, outer)
    H[:4, :4] += np.einsum('g,gij->ij', p / (beta * len(good)), hes) + lam * np.eye(4)
    return loss, grad, H


def fit(rows, base_models, frozen_hard, weighted):
    part, z, y = data(rows, base_models)
    if hard_set(rows) != frozen_hard: raise ValueError('Fixed hard set changed')
    bad_rows = [r for r, bad in zip(part, y) if bad]
    hard = set(frozen_hard['images'])
    raw = np.array([SETTINGS['hard_multiplier'] if weighted and r['image'] in hard else 1. for r in bad_rows])
    q = raw / raw.sum()
    v = np.r_[np.zeros(4), math.log(2) + SETTINGS['temperature'] * math.log(19)]
    initial = objective(v, z[~y], z[y], q)[0]; trace = []
    for step in range(SETTINGS['max_iterations']):
        loss, grad, H = objective(v, z[~y], z[y], q)
        trace.append(dict(step=step, objective=loss, max_gradient=float(abs(grad).max())))
        if abs(grad).max() <= SETTINGS['gradient_tolerance']: break
        direction = np.linalg.solve(H, grad)
        for k in range(30):
            trial = v - .5 ** k * direction
            if objective(trial, z[~y], z[y], q)[0] <= loss - 1e-4 * .5 ** k * grad.dot(direction):
                v = trial; break
        else: raise RuntimeError('Fixed line search failed')
    loss, grad, _ = objective(v, z[~y], z[y], q)
    if abs(grad).max() > SETTINGS['gradient_tolerance']: raise RuntimeError('Fixed fitting budget exhausted')
    m = base_models['fusion4']
    return dict(weights=[m['weights'][0]] + v[:4].tolist(), tail_threshold=float(v[4]),
        mean=deepcopy(m['mean']), scale=deepcopy(m['scale']), train_images=[r['image'] for r in part],
        wrong_images=[r['image'] for r in bad_rows], error_weights=q.tolist(), weighted=weighted,
        train_rows=len(part), bad_rows=int(y.sum()), trace=trace, settings=deepcopy(SETTINGS),
        initial_objective=initial, final_objective=loss, max_abs_gradient=float(abs(grad).max()))


def score(rows, models):
    result = deepcopy(rows)
    for r in result:
        for name, m in models.items():
            value = r['risks']['partial_auc4']
            if r['pred'] is not None and r['temporal']['available']:
                value = float(parent.base.simple.linear_risk(m, np.asarray([parent.base.features(r['pred'], r['image_size'], r['temporal'], 4)]))[0])
            r['risks'][name] = value
    return result


def supported(r): return r['pred'] is not None and r['temporal']['available']


def fixed(rows):
    return {r['image'] for r in rows if r['pred'] is not None and not r['temporal']['available'] and r['risks']['partial_auc4'] <= FALLBACK_CUTOFF}


def accepted(rows, method, cutoff):
    return fixed(rows) | {r['image'] for r in rows if supported(r) and r['risks'][method] <= cutoff}


def calibrate(rows, method):
    if len(rows) != 887 or any(r['split'] != 'val' or r['reliability_role'] != 'val' for r in rows):
        raise ValueError('One complete VAL calibration only')
    entries = {}
    for group, part in level.metrics.grouped(rows).items():
        good = [r for r in part if level.metrics.bad(r) is False]
        retained = sum(r['image'] in fixed(part) for r in good)
        need = max(0, math.ceil(.95 * len(good)) - retained)
        values = sorted(r['risks'][method] for r in good if supported(r))
        if need > len(values): raise ValueError('Frozen fallback cannot satisfy group95')
        entries[group] = dict(good=len(good), fixed_correct=retained, active_need=need,
            risk_le=values[need - 1] if need else 0.)
    return dict(risk_le=max(v['risk_le'] for v in entries.values()), fallback_risk_le=FALLBACK_CUTOFF,
        groups=entries, role='previously_exposed_VAL', not_calibrated_probability=True)


def locked_point(rows, method, count=None, correct=None):
    active = sorted([r for r in rows if supported(r)], key=lambda r: (r['risks'][method], r['image']))
    hold = fixed(rows); fixed_bad = sum(r['image'] in hold and level.metrics.bad(r) is True for r in rows)
    if count is not None:
        n = count - len(hold)
        if not 0 <= n <= len(active): raise ValueError('Invalid locked count')
        ids = hold | {r['image'] for r in active[:n]}
        result = level.summarize(rows, ids)
        tie = level.point(active, method, count=n)['tie_bounds']
        result['tie_bounds'] = dict(tie, FA_min=tie['FA_min'] + fixed_bad, FA_max=tie['FA_max'] + fixed_bad)
    else:
        held_good = len(hold) - fixed_bad; need = correct - held_good
        if not 0 <= need <= sum(level.metrics.bad(r) is False for r in active): raise ValueError('Invalid locked CR')
        cut = level.point(active, method, correct=need)['offline_GT_threshold']
        ids = hold | {r['image'] for r in active if cut is not None and r['risks'][method] <= cut}
        result = level.summarize(rows, ids)
        result.update(exact_CR=result['states']['CR'] == correct, offline_GT_threshold=cut)
    result['fallback_decisions_locked'] = True
    return result


def evaluate(rows, cutoffs):
    groups = {}
    for group, part in level.metrics.grouped(rows).items():
        entry = {}
        for method, cal in cutoffs.items():
            ids = accepted(part, method, cal['risk_le']); c = level.summarize(part, ids)
            controls = {}
            for name in (CONTROL, 'partial_auc4', 'crossfit_fusion', 'fusion4', 'full_simple', 'score_only'):
                # crossfit_fusion control holds the corrected historical fallback decision.
                point = locked_point if name in (CONTROL, 'partial_auc4', 'crossfit_fusion') else level.point
                controls[name] = dict(same_count=point(part, name, count=len(ids)), same_CR=point(part, name, correct=c['states']['CR']))
            ref = level.summarize(part, accepted(part, 'crossfit_fusion', FUSION_CUTOFF))
            entry[method] = dict(candidate=c, controls=controls, frozen_fallback_reference=ref,
                candidate_at_reference_CR=locked_point(part, method, correct=ref['states']['CR']),
                supported_AUROC=level.metrics.auc([r for r in part if supported(r)], method),
                FA_regimes={k: sum(r['image'] in ids and level.metrics.bad(r) is True and
                    ('unsupported' if not supported(r) else 'low' if r['temporal']['primary'] <= parent.base.LEVEL_CUTOFF else 'high') == k for r in part)
                    for k in ('low', 'high', 'unsupported')})
        groups[group] = entry
    return dict(groups=groups, center=level.metrics.center(rows))


def gate(evaluation):
    failures = []
    for group, arms in evaluation['groups'].items():
        c = arms[METHOD]['candidate']; s = c['states']
        if s['CR'] < math.ceil(.95 * (s['CR'] + s['FR'])): failures.append(dict(group=group, check='group95'))
        for method, matches in arms[METHOD]['controls'].items():
            for kind, control in matches.items():
                bound = control['tie_bounds']['FA_min'] if kind == 'same_count' else control['states']['FA']
                if (kind == 'same_CR' and not control['exact_CR']) or s['FA'] > bound or (group == 'all' and s['FA'] >= bound):
                    failures.append(dict(group=group, method=method, check=kind + '_FA'))
                if c['runs']['correct_rejection']['longest'] > control['runs']['correct_rejection']['longest']:
                    failures.append(dict(group=group, method=method, check=kind + '_continuous_FR'))
    return dict(passed=not failures, failures=failures, TEST_opened=False, candidate_adopted=False, stop_if_failed=True)


class CausalHardErrorPAUC(parent.CausalPartialAUC):
    def decide(self, pred, image_size, frame_id, reset=False):
        result = super().decide(pred, image_size, frame_id, reset)
        value = result['size_partial_auc_risk']
        if not result['size_partial_auc_history_available']:
            result['size_accepted'] = value is not None and value <= FALLBACK_CUTOFF
        result['size_hard_error_pauc_version'] = VERSION
        result['size_partial_auc_version'] = VERSION
        return result
