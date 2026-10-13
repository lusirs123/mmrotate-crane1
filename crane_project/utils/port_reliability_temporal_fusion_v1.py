"""Bounded supervised static/causal-size fusion. Online APIs never consume GT."""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_temporal_evidence_v1 as level
from crane_project.utils import port_reliability_temporal_trend_v1 as trend

VERSION = 'port_reliability_temporal_fusion_v1'
FITTING = dict(l2=.1, max_iterations=100, gradient_tolerance=1e-8)
FEATURES = tuple(simple.FEATURES) + ('past3_log_edge_deviation',)
LEVEL_CUTOFF = .0785511443740936
ARMS = ('static3', 'fusion4')
CONTROLS = ('static3', 'full_simple', 'score_only')


def features(pred, image_size, point, dimensions):
    x = simple.descriptor(pred, image_size).tolist()
    if dimensions == 4:
        if not point['available']:
            raise ValueError('Missing history must use shared static fallback')
        x.append(point['primary'])
    elif dimensions != 3:
        raise ValueError('Only prespecified dimensions permitted')
    return x


def fit_matrix(x, y, dimensions):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 2 or x.shape[1] != dimensions or y.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isin(y, [0., 1.]).all()):
        raise ValueError('Invalid supervised design')
    bad = int(y.sum()); good = len(y)-bad
    if not bad or not good:
        raise ValueError('Both genuine error classes required')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-8, 1., scale)
    design = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    sample_weights = np.where(y == 1, .5/bad, .5/good)
    weights = np.zeros(dimensions+1)
    initial = simple.logistic_objective(weights, design, y, sample_weights, FITTING['l2'])[0]
    converged = False
    for step in range(FITTING['max_iterations']):
        loss, gradient, hessian = simple.logistic_objective(weights, design, y, sample_weights, FITTING['l2'])
        if np.max(np.abs(gradient)) <= FITTING['gradient_tolerance']:
            converged = True; break
        direction = np.linalg.solve(hessian, gradient)
        for backtrack in range(30):
            trial = weights-(.5**backtrack)*direction
            candidate = simple.logistic_objective(trial, design, y, sample_weights, FITTING['l2'])[0]
            if candidate <= loss-1e-4*(.5**backtrack)*gradient.dot(direction):
                weights = trial; break
        else:
            raise RuntimeError('Fixed Newton line search failed')
    final, gradient, _ = simple.logistic_objective(weights, design, y, sample_weights, FITTING['l2'])
    if not (converged or np.max(np.abs(gradient)) <= FITTING['gradient_tolerance']):
        raise RuntimeError('Fixed fitting budget exhausted')
    return dict(feature_names=list(FEATURES[:dimensions]), mean=mean.tolist(), scale=scale.tolist(),
                weights=weights.tolist(), train_rows=len(y), bad_rows=bad, good_rows=good,
                initial_objective=initial, final_objective=final, iterations=step+1,
                max_abs_gradient=float(np.max(np.abs(gradient))), converged=True)


def fit(train):
    if not train or any(r['reliability_role'] != 'train' or r['split'] not in ('train', 'train_sim') for r in train):
        raise ValueError('Full TRAIN only; no VAL fitting')
    outputs = [r for r in train if r['pred'] is not None]
    supported = [r for r in outputs if r['temporal']['available']]
    models = {}
    for name, part, dim in [('fallback3', outputs, 3), ('static3', supported, 3), ('fusion4', supported, 4)]:
        x = [features(r['pred'], r['image_size'], r['temporal'], dim) for r in part]
        models[name] = fit_matrix(x, [int(level.metrics.bad(r)) for r in part], dim)
        models[name]['train_images'] = [r['image'] for r in part]
    if models['static3']['train_images'] != models['fusion4']['train_images']:
        raise ValueError('Paired arms must share all supported TRAIN identities')
    return models


def online_risks(pred, image_size, point, models):
    if pred is None:
        return {a: None for a in ARMS}
    result = {}
    for a in ARMS:
        name = a if point['available'] else 'fallback3'
        dim = 4 if name == 'fusion4' else 3
        result[a] = float(simple.linear_risk(models[name], np.asarray([features(pred, image_size, point, dim)]))[0])
    return result


class CausalFusion:
    def __init__(self, models, cutoff, original_policy):
        if set(models) != {'fallback3', 'static3', 'fusion4'} or not 0 <= cutoff <= 1:
            raise ValueError('Invalid frozen fusion policy')
        self.models = deepcopy(models); self.cutoff = cutoff
        self.original = simple.SimpleComponentReliability(original_policy)
        self.tracker = level.CausalSizeEvidence()

    def decide(self, pred, image_size, frame_id, reset=False):
        original = self.original.decide(pred, image_size)
        point = self.tracker.observe(pred, frame_id, image_size, reset)
        risks = online_risks(pred, image_size, point, self.models)
        original['size_accepted'] = pred is not None and risks['fusion4'] <= self.cutoff
        original['size_fusion_risk'] = risks['fusion4']
        original['size_fusion_version'] = VERSION
        original['size_fusion_history_available'] = point['available']
        return original


def score(rows, models):
    result = deepcopy(rows)
    for r in result:
        r['risks'].update(online_risks(r['pred'], r['image_size'], r['temporal'], models))
    return result


def select_cutoff(val, method):
    if not val or any(r['reliability_role'] != 'val' or r['split'] != 'val' for r in val):
        raise ValueError('One complete VAL calibration only')
    groups = {}
    for group, part in level.metrics.grouped(val).items():
        good = sorted(r['risks'][method] for r in part if level.metrics.bad(r) is False)
        required = math.ceil(.95*len(good))
        groups[group] = dict(good_outputs=len(good), required=required,
                             cutoff=good[required-1] if good else None)
    cutoff = max(g['cutoff'] for g in groups.values() if g['cutoff'] is not None)
    return dict(risk_le=cutoff, group_protection=groups, role='exposed_VAL_calibration',
                single_global_threshold=True, not_a_calibrated_error_probability=True)


def accepted(rows, method, cutoff):
    return {r['image'] for r in rows if r['pred'] is not None and r['risks'][method] <= cutoff}


def evaluate(rows, cutoffs):
    groups = {}
    for g, part in level.metrics.grouped(rows).items():
        ids = accepted(part, 'fusion4', cutoffs['fusion4']['risk_le'])
        candidate = level.summarize(part, ids)
        cr, n = candidate['states']['CR'], len(ids)
        controls = {m:dict(same_count=level.point(part, m, count=n), same_CR=level.point(part, m, correct=cr))
                    for m in CONTROLS}
        old_ids = level.veto_accept(part, LEVEL_CUTOFF)
        shared_cr = min(cr, sum(level.metrics.bad(r) is False for r in part if r['image'] in level.formal_accept(part)))
        matched = trend.matched_veto(part, correct=shared_cr)
        groups[g] = dict(candidate=candidate,
            static3=level.summarize(part, accepted(part, 'static3', cutoffs['static3']['risk_le'])),
            formal=level.summarize(part, level.formal_accept(part)),
            old_level=level.summarize(part, old_ids), old_same_CR=matched,
            candidate_at_old_shared_CR=level.point(part, 'fusion4', correct=shared_cr),
            old_same_count=trend.matched_veto(part, count=n),
            controls=controls, AUROC={m:level.metrics.auc(part, m) for m in ('fusion4',)+CONTROLS})
    return dict(groups=groups, center=level.metrics.center(rows))


def gate(evaluation):
    failures = []
    for g, data in evaluation['groups'].items():
        c = data['candidate']; counts = c['states']
        if counts['CR'] < math.ceil(.95*(counts['CR']+counts['FR'])):
            failures.append(dict(group=g, check='correct_retention95'))
        for m, matches in data['controls'].items():
            for kind in ('same_count', 'same_CR'):
                control = matches[kind]
                bound = control['tie_bounds']['FA_min'] if kind == 'same_count' else control['states']['FA']
                if (kind == 'same_CR' and not control['exact_CR']) or counts['FA'] > bound or (g == 'all' and counts['FA'] >= bound):
                    failures.append(dict(group=g, control=m, check=kind+'_FA_gain'))
                if c['runs']['correct_rejection']['longest'] > control['runs']['correct_rejection']['longest']:
                    failures.append(dict(group=g, control=m, check=kind+'_correct_FR_continuity'))
        old = data['old_same_CR']
        if old is None or not old.get('exact_CR', False):
            failures.append(dict(group=g, check='old_same_CR_unavailable_or_tie'))
        else:
            shared = data['candidate_at_old_shared_CR']
            if not shared['exact_CR'] or shared['states']['FA'] > old['states']['FA'] or (g == 'all' and shared['states']['FA'] >= old['states']['FA']):
                failures.append(dict(group=g, check='old_same_CR_FA_gain'))
            if shared['runs']['correct_rejection']['longest'] > old['runs']['correct_rejection']['longest']:
                failures.append(dict(group=g, check='old_same_CR_correct_FR_continuity'))
    return dict(passed=not failures, failures=failures, candidate_adopted=False,
                TEST_opened=False, stop_if_failed=True)
