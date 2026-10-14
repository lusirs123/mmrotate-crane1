"""Fixed 3-D convex constrained objective; GT is fitting/evaluation only."""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_reliability_video_transfer_v1 as video

VERSION = 'port_reliability_fr_constraint_v1'
SETTINGS = dict(l2=.1, correct_retention=.95, surrogate_FR_limit=.05,
    barrier_schedule=[.1, .01, .001, .0001, .00001, .000001, .0000001],
    max_iterations_per_stage=100, gradient_tolerance=2e-7,
    max_backtracks=60, stationarity_tolerance=5e-6,
    feasibility_tolerance=1e-10, duality_gap_tolerance=1e-6,
    control_max_iterations=100, control_gradient_tolerance=1e-8,
    arms=['control3', 'constrained3'])


def prepare(rows):
    part = [r for r in rows if r['pred'] is not None]
    x = np.array([video.fusion.simple.descriptor(r['pred'], r['image_size']) for r in part])
    y = np.array([video.wrong(r) for r in part], dtype=bool)
    if not y.any() or y.all():
        raise ValueError('Both real fitting error classes required')
    mean = x.mean(0); scale = x.std(0); scale[scale < 1e-8] = 1.
    d = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    groups = {seq: d[[not y[i] and r['sequence'] == seq for i, r in enumerate(part)]]
              for seq in sorted({r['sequence'] for r in part})}
    if any(not len(g) for g in groups.values()):
        raise ValueError('Every fitting video needs correct support')
    return part, mean, scale, d[y], groups


def objective(w, errors):
    z = errors@w; p = video.fusion.simple.sigmoid(z)
    penalty = w.copy(); penalty[0] = 0
    loss = float(np.logaddexp(0., -z).mean()+.5*SETTINGS['l2']*(penalty@penalty))
    grad = errors.T@(p-1)/len(errors)+SETTINGS['l2']*penalty
    hess = errors.T@((p*(1-p))[:, None]*errors)/len(errors)
    hess[1:, 1:] += np.eye(len(w)-1)*SETTINGS['l2']
    return loss, grad, hess


def constraints(w, groups):
    values, gradients, hessians = [], [], []
    for d in groups.values():
        z = d@w; p = video.fusion.simple.sigmoid(z)
        factor = len(d)*math.log(2.)
        values.append(float(np.logaddexp(0., z).sum()/factor))
        gradients.append(d.T@p/factor)
        hessians.append(d.T@((p*(1-p))[:, None]*d)/factor)
    return np.array(values), np.array(gradients), np.array(hessians)


def barrier(w, errors, groups, mu):
    loss, grad, hess = objective(w, errors)
    q, j, h = constraints(w, groups); slack = SETTINGS['surrogate_FR_limit']-q
    if min(slack) <= 0:
        return math.inf, None, None
    loss -= mu*float(np.log(slack).sum())
    grad += (mu/slack)@j
    hess += np.sum((mu/slack)[:, None, None]*h, axis=0)
    hess += np.einsum('v,vi,vj->ij', mu/slack**2, j, j)
    return loss, grad, hess


def certificate(w, errors, groups, mu):
    q, j, _ = constraints(w, groups); slack = SETTINGS['surrogate_FR_limit']-q
    if min(slack) <= 0:
        raise ValueError('Non-interior constrained solution')
    multipliers = mu/slack
    loss, grad, _ = objective(w, errors)
    residual = float(np.max(np.abs(grad+multipliers@j)))
    gap = float(multipliers@slack)
    if (residual > SETTINGS['stationarity_tolerance'] or gap > SETTINGS['duality_gap_tolerance']
            or not np.isfinite(w).all()):
        raise RuntimeError('Fixed budget failed convex KKT certificate')
    return dict(objective=loss, surrogate_by_video=dict(zip(groups, q.tolist())),
        dual_multipliers=dict(zip(groups, multipliers.tolist())),
        max_abs_stationarity=residual, duality_gap=gap,
        empirical_fit_only=True, guarantee_on_unseen_video=False)


def fit_constrained(rows):
    part, mean, scale, errors, groups = prepare(rows)
    w = np.zeros(4)
    w[0] = math.log(math.expm1(.025*math.log(2.)))
    stages = []
    initial = objective(w, errors)[0]
    for mu in SETTINGS['barrier_schedule']:
        done = False
        for iteration in range(SETTINGS['max_iterations_per_stage']):
            loss, grad, hess = barrier(w, errors, groups, mu)
            if np.max(np.abs(grad)) <= SETTINGS['gradient_tolerance']:
                done = True; break
            direction = np.linalg.solve(hess, grad)
            if float(grad@direction) < 0:
                raise RuntimeError('Non-descent Newton direction')
            for bt in range(SETTINGS['max_backtracks']):
                rate = .5**bt; trial = w-rate*direction
                value = barrier(trial, errors, groups, mu)[0]
                if value <= loss-1e-4*rate*float(grad@direction):
                    w = trial; break
            else:
                raise RuntimeError('Fixed interior line search failed')
        # Check after the final permitted update too.
        residual = float(np.max(np.abs(barrier(w, errors, groups, mu)[1])))
        if not done and residual > SETTINGS['gradient_tolerance']:
            raise RuntimeError('Fixed barrier stage did not converge: %g, %g' % (mu, residual))
        stages.append(dict(mu=mu, iterations=iteration+1, max_abs_gradient=residual))
    cert = certificate(w, errors, groups, SETTINGS['barrier_schedule'][-1])
    return dict(feature_names=list(video.fusion.simple.FEATURES), mean=mean.tolist(), scale=scale.tolist(),
        weights=w.tolist(), train_rows=len(part), bad_rows=len(errors), good_rows=len(part)-len(errors),
        initial_objective=initial, converged=True, stages=stages, certificate=cert,
        decision='accept iff logit < 0; logit==0 rejected', fitting_videos=list(groups))


def logit(model, row):
    if row['pred'] is None:
        return None
    x = video.fusion.simple.descriptor(row['pred'], row['image_size'])
    return float(model['weights'][0]+((x-np.array(model['mean']))/np.array(model['scale']))@np.array(model['weights'][1:]))


def fit_score(fitting, testing):
    part = [r for r in fitting if r['pred'] is not None]
    control = video.fusion.simple.fit_linear_risk(
        np.array([video.fusion.simple.descriptor(r['pred'], r['image_size']) for r in part]),
        [video.wrong(r) for r in part], dict(l2=.1, max_iterations=100, gradient_tolerance=1e-8))
    candidate = fit_constrained(fitting)
    models = dict(control3=control, constrained3=candidate)
    def scored(rows):
        result = deepcopy(rows)
        for r in result:
            for key, model in models.items(): r['risks'][key] = logit(model, r)
        return result
    fit, held = scored(fitting), scored(testing)
    thresholds = {m: video.cutoff(fit, m) for m in ('control3', 'full_simple', 'score_only')}
    thresholds['constrained3'] = dict(logit_lt=0., rule='fixed_zero_no_posthoc_tuning')
    return models, thresholds, fit, held


def accepted(rows, method, thresholds):
    return {r['image'] for r in rows if r['pred'] is not None and
            (r['risks'][method] < 0 if method == 'constrained3' else
             r['risks'][method] <= thresholds[method]['risk_le'])}


def evaluate(rows, thresholds):
    result = {}
    for group, part in video.fusion.level.metrics.grouped(rows).items():
        data = dict(formal=video.counts(part, video.fusion.level.formal_accept(part)), methods={})
        required = math.ceil(.95*sum(video.wrong(r) is False for r in part))
        for method in ('control3', 'constrained3', 'full_simple', 'score_only'):
            delivery = video.counts(part, accepted(part, method, thresholds))
            controls = {}
            if method in SETTINGS['arms']:
                for c in ('control3', 'full_simple', 'score_only'):
                    controls[c] = dict(same_count=video.point(part, c, count=delivery['accepted']),
                                      same_CR=video.point(part, c, correct=delivery['states']['CR']))
            data['methods'][method] = dict(delivery=delivery, AUROC=video.fusion.level.metrics.auc(part, method),
                offline_correct95=video.point(part, method, correct=required), controls=controls)
        result[group] = data
    return result


def gate(folds):
    failures = []; totals = dict(control3=0, constrained3=0)
    for seq, fold in folds.items():
        methods = fold['held_evaluation']['all']['methods']
        a, b = methods['constrained3'], methods['control3']
        for m in totals: totals[m] += methods[m]['offline_correct95']['states']['FA']
        if a['delivery']['correct_retention'] < .95:
            failures.append(seq+': transferred correct retention <95%')
        # Strong protection fixed before results; no per-video degradation allowance.
        if a['offline_correct95']['states']['FA'] > b['offline_correct95']['states']['FA']:
            failures.append(seq+': same-correct95 FA worse')
        for c in ('control3', 'full_simple', 'score_only'):
            if a['delivery']['states']['FA'] > a['controls'][c]['same_count']['states']['FA']:
                failures.append(seq+': same-count FA worse than '+c)
            same = video.point(fold['held_rows_for_gate'], c, correct=a['delivery']['states']['CR'])
            if a['delivery']['states']['FA'] > same['states']['FA']:
                failures.append(seq+': same-CR FA worse than '+c)
            if a['delivery']['longest_correct_FR'] > same['longest_correct_FR']:
                failures.append(seq+': same-CR continuous FR worse than '+c)
    if totals['constrained3'] >= totals['control3']:
        failures.append('No strict pooled per-video correct95 FA improvement')
    if totals['control3'] != 54:
        raise ValueError('Reviewed 3-D leave-video-out control not reproduced')
    return dict(passed=not failures, failures=failures, per_video_correct95_FA=totals,
                rule='all held videos retention95 + strict sameCR FA increment + per-video protections',
                statistical_guarantee=False)


def audit(rows, formal_model):
    video.support.validate(rows)
    original = deepcopy(rows)
    seqs = sorted({r['sequence'] for r in rows})
    if len(seqs) != 6:
        raise ValueError('Reviewed six TRAIN videos required')
    models, thresholds, fit, reference = fit_score(rows, rows)
    for k in ('mean', 'scale', 'weights'):
        if not np.allclose(models['control3'][k], formal_model[k], atol=1e-12, rtol=0):
            raise ValueError('Full TRAIN control not formal simple')
    ref = dict(models=models, thresholds=thresholds, evaluation=evaluate(reference, thresholds))
    folds, held = {}, []
    for seq in seqs:
        fitting = [r for r in rows if r['sequence'] != seq]
        testing = [r for r in rows if r['sequence'] == seq]
        m, t, sf, sh = fit_score(fitting, testing)
        folds[seq] = dict(fit_images=[r['image'] for r in fitting], held_images=[r['image'] for r in testing],
            models=m, thresholds=t, fit_risks={r['image']:r['risks'] for r in sf},
            fit_evaluation=evaluate(sf, t), held_evaluation=evaluate(sh, t), held_rows_for_gate=sh)
        for r in sh: r['diagnostic_fold'] = seq
        held.extend(sh)
    g = gate(folds)
    for f in folds.values(): del f['held_rows_for_gate']
    held.sort(key=lambda r: (r['sequence'], r['frame_id'], r['image']))
    pooled = {m:{g:video.counts(part, {r['image'] for r in held
        if r['image'] in accepted([r], m, folds[r['sequence']]['thresholds'])})
        for g, part in video.fusion.level.metrics.grouped(held).items()}
        for m in ('control3', 'constrained3', 'full_simple', 'score_only')}
    if rows != original: raise ValueError('Input mutation')
    return dict(status='TRAIN_GATE_PASSED_PENDING_VAL' if g['passed'] else 'TRAIN_GATE_FAILED_STOP',
        reference=ref, folds=folds, gate=g, pooled_transfer_delivery=pooled, fits=14,
        held_arm_evaluations=12, detector_inferences=0, neural_updates=0,
        VAL_opened=False, TEST_opened=False, candidate_adopted=False), held, reference
