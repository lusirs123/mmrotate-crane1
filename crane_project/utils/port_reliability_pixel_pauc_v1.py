"""Paired four/six-input smooth pAUC fit with exact frozen fallback decisions."""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_reliability_partial_auc_v1 as parent
from crane_project.utils import port_reliability_pixel_border_v1 as pixels

VERSION = 'port_reliability_pixel_pauc_v1'
CONTROL = 'paired_pauc4'
METHOD = 'pixel_pauc6'
SETTINGS = deepcopy(parent.SETTINGS)
FALLBACK_CUTOFF = .4094748402105834
FUSION_CUTOFF = .5317310050227115
level = parent.base.level


def pixel_supported(evidence):
    if evidence is None:
        return False
    for axis in ('L', 'S'):
        item = evidence[axis]
        if not isinstance(item['supported'], bool):
            raise ValueError('Boolean native-pixel support required')
        if not item['supported']:
            if item['mismatch'] is not None:
                raise ValueError('Unsupported pixel evidence has a value')
        elif item['mismatch'] is None or not math.isfinite(item['mismatch']) or item['mismatch'] < 0:
            raise ValueError('Finite nonnegative pixel distance required')
    return all(evidence[k]['supported'] for k in ('L', 'S'))


def active(row):
    return row['pred'] is not None and row['temporal']['available'] and pixel_supported(row['pixel_evidence'])


def features(pred, image_size, point, evidence, dim):
    if dim not in (4, 6) or pred is None or not point['available'] or not pixel_supported(evidence):
        raise ValueError('Both arms require identical history/pixel-supported observations')
    x = list(parent.base.features(pred, image_size, point, 4))
    if dim == 6:
        x += [evidence[k]['mismatch'] for k in ('L', 'S')]
    return np.asarray(x, dtype=float)


def attach(rows, pixel_rows, role):
    expected = 2558 if role == 'TRAIN' else 887 if role == 'VAL' else None
    if expected is None or len(rows) != expected or len(pixel_rows) != expected:
        raise ValueError('Complete original TRAIN/VAL only')
    lookup = {r['image']: r for r in pixel_rows}
    if len(lookup) != expected or len({r['image'] for r in rows}) != expected or set(lookup) != {r['image'] for r in rows}:
        raise ValueError('Duplicated/different pixel membership')
    output = deepcopy(rows)
    for row in output:
        if row['reliability_role'] != role.lower() or row['split'] not in (('train', 'train_sim') if role == 'TRAIN' else ('val',)):
            raise ValueError('Role changed')
        pix = lookup[row['image']]
        if any(row[k] != pix[k] for k in ('pred', 'domain', 'sequence')):
            raise ValueError('Pixel observation identity changed')
        if level.metrics.bad(row) != pix['wrong'] or row['fallback_decision'] != pix['accepted']:
            raise ValueError('Offline label/frozen decision changed')
        low = row['temporal']['available'] and row['temporal']['primary'] <= parent.base.LEVEL_CUTOFF
        if low != pix['low_temporal']:
            raise ValueError('History identity changed')
        row['pixel_evidence'] = deepcopy(pix['evidence'])
        pixel_supported(row['pixel_evidence'])
    return output


def prepare(train, base_models):
    if len(train) != 2558 or len({r['image'] for r in train}) != 2558 or any(r['reliability_role'] != 'train' or r['split'] not in ('train', 'train_sim') for r in train):
        raise ValueError('Complete original TRAIN only')
    base = base_models['fusion4']
    history_ids = [r['image'] for r in train if r['pred'] is not None and r['temporal']['available']]
    if history_ids != base['train_images']:
        raise ValueError('Frozen four-input TRAIN identity differs')
    part = [r for r in train if active(r)]
    if not part:
        raise ValueError('No active TRAIN observations')
    x = np.array([features(r['pred'], r['image_size'], r['temporal'], r['pixel_evidence'], 6) for r in part])
    y = np.array([bool(level.metrics.bad(r)) for r in part])
    if not y.any() or y.all():
        raise ValueError('Genuine TRAIN correct and wrong examples required')
    extra_scale = x[:, 4:].std(axis=0)
    extra_scale[extra_scale < 1e-8] = 1.
    mean = list(base['mean']) + x[:, 4:].mean(axis=0).tolist()
    scale = list(base['scale']) + extra_scale.tolist()
    if not np.isfinite(x).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or min(scale) <= 0:
        raise ValueError('Invalid TRAIN normalization')
    return dict(train_images=[r['image'] for r in part], active_rows=len(part),
                wrong_images=[r['image'] for r in part if level.metrics.bad(r) is True],
                fallback_images=[r['image'] for r in train if not active(r)],
                mean=mean, scale=scale, intercept=base['weights'][0],
                feature_names=list(base['feature_names']) + ['native_L_border_mismatch', 'native_S_border_mismatch'],
                additional_normalization_role='TRAIN_ONLY', settings=deepcopy(SETTINGS))


def objective(v, good, wrong):
    """Original uniformly weighted objective generalized to four/six slopes."""
    v, good, wrong = map(lambda a: np.asarray(a, dtype=float), (v, good, wrong))
    if good.ndim != 2 or wrong.ndim != 2 or good.shape[1] not in (4, 6) or wrong.shape[1] != good.shape[1] or v.shape != (good.shape[1]+1,) or not len(good) or not len(wrong):
        raise ValueError('Four/six slopes plus offline tail threshold, two genuine classes')
    if not all(np.isfinite(a).all() for a in (v, good, wrong)):
        raise ValueError('Nonfinite objective input')
    dim = good.shape[1]
    d = good[:, None, :] - wrong[None, :, :]
    u = d @ v[:dim]; prob = parent.base.simple.sigmoid(u)
    h = np.logaddexp(0., u).mean(axis=1)
    jac = np.mean(prob[:, :, None]*d, axis=1)
    hes = np.einsum('gi,gij,gik->gjk', prob*(1-prob), d, d)/len(wrong)
    T, beta, lam = (SETTINGS[k] for k in ('temperature', 'beta', 'l2'))
    p = parent.base.simple.sigmoid((h-v[-1])/T)
    loss = float(v[-1]+T*np.logaddexp(0., (h-v[-1])/T).mean()/beta+.5*lam*v[:dim].dot(v[:dim]))
    grad = np.r_[np.mean(p[:, None]*jac, axis=0)/beta+lam*v[:dim], 1-p.mean()/beta]
    outer = np.column_stack((jac, -np.ones(len(good))))
    H = np.einsum('g,gi,gj->ij', p*(1-p)/(beta*T*len(good)), outer, outer)
    H[:dim, :dim] += np.einsum('g,gij->ij', p/(beta*len(good)), hes)+lam*np.eye(dim)
    return loss, grad, H


def fit_arrays(good, wrong):
    """Pure numerical solver; used by synthetic tests, no dataset access."""
    dim = np.asarray(good).shape[1]
    v = np.r_[np.zeros(dim), math.log(2)+SETTINGS['temperature']*math.log(19)]
    initial = objective(v, good, wrong)[0]; trace = []
    for step in range(SETTINGS['max_iterations']):
        loss, grad, H = objective(v, good, wrong)
        trace.append(dict(step=step, objective=loss, max_abs_gradient=float(abs(grad).max())))
        if abs(grad).max() <= SETTINGS['gradient_tolerance']:
            break
        direction = np.linalg.solve(H, grad)
        for k in range(30):
            trial = v-.5**k*direction
            if objective(trial, good, wrong)[0] <= loss-1e-4*.5**k*grad.dot(direction):
                v = trial; break
        else:
            raise RuntimeError('Fixed line-search budget exhausted')
    loss, grad, _ = objective(v, good, wrong)
    if abs(grad).max() > SETTINGS['gradient_tolerance']:
        raise RuntimeError('Fixed fit budget exhausted')
    return dict(slopes=v[:dim].tolist(), tail_threshold=float(v[-1]), initial_objective=initial,
                final_objective=loss, max_abs_gradient=float(abs(grad).max()), trace=trace)


def fit(train, prepared, dim):
    if dim not in (4, 6) or len(train) != 2558 or any(r['reliability_role'] != 'train' or r['split'] not in ('train', 'train_sim') for r in train):
        raise ValueError('Paired complete TRAIN fit required')
    part = [r for r in train if active(r)]
    if prepared['train_images'] != [r['image'] for r in part] or prepared['settings'] != SETTINGS:
        raise ValueError('Prepared TRAIN identities/settings changed')
    mean = np.asarray(prepared['mean'][:dim]); scale = np.asarray(prepared['scale'][:dim])
    z = (np.array([features(r['pred'], r['image_size'], r['temporal'], r['pixel_evidence'], dim) for r in part])-mean)/scale
    y = np.array([bool(level.metrics.bad(r)) for r in part])
    if prepared['wrong_images'] != [r['image'] for r, bad in zip(part, y) if bad]:
        raise ValueError('Prepared label alignment changed')
    numerical = fit_arrays(z[~y], z[y])
    return dict(numerical, weights=[prepared['intercept']]+numerical['slopes'],
                mean=mean.tolist(), scale=scale.tolist(), feature_names=prepared['feature_names'][:dim],
                train_images=deepcopy(prepared['train_images']), wrong_images=deepcopy(prepared['wrong_images']),
                dim=dim, settings=deepcopy(SETTINGS), error_weighting='uniform')


def score(rows, models):
    result = deepcopy(rows)
    for row in result:
        for name, model in models.items():
            value = row['risks']['partial_auc4']
            if active(row):
                x = features(row['pred'], row['image_size'], row['temporal'], row['pixel_evidence'], model['dim'])
                value = float(parent.base.simple.linear_risk(model, x[None])[0])
            row['risks'][name] = value
    return result


def fixed(rows):
    return {r['image'] for r in rows if r['pred'] is not None and not active(r) and r['risks']['partial_auc4'] <= FALLBACK_CUTOFF}


def accepted(rows, method, cutoff):
    return fixed(rows) | {r['image'] for r in rows if active(r) and r['risks'][method] <= cutoff}


def calibrate(rows, method):
    if len(rows) != 887 or len({r['image'] for r in rows}) != 887 or any(r['split'] != 'val' or r['reliability_role'] != 'val' for r in rows):
        raise ValueError('Single complete VAL calibration required')
    groups = {}
    for name, part in level.metrics.grouped(rows).items():
        good = [r for r in part if level.metrics.bad(r) is False]
        held = sum(r['image'] in fixed(part) for r in good)
        need = max(0, math.ceil(.95*len(good))-held)
        values = sorted(r['risks'][method] for r in good if active(r))
        if need > len(values):
            raise ValueError('Frozen fallback cannot meet group95')
        groups[name] = dict(good=len(good), fixed_correct=held, active_need=need,
                            risk_le=values[need-1] if need else 0.)
    return dict(risk_le=max(v['risk_le'] for v in groups.values()), groups=groups,
                fallback_risk_le=FALLBACK_CUTOFF, role='previously_exposed_VAL', not_calibrated_probability=True)


def locked_point(rows, method, count=None, correct=None):
    supported = [r for r in rows if active(r)]
    hold = fixed(rows)
    held_bad = sum(level.metrics.bad(r) is True and r['image'] in hold for r in rows)
    if count is not None:
        need = count-len(hold)
        if not 0 <= need <= len(supported):
            raise ValueError('Infeasible branch-locked count')
        point = level.point(supported, method, count=need)
        chosen = sorted(supported, key=lambda r: (r['risks'][method], r['image']))[:need]
        result = level.summarize(rows, hold | {r['image'] for r in chosen})
        tie = point['tie_bounds']
        result['tie_bounds'] = dict(tie, FA_min=tie['FA_min']+held_bad, FA_max=tie['FA_max']+held_bad)
    else:
        need = correct-(len(hold)-held_bad)
        if not 0 <= need <= sum(level.metrics.bad(r) is False for r in supported):
            return dict(feasible=False, reason='Frozen fallback cannot reach requested CR', fallback_decisions_locked=True)
        point = level.point(supported, method, correct=need); cut = point['offline_GT_threshold']
        result = level.summarize(rows, hold | {r['image'] for r in supported if cut is not None and r['risks'][method] <= cut})
        result.update(exact_CR=result['states']['CR']==correct, offline_GT_threshold=cut)
    result.update(feasible=True, fallback_decisions_locked=True)
    return result


def evaluate(rows, cutoffs):
    groups = {}
    for name, part in level.metrics.grouped(rows).items():
        entries = {}
        ref_ids = {r['image'] for r in part if r['pred'] is not None and r['fallback_decision']}
        original_reference = level.summarize(part, ref_ids)
        for method, cal in cutoffs.items():
            ids = accepted(part, method, cal['risk_le']); summary = level.summarize(part, ids)
            controls = {}
            for control in (CONTROL, 'partial_auc4', 'crossfit_fusion', 'fusion4', 'full_simple', 'score_only'):
                point = locked_point if control in (CONTROL, 'partial_auc4', 'crossfit_fusion') else level.point
                controls[control] = dict(same_count=point(part, control, count=len(ids)), same_CR=point(part, control, correct=summary['states']['CR']))
            entries[method] = dict(candidate=summary, controls=controls,
                frozen_fallback_reference=original_reference,
                candidate_at_reference_CR=locked_point(part, method, correct=original_reference['states']['CR']),
                active_AUROC=level.metrics.auc([r for r in part if active(r)], method),
                FA_regimes={kind:sum(r['image'] in ids and level.metrics.bad(r) is True and
                    ('history_unavailable' if not r['temporal']['available'] else 'pixel_unavailable' if not active(r) else
                     'low' if r['temporal']['primary']<=parent.base.LEVEL_CUTOFF else 'high')==kind for r in part)
                    for kind in ('low','high','history_unavailable','pixel_unavailable')})
        groups[name] = entries
    return dict(groups=groups, center=level.metrics.center(rows))


def gate(evaluation):
    failures = []
    for group, arms in evaluation['groups'].items():
        c = arms[METHOD]['candidate']; s = c['states']
        if s['CR'] < math.ceil(.95*(s['CR']+s['FR'])):
            failures.append(dict(group=group, check='group95'))
        for name, matches in arms[METHOD]['controls'].items():
            for kind, control in matches.items():
                if not control.get('feasible',True) or (kind=='same_CR' and not control['exact_CR']):
                    failures.append(dict(group=group, method=name, check=kind+'_infeasible_or_tied')); continue
                bound = control['tie_bounds']['FA_min'] if kind=='same_count' else control['states']['FA']
                if s['FA']>bound or (group=='all' and s['FA']>=bound):
                    failures.append(dict(group=group, method=name, check=kind+'_FA'))
                if c['runs']['correct_rejection']['longest']>control['runs']['correct_rejection']['longest']:
                    failures.append(dict(group=group, method=name, check=kind+'_continuous_FR'))
        ref = arms[METHOD]['frozen_fallback_reference']
        matched = arms[METHOD]['candidate_at_reference_CR']
        if not matched.get('feasible', True) or not matched['exact_CR']:
            failures.append(dict(group=group, check='actual_reference_CR_infeasible_or_tied'))
        else:
            if matched['states']['FA']>ref['states']['FA'] or (group=='all' and matched['states']['FA']>=ref['states']['FA']):
                failures.append(dict(group=group, check='actual_reference_CR_FA'))
            if matched['runs']['correct_rejection']['longest']>ref['runs']['correct_rejection']['longest']:
                failures.append(dict(group=group, check='actual_reference_CR_continuous_FR'))
    all_arms = evaluation['groups']['all']
    if all_arms[METHOD]['FA_regimes']['low'] >= all_arms[CONTROL]['FA_regimes']['low']:
        failures.append(dict(group='all', check='low_deviation_FA_increment'))
    return dict(passed=not failures, failures=failures, TEST_opened=False, candidate_adopted=False, stop_if_failed=True)


class CausalPixelPAUC:
    """Gray pixels + current box + past-only tracker. No GT/domain/video router."""
    def __init__(self, base_models, partial_model, model, cutoff, original_policy):
        if model['dim']!=6 or not 0<=cutoff<=1:
            raise ValueError('Six-input model/global cutoff required')
        self.base_models=deepcopy(base_models);self.partial_model=deepcopy(partial_model)
        self.model=deepcopy(model);self.cutoff=cutoff
        self.original=parent.base.simple.SimpleComponentReliability(original_policy)
        self.tracker=level.CausalSizeEvidence()

    def decide(self, pred, image_size, frame_id, gray=None, reset=False):
        result=self.original.decide(pred,image_size)
        point=self.tracker.observe(pred,frame_id,image_size,reset)
        evidence=None
        if gray is not None and pred is not None:
            if np.asarray(gray).shape!=(image_size[1],image_size[0]):
                raise ValueError('Use native original-coordinate gray image')
            evidence=pixels.evidence(gray,pred)
        is_active=pred is not None and point['available'] and pixel_supported(evidence)
        if pred is None:
            risk=None
        elif is_active:
            x=features(pred,image_size,point,evidence,6)
            risk=float(parent.base.simple.linear_risk(self.model,x[None])[0])
        else:
            risk=parent.online_risk(pred,image_size,point,self.base_models,self.partial_model)
        result['size_accepted']=risk is not None and risk <= (self.cutoff if is_active else FALLBACK_CUTOFF)
        result.update(size_pixel_pauc_risk=risk,size_pixel_pauc_active=is_active,size_pixel_pauc_version=VERSION)
        return result
