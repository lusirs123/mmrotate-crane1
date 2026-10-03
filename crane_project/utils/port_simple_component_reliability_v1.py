"""Current-frame three-flag interface; NumPy only, no detector or image head.

The two linear risk models are fitted offline on genuine TRAIN outputs.
Their balanced sigmoid scores are NOT calibrated correctness probabilities.
"""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np

VERSION = 'port_simple_component_reliability_v1'
FEATURES = ('score_logit', 'log_relative_geometric_size', 'log_aspect')
COMPONENTS = ('center', 'size', 'angle')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def prediction(value):
    if value is None:
        return None
    p = np.asarray(value, dtype=np.float64)
    if (p.shape != (6,) or not np.isfinite(p).all() or min(p[2:4]) <= 0
            or not .05 < p[5] <= 1.):
        raise ValueError('Expected original-coordinate genuine B top1 or None')
    return p.copy()


def canonical(box):
    b = np.asarray(box, dtype=np.float64).copy()
    if b.shape != (5,) or not np.isfinite(b).all() or min(b[2:4]) <= 0:
        raise ValueError('Expected a finite positive OBB')
    if b[2] < b[3]:
        b[2], b[3] = b[3], b[2]
        b[4] += math.pi/2
    b[4] = (b[4]+math.pi/2) % math.pi-math.pi/2
    return b


def descriptor(pred_original, image_size):
    p = prediction(pred_original)
    size = np.asarray(image_size, dtype=np.float64)
    if (p is None or size.shape != (2,) or not np.isfinite(size).all()
            or min(size) <= 0):
        raise ValueError('Features require an output and original [width,height]')
    b = canonical(p[:5])
    score = float(np.clip(p[5], 1e-6, 1-1e-6))
    # Sums of logs avoid overflow and preserve isotropic unit invariance.
    return np.array([math.log(score)-math.log1p(-score),
        .5*(math.log(b[2])+math.log(b[3])-math.log(size[0])-math.log(size[1])),
        math.log(b[2])-math.log(b[3])], dtype=np.float64)


def geometry_errors(gt_original, pred_original):
    g, b = canonical(gt_original), canonical(prediction(pred_original)[:5])
    return dict(center_px=float(np.linalg.norm(b[:2]-g[:2])),
        size_max_relative=float(np.max(np.abs(b[2:4]-g[2:4])/g[2:4])),
        angle_deg=float(abs((b[4]-g[4]+math.pi/2) % math.pi-math.pi/2)*180/math.pi))


def sigmoid(value):
    z = np.asarray(value, dtype=np.float64)
    return np.exp(-np.logaddexp(0., -z))


def logistic_objective(weights, design, labels, sample_weights, l2):
    logits = design.dot(weights)
    penalty = weights.copy(); penalty[0] = 0.
    risk = sigmoid(logits)
    loss = float(sample_weights.dot(np.logaddexp(0., logits)-labels*logits)
                 +.5*l2*penalty.dot(penalty))
    gradient = design.T.dot(sample_weights*(risk-labels))+l2*penalty
    hessian = design.T.dot((sample_weights*risk*(1-risk))[:, None]*design)
    hessian[1:, 1:] += np.eye(design.shape[1]-1)*l2
    return loss, gradient, hessian


def fit_linear_risk(features, labels, settings):
    x, y = np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != len(FEATURES) or y.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isin(y, [0., 1.]).all()):
        raise ValueError('Invalid offline linear-risk fitting inputs')
    bad, good = int(y.sum()), int(len(y)-y.sum())
    if not bad or not good:
        raise ValueError('Both genuine TRAIN error classes are required')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-8, 1., scale)
    design = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    sample_weights = np.where(y == 1, .5/bad, .5/good)
    weights = np.zeros(1+len(FEATURES), dtype=np.float64)
    initial_loss = logistic_objective(weights, design, y, sample_weights, settings['l2'])[0]
    converged = False
    for step in range(settings['max_iterations']):
        loss, gradient, hessian = logistic_objective(weights, design, y, sample_weights, settings['l2'])
        if np.max(np.abs(gradient)) <= settings['gradient_tolerance']:
            converged = True
            break
        direction = np.linalg.solve(hessian, gradient)
        for backtrack in range(30):
            rate = .5**backtrack
            candidate = weights-rate*direction
            candidate_loss = logistic_objective(candidate, design, y, sample_weights, settings['l2'])[0]
            if candidate_loss <= loss-1e-4*rate*gradient.dot(direction):
                weights = candidate
                break
        else:
            raise RuntimeError('Fixed Newton line search failed; no parameter search permitted')
    final_loss, gradient, _ = logistic_objective(weights, design, y, sample_weights, settings['l2'])
    converged = converged or bool(np.max(np.abs(gradient)) <= settings['gradient_tolerance'])
    if not converged or not np.isfinite(weights).all():
        raise RuntimeError('Fixed linear-risk fit did not converge')
    return dict(feature_names=list(FEATURES), mean=mean.tolist(), scale=scale.tolist(),
        weights=weights.tolist(), train_rows=len(y), bad_rows=bad, good_rows=good,
        initial_objective=initial_loss, final_objective=final_loss,
        iterations=step+1, max_abs_gradient=float(np.max(np.abs(gradient))), converged=True)


def linear_risk(model, features):
    x = np.asarray(features, dtype=np.float64)
    return sigmoid(model['weights'][0]+((x-np.asarray(model['mean']))/
        np.asarray(model['scale'])).dot(np.asarray(model['weights'][1:])))


def coverage_cutoff(risks, total_frames, target):
    """One pooled VAL cutoff. Ties stay together; never route by video/domain."""
    r = np.asarray(risks, dtype=np.float64)
    if (r.ndim != 1 or not len(r) or not np.isfinite(r).all()
            or ((r < 0) | (r > 1)).any() or len(r) > total_frames
            or not 0 < target <= 1):
        raise ValueError('Invalid calibration risks or coverage')
    count = min(len(r), int(math.ceil(target*total_frames)))
    threshold = float(np.sort(r)[count-1])
    accepted = int((r <= threshold).sum())
    return dict(risk_le=threshold, requested_full_frame_coverage=target,
        total_frames=total_frames, outputs=len(r), requested_count=int(math.ceil(target*total_frames)),
        capped_count=count, actual_accepted=accepted, actual_full_frame_coverage=accepted/total_frames,
        target_reachable=len(r)/total_frames >= target, tie_policy='accept_entire_cutoff_tie')


def validate_policy(policy):
    if (policy.get('protocol') != VERSION or policy.get('feature_names') != list(FEATURES)
            or policy.get('center_policy') != 'retain_valid_B_output_no_extra_rejection'
            or set(policy.get('models', {})) != {'size', 'angle'}):
        raise ValueError('Unexpected simple three-component policy')
    for name in ('size', 'angle'):
        m = policy['models'][name]
        if (m.get('feature_names') != list(FEATURES) or m.get('converged') is not True
                or len(m.get('weights', [])) != 4 or len(m.get('mean', [])) != 3
                or len(m.get('scale', [])) != 3):
            raise ValueError('Invalid linear risk model')
        numbers = np.asarray(m['weights']+m['mean']+m['scale'], dtype=float)
        if not np.isfinite(numbers).all() or min(m['scale']) <= 0:
            raise ValueError('Nonfinite or invalid model normalization')
        for method in ('simple', 'score_only'):
            cutoff = policy['cutoffs'][method][name]['risk_le']
            if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
                raise ValueError('Invalid fixed working cutoff')


class SimpleComponentReliability:
    """Online API accepts ONLY the current original B output and image size."""

    def __init__(self, policy):
        validate_policy(policy)
        self._policy = deepcopy(policy)

    def decide(self, pred_original, image_size, method='simple'):
        if method not in ('simple', 'score_only', 'raw'):
            raise ValueError('Unknown fixed method')
        p = prediction(pred_original)
        if p is None:
            return dict(raw_b_output=None, center_accepted=False, size_accepted=False,
                        angle_accepted=False, risks=dict(size=None, angle=None))
        x = descriptor(p, image_size)
        risks = {name: float(linear_risk(self._policy['models'][name], x))
                 if method == 'simple' else float(1-p[5]) for name in ('size', 'angle')}
        flags = {name: method == 'raw' or risks[name] <= self._policy['cutoffs'][method][name]['risk_le']
                 for name in ('size', 'angle')}
        return dict(raw_b_output=deepcopy(list(pred_original)), center_accepted=True,
            size_accepted=bool(flags['size']), angle_accepted=bool(flags['angle']), risks=risks)


def create_policy(train_rows, val_rows, protocol):
    models = {}
    for component, error_key in (('size', 'size_max_relative'), ('angle', 'angle_deg')):
        out = [r for r in train_rows if r['pred'] is not None and
               (component != 'angle' or r['train_angle_eligible'])]
        x = np.asarray([descriptor(r['pred'], r['image_size']) for r in out])
        labels = [geometry_errors(r['gt'], r['pred'])[error_key] > protocol['error_limits'][error_key]
                  for r in out]
        models[component] = fit_linear_risk(x, labels, protocol['fitting'])
    outputs = [r for r in val_rows if r['pred'] is not None]
    x = np.asarray([descriptor(r['pred'], r['image_size']) for r in outputs])
    cutoffs = {}
    for method in ('simple', 'score_only'):
        cutoffs[method] = {}
        for component in ('size', 'angle'):
            risks = linear_risk(models[component], x) if method == 'simple' else np.array([1-r['pred'][5] for r in outputs])
            cutoffs[method][component] = coverage_cutoff(risks, len(val_rows), protocol['working_coverage'])
    policy = dict(protocol=VERSION, feature_names=list(FEATURES),
        center_policy=protocol['center_policy'], models=models, cutoffs=cutoffs,
        parameter_count=8, calibration_role='pooled_source_VAL_coverage_only_no_GT_in_cutoff',
        evidence_role='exploratory_not_independent_validation', test_repeatedly_exposed=True)
    validate_policy(policy)
    return policy


def _component_stats(rows, component, accepted):
    eligible = [r for r in rows if component != 'angle' or r['angle_axis_well_defined']]
    out = [r for r in eligible if r['pred'] is not None]
    error_key, limit = {'center': ('center_px', 15.), 'size': ('size_max_relative', .1),
                        'angle': ('angle_deg', 3.)}[component]
    def good(r):
        e = geometry_errors(r['gt'], r['pred'])[error_key]
        return e < limit if component == 'center' else e <= limit
    kept = [r for r in out if r['image'] in accepted]
    hits = sum(good(r) for r in kept)
    errors = [geometry_errors(r['gt'], r['pred'])[error_key] for r in kept]
    run = best = 0; previous = None
    for r in sorted(rows, key=lambda a: (a['sequence'], a['frame_id'])):
        if previous is None or r['sequence'] != previous['sequence'] or r['frame_id'] != previous['frame_id']+1:
            run = 0
        run = 0 if r['image'] in accepted else run+1
        best = max(best, run); previous = r
    return dict(eligible_frames=len(eligible), output_frames=len(out), accepted_frames=len(kept),
        unassessed_accepted=sum(r['image'] in accepted and not r['angle_axis_well_defined']
            for r in rows) if component == 'angle' else 0,
        accepted_coverage=len(kept)/len(eligible) if eligible else None,
        correct_accepted=hits, incorrect_accepted=len(kept)-hits,
        correctness_on_accepted=hits/len(kept) if kept else None,
        full_frame_correct_coverage=hits/len(eligible) if eligible else None,
        correct_rejected=sum(good(r) and r['image'] not in accepted for r in out),
        incorrect_rejected=sum(not good(r) and r['image'] not in accepted for r in out),
        missing_outputs=len(eligible)-len(out),
        error_mean=float(np.mean(errors)) if errors else None,
        error_rmse=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        error_p90=float(np.percentile(errors, 90)) if errors else None,
        longest_flag_unavailable_all_frames=best)


def evaluate(rows, policy):
    runtime = SimpleComponentReliability(policy)
    records = [dict(image=r['image'], methods={m: runtime.decide(r['pred'], r['image_size'], m)
                   for m in ('raw', 'score_only', 'simple')}) for r in rows]
    indexed = {r['image']: r['methods'] for r in records}
    groups = dict(all=rows)
    for field in ('domain', 'sequence'):
        groups.update({field+':'+v: [r for r in rows if r[field] == v] for v in sorted({r[field] for r in rows})})
    strata = {}
    for group, values in groups.items():
        out = [r for r in values if r['pred'] is not None]
        hits = sum(geometry_errors(r['gt'], r['pred'])['center_px'] < 15 for r in out)
        summary = dict(frames=len(values), output_frames=len(out), center_hits=hits,
            output_coverage=len(out)/len(values), center_hit_rate_on_outputs=hits/len(out) if out else None,
            all_frame_center_correct_coverage=hits/len(values), components={})
        for component in COMPONENTS:
            stats = {}
            for method in ('raw', 'score_only', 'simple'):
                accepted = {r['image'] for r in values if indexed[r['image']][method][component+'_accepted']}
                stats[method] = _component_stats(values, component, accepted)
            count = stats['simple']['accepted_frames']
            ranked = sorted((r for r in out if component != 'angle' or r['angle_axis_well_defined']),
                            key=lambda r: (-r['pred'][5], r['image']))
            stats['matched_score_diagnostic'] = _component_stats(values, component,
                {r['image'] for r in ranked[:count]})
            stats['simple_minus_matched_score_incorrect_accepted'] = (
                stats['simple']['incorrect_accepted']-stats['matched_score_diagnostic']['incorrect_accepted'])
            summary['components'][component] = stats
        summary['complete_obb'] = {}
        for method in ('raw', 'score_only', 'simple'):
            kept = [r for r in values if all(indexed[r['image']][method][c+'_accepted'] for c in COMPONENTS)]
            assessed = [r for r in kept if r['angle_axis_well_defined']]
            joint_hits = 0
            for r in assessed:
                e = geometry_errors(r['gt'], r['pred'])
                joint_hits += e['center_px'] < 15 and e['size_max_relative'] <= .1 and e['angle_deg'] <= 3
            summary['complete_obb'][method] = dict(accepted_frames=len(kept),
                all_frame_accepted_coverage=len(kept)/len(values), joint_assessed=len(assessed),
                joint_unassessed=len(kept)-len(assessed), joint_correct=joint_hits,
                joint_incorrect=len(assessed)-joint_hits,
                all_frame_joint_correct_coverage=joint_hits/len(values))
        strata[group] = summary
    return records, strata
