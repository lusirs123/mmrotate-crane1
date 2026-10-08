"""Two TRAIN-fitted size-score ablations; frozen center/angle and final M boxes."""
from copy import deepcopy

import numpy as np

from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_separability_v1 as metrics
from crane_project.utils.port_reliability_tradeoff_v1 import required_good
from crane_project.utils import port_midpoint_sigma15_reliability_v1 as binding

VERSION = 'port_reliability_feature_ablation_v1'
ARMS = {'drop_relative_size': (0, 2), 'drop_aspect': (0, 1)}
CONTROLS = ('full_simple', 'score_only')
FIT = dict(l2=.1, max_iterations=100, gradient_tolerance=1e-8)


def fit(features, labels, indices):
    """Same zero-init, class balance, normalization, L2 and Newton as simple-v1."""
    all_x, y = np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    indices = tuple(indices)
    if (indices not in tuple(ARMS.values())+((0, 1, 2),) or all_x.ndim != 2
            or all_x.shape[1] != 3 or y.shape != (len(all_x),)
            or not np.isfinite(all_x).all() or not np.isin(y, [0., 1.]).all()):
        raise ValueError('Invalid fixed size fitting data/features')
    bad, good = int(y.sum()), int(len(y)-y.sum())
    if not bad or not good:
        raise ValueError('Both genuine TRAIN classes required')
    x = all_x[:, indices]
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-8, 1., scale)
    design = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    sw = np.where(y == 1, .5/bad, .5/good)
    weights = np.zeros(1+len(indices), dtype=np.float64)
    initial = simple.logistic_objective(weights, design, y, sw, FIT['l2'])[0]
    converged = False
    for step in range(FIT['max_iterations']):
        loss, grad, hessian = simple.logistic_objective(weights, design, y, sw, FIT['l2'])
        if np.max(np.abs(grad)) <= FIT['gradient_tolerance']:
            converged = True
            break
        direction = np.linalg.solve(hessian, grad)
        for backtrack in range(30):
            rate = .5**backtrack
            candidate = weights-rate*direction
            next_loss = simple.logistic_objective(candidate, design, y, sw, FIT['l2'])[0]
            if next_loss <= loss-1e-4*rate*grad.dot(direction):
                weights = candidate
                break
        else:
            raise RuntimeError('Fixed Newton line search failed')
    final, grad, _ = simple.logistic_objective(weights, design, y, sw, FIT['l2'])
    converged = converged or bool(np.max(np.abs(grad)) <= FIT['gradient_tolerance'])
    if not converged or not np.isfinite(weights).all():
        raise RuntimeError('Fixed fit did not converge; no search/retry budget')
    return dict(feature_indices=list(indices), feature_names=[simple.FEATURES[i] for i in indices],
        mean=mean.tolist(), scale=scale.tolist(), weights=weights.tolist(),
        train_rows=len(y), bad_rows=bad, good_rows=good, initial_objective=initial,
        final_objective=final, iterations=step+1, max_abs_gradient=float(np.max(np.abs(grad))),
        converged=True, settings=deepcopy(FIT), initialization='all_zero')


def risk(model, features):
    indices = tuple(model['feature_indices'])
    if (indices not in tuple(ARMS.values())+((0, 1, 2),)
            or model['feature_names'] != [simple.FEATURES[i] for i in indices]
            or model['converged'] is not True or model['settings'] != FIT):
        raise ValueError('Wrong size model/features/settings')
    mean, scale, weights = [np.asarray(model[k], dtype=np.float64) for k in ('mean','scale','weights')]
    if (mean.shape != (len(indices),) or scale.shape != mean.shape
            or weights.shape != (len(indices)+1,) or not np.isfinite(np.r_[mean,scale,weights]).all()
            or np.any(scale <= 0)):
        raise ValueError('Invalid size model parameters')
    x = np.asarray(features, dtype=np.float64)
    if x.ndim not in (1, 2) or x.shape[-1] != 3 or not np.isfinite(x).all():
        raise ValueError('Expected the original three descriptors')
    return simple.sigmoid(weights[0]+((x[..., indices]-mean)/scale).dot(weights[1:]))


def groups(rows):
    result = {'all': rows}
    for field in ('domain', 'sequence'):
        for value in sorted({r[field] for r in rows}):
            result[field+':'+value] = [r for r in rows if r[field] == value]
    return result


def size_bad(row):
    return simple.geometry_errors(row['gt'], row['pred'])['size_max_relative'] > .1


def calibrate(rows, risks, target=.95):
    if not rows or any(r['split'] != 'val' for r in rows):
        raise ValueError('Supervised cutoff calibration is VAL-only, never TRAIN/TEST')
    requirements = {}
    for name, values in groups(rows).items():
        good = sorted(risks[r['image']] for r in values if r['pred'] is not None and not size_bad(r))
        count = required_good(len(good), target)
        requirements[name] = dict(good_outputs=len(good), required_good=count,
            first_whole_tie_risk_le=good[count-1] if count else None)
    points = [v['first_whole_tie_risk_le'] for v in requirements.values() if v['good_outputs']]
    if not points:
        raise ValueError('No correct VAL size output for calibration')
    return dict(risk_le=float(max(points)), good_retention_target=target,
        requirements=requirements, uses_VAL_GT=True, single_global_cutoff=True,
        calibration_not_independent_validation=True, tie_policy='accept_entire_cutoff_tie')


def unavailable(rows, accepted):
    run = longest = 0; previous = None
    for row in sorted(rows, key=lambda r: (r['sequence'],r['frame_id'],r['image'])):
        if previous is None or row['sequence'] != previous[0] or row['frame_id'] != previous[1]+1:
            run = 0
        run = 0 if row['image'] in accepted else run+1
        longest = max(longest, run)
        previous = (row['sequence'], row['frame_id'])
    return longest


def describe(rows, risks, cutoff, controls):
    output = {}
    for name, values in groups(rows).items():
        out = [r for r in values if r['pred'] is not None]
        ids = [r['image'] for r in out]
        bad = [size_bad(r) for r in out]
        kept = {image for image in ids if risks[image] <= cutoff}
        result = metrics.confusion(bad, [i in kept for i in ids], len(values), len(values))
        result.update(frames=len(values), output_frames=len(out), missing_outputs=len(values)-len(out),
            longest_flag_unavailable=unavailable(values, kept), global_risk_le=cutoff,
            rank=metrics.rank_metrics([risks[i] for i in ids], bad), matched_controls={})
        k = len(kept)
        for control, scores in controls.items():
            matched = metrics.same_count_reference([scores[i] for i in ids], bad, ids, [k])[k]
            ranked = sorted(ids, key=lambda i: (scores[i],i))[:k]
            matched.update(correct_accepted=k-matched['incorrect_accepted'],
                correct_rejected=result['good_outputs']-k+matched['incorrect_accepted'],
                longest_flag_unavailable=unavailable(values, set(ranked)))
            result['matched_controls'][control] = matched
        output[name] = result
    return output


def gate(stats, target=.95):
    """Predeclared strict benefit, no per-group/tie/continuity deterioration."""
    failures = []
    for group, p in stats.items():
        if p['correct_accepted'] < required_good(p['good_outputs'], target):
            failures.append(dict(group=group, check='good_retention'))
        for name in CONTROLS:
            control = p['matched_controls'][name]
            # Be conservative about a cut score tie; never choose ties using GT.
            if p['incorrect_accepted'] > control['bad_min_over_tie']:
                failures.append(dict(group=group, control=name, check='matched_FA_nonincrease'))
            if group == 'all' and p['incorrect_accepted'] >= control['bad_min_over_tie']:
                failures.append(dict(group=group, control=name, check='strict_overall_matched_FA_gain'))
            if p['longest_flag_unavailable'] > control['longest_flag_unavailable']:
                failures.append(dict(group=group, control=name, check='matched_continuity_nonincrease'))
    return dict(passed=not failures, failures=failures, target=target,
        continuity_allowed_extra_frames=0, score_tie_comparison='conservative_min_FA_bound')


def choose(results):
    passed = [(v['stats']['all']['incorrect_accepted'], -v['stats']['all']['correct_accepted'],name)
              for name,v in results.items() if v['gate']['passed']]
    if not passed:
        return None
    passed.sort()
    if len(passed) > 1 and passed[0][:2] == passed[1][:2]:
        return None  # Tie: preserve original policy, no TEST-based tiebreaking.
    return passed[0][2]


def scores(rows, original, models):
    api = binding.Sigma15Reliability(original, original['front_end'])
    output = {name: {} for name in CONTROLS+tuple(models)}
    for r in rows:
        if r['pred'] is None:
            for values in output.values(): values[r['image']] = None
            continue
        base = api.decide(r['pred'], r['image_size'])
        output['full_simple'][r['image']] = base['risks']['size']
        output['score_only'][r['image']] = float(1-r['pred'][5])
        features = simple.descriptor(r['pred'],r['image_size'])
        for name,model in models.items(): output[name][r['image']] = float(risk(model,features))
    return output


def decide(policy, pred_original, image_size):
    """Only current delivered OBB+score/image size; never GT/domain/frame state."""
    original = policy['frozen_original_policy']
    if (policy['protocol'] != VERSION or policy['arm'] not in ARMS
            or tuple(policy['size_model']['feature_indices']) != ARMS[policy['arm']]
            or policy['size_cutoff']['single_global_cutoff'] is not True):
        raise ValueError('Invalid frozen ablation policy')
    cutoff = policy['size_cutoff']['risk_le']
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError('Invalid frozen global size cutoff')
    base = binding.Sigma15Reliability(original, original['front_end']).decide(pred_original,image_size)
    if pred_original is not None:
        value = float(risk(policy['size_model'],simple.descriptor(pred_original,image_size)))
        base['risks']['size'] = value
        base['size_accepted'] = value <= cutoff
    return base


def component_report(rows, original, policies):
    """Full three-flag/joint reporting; original geometry and qualified angles."""
    api = binding.Sigma15Reliability(original,original['front_end'])
    decisions = {'original_simple': {r['image']:api.decide(r['pred'],r['image_size']) for r in rows}}
    decisions.update({name:{r['image']:decide(p,r['pred'],r['image_size']) for r in rows}
                      for name,p in policies.items()})
    result = {}
    for group,values in groups(rows).items():
        out = [r for r in values if r['pred'] is not None]
        hits = sum(simple.geometry_errors(r['gt'],r['pred'])['center_px']<15 for r in out)
        result[group] = dict(frames=len(values),output_frames=len(out),center_hits=hits,
            output_coverage=metrics.ratio(len(out),len(values)),
            center_hit_rate_on_outputs=metrics.ratio(hits,len(out)),
            all_frame_center_correct_coverage=metrics.ratio(hits,len(values)),methods={})
        for name,records in decisions.items():
            stats = {c:simple._component_stats(values,c,{r['image'] for r in values
                     if records[r['image']][c+'_accepted']}) for c in simple.COMPONENTS}
            kept = [r for r in out if all(records[r['image']][c+'_accepted'] for c in simple.COMPONENTS)]
            qualified = [r for r in kept if r['angle_axis_well_defined']]
            joint_correct = 0
            for row in qualified:
                e=simple.geometry_errors(row['gt'],row['pred'])
                joint_correct += int(e['center_px']<15 and e['size_max_relative']<=.1 and e['angle_deg']<=3)
            result[group]['methods'][name] = dict(components=stats,complete_obb=dict(
                accepted_frames=len(kept),qualified_frames=len(qualified),
                unassessed_accepted=len(kept)-len(qualified),joint_correct=joint_correct,
                joint_incorrect=len(qualified)-joint_correct,
                full_frame_joint_correct_coverage=metrics.ratio(joint_correct,len(values))))
    return result
