"""Descriptive frozen-score diagnostics, with NumPy only and no fitting.

Bad geometry is the positive class. Larger risk means more likely bad.
All curve thresholds are offline diagnostic points, never a new policy.
"""
from copy import deepcopy

import numpy as np

from crane_project.utils import port_simple_component_reliability_v1 as simple

VERSION = 'port_reliability_separability_v1'
METHODS = ('simple', 'score_only')


def ratio(a, b):
    return a/b if b else None


def checked_arrays(risks, bad):
    r = np.asarray(risks, dtype=np.float64)
    y = np.asarray(bad)
    if (r.ndim != 1 or y.shape != r.shape or not np.isfinite(r).all()
            or np.any((r < 0) | (r > 1)) or not np.isin(y, [0, 1]).all()):
        raise ValueError('Require finite risk [0,1] and binary bad-geometry labels')
    return r, y.astype(np.int64)


def rank_metrics(risks, bad):
    """Error-class AP (step integration) and AUROC, grouping exact score ties."""
    r, y = checked_arrays(risks, bad)
    positive, negative = int(y.sum()), int(len(y)-y.sum())
    result = dict(outputs=len(y), bad=positive, good=negative,
        error_prevalence=ratio(positive, len(y)), both_classes=bool(positive and negative),
        unique_risks=len(np.unique(r)), error_average_precision=None, error_auroc=None)
    if not len(r):
        return result
    order = np.argsort(-r, kind='stable')
    sorted_r, sorted_y = r[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_r[:-1] != sorted_r[1:])+1, len(r)]
    tp = np.cumsum(sorted_y)[ends-1].astype(float)
    fp = ends-tp
    if positive:
        recalls = tp/positive
        result['error_average_precision'] = float(np.sum(np.diff(np.r_[0., recalls])*(tp/ends)))
    if positive and negative:
        tpr, fpr = np.r_[0., tp/positive], np.r_[0., fp/negative]
        result['error_auroc'] = float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
    return result


def distribution(risks):
    x = np.asarray(risks, dtype=float)
    if not len(x):
        return dict(count=0, mean=None, std=None, min=None, max=None, quantiles=None,
                    histogram_edges=np.linspace(0, 1, 11).tolist(), histogram_counts=[0]*10)
    return dict(count=len(x), mean=float(x.mean()), std=float(x.std()),
        min=float(x.min()), max=float(x.max()),
        quantiles={name: float(value) for name, value in zip(
            ('p05', 'p25', 'p50', 'p75', 'p95'), np.percentile(x, [5, 25, 50, 75, 95]))},
        histogram_edges=np.linspace(0, 1, 11).tolist(),
        histogram_counts=np.histogram(x, bins=np.linspace(0, 1, 11))[0].tolist())


def confusion(bad, accepted, total_frames, eligible_frames):
    y = np.asarray(bad, dtype=bool); a = np.asarray(accepted, dtype=bool)
    if (a.shape != y.shape or y.ndim != 1 or not 0 <= len(y) <= eligible_frames <= total_frames):
        raise ValueError('Invalid assessment mask or coverage denominators')
    ca, ba = int(np.sum(~y & a)), int(np.sum(y & a))
    gr, br = int(np.sum(~y & ~a)), int(np.sum(y & ~a))
    return counts_metrics(ca, ba, gr, br, total_frames, eligible_frames)


def counts_metrics(ca, ba, gr, br, total_frames, eligible_frames):
    n, k, good, errors = ca+ba+gr+br, ca+ba, ca+gr, ba+br
    return dict(assessed_outputs=n, good_outputs=good, bad_outputs=errors,
        accepted_assessed_outputs=k, correct_accepted=ca, incorrect_accepted=ba,
        correct_rejected=gr, incorrect_rejected=br, state_errors=ba+gr,
        state_accuracy_on_assessed_outputs=ratio(ca+br, n),
        balanced_state_accuracy=(ca/good+br/errors)/2 if good and errors else None,
        incorrect_acceptance_rate_on_bad=ratio(ba, errors),
        error_detection_rate=ratio(br, errors), good_retention_rate=ratio(ca, good),
        incorrect_rejection_rate_on_good=ratio(gr, good),
        correctness_on_accepted=ratio(ca, k), error_fraction_on_accepted=ratio(ba, k),
        error_precision_on_rejected=ratio(br, n-k),
        acceptance_coverage_on_assessed_outputs=ratio(k, n),
        assessed_acceptance_coverage_on_eligible_frames=ratio(k, eligible_frames),
        assessed_acceptance_coverage_on_all_frames=ratio(k, total_frames),
        correct_accepted_coverage_on_eligible_frames=ratio(ca, eligible_frames),
        correct_accepted_coverage_on_all_frames=ratio(ca, total_frames),
        unavoidable_bad_at_same_count=max(0, k-good),
        max_correctness_at_same_count=ratio(min(k, good), k))


def same_count_reference(risks, bad, images, counts):
    """Exact-count score comparison, with label-free image tie ordering.

    Tie min/max are offline bounds only; they never choose accepted frames.
    Expected count assumes a random subset of the boundary tie, not deployment.
    """
    r, y = checked_arrays(risks, bad)
    if len(images) != len(y) or len(set(images)) != len(images):
        raise ValueError('Same-count comparison requires unique image identities')
    order = np.lexsort((np.asarray(images, dtype=str), r))
    sr, sy = r[order], y[order]
    prefix = np.r_[0, np.cumsum(sy)]
    result = {}
    for k in sorted(set(counts)):
        if not 0 <= k <= len(r):
            raise ValueError('Invalid matched acceptance count')
        if k == 0:
            result[k] = dict(accepted=0, incorrect_accepted=0, boundary_risk=None,
                boundary_tie_size=0, boundary_tie_selected=0, bad_min_over_tie=0,
                bad_max_over_tie=0, bad_expected_random_tie=0.)
            continue
        value = sr[k-1]
        lo, hi = int(np.searchsorted(sr, value, 'left')), int(np.searchsorted(sr, value, 'right'))
        before, tie_bad, selected = int(prefix[lo]), int(prefix[hi]-prefix[lo]), k-lo
        tie_good = hi-lo-tie_bad
        result[k] = dict(accepted=k, incorrect_accepted=int(prefix[k]), boundary_risk=float(value),
            boundary_tie_size=hi-lo, boundary_tie_selected=selected,
            bad_min_over_tie=before+max(0, selected-tie_good),
            bad_max_over_tie=before+min(selected, tie_bad),
            bad_expected_random_tie=before+selected*tie_bad/(hi-lo))
    return result


def acceptance_curve(risks, bad, images, reference_risks, total_frames, eligible_frames):
    """Whole-tie risk<=threshold sweep; missing/ineligible outputs are excluded."""
    r, y = checked_arrays(risks, bad)
    if not 0 <= len(y) <= eligible_frames <= total_frames:
        raise ValueError('Invalid curve denominators')
    order = np.argsort(r, kind='stable'); sr, sy = r[order], y[order]
    ends = (np.r_[np.flatnonzero(sr[:-1] != sr[1:])+1, len(sr)] if len(sr)
            else np.array([], dtype=int))
    prefix = np.r_[0, np.cumsum(sy)]
    matched = same_count_reference(reference_risks, y, images, [0]+ends.tolist())
    n, all_bad = len(y), int(y.sum())
    points = []
    for k in [0]+ends.tolist():
        ba = int(prefix[k]); ca = k-ba
        point = counts_metrics(ca, ba, n-all_bad-ca, all_bad-ba, total_frames, eligible_frames)
        reference = matched[k]
        point.update(accept_rule='none' if k == 0 else 'risk_le',
            risk_le=None if k == 0 else float(sr[k-1]),
            same_count_score=reference,
            bad_minus_same_count_score=ba-reference['incorrect_accepted'],
            bad_minus_same_count_score_min_bound=ba-reference['bad_max_over_tie'],
            bad_minus_same_count_score_max_bound=ba-reference['bad_min_over_tie'])
        points.append(point)
    return points


def prepare_rows(rows, policy):
    runtime = simple.SimpleComponentReliability(policy)
    prepared = []
    for row in rows:
        if (row['split'] not in ('train', 'train_sim', 'val')
                or type(row['angle_axis_well_defined']) is not bool
                or type(row['train_angle_eligible']) is not bool):
            raise ValueError('Only typed TRAIN/VAL qualification records allowed')
        g = simple.canonical(row['gt'])
        if row['angle_axis_well_defined'] != bool(g[2]/g[3] >= 1.2):
            raise ValueError('Saved evaluation direction qualification differs: '+row['image'])
        size = np.asarray(row['image_size'], dtype=float)
        if size.shape != (2,) or not np.isfinite(size).all() or min(size) <= 0:
            raise ValueError('Invalid original image size')
        methods = {m: runtime.decide(row['pred'], row['image_size'], m) for m in ('raw',)+METHODS}
        if any(v['raw_b_output'] != row['pred'] or v['center_accepted'] != (row['pred'] is not None)
               for v in methods.values()):
            raise ValueError('Raw B output or centre retention changed')
        prepared.append(dict(deepcopy(row), methods=methods,
            offline_errors=simple.geometry_errors(row['gt'], row['pred']) if row['pred'] is not None else None))
    return prepared


def summarize_component(values, component, policy):
    training_direction = component == 'angle_training_qualification'
    base = 'angle' if training_direction else component
    eligible = [r for r in values if base != 'angle' or
        r['train_angle_eligible' if training_direction else 'angle_axis_well_defined']]
    outputs = [r for r in eligible if r['pred'] is not None]
    key, limit = ('size_max_relative', .1) if base == 'size' else ('angle_deg', 3.)
    bad = np.array([r['offline_errors'][key] > limit for r in outputs], dtype=bool)
    images = [r['image'] for r in outputs]
    risks = {m: np.array([r['methods'][m]['risks'][base] for r in outputs]) for m in METHODS}
    summary = dict(frames=len(values), eligible_frames=len(eligible),
        assessed_outputs=len(outputs), missing_eligible_outputs=len(eligible)-len(outputs),
        ineligible_frames=len(values)-len(eligible), bad_outputs=int(bad.sum()),
        good_outputs=int(len(bad)-bad.sum()),
        qualification='saved_train_direction_eligibility' if training_direction else
            ('saved_GT_aspect_ge_1.2' if base == 'angle' else 'all_frames'),
        raw=confusion(bad, np.ones(len(bad), dtype=bool), len(values), len(eligible)), methods={})
    curves = {}
    eligible_images = {r['image'] for r in eligible}
    for method in METHODS:
        risk = risks[method]
        # Use the original global cutoffs, never a cutoff fitted to this group.
        cutoff = policy['cutoffs'][method][base]['risk_le']
        accepted = risk <= cutoff
        fixed = confusion(bad, accepted, len(values), len(eligible))
        all_kept = [r for r in values if r['methods'][method][base+'_accepted']]
        fixed.update(frozen_global_risk_le=cutoff, flag_accepted_all_frames=len(all_kept),
            flag_coverage_all_frames=ratio(len(all_kept), len(values)),
            unassessed_accepted=sum(r['image'] not in eligible_images for r in all_kept))
        reference = same_count_reference(risks['score_only'], bad, images,
                                        [fixed['accepted_assessed_outputs']])[fixed['accepted_assessed_outputs']]
        fixed.update(same_count_score=reference,
            bad_minus_same_count_score=fixed['incorrect_accepted']-reference['incorrect_accepted'])
        summary['methods'][method] = dict(rank=rank_metrics(risk, bad),
            distributions=dict(good=distribution(risk[~bad]), bad=distribution(risk[bad])),
            fixed_workpoint=fixed)
        curves[method] = acceptance_curve(risk, bad, images, risks['score_only'], len(values), len(eligible))
    summary['simple_minus_score'] = {
        metric: (summary['methods']['simple']['rank'][metric]-summary['methods']['score_only']['rank'][metric]
                 if summary['methods']['simple']['rank'][metric] is not None
                 and summary['methods']['score_only']['rank'][metric] is not None else None)
        for metric in ('error_auroc', 'error_average_precision')}
    return summary, curves


def analyze(rows, policy, role):
    if role not in ('train', 'val') or not rows:
        raise ValueError('Only nonempty TRAIN/VAL diagnostics allowed')
    if any(r['split'] not in (('train', 'train_sim') if role == 'train' else ('val',)) for r in rows):
        raise ValueError('Split roles cannot be mixed or replaced by TEST')
    before = simple.fingerprint([rows, policy])
    prepared = prepare_rows(rows, policy)
    groups = dict(all=prepared)
    for field in ('domain', 'sequence'):
        groups.update({field+':'+v: [r for r in prepared if r[field] == v]
                       for v in sorted({r[field] for r in prepared})})
    strata, curve_rows = {}, []
    for group, values in groups.items():
        outputs = [r for r in values if r['pred'] is not None]
        hits = sum(r['offline_errors']['center_px'] < 15 for r in outputs)
        strata[group] = dict(frames=len(values), output_frames=len(outputs), center_hits=hits,
            output_coverage=ratio(len(outputs), len(values)),
            center_hit_rate_on_outputs=ratio(hits, len(outputs)),
            all_frame_center_correct_coverage=ratio(hits, len(values)), components={})
        for component in ('size', 'angle')+(('angle_training_qualification',) if role == 'train' else ()):
            summary, curves = summarize_component(values, component, policy)
            strata[group]['components'][component] = summary
            for method, points in curves.items():
                curve_rows.extend(dict(role=role, group=group, component=component, method=method, **p)
                                  for p in points)
    if before != simple.fingerprint([rows, policy]):
        raise ValueError('Frozen input or policy changed during diagnostics')
    return strata, curve_rows, prepared
