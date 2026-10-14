"""TRAIN-internal video transfer diagnostic; never exports an online policy."""
import math
from copy import deepcopy
import numpy as np
from crane_project.utils import port_reliability_remaining_error_support_v1 as support
from crane_project.utils import port_reliability_temporal_fusion_v1 as fusion

VERSION = 'port_reliability_video_transfer_v1'
SETTINGS = dict(correct_retention=.95, l2=.1, max_iterations=100,
                gradient_tolerance=1e-8, arms=['static3', 'fusion4'])
STATES = ('FA', 'FR', 'ED', 'CR', 'MISSING')


def wrong(row):
    return fusion.level.metrics.bad(row)


def counts(rows, accepted):
    values = {s: 0 for s in STATES}
    longest = current = 0
    prev = None
    for r in rows:
        bad = wrong(r)
        keep = r['image'] in accepted
        if bad is None:
            if keep:
                raise ValueError('A missing box cannot be accepted')
            state = 'MISSING'
        else:
            state = ('FA' if bad else 'CR') if keep else ('ED' if bad else 'FR')
        values[state] += 1
        adjacent = (prev is not None and prev['sequence'] == r['sequence']
                    and r['frame_id'] == prev['frame_id']+1
                    and not r['temporal'].get('acquisition_boundary', False))
        current = (current+1 if adjacent else 1) if state == 'FR' else 0
        longest = max(longest, current)
        prev = r
    good = values['CR']+values['FR']
    accepted_n = values['FA']+values['CR']
    outputs = len(rows)-values['MISSING']
    return dict(states=values, frames=len(rows), outputs=outputs, accepted=accepted_n,
                correct_retention=values['CR']/good if good else None,
                accepted_error_rate=values['FA']/accepted_n if accepted_n else None,
                acceptance_coverage=accepted_n/len(rows), output_coverage=outputs/len(rows),
                full_frame_correct_coverage=values['CR']/len(rows), longest_correct_FR=longest)


def point(rows, method, count=None, correct=None):
    """Offline matching. Whole cutoff ties for CR; identity ties for exact count."""
    ordered = sorted((r for r in rows if r['pred'] is not None),
                     key=lambda r: (r['risks'][method], r['image']))
    if (count is None) == (correct is None):
        raise ValueError('Exactly one matching target required')
    if count is not None:
        if not 0 <= count <= len(ordered):
            raise ValueError('Invalid count target')
        chosen = ordered[:count]
        boundary = chosen[-1]['risks'][method] if chosen else None
        before = [r for r in ordered if boundary is not None and r['risks'][method] < boundary]
        tied = [r for r in ordered if boundary is not None and r['risks'][method] == boundary]
        slots = count-len(before)
        fa = sum(wrong(r) for r in before)
        bounds = dict(FA_min=fa+max(0, slots-sum(not wrong(r) for r in tied)),
                      FA_max=fa+min(slots, sum(wrong(r) for r in tied)),
                      partial_tie=0 < slots < len(tied))
    else:
        if not 0 <= correct <= sum(wrong(r) is False for r in ordered):
            raise ValueError('Invalid correct target')
        good_risks = sorted(r['risks'][method] for r in ordered if not wrong(r))
        boundary = good_risks[correct-1] if correct else None
        chosen = [r for r in ordered if boundary is not None and r['risks'][method] <= boundary]
        bounds = None
    result = counts(rows, {r['image'] for r in chosen})
    result.update(offline_GT_comparison_only=True, risk_le=boundary,
                  exact_CR=result['states']['CR'] == correct if correct is not None else None,
                  tie_bounds=bounds)
    return result


def cutoff(fit_rows, method):
    """One global cutoff protecting 95% correct outputs in every FIT video."""
    by_video = {}
    for seq in sorted({r['sequence'] for r in fit_rows}):
        good = sorted(r['risks'][method] for r in fit_rows
                      if r['sequence'] == seq and wrong(r) is False)
        if not good:
            raise ValueError('No correct support in a fitting video')
        required = math.ceil(SETTINGS['correct_retention']*len(good))
        by_video[seq] = dict(correct=len(good), required=required, risk_le=good[required-1])
    return dict(risk_le=max(v['risk_le'] for v in by_video.values()),
                fit_videos=by_video, rule='max_of_fit_video_correct95',
                diagnostic_only=True, not_a_calibrated_probability=True)


def evaluate(rows, thresholds):
    result = {}
    for group, part in fusion.level.metrics.grouped(rows).items():
        required = math.ceil(.95*sum(wrong(r) is False for r in part))
        data = dict(formal=counts(part, fusion.level.formal_accept(part)), methods={})
        for method in ('static3', 'fusion4', 'full_simple', 'score_only'):
            kept = {r['image'] for r in part if r['pred'] is not None
                    and r['risks'][method] <= thresholds[method]['risk_le']}
            delivery = counts(part, kept)
            controls = {}
            if method in SETTINGS['arms']:
                for control in ('full_simple', 'score_only', 'static3'):
                    controls[control] = dict(
                        same_count=point(part, control, count=delivery['accepted']),
                        same_CR=point(part, control, correct=delivery['states']['CR']))
            data['methods'][method] = dict(
                delivery=delivery, AUROC=fusion.level.metrics.auc(part, method),
                offline_correct95=point(part, method, correct=required), controls=controls)
        result[group] = data
    return result


def fit_score(fit_rows, eval_rows):
    """Reuses old three-model fit: full static fallback + supported static/fusion."""
    models = fusion.fit(fit_rows)
    scored_fit = fusion.score(fit_rows, models)
    scored_eval = fusion.score(eval_rows, models)
    # static3 is the full-TRAIN 3-D judge; fusion4 keeps its original static fallback.
    for rows in (scored_fit, scored_eval):
        for r in rows:
            if r['pred'] is not None:
                x = fusion.simple.descriptor(r['pred'], r['image_size'])
                r['risks']['static3'] = float(fusion.simple.linear_risk(models['fallback3'], x[None])[0])
    thresholds = {m: cutoff(scored_fit, m) for m in ('static3', 'fusion4', 'full_simple', 'score_only')}
    return models, thresholds, scored_fit, scored_eval


def audit(rows, formal_model):
    support.validate(rows)
    original = deepcopy(rows)
    sequences = sorted({r['sequence'] for r in rows})
    if len(sequences) != 6:
        raise ValueError('Fixed six-video TRAIN required')
    models, thresholds, fit_scored, reference_rows = fit_score(rows, rows)
    for k in ('mean', 'scale', 'weights'):
        if not np.allclose(models['fallback3'][k], formal_model[k], atol=1e-12, rtol=0):
            raise ValueError('Full TRAIN static model does not reproduce formal simple')
    reference = dict(models=models, thresholds=thresholds, evaluation=evaluate(reference_rows, thresholds))
    folds = {}
    held_rows = []
    for seq in sequences:
        fitting = [r for r in rows if r['sequence'] != seq]
        held = [r for r in rows if r['sequence'] == seq]
        models, thresholds, scored_fit, scored_held = fit_score(fitting, held)
        if set(r['image'] for r in fitting) & set(r['image'] for r in held):
            raise ValueError('Held-video contamination')
        folds[seq] = dict(fit_images=[r['image'] for r in fitting],
                          held_images=[r['image'] for r in held], models=models,
                          fit_risks={r['image']: {m: r['risks'][m] for m in
                              ('static3', 'fusion4', 'full_simple', 'score_only')} for r in scored_fit},
                          thresholds=thresholds, fit_evaluation=evaluate(scored_fit, thresholds),
                          held_evaluation=evaluate(scored_held, thresholds))
        for r in scored_held:
            r['diagnostic_fold'] = seq
            held_rows.append(r)
    if rows != original:
        raise ValueError('Inputs were mutated')
    held_rows.sort(key=lambda r: (r['sequence'], r['frame_id'], r['image']))
    # Pooled transfer delivery uses each fold's pre-determined cutoff; no pooled
    # recalibration/AUC across differently fitted risk scales is reported.
    pooled = {}
    for method in ('static3', 'fusion4', 'full_simple', 'score_only'):
        accepted = {r['image'] for r in held_rows if r['pred'] is not None and
                    r['risks'][method] <= folds[r['sequence']]['thresholds'][method]['risk_le']}
        pooled[method] = {g: counts(part, accepted) for g, part in fusion.level.metrics.grouped(held_rows).items()}
    return dict(status='TRAIN_VIDEO_TRANSFER_DIAGNOSTIC_NO_ADOPTION', reference=reference,
                folds=folds, pooled_transfer_delivery=pooled, fits=21,
                held_arm_evaluations=12, detector_inferences=0, neural_updates=0,
                VAL_opened=False, TEST_opened=False, candidate_adopted=False), held_rows, reference_rows
