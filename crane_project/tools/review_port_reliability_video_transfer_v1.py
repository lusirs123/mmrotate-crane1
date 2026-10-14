#!/usr/bin/env python3
"""Independent fold isolation, stationary objective, scoring and metric review."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_reliability_video_transfer_v1 as runner


def bad(r):
    if r['pred'] is None:
        return None
    p, g = sorted(r['pred'][2:4]), sorted(r['gt'][2:4])
    return any(abs(a-b)/b > .1 for a, b in zip(p, g))


def descriptor(r, dimension):
    p = r['pred']; w, h = sorted(p[2:4], reverse=True)
    s = min(1-1e-6, max(1e-6, p[5]))
    x = [math.log(s/(1-s)), .5*math.log(w*h/(r['image_size'][0]*r['image_size'][1])), math.log(w/h)]
    if dimension == 4:
        x.append(r['temporal']['primary'])
    return x


def predict(model, r, dim):
    x = descriptor(r, dim)
    logit = model['weights'][0]+sum((a-m)/s*w for a, m, s, w in zip(
        x, model['mean'], model['scale'], model['weights'][1:]))
    return float(np.exp(-np.logaddexp(0., -logit)))


def check_models(models, fit):
    checks = 0
    for name, dim in [('fallback3', 3), ('static3', 3), ('fusion4', 4)]:
        part = [r for r in fit if r['pred'] is not None and
                (name == 'fallback3' or r['temporal']['available'])]
        m = models[name]; x = np.array([descriptor(r, dim) for r in part]); y = np.array([int(bad(r)) for r in part])
        mean = x.mean(0); scale = x.std(0); scale[scale < 1e-8] = 1
        if not np.allclose(m['mean'], mean, atol=1e-12, rtol=0) or not np.allclose(m['scale'], scale, atol=1e-12, rtol=0):
            raise ValueError('Normalizer uses wrong fitting membership')
        if name != 'fallback3' and m['train_images'] != [r['image'] for r in part]:
            raise ValueError('Supported fitting membership differs')
        if m['train_rows'] != len(part) or m['bad_rows'] != int(y.sum()) or m['good_rows'] != int((1-y).sum()):
            raise ValueError('Class counts differ')
        d = np.column_stack((np.ones(len(part)), (x-mean)/scale)); w = np.array(m['weights'])
        a = np.where(y == 1, .5/y.sum(), .5/(len(y)-y.sum()))
        logits = d@w; prob = np.exp(-np.logaddexp(0., -logits))
        penalty = w.copy(); penalty[0] = 0
        gradient = d.T@(a*(prob-y))+.1*penalty
        loss = float(a@(np.logaddexp(0., logits)-y*logits)+.05*(penalty@penalty))
        if np.max(np.abs(gradient)) > 1.0001e-8 or abs(loss-m['final_objective']) > 1e-12:
            raise ValueError('Not the prescribed converged balanced objective')
        checks += 1
    return checks


def score(rows, models):
    result = deepcopy(rows)
    for r in result:
        if r['pred'] is None:
            r['risks'].update(static3=None, fusion4=None)
        else:
            r['risks']['static3'] = predict(models['fallback3'], r, 3)
            supported = r['temporal']['available']
            r['risks']['fusion4'] = predict(models['fusion4'] if supported else models['fallback3'], r, 4 if supported else 3)
    return result


def states(rows, chosen):
    c = Counter(); prev = None; run = longest = 0
    for r in rows:
        error = bad(r); keep = r['image'] in chosen
        s = 'MISSING' if error is None else ('FA' if error else 'CR') if keep else ('ED' if error else 'FR')
        c[s] += 1
        adjacent = (prev is not None and prev['sequence'] == r['sequence'] and
                    r['frame_id'] == prev['frame_id']+1 and not r['temporal'].get('acquisition_boundary'))
        run = (run+1 if adjacent else 1) if s == 'FR' else 0
        longest = max(run, longest); prev = r
    accepted = c['FA']+c['CR']; good = c['CR']+c['FR']
    return dict(states={s: c[s] for s in ('FA', 'FR', 'ED', 'CR', 'MISSING')},
        frames=len(rows), outputs=len(rows)-c['MISSING'], accepted=accepted,
        correct_retention=c['CR']/good if good else None,
        accepted_error_rate=c['FA']/accepted if accepted else None,
        acceptance_coverage=accepted/len(rows), output_coverage=(len(rows)-c['MISSING'])/len(rows),
        full_frame_correct_coverage=c['CR']/len(rows), longest_correct_FR=longest)


def check_summary(actual, rows, chosen):
    for k, v in states(rows, chosen).items():
        if actual[k] != v:
            raise ValueError('Independent four states/coverage/FR run differ: '+k)


def check_point(actual, rows, method, count=None, correct=None):
    ordered = sorted((r for r in rows if r['pred'] is not None), key=lambda r: (r['risks'][method], r['image']))
    if count is not None:
        chosen = ordered[:count]; boundary = chosen[-1]['risks'][method] if chosen else None
        before = [r for r in ordered if boundary is not None and r['risks'][method] < boundary]
        tied = [r for r in ordered if boundary is not None and r['risks'][method] == boundary]
        slots = count-len(before); n = sum(bad(r) for r in before)
        bounds = dict(FA_min=n+max(0, slots-sum(not bad(r) for r in tied)),
                      FA_max=n+min(slots, sum(bad(r) for r in tied)), partial_tie=0 < slots < len(tied))
        if actual['tie_bounds'] != bounds:
            raise ValueError('Tie bounds differ')
    else:
        values = sorted(r['risks'][method] for r in ordered if not bad(r))
        boundary = values[correct-1] if correct else None
        chosen = [r for r in ordered if boundary is not None and r['risks'][method] <= boundary]
        if actual['exact_CR'] != (sum(not bad(r) for r in chosen) == correct):
            raise ValueError('CR tie handling differs')
    if actual['risk_le'] != boundary or not actual['offline_GT_comparison_only']:
        raise ValueError('Offline point identity differs')
    check_summary(actual, rows, {r['image'] for r in chosen})


def check_thresholds(thresholds, fitting):
    seqs = sorted({r['sequence'] for r in fitting})
    for method, threshold in thresholds.items():
        expected = {}
        for seq in seqs:
            good = sorted(r['risks'][method] for r in fitting if r['sequence'] == seq and bad(r) is False)
            n = math.ceil(.95*len(good))
            expected[seq] = dict(correct=len(good), required=n, risk_le=good[n-1])
        # Independent log arithmetic differs by roundoff: membership is exact,
        # values checked numerically; actual saved thresholds define deliveries.
        for seq in seqs:
            a, e = threshold['fit_videos'][seq], expected[seq]
            if a['correct'] != e['correct'] or a['required'] != e['required'] or abs(a['risk_le']-e['risk_le']) > 1e-12:
                raise ValueError('Held-video leakage into threshold')
        if set(threshold['fit_videos']) != set(seqs) or abs(threshold['risk_le']-max(v['risk_le'] for v in expected.values())) > 1e-12:
            raise ValueError('Wrong global fitting threshold')


def check_evaluation(evaluation, rows, thresholds):
    checks = 0
    groups = dict(all=rows)
    for field in ('domain', 'sequence'):
        for value in sorted({r[field] for r in rows}):
            groups[field+':'+value] = [r for r in rows if r[field] == value]
    if set(evaluation) != set(groups):
        raise ValueError('Group scope differs')
    for g, part in groups.items():
        data = evaluation[g]
        check_summary(data['formal'], part, {r['image'] for r in part if r['original_simple_decision']['size_accepted']})
        for method, entry in data['methods'].items():
            kept = {r['image'] for r in part if r['pred'] is not None and r['risks'][method] <= thresholds[method]['risk_le']}
            check_summary(entry['delivery'], part, kept)
            check_point(entry['offline_correct95'], part, method, correct=math.ceil(.95*sum(bad(r) is False for r in part)))
            good = np.array([r['risks'][method] for r in part if bad(r) is False])
            errors = np.array([r['risks'][method] for r in part if bad(r) is True])
            auc = float(np.mean((errors[:, None] > good[None]).astype(float)+.5*(errors[:, None] == good[None]))) if len(errors) and len(good) else None
            if (auc is None) != (entry['AUROC'] is None) or (auc is not None and abs(auc-entry['AUROC']) > 1e-12):
                raise ValueError('Pairwise AUROC differs')
            for control, values in entry['controls'].items():
                check_point(values['same_count'], part, control, count=entry['delivery']['accepted'])
                check_point(values['same_CR'], part, control, correct=entry['delivery']['states']['CR'])
            checks += 1
    return checks


def review(out, allow_absent=False):
    p, pins, absent, original, policy = runner.load(allow_absent)
    report = json.loads((out/'report.json').read_text()); result = report['result']
    held = runner.parent.io.read(out/'held_TRAIN.jsonl.gz')
    reference = runner.parent.io.read(out/'reference_TRAIN.jsonl.gz')
    if report['input_pins'] != pins or report['settings'] != runner.core.SETTINGS or report['frames'] != len(original):
        raise ValueError('Report binding differs')
    if result['fits'] != 21 or result['held_arm_evaluations'] != 12 or any(result[k] for k in ('VAL_opened', 'TEST_opened', 'neural_updates', 'detector_inferences', 'candidate_adopted')):
        raise ValueError('Scope differs')
    if any(report[k] for k in ('formal_policy_changed', 'geometry_changed', 'output_identity_changed')):
        raise ValueError('Formal pipeline changed')
    seqs = sorted({r['sequence'] for r in original})
    if set(result['folds']) != set(seqs):
        raise ValueError('Fold scope differs')
    checks = 0
    for actual in (held, reference):
        if [r['image'] for r in actual] != [r['image'] for r in original]:
            raise ValueError('Output identity differs')
        for a, r in zip(actual, original):
            if any(a[k] != v for k, v in r.items() if k != 'risks'):
                raise ValueError('Input/prediction/GT/history/formal decision changed')
            models = result['reference']['models'] if actual is reference else result['folds'][r['sequence']]['models']
            predicted = score([r], models)[0]
            for key in ('static3', 'fusion4', 'full_simple', 'score_only'):
                x, y = a['risks'][key], predicted['risks'][key]
                if (x is None) != (y is None) or (x is not None and abs(x-y) > 1e-12):
                    raise ValueError('Independent risk differs')
            checks += 1
    ref = result['reference']; checks += check_models(ref['models'], original)
    for k in ('mean', 'scale', 'weights'):
        if not np.allclose(ref['models']['fallback3'][k], policy['models']['size'][k], atol=1e-12, rtol=0):
            raise ValueError('In-sample static3 not formal simple')
    check_thresholds(ref['thresholds'], reference)
    checks += check_evaluation(ref['evaluation'], reference, ref['thresholds'])
    for seq, fold in result['folds'].items():
        fitting = [r for r in original if r['sequence'] != seq]
        testing = [r for r in held if r['sequence'] == seq]
        if fold['fit_images'] != [r['image'] for r in fitting] or fold['held_images'] != [r['image'] for r in testing]:
            raise ValueError('Video exclusion differs')
        checks += check_models(fold['models'], fitting)
        scored_fit = score(fitting, fold['models'])
        check_thresholds(fold['thresholds'], scored_fit)
        # Check independent arithmetic, then use saved values for exact
        # threshold ties. No optimizer is rerun by the reviewer.
        if set(fold['fit_risks']) != {r['image'] for r in fitting}:
            raise ValueError('Fit-score membership differs')
        saved_fit = deepcopy(fitting)
        for a, b in zip(saved_fit, scored_fit):
            saved = fold['fit_risks'][a['image']]
            if set(saved) != {'static3', 'fusion4', 'full_simple', 'score_only'}:
                raise ValueError('Fit risk schema differs')
            for m, x in saved.items():
                y = b['risks'][m]
                if (x is None) != (y is None) or (x is not None and abs(x-y) > 1e-12):
                    raise ValueError('Independent fit risk differs')
            a['risks'].update(saved)
        checks += check_evaluation(fold['fit_evaluation'], saved_fit, fold['thresholds'])
        checks += check_evaluation(fold['held_evaluation'], testing, fold['thresholds'])
    for method, groups in result['pooled_transfer_delivery'].items():
        kept = {r['image'] for r in held if r['pred'] is not None and r['risks'][method] <= result['folds'][r['sequence']]['thresholds'][method]['risk_le']}
        for group, actual in groups.items():
            part = held if group == 'all' else [r for r in held if r[group.split(':')[0]] == group.split(':')[1]]
            check_summary(actual, part, kept); checks += 1
    return dict(passed=True, rows=len(original), independently_checked_scores=2*len(original),
        independent_checks=checks, report_sha256=runner.parent.io.sha(out/'report.json'),
        absent_frontend=absent, VAL_opened=False, TEST_opened=False,
        checked=['fold isolation', 'fit-only normalizers', 'balanced objective stationary gradient',
                 'score/fallback', 'fit-only cutoff', 'four states', 'pairwise AUC', 'ties', 'FR runs'])


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--out', type=Path, required=True)
    a.add_argument('--allow-absent-frontend', action='store_true'); a.add_argument('--receipt', default='server_review.json')
    args = a.parse_args(); value = review(args.out, args.allow_absent_frontend)
    runner.parent.io.write(args.out/args.receipt, value); print(value)
