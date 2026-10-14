#!/usr/bin/env python3
"""Independent weighted objective stationarity, frozen flags and full counts."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_reliability_hard_error_pauc_v1 as runner
from crane_project.tools.review_port_reliability_partial_auc_v1 import vector, equivalent_metrics
from crane_project.tools.review_port_reliability_temporal_fusion_v1 import check_summary, bad, rank_ids
core = runner.core


def fit_check(model, rows, hard, base):
    supported = [r for r in rows if r['pred'] is not None and r['temporal']['available']]
    if model['train_images'] != [r['image'] for r in supported]: raise ValueError('Fit membership changed')
    y = np.array([bool(bad(r)) for r in supported]); wrong_rows = [r for r in supported if bad(r)]
    if model['wrong_images'] != [r['image'] for r in wrong_rows]: raise ValueError('Error weight alignment differs')
    q = np.array([2. if model['weighted'] and r['image'] in hard['images'] else 1. for r in wrong_rows]); q /= q.sum()
    if not np.array_equal(q, model['error_weights']): raise ValueError('Weights differ')
    m = base['fusion4']; z = (np.array([vector(r) for r in supported]) - m['mean']) / m['scale']
    d = z[~y, None, :] - z[None, y, :]; w = np.array(model['weights'][1:])
    u = np.einsum('abi,i->ab', d, w); probability = np.exp(-np.logaddexp(0., -u))
    h = np.sum(np.logaddexp(0., u) * q[None, :], axis=1)
    jac = np.sum(probability[:, :, None] * d * q[None, :, None], axis=1)
    a = (h - model['tail_threshold']) / .1; outer = np.exp(-np.logaddexp(0., -a))
    loss = model['tail_threshold'] + .1 * np.logaddexp(0., a).mean() / .05 + .05 * w.dot(w)
    grad = np.r_[np.mean(outer[:, None] * jac, axis=0) / .05 + .1 * w, 1 - outer.mean() / .05]
    if abs(grad).max() > 1.1e-8 or abs(loss - model['final_objective']) > 1e-12:
        raise ValueError('Independent objective not stationary')
    if model['mean'] != m['mean'] or model['scale'] != m['scale'] or model['weights'][0] != m['weights'][0]:
        raise ValueError('Normalization/intercept changed')
    if model['settings'] != core.SETTINGS: raise ValueError('Fit contract changed')
    return float(abs(grad).max())


def review(result, allow_absent=False):
    p, pins, absent, previous = runner.load(allow_absent)
    report = json.loads((result / 'report.json').read_text())
    models = json.loads((result / 'models.json').read_text()); hard = json.loads((result / 'hard_TRAIN.json').read_text())
    cal = json.loads((result / 'calibration.json').read_text())
    if report['input_pins'] != pins or report['models'] != models or report['hard_TRAIN'] != hard or report['calibration'] != cal:
        raise ValueError('Saved artifacts differ')
    checks = total = 0; original_train = runner.io.read(ROOT / p['reference_result'] / 'scored_TRAIN.jsonl.gz')
    wrong = [r for r in original_train if r['temporal']['available'] and bad(r) is True]
    boundary = sorted(r['risks']['partial_auc4'] for r in wrong)[math.ceil(.25 * len(wrong)) - 1]
    expected = [r['image'] for r in wrong if r['risks']['partial_auc4'] <= boundary]
    if hard['images'] != expected or hard['TRAIN_risk_boundary'] != boundary or not hard['no_VAL_threshold_used']:
        raise ValueError('Hard set used wrong role/threshold')
    gradients = {m: fit_check(model, original_train, hard, report['base_models']) for m, model in models.items()}
    if not np.allclose(models[core.CONTROL]['weights'], previous['model']['weights'], atol=1e-11, rtol=0):
        raise ValueError('Frozen partial-AUC reproduction differs')
    for role in ('TRAIN', 'VAL'):
        saved = runner.io.read(result / ('scored_' + role + '.jsonl.gz'))
        original = runner.io.read(ROOT / p['reference_result'] / ('scored_' + role + '.jsonl.gz'))
        if len(saved) != len(original): raise ValueError('Coverage changed')
        for r, old in zip(saved, original):
            if {k: v for k, v in r.items() if k != 'risks'} != {k: v for k, v in old.items() if k != 'risks'}:
                raise ValueError('Frozen frame/decision changed')
            if any(r['risks'][k] != v for k, v in old['risks'].items()): raise ValueError('Old risk changed')
            for method, model in models.items():
                expected = old['risks']['partial_auc4']
                if old['pred'] is not None and old['temporal']['available']:
                    z = (vector(old) - model['mean']) / model['scale']
                    expected = math.exp(-float(np.logaddexp(0., -(model['weights'][0] + z.dot(model['weights'][1:])))))
                actual = r['risks'][method]
                if (actual is None) != (expected is None) or (actual is not None and abs(actual - expected) > 1e-12):
                    raise ValueError('Independent prediction differs')
            total += 1; checks += 1
        for group, part in core.level.metrics.grouped(saved).items():
            for method in models:
                ids = {r['image'] for r in part if r['pred'] is not None and r['risks'][method] <=
                    (cal[method]['risk_le'] if r['temporal']['available'] else core.FALLBACK_CUTOFF)}
                check_summary(part, ids, report['roles'][role]['groups'][group][method]['candidate']); checks += 1
                entry = report['roles'][role]['groups'][group][method]
                for name, matches in entry['controls'].items():
                    held = {r['image'] for r in part if r['pred'] is not None and not r['temporal']['available'] and r['risks']['partial_auc4'] <= core.FALLBACK_CUTOFF} if name in (core.CONTROL, 'partial_auc4', 'crossfit_fusion') else set()
                    active = [r for r in part if r['pred'] is not None and r['temporal']['available']] if name in (core.CONTROL, 'partial_auc4', 'crossfit_fusion') else part
                    held_good = sum(r['image'] in held and bad(r) is False for r in part)
                    for kind, summary in matches.items():
                        target = len(ids) - len(held) if kind == 'same_count' else entry['candidate']['states']['CR'] - held_good
                        chosen = held | (rank_ids(active, name, count=target) if kind == 'same_count' else rank_ids(active, name, correct=target))
                        check_summary(part, chosen, summary); checks += 1
                held = core.fixed(part)
                ref_ids = held | {r['image'] for r in part if r['pred'] is not None and r['temporal']['available'] and r['risks']['crossfit_fusion'] <= core.FUSION_CUTOFF}
                check_summary(part, ref_ids, entry['frozen_fallback_reference']); checks += 1
                active = [r for r in part if r['pred'] is not None and r['temporal']['available']]
                need = sum(r['image'] in ref_ids and bad(r) is False for r in part) - sum(r['image'] in held and bad(r) is False for r in part)
                check_summary(part, held | rank_ids(active, method, correct=need), entry['candidate_at_reference_CR']); checks += 1
        if not equivalent_metrics(report['roles'][role], core.evaluate(saved, cal)):
            raise ValueError('Full metrics/matched counts differ')
        if role == 'VAL':
            for method in models:
                thresholds = []
                for group, part in core.level.metrics.grouped(saved).items():
                    good = [r for r in part if bad(r) is False]
                    held = sum(not r['temporal']['available'] and r['risks']['partial_auc4'] <= core.FALLBACK_CUTOFF for r in good)
                    need = max(0, math.ceil(.95 * len(good)) - held)
                    values = sorted(r['risks'][method] for r in good if r['temporal']['available'])
                    thresholds.append(values[need - 1] if need else 0.)
                if cal[method]['risk_le'] != max(thresholds): raise ValueError('Calibration differs')
    normalize = lambda g: dict(g, failures=sorted(g['failures'], key=lambda v: json.dumps(v, sort_keys=True)))
    if not equivalent_metrics(normalize(report['gate']), normalize(core.gate(report['roles']['VAL']))): raise ValueError('Gate differs')
    if report['fit_calls'] != 2 or not report['model_saved_before_VAL'] or not report['hard_set_saved_before_fit_and_VAL']:
        raise ValueError('Fit order/number differs')
    for k in ('detector_inferences', 'detector_updates', 'neural_training_updates', 'TEST_opened', 'formal_policy_changed', 'candidate_adopted'):
        if report[k]: raise ValueError('Scope exceeded')
    return dict(passed=True, rows=total, independent_checks=checks, gradients=gradients,
        absent_frontend=absent, report_sha256=runner.io.sha(result / 'report.json'),
        fallback_flags_preserved=True, TEST_opened=False)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--result', type=Path, required=True); a.add_argument('--out', type=Path, required=True)
    a.add_argument('--allow-absent-frontend', action='store_true'); args = a.parse_args()
    runner.io.write(args.out, review(args.result, args.allow_absent_frontend)); print('Independent hard-error pAUC review passed')
