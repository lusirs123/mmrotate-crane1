#!/usr/bin/env python3
"""Independent formulas for normalization, stationarity, scores and state counts."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_reliability_pixel_pauc_v1 as runner
from crane_project.tools.review_port_reliability_partial_auc_v1 import vector, equivalent_metrics
from crane_project.tools.review_port_reliability_temporal_fusion_v1 import check_summary, bad, rank_ids
core = runner.core


def supported(row):
    e = row['pixel_evidence']
    return row['pred'] is not None and row['temporal']['available'] and e is not None and all(e[k]['supported'] for k in ('L', 'S'))


def raw_vector(row, dim):
    x = vector(row).tolist()
    if dim == 6:
        x += [row['pixel_evidence'][k]['mismatch'] for k in ('L', 'S')]
    return np.asarray(x)


def held_ids(rows):
    return {r['image'] for r in rows if r['pred'] is not None and not supported(r) and r['risks']['partial_auc4'] <= .4094748402105834}


def matched_ids(rows, method, count=None, correct=None, locked=True):
    held = held_ids(rows) if locked else set()
    part = [r for r in rows if supported(r)] if locked else rows
    if count is not None:
        return held | rank_ids(part, method, count=count-len(held))
    required = correct-sum(r['image'] in held and bad(r) is False for r in rows)
    if not 0 <= required <= sum(bad(r) is False for r in part):
        return None
    return held | rank_ids(part, method, correct=required)


def fit_check(model, rows, prepared):
    part = [r for r in rows if supported(r)]; dim = model['dim']
    if model['train_images'] != [r['image'] for r in part] or model['wrong_images'] != [r['image'] for r in part if bad(r) is True]:
        raise ValueError('Paired TRAIN fit identity/labels differ')
    if model['mean'] != prepared['mean'][:dim] or model['scale'] != prepared['scale'][:dim] or model['weights'][0] != prepared['intercept'] or model['settings'] != core.SETTINGS or model['error_weighting'] != 'uniform':
        raise ValueError('Normalization/intercept/objective contract differs')
    z = (np.array([raw_vector(r, dim) for r in part])-model['mean'])/model['scale']
    y = np.array([bool(bad(r)) for r in part]); d = z[~y, None, :]-z[None, y, :]
    slopes = np.asarray(model['weights'][1:]); tau = model['tail_threshold']
    u = np.einsum('gbi,i->gb', d, slopes)
    probability = np.exp(-np.logaddexp(0., -u))
    h = np.sum(np.logaddexp(0., u), axis=1)/int(y.sum())
    jac = np.einsum('gb,gbi->gi', probability, d)/int(y.sum())
    tail = np.exp(-np.logaddexp(0., -(h-tau)/.1))
    loss = tau+.1*np.logaddexp(0., (h-tau)/.1).mean()/.05+.05*slopes.dot(slopes)
    grad = np.r_[np.mean(tail[:, None]*jac, axis=0)/.05+.1*slopes, 1-tail.mean()/.05]
    initial = math.log(2)-.1*math.log(.05)+.1*.95/.05*(-math.log(.95))
    if abs(grad).max() > 1.1e-8 or abs(loss-model['final_objective']) > 1e-12 or abs(initial-model['initial_objective']) > 1e-12:
        raise ValueError('Independent original pAUC formula not stationary')
    if len(model['trace']) > 100 or model['slopes'] != model['weights'][1:]:
        raise ValueError('Fixed budget or slopes differ')
    return float(abs(grad).max())


def review(result, allow_absent=False):
    p, pins, absent, previous = runner.load(allow_absent)
    report = json.loads((result/'report.json').read_text())
    models = json.loads((result/'models.json').read_text())
    prepared = json.loads((result/'TRAIN_prepare.json').read_text())
    calibration = json.loads((result/'calibration.json').read_text())
    if report['input_pins'] != pins or report['models'] != models or report['prepared'] != prepared or report['calibration'] != calibration or report['base_models'] != previous['base_models'] or report['original_partial_model'] != previous['model']:
        raise ValueError('Saved artifacts differ')
    original_train, expected_prepare = runner.prepare(p, previous)
    if not equivalent_metrics(prepared, expected_prepare):
        raise ValueError('Prepared membership/normalization differs')
    active_train = [r for r in original_train if supported(r)]
    extra = np.array([[r['pixel_evidence'][a]['mismatch'] for a in ('L', 'S')] for r in active_train])
    scale = extra.std(axis=0); scale[scale < 1e-8] = 1.
    if not np.allclose(prepared['mean'][4:], extra.mean(axis=0), atol=1e-13, rtol=0) or not np.allclose(prepared['scale'][4:], scale, atol=1e-13, rtol=0):
        raise ValueError('Additional normalizer not TRAIN-derived')
    gradients = {name: fit_check(m, original_train, prepared) for name, m in models.items()}
    total = checks = 0
    for role in ('TRAIN', 'VAL'):
        saved = runner.io.read(result/('scored_'+role+'.jsonl.gz'))
        original = runner.io.read(ROOT/p['reference_result']/('scored_'+role+'.jsonl.gz'))
        pix = json.loads((ROOT/p['pixel_result']/'rows.json').read_text()) if role == 'TRAIN' else json.loads((result/'pixels_VAL.json').read_text())
        expected = core.attach(original, pix, role)
        if len(saved) != len(expected):
            raise ValueError('Full role coverage changed')
        for r, old in zip(saved, expected):
            if {k:v for k,v in r.items() if k != 'risks'} != {k:v for k,v in old.items() if k != 'risks'} or any(r['risks'][k] != v for k,v in old['risks'].items()):
                raise ValueError('Frame/box/flags/evidence/original score changed')
            for name, model in models.items():
                risk = old['risks']['partial_auc4']
                if supported(old):
                    z = (raw_vector(old, model['dim'])-model['mean'])/model['scale']
                    risk = math.exp(-float(np.logaddexp(0., -(model['weights'][0]+z.dot(model['weights'][1:])))))
                actual = r['risks'][name]
                if (risk is None) != (actual is None) or (risk is not None and abs(risk-actual) > 1e-12):
                    raise ValueError('Independent prediction/fallback identity differs')
            total += 1; checks += 1
        if role == 'VAL':
            manifest = json.loads((ROOT/p['manifest']).read_text())
            specs = {r['id']:r for r in manifest['records'] if r['split'] == 'val'}
            if any(r['image_sha256'] != specs[r['image']]['images']['sha256'] for r in pix):
                raise ValueError('Native VAL source identity differs')
            # Re-extract only during an explicitly invoked result review on the server.
            if not allow_absent:
                fresh = runner.extract_val(original, p, ROOT/'crane_project/data/crane_grab_port_day2night_v1')
                if not equivalent_metrics(pix, fresh):
                    raise ValueError('Saved VAL evidence not from native image extractor')
            for method, cal in calibration.items():
                thresholds = []
                for group, part in core.level.metrics.grouped(saved).items():
                    good = [r for r in part if bad(r) is False]
                    held = sum(r['image'] in held_ids(part) for r in good)
                    need = max(0, math.ceil(.95*len(good))-held)
                    scores = sorted(r['risks'][method] for r in good if supported(r))
                    thresholds.append(scores[need-1] if need else 0.)
                if cal['risk_le'] != max(thresholds) or cal['fallback_risk_le'] != .4094748402105834:
                    raise ValueError('Global group95 calibration differs')
        for group, part in core.level.metrics.grouped(saved).items():
            for method, cal in calibration.items():
                ids = held_ids(part) | {r['image'] for r in part if supported(r) and r['risks'][method] <= cal['risk_le']}
                entry = report['roles'][role]['groups'][group][method]
                check_summary(part, ids, entry['candidate']); checks += 1
                for name, matches in entry['controls'].items():
                    locked = name in (core.CONTROL, 'partial_auc4', 'crossfit_fusion')
                    for kind, summary in matches.items():
                        chosen = matched_ids(part, name, count=len(ids), locked=locked) if kind == 'same_count' else matched_ids(part, name, correct=entry['candidate']['states']['CR'], locked=locked)
                        if chosen is None:
                            if summary.get('feasible', True): raise ValueError('Infeasible CR incorrectly reported')
                        else: check_summary(part, chosen, summary)
                        checks += 1
                ref_ids = {r['image'] for r in part if r['pred'] is not None and r['fallback_decision']}
                check_summary(part, ref_ids, entry['frozen_fallback_reference']); checks += 1
                chosen = matched_ids(part, method, correct=entry['frozen_fallback_reference']['states']['CR'])
                if chosen is None:
                    if entry['candidate_at_reference_CR'].get('feasible', True): raise ValueError('Reference CR infeasible')
                else: check_summary(part, chosen, entry['candidate_at_reference_CR'])
                checks += 1
        if not equivalent_metrics(report['roles'][role], core.evaluate(saved, calibration)):
            raise ValueError('Metrics/continuity/matched comparison differ')
    normalized = lambda g: dict(g, failures=sorted(g['failures'], key=lambda x:json.dumps(x,sort_keys=True)))
    if not equivalent_metrics(normalized(report['gate']), normalized(core.gate(report['roles']['VAL']))):
        raise ValueError('Gate differs')
    if report['fit_calls'] != 2 or not report['model_saved_before_VAL']:
        raise ValueError('Fit count/order declaration differs')
    for k in ('TEST_opened','formal_policy_changed','candidate_adopted','detector_inferences','detector_updates','neural_training_updates'):
        if report[k]: raise ValueError('Scope exceeded')
    return dict(passed=True,rows=total,independent_checks=checks,gradients=gradients,
                absent_frontend=absent,report_sha256=runner.io.sha(result/'report.json'),
                fallback_flags_preserved=True,TEST_opened=False)


if __name__ == '__main__':
    a=argparse.ArgumentParser();a.add_argument('--result',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--allow-absent-frontend',action='store_true');args=a.parse_args()
    runner.io.write(args.out,review(args.result,args.allow_absent_frontend));print('Independent pixel pAUC review passed')
