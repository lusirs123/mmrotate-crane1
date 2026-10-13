#!/usr/bin/env python3
"""Independent causal replay, convex-fit stationarity, calibration and state review."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.review_port_reliability_temporal_history_gate_v1 import check_summary as check_counts_runs, rank_ids, matched_veto_ids, part_for, bad, state

VERSION = 'port_reliability_temporal_fusion_v1'


def check_summary(rows, ids, summary):
    check_counts_runs(rows, ids, summary)
    c = summary['states']; n = len(rows); outputs = n-c['MISSING']; good = c['CR']+c['FR']
    expected = dict(frames=n, outputs=outputs, output_coverage=outputs/n,
        acceptance_coverage_all_frames=len(ids)/n, correct_coverage_all_frames=c['CR']/n,
        good_retention=c['CR']/good if good else None,
        accepted_error_fraction=c['FA']/len(ids) if ids else None)
    for k, value in expected.items():
        if summary[k] != value: raise ValueError('Coverage/risk scalar differs '+k)
    pairs = switches = 0
    for a, b in zip(rows, rows[1:]):
        if a['sequence'] == b['sequence'] and a['frame_id']+1 == b['frame_id'] and not b['temporal']['acquisition_boundary']:
            pairs += 1; switches += int((a['image'] in ids) != (b['image'] in ids))
    if summary['adjacent_observed_pairs'] != pairs or summary['flag_switches'] != switches:
        raise ValueError('Independent switching counts differ')


def read(path):
    with gzip.open(path, 'rt') as f: return [json.loads(line) for line in f]


def vector(r, dim):
    p = r['pred']; long, short = sorted(p[2:4], reverse=True)
    score = min(max(p[5], 1e-6), 1-1e-6)
    v = [math.log(score/(1-score)), .5*math.log(long*short/(r['image_size'][0]*r['image_size'][1])), math.log(long/short)]
    if dim == 4: v.append(r['temporal']['primary'])
    return v


def predict(model, r):
    dim = len(model['mean'])
    x = (np.array(vector(r, dim))-model['mean'])/model['scale']
    z = model['weights'][0]+x.dot(model['weights'][1:])
    return 1/(1+math.exp(-float(z)))


def check_fit(model, rows):
    ids = [r['image'] for r in rows]
    if ids != model['train_images']: raise ValueError('Training identities differ')
    x = np.array([vector(r, len(model['mean'])) for r in rows]); y = np.array([float(bad(r)) for r in rows])
    mean = x.mean(axis=0); scale = x.std(axis=0); scale[scale < 1e-8] = 1.
    if not np.allclose(mean, model['mean'], atol=1e-13, rtol=0) or not np.allclose(scale, model['scale'], atol=1e-13, rtol=0):
        raise ValueError('TRAIN standardization differs')
    design = np.column_stack((np.ones(len(x)), (x-mean)/scale)); w = np.array(model['weights'])
    sw = np.where(y == 1, .5/y.sum(), .5/(len(y)-y.sum()))
    z = design.dot(w); prob = 1/(1+np.exp(-z)); penalty = w.copy(); penalty[0] = 0
    grad = design.T.dot(sw*(prob-y))+.1*penalty
    objective = float(sw.dot(np.logaddexp(0,z)-y*z)+.05*penalty.dot(penalty))
    if np.max(abs(grad)) > 1.1e-8 or abs(objective-model['final_objective']) > 1e-12:
        raise ValueError('Model not stationary for prescribed convex TRAIN objective')
    if int(y.sum()) != model['bad_rows'] or len(y) != model['train_rows']: raise ValueError('Label counts differ')
    return float(np.max(abs(grad)))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--result', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--allow-absent-frontend', action='store_true')
    a = p.parse_args(); report = json.loads((a.result/'report.json').read_text())
    protocol = json.loads((ROOT/('crane_project/tools/'+VERSION+'_protocol.json')).read_text())
    if report['models'] != json.loads((a.result/'models.json').read_text()) or report['cutoffs'] != json.loads((a.result/'calibration.json').read_text()):
        raise ValueError('Saved model/calibration artifact differs')
    absent = []
    for name, digest in dict(report['input_pins'], **report['sources']['sources']).items():
        path = ROOT/name
        if a.allow_absent_frontend and not path.exists() and name in protocol['frozen_frontend_identity']:
            if digest != protocol['frozen_frontend_identity'][name]: raise ValueError('Frontend pin differs')
            absent.append(name); continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest: raise ValueError('Input/source changed '+name)
    raw = {r['image']:r for r in read(ROOT/protocol['rows'])}; scored = {}; total = checks = 0; maxdiff = 0.
    for role in ('TRAIN', 'VAL'):
        rows = read(a.result/('scored_'+role+'.jsonl.gz')); scored[role] = rows
        past = []; prev = None
        for r in rows:
            original = raw[r['image']]
            for key in ('pred','gt','image_size','original_simple_decision','split','reliability_role','frame_id','sequence','domain','size_risks'):
                if r[key] != original[key]: raise ValueError('Original evidence mutated')
            boundary = (prev is None or r['sequence'] != prev['sequence'] or r['frame_id'] != prev['frame_id']+1
                or r['image_size'] != prev['image_size'] or r['frame_id'] in protocol['boundaries'].get(r['sequence'], []))
            if boundary or r['pred'] is None: past = []
            available = r['pred'] is not None and len(past) >= 3
            if r['temporal']['available'] != available: raise ValueError('Causal availability differs')
            if r['temporal']['acquisition_boundary'] != (r['frame_id'] in protocol['boundaries'].get(r['sequence'], [])):
                raise ValueError('Known boundary differs')
            if available:
                edge = [math.log(x) for x in sorted(r['pred'][2:4], reverse=True)]
                d = max(abs(edge[i]-median(h[1][i] for h in past[-3:])) for i in (0,1))
                diff = abs(d-r['temporal']['primary']); maxdiff = max(maxdiff, diff)
                if diff > 1e-12 or r['temporal']['history_frames'] != [h[0] for h in past[-3:]]: raise ValueError('History replay differs')
            for arm in ('static3', 'fusion4'):
                expected = None if r['pred'] is None else predict(report['models'][arm if available else 'fallback3'], r)
                actual = r['risks'][arm]
                if expected is None:
                    if actual is not None: raise ValueError('Missing output has risk')
                elif abs(expected-actual) > 1e-12: raise ValueError('Independent learned score differs')
            if r['pred'] is not None: past.append((r['frame_id'], [math.log(x) for x in sorted(r['pred'][2:4], reverse=True)]))
            prev = r; total += 1
    train = [r for r in scored['TRAIN'] if r['pred'] is not None]
    supported = [r for r in train if r['temporal']['available']]
    gradients = {name:check_fit(report['models'][name], train if name == 'fallback3' else supported)
                 for name in ('fallback3', 'static3', 'fusion4')}
    if report['models']['static3']['train_images'] != report['models']['fusion4']['train_images']: raise ValueError('Unequal arms')
    for arm, calibration in report['cutoffs'].items():
        thresholds = []
        for group, entry in calibration['group_protection'].items():
            good = sorted(r['risks'][arm] for r in part_for(scored['VAL'], group) if bad(r) is False)
            required = math.ceil(.95*len(good)); threshold = good[required-1] if good else None
            if entry != dict(good_outputs=len(good), required=required, cutoff=threshold): raise ValueError('VAL calibration differs')
            if threshold is not None: thresholds.append(threshold)
        if max(thresholds) != calibration['risk_le']: raise ValueError('Global group-protection cutoff differs')
    for role, rows in scored.items():
        out = [r for r in rows if r['pred'] is not None]
        hits = sum(math.dist(r['pred'][:2],r['gt'][:2]) < 15 for r in out)
        center = dict(frames=len(rows),outputs=len(out),hits=hits,hit_rate_on_outputs=hits/len(out),
            output_coverage=len(out)/len(rows),correct_coverage_all_frames=hits/len(rows))
        if center != report['roles'][role]['center']: raise ValueError('Center coverage differs')
        checks += 1
        for group, data in report['roles'][role]['groups'].items():
            part = part_for(rows, group)
            for name, method in [('candidate','fusion4'), ('static3','static3')]:
                ids = {r['image'] for r in part if r['pred'] is not None and r['risks'][method] <= report['cutoffs'][method]['risk_le']}
                check_summary(part, ids, data[name]); checks += 1
            formal = {r['image'] for r in part if r['original_simple_decision']['size_accepted']}
            old = {r['image'] for r in part if r['image'] in formal and (not r['temporal']['available'] or r['temporal']['primary'] <= .0785511443740936)}
            check_summary(part, formal, data['formal']); check_summary(part, old, data['old_level']); checks += 2
            for method, matches in data['controls'].items():
                for kind, target in [('same_count', data['candidate']['accepted_outputs']), ('same_CR', data['candidate']['states']['CR'])]:
                    ids = rank_ids(part, method, count=target) if kind == 'same_count' else rank_ids(part, method, correct=target)
                    check_summary(part, ids, matches[kind]); checks += 1
                    if kind == 'same_count':
                        out = sorted((r for r in part if r['pred'] is not None), key=lambda r:(r['risks'][method],r['image']))
                        t = out[target-1]['risks'][method] if target else None
                        before = [r for r in out if t is not None and r['risks'][method] < t]
                        tied = [r for r in out if t is not None and r['risks'][method] == t]
                        slots = target-len(before); wrong = sum(bool(bad(r)) for r in before)
                        b = matches[kind]['tie_bounds']
                        if b['FA_min'] != wrong+max(0,slots-sum(not bad(r) for r in tied)) or b['FA_max'] != wrong+min(slots,sum(bool(bad(r)) for r in tied)):
                            raise ValueError('Tie bounds differ')
            shared = min(data['candidate']['states']['CR'], data['formal']['states']['CR'])
            ids = rank_ids(part, 'fusion4', correct=shared)
            check_summary(part, ids, data['candidate_at_old_shared_CR']); checks += 1
            for kind, target, key in [('CR', shared, 'old_same_CR'), ('count', data['candidate']['accepted_outputs'], 'old_same_count')]:
                ids = matched_veto_ids(part, target, kind, key='temporal')
                if ids is None:
                    if data[key]['feasible']: raise ValueError('Old-rule matching feasibility differs')
                else: check_summary(part, ids, data[key])
                checks += 1
            for method, value in data['AUROC'].items():
                good = [r['risks'][method] for r in part if bad(r) is False]
                wrong = [r['risks'][method] for r in part if bad(r) is True]
                auc = sum((x>y)+.5*(x==y) for x in wrong for y in good)/(len(good)*len(wrong)) if good and wrong else None
                if auc is not None and abs(auc-value) > 1e-12: raise ValueError('Independent pairwise AUROC differs')
                checks += 1
    if total != 3445 or set(raw) != {r['image'] for rows in scored.values() for r in rows} or report['TEST_opened'] or report['neural_training_updates'] or report['detector_inferences']:
        raise ValueError('Scope differs')
    from crane_project.utils import port_reliability_temporal_fusion_v1 as core
    if core.gate(report['roles']['VAL']) != report['gate']: raise ValueError('Verified gate differs')
    result = dict(passed=True, rows=total, independent_checks=checks, fit_gradient_max=gradients,
                  max_temporal_replay_difference=maxdiff, absent_frontend_files=absent,
                  report_sha256=hashlib.sha256((a.result/'report.json').read_bytes()).hexdigest(), TEST_opened=False)
    with a.out.open('x') as f: json.dump(result,f,indent=2,sort_keys=True); f.write('\n')
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__': main()
