"""Independent scalar replay: source identities, objective gradient and metrics."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rows(p):
    with gzip.open(p, 'rt') as f:
        return [json.loads(s) for s in f]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def wrong(r):
    if r['pred'] is None:
        return None
    return any(abs(p-g)/g > .1 for p, g in zip(sorted(r['pred'][2:4]), sorted(r['gt'][2:4])))


def features(r):
    p = r['pred']; a, b = sorted(p[2:4], reverse=True); w, h = r['image_size']
    score = min(1-1e-6, max(1e-6, p[5]))
    return [math.log(score)-math.log1p(-score),
            .5*(math.log(a)+math.log(b)-math.log(w)-math.log(h)), math.log(a)-math.log(b)]


def risk(model, x):
    z = model['weights'][0]+sum(w*(v-m)/s for w, v, m, s in zip(
        model['weights'][1:], x, model['mean'], model['scale']))
    return 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))


def verify(out):
    read = lambda name: json.loads((out/name).read_text())
    m, report, models, freeze, completion = map(read, (
        'protocol.json', 'report.json', 'models.json', 'freeze.json', 'completion.json'))
    for p, h in completion['artifacts'].items():
        assert sha(out/p) == h, p
    for p, h in m['development_pins'].items():
        assert sha(ROOT/p) == h, p
    assert freeze['model_sha256'] == sha(out/'models.json')
    saved = {role: rows(out/('scored_'+role+'.jsonl.gz')) for role in ('TRAIN', 'PILOT_TRAIN', 'VAL')}
    for role, source_role in (('TRAIN', 'TRAIN'), ('PILOT_TRAIN', 'PILOT'), ('VAL', 'VAL')):
        original = rows(ROOT/m['roles'][source_role]['rows'])
        assert len(saved[role]) == len(original)
        for a, b in zip(saved[role], original):
            assert {k: v for k, v in a.items() if k != 'audit_risks'} == b
    if report['TEST_read']:
        assert freeze['decision']['passed']
        for p, h in m['test_pins'].items(): assert sha(ROOT/p) == h
        saved['TEST'] = rows(out/'scored_TEST.jsonl.gz')
        original = rows(ROOT/m['roles']['TEST']['rows'])
        for a, b in zip(saved['TEST'], original):
            assert {k: v for k, v in a.items() if k != 'audit_risks'} == b
    else:
        assert not (out/'scored_TEST.jsonl.gz').exists()
        assert not freeze['decision']['passed'] and 'TEST' not in report['results']
    saved['COMBINED_TRAIN'] = saved['TRAIN']+saved['PILOT_TRAIN']
    fit_proof = {}
    for name, material in (('refit', saved['TRAIN']), ('augmented', saved['COMBINED_TRAIN'])):
        model = models[name]; material = [r for r in material if r['pred'] is not None]
        xs = [features(r) for r in material]; labels = [int(wrong(r)) for r in material]
        n = len(xs); nb = sum(labels); ng = n-nb
        means = [sum(x[i] for x in xs)/n for i in range(3)]
        scales = [math.sqrt(sum((x[i]-means[i])**2 for x in xs)/n) for i in range(3)]
        assert max(abs(a-b) for a, b in zip(means, model['mean'])) < 1e-11
        assert max(abs(a-b) for a, b in zip(scales, model['scale'])) < 1e-11
        assert (model['train_rows'], model['bad_rows'], model['good_rows']) == (n, nb, ng)
        gradient = [0.]+[m['fitting']['l2']*w for w in model['weights'][1:]]
        objective = .5*m['fitting']['l2']*sum(w*w for w in model['weights'][1:])
        for x, y in zip(xs, labels):
            a = [1.]+[(v-mu)/s for v, mu, s in zip(x, model['mean'], model['scale'])]
            z = sum(w*v for w, v in zip(model['weights'], a)); p = risk(model, x)
            weight = .5/(nb if y else ng)
            objective += weight*(max(z, 0)+math.log1p(math.exp(-abs(z)))-y*z)
            for i in range(4): gradient[i] += weight*(p-y)*a[i]
        assert max(map(abs, gradient)) <= m['fitting']['gradient_tolerance']+1e-12
        assert abs(objective-model['final_objective']) < 1e-11
        fit_proof[name] = dict(rows=n, bad=nb, good=ng, max_abs_gradient=max(map(abs, gradient)), objective=objective)
    replay = 0.; cells = pairs = curve_points = match_checks = 0
    for role, material in saved.items():
        parts = {'all': material}
        for field in ('domain', 'sequence'):
            for value in {r[field] for r in material}:
                parts[field+':'+value] = [r for r in material if r[field] == value]
        for method, model in models.items():
            cut = freeze['cutoffs'][method]
            for r in material:
                if r['pred'] is None:
                    assert r['audit_risks'][method] is None
                else:
                    delta = abs(risk(model, features(r))-r['audit_risks'][method])
                    replay = max(replay, delta); assert delta < 1e-12
            metrics = report['results'][role][method]
            for group, part in parts.items():
                d = metrics['at_frozen_VAL_cut'][group]
                count = Counter({k: 0 for k in ('FA', 'FR', 'ED', 'CR', 'MISSING')})
                run = strict = gap = longest = longest_strict = longest_gap = 0; previous = None
                good_scores, bad_scores = [], []
                for r in sorted(part, key=lambda r: (r['sequence'], r['frame_id'], r['image'])):
                    b = wrong(r); score = r['audit_risks'][method]
                    keep = b is not None and cut is not None and score <= cut
                    state = 'MISSING' if b is None else ('FA' if b else 'CR') if keep else ('ED' if b else 'FR')
                    count[state] += 1
                    same_video = previous is not None and previous[0] == r['sequence']
                    adjacent = same_video and previous[1]+1 == r['frame_id']
                    run = (run+1 if same_video else 1) if state == 'FR' else 0
                    strict = (strict+1 if adjacent else 1) if state == 'FR' else 0
                    gap = 0 if state == 'CR' else (gap+1 if same_video else 1)
                    longest = max(longest, run); longest_strict = max(longest_strict, strict); longest_gap = max(longest_gap, gap)
                    previous = r['sequence'], r['frame_id']
                    if b is not None: (bad_scores if b else good_scores).append(score)
                assert dict(count) == d['states'], (role, method, group)
                assert (longest, longest_strict, longest_gap) == (d['longest_FR_observations'], d['longest_correct_FR'], d['longest_without_CR_observations'])
                g = count['CR']+count['FR']; a = count['CR']+count['FA']; n = len(part)
                expected = dict(correct_retention=count['CR']/g if g else None,
                    accepted_error_rate=count['FA']/a if a else None, acceptance_coverage=a/n,
                    output_coverage=(n-count['MISSING'])/n, full_frame_correct_coverage=count['CR']/n)
                assert all(d[k] == v for k, v in expected.items())
                auc = (sum((e > c)+.5*(e == c) for e in bad_scores for c in good_scores)/(len(bad_scores)*len(good_scores))) if bad_scores and good_scores else None
                assert auc == metrics['group_AUROC'][group]
                pairs += len(bad_scores)*len(good_scores); cells += 1
            ordered = sorted([r for r in material if r['pred'] is not None], key=lambda r: (r['audit_risks'][method], r['image']))
            histogram = {}
            for r in ordered:
                pair = histogram.setdefault(r['audit_risks'][method], [0, 0]); pair[int(wrong(r))] += 1
            expected = [(0, 0, None)]; cr = fa = 0
            for score, pair in sorted(histogram.items()):
                cr += pair[0]; fa += pair[1]; expected.append((cr, fa, score))
            assert expected == [(p['CR'], p['FA'], p['risk_le']) for p in metrics['curve']]
            curve_points += len(expected)
            for key in ('same_CR_as_formal', 'same_count_as_formal'):
                p = metrics[key]
                if key.startswith('same_CR'):
                    goods = sorted(r['audit_risks'][method] for r in ordered if not wrong(r))
                    threshold = goods[p['requested_CR']-1] if p['requested_CR'] else None
                    kept = [r for r in ordered if threshold is not None and r['audit_risks'][method] <= threshold]
                    assert p['risk_le'] == threshold
                else:
                    kept = ordered[:p['accepted']]
                assert sum(wrong(r) for r in kept) == p['FA']
                assert len(kept)-p['FA'] == p['CR']; match_checks += 1
    val = report['results']['VAL']
    for method in ('refit', 'augmented'):
        good = sorted(r['audit_risks'][method] for r in saved['VAL'] if wrong(r) is False)
        assert freeze['cutoffs'][method] == good[652]
    a = val['augmented']['at_frozen_VAL_cut']; f = val['formal']['at_frozen_VAL_cut']
    checks = dict(protects_CR=a['all']['states']['CR'] >= 653,
        reduces_FA=a['all']['states']['FA'] < f['all']['states']['FA'],
        protects_real=a['domain:real']['states']['CR'] >= 189)
    assert freeze['decision']['checks'] == checks
    assert report['TEST_read'] == all(checks.values())
    assert report['detector_inference'] == report['detector_updates'] == report['feature_extraction'] == 0
    return dict(status='PASS', scalar_risk_max_abs=replay, group_state_continuity_checks=cells,
        AUROC_pairs=pairs, curve_points=curve_points, matched_checks=match_checks,
        fitting_stationarity=fit_proof, TEST_read=report['TEST_read'], source_identity_preserved=True,
        report_sha256=sha(out/'report.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True, type=Path)
    p.add_argument('--receipt', required=True)
    a = p.parse_args(); result = verify(a.out.resolve())
    with (a.out/a.receipt).open('x') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))
