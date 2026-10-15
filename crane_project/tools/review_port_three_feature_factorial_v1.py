"""Independent scalar forward/backprop and metric replay, without Torch."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
from crane_project.tools.review_port_simple_pilot_increment_v1 import features, wrong, risk as linear_risk

ROOT = Path(__file__).resolve().parents[2]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def readrows(p):
    with gzip.open(p, 'rt') as f: return [json.loads(s) for s in f]


def sigmoid(z):
    return 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))


def forward(model, x):
    x = [(v-m)/s for v, m, s in zip(x, model['mean'], model['scale'])]
    w = model['weights']; w1 = w['network.0.weight']; b1 = w['network.0.bias']
    w2 = w['network.2.weight'][0]; b2 = w['network.2.bias'][0]
    pre = [sum(a*b for a, b in zip(row, x))+bias for row, bias in zip(w1, b1)]
    hidden = [max(0., v) for v in pre]
    z = sum(a*b for a, b in zip(w2, hidden))+b2
    return sigmoid(z), z, x, pre, hidden


def training_proof(model, material, settings):
    material = [r for r in material if r['pred'] is not None]
    xs = [features(r) for r in material]; ys = [int(wrong(r)) for r in material]
    n = len(xs); nb = sum(ys); ng = n-nb
    mean = [sum(x[i] for x in xs)/n for i in range(3)]
    scale = [math.sqrt(sum((x[i]-mean[i])**2 for x in xs)/n) for i in range(3)]
    assert max(abs(a-b) for a, b in zip(mean, model['mean'])) < 1e-11
    assert max(abs(a-b) for a, b in zip(scale, model['scale'])) < 1e-11
    assert (n, nb, ng) == (model['train_rows'], model['bad_rows'], model['good_rows'])
    ids = ('\n'.join(r['image'] for r in material)+'\n').encode()
    assert hashlib.sha256(ids).hexdigest() == model['train_identity_sha256']
    w = model['weights']; a = w['network.0.weight']; b = w['network.2.weight'][0]
    l2 = settings['l2']; ga = [[l2*v for v in row] for row in a]
    gb = [l2*v for v in b]; gc = [0.]*8; gd = 0.
    loss = .5*l2*(sum(v*v for row in a for v in row)+sum(v*v for v in b))
    for x, y in zip(xs, ys):
        p, z, normalized, pre, hidden = forward(model, x)
        weight = .5/(nb if y else ng); delta = weight*(p-y)
        loss += weight*(max(z, 0)+math.log1p(math.exp(-abs(z)))-y*z)
        gd += delta
        for i in range(8):
            gb[i] += delta*hidden[i]
            derivative = delta*b[i] if pre[i] > 0 else 0.
            gc[i] += derivative
            for j in range(3): ga[i][j] += derivative*normalized[j]
    grad = max(map(abs, [v for row in ga for v in row]+gb+gc+[gd]))
    assert abs(loss-model['objective_final']) < 1e-10
    assert abs(grad-model['max_abs_gradient']) < 1e-10
    assert model['converged'] == (grad <= settings['stationarity_limit'])
    assert 0 < model['optimizer_iterations'] <= settings['max_iterations']
    assert loss <= model['objective_initial']+1e-12
    return dict(rows=n, bad=nb, good=ng, objective=loss, max_abs_gradient=grad,
                approximate_stationarity=model['converged'])


def verify(out):
    read = lambda name: json.loads((out/name).read_text())
    m, report, models, freeze, completion = map(read, (
        'protocol.json', 'report.json', 'models.json', 'freeze.json', 'completion.json'))
    for name, h in completion['artifacts'].items(): assert sha(out/name) == h, name
    for p, h in m['development_pins'].items(): assert sha(ROOT/p) == h, p
    for p, h in m['test_pins'].items(): assert sha(ROOT/p) == h, p
    assert sha(out/'models.json') == freeze['models_sha256']
    assert freeze['all_training_complete_before_TEST']
    saved = {}
    for role, source in (('TRAIN', 'TRAIN'), ('PILOT_TRAIN', 'PILOT'), ('VAL', 'VAL'), ('TEST', 'TEST')):
        saved[role] = readrows(out/('scored_'+role+'.jsonl.gz'))
        original = readrows(ROOT/m['roles'][source]['rows'])
        assert len(original) == len(saved[role]) == m['roles'][source]['frames']
        for a, b in zip(saved[role], original):
            assert {k: v for k, v in a.items() if k != 'audit_risks'} == b
    saved['COMBINED_TRAIN'] = saved['TRAIN']+saved['PILOT_TRAIN']
    prior = ROOT/m['linear_result']; old_models = json.loads((prior/'models.json').read_text())
    old_freeze = json.loads((prior/'freeze.json').read_text())
    for name, original in (('formal', 'formal'), ('linear_old', 'refit'), ('linear_augmented', 'augmented')):
        assert models[name] == old_models[original]
        assert freeze['cutoffs'][name] == old_freeze['cutoffs'][original]
    training = {name: training_proof(model, saved['TRAIN' if '_old_' in name else 'COMBINED_TRAIN'], m['fitting'])
        for name, model in models.items() if name.startswith('mlp_')}
    logs = [json.loads(s) for s in (out/'train_log.jsonl').read_text().splitlines()]
    for name in training:
        assert sum(v['method'] == name for v in logs) == models[name]['function_evaluations']
    cells = pairs = curves = matches = 0; delta = 0.
    for role, material in saved.items():
        parts = {'all': material}
        for field in ('domain', 'sequence'):
            for value in {r[field] for r in material}:
                parts[field+':'+value] = [r for r in material if r[field] == value]
        for name, metrics in report['results'][role].items():
            model = models[name]; cut = freeze['cutoffs'][name]
            for r in material:
                score = r['audit_risks'][name]
                if r['pred'] is None:
                    assert score is None
                else:
                    value = forward(model, features(r))[0] if name.startswith('mlp_') else linear_risk(model, features(r))
                    delta = max(delta, abs(value-score)); assert abs(value-score) < 1e-12
            for group, part in parts.items():
                d = metrics['at_frozen_VAL_cut'][group]
                count = Counter({k: 0 for k in ('FA', 'FR', 'ED', 'CR', 'MISSING')})
                run = strict = gap = longest = longest_strict = longest_gap = 0; previous = None
                good_scores, bad_scores = [], []
                for r in sorted(part, key=lambda r: (r['sequence'], r['frame_id'], r['image'])):
                    b = wrong(r); score = r['audit_risks'][name]
                    keep = b is not None and cut is not None and score <= cut
                    state = 'MISSING' if b is None else ('FA' if b else 'CR') if keep else ('ED' if b else 'FR')
                    count[state] += 1
                    same = previous is not None and previous[0] == r['sequence']
                    adjacent = same and previous[1]+1 == r['frame_id']
                    run = (run+1 if same else 1) if state == 'FR' else 0
                    strict = (strict+1 if adjacent else 1) if state == 'FR' else 0
                    gap = 0 if state == 'CR' else (gap+1 if same else 1)
                    longest = max(longest, run); longest_strict = max(longest_strict, strict); longest_gap = max(longest_gap, gap)
                    previous = r['sequence'], r['frame_id']
                    if b is not None: (bad_scores if b else good_scores).append(score)
                assert dict(count) == d['states'], (role, name, group)
                assert (longest, longest_strict, longest_gap) == (d['longest_FR_observations'], d['longest_correct_FR'], d['longest_without_CR_observations'])
                g = count['CR']+count['FR']; a = count['CR']+count['FA']; n = len(part)
                expected = dict(correct_retention=count['CR']/g if g else None,
                    accepted_error_rate=count['FA']/a if a else None, acceptance_coverage=a/n,
                    output_coverage=(n-count['MISSING'])/n, full_frame_correct_coverage=count['CR']/n)
                assert all(d[k] == v for k, v in expected.items())
                auc = (sum((e > c)+.5*(e == c) for e in bad_scores for c in good_scores)/(len(bad_scores)*len(good_scores))) if bad_scores and good_scores else None
                assert auc == metrics['group_AUROC'][group]
                pairs += len(bad_scores)*len(good_scores); cells += 1
            ordered = sorted([r for r in material if r['pred'] is not None], key=lambda r: (r['audit_risks'][name], r['image']))
            histogram = {}
            for r in ordered:
                pair = histogram.setdefault(r['audit_risks'][name], [0, 0]); pair[int(wrong(r))] += 1
            expected = [(0, 0, None)]; cr = fa = 0
            for score, pair in sorted(histogram.items()):
                cr += pair[0]; fa += pair[1]; expected.append((cr, fa, score))
            assert expected == [(p['CR'], p['FA'], p['risk_le']) for p in metrics['curve']]
            curves += len(expected)
            for key in ('same_CR_as_formal', 'same_count_as_formal', 'formal_at_candidate_count'):
                point = metrics[key]
                method = 'formal' if key == 'formal_at_candidate_count' else name
                order = sorted(ordered, key=lambda r: (r['audit_risks'][method], r['image']))
                if key == 'same_CR_as_formal':
                    goods = sorted(r['audit_risks'][method] for r in order if not wrong(r))
                    boundary = goods[point['requested_CR']-1] if point['requested_CR'] else None
                    kept = [r for r in order if boundary is not None and r['audit_risks'][method] <= boundary]
                    assert boundary == point['risk_le']
                else:
                    kept = order[:point['accepted']]
                assert sum(wrong(r) for r in kept) == point['FA']
                assert len(kept)-point['FA'] == point['CR']; matches += 1
            for point in metrics['auxiliary'].values():
                kept = [r for r in ordered if point['risk_le'] is not None and r['audit_risks'][name] <= point['risk_le']]
                assert sum(wrong(r) for r in kept) == point['FA']
                assert len(kept)-point['FA'] == point['CR']; matches += 1
    val = report['results']['VAL']; checks = []
    for seed in m['seeds']:
        old, aug = ('mlp_'+a+'_'+str(seed) for a in ('old', 'augmented'))
        for name in (old, aug):
            goods = sorted(r['audit_risks'][name] for r in saved['VAL'] if wrong(r) is False)
            assert freeze['cutoffs'][name] == goods[652]
        a = val[aug]['at_frozen_VAL_cut']; o = val[old]['at_frozen_VAL_cut']
        checks.append(dict(seed=seed, beats_old=a['all']['states']['FA'] < o['all']['states']['FA'],
            beats_formal=a['all']['states']['FA'] < val['formal']['at_frozen_VAL_cut']['all']['states']['FA'],
            protects_CR=a['all']['states']['CR'] >= 653, protects_real=a['domain:real']['states']['CR'] >= 189,
            TRAIN_stationary=models[old]['converged'] and models[aug]['converged']))
    assert freeze['decision']['checks'] == checks
    passed = all(all(v for k, v in c.items() if k != 'seed') for c in checks)
    assert report['MLP_TEST_reported'] == passed == freeze['decision']['passed']
    expected = {'formal', 'linear_old', 'linear_augmented'} | ({'mlp_old_1701', 'mlp_augmented_1701'} if passed else set())
    assert set(report['results']['TEST']) == expected
    assert all(set(r['audit_risks']) == expected for r in saved['TEST'])
    assert report['detector_updates'] == report['detector_inference'] == report['feature_extraction'] == 0
    return dict(status='PASS', scalar_risk_max_abs=delta, metric_group_checks=cells,
        AUROC_pairs=pairs, curve_points=curves, matched_auxiliary_checks=matches,
        training_gradient_proofs=training, MLP_TEST_reported=passed, linear_TEST_reported=True,
        source_identity_preserved=True, report_sha256=sha(out/'report.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True, type=Path)
    p.add_argument('--receipt', required=True); a = p.parse_args()
    result = verify(a.out.resolve())
    with (a.out/a.receipt).open('x') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))
