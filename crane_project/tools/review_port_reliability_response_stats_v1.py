"""Independent scalar statistics/model gradients plus complete saved-score review."""
import argparse
import gzip
import json
import math
from pathlib import Path
import numpy as np
from crane_project.tools.run_port_reliability_response_stats_v1 import ROOT, sha, write
from crane_project.utils import port_reliability_response_stats_v1 as core
from crane_project.utils import port_simple_pilot_increment_v1 as base
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import (
    Ranking, bad, groups, independent_check)


def scalar_statistics(p):
    values = [float(v) for v in np.asarray(p).reshape(-1)]
    total = sum(values); values = [v/total for v in values]
    mx = sum(v*((i % 9)/8.-.5) for i, v in enumerate(values))
    my = sum(v*((i//9)/8.-.5) for i, v in enumerate(values))
    return [sum(-v*math.log(v) for v in values if v)/math.log(81.), max(values),
            sum(v*(((i % 9)/8.-.5-mx)**2+((i//9)/8.-.5-my)**2) for i, v in enumerate(values)),
            sum(v for i, v in enumerate(values) if i//9 in (0, 8) or i % 9 in (0, 8))]


def independent_model(model, raw, labels=None):
    normalizer = model['normalizer']; x = (raw-normalizer['mean'])/normalizer['scale']
    w = model['weights']; a = np.asarray(w['network.0.weight']); b = np.asarray(w['network.0.bias'])
    c = np.asarray(w['network.2.weight']); d = float(w['network.2.bias'][0])
    pre = np.einsum('ij,kj->ik', x, a)+b; h = np.maximum(pre, 0)
    logits = np.einsum('ij,j->i', h, c[0])+d
    risks = np.array([1/(1+math.exp(-v)) if v >= 0 else math.exp(v)/(1+math.exp(v)) for v in logits])
    proof = {}
    if labels is not None:
        np.testing.assert_allclose(raw.mean(0), normalizer['mean'], rtol=0, atol=1e-13)
        scale = raw.std(0); scale[scale < 1e-8] = 1
        np.testing.assert_allclose(scale, normalizer['scale'], rtol=0, atol=1e-13)
        y = np.asarray(labels); weights = np.where(y, .5/y.sum(), .5/(len(y)-y.sum()))
        dz = (risks-y)*weights
        dh = dz[:, None]*c*(pre > 0)
        grads = [dh.T@x+core.SETTINGS['l2']*a, dh.sum(0),
                 dz[None]@h+core.SETTINGS['l2']*c, np.array([dz.sum()])]
        gradient = max(float(np.abs(g).max()) for g in grads)
        data = float(np.sum(weights*(np.maximum(logits, 0)-y*logits+np.log1p(np.exp(-np.abs(logits))))))
        penalty = .5*core.SETTINGS['l2']*float((a*a).sum()+(c*c).sum())
        if abs(gradient-model['max_abs_gradient']) > 1e-10 or abs(data+penalty-model['objective_final']) > 1e-10:
            raise ValueError('Independent TRAIN objective/gradient mismatch')
        proof = dict(data_BCE=data, L2_penalty=penalty, max_abs_gradient=gradient)
    return risks, proof


def review(out, receipt):
    complete = json.loads((out/'completion.json').read_text())
    for name, expected in complete['artifacts'].items():
        if sha(out/name) != expected:
            raise ValueError('Changed artifact '+name)
    report = json.loads((out/'report.json').read_text())
    manifest = json.loads((out/'protocol.json').read_text())
    freeze = json.loads((out/'freeze.json').read_text())
    models = json.loads((out/'models.json').read_text()); cuts = report['cutoffs']
    if sha(out/'models.json') != freeze['models_sha256'] or cuts != freeze['cutoffs']:
        raise ValueError('Freeze mismatch')
    checks = []; statistic_error = 0.; model_error = 0.; train_gradients = {}
    for role, results in report['results'].items():
        with gzip.open(out/('scored_'+role+'.jsonl.gz'), 'rt') as f:
            rows = [json.loads(s) for s in f]
        with gzip.open(ROOT/manifest['roles'][role]['rows'], 'rt') as f:
            source = {r['image']: r for r in map(json.loads, f)}
        if len(rows) != len(source) or {r['image'] for r in rows} != set(source):
            raise ValueError('Source membership mismatch')
        fields = ('pred', 'gt', 'image_size', 'original_simple_decision', 'sequence', 'domain', 'split', 'frame_id')
        for row in rows:
            if any(row[k] != source[row['image']][k] for k in fields):
                raise ValueError('Original frame/box/label/flags changed')
            for name, decision in row['response_decisions'].items():
                expected = core.size_decision(row['original_simple_decision'], row['audit_risks'][name], cuts[name])
                if decision != expected:
                    raise ValueError('Candidate size/other flags mismatch')
        cache = np.load(out/(role.lower()+'_responses.npz'))
        ids, raw = list(cache['images']), cache['features']
        outputs = {r['image']: r for r in rows if r['pred'] is not None}
        if set(ids) != set(outputs) or len(ids) != len(outputs):
            raise ValueError('Response completeness mismatch')
        for i, name in enumerate(ids):
            np.testing.assert_array_equal(raw[i, :3], simple.descriptor(outputs[name]['pred'], outputs[name]['image_size']))
            p, q = cache['probability'][i], cache['prior'][i]
            expected = np.array([a-b for pm, qm in zip(p, q)
                for a, b in zip(scalar_statistics(pm), scalar_statistics(qm))])
            statistic_error = max(statistic_error, float(np.max(np.abs(expected-raw[i, 3:]))))
        if statistic_error > 1e-12:
            raise ValueError('Independent scalar response statistics mismatch')
        for name, model in models.items():
            if name not in results:
                continue
            labels = [bad(outputs[i]) for i in ids] if role == 'TRAIN' else None
            values, proof = independent_model(model, raw[:, :model['dimensions']], labels)
            if proof:
                train_gradients[name] = proof
            model_error = max(model_error, max(abs(v-outputs[i]['audit_risks'][name]) for i, v in zip(ids, values)))
        if model_error > 1e-11:
            raise ValueError('Independent model risk mismatch')
        for method, value in results.items():
            recalculated = base.diagnostics(rows, method, cuts)
            if recalculated != value:
                raise ValueError('Saved diagnostics mismatch '+role+'/'+method)
            for group, part in groups(rows).items():
                checks.append(independent_check(Ranking(part, method), value['at_frozen_VAL_cut'][group], cuts[method]))
    if not report['TEST_read'] and (out/'scored_TEST.jsonl.gz').exists():
        raise ValueError('Unexpected TEST output')
    proof = dict(status='INDEPENDENT_REVIEW_PASS', groups=len(checks),
        AUROC_pair_comparisons=sum(c['pair_comparisons'] for c in checks),
        curve_points=sum(c['curve_points'] for c in checks),
        scalar_statistics_max_abs=statistic_error, model_risk_max_abs=model_error,
        train_objective_gradients=train_gradients, report_sha256=sha(out/'report.json'))
    write(out/receipt, proof); print(json.dumps(proof, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--receipt', default='independent_review.json'); args = parser.parse_args()
    review(args.out.resolve(), args.receipt)
