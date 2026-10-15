"""Independent NumPy objectives/gradients, source pairing and scalar metrics."""
import argparse
import gzip
import json
import math
from pathlib import Path
import numpy as np
from crane_project.tools import run_port_reliability_response_tail_v1 as runner
from crane_project.tools.review_port_reliability_response_stats_v1 import scalar_statistics
from crane_project.utils import port_reliability_response_tail_v1 as core
from crane_project.utils import port_reliability_response_stats_v1 as response
from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_simple_pilot_increment_v1 as base
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import (
    Ranking, bad, groups, independent_check, identity)


def independent_model(model, raw, labels=None):
    n = model['normalizer']; x = (raw-n['mean'])/n['scale']
    w = model['weights']; a = np.asarray(w['network.0.weight']); b = np.asarray(w['network.0.bias'])
    c = np.asarray(w['network.2.weight']); d = w['network.2.bias'][0]
    pre = np.einsum('ij,kj->ik', x, a)+b; h = np.maximum(pre, 0)
    z = np.einsum('ij,j->i', h, c[0])+d
    risk = simple.sigmoid(z); proof = {}
    if labels is not None:
        np.testing.assert_allclose(raw.mean(0), n['mean'], atol=1e-13, rtol=0)
        scale = raw.std(0); scale[scale < 1e-8] = 1
        np.testing.assert_allclose(scale, n['scale'], atol=1e-13, rtol=0)
        good, wrong = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
        dz = np.zeros(len(z))
        if model['objective_kind'] == 'bce':
            data = .5*(np.logaddexp(0, z[good]).mean()+np.logaddexp(0, -z[wrong]).mean())
            dz[good] = .5*risk[good]/len(good); dz[wrong] = .5*(risk[wrong]-1)/len(wrong)
        else:
            delta = z[good, None]-z[wrong][None, :]
            losses = np.logaddexp(0, delta).mean(1)
            k = math.ceil(core.SETTINGS['tail_fraction']*len(good))
            # Use recorded active set to replay valid top-k subgradients at ties.
            selected = np.asarray(model.get('final_tail_indices_within_goods', np.argsort(losses)[-k:]), dtype=int)
            if len(set(selected)) != k or selected.min() < 0 or selected.max() >= len(good) or losses[selected].min() < np.sort(losses)[-k]-1e-12:
                raise ValueError('Saved tail active set is not top-k')
            data = losses[selected].mean()
            derivatives = simple.sigmoid(delta[selected])/(k*len(wrong))
            dz[good[selected]] = derivatives.sum(1); dz[wrong] = -derivatives.sum(0)
            proof['tail_correct_count'] = k
            proof['tail_boundary_margin'] = float(np.sort(losses)[-k]-np.sort(losses)[-k-1]) if k < len(good) else None
        dh = dz[:, None]*c*(pre > 0)
        gradients = [dh.T@x+core.SETTINGS['l2']*a, dh.sum(0),
                     dz[None]@h+core.SETTINGS['l2']*c, np.array([dz.sum()])]
        gradient = max(float(np.abs(g).max()) for g in gradients)
        penalty = .5*core.SETTINGS['l2']*((a*a).sum()+(c*c).sum())
        if abs(data+penalty-model['objective_final']) > 1e-10 or abs(gradient-model['max_abs_gradient']) > 1e-9:
            raise ValueError('Independent objective/gradient mismatch')
        proof.update(data_loss=float(data), penalty=float(penalty), max_abs_gradient=gradient,
                     objective=float(data+penalty))
    return risk, proof


def review(out, receipt):
    complete = json.loads((out/'completion.json').read_text())
    for name, expected in complete['artifacts'].items():
        if runner.sha(out/name) != expected:
            raise ValueError('Changed frozen artifact '+name)
    report = json.loads((out/'report.json').read_text()); protocol = json.loads((out/'protocol.json').read_text())
    models = json.loads((out/'models.json').read_text()); freeze = json.loads((out/'freeze.json').read_text())
    train_frozen = json.loads((out/'TRAIN_frozen.json').read_text())
    if (runner.sha(out/'models.json') != freeze['models_sha256']
            or freeze['models_sha256'] != train_frozen['models_sha256']
            or report['cutoffs'] != freeze['cutoffs']
            or report['decision'] != freeze['decision']
            or not train_frozen['all_training_complete_before_VAL']):
        raise ValueError('Frozen stage mismatch')
    cuts = report['cutoffs']; checks = []; risk_error = stat_error = descriptor_error = 0.; objectives = {}
    for role, results in report['results'].items():
        with gzip.open(out/('scored_'+role+'.jsonl.gz'), 'rt') as f:
            rows = [json.loads(s) for s in f]
        with gzip.open(runner.ROOT/protocol['roles'][role]['rows'], 'rt') as f:
            original = {r['image']: r for r in map(json.loads, f)}
        if len(rows) != len(original) or {r['image'] for r in rows} != set(original):
            raise ValueError('Frame membership mismatch')
        for row in rows:
            if identity(row) != identity(original[row['image']]):
                raise ValueError('Original output/label/flags changed')
            for method, decision in row['response_decisions'].items():
                if decision != response.size_decision(row['original_simple_decision'], row['audit_risks'][method], cuts[method]):
                    raise ValueError('Size flag protection failure')
        with np.load(out/(role.lower()+'_responses.npz'), allow_pickle=False) as cache:
            ids, raw = cache['images'].tolist(), cache['features']
            p, q = cache['probability'], cache['prior']
        outputs = {r['image']: r for r in rows if r['pred'] is not None}
        if len(ids) != len(outputs) or set(ids) != set(outputs):
            raise ValueError('Response membership mismatch')
        if role in protocol['saved_responses'] and runner.sha(out/(role.lower()+'_responses.npz')) != protocol['saved_responses'][role]['sha256']:
            raise ValueError('Saved response bytes differ')
        for i, name in enumerate(ids):
            r = outputs[name]
            descriptor_error = max(descriptor_error, float(np.max(np.abs(raw[i, :3]-simple.descriptor(r['pred'], r['image_size'])))))
            expected = np.asarray([a-b for pm, qm in zip(p[i], q[i])
                for a, b in zip(scalar_statistics(pm), scalar_statistics(qm))])
            stat_error = max(stat_error, float(np.max(np.abs(expected-raw[i, 3:]))))
        if stat_error > 1e-12 or descriptor_error > 1e-12:
            raise ValueError('Scalar features differ')
        for method, model in models.items():
            if method not in results:
                continue
            labels = np.asarray([bad(outputs[i]) for i in ids], dtype=float) if role == 'TRAIN' else None
            values, proof = independent_model(model, raw[:, :model['dimensions']], labels)
            if proof:
                objectives[method] = proof
            risk_error = max(risk_error, max(abs(v-outputs[i]['audit_risks'][method]) for i, v in zip(ids, values)))
        if risk_error > 1e-11:
            raise ValueError('Independent scoring differs')
        if runner.metrics(rows, tuple(results), cuts) != results:
            raise ValueError('Diagnostics differ')
        for method, value in results.items():
            for group, part in groups(rows).items():
                checks.append(independent_check(Ranking(part, method), value['at_frozen_VAL_cut'][group], cuts[method]))
        if role == 'VAL':
            if core.gate(rows, cuts) != report['decision']:
                raise ValueError('Performance gate differs')
            # Independently sort correct risks for the two protection quantiles.
            for method in results:
                if method == 'formal':
                    continue
                good = sorted(r['audit_risks'][method] for r in rows if bad(r) is False)
                real = sorted(r['audit_risks'][method] for r in rows if bad(r) is False and r['domain'] == 'real')
                if cuts[method] != max(good[652], real[188]):
                    raise ValueError('Unified protected cutoff differs')
    if not report['TEST_read'] and (out/'scored_TEST.jsonl.gz').exists():
        raise ValueError('Unexpected TEST run')
    for seed in core.SEEDS:
        if models['A_%d' % seed]['initial_state_sha256'] != models['C_%d' % seed]['initial_state_sha256'] or models['B_%d' % seed]['initial_state_sha256'] != models['D_%d' % seed]['initial_state_sha256']:
            raise ValueError('Paired initialization mismatch')
    proof = dict(status='INDEPENDENT_REVIEW_PASS', groups=len(checks),
        AUROC_pair_comparisons=sum(c['pair_comparisons'] for c in checks),
        curve_points=sum(c['curve_points'] for c in checks), model_risk_max_abs=risk_error,
        scalar_statistics_max_abs=stat_error, descriptor_max_abs=descriptor_error,
        train_objective_gradients=objectives, report_sha256=runner.sha(out/'report.json'),
        review_source_sha256=runner.sha(Path(__file__)))
    runner.write(out/receipt, proof); print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--receipt', default='independent_review.json'); args = parser.parse_args()
    review(args.out.resolve(), args.receipt)
