"""Independently verify downloaded frozen-diagnostic metadata (stdlib only).

Raw arrays remain on the server and are checked by the separate CPU analyzer.
This verifier reconstructs published aggregates from frame/batch JSONL records.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics


def main(root):
    checks = 0

    def eq(a, b):
        nonlocal checks
        assert a == b, (a, b)
        checks += 1

    def close(a, b):
        nonlocal checks
        assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), (a, b)
        checks += 1

    def jl(path):
        return [json.loads(s) for s in path.read_text().splitlines()]

    analysis = root / 'analysis_v1'
    provenance = json.loads((analysis / 'provenance.json').read_text())
    for name, digest in provenance['outputs'].items():
        eq(hashlib.sha256((analysis / name).read_bytes()).hexdigest(), digest)
    for name in ('protocol', 'draft'):
        eq(hashlib.sha256((root / (name + '.json')).read_bytes()).hexdigest(), provenance[name + '_sha256'])
    complete = json.loads((root / 'complete.json').read_text())
    eq(complete['passed'], True)
    eq(complete['updates'], 0)
    draft = json.loads((root / 'draft.json').read_text())
    data = json.loads((analysis / 'analysis.json').read_text())
    state = jl(root / 'state_checks.jsonl')
    state += [json.loads(f.read_text()) for f in root.glob('reader_*.json')]
    unique = {(r['group'], r['phase']): r for r in state}
    eq(len(unique), 12)
    for r in unique.values():
        eq(r['before'], r['after'])
        eq(r['passed'], True)

    candidate_total = gradient_total = 0
    for g in ('100', '101', '110', '111'):
        for split, n in (('val', 887), ('train', 2558)):
            frames = jl(root / g / ('candidates_' + split + '_summary.jsonl'))
            eq(len(frames), n)
            eq(len({r['frame'] for r in frames}), n)
            for r in frames:
                category = ('no_qualified' if not r['qualified'] else
                            'score_filter_lost' if not r['qualified_passed'] else
                            'success' if r['selected'] >= 0 and r['selected_iou'] >= .5 else 'ranking_lost')
                eq(r['category'], category)
            candidate_total += n
            for row in [x for x in data['candidates'] if x['group'] == g and x['split'] == split]:
                sub = [r for r in frames if row['domain'] == 'all' or r['frame'].startswith(row['domain'] + '_')]
                counts = collections.Counter(r['category'] for r in sub)
                eq(row['frames'], len(sub))
                eq(row['outputs'], sum(r['selected'] >= 0 for r in sub))
                eq(row['qualified_available'], sum(r['qualified'] > 0 for r in sub))
                eq(row['center_correct'], sum(r['selected'] >= 0 and r['selected_center'] < 15 for r in sub))
                for k in ('no_qualified', 'score_filter_lost', 'ranking_lost', 'success'):
                    eq(row[k], counts[k])
        batches = jl(root / g / 'gradient_summary.jsonl')
        eq(len(batches), 128)
        gradient_total += len(batches)
        for r, view in zip(batches, draft['training_views']):
            eq(r['batch'], view['batch'])
            eq(r['frames'], [s['frame'] for s in view['samples']])
            eq(r['strength_degenerate'], False)
            # S is evaluated independently in float32, not copied from G.
            # Use the frozen CPU analyzer's arithmetic tolerance here.
            assert math.isclose(r['S'], r['G'], rel_tol=1e-6, abs_tol=1e-8)
            checks += 1
        for row in [r for r in data['gradients'] if r['group'] == g]:
            space = row['space']
            for key, values in (
                ('alpha', [r['alpha'] for r in batches]),
                ('cosine', [r['gradient'][space]['G_vs_S']['cosine'] for r in batches]),
                ('ratio', [r['gradient'][space]['G_vs_S']['ratio'] for r in batches]),
            ):
                close(row[key + '_mean'], statistics.mean(values))
                close(row[key + '_min'], min(values))
                if key != 'cosine':
                    close(row[key + '_max'], max(values))
            for objective in ('U', 'G', 'S'):
                close(row[objective + '_preclip_mean'], statistics.mean(r['total'][objective]['preclip_norm'] for r in batches))
                eq(row[objective + '_clipped'], sum(r['total'][objective]['clip_coefficient'] < 1 for r in batches))
        eq(data['empty_gt_calls'][g], sum(x['empty_gt'] for r in batches for x in r['levels']))
        for kind in ('positive', 'negative', 'ignored'):
            mu = next(r for r in data['mu'] if r['group'] == g and r['domain'] == 'all' and r['kind'] == kind)
            for field, source in (('candidates', kind), ('topk', kind + '_topk'),
                                  ('loss_U', kind + '_loss_U'), ('loss_G', kind + '_loss_G')):
                # Layer sums are float32 while the raw analyzer sums float64 elements.
                expected = sum(x[source] for r in batches for x in r['levels'])
                assert math.isclose(mu[field], expected, rel_tol=1e-5, abs_tol=1e-7), (g, kind, field)
                checks += 1
            for k in ('candidates', 'topk', 'loss_U', 'loss_G', 'loss_increment'):
                close(mu[k], sum(r[k] for r in data['mu'] if r['group'] == g and r['domain'] in ('real', 'sim') and r['kind'] == kind))
            close(sum(mu['kld_reference_' + str(i)] for i in range(5)), mu['topk'])
            close(sum(mu[k] for k in ('mu_near1', 'mu_middle', 'mu_near2')), mu['topk'])
        for row in [r for r in data['label_gradients'] if r['group'] == g]:
            for obj in ('U', 'G', 'S'):
                close(row[obj + '_L2_mean'], statistics.mean(r['label_partition'][row['kind']][obj] for r in batches))
                energy = sum(r['label_partition'][row['kind']][obj] ** 2 for r in batches)
                total = sum(r['label_partition'][k][obj] ** 2 for r in batches for k in ('positive', 'negative', 'ignored'))
                close(row[obj + '_energy_share_pct'], 100 * energy / total)
    eq(candidate_total, complete['candidate_frames'])
    eq(gradient_total, complete['gradient_batches'])
    receipt = dict(passed=True, checks=checks, candidate_frames=candidate_total,
                   gradient_batches=gradient_total, updates=0,
                   scope='Downloaded metadata only; full raw NPZ checked separately on server CPU',
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   provenance_sha256=hashlib.sha256((analysis / 'provenance.json').read_bytes()).hexdigest())
    out = root / 'local_metadata_verification.json'
    with out.open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
