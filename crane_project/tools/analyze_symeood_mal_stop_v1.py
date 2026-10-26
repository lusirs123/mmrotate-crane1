"""CPU-only audit of a stopped MAL/S_MAL pair; never loads a checkpoint.

Reads archived scalar/probe/input logs. No inference, epoch selection, TEST,
training repair or restart. Writes a new directory and preserves all evidence.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, values):
    with Path(path).open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in values for k in r}))
        writer.writeheader()
        writer.writerows(values)


def describe(values):
    import numpy as np
    if not values:
        return dict(n=0)
    a = np.asarray(values, dtype=float)
    assert np.isfinite(a).all()
    return dict(n=len(a), minimum=float(a.min()), maximum=float(a.max()),
                mean=float(a.mean()), quantiles=dict(zip(
                    ('p01', 'p25', 'p50', 'p75', 'p95', 'p99'),
                    map(float, np.quantile(a, [.01, .25, .5, .75, .95, .99])))))


def algebra_examples():
    """Counterexamples to 'U small implies MAL small', not real TRAIN replay."""
    import torch
    import torch.nn.functional as F
    from crane_project.utils.symeood_mal_math_v1 import mal_terms, match_strength
    result = []
    for dtype in (torch.float32, torch.float64):
        for logit in (2., 5., 8., 10., 20.):
            z = torch.tensor([[logit]], dtype=dtype, requires_grad=True)
            p = z.sigmoid()
            U = .25 * .25 * (1-p).pow(2).mul(
                F.binary_cross_entropy_with_logits(z, torch.ones_like(z), reduction='none')).sum()
            M = .25 * mal_terms(z, torch.tensor([0]), torch.tensor([.5], dtype=dtype),
                                torch.ones(1, dtype=dtype)).sum()
            gu = torch.autograd.grad(U, z, retain_graph=True)[0]
            gm = torch.autograd.grad(M, z, retain_graph=True)[0]
            row = dict(dtype=str(dtype), logit=logit, p=float(p.detach()),
                       q=.5, target=.5**1.5, U=float(U.detach()), M=float(M.detach()),
                       U_logit_gradient=float(gu), MAL_logit_gradient=float(gm))
            try:
                a, S, degenerate = match_strength(U, M)
                gs = torch.autograd.grad(S, z)[0]
                torch.testing.assert_close(gs, a*gu)
                row.update(guard='passed', alpha=float(a), S_logit_gradient=float(gs),
                           alpha_times_U_gradient=float(a*gu))
            except FloatingPointError as exc:
                row.update(guard=str(exc))
            result.append(row)
    return result


def analyze(base, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((base/'transfer_failure_review_20261026_v1.manifest.json').read_text())
    for item in manifest['files']:
        assert sha(base/item['path']) == item['sha256'], item['path']
    contract = json.loads((base/'protocol.json').read_text())
    sources = []
    for name, expected in contract['sources'].items():
        assert sha(ROOT/name) == expected, ('local frozen source changed', name)
        sources.append(name)
    status = json.loads((base/'status.json').read_text())
    assert status['state'] == 'stopped_on_error'
    alpha_summary = {}; epoch_rows = []; checks = collections.Counter()
    initial = {}; inputs = {}; online = []; probe_rows = []; steps = {}; clip_summary = {}
    for group in contract['groups']:
        work = base/group
        alpha_summary[group] = {}; clip_summary[group] = {}
        initial[group] = {}; inputs[group] = {}; steps[group] = {}
        for rank in (0, 1):
            identity = list(rows(work/('identity_rank%d.jsonl' % rank)))
            assert len(identity) == 1 and not identity[0]['resumed']
            initial[group][rank] = list(rows(work/('initial_state_rank%d.jsonl' % rank)))[0]['full_state_sha256']
            inputs[group][rank] = list(rows(work/('inputs_rank%d.jsonl' % rank)))
            all_stats = []; per_epoch = collections.defaultdict(list)
            for i, r in enumerate(rows(work/('alpha_rank%d.jsonl' % rank))):
                assert r['iteration'] == i and r['epoch'] == i//640+1
                for key in ('U', 'M', 'S', 'alpha', 'Upos', 'Uneg', 'Mpos', 'Mneg'):
                    assert math.isfinite(r[key]) and r[key] >= 0, (group, rank, i, key)
                assert math.isclose(r['S'], r['alpha']*r['U'], rel_tol=1e-5, abs_tol=1e-7)
                if not r['degenerate']:
                    assert math.isclose(r['S'], r['M'], rel_tol=1e-5, abs_tol=1e-7)
                else:
                    assert r['U'] <= 1e-12 and r['M'] <= 1e-12 and r['alpha'] == 1
                assert math.isclose(r['M'], r['Mpos']+r['Mneg'], rel_tol=1e-5, abs_tol=1e-7)
                assert len(r['q']) == len(r['target']) == len(r['positive_score']) == sum(r['positive'])
                assert all(0 <= q <= 1 for q in r['q'])
                for q, t in zip(r['q'], r['target']):
                    assert math.isclose(t, q**1.5, rel_tol=1e-5, abs_tol=1e-7)
                checks['successful_update_scalar_contracts'] += 1
                all_stats.append(r); per_epoch[r['epoch']].append(r)
            steps[group][rank] = len(all_stats)
            for epoch, rr in sorted(per_epoch.items()):
                q = [v for r in rr for v in r['q']]
                p = [v for r in rr for v in r['positive_score']]
                low = [v for r in rr for v in r['lowq_score']]
                mm = sum(r['M'] for r in rr)
                epoch_rows.append(dict(group=group, rank=rank, epoch=epoch, updates=len(rr),
                    complete_epoch=len(rr)==640, alpha_min=min(r['alpha'] for r in rr),
                    alpha_max=max(r['alpha'] for r in rr), alpha_median=describe([r['alpha'] for r in rr])['quantiles']['p50'],
                    U_min=min(r['U'] for r in rr), U_mean=sum(r['U'] for r in rr)/len(rr),
                    M_mean=mm/len(rr), M_negative_fraction=sum(r['Mneg'] for r in rr)/mm if mm else 0,
                    q_mean=sum(q)/len(q) if q else None,
                    positive_score_mean=sum(p)/len(p) if p else None,
                    low_q_score_mean=sum(low)/len(low) if low else None,
                    alpha_above_1e3=sum(r['alpha'] > 1e3 for r in rr),
                    alpha_above_1e6=sum(r['alpha'] > 1e6 for r in rr)))
            extreme = max(all_stats, key=lambda r:r['alpha'])
            keep = ('iteration','epoch','U','M','S','alpha','positive','q','target','positive_score')
            alpha_summary[group][str(rank)] = dict(updates=len(all_stats),
                alpha=describe([r['alpha'] for r in all_stats]),
                U=describe([r['U'] for r in all_stats]), M=describe([r['M'] for r in all_stats]),
                above_1e3=sum(r['alpha'] > 1e3 for r in all_stats),
                above_1e6=sum(r['alpha'] > 1e6 for r in all_stats),
                degenerate=sum(r['degenerate'] for r in all_stats),
                extreme={k:extreme[k] for k in keep}, last={k:all_stats[-1][k] for k in keep})
            clips=list(rows(work/('clip_rank%d.jsonl'%rank)))
            for r in clips:
                assert r['preclip_norm'] is not None and math.isfinite(r['preclip_norm'])
                assert math.isclose(r['clip_coefficient'],min(1.,10/(r['preclip_norm']+1e-6)),abs_tol=1e-12)
                checks['sampled_clip_contracts'] += 1
            clip_summary[group][str(rank)] = dict(samples=len(clips), clipped=sum(r['clip_coefficient']<1 for r in clips),
                norm=describe([r['preclip_norm'] for r in clips]),
                peak_allocated_mib=max(r['peak_allocated_mib'] for r in clips),
                peak_reserved_mib=max(r['peak_reserved_mib'] for r in clips))
            for r in rows(work/('probes_rank%d.jsonl'%rank)):
                probe_rows.append(dict(group=group,rank=rank,epoch=r['epoch'],probe=r['probe'],
                    alpha=r['alpha'],U=r['U'],M=r['M'],
                    parameters_cosine=r['gradients']['parameters']['cosine'],
                    logits_cosine=r['gradients']['logits']['cosine'],
                    local_total_preclip=r['total_gradient']['preclip_norm']))
            conservation=list(rows(work/('probe_conservation_rank%d.jsonl'%rank)))
            assert all(r['passed'] for r in conservation)
            checks['probe_conservation_receipts'] += len(conservation)
        for log in work.glob('*.log.json'):
            for r in rows(log):
                if r.get('mode')=='val':online.append(dict(group=group,**r))
    initial_values=[v for per_group in initial.values() for v in per_group.values()]
    assert len(set(initial_values))==1
    paired_epochs=[]
    for rank in (0,1):
        left={r['epoch']:r for r in inputs['MAL'][rank]}
        right={r['epoch']:r for r in inputs['S_MAL'][rank]}
        assert len(left)==24 and len(right)==10
        for epoch,r in right.items():
            assert left[epoch]['sha256']==r['sha256'] and left[epoch]['batches']==r['batches']==640
            paired_epochs.append(dict(rank=rank,epoch=epoch,sha256=r['sha256'],batches=640))
    assert all(steps['MAL'][rank]==15360 and steps['S_MAL'][rank]==7022 for rank in (0,1))
    assert {r['epoch'] for r in online if r['group']=='MAL'}==set(range(2,25,2))
    assert {r['epoch'] for r in online if r['group']=='S_MAL'}==set(range(2,11,2))
    # Failed forward is not logged by after_train_iter: don't invent its tensors.
    text=(base/'S_MAL_train.log').read_text()
    assert 'U near zero but counterfactual MAL nonzero' in text
    failure=dict(rank=0,stage='forward before backward/optimizer on failed step',
        error='U near zero but counterfactual MAL nonzero',
        guard_implication=dict(U_lte=1e-12, M_gt=1e-12),
        completed_updates_per_rank=7022,last_logged_iteration=7021,
        last_completed_epoch=10,partial_epoch=11,partial_updates=622,
        failing_U_M_logits_quality='not saved; exact values unknown',
        cannot_conclude='no claim that the exact failing U was zero or a unique cause was proven')
    examples=algebra_examples()
    assert any(r['guard']==failure['error'] for r in examples)
    paired_clips=[]
    for group in contract['groups']:
        for rank in (0,1):
            rr=[r for r in rows(base/group/('clip_rank%d.jsonl'%rank)) if r['iteration']<7022]
            paired_clips.append(dict(group=group,rank=rank,window='first7022_successful_updates',
                samples=len(rr),clipped=sum(r['clip_coefficient']<1 for r in rr),
                norm=describe([r['preclip_norm'] for r in rr])))
    write_csv(out/'per_epoch_strength.csv',epoch_rows)
    write_csv(out/'online_VAL.csv',online)
    write_csv(out/'probe_gradient_summary.csv',probe_rows)
    write_csv(out/'algebra_counterexamples.csv',examples)
    (out/'alpha_summary.json').write_text(json.dumps(alpha_summary,indent=2))
    (out/'clip_summary.json').write_text(json.dumps(clip_summary,indent=2))
    (out/'same_window_clip_summary.json').write_text(json.dumps(paired_clips,indent=2))
    (out/'failure.json').write_text(json.dumps(failure,indent=2))
    (out/'identity_verification.json').write_text(json.dumps(dict(passed=True,
        local_source_count=len(sources),source_names=sources,
        initial_full_state_sha256=initial_values[0],paired_completed_epochs=paired_epochs,
        partial_epoch11_inputs='per-epoch hashes only: unavailable for incomplete epoch',
        checks=dict(checks),manifest_files_verified=len(manifest['files'])),indent=2))
    (out/'complete.json').write_text(json.dumps(dict(
        audit_passed=True,experiment_complete=False,MAL_complete=True,S_MAL_complete=False,
        epoch_budget_consumed=24+7022/640,formal_optimizer_updates=15360+7022,
        candidate_selection_complete=False,TEST_complete=False,depth=False,
        training_or_inference_performed_by_this_audit=False),indent=2))
    print(json.dumps(dict(steps=steps,checks=dict(checks),failure=failure),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();analyze(args.base.resolve(),args.out.resolve())
