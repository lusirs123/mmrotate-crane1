"""Audit C using saved outputs only; stdlib, no model import/inference/training.

Fixed comparisons: 100->101 and 110->111. TEST remains descriptive.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


GROUPS = ('100', '101', '110', '111')
PAIRS = (('100', '101'), ('110', '111'))


def mean(xs):
    return statistics.mean(xs) if xs else None


def diff_angle(a, b):
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def partition(rows):
    """Never join a sequence boundary or a missing frame ID."""
    result = []
    for row in sorted(rows, key=lambda r: (r['domain'], r['seq'], r['fid'])):
        if (not result or row['seq'] != result[-1][-1]['seq'] or
                row['domain'] != result[-1][-1]['domain'] or
                row['fid'] != result[-1][-1]['fid'] + 1):
            result.append([])
        result[-1].append(row)
    return result


def events(rows):
    result = []
    for segment in partition(rows):
        run = []
        for row in segment:
            if row['riou_export'] < .5:
                run.append(row)
            elif run:
                result.append((run, True))
                run = []
        if run:
            result.append((run, False))
    return result


def aggregate(rows):
    ev = events(rows)
    maxima = [max([len(run) for run, _ in events(seg)], default=0)
              for seg in partition(rows)]
    pairs = {}
    windows = []
    for segment in partition(rows):
        hits = [r['riou_export'] >= .5 for r in segment]
        windows.extend(any(hits[i:i+10]) for i in range(max(0, len(hits)-9)))
        for a, b in zip(segment, segment[1:]):
            if a['output'] and b['output']:
                pairs[(a['frame'], b['frame'])] = (
                    100 * abs(b['diag_export']-a['diag_export'])/a['diag_export'],
                    max(0., 1-diff_angle(b['angle_export'], a['angle_export']) /
                        (math.radians(35)+1e-9)))
    return dict(frames=len(rows), outputs=sum(r['output'] for r in rows),
                center=sum(r['center_export'] < 15 for r in rows),
                hits=sum(r['riou_export'] >= .5 for r in rows),
                riou=mean([r['riou_export'] for r in rows]),
                DFR=mean([v[0] for v in pairs.values()]),
                ACI=mean([v[1] for v in pairs.values()]),
                TDR_hits=sum(windows), TDR_windows=len(windows),
                MCML_max=max(maxima), MCML_mean=mean(maxima),
                MRF=mean([len(run) for run, recovered in ev if recovered]),
                pairs=pairs, events=ev,
                gated_RMSE=math.sqrt(mean([r['gated_angle_sq'] for r in rows])))


def compare(off, on):
    a, b = aggregate(off), aggregate(on)
    am = {r['frame']: r for r in off}
    bm = {r['frame']: r for r in on}
    assert am.keys() == bm.keys()
    n = len(off)
    gains, losses = {}, {}
    for key, predicate in (
            ('hit', lambda r: r['riou_export'] >= .5),
            ('center', lambda r: r['center_export'] < 15),
            ('output', lambda r: r['output'])):
        gains[key] = [f for f in am if not predicate(am[f]) and predicate(bm[f])]
        losses[key] = [f for f in am if predicate(am[f]) and not predicate(bm[f])]
    common_pairs = sorted(a['pairs'].keys() & b['pairs'].keys())
    shared = dict(count=len(common_pairs))
    for i, metric in enumerate(('DFR', 'ACI')):
        shared[metric+'_off'] = mean([a['pairs'][k][i] for k in common_pairs])
        shared[metric+'_on'] = mean([b['pairs'][k][i] for k in common_pairs])
        shared['delta_'+metric] = shared[metric+'_on']-shared[metric+'_off']
    gate_a = {k for k, r in am.items() if r['center_export'] < 10}
    gate_b = {k for k, r in bm.items() if r['center_export'] < 10}
    both_gate = gate_a & gate_b
    common_output = [k for k in am if am[k]['output'] and bm[k]['output']]
    angle = dict(frames=n, penalty_off=n-len(gate_a), penalty_on=n-len(gate_b),
                 gate_gained=len(gate_b-gate_a), gate_lost=len(gate_a-gate_b),
                 gated_RMSE_off=a['gated_RMSE'], gated_RMSE_on=b['gated_RMSE'],
                 penalty_delta_MSE=8100*(len(gate_a)-len(gate_b))/n,
                 pass_angle_delta_MSE=(sum(bm[k]['gated_angle_sq'] for k in gate_b)-
                                       sum(am[k]['gated_angle_sq'] for k in gate_a))/n,
                 both_pass_count=len(both_gate),
                 both_pass_RMSE_off=math.sqrt(mean([am[k]['gated_angle_sq'] for k in both_gate])),
                 both_pass_RMSE_on=math.sqrt(mean([bm[k]['gated_angle_sq'] for k in both_gate])),
                 common_output_count=len(common_output),
                 common_output_native_RMSE_off=math.sqrt(mean([am[k]['angle_error_deg']**2 for k in common_output])),
                 common_output_native_RMSE_on=math.sqrt(mean([bm[k]['angle_error_deg']**2 for k in common_output])))
    delta_mse = b['gated_RMSE']**2-a['gated_RMSE']**2
    assert math.isclose(delta_mse, angle['penalty_delta_MSE']+angle['pass_angle_delta_MSE'], abs_tol=1e-9)
    for key, metric in (('hit', 'hits'), ('center', 'center'), ('output', 'outputs')):
        assert len(gains[key])-len(losses[key]) == b[metric]-a[metric]
    # Net gains and losses do not alone identify split/merge changes in runs.
    off_failure = {r['frame'] for r in off if r['riou_export'] < .5}
    on_failure = {r['frame'] for r in on if r['riou_export'] < .5}
    result = dict(frames=n, off={k:v for k,v in a.items() if k not in ('pairs','events')},
                  on={k:v for k,v in b.items() if k not in ('pairs','events')},
                  gains=gains, losses=losses, common_pairs=shared, angle=angle,
                  off_events=len(a['events']), on_events=len(b['events']),
                  fully_resolved_off_events=sum(not any(r['frame'] in on_failure for r in run) for run,_ in a['events']),
                  wholly_new_on_events=sum(not any(r['frame'] in off_failure for r in run) for run,_ in b['events']),
                  diagonal_changed_common_outputs=sum(abs(am[k]['diag_export']-bm[k]['diag_export'])>1e-6 for k in common_output))
    return result


def write_csv(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, out):
    base = root/'work_dirs/symeood_ablation_v1'
    source = base/'analysis_all8_20261024'
    all8 = json.loads((source/'analysis_all8.json').read_text())['groups']
    groups = defaultdict(dict)
    for r in csv.DictReader((source/'per_frame_all8.csv').open()):
        if r['group'] not in GROUPS:
            continue
        r = {k:(v if k in ('group','split','epoch','frame','domain','seq') else
                  v == 'True' if k in ('output','angle_eligible') else
                  float(v) if v else None) for k,v in r.items()}
        r['fid'] = int(r['fid'])
        groups[(r['group'], r['split'], r['epoch'])][r['frame']] = r
    contexts = [('selected_val','val',None), ('test','test',None)] + [
        ('val_ep'+str(e), 'val', 'epoch_'+str(e)) for e in (16,18,20,22,24)]
    out.mkdir(parents=True, exist_ok=False)
    comparisons, by_video, transitions, event_rows, angle_rows, pair_rows = [],[],[],[],[],[]
    pair_details = []
    checks = 0
    for off, on in PAIRS:
        for context, split, epoch in contexts:
            off_epoch = epoch or all8[off]['selected']
            on_epoch = epoch or all8[on]['selected']
            a = list(groups[off,split,off_epoch].values())
            b = list(groups[on,split,on_epoch].values())
            assert len(a)==len(b)==(1440 if split=='test' else 887)
            meta = dict(off_group=off,on_group=on,context=context,split=split,
                        off_epoch=off_epoch,on_epoch=on_epoch)
            for domain in ('real','sim','all'):
                av = [r for r in a if domain=='all' or r['domain']==domain]
                bv = [r for r in b if domain=='all' or r['domain']==domain]
                comp = compare(av,bv)
                # Reaggregation against already independently verified summaries.
                for g,rs,ep in ((off,comp['off'],off_epoch),(on,comp['on'],on_epoch)):
                    expected=all8[g]['test' if split=='test' else ep][domain]
                    for k, ek in (('outputs','outputs'),('center','center_correct'),
                                  ('DFR','DFR'),('ACI','ACI'),('riou','mean_RIoU'),
                                  ('MCML_max','MCML_max'),('MCML_mean','MCML_mean'),
                                  ('TDR_hits','TDR_hits'),('TDR_windows','TDR_windows')):
                        assert math.isclose(rs[k],expected[ek],abs_tol=1e-10), (g,context,domain,k)
                        checks += 1
                comparisons.append(dict(**meta,domain=domain,**comp))
                angle_rows.append(dict(**meta,domain=domain,**comp['angle']))
                pair_rows.append(dict(**meta,domain=domain,**comp['common_pairs']))
            for domain,seq in sorted({(r['domain'],r['seq']) for r in a}):
                av=[r for r in a if (r['domain'],r['seq'])==(domain,seq)]
                bv=[r for r in b if (r['domain'],r['seq'])==(domain,seq)]
                comp=compare(av,bv)
                by_video.append(dict(**meta,domain=domain,seq=seq,**comp))
            am={r['frame']:r for r in a};bm={r['frame']:r for r in b}
            ap, bp = aggregate(a)['pairs'], aggregate(b)['pairs']
            for start, end in sorted(ap.keys() & bp.keys()):
                pair_details.append(dict(**meta, domain=am[start]['domain'],
                                         seq=am[start]['seq'], start=start, end=end,
                                         off_DFR=ap[start,end][0], on_DFR=bp[start,end][0],
                                         off_ACI=ap[start,end][1], on_ACI=bp[start,end][1]))
            for frame,ra in sorted(am.items()):
                rb=bm[frame]
                transitions.append(dict(**meta,domain=ra['domain'],seq=ra['seq'],frame=frame,
                                        off_hit=ra['riou_export']>=.5,on_hit=rb['riou_export']>=.5,
                                        off_center=ra['center_export']<15,on_center=rb['center_export']<15,
                                        off_output=ra['output'],on_output=rb['output'],
                                        off_riou=ra['riou_export'],on_riou=rb['riou_export']))
            for model,rs in ((off,a),(on,b)):
                for run,recovered in events(rs):
                    event_rows.append(dict(**meta,model=model,domain=run[0]['domain'],seq=run[0]['seq'],
                                           start=run[0]['frame'],end=run[-1]['frame'],length=len(run),
                                           recovered=recovered,
                                           frames=';'.join(r['frame'] for r in run)))
    # Existing loss logs are descriptive only, not per-sample or counterfactual gradients.
    loss_logs=[]
    for g in GROUPS:
        snapshot=base/('analysis_first4_20261023/snapshot' if g=='100' else 'analysis_all8_20261024/last4_snapshot')
        directory=snapshot/'work_dirs/symeood_ablation_v1'/g
        logfiles=list(directory.glob('*.log.json'));assert len(logfiles)==1
        logs=[json.loads(line) for line in logfiles[0].read_text().splitlines()]
        train=[r for r in logs if r.get('mode')=='train']
        for epoch in range(1,25):
            rs=[r for r in train if r['epoch']==epoch];assert rs
            loss_logs.append(dict(group=g,epoch=epoch,logged_intervals=len(rs),
                                  **{key:mean([r[key] for r in rs]) for key in ('loss_cls','loss_bbox','aux0_loss_cls','aux0_loss_bbox','grad_norm')}))
    # Verify audited code is exactly the recorded runtime source for both queues.
    source_hash={}
    for relative in ('mmrotate/models/losses/sym_nfl_loss.py',
                     'mmrotate/models/dense_heads/sym_eood_head.py',
                     'mmrotate/models/losses/sym_kld_calculator.py'):
        digest=sha(root/relative)
        for batch in ('first4','last4'):
            manifest=json.loads((base/('launch_'+batch+'_20261023')/'manifest.json').read_text())
            assert manifest['sha256'][relative]==digest,relative
        source_hash[relative]=digest
    payload=dict(scope='Existing fixed outputs only; no new inference, training, checkpoint selection, or depth access',
                 comparisons=comparisons,by_video=by_video,
                 verification=dict(passed=True,reaggregated_metric_checks=checks,
                                   common_pair_matching=True,frame_transition_conservation=True,
                                   gated_MSE_decomposition=True,frozen_source_hash_match=source_hash))
    (out/'analysis.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    write_csv(out/'angle_decomposition.csv',angle_rows)
    write_csv(out/'common_pairs.csv',pair_rows)
    write_csv(out/'common_pair_details.csv',pair_details)
    write_csv(out/'frame_transitions.csv',transitions)
    write_csv(out/'failure_events.csv',event_rows)
    write_csv(out/'training_log_means.csv',loss_logs)
    (out/'provenance.json').write_text(json.dumps(dict(
        inputs={str(p.relative_to(root)):sha(p) for p in (source/'per_frame_all8.csv',source/'analysis_all8.json')},
        script_sha256=sha(Path(__file__)),
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2))
    print(json.dumps(payload['verification'],indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.root,args.out)
