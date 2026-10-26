"""Independent stdlib CPU aggregation of transferred paired-run receipts.

Uses exported per-frame values; does not independently recompute polygon IoU.
No model, CUDA, inference, selection change, or training imports.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(xs):return sum(xs)/len(xs) if xs else None
def percentile(xs,q):
    xs=sorted(xs);position=(len(xs)-1)*q;lo=math.floor(position);hi=math.ceil(position)
    return xs[lo]*(hi-position)+xs[hi]*(position-lo) if hi!=lo else xs[lo]


def aggregate(rr):
    groups=collections.defaultdict(list)
    for r in rr:groups[r['domain'],r['seq']].append(r)
    maxima=[];events=[];pairs={};windows=[]
    for rows in groups.values():
        segments=[]
        for r in sorted(rows,key=lambda x:x['fid']):
            if not segments or r['fid']!=segments[-1][-1]['fid']+1:segments.append([])
            segments[-1].append(r)
        for seg in segments:
            failure=0;maximum=0
            for r in seg:
                if r['riou_export']<.5:failure+=1;maximum=max(maximum,failure)
                elif failure:events.append(failure);failure=0
            if failure:events.append(failure)
            maxima.append(maximum)
            windows.extend(any(r['riou_export']>=.5 for r in seg[i:i+10]) for i in range(max(0,len(seg)-9)))
            for a,b in zip(seg,seg[1:]):
                if a['output'] and b['output']:
                    da=abs((b['angle_export']-a['angle_export']+math.pi/2)%math.pi-math.pi/2)
                    pairs[a['frame'],b['frame']]=(100*abs(b['diag_export']-a['diag_export'])/a['diag_export'],max(0.,1-da/(math.radians(35)+1e-9)))
    valid=[r for r in rr if r['output']];eligible=[r for r in valid if r['angle_eligible']]
    m=dict(frames=len(rr),outputs=len(valid),missing=len(rr)-len(valid),coverage_pct=100*len(valid)/len(rr),
        center=sum(r['center_export']<15 for r in rr),center_correct=sum(r['center_error']<15 for r in valid),
        localization_hits=sum(r['riou_export']>=.5 for r in rr),failure_events=len(events),
        mean_RIoU=mean([r['riou_export'] for r in rr]),mean_RIoU_native=mean([r['riou_native'] for r in rr]),
        MCML_max=max(maxima),MCML_mean=mean(maxima),MCML_segments=len(maxima),
        DFR=mean([v[0] for v in pairs.values()]),ACI=mean([v[1] for v in pairs.values()]),DFR_pairs=len(pairs),
        TDR_hits=sum(windows),TDR_windows=len(windows),TDR_pct=100*mean(windows) if windows else None,
        gated_angle_RMSE_deg=math.sqrt(mean([r['gated_angle_sq'] for r in rr])),
        size_correct=sum(max(r['long_error'],r['short_error'])<=.1 for r in valid),angle_eligible=len(eligible),
        angle_correct=sum(r['angle_error_deg']<=3 for r in eligible),
        meanabs_a=mean([abs(r['a']) for r in valid]),meanabs_r=mean([abs(r['r']) for r in valid]))
    for side in ('long','short'):
        vals=[r[side+'_error'] for r in valid]
        m.update({side+'_MAE_pct':100*mean(vals),side+'_P95_pct':100*percentile(vals,.95),
                  side+'_bias_pct':100*mean([r[side+'_bias'] for r in valid])})
    return m,pairs


def verify(base,out=None):
    manifest=json.loads((base/'transfer_manifest.json').read_text())
    assert all(sha(base/p)==h for p,h in manifest.items())
    parsed=[]
    strings={'group','split','epoch','frame','domain','seq'}
    for row in csv.DictReader((base/'analysis_v2/per_frame.csv').open()):
        r={k:(v if k in strings else v=='True' if k in ('output','angle_eligible') else float(v) if v else None) for k,v in row.items()}
        r['fid']=int(r['fid']);parsed.append(r)
    assert len(parsed)==11750
    buckets=collections.defaultdict(list)
    for r in parsed:buckets[r['group'],r['split'],r['epoch']].append(r)
    assert len(buckets)==12
    checks=[]
    for row in csv.DictReader((base/'analysis_v2/all_candidate_VAL_and_selected_TEST.csv').open()):
        rr=[r for r in buckets[row['group'],row['split'],row['epoch']] if row['domain']=='all' or r['domain']==row['domain']]
        assert len(rr)==len({r['frame'] for r in rr})
        got,_=aggregate(rr)
        for k,v in got.items():
            delta=abs(v-float(row[k]));assert delta<1e-10,(row['group'],row['split'],row['epoch'],row['domain'],k,delta)
            checks.append(delta)
    comparisons=json.loads((base/'analysis_v2/paired_comparisons.json').read_text());pairchecks=0
    chosen=json.loads((base/'analysis_v2/complete.json').read_text())['selected']
    for c in comparisons:
        context=c['context'];split='test' if context=='selected_TEST' else 'val'
        same=None if context.startswith('selected_') else 'epoch_'+context.removeprefix('VAL_ep')
        sides=[]
        for g in ('110','111'):
            rr=[r for r in buckets[g,split,same or chosen[g]] if (c['domain']=='all' or r['domain']==c['domain']) and ('seq' not in c or r['seq']==c['seq'])]
            sides.append(rr)
        (am,ap),(bm,bp)=(aggregate(r) for r in sides);common=ap.keys()&bp.keys()
        assert len(common)==c['common_pairs']['count']
        for index,name in enumerate(('DFR','ACI')):
            av,bv=(mean([p[k][index] for k in common]) for p in (ap,bp))
            for key,val in ((name+'_off',av),(name+'_on',bv),('delta_'+name,bv-av)):
                assert abs(val-c['common_pairs'][key])<1e-10;pairchecks+=1
        for metric,predicate in (('hit',lambda r:r['riou_export']>=.5),('center',lambda r:r['center_export']<15),('output',lambda r:r['output'])):
            a,b=({r['frame']:predicate(r) for r in rs} for rs in sides)
            gains={k for k in a if not a[k] and b[k]};losses={k for k in a if a[k] and not b[k]}
            assert gains==set(c['gains'][metric]) and losses==set(c['losses'][metric]);pairchecks+=2
        angle=c['angle'];delta=bm['gated_angle_RMSE_deg']**2-am['gated_angle_RMSE_deg']**2
        assert abs(delta-angle['penalty_delta_MSE']-angle['pass_angle_delta_MSE'])<1e-9;pairchecks+=1
    audit=json.loads((base/'completion_audit.json').read_text());assert audit['passed'] and audit['all_paired_inputs_identical']
    result=dict(passed=True,receipt_files=len(manifest),per_frame_records=len(parsed),aggregate_metric_checks=len(checks),
        max_aggregate_abs_delta=max(checks),paired_transition_common_pair_and_angle_checks=pairchecks,
        selected=chosen,formal_runs=2,additional_training=False,additional_inference=False,GPU=False,
        limits='Independent aggregation from transferred per-frame geometry; polygon intersection was not independently recomputed locally.',
        source_sha256=sha(Path(__file__)))
    with (out or base/'local_completion_verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path)
    a=p.parse_args();verify(a.base.resolve(),a.out.resolve() if a.out else None)
