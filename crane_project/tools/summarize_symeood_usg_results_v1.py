"""Post-run explanatory tables; does not select weights or run models."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def records(path):return list(csv.DictReader(path.open()))
def avg(xs):return statistics.mean(xs) if xs else None
def delta_angle(a,b):return abs((a-b+math.pi/2)%math.pi-math.pi/2)


def temporal_pairs(rows):
    ordered=sorted(rows,key=lambda r:(r['domain'],r['seq'],r['fid']));pairs={}
    for a,b in zip(ordered,ordered[1:]):
        if (a['domain'],a['seq'])!=(b['domain'],b['seq']) or b['fid']!=a['fid']+1:continue
        if a['output'] and b['output']:
            pairs[a['frame'],b['frame']]=(100*abs(b['diag_export']-a['diag_export'])/a['diag_export'],
                max(0.,1-delta_angle(a['angle_export'],b['angle_export'])/(math.radians(35)+1e-9)))
    return pairs


def run(base,out):
    out.mkdir(exist_ok=False)
    strings={'run','epoch','frame','domain','seq'};flags={'output','angle_eligible'}
    frames=collections.defaultdict(list)
    for rec in records(base/'analysis_v1/per_frame.csv'):
        r={k:(v if k in strings else v=='True' if k in flags else float(v) if v else None) for k,v in rec.items()}
        r['fid']=int(r['fid']);frames[r['run'],r['epoch']].append(r)
    pairs=records(base/'analysis_v1/paired_metrics.csv');detail=[];checks=0
    for meta in pairs:
        off='s0_'+meta['regression']+'_'+meta['off'].lower();on='s0_'+meta['regression']+'_'+meta['on'].lower()
        a=[r for r in frames[off,meta['off_epoch']] if meta['domain']=='all' or r['domain']==meta['domain']]
        b=[r for r in frames[on,meta['on_epoch']] if meta['domain']=='all' or r['domain']==meta['domain']]
        am={r['frame']:r for r in a};bm={r['frame']:r for r in b};assert am.keys()==bm.keys()
        ap,bp=temporal_pairs(a),temporal_pairs(b);common=ap.keys()&bp.keys()
        assert len(common)==int(meta['common_pairs']);checks+=1
        for i,k in enumerate(('DFR','ACI')):
            value=avg([bp[p][i]-ap[p][i] for p in common])
            assert math.isclose(value,float(meta['common_delta_'+k]),rel_tol=1e-10,abs_tol=1e-10)
            checks+=1
        both=[f for f in am if am[f]['center_export']<10 and bm[f]['center_export']<10]
        qa=sum(r['center_export']>=10 for r in a);qb=sum(r['center_export']>=10 for r in b)
        row=dict(meta,off_penalty_frames=qa,on_penalty_frames=qb,common_pass_frames=len(both),
                 common_pass_RMSE_off=math.sqrt(avg([am[f]['gated_angle_sq'] for f in both])) if both else None,
                 common_pass_RMSE_on=math.sqrt(avg([bm[f]['gated_angle_sq'] for f in both])) if both else None)
        for name,predicate in [('center',lambda r:r['center_export']<15),('hit',lambda r:r['riou_export']>=.5),('output',lambda r:r['output'])]:
            gained=[f for f in am if not predicate(am[f]) and predicate(bm[f])]
            lost=[f for f in am if predicate(am[f]) and not predicate(bm[f])]
            assert len(gained)-len(lost)==int(meta['delta_'+('hits' if name=='hit' else 'outputs' if name=='output' else name)])
            row[name+'_gained']=gained;row[name+'_lost']=lost;checks+=1
        mse=avg([r['gated_angle_sq'] for r in b])-avg([r['gated_angle_sq'] for r in a])
        assert math.isclose(mse,float(meta['angle_penalty_delta_MSE'])+float(meta['angle_pass_delta_MSE']),rel_tol=1e-10,abs_tol=1e-10)
        checks+=1;detail.append(row)
    probes=json.loads((base/'analysis_v1/probes.json').read_text());probe_rows=[]
    for name in sorted({r['run'] for r in probes}):
        for ep in (1,12,24):
            # Both ranks read the same four fixed batches. Use rank0 to avoid
            # presenting those rank repetitions as additional unique inputs.
            rs=[r for r in probes if r['run']==name and r['epoch']==ep and r['rank']==0]
            assert len(rs)==4 and len({r['input'] for r in rs})==4
            assert all(not r['degenerate'] and math.isclose(r['S'],r['G'],rel_tol=1e-6,abs_tol=1e-8) for r in rs)
            pos=sum(x['positive_increment'] for r in rs for x in r['levels'])
            neg=sum(x['negative_increment'] for r in rs for x in r['levels'])
            row=dict(run=name,epoch=ep,rank=0,unique_batches=4,alpha_mean=avg([r['alpha'] for r in rs]),
                     alpha_min=min(r['alpha'] for r in rs),alpha_max=max(r['alpha'] for r in rs),
                     logits_GS_cosine_mean=avg([r['gradient']['logits']['G_vs_S']['cosine'] for r in rs]),
                     FPN_GS_cosine_mean=avg([r['gradient']['FPN_parameters']['G_vs_S']['cosine'] for r in rs]),
                     FPN_GS_cosine_min=min(r['gradient']['FPN_parameters']['G_vs_S']['cosine'] for r in rs),
                     negative_increment_pct=100*neg/(pos+neg),
                     **{k+'_clipped':sum(r['total'][k]['clip_coefficient']<1 for r in rs) for k in ('U','G','S')})
            probe_rows.append(row);checks+=1
    with (out/'paired_details.json').open('x') as f:json.dump(detail,f,indent=2,allow_nan=False)
    with (out/'probes_rank0_summary.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(probe_rows[0]));w.writeheader();w.writerows(probe_rows)
    curve=[]
    for reg in ('iou','kld'):
        for off,on in [('U','S'),('S','G'),('U','G')]:
            for dom in ('real','sim','all'):
                rs=[r for r in pairs if r['regression']==reg and r['off']==off and r['on']==on and r['domain']==dom and r['context']!='selected']
                assert len(rs)==5
                row=dict(regression=reg,off=off,on=on,domain=dom,epochs=5)
                for key in ('center','hits','outputs','riou','MCML_max','failure_events','DFR','ACI','gated_RMSE'):
                    vals=[float(r['delta_'+key]) for r in rs];sign=1 if key in ('center','hits','outputs','riou','ACI') else -1
                    row[key+'_improved']=sum(v*sign>0 for v in vals)
                    row[key+'_equal']=sum(v==0 for v in vals)
                curve.append(row)
    with (out/'curve_counts.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
    receipt=dict(passed=True,paired_contexts=len(detail),probe_summaries=len(probe_rows),checks=checks,
        scope='same-epoch sensitivity and fixed TRAIN probes, single seed; no re-selection or causal root-cause claim',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.base.resolve(),a.out.resolve())
