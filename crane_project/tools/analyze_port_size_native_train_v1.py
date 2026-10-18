#!/usr/bin/env python3
"""Revise S030 coordinate-only evidence, or analyze saved S031 epochs locally.

Never overwrites original receipts; never runs a detector or fits a model.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csv_write(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def revision(source,out):
    import torch
    from crane_project.tools.audit_port_size_native_p3_v1 import linear_evidence
    from crane_project.utils import port_geometry_refine_g_v1 as g
    from crane_project.utils import port_size_native_p3_v1 as n
    torch.set_num_threads(1);r=read(source);values=[]
    for v in r['views']:
        x=dict(image=v['image'],scale=v['scale'],missing=v.get('missing',False))
        if not x['missing']:
            b,bm,xy,m=[torch.tensor(v[k],dtype=torch.float32) for k in ('b','bm','actual_sxsy','m')]
            _,cb,xy,sides,rot=n.old.residual.original.frame(b,bm,xy)
            points=n.old.border_points(m,bm,n.old.bucket_grid(m)).reshape(1,-1,2)*xy[:,None]
            local=torch.bmm(points-cb[:,None,:2],rot)/sides[:,None]
            pts=points[(local.abs()<=.5).all(-1)]
            _,old=g.sampling_grid(bm,(128,128),'aligned')
            linear=linear_evidence(old,pts,128,128,v['img_hw'])
            if linear['witness']!=v['linear_basis']['witness']:raise ValueError('Positive/negative conclusion unexpectedly changed')
            x.update(linear,original_old_null_response=v['linear_basis']['old_null_response'],original_native_response=v['linear_basis']['native_response'])
        values.append(x)
    out.mkdir(parents=True,exist_ok=False)
    report=dict(source=str(source),source_sha256=sha(source),revision='Noise residual<=1e-9 is not normalized; invalid witness responses null. SVD threshold is absolute1e-9. Original receipts unchanged.',
        views=values,positive_count=sum(v.get('witness',False) for v in values),
        corrected_invalid_response_count=sum(not v.get('missing') and not v.get('valid_witness_constructed') for v in values),
        old_nominal_latency_bool_is_formal_pass=False,latency_status='UNCONFIRMED_S030_SHARED_LOAD',
        scene_identity_count=6,view_count=12,independent_scale_samples=False,model_calls=0,training_updates=0)
    write(out/'revision.json',report);csv_write(out/'revised_evidence.csv',values)
    write(out/'artifacts.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print('Revision:',report['positive_count'],'positives unchanged;',report['corrected_invalid_response_count'],'invalid witness responses set null; no model calls')


def results(source,out):
    report=read(source/'training.json')
    if report['status']!='COMPLETE' or report['formal_head_updates']!=30672:raise ValueError('Not a completed paired run')
    baseline=read(source/'baseline_val.summary.json');rows=[];comparisons={}
    for arm,v in report['arms'].items():
        if len(v['epochs'])!=24:raise ValueError('Missing full VAL epoch')
        for ep in v['epochs']:
            for group,cell in ep['metrics'].items():
                method=cell['edge_residual'];sizes=method['sizes'];component=ep['components'][group]['edge_residual']
                x=dict(arm=arm,epoch=ep['epoch'],group=group,eligible=int(ep['eligibility']['pass_all']),
                    riou=method['mean_riou_full_frame'],outputs=sizes['outputs'],frames=sizes['frames'],missing=sizes['missing'],
                    long_mae_pct=100*sizes['edges']['long']['mae'],short_mae_pct=100*sizes['edges']['short']['mae'],
                    long_p95_pct=100*sizes['edges']['long']['p95'],short_p95_pct=100*sizes['edges']['short']['p95'],
                    long_bias_pct=100*sizes['edges']['long']['bias'],short_bias_pct=100*sizes['edges']['short']['bias'],
                    meanabs_a=sizes['a']['abs_mae'],meanabs_r=sizes['r']['abs_mae'],size_correct=sizes['joint10'],
                    size_correct_rate=sizes['joint10']/sizes['outputs'],wrong_to_right=cell['paired']['wrong_to_right'],right_to_wrong=cell['paired']['right_to_wrong'],
                    center_correct=component['center_correct'],angle_correct=component['angle_correct'],angle_eligible=component['angle_eligible_outputs'],
                    ap50=method['cache_gt_ap11']['0.5'],ap75=method['cache_gt_ap11']['0.75'])
                for d,t in method['temporal'].items():
                    for k in ('dfr_percent_per_frame','aci','mcml_max','mcml_mean'):x[d+'_'+k]=t[k]
                    x[d+'_tdr_hits']=t['tdr_w10']['numerator'];x[d+'_tdr_windows']=t['tdr_w10']['denominator']
                if group in ep['diagnostics']:
                    for e in ('long','short'):
                        near=ep['diagnostics'][group]['near'][e]
                        x['near_'+e+'_count']=near['count'];x['near_'+e+'_mae_pct']=100*near['after_mae'] if near['after_mae'] is not None else None;x['near_'+e+'_harmed']=near['harmed']
                rows.append(x)
        chosen=v['selected'] if v['selected'] is not None else v['representative']
        ep=next(e for e in v['epochs'] if e['epoch']==chosen)
        comparisons[arm]=dict(chosen_epoch=chosen,selected=v['selected'],representative=v['representative'],eligible_epochs=v['eligible_epochs'],
            eligibility=ep['eligibility'],metrics=ep['metrics'],components=ep['components'],diagnostics=ep['diagnostics'])
    a,c=[comparisons[k] for k in ('cached_p3','native_p3')];mechanism={}
    for e in ('long','short'):
        mechanism['sim/'+e]=c['metrics']['sim']['edge_residual']['sizes']['edges'][e]['mae']<a['metrics']['sim']['edge_residual']['sizes']['edges'][e]['mae']-1e-6
        mechanism['real/'+e]=c['metrics']['real']['edge_residual']['sizes']['edges'][e]['mae']<=a['metrics']['real']['edge_residual']['sizes']['edges'][e]['mae']+.001
    mechanism['overall/riou']=c['metrics']['overall']['edge_residual']['mean_riou_full_frame']>=a['metrics']['overall']['edge_residual']['mean_riou_full_frame']-1e-5
    out.mkdir(parents=True,exist_ok=False)
    csv_write(out/'all_epochs_val.csv',rows)
    write(out/'analysis.json',dict(protocol=report['protocol'],training_sha256=sha(source/'training.json'),formal_updates=30672,
        arms=comparisons,baseline=baseline,mechanism=dict(checks=mechanism,pass_all=all(mechanism.values())),
        efficiency=report['efficiency'],limits='No TEST/depth; new0.1pp tolerance is not S025 passing. RIoU epsilon exceedance alone is not practical gain.',
        recommendation='Keep formal M; no automatic replacement. Interpret selected/representative and mechanism separately; stop fixed-budget hypothesis if no usable benefit.'))
    write(out/'artifacts.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(arms={k:{f:v[f] for f in ('selected','representative','eligible_epochs')} for k,v in report['arms'].items()},mechanism=mechanism),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('revision','results'));p.add_argument('source',type=Path);p.add_argument('out',type=Path);args=p.parse_args()
    (revision if args.stage=='revision' else results)(args.source,args.out)
