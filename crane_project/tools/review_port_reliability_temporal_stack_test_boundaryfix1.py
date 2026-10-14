#!/usr/bin/env python3
"""Independent boundary-safe audit of the unchanged frozen TEST report.

At <=1e-10 threshold distance use the pinned delivered risk for exact tie
semantics, after independent risk verification. Far-boundary flags must agree.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import eval_port_reliability_temporal_stack_test_v1 as runner
from crane_project.tools import run_port_reliability_conditional_evidence_v1 as io


def sigmoid(x):return 1/(1+math.exp(-float(x)))

def linear(m,x):return sigmoid(m['weights'][0]+sum(w*(v-u)/s for w,v,u,s in zip(m['weights'][1:],x,m['mean'],m['scale'])))


def verify(out,allow_absent=False):
    p,pins,absent,paths,b=runner.checked(allow_absent)
    report=json.loads((out/'report.json').read_text());checks=0
    if report['input_pins']!=pins or report['frozen_cutoffs']!=b['cutoffs'] or json.loads((out/'frozen_bundle.json').read_text())!=b:
        raise ValueError('Frozen bundle/report changed')
    if any(report[k] for k in ('fits','calibration_calls','detector_inferences','threshold_updates','formal_policy_changed','TEST_used_for_selection','automatic_promotion')):
        raise ValueError('Scope violation')
    with np.load(paths['native_test']/'test_native_projected.npz',allow_pickle=False) as a:
        vectors=dict(zip(a['images'].tolist(),np.asarray(a['features'])))
    total=0
    for role,n in [('VAL',887),('TEST',1440)]:
        rows=io.read(out/('scored_'+role+'.jsonl.gz'));total+=len(rows)
        if len(rows)!=n or len({r['image'] for r in rows})!=n or any(r['split']!=role.lower() for r in rows):raise ValueError('Role/count mismatch')
        history=[];prev=None
        for r in rows:
            pred=r['pred'];pos=(r['sequence'],r['frame_id'],r['image_size'])
            if prev is None or pos[0]!=prev[0] or pos[1]!=prev[1]+1 or pos[2]!=prev[2]:history=[]
            available=pred is not None and len(history)==3
            if r['temporal']['available']!=available:raise ValueError('Causal history support differs')
            if pred is None:
                history=[]
                if any(r['flags'].values()) or any(v is not None for v in r['risks'].values()):raise ValueError('Missing output classified')
                prev=pos;continue
            long,short=sorted(pred[2:4],reverse=True);logs=[math.log(long),math.log(short)]
            d=[math.log(pred[5]/(1-pred[5])) if pred[5]<1 else math.log((1-1e-6)/1e-6),
               math.log(math.sqrt(long*short)/math.sqrt(r['image_size'][0]*r['image_size'][1])),math.log(long/short)]
            # Descriptor score clipping follows the frozen simple contract.
            score_clip=min(max(pred[5],1e-6),1-1e-6);d[0]=math.log(score_clip/(1-score_clip))
            primary=max(abs(logs[i]-float(np.median([v[i] for v in history]))) for i in (0,1)) if available else None
            if available and abs(primary-r['temporal']['primary'])>1e-12:raise ValueError('History median differs')
            f4=linear(b['models']['fusion4'],d+[primary]) if available else linear(b['models']['fallback3'],d)
            rp=linear(b['pauc'],d+[primary]) if available else linear(b['models']['fallback3'],d)
            ri=r['image_crossfit_risk']
            if role=='TEST':
                head=b['head'];v=vectors[r['image']];h=((v-np.array(head['normalizer']['mean']))/np.array(head['normalizer']['scale']))[1:]
                for i in (0,2,4):
                    h=np.array(head['model']['network.%d.weight'%i])@h+np.array(head['model']['network.%d.bias'%i])
                    if i!=4:h=np.maximum(h,0)
                independent=sigmoid(-v[0]+h[0]+head['model']['beta'][0])
                if abs(ri-independent)>1e-10:raise ValueError('Independent image head differs')
            rf=rp
            if available:
                def logit(v):return min(12,max(-12,math.log(min(max(v,1e-12),1-1e-12)/(1-min(max(v,1e-12),1-1e-12)))))
                f=b['fusion'];vals=[logit(rp),logit(ri)];z=[(v-u)/s for v,u,s in zip(vals,f['normalizer']['mean'],f['normalizer']['scale'])]
                rf=sigmoid(vals[0]+f['weights'][0]+sum(w*v for w,v in zip(f['weights'][1:],z)))
            for name,value in [('fusion4',f4),('partial_auc4',rp),('crossfit_fusion',rf)]:
                if abs(value-r['risks'][name])>1e-10:raise ValueError('Independent linear/fusion risk differs: '+name)
                reference=r['risks'][name] if abs(value-b['cutoffs'][name])<=1e-10 else value
                if r['flags'][name]!=(reference<=b['cutoffs'][name]):raise ValueError('Frozen flag differs')
            branch=b['cutoffs']['crossfit_fusion'] if available else b['cutoffs']['partial_auc4']
            ref=r['risks']['crossfit_fusion'] if abs(rf-branch)<=1e-10 else rf
            if abs(r['risks']['fallback_margin']-rf/branch)>1e-10 or r['flags']['fallback_margin']!=(ref<=branch):raise ValueError('Fallback identity differs')
            if r['flags']['temporal_veto']!=(r['flags']['simple'] and (not available or primary<=b['cutoffs']['temporal_veto'])):raise ValueError('Veto differs')
            if r['original_simple_decision']['final_box_original']!=pred:raise ValueError('M changed')
            history=(history+[logs])[-3:];prev=pos;checks+=1
        for g,part in runner.core.base.level.metrics.grouped(rows).items():
            saved=report['roles'][role]['groups'][g]
            for m in runner.core.METHODS:
                counts={k:0 for k in ('FA','FR','ED','CR','MISSING')}
                for r in part:
                    if r['pred'] is None:state='MISSING'
                    else:
                        pe,ge=sorted(r['pred'][2:4]),sorted(r['gt'][2:4]);bad=max(abs(x-y)/y for x,y in zip(pe,ge))>.1
                        state=('FA' if bad else 'CR') if r['flags'][m] else ('ED' if bad else 'FR')
                    counts[state]+=1
                if counts!=saved['actual'][m]['states']:raise ValueError('Independent confusion differs')
                for control,pair in saved['comparisons'][m].items():
                    outputs=sorted([r for r in part if r['pred'] is not None],key=lambda r:(r['risks'][control],r['image']))
                    k=counts['FA']+counts['CR'];ids={r['image'] for r in outputs[:k]}
                    if runner.core.summary(part,ids)['states']!=pair['same_count']['states']:raise ValueError('Matched count differs')
                    target=counts['CR'];seen=0;bound=None
                    if target:
                        for r in outputs:
                            pe,ge=sorted(r['pred'][2:4]),sorted(r['gt'][2:4]);seen+=int(max(abs(x-y)/y for x,y in zip(pe,ge))<=.1)
                            if seen==target:bound=r['risks'][control];break
                    ids={r['image'] for r in outputs if bound is not None and r['risks'][control]<=bound}
                    if runner.core.summary(part,ids)['states']!=pair['same_CR']['states']:raise ValueError('Matched CR differs')
                    checks+=2
        # Full saved continuity/AUC/coverage/tie payload replay supplements independent counts.
        # JSON object keys (e.g. run-length histogram) are strings on disk.
        replay=json.loads(json.dumps(runner.core.evaluate(rows),sort_keys=True,allow_nan=False))
        if replay!=report['roles'][role]:raise ValueError('Metric payload differs')
    return dict(passed=True,rows=total,independent_checks=checks,report_sha256=io.sha(out/'report.json'),absent_frontend=absent,
        independently_verified=['history','linear/slopes','TEST quality head','fusion','fallback branch','four states','same-count','same-CR'],
        reused_metric_replay=['continuity','coverage','AUC','ties'],TEST_selected=False,
        boundary_policy='Independent risk tolerance1e-10; exact-cutoff flags use unchanged pinned delivered risk',
        review_source_sha256=io.sha(Path(__file__)))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--allow-absent-frontend',action='store_true');a.add_argument('--receipt',default='server_review.json');args=a.parse_args()
    value=verify(args.out,args.allow_absent_frontend);io.write(args.out/args.receipt,value);print(value)
