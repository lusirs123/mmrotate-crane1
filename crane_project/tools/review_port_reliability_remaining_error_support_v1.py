#!/usr/bin/env python3
"""Independent states/runs/donor membership/distances; complete audit replay."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_remaining_error_support_v1 as runner
from crane_project.tools.review_port_reliability_conditional_evidence_v1 import equivalent_payload


def review(out,allow_absent=False):
    p,pins,absent,rows,x,ids=runner.load(allow_absent);report=json.loads((out/'report.json').read_text());result=report['result'];checks=0
    if report['input_pins']!=pins or report['settings']!=p['settings'] or report['frames']!=2558:raise ValueError('Source/settings differ')
    if any(report[k] for k in ('fits','threshold_updates','detector_inferences','neural_updates','policy_changed','formal_flags_changed','VAL_opened','TEST_opened','candidate_adopted')):raise ValueError('Scope violation')
    byid={r['image']:r for r in rows};states={};signs={};sizes={};regimes={};episodes=[];current=None;prev=None
    for r in rows:
        if r['pred'] is None:states[r['image']]='MISSING';wrong=False
        else:
            pe,ge=[sorted(r[k][2:4],reverse=True) for k in ('pred','gt')];e=[(a-b)/b for a,b in zip(pe,ge)];wrong=any(abs(a)>.1 for a in e)
            states[r['image']]=('FA' if wrong else 'CR') if r['original_simple_decision']['size_accepted'] else ('ED' if wrong else 'FR')
            signs[r['image']]='/'.join('small' if a<-.1 else 'large' if a>.1 else 'within' for a in e)
            sizes[r['image']]=.5*(math.log(pe[0])+math.log(pe[1])-math.log(r['image_size'][0])-math.log(r['image_size'][1]))
        regimes[r['image']]='unsupported' if not r['temporal']['available'] else 'low' if r['temporal']['primary']<=p['settings']['level_cutoff'] else 'high'
        if not wrong or prev is None or prev['sequence']!=r['sequence'] or prev['frame_id']+1!=r['frame_id'] or r['temporal'].get('acquisition_boundary'):current=None
        if wrong:
            if current is None:current=[];episodes.append(current)
            current.append(r['image'])
        prev=r;checks+=1
    if episodes!=[v['images'] for v in result['error_runs']]:raise ValueError('Independent error runs differ')
    episode={i:k for k,ep in enumerate(episodes) for i in ep}
    mean=x.mean(0);std=x.std(0);std[std<1e-8]=1.;z=(x-mean)/std;index={s:i for i,s in enumerate(ids)}
    if not np.allclose(result['normalizer']['mean'],mean,atol=1e-12,rtol=0) or not np.allclose(result['normalizer']['scale'],std,atol=1e-12,rtol=0):raise ValueError('TRAIN-only descriptive normalization differs')
    if {v['image'] for v in result['remaining_FA']}!={i for i,s in states.items() if s=='FA'}:raise ValueError('FA scope differs')
    for d in result['remaining_FA']:
        r=byid[d['image']];i=r['image']
        def match(q):
            j=q['image'];return q['sequence']==r['sequence'] and regimes[j]==regimes[i] and abs(sizes[j]-sizes[i])<=.1 and abs(q['frame_id']-r['frame_id'])>3
        semantic=[q['image'] for q in rows if states[q['image']]=='ED' and signs[q['image']]==signs[i]]
        donors=[j for j in semantic if match(byid[j]) and episode[j]!=episode[i]]
        correct=[q['image'] for q in rows if states[q['image']]=='CR' and match(q)]
        if donors!=d['matched_ED_images'] or correct!=d['matched_CR_images'] or semantic!=d['same_signature_ED_images']:raise ValueError('Independent donor membership differs')
        for block,begin,end in [('static3',0,3),('native256',3,259)]:
            for key,part in [('rejected_error',donors),('accepted_correct',correct)]:
                pairs=[]
                for j in part:
                    distance=math.sqrt(sum((float(z[index[i],k])-float(z[index[j],k]))**2 for k in range(begin,end))/(end-begin));pairs.append((distance,j))
                v=d['nearest'][block][key]
                if not pairs:
                    if v is not None:raise ValueError('Missing distance replaced')
                else:
                    value,j=min(pairs)
                    if j!=v['image'] or abs(value-v['distance'])>1e-12:raise ValueError('Independent native/static distance differs')
                checks+=1
    # Counts, signed errors, regimes and summaries are also independently recomputed.
    for g,part in runner.core.temporal.metrics.grouped(rows).items():
        actual={s:sum(states[r['image']]==s for r in part) for s in runner.core.STATES}
        if actual!=result['groups'][g]['states']:raise ValueError('Independent group states differ')
    rebuilt=json.loads(json.dumps(runner.core.audit(rows,x,ids),sort_keys=True,allow_nan=False))
    if not equivalent_payload(rebuilt,result):raise ValueError('Full audit payload differs')
    return dict(passed=True,rows=len(rows),independent_checks=checks,report_sha256=runner.io.sha(out/'report.json'),absent_frontend=absent,
        independent=['states','error runs','donor membership','nearest distances','TRAIN normalization'],full_payload_replayed=True,VAL_opened=False,TEST_opened=False)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--allow-absent-frontend',action='store_true');a.add_argument('--receipt',default='server_review.json');args=a.parse_args()
    value=review(args.out,args.allow_absent_frontend);runner.io.write(args.out/args.receipt,value);print(value)
