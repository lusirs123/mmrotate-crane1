#!/usr/bin/env python3
"""Independent candidate geometry/median, native index, frozen frames and metrics audit."""
import argparse
import json
import math
from pathlib import Path
from statistics import median
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_candidate_support_v1 as runner
from crane_project.tools.review_port_reliability_temporal_fusion_v1 import check_summary,rank_ids,part_for
from crane_project.tools.review_port_reliability_partial_auc_v1 import equivalent_metrics
core=runner.core


def independent_evidence(row):
    pred=row['pred'];pool=row['candidate_pool'];chosen=row['candidate_chosen'];ev=row['candidate_evidence']
    if pred is None:
        if pool or chosen is not None or ev['available'] or ev['reason']!='missing':raise ValueError('Missing given evidence')
        return
    if not 0<=chosen<len(pool):raise ValueError('Native selected index lost')
    b=pool[chosen];ref=row['b_original']
    if not np.allclose(b[:4],ref[:4],atol=5e-4,rtol=2e-5) or abs((b[4]-ref[4]+math.pi/2)%math.pi-math.pi/2)>2e-6 or b[5]!=ref[5] or b[5]!=pred[5]:raise ValueError('Independent native B identity differs')
    if b[5]!=max(x[5] for x in pool):raise ValueError('Observed source is not actual Top1')
    if any(x[5]<=row['candidate_trace']['score_thr'] for x in pool):raise ValueError('Below native score threshold')
    L=max(pred[2:4]);S=min(pred[2:4]);theta=pred[4]+(math.pi/2 if pred[2]<pred[3] else 0)
    eligible=[]
    for i,box in enumerate(pool):
        if i==chosen:continue
        angle=box[4]+(math.pi/2 if box[2]<box[3] else 0)
        delta=math.atan2(math.sin(2*(angle-theta)),math.cos(2*(angle-theta)))/2
        dx=box[0]-pred[0];dy=box[1]-pred[1]
        u=math.cos(theta)*dx+math.sin(theta)*dy;v=-math.sin(theta)*dx+math.cos(theta)*dy
        if abs(u)<=L/2 and abs(v)<=S/2 and abs(delta)<=math.pi/12+1e-14:eligible.append(i)
    selected=sorted(eligible,key=lambda i:(-pool[i][5],i))[:8]
    if [n['index'] for n in ev['neighbors']]!=selected or any(n['box']!=pool[n['index']] for n in ev['neighbors']):raise ValueError('Independent neighbor membership differs')
    available=len(selected)>=3
    if ev['available']!=available or ev['eligible']!=len(eligible) or ev['surviving']!=len(pool):raise ValueError('Support/count mismatch')
    features=[median(math.log((max if j==0 else min)(pool[i][2:4])/(L if j==0 else S)) for i in selected) for j in range(2)] if available else None
    if features is None:
        if ev['features'] is not None:raise ValueError('Unsupported features synthesized')
    elif not np.allclose(features,ev['features'],rtol=0,atol=1e-12):raise ValueError('Independent median differs')


def fit_check(model,rows,dim):
    part=[r for r in rows if core.active(r)]
    if model['train_images']!=[r['image'] for r in part]:raise ValueError('Fit membership changed')
    x=np.asarray([core.vector(r,dim) for r in part]);mean=x.mean(0);scale=x.std(0);scale=np.where(scale<1e-8,1.,scale)
    if not np.allclose(mean,model['mean'],atol=1e-13,rtol=0) or not np.allclose(scale,model['scale'],atol=1e-13,rtol=0):raise ValueError('TRAIN normalization differs')
    z=(x-mean)/scale;y=np.array([core.parent.base.level.metrics.bad(r) for r in part])
    d=z[~y][:,None,:]-z[y][None,:,:];w=np.asarray(model['weights'][1:]);tau=model['tail_threshold']
    u=np.einsum('gbi,i->gb',d,w);p=np.exp(-np.logaddexp(0.,-u));h=np.logaddexp(0.,u).mean(1)
    outer=np.exp(-np.logaddexp(0.,-(h-tau)/.1));jac=np.einsum('gb,gbi->gi',p,d)/y.sum()
    gradient=np.r_[np.mean(outer[:,None]*jac,axis=0)/.05+.1*w,1-outer.mean()/.05]
    if abs(gradient).max()>1.1e-8 or int(y.sum())!=model['bad_rows']:raise ValueError('Independent fit stationarity differs')
    return float(abs(gradient).max())


def review(result,allow_absent=False):
    p,pins,absent,previous=runner.load(allow_absent)
    report=json.loads((result/'report.json').read_text())
    if report['input_pins']!=pins or report['TEST_opened'] or report['candidate_adopted'] or report['formal_policy_changed'] or report['detector_updates']:raise ValueError('Scope/pins changed')
    if report['absent_frontend']:raise ValueError('Execution did not verify actual frontend')
    parts={};checks=0
    for role in ('TRAIN','VAL'):
        if role=='VAL' and not report['VAL_opened']:
            if any((result/name).exists() for name in ('candidates_VAL.jsonl.gz','scored_VAL.jsonl.gz','models.json','calibration.json')):raise ValueError('Stop contract violated')
            continue
        rows=runner.io.read(result/('candidates_'+role+'.jsonl.gz'));original=runner.io.read(ROOT/p['previous_result']/('scored_'+role+'.jsonl.gz'))
        if len(rows)!=len(original) or [r['image'] for r in rows]!=[r['image'] for r in original]:raise ValueError('Full role identities differ')
        for row,old in zip(rows,original):
            if any(row[k]!=v for k,v in old.items()):raise ValueError('Frozen frame/risks changed')
            independent_evidence(row);checks+=1
        parts[role]=rows
        proof=report['train_proof' if role=='TRAIN' else 'val_proof']
        if proof['frames']!=len(rows) or proof['detector_inferences']!=len(rows) or proof['detector_updates'] or proof['GT_online']:raise ValueError('Inference proof mismatch')
    screen=core.screen(parts['TRAIN'])
    if screen!=report['screen'] or screen!=json.loads((result/'TRAIN_screen.json').read_text()):raise ValueError('TRAIN prerequisite screen differs')
    gradients={}
    if not screen['passed']:
        if report['fit_calls'] or report['VAL_opened'] or report['status']!='TRAIN_EVIDENCE_FAILED_STOP':raise ValueError('Failed prerequisite did not stop')
    else:
        models=json.loads((result/'models.json').read_text());cuts=json.loads((result/'calibration.json').read_text())
        if report['fit_calls']!=2 or not report['models_saved_before_VAL'] or models!=report['models'] or cuts!=report['calibration']:raise ValueError('Fit/VAL ordering artifacts differ')
        for method,dim in ((core.CONTROL,4),(core.METHOD,6)):gradients[method]=fit_check(models[method],parts['TRAIN'],dim)
        for role,rows in parts.items():
            scored=runner.io.read(result/('scored_'+role+'.jsonl.gz'))
            if not equivalent_metrics(scored,core.score(rows,models)):raise ValueError('Score replay differs')
            evaluation=report['roles'][role]
            for group,arms in evaluation['groups'].items():
                part=part_for(scored,group)
                for method,arm in arms.items():
                    cutoff=cuts[method];ids=core.accepted(part,method,cutoff['risk_le'],cutoff['fallback_risk_le'])
                    candidate_count=len(ids)
                    check_summary(part,ids,arm['candidate']);checks+=1
                    for control,matches in arm['controls'].items():
                        for kind,data in matches.items():
                            ids=rank_ids(part,control,count=candidate_count) if kind=='same_count' else rank_ids(part,control,correct=arm['candidate']['states']['CR'])
                            check_summary(part,ids,data);checks+=1
            if not equivalent_metrics(evaluation,core.evaluate(scored,cuts)):raise ValueError('Full evaluation differs')
        for method in models:
            if cuts[method]!=core.calibrate(runner.io.read(result/'scored_VAL.jsonl.gz'),method,previous['calibration']['risk_le']):raise ValueError('VAL calibration differs')
        if report['gate']!=core.gate(report['roles']['VAL']):raise ValueError('Gate differs')
    return dict(passed=True,rows=sum(map(len,parts.values())),independent_checks=checks,
        objective_gradients=gradients,local_absent_frontend=absent,report_sha256=runner.io.sha(result/'report.json'),
        TEST_opened=False,status=report['status'])


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--result',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--allow-absent-frontend',action='store_true');args=a.parse_args()
    runner.io.write(args.out,review(args.result,args.allow_absent_frontend));print('Independent candidate support review passed')
