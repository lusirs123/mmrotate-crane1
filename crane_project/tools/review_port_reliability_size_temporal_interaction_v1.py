#!/usr/bin/env python3
"""Independent scalar stationarity, neutral/base identity, calibration and metrics."""
import argparse
import json
import math
from pathlib import Path
import sys
from statistics import median
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_size_temporal_interaction_v1 as runner
from crane_project.tools.review_port_reliability_temporal_fusion_v1 import check_summary,rank_ids,part_for,bad
from crane_project.tools.review_port_reliability_conditional_evidence_v1 import equivalent_payload
from crane_project.utils import port_reliability_size_temporal_interaction_v1 as core


def vector(row):
    a,b=sorted(row['pred'][2:4],reverse=True);w,h=row['image_size'];s=min(max(row['pred'][5],1e-6),1-1e-6)
    return np.array([math.log(s)-math.log1p(-s),.5*(math.log(a)+math.log(b)-math.log(w)-math.log(h)),math.log(a)-math.log(b),row['temporal']['primary']])


def scalar_check(model,rows,base):
    if [r['image'] for r in rows]!=base['fusion4']['train_images'] or [r['image'] for r in rows]!=model['train_images']:raise ValueError('Supported TRAIN identities differ')
    m=base['fusion4'];z=(np.array([vector(r) for r in rows])-m['mean'])/m['scale']
    logits=m['weights'][0]+z.dot(m['weights'][1:]);phi=z[:,1]*z[:,3]
    y=np.array([float(bad(r)) for r in rows]);w=np.where(y==1,.5/y.sum(),.5/(len(y)-y.sum()))
    gamma=model['gamma'];u=logits+gamma*phi;prob=np.exp(-np.logaddexp(0.,-u))
    loss=float(w.dot(np.logaddexp(0.,u)-y*u)+.05*gamma*gamma)
    gradient=float(w.dot((prob-y)*phi)+.1*gamma)
    hessian=float(w.dot(prob*(1-prob)*phi*phi)+.1)
    initial=float(w.dot(np.logaddexp(0.,logits)-y*logits))
    if abs(gradient)>1.1e-8 or abs(loss-model['final_objective'])>1e-12 or abs(initial-model['initial_objective'])>1e-12:
        raise ValueError('Scalar not stationary for prespecified objective')
    if hessian<=0 or loss>initial+1e-12 or model['train_rows']!=len(rows) or model['bad_rows']!=int(y.sum()):raise ValueError('Scalar fit contract differs')
    return gradient


def equivalent_metrics(left,right):
    # JSON converts integer histogram keys to strings. Normalize representation,
    # while the shared comparator keeps all counts/identities exact.
    normalize=lambda x:json.loads(json.dumps(x,sort_keys=True,allow_nan=False))
    return equivalent_payload(normalize(left),normalize(right))


def review(result,allow_absent=False,original_review=None):
    p,pins,absent,models,oldcutoffs=runner.load(allow_absent)
    report=json.loads((result/'report.json').read_text());model=json.loads((result/'scalar_model.json').read_text())
    cutoff=json.loads((result/'calibration.json').read_text())
    if set(report['input_pins'])!=set(pins) or report['model']!=model or report['calibration']!=cutoff or report['base_models']!=models:raise ValueError('Saved artifacts/pins differ')
    repairs={}
    for name,digest in pins.items():
        if report['input_pins'][name]==digest:continue
        allowed={'crane_project/tools/review_'+core.VERSION+'.py':'original_review.py',
                 'tests/test_'+core.VERSION+'.py':'original_tests.py'}
        if original_review is None or name not in allowed:raise ValueError('Non-review source changed')
        original=original_review if name.startswith('crane_project/') else result/allowed[name]
        if original.is_symlink() or original.resolve().parent!=result.resolve() or runner.io.sha(original)!=report['input_pins'][name]:raise ValueError('Original audit source not preserved')
        repairs[name]=dict(original_sha256=report['input_pins'][name],current_sha256=digest,
                           scope='audit JSON key normalization only; no refit/rescore/calibration')
    if not set(report['absent_frontend'])<=set(p['frontend_pins']) or (not allow_absent and report['absent_frontend']):raise ValueError('Unverified frontend')
    previous=ROOT/p['previous_result'];parts={role:runner.io.read(result/('scored_'+role+'.jsonl.gz')) for role in ('TRAIN','VAL')}
    checks=0;maxdiff=0.
    for role,rows in parts.items():
        original=runner.io.read(previous/('scored_'+role+'.jsonl.gz'))
        if len(rows)!=len(original):raise ValueError('Complete role missing')
        replay=core.base.level.enrich(original,p['boundaries'])
        for r,old,history in zip(rows,original,replay):
            for key in old:
                if key!='risks' and r[key]!=old[key]:raise ValueError('Frozen frame mutated: '+key)
            if set(r['risks'])!=set(old['risks'])|{core.METHOD}:raise ValueError('Unexpected risk features')
            if any(r['risks'][k]!=v for k,v in old['risks'].items()):raise ValueError('Frozen baseline risk changed')
            if r['temporal']!=history['temporal']:raise ValueError('Causal history replay differs')
            expected=old['risks']['fusion4']
            if r['pred'] is not None and r['temporal']['available']:
                m=models['fusion4'];z=(vector(r)-m['mean'])/m['scale']
                logit=float(m['weights'][0]+z.dot(m['weights'][1:]));expected=math.exp(-float(np.logaddexp(0.,-(logit+model['gamma']*z[1]*z[3]))))
            actual=r['risks'][core.METHOD]
            if expected is None:
                if actual is not None:raise ValueError('Missing output given risk')
            else:
                error=abs(actual-expected);maxdiff=max(maxdiff,error)
                if error>1e-12:raise ValueError('Independent scalar score differs')
            if core.online_risk(r['pred'],r['image_size'],r['temporal'],models,0.)!=core.base.online_risks(r['pred'],r['image_size'],r['temporal'],models)['fusion4']:raise ValueError('Neutral identity differs')
            checks+=1
    gradient=scalar_check(model,[r for r in parts['TRAIN'] if r['pred'] is not None and r['temporal']['available']],models)
    thresholds=[]
    for group,entry in cutoff['group_protection'].items():
        values=sorted(r['risks'][core.METHOD] for r in part_for(parts['VAL'],group) if bad(r) is False)
        required=math.ceil(.95*len(values));threshold=values[required-1] if values else None
        if entry!=dict(good_outputs=len(values),required=required,cutoff=threshold):raise ValueError('VAL group threshold differs')
        if threshold is not None:thresholds.append(threshold)
    if cutoff['risk_le']!=max(thresholds):raise ValueError('Single VAL threshold differs')
    for role,rows in parts.items():
        for group,data in report['roles'][role]['groups'].items():
            part=part_for(rows,group)
            ids={r['image'] for r in part if r['pred'] is not None and r['risks'][core.METHOD]<=cutoff['risk_le']}
            check_summary(part,ids,data['candidate']);checks+=1
            for method,matches in data['controls'].items():
                for kind,target in [('same_count',len(ids)),('same_CR',data['candidate']['states']['CR'])]:
                    matched=rank_ids(part,method,count=target) if kind=='same_count' else rank_ids(part,method,correct=target)
                    check_summary(part,matched,matches[kind]);checks+=1
        if not equivalent_metrics(report['roles'][role],core.evaluate(rows,cutoff['risk_le'],oldcutoffs['fusion4']['risk_le'])):raise ValueError('Full metrics, ties, AUROC or old comparison differs')
    def normalized(g):
        g=dict(g);g['failures']=sorted(g['failures'],key=lambda x:json.dumps(x,sort_keys=True));return g
    if normalized(report['gate'])!=normalized(core.gate(report['roles']['VAL'])):raise ValueError('Gate differs')
    if (report['fit_calls']!=1 or report['trainable_parameters']!=1 or report['neural_training_updates']
        or report['detector_inferences'] or report['TEST_opened'] or report['formal_policy_changed']
        or report['candidate_adopted'] or not report['scalar_saved_before_VAL']):raise ValueError('Run scope differs')
    return dict(passed=True,rows=sum(len(x) for x in parts.values()),independent_checks=checks,
                scalar_gradient=gradient,max_score_difference=maxdiff,absent_frontend=absent,
                report_sha256=runner.io.sha(result/'report.json'),TEST_opened=False,review_only_repairs=repairs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-absent-frontend',action='store_true');p.add_argument('--original-review-source',type=Path);a=p.parse_args()
    runner.io.write(a.out,review(a.result,a.allow_absent_frontend,a.original_review_source));print('Independent scalar review passed',flush=True)


if __name__=='__main__':main()
