#!/usr/bin/env python3
"""Independent label exclusion, frozen-frame/head/fusion replay and metrics."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_image_temporal_crossfit_v1 as runner
from crane_project.tools.review_port_reliability_temporal_fusion_v1 import check_summary,rank_ids,part_for,bad
from crane_project.tools.review_port_reliability_conditional_evidence_v1 import equivalent_payload
from crane_project.utils import port_reliability_image_temporal_crossfit_v1 as core


def equivalent(a,b):
    clean=lambda x:json.loads(json.dumps(x,sort_keys=True,allow_nan=False))
    return equivalent_payload(clean(a),clean(b))


def head_predict(head,matrix):
    n=head['normalizer'];x=((matrix-n['mean'])/n['scale'])[:,1:]
    m=head['model']
    for i in (0,2,4):
        x=x@np.array(m['network.%d.weight'%i]).T+np.array(m['network.%d.bias'%i])
        if i!=4:x=np.maximum(x,0)
    logits=-matrix[:,0]+x[:,0]+m['beta'][0]
    return np.exp(-np.logaddexp(0,-logits))


def logit(risks):
    eps=1/(1+math.exp(12));r=np.clip(risks,eps,1-eps)
    return np.log(r)-np.log1p(-r)


def fusion_check(model,rows,rp,ri):
    ids=[r['image'] for r in rows]
    if ids!=model['train_images']:raise ValueError('Fusion TRAIN membership differs')
    x=np.column_stack((logit(rp),logit(ri)));mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1
    if not np.allclose(mean,model['normalizer']['mean'],atol=1e-13,rtol=0) or not np.allclose(scale,model['normalizer']['scale'],atol=1e-13,rtol=0):raise ValueError('Fusion TRAIN normalizer differs')
    X=np.column_stack((np.ones(len(x)),(x-mean)/scale));w=np.array(model['weights']);offset=x[:,0]
    y=np.array([float(bad(r)) for r in rows]);sw=np.where(y==1,.5/y.sum(),.5/(len(y)-y.sum()))
    u=offset+X@w;p=np.exp(-np.logaddexp(0,-u));penalty=w.copy();penalty[0]=0
    gradient=X.T@(sw*(p-y))+.1*penalty
    objective=float(sw@(np.logaddexp(0,u)-y*u)+.05*penalty@penalty)
    initial=float(sw@(np.logaddexp(0,offset)-y*offset))
    if abs(gradient).max()>1.1e-8 or abs(objective-model['final_objective'])>1e-12 or abs(initial-model['initial_objective'])>1e-12:raise ValueError('Fusion not stationary for fixed objective')
    return float(abs(gradient).max())


def review(result,allow_absent=False):
    p,pins,absent,prior=runner.load(allow_absent)
    report=json.loads((result/'report.json').read_text());heads=json.loads((result/'quality_heads.json').read_text());fusion=json.loads((result/'fusion_models.json').read_text());cut=json.loads((result/'calibration.json').read_text());scores=json.loads((result/'TRAIN_image_scores.json').read_text())
    if report['input_pins']!=pins or report['calibration']!=cut:raise ValueError('Run source/calibration differs')
    native=ROOT/p['native_result'];parent=ROOT/p['temporal_result']
    originals={role:runner.io.read(parent/('scored_'+role+'.jsonl.gz')) for role in ('TRAIN','VAL')}
    matrix,train=runner.feature_matrix(native/'train_native_projected.npz',originals['TRAIN'])
    if len(train)!=2558 or scores['images']!=[r['image'] for r in train]:raise ValueError('Complete TRAIN score membership differs')
    if heads['partition']!=core.folds(train):raise ValueError('Heldout video partitions differ')
    oof=np.full(len(train),np.nan);init=set();updates=0;expected_tags={}
    for split in heads['partition']:
        tag=split['heldout'];head=heads['folds'][tag];fit=split['fit'];test=split['test']
        if head['fit_indices']!=fit or head['train_images']!=[train[i]['image'] for i in fit] or head['heldout_images']!=[train[i]['image'] for i in test] or tag in head['training_sequences']:raise ValueError('Heldout label exclusion failed')
        if set(head['training_sequences'])!=set(train[i]['sequence'] for i in fit):raise ValueError('Training-video identity differs')
        oof[test]=head_predict(head,matrix[test]);expected_tags[tag]=len(fit)
    full=heads['full']
    if full['fit_indices']!=list(range(len(train))) or full['train_images']!=scores['images']:raise ValueError('Full TRAIN head identity differs')
    insample=head_predict(full,matrix);expected_tags['FULL_TRAIN']=len(train)
    for tag,head in list(heads['folds'].items())+[('FULL_TRAIN',full)]:
        x=matrix[head['fit_indices']];mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1
        if not np.allclose(mean,head['normalizer']['mean'],atol=1e-12,rtol=0) or not np.allclose(scale,head['normalizer']['scale'],atol=1e-12,rtol=0):raise ValueError('Fold normalizer uses heldout features')
        if head['bad_rows']!=sum(bool(bad(train[i])) for i in head['fit_indices']) or head['good_rows']!=len(x)-head['bad_rows']:raise ValueError('Fold label count differs')
        budget=100*math.ceil(len(x)/256)
        if head['updates']!=budget or head['final_epoch']!=100 or head['parameter_count']!=4290 or head['detector_updates']:raise ValueError('Quality budget/capacity differs')
        rng=np.random.RandomState(1701);digest=hashlib.sha256()
        for epoch in range(100):digest.update(rng.permutation(len(x)).astype('<i8').tobytes())
        if digest.hexdigest()!=head['batch_order_sha256']:raise ValueError('Quality batch order differs')
        init.add(head['initial_sha256']);updates+=budget
    if len(init)!=1 or not np.isfinite(oof).all() or not np.allclose(oof,scores['crossfit'],atol=1e-12,rtol=0) or not np.allclose(insample,scores['in_sample'],atol=1e-12,rtol=0):raise ValueError('Fresh initialization or head score replay differs')
    records=[json.loads(s) for s in (result/'quality_train_log.jsonl').read_text().splitlines()]
    for tag,n in expected_tags.items():
        logs=[r for r in records if r['tag']==tag];batches=math.ceil(n/256)
        if len(logs)!=100*batches:raise ValueError('Missing quality updates')
        for i,r in enumerate(logs):
            if r['epoch']!=i//batches+1 or r['slot']!=i%batches or r['batch_count']!=min(256,n-256*(i%batches)):raise ValueError('Quality update sequence differs')
            if not all(math.isfinite(v) for v in (r['loss'],r['gradient_norm_before'],r['gradient_norm_after'])) or r['gradient_norm_after']>5.001 or not all(math.isfinite(v) for v in r['gradient_by_parameter'].values()):raise ValueError('Quality nonfinite/clip audit failed')
    if len(records)!=updates:raise ValueError('Unexpected quality updates')
    sup=[i for i,r in enumerate(train) if r['temporal']['available']];rows=[train[i] for i in sup];rp=[r['risks']['partial_auc4'] for r in rows]
    gradients={arm:fusion_check(fusion[arm],rows,rp,(insample if arm==core.ARMS[0] else oof)[sup]) for arm in core.ARMS}
    parts={role:runner.io.read(result/('scored_'+role+'.jsonl.gz')) for role in ('TRAIN','VAL')};checks=0;maxdiff=0
    for role,saved in parts.items():
        original=originals[role];x,outputs=runner.feature_matrix(native/(role.lower()+'_native_projected.npz'),original)
        ri=dict(zip((r['image'] for r in outputs),head_predict(full,x)))
        if len(saved)!=len(original):raise ValueError('Complete role missing')
        replay=core.temporal.base.level.enrich(original,p['boundaries'])
        for r,old,history in zip(saved,original,replay):
            for k in old:
                if k!='risks' and r[k]!=old[k]:raise ValueError('Frozen frame changed: '+k)
            if set(r['risks'])!=set(old['risks'])|set(core.ARMS) or any(r['risks'][k]!=v for k,v in old['risks'].items()):raise ValueError('Frozen risks changed')
            if not equivalent_payload(r['temporal'],history['temporal']):raise ValueError('Causal replay differs')
            image_risk=ri.get(r['image'])
            if (image_risk is None)!=(r['image_crossfit_risk'] is None) or (image_risk is not None and abs(image_risk-r['image_crossfit_risk'])>1e-12):raise ValueError('Full-head delivered risk differs')
            for arm in core.ARMS:
                expected=old['risks']['partial_auc4']
                if old['pred'] is not None and old['temporal']['available']:
                    model=fusion[arm];v=np.array([logit([expected])[0],logit([image_risk])[0]])
                    z=(v-model['normalizer']['mean'])/model['normalizer']['scale'];u=v[0]+np.r_[1,z]@model['weights'];expected=float(np.exp(-np.logaddexp(0,-u)))
                actual=r['risks'][arm]
                if (expected is None)!=(actual is None):raise ValueError('Missing score differs')
                if actual is not None:
                    maxdiff=max(maxdiff,abs(actual-expected))
                    if abs(actual-expected)>1e-12:raise ValueError('Independent fusion score differs')
            checks+=1
        for arm in core.ARMS:
            groups=[]
            for group,entry in cut[arm]['group_protection'].items():
                values=sorted(r['risks'][arm] for r in part_for(parts['VAL'],group) if bad(r) is False);required=math.ceil(.95*len(values));threshold=values[required-1] if values else None
                if entry!=dict(good_outputs=len(values),required=required,cutoff=threshold):raise ValueError('Group95 cutoff differs')
                if threshold is not None:groups.append(threshold)
            if cut[arm]['risk_le']!=max(groups):raise ValueError('Single cutoff differs')
            for group,data in report['roles'][role][arm]['groups'].items():
                part=part_for(saved,group);ids={r['image'] for r in part if r['pred'] is not None and r['risks'][arm]<=cut[arm]['risk_le']}
                check_summary(part,ids,data['candidate']);checks+=1
                for method,match in data['controls'].items():
                    for kind in ('same_count','same_CR'):
                        matched=rank_ids(part,method,count=len(ids)) if kind=='same_count' else rank_ids(part,method,correct=data['candidate']['states']['CR'])
                        check_summary(part,matched,match[kind]);checks+=1
        if not equivalent(report['roles'][role],core.evaluate(saved,cut,prior['calibration']['risk_le'])):raise ValueError('Full metrics/ties/AUROC differ')
    norm=lambda g:dict(g,failures=sorted(g['failures'],key=lambda x:json.dumps(x,sort_keys=True)))
    if norm(report['gate'])!=norm(core.gate(report['roles']['VAL'])):raise ValueError('Gate differs')
    if report['quality_updates']!=updates or report['fit_calls']!=len(heads['partition'])+3 or report['quality_head_count']!=len(heads['partition'])+1 or report['detector_inferences'] or report['detector_updates'] or report['TEST_opened'] or report['formal_policy_changed'] or report['candidate_adopted'] or not report['model_saved_before_VAL']:raise ValueError('Run scope differs')
    return dict(passed=True,rows=sum(len(r) for r in parts.values()),independent_checks=checks,quality_updates=updates,fusion_gradients=gradients,max_score_difference=maxdiff,absent_frontend=absent,report_sha256=runner.io.sha(result/'report.json'),TEST_opened=False,head_label_exclusion_verified=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-absent-frontend',action='store_true');a=p.parse_args()
    runner.io.write(a.out,review(a.result,a.allow_absent_frontend));print('Independent image/temporal crossfit review passed',flush=True)


if __name__=='__main__':main()
