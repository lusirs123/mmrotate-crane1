#!/usr/bin/env python3
"""Independent scalar offline audit of the fixed run; never refit or select."""
from collections import Counter
from copy import deepcopy
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from crane_project.utils import port_reliability_redc_size_v1 as core
from crane_project.utils import port_simple_component_reliability_v1 as simple


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bad(row):
    if row['pred'] is None:return None
    p=sorted(row['pred'][2:4]);g=sorted(row['gt'][2:4])
    return any(abs(a-b)/b>.1 for a,b in zip(p,g))


def counts(rows, accepted):
    states=Counter(dict(FA=0,FR=0,ED=0,CR=0,MISSING=0))
    longest=run=0;previous=None
    for r in sorted(rows,key=lambda r:(r['sequence'],r['frame_id'],r['image'])):
        b=bad(r);keep=r['image'] in accepted
        state='MISSING' if b is None else (('FA' if b else 'CR') if keep else ('ED' if b else 'FR'))
        assert b is not None or not keep
        states[state]+=1
        if previous is None or previous[0]!=r['sequence'] or previous[1]+1!=r['frame_id']:run=0
        run=run+1 if state=='FR' else 0
        longest=max(longest,run);previous=r['sequence'],r['frame_id']
    return dict(states),longest


def grouped(rows):
    return dict(all=rows,**{f+':'+k:[r for r in rows if r[f]==k]
        for f in ('domain','sequence') for k in sorted({r[f] for r in rows})})


def auc(rows,method):
    good=[r['experiment_risks'][method] for r in rows if bad(r) is False]
    wrong=[r['experiment_risks'][method] for r in rows if bad(r) is True]
    if not good or not wrong:return None
    pairs=np.asarray(wrong)[:,None]-np.asarray(good)[None,:]
    return float(np.mean((pairs>0)+.5*(pairs==0)))


def audit(directory):
    directory=Path(directory)
    receipt=json.loads((directory/'completion.json').read_text())
    for name,pin in receipt['artifacts'].items():assert digest(directory/name)==pin,(name,'artifact SHA')
    report=json.loads((directory/'report.json').read_text())
    model=json.loads((directory/'models.json').read_text())
    rows={role:[json.loads(s) for s in gzip.open(directory/('scored_'+role+'.jsonl.gz'),'rt')]
          for role in ('TRAIN','VAL')}
    maximum=0.;statuses=0;rank_checks=0;failures=[]
    train=np.load(directory/'train_features.npz',allow_pickle=False)
    expected=core.normalization(train['features'])
    assert simple.fingerprint(expected)==simple.fingerprint(model['normalizer'])
    training={};activity={}
    for role,part in rows.items():
        assert len(part)==(2558 if role=='TRAIN' else 887)
        assert all(r['reliability_role']==role.lower() for r in part)
        data=np.load(directory/(role.lower()+'_features.npz'),allow_pickle=False)
        ids=data['images'].tolist();by_id={r['image']:r for r in part}
        assert set(ids)=={r['image'] for r in part if r['pred'] is not None}
        h=core.normalize(data['features'],model['normalizer'])[:,1:]
        a=model['models']['redc']
        for i in (0,2):
            h=np.maximum(h@np.asarray(a['network.%d.weight'%i]).T+a['network.%d.bias'%i],0.)
        activity[role.lower()]=dict(outputs=len(h),second_hidden_all_zero=int(np.all(h==0,axis=1).sum()))
        for i,image in enumerate(ids):
            np.testing.assert_allclose(data['features'][i,:3],simple.descriptor(by_id[image]['pred'],by_id[image]['image_size']),rtol=0,atol=1e-12)
        for arm in core.ARMS:
            risks=simple.sigmoid(core.numpy_logits(model['models'][arm],data['features'],model['normalizer'],arm))
            for image,risk in zip(ids,risks):
                delta=abs(float(risk)-by_id[image]['experiment_risks'][arm]);maximum=max(maximum,delta)
                assert delta<1e-12
            for r in part:
                d=deepcopy(r['original_simple_decision'])
                if r['pred'] is not None:
                    d['risks']['size']=r['experiment_risks'][arm]
                    d['size_accepted']=d['risks']['size']<=report['VAL_cutoffs'][arm]['risk_le']
                assert d==r['candidate_decisions'][arm]
        groups=grouped(part)
        stats=report['VAL_statistics'] if role=='VAL' else report['train_statistics']
        for arm in core.ARMS:
            for name,group in groups.items():
                out=[r for r in group if r['pred'] is not None]
                point=stats[arm][name];cut=report['VAL_cutoffs'][arm]['risk_le']
                kept={r['image'] for r in out if r['experiment_risks'][arm]<=cut}
                c,l=counts(group,kept);assert c==point['actual']['states'];assert l==point['actual']['runs']['correct_rejection']['longest'];statuses+=1
                for method,rank in point['rank'].items():
                    value=auc(group,method)
                    assert value is None and rank['error_auroc'] is None or value is not None and abs(value-rank['error_auroc'])<1e-12
                    rank_checks+=1
                for method in point['matched_CR_controls']:
                    ranked=sorted(out,key=lambda r:(r['experiment_risks'][method],r['image']))
                    selected={r['image'] for r in ranked[:len(kept)]}
                    c2,l2=counts(group,selected)
                    assert c2==point['same_count_controls'][method]['states'];statuses+=1
                    target=c['CR'];acc_good=0;boundary=None
                    if target:
                        for r in ranked:
                            acc_good+=int(not bad(r))
                            if acc_good==target:boundary=r['experiment_risks'][method];break
                    selected={r['image'] for r in out if boundary is not None and r['experiment_risks'][method]<=boundary}
                    c3,l3=counts(group,selected)
                    assert c3==point['matched_CR_controls'][method]['states'];assert l3==point['matched_CR_controls'][method]['runs']['correct_rejection']['longest'];statuses+=1
                    k=len(kept)
                    if k:
                        boundary=ranked[k-1]['experiment_risks'][method]
                        before=[r for r in ranked if r['experiment_risks'][method]<boundary]
                        tie=[r for r in ranked if r['experiment_risks'][method]==boundary]
                        lower=sum(bad(r) for r in before)+max(0,k-len(before)-sum(not bad(r) for r in tie))
                    else:lower=0
                    assert lower==point['control_tie_bounds'][method]['bad_min_over_tie']
                    if role=='VAL' and arm=='redc' and method in core.CONTROLS:
                        if c['FA']>lower:failures.append(dict(group=name,control=method,check='same_count_FA_nonincrease'))
                        if name=='all' and c['FA']>=lower:failures.append(dict(group=name,control=method,check='strict_overall_same_count_FA_gain'))
                        if c['FA']>c3['FA']:failures.append(dict(group=name,control=method,check='same_CR_FA_nonincrease'))
                        if name=='all' and c['FA']>=c3['FA']:failures.append(dict(group=name,control=method,check='strict_overall_same_CR_FA_gain'))
                        if c3['CR']!=c['CR']:failures.append(dict(group=name,control=method,check='same_CR_tie_inconclusive'))
                        if l>l3:failures.append(dict(group=name,control=method,check='longest_FR_same_CR_nonincrease'))
                if role=='VAL' and arm=='redc' and c['CR']<int(np.ceil(.95*(c['CR']+c['FR']))):
                    failures.append(dict(group=name,check='correct_retention'))
        if role=='TRAIN':
            y=np.array([bad(by_id[i]) for i in ids],dtype=float);weights=core.class_weights(y)
            for arm in core.ARMS:
                logits=core.numpy_logits(model['models'][arm],data['features'],model['normalizer'],arm)
                training[arm]=dict(balanced_final_loss=float(np.mean((np.logaddexp(0.,logits)-y*logits)*weights)),
                    initial_loss=float(np.mean((np.logaddexp(0.,-data['features'][:,0])+y*data['features'][:,0])*weights)),
                    TRAIN_error_AUROC=auc(part,arm))
    for arm,point in report['VAL_cutoffs'].items():
        requirements=[]
        for name,group in grouped(rows['VAL']).items():
            good=sorted(r['experiment_risks'][arm] for r in group if bad(r) is False)
            need=int(np.ceil(.95*len(good)))
            if need:requirements.append(good[need-1])
        assert point['risk_le']==max(requirements)
    assert failures==report['gate']['failures']
    logs=[json.loads(s) for s in (directory/'train_log.jsonl').read_text().splitlines()]
    assert len(logs)==2000 and Counter(r['arm'] for r in logs)=={'linear':1000,'redc':1000}
    assert {(r['epoch'],r['slot'],r['arm']) for r in logs}=={(e,s,a) for e in range(1,101) for s in range(10) for a in core.ARMS}
    assert all(r['batch_count']==(254 if r['slot']==9 else 256) for r in logs)
    order=hashlib.sha256();rng=np.random.RandomState(1701)
    for _ in range(100):order.update(rng.permutation(2558).astype('<i8').tobytes())
    assert order.hexdigest()==report['batch_order_sha256']
    assert all(r['gradient_norm_after']<=5.0001 and np.isfinite(r['loss']) for r in logs)
    v=rows['VAL'];redc={r['image'] for r in v if r['candidate_decisions']['redc']['size_accepted']}
    score_kept={r['image'] for r in v if r['pred'] is not None and r['experiment_risks']['score_only']<=report['VAL_cutoffs']['score_only']['risk_le']}
    return dict(passed=True,rows=3445,risk_replay_maximum=maximum,state_and_matched_groups_checked=statuses,
        rank_AUROC_checks=rank_checks,gate_failures_exact=failures,training=training,
        updates_checked=2000,redc_and_score_fixed_point_identical=redc==score_kept,
        report_sha256=digest(directory/'report.json'),models_sha256=digest(directory/'models.json'),
        independent_scalar_states_and_pairwise_AUROC=True,observed_hidden_activity=activity,
        fitting_updates=0,TEST_read=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    result=audit(args.directory)
    path=args.directory/'local_review.json'
    with path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
