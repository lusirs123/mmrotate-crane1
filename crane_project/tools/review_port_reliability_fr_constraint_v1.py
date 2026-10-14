#!/usr/bin/env python3
"""Independent numerical KKT, membership, delivery and metric review; no refitting."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_reliability_fr_constraint_v1 as runner
from crane_project.tools import review_port_reliability_video_transfer_v1 as old


def check_models(models, fitting):
    rows=[r for r in fitting if r['pred'] is not None]
    x=np.array([old.descriptor(r,3) for r in rows]); y=np.array([old.bad(r) for r in rows],dtype=bool)
    mean=x.mean(0); scale=x.std(0); scale[scale<1e-8]=1
    d=np.column_stack((np.ones(len(rows)),(x-mean)/scale)); errors=d[y]
    for name,m in models.items():
        if not np.allclose(mean,m['mean'],atol=1e-12,rtol=0) or not np.allclose(scale,m['scale'],atol=1e-12,rtol=0):
            raise ValueError('Fit-only normalizer differs')
        if m['train_rows']!=len(rows) or m['bad_rows']!=int(y.sum()) or m['good_rows']!=int((~y).sum()):
            raise ValueError('Class count differs')
        w=np.array(m['weights']); penalty=w.copy(); penalty[0]=0; z=d@w
        if name=='control3':
            a=np.where(y,.5/y.sum(),.5/(~y).sum()); p=np.exp(-np.logaddexp(0.,-z))
            grad=d.T@(a*(p-y))+.1*penalty
            loss=float(a@(np.logaddexp(0.,z)-y*z)+.05*(penalty@penalty))
            if max(abs(grad))>1.0001e-8 or abs(loss-m['final_objective'])>1e-12:
                raise ValueError('Control balanced optimizer differs')
        else:
            cert=m['certificate']; zz=errors@w
            loss=float(np.logaddexp(0.,-zz).mean()+.05*(penalty@penalty))
            grad=errors.T@(np.exp(-np.logaddexp(0.,-zz))-1)/len(errors)+.1*penalty
            complement=0; seqs=sorted({r['sequence'] for r in rows})
            if set(cert['dual_multipliers'])!=set(seqs) or m['fitting_videos']!=seqs:
                raise ValueError('Wrong video constraints')
            for seq in seqs:
                mask=np.array([not yy and r['sequence']==seq for yy,r in zip(y,rows)])
                dd=d[mask]; zc=dd@w
                q=float(np.logaddexp(0.,zc).mean()/math.log(2))
                j=dd.T@np.exp(-np.logaddexp(0.,-zc))/(len(dd)*math.log(2))
                lam=cert['dual_multipliers'][seq]
                if not 0<=q<.05 or lam<0 or abs(q-cert['surrogate_by_video'][seq])>1e-12:
                    raise ValueError('Nonfeasible constraint or multiplier')
                if float((zc>=0).mean())>q+1e-12:
                    raise ValueError('Surrogate fails empirical FR upper bound')
                grad+=lam*j; complement+=lam*(.05-q)
            if max(abs(grad))>5e-6 or complement>1e-6 or abs(loss-cert['objective'])>1e-12:
                raise ValueError('Independent convex KKT certificate failed')
            if abs(max(abs(grad))-cert['max_abs_stationarity'])>1e-10 or abs(complement-cert['duality_gap'])>1e-12:
                raise ValueError('Reported certificate residual differs')
            if cert['guarantee_on_unseen_video'] or not cert['empirical_fit_only']:
                raise ValueError('Unsupported guarantee')
            if [s['mu'] for s in m['stages']]!=runner.core.SETTINGS['barrier_schedule'] or any(s['iterations']>100 or s['max_abs_gradient']>2e-7 for s in m['stages']):
                raise ValueError('Fixed optimization budget changed')
    return 2


def check_risks(actual, original, models):
    for a,r in zip(actual,original):
        if a['image']!=r['image'] or any(a[k]!=v for k,v in r.items() if k!='risks'):
            raise ValueError('Prediction/GT/history/output/formal identity changed')
        if any(a['risks'][k]!=v for k,v in r['risks'].items()):
            raise ValueError('Historical scores changed')
        for name,m in models.items():
            expected=None
            if r['pred'] is not None:
                x=old.descriptor(r,3)
                expected=m['weights'][0]+sum((xx-mu)/s*w for xx,mu,s,w in zip(x,m['mean'],m['scale'],m['weights'][1:]))
            value=a['risks'][name]
            if (value is None)!=(expected is None) or (value is not None and abs(value-expected)>1e-12):
                raise ValueError('Independent logit differs')
    return len(actual)*2


def check_evaluation(evaluation,rows,thresholds):
    groups={'all':rows}
    for f in ('domain','sequence'):
        for v in sorted({r[f] for r in rows}): groups[f+':'+v]=[r for r in rows if r[f]==v]
    if set(evaluation)!=set(groups): raise ValueError('Wrong reporting scope')
    n=0
    for g,part in groups.items():
        data=evaluation[g]
        old.check_summary(data['formal'],part,{r['image'] for r in part if r['original_simple_decision']['size_accepted']})
        for method,e in data['methods'].items():
            keep={r['image'] for r in part if r['pred'] is not None and
                  (r['risks'][method]<0 if method=='constrained3' else r['risks'][method]<=thresholds[method]['risk_le'])}
            old.check_summary(e['delivery'],part,keep)
            old.check_point(e['offline_correct95'],part,method,correct=math.ceil(.95*sum(old.bad(r) is False for r in part)))
            good=np.array([r['risks'][method] for r in part if old.bad(r) is False]); err=np.array([r['risks'][method] for r in part if old.bad(r) is True])
            auc=float(((err[:,None]>good[None])+0.5*(err[:,None]==good[None])).mean()) if len(err) and len(good) else None
            if (auc is None)!=(e['AUROC'] is None) or (auc is not None and abs(auc-e['AUROC'])>1e-12):
                raise ValueError('AUROC differs')
            for control,value in e['controls'].items():
                old.check_point(value['same_count'],part,control,count=e['delivery']['accepted'])
                old.check_point(value['same_CR'],part,control,correct=e['delivery']['states']['CR'])
            n+=1
    return n


def review(out,allow_absent=False):
    p,pins,absent,original,policy=runner.load(allow_absent)
    report=json.loads((out/'report.json').read_text()); result=report['result']
    if report['input_pins']!=pins or report['settings']!=runner.core.SETTINGS or report['frames']!=len(original):
        raise ValueError('Report binding changed')
    if result['fits']!=14 or any(result[k] for k in ('VAL_opened','TEST_opened','neural_updates','detector_inferences','candidate_adopted')):
        raise ValueError('Experiment scope changed')
    if any(report[k] for k in ('formal_policy_changed','geometry_changed','output_identity_changed')):
        raise ValueError('Formal pipeline changed')
    held=runner.parent.io.read(out/'held_TRAIN.jsonl.gz'); refrows=runner.parent.io.read(out/'reference_TRAIN.jsonl.gz')
    if len(held)!=len(original) or len(refrows)!=len(original): raise ValueError('Wrong frame count')
    n=0; ref=result['reference']
    n+=check_risks(refrows,original,ref['models']); n+=check_models(ref['models'],original)
    for k in ('mean','scale','weights'):
        if not np.allclose(ref['models']['control3'][k],policy['models']['size'][k],atol=1e-12,rtol=0): raise ValueError('Full control identity differs')
    old.check_thresholds({k:v for k,v in ref['thresholds'].items() if k!='constrained3'},refrows)
    if ref['thresholds']['constrained3']!={'logit_lt':0.,'rule':'fixed_zero_no_posthoc_tuning'}: raise ValueError('Tuned candidate cutoff')
    n+=check_evaluation(ref['evaluation'],refrows,ref['thresholds'])
    seqs=sorted({r['sequence'] for r in original})
    if set(result['folds'])!=set(seqs): raise ValueError('Wrong folds')
    failures=[]; totals={'control3':0,'constrained3':0}
    for seq,f in result['folds'].items():
        fit=[r for r in original if r['sequence']!=seq]; origheld=[r for r in original if r['sequence']==seq]
        h=[r for r in held if r['sequence']==seq]
        if f['fit_images']!=[r['image'] for r in fit] or f['held_images']!=[r['image'] for r in origheld]: raise ValueError('Fold contamination')
        if any(r['diagnostic_fold']!=seq for r in h): raise ValueError('Wrong fold identity')
        n+=check_models(f['models'],fit); n+=check_risks(h,origheld,f['models'])
        sf=deepcopy(fit)
        if set(f['fit_risks'])!={r['image'] for r in fit}: raise ValueError('Wrong fit scores')
        for r in sf: r['risks']=f['fit_risks'][r['image']]
        n+=check_risks(sf,fit,f['models'])
        old.check_thresholds({k:v for k,v in f['thresholds'].items() if k!='constrained3'},sf)
        if f['thresholds']['constrained3']!=ref['thresholds']['constrained3']: raise ValueError('Fold candidate cutoff changed')
        n+=check_evaluation(f['fit_evaluation'],sf,f['thresholds']); n+=check_evaluation(f['held_evaluation'],h,f['thresholds'])
        methods=f['held_evaluation']['all']['methods']; a=methods['constrained3']; b=methods['control3']
        for m in totals: totals[m]+=methods[m]['offline_correct95']['states']['FA']
        if a['delivery']['correct_retention']<.95: failures.append(seq+': transferred correct retention <95%')
        if a['offline_correct95']['states']['FA']>b['offline_correct95']['states']['FA']: failures.append(seq+': same-correct95 FA worse')
        for c in ('control3','full_simple','score_only'):
            if a['delivery']['states']['FA']>a['controls'][c]['same_count']['states']['FA']: failures.append(seq+': same-count FA worse than '+c)
            sc=a['controls'][c]['same_CR']
            if a['delivery']['states']['FA']>sc['states']['FA']: failures.append(seq+': same-CR FA worse than '+c)
            if a['delivery']['longest_correct_FR']>sc['longest_correct_FR']: failures.append(seq+': same-CR continuous FR worse than '+c)
    if totals['constrained3']>=totals['control3']: failures.append('No strict pooled per-video correct95 FA improvement')
    if totals['control3']!=54 or result['gate']['failures']!=failures or result['gate']['per_video_correct95_FA']!=totals or result['gate']['passed']!=(not failures):
        raise ValueError('Pre-registered gate differs')
    expected='TRAIN_GATE_PASSED_PENDING_VAL' if not failures else 'TRAIN_GATE_FAILED_STOP'
    if result['status']!=expected: raise ValueError('Wrong stopping decision')
    for method,groups in result['pooled_transfer_delivery'].items():
        keep={r['image'] for r in held if r['pred'] is not None and
              (r['risks'][method]<0 if method=='constrained3' else r['risks'][method]<=result['folds'][r['sequence']]['thresholds'][method]['risk_le'])}
        for g,stats in groups.items():
            part=held if g=='all' else [r for r in held if r[g.split(':')[0]]==g.split(':')[1]]
            old.check_summary(stats,part,keep); n+=1
    return dict(passed=True,rows=len(original),independent_checks=n,
        report_sha256=runner.parent.io.sha(out/'report.json'),absent_frontend=absent,
        candidate_gate_passed=not failures,VAL_opened=False,TEST_opened=False,
        checked=['analytic KKT certificate','fit-only video constraints','empirical FR bound','balanced control',
                 'zero-cutoff and transferred cutoff','independent logit','four states/coverage/FR runs',
                 'pairwise AUC','whole ties and matched count/CR','pre-registered gate'])

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--out',type=Path,required=True)
    a.add_argument('--allow-absent-frontend',action='store_true'); a.add_argument('--receipt',default='server_review.json')
    args=a.parse_args(); v=review(args.out,args.allow_absent_frontend)
    runner.parent.io.write(args.out/args.receipt,v); print(v)
