#!/usr/bin/env python3
"""Independent fit membership, actual counts/margins and score/metric review."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_actual_fr_v1 as runner
from crane_project.tools import review_port_reliability_video_transfer_v1 as old
from crane_project.tools import review_port_reliability_fr_constraint_v1 as previous_review


def check_models(models,fitting,previous):
    rows=[r for r in fitting if r['pred'] is not None]
    x=np.array([old.descriptor(r,3) for r in rows]); y=np.array([old.bad(r) for r in rows],dtype=bool)
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
    d=np.column_stack((np.ones(len(rows)),(x-mean)/scale))
    if models['control3']!=previous['models']['control3']:raise ValueError('Reviewed control weights modified')
    for name,m in models.items():
        if not np.allclose(m['mean'],mean,atol=1e-12,rtol=0) or not np.allclose(m['scale'],scale,atol=1e-12,rtol=0):raise ValueError('Fit-only normalization changed')
        if m['train_rows']!=len(rows) or m['bad_rows']!=int(y.sum()) or m['good_rows']!=int((~y).sum()):raise ValueError('Fit class count changed')
    m=models['actual_count3'];w=np.array(m['weights']);z=d@w;reject=z>=0
    if not np.isfinite(w).all() or max(abs(w))>10.+1e-7 or min(abs(z))<1e-4-1e-7:raise ValueError('Fixed coefficient bound/sign gap violated')
    if m['fit_rejections']!={r['image']:bool(zz) for r,zz in zip(rows,reject)}:raise ValueError('Actual vs binary rejection differs')
    seqs=sorted({r['sequence'] for r in rows});by={}
    for seq in seqs:
        mask=np.array([r['sequence']==seq for r in rows])&~y
        n=int(mask.sum());fr=int(reject[mask].sum());budget=n//20
        if fr>budget:raise ValueError('Actual FR budget violated')
        by[seq]=dict(correct=n,max_FR=budget,actual_FR=fr)
    expected=dict(FA=int((y&~reject).sum()),ED=int((y&reject).sum()),
        FR=int((~y&reject).sum()),CR=int((~y&~reject).sum()),per_video=by,
        minimum_signed_margin=float(min(abs(z))),max_abs_coefficient=float(max(abs(w))),actual_binary_signs_verified=True)
    for k,v in expected.items():
        a=m['certificate'][k]
        if isinstance(v,float):
            if abs(v-a)>1e-10:raise ValueError('Solution certificate float differs')
        elif v!=a:raise ValueError('Solution certificate counts differ')
    if m['fitting_videos']!=seqs or m['NP_statistical_guarantee'] or m['unrestricted_capacity_claim']:raise ValueError('Unsupported scope claim')
    solver=m['solver'];upper=expected['FA'];dual=solver['raw_dual_bound']
    lower=max(0,math.ceil(int(y.sum())+dual-1e-6)) if dual is not None else 0
    if not 0<=lower<=upper or solver['FA_upper']!=upper or solver['FA_lower']!=lower or solver['absolute_FA_gap']!=upper-lower:raise ValueError('Invalid objective bounds')
    opt=bool(solver['incumbent'] and solver['status']==0 and upper==lower)
    if solver['optimal_within_fixed_domain']!=opt:raise ValueError('Unsupported optimality claim')
    outcome='OPTIMAL_WITHIN_FIXED_DOMAIN' if opt else 'FEASIBLE_BUDGET_UNRESOLVED' if solver['incumbent'] else 'NO_SOLVER_INCUMBENT_ALL_ACCEPT_REFERENCE'
    if solver['outcome']!=outcome or solver['options']!={'time_limit':60.,'node_limit':100000,'mip_rel_gap':0.,'presolve':True}:raise ValueError('Fixed solver contract changed')
    if not solver['incumbent'] and (expected['ED'] or expected['FR']):raise ValueError('No-incumbent reference misrepresented')
    return 2

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
                  (r['risks'][method]<0 if method=='actual_count3' else r['risks'][method]<=thresholds[method]['risk_le'])}
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
    p,pins,absent,original,policy,old_result=runner.load(allow_absent)
    report=json.loads((out/'report.json').read_text());result=report['result']
    if report['input_pins']!=pins or report['settings']!=runner.core.SETTINGS or report['frames']!=len(original):raise ValueError('Report provenance differs')
    if result['candidate_solver_calls']!=7 or result['control_refits']!=0 or any(result[k] for k in ('VAL_opened','TEST_opened','candidate_adopted','GPU_used','detector_inferences')):raise ValueError('Scope changed')
    if any(report[k] for k in ('formal_policy_changed','geometry_changed','output_identity_changed')):raise ValueError('Formal prediction changed')
    held=runner.parent.io.read(out/'held_TRAIN.jsonl.gz');refrows=runner.parent.io.read(out/'reference_TRAIN.jsonl.gz')
    if len(held)!=len(original) or len(refrows)!=len(original):raise ValueError('Output count changed')
    n=0;ref=result['reference'];n+=check_models(ref['models'],original,old_result['reference'])
    n+=previous_review.check_risks(refrows,original,ref['models'])
    if ref['thresholds']['actual_count3']!={'logit_lt':0.,'rule':'fixed_zero_no_posthoc_tuning'}:raise ValueError('Candidate cutoff tuning')
    for k,v in old_result['reference']['thresholds'].items():
        if k!='constrained3' and ref['thresholds'][k]!=v:raise ValueError('Control cutoff changed')
    old.check_thresholds({k:v for k,v in ref['thresholds'].items() if k!='actual_count3'},refrows)
    n+=check_evaluation(ref['fit_evaluation'],refrows,ref['thresholds']);n+=check_evaluation(ref['held_evaluation'],refrows,ref['thresholds'])
    seqs=sorted({r['sequence'] for r in original})
    if set(result['folds'])!=set(seqs):raise ValueError('Wrong fold set')
    failures=[];totals={'control3':0,'actual_count3':0}
    contexts=[ref]+list(result['folds'].values())
    if any(not x['models']['actual_count3']['solver']['incumbent'] for x in contexts):failures.append('At least one fit has no solver candidate')
    a=ref['fit_evaluation']['all']['methods']['actual_count3']
    if a['delivery']['states']['FA']>=a['controls']['control3']['same_CR']['states']['FA']:failures.append('Full TRAIN candidate has no strict same-CR FA improvement')
    for seq,f in result['folds'].items():
        fitting=[r for r in original if r['sequence']!=seq];origheld=[r for r in original if r['sequence']==seq];h=[r for r in held if r['sequence']==seq]
        if f['fit_images']!=[r['image'] for r in fitting] or f['held_images']!=[r['image'] for r in origheld]:raise ValueError('Held video entered fit')
        if any(r['diagnostic_fold']!=seq for r in h):raise ValueError('Wrong fold ID')
        n+=check_models(f['models'],fitting,old_result['folds'][seq]);n+=previous_review.check_risks(h,origheld,f['models'])
        sf=deepcopy(fitting)
        if set(f['fit_risks'])!={r['image'] for r in sf}:raise ValueError('Wrong fit scores')
        for r in sf:r['risks']=f['fit_risks'][r['image']]
        n+=previous_review.check_risks(sf,fitting,f['models'])
        old.check_thresholds({k:v for k,v in f['thresholds'].items() if k!='actual_count3'},sf)
        if f['thresholds']['actual_count3']!=ref['thresholds']['actual_count3']:raise ValueError('Fold candidate threshold modified')
        n+=check_evaluation(f['fit_evaluation'],sf,f['thresholds']);n+=check_evaluation(f['held_evaluation'],h,f['thresholds'])
        methods=f['held_evaluation']['all']['methods'];a=methods['actual_count3'];b=methods['control3']
        for m in totals:totals[m]+=methods[m]['offline_correct95']['states']['FA']
        if a['delivery']['correct_retention']<.95:failures.append(seq+': transferred correct retention <95%')
        if a['offline_correct95']['states']['FA']>b['offline_correct95']['states']['FA']:failures.append(seq+': same-correct95 FA worse')
        for c in ('control3','full_simple','score_only'):
            if a['delivery']['states']['FA']>a['controls'][c]['same_count']['states']['FA']:failures.append(seq+': same-count FA worse than '+c)
            sc=a['controls'][c]['same_CR']
            if a['delivery']['states']['FA']>sc['states']['FA']:failures.append(seq+': same-CR FA worse than '+c)
            if a['delivery']['longest_correct_FR']>sc['longest_correct_FR']:failures.append(seq+': same-CR continuous FR worse than '+c)
    if totals['control3']!=54:raise ValueError('Held control54 not reproduced')
    if totals['actual_count3']>=totals['control3']:failures.append('No strict pooled per-video correct95 FA improvement')
    if result['gate']['failures']!=failures or result['gate']['passed']!=(not failures) or result['gate']['per_video_correct95_FA']!=totals:raise ValueError('Gate changed')
    if result['gate']['solver_all_optimal']!=all(x['models']['actual_count3']['solver']['optimal_within_fixed_domain'] for x in contexts):raise ValueError('Pooled solver status differs')
    expected='TRAIN_GATE_PASSED_PENDING_VAL' if not failures else 'TRAIN_GATE_FAILED_STOP'
    if result['status']!=expected:raise ValueError('Wrong stopping decision')
    for method,groups in result['pooled_transfer_delivery'].items():
        keep={r['image'] for r in held if r['pred'] is not None and
            (r['risks'][method]<0 if method=='actual_count3' else r['risks'][method]<=result['folds'][r['sequence']]['thresholds'][method]['risk_le'])}
        for g,actual in groups.items():
            part=held if g=='all' else [r for r in held if r[g.split(':')[0]]==g.split(':')[1]]
            old.check_summary(actual,part,keep);n+=1
    return dict(passed=True,rows=len(original),independent_checks=n,report_sha256=runner.parent.io.sha(out/'report.json'),
        absent_frontend=absent,candidate_gate_passed=not failures,VAL_opened=False,TEST_opened=False,
        solver_bound_is_reported_not_independently_reproved=True,
        checked=['fit exclusion/normalizers','actual FR integer counts','weight bound/margin and assignment signs',
          'saved solver objective bounds/status','reviewed controls unchanged','independent scores/FA/FR/ED/CR',
          'whole ties/same-count/same-CR/AUC/continuous FR','pre-registered gate'])

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--allow-absent-frontend',action='store_true')
    a.add_argument('--receipt',default='server_review.json');args=a.parse_args();v=review(args.out,args.allow_absent_frontend)
    runner.parent.io.write(args.out/args.receipt,v);print(v)
