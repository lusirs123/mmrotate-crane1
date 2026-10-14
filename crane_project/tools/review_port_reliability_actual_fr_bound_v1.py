#!/usr/bin/env python3
"""Reconstruct scores/counts and verify objective cut without fitting/solving."""
import argparse
import gzip
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from crane_project.tools import run_port_reliability_actual_fr_bound_v1 as run
from crane_project.tools import review_port_reliability_video_transfer_v1 as independent


def check_model(model, rows):
    part=[r for r in rows if r['pred'] is not None]
    x=np.array([independent.descriptor(r,3) for r in part]);mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
    if not np.allclose(model['mean'],mean,atol=1e-12,rtol=0) or not np.allclose(model['scale'],scale,atol=1e-12,rtol=0):
        raise ValueError('Normalizer differs from complete fitting TRAIN')
    w=np.array(model['weights']);s=w[0]+((x-mean)/scale)@w[1:]
    bad=np.array([independent.bad(r) for r in part]);reject=s>=0
    if not np.isfinite(w).all() or max(abs(w))>10.+1e-7 or min(abs(s))<1e-4-1e-7:
        raise ValueError('Coefficient bound or sign margin violated')
    cert=model['certificate']
    counts=dict(FA=int((bad&~reject).sum()),FR=int((~bad&reject).sum()),
                ED=int((bad&reject).sum()),CR=int((~bad&~reject).sum()))
    if counts['FA']>59 or any(cert[k]!=v for k,v in counts.items()):
        raise ValueError('Actual counts or FA59 constraint violated')
    expected={}
    for seq in sorted({r['sequence'] for r in part}):
        mask=np.array([r['sequence']==seq for r in part])&~bad
        good=int(mask.sum());fr=int(reject[mask].sum())
        if fr>good//20:raise ValueError('Per-video actual FR violated')
        expected[seq]=dict(correct=good,max_FR=good//20,actual_FR=fr)
    if cert['per_video']!=expected or model['fit_rejections']!={r['image']:bool(v) for r,v in zip(part,reject)}:
        raise ValueError('Certificate actual membership differs')
    if abs(cert['minimum_signed_margin']-float(min(abs(s))))>1e-10 or abs(cert['max_abs_coefficient']-max(abs(w)))>1e-10:
        raise ValueError('Certificate numeric values differ')
    if model['train_rows']!=len(part) or model['bad_rows']!=int(bad.sum()) or model['good_rows']!=int((~bad).sum()):
        raise ValueError('Training support differs')
    return len(part)


def check_evaluation(rows, frozen, saved, curves):
    scored=run.val.apply(rows,frozen);checks=0
    for original,row in zip(rows,scored):
        if any(row[k]!=v for k,v in original.items() if k!='risks'):
            raise ValueError('Original box/flags/output identity changed')
        if row['pred'] is not None:
            model=frozen['candidate_model'];x=independent.descriptor(row,3)
            z=model['weights'][0]+sum((a-m)/s*w for a,m,s,w in zip(x,model['mean'],model['scale'],model['weights'][1:]))
            if abs(z-row['risks']['actual_count_logit'])>1e-12:raise ValueError('Independent logit differs')
        checks+=1
    groups={'all':scored}
    for key in ('domain','sequence'):
        for value in sorted({r[key] for r in scored}):groups[key+':'+value]=[r for r in scored if r[key]==value]
    if set(saved['groups'])!=set(groups) or set(curves)!=set(groups) or saved['frames']!=len(rows):
        raise ValueError('Reporting scope differs')
    for name,part in groups.items():
        data=saved['groups'][name]
        for method in run.val.METHODS:
            delivery=data['methods'][method]['delivery']
            keep={r['image'] for r in part if r['supplement_flags'][method]}
            independent.check_summary(delivery,part,keep)
            for risk in run.val.RISKS:
                value=data['methods'][method]['controls'][risk]
                independent.check_point(value['same_count'],part,risk,count=delivery['accepted'])
                independent.check_point(value['same_CR'],part,risk,correct=delivery['states']['CR'])
                checks+=2
        for risk in run.val.RISKS:
            good=np.array([r['risks'][risk] for r in part if independent.bad(r) is False])
            bad=np.array([r['risks'][risk] for r in part if independent.bad(r) is True])
            auc=float(((bad[:,None]>good[None])+.5*(bad[:,None]==good[None])).mean()) if len(good) and len(bad) else None
            if auc!=data['AUROC'][risk]:raise ValueError('AUC differs')
            independent.check_point(data['offline_correct95'][risk],part,risk,correct=math.ceil(.95*len(good)))
            thresholds=sorted({r['risks'][risk] for r in part if r['pred'] is not None})
            points=curves[name][risk]
            if len(points)!=len(thresholds)+1 or [p['threshold'] for p in points]!=[None]+thresholds:
                raise ValueError('Curve missing whole-tie point')
            for point in points:
                keep={r['image'] for r in part if r['pred'] is not None and point['threshold'] is not None and r['risks'][risk]<=point['threshold']}
                independent.check_summary(point,part,keep);checks+=1
        outputs=[r for r in part if r['pred'] is not None]
        hits=sum(math.hypot(r['pred'][0]-r['gt'][0],r['pred'][1]-r['gt'][1])<15 for r in outputs)
        center=dict(output_frames=len(outputs),correct_output_frames=hits,hit_rate_outputs=hits/len(outputs) if outputs else None,
            output_coverage=len(outputs)/len(part),full_frame_correct_coverage=hits/len(part))
        if data['center']!=center or not data['center_angle_flags_boxes_scores_output_identity_unchanged']:
            raise ValueError('Center coverage/flag preservation differs')
    return checks


def review(out, allow_absent=False):
    contract,pins,absent,train,witness,base=run.load(allow_absent)
    r=run.val.read(out/'report.json');solved=r['solving'];m=solved['model'];solver=solved['solver']
    if r['input_pins']!=pins or r['settings']!=run.core.SETTINGS or r['protocol']!=run.core.VERSION:
        raise ValueError('Source/contract changed')
    if r['solver_calls']!=1 or r['control_refits']!=0 or any(r[k] for k in ('threshold_updates','TEST_opened','GPU_used','detector_inferences','candidate_adopted','formal_policy_changed')):
        raise ValueError('Scope violation')
    if solved['known_control_weights']!=witness['weights'] or solved['known_control_certificate']!=witness['certificate'] or solved['known_control_is_candidate']:
        raise ValueError('Control misrepresented')
    if solver['options']!=dict(time_limit=60.,node_limit=100000,mip_rel_gap=0.,presolve=True) or solver['warm_start_supplied'] or solver['objective_cut_FA_le']!=59:
        raise ValueError('Solver contract changed')
    lower=max(0,math.ceil(71+solver['raw_dual_bound']-1e-6)) if solver['raw_dual_bound'] is not None else 0
    if solver['FA_lower']!=lower or not 0<=lower<=59 or solver['known_feasible_FA_upper']!=59:
        raise ValueError('Invalid reported objective bound')
    candidate=m is not None;optimal=bool(candidate and solver['status']==0 and lower==m['certificate']['FA'])
    expected='OPTIMAL_WITHIN_FIXED_DOMAIN' if optimal else 'FEASIBLE_BUDGET_UNRESOLVED' if candidate else 'NO_SOLVER_CANDIDATE'
    if solver['incumbent']!=candidate or solver['optimal_within_fixed_domain']!=optimal or solver['outcome']!=expected:
        raise ValueError('Solver status differs')
    if r['VAL_opened']!=candidate or r['training_objective_improved']!=bool(candidate and m['certificate']['FA']<59):
        raise ValueError('Training/VAL state differs')
    frozen=run.val.read(out/'frozen_model_before_VAL.json');curves=run.val.read(out/'risk_correct_retention_curves.json');checks=0;total=len(train)
    if candidate:
        checks+=check_model(m,train)
        if m['solver']!=solver or solver['FA_upper']!=m['certificate']['FA'] or solver['absolute_FA_gap']!=solver['FA_upper']-lower:
            raise ValueError('Saved solver/candidate bound differs')
        expected_frozen=dict(base,candidate_model=m,original_solver_status=solver)
        if frozen!=expected_frozen or r['frozen_workpoints']!=frozen:
            raise ValueError('Model/cutoff modified after fitting')
        with gzip.open(ROOT/(run.val.VAL+'scored_VAL.jsonl.gz'),'rt') as stream:rows=[run.val.json.loads(line) for line in stream]
        if len(rows)!=887 or any(x['split']!='val' or x['reliability_role']!='val' for x in rows):raise ValueError('Wrong VAL')
        for role,part in [('TRAIN',train),('VAL',rows)]:checks+=check_evaluation(part,frozen,r['evaluations'][role],curves[role])
        total+=len(rows)
    elif frozen is not None or r['frozen_workpoints'] is not None or r['evaluations'] or curves or solver['FA_upper'] is not None:
        raise ValueError('No-incumbent control labeled candidate')
    status='VALID_CANDIDATE_COMPLETE_VAL_REPORT' if candidate else 'NO_NEW_CANDIDATE_STOP'
    if r['status']!=status:raise ValueError('Wrong stopping state')
    return dict(passed=True,rows=total,independent_checks=checks,absent_frontend=absent,
        report_sha256=run.val.sha(out/'report.json'),VAL_opened=candidate,TEST_opened=False,
        solver_bound_reported_not_branch_and_bound_reproved=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-absent-frontend',action='store_true')
    p.add_argument('--receipt',default='server_review.json');a=p.parse_args()
    result=review(a.out,a.allow_absent_frontend);run.val.write(a.out/a.receipt,result);print(result)
