"""Bounded empirical-count linear feasibility experiment. No NP guarantee."""
from copy import deepcopy
import math
import time
import numpy as np
from crane_project.utils import port_reliability_video_transfer_v1 as video
from crane_project.utils import port_reliability_fr_constraint_v1 as proxy

VERSION='port_reliability_actual_fr_v1'
SETTINGS=dict(correct_retention=.95,coefficient_bound=10.,classification_margin=1e-4,
    time_limit_seconds=60.,node_limit=100000,mip_relative_gap=0.,presolve=True,
    verification_tolerance=1e-7,integrality_tolerance=1e-6,
    arms=['control3','actual_count3'])


def build(rows):
    """Only fitting labels enter mixed-integer constraints."""
    part=[r for r in rows if r['pred'] is not None]
    if not part:raise ValueError('No fitting outputs')
    x=np.array([video.fusion.simple.descriptor(r['pred'],r['image_size']) for r in part])
    y=np.array([video.wrong(r) for r in part],dtype=bool)
    if not y.any() or y.all():raise ValueError('Both fitting classes required')
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
    d=np.column_stack((np.ones(len(x)),(x-mean)/scale))
    limits={}
    for seq in sorted({r['sequence'] for r in part}):
        ids=[i for i,r in enumerate(part) if r['sequence']==seq and not y[i]]
        if not ids:raise ValueError('Every fitting video needs correct support')
        # Exact integer floor avoids a floating ceil(.95*n) off-by-one.
        limits[seq]=dict(indices=ids,correct=len(ids),max_FR=len(ids)//20)
    big_m=SETTINGS['coefficient_bound']*np.abs(d).sum(1)+SETTINGS['classification_margin']
    return dict(rows=part,mean=mean,scale=scale,design=d,labels=y,limits=limits,big_m=big_m)


def validate_solution(data,w,z):
    """Independent deployment signs must equal all binary assignments."""
    w=np.asarray(w,dtype=float);z=np.asarray(z,dtype=float)
    if w.shape!=(4,) or z.shape!=(len(data['rows']),) or not np.isfinite(w).all() or not np.isfinite(z).all():
        raise ValueError('Invalid finite solution shape')
    tol=SETTINGS['verification_tolerance'];it=SETTINGS['integrality_tolerance']
    rounded=np.rint(z)
    if max(np.abs(w))>SETTINGS['coefficient_bound']+tol or max(abs(z-rounded))>it or not np.isin(rounded,[0,1]).all():
        raise ValueError('Weight bounds or integrality violated')
    score=data['design']@w;reject=score>=0;binary=rounded.astype(bool)
    margin=SETTINGS['classification_margin']
    signed=np.where(binary,score,-score)
    if not np.array_equal(reject,binary) or min(signed)<margin-tol:
        raise ValueError('Binary assignments differ from actual decision/margin')
    group={}
    for seq,v in data['limits'].items():
        fr=int(reject[v['indices']].sum())
        if fr>v['max_FR']:raise ValueError('Actual per-video FR budget violated')
        group[seq]=dict(correct=v['correct'],max_FR=v['max_FR'],actual_FR=fr)
    bad=data['labels'];fa=int((bad&~reject).sum());ed=int((bad&reject).sum())
    return dict(FA=fa,ED=ed,FR=int((~bad&reject).sum()),CR=int((~bad&~reject).sum()),
        per_video=group,minimum_signed_margin=float(min(signed)),
        max_abs_coefficient=float(max(abs(w))),actual_binary_signs_verified=True)


def solve(rows):
    # Lazy dependency: local numeric review/tests do not require SciPy.
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import csr_matrix,diags,bmat
    import scipy
    data=build(rows);d=data['design'];n=len(d);y=data['labels'];m=data['big_m'];eps=SETTINGS['classification_margin']
    # z=1 rejects; both inequalities enforce sign with a symmetric gap.
    # s-M*z <= -eps; s-M*z >= eps-M.
    matrix= bmat([[csr_matrix(d),-diags(m)]],format='csr')
    gr=np.zeros((len(data['limits']),4+n));upp=[]
    for j,v in enumerate(data['limits'].values()):
        gr[j,4+np.array(v['indices'])]=1.;upp.append(v['max_FR'])
    constraints=[LinearConstraint(matrix,eps-m,np.full(n,-eps)),
                 LinearConstraint(csr_matrix(gr),np.zeros(len(upp)),np.array(upp))]
    objective=np.r_[np.zeros(4),-y.astype(float)] # min -ED == min FA
    options=dict(time_limit=SETTINGS['time_limit_seconds'],node_limit=SETTINGS['node_limit'],
        mip_rel_gap=SETTINGS['mip_relative_gap'],presolve=SETTINGS['presolve'])
    started=time.monotonic()
    result=milp(c=objective,integrality=np.r_[np.zeros(4),np.ones(n)],
        bounds=Bounds(np.r_[np.full(4,-SETTINGS['coefficient_bound']),np.zeros(n)],
                      np.r_[np.full(4,SETTINGS['coefficient_bound']),np.ones(n)]),
        constraints=constraints,options=options)
    elapsed=time.monotonic()-started
    incumbent=result.x is not None
    if incumbent:
        w=result.x[:4];z=result.x[4:]
    else:
        # Known feasible reference, never mislabeled as a solver-trained candidate.
        w=np.array([-eps,0.,0.,0.]);z=np.zeros(n)
    cert=validate_solution(data,w,z)
    raw_dual=getattr(result,'mip_dual_bound',None)
    dual=float(raw_dual) if raw_dual is not None and np.isfinite(raw_dual) else None
    lower=max(0,int(math.ceil(int(y.sum())+dual-1e-6))) if dual is not None else 0
    if lower>cert['FA']:raise ValueError('Invalid solver lower bound')
    if incumbent and abs(float(result.fun)+cert['ED'])>1e-6:raise ValueError('Objective differs from actual ED')
    optimal=bool(incumbent and result.status==0 and cert['FA']==lower)
    outcome='OPTIMAL_WITHIN_FIXED_DOMAIN' if optimal else 'FEASIBLE_BUDGET_UNRESOLVED' if incumbent else 'NO_SOLVER_INCUMBENT_ALL_ACCEPT_REFERENCE'
    return dict(feature_names=list(video.fusion.simple.FEATURES),mean=data['mean'].tolist(),scale=data['scale'].tolist(),
        weights=w.tolist(),train_rows=n,bad_rows=int(y.sum()),good_rows=int((~y).sum()),
        fitting_videos=list(data['limits']),certificate=cert,
        fit_rejections={r['image']:bool(v) for r,v in zip(data['rows'],np.rint(z))},
        solver=dict(outcome=outcome,status=int(result.status),message=str(result.message),
            incumbent=incumbent,optimal_within_fixed_domain=optimal,
            FA_upper=cert['FA'],FA_lower=lower,absolute_FA_gap=cert['FA']-lower,
            raw_dual_bound=dual,node_count=int(getattr(result,'mip_node_count',0) or 0),
            elapsed_seconds=elapsed,scipy_version=scipy.__version__,options=options),
        decision='accept iff logit < 0; logit==0 rejected',
        unrestricted_capacity_claim=False,NP_statistical_guarantee=False)


def score(rows,models):
    values=deepcopy(rows)
    for r in values:
        r['risks']['control3']=proxy.logit(models['control3'],r)
        r['risks']['actual_count3']=proxy.logit(models['actual_count3'],r)
    return values


def accepted(rows,method,thresholds):
    return {r['image'] for r in rows if r['pred'] is not None and
        (r['risks'][method]<0 if method=='actual_count3' else r['risks'][method]<=thresholds[method]['risk_le'])}


def evaluate(rows,thresholds):
    result={}
    for g,part in video.fusion.level.metrics.grouped(rows).items():
        data=dict(formal=video.counts(part,video.fusion.level.formal_accept(part)),methods={})
        required=math.ceil(.95*sum(video.wrong(r) is False for r in part))
        for m in ('control3','actual_count3','full_simple','score_only'):
            delivery=video.counts(part,accepted(part,m,thresholds));controls={}
            if m in SETTINGS['arms']:
                for c in ('control3','full_simple','score_only'):
                    controls[c]=dict(same_count=video.point(part,c,count=delivery['accepted']),
                                    same_CR=video.point(part,c,correct=delivery['states']['CR']))
            data['methods'][m]=dict(delivery=delivery,controls=controls,
                AUROC=video.fusion.level.metrics.auc(part,m),
                offline_correct95=video.point(part,m,correct=required))
        result[g]=data
    return result


def context(fitting,held,previous,progress):
    print('MILP begin',progress,'fitting',len(fitting),'held',len(held),flush=True)
    candidate=solve(fitting);models=dict(control3=deepcopy(previous['models']['control3']),actual_count3=candidate)
    thresholds={k:deepcopy(v) for k,v in previous['thresholds'].items() if k!='constrained3'}
    thresholds['actual_count3']=dict(logit_lt=0.,rule='fixed_zero_no_posthoc_tuning')
    sf=score(fitting,models);sh=score(held,models)
    # Exact normalizer and control cutoff replay; no old control optimizer rerun.
    for k in ('mean','scale'):
        if not np.allclose(candidate[k],models['control3'][k],atol=1e-12,rtol=0):raise ValueError('Control/candidate fitting membership mismatch')
    replay=video.cutoff(sf,'control3')
    if replay!=thresholds['control3']:raise ValueError('Old fitting cutoff not reproduced')
    print('MILP finish',progress,candidate['solver'],candidate['certificate'],flush=True)
    return dict(models=models,thresholds=thresholds,fit_images=[r['image'] for r in fitting],
        held_images=[r['image'] for r in held],fit_risks={r['image']:r['risks'] for r in sf},
        fit_evaluation=evaluate(sf,thresholds),held_evaluation=evaluate(sh,thresholds)),sf,sh


def gate(result):
    failures=[];totals=dict(control3=0,actual_count3=0)
    contexts=[result['reference']]+list(result['folds'].values())
    if any(not x['models']['actual_count3']['solver']['incumbent'] for x in contexts):failures.append('At least one fit has no solver candidate')
    ref=result['reference']['fit_evaluation']['all']['methods'];cand=ref['actual_count3'];ctrl=ref['control3']
    if cand['delivery']['states']['FA']>=cand['controls']['control3']['same_CR']['states']['FA']:
        failures.append('Full TRAIN candidate has no strict same-CR FA improvement')
    for seq,f in result['folds'].items():
        methods=f['held_evaluation']['all']['methods'];a=methods['actual_count3'];b=methods['control3']
        for m in totals:totals[m]+=methods[m]['offline_correct95']['states']['FA']
        if a['delivery']['correct_retention']<.95:failures.append(seq+': transferred correct retention <95%')
        if a['offline_correct95']['states']['FA']>b['offline_correct95']['states']['FA']:failures.append(seq+': same-correct95 FA worse')
        for c in ('control3','full_simple','score_only'):
            if a['delivery']['states']['FA']>a['controls'][c]['same_count']['states']['FA']:failures.append(seq+': same-count FA worse than '+c)
            sc=a['controls'][c]['same_CR']
            if a['delivery']['states']['FA']>sc['states']['FA']:failures.append(seq+': same-CR FA worse than '+c)
            if a['delivery']['longest_correct_FR']>sc['longest_correct_FR']:failures.append(seq+': same-CR continuous FR worse than '+c)
    if totals['control3']!=54:raise ValueError('Reviewed leave-out control54 not reproduced')
    if totals['actual_count3']>=totals['control3']:failures.append('No strict pooled per-video correct95 FA improvement')
    return dict(passed=not failures,failures=failures,per_video_correct95_FA=totals,
        solver_all_optimal=all(x['models']['actual_count3']['solver']['optimal_within_fixed_domain'] for x in contexts),
        statistical_guarantee=False)


def audit(rows,old):
    video.support.validate(rows);before=deepcopy(rows);seqs=sorted({r['sequence'] for r in rows})
    if len(seqs)!=6 or set(old['folds'])!=set(seqs):raise ValueError('Reviewed six videos required')
    ref,sf,reference=context(rows,rows,old['reference'],'full_TRAIN')
    result=dict(reference=ref,folds={});held=[]
    for seq in seqs:
        fitting=[r for r in rows if r['sequence']!=seq];testing=[r for r in rows if r['sequence']==seq]
        previous=old['folds'][seq]
        if previous['fit_images']!=[r['image'] for r in fitting] or previous['held_images']!=[r['image'] for r in testing]:raise ValueError('Prior fold binding differs')
        fold,_,sh=context(fitting,testing,previous,seq);result['folds'][seq]=fold
        for r in sh:r['diagnostic_fold']=seq
        held.extend(sh)
    held.sort(key=lambda r:(r['sequence'],r['frame_id'],r['image']))
    result['gate']=gate(result)
    result.update(status='TRAIN_GATE_PASSED_PENDING_VAL' if result['gate']['passed'] else 'TRAIN_GATE_FAILED_STOP',
        candidate_solver_calls=7,control_refits=0,controls_reused_from_reviewed_previous=True,
        VAL_opened=False,TEST_opened=False,candidate_adopted=False,detector_inferences=0,GPU_used=False)
    result['pooled_transfer_delivery']={m:{g:video.counts(part,{r['image'] for r in held if r['image'] in accepted([r],m,result['folds'][r['sequence']]['thresholds'])})
        for g,part in video.fusion.level.metrics.grouped(held).items()} for m in ('control3','actual_count3','full_simple','score_only')}
    if rows!=before:raise ValueError('Input mutation')
    return result,held,reference
