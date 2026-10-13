"""Fixed linear size-risk objective comparison; no GT in online API.

Smooth upper-tail pairwise logistic surrogate, not the PAUCI paper algorithm.
Wrong sizes are positives. The highest-risk 5% of correct sizes form the
one-way partial-AUC region. Smoothing biases the tail objective; no guarantee.
"""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_reliability_temporal_fusion_v1 as base

VERSION = 'port_reliability_partial_auc_v1'
METHOD = 'partial_auc4'
CONTROLS = ('fusion4', 'full_simple', 'score_only', 'static3')
SETTINGS = dict(beta=.05, temperature=.1, l2=.1, max_iterations=100,
                gradient_tolerance=1e-8, initialization=0., correct_retention=.95)


def objective(v, good, wrong):
    """Return value/gradient/Hessian for 4 slopes and offline tail threshold.

    h_j=mean_i softplus(w.(good_j-wrong_i));
    J=tau+T/(beta*N_good)*sum softplus((h_j-tau)/T)+l2/2*||w||².
    Intercept cancels in ranking, so retain the frozen BCE intercept online.
    """
    v=np.asarray(v,dtype=float);good=np.asarray(good);wrong=np.asarray(wrong)
    if v.shape!=(5,) or good.ndim!=2 or wrong.ndim!=2 or good.shape[1]!=4 or wrong.shape[1]!=4 or not len(good) or not len(wrong):
        raise ValueError('Two genuine classes and four features required')
    if not all(np.isfinite(a).all() for a in (v,good,wrong)):raise ValueError('Nonfinite objective input')
    d=good[:,None,:]-wrong[None,:,:]
    u=d@v[:4];prob=base.simple.sigmoid(u)
    h=np.logaddexp(0.,u).mean(axis=1)
    jac=np.mean(prob[:,:,None]*d,axis=1)
    hes=np.einsum('gi,gij,gik->gjk',prob*(1-prob),d,d)/len(wrong)
    t=SETTINGS['temperature'];beta=SETTINGS['beta'];lam=SETTINGS['l2']
    a=(h-v[4])/t;p=base.simple.sigmoid(a)
    loss=float(v[4]+t*np.logaddexp(0.,a).mean()/beta+.5*lam*v[:4].dot(v[:4]))
    grad=np.r_[np.mean(p[:,None]*jac,axis=0)/beta+lam*v[:4],1-p.mean()/beta]
    outer=np.column_stack((jac,-np.ones(len(good))))
    H=np.einsum('g,gi,gj->ij',p*(1-p)/(beta*t*len(good)),outer,outer)
    H[:4,:4]+=np.einsum('g,gij->ij',p/(beta*len(good)),hes)+lam*np.eye(4)
    return loss,grad,H


def fit(train, models):
    if not train or any(r['reliability_role']!='train' or r['split'] not in ('train','train_sim') for r in train):
        raise ValueError('Complete TRAIN only')
    rows=[r for r in train if r['pred'] is not None and r['temporal']['available']]
    m=models['fusion4']
    if [r['image'] for r in rows]!=m['train_images']:raise ValueError('Paired TRAIN identities differ')
    x=np.array([base.features(r['pred'],r['image_size'],r['temporal'],4) for r in rows])
    z=(x-m['mean'])/m['scale'];y=np.array([int(base.level.metrics.bad(r)) for r in rows])
    good,wrong=z[y==0],z[y==1]
    if not len(good) or not len(wrong):raise ValueError('Both genuine classes required')
    v=np.r_[np.zeros(4),math.log(2)+SETTINGS['temperature']*math.log((1-SETTINGS['beta'])/SETTINGS['beta'])]
    initial=objective(v,good,wrong)[0];trace=[]
    for step in range(SETTINGS['max_iterations']):
        loss,g,H=objective(v,good,wrong)
        trace.append(dict(iteration=step,objective=loss,max_abs_gradient=float(abs(g).max())))
        if abs(g).max()<=SETTINGS['gradient_tolerance']:break
        delta=np.linalg.solve(H,g)
        for backtrack in range(30):
            trial=v-.5**backtrack*delta
            if objective(trial,good,wrong)[0]<=loss-1e-4*.5**backtrack*g.dot(delta):
                v=trial;break
        else:raise RuntimeError('Fixed line search failed')
    final,g,H=objective(v,good,wrong)
    if abs(g).max()>SETTINGS['gradient_tolerance']:raise RuntimeError('Fixed budget exhausted')
    return dict(weights=[m['weights'][0]]+v[:4].tolist(),tail_threshold=float(v[4]),
        mean=deepcopy(m['mean']),scale=deepcopy(m['scale']),feature_names=deepcopy(m['feature_names']),
        train_images=[r['image'] for r in rows],train_rows=len(rows),bad_rows=int(y.sum()),good_rows=int((y==0).sum()),
        initial_objective=initial,final_objective=final,max_abs_gradient=float(abs(g).max()),
        iterations=step+1,trace=trace,settings=deepcopy(SETTINGS),
        online_trainable_slopes=4,offline_nuisance_parameters=1,
        intercept_frozen_for_shared_fallback=True,objective_is_biased_smooth_surrogate=True)


def online_risk(pred,image_size,point,models,model):
    if pred is None:return None
    if not point['available']:return base.online_risks(pred,image_size,point,models)['fusion4']
    return float(base.simple.linear_risk(model,np.array([base.features(pred,image_size,point,4)]))[0])


class CausalPartialAUC:
    def __init__(self,models,model,cutoff,original_policy):
        if not 0<=cutoff<=1:raise ValueError('Invalid cutoff')
        self.models=deepcopy(models);self.model=deepcopy(model);self.cutoff=cutoff
        self.original=base.simple.SimpleComponentReliability(original_policy)
        self.tracker=base.level.CausalSizeEvidence()

    def decide(self,pred,image_size,frame_id,reset=False):
        result=self.original.decide(pred,image_size)
        point=self.tracker.observe(pred,frame_id,image_size,reset)
        risk=online_risk(pred,image_size,point,self.models,self.model)
        result['size_accepted']=risk is not None and risk<=self.cutoff
        result['size_partial_auc_risk']=risk
        result['size_partial_auc_history_available']=point['available']
        result['size_partial_auc_version']=VERSION
        return result


def score(rows,models,model):
    result=deepcopy(rows)
    for r in result:
        old=base.online_risks(r['pred'],r['image_size'],r['temporal'],models)['fusion4']
        saved=r['risks']['fusion4']
        if (old is None)!=(saved is None) or (old is not None and abs(old-saved)>1e-12):raise ValueError('Frozen base identity differs')
        r['risks'][METHOD]=online_risk(r['pred'],r['image_size'],r['temporal'],models,model)
    return result


def partial_auc(rows,method):
    """Normalized exact empirical trapezoidal ROC area on FPR [0,.05].
    Equal scores are one group; ties linearly interpolate, no image tie-break.
    """
    rows=[r for r in rows if r['pred'] is not None]
    good=sum(base.level.metrics.bad(r) is False for r in rows);wrong=len(rows)-good
    if not good or not wrong:return None
    grouped={}
    for r in rows:
        pair=grouped.setdefault(r['risks'][method],[0,0]);pair[int(base.level.metrics.bad(r))]+=1
    x=y=area=0.;beta=SETTINGS['beta']
    for risk,(ng,nb) in sorted(grouped.items(),reverse=True):
        nx=x+ng/good;ny=y+nb/wrong
        if nx==x:y=ny;continue
        stop=min(nx,beta);sy=y+(ny-y)*(stop-x)/(nx-x)
        area+=(stop-x)*(y+sy)/2
        if nx>=beta:break
        x,y=nx,ny
    return float(area/beta)


def evaluate(rows, cutoff, base_cutoff):
    groups={}
    for group,part in base.level.metrics.grouped(rows).items():
        ids=base.accepted(part,METHOD,cutoff)
        candidate=base.level.summarize(part,ids);cr=candidate['states']['CR'];n=len(ids)
        controls={m:dict(same_count=base.level.point(part,m,count=n),same_CR=base.level.point(part,m,correct=cr)) for m in CONTROLS}
        formal=base.level.summarize(part,base.level.formal_accept(part))
        shared=min(cr,formal['states']['CR'])
        groups[group]=dict(candidate=candidate,controls=controls,
            frozen4=base.level.summarize(part,base.accepted(part,'fusion4',base_cutoff)),formal=formal,
            old_veto_shared_CR=base.trend.matched_veto(part,correct=shared),
            candidate_at_old_shared_CR=base.level.point(part,METHOD,correct=shared),
            AUROC={m:base.level.metrics.auc(part,m) for m in (METHOD,)+CONTROLS},
            FA_regimes={kind:sum(r['image'] in ids and base.level.metrics.bad(r) is True
                and ('unsupported' if not r['temporal']['available'] else 'low' if r['temporal']['primary']<=base.LEVEL_CUTOFF else 'high')==kind
                for r in part) for kind in ('low','high','unsupported')})
    return dict(groups=groups,center=base.level.metrics.center(rows),partial_AUC={g:{m:partial_auc(part,m) for m in (METHOD,)+CONTROLS} for g,part in base.level.metrics.grouped(rows).items()})


def gate(evaluation):
    failures=[]
    for group,data in evaluation['groups'].items():
        c=data['candidate'];s=c['states']
        if s['CR']<math.ceil(.95*(s['CR']+s['FR'])):failures.append(dict(group=group,check='correct_retention95'))
        for method in ('fusion4','full_simple','score_only'):
            for match in ('same_count','same_CR'):
                control=data['controls'][method][match]
                bound=control['tie_bounds']['FA_min'] if match=='same_count' else control['states']['FA']
                if (match=='same_CR' and not control['exact_CR']) or s['FA']>bound or (group=='all' and s['FA']>=bound):
                    failures.append(dict(group=group,control=method,check=match+'_FA_gain'))
                if c['runs']['correct_rejection']['longest']>control['runs']['correct_rejection']['longest']:
                    failures.append(dict(group=group,control=method,check=match+'_correct_FR_continuity'))
    return dict(passed=not failures,failures=failures,candidate_adopted=False,
                TEST_opened=False,stop_if_failed=True)
