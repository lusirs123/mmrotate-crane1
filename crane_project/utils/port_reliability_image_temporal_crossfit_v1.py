"""Two-score anchored fusion; frozen frontend, quality-head cross-fitting only."""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_reliability_partial_auc_v1 as temporal
from crane_project.utils import port_reliability_feature_source_v1 as image

VERSION='port_reliability_image_temporal_crossfit_v1'
ARMS=('in_sample_fusion','crossfit_fusion')
METHOD=ARMS[1]
CONTROLS=(ARMS[0],'partial_auc4','fusion4','full_simple','score_only','static3')
SETTINGS=dict(seed=1701,epochs=100,batch_size=256,lr=.001,weight_decay=.0001,
              clip_norm=5.,parameter_count=4290,l2=.1,max_iterations=100,
              gradient_tolerance=1e-8,logit_clip=12.,correct_retention=.95)


def folds(rows):
    if not rows or any(r['reliability_role']!='train' or r['split'] not in ('train','train_sim') or r['pred'] is None for r in rows):
        raise ValueError('Full output TRAIN only')
    if len({r['image'] for r in rows})!=len(rows):raise ValueError('Duplicate TRAIN image')
    seq=np.array([r['sequence'] for r in rows]);result=[]
    for heldout in sorted(set(seq)):
        fit=np.flatnonzero(seq!=heldout);test=np.flatnonzero(seq==heldout)
        labels=[int(temporal.base.level.metrics.bad(rows[i])) for i in fit]
        if not labels or not any(labels) or all(labels):raise ValueError('Both fit classes needed for each fold')
        result.append(dict(heldout=str(heldout),fit=fit.tolist(),test=test.tolist()))
    return result


def bounded_logit(risk):
    r=np.asarray(risk,dtype=float)
    if not np.isfinite(r).all() or np.any((r<0)|(r>1)):raise ValueError('Finite risks in [0,1] required')
    eps=1/(1+math.exp(SETTINGS['logit_clip']))
    r=np.clip(r,eps,1-eps)
    return np.log(r)-np.log1p(-r)


def fusion_design(rp,ri,normalizer=None):
    x=np.column_stack((bounded_logit(rp),bounded_logit(ri)))
    if normalizer is None:
        scale=x.std(0);normalizer=dict(mean=x.mean(0).tolist(),scale=np.where(scale<1e-8,1.,scale).tolist())
    mean,scale=np.array(normalizer['mean']),np.array(normalizer['scale'])
    if mean.shape!=(2,) or scale.shape!=(2,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Invalid two-score normalizer')
    return np.column_stack((np.ones(len(x)),(x-mean)/scale)),x[:,0],normalizer


def objective(weights,design,offset,labels,sample_weights):
    u=offset+design@weights;p=temporal.base.simple.sigmoid(u)
    penalty=np.array(weights,copy=True);penalty[0]=0
    loss=float(sample_weights@(np.logaddexp(0,u)-labels*u)+.5*SETTINGS['l2']*penalty@penalty)
    grad=design.T@(sample_weights*(p-labels))+SETTINGS['l2']*penalty
    H=design.T@((sample_weights*p*(1-p))[:,None]*design)+SETTINGS['l2']*np.diag([0,1,1])
    return loss,grad,H


def fit_fusion(rows,rp,ri):
    if len(rows)!=len(rp) or len(rows)!=len(ri) or not rows or any(r['reliability_role']!='train' or r['split'] not in ('train','train_sim') or not r['temporal']['available'] or r['pred'] is None for r in rows):raise ValueError('History-supported TRAIN only')
    X,offset,norm=fusion_design(rp,ri)
    y=np.array([float(temporal.base.level.metrics.bad(r)) for r in rows]);bad=int(y.sum());good=len(y)-bad
    if not bad or not good:raise ValueError('Both genuine classes required')
    sw=np.where(y==1,.5/bad,.5/good);weights=np.zeros(3);initial=objective(weights,X,offset,y,sw)[0];trace=[]
    for step in range(SETTINGS['max_iterations']):
        loss,g,H=objective(weights,X,offset,y,sw);trace.append(dict(iteration=step,objective=loss,max_abs_gradient=float(abs(g).max())))
        if abs(g).max()<=SETTINGS['gradient_tolerance']:break
        direction=np.linalg.solve(H,g)
        for bt in range(30):
            trial=weights-.5**bt*direction
            if objective(trial,X,offset,y,sw)[0]<=loss-1e-4*.5**bt*g@direction:weights=trial;break
        else:raise RuntimeError('Fixed fusion line search failed')
    final,g,H=objective(weights,X,offset,y,sw)
    if abs(g).max()>SETTINGS['gradient_tolerance']:raise RuntimeError('Fusion budget exhausted')
    return dict(weights=weights.tolist(),normalizer=norm,train_images=[r['image'] for r in rows],train_rows=len(rows),bad_rows=bad,good_rows=good,initial_objective=initial,final_objective=final,max_abs_gradient=float(abs(g).max()),iterations=step+1,trace=trace)


def fused_risk(rp,ri,model,history_available):
    if rp is None:return None
    if not history_available:return float(rp)
    X,offset,_=fusion_design([rp],[ri],model['normalizer']);weights=np.array(model['weights'])
    if not np.isfinite(weights).all() or weights.shape!=(3,):raise ValueError('Invalid three fusion weights')
    if not np.any(weights):return float(rp)
    return float(temporal.base.simple.sigmoid(offset+X@weights)[0])


def score(rows,image_risks,fusion_models):
    result=deepcopy(rows)
    for r in result:
        ri=image_risks[r['image']]
        if (ri is None)!=(r['pred'] is None):raise ValueError('Image risk/output identity mismatch')
        r['image_crossfit_risk']=ri
        for arm in ARMS:r['risks'][arm]=fused_risk(r['risks']['partial_auc4'],ri,fusion_models[arm],r['temporal']['available'])
    return result


class CausalImageTemporal:
    def __init__(self,temporal_models,temporal_model,pca,head,head_normalizer,fusion,cutoff,original_policy):
        if not 0<=cutoff<=1:raise ValueError('Invalid cutoff')
        self.temporal_models=deepcopy(temporal_models);self.temporal_model=deepcopy(temporal_model)
        self.pca=deepcopy(pca);self.head=deepcopy(head);self.head_normalizer=deepcopy(head_normalizer);self.fusion=deepcopy(fusion);self.cutoff=cutoff
        self.original=temporal.base.simple.SimpleComponentReliability(original_policy);self.tracker=temporal.base.level.CausalSizeEvidence()

    def decide(self,pred,image_size,frame_id,native_patch=None,reset=False):
        result=self.original.decide(pred,image_size);point=self.tracker.observe(pred,frame_id,image_size,reset)
        rp=temporal.online_risk(pred,image_size,point,self.temporal_models,self.temporal_model);ri=None
        if pred is not None and point['available']:
            raw=np.asarray(native_patch,dtype=float)
            if raw.shape!=(2304,) or not np.isfinite(raw).all():raise ValueError('Actual selected native B patch required')
            descriptors=np.array([temporal.base.simple.descriptor(pred,image_size)])
            matrix=image.features(descriptors,raw[None],self.pca)
            ri=float(temporal.base.simple.sigmoid(image.numpy_logits(self.head,matrix,self.head_normalizer))[0])
        risk=fused_risk(rp,ri,self.fusion,point['available'])
        result['size_accepted']=risk is not None and risk<=self.cutoff
        result['size_image_temporal_risk']=risk;result['size_image_temporal_history_available']=point['available'];result['size_image_temporal_version']=VERSION
        return result


def evaluate(rows,cutoffs,original_cutoff):
    all_arms={}
    for arm in ARMS:
        groups={}
        for g,part in temporal.base.level.metrics.grouped(rows).items():
            ids=temporal.base.accepted(part,arm,cutoffs[arm]['risk_le']);summary=temporal.base.level.summarize(part,ids)
            controls={m:dict(same_count=temporal.base.level.point(part,m,count=len(ids)),same_CR=temporal.base.level.point(part,m,correct=summary['states']['CR'])) for m in CONTROLS if m!=arm}
            groups[g]=dict(candidate=summary,controls=controls,formal=temporal.base.level.summarize(part,temporal.base.level.formal_accept(part)),
                original_partial_auc=temporal.base.level.summarize(part,temporal.base.accepted(part,'partial_auc4',original_cutoff)),
                AUROC={m:temporal.base.level.metrics.auc(part,m) for m in (arm,)+tuple(controls)},
                FA_regimes={kind:sum(r['image'] in ids and temporal.base.level.metrics.bad(r) is True and ('unsupported' if not r['temporal']['available'] else 'low' if r['temporal']['primary']<=temporal.base.LEVEL_CUTOFF else 'high')==kind for r in part) for kind in ('low','high','unsupported')})
        all_arms[arm]=dict(groups=groups,center=temporal.base.level.metrics.center(rows))
    return all_arms


def gate(evaluation):
    failures=[]
    for group,data in evaluation[METHOD]['groups'].items():
        c=data['candidate'];s=c['states']
        if s['CR']<math.ceil(.95*(s['CR']+s['FR'])):failures.append(dict(group=group,check='correct_retention95'))
        for m in (ARMS[0],'partial_auc4','full_simple','score_only'):
            for match in ('same_count','same_CR'):
                control=data['controls'][m][match];bound=control['tie_bounds']['FA_min'] if match=='same_count' else control['states']['FA']
                if (match=='same_CR' and not control['exact_CR']) or s['FA']>bound or (group=='all' and s['FA']>=bound):failures.append(dict(group=group,control=m,check=match+'_FA_gain'))
                if c['runs']['correct_rejection']['longest']>control['runs']['correct_rejection']['longest']:failures.append(dict(group=group,control=m,check=match+'_correct_FR_continuity'))
    return dict(passed=not failures,failures=failures,candidate_adopted=False,TEST_opened=False,stop_if_failed=True)
