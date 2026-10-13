"""One scalar correction to frozen four-dimensional size risk; GT offline only."""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_reliability_temporal_fusion_v1 as base

VERSION = 'port_reliability_size_temporal_interaction_v1'
SETTINGS = dict(l2=.1, max_iterations=100, gradient_tolerance=1e-8,
                interaction_indices=[1, 3], initialization=0., correct_retention=.95)
METHOD = 'size_temporal_interaction'
CONTROLS = ('fusion4', 'full_simple', 'score_only', 'static3')


def terms(pred, image_size, point, models):
    if pred is None or not point['available']:
        raise ValueError('Interaction requires actual output and sufficient history')
    m = models['fusion4']
    z = (np.asarray(base.features(pred, image_size, point, 4))-m['mean'])/m['scale']
    return float(m['weights'][0]+z.dot(m['weights'][1:])), float(z[1]*z[3])


def objective(gamma, logits, phi, labels, sample_weights):
    u = logits+gamma*phi
    p = base.simple.sigmoid(u)
    loss = float(sample_weights.dot(np.logaddexp(0.,u)-labels*u)+.5*SETTINGS['l2']*gamma*gamma)
    gradient = float(sample_weights.dot((p-labels)*phi)+SETTINGS['l2']*gamma)
    hessian = float(sample_weights.dot(p*(1-p)*phi*phi)+SETTINGS['l2'])
    return loss, gradient, hessian


def fit(train, models):
    if not train or any(r['reliability_role']!='train' or r['split'] not in ('train','train_sim') for r in train):
        raise ValueError('Complete TRAIN only; VAL/TEST forbidden')
    part = [r for r in train if r['pred'] is not None and r['temporal']['available']]
    if [r['image'] for r in part]!=models['fusion4']['train_images']:
        raise ValueError('Frozen supported TRAIN identities changed')
    array = np.array([terms(r['pred'],r['image_size'],r['temporal'],models) for r in part])
    y = np.array([int(base.level.metrics.bad(r)) for r in part],dtype=float)
    nbad = int(y.sum()); ngood = len(y)-nbad
    if not nbad or not ngood: raise ValueError('Both genuine classes required')
    logits,phi = array.T
    weights = np.where(y==1,.5/nbad,.5/ngood)
    gamma = 0.; initial, g0, _ = objective(gamma,logits,phi,y,weights)
    for step in range(SETTINGS['max_iterations']):
        loss,g,h = objective(gamma,logits,phi,y,weights)
        if not all(math.isfinite(v) for v in (loss,g,h)) or h<=0: raise ValueError('Invalid scalar derivatives')
        if abs(g)<=SETTINGS['gradient_tolerance']: break
        for backtrack in range(30):
            candidate = gamma-(.5**backtrack)*g/h
            if objective(candidate,logits,phi,y,weights)[0]<=loss-1e-4*(.5**backtrack)*g*g/h:
                gamma = candidate; break
        else: raise RuntimeError('Fixed scalar line search failed')
    final,g,h = objective(gamma,logits,phi,y,weights)
    if abs(g)>SETTINGS['gradient_tolerance']: raise RuntimeError('Fixed fitting budget exhausted')
    return dict(gamma=float(gamma),train_images=[r['image'] for r in part],
                train_rows=len(part),bad_rows=nbad,good_rows=ngood,
                initial_objective=initial,final_objective=final,initial_gradient=g0,
                final_gradient=g,final_hessian=h,iterations=step+1,
                settings=deepcopy(SETTINGS),frozen_base_not_refitted=True)


def online_risk(pred, image_size, point, models, gamma):
    if not math.isfinite(gamma): raise ValueError('Gamma must be finite')
    risk = base.online_risks(pred,image_size,point,models)['fusion4']
    if pred is None or not point['available'] or gamma==0.:
        return risk  # Exact neutral identity; shared static fallback, never fake d=0.
    logit,phi = terms(pred,image_size,point,models)
    return float(base.simple.sigmoid(logit+gamma*phi))


class CausalInteraction:
    def __init__(self, models, gamma, cutoff, original_policy):
        if not math.isfinite(gamma) or not 0<=cutoff<=1: raise ValueError('Invalid frozen scalar policy')
        self.models=deepcopy(models);self.gamma=gamma;self.cutoff=cutoff
        self.original=base.simple.SimpleComponentReliability(original_policy)
        self.tracker=base.level.CausalSizeEvidence()

    def decide(self, pred, image_size, frame_id, reset=False):
        result=self.original.decide(pred,image_size)
        point=self.tracker.observe(pred,frame_id,image_size,reset)
        risk=online_risk(pred,image_size,point,self.models,self.gamma)
        result['size_accepted']=risk is not None and risk<=self.cutoff
        result['size_interaction_risk']=risk
        result['size_interaction_history_available']=point['available']
        result['size_interaction_version']=VERSION
        return result


def score(rows, models, gamma):
    copied=deepcopy(rows)
    for r in copied:
        neutral=online_risk(r['pred'],r['image_size'],r['temporal'],models,0.)
        previous=r['risks']['fusion4']
        if (neutral is None)!=(previous is None) or (neutral is not None and abs(neutral-previous)>1e-12):
            raise ValueError('Frozen base scoring identity differs')
        r['risks'][METHOD]=online_risk(r['pred'],r['image_size'],r['temporal'],models,gamma)
    return copied


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
    return dict(groups=groups,center=base.level.metrics.center(rows))


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
