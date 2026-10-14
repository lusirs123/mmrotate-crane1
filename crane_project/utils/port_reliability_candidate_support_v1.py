"""GT-free same-forward candidate evidence; final M geometry stays immutable."""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_reliability_partial_auc_v1 as parent

VERSION = 'port_reliability_candidate_support_v1'
METHOD = 'candidate_support6'
CONTROL = 'candidate_support4'
SETTINGS = dict(max_neighbors=8, min_neighbors=3, angle_degrees=15.,
                min_domain_coverage=.5, min_wrong=20, min_video_wrong=3,
                min_video_good=20, min_real_videos=2, min_direction_auc=.55)


def canonical(box):
    b=np.asarray(box,dtype=float)
    if b.shape!=(6,) or not np.isfinite(b).all() or min(b[2:4])<=0 or not 0<=b[5]<=1:
        raise ValueError('Finite positive original-coordinate box required')
    long,short=b[2:4];angle=b[4]
    if long<short:long,short=short,long;angle+=math.pi/2
    angle=(angle+math.pi/2)%math.pi-math.pi/2
    return float(long),float(short),float(angle)


def evidence(pred,candidates,chosen):
    """Candidates already obey native padding/score masks, before native Top-K.

    chosen is the actual surviving-array index, never a nearest-box match.
    Candidate sizes are B predictions; evidence is relative to delivered M.
    No score threshold relaxation, GT, domain, sequence, or temporal inputs.
    """
    pool=np.asarray(candidates,dtype=float)
    if pool.ndim!=2 or pool.shape[1]!=6 or not np.isfinite(pool).all():
        raise ValueError('Invalid native candidate pool')
    if pred is None:
        if len(pool) or chosen is not None:raise ValueError('Missing output has candidates')
        return dict(available=False,reason='missing',neighbors=[],features=None,surviving=0)
    if not isinstance(chosen,(int,np.integer)) or not 0<=chosen<len(pool):
        raise ValueError('Actual selected index required')
    L,S,a=canonical(pred);ca,sa=math.cos(a),math.sin(a);eligible=[]
    for i,b in enumerate(pool):
        if i==chosen:continue
        l,s,t=canonical(b);dx,dy=b[:2]-np.asarray(pred[:2])
        along=ca*dx+sa*dy;across=-sa*dx+ca*dy
        angle=abs((t-a+math.pi/2)%math.pi-math.pi/2)
        if abs(along)<=L/2 and abs(across)<=S/2 and angle<=math.radians(SETTINGS['angle_degrees'])+1e-14:
            eligible.append((i,b.tolist(),math.log(l/L),math.log(s/S)))
    # Deterministic score ties by native surviving index; never GT selection.
    eligible.sort(key=lambda v:(-v[1][5],v[0]));selected=eligible[:SETTINGS['max_neighbors']]
    available=len(selected)>=SETTINGS['min_neighbors']
    values=np.asarray([[v[2],v[3]] for v in selected])
    return dict(available=available,reason='supported' if available else 'insufficient_neighbors',
        neighbors=[dict(index=i,box=b) for i,b,_,_ in selected],
        features=np.median(values,axis=0).tolist() if available else None,
        surviving=len(pool),eligible=len(eligible),selected_index=int(chosen))


def auc(values,labels):
    values=np.asarray(values);labels=np.asarray(labels,dtype=bool)
    good=values[~labels];wrong=values[labels]
    if not len(good) or not len(wrong):return None
    return float(np.mean((wrong[:,None]>good[None,:])+.5*(wrong[:,None]==good[None,:])))


def screen(rows):
    if len(rows)!=2558 or any(r['reliability_role']!='train' for r in rows):raise ValueError('Full TRAIN only')
    supported=[r for r in rows if r['candidate_evidence']['available'] and r['temporal']['available']]
    bad=parent.base.level.metrics.bad
    groups={};fail=[]
    for name,part in parent.base.level.metrics.grouped(rows).items():
        out=[r for r in part if r['pred'] is not None]
        ready=[r for r in out if r['candidate_evidence']['available']]
        groups[name]=dict(outputs=len(out),supported=len(ready),coverage=len(ready)/len(out) if out else 0.,
            supported_wrong=sum(bad(r) is True for r in ready),supported_good=sum(bad(r) is False for r in ready))
    for name in ('real','sim'):
        # Historical grouped() uses capitalized domains; support either exact key.
        key='domain:'+name
        if groups[key]['coverage']<SETTINGS['min_domain_coverage']:fail.append('coverage_'+name)
    y=[bad(r) for r in supported];x=np.asarray([r['candidate_evidence']['features'] for r in supported]).reshape(-1,2)
    if sum(y)<SETTINGS['min_wrong']:fail.append('too_few_supported_wrong')
    pooled=[auc(x[:,j],y) for j in range(2)]
    signs=[1 if a is None or a>=.5 else -1 for a in pooled]
    videos={};passing=[[],[]]
    for seq in sorted({r['sequence'] for r in supported}):
        part=[r for r in supported if r['sequence']==seq];labels=[bad(r) for r in part]
        xx=np.asarray([r['candidate_evidence']['features'] for r in part]);a=[auc(signs[j]*xx[:,j],labels) for j in range(2)]
        ng=len(labels)-sum(labels);nb=sum(labels)
        videos[seq]=dict(good=ng,wrong=nb,directed_AUC=a)
        if seq.startswith('real_') and ng>=SETTINGS['min_video_good'] and nb>=SETTINGS['min_video_wrong']:
            for j in range(2):
                if a[j]>=SETTINGS['min_direction_auc']:passing[j].append(seq)
    if max(map(len,passing))<SETTINGS['min_real_videos']:fail.append('no_consistent_two_video_signal')
    return dict(passed=not fail,failures=fail,groups=groups,pooled_AUC=pooled,directions=signs,
        videos=videos,passing_videos_by_axis=passing,paired_fit_rows=len(supported),
        paired_fit_wrong=sum(y),not_a_performance_guarantee=True)


def objective(v,good,wrong):
    """Same fixed smooth partial-AUC objective generalized to 4/6 slopes."""
    v=np.asarray(v,dtype=float);good=np.asarray(good);wrong=np.asarray(wrong);dimen=len(v)-1
    if dimen not in (4,6) or good.shape[1]!=dimen or wrong.shape[1]!=dimen or not len(good) or not len(wrong):raise ValueError('Paired genuine classes required')
    if not all(np.isfinite(a).all() for a in (v,good,wrong)):raise ValueError('Nonfinite objective')
    d=good[:,None,:]-wrong[None,:,:];u=d@v[:-1];prob=parent.base.simple.sigmoid(u)
    h=np.logaddexp(0.,u).mean(axis=1);jac=np.mean(prob[:,:,None]*d,axis=1)
    hes=np.einsum('gi,gij,gik->gjk',prob*(1-prob),d,d)/len(wrong)
    T=parent.SETTINGS['temperature'];beta=parent.SETTINGS['beta'];lam=parent.SETTINGS['l2'];p=parent.base.simple.sigmoid((h-v[-1])/T)
    loss=float(v[-1]+T*np.logaddexp(0.,(h-v[-1])/T).mean()/beta+.5*lam*v[:-1].dot(v[:-1]))
    grad=np.r_[np.mean(p[:,None]*jac,axis=0)/beta+lam*v[:-1],1-p.mean()/beta]
    outer=np.column_stack((jac,-np.ones(len(good))))
    H=np.einsum('g,gi,gj->ij',p*(1-p)/(beta*T*len(good)),outer,outer)
    H[:-1,:-1]+=np.einsum('g,gij->ij',p/(beta*len(good)),hes)+lam*np.eye(dimen)
    return loss,grad,H


def vector(row,dim):
    x=parent.base.features(row['pred'],row['image_size'],row['temporal'],4)
    return x+(row['candidate_evidence']['features'] if dim==6 else [])


def fit(rows,dim,intercept):
    if any(r['reliability_role']!='train' for r in rows):raise ValueError('TRAIN-only fit')
    part=[r for r in rows if r['candidate_evidence']['available'] and r['temporal']['available']]
    x=np.array([vector(r,dim) for r in part]);y=np.array([parent.base.level.metrics.bad(r) for r in part])
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale<1e-8,1.,scale);z=(x-mean)/scale
    v=np.r_[np.zeros(dim),math.log(2)+.1*math.log(19)];trace=[]
    for step in range(parent.SETTINGS['max_iterations']):
        loss,g,H=objective(v,z[~y],z[y]);trace.append(dict(step=step,loss=loss,max_gradient=float(abs(g).max())))
        if abs(g).max()<=1e-8:break
        delta=np.linalg.solve(H,g)
        for k in range(30):
            trial=v-.5**k*delta
            if objective(trial,z[~y],z[y])[0]<=loss-1e-4*.5**k*g.dot(delta):v=trial;break
        else:raise RuntimeError('Fixed line search failed')
    loss,g,_=objective(v,z[~y],z[y])
    if abs(g).max()>1e-8:raise RuntimeError('Fixed fitting budget exhausted')
    return dict(weights=[intercept]+v[:-1].tolist(),tail_threshold=float(v[-1]),mean=mean.tolist(),scale=scale.tolist(),
        train_images=[r['image'] for r in part],bad_rows=int(y.sum()),trace=trace,final_objective=loss,
        max_abs_gradient=float(abs(g).max()),settings=deepcopy(parent.SETTINGS))


def risk(pred,image_size,point,ev,model,fallback):
    if pred is None:return None
    if not point['available'] or not ev['available']:return fallback
    dim=len(model['mean']);x=parent.base.features(pred,image_size,point,4)+(ev['features'] if dim==6 else [])
    return float(parent.base.simple.linear_risk(model,np.asarray([x]))[0])


def score(rows,models):
    out=deepcopy(rows)
    for r in out:
        for name,m in models.items():
            r['risks'][name]=risk(r['pred'],r['image_size'],r['temporal'],r['candidate_evidence'],m,r['risks'][parent.METHOD])
    return out


def active(row):
    return row['pred'] is not None and row['temporal']['available'] and row['candidate_evidence']['available']


def accepted(rows,method,cutoff,fallback_cutoff):
    return {r['image'] for r in rows if r['pred'] is not None and
            (r['risks'][method]<=cutoff if active(r) else r['risks'][parent.METHOD]<=fallback_cutoff)}


def calibrate(rows,method,fallback_cutoff):
    if any(r['reliability_role']!='val' for r in rows):raise ValueError('VAL-only calibration')
    groups={};thresholds=[]
    for group,part in parent.base.level.metrics.grouped(rows).items():
        good=[r for r in part if parent.base.level.metrics.bad(r) is False]
        retained=sum(not active(r) and r['risks'][parent.METHOD]<=fallback_cutoff for r in good)
        need=max(0,math.ceil(.95*len(good))-retained)
        values=sorted(r['risks'][method] for r in good if active(r))
        if need>len(values):raise ValueError('Fixed fallback cannot satisfy group95')
        threshold=values[need-1] if need else 0.
        groups[group]=dict(good=len(good),fallback_correct=retained,required_active=need,cutoff=threshold)
        thresholds.append(threshold)
    return dict(risk_le=max(thresholds),fallback_risk_le=fallback_cutoff,groups=groups,
        not_a_calibrated_probability=True,role='previously_exposed_VAL')


def evaluate(rows,cutoffs):
    groups={}
    for group,part in parent.base.level.metrics.grouped(rows).items():
        data={}
        for name,cutoff in cutoffs.items():
            ids=accepted(part,name,cutoff['risk_le'],cutoff['fallback_risk_le'])
            c=parent.base.level.summarize(part,ids);cr=c['states']['CR']
            methods=(parent.METHOD,'fusion4','full_simple','score_only')
            controls={m:dict(same_count=parent.base.level.point(part,m,count=len(ids)),
                            same_CR=parent.base.level.point(part,m,correct=cr)) for m in methods}
            data[name]=dict(candidate=c,controls=controls,FA_regimes={kind:sum(r['image'] in ids and parent.base.level.metrics.bad(r) is True and
                ('unsupported' if not r['temporal']['available'] else 'low' if r['temporal']['primary']<=parent.base.LEVEL_CUTOFF else 'high')==kind for r in part) for kind in ('low','high','unsupported')})
        # Paired 4D control preserves the exact same support/fallback contract.
        groups[group]=data
    return dict(groups=groups,center=parent.base.level.metrics.center(rows))


def gate(evaluation):
    failures=[]
    for group,arms in evaluation['groups'].items():
        c=arms[METHOD]['candidate'];s=c['states']
        if s['CR']<math.ceil(.95*(s['CR']+s['FR'])):failures.append(dict(group=group,check='correct_retention95'))
        for method,matches in arms[METHOD]['controls'].items():
            for kind,control in matches.items():
                bound=control['tie_bounds']['FA_min'] if kind=='same_count' else control['states']['FA']
                if (kind=='same_CR' and not control['exact_CR']) or s['FA']>bound or (group=='all' and s['FA']>=bound):failures.append(dict(group=group,control=method,check=kind+'_FA'))
                if c['runs']['correct_rejection']['longest']>control['runs']['correct_rejection']['longest']:
                    failures.append(dict(group=group,control=method,check=kind+'_continuous_FR'))
        # Delivery-level comparison with paired control, alongside matched base scores.
        paired=arms[CONTROL]['candidate']['states']
        if group=='all' and not (s['FA']<paired['FA'] and s['CR']>=paired['CR']):failures.append(dict(group=group,check='paired_control_gain'))
        if s['FA']>paired['FA'] or s['FR']>paired['FR']:failures.append(dict(group=group,check='paired_control_protection'))
    low=evaluation['groups']['all'][METHOD]['FA_regimes']['low']
    control=evaluation['groups']['all'][CONTROL]['FA_regimes']['low']
    if low>=control:failures.append(dict(group='all',check='stable_error_increment'))
    return dict(passed=not failures,failures=failures,candidate_adopted=False,TEST_opened=False)


class CausalCandidateSupport:
    def __init__(self,base_models,base_model,base_cutoff,model,cutoff,policy):
        self.base=parent.CausalPartialAUC(base_models,base_model,base_cutoff,policy)
        self.model=deepcopy(model);self.cutoff=cutoff
    def decide(self,pred,image_size,frame_id,candidates,chosen,reset=False):
        result=self.base.original.decide(pred,image_size)
        point=self.base.tracker.observe(pred,frame_id,image_size,reset)
        fallback=parent.online_risk(pred,image_size,point,self.base.models,self.base.model)
        result['size_accepted']=fallback is not None and fallback<=self.base.cutoff
        ev=evidence(pred,candidates,chosen)
        value=risk(pred,image_size,point,ev,self.model,fallback)
        if pred is not None and point['available'] and ev['available']:result['size_accepted']=value<=self.cutoff
        result.update(size_candidate_support_risk=value,size_candidate_support_available=ev['available'])
        return result
