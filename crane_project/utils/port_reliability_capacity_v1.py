"""Capacity-only paired experiment on frozen TRAIN PCA features."""
import math
from crane_project.utils.port_simple_component_reliability_v1 import sigmoid
import numpy as np
import torch
from torch import nn
from crane_project.utils.port_reliability_feature_source_v1_torch import ResidualRisk
from crane_project.utils import port_reliability_feature_source_v1 as feature
from crane_project.utils import port_reliability_redc_size_v1_torch as training
from crane_project.utils.port_reliability_paired_ranking_audit_v1 import Ranking,bad,groups,summarize

SEEDS=(1701,1702,1703)
class LargeRisk(ResidualRisk):
    def __init__(self):
        # Same layer construction order, nonlinearities and neutral initialization.
        nn.Module.__init__(self)
        self.network=nn.Sequential(nn.Linear(258,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
        nn.init.zeros_(self.network[4].weight);nn.init.zeros_(self.network[4].bias)
        self.beta=nn.Parameter(torch.zeros(1))

def make_model(arm,seed):
    torch.manual_seed(seed)
    return {'small':ResidualRisk,'large':LargeRisk}[arm]()

def prepare(rows,features,images,normalizer):
    """Identity-join labels, never infer label order from saved arrays."""
    images=list(map(str,images));by_id={r['image']:r for r in rows}
    if len(by_id)!=len(rows) or len(set(images))!=len(images):raise ValueError('Duplicate identity')
    if set(images)!={r['image'] for r in rows if r['pred'] is not None}:raise ValueError('Feature/output material mismatch')
    features=np.asarray(features,dtype=np.float64)
    if features.shape!=(len(images),259):raise ValueError('Expected frozen259 input')
    x=feature.normalize(features,normalizer)
    labels=np.asarray([bad(by_id[i]) for i in images],dtype=np.float32)
    return x,features[:,0].copy(),labels

def fit(x,z,y,seed):
    if len(y)!=2558 or int(y.sum())!=71:raise ValueError('Complete TRAIN only')
    tensors=[torch.tensor(v,dtype=torch.float32) for v in (x,z,y,feature.class_weights(y))]
    tx,tz,ty,tw=tensors
    models={arm:make_model(arm,seed) for arm in ('small','large')}
    for m in models.values():
        if not torch.equal(m(tx,tz),-tz):raise ValueError('Neutral score identity failed')
    optimizers={arm:training.optimizer(m) for arm,m in models.items()}
    rng=np.random.RandomState(seed);logs=[];orders=[];counts=dict(small=0,large=0)
    for epoch in range(1,101):
        order=rng.permutation(len(y));orders.append(order.astype('<i8').tobytes())
        for slot,offset in enumerate(range(0,len(y),256)):
            index=torch.tensor(order[offset:offset+256],dtype=torch.long)
            for arm,m in models.items():
                v=training.update(m,optimizers[arm],tx[index],tz[index],ty[index],tw[index]);counts[arm]+=1
                logs.append(dict(seed=seed,arm=arm,epoch=epoch,slot=slot,**v))
        if epoch%25==0:print('seed',seed,'epoch',epoch,'updates',counts,flush=True)
    if counts!=dict(small=1000,large=1000):raise ValueError('Budget mismatch')
    return models,logs,b''.join(orders)

def score(rows,matrix,images,normalizer,exported,method):
    logits=feature.numpy_logits(exported,matrix,normalizer)
    risks=sigmoid(logits)
    values=dict(zip(map(str,images),map(float,risks)))
    for r in rows:r['audit_risks'][method]=values.get(r['image'])
    return values

def at_cut(rows,method,cut):
    ids={r['image'] for r in rows if r['pred'] is not None and cut is not None and r['audit_risks'][method]<=cut}
    result={g:summarize(part,ids) for g,part in groups(rows).items()}
    for g,part in groups(rows).items():
        longest=run=gap=longgap=0;prev=None
        for r in sorted(part,key=lambda r:(r['sequence'],r['frame_id'],r['image'])):
            if r['sequence']!=prev:run=gap=0
            run=run+1 if bad(r) is False and r['image'] not in ids else 0
            gap=0 if bad(r) is False and r['image'] in ids else gap+1
            longest=max(longest,run);longgap=max(longgap,gap);prev=r['sequence']
        result[g].update(longest_FR_observations=longest,longest_without_CR_observations=longgap)
    return result

def points(rows,method,target):
    ranking=Ranking(rows,method)
    if target>ranking.good:raise ValueError('CR target beyond available correct observations')
    primary=ranking.same_cr(target)
    return dict(primary_point=primary,primary=at_cut(rows,method,primary['risk_le']),
                auxiliary={str(p):ranking.same_cr(math.ceil(ranking.good*p/100)) for p in (95,97)},
                AUROC=ranking.auc(),curve=ranking.curve())

def gate(results,baseline):
    """Predeclared conservative decision, no all-video-win requirement."""
    checks=[]
    for seed in SEEDS:
        s,l=(results[str(seed)][a]['VAL']['primary'] for a in ('small','large'))
        checks.append(dict(seed=seed,beats_small=l['all']['states']['FA']<s['all']['states']['FA'],
                           beats_simple=l['all']['states']['FA']<baseline['all']['states']['FA'],
                           protects_real=l['domain:real']['states']['CR']>=baseline['domain:real']['states']['CR']))
    passed=all(c['beats_small'] and c['beats_simple'] and c['protects_real'] for c in checks)
    return dict(passed=passed,checks=checks,representative_seed=1701 if passed else None,
                selected_arm='large' if passed else None,selection='fixed seed1701; no best-seed selection; TEST only after PASS')
