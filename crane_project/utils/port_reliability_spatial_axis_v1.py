"""Spatial quality representation and offline axis supervision; no online GT."""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_reliability_within_video_rank_v1 as old
from crane_project.utils import port_reliability_complementarity_v1 as metrics
from crane_project.utils import port_reliability_state_continuity_v1 as states

VERSION='port_reliability_spatial_axis_v1'
ARMS=('coarse_bce','spatial_bce','spatial_axis')
METHODS=ARMS+('full_simple','score_only')
SETTINGS=dict(seed=1701,epochs=100,batch_size=256,lr=.001,weight_decay=.0001,
 clip_norm=5.,residual_bound=.5,auxiliary_weight=.25,smooth_l1_beta=1.,axis_unit=.1,
 roi_size=9,context_multiplier=1.5,min_context_model_px=16,channels=256,
 stem_channels=8,hidden=16,parameter_count=13115,correct_retention=.95,
 spatial_dtype='float32',risk_loss_dtype='float64',normalization='TRAIN-only channel moments on support>0')
readout,checked_anchor,class_weights,decide,calibrate,ranking_summary=(
 old.readout,old.checked_anchor,old.class_weights,old.decide,old.calibrate,old.ranking_summary)


def targets(rows):
    """Offline only: canonical long/short signed relative error / 10%."""
    result=[]
    for r in rows:
        if r['pred'] is None: raise ValueError('No target for a missing output')
        p=np.sort(np.asarray(r['pred'][2:4],dtype=np.float64))[::-1]
        g=np.sort(np.asarray(r['gt'][2:4],dtype=np.float64))[::-1]
        if np.any(g<=0) or not np.isfinite(np.r_[p,g]).all():raise ValueError('Invalid edges')
        result.append((p-g)/g/.1)
    return np.asarray(result)


def coarse(x):
    x=np.asarray(x)
    if x.ndim!=4 or x.shape[1:]!=(256,9,9):raise ValueError('Expected Nx256x9x9')
    b=x.reshape(len(x),256,3,3,3,3).mean((3,5))
    return np.repeat(np.repeat(b,3,2),3,3)


def fit_normalizer(roi,support,descriptors,role):
    if role!='TRAIN':raise ValueError('Only TRAIN may fit normalization')
    validate_inputs(roi,support,descriptors)
    v=(support>0).astype(np.float64);x=roi.astype(np.float64)
    mass=float(v.sum())
    if not mass:raise ValueError('No TRAIN feature support')
    mean=(x*v).sum((0,2,3))/mass
    var=(((x-mean[None,:,None,None])*v)**2).sum((0,2,3))/mass
    scale=np.sqrt(var);scale=np.where(scale<1e-8,1.,scale)
    ds=descriptors.std(0);ds=np.where(ds<1e-8,1.,ds)
    return dict(channel_mean=mean.tolist(),channel_scale=scale.tolist(),
                descriptor_mean=descriptors.mean(0).tolist(),descriptor_scale=ds.tolist(),fit_role='TRAIN')


def validate_inputs(roi,support,descriptors):
    n=len(roi)
    if (roi.shape!=(n,256,9,9) or support.shape!=(n,1,9,9) or descriptors.shape!=(n,2)
        or not all(np.isfinite(z).all() for z in (roi,support,descriptors))
        or np.any(support<0) or np.any(support>1+1e-6)):
        raise ValueError('Invalid spatial inputs')


def normalize(roi,support,descriptors,norm):
    validate_inputs(roi,support,descriptors)
    if norm['fit_role']!='TRAIN':raise ValueError('Invalid normalization role')
    m=np.asarray(norm['channel_mean']);s=np.asarray(norm['channel_scale'])
    dm=np.asarray(norm['descriptor_mean']);ds=np.asarray(norm['descriptor_scale'])
    if m.shape!=(256,) or s.shape!=(256,) or dm.shape!=(2,) or ds.shape!=(2,) or np.any(s<=0) or np.any(ds<=0):
        raise ValueError('Invalid normalization dimensions')
    x=((roi-m[None,:,None,None])/s[None,:,None,None])*(support>0)
    return x.astype(np.float32),((descriptors-dm)/ds).astype(np.float32)


def conv(x,w,b,padding=0):
    """NumPy convolution for full saved-head replay, independent of Torch."""
    if padding:x=np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)))
    k=w.shape[-1];out=x.shape[2]-k+1
    z=np.zeros((len(x),w.shape[0],out,out),dtype=np.float64)
    for i in range(k):
        for j in range(k):
            z+=np.einsum('nchw,oc->nohw',x[:,:,i:i+out,j:j+out],w[:,:,i,j],optimize=True)
    return z+b[None,:,None,None]


def numpy_forward(model,x,support,descriptors,anchor,arm):
    if arm not in ARMS:raise ValueError('Unknown arm')
    if arm=='coarse_bce':x=coarse(x)
    w=lambda k:np.asarray(model[k],dtype=np.float64)
    h=np.concatenate((x,support),axis=1).astype(np.float64)
    h=np.maximum(conv(h,w('stem.0.weight'),w('stem.0.bias')),0)
    h=np.maximum(conv(h,w('stem.2.weight'),w('stem.2.bias'),1),0)
    h=np.concatenate((h.reshape(len(h),-1),descriptors),axis=1)
    h=np.maximum(h@w('hidden.weight').T+w('hidden.bias'),0)
    delta=.5*np.tanh((h@w('risk_head.weight').T+w('risk_head.bias'))[:,0])
    axis=h@w('axis_head.weight').T+w('axis_head.bias')
    return readout(anchor,delta),delta,axis


def statistics(rows,cutoffs):
    result={}
    for method in METHODS:
        result[method]={}
        for name,group in metrics.grouped(rows).items():
            actual=states.summarize(group,metrics.accepted(group,method,cutoffs[method]['risk_le']))
            result[method][name]=dict(actual=actual,error_AUROC=metrics.auc(group,method),
              controls={m:metrics.matched(group,m,actual) for m in METHODS if m!=method})
    return result


def gate(statistics):
    failures=[];gain=False
    for group,value in statistics['spatial_axis'].items():
        actual=value['actual'];c=actual['states']
        if c['CR']<math.ceil(.95*(c['CR']+c['FR'])):failures.append(dict(group=group,check='correct_retention95'))
        for method in ('spatial_bce','full_simple','score_only'):
            comp=value['controls'][method];sc=comp['same_CR']
            checks=dict(same_count_FA_nonincrease=c['FA']<=comp['same_count_tie_bounds']['bad_min'],
              same_CR_exact=comp['same_CR_exact'],same_CR_FA_nonincrease=c['FA']<=sc['states']['FA'],
              longest_correct_FR_nonincrease=actual['runs']['correct_rejection']['longest']<=sc['runs']['correct_rejection']['longest'])
            if group=='all':checks.update(overall_strict_same_count_FA_gain=c['FA']<comp['same_count_tie_bounds']['bad_min'],overall_strict_same_CR_FA_gain=c['FA']<sc['states']['FA'])
            failures.extend(dict(group=group,control=method,check=k) for k,v in checks.items() if not v)
            if group.startswith('sequence:real_') and method=='full_simple' and comp['same_CR_exact'] and c['FA']<sc['states']['FA']:
                a=value['error_AUROC'];refs=[statistics[m][group]['error_AUROC'] for m in ('spatial_bce','full_simple')]
                if a is not None and all(v is not None and a>v for v in refs):gain=True
    if not gain:failures.append(dict(group='Real_videos',check='same_CR_FA_and_video_AUROC_gain'))
    return dict(passed=not failures,failures=failures,selected_arm='spatial_axis' if not failures else None,
                no_control_reselection=True,TEST_allowed=not failures)


def axis_metrics(rows,predictions):
    out={};lookup={r['image']:i for i,r in enumerate(rows)};t=targets(rows)
    for name,group in metrics.grouped(rows).items():
        ix=[lookup[r['image']] for r in group];e=(predictions[ix]-t[ix])*.1
        out[name]=dict(n=len(ix),signed_bias=e.mean(0).tolist(),relative_MAE=np.abs(e).mean(0).tolist(),
                      absolute_error_P95=np.quantile(np.abs(e),.95,axis=0).tolist(),
                      prediction_is_not_correct_probability=True)
    return out


def markdown(report):
    lines=['# 空间质量表示与长短轴连续辅助监督','', '**状态：%s**。'%report['status'],
      '完整TRAIN→完整VAL；未读取TEST。只改变尺寸评分/标志，正式M/simple保持。','',
      '| VAL方法 | 接受数 | FA | FR | ED | CR | AUROC |','|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        v=report['statistics']['VAL'][m]['all'];c=v['actual']['states']
        lines.append('|%s|%d|%d|%d|%d|%d|%.6f|'%(m,v['actual']['accepted_outputs'],c['FA'],c['FR'],c['ED'],c['CR'],v['error_AUROC']))
    lines+=['','| 独立同CR比较 | CR | 候选FA | 空间BCE FA | simple FA | score FA |','|---|---:|---:|---:|---:|---:|']
    for g,v in report['statistics']['VAL']['spatial_axis'].items():
        c=v['actual']['states'];lines.append('|%s|%d|%d|%d|%d|%d|'%(g,c['CR'],c['FA'],*(v['controls'][m]['same_CR']['states']['FA'] for m in ('spatial_bce','full_simple','score_only'))))
    lines+=['','分组同CR参照不可相加；原正式policy另报。辅助误差不产生接受标志、不修框。',
      'A/B/C相同容量、初始化与1000次更新，不声称相同反向成本；A固定3×3压缩后展开。',
      '候选失败不反选A/B，不调系数、预算或门限。当前VAL/TEST已暴露，不声称独立泛化保证。']
    return '\n'.join(lines)+'\n'
