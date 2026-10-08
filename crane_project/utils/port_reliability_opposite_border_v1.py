"""Finite frozen-ROI size quality readout. Sampling/scoring APIs accept no GT."""
from copy import deepcopy
import math

import numpy as np

from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_reliability_feature_ablation_v1 as ab
from crane_project.utils import port_reliability_separability_v1 as sep
from crane_project.utils import port_reliability_state_continuity_v1 as states
from crane_project.utils.port_reliability_tradeoff_v1 import required_good

VERSION = 'port_reliability_opposite_border_v1'
ARM = 'opposite_border'
CONTROLS = ('refit_simple', 'ordinary_roi', 'full_simple', 'score_only')
FIT = dict(l2=.1, max_iterations=100, gradient_tolerance=1e-8)
SAMPLING = dict(roi_size=9, channels=256, context_multiplier=1.5,
               min_context_model_px=16., stride_model_px=8., ordinary_grid=9,
               ordinary_pool=3, tangent_fractions=[-.35, 0., .35],
               band_names=['inner', 'edge', 'outer'],
               fields=['channel_mean', 'channel_rms'], offset_min_cache_intervals=1.)
ROI_NAMES = tuple('roi_r%d_c%d_%s' % (r,c,k) for r in range(3) for c in range(3)
                  for k in ('mean','rms','support'))
BORDER_NAMES = tuple('%s_%s_%s_%s' % (axis,side,band,k)
    for axis in ('long_normal','short_normal') for side in ('positive','negative')
    for band in ('inner','edge','outer') for k in ('mean','rms','support'))
SCHEMAS = dict(refit_simple=simple.FEATURES,
               ordinary_roi=simple.FEATURES+ROI_NAMES,
               opposite_border=simple.FEATURES+BORDER_NAMES)


def summarize_channels(roi):
    """Fixed channel mean/RMS, no learned stem/projection or GT labels."""
    x = np.asarray(roi, dtype=np.float64)
    if x.shape != (256,9,9) or not np.isfinite(x).all():
        raise ValueError('Expected finite native cached 256x9x9 ROI')
    return np.stack((x.mean(axis=0), np.sqrt(np.mean(x*x, axis=0))))


def transform(base_original, base_model, scale_xy):
    """Exact native raw w/h restoration; canonical axes only for cache frame."""
    b = simple.prediction(base_original)
    bm = np.asarray(base_model, dtype=np.float64); xy = np.asarray(scale_xy, dtype=np.float64)
    if (b is None or bm.shape != (5,) or xy.shape != (2,) or not np.isfinite(xy).all()
            or not np.isfinite(bm).all() or np.any(xy<=0)):
        raise ValueError('Invalid original/model cache frame')
    expected = b[:4]*xy[[0,1,0,1]]
    if (not np.allclose(bm[:4], expected, atol=1e-4, rtol=1e-6)
            or abs((bm[4]-b[4]+math.pi/2)%math.pi-math.pi/2)>1e-5):
        raise ValueError('Native sx/sy or raw w/h-angle association differs')
    cb = simple.canonical(bm); c,s = math.cos(cb[4]),math.sin(cb[4])
    rotation = np.array([[c,-s],[s,c]])
    sides = np.maximum(cb[2:4]*1.5,16.)
    return cb, xy, sides, rotation


def bilinear(values, local):
    """Stored grid endpoints are +/-0.5. No extrapolation or extra resolution."""
    x=np.asarray(values,dtype=np.float64); p=np.asarray(local,dtype=np.float64)
    if (x.ndim!=3 or x.shape[1:]!=(9,9) or p.shape[-1]!=2
            or not np.isfinite(x).all() or not np.isfinite(p).all()):
        raise ValueError('Invalid fixed cached grid')
    flat=p.reshape(-1,2); index=(flat+.5)*8
    inside=(np.abs(flat)<=.5+1e-12).all(axis=1)
    index=np.clip(index,0,8); low=np.floor(index).astype(int); high=np.minimum(low+1,8)
    weight=index-low; vx,vy=weight[:,0],weight[:,1]
    a=x[:,low[:,1],low[:,0]]; b=x[:,low[:,1],high[:,0]]
    c=x[:,high[:,1],low[:,0]]; d=x[:,high[:,1],high[:,0]]
    value=(a*(1-vx)+b*vx)*(1-vy)+(c*(1-vx)+d*vx)*vy
    return (value*inside[None]).T.reshape(p.shape[:-1]+(x.shape[0],))


def pooled(fields,support,local):
    """Supported tangent/region mean; zero evidence never silently drops a frame."""
    weight=bilinear(support[None],local)[...,0]
    value=bilinear(fields,local)
    mass=weight.sum(axis=-1)
    total=(value*weight[...,None]).sum(axis=-2)
    mean=np.divide(total,mass[...,None],out=np.zeros_like(total),where=mass[...,None]>0)
    return np.concatenate((mean,weight.mean(axis=-1)[...,None]),axis=-1)


def sample_evidence(main,base_original,base_model,scale_xy,fields,support):
    """GT-free query of actual M borders inside the frozen B cache frame.

    Bands are one stored-grid interval AND one native P3 stride apart or more.
    Unsupported/out-of-cache samples have zero support, explicitly reported.
    This tests spatial organization of the SAME cached signal, not sharper P3.
    """
    if main is None:
        return None
    m=simple.canonical(simple.prediction(main)[:5])
    f=np.asarray(fields,dtype=np.float64); support=np.asarray(support,dtype=np.float64)
    if (f.shape!=(2,9,9) or support.shape!=(9,9) or not np.isfinite(f).all()
            or not np.isfinite(support).all() or np.any(f[1]<0) or np.any((support<0)|(support>1))):
        raise ValueError('Invalid fixed fields/support')
    cb,xy,sides,rotation=transform(base_original,base_model,scale_xy)
    c,s=math.cos(m[4]),math.sin(m[4]); axes=np.array([[c,s],[-s,c]])
    localize=lambda p: ((p*xy-cb[:2])@rotation)/sides
    # Offset is set by source resolution, never by labels or an error threshold.
    derivative=((axes*xy)@rotation)/sides*8
    cache_intervals=np.max(np.abs(derivative),axis=1)
    model_lengths=np.linalg.norm(axes*xy,axis=1)
    offsets=np.maximum(1/cache_intervals,8/model_lengths)
    grid=np.linspace(-.5,.5,9); yy,xx=np.meshgrid(grid,grid,indexing='ij')
    points=m[:2]+xx[...,None]*(m[2]*1.5)*axes[0]+yy[...,None]*(m[3]*1.5)*axes[1]
    local=localize(points)
    # True 3x3 region pooling, not one global average; both arms share fields.
    regions=local.reshape(3,3,3,3,2).transpose(0,2,1,3,4).reshape(3,3,9,2)
    ordinary=pooled(f,support,regions)
    border_points=np.empty((2,2,3,3,2),dtype=float)
    for axis in range(2):
        for side,sign in enumerate((1.,-1.)):
            for band,k in enumerate((-1.,0.,1.)):
                distance=m[2+axis]*.5+k*offsets[axis]
                border_points[axis,side,band]=(m[:2]+sign*distance*axes[axis]
                    +np.array(SAMPLING['tangent_fractions'])[:,None]*m[3-axis]*axes[1-axis])
    border_local=localize(border_points)
    border=pooled(f,support,border_local)
    # If source spacing pushes inner samples beyond the center, this axis has
    # no resolved inner/edge/outer evidence. Keep the frame, expose zero support.
    qualified=offsets<=m[2:4]*.5+1e-12
    border[~qualified]=0.
    return dict(ordinary_roi=ordinary.reshape(-1),opposite_border=border.reshape(-1),
        ordinary_support=ordinary[...,2].reshape(-1),border_support=border[...,2].reshape(-1),
        offsets_original=offsets,axis_resolution_qualified=qualified,cache_band_separation_Linf=offsets*cache_intervals,
        native_P3_band_separation=offsets*model_lengths/8,
        M_short_native_P3_cells=m[3]*model_lengths[1]/8,
        border_local=border_local)


def descriptors(main,image_size,evidence):
    if main is None:
        if evidence is not None: raise ValueError('No main output can have image evidence')
        return {arm:None for arm in SCHEMAS}
    base=simple.descriptor(main,image_size)
    if evidence is None: raise ValueError('Main output requires explicit supported fields')
    result=dict(refit_simple=base)
    for arm in ('ordinary_roi','opposite_border'):
        x=np.asarray(evidence[arm],dtype=float)
        if x.shape!=(len(SCHEMAS[arm])-3,) or not np.isfinite(x).all():
            raise ValueError('Fixed image descriptor schema differs')
        result[arm]=np.r_[base,x]
    return result


def fit(features, labels, arm):
    """Fixed zero-init/class-balanced Newton; fixed arm schema only."""
    x, y = np.asarray(features, dtype=float), np.asarray(labels, dtype=float)
    if (x.ndim != 2 or arm not in SCHEMAS or x.shape[1] != len(SCHEMAS[arm]) or y.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isin(y, [0., 1.]).all()):
        raise ValueError('Invalid fixed fitting inputs')
    bad, good = int(y.sum()), int(len(y)-y.sum())
    if not bad or not good:
        raise ValueError('Both TRAIN error classes required')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-8, 1., scale)
    design = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    sw = np.where(y == 1, .5/bad, .5/good)
    w = np.zeros(x.shape[1]+1)
    initial = simple.logistic_objective(w, design, y, sw, FIT['l2'])[0]
    for step in range(FIT['max_iterations']):
        loss, grad, h = simple.logistic_objective(w, design, y, sw, FIT['l2'])
        if np.max(np.abs(grad)) <= FIT['gradient_tolerance']:
            break
        direction = np.linalg.solve(h, grad)
        for k in range(30):
            candidate = w-.5**k*direction
            value = simple.logistic_objective(candidate, design, y, sw, FIT['l2'])[0]
            if value <= loss-1e-4*.5**k*grad.dot(direction):
                w = candidate
                break
        else:
            raise RuntimeError('Fixed Newton line search failed')
    loss, grad, _ = simple.logistic_objective(w, design, y, sw, FIT['l2'])
    if not np.isfinite(w).all() or np.max(np.abs(grad)) > FIT['gradient_tolerance']:
        raise RuntimeError('Fixed fit did not converge; no retry/search')
    return dict(feature_names=list(SCHEMAS[arm]), mean=mean.tolist(),
                scale=scale.tolist(), weights=w.tolist(), settings=deepcopy(FIT),
                train_rows=len(y), bad_rows=bad, good_rows=good, iterations=step+1,
                initial_objective=initial, final_objective=loss,
                max_abs_gradient=float(np.max(np.abs(grad))), initialization='all_zero')


def risk(model, features, arm):
    names = model['feature_names']
    if arm not in SCHEMAS or names != list(SCHEMAS[arm]) or model['settings'] != FIT:
        raise ValueError('Invalid fixed model contract')
    mean, scale, w = [np.asarray(model[k], dtype=float) for k in ('mean', 'scale', 'weights')]
    x = np.asarray(features, dtype=float)
    n = len(names)
    if (mean.shape != (n,) or scale.shape != (n,) or w.shape != (n+1,)
            or x.ndim not in (1, 2) or x.shape[-1] != n
            or not np.isfinite(np.r_[mean, scale, w]).all() or not np.isfinite(x).all()
            or np.any(scale <= 0)):
        raise ValueError('Invalid model or features')
    return simple.sigmoid(w[0]+((x-mean)/scale).dot(w[1:]))


def decide(model,cutoff,original_decision,features,arm=ARM):
    """Only size flag/risk changes. No GT, role, domain, video or auxiliary API."""
    if not math.isfinite(cutoff) or not 0<=cutoff<=1:
        raise ValueError('Invalid frozen cutoff')
    d=deepcopy(original_decision)
    present=d['final_box_original'] is not None
    if d['center_accepted']!=present: raise ValueError('Original center identity differs')
    if not present:
        if features is not None or d['size_accepted'] or d['angle_accepted'] or any(v is not None for v in d['risks'].values()):
            raise ValueError('Missing output cannot have quality')
        return d
    value=float(risk(model,features,arm)); d['risks']['size']=value; d['size_accepted']=value<=cutoff
    return d


def fit_cutoff(rows,risks):
    """Fit-only global whole-tie cutoff; probe/VAL can NEVER refit it."""
    if not rows or any(r['reliability_role']!='train' or r['sample_role']!='fit' for r in rows):
        raise ValueError('Cutoff is fixed on fit TRAIN only')
    requirements={}
    for name,group in ab.groups(rows).items():
        good=sorted(risks[r['image']] for r in group if r['pred'] is not None and not ab.size_bad(r))
        n=required_good(len(good),.95)
        requirements[name]=dict(good=len(good),required=n,risk_le=good[n-1] if n else None)
    points=[v['risk_le'] for v in requirements.values() if v['risk_le'] is not None]
    if not points: raise ValueError('No fit correct size observations')
    return dict(risk_le=max(points),requirements=requirements,one_global_cutoff=True,
        cutoff_role='fit',probe_calibration=False,deployable=False,correct_retention_target=.95)


def describe(rows,scores,threshold,arm=ARM):
    result={}
    controls=tuple(m for m in CONTROLS if m!=arm)
    for name,group in ab.groups(rows).items():
        out=[r for r in group if r['pred'] is not None]; ids=[r['image'] for r in out]
        bad=[ab.size_bad(r) for r in out]
        accepted={i for i in ids if scores[arm][i]<=threshold}
        value=states.compare(group,accepted,scores,controls)
        value['rank']={m:sep.rank_metrics([scores[m][i] for i in ids],bad) for m in scores}
        value['control_tie_bounds']={}; value['matched_CR_controls']={}
        for m in controls:
            k=len(accepted)
            value['control_tie_bounds'][m]=sep.same_count_reference([scores[m][i] for i in ids],bad,ids,[k])[k]
            target=value['actual']['states']['CR']; good=0; bound=None
            ranked=sorted(out,key=lambda r:(scores[m][r['image']],r['image']))
            if target:
                for r in ranked:
                    good+=int(not ab.size_bad(r))
                    if good>=target: bound=scores[m][r['image']]; break
            kept={i for i in ids if bound is not None and scores[m][i]<=bound}
            value['matched_CR_controls'][m]=states.summarize(group,kept)
        result[name]=value
    return result


def gate(stats):
    failures=[]
    for name,v in stats.items():
        a=v['actual']; c=a['states']; strict=name in ('all','domain:real')
        if c['CR']<required_good(c['CR']+c['FR'],.95):
            failures.append(dict(group=name,check='correct_retention',CR=c['CR'],good=c['CR']+c['FR']))
        for control in CONTROLS:
            bound=v['control_tie_bounds'][control]['bad_min_over_tie']
            if (c['FA']>=bound if strict else c['FA']>bound):
                failures.append(dict(group=name,control=control,check='same_count_FA',candidate=c['FA'],control_best_tie_FA=bound,strict=strict))
            b=v['matched_CR_controls'][control]['states']
            if (c['FA']>=b['FA'] if strict else c['FA']>b['FA']):
                failures.append(dict(group=name,control=control,check='matched_CR_FA',candidate=c['FA'],control_FA=b['FA'],strict=strict))
            if a['runs']['correct_rejection']['longest']>v['same_count_controls'][control]['runs']['correct_rejection']['longest']:
                failures.append(dict(group=name,control=control,check='longest_FR'))
    return dict(passed=not failures,failures=failures,selected=False,TEST_used=False,
        gate_scope='exposed_TRAIN_probe_fit_fixed_cutoffs')
