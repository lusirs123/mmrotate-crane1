"""GT-fixed half-peak range samples for TRAIN; existing reader stays image-only."""
import math
import numpy as np
from crane_project.utils import port_size_reference_v1 as geometry
from crane_project.utils.port_size_core_curvature_v2 import (
    GRADIENT_CHECK, gradient_consistency, acceptance_bound, online_reading)

VERSION = 'port_size_halfpeak_range_v3'
ARMS = ('a0', 'a1')
COEFFICIENTS = dict(v1=1., inner=.125, outer=.125)
RANGE_SETTINGS = dict(radial_factors=[.85, 1., 1.15], transverse_sigma_offsets=[-.25, 0., .25],
    half_peak_fraction=.5, contrast_floor=.05, smooth_l1_beta=.1,
    GT_min_short_model_px=16., min_background_cells=16,
    context_multiple=3., context_min_original_px=64.)


def projection(gt, meta):
    """GT supplies ONLY fixed positions/interpolation/targets, not predictions.

    Coordinates are sigma-normalized. Along either edge and at transverse t,
    the half-peak location is sqrt(2 log 2 - t^2). Both signs of each edge are
    sampled at .85/1/1.15 times that location, with equal edge/side weights.
    The target uses the same bilinear grid sampling, removing grid-phase bias.
    """
    points, valid = geometry.grid(meta); b = geometry.box_model(gt, meta)
    target, _ = geometry.target_map(gt, meta)
    original = geometry.points_original(points, meta); g = geometry.canonical(gt)
    side = max(RANGE_SETTINGS['context_multiple']*g[2], RANGE_SETTINGS['context_min_original_px'])
    background = valid & (target == 0) & (np.max(np.abs(original-g[:2]), axis=-1) <= side/2)
    info = dict(eligible=False, reason='unresolved_GT_short', cells=36,
        short_model_px=float(b[3]), background_cells=int(background.sum()))
    if b[3] < RANGE_SETTINGS['GT_min_short_model_px']: return info
    if info['background_cells'] < RANGE_SETTINGS['min_background_cells']:
        info['reason']='insufficient_GT_background'; return info
    normalized=[]; inner=[]; outer=[]
    for axis in (0, 1):
        for sign in (-1, 1):
            for transverse in RANGE_SETTINGS['transverse_sigma_offsets']:
                half=math.sqrt(2*math.log(2)-transverse**2)
                for k,factor in enumerate(RANGE_SETTINGS['radial_factors']):
                    coordinate=[transverse, transverse]; coordinate[axis]=sign*factor*half
                    index=len(normalized);normalized.append(coordinate)
                    if k <= 1: inner.append(index)
                    if k >= 1: outer.append(index)
    normalized=np.asarray(normalized+[[0.,0.]],dtype=float)
    delta=normalized*b[2:4]/4.;c,s=math.cos(b[4]),math.sin(b[4])
    xy=b[:2]+delta @ np.array([[c,s],[-s,c]])
    # Frozen stride2 map origin is -3, not 0; no grid_sample convention drift.
    index=(xy+3.)/2.; low=np.floor(index).astype(int); fraction=index-low
    indices=[];weights=[];height,width=valid.shape
    for (x,y),(fx,fy) in zip(low,fraction):
        neighbours=[(y,x),(y,x+1),(y+1,x),(y+1,x+1)]
        if any(not (0<=yy<height and 0<=xx<width and valid[yy,xx]) for yy,xx in neighbours):
            info['reason']='GT_sample_footprint_clipped';return info
        indices.append([yy*width+xx for yy,xx in neighbours])
        weights.append([(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy])
    indices=np.asarray(indices,dtype=np.int64);weights=np.asarray(weights,dtype=float)
    sampled=(target.ravel()[indices]*weights).sum(axis=1)
    if sampled[-1] <= RANGE_SETTINGS['contrast_floor']:
        info['reason']='unresolved_GT_center';return info
    info.update(eligible=True,reason='eligible',indices=indices,weights=weights,
        background_mask=background,shape=valid.shape,normalized=normalized,
        sample_target=sampled[:-1]/sampled[-1],inner=np.asarray(inner),outer=np.asarray(outer))
    return info


def public_projection(value):
    return {k:value[k] for k in ('eligible','reason','cells','short_model_px','background_cells')}


def huber(value):
    value=np.asarray(value,dtype=float)
    return np.where(np.abs(value)<=.1,value**2/.2,np.abs(value)-.05)


def numpy_terms(probability, constants):
    """Independent NumPy forward oracle, matching torch.median's lower median."""
    p=np.asarray(probability,dtype=float)
    if p.shape != constants['shape'] or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid range probability/shape')
    values=p[constants['background_mask']];background=float(np.partition(values,(len(values)-1)//2)[(len(values)-1)//2])
    samples=(p.ravel()[constants['indices']]*constants['weights']).sum(axis=1)
    amplitude=float(samples[-1]-background)
    denominator=max(amplitude,RANGE_SETTINGS['contrast_floor'])
    ratios=(samples[:-1]-background)/denominator
    errors=ratios-constants['sample_target']
    return dict(inner=float(huber(errors[constants['inner']]).mean()),
        outer=float(huber(errors[constants['outer']]).mean()),background=background,
        center_probability=float(samples[-1]),amplitude=amplitude,
        contrast_floor_active=amplitude<RANGE_SETTINGS['contrast_floor'],
        mean_absolute_profile_error=float(np.abs(errors).mean()),ratios=ratios)


def synthetic_probability(constants, factors=(1.,1.), amplitude=.8, background=.01):
    """Full-grid toy Gaussian; NOT learned-image evidence or online GT input."""
    target=constants['_toy_target'] if '_toy_target' in constants else None
    if target is None: raise ValueError('Numerical probe needs its fixed GT toy coordinates')
    x,y=target;f0,f1=factors
    response=np.exp(-.5*((x/f0)**2+(y/f1)**2))
    response*=((np.abs(x)<=2*f0)&(np.abs(y)<=2*f1))
    return background+amplitude*response


def toy_constants(gt, meta):
    constants=projection(gt,meta)
    if not constants['eligible']:return constants
    points,_=geometry.grid(meta);b=geometry.box_model(gt,meta);d=points-b[:2]
    c,s=math.cos(b[4]),math.sin(b[4])
    constants['_toy_target']=(4*(d[...,0]*c+d[...,1]*s)/b[2],4*(-d[...,0]*s+d[...,1]*c)/b[3])
    return constants


def numerical_probe(constants):
    ideal=numpy_terms(synthetic_probability(constants),constants);cases=[]
    def objective(factors):
        values=numpy_terms(synthetic_probability(constants,factors),constants)
        return .125*(values['inner']+values['outer'])
    for axis in (0,1):
        for factor in (.85,1.15):
            factors=[1.,1.];factors[axis]=factor
            positive=factors.copy();negative=factors.copy();positive[axis]+=1e-5;negative[axis]-=1e-5
            derivative=(objective(positive)-objective(negative))/2e-5
            cases.append(dict(axis=axis,factor=factor,scale_derivative=float(derivative),
                passed=bool(math.isfinite(derivative) and derivative*(factor-1)>0)))
    boundary=constants['sample_target'][np.intersect1d(constants['inner'],constants['outer'])]
    error=max(ideal['inner'],ideal['outer']);half_error=float(np.max(np.abs(boundary-.5)))
    return dict(ideal_loss_max=error,half_peak_target_max_grid_error=half_error,cases=cases,
        passed=bool(error<=1e-12 and half_error<=.03 and all(r['passed'] for r in cases)))


def offline_evidence(probability, gt, meta):
    """Attach GT proxy diagnostics AFTER online reading; never gates a flag."""
    constants=projection(gt,meta)
    info=public_projection(constants)
    if constants['eligible']:
        terms=numpy_terms(probability,constants)
        info.update({k:v for k,v in terms.items() if k!='ratios'})
    return info
