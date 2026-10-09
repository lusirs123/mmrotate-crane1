"""Fixed source comparison; TRAIN-only PCA, identical size residual heads."""
from copy import deepcopy
import numpy as np
from crane_project.utils import port_reliability_readout_compare_v1 as previous
from crane_project.utils import port_reliability_redc_size_v1 as old

VERSION = 'port_reliability_feature_source_v1'
ARMS = ('midpoint', 'native')
CONTROLS = ('midpoint', 'full_simple', 'score_only')
SETTINGS = dict(seed=1701, epochs=100, batch_size=256, lr=.001,
    weight_decay=.0001, clip_norm=5., pca_dimensions=256,
    hidden=[16,8], correct_retention=.95, parameter_count=4290)
class_weights = old.class_weights
decide = old.decide
describe = old.describe


def fit_pca(raw, role):
    x = np.asarray(raw, dtype=np.float64)
    if role != 'TRAIN' or x.ndim != 2 or min(x.shape) < 256 or not np.isfinite(x).all():
        raise ValueError('PCA requires finite full TRAIN with at least256 dimensions')
    mean = x.mean(0); centered = x-mean
    _, singular, v = np.linalg.svd(centered, full_matrices=False)
    components = v[:256].copy()
    signs = np.sign(components[np.arange(256), np.abs(components).argmax(1)])
    components *= signs[:,None]
    return dict(mean=mean.tolist(), components=components.tolist(), train_count=len(x),
        raw_dimensions=x.shape[1], dimensions=256, whiten=False,
        eigenvalues=(singular[:256]**2/(len(x)-1)).tolist(),
        explained_variance_fraction=float((singular[:256]**2).sum()/(singular**2).sum()))


def project(raw, pca):
    x = np.asarray(raw, dtype=np.float64); c = np.asarray(pca['components']); mean = np.asarray(pca['mean'])
    if (x.ndim != 2 or x.shape[1] != len(mean) or c.shape != (256,len(mean))
            or not np.isfinite(x).all() or not np.isfinite(c).all() or not np.isfinite(mean).all()):
        raise ValueError('PCA inference dimension/value mismatch')
    return (x-mean)@c.T


def features(descriptors, raw, pca):
    d = np.asarray(descriptors, dtype=np.float64)
    if d.shape != (len(raw),3) or not np.isfinite(d).all():
        raise ValueError('Delivered M descriptor pairing mismatch')
    return np.c_[d,project(raw,pca)]


def normalization(train):
    x = np.asarray(train, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 259 or not len(x) or not np.isfinite(x).all():
        raise ValueError('TRAIN normalization shape/value mismatch')
    scale = x.std(0)
    return dict(mean=x.mean(0).tolist(),scale=np.where(scale<1e-8,1.,scale).tolist())


def normalize(features, normalizer):
    x = np.asarray(features, dtype=np.float64)
    mean, scale = (np.asarray(normalizer[k]) for k in ('mean','scale'))
    if (x.shape[-1:] != (259,) or mean.shape != (259,) or scale.shape != (259,)
            or not np.isfinite(x).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale<=0)):
        raise ValueError('Invalid259-dimensional standardization')
    return (x-mean)/scale


def numpy_logits(model, features, normalizer):
    h = normalize(features,normalizer)[:,1:]
    for n in (0,2,4):
        h = h@np.asarray(model['network.%d.weight'%n]).T+model['network.%d.bias'%n]
        if n != 4: h = np.maximum(h,0.)
    return -np.asarray(features)[:,0]+h[:,0]+float(model['beta'][0])


def gate(statistics):
    # Reuse the predeclared conservative matching semantics, replace only the
    # fair control name. No selection of midpoint as an alternative winner.
    translated = deepcopy(statistics)
    for value in translated.values():
        for key in ('control_tie_bounds','matched_CR_controls'):
            value[key]['temperature'] = value[key].pop('midpoint')
    value = previous.gate(translated)
    for failure in value['failures']:
        if failure.get('control') == 'temperature': failure['control'] = 'midpoint'
    value['selected_arm'] = 'native' if value['passed'] else None
    return value


def patch_reference(feature_map, y, x):
    """Independent CPU channel-major patch, including true convolution zeros."""
    a = np.asarray(feature_map)
    if a.ndim != 3 or a.shape[0] != 256 or not (0<=y<a.shape[1] and 0<=x<a.shape[2]):
        raise ValueError('Invalid winning cell')
    return np.pad(a,((0,0),(1,1),(1,1)))[:,y:y+3,x:x+3].reshape(2304)
