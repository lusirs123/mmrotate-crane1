"""NumPy geometry for a bounded PQA-inspired SIZE reference experiment.

Online functions take predicted image evidence and a B context, never GT.
OBB-derived targets are geometry proxies, not visible masks. No box correction,
classification probability, deployment cutoff, or candidate ordering is emitted.
"""
import math

import numpy as np

from crane_project.utils.port_simple_component_reliability_v1 import canonical

VERSION = 'port_size_reference_v1'
STRIDE = 2
# A Gaussian truncated to +/-2 sigma in each principal direction has this
# variance factor. Using 4*sqrt(moment) without this correction biases size.
TRUNCATED_VARIANCE = 1.-4.*math.exp(-2.)/math.sqrt(2.*math.pi)/math.erf(math.sqrt(2.))


def geometry(meta):
    scale = np.asarray(meta['scale_factor'], dtype=float)
    ih, iw = meta['img_shape'][:2]; oh, ow = meta['ori_shape'][:2]
    ph, pw = meta['pad_shape'][:2]
    if min(ih, iw, oh, ow, ph, pw) <= 0: raise ValueError('Empty image geometry')
    if (scale.shape != (4,) or not np.isfinite(scale).all() or np.any(scale <= 0)
            or scale[0] != scale[2] or scale[1] != scale[3]
            or abs(scale[0]-scale[1]) > 1./min(oh, ow)+1e-6
            or min(ih, iw, oh, ow) <= 0 or ih > ph or iw > pw
            or ph % STRIDE or pw % STRIDE):
        raise ValueError('Only the frozen isotropic B geometry is supported')
    if meta.get('flip', False) and meta.get('flip_direction') not in ('horizontal', 'vertical', 'diagonal'):
        raise ValueError('Unknown flip')
    return scale


def points_original(points, meta):
    scale = geometry(meta); p = np.asarray(points, dtype=float).copy()
    if meta.get('flip', False):
        direction = meta['flip_direction']
        if direction in ('horizontal', 'diagonal'): p[..., 0] = meta['img_shape'][1]-1-p[..., 0]
        if direction in ('vertical', 'diagonal'): p[..., 1] = meta['img_shape'][0]-1-p[..., 1]
    return p/scale[:2]


def box_model(box, meta):
    scale = geometry(meta); b = canonical(box)
    b[:2] *= scale[:2]; b[2:4] *= math.sqrt(scale[0]*scale[1])
    if meta.get('flip', False):
        direction = meta['flip_direction']
        if direction in ('horizontal', 'diagonal'): b[0] = meta['img_shape'][1]-1-b[0]
        if direction in ('vertical', 'diagonal'): b[1] = meta['img_shape'][0]-1-b[1]
        if direction != 'diagonal': b[4] = (math.pi-b[4]+math.pi/2) % math.pi-math.pi/2
    return b


def grid(meta):
    geometry(meta); ph, pw = meta['pad_shape'][:2]
    y, x = np.mgrid[:ph//STRIDE, :pw//STRIDE]
    # P3 has anchor coordinate i*8. align_corners=False interpolation gives
    # high-resolution cell coordinate (j+.5)*2-4, not j*2.
    points = np.stack((x, y), axis=-1)*STRIDE+(STRIDE-8)/2.
    ih, iw = meta['img_shape'][:2]
    valid = (points[..., 0] >= 0) & (points[..., 1] >= 0) & (points[..., 0] < iw) & (points[..., 1] < ih)
    return points, valid


def target_map(gt_original, meta):
    """Offline full-OBB truncated Gaussian; no constant short-edge/axis tube."""
    p, valid = grid(meta); b = box_model(gt_original, meta)
    c, s = math.cos(b[4]), math.sin(b[4]); d = p-b[:2]
    u = d[..., 0]*c+d[..., 1]*s; v = -d[..., 0]*s+d[..., 1]*c
    inside = (np.abs(u) <= b[2]/2) & (np.abs(v) <= b[3]/2) & valid
    target = np.exp(-.5*((u/(b[2]/4))**2+(v/(b[3]/4))**2))*inside
    return target.astype(np.float32), valid


def loss_numpy(logits, target, valid):
    """Same target-preserving soft focal BCE as the Torch reference learner."""
    z, t = np.asarray(logits, float), np.asarray(target, float)
    v = np.asarray(valid, bool)
    if z.shape != t.shape or v.shape != t.shape or not np.isfinite(z).all() or not np.isfinite(t).all() or np.any((t < 0) | (t > 1)):
        raise ValueError('Invalid reference loss inputs')
    mass = float(t[v].sum())
    if mass <= 0: raise ValueError('No valid positive geometry target')
    p = np.exp(-np.logaddexp(0., -z))
    positive = t > 0
    weight = np.where(positive, .25*np.abs(t-p), .75*p**2)
    return float(((np.logaddexp(0., z)-t*z)*weight)[v].sum()/mass)


def reference_from_map(probability, context_box, meta, settings):
    """Infer reference principal sizes ONCE per genuine B context.

    All offline candidates share this fixed context, so changing a probe cannot
    change what is called image evidence. It is independent of candidate angle
    and short edge. Undefined evidence is explicit, never a trusted size.
    """
    p, valid = grid(meta); a = np.asarray(probability, dtype=float)
    if a.shape != valid.shape or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError('Expected a finite probability map on the stride2 grid')
    b = canonical(context_box); original = points_original(p, meta)
    side = max(settings['context_multiple']*b[2], settings['context_min_original_px'])
    region = valid & (np.abs(original[..., 0]-b[0]) <= side/2) & (np.abs(original[..., 1]-b[1]) <= side/2)
    info = dict(defined=False, reason='empty_context', long_original_px=None, short_original_px=None,
                center_original=None, angle_rad=None, peak=0., mass=0., background=0., edge_mass_fraction=None)
    if int(region.sum()) < settings['min_context_cells']: return info
    values = a[region]; background = float(np.median(values)); peak = float(values.max())
    weights = np.maximum(values-background, 0.)
    mass = float(weights.sum()); info.update(peak=peak, mass=mass, background=background)
    if peak < settings['min_peak'] or mass < settings['min_mass']:
        info['reason'] = 'weak_evidence'; return info
    xy = original[region]; mu = (weights[:, None]*xy).sum(axis=0)/mass
    d = xy-mu; cov = (d*weights[:, None]).T.dot(d)/mass
    eigen, vectors = np.linalg.eigh(cov)
    if eigen[0] <= 0 or not np.isfinite(eigen).all():
        info['reason'] = 'degenerate_moments'; return info
    sizes = 4.*np.sqrt(eigen[::-1]/TRUNCATED_VARIANCE)
    u = vectors[:, 1]; angle = (math.atan2(u[1], u[0])+math.pi/2) % math.pi-math.pi/2
    margin = STRIDE/min(geometry(meta)[:2])*2
    ow, oh = meta['ori_shape'][1], meta['ori_shape'][0]
    edge = ((np.max(np.abs(xy-b[:2]), axis=1) >= side/2-margin)
            | (np.min(xy, axis=1) < margin)
            | (xy[:, 0] > ow-1-margin) | (xy[:, 1] > oh-1-margin))
    edge_fraction = float(weights[edge].sum()/mass)
    info.update(long_original_px=float(sizes[0]), short_original_px=float(sizes[1]),
        center_original=mu.tolist(), angle_rad=float(angle), edge_mass_fraction=edge_fraction)
    short_model = sizes[1]*math.sqrt(float(np.prod(geometry(meta)[:2])))
    if short_model < settings['min_reference_short_model_px']:
        info['reason'] = 'unresolved_short_edge'; return info
    if edge_fraction > settings['max_edge_mass_fraction']:
        info['reason'] = 'context_clipped_or_contaminated'; return info
    info.update(defined=True, reason='defined')
    return info


def size_reading(candidate_original, reference, risk_scale=.1):
    """Size-only canonical length ratios. Not PQA whole-box IoU or calibrated P."""
    b = canonical(candidate_original)
    if not reference['defined']:
        return dict(risk=None, long_log_ratio=None, short_log_ratio=None, defined=False)
    r = np.asarray([reference['long_original_px'], reference['short_original_px']], float)
    if not np.isfinite(r).all() or np.any(r <= 0) or not math.isfinite(risk_scale) or risk_scale <= 0:
        raise ValueError('Invalid reference size/risk scale')
    delta = np.log(b[2:4]/r)
    return dict(risk=float(-math.expm1(-float(np.abs(delta).max())/risk_scale)),
        long_log_ratio=float(delta[0]), short_log_ratio=float(delta[1]), defined=True)


def probes(gt):
    b = canonical(gt); out = [('gt', b.copy())]
    for index, label in ((2, 'long'), (3, 'short')):
        for sign in (-1, 1):
            p = b.copy(); p[index] *= 1+sign*.15; out.append((label+('_minus' if sign < 0 else '_plus'), p))
    for axis in (0, 1):
        p = b.copy(); p[axis] += 15.; out.append(('center_'+str(axis), p))
    for sign in (-1, 1):
        p = b.copy(); p[4] += sign*math.radians(5); out.append(('angle_'+str(sign), p))
    p = b.copy(); p[2:4] = p[[3, 2]]; p[4] += math.pi/2
    out.append(('equivalent', p))
    return out


def partition(rows, settings):
    """Fixed grouped split before fitting; labels never choose frames."""
    grouped = {}
    for r in rows: grouped.setdefault(r['sequence'], []).append(r)
    if set(grouped) != set(settings['train_sequences']): raise ValueError('Unexpected TRAIN sequence roles')
    for seq in grouped: grouped[seq].sort(key=lambda r: r['frame_id'])
    fit, holdout, guard = [], [], []
    for seq, values in sorted(grouped.items()):
        if seq == settings['holdout_real_sequence']:
            holdout.extend(values); continue
        if seq == 'sim_seq08':
            start = len(values)-settings['sim_holdout_frames']
            holdout.extend(values[start:]); guard.extend(values[start-settings['sim_guard_frames']:start])
            values = values[:start-settings['sim_guard_frames']]
            cap = settings['sim_fit_frames']
        else: cap = settings['real_fit_frames_per_sequence']
        indices = np.linspace(0, len(values)-1, cap).astype(int)
        if len(set(indices)) != cap: raise ValueError('Insufficient fixed fitting rows')
        fit.extend(values[int(i)] for i in indices)
    return dict(fit=fit, holdout=holdout, guard=guard)
