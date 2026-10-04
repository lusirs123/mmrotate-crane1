"""Frozen midpoint front-end adapter and a bounded image-reference reader.

Neither online API accepts GT. The reference fits the main response to the same
Gaussian family used by the already trained image branch, rather than assuming
that every positive context pixel belongs to that Gaussian. No box is corrected.
"""
from copy import deepcopy
import math

import numpy as np

from crane_project.utils import port_simple_component_reliability_v1 as simple
from crane_project.utils import port_size_reference_v1 as old

VERSION = 'port_midpoint_reliability_v1'


def component_at_peak(mask, peak):
    """Eight-connected main response; no GT or candidate dimensions select it."""
    selected = np.zeros(mask.shape, dtype=bool)
    pending = [tuple(peak)]; selected[tuple(peak)] = True
    while pending:
        y, x = pending.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y+dy, x+dx
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not selected[yy, xx]):
                    selected[yy, xx] = True; pending.append((yy, xx))
    return selected


def template_reference(probability, context_box, meta, protocol):
    """Fixed robust log-Gaussian fit on the connected half-peak core.

    log response = c + b.x + x'Qx, covariance = -.5 inv(Q). This covariance is
    untruncated, so lengths are 4 sigma, without the old moment correction.
    Candidate angle/short edge and GT never enter core selection or fitting.
    """
    points, valid = old.grid(meta); p = np.asarray(probability, dtype=float)
    if p.shape != valid.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected finite probability on the frozen stride2 grid')
    settings = protocol['template_reader']; context = simple.canonical(context_box)
    original = old.points_original(points, meta)
    side = max(protocol['context_multiple']*context[2], protocol['context_min_original_px'])
    region = valid & (np.max(np.abs(original-context[:2]), axis=-1) <= side/2)
    info = dict(defined=False, reason='empty_context', long_original_px=None,
        short_original_px=None, center_original=None, angle_rad=None,
        peak=0., background=0., core_cells=0, fit_log_rmse=None, condition=None)
    if int(region.sum()) < protocol['min_context_cells']: return info
    background = float(np.median(p[region])); peak = float(p[region].max())
    response = np.maximum(p-background, 0.)
    info.update(background=background, peak=peak)
    if peak < protocol['min_peak'] or float(response[region].sum()) < protocol['min_mass']:
        info['reason'] = 'weak_evidence'; return info
    amplitude = float(response[region].max())
    core_mask = region & (response >= settings['core_peak_fraction']*amplitude)
    peak_index = np.unravel_index(np.argmax(np.where(region, response, -1.)), p.shape)
    core = component_at_peak(core_mask, peak_index)
    info['core_cells'] = int(core.sum())
    if info['core_cells'] < settings['min_core_cells']:
        info['reason'] = 'insufficient_core'; return info
    xy = original[core]; origin = xy.mean(axis=0)
    unit = float(np.sqrt(np.mean(np.sum((xy-origin)**2, axis=1))))
    if not math.isfinite(unit) or unit <= 0:
        info['reason'] = 'degenerate_core'; return info
    x, y = ((xy-origin)/unit).T
    design = np.column_stack((np.ones(len(x)), x, y, x*x, x*y, y*y))
    condition = float(np.linalg.cond(design)); info['condition'] = condition
    if not math.isfinite(condition) or condition > settings['max_condition']:
        info['reason'] = 'ill_conditioned_core'; return info
    target = np.log(response[core]); weights = np.ones(len(x))
    for _ in range(settings['irls_steps']):
        coefficients = np.linalg.lstsq(design*np.sqrt(weights[:, None]),
            target*np.sqrt(weights), rcond=None)[0]
        residual = design.dot(coefficients)-target
        weights = np.minimum(1., settings['huber_log_delta']/np.maximum(np.abs(residual), 1e-12))
    rmse = float(np.sqrt(np.mean(residual**2))); info['fit_log_rmse'] = rmse
    q = np.array([[coefficients[3], coefficients[4]/2], [coefficients[4]/2, coefficients[5]]])
    if np.linalg.eigvalsh(q).max() >= -1e-8:
        info['reason'] = 'not_a_gaussian_peak'; return info
    covariance = -.5*np.linalg.inv(q)*unit**2
    eigen, vectors = np.linalg.eigh(covariance)
    mu = origin-.5*np.linalg.solve(q, coefficients[1:3])*unit
    lengths = 4.*np.sqrt(eigen[::-1]); axis = vectors[:, 1]
    theta = (math.atan2(axis[1], axis[0])+math.pi/2) % math.pi-math.pi/2
    info.update(long_original_px=float(lengths[0]), short_original_px=float(lengths[1]),
        center_original=mu.tolist(), angle_rad=float(theta))
    if not np.isfinite(np.r_[lengths, mu]).all():
        info['reason'] = 'nonfinite_fit'; return info
    if rmse > settings['max_fit_log_rmse']:
        info['reason'] = 'poor_template_fit'; return info
    if lengths[1]*math.sqrt(float(np.prod(old.geometry(meta)[:2]))) < protocol['min_reference_short_model_px']:
        info['reason'] = 'unresolved_short_edge'; return info
    margin = 2*old.STRIDE/min(old.geometry(meta)[:2]); height, width = meta['ori_shape'][:2]
    if (np.max(np.abs(mu-context[:2])) >= side/2-margin or
            np.any(xy.min(axis=0) < margin) or xy[:, 0].max() > width-1-margin or
            xy[:, 1].max() > height-1-margin or
            np.max(np.abs(xy-context[:2])) >= side/2-margin):
        info['reason'] = 'core_clipped'; return info
    info.update(defined=True, reason='defined')
    return info


def paired_rows(sources, predictions):
    """Bind genuine formal midpoint outputs to reviewed labels, keeping missing."""
    indexed = {p['image']: p for p in predictions}
    if len(indexed) != len(predictions) or set(indexed) != {r['image'] for r in sources}:
        raise ValueError('Midpoint prediction identities differ')
    result = []
    for source in sources:
        value = indexed[source['image']]
        for key in ('sequence', 'domain', 'frame_id'):
            if value[key] != source[key]: raise ValueError('Midpoint frame metadata differs')
        if not np.allclose(value['gt'], source['gt'], atol=1e-4, rtol=1e-6):
            raise ValueError('Reviewed GT differs from formal cache')
        b, final = simple.prediction(value['b']), simple.prediction(value['midpoint'])
        if (b is None) != (final is None): raise ValueError('Midpoint changed output coverage')
        if b is None and (value['accepted'] is not None or value['candidate'] is not None):
            raise ValueError('Missing B cannot create a candidate/accepted output')
        if b is not None:
            if type(value['accepted']) is not bool:
                raise ValueError('Present B requires a boolean midpoint acceptance')
            if b[5] != final[5]: raise ValueError('Midpoint changed detector score')
            if value['accepted'] is False and value['midpoint'] != value['b']:
                raise ValueError('Midpoint full-frame fallback differs')
            if value['accepted'] is True and value['candidate'] != value['midpoint']:
                raise ValueError('Accepted midpoint candidate differs')
        row = deepcopy(source)
        row.update(pred=deepcopy(value['midpoint']), b_original=deepcopy(value['b']),
            midpoint_accepted=value['accepted'], midpoint_candidate=deepcopy(value['candidate']))
        result.append(row)
    return result


class MidpointReliability:
    """Three flags for the FINAL midpoint box; never writes back to detection."""
    def __init__(self, policy, front_end):
        if policy.get('protocol') != VERSION or policy.get('front_end') != front_end:
            raise ValueError('Reliability policy belongs to another front end')
        self.base = simple.SimpleComponentReliability(policy['simple_policy'])

    def decide(self, final_box_original, image_size, method='simple'):
        value = self.base.decide(final_box_original, image_size, method)
        value['final_box_original'] = value.pop('raw_b_output')
        return value


def fixed_probe_rows(partition):
    selected = []
    for role in ('fit', 'holdout'):
        groups = {}
        for r in partition[role]: groups.setdefault(r['sequence'], []).append(r)
        for seq in sorted(groups):
            values = sorted(groups[seq], key=lambda r: r['frame_id'])
            for r in (values[0], values[-1]): selected.append(dict(r, reference_role=role))
    if len(selected) != 14 or len({r['image'] for r in selected}) != 14:
        raise ValueError('Requires predeclared 10 fitting + 4 held-out TRAIN frames')
    return selected


def offline_map_evidence(probability, gt, final_box, meta, protocol):
    """GT-only diagnostic of signal mass and moment leverage; never online input."""
    p, valid = old.grid(meta); xy = old.points_original(p, meta)
    context = simple.canonical(final_box)
    side = max(protocol['context_multiple']*context[2], protocol['context_min_original_px'])
    region = valid & (np.max(np.abs(xy-context[:2]),axis=-1) <= side/2)
    if not region.any(): return dict(defined=False,reason='empty_context')
    response = np.maximum(np.asarray(probability,float)-np.median(probability[region]),0.)
    weight=response[region]; mass=float(weight.sum())
    if mass <= 0: return dict(defined=False,reason='zero_mass')
    target,_=old.target_map(gt,meta); inside=(target[region]>0)
    points=xy[region]; mu=(points*weight[:,None]).sum(axis=0)/mass
    delta=points-mu; g=simple.canonical(gt)
    c,s=math.cos(g[4]),math.sin(g[4])
    along=delta[:,0]*c+delta[:,1]*s; across=-delta[:,0]*s+delta[:,1]*c
    def fraction(values):
        total=float(values.sum())
        return float(values[~inside].sum()/total) if total>0 else None
    peak=np.unravel_index(np.argmax(np.where(region,response,-1.)),response.shape)
    core=component_at_peak(region & (response>=protocol['template_reader']['core_peak_fraction']*response[peak]),peak)
    return dict(defined=True,response_mass=mass,gt_outside_mass_fraction=fraction(weight),
        gt_outside_long_moment_fraction=fraction(weight*along**2),
        gt_outside_short_moment_fraction=fraction(weight*across**2),
        core_cells=int(core.sum()),core_outside_gt_cells=int((core & (target==0)).sum()),
        response_peak_original=xy[peak].tolist(),peak_center_error_px=float(np.linalg.norm(xy[peak]-g[:2])),
        moment_center_error_px=float(np.linalg.norm(mu-g[:2])),GT_used_only_offline=True)


def save_map_preview(path, probability, target, valid):
    """Same fixed colour scale for predicted map, offline GT target, difference."""
    from PIL import Image, ImageDraw
    height,width=target.shape
    canvas=Image.new('RGB',(3*width,height+24),(24,24,24)); draw=ImageDraw.Draw(canvas)
    for i,(label,values) in enumerate((('prediction',probability),('GT target (offline)',target),
                                     ('absolute difference',np.abs(probability-target)))):
        v=np.clip(np.asarray(values,float),0.,1.)
        rgb=np.stack((v, .3*v, 1-v),axis=-1)
        rgb[~valid]=0
        canvas.paste(Image.fromarray((255*rgb).astype(np.uint8)),(i*width,24))
        draw.text((i*width+5,5),label,fill=(255,255,255))
    canvas.save(path)
