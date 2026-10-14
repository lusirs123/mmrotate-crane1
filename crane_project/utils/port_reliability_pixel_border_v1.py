"""GT-free native-pixel border evidence; descriptive TRAIN screening only."""
import math
import numpy as np
from crane_project.utils.port_reliability_sparse_candidate_v1 import pairs

VERSION = 'port_reliability_pixel_border_v1'
SETTINGS = dict(tangents=7, tangent_extent=.30, radial_points=129,
                radial_min=.25, radial_max=.75, derivative_pixels=1.,
                min_valid_profiles=5, low_temporal=.0785511443740936)


def sample(im, points):
    """Bilinear interpolation, explicit unsupported mask, no extrapolation."""
    p = np.asarray(points, dtype=float)
    h, w = im.shape
    valid = ((p[..., 0] >= 0) & (p[..., 0] <= w-1) &
             (p[..., 1] >= 0) & (p[..., 1] <= h-1))
    x = np.clip(p[..., 0], 0, w-1); y = np.clip(p[..., 1], 0, h-1)
    x0 = x.astype(int); y0 = y.astype(int)
    x1 = np.minimum(x0+1, w-1); y1 = np.minimum(y0+1, h-1)
    a = x-x0; b = y-y0
    v = (im[y0,x0]*(1-a)+im[y0,x1]*a)*(1-b)+(im[y1,x0]*(1-a)+im[y1,x1]*a)*b
    return v, valid


def evidence(gray, pred):
    """Only native image and delivered box. Never labels/history/video identity."""
    if pred is None:
        return None
    im = np.asarray(gray, dtype=float)
    b = np.asarray(pred, dtype=float)
    if im.ndim != 2 or not np.isfinite(im).all() or b.shape not in ((5,), (6,)) or not np.isfinite(b).all() or min(b[2:4]) <= 0:
        raise ValueError('Finite native gray image and positive OBB required')
    cx, cy, L, S, angle = b[:5]
    if L < S:
        L, S, angle = S, L, angle+math.pi/2
    angle = (angle+math.pi/2) % math.pi-math.pi/2
    u = np.array([math.cos(angle),math.sin(angle)]); v = np.array([-u[1],u[0]])
    t = np.linspace(-SETTINGS['tangent_extent'], SETTINGS['tangent_extent'], SETTINGS['tangents'])
    radius = np.linspace(SETTINGS['radial_min'], SETTINGS['radial_max'], SETTINGS['radial_points'])
    result = {}
    for name, normal, tangent, size, span in [('L',u,v,L,S),('S',v,u,S,L)]:
        sides = []
        for sign in (-1,1):
            pts = np.array([cx,cy])+t[:,None,None]*span*tangent+sign*radius[None,:,None]*size*normal
            delta = SETTINGS['derivative_pixels']
            hi, a = sample(im,pts+delta*normal); lo, bmask = sample(im,pts-delta*normal)
            g = np.abs(hi-lo)/(2*delta)
            ok = (a & bmask).all(axis=1) & (g.max(axis=1)>1e-8)
            offsets=[]; prominence=[]; peak_positions=[]
            for i in np.flatnonzero(ok):
                # Tied maxima use their central location, no GT/direction preference.
                peak = float(np.median(radius[np.isclose(g[i],g[i].max(),rtol=0,atol=1e-12)]))
                peak_positions.append(peak)
                offsets.append(math.log(2*peak))
                prominence.append(float(g[i].max()/(g[i].mean()+1e-12)))
            sides.append(dict(valid_profiles=int(ok.sum()), signed_log_offsets=offsets,
                              peak_fractions=peak_positions, prominence=prominence))
        supported = all(s['valid_profiles']>=SETTINGS['min_valid_profiles'] for s in sides)
        vals = [x for side in sides for x in side['signed_log_offsets']]
        result[name] = dict(supported=supported, sides=sides,
            mismatch=float(np.median(np.abs(vals))) if supported else None,
            signed_offset=float(np.median(vals)) if supported else None,
            dispersion=float(np.median(np.abs(np.array(vals)-np.median(vals)))) if supported else None)
    return result


def analyze(rows):
    """Prespecified positive mismatch direction, labels only here offline."""
    low = [r for r in rows if r['low_temporal'] and r['wrong'] is not None]
    groups = {'all':low}
    for key in ('domain','sequence'):
        for val in sorted({r[key] for r in low}):
            groups[key+':'+val]=[r for r in low if r[key]==val]
    result = {}
    for name, part in groups.items():
        axes={}
        for axis in ('L','S'):
            valid=[r for r in part if r['evidence'] and r['evidence'][axis]['supported']]
            bad=[r['evidence'][axis]['mismatch'] for r in valid if r['wrong']]
            good=[r['evidence'][axis]['mismatch'] for r in valid if not r['wrong']]
            accepted=[r for r in valid if r['accepted']]
            # Match video and predicted geometric size within +/-10%; not GT size.
            wins=0.; count=0
            for r in valid:
                if not r['wrong']:continue
                controls=[q for q in valid if not q['wrong'] and q['sequence']==r['sequence'] and abs(math.log(q['scale']/r['scale']))<=math.log(1.1)]
                p=pairs([r['evidence'][axis]['mismatch']],[q['evidence'][axis]['mismatch'] for q in controls])
                wins+=p['wins'];count+=p['pairs']
            axes[axis]=dict(supported=len(valid), pooled=pairs(bad,good),
                size_matched_pairs=count, size_matched_auc=wins/count if count else None,
                remaining_FA_vs_CR=pairs([r['evidence'][axis]['mismatch'] for r in accepted if r['wrong']],
                                        [r['evidence'][axis]['mismatch'] for r in accepted if not r['wrong']]))
        result[name]=dict(n=len(part), wrong=sum(r['wrong'] for r in part),
            error_types={k:sum(r['error_type']==k for r in part) for k in ('L_only','S_only','both','correct')},axes=axes)
    # Feasibility only, no fitted rule/VAL phase exists in this executor.
    qualifying={}
    for axis in ('L','S'):
        qualifying[axis]=[name for name,g in result.items() if name.startswith('sequence:real_')
            and g['axes'][axis]['pooled']['wrong']>=3 and g['axes'][axis]['pooled']['good']>=20
            and g['axes'][axis]['size_matched_auc'] is not None and g['axes'][axis]['size_matched_auc']>=.55]
    passed=any(result['all']['axes'][a]['pooled']['wrong']>=20 and len(qualifying[a])>=2 for a in ('L','S'))
    return dict(groups=result, qualifying_videos=qualifying, feasibility_passed=passed,
                status='TRAIN_SCREEN_ONLY_NO_FIT', caution='Correlated previously exposed TRAIN; descriptive evidence, no performance gain or independent validation.')
