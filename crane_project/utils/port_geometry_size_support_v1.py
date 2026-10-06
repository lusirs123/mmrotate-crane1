"""Read-only original-coordinate size diagnostics. Standard library only.

GT is offline evidence. This module trains no model and creates no policy.
Consecutive frame-index runs are not independent scenes or physical time.
"""
from collections import Counter
import math

VERSION = 'port_geometry_size_support_v1'
COUNTS = {
    'train': dict(real_seq01=339, real_seq05=560, real_seq06=466,
                  real_seq12=141, real_seq13=304, sim_seq08=748),
    'val': dict(real_seq07=226, real_seq14=149, sim_seq10=512)}
SETTINGS = dict(size_relative_limit=.1, center_px_limit=15.,
                annotation_pixel_tolerance=.25,
                known_sampling_step_seconds=1./3., sampling_gap_multiple=2.)
FLAGS = ('size_wrong', 'long_only_wrong', 'short_only_wrong', 'both_wrong',
         'both_small', 'both_large', 'short_too_small', 'long_too_small',
         'opposite_signed_axes', 'ratio_relative_over_10pct',
         'size_relative_over_20pct', 'size_relative_over_30pct', 'size_relative_over_50pct')


def finite(values, length, name):
    if (not isinstance(values, list) or len(values) != length or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) for v in values)):
        raise ValueError('Expected finite '+name)
    return values


def canonical(box):
    finite(box, 5, 'OBB')
    if min(box[2:4]) <= 0:
        raise ValueError('Nonpositive OBB edges')
    cx, cy, w, h, angle = box
    if w < h:
        w, h, angle = h, w, angle+math.pi/2
    return [cx, cy, w, h, (angle+math.pi/2) % math.pi-math.pi/2]


def prediction(box):
    if box is None:
        return None
    finite(box, 6, 'prediction')
    if not .05 < box[5] <= 1:
        raise ValueError('Prediction score outside frozen output contract')
    return canonical(box[:5])


def checked_rows(rows, counts=COUNTS):
    if len({r['image'] for r in rows}) != len(rows):
        raise ValueError('Duplicate collection image')
    for role, expected in counts.items():
        if Counter(r['sequence'] for r in rows if r['reliability_role'] == role) != Counter(expected):
            raise ValueError('Collection role/sequence counts differ: '+role)
    for r in rows:
        role = r['reliability_role']
        sequence, number = r['image'].rsplit('_', 1)
        if (role not in counts or sequence not in counts[role] or
                sequence != r['sequence'] or sequence.split('_')[0] != r['domain'] or
                not number.isdigit() or len(number) != 5 or
                type(r['frame_id']) is not int or int(number) != r['frame_id'] or
                r['split'] != ('train_sim' if role == 'train' and r['domain'] == 'sim' else role)):
            raise ValueError('Non-allowlisted TRAIN/VAL metadata: '+r['image'])
        finite(r['image_size'], 2, 'original image size')
        if min(r['image_size']) <= 0:
            raise ValueError('Nonpositive image size')
        canonical(r['gt'])
        for key in ('pred', 'b_original', 'midpoint_candidate'):
            prediction(r[key])
        if any(type(r[k]) is not bool for k in
               ('train_angle_eligible', 'angle_axis_well_defined')):
            raise ValueError('Nonboolean collection decision')
        b, m, candidate = r['b_original'], r['pred'], r['midpoint_candidate']
        if (b is None) != (m is None) or (b is None) != (candidate is None):
            raise ValueError('Frozen B/M output presence differs')
        if b is None:
            if r['midpoint_accepted'] is not None and r['midpoint_accepted'] is not False:
                raise ValueError('Missing output accepted by midpoint')
        elif (type(r['midpoint_accepted']) is not bool or b[5] != m[5] or b[5] != candidate[5] or
              m != (candidate if r['midpoint_accepted'] else b)):
            raise ValueError('Frozen midpoint score/delivery changed')
    return rows


def size_errors(gt, box):
    g, p = canonical(gt), prediction(box)
    if p is None:
        return None
    el, es = math.log(p[2])-math.log(g[2]), math.log(p[3])-math.log(g[3])
    # Match the existing evaluator at exact 10% boundaries (avoid x/y-1
    # cancellation turning 110 vs100 into 0.10000000000000009).
    rl, rs = (p[2]-g[2])/g[2], (p[3]-g[3])/g[3]
    ratio = math.expm1(el-es)
    limit = SETTINGS['size_relative_limit']
    wl, ws = abs(rl) > limit, abs(rs) > limit
    return dict(long_relative=rl, short_relative=rs, long_log=el, short_log=es,
        scale_log=(el+es)/2, ratio_log=el-es, ratio_relative=ratio,
        size_max_relative=max(abs(rl), abs(rs)),
        center_px=math.hypot(p[0]-g[0], p[1]-g[1]),
        gt_long_px=g[2], gt_short_px=g[3], gt_aspect=g[2]/g[3],
        size_wrong=wl or ws, long_only_wrong=wl and not ws,
        short_only_wrong=ws and not wl, both_wrong=wl and ws,
        both_small=rl < -limit and rs < -limit,
        both_large=rl > limit and rs > limit,
        short_too_small=rs < -limit, long_too_small=rl < -limit,
        opposite_signed_axes=rl*rs < 0,
        ratio_relative_over_10pct=abs(ratio) > limit,
        size_relative_over_20pct=max(abs(rl),abs(rs)) > .2,
        size_relative_over_30pct=max(abs(rl),abs(rs)) > .3,
        size_relative_over_50pct=max(abs(rl),abs(rs)) > .5)


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*q
    lower = int(index)
    return values[lower]+(values[min(lower+1, len(values)-1)]-values[lower])*(index-lower)


def distribution(values):
    if not values:
        return dict(n=0, mean=None, median=None, p05=None, p95=None,
                    mean_abs=None, p95_abs=None, max_abs=None)
    return dict(n=len(values), mean=math.fsum(values)/len(values),
        median=quantile(values, .5), p05=quantile(values, .05), p95=quantile(values, .95),
        mean_abs=math.fsum(abs(v) for v in values)/len(values),
        p95_abs=quantile([abs(v) for v in values], .95), max_abs=max(abs(v) for v in values))


def summary(records, arm):
    errors = [r[arm] for r in records if r[arm] is not None]
    n, output = len(records), len(errors)
    correct = sum(not e['size_wrong'] for e in errors)
    center = sum(e['center_px'] < SETTINGS['center_px_limit'] for e in errors)
    return dict(frames=n, outputs=output, missing=n-output,
        output_coverage=output/n if n else None,
        center_hits_on_outputs=center,
        center_hit_rate_on_outputs=center/output if output else None,
        all_frame_center_correct_coverage=center/n if n else None,
        both_size_correct=correct, both_size_correct_on_outputs=correct/output if output else None,
        all_frame_both_size_correct_coverage=correct/n if n else None,
        error_flags={flag: sum(e[flag] for e in errors) for flag in FLAGS},
        continuous={key: distribution([e[key] for e in errors]) for key in
            ('long_relative', 'short_relative', 'long_log', 'short_log',
             'scale_log', 'ratio_log', 'ratio_relative', 'size_max_relative',
             'gt_long_px', 'gt_short_px', 'gt_aspect')})


def summaries(records):
    groups = {}
    for role in ('train', 'val'):
        subset = [r for r in records if r['role'] == role]
        groups[role+'/all'] = subset
        for field in ('domain', 'sequence', 'annotation_family'):
            for name in sorted({r[field] for r in subset}):
                groups[role+'/'+field+':'+name] = [r for r in subset if r[field] == name]
        if role == 'train':
            for name in ('legacy64_fit', 'legacy64_probe', 'remaining_train'):
                groups[role+'/sampling_role:'+name] = [r for r in subset if r['sampling_role'] == name]
    results = {}
    for name, values in groups.items():
        paired = [r for r in values if r['b'] is not None]
        results[name] = dict(b=summary(values, 'b'), midpoint=summary(values, 'midpoint'),
            midpoint_correct_wrong_descriptive={
                label:summary([r for r in values if r['midpoint'] is not None and
                               r['midpoint']['size_wrong'] == wrong],'midpoint')
                for label,wrong in (('size_correct',False),('size_wrong',True))},
            paired=dict(outputs=len(paired),
                recovered=sum(r['b']['size_wrong'] and not r['midpoint']['size_wrong'] for r in paired),
                introduced=sum(not r['b']['size_wrong'] and r['midpoint']['size_wrong'] for r in paired)))
    return results


def contiguous(previous, current):
    if (previous['sequence'] != current['sequence'] or
            current['frame_id'] != previous['frame_id']+1):
        return False
    a, b = previous['sampling'], current['sampling']
    if (a is None) != (b is None):
        return False
    if a is None:
        return True  # Index adjacency only, not verified physical continuity.
    return (a['source_video'] == b['source_video'] and a['segment'] == b['segment'] and
            b['source_frame_index'] > a['source_frame_index'] and
            0 < b['source_time_seconds']-a['source_time_seconds'] <=
            SETTINGS['sampling_gap_multiple']*SETTINGS['known_sampling_step_seconds'])


def error_runs(records):
    """Maximal error runs, split at role/video/source/time/index boundaries.

    Different flags overlap. Run counts never establish independent support.
    """
    result = []
    for role in ('train', 'val'):
        values = sorted((r for r in records if r['role'] == role),
                        key=lambda r: (r['sequence'], r['frame_id']))
        for arm in ('b', 'midpoint'):
            for flag in FLAGS:
                run = []
                for r in values+[None]:
                    active = r is not None and r[arm] is not None and r[arm][flag]
                    if run and (not active or not contiguous(run[-1], r)):
                        result.append(dict(role=role, arm=arm, flag=flag,
                            sequence=run[0]['sequence'], start_image=run[0]['image'],
                            end_image=run[-1]['image'], frames=len(run),
                            sampling_roles=dict(Counter(v['sampling_role'] for v in run)),
                            continuity_basis=('verified_source_frame_time' if run[0]['sampling']
                                              else 'frame_index_only_unverified_time'),
                            independent_scene_confirmed=False))
                        run = []
                    if active:
                        run.append(r)
    return result


def polygon_geometry(text):
    lines = [line.split() for line in text.splitlines() if line.strip()]
    if (len(lines) != 1 or len(lines[0]) != 10 or lines[0][8] != 'grab' or
            lines[0][9] not in ('0', '1')):
        raise ValueError('Expected one fixed grab DOTA quadrilateral')
    coords = [float(v) for v in lines[0][:8]]
    finite(coords, 8, 'DOTA vertices')
    points = list(zip(coords[::2], coords[1::2]))
    edges = [math.hypot(points[(i+1)%4][0]-points[i][0],
                        points[(i+1)%4][1]-points[i][1]) for i in range(4)]
    if min(edges) <= 0:
        raise ValueError('Degenerate DOTA edge')
    sides = sorted(((edges[0]+edges[2])/2, (edges[1]+edges[3])/2), reverse=True)
    return dict(center=[math.fsum(coords[::2])/4, math.fsum(coords[1::2])/4],
                long_px=sides[0], short_px=sides[1], aspect=sides[0]/sides[1])


def annotation_comparison(gt, geometry):
    g = canonical(gt)
    delta = [geometry['center'][0]-g[0], geometry['center'][1]-g[1],
             geometry['long_px']-g[2], geometry['short_px']-g[3]]
    # DOTA corners are rounded to 0.1 px; this is a diagnostic approximation
    # to native minAreaRect, not a replacement GT or inference conversion.
    return dict(delta_center_long_short_px=delta,
        within_rounding_tolerance=all(abs(v) <= SETTINGS['annotation_pixel_tolerance'] for v in delta),
        scope='rounded_polygon_geometry_check_not_native_GT_reconstruction')
