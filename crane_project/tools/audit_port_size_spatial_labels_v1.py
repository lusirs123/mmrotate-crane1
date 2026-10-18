#!/usr/bin/env python3
"""S033 spatial-label adaptation review, not a model or training entry.

Saved TRAIN boxes only; analytic polygon labels and predeclared synthetic
counterexamples. M's 1.5x search is an IDEAL proposed window, not the actual
B ROI, feature support, visible contour or a tested inference system.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.analyze_port_size_joint_v1 import (
    canonical, csv_write, mean, quantile, read, sha, write)

VERSION = 'port_size_spatial_label_review_v1'
SEARCH = 1.5


def rectangle(cx, cy, long, short, theta):
    c, s = math.cos(theta), math.sin(theta)
    return [(cx+c*x-s*y, cy+s*x+c*y) for x, y in
            [(-long/2, -short/2), (long/2, -short/2),
             (long/2, short/2), (-long/2, short/2)]]


def gt_in_m(gt, m):
    g, m = canonical(gt), canonical(m)
    dx, dy = g[0]-m[0], g[1]-m[1]
    c, s = math.cos(m[4]), math.sin(m[4])
    return rectangle(c*dx+s*dy, -s*dx+c*dy, g[2], g[3], g[4]-m[4])


def clip_rect(poly, left, right, bottom, top):
    poly = list(poly)
    for axis, value, lower in ((0, left, True), (0, right, False),
                               (1, bottom, True), (1, top, False)):
        if not poly:
            break
        result = []
        previous = poly[-1]
        inside = lambda p: p[axis] >= value if lower else p[axis] <= value
        for current in poly:
            a, b = inside(previous), inside(current)
            if a != b:
                t = (value-previous[axis])/(current[axis]-previous[axis])
                result.append(tuple(previous[j]+t*(current[j]-previous[j]) for j in (0, 1)))
            if b:
                result.append(current)
            previous = current
        poly = result
    return poly


def area(poly):
    if not poly:
        return 0.
    return abs(math.fsum(a[0]*b[1]-b[0]*a[1] for a, b in zip(poly, poly[1:]+poly[:1]))/2)


def intrinsic_from_polygon(poly):
    """Ideal UNIFORM polygon mass: central covariance eigenvalues, in pixels.

    For a full rectangle, lengths=sqrt(12*eigenvalues), independent of its
    center/orientation. For a clipped polygon this is NOT rectangle recovery.
    No network/probability field is implemented by this reference arithmetic.
    """
    if len(poly) < 3 or area(poly) < 1e-9:
        return None
    edges = [(p, q, p[0]*q[1]-q[0]*p[1]) for p, q in zip(poly, poly[1:]+poly[:1])]
    signed = math.fsum(z for p, q, z in edges)/2
    cx = math.fsum((p[0]+q[0])*z for p, q, z in edges)/(6*signed)
    cy = math.fsum((p[1]+q[1])*z for p, q, z in edges)/(6*signed)
    exx = math.fsum((p[0]*p[0]+p[0]*q[0]+q[0]*q[0])*z for p, q, z in edges)/(12*signed)
    eyy = math.fsum((p[1]*p[1]+p[1]*q[1]+q[1]*q[1])*z for p, q, z in edges)/(12*signed)
    exy = math.fsum((2*p[0]*p[1]+p[0]*q[1]+q[0]*p[1]+2*q[0]*q[1])*z for p, q, z in edges)/(24*signed)
    xx, yy, xy = exx-cx*cx, eyy-cy*cy, exy-cx*cy
    disc = math.hypot(xx-yy, 2*xy)
    lambdas = [(xx+yy+disc)/2, (xx+yy-disc)/2]
    if min(lambdas) <= 0:
        return None
    return [math.sqrt(12*x) for x in lambdas]


def point_mask(poly_box, search_long=150., search_short=75., k=32):
    cx, cy, l, s, angle = poly_box
    c, sn = math.cos(angle), math.sin(angle)
    values = []
    for j in range(k):
        for i in range(k):
            x, y = ((i+.5)/k-.5)*search_long-cx, ((j+.5)/k-.5)*search_short-cy
            values.append(abs(c*x+sn*y) <= l/2 and abs(-sn*x+c*y) <= s/2)
    return values


def inspect(gt, m):
    g, m = canonical(gt), canonical(m)
    poly = gt_in_m(g, m)
    xs, ys = zip(*poly)
    projected = [max(xs)-min(xs), max(ys)-min(ys)]
    expected = [abs(math.cos(g[4]-m[4]))*g[2]+abs(math.sin(g[4]-m[4]))*g[3],
                abs(math.sin(g[4]-m[4]))*g[2]+abs(math.cos(g[4]-m[4]))*g[3]]
    if max(abs(a-b) for a, b in zip(projected, expected)) > 1e-7:
        raise ValueError('Projection formula/corner disagreement')
    virtual_intersection = area(clip_rect(poly, -g[2]/2, g[2]/2, -g[3]/2, g[3]/2))
    virtual_iou = virtual_intersection/(2*g[2]*g[3]-virtual_intersection)
    rx, ry = SEARCH*m[2]/2, SEARCH*m[3]/2
    clipped = clip_rect(poly, -rx, rx, -ry, ry)
    visible = min(1., area(clipped)/(g[2]*g[3]))
    full = intrinsic_from_polygon(poly)
    if full is None:
        raise ValueError('Positive rectangle has no ideal moments')
    reduced = intrinsic_from_polygon(clipped)
    return dict(virtual_real_box_iou=virtual_iou,
        virtual_fully_in_ideal_M_search=(g[2] <= 2*rx+1e-9 and g[3] <= 2*ry+1e-9),
        true_fully_in_ideal_M_search=all(abs(x) <= rx+1e-9 and abs(y) <= ry+1e-9 for x, y in poly),
        true_area_fraction_in_ideal_M_search=visible,
        projection_relative_errors=[projected[j]/g[j+2]-1 for j in (0, 1)],
        projected_widths=projected,
        projected_axis_order_reversed=projected[0] < projected[1],
        projection_direction_disagrees=[(projected[j]-m[j+2])*(g[j+2]-m[j+2]) < -1e-12 for j in (0, 1)],
        full_moment_relative_errors=[full[j]/g[j+2]-1 for j in (0, 1)],
        clipped_moment_relative_errors=None if reduced is None else [reduced[j]/g[j+2]-1 for j in (0, 1)])


def summarize(rows):
    good = [r['geometry'] for r in rows if r['geometry'] is not None]
    truncated = [x for x in good if not x['true_fully_in_ideal_M_search']]
    result = dict(views=len(rows), outputs=len(good), missing=len(rows)-len(good),
        virtual_fully_covered=sum(x['virtual_fully_in_ideal_M_search'] for x in good),
        true_fully_covered=sum(x['true_fully_in_ideal_M_search'] for x in good),
        true_clipped=len(truncated), empty_true_target=sum(x['true_area_fraction_in_ideal_M_search'] < 1e-12 for x in good),
        mean_virtual_real_box_iou=mean([x['virtual_real_box_iou'] for x in good]),
        mean_true_area_fraction=mean([x['true_area_fraction_in_ideal_M_search'] for x in good]),
        ideal_full_moment_max_abs_relative_error=max((abs(e) for x in good for e in x['full_moment_relative_errors']), default=None),
        projected_axis_order_reversed=sum(x['projected_axis_order_reversed'] for x in good))
    for j, edge in enumerate(('long_axis', 'short_axis')):
        errors = [abs(x['projection_relative_errors'][j]) for x in good]
        clip = [abs(x['clipped_moment_relative_errors'][j]) for x in truncated if x['clipped_moment_relative_errors'] is not None]
        result.update({edge+'_projection_MAE_pct':100*mean(errors) if errors else None,
                       edge+'_projection_P95_pct':100*quantile(errors) if errors else None,
                       edge+'_projection_direction_disagrees':sum(x['projection_direction_disagrees'][j] for x in good),
                       edge+'_clipped_moment_evaluable':len(clip),
                       edge+'_clipped_moment_MAE_pct':100*mean(clip) if clip else None})
    return result


def synthetic():
    m = [0., 0., 100., 50., 0.]
    cases = dict(identity=m, translation_only=[10., 5., 100., 50., 0.],
                 angle5_size_exact=[0., 0., 100., 50., math.radians(5)],
                 angle45_size_exact=[0., 0., 100., 50., math.pi/4],
                 off_window=[500., 500., 100., 50., 0.])
    values = {name:inspect(gt, m) for name, gt in cases.items()}
    values['fixed32_binary_alias'] = dict(first_box=[0,0,94,40,0], second_box=[0,0,96,41,0],
        same_point_mask=point_mask([0,0,94,40,0]) == point_mask([0,0,96,41,0]),
        scope='Fixed32 point occupancy only. Fractional cell-area labels can distinguish these; not an impossibility theorem for spatial supervision.')
    return values


def run(source, out):
    source, out = source.resolve(), out.resolve()
    if out.exists() or out.parent != source.parent or out == source:
        raise ValueError('Fresh sibling result directory required')
    p = source/'baseline_train.rows.jsonl.gz'
    index = read(source/'artifacts.json')['files']
    if sha(p) != index[p.name]:
        raise ValueError('Saved TRAIN identity drift')
    values = []
    with gzip.open(p, 'rt') as f:
        for s in f:
            r = json.loads(s)
            if r['split'] not in ('train', 'train_sim') or r['scale'] not in (1., .5):
                raise ValueError('Only original TRAIN two views permitted')
            values.append(dict(image=r['image'], sequence=r['sequence'], domain=r['domain'], scale=r['scale'],
                               geometry=None if r['midpoint'] is None else inspect(r['gt'], r['midpoint'])))
    if len(values) != 5116 or len({(r['image'],r['scale']) for r in values}) != 5116:
        raise ValueError('TRAIN coverage differs')
    summaries = []
    for scale in (1., .5):
        rs = [r for r in values if r['scale'] == scale]
        if len(rs) != 2558:
            raise ValueError('TRAIN scale count differs')
        groups = dict(overall=rs)
        groups.update({d:[r for r in rs if r['domain'] == d] for d in ('real','sim')})
        groups.update({s:[r for r in rs if r['sequence'] == s] for s in sorted({r['sequence'] for r in rs})})
        summaries.extend(dict(scale=scale, group=g, **summarize(x)) for g, x in groups.items())
    out.mkdir(parents=True, exist_ok=False)
    csv_write(out/'train_label_geometry.csv', summaries)
    with gzip.open(out/'train_label_geometry.rows.jsonl.gz', 'xt') as f:
        for r in values:
            f.write(json.dumps(r, allow_nan=False)+'\n')
    references = ['crane_project/utils/port_geometry_size_boundary_v1.py',
                  'crane_project/utils/port_geometry_size_boundary_continuous_v1.py',
                  'crane_project/utils/port_geometry_size_distribution_v1.py',
                  'crane_project/utils/port_geometry_size_condition_v1.py',
                  'crane_project/utils/port_size_native_p3_v1.py',
                  'crane_project/utils/port_geometry_refine_g_v1.py',
                  'crane_project/utils/port_geometry_midpoint_edge_residual_v1.py',
                  'crane_project/tools/analyze_port_size_joint_v1.py',
                  'tests/test_port_size_spatial_labels_v1.py']
    write(out/'analysis.json', dict(protocol=VERSION, status='GEOMETRY_ADAPTATION_REVIEW_ONLY',
        inputs={p.name:sha(p)}, sources={n:sha(ROOT/n) for n in references},
        script_sha256=sha(Path(__file__).resolve()), summaries=summaries, synthetic=synthetic(),
        scope='Ideal1.5x M search only; actual B-ROI resize/support not audited; full polygon moments are oracle arithmetic, not learned performance.',
        actual_roi_support_audited=False, image_contour_labels_available=False,
        grid_resolution_search=False, coefficients_fitted=False, val_reads=0, test_reads=0,
        depth_reads=0, model_calls=0, gpu_calls=0, training_updates=0))
    write(out/'artifacts.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(status='COMPLETE',views=len(values),groups=len(summaries),model_calls=0)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    run(args.run, args.out)
