"""Full-denominator size diagnostics and predeclared selection; no fitting."""
from collections import Counter
import math
from crane_project.utils import port_geometry_midpoint_edge_residual_metrics_v1 as old

METHODS = ('b', 'midpoint', 'edge_residual')
TOL = 1e-6
RIOU_TOL = 1e-5


def group(gt, m):
    if m is None:
        return 'missing'
    g, p = sorted(gt[2:4], reverse=True), sorted(m[2:4], reverse=True)
    errors = [x/y-1 for x, y in zip(p, g)]
    if max(map(abs, errors)) <= .03:
        return 'near_correct'
    if max(errors) < 0:
        return 'need_enlarge'
    if min(errors) > 0:
        return 'need_shrink'
    return 'mixed'


def size_summary(rows, method):
    values, signed, logs, projected = [], [], [], []
    for r in rows:
        if r[method] is None:
            continue
        g, p = old.canonical(r['gt']), old.canonical(r[method])
        e = [p[i]/g[i]-1 for i in (2, 3)]
        z = [math.log(p[i]/g[i]) for i in (2, 3)]
        signed.append(e); values.append(list(map(abs, e))); logs.append(z)
        angle = abs(old.wrap_pi(p[4]-g[4]))
        pl = (abs(math.cos(angle))*p[2]+abs(math.sin(angle))*p[3])/g[2]
        ps = (abs(math.sin(angle))*p[2]+abs(math.cos(angle))*p[3])/g[3]
        projected.append([pl, ps])
    n = len(values)
    stats = lambda x: dict(mean=sum(x)/len(x), p95=old.percentile(x)) if x else dict(mean=None, p95=None)
    a, ratio = [(x+y)/2 for x, y in logs], [x-y for x, y in logs]
    return dict(frames=len(rows), outputs=n, missing=len(rows)-n,
        joint10=sum(max(v) <= .1 for v in values),
        edges={name:dict(mae=sum(v[i] for v in values)/n if n else None,
            p95=old.percentile([v[i] for v in values]),
            bias=sum(v[i] for v in signed)/n if n else None,
            positive=stats([v[i] for v in signed if v[i]>0]),
            negative_abs=stats([-v[i] for v in signed if v[i]<0]))
            for i, name in enumerate(('long','short'))},
        a=dict(abs_mae=stats(list(map(abs,a)))['mean'], abs_p95=old.percentile(list(map(abs,a))),
               bias=sum(a)/n if n else None),
        r=dict(abs_mae=stats(list(map(abs,ratio)))['mean'], abs_p95=old.percentile(list(map(abs,ratio))),
               bias=sum(ratio)/n if n else None),
        projected_gt_axes={name:stats([v[i] for v in projected]) for i,name in enumerate(('long','short'))})


def paired(rows, before='midpoint', after='edge_residual'):
    present = [r for r in rows if r[before] is not None and r[after] is not None]
    correct = lambda r,k:max(abs(x/y-1) for x,y in zip(sorted(r[k][2:4]),sorted(r['gt'][2:4]))) <= .1
    return dict(wrong_to_right=sum(not correct(r,before) and correct(r,after) for r in present),
        right_to_wrong=sum(correct(r,before) and not correct(r,after) for r in present),
        new_output=sum(r[before] is None and r[after] is not None for r in rows),
        lost_output=sum(r[before] is not None and r[after] is None for r in rows),
        frozen_identity=all(old.frozen_identity(r) for r in rows))


def train_report(rows):
    result, checks = {}, {}
    for scale in (1., .5):
        for domain in ('real','sim'):
            cell = [r for r in rows if r['scale']==scale and r['domain']==domain]
            name = domain+'/'+str(scale)
            result[name] = dict(all={k:size_summary(cell,k) for k in METHODS}, groups={})
            checks[name+'/identity'] = paired(cell)['frozen_identity']
            for label in ('need_enlarge','need_shrink','near_correct','mixed','missing'):
                subset = [r for r in cell if group(r['gt'],r['midpoint'])==label]
                values = {k:size_summary(subset,k) for k in METHODS}
                result[name]['groups'][label] = dict(count=len(subset), methods=values, paired=paired(subset))
                if label in ('need_enlarge','need_shrink','near_correct'):
                    x, y = values['midpoint']['a']['abs_mae'], values['edge_residual']['a']['abs_mae']
                    checks[name+'/'+label+'/support'] = bool(subset)
                    checks[name+'/'+label+'/abs_a'] = (x is not None and y is not None and
                        (y<=x+TOL if label=='near_correct' else y<x-TOL))
                    if label=='near_correct':
                        checks[name+'/'+label+'/new_size10_failures'] = paired(subset)['right_to_wrong']==0
    grouped = {'overall': rows}
    grouped.update({d:[r for r in rows if r['domain']==d] for d in ('real','sim')})
    grouped.update({v:[r for r in rows if r['sequence']==v] for v in sorted({r['sequence'] for r in rows})})
    return dict(cells=result, groups={name:dict(methods={k:size_summary(rs,k) for k in METHODS},paired=paired(rs))
                for name,rs in grouped.items()}, checks=checks, failed_checks=[k for k,v in checks.items() if not v],
                train_pass=all(checks.values()))


def summary(rows, riou_cache=None):
    import numpy as np
    from crane_project.tools.eval_crane_offline import CraneOfflineEvaluator, compute_riou
    from crane_project.tools.eval_port_geometry_ap_v1 import evaluate_images
    evaluator = CraneOfflineEvaluator(mode='test')  # calculation mode only, supplied VAL rows
    if riou_cache is None:
        riou_cache = {}
    for r in rows:
        r['metrics'] = {}
        for method in METHODS:
            key = (tuple(r['gt']),tuple(r[method]) if r[method] is not None else None)
            if key not in riou_cache:
                riou_cache[key] = compute_riou(np.asarray(old.canonical(r[method])),
                    np.asarray(old.canonical(r['gt']))) if r[method] is not None else 0.
            r['metrics'][method] = old.row_metrics(r['gt'],r[method],riou_cache[key])
    subsets = dict(overall=rows)
    subsets.update({d:[r for r in rows if r['domain']==d] for d in ('real','sim')})
    subsets.update({s:[r for r in rows if r['sequence']==s] for s in sorted({r['sequence'] for r in rows})})
    result = {}
    for name, subset in subsets.items():
        result[name] = dict(paired=paired(subset))
        for method in METHODS:
            value = old.method_summary(subset,method,evaluator,np)
            value['sizes'] = size_summary(subset,method)
            ordered = sorted(subset,key=lambda r:r['image'])
            dets = [[old.canonical(r[method])+[r[method][5]]] if r[method] is not None else [] for r in ordered]
            gt = [[old.canonical(r['gt'])] for r in ordered]
            value['cache_gt_ap11'] = {str(th):evaluate_images(dets,gt,th)['ap_percent'] for th in (.5,.75)}
            result[name][method] = value
    return result


def val_gate(value):
    checks = {}
    for domain in ('real','sim'):
        cell = value[domain]
        m,b,c = [cell[k] for k in ('midpoint','b','edge_residual')]
        x,y = m['sizes'],c['sizes']
        p = domain+'/'
        checks[p+'identity'] = cell['paired']['frozen_identity'] and cell['paired']['new_output']==cell['paired']['lost_output']==0
        checks[p+'joint10'] = y['joint10'] >= x['joint10']
        for edge in ('long','short'):
            checks[p+edge+'/mae'] = y['edges'][edge]['mae'] <= x['edges'][edge]['mae']+TOL
            checks[p+edge+'/p95'] = y['edges'][edge]['p95'] <= x['edges'][edge]['p95']+TOL
            if domain=='real':
                checks[p+edge+'/strict_mae_vs_m_b'] = y['edges'][edge]['mae'] < min(x['edges'][edge]['mae'],b['sizes']['edges'][edge]['mae'])-TOL
        for metric in ('a','r'):
            checks[p+metric] = y[metric]['abs_mae'] <= x[metric]['abs_mae']+TOL
        checks[p+'riou'] = (c['mean_riou_full_frame'] > m['mean_riou_full_frame']+RIOU_TOL if domain=='real' else
                            c['mean_riou_full_frame'] >= m['mean_riou_full_frame']-RIOU_TOL)
        mt,ct = m['temporal'][domain],c['temporal'][domain]
        for key in ('mcml_max','mcml_mean'):
            checks[p+key] = ct[key] <= mt[key]
        checks[p+'tdr'] = ct['tdr_w10']['denominator']==mt['tdr_w10']['denominator'] and ct['tdr_w10']['numerator']>=mt['tdr_w10']['numerator']
        checks[p+'aci'] = ct['aci']==mt['aci'] and ct['aci_adjacent_pairs']==mt['aci_adjacent_pairs']
        checks[p+'center_direction'] = c['center_correct_full_frame']==m['center_correct_full_frame'] and c['pure_angle_rmse_deg']==m['pure_angle_rmse_deg']
    return dict(checks=checks,failed_checks=[k for k,v in checks.items() if not v],val_pass=all(checks.values()))


def selection_key(summary,epoch):
    e=summary['real']['edge_residual']['sizes']['edges']
    return e['long']['mae'],e['short']['mae'],epoch
