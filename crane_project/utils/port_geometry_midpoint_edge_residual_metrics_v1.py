"""Paired, bounded TRAIN/VAL reporting and fixed finite acceptance gates.

Pure reporting/gates need only the standard library. evaluate() lazily uses the
existing strict rotated-IoU/sequence evaluator; its test *mode* requests full
temporal metrics from supplied TRAIN/VAL rows, never reads any TEST input.
"""
from collections import Counter, defaultdict
import math

from crane_project.utils.port_geometry_midpoint_edge_feasibility_v1 import wrap_pi

METHODS = ('midpoint', 'edge_residual')
TOLERANCES = dict(riou=1e-5, continuous=1e-6)
ENGINEERING_REQUIRED = {
    'neutral_exact', 'gpu_save_reload', 'frozen_midpoint_state', 'frozen_midpoint_no_grad',
    'whole_run_frozen_identity', 'independent_parameter_count', 'first_stem_zero',
    'first_terminal_learns', 'subsequent_stem_learns', 'fixed_head_weight_sha',
    'cache_manifest_sha', 'source_closure', 'fixed_val_replay', 'smoke_not_reused', 'final_budget'}


def canonical(box):
    values = [float(v) for v in box[:5]]
    if len(values) != 5 or not all(math.isfinite(v) for v in values) or min(values[2:4]) <= 0:
        raise ValueError('Expected a finite positive raw OBB')
    if values[2] < values[3]:
        values[2], values[3] = values[3], values[2]
        values[4] += math.pi/2
    values[4] = wrap_pi(values[4])
    return values


def rate(hits, total):
    return dict(numerator=hits, denominator=total, fraction=hits/total if total else None)


def percentile(values, q=.95):
    if not values:
        return None
    ordered = sorted(values)
    x = (len(ordered)-1)*q
    lo, hi = math.floor(x), math.ceil(x)
    return ordered[lo]*(hi-x)+ordered[hi]*(x-lo) if hi != lo else ordered[lo]


def describe(values):
    return dict(n=len(values), mae=sum(values)/len(values) if values else None,
                p95=percentile(values), maximum=max(values) if values else None)


def row_metrics(gt_raw, pred_raw, riou):
    gt = canonical(gt_raw)
    if pred_raw is None:
        return dict(output=False, center_hit=False, joint_size10=False,
                    center_and_size10=False, riou=0., penalty_angle_deg=90.)
    pred = canonical(pred_raw)
    center = math.hypot(pred[0]-gt[0], pred[1]-gt[1])
    errors = [abs(pred[i]/gt[i]-1.) for i in (2, 3)]
    angle = math.degrees(abs(wrap_pi(pred[4]-gt[4])))
    joint = max(errors) <= .10
    if not math.isfinite(riou) or not 0 <= riou <= 1:
        raise ValueError('Invalid strict RIoU')
    return dict(output=True, center_hit=center < 15., joint_size10=joint,
        center_and_size10=center < 15. and joint, riou=riou,
        center_error_px=center, long_relative_error=errors[0], short_relative_error=errors[1],
        long_signed_log_ratio=math.log(pred[2]/gt[2]),
        short_signed_log_ratio=math.log(pred[3]/gt[3]), angle_error_deg=angle,
        penalty_angle_deg=angle if center < 10. else 90.)


def frozen_identity(row):
    m, c = row['midpoint'], row['edge_residual']
    if m is None or c is None:
        return m is None and c is None
    if (len(m) != 6 or len(c) != 6 or
            not all(math.isfinite(v) for v in c) or min(c[2:4]) <= 0):
        return False
    return (all(m[i] == c[i] for i in (0, 1, 4, 5)) and
            (m[2] < m[3]) == (c[2] < c[3]) and
            abs(wrap_pi(canonical(c)[4]-canonical(m)[4])) <= 1e-6)


def failure_intervals(rows, method, kind):
    result, current, previous = [], None, None
    for r in sorted(rows, key=lambda x: (x['sequence'], x['frame_id'])):
        metric = r['metrics'][method]
        fail = (not metric['output'] if kind == 'no_output' else
                not metric['center_hit'] if kind == 'center' else metric['riou'] < .5)
        contiguous = previous is not None and r['sequence'] == previous['sequence'] and r['frame_id'] == previous['frame_id']+1
        if current is not None and (not fail or not contiguous):
            result.append(current)
            current = None
        if fail:
            if current is None:
                current = dict(sequence=r['sequence'], start=r['frame_id'], end=r['frame_id'], length=1)
            else:
                current['end'] = r['frame_id']
                current['length'] += 1
        previous = r
    if current is not None:
        result.append(current)
    return result


def temporal_summary(rows, method, evaluator, np):
    """Reuse native sequence formulas; retain raw precision and event counts."""
    offline = [dict(domain=r['domain'], seq_id=r['sequence'], frame_id=r['frame_id'],
        gt_box=np.asarray(canonical(r['gt'])),
        pred_box=None if r[method] is None else np.asarray(canonical(r[method])),
        score=0. if r[method] is None else r[method][5], plc_rope=None, image=r['image']) for r in rows]
    series = defaultdict(list)
    for r in sorted(offline, key=lambda x: (x['domain'], x['seq_id'], x['frame_id'])):
        series[(r['domain'], r['seq_id'])].append(r)
    buckets = defaultdict(lambda: defaultdict(list))
    for (domain, _), frames in series.items():
        for segment in evaluator._split_contiguous_frames(frames):
            value = evaluator._compute_sequence_metrics(segment)
            for key in ('angle_errors', 'dfr_vals', 'aci_vals', 'tdr_hits', 'mrf_vals'):
                buckets[domain][key].extend(value[key])
            buckets[domain]['mcml'].append(value['mcml'])
    result = {}
    for domain, values in buckets.items():
        mean = lambda key: sum(values[key])/len(values[key]) if values[key] else None
        angles = values['angle_errors']
        dfr, aci, mcml, mrf = mean('dfr_vals'), mean('aci_vals'), values['mcml'], mean('mrf_vals')
        result[domain] = dict(
            dfr_percent_per_frame=None if dfr is None else 100*dfr,
            dfr_adjacent_pairs=len(values['dfr_vals']), aci=aci,
            aci_adjacent_pairs=len(values['aci_vals']),
            a_rmse_deg=math.degrees(math.sqrt(sum(v*v for v in angles)/len(angles))) if angles and domain == 'sim' else None,
            angle_penalty_frames=len(angles) if domain == 'sim' else 0,
            tdr_w10=rate(sum(values['tdr_hits']), len(values['tdr_hits'])),
            mcml_max=max(mcml) if mcml else None,
            mcml_mean=sum(mcml)/len(mcml) if mcml else None,
            mcml_segments=len(mcml),
            mcml_pass_limit5=int(max(mcml) <= 5) if mcml else None,
            mrf_mean=mrf, mrf_events=len(values['mrf_vals']))
    # Original rounded protocol output is retained for comparison/readback;
    # finite gates use the unrounded values above, not rounded displayed means.
    before = evaluator.logger.disabled
    evaluator.logger.disabled = True
    try:
        protocol = evaluator.evaluate_records(offline)
    finally:
        evaluator.logger.disabled = before
    return result, protocol


def method_summary(rows, method, evaluator=None, np=None):
    values = [r['metrics'][method] for r in rows]
    outputs = [v for v in values if v['output']]
    n, out = len(values), len(outputs)
    hits = sum(v['center_hit'] for v in values)
    joint = sum(v['joint_size10'] for v in values)
    both = sum(v['center_and_size10'] for v in values)
    result = dict(frames=n, output_coverage=rate(out, n),
        center_correct_conditional=rate(hits, out), center_correct_full_frame=rate(hits, n),
        joint_size10_conditional=rate(joint, out), joint_size10_full_frame=rate(joint, n),
        center_and_size10_full_frame=rate(both, n),
        mean_riou_full_frame=sum(v['riou'] for v in values)/n if n else None,
        edges={name: dict(describe([v[name+'_relative_error'] for v in outputs]),
            signed_log_mean=sum(v[name+'_signed_log_ratio'] for v in outputs)/out if out else None)
            for name in ('long', 'short')},
        pure_angle_rmse_deg=math.sqrt(sum(v['angle_error_deg']**2 for v in outputs)/out) if out else None,
        center_error=describe([v['center_error_px'] for v in outputs]))
    result['failure_intervals'] = {kind: failure_intervals(rows, method, kind)
                                  for kind in ('no_output', 'center', 'riou')}
    result['longest_failure_run'] = {kind: max((v['length'] for v in runs), default=0)
                                    for kind, runs in result['failure_intervals'].items()}
    if evaluator is not None and rows:
        result['temporal'], result['metric_protocol_v2'] = temporal_summary(rows, method, evaluator, np)
    return result


def grouped(rows, evaluator=None, np=None):
    groups = dict(overall=rows)
    groups.update({d: [r for r in rows if r['domain'] == d] for d in ('real', 'sim')})
    groups.update({s: [r for r in rows if r['sequence'] == s] for s in sorted({r['sequence'] for r in rows})})
    result = {}
    for name, subset in groups.items():
        if not subset:
            continue
        before, after = ([r['metrics'][method] for r in subset] for method in METHODS)
        result[name] = {method: method_summary(subset, method, evaluator, np) for method in METHODS}
        result[name]['paired'] = dict(
            frozen_identity_preserved=all(frozen_identity(r) for r in subset),
            recovered_size10=sum(not a['joint_size10'] and b['joint_size10'] for a, b in zip(before, after)),
            new_size10_failures=sum(a['joint_size10'] and not b['joint_size10'] for a, b in zip(before, after)),
            new_riou_below_0_5=sum(a['riou'] >= .5 and b['riou'] < .5 for a, b in zip(before, after)),
            accepted_changes=sum(r.get('size_delivery') == 'accepted' for r in subset),
            delivery_counts=dict(Counter(r.get('size_delivery', 'unknown') for r in subset)),
            fallback_check_counts=dict(Counter(k for r in subset if r.get('size_delivery') == 'fallback' for k in r['failed_checks'])))
        targets = [r['target_diagnostic'] for r in subset if r.get('target_diagnostic') is not None]
        result[name]['target_diagnostics'] = dict(output_views=len(targets),
            exact_target_outside_original_b=sum(t['exact_target_outside_original_b'] for t in targets),
            exact_target_m_order_conflicts=sum(t['exact_target_m_order_conflict'] for t in targets),
            fixed_b_target_m_order_conflicts=sum(t['fixed_b_target_m_order_conflict'] for t in targets),
            fixed_b_target_axis_differences=sum(t['fixed_b_target_axis_difference'] for t in targets),
            original_model_b_swap_differences=sum(t['original_b_swap'] != t['model_b_swap'] for t in targets),
            gt_exact_squares=sum(t['gt_exact_square'] for t in targets),
            note='Axis-difference flags include numerical differences; they are not all semantic swaps. Exact-target out-of-bound does not imply joint10 infeasible. No target filter.')
    return result


def summarize_shards(rows, evaluator=None, np=None):
    result = {}
    for shard in sorted({r['shard'] for r in rows}):
        subset = [r for r in rows if r['shard'] == shard]
        subsets = dict(all=subset)
        if shard.startswith('train_'):
            for role in ('fit', 'probe'):
                subsets[role] = [r for r in subset if r['sample_role'] == role]
                subsets[role+'_eligible'] = [r for r in subsets[role] if r['eligible_train']]
        result[shard] = {name: grouped(value, evaluator, np) for name, value in subsets.items()}
    return result


def evaluate(rows):
    import numpy as np
    from crane_project.tools.eval_crane_offline import CraneOfflineEvaluator, compute_riou
    evaluator = CraneOfflineEvaluator(mode='test')
    for r in rows:
        r['metrics'] = {method: row_metrics(r['gt'], r[method],
            compute_riou(np.asarray(canonical(r[method])), np.asarray(canonical(r['gt'])))
            if r[method] is not None else 0.) for method in METHODS}
    return summarize_shards(rows, evaluator, np)


def finite_gates(summary, engineering):
    """One predeclared gate; no selection/retry/tuning or automatic promotion."""
    checks = {'engineering/'+k: v is True for k, v in engineering.items()}
    checks['engineering/all_required_proofs'] = ENGINEERING_REQUIRED <= set(engineering)
    tol, riou_tol = TOLERANCES['continuous'], TOLERANCES['riou']
    def compare(a, b, relation, tolerance=tol):
        if a is None or b is None:
            return a is None and b is None
        if not math.isfinite(a) or not math.isfinite(b):
            return False
        return abs(b-a) <= tolerance if relation == 'equal' else (b <= a+tolerance if relation == 'down' else b >= a-tolerance)
    val = summary['val_s1']['all']
    required = {'real', 'sim', 'real_seq07', 'real_seq14', 'sim_seq10'}
    checks['val/groups_complete'] = required <= set(val)
    for name in sorted(required):
        if name not in val:
            continue
        group = val[name]
        a, b = (group[method] for method in METHODS)
        prefix = 'val/'+name+'/'
        expected_frames = {'real':375, 'sim':512, 'real_seq07':226, 'real_seq14':149, 'sim_seq10':512}[name]
        checks[prefix+'all_frames'] = a['frames'] == b['frames'] == expected_frames
        checks[prefix+'frozen_identity'] = group['paired']['frozen_identity_preserved'] is True
        checks[prefix+'output_coverage'] = a['output_coverage'] == b['output_coverage']
        checks[prefix+'center_denominators'] = all(a[key] == b[key] for key in ('center_correct_conditional', 'center_correct_full_frame'))
        aj, bj = (v['joint_size10_full_frame']['numerator'] for v in (a, b))
        checks[prefix+'joint_size10'] = bj > aj if name in ('real', 'sim') else bj >= aj
        checks[prefix+'center_and_size10'] = b['center_and_size10_full_frame']['numerator'] >= a['center_and_size10_full_frame']['numerator']
        for edge in ('long', 'short'):
            for statistic in ('mae', 'p95', 'maximum'):
                checks[prefix+edge+'/'+statistic] = compare(a['edges'][edge][statistic], b['edges'][edge][statistic], 'down')
        checks[prefix+'mean_riou'] = compare(a['mean_riou_full_frame'], b['mean_riou_full_frame'], 'up', riou_tol)
        checks[prefix+'no_new_riou_failures'] = group['paired']['new_riou_below_0_5'] == 0
        checks[prefix+'longest_riou_run'] = b['longest_failure_run']['riou'] <= a['longest_failure_run']['riou']
        domain = name.split('_')[0]
        x, y = a['temporal'][domain], b['temporal'][domain]
        for key, direction in (('dfr_percent_per_frame', 'down'), ('aci', 'equal'),
                ('a_rmse_deg', 'equal'), ('mcml_mean', 'down'), ('mrf_mean', 'down')):
            checks[prefix+key] = compare(x[key], y[key], direction)
        for key in ('aci_adjacent_pairs', 'dfr_adjacent_pairs', 'angle_penalty_frames', 'mcml_segments'):
            checks[prefix+key] = x[key] == y[key]
        checks[prefix+'tdr_w10'] = (x['tdr_w10']['denominator'] == y['tdr_w10']['denominator'] and
            y['tdr_w10']['numerator'] >= x['tdr_w10']['numerator'])
        checks[prefix+'mcml_pass_limit5'] = compare(x['mcml_pass_limit5'], y['mcml_pass_limit5'], 'up', 0.)
        checks[prefix+'mcml_max'] = compare(x['mcml_max'], y['mcml_max'], 'down', 0.)
        checks[prefix+'mrf_definition'] = (x['mrf_events'] == 0) == (y['mrf_events'] == 0)
    for shard in ('train_s1', 'train_s05'):
        for domain in ('real', 'sim'):
            a, b = (summary[shard]['probe'][domain][method] for method in METHODS)
            checks[shard+'/'+domain+'/probe_joint_size10'] = b['joint_size10_full_frame']['numerator'] >= a['joint_size10_full_frame']['numerator']
        checks[shard+'/all_frozen_identity'] = summary[shard]['all']['overall']['paired']['frozen_identity_preserved'] is True
    failed = [key for key, passed in checks.items() if not passed]
    return dict(finite_joint_gate_pass=not failed, checks=checks, failed_checks=failed,
        automatic_promotion=False, formal_training_approved=False,
        status='FINITE_GATE_PASS_REVIEW_REQUIRED' if not failed else 'FINITE_GATE_NOT_PASSED_KEEP_M',
        note='A finite failure does not refute all independent size learning; no budget/structure/threshold sweep follows automatically.')
