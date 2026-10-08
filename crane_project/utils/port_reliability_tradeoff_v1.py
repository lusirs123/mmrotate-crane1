"""Frozen M/score diagnostics. No fitting or deployable threshold selection."""
from bisect import bisect_right
from copy import deepcopy
from decimal import Decimal, ROUND_CEILING

from crane_project.utils import port_reliability_separability_v1 as sep
from crane_project.utils import port_simple_component_reliability_v1 as simple

VERSION = 'port_reliability_tradeoff_v1'


def required_good(good, target):
    if not 0 < target <= 1 or good < 0:
        raise ValueError('Invalid retention target or good count')
    return int((Decimal(str(target))*good).to_integral_value(rounding=ROUND_CEILING))


def retention_point(points, target):
    """First whole-tie point retaining ceil(target*good); GT diagnostic only."""
    good = points[-1]['good_outputs']
    if not good:
        return None
    needed = required_good(good, target)
    return deepcopy(next(p for p in points if p['correct_accepted'] >= needed))


def point_at(points, threshold):
    if threshold is None:
        return deepcopy(points[0])
    i = bisect_right([p['risk_le'] for p in points[1:]], threshold)
    return deepcopy(points[i])


def continuity(rows, method, component, threshold):
    """Unavailable includes missing; gaps reset runs, no unobserved-frame fill."""
    run = longest = accepted = links = switches = 0
    prev = None
    for row in sorted(rows, key=lambda r: (r['sequence'], r['frame_id'], r['image'])):
        adjacent = (prev is not None and prev[0] == row['sequence']
                    and prev[1]+1 == row['frame_id'])
        flag = bool(row['pred'] is not None and threshold is not None
                    and row['methods'][method]['risks'][component] <= threshold)
        if adjacent:
            links += 1
            switches += int(flag != prev[2])
        else:
            run = 0
        accepted += int(flag)
        run = 0 if flag else run+1
        longest = max(longest, run)
        prev = (row['sequence'], row['frame_id'], flag)
    return dict(online_flag_accepted_all_frames=accepted,
        online_flag_coverage_all_frames=sep.ratio(accepted, len(rows)),
        longest_flag_unavailable_observed_consecutive_frames=longest,
        adjacent_observed_pairs=links, flag_switches_on_adjacent_pairs=switches,
        flag_switch_rate=sep.ratio(switches, links), gaps_reset_runs=True)


def replay_val(prepared, saved):
    expected = {r['image']: r for r in saved}
    if len(expected) != len(saved) or set(expected) != {r['image'] for r in prepared}:
        raise ValueError('Frozen VAL replay identity mismatch')
    maximum = 0.
    for row in prepared:
        for method in ('raw',)+sep.METHODS:
            a, b = row['methods'][method], expected[row['image']]['methods'][method]
            if (a['raw_b_output'] != b['final_box_original'] or
                    any(a[k] != b[k] for k in ('center_accepted', 'size_accepted', 'angle_accepted'))):
                raise ValueError('Frozen VAL flags/box changed: '+row['image'])
            for key in ('size', 'angle'):
                if (a['risks'][key] is None) != (b['risks'][key] is None):
                    raise ValueError('Frozen VAL missing risk changed')
                if a['risks'][key] is not None:
                    maximum = max(maximum, abs(a['risks'][key]-b['risks'][key]))
    if maximum > 1e-12:
        raise ValueError('Frozen VAL risk replay exceeds 1e-12')
    return dict(frames=len(saved), flags_boxes_exact=True, risk_max_abs_delta=maximum)


def analyze(rows, policy, role, targets):
    before = simple.fingerprint([rows, policy])
    strata, curves, prepared = sep.analyze(rows, policy, role)
    indexed = {}
    for p in curves:
        indexed.setdefault((p['group'], p['component'], p['method']), []).append(p)
    groups = {'all': prepared}
    for name in strata:
        if name != 'all':
            field, value = name.split(':', 1)
            groups[name] = [r for r in prepared if r[field] == value]
    for name, group in strata.items():
        for component in ('size', 'angle'):
            for method in sep.METHODS:
                fixed = group['components'][component]['methods'][method]['fixed_workpoint']
                fixed['continuity'] = continuity(groups[name], method, component,
                                                 fixed['frozen_global_risk_le'])
    assessments = []
    for component in ('size', 'angle'):
        for method in sep.METHODS:
            for target in targets:
                local = {g: retention_point(indexed[g, component, method], target) for g in groups}
                # Group-specific points are capacity descriptions, never routed online.
                for scope in ('pooled', 'all_domains_and_videos'):
                    requirements = ['all'] if scope == 'pooled' else list(groups)
                    thresholds = [local[g]['risk_le'] for g in requirements if local[g] is not None]
                    threshold = max(thresholds) if thresholds else None
                    at = {}
                    for name in groups:
                        point = point_at(indexed[name, component, method], threshold)
                        point.update(global_diagnostic_risk_le=threshold,
                            required_good=required_good(point['good_outputs'], target),
                            retention_constraint_met=(point['correct_accepted'] >=
                                required_good(point['good_outputs'], target)) if point['good_outputs'] else None,
                            continuity=continuity(groups[name], method, component, threshold))
                        at[name] = point
                    assessments.append(dict(component=component, method=method, target=target,
                        constraint_scope=scope, global_diagnostic_risk_le=threshold,
                        groups=at, group_specific_capacity_only=local,
                        deployable=False, uses_GT_to_describe_ordering_capacity=True))
    if before != simple.fingerprint([rows, policy]):
        raise ValueError('Inputs/policy changed')
    return dict(role=role, strata=strata, retention_assessments=assessments,
                inputs_policy_unchanged=True), curves, prepared
