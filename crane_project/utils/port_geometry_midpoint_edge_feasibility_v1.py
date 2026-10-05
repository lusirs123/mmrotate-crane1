"""OFFLINE size feasibility for frozen B24 + midpoint sigma1.5/epoch03.

Standard-library geometry only, Python 3.8+. Certificates are GT diagnostics,
never predictions. No learning, detector, reliability policy or TEST access.
The runtime adapter must recheck witnesses in the source tensor dtype/device.
"""
import math


SETTINGS = dict(relative_error_limit=.10, edge_ratio_min=.8,
    edge_ratio_max=1.25, context_multiplier=1.5, min_context_pixels=16.,
    roi_bound=.5, roi_bound_tolerance=1e-5,
    max_center_over_b_short=.30, max_angle_residual_rad=math.pi/18,
    legacy_guard_tolerance=1e-6, boundary_atol_px=1e-4, boundary_rtol=1e-6)


def _numbers(value, length, name):
    if value is None or len(value) != length:
        raise ValueError('Invalid '+name+' shape')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) for v in value):
        raise ValueError('Expected finite numeric '+name)
    return tuple(float(v) for v in value)


def _box(value, length, name):
    result = _numbers(value, length, name)
    if min(result[2:4]) <= 0:
        raise ValueError('Nonpositive '+name+' edges')
    return result


def wrap_pi(value):
    return (value+math.pi/2) % math.pi-math.pi/2


def make_frame(b_original, b_model, scale_xy, runtime_frame=None):
    """Actual raw association first; only model B defines the ROI axes.

    runtime_frame carries context/rotation from the existing torch frame()
    function, so float32 trig/resize rounding is retained, not guessed.
    """
    b = _box(b_original, 6, 'B original')
    bm = _box(b_model, 5, 'B model')
    xy = _numbers(scale_xy, 2, 'scale_xy')
    if min(xy) <= 0:
        raise ValueError('Nonpositive actual sx/sy')
    for i, scale in enumerate((xy[0], xy[1], xy[0], xy[1])):
        if not math.isclose(bm[i], b[i]*scale, rel_tol=1e-6, abs_tol=1e-4):
            raise ValueError('B raw w/h restoration differs')
    if abs(wrap_pi(bm[4]-b[4])) > 1e-5:
        raise ValueError('B raw angle restoration differs')
    swap = bm[2] < bm[3]
    angle = wrap_pi(bm[4]+(math.pi/2 if swap else 0.))
    context = tuple(max(v*SETTINGS['context_multiplier'],
                        SETTINGS['min_context_pixels'])
                    for v in (max(bm[2:4]), min(bm[2:4])))
    c, s = math.cos(angle), math.sin(angle)
    rotation = ((c, -s), (s, c))
    if runtime_frame is not None:
        actual_context = _numbers(runtime_frame['context_sides_model'], 2, 'context')
        actual_rotation = tuple(_numbers(v, 2, 'rotation')
                                for v in runtime_frame['rotation_model'])
        if len(actual_rotation) != 2 or min(actual_context) <= 0:
            raise ValueError('Invalid runtime ROI frame')
        if any(not math.isclose(a, e, rel_tol=1e-6, abs_tol=1e-4)
               for a, e in zip(actual_context, context)):
            raise ValueError('Runtime context differs from original frame')
        if any(abs(actual_rotation[i][j]-rotation[i][j]) > 1e-5
               for i in range(2) for j in range(2)):
            raise ValueError('Runtime model-B rotation differs')
        context, rotation = actual_context, actual_rotation
    return dict(center_model=list(bm[:2]), scale_xy=list(xy),
        context_sides_model=list(context), rotation_model=[list(v) for v in rotation],
        original_b_swap=b[2] < b[3], model_b_swap=swap)


def to_roi(point, frame):
    """Row-vector version of the existing original_to_roi affine map."""
    x = point[0]*frame['scale_xy'][0]-frame['center_model'][0]
    y = point[1]*frame['scale_xy'][1]-frame['center_model'][1]
    r, side = frame['rotation_model'], frame['context_sides_model']
    return tuple((x*r[0][j]+y*r[1][j])/side[j] for j in (0, 1))


def _linear(vector, frame):
    x, y = (vector[i]*frame['scale_xy'][i] for i in (0, 1))
    r, side = frame['rotation_model'], frame['context_sides_model']
    return tuple((x*r[0][j]+y*r[1][j])/side[j] for j in (0, 1))


def roi_caps(m, frame):
    q = to_roi(m[:2], frame)
    c, s = math.cos(m[4]), math.sin(m[4])
    a, b = _linear((c*.5, s*.5), frame), _linear((-s*.5, c*.5), frame)
    limit = SETTINGS['roi_bound']+SETTINGS['roi_bound_tolerance']
    def cap(vector):
        if any(abs(v) > limit for v in q):
            return 0.
        bounds = [(limit-abs(q[j]))/abs(vector[j])
                  for j in (0, 1) if vector[j] != 0.]
        # A positive sx/sy and valid rotation cannot erase an entire axis.
        if not bounds:
            raise ValueError('Degenerate affine size axis')
        return min(bounds)
    return dict(q0=list(q), width_coefficient=list(a), height_coefficient=list(b),
        raw_width_upper_px=cap(a), raw_height_upper_px=cap(b), limit=limit)


def goal_edges(gt, m):
    gt = _box(gt, 5, 'GT')
    long, short = max(gt[2:4]), min(gt[2:4])
    return (short, long) if m[2] < m[3] else (long, short)


def size_errors(edges, goal):
    return tuple(abs(edges[i]/goal[i]-1.) for i in (0, 1))


def joint_correct(edges, goal):
    return max(size_errors(edges, goal)) <= SETTINGS['relative_error_limit']


def intervals(b, goal, caps=None):
    error = SETTINGS['relative_error_limit']
    result = []
    for i in (0, 1):
        lower = max(SETTINGS['edge_ratio_min']*b[i+2], (1-error)*goal[i])
        upper = min(SETTINGS['edge_ratio_max']*b[i+2], (1+error)*goal[i])
        if caps is not None:
            upper = min(upper, caps[i])
        result.append([lower, upper])
    return result


def solve_intervals(bounds, swapped):
    """Exact nominal inequalities; boundary warnings do not widen intervals."""
    (lw, uw), (lh, uh) = bounds
    failures = []
    if lw > uw:
        failures.append('raw_width_interval_empty')
    if lh > uh:
        failures.append('raw_height_interval_empty')
    order_margin = uh-lw if swapped else uw-lh
    if not failures and (order_margin <= 0 if swapped else order_margin < 0):
        failures.append('strict_order_unavailable' if swapped else 'order_unavailable')
    tolerance = SETTINGS['boundary_atol_px']+SETTINGS['boundary_rtol']*max(
        abs(v) for pair in bounds for v in pair)
    margins = [uw-lw, uh-lh, order_margin]
    return dict(feasible=not failures, failed_constraints=failures,
        width_interval_px=list(bounds[0]), height_interval_px=list(bounds[1]),
        width_margin_px=margins[0], height_margin_px=margins[1],
        order_margin_px=order_margin, strict_order=swapped,
        numeric_boundary=any(abs(v) <= tolerance for v in margins),
        boundary_warning_px=tolerance)


def certificate_candidates(bounds, swapped, m):
    """Fixed numerical certificates, not a size/model/threshold sweep."""
    (lw, uw), (lh, uh) = bounds
    mid = [(lw+uw)*.5, (lh+uh)*.5]
    wrong_order = mid[0] >= mid[1] if swapped else mid[0] < mid[1]
    if wrong_order:
        t = (max(lw, lh)+min(uw, uh))*.5
        mid = ([(lw+min(uw, t))*.5, (max(lh, t)+uh)*.5] if swapped
               else [(max(lw, t)+uw)*.5, (lh+min(uh, t))*.5])
    nearest = [min(max(m[i+2], bounds[i][0]), bounds[i][1]) for i in (0, 1)]
    extreme = [lw, uh] if swapped else [uw, lh]
    values = []
    for pair in (mid, nearest, extreme):
        if pair not in values and all(math.isfinite(v) and v > 0 for v in pair):
            values.append(pair)
    return values


def check_sizes(edges, b, m, goal, frame):
    """Continuous nominal candidate check, separate from baseline bypass."""
    edges = _numbers(edges, 2, 'witness edges')
    positive = min(edges) > 0
    if not positive:
        return dict(passed=False, checks=dict(positive=False))
    order = (edges[0] < edges[1]) == (m[2] < m[3])
    ratios = [edges[i]/b[i+2] for i in (0, 1)]
    c, s = math.cos(m[4]), math.sin(m[4])
    u, v = (edges[0]*c*.5, edges[0]*s*.5), (-edges[1]*s*.5, edges[1]*c*.5)
    points = [[m[0]+sign*p[0], m[1]+sign*p[1]] for p in (u, v) for sign in (-1, 1)]
    local = [to_roi(p, frame) for p in points]
    limit = SETTINGS['roi_bound']+SETTINGS['roi_bound_tolerance']
    checks = dict(positive=positive, nominal_b_edge_bound=all(
        SETTINGS['edge_ratio_min'] <= v <= SETTINGS['edge_ratio_max'] for v in ratios),
        raw_order=order, joint10=joint_correct(edges, goal),
        roi_midpoints=all(abs(v) <= limit for p in local for v in p))
    return dict(passed=all(checks.values()), checks=checks,
        size_errors=list(size_errors(edges, goal)), raw_width_height=list(edges),
        points_roi=local, roi_min_margin=limit-max(abs(v) for p in local for v in p))


def _check_m(b, m, runtime_checks=None):
    tolerance = SETTINGS['legacy_guard_tolerance']
    if m[5] != b[5]:
        raise ValueError('Frozen midpoint changed B score')
    checks = dict(center_bound=math.hypot(m[0]-b[0], m[1]-b[1]) <=
        SETTINGS['max_center_over_b_short']*min(b[2:4])+tolerance,
        angle_bound=abs(wrap_pi(m[4]-b[4])) <= SETTINGS['max_angle_residual_rad']+tolerance,
        edge_bound=all(SETTINGS['edge_ratio_min']-tolerance <= m[i+2]/b[i+2] <=
            SETTINGS['edge_ratio_max']+tolerance for i in (0, 1)))
    # Native float32 guard arithmetic is authoritative for native M. It can
    # differ from scalar float64 by one ULP at an inherited guard boundary.
    # This never relaxes the nominal candidate bounds or the10% metric.
    if runtime_checks is not None:
        if set(runtime_checks) != set(checks) or any(type(v) is not bool for v in runtime_checks.values()):
            raise ValueError('Invalid native frozen-M source guard checks')
        actual = runtime_checks
    else:
        actual = checks
    if not all(actual.values()):
        raise ValueError('Frozen M violates original B guard: '+','.join(k for k, v in actual.items() if not v))
    return dict(float64_checks=checks, runtime_checks=runtime_checks,
        numeric_disagreements=[k for k in checks if runtime_checks is not None and checks[k] != runtime_checks[k]])


def diagnose_frame(b_original, m_original, gt_original, b_model=None, scale_xy=None,
                   runtime_frame=None, verify_witness=None, matched_gt_raw=None,
                   source_guard_checks=None):
    """GT-only analysis. Never output the certificate as a replacement box.

    verify_witness(edges, goal) checks log/exp reconstruction and the existing
    tensor ROI map. None means runtime feasibility has not been established.
    """
    gt = _box(gt_original, 5, 'GT')
    if b_original is None or m_original is None:
        if b_original is not None or m_original is not None:
            raise ValueError('B/M output frame identity differs')
        return dict(output=False, baseline_joint10=False, baseline_center_hit=False,
            baseline_center_and_joint10=False, constraint_status='no_output',
            delivered_size_reachable=False, exact_square_bypass=False,
            stages={}, witness=None, certificates_attempted=[])
    b, m = _box(b_original, 6, 'B original'), _box(m_original, 6, 'M original')
    source_guards = _check_m(b, m, source_guard_checks)
    frame = make_frame(b, b_model, scale_xy, runtime_frame)
    goal, swapped = goal_edges(gt, m), m[2] < m[3]
    caps = roi_caps(m, frame)
    only_bounds = intervals(b, goal)
    full_bounds = intervals(b, goal, (caps['raw_width_upper_px'], caps['raw_height_upper_px']))
    first, full = solve_intervals(only_bounds, swapped), solve_intervals(full_bounds, swapped)
    stages = dict(b_bounds_only=all(v[0] <= v[1] for v in only_bounds),
        b_bounds_and_order=first['feasible'], b_roi_and_order=full['feasible'])
    good = joint_correct(m[2:4], goal)
    center = math.hypot(m[0]-gt[0], m[1]-gt[1]) < 15.
    exact = check_sizes(goal, b, m, goal, frame)
    square = m[2] == m[3]
    attempted, witness = [], None
    if full['feasible'] and not square:
        for pair in certificate_candidates(full_bounds, swapped, m):
            nominal = check_sizes(pair, b, m, goal, frame)
            runtime = verify_witness(pair, goal) if verify_witness is not None else None
            attempted.append(dict(requested_raw_width_height=pair,
                nominal=nominal, runtime=runtime))
            if (verify_witness is not None and isinstance(runtime, dict) and
                    runtime.get('passed') is True) or (
                    verify_witness is None and nominal['passed']):
                witness = attempted[-1]
                break
    if square:
        status = 'frozen_square_baseline'
    elif not full['feasible']:
        status = 'numeric_unresolved' if full['numeric_boundary'] else 'nominal_infeasible'
    elif witness is None:
        status = 'numeric_unresolved'
    else:
        status = 'runtime_witness_verified' if verify_witness is not None else 'continuous_witness_only'
    # The zero-residual/full-M fallback may deliver a good baseline even when
    # the newly reconstructed rectangle midpoints fail a new ROI constraint.
    reachable = (True if good else False if square else
                 True if status == 'runtime_witness_verified' else
                 False if status == 'nominal_infeasible' else None)
    result = dict(output=True, baseline_joint10=good, baseline_center_hit=center,
        baseline_center_and_joint10=good and center,
        baseline_size_errors=list(size_errors(m[2:4], goal)),
        goal_raw_width_height=list(goal), m_raw_swap=swapped,
        exact_square_bypass=square, frame=frame, roi=caps, source_guards=source_guards,
        stages=stages, bound_order_intersection=first, full_intersection=full,
        exact_sorted_gt_size_inside_b_bound=all(SETTINGS['edge_ratio_min'] <= goal[i]/b[i+2]
            <= SETTINGS['edge_ratio_max'] for i in (0, 1)),
        exact_sorted_gt_size_passes_all_constraints=exact['passed'],
        baseline_rebuilt_midpoints_pass_nominal_constraints=check_sizes(m[2:4], b, m, goal, frame),
        constraint_status=status, delivered_size_reachable=reachable,
        baseline_bypass_reachable=good, witness=witness,
        certificates_attempted=attempted,
        note='OFFLINE GT certificate only; no box repair, model effect or joint RIoU/temporal guarantee.')
    if matched_gt_raw is not None:
        target = _numbers(matched_gt_raw, 2, 'matched GT raw edges')
        if min(target) <= 0:
            raise ValueError('Nonpositive matched GT edges')
        result['matched_training_target'] = dict(raw_width_height=list(target),
            inside_b_edge_bound=all(SETTINGS['edge_ratio_min'] <= target[i]/b[i+2] <=
                SETTINGS['edge_ratio_max'] for i in (0, 1)),
            keeps_m_raw_order=(target[0] < target[1]) == swapped,
            agrees_with_metric_axis_target=all(math.isclose(target[i], goal[i],
                rel_tol=1e-6, abs_tol=1e-4) for i in (0, 1)))
    return result
