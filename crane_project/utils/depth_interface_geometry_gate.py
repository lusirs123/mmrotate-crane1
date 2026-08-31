"""Source-val-only geometry gate for the analytical depth interface.

The gate measures whether an OBB stream preserves the metric geometry needed
by downstream scale/depth formulae.  It deliberately consumes only labelled
source validation boxes and a frozen source reference stream; target/fixed-dev
depth truth is neither required nor accepted.
"""

import math

import numpy as np


DEFAULT_TOLERANCES = dict(
    missing_rate_increase=0.005,
    center_mean_increase_px=0.5,
    center_p95_increase_px=2.0,
    angle_mean_increase_deg=0.5,
    angle_p95_increase_deg=2.0,
    relative_size_mean_abs_increase=0.01,
    relative_size_p95_abs_increase=0.03,
    q_mean_abs_increase=0.01,
    q_p95_abs_increase=0.03,
    q_reference_p99_expansion=0.02,
    q_envelope_exceedance_max=0.01,
)


def _canonical(box):
    """Return center, long edge, short edge, and long-edge angle."""
    cx, cy, width, height, angle = [float(value) for value in box[:5]]
    if width >= height:
        return cx, cy, width, height, angle
    return cx, cy, height, width, angle + math.pi / 2.0


def _periodic_angle_error(first, second):
    delta = float(first) - float(second)
    return abs(0.5 * math.atan2(math.sin(2.0 * delta),
                                math.cos(2.0 * delta)))


def _percentile(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=np.float64),
                               percentile)) if values else None


def _mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64))) \
        if values else None


def _summarize(metadata, boxes, domain=None):
    center = []
    angle_deg = []
    long_rel = []
    short_rel = []
    diagonal_rel = []
    q = []
    selected = 0
    missing = 0
    for meta, prediction in zip(metadata, boxes):
        if domain is not None and meta['domain'] != domain:
            continue
        selected += 1
        if prediction is None:
            missing += 1
            continue
        px, py, plong, pshort, pangle = _canonical(prediction)
        gx, gy, glong, gshort, gangle = _canonical(meta['gt_box'])
        if min(plong, pshort, glong, gshort) <= 0.0:
            raise RuntimeError('Depth-interface geometry contains zero edge')
        center.append(math.hypot(px - gx, py - gy))
        angle_deg.append(math.degrees(
            _periodic_angle_error(pangle, gangle)))
        long_rel.append((plong - glong) / glong)
        short_rel.append((pshort - gshort) / gshort)
        prediction_diagonal = math.hypot(plong, pshort)
        target_diagonal = math.hypot(glong, gshort)
        diagonal_rel.append(
            (prediction_diagonal - target_diagonal) / target_diagonal)
        q.append(math.log((plong / pshort) / (glong / gshort)))
    if selected == 0:
        raise RuntimeError('Depth-interface geometry scope is empty')

    def error_summary(values, include_signed=False):
        absolute = [abs(value) for value in values]
        result = dict(
            mean_abs=_mean(absolute),
            p95_abs=_percentile(absolute, 95.0),
            p99_abs=_percentile(absolute, 99.0))
        if include_signed:
            result['mean_signed'] = _mean(values)
        return result

    return dict(
        frame_count=selected,
        valid_count=selected - missing,
        missing_count=missing,
        missing_rate=float(missing / selected),
        center_error_px=error_summary(center),
        angle_error_deg=error_summary(angle_deg),
        long_relative_error=error_summary(long_rel, include_signed=True),
        short_relative_error=error_summary(short_rel, include_signed=True),
        diagonal_relative_error=error_summary(
            diagonal_rel, include_signed=True),
        q_log_aspect_residual=error_summary(q, include_signed=True),
        _q_values=q)


def geometry_metrics(metadata, boxes):
    """Compute reportable geometry metrics for all, real, and sim scopes."""
    if len(metadata) != len(boxes):
        raise RuntimeError('Geometry metadata/result length mismatch')
    scopes = dict(
        all=_summarize(metadata, boxes),
        real=_summarize(metadata, boxes, domain='real'),
        sim=_summarize(metadata, boxes, domain='sim'))
    for value in scopes.values():
        value.pop('_q_values', None)
    return scopes


def depth_interface_geometry_gate(metadata, candidate_boxes, reference_boxes,
                                  tolerances=None):
    """Compare a candidate with the frozen K1/SymEOOD source-val reference."""
    limits = dict(DEFAULT_TOLERANCES)
    if tolerances:
        unknown = sorted(set(tolerances) - set(limits))
        if unknown:
            raise RuntimeError(
                'Unknown depth-interface tolerance: ' + ', '.join(unknown))
        limits.update({key: float(value)
                       for key, value in tolerances.items()})
    if len(metadata) != len(candidate_boxes) or len(metadata) != len(
            reference_boxes):
        raise RuntimeError('Geometry gate input length mismatch')

    candidate_raw = {
        scope: _summarize(metadata, candidate_boxes, domain=domain)
        for scope, domain in (('all', None), ('real', 'real'), ('sim', 'sim'))}
    reference_raw = {
        scope: _summarize(metadata, reference_boxes, domain=domain)
        for scope, domain in (('all', None), ('real', 'real'), ('sim', 'sim'))}

    checks = {}
    envelope = {}

    def within(candidate_value, reference_value, allowance):
        return bool(
            candidate_value is not None and reference_value is not None
            and candidate_value <= reference_value + allowance)

    # The depth claim is simulation-only.  ``all`` prevents a small sim slice
    # from hiding broad source geometry damage; ``sim`` protects the physical
    # depth interface directly.  Real-domain continuity stays in the main gate.
    for scope in ('all', 'sim'):
        candidate = candidate_raw[scope]
        reference = reference_raw[scope]
        prefix = scope + '_'
        checks[prefix + 'missing_rate'] = (
            candidate['missing_rate'] <= reference['missing_rate']
            + limits['missing_rate_increase'])
        for metric, mean_limit, p95_limit in (
                ('center_error_px', 'center_mean_increase_px',
                 'center_p95_increase_px'),
                ('angle_error_deg', 'angle_mean_increase_deg',
                 'angle_p95_increase_deg')):
            checks[prefix + metric + '_mean'] = within(
                candidate[metric]['mean_abs'],
                reference[metric]['mean_abs'], limits[mean_limit])
            checks[prefix + metric + '_p95'] = within(
                candidate[metric]['p95_abs'],
                reference[metric]['p95_abs'], limits[p95_limit])
        for metric in ('long_relative_error', 'short_relative_error',
                       'diagonal_relative_error'):
            checks[prefix + metric + '_mean'] = within(
                candidate[metric]['mean_abs'],
                reference[metric]['mean_abs'],
                limits['relative_size_mean_abs_increase'])
            checks[prefix + metric + '_p95'] = within(
                candidate[metric]['p95_abs'],
                reference[metric]['p95_abs'],
                limits['relative_size_p95_abs_increase'])
        q_metric = 'q_log_aspect_residual'
        checks[prefix + 'q_mean'] = within(
            candidate[q_metric]['mean_abs'],
            reference[q_metric]['mean_abs'],
            limits['q_mean_abs_increase'])
        checks[prefix + 'q_p95'] = within(
            candidate[q_metric]['p95_abs'],
            reference[q_metric]['p95_abs'],
            limits['q_p95_abs_increase'])
        reference_p99 = reference[q_metric]['p99_abs']
        q_limit = (None if reference_p99 is None else
                   reference_p99 + limits['q_reference_p99_expansion'])
        candidate_q = candidate['_q_values']
        exceedance = (float(np.mean(
            np.abs(np.asarray(candidate_q, dtype=np.float64)) > q_limit))
            if candidate_q and q_limit is not None else 1.0)
        envelope[scope] = dict(
            reference_p99_abs=reference_p99,
            expansion=limits['q_reference_p99_expansion'],
            candidate_limit_abs=q_limit,
            candidate_exceedance_rate=exceedance,
            maximum_exceedance_rate=limits['q_envelope_exceedance_max'])
        checks[prefix + 'q_envelope_exceedance'] = (
            exceedance <= limits['q_envelope_exceedance_max'])

    def clean(scopes):
        result = {}
        for scope, metrics in scopes.items():
            result[scope] = dict(metrics)
            result[scope].pop('_q_values', None)
        return result

    return dict(
        definition=(
            'q=log((predicted_long/predicted_short)/'
            '(gt_long/gt_short)); source-val OBB geometry only'),
        gated_scopes=['all', 'sim'],
        real_scope_role='reported_only; real depth has no metric truth claim',
        tolerances=limits,
        q_reference_envelope=envelope,
        candidate=clean(candidate_raw),
        reference=clean(reference_raw),
        checks=checks,
        passed=all(checks.values()))
