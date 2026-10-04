"""GT-only training projection and offline paired SIZE evidence. No TEST API."""
from copy import deepcopy
import math
import numpy as np
from crane_project.utils import port_size_reference_v1 as geometry
from crane_project.utils import port_simple_component_reliability_v1 as simple

VERSION = 'port_size_core_curvature_v2'
ARMS = ('a0', 'a1')
COEFFICIENTS = {'v1': 1., 'curv': .25, 'quad': .05}


def projection(gt, meta):
    """Constants only; NEVER called by online size reading."""
    points, valid = geometry.grid(meta)
    b = geometry.box_model(gt, meta)
    target, _ = geometry.target_map(gt, meta)
    core = valid & (target >= .5)
    info = dict(eligible=False, reason='unresolved_GT_short', cells=int(core.sum()),
                short_model_px=float(b[3]), condition=None)
    if b[3] < 16.: return info
    if core.sum() < 16:
        info['reason'] = 'insufficient_GT_core'; return info
    d = points[core]-b[:2]; c, s = math.cos(b[4]), math.sin(b[4])
    x = 4*(d[:, 0]*c+d[:, 1]*s)/b[2]
    y = 4*(-d[:, 0]*s+d[:, 1]*c)/b[3]
    design = np.column_stack((np.ones(len(x)), x, y, x*x, x*y, y*y))
    # Exact geometry weights, not float32 target rounding, preserve ideal K=I.
    weight = np.exp(-.5*(x*x+y*y)); weight /= weight.sum()
    a = design*np.sqrt(weight[:, None])
    condition = float(np.linalg.cond(a)); info['condition'] = condition
    if np.linalg.matrix_rank(a) < 6 or not np.isfinite(condition) or condition > 1e6:
        info['reason'] = 'ill_conditioned_GT_core'; return info
    q, r = np.linalg.qr(a, mode='reduced')
    operator = np.linalg.solve(r, q.T)*np.sqrt(weight)[None, :]
    info.update(eligible=True, reason='eligible', mask=core, design=design,
                weight=weight, operator=operator, coordinates=np.column_stack((x, y)))
    return info


def public_projection(value):
    return {k: value[k] for k in ('eligible', 'reason', 'cells', 'short_model_px', 'condition')}


def huber(a):
    a = np.asarray(a, dtype=float)
    return np.where(np.abs(a) <= .1, a*a/.2, np.abs(a)-.05)


def huber_gradient(a):
    return np.clip(np.asarray(a, dtype=float)/.1, -1., 1.)


def numpy_terms(y, constants, gradient=False):
    """Independent float64 oracle for the auxiliary y -> loss derivative."""
    y = np.asarray(y, dtype=float)
    p, x, w = (constants[k] for k in ('operator', 'design', 'weight'))
    if y.shape != (x.shape[0],) or not np.isfinite(y).all():
        raise ValueError('Nonfinite/core shape mismatch')
    beta = p.dot(y)
    k = np.array([[-2*beta[3], -beta[4]], [-beta[4], -2*beta[5]]])
    errors = np.array([k[0, 0]-1., k[1, 1]-1., k[0, 1]])
    residual = y-x.dot(beta)
    result = dict(curv=float(np.array([.5, .5, 1.]).dot(huber(errors))),
                  quad=float(w.dot(huber(residual))), beta=beta, curvature=k)
    if gradient:
        gb = np.zeros(6)
        h = huber_gradient(errors)
        gb[3], gb[5], gb[4] = -h[0], -h[1], -h[2]
        gcurv = p.T.dot(gb)
        gr = w*huber_gradient(residual)
        gquad = gr-p.T.dot(x.T.dot(gr))
        result['gradient_y'] = .25*gcurv+.05*gquad
    return result


def numerical_probe(constants):
    """New loss only, analytic toy y; no learned-image/gradient-strength claim."""
    x, y = constants['coordinates'].T
    ideal = numpy_terms(-.5*(x*x+y*y), constants)
    cases = []
    for axis in (0, 1):
        for factor in (.85, 1.15):
            scale = [1., 1.]; scale[axis] = factor
            response = math.log(.9)-.5*((x/scale[0])**2+(y/scale[1])**2)
            terms = numpy_terms(response, constants, True)
            expected = np.diag(1/np.square(scale))
            derivative = terms['gradient_y'].dot((x if axis == 0 else y)**2/factor**3)
            direction = np.sin(np.arange(len(response))*.37)
            step = 1e-6
            def loss(v):
                t = numpy_terms(v, constants); return .25*t['curv']+.05*t['quad']
            finite = (loss(response+step*direction)-loss(response-step*direction))/(2*step)
            analytic = float(terms['gradient_y'].dot(direction))
            relative = abs(finite-analytic)/max(abs(finite), abs(analytic), 1e-8)
            cases.append(dict(axis=axis, factor=factor,
                curvature_error=float(np.max(np.abs(terms['curvature']-expected))),
                scale_derivative=float(derivative), finite_difference_relative_error=float(relative),
                passed=bool(np.max(np.abs(terms['curvature']-expected)) <= 1e-6
                    and derivative*(factor-1) > 0 and relative <= 1e-4)))
    error = float(np.max(np.abs(ideal['curvature']-np.eye(2))))
    return dict(ideal_curvature_error=error, cases=cases,
                passed=error <= 1e-6 and all(c['passed'] for c in cases))


GRADIENT_CHECK = dict(
    version='uncancelled_component_scale_v1',
    relative_dtype_epsilon_multiplier=128., absolute_tolerance=1e-12,
    backward_precision='reference_only_forward_backward_TF32_off_restore_flags',
    norm_checks='per_parameter_L2_and_max_uncancelled_weighted_components')


def gradient_consistency(tensors):
    """Roundoff check relative to operands, never the cancelled result.

    Scalars come from float64 reductions of parameter gradients. Both L2 and
    maximum errors are bounded for EACH tensor so another tensor cannot mask
    a missing/wrong gradient. This is an engineering check, not a loss weight.
    """
    if not tensors:
        raise ValueError('No parameter gradient comparisons')
    records = []
    for row in tensors:
        keys = ('dtype_epsilon', 'error_norm', 'error_max',
                'uncancelled_norm', 'uncancelled_max')
        if any(not math.isfinite(row[k]) or row[k] < 0 for k in keys) or row['dtype_epsilon'] <= 0:
            raise ValueError('Invalid/nonfinite gradient comparison')
        relative = GRADIENT_CHECK['relative_dtype_epsilon_multiplier']*row['dtype_epsilon']
        limits = {key: relative*row['uncancelled_'+key]+GRADIENT_CHECK['absolute_tolerance']
                  for key in ('norm', 'max')}
        records.append(dict(row, relative_tolerance=relative,
            allowed_norm_error=limits['norm'], allowed_max_error=limits['max'],
            norm_error_fraction=row['error_norm']/limits['norm'],
            max_error_fraction=row['error_max']/limits['max'],
            passed=row['error_norm'] <= limits['norm'] and row['error_max'] <= limits['max']))
    return dict(version=GRADIENT_CHECK['version'], tensors=records,
                passed=all(row['passed'] for row in records))


def acceptance_bound(total_frames, outputs, good_outputs, count):
    if not 0 <= good_outputs <= outputs <= total_frames or not 0 <= count <= outputs:
        raise ValueError('Invalid support/acceptance count')
    return dict(frames=total_frames, outputs=outputs, good_outputs=good_outputs,
                accepted=count, minimum_bad_accepted=max(0, count-good_outputs),
                role='offline_coverage_constraint_not_prediction_or_deployment_goal')


def online_reading(probability, final_box, image_size, meta, runtime, settings):
    """Image/box-only API. GT/domain/video/time explicitly absent."""
    from crane_project.utils import port_midpoint_reliability_v1 as midpoint
    before = deepcopy(final_box)
    frozen = runtime.decide(final_box, image_size)
    score_only = runtime.decide(final_box, image_size, 'score_only')
    if frozen['final_box_original'] != before or frozen['center_accepted'] != (before is not None):
        raise ValueError('Frozen decision changed output/center')
    ref = (midpoint.template_reference(probability, final_box[:5], meta, settings)
           if final_box is not None else dict(defined=False, reason='no_detector_output'))
    reading = (geometry.size_reading(final_box[:5], ref, settings['risk_scale'])
               if final_box is not None else dict(defined=False, risk=None))
    if before != final_box: raise ValueError('Reference modified frozen box')
    if score_only['final_box_original'] != before or score_only['center_accepted'] != (before is not None):
        raise ValueError('Frozen score-only changed output/center')
    return dict(reference=ref, reading=reading, frozen=frozen, frozen_score_only=score_only)
