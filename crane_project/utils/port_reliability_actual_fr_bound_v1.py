"""One fixed-budget count solve restricted by a verified feasible FA upper bound."""
import math
import time
import numpy as np
from crane_project.utils import port_reliability_actual_fr_v1 as prior

VERSION = 'port_reliability_actual_fr_bound_v1'
SETTINGS = dict(prior.SETTINGS, fitting_FA_upper=59, solver_calls=1,
                known_solution_warm_start=False, fallback_is_new_candidate=False)


def witness_certificate(data, weights, upper):
    z = (data['design'] @ np.asarray(weights) >= 0).astype(float)
    cert = prior.validate_solution(data, weights, z)
    if not isinstance(upper, int) or upper < 0 or cert['FA'] > upper:
        raise ValueError('FA upper bound lacks a feasible witness')
    return cert


def solve(rows, witness, upper=59):
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import csr_matrix, diags, bmat
    import scipy
    data = prior.build(rows)
    known = witness_certificate(data, witness, upper)
    d = data['design']; n = len(d); y = data['labels']; m = data['big_m']
    eps = prior.SETTINGS['classification_margin']
    matrix = bmat([[csr_matrix(d), -diags(m)]], format='csr')
    groups = np.zeros((len(data['limits']), 4+n)); limits = []
    for j, value in enumerate(data['limits'].values()):
        groups[j, 4+np.array(value['indices'])] = 1.
        limits.append(value['max_FR'])
    error_rejections = csr_matrix(np.r_[np.zeros(4), y.astype(float)][None, :])
    constraints = [LinearConstraint(matrix, eps-m, np.full(n, -eps)),
        LinearConstraint(csr_matrix(groups), np.zeros(len(limits)), np.array(limits)),
        LinearConstraint(error_rejections, int(y.sum())-upper, np.inf)]
    objective = np.r_[np.zeros(4), -y.astype(float)]
    options = dict(time_limit=60., node_limit=100000, mip_rel_gap=0., presolve=True)
    start = time.monotonic()
    answer = milp(c=objective, integrality=np.r_[np.zeros(4), np.ones(n)],
        bounds=Bounds(np.r_[np.full(4, -10.), np.zeros(n)],
                      np.r_[np.full(4, 10.), np.ones(n)]),
        constraints=constraints, options=options.copy())
    raw = getattr(answer, 'mip_dual_bound', None)
    dual = float(raw) if raw is not None and np.isfinite(raw) else None
    lower = max(0, math.ceil(int(y.sum())+dual-1e-6)) if dual is not None else 0
    if answer.status == 2:
        raise ValueError('Solver infeasibility contradicts verified witness')
    model = None; cert = None
    incumbent = answer.x is not None
    if incumbent:
        cert = prior.validate_solution(data, answer.x[:4], answer.x[4:])
        if cert['FA'] > upper or lower > cert['FA'] or abs(float(answer.fun)+cert['ED']) > 1e-6:
            raise ValueError('Actual objective/signs or FA bound violated')
    if lower > upper:
        raise ValueError('Solver bound contradicts known feasible witness')
    optimal = bool(incumbent and answer.status == 0 and lower == cert['FA'])
    solver = dict(status=int(answer.status), message=str(answer.message), incumbent=incumbent,
        outcome='OPTIMAL_WITHIN_FIXED_DOMAIN' if optimal else 'FEASIBLE_BUDGET_UNRESOLVED' if incumbent else 'NO_SOLVER_CANDIDATE',
        FA_lower=lower, FA_upper=cert['FA'] if cert else None,
        known_feasible_FA_upper=known['FA'], raw_dual_bound=dual,
        absolute_FA_gap=cert['FA']-lower if cert else None,
        optimal_within_fixed_domain=optimal, options=options,
        elapsed_seconds=time.monotonic()-start,
        node_count=int(getattr(answer, 'mip_node_count', 0) or 0), scipy_version=scipy.__version__,
        objective_cut_FA_le=upper, warm_start_supplied=False)
    if incumbent:
        model = dict(feature_names=list(prior.video.fusion.simple.FEATURES),
            weights=answer.x[:4].tolist(), mean=data['mean'].tolist(), scale=data['scale'].tolist(),
            certificate=cert, solver=solver, train_rows=n, bad_rows=int(y.sum()), good_rows=int((~y).sum()),
            fit_rejections={r['image']:bool(z) for r,z in zip(data['rows'], answer.x[4:] >= .5)},
            fitting_videos=list(data['limits']), decision='accept iff logit < 0',
            NP_statistical_guarantee=False, unrestricted_capacity_claim=False)
    return dict(model=model, solver=solver, known_control_certificate=known,
                known_control_weights=list(witness), known_control_is_candidate=False)
