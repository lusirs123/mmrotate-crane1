"""Auxiliary TRAIN snapshot attribution; no formal detector changes."""
import math

VERSION = 'port_reliability_oof_gradient_chain_v1'
LIMIT_BYTES = 4 * 2**30
REGRESSION_BUDGET = 10.0  # inherited global clip norm; no coefficient search
FINITE_UPDATES = 64  # original32 warmup plus32 updates to verify the plateau


def analytic_kld_terms(p, q):
    """Same symmetric Gaussian distance, expressed in principal-axis coordinates.

    This is a diagnostic reference, not a replacement for the frozen operator.
    Inputs may be FP32 or FP64. Preserve the original one-pixel size floor.
    """
    import torch
    pw, ph = p[:, 2].clamp(min=1), p[:, 3].clamp(min=1)
    qw, qh = q[:, 2].clamp(min=1), q[:, 3].clamp(min=1)
    angle = p[:, 4] - q[:, 4]
    c2, s2 = angle.cos().square(), angle.sin().square()
    shape = .5 * (c2 * ((pw/qw).square() + (ph/qh).square() +
                        (qw/pw).square() + (qh/ph).square()) +
                  s2 * ((pw/qh).square() + (ph/qw).square() +
                        (qw/ph).square() + (qh/pw).square()) - 4)
    delta = p[:, :2] - q[:, :2]
    def projection(a):
        x = delta[:, 0]*a.cos() + delta[:, 1]*a.sin()
        y = -delta[:, 0]*a.sin() + delta[:, 1]*a.cos()
        return x, y
    px, py = projection(p[:, 4]); qx, qy = projection(q[:, 4])
    center_p = 2*((px/pw).square() + (py/ph).square())
    center_q = 2*((qx/qw).square() + (qy/qh).square())
    return dict(shape=shape, center_p=center_p, center_q=center_q,
                raw=shape + center_p + center_q)


def reference_loss(p, q, weight, avg_factor, loss_weight):
    """Exact original compression/reduction, diagnostic principal-axis form."""
    raw = analytic_kld_terms(p, q)['raw'].clamp(0, 1e4)
    loss = ((1+raw).sqrt()-1).clamp(max=10)
    if weight is not None: loss = loss*weight
    return loss.sum()/float(avg_factor)*float(loss_weight)


def bounded_decode(coder, anchors, deltas, limit):
    """Independent differentiable copy of head bounds for VJP verification."""
    import torch
    b = coder.decode(anchors, deltas)
    return torch.stack((b[:, 0].clamp(0, limit-1), b[:, 1].clamp(0, limit-1),
                        b[:, 2].clamp(1, limit), b[:, 3].clamp(1, limit),
                        b[:, 4]), -1)


def canonical_error(p, q):
    import torch
    ps = p[:, 2:4].sort(dim=1, descending=True)[0]
    qs = q[:, 2:4].sort(dim=1, descending=True)[0]
    pa = p[:, 4] + (p[:, 2] < p[:, 3]).to(p.dtype)*math.pi/2
    qa = q[:, 4] + (q[:, 2] < q[:, 3]).to(q.dtype)*math.pi/2
    angle = torch.remainder(pa-qa+math.pi/2, math.pi)-math.pi/2
    return dict(relative_long_short_error=(ps/qs-1).detach().cpu().tolist(),
                center_error_px=(p[:, :2]-q[:, :2]).norm(dim=1).detach().cpu().tolist(),
                canonical_axis_error_deg=(angle.abs()*180/math.pi).detach().cpu().tolist(),
                covariance_condition=((ps[:, 0]/ps[:, 1])**2).detach().cpu().tolist())


def validate_boundary(snapshot, completion, name):
    expected = completion['completed_updates'] + (0 if name == 'previous' else 1)
    if name not in ('previous', 'anomaly') or snapshot['iteration']+1 != expected:
        raise ValueError('Snapshot boundary differs from diagnostic stop')
    if not snapshot['diagnostic_only']:
        raise ValueError('Only captured diagnostic snapshots allowed')
    if completion['status'] != 'ANOMALY_QUARANTINED_BEFORE_OPTIMIZER':
        raise ValueError('Only the captured finite-gradient anomaly is supported')


def regression_budget_backward(total, regression, parameters, budget=REGRESSION_BUDGET):
    """Auxiliary-only gradient repair: bound main-regression contribution first.

    Forward losses and targets remain unchanged. The detached multiplier makes
    backward use g_other + min(1, 10/||g_reg||)*g_reg. The ordinary global clip
    and optimizer follow. This does not prove training or downstream accuracy.
    """
    import torch
    if budget != REGRESSION_BUDGET:
        raise ValueError('Fixed inherited budget required; no tuning here')
    if any(p.grad is not None for p in parameters):
        raise ValueError('Clear gradients before auxiliary budget backward')
    values = torch.autograd.grad(regression,parameters,retain_graph=True,allow_unused=True)
    present = [g for g in values if g is not None]
    if not present or not all(bool(torch.isfinite(g).all()) for g in present):
        raise FloatingPointError('Invalid regression gradient before budget scaling')
    norm = torch.stack([g.detach().norm() for g in present]).norm()
    if not torch.isfinite(norm): raise FloatingPointError('Invalid regression gradient norm')
    scale = min(1.,budget/(float(norm)+1e-12))
    # Cancel the original regression branch before backward, then add its
    # scaled parameter-gradient explicitly. A surrogate with coefficient
    # (scale-1) loses precision when FP32 scale is near machine epsilon.
    (total-regression).backward()
    for p,g in zip(parameters,values):
        if g is None: continue
        if p.grad is None: p.grad=g.detach().mul(scale)
        else: p.grad.add_(g.detach(),alpha=scale)
    del values, present
    return dict(regression_gradient_norm=float(norm),regression_gradient_scale=scale,
                regression_gradient_budgeted_norm=float(norm)*scale,
                regression_budget=budget,forward_loss_unchanged=True)

