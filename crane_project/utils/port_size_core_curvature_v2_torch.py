"""Server Torch losses and measured parameter gradients; no detector updates."""
import math
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import functional as F
from crane_project.utils.port_size_reference_v1_torch import SizeReference as OriginalSizeReference, reference_loss
from crane_project.utils import port_size_core_curvature_v2 as core


@contextmanager
def reference_precision():
    """Full FP32 candidate reference; restore B/midpoint's precision settings.

    TF32 operand rounding need not distribute across three separate VJPs and
    one combined VJP. Convolution backward can save the forward's TF32 flag,
    so the candidate forward must use the same local setting too.
    """
    matmul = torch.backends.cuda.matmul
    previous = (matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        (matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark) = previous


class SizeReference(OriginalSizeReference):
    """Same architecture/parameters/initialization; explicit candidate precision."""
    def forward(self, detached_p3):
        with reference_precision():
            return super().forward(detached_p3)


def loss_terms(logits, target, valid, constants):
    v1 = reference_loss(logits, target, valid)
    if not constants['eligible']:
        zero = logits.sum()*0.
        return dict(v1=v1, curv=zero, quad=zero)
    device = logits.device
    mask = torch.as_tensor(constants['mask'], dtype=torch.bool, device=device)
    if tuple(mask.shape) != tuple(logits.shape[2:]) or logits.shape[:2] != (1, 1):
        raise ValueError('Core loss expects one stride2 map')
    y = F.logsigmoid(logits[0, 0][mask].double())
    p = torch.as_tensor(constants['operator'], device=device, dtype=torch.float64)
    x = torch.as_tensor(constants['design'], device=device, dtype=torch.float64)
    w = torch.as_tensor(constants['weight'], device=device, dtype=torch.float64)
    beta = p @ y
    errors = torch.stack((-2*beta[3]-1., -2*beta[5]-1., -beta[4]))
    curv = (F.smooth_l1_loss(errors, torch.zeros_like(errors), reduction='none', beta=.1)
            *errors.new_tensor([.5, .5, 1.])).sum()
    residual = y-x @ beta
    quad = (w*F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction='none', beta=.1)).sum()
    return dict(v1=v1, curv=curv, quad=quad)


def coefficients(arm):
    if arm not in core.ARMS: raise ValueError('Unknown fixed arm')
    return dict(v1=1., curv=.25 if arm == 'a1' else 0., quad=.05 if arm == 'a1' else 0.)


def verify_auxiliary_autograd(constants, device):
    """Compare real Torch dy/dlogits with independent NumPy oracle, no update."""
    x, y = constants['coordinates'].T
    response = math.log(.8)-.5*((x/1.15)**2+y*y)+.05*np.sin(3*x)
    probabilities = np.exp(response)
    values = np.full(constants['mask'].shape, -4.)
    values[constants['mask']] = np.log(probabilities)-np.log1p(-probabilities)
    logits = torch.tensor(values, device=device, dtype=torch.float64)[None,None].requires_grad_()
    target = torch.as_tensor(constants['mask'], device=device, dtype=torch.float64)[None,None]
    terms = loss_terms(logits, target, target, constants)
    actual = torch.autograd.grad(.25*terms['curv']+.05*terms['quad'], logits)[0][0,0].detach().cpu().numpy()
    expected = core.numpy_terms(response, constants, True)['gradient_y']*(1-probabilities)
    difference = float(np.max(np.abs(actual[constants['mask']]-expected)))
    relative = difference/max(float(np.max(np.abs(expected))), 1e-8)
    outside = float(np.max(np.abs(actual[~constants['mask']]))) if (~constants['mask']).any() else 0.
    if relative > 1e-4 or outside != 0.:
        raise ValueError('Actual Torch auxiliary derivative differs from independent oracle')
    return dict(relative_max_error=relative, maximum_absolute_error=difference,
                outside_core_gradient_max=outside, passed=True)


def gradient_measurement(terms, model, arm, clip=10.):
    """autograd.grad does not populate .grad or consume the subsequent backward.

    All component post-clip norms are projections using the SAME measured total
    clip scale; components are not separately clipped. Zero quad gradients are
    legal on an exact quadratic response. Ratios are measurements, not tuning.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    weights = coefficients(arm)
    with reference_precision():
        raw = {k: torch.autograd.grad(v, [p for _, p in named], retain_graph=True)
               for k, v in terms.items()}
    if any(not bool(torch.isfinite(g).all()) for gs in raw.values() for g in gs):
        raise ValueError('Nonfinite component gradient')
    def norm(gs):
        return math.sqrt(sum(float(g.double().square().sum()) for g in gs))
    weighted = {k: tuple(g*weights[k] for g in gs) for k, gs in raw.items()}
    new = tuple(a+b for a, b in zip(weighted['curv'], weighted['quad']))
    total = tuple(a+b for a, b in zip(weighted['v1'], new))
    before = norm(total)
    if not math.isfinite(before) or before <= 0:
        raise ValueError('Inactive total parameter gradient')
    scale = min(1., clip/(before+1e-6))
    original, added = norm(raw['v1']), norm(new)
    dot = sum(float(a.double().mul(b.double()).sum()) for a, b in zip(raw['v1'], new))
    groups = {}
    for group in ('stem', 'output'):
        idx = [i for i, (name, _) in enumerate(named) if name.startswith(group+'.')]
        def select(gs): return [gs[i] for i in idx]
        baseline = norm(select(raw['v1'])); extra = norm(select(new))
        groups[group] = dict(raw_norms={k:norm(select(v)) for k,v in raw.items()},
            weighted_norms={k:norm(select(v)) for k,v in weighted.items()},
            new_norm=extra, new_to_v1=None if baseline == 0 else extra/baseline,
            total_before=norm(select(total)), total_after_projected=norm(select(total))*scale)
    measurement = dict(losses={k:float(v.detach()) for k,v in terms.items()}, coefficients=weights,
        raw_gradient_norms={k:norm(v) for k,v in raw.items()},
        weighted_gradient_norms={k:norm(v) for k,v in weighted.items()},
        new_gradient_norm=added, original_gradient_norm=original,
        new_to_original=None if original == 0 else added/original,
        new_original_cosine=None if original*added == 0 else dot/(original*added),
        total_preclip_norm=before, clip_norm=clip, common_clip_scale=scale,
        component_postclip_projected_norms={k:norm(v)*scale for k,v in weighted.items()},
        new_postclip_projected_norm=added*scale, groups=groups,
        uncancelled_tensor_scales=[dict(name=name, dtype=str(p.dtype),
            dtype_epsilon=float(torch.finfo(p.dtype).eps),
            uncancelled_norm=sum(norm([gs[i]]) for gs in weighted.values()),
            uncancelled_max=sum(float(gs[i].abs().max()) for gs in weighted.values()))
            for i,(name,p) in enumerate(named)],
        backward_precision=core.GRADIENT_CHECK['backward_precision'],
        strength_role='measured_effective_gradients_not_coefficient_optimality')
    return measurement, total


def tensor_comparison(name, actual, expected, operands):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('Gradient comparison shape/dtype differs')
    if any(not bool(torch.isfinite(g).all()) for g in [actual,expected]+list(operands)):
        raise ValueError('Nonfinite gradient comparison')
    delta = actual.double()-expected.double()
    return dict(name=name, dtype=str(actual.dtype), dtype_epsilon=float(torch.finfo(actual.dtype).eps),
        error_norm=float(delta.square().sum().sqrt()), error_max=float(delta.abs().max()),
        uncancelled_norm=sum(float(g.double().square().sum().sqrt()) for g in operands),
        uncancelled_max=sum(float(g.abs().max()) for g in operands))


def consistency_failure(arm, consistency, measured):
    worst = max(consistency['tensors'], key=lambda r:max(r['norm_error_fraction'], r['max_error_fraction']))
    failure = ValueError('Gradient verification failed: arm=%s parameter=%s '
        'max_error=%.9g allowed_max=%.9g norm_error=%.9g allowed_norm=%.9g' %
        (arm,worst['name'],worst['error_max'],worst['allowed_max_error'],
         worst['error_norm'],worst['allowed_norm_error']))
    failure.details = dict(arm=arm, gradient_consistency=consistency, measurement=measured)
    return failure


def update(terms, model, optimizer, arm, logits):
    """Keep native total.backward; verify at logits and with one same-seed VJP.

    Three separately seeded convolution backwards do not give a robust
    equality oracle for a single combined backward. Their difference remains
    recorded, while the strict checks below use identical upstream seeds.
    """
    measured, component_sum = gradient_measurement(terms, model, arm)
    weights = coefficients(arm)
    total = sum(terms[k]*weights[k] for k in weights)
    with reference_precision():
        seed_terms = [torch.autograd.grad(terms[k],logits,retain_graph=True)[0]*weights[k]
                      for k in weights]
    seed_expected = sum(seed_terms)
    observed = []
    hook = logits.register_hook(lambda g: observed.append(g.detach().clone()))
    optimizer.zero_grad()
    try:
        with reference_precision():
            total.backward(retain_graph=True)
    finally:
        hook.remove()
    if len(observed) != 1:
        raise ValueError('Missing/repeated shared-logit backward seed')
    seed_check = core.gradient_consistency([
        tensor_comparison('shared_logits', observed[0],seed_expected,seed_terms)])
    if not seed_check['passed']:
        raise consistency_failure(arm,seed_check,measured)
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    with reference_precision():
        expected = torch.autograd.grad(logits,[p for _,p in named],
                                       grad_outputs=observed[0],retain_graph=True)
    actual = [p.grad for p in model.parameters() if p.requires_grad]
    if any(g is None or not bool(torch.isfinite(g).all()) for g in actual):
        raise ValueError('Missing/nonfinite total gradient')
    if len(actual) != len(expected):
        raise ValueError('Parameter gradient count differs')
    comparisons = [tensor_comparison(name,g,e,[e])
                   for (name,_),g,e in zip(named,actual,expected)]
    consistency = core.gradient_consistency(seed_check['tensors']+comparisons)
    if not consistency['passed']:
        raise consistency_failure(arm,consistency,measured)
    component_comparisons = []
    for g,e,scale in zip(actual, component_sum, measured['uncancelled_tensor_scales']):
        if g.shape != e.shape or g.dtype != e.dtype:
            raise ValueError('Parameter gradient shape/dtype differs')
        difference = g.double()-e.double()
        component_comparisons.append(dict(scale, error_norm=float(difference.square().sum().sqrt()),
            error_max=float(difference.abs().max())))
    component_diagnostic = core.gradient_consistency(component_comparisons)
    component_diagnostic['role'] = 'diagnostic_only_different_convolution_backward_seeds'
    error = max(row['error_max'] for row in component_comparisons)
    actual_before = math.sqrt(sum(float(g.double().square().sum()) for g in actual))
    if not math.isfinite(actual_before) or actual_before <= 0:
        raise ValueError('Inactive/nonfinite actual total gradient')
    group_before={group:math.sqrt(sum(float(g.double().square().sum())
        for (name,_),g in zip(named,actual) if name.startswith(group+'.')))
        for group in measured['groups']}
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    after = math.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters()))
    if after > 10.00001 or after <= 0 or not math.isfinite(after):
        raise ValueError('Invalid clipped gradient')
    retention=after/actual_before
    measured['component_sum_preclip_norm']=measured['total_preclip_norm']
    measured['component_sum_projected_clip_scale']=measured['common_clip_scale']
    measured['total_preclip_norm']=actual_before
    measured['common_clip_scale']=retention
    measured['component_postclip_projected_norms']={k:v*retention for k,v in measured['weighted_gradient_norms'].items()}
    measured['new_postclip_projected_norm']=measured['new_gradient_norm']*retention
    for group,report in measured['groups'].items():
        report.update(actual_preclip_norm=group_before[group],
            actual_postclip_norm=math.sqrt(sum(float(g.double().square().sum())
                for (name,_),g in zip(named,actual) if name.startswith(group+'.'))),
            total_after_projected=report['total_before']*retention)
    measured.update(actual_preclip_norm=actual_before, actual_postclip_norm=after,
                    component_sum_backward_max_error=error, gradient_consistency=consistency,
                    parameter_component_sum_diagnostic=component_diagnostic,
                    actual_clip_retention=retention)
    optimizer.step()
    return measured
