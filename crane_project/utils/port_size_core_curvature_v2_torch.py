"""Server Torch losses and measured parameter gradients; no detector updates."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from crane_project.utils.port_size_reference_v1_torch import SizeReference, reference_loss
from crane_project.utils import port_size_core_curvature_v2 as core


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
        strength_role='measured_effective_gradients_not_coefficient_optimality')
    return measurement, total


def update(terms, model, optimizer, arm):
    measured, expected = gradient_measurement(terms, model, arm)
    weights = coefficients(arm)
    total = sum(terms[k]*weights[k] for k in weights)
    optimizer.zero_grad(); total.backward()
    actual = [p.grad for p in model.parameters() if p.requires_grad]
    if any(g is None or not bool(torch.isfinite(g).all()) for g in actual):
        raise ValueError('Missing/nonfinite total gradient')
    error = max(float((g-e).abs().max()) for g,e in zip(actual, expected))
    maximum = max(float(e.abs().max()) for e in expected)
    if error > 1e-5*max(maximum, 1e-7)+1e-9:
        raise ValueError('Component decomposition differs from actual backward')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    after = math.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters()))
    if after > 10.00001 or after <= 0 or not math.isfinite(after):
        raise ValueError('Invalid clipped gradient')
    measured.update(actual_postclip_norm=after, component_sum_backward_max_error=error,
                    actual_clip_retention=after/measured['total_preclip_norm'])
    optimizer.step()
    return measured
