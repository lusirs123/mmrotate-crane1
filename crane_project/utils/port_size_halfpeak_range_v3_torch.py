"""Half-peak range loss and existing strict same-seed gradient verification."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from crane_project.utils.port_size_reference_v1_torch import reference_loss
from crane_project.utils.port_size_core_curvature_v2_torch import SizeReference, reference_precision
from crane_project.utils import port_size_halfpeak_range_v3 as core


def coefficients(arm):
    if arm not in core.ARMS: raise ValueError('Unknown fixed arm')
    return {k:(v if arm=='a1' or k=='v1' else 0.) for k,v in core.COEFFICIENTS.items()}


def loss_terms(logits, target, valid, constants):
    v1=reference_loss(logits,target,valid)
    if not constants['eligible']:
        zero=logits.sum()*0.
        return dict(v1=v1,inner=zero,outer=zero)
    if tuple(logits.shape[2:])!=tuple(constants['shape']) or logits.shape[:2]!=(1,1):
        raise ValueError('Range supervision expects one fixed stride2 map')
    p=logits.sigmoid()[0,0].double()
    device=logits.device
    indices=torch.as_tensor(constants['indices'],dtype=torch.long,device=device)
    weights=torch.as_tensor(constants['weights'],dtype=torch.float64,device=device)
    mask=torch.as_tensor(constants['background_mask'],dtype=torch.bool,device=device)
    background=p[mask].median()
    samples=(p.reshape(-1)[indices]*weights).sum(dim=1)
    # Only weak contrast uses a fixed floor. Ratios stay signed/unclipped so
    # below-background samples retain a gradient; center/background stay live.
    amplitude=(samples[-1]-background).clamp_min(core.RANGE_SETTINGS['contrast_floor'])
    ratios=(samples[:-1]-background)/amplitude
    desired=torch.as_tensor(constants['sample_target'],dtype=torch.float64,device=device)
    point_loss=F.smooth_l1_loss(ratios,desired,reduction='none',beta=.1)
    inner=torch.as_tensor(constants['inner'],dtype=torch.long,device=device)
    outer=torch.as_tensor(constants['outer'],dtype=torch.long,device=device)
    return dict(v1=v1,inner=point_loss[inner].mean(),outer=point_loss[outer].mean())


def verify_auxiliary_autograd(constants, device):
    """Real Torch derivatives vs independent NumPy directional differences.

    Test BOTH contrast-floor branches, each loss, peak/background gradients
    and nonlocal support. No optimizer/model update and no coefficient search.
    """
    yy,xx=np.mgrid[:constants['shape'][0],:constants['shape'][1]]
    records=[]
    for peak in (.8,.115):
        p=.1+.007*np.sin(xx*.019+yy*.037)+1e-10*(yy*len(xx[0])+xx)
        # A UNIQUE lower median with a fixed gap avoids a finite difference
        # crossing a sorting boundary on large real-view background supports.
        # This controls the derivative toy only, never the learned image loss.
        background_indices=np.flatnonzero(constants['background_mask'].reshape(-1))
        middle=(len(background_indices)-1)//2
        p.reshape(-1)[background_indices[:middle]]=.08
        p.reshape(-1)[background_indices[middle]]=.10
        p.reshape(-1)[background_indices[middle+1:]]=.12
        p.reshape(-1)[constants['indices'][-1]]=peak
        z=np.log(p)-np.log1p(-p)
        logits=torch.tensor(z,dtype=torch.float64,device=device)[None,None].requires_grad_()
        target=torch.ones_like(logits)*.5;valid=torch.ones_like(logits)
        terms=loss_terms(logits,target,valid,constants)
        floor=core.numpy_terms(p,constants)['contrast_floor_active']
        for name in ('inner','outer'):
            derivative=torch.autograd.grad(terms[name],logits,retain_graph=True)[0][0,0].detach().cpu().numpy()
            direction=np.sin(xx*.071+yy*.053+.3)
            step=1e-6
            def objective(values):
                return core.numpy_terms(np.exp(-np.logaddexp(0.,-values)),constants)[name]
            finite=(objective(z+step*direction)-objective(z-step*direction))/(2*step)
            actual=float(np.sum(derivative*direction))
            relative=abs(finite-actual)/max(abs(actual),abs(finite),1e-8)
            support=constants['background_mask'].copy();support.reshape(-1)[constants['indices'].ravel()]=True
            outside=float(np.max(np.abs(derivative[~support]))) if (~support).any() else 0.
            if relative>1e-4 or outside!=0. or not np.isfinite(derivative).all():
                raise ValueError('Actual range derivative differs from independent NumPy oracle')
            records.append(dict(term=name,contrast_floor_active=floor,
                finite_difference_relative_error=relative,outside_support_gradient_max=outside,passed=True))
    return dict(cases=records,passed=True)


def gradient_measurement(terms, model, arm, clip=10.):
    """autograd.grad does not populate .grad or consume the subsequent backward.

    All component post-clip norms are projections using the SAME measured total
    clip scale; components are not separately clipped. Zero auxiliary gradients are
    legal on an exactly matching range profile. Ratios are measurements, not tuning.
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
    new = tuple(a+b for a, b in zip(weighted['inner'], weighted['outer']))
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
