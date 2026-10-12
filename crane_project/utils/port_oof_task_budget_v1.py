"""Opt-in auxiliary task-gradient budgets; forward loss contracts stay intact."""
from copy import deepcopy

VERSION = 'per_task_gradient_budget_v1'
MEMORY_REVISION = 'release_loss_graphs_optional_raw_v2'
TASKS = ('main_cls', 'main_reg', 'auxiliary')
BUDGET = 10.  # inherited global clip scale, fixed before the counterfactual


def task_budget_backward(terms, parameters, measure_raw=False):
    """Accumulate sum_t min(1,10/||g_t||)*g_t explicitly in FP32.

    No total-minus-task cancellation, no detached targets, no loss rescaling in
    forward. Temporaries are bounded by parameter count, not iteration count.
    """
    import torch
    from crane_project.utils.port_reliability_oof_prevention_v1 import finite_gradient_norm
    if tuple(terms) != TASKS:
        raise ValueError('Exactly the three reviewed auxiliary task losses required')
    if any(p.grad is not None for p in parameters):
        raise ValueError('Clear gradients before task budgeting')
    if not isinstance(measure_raw, bool):
        raise ValueError('Raw gradient measurement must be explicitly boolean')
    # Full raw accumulation is diagnostic-only and never controls the update.
    raw = [None]*len(parameters) if measure_raw else None
    stats = {}
    for index,(name,loss) in enumerate(terms.items()):
        if loss.numel() != 1 or not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError('Invalid '+name+' scalar loss')
        gs = torch.autograd.grad(loss,parameters,retain_graph=index<len(TASKS)-1,allow_unused=True)
        norm = finite_gradient_norm(gs)
        norm_value = float(norm)
        scale = min(1.,BUDGET/(norm_value+1e-12))
        stats[name] = dict(raw_norm=norm_value,scale=scale,budgeted_norm=norm_value*scale)
        for i,(p,g) in enumerate(zip(parameters,gs)):
            if g is None: continue
            g = g.detach()
            if raw is not None:
                if raw[i] is None: raw[i] = g.clone()
                else: raw[i].add_(g)
            # Materialize every scaled contribution with the same rounding.
            # Mixing mul for the first task with fused add(alpha=...) later
            # leaves a residual for equal opposite large FP32 gradients.
            scaled = g.mul(scale)
            if p.grad is None: p.grad = scaled
            else: p.grad.add_(scaled)
        del gs
    raw_norm = float(finite_gradient_norm(raw)) if raw is not None else None
    mixed = float(finite_gradient_norm(p.grad for p in parameters))
    return dict(task_budget_revision=VERSION,task_memory_revision=MEMORY_REVISION,
        raw_gradient_measured=measure_raw,component_gradients=stats,
        raw_gradient_before_budget=raw_norm,gradient_before_clip=mixed,
        regression_gradient_norm=stats['main_reg']['raw_norm'],
        regression_gradient_scale=stats['main_reg']['scale'],
        regression_gradient_budgeted_norm=stats['main_reg']['budgeted_norm'],regression_budget=BUDGET)


def release_loss_graphs(runner, hook):
    """Drop only completed loss references; keep batch/buffers for quarantine.

    Caller must also delete its local terms/auxiliary/total references. This
    never modifies parameters, gradients, momentum, RNG or forward scalars.
    """
    runner.outputs['loss'] = runner.outputs['loss'].detach()
    hook.losses = hook._regression_loss = None


def configure(cfg):
    """Separate auxiliary hook only; original prevention configs retain identity."""
    from crane_project.utils.port_reliability_oof_prevention_v1 import check_config
    check_config(cfg)
    cfg.optimizer_config.type = 'PortOOFTaskBudgetHook'
    cfg.optimizer_config.task_budget_revision = VERSION
    cfg.optimizer_config.task_memory_revision = MEMORY_REVISION
    cfg.custom_imports['imports'] = list(cfg.custom_imports['imports']) + ['mmrotate.core.hooks.port_oof_task_budget_hook']
    check_task_config(cfg)
    return cfg


def check_task_config(cfg):
    from crane_project.utils.port_reliability_oof_prevention_v1 import check_config
    if (cfg.optimizer_config.type != 'PortOOFTaskBudgetHook' or
            cfg.optimizer_config.task_budget_revision != VERSION or
            cfg.optimizer_config.get('task_memory_revision') != MEMORY_REVISION or
            'mmrotate.core.hooks.port_oof_task_budget_hook' not in cfg.custom_imports['imports']):
        raise ValueError('Explicit auxiliary per-task budget revision required')
    base = deepcopy(cfg)
    base.optimizer_config.type = 'PortOOFPreventionHook'
    base.optimizer_config.pop('task_budget_revision')
    base.optimizer_config.pop('task_memory_revision')
    check_config(base)
