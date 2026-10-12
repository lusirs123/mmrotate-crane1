"""Fixed auxiliary-A prevention window; not an OOF source or reliability result."""
from copy import deepcopy
import math
from crane_project.utils.port_reliability_oof_followup_v1 import TRACE, anomaly
from crane_project.utils.port_reliability_oof_training_v2 import validate_config

VERSION = 'port_reliability_oof_prevention_v1'
SETTINGS = dict(max_updates=7000, gradient_limit=1000., main_cls_limit=5.,
                main_reg_limit=10., regression_budget=10., seed=1701)
HOOK = 'mmrotate.core.hooks.port_oof_prevention_hook'
ALLOCATION_REVISION = 'scalar_norm_finite_checks_v1'


def finite_gradient_norm(gradients):
    """Reject NaN/Inf (including norm overflow) without gradient-sized masks."""
    import torch
    norms = [g.detach().norm() for g in gradients if g is not None]
    if not norms:
        raise FloatingPointError('Missing gradients')
    norm = torch.stack(norms).norm()
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Nonfinite gradient norm')
    return norm


def reasons(losses, mixed_norm):
    """Only the gradient actually delivered to global clip is a stop statistic."""
    result = anomaly(losses, mixed_norm, dict(TRACE, **SETTINGS))
    if any(not math.isfinite(float(v)) for v in losses.values()) and 'nonfinite' not in result:
        result.append('nonfinite')
    return result


def configure(cfg, directory):
    validate_config(cfg)
    cfg.checkpoint_config = None
    cfg.optimizer_config.type = 'PortOOFPreventionHook'
    cfg.optimizer_config.directory = str(directory)
    cfg.optimizer_config.settings = deepcopy(SETTINGS)
    cfg.custom_imports['imports'] = list(cfg.custom_imports['imports']) + [HOOK]
    cfg.custom_hooks = []
    check_config(cfg)
    return cfg


def check_config(cfg):
    if cfg.optimizer_config.type != 'PortOOFPreventionHook' or cfg.optimizer_config.settings != SETTINGS:
        raise ValueError('Unregistered prevention optimizer/limits')
    if HOOK not in cfg.custom_imports['imports'] or cfg.custom_hooks or cfg.checkpoint_config is not None:
        raise ValueError('No extra trace/checkpoint/selection hooks allowed')
    if cfg.data.val or cfg.data.test or cfg.evaluation is not None or cfg.workflow != [('train', 1)]:
        raise ValueError('TRAIN-only workflow required')
    if cfg.seed != SETTINGS['seed'] or cfg.runner.max_epochs != 24:
        raise ValueError('Preserve original seed and 24-epoch LR schedule')
    if cfg.optimizer_config.grad_clip != dict(max_norm=10, norm_type=2):
        raise ValueError('Original global clip10 required')
    copied = deepcopy(cfg)
    copied.optimizer_config.type = 'PortStrictOptimizerHook'
    copied.optimizer_config.pop('directory')
    copied.optimizer_config.pop('settings')
    validate_config(copied)


def measured_budget_backward(total, regression, parameters):
    """Same fixed repair, additionally measure raw and delivered mixed gradients.

    No extra total-loss backward is needed. Raw = other + regression; delivered
    = other + alpha*regression. Explicit addition preserves FP32 budget precision.
    """
    import torch
    if any(p.grad is not None for p in parameters):
        raise ValueError('Clear gradients before prevention backward')
    gradients = torch.autograd.grad(regression, parameters, retain_graph=True, allow_unused=True)
    norm = finite_gradient_norm(gradients)
    scale = min(1., SETTINGS['regression_budget'] / (float(norm) + 1e-12))
    (total - regression).backward()
    raw_norms = []
    for p, g in zip(parameters, gradients):
        other = p.grad
        if other is not None and not bool(torch.isfinite(other.detach().norm())):
            raise FloatingPointError('Nonfinite other gradients')
        raw = g.detach() if other is None and g is not None else other
        if other is not None and g is not None:
            raw = other.detach() + g.detach()
        if raw is not None:
            raw_norms.append(raw.norm())
        if g is not None:
            if other is None:
                p.grad = g.detach().mul(scale)
            else:
                p.grad.add_(g.detach(), alpha=scale)
        del raw
    mixed = finite_gradient_norm(p.grad for p in parameters)
    raw_norm = torch.stack(raw_norms).norm()
    if not torch.isfinite(mixed) or not torch.isfinite(raw_norm):
        raise FloatingPointError('Nonfinite gradient norm')
    return dict(raw_gradient_before_budget=float(raw_norm),
                gradient_before_clip=float(mixed), regression_gradient_norm=float(norm),
                regression_gradient_scale=scale, regression_gradient_budgeted_norm=float(norm)*scale,
                regression_budget=SETTINGS['regression_budget'])
