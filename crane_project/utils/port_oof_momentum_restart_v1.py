"""Opt-in auxiliary SGD restart on opposing carry-over; no loss rescaling."""
from copy import deepcopy
import math

VERSION = 'opposing_sgd_momentum_restart_v1'


def momentum_restart_decision(optimizer):
    """Read-only dot(current clipped gradient + decay, beta * old momentum).

    All reviewed parameter groups share a learning rate. Thus a negative dot
    means the stored carry-over opposes the current update direction. This is
    not a proof of finite-step descent or classification generalization.
    """
    import torch
    if not isinstance(optimizer, torch.optim.SGD):
        raise ValueError('Reviewed SGD required')
    if len({float(g['lr']) for g in optimizer.param_groups}) != 1:
        raise ValueError('Uniform reviewed learning rate required')
    dots, count = [], 0
    for group in optimizer.param_groups:
        if group.get('nesterov', False) or group.get('dampening', 0.) != 0.:
            raise ValueError('Ordinary undampened SGD required')
        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise ValueError('Dense auxiliary gradients required')
            old = optimizer.state.get(p, {}).get('momentum_buffer')
            if old is None or not group['momentum']:
                continue
            # Dot reductions allocate scalars, not another full-model vector.
            dot = torch.dot(p.grad.detach().reshape(-1), old.detach().reshape(-1))
            if group['weight_decay']:
                dot = dot + group['weight_decay']*torch.dot(p.detach().reshape(-1), old.detach().reshape(-1))
            dots.append(dot*group['momentum'])
            count += 1
    alignment = float(torch.stack(dots).sum()) if dots else 0.
    if not math.isfinite(alignment):
        raise FloatingPointError('Nonfinite SGD momentum alignment')
    return dict(momentum_restart_revision=VERSION, gradient_momentum_dot=alignment,
                active_momentum_tensors=count, restart_required=alignment < 0.)


def restart_opposing_momentum(optimizer):
    """Called immediately before SGD.step, after all production stop guards."""
    decision = momentum_restart_decision(optimizer)
    if decision['restart_required']:
        for group in optimizer.param_groups:
            for p in group['params']:
                old = optimizer.state.get(p, {}).get('momentum_buffer')
                if p.grad is not None and old is not None:
                    old.zero_()
    return dict(decision,momentum_restarted=decision['restart_required'])


def configure(cfg):
    from crane_project.utils.port_oof_task_budget_v1 import check_task_config
    check_task_config(cfg)
    cfg.optimizer.type = 'PortOOFMomentumRestartSGD'
    cfg.optimizer.momentum_restart_revision = VERSION
    cfg.optimizer_config.type = 'PortOOFMomentumRestartHook'
    cfg.optimizer_config.momentum_restart_revision = VERSION
    cfg.custom_imports['imports'] = list(cfg.custom_imports['imports']) + [
        'mmrotate.core.hooks.port_oof_momentum_restart_hook']
    check_config(cfg)


def check_config(cfg):
    from crane_project.utils.port_oof_task_budget_v1 import check_task_config
    if (cfg.optimizer.type != 'PortOOFMomentumRestartSGD' or
            cfg.optimizer.get('momentum_restart_revision') != VERSION or
            cfg.optimizer_config.type != 'PortOOFMomentumRestartHook' or
            cfg.optimizer_config.get('momentum_restart_revision') != VERSION or
            'mmrotate.core.hooks.port_oof_momentum_restart_hook' not in cfg.custom_imports['imports']):
        raise ValueError('Explicit auxiliary momentum-restart revision required')
    base = deepcopy(cfg)
    base.optimizer.type = 'SGD'
    base.optimizer.pop('momentum_restart_revision')
    base.optimizer_config.type = 'PortOOFTaskBudgetHook'
    base.optimizer_config.pop('momentum_restart_revision')
    check_task_config(base)
    if (base.optimizer.lr != .0025 or base.optimizer.momentum != .9 or
            base.optimizer.weight_decay != .0001 or base.optimizer.get('nesterov',False) or
            base.optimizer.get('dampening',0.) != 0.):
        raise ValueError('Original SGD hyperparameters required')
