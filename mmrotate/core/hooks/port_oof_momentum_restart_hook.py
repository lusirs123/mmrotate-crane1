"""Auxiliary-only momentum restart; original task/loss/resource guards remain."""
import json
from pathlib import Path
import torch
from mmcv.runner import HOOKS
from mmcv.runner.optimizer.builder import OPTIMIZERS
from mmrotate.core.hooks.port_oof_task_budget_hook import PortOOFTaskBudgetHook
from crane_project.utils.port_oof_momentum_restart_v1 import VERSION, restart_opposing_momentum


@OPTIMIZERS.register_module()
class PortOOFMomentumRestartSGD(torch.optim.SGD):
    def __init__(self, params, momentum_restart_revision, **kwargs):
        if momentum_restart_revision != VERSION:
            raise ValueError('Explicit fixed momentum-restart revision required')
        super().__init__(params, **kwargs)
        self.restart_count = self.actual_steps = 0
        self.last_restart = None

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Reviewed auxiliary SGD has no closure')
        # The parent hook already passed loss, gradient and memory guards.
        self.last_restart = restart_opposing_momentum(self)
        result = super().step()
        self.actual_steps += 1
        self.restart_count += int(self.last_restart['momentum_restarted'])
        return result


@HOOKS.register_module()
class PortOOFMomentumRestartHook(PortOOFTaskBudgetHook):
    def __init__(self, momentum_restart_revision, **kwargs):
        if momentum_restart_revision != VERSION:
            raise ValueError('Explicit fixed momentum-restart revision required')
        super().__init__(**kwargs)

    def record(self, value):
        value = dict(value, momentum_restart_revision=VERSION)
        if value.get('optimizer_updated'):
            optimizer = self.runner.optimizer
            if not isinstance(optimizer, PortOOFMomentumRestartSGD) or optimizer.last_restart is None:
                raise ValueError('Registered restart optimizer required')
            value['momentum_restart'] = dict(optimizer.last_restart)
        super().record(value)

    def finish(self, status, runner):
        already = self.finished
        super().finish(status, runner)
        if not already:
            path = self.directory/'completion.json'
            report = json.loads(path.read_text())
            report.update(momentum_restart_revision=VERSION,
                          momentum_restart_count=runner.optimizer.restart_count)
            path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
