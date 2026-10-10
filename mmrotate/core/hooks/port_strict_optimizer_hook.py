"""Opt-in finite-gradient guard for auxiliary training, with unchanged clip10."""
import torch
from mmcv.runner import HOOKS, OptimizerHook
from crane_project.utils.port_reliability_oof_training_v2 import MemoryPlateau


@HOOKS.register_module()
class PortStrictOptimizerHook(OptimizerHook):
    def __init__(self, memory_guard=None, **kwargs):
        super().__init__(**kwargs)
        if self.grad_clip is None:
            raise ValueError('Strict auxiliary training requires explicit gradient clipping')
        self.nonfinite_gradient_events = 0
        self._runner = None
        self.memory_guard = MemoryPlateau(**memory_guard) if memory_guard is not None else None

    def after_train_iter(self, runner):
        self._runner = runner
        try:
            super().after_train_iter(runner)
            if self.memory_guard is not None:
                torch.cuda.synchronize()
                memory = self.memory_guard.observe(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
                runner.log_buffer.update(dict(cuda_allocated_mib=memory['allocated_bytes']/2**20,
                                              cuda_reserved_mib=memory['reserved_bytes']/2**20),
                                         runner.outputs['num_samples'])
        finally:
            self._runner = None

    def clip_grads(self, params):
        params = [p for p in params if p.requires_grad and p.grad is not None]
        if not params:
            raise ValueError('Auxiliary training produced no parameter gradients')
        finite = torch.stack([torch.isfinite(p.grad).all() for p in params]).all()
        if not finite:
            self.nonfinite_gradient_events += 1
            raise FloatingPointError('Nonfinite gradients before optimizer step; events=%d' %
                                     self.nonfinite_gradient_events)
        before = super().clip_grads(params)
        if not torch.isfinite(before):
            self.nonfinite_gradient_events += 1
            raise FloatingPointError('Nonfinite gradient norm before optimizer step')
        norm_type = self.grad_clip.get('norm_type', 2.)
        after = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]), norm_type)
        if self._runner is not None:
            fraction = min(1., float(self.grad_clip['max_norm']) / (float(before) + 1e-6))
            self._runner.log_buffer.update(dict(
                gradient_before_clip=float(before), gradient_after_clip=float(after),
                gradient_clip_fraction=fraction,
                nonfinite_gradient_events=self.nonfinite_gradient_events),
                self._runner.outputs['num_samples'])
        return before
