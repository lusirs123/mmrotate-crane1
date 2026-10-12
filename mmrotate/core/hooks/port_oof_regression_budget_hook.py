"""Opt-in auxiliary optimizer hook; never activated in a formal model config."""
import types
import torch
from mmcv.runner import HOOKS
from mmrotate.core.hooks.port_strict_optimizer_hook import PortStrictOptimizerHook
from crane_project.utils.port_reliability_oof_gradient_chain_v1 import regression_budget_backward


@HOOKS.register_module()
class PortOOFRegressionBudgetHook(PortStrictOptimizerHook):
    def before_run(self,runner):
        if self.grad_clip != dict(max_norm=10,norm_type=2):
            raise ValueError('Auxiliary budget requires unchanged global clip10/norm2')
        if getattr(runner.model,'device_ids',[]) and len(runner.model.device_ids)!=1:
            raise ValueError('Auxiliary gradient budget is single-GPU only')
        self._regression_loss = None
        model=runner.model.module if hasattr(runner.model,'module') else runner.model
        original=model.forward_train
        def forward(_model,*args,**kwargs):
            result=original(*args,**kwargs)
            value=result['loss_bbox']
            self._regression_loss=sum(v.mean() for v in value) if isinstance(value,(list,tuple)) else value.mean()
            return result
        model.forward_train=types.MethodType(forward,model)

    def before_train_iter(self,runner):
        self._regression_loss=None

    def after_train_iter(self,runner):
        self._runner=runner
        try:
            if self._regression_loss is None: raise ValueError('Missing actual main regression loss')
            runner.optimizer.zero_grad(set_to_none=True)
            stats=regression_budget_backward(runner.outputs['loss'],self._regression_loss,
                [p for p in runner.model.parameters() if p.requires_grad])
            self.clip_grads(runner.model.parameters())  # inherited finite checks, then global10
            if self.memory_guard is not None:
                torch.cuda.synchronize()
                memory=self.memory_guard.observe(torch.cuda.memory_allocated(),torch.cuda.memory_reserved())
                stats.update(cuda_allocated_mib=memory['allocated_bytes']/2**20,
                             cuda_reserved_mib=memory['reserved_bytes']/2**20)
            runner.log_buffer.update(stats,runner.outputs['num_samples'])
            runner.optimizer.step()
        finally:
            self._regression_loss=None;self._runner=None
