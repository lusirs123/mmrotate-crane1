"""Independent auxiliary repair: unchanged loss/resource stops, three budgets."""
from pathlib import Path
import torch
from mmcv.runner import HOOKS
from mmrotate.core.hooks.port_oof_prevention_hook import PortOOFPreventionHook, PreventionStopped
from mmrotate.core.hooks.port_oof_trace_hook import sum_loss
from crane_project.utils.port_reliability_oof_prevention_v1 import SETTINGS, reasons
from crane_project.utils.port_oof_task_budget_v1 import VERSION, task_budget_backward


@HOOKS.register_module()
class PortOOFTaskBudgetHook(PortOOFPreventionHook):
    def __init__(self, task_budget_revision, **kwargs):
        if task_budget_revision != VERSION:
            raise ValueError('Explicit fixed per-task budget revision required')
        super().__init__(**kwargs)

    def finish(self,status,runner):
        from crane_project.utils.port_reliability_oof_followup_v1 import json_safe
        import json
        already = self.finished
        super().finish(status,runner)
        if not already:
            p=self.directory/'completion.json'
            report=json.loads(p.read_text())
            report['task_budget_revision']=VERSION
            p.write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')

    def after_train_iter(self,runner):
        self._runner=runner
        record=dict(step=runner.iter+1,epoch=runner.epoch+1,
            lr=[g['lr'] for g in runner.optimizer.param_groups],
            images=[Path(m['filename']).stem for m in self.batch['img_metas']],optimizer_updated=False,
            task_budget_revision=VERSION)
        try:
            terms=dict(main_cls=sum_loss(self.losses['loss_cls']),main_reg=self._regression_loss)
            aux=[sum_loss(v) for k,v in self.losses.items() if 'loss' in k and k not in ('loss_cls','loss_bbox')]
            if not aux: raise ValueError('Reviewed auxiliary branch required')
            terms['auxiliary']=sum(aux)
            total=sum(terms.values())
            if not torch.allclose(total.detach(),runner.outputs['loss'].detach(),atol=1e-6,rtol=1e-6):
                raise ValueError('Task losses do not reconstruct actual total')
            values={k:float(v.detach()) for k,v in terms.items()}
            values['total']=float(runner.outputs['loss'].detach())
            record['losses']=values
            failure=reasons(values,0.)
            if failure:
                record['anomaly']=failure
                self.quarantine('ANOMALY_QUARANTINED_BEFORE_OPTIMIZER',record,runner)
                raise PreventionStopped('Unchanged loss limit: '+','.join(failure))
            runner.optimizer.zero_grad(set_to_none=True)
            params=[p for p in runner.model.parameters() if p.requires_grad]
            stats=task_budget_backward(terms,params)
            record.update(stats)
            self.peak_raw=max(self.peak_raw,stats['raw_gradient_before_budget'])
            self.peak_mixed=max(self.peak_mixed,stats['gradient_before_clip'])
            self.raw_exceedances+=int(stats['raw_gradient_before_budget']>=SETTINGS['gradient_limit'])
            self.budgeted_steps+=int(any(x['scale']<1. for x in stats['component_gradients'].values()))
            failure=reasons(values,stats['gradient_before_clip'])
            if failure:
                record['anomaly']=failure
                self.quarantine('ANOMALY_QUARANTINED_BEFORE_OPTIMIZER',record,runner)
                raise PreventionStopped('Unchanged mixed-gradient limit: '+','.join(failure))
            self.clip_grads(params)
            record['gradient_after_clip']=float(torch.stack([p.grad.detach().norm() for p in params if p.grad is not None]).norm())
            torch.cuda.synchronize()
            a=torch.cuda.memory_stats()
            allocated,reserved=torch.cuda.memory_allocated(),torch.cuda.memory_reserved()
            pa,pr=torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
            record['allocator']=dict(allocated_bytes=allocated,reserved_bytes=reserved,
                peak_allocated_bytes=pa,peak_reserved_bytes=pr,
                inactive_split_bytes=a.get('inactive_split_bytes.all.current'),
                active_bytes=a.get('active_bytes.all.current'),allocation_retries=a.get('num_alloc_retries'),OOM_events=a.get('num_ooms'))
            record['memory']=self.memory_guard.observe(allocated,reserved,peak_allocated=pa,peak_reserved=pr)
            runner.log_buffer.update({k:v for k,v in stats.items() if isinstance(v,(int,float))},runner.outputs['num_samples'])
            runner.log_buffer.update(dict(cuda_allocated_mib=allocated/2**20,cuda_reserved_mib=reserved/2**20),runner.outputs['num_samples'])
            runner.optimizer.step()
            self.completed+=1
            record.update(optimizer_updated=True,completed_updates=self.completed)
            self.record(record)
        except PreventionStopped:
            raise
        except Exception as exc:
            record.update(error=repr(exc),anomaly=['update_or_resource_failure'])
            self.quarantine('UPDATE_OR_RESOURCE_FAILURE_STOP',record,runner)
            raise
        finally:
            self._runner=self._regression_loss=self.losses=self.batch=None
            self.pre_buffers=self.pre_rng=self.pre_runtime=None
