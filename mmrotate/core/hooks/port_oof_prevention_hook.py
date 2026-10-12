"""TRAIN-only budgeted update, stop before an anomalous optimizer step."""
import hashlib
import inspect
import json
from pathlib import Path
import types
import torch
from mmcv.runner import HOOKS, OptimizerHook
from mmrotate.core.hooks.port_oof_regression_budget_hook import PortOOFRegressionBudgetHook
from mmrotate.core.hooks.port_oof_trace_hook import cpu_tree, rng_state, runtime_state, sum_loss
from crane_project.utils.port_reliability_oof_followup_v1 import json_safe
from crane_project.utils.port_reliability_oof_prevention_v1 import SETTINGS, VERSION, ALLOCATION_REVISION, reasons, measured_budget_backward, finite_gradient_norm


class PreventionStopped(RuntimeError):
    pass


@HOOKS.register_module()
class PortOOFPreventionHook(PortOOFRegressionBudgetHook):
    def __init__(self, directory, settings, **kwargs):
        super().__init__(**kwargs)
        if settings != SETTINGS or self.memory_guard is None:
            raise ValueError('Fixed limits and memory plateau required')
        self.directory = Path(directory)
        self.completed = 0
        self.finished = False
        self.peak_raw = self.peak_mixed = 0.
        self.raw_exceedances = 0
        self.budgeted_steps = 0

    def clip_grads(self, params):
        # Identical clip10 operation; scalar finite checks avoid full-size
        # boolean masks introduced by the generic strict hook.
        params = [p for p in params if p.requires_grad and p.grad is not None]
        finite_gradient_norm(p.grad for p in params)
        before = OptimizerHook.clip_grads(self, params)
        after = finite_gradient_norm(p.grad for p in params)
        if not bool(torch.isfinite(before)):
            raise FloatingPointError('Nonfinite gradient norm before optimizer')
        if self._runner is not None:
            self._runner.log_buffer.update(dict(gradient_before_clip=float(before),
                gradient_after_clip=float(after),
                gradient_clip_fraction=min(1.,10./(float(before)+1e-6)),
                nonfinite_gradient_events=self.nonfinite_gradient_events),self._runner.outputs['num_samples'])
        return before

    def before_run(self, runner):
        self.model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        signature = inspect.signature(self.model.forward_train)
        super().before_run(runner)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.log = (self.directory/'updates.jsonl').open('x')
        self.runner = runner
        original = self.model.forward_train
        def forward(_model, *args, **kwargs):
            bound = signature.bind(*args, **kwargs); bound.apply_defaults()
            self.batch = dict(bound.arguments)
            for key, parameter in signature.parameters.items():
                if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                    self.batch.update(self.batch.pop(key, {}))
            try:
                self.losses = original(*args, **kwargs)
                return self.losses
            except Exception as exc:
                self.quarantine('FORWARD_FAILURE_STOP', dict(error=repr(exc)), runner)
                raise
        self.model.forward_train = types.MethodType(forward, self.model)

    def before_train_iter(self, runner):
        if self.completed >= SETTINGS['max_updates']:
            self.finish('PREVENTION_WINDOW_PASS', runner)
            raise PreventionStopped('7000 completed updates; no automatic continuation')
        super().before_train_iter(runner)
        self.losses = self.batch = None
        # Parameters and momentum stay unchanged until step(); save only the
        # small mutable buffers/RNG here, not a whole CPU model every iteration.
        self.pre_buffers = {n:b.detach().cpu().clone() for n,b in self.model.named_buffers()}
        self.pre_rng = rng_state()
        self.pre_runtime = runtime_state(self.model)

    def snapshot(self, runner, before):
        state = cpu_tree(self.model.state_dict())
        if before:
            state.update(self.pre_buffers)
        return dict(protocol=VERSION, diagnostic_only=True, iteration=runner.iter,
                    model=state, optimizer=cpu_tree(runner.optimizer.state_dict()),
                    rng=self.pre_rng if before else rng_state(),
                    runtime=self.pre_runtime if before else runtime_state(self.model),
                    batch=cpu_tree(self.batch))

    def quarantine(self, status, record, runner):
        if self.finished:
            return
        torch.save(self.snapshot(runner, True), str(self.directory/'before_anomaly.pt'))
        torch.save(dict(self.snapshot(runner, False), trace=record), str(self.directory/'after_forward_before_update.pt'))
        self.record(dict(record, optimizer_updated=False))
        self.finish(status, runner)

    def record(self, value):
        self.log.write(json.dumps(json_safe(value), allow_nan=False)+'\n')
        self.log.flush()

    def finish(self, status, runner):
        if self.finished:
            return
        self.finished = True
        pins = {}
        for path in self.directory.glob('*.pt'):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024), b''):
                    digest.update(block)
            pins[path.name] = digest.hexdigest()
        report = dict(protocol=VERSION, allocation_revision=ALLOCATION_REVISION,
                      status=status, completed_updates=self.completed,
                      max_updates=SETTINGS['max_updates'], peak_raw_gradient=self.peak_raw,
                      peak_mixed_gradient=self.peak_mixed, raw_gradient_exceedances=self.raw_exceedances,
                      budgeted_steps=self.budgeted_steps, snapshot_sha256=pins,
                      VAL_scored=False, TEST_read=False, adopted=False,
                      diagnostic_weights_usable_as_OOF=False, full_training_completed=False,
                      gradient_stop_definition='budgeted_mixed_before_global_clip',
                      memory_guard=self.memory_guard.settings)
        (self.directory/'completion.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        self.log.close()

    def after_train_iter(self, runner):
        self._runner = runner
        record = dict(step=runner.iter+1, epoch=runner.epoch+1,
                      lr=[g['lr'] for g in runner.optimizer.param_groups],
                      images=[Path(m['filename']).stem for m in self.batch['img_metas']],
                      optimizer_updated=False)
        try:
            values = dict(main_cls=float(sum_loss(self.losses['loss_cls']).detach()),
                          main_reg=float(self._regression_loss.detach()),
                          total=float(runner.outputs['loss'].detach()))
            record['losses'] = values
            failure = reasons(values, 0.)
            if failure:
                record['anomaly'] = failure
                self.quarantine('ANOMALY_QUARANTINED_BEFORE_OPTIMIZER', record, runner)
                raise PreventionStopped('Loss limit: '+','.join(failure))
            runner.optimizer.zero_grad(set_to_none=True)
            params = [p for p in runner.model.parameters() if p.requires_grad]
            stats = measured_budget_backward(runner.outputs['loss'], self._regression_loss, params)
            record.update(stats)
            self.peak_raw = max(self.peak_raw, stats['raw_gradient_before_budget'])
            self.peak_mixed = max(self.peak_mixed, stats['gradient_before_clip'])
            self.raw_exceedances += int(stats['raw_gradient_before_budget'] >= SETTINGS['gradient_limit'])
            self.budgeted_steps += int(stats['regression_gradient_scale'] < 1.)
            failure = reasons(values, stats['gradient_before_clip'])
            if failure:
                record['anomaly'] = failure
                self.quarantine('ANOMALY_QUARANTINED_BEFORE_OPTIMIZER', record, runner)
                raise PreventionStopped('Mixed-gradient limit: '+','.join(failure))
            self.clip_grads(params)
            record['gradient_after_clip'] = float(torch.stack([p.grad.detach().norm() for p in params if p.grad is not None]).norm())
            torch.cuda.synchronize()
            record['memory'] = self.memory_guard.observe(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
            runner.log_buffer.update(dict(stats, gradient_after_clip=record['gradient_after_clip'],
                cuda_allocated_mib=record['memory']['allocated_bytes']/2**20,
                cuda_reserved_mib=record['memory']['reserved_bytes']/2**20), runner.outputs['num_samples'])
            runner.optimizer.step()
            self.completed += 1
            record.update(optimizer_updated=True, completed_updates=self.completed)
            self.record(record)
        except PreventionStopped:
            raise
        except Exception as exc:
            record.update(error=repr(exc), anomaly=['update_or_resource_failure'])
            self.quarantine('UPDATE_OR_RESOURCE_FAILURE_STOP', record, runner)
            raise
        finally:
            self._runner = self._regression_loss = self.losses = self.batch = None
            self.pre_buffers = self.pre_rng = self.pre_runtime = None
