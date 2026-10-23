"""Run tools/train.py unchanged; stop after ten validated DDP updates.

The hook is installed only in this process, after the optimizer hook. No
preflight checkpoint is used for formal training. Outputs stay in --work-dir.
"""
import json
import math
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from mmcv.runner import BaseRunner, Hook


class PreflightComplete(Exception):
    pass


class VerifyUpdates(Hook):
    def before_run(self, runner):
        assert dist.get_world_size() == 2
        self.rows = []

    def before_train_iter(self, runner):
        self.before = {n: p.detach().clone() for n, p in
                       runner.model.named_parameters() if p.requires_grad}

    def after_train_iter(self, runner):
        loss = float(runner.outputs['loss'])
        lr = runner.optimizer.param_groups[0]['lr']
        expected = .0025 * (.001 + .999 * runner.iter / 1000)
        assert abs(lr - expected) < 1e-10, (lr, expected)
        assert runner.outputs['num_samples'] == 2
        assert math.isfinite(loss)
        grads, changed = 0, 0
        for n, p in runner.model.named_parameters():
            assert bool(torch.isfinite(p).all()), n
            if p.grad is not None:
                assert bool(torch.isfinite(p.grad).all()), n
                grads += int(bool(torch.count_nonzero(p.grad)))
            if n in self.before:
                changed += int(not torch.equal(p.detach(), self.before[n]))
        assert grads > 0 and changed > 0, (grads, changed)
        self.rows.append(dict(iter=runner.iter, loss=loss, lr=lr,
                              nonzero_grad_tensors=grads, updated=changed))
        if runner.iter == 9:
            rank = dist.get_rank()
            payload = dict(passed=True, rank=rank, world_size=2,
                           samples_per_gpu=2, iterations=self.rows,
                           scope='formal train.py entry, init_weights, DDP, optimizer hooks; TRAIN only')
            path = Path(runner.work_dir) / ('formal_preflight_rank%d.json' % rank)
            path.write_text(json.dumps(payload, indent=2))
            dist.barrier()
            raise PreflightComplete()


original = BaseRunner.register_training_hooks


def register(self, *args, **kwargs):
    original(self, *args, **kwargs)
    self.register_hook(VerifyUpdates(), priority='LOW')


BaseRunner.register_training_hooks = register
try:
    runpy.run_path(str(ROOT / 'tools/train.py'), run_name='__main__')
except PreflightComplete:
    if dist.is_initialized():
        dist.destroy_process_group()
    print('FORMAL_ENTRY_PREFLIGHT_PASS', flush=True)
finally:
    BaseRunner.register_training_hooks = original
