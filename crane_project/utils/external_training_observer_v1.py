"""Resource/progress hook for native MMRotate external training. No method math."""
import json
import math
import os
from pathlib import Path
import time

import torch
from mmcv.runner import HOOKS, Hook


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(str(temporary), str(path))


@HOOKS.register_module()
class ExternalTrainingObserverV1(Hook):
    def before_run(self, runner):
        self.out = Path(os.environ['EXTERNAL_RUN_DIR'])
        self.rank = int(os.environ['LOCAL_RANK'])
        self.updates = 0
        self.started = time.monotonic()
        self.latest = {}
        step = runner.optimizer.step
        def counted(*args, **kwargs):
            value = step(*args, **kwargs)
            self.updates += 1
            return value
        runner.optimizer.step = counted
        self.write(runner, 'INITIALIZED')

    def after_train_iter(self, runner):
        loss = float(runner.outputs['loss'].detach())
        if not math.isfinite(loss):
            raise RuntimeError('Nonfinite loss. Stop without automatic restart.')
        self.latest = dict(loss=loss, loss_terms=runner.outputs['log_vars'])
        if runner.iter < 10 or (runner.iter + 1) % 10 == 0:
            self.write(runner, 'TRAINING')

    def before_val_epoch(self, runner):
        self.write(runner, 'ONLINE_VAL_MONITORING')

    def after_train_epoch(self, runner):
        self.write(runner, 'EPOCH_FINISHED')

    def write(self, runner, state):
        atomic(self.out / ('rank_%d_heartbeat.json' % self.rank), dict(
            state=state, pid=os.getpid(), rank=self.rank, epoch=int(runner.epoch)+1,
            max_epochs=runner.max_epochs, completed_microbatches=int(runner.iter)+1,
            optimizer_updates=self.updates, elapsed_seconds=time.monotonic()-self.started,
            allocated_mib=torch.cuda.memory_allocated(self.rank)/2**20,
            reserved_mib=torch.cuda.memory_reserved(self.rank)/2**20,
            peak_allocated_mib=torch.cuda.max_memory_allocated(self.rank)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(self.rank)/2**20,
            **self.latest))

    def after_run(self, runner):
        self.write(runner, 'COMPLETED')
        atomic(self.out / ('rank_%d_complete.json' % self.rank), dict(
            completed_epochs=int(runner.epoch), optimizer_updates=self.updates,
            rank=self.rank, passed=int(runner.epoch) == runner.max_epochs))
