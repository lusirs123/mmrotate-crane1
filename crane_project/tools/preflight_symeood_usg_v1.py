"""Two-rank real train.py entry, two mini epochs, disposable engineering run.

Only this process limits the loader. Formal training does not import this file.
"""
import itertools
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mmcv.runner import EpochBasedRunner, Hook, BaseRunner


class LimitedLoader:
    def __init__(self, original):self.original=original
    def __getattr__(self,k):return getattr(self.original,k)
    def __len__(self):return 2
    def __iter__(self):return itertools.islice(iter(self.original),2)


original_train=EpochBasedRunner.train
def small_train(self, loader, **kwargs):
    return original_train(self, LimitedLoader(loader), **kwargs)
EpochBasedRunner.train=small_train


class Trace(Hook):
    def after_train_iter(self,runner):
        import torch
        import torch.distributed as dist
        from crane_project.utils.symeood_usg_v1 import tensor_digest
        model=runner.model.module
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert any(bool(g.count_nonzero()) for g in grads)
        assert all(bool(torch.isfinite(g).all()) for g in grads)
        record=dict(epoch=runner.epoch+1,iteration=runner.iter,
                    loss=float(runner.outputs['loss']), learning_rate=runner.optimizer.param_groups[0]['lr'],
                    parameters_sha256=tensor_digest(model.named_parameters()),
                    assigner_calls=model.bbox_head.assigner._local_call_count,
                    loss_calls=int(model.bbox_head.loss_cls._local_iter))
        with (Path(runner.work_dir)/('preflight_trace_rank%d.jsonl'%dist.get_rank())).open('a') as f:f.write(json.dumps(record)+'\n')


original_register=BaseRunner.register_training_hooks
def register(self,*args,**kwargs):
    original_register(self,*args,**kwargs)
    self.register_hook(Trace(),priority='LOW')
BaseRunner.register_training_hooks=register
runpy.run_path(str(ROOT/'tools/train.py'),run_name='__main__')
