"""Deferred native-GPU gate: actual train.py, four disposable updates.

CPU tensor offload, checkpoint recomputation and precision changes are absent.
This file is not executed by CPU preparation. Explicit GPU authorization and
the engineering-only CLI arguments are required before constructing anything.
"""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    if '--execute-native-check' not in sys.argv:
        print('NOT_EXECUTED: native GPU acceptance pending; explicit --execute-native-check required')
        return
    sys.argv.remove('--execute-native-check')
    assert '--no-validate' in sys.argv and '--deterministic' in sys.argv
    assert 'runner.max_epochs=2' in sys.argv and '--work-dir' in sys.argv
    dest=Path(sys.argv[sys.argv.index('--work-dir')+1]);assert not dest.exists(),'Never overwrite an engineering run'
    import gc
    import itertools
    import json
    import runpy
    import torch
    import torch.distributed as dist
    from mmcv.runner import EpochBasedRunner,Hook,BaseRunner,OptimizerHook
    from crane_project.utils.symeood_repro_trace_v1 import tensor_digest,input_record,rng_digest,runtime
    from audit_symeood_cross_entry_short_v1 import Tape

    class LimitedLoader:
        def __init__(self,source):self.source=source
        def __getattr__(self,k):return getattr(self.source,k)
        def __len__(self):return 2
        def __iter__(self):return itertools.islice(iter(self.source),2)

    original_train=EpochBasedRunner.train
    def limited(self,loader,**kwargs):return original_train(self,LimitedLoader(loader),**kwargs)
    EpochBasedRunner.train=limited

    def write(path,row):
        with Path(path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

    class Observer(Hook):
        def before_run(self,runner):
            self.model=runner.model.module;self.rank=dist.get_rank();self.tape=Tape(self.model)
            write(Path(runner.work_dir)/('native_initial_rank%d.jsonl'%self.rank),dict(
                state_sha256=tensor_digest(self.model.state_dict().items()),rng_sha256=rng_digest(),
                parameters=tensor_digest(self.model.named_parameters()),buffers=tensor_digest(self.model.named_buffers()),
                parameter_order=[n for n,_ in self.model.named_parameters()]))
            opt=next(h for h in runner._hooks if isinstance(h,OptimizerHook));original=opt.clip_grads
            def capture(params):
                ps=list(params);self.raw_gradient=tensor_digest((str(i),p.grad) for i,p in enumerate(ps) if p.grad is not None)
                self.none=[p.grad is None for p in ps]
                self.nonzero=sum(p.grad is not None and bool(p.grad.count_nonzero()) for p in ps)
                assert self.nonzero>0 and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in ps)
                return original(ps)
            opt.clip_grads=capture

        def before_train_iter(self,runner):
            self.tape.clear();self.input=input_record(runner.data_batch);self.before_rng=rng_digest()

        def after_train_iter(self,runner):
            t=self.tape
            optimizer=runner.optimizer.state_dict();states=[]
            for key,values in sorted(optimizer['state'].items()):
                for name,value in sorted(values.items()):
                    if torch.is_tensor(value):states.append((str(key)+'.'+name,value))
            row=dict(iteration=runner.iter,epoch=runner.epoch+1,input=self.input['combined_sha256'],rng_before=self.before_rng,rng_after=rng_digest(),
                predictions=tensor_digest((str(i),v) for i,v in enumerate(t.predictions)),
                labels=tensor_digest((str(i),v) for i,v in enumerate(t.labels)),
                assignment=tensor_digest((str(i),v) for i,v in enumerate(t.assign_int)),
                loss_parts={k:[float(x) for x in values] for k,values in t.losses.items()},
                logits_gradients=tensor_digest((str(i),v.grad) for i,v in enumerate(t.scores)),
                parameter_gradients_before_clip=self.raw_gradient,gradient_none=self.none,
                grad_norm=float(runner.log_buffer.val_history['grad_norm'][-1]),
                optimizer_tensor_state=tensor_digest(states),optimizer_groups=optimizer['param_groups'],
                state_sha256=tensor_digest(self.model.state_dict().items()),parameters_sha256=tensor_digest(self.model.named_parameters()),buffers_sha256=tensor_digest(self.model.named_buffers()),
                counters=[self.model.bbox_head.assigner._local_call_count,int(self.model.bbox_head.loss_cls._local_iter)],
                nonzero_gradient_tensors=self.nonzero,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            write(Path(runner.work_dir)/('native_steps_rank%d.jsonl'%self.rank),row)
            t.clear()

        def after_run(self,runner):
            self.tape.close();write(Path(runner.work_dir)/('native_runtime_rank%d.jsonl'%self.rank),runtime())
            gc.collect();torch.cuda.empty_cache()

    original_register=BaseRunner.register_training_hooks
    def register(self,*args,**kwargs):
        original_register(self,*args,**kwargs);self.register_hook(Observer(),priority='LOW')
    BaseRunner.register_training_hooks=register
    # Exact production entry; only loader duration and observer are engineering
    # instrumentation. No save_on_cpu or alternative optimizer is installed.
    runpy.run_path(str(ROOT/'tools/train.py'),run_name='__main__')


if __name__=='__main__':main()
