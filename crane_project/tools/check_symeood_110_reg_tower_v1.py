"""Disposable native-entry R4 forward/backward gate, optimizer step forbidden."""
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


class GateComplete(Exception):
    pass


def main():
    import torch
    import torch.distributed as dist
    from mmcv.runner import BaseRunner,Hook
    from crane_project.utils.symeood_repro_trace_v1 import input_record,tensor_digest
    work=Path(sys.argv[sys.argv.index('--work-dir')+1]);base=work.parent
    c=json.loads((base/'protocol.json').read_text())
    # The parent controller checks nonexistence once before launching ranks;
    # rank1 may legitimately see files already created by rank0 here.

    class Gate(Hook):
        def before_run(self,runner):
            self.updates=0
            def forbidden(*args,**kwargs):
                self.updates+=1
                raise AssertionError('Optimizer updates forbidden in disposable gate')
            runner.optimizer.step=forbidden
            self.model=runner.model.module
            assert tensor_digest(self.model.state_dict().items())==c['initial_state_sha256']
        def before_train_iter(self,runner):
            rank=dist.get_rank();assert runner.iter==0
            reference=ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/110'
            with (reference/('repro_inputs_rank%d.jsonl'%rank)).open() as f:old=json.loads(next(f))
            current=input_record(runner.data_batch)
            assert current['combined_sha256']==old['combined_sha256'],'TRAIN input stream differs'
            before=tensor_digest(self.model.named_parameters())
            result=runner.model.train_step(runner.data_batch,runner.optimizer)
            assert torch.isfinite(result['loss'])
            result['loss'].backward()
            norms={}
            for branch in ('reg_convs',):
                pp=list(getattr(self.model.bbox_head,branch).parameters())
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pp)
                norms[branch]=float(sum(p.grad.square().sum() for p in pp).sqrt())
                assert norms[branch]>0
            assert len(self.model.bbox_head.cls_convs)==0
            cls_grad=self.model.bbox_head.retina_cls.weight.grad
            assert cls_grad is not None and torch.isfinite(cls_grad).all()
            assert self.updates==0 and before==tensor_digest(self.model.named_parameters())
            (work/('gate_rank%d.json'%rank)).write_text(json.dumps(dict(passed=True,
                optimizer_updates=0,backward_calls=1,rank=rank,tower_gradients=norms,
                common_initial_and_first_input_exact=True,finite_loss=float(result['loss'].detach()),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20),indent=2))
            dist.barrier();raise GateComplete()

    original=BaseRunner.register_training_hooks
    def register(self,*args,**kwargs):
        original(self,*args,**kwargs);self.register_hook(Gate(),priority='LOWEST')
    BaseRunner.register_training_hooks=register
    try:runpy.run_path(str(ROOT/'tools/train.py'),run_name='__main__')
    except GateComplete:
        dist.destroy_process_group();print('R4_GATE_PASS_DISCARDED',flush=True)
    else:raise AssertionError('Gate did not stop before optimizer update')


if __name__=='__main__':main()
