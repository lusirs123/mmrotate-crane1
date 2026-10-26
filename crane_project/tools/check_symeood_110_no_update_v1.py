"""Disposable native-entry check: one real batch/rank, zero optimizer updates.

The formal process never imports this instrumentation. Initial state and first
input must match retained old110 records; a no-grad loss forward loads GPU libs.
"""
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'crane_project/tools'))


class CheckCompleted(Exception):
    pass


def main():
    assert '--work-dir' in sys.argv and '--seed' in sys.argv
    assert sys.argv[sys.argv.index('--seed') + 1] == '0'
    assert '--deterministic' not in sys.argv and '--resume-from' not in sys.argv
    work = Path(sys.argv[sys.argv.index('--work-dir') + 1])
    assert not work.exists(), 'Never reuse an engineering directory'
    import torch
    import torch.distributed as dist
    from mmcv.runner import BaseRunner, Hook
    from crane_project.utils.symeood_repro_trace_v1 import (
        append, assert_runtime_lock, input_record, rng_digest, tensor_digest)
    from audit_symeood_c_pair_completion_v1 import rng_identity

    class Audit(Hook):
        def before_run(self, runner):
            self.rank = dist.get_rank()
            assert dist.get_world_size() == 2
            self.steps = 0

            def forbidden(*args, **kwargs):
                self.steps += 1
                raise AssertionError('Optimizer update forbidden in this gate')

            runner.optimizer.step = forbidden
            old = ROOT / 'work_dirs/symeood_c_pair_seed0_20261025_v2/110'
            self.old = old
            a = json.loads((old / ('repro_initial_rank%d.jsonl' % self.rank)).read_text().splitlines()[0])
            b = json.loads((work / ('repro_initial_rank%d.jsonl' % self.rank)).read_text().splitlines()[0])
            assert a['initial_state_sha256'] == b['initial_state_sha256']
            assert rng_identity(a['rng']) == rng_identity(b['rng'])
            self.initial = b['initial_state_sha256']

        def before_train_iter(self, runner):
            assert runner.iter == 0
            before = rng_digest()
            actual = input_record(runner.data_batch)
            assert before == rng_digest(), 'Input logging changed RNG'
            with (self.old / ('repro_inputs_rank%d.jsonl' % self.rank)).open() as f:
                old = json.loads(next(f))
            assert old['iteration'] == 0 and actual['combined_sha256'] == old['combined_sha256']
            model = runner.model.module
            params = tensor_digest(model.named_parameters())
            assert all(p.grad is None for p in model.parameters())
            with torch.no_grad():
                output = runner.model.train_step(runner.data_batch, runner.optimizer)
            torch.cuda.synchronize()
            assert bool(torch.isfinite(output['loss']).all())
            assert self.steps == 0 and params == tensor_digest(model.named_parameters())
            assert all(p.grad is None for p in model.parameters())
            lock = next(h.runtime_lock for h in runner._hooks if hasattr(h, 'runtime_lock'))
            loaded = assert_runtime_lock(lock)
            append(work / ('no_update_rank%d.jsonl' % self.rank), dict(
                passed=True, rank=self.rank, optimizer_updates=0, backward_calls=0,
                initial_state_sha256=self.initial, initial_rng_values_match=True,
                first_input_sha256=actual['combined_sha256'], first_input_matches=True,
                logging_rng_conserved=True, parameters_conserved=True,
                finite_loss=float(output['loss']), runtime=loaded,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                limit='One real batch per rank only; not full input trajectory or performance reproduction'))
            dist.barrier()
            raise CheckCompleted()

    register = BaseRunner.register_training_hooks

    def observed(self, *args, **kwargs):
        register(self, *args, **kwargs)
        self.register_hook(Audit(), priority='LOWEST')

    BaseRunner.register_training_hooks = observed
    try:
        runpy.run_path(str(ROOT / 'tools/train.py'), run_name='__main__')
    except CheckCompleted:
        assert (work / ('no_update_rank%d.jsonl' % dist.get_rank())).is_file()
        dist.destroy_process_group()
        print('NO_UPDATE_GATE_COMPLETED', flush=True)
    else:
        raise AssertionError('Gate did not stop before the first native update')


if __name__ == '__main__':
    main()
