"""Memory-limited native gate/train/VAL entries for the single E+B run."""
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


class GateComplete(Exception):
    pass


def main():
    stage, base = sys.argv[1], Path(sys.argv[2]).resolve()
    extra = sys.argv[3:]
    c = json.loads((base / 'launch_protocol_gpus01.json').read_text())
    backend = c.get('distributed_backend', 'nccl')
    assert backend in ('nccl', 'gloo')
    assert stage in ('gate', 'train', 'sweep', 'eval')
    for name, digest in c['sources'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in c['source_tree'].items():
        assert sha(ROOT / name) == digest, name
    config = ROOT / c['config']
    assert sha(config) == c['config_sha256']
    if stage == 'sweep':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        sys.path.insert(0, str(ROOT / 'crane_project/tools'))
        import crane_project.tools.ckpt_sweep as sweep
        original_run = subprocess.run

        def limited_run(command, *args, **kwargs):
            if (isinstance(command, list) and len(command) > 1
                    and Path(command[1]).resolve() == ROOT / 'tools/test.py'):
                command = [command[0], str(Path(__file__).resolve()), 'eval', str(base)] + command[2:]
            return original_run(command, *args, **kwargs)

        sweep.subprocess = SimpleNamespace(run=limited_run)
        sys.argv = ['ckpt_sweep.py', '--config', str(config), '--work-dir', str(base / 'train'),
                    '--epochs', '16', '18', '20', '22', '24', '--gpus', '0', '1',
                    '--center-thresh', '15', '--mcml-limit', '5']
        sweep.main()
        return

    import torch
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert (visible == '0,1' if stage in ('gate', 'train') else visible in ('0', '1'))
    device = int(os.environ.get('LOCAL_RANK', 0)) if stage in ('gate', 'train') else 0
    torch.cuda.set_device(device)
    cap = c['allocator_limit_mib'] if stage in ('gate', 'train') else 3072
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(cap * 2**20 / total, device)
    torch.cuda.reset_peak_memory_stats(device)
    assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
    probe = torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    physical = (0,1)[device] if stage in ('gate','train') else int(visible)
    apps = subprocess.check_output(['nvidia-smi', '-i', str(physical), '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    observed = [line.split(',')[1].strip() for line in apps.splitlines() if line.split(',')[0].strip() == str(os.getpid())]
    assert observed == [c['GPU_UUIDs'][str(physical)]], observed
    del probe
    if stage == 'eval':
        assert len(extra) > 2 and Path(extra[0]).resolve() == config
        val_ann = str((ROOT / 'crane_project/data/crane_grab_port_day2night_v1/val/annfiles').resolve())
        val_img = str((ROOT / 'crane_project/data/crane_grab_port_day2night_v1/val/images').resolve())
        assert 'data.test.ann_file=' + val_ann in extra
        assert 'data.test.img_prefix=' + val_img in extra
        sys.argv = ['tools/test.py'] + extra
        runpy.run_path(str(ROOT / 'tools/test.py'), run_name='__main__')
        return

    import torch.distributed as dist
    from mmcv.runner import BaseRunner, Hook
    from crane_project.utils.symeood_repro_trace_v1 import tensor_digest, input_record

    class Trace(Hook):
        def before_run(self, runner):
            assert dist.get_backend() == backend
            self.model = runner.model.module
            self.rank = dist.get_rank()
            self.work = Path(runner.work_dir)
            self.updates = 0
            self.step = runner.optimizer.step

            def counted(*args, **kwargs):
                if stage == 'gate':
                    raise AssertionError('Optimizer updates forbidden in gate')
                value = self.step(*args, **kwargs)
                self.updates += 1
                return value

            runner.optimizer.step = counted
            assert len(self.model.bbox_head.cls_convs) == len(self.model.bbox_head.reg_convs) == 4
            assert sum(p.numel() for p in self.model.parameters()) == 36281432
            head = self.model.bbox_head.predictors[0]
            assert type(head.pola).__name__ == 'PolaAssigner'
            assert type(head.loss_iou).__name__ == 'EOODSymKLDPositiveLossV1'
            digest = tensor_digest(self.model.state_dict().items())
            assert digest == c['expected_new_reference_initial_state_sha256']
            digests = [None] * 2
            dist.all_gather_object(digests, digest)
            assert len(set(digests)) == 1
            save(self.work / ('initial_rank%d.json' % self.rank), dict(state_sha256=digest,
                rank=self.rank, optimizer_updates=0, strict_old_baseline_pair=False,
                same_initial_as_EA=True, distributed_backend=backend))
            if stage == 'train' and self.rank == 0:
                path = self.work / 'complete_initial_model.pth'
                assert not path.exists()
                torch.save(dict(state_dict={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                    meta=dict(seed=0, purpose='new E+B before first update; not certified old EOOD initialization')), str(path))
            dist.barrier()

        def before_train_iter(self, runner):
            assert self.model.bbox_head.predictors[0].epoch == runner.epoch
            if stage == 'train':
                if runner.inner_iter == 0:
                    record = dict(epoch=runner.epoch+1, iteration=runner.iter,
                        assignment='MaxIoU' if runner.epoch < 4 else 'POLA-IoU',
                        input=input_record(runner.data_batch))
                    with (self.work / ('epoch_inputs_rank%d.jsonl' % self.rank)).open('a') as f:
                        f.write(json.dumps(record) + '\n')
                return
            before = tensor_digest(self.model.named_parameters())
            phases = []
            # Extra allocation represents optimizer/bucket headroom; no optimizer
            # is stepped. Both actual full-model assignment paths are tested.
            reserve = torch.empty(256 * 2**20, dtype=torch.uint8, device=device)
            for epoch in (0, 4):
                self.model.set_epoch(epoch)
                runner.optimizer.zero_grad(set_to_none=True)
                head = self.model.bbox_head.predictors[0]
                original_assign = head.pola.assign
                calls = []

                def observed(*args, **kwargs):
                    value = original_assign(*args, **kwargs)
                    calls.append(int((value.gt_inds > 0).sum()))
                    return value

                head.pola.assign = observed
                try:
                    result = runner.model.train_step(runner.data_batch, runner.optimizer)
                finally:
                    head.pola.assign = original_assign
                assert calls == ([] if epoch == 0 else [1, 1]), calls
                assert torch.isfinite(result['loss'])
                result['loss'].backward()
                gradients = {}
                for name, module in (
                    ('main_cls', self.model.bbox_head.predictors[0].retina_cls),
                    ('main_reg', self.model.bbox_head.predictors[0].retina_reg),
                    ('cls_tower', self.model.bbox_head.cls_convs),
                    ('reg_tower', self.model.bbox_head.reg_convs)):
                    gg = [p.grad for p in module.parameters() if p.grad is not None]
                    assert gg and all(torch.isfinite(g).all() for g in gg)
                    gradients[name] = float(sum(g.detach().square().sum() for g in gg).sqrt())
                    assert gradients[name] > 0
                phases.append(dict(epoch_index=epoch, finite_loss=float(result['loss'].detach()),
                                   gradients=gradients, POLA_positive_counts=calls,
                                   regression_type=type(head.loss_iou).__name__,
                                   regression_term=result['log_vars']['loss_0_loss_iou']))
            assert self.updates == 0 and before == tensor_digest(self.model.named_parameters())
            save(self.work / ('gate_rank%d.json' % self.rank), dict(passed=True, rank=self.rank,
                optimizer_updates=0, phases=phases, reserve_mib=256,
                peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20,
                allocator_limit_mib=cap, discarded=True, distributed_backend=backend))
            del reserve
            dist.barrier()
            raise GateComplete()

        def after_train_iter(self, runner):
            # VERY_HIGH runs before OptimizerHook: reject nonfinite loss before step.
            assert torch.isfinite(runner.outputs['loss']).all()

        def after_train_epoch(self, runner):
            row = dict(epoch=runner.epoch+1, updates=self.updates, rank=self.rank,
                batches=len(runner.data_loader),
                peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20)
            with (self.work / ('epochs_rank%d.jsonl' % self.rank)).open('a') as f:
                f.write(json.dumps(row) + '\n')

        def after_run(self, runner):
            assert self.updates == 15360
            save(self.work / ('complete_rank%d.json' % self.rank), dict(passed=True,
                updates=self.updates, epochs=24, TEST=False, distributed_backend=backend,
                peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20))

    original = BaseRunner.register_training_hooks

    def register(runner, *args, **kwargs):
        original(runner, *args, **kwargs)
        runner.register_hook(Trace(), priority='VERY_HIGH')

    BaseRunner.register_training_hooks = register
    work = base / ('gate_work' if stage == 'gate' else 'train')
    from crane_project.utils.eood_B_regression_protocol_v1 import training_arguments
    sys.argv = training_arguments(config, work, backend)
    try:
        runpy.run_path(str(ROOT / 'tools/train.py'), run_name='__main__')
    except GateComplete:
        assert stage == 'gate'
        dist.destroy_process_group()
        print('E_B_NATIVE_GATE_PASS_ZERO_UPDATES', flush=True)
    else:
        assert stage == 'train'


if __name__ == '__main__':
    main()
