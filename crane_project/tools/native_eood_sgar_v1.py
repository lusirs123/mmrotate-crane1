"""Memory-limited native gate/train/VAL entries for the single EOOD+SGAR run."""
import hashlib
import json
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
    from crane_project.utils.eood_sgar_protocol_v1 import verify, save as save_json
    c = verify(base, launch=True)
    backend = c['distributed_backend']
    assert backend in ('nccl', 'gloo')
    assert stage in ('gate', 'train', 'sweep', 'eval')
    if stage in ('sweep', 'eval'):
        completed = json.loads((base / 'training_verification.json').read_text())
        assert completed['passed'] and completed['epochs'] == 24
    config = ROOT / c['config']
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

        # Keep the historical selector and its full config; preserve an empty
        # selection too (the shared CLI returns before saving in that case).
        selection = c['checkpoint_selection']['selection_config']
        assert dict(sweep.SELECTION_CONFIG, mcml_limit=selection['mcml_limit']) == selection
        original_select = sweep.select_best_checkpoint
        captured = {}

        def observed_select(results, config_values):
            assert config_values == selection
            winner, data, info = original_select(results, config_values)
            captured.update(winner=winner, results=results, info=info)
            return winner, data, info

        sweep.select_best_checkpoint = observed_select
        sweep.subprocess = SimpleNamespace(run=limited_run)
        sys.argv = ['ckpt_sweep.py', '--config', str(config), '--work-dir', str(base / 'train'),
                    '--epochs', *map(str, c['candidate_epochs']),
                    '--gpus', str(c['physical_gpus'][0]),
                    '--center-thresh', str(c['checkpoint_selection']['center_thresh_px']),
                    '--mcml-limit', str(selection['mcml_limit'])]
        sweep.main()
        target = base / 'train/ckpt_sweep/sweep_results.json'
        if not target.exists():
            assert captured and captured['winner'] is None
            results = captured['results']
            assert set(results) == {'epoch_%d' % e for e in c['candidate_epochs']}
            save_json(target, dict(selection_config=selection,
                metric_protocol_version=sweep.METRIC_PROTOCOL_VERSION,
                evidence_role='source_val_checkpoint_selection',
                candidate_epochs=c['candidate_epochs'],
                center_thresh_px=c['checkpoint_selection']['center_thresh_px'],
                config_sha256=c['config_sha256'],
                source_val_annotations_sha256=c['val_annotations_sha256'],
                selection_info=captured['info'], selected_checkpoint=None,
                selected_path=None, all_checkpoints={name: {
                    key: row[key] for key in ('checkpoint', 'checkpoint_sha256',
                                              'results_pkl', 'results_pkl_sha256', 'metrics')}
                    for name, row in results.items()}, TEST=False))
        saved = json.loads(target.read_text())
        assert saved['selection_config'] == selection
        assert set(saved['all_checkpoints']) == {'epoch_%d' % e for e in c['candidate_epochs']}
        return

    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if stage in ('gate', 'train'):
        assert visible == ','.join(map(str, c['visible_cuda_ordinals']))
        device = int(os.environ.get('LOCAL_RANK', 0))
        assert device in (0, 1)
        physical = c['physical_gpus'][device]
    else:
        assert visible in {str(i) for i in c['physical_gpus']}
        physical, device = int(visible), 0
        # Old PyTorch cannot parse GPU UUID masks. Use the verified PCI-order
        # CUDA ordinal, and still check the actual process GPU UUID below.
        ordinal = c['visible_cuda_ordinals'][c['physical_gpus'].index(physical)]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(ordinal)
    import torch
    assert torch.cuda.device_count() == (2 if stage in ('gate', 'train') else 1)
    torch.cuda.set_device(device)
    cap = c['allocator_limit_mib'] if stage in ('gate', 'train') else 3072
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(cap * 2**20 / total, device)
    torch.cuda.reset_peak_memory_stats(device)
    assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
    probe = torch.empty(1, device=device)
    torch.cuda.synchronize(device)
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
        checkpoint = Path(extra[1]).resolve()
        assert checkpoint.parent == (base / 'train').resolve()
        assert checkpoint.name in {'epoch_%d.pth' % e for e in c['candidate_epochs']}
        sys.argv = ['tools/test.py'] + extra
        runpy.run_path(str(ROOT / 'tools/test.py'), run_name='__main__')
        return

    import torch.distributed as dist
    from mmcv.runner import BaseRunner, Hook
    from crane_project.utils.symeood_repro_trace_v1 import tensor_digest, input_record, rng_payload

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
                for parameter in self.model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError('Nonfinite gradient before optimizer step')
                value = self.step(*args, **kwargs)
                self.updates += 1
                return value

            runner.optimizer.step = counted
            assert len(self.model.bbox_head.cls_convs) == len(self.model.bbox_head.reg_convs) == 4
            assert sum(p.numel() for p in self.model.parameters()) == 36281432
            head = self.model.bbox_head.predictors[0]
            assert type(head.pola).__name__ == 'PolaKLDGeometryV1'
            assert head.pola.geometry_mode == 'kld'
            assert head.pola.cost_class == head.pola.cost_riou == 1.
            assert not hasattr(head.pola, 'tau_init')
            assert type(head.loss_iou).__name__ == 'EOODSymKLDPositiveLossV1'
            digest = tensor_digest(self.model.state_dict().items())
            if stage == 'train':
                acceptance = json.loads((base / 'gate_acceptance.json').read_text())
                assert acceptance['passed']
                gate_initial = json.loads((base / 'gate_work' / ('initial_rank%d.json' % self.rank)).read_text())
                assert digest == gate_initial['state_sha256'], 'Formal initialization differs from gate'
            digests = [None] * 2
            dist.all_gather_object(digests, digest)
            assert len(set(digests)) == 1
            save(self.work / ('initial_rank%d.json' % self.rank), dict(state_sha256=digest,
                rank=self.rank, optimizer_updates=0, strict_old_baseline_pair=False,
                same_initial_as_gate=stage == 'train', distributed_backend=backend))
            if stage == 'train' and self.rank == 0:
                path = self.work / 'complete_initial_model.pth'
                assert not path.exists()
                torch.save(dict(state_dict={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                    rng_state=rng_payload(),
                    meta=dict(seed=0, purpose='new EOOD+SGAR before first update; not certified old EOOD initialization')), str(path))
            dist.barrier()

        def before_train_iter(self, runner):
            assert self.model.bbox_head.predictors[0].epoch == runner.epoch
            if stage == 'train':
                if runner.inner_iter == 0:
                    if runner.iter == 0:
                        gate_row = json.loads((base / 'gate_work' / ('gate_rank%d.json' % self.rank)).read_text())
                        current_input = input_record(runner.data_batch)
                        same_input = current_input == gate_row['input']
                        save(self.work / ('first_batch_comparison_rank%d.json' % self.rank),
                             dict(passed=same_input, gate_input=gate_row['input'],
                                  formal_input=current_input, optimizer_updates=self.updates))
                        assert same_input, 'Gate/formal first batch differs'
                    record = dict(epoch=runner.epoch+1, iteration=runner.iter,
                        assignment='MaxIoU' if runner.epoch < 4 else 'POLA-SymKLD',
                        input=input_record(runner.data_batch))
                    with (self.work / ('epoch_inputs_rank%d.jsonl' % self.rank)).open('a') as f:
                        f.write(json.dumps(record) + '\n')
                return
            before = tensor_digest(self.model.named_parameters())
            # Record before train_step: BaseDetector.forward_train mutates
            # img_metas by adding batch_input_shape. Formal traces are also
            # pre-forward, so both captures must use the same timing.
            gate_input = input_record(runner.data_batch)
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
                input=gate_input,
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
            assert self.updates == c['expected_updates_per_rank']
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
    from crane_project.utils.eood_sgar_protocol_v1 import training_arguments
    sys.argv = training_arguments(config, work, backend)
    try:
        runpy.run_path(str(ROOT / 'tools/train.py'), run_name='__main__')
    except GateComplete:
        assert stage == 'gate'
        dist.destroy_process_group()
        print('EOOD_SGAR_NATIVE_GATE_PASS_ZERO_UPDATES', flush=True)
    else:
        assert stage == 'train'


if __name__ == '__main__':
    main()
