"""Bounded native-entry E+A trial: GPU3, two disposable SGD steps, no VAL/TEST.

Controller checks free memory, launches the native entry in its own process
group, monitors only that child's GPU usage and stops it on resource pressure.
The trial never writes a checkpoint and cannot start a formal training run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ALLOCATOR_MIB = 5632
PROCESS_LIMIT_MIB = 6144
MIN_START_FREE_MIB = 6656
MIN_RUNTIME_FREE_MIB = 1024


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def gpu_info():
    output = subprocess.check_output(['nvidia-smi', '--id=3',
        '--query-gpu=index,uuid,memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    index, uuid, total, used, free = [part.strip() for part in output.strip().split(',')]
    return dict(index=int(index), uuid=uuid, total_mib=int(total),
                used_mib=int(used), free_mib=int(free))


def cuda_ordinal(target_uuid):
    # torch1.13's NVML parser accepts only integer CUDA_VISIBLE_DEVICES.
    # PCI_BUS_ID ordering plus a post-context PID/UUID check avoids ambiguity.
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=pci.bus_id,uuid', '--format=csv,noheader'], text=True)
    devices = [tuple(part.strip() for part in line.split(','))
               for line in output.splitlines()]
    devices.sort(key=lambda row: row[0])
    return next(i for i, (_, uuid) in enumerate(devices) if uuid == target_uuid)


def controller(base, out):
    protocol_path = base / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['control_retrain'] is False
    assert protocol['planned_new_training_runs'] == 1
    config = ROOT / protocol['config']
    assert sha(config) == protocol['config_sha256']
    for relative, digest in protocol['sources'].items():
        assert sha(ROOT / relative) == digest, relative
    cpu = json.loads((base / 'native_CPU_receipt.json').read_text())
    assert cpu['passed'] and cpu['native_tests'] == 6
    assert cpu['protocol_sha256'] == sha(protocol_path)
    initial_gpu = gpu_info()
    out.mkdir(parents=True, exist_ok=False)
    trial = dict(protocol='eood_A_geometry_GPU3_smoke_v1',
        purpose='Native-entry functionality/resource test, not detector performance',
        physical_gpu=3, gpu_uuid=initial_gpu['uuid'], logical_gpu=0,
        cuda_visible_ordinal=cuda_ordinal(initial_gpu['uuid']),
        device_mapping='PCI_BUS_ID ordinal; verify child PID GPU UUID before model build',
        allocator_limit_mib=ALLOCATOR_MIB, process_limit_mib=PROCESS_LIMIT_MIB,
        min_start_free_mib=MIN_START_FREE_MIB, min_runtime_free_mib=MIN_RUNTIME_FREE_MIB,
        parent_protocol_sha256=sha(protocol_path), config_sha256=sha(config),
        script_sha256=sha(__file__), start_gpu=initial_gpu,
        single_gpu=True, per_gpu_batch=2, intended_disposable_updates=2,
        phases=[0, 4], phase_note='Force assignment phase for diagnostics, not actual epoch4 training.',
        formal_training=False, checkpoints_allowed=False, VAL=False, TEST=False,
        auto_retry=False, auto_formal_launch=False)
    save(out / 'trial_protocol.json', trial)
    if initial_gpu['free_mib'] < MIN_START_FREE_MIB:
        save(out / 'controller_result.json', dict(status='NOT_STARTED_LOW_MEMORY',
            passed=False, child_started=False, snapshot=initial_gpu))
        print('NOT_STARTED_LOW_MEMORY', flush=True)
        return 2
    env = dict(os.environ)
    launch = json.loads((ROOT / 'work_dirs/symeood_110_retrain_seed0_20261026_v1/protocol.json').read_text())['launch_environment']
    for key, value in launch.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    env.update(CUDA_VISIBLE_DEVICES=str(trial['cuda_visible_ordinal']),
               CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONPATH=str(ROOT),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # The current native mmrotljj interpreter launches the same interpreter.
    command = [sys.executable, str(Path(__file__).resolve()), '--child',
               '--base', str(base), '--out', str(out)]
    reason = None
    maximum_process = 0
    minimum_free = initial_gpu['free_mib']
    start = time.monotonic()
    with (out / 'native_entry.log').open('x') as log, (out / 'memory_samples.jsonl').open('x') as samples:
        child = subprocess.Popen(command, cwd=str(ROOT), env=env,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        save(out / 'child_identity.json', dict(pid=child.pid, process_group=child.pid,
             GPU_uuid=initial_gpu['uuid'], command=command))
        while child.poll() is None:
            try:
                current = gpu_info()
                apps = subprocess.check_output(['nvidia-smi',
                    '--query-compute-apps=gpu_uuid,pid,used_memory',
                    '--format=csv,noheader,nounits'], text=True)
                owned = []
                for line in apps.splitlines():
                    uuid, pid, memory = [part.strip() for part in line.split(',')]
                    if int(pid) == child.pid:
                        assert uuid == initial_gpu['uuid'], 'Child used an unauthorized GPU'
                        owned.append(int(memory))
                used = sum(owned)
                maximum_process = max(maximum_process, used)
                minimum_free = min(minimum_free, current['free_mib'])
                samples.write(json.dumps(dict(seconds=time.monotonic()-start,
                    child_gpu_mib=used, GPU=current)) + '\n')
                samples.flush()
                if used > PROCESS_LIMIT_MIB:
                    reason = 'CHILD_PROCESS_MEMORY_LIMIT'
                elif current['free_mib'] < MIN_RUNTIME_FREE_MIB:
                    reason = 'LOW_GPU_FREE_MEMORY'
                elif time.monotonic() - start > 300:
                    reason = 'TRIAL_TIMEOUT'
            except Exception as error:
                reason = 'MONITOR_ERROR:' + repr(error)
            if reason:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            time.sleep(.5)
        code = child.wait()
    checkpoints = list(out.rglob('*.pth'))
    result_path = out / 'native_trial_result.json'
    passed = code == 0 and not reason and result_path.exists() and not checkpoints
    if passed:
        native = json.loads(result_path.read_text())
        assert native['passed'] and native['disposable_optimizer_updates'] == 2
        assert native['parent_protocol_sha256'] == sha(protocol_path)
        for relative, digest in protocol['sources'].items():
            assert sha(ROOT / relative) == digest, relative
    save(out / 'controller_result.json', dict(status='PASS' if passed else 'STOPPED_OR_FAILED',
        passed=passed, exit_code=code, stop_reason=reason,
        elapsed_seconds=time.monotonic()-start,
        monitored_process_peak_mib=maximum_process,
        minimum_observed_free_mib=minimum_free, memory_sample_interval_seconds=.5,
        driver_peak_note='Sampled nvidia-smi peak, not an exact continuous maximum.',
        final_gpu=gpu_info(), checkpoints_written=[str(p) for p in checkpoints],
        formal_training=False, auto_retry=False, VAL=False, TEST=False))
    print(json.dumps(json.loads((out / 'controller_result.json').read_text())), flush=True)
    return 0 if passed else 1


def native_child(base, out):
    import runpy
    import torch
    from mmcv.runner import BaseRunner, EpochBasedRunner, Hook
    from mmcv.runner.checkpoint import get_torchvision_models
    from crane_project.utils.symeood_repro_trace_v1 import input_record, tensor_digest
    protocol = json.loads((base / 'protocol.json').read_text())
    trial = json.loads((out / 'trial_protocol.json').read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(trial['cuda_visible_ordinal'])
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_MIB*2**20 / total, 0)
    torch.set_num_threads(1)
    probe = torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    owned_uuids = set()
    for _ in range(10):
        output = subprocess.check_output(['nvidia-smi',
            '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        owned_uuids = {parts[0].strip() for line in output.splitlines()
                       for parts in [line.split(',')]
                       if int(parts[1].strip()) == os.getpid()}
        if owned_uuids:
            break
        time.sleep(.1)
    assert owned_uuids == {trial['gpu_uuid']}, 'Actual CUDA device is not authorized GPU3'
    del probe
    # Fail rather than silently download a new pretrained backbone.
    url = get_torchvision_models()['resnet50']
    pretrained = Path(torch.hub.get_dir()) / 'checkpoints' / url.rsplit('/', 1)[-1]
    assert pretrained.is_file(), 'Expected existing pretrained cache; no download authorized here'
    pretrained_sha = sha(pretrained)

    class TrialComplete(Exception):
        pass

    class TrialHook(Hook):
        def before_run(self, runner):
            self.runner = runner
            self.model = runner.model.module
            self.main = self.model.bbox_head.predictors[0]
            assert self.main.pola.__class__.__name__ == 'PolaKLDGeometryV1'
            assert len(self.model.bbox_head.cls_convs) == len(self.model.bbox_head.reg_convs) == 4
            assert sum(p.numel() for p in self.model.parameters()) == 36281432
            assert runner.optimizer.__class__.__name__ == 'SGD'
            self.initial_parameters = tensor_digest(self.model.named_parameters())
            self.previous_parameters = self.initial_parameters
            self.updates, self.rows = 0, []
            self.pola_calls, self.max_iou_calls = 0, 0
            old_pola, old_maxiou = self.main.pola.assign, self.main.assigner.assign

            def pola(*args, **kwargs):
                self.pola_calls += 1
                return old_pola(*args, **kwargs)

            def maxiou(*args, **kwargs):
                self.max_iou_calls += 1
                return old_maxiou(*args, **kwargs)

            self.main.pola.assign, self.main.assigner.assign = pola, maxiou
            original_step = runner.optimizer.step

            def step(*args, **kwargs):
                assert self.updates < 2, 'Third trial update forbidden'
                groups = dict(cls_tower=self.model.bbox_head.cls_convs,
                    reg_tower=self.model.bbox_head.reg_convs,
                    main_cls=self.main.retina_cls, main_reg=self.main.retina_reg,
                    auxiliary=self.model.bbox_head.predictors[1], FPN=self.model.neck)
                self.gradient_norms = {}
                for name, module in groups.items():
                    gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
                    assert gradients and all(torch.isfinite(g).all() for g in gradients), name
                    norm = float(sum(g.detach().float().square().sum() for g in gradients).sqrt())
                    assert norm > 0, name
                    self.gradient_norms[name] = norm
                value = original_step(*args, **kwargs)
                self.updates += 1
                return value

            runner.optimizer.step = step
            torch.cuda.reset_peak_memory_stats()

        def after_train_iter(self, runner):
            torch.cuda.synchronize()
            assert self.updates == runner.iter + 1
            assert torch.isfinite(runner.outputs['loss']).all()
            digest = tensor_digest(self.model.named_parameters())
            assert digest != self.previous_parameters
            self.previous_parameters = digest
            if runner.iter == 0:
                assert self.pola_calls == 0 and self.max_iou_calls == 2
            else:
                assert self.pola_calls == 2 and self.max_iou_calls == 2
            self.rows.append(dict(iteration=runner.iter,
                forced_assignment_epoch=[0, 4][runner.iter],
                input=self.current_input, loss=float(runner.outputs['loss'].detach()),
                loss_terms=runner.outputs['log_vars'],
                clipped_gradient_l2=self.gradient_norms,
                lr=[group['lr'] for group in runner.optimizer.param_groups],
                parameters_sha256=digest, optimizer_updates=self.updates,
                main_POLA_calls=self.pola_calls, main_MaxIoU_calls=self.max_iou_calls,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
            save(out / ('step_%d.json' % runner.iter), self.rows[-1])
            print('DISPOSABLE_NATIVE_STEP', runner.iter, 'PASS', flush=True)
            if self.updates == 2:
                assert sha(pretrained) == pretrained_sha
                save(out / 'native_trial_result.json', dict(passed=True,
                    physical_gpu=3, logical_gpu=0, per_gpu_batch=2,
                    disposable_optimizer_updates=2, formal_training=False,
                    parameters_discarded=True, weights_saved=False,
                    parent_protocol_sha256=sha(base/'protocol.json'),
                    source_config_sha256=protocol['config_sha256'],
                    initial_parameters_sha256=self.initial_parameters,
                    final_parameters_sha256=digest,
                    pretrained_cache=str(pretrained), pretrained_cache_sha256=pretrained_sha,
                    parameter_count=36281432, steps=self.rows,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    training_batch_shape=[2,3,1024,1024],
                    phase_note='Epoch4 assignment forced after one step, not a convergence or performance test.',
                    multi_gpu_DDP_checked=False, VAL=False, TEST=False))
                raise TrialComplete()

    hook = TrialHook()
    original_register = BaseRunner.register_training_hooks
    original_run_iter = EpochBasedRunner.run_iter

    def register(runner, *args, **kwargs):
        original_register(runner, *args, **kwargs)
        runner.register_hook(hook, priority='LOWEST')

    def run_iter(runner, data_batch, train_mode, **kwargs):
        assert train_mode and runner.iter in (0,1)
        batch = data_batch['img'].data[0]
        assert tuple(batch.shape) == (2,3,1024,1024), batch.shape
        hook.current_input = input_record(data_batch)
        for boxes in data_batch['gt_bboxes'].data[0]:
            assert len(boxes) == 1
        hook.main.epoch = [0,4][runner.iter]
        return original_run_iter(runner, data_batch, train_mode, **kwargs)

    BaseRunner.register_training_hooks = register
    EpochBasedRunner.run_iter = run_iter
    sys.argv = [str(ROOT/'tools/train.py'), str(ROOT/protocol['config']),
        '--work-dir', str(out/'native_entry'), '--seed','0',
        '--gpu-ids','0','--no-validate']
    try:
        runpy.run_path(str(ROOT/'tools/train.py'), run_name='__main__')
    except TrialComplete:
        print('E_A_NATIVE_GPU3_SMOKE_PASS_DISCARDED', flush=True)
    else:
        raise AssertionError('Native trial did not stop after two disposable steps')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.child:
        native_child(args.base.resolve(), args.out.resolve())
    else:
        sys.exit(controller(args.base.resolve(), args.out.resolve()))
