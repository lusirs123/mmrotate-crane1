"""Resource/status observer for the formal FAA run; no method/loss changes."""
import json
import math
import os
from pathlib import Path
import time
import subprocess
import signal
import sys
import hashlib
from datetime import datetime, timezone, timedelta

import torch
from mmcv.runner import HOOKS, Hook


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.%d.tmp' % os.getpid())
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(str(tmp), str(path))


@HOOKS.register_module()
class FAATrainingMemoryHookV1(Hook):
    def before_run(self, runner):
        self.out = Path(os.environ['FAA_RUN_DIR'])
        self.rank = int(os.environ['LOCAL_RANK'])
        self.started = time.monotonic()
        self.updates = 0
        self.current = {}
        assert runner.max_epochs == 24
        assert runner.optimizer.__class__.__name__ == 'AdamW'
        assert torch.cuda.current_device() == self.rank
        protocol = json.loads((self.out/'protocol.json').read_text())
        atomic_json(self.out/('rank_%d_identity.json' % self.rank),
                    dict(pid=os.getpid(), rank=self.rank))
        total = torch.cuda.get_device_properties(self.rank).total_memory
        torch.cuda.set_per_process_memory_fraction(protocol['allocator_limit_mib']*2**20/total, self.rank)
        torch.cuda.reset_peak_memory_stats(self.rank)
        step = runner.optimizer.step
        def counted_step(*args, **kwargs):
            value = step(*args, **kwargs)
            self.updates += 1
            return value
        runner.optimizer.step = counted_step
        self.write(runner, 'INITIALIZED')

    def before_train_epoch(self, runner):
        assert len(runner.data_loader.dataset) == 2558
        assert runner.data_loader.batch_size == 1
        assert torch.distributed.get_world_size() == 2
        self.steps_per_epoch = len(runner.data_loader)
        torch.cuda.empty_cache()
        self.write(runner, 'TRAINING')

    def before_train_iter(self, runner):
        batch = runner.data_batch['img'].data[0]
        assert tuple(batch.shape) == (1, 3, 1024, 1024), batch.shape

    def after_train_iter(self, runner):
        loss = float(runner.outputs['loss'].detach())
        if not math.isfinite(loss):
            raise RuntimeError('Nonfinite FAA training loss; stopping without automatic restart')
        self.current = dict(loss=loss,
                            loss_terms={k: float(v) for k, v in runner.outputs['log_vars'].items()})
        if runner.iter < 10 or (runner.iter+1) % 10 == 0:
            self.write(runner, 'TRAINING', completed_iteration=runner.iter+1)

    def after_train_epoch(self, runner):
        self.write(runner, 'EPOCH_FINISHED')

    def after_run(self, runner):
        self.write(runner, 'COMPLETED')
        atomic_json(self.out/('rank_%d_complete.json' % self.rank),
                    dict(self.status, completed_epochs=runner.epoch))

    def write(self, runner, state, completed_iteration=None):
        self.status = dict(state=state, rank=self.rank, pid=os.getpid(),
            unix_time=time.time(), elapsed_seconds=time.monotonic()-self.started,
            epoch=min(int(runner.epoch)+1, 24), total_epochs=24,
            steps_per_epoch=getattr(self, 'steps_per_epoch', None),
            completed_microbatches=int(runner.iter if completed_iteration is None else completed_iteration),
            optimizer_updates=self.updates,
            allocated_mib=torch.cuda.memory_allocated(self.rank)/2**20,
            reserved_mib=torch.cuda.memory_reserved(self.rank)/2**20,
            peak_allocated_mib=torch.cuda.max_memory_allocated(self.rank)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(self.rank)/2**20,
            **self.current)
        atomic_json(self.out/('rank_%d_heartbeat.json' % self.rank), self.status)


# The following controller only launches/observes the existing native trainer.
# It does not implement a model, dataloader, optimizer or training loop.
ROOT = Path(__file__).resolve().parents[2]
CONFIG = 'configs/faa/crane_faa_oriented_rcnn_r50_fpn_24e_port_day2night_2gpu_v1.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(out):
    from crane_project.tools.check_faa_adaptation_v1 import check_config, resolve
    from crane_project.tools.smoke_faa_gpu_v1 import gpu_rows
    _, earlier = check_config()
    formal, closure = resolve(ROOT/CONFIG)
    base, _ = resolve(ROOT/earlier['config'])
    for key in ('model', 'data', 'optimizer', 'optimizer_config', 'lr_config',
                'runner', 'evaluation', 'checkpoint_config'):
        assert formal[key] == base[key], key
    assert formal['gpu_ids'] == [0, 1]
    before = {r['index']: r for r in gpu_rows() if r['index'] in (2, 3)}
    assert set(before) == {2, 3}
    assert all(r['free_mib'] >= 7168 for r in before.values()), before
    ordered = sorted(gpu_rows(), key=lambda r: r['pci'])
    ordinals = [next(i for i, r in enumerate(ordered) if r['uuid'] == before[g]['uuid'])
                for g in (2, 3)]
    # Verify existing cache before native init_weights; do not download weights.
    pretrained = Path(torch.hub.get_dir())/'checkpoints/resnet50-0676ba61.pth'
    assert sha(pretrained) == '0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a'
    dataset = ROOT/base['data_root']
    manifest = load(dataset/'manifest.json')
    verified = 0
    for row in manifest['records']:
        if row['split'] == 'test':
            continue
        assert row['split'] in ('train', 'train_sim', 'val')
        for kind in ('images', 'annfiles'):
            assert sha(dataset/row[kind]['path']) == row[kind]['sha256'], row['id']
            verified += 1
    assert verified == (1810+748+887)*2
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'mmrotate').rglob('*.py')}
    for name in list(earlier['source_sha256'])+closure+[
            'tools/train.py', 'mmrotate/apis/train.py',
            'crane_project/utils/faa_training_memory_v1.py',
            'crane_project/tools/smoke_faa_gpu_v1.py']:
        sources[name] = sha(ROOT/name)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'protocol.json', dict(protocol='faa_gpu23_formal_train_v1',
        created_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        config=CONFIG, inherited_configs=closure,
        source_sha256=sources, references_sha256=earlier['references_sha256'],
        physical_gpus=[2, 3], ordinals=ordinals, before=before,
        seed=0, deterministic=True, dtype='float32', samples_per_gpu=1,
        cumulative_iters=2, effective_global_batch=4, max_epochs=24,
        optimizer=formal['optimizer'], lr_config=formal['lr_config'],
        allocator_limit_mib=6144, driver_limit_mib=6656, minimum_free_mib=1536,
        minimum_start_free_mib=7168, auto_restart=False, auto_resume=False,
        train_images=2558, val_images=887, verified_train_val_files=verified,
        pretrained_path=str(pretrained), pretrained_sha256=sha(pretrained),
        checkpoint_selection=dict(split='val', candidates=[16, 18, 20, 22, 24],
            implementation='crane_project/tools/ckpt_sweep.py',
            selection_config=load(ROOT/'configs/faa/upstream_manifest.json')['checkpoint_selection']['selection_config'],
            automatic_sweep=False),
        native_entry='tools/train.py', TEST_inference=False, auto_TEST=False))


def verify_sources(protocol):
    for name, digest in {**protocol['source_sha256'], **protocol['references_sha256']}.items():
        assert sha(ROOT/name) == digest, name
    assert sha(protocol['pretrained_path']) == protocol['pretrained_sha256']


def identities(out):
    result = set()
    for path in out.glob('rank_*_identity.json'):
        result.add(load(path)['pid'])
    return result


def kill_training(child, out):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for pid in identities(out):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pid in identities(out):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.wait(timeout=10)


def monitor(out):
    from crane_project.tools.smoke_faa_gpu_v1 import gpu_rows, processes
    protocol = load(out/'protocol.json')
    verify_sources(protocol)
    current = {r['index']: r for r in gpu_rows() if r['index'] in (2, 3)}
    assert all(r['free_mib'] >= protocol['minimum_start_free_mib'] for r in current.values())
    env = dict(os.environ, FAA_RUN_DIR=str(out), CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUDA_VISIBLE_DEVICES=','.join(map(str, protocol['ordinals'])),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(ROOT), PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nnodes=1', '--nproc_per_node=2', str(ROOT/'tools/train.py'),
        str(ROOT/CONFIG), '--launcher', 'pytorch', '--seed', '0', '--deterministic',
        '--work-dir', str(out/'train')]
    peaks = {str(g): dict(process_mib=0, total_used_mib=0, minimum_free_mib=8192,
                         utilization_peak_pct=0) for g in (2, 3)}
    started = time.monotonic()
    reason = None
    with (out/'native_train.log').open('x') as log, (out/'gpu_samples.jsonl').open('x') as samples:
        child = subprocess.Popen(command, cwd=str(ROOT), env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        write_new(out/'trainer_process.json', dict(pid=child.pid, command=command))
        try:
            while child.poll() is None:
                rows = {r['index']: r for r in gpu_rows() if r['index'] in (2, 3)}
                pids = identities(out)
                apps = [p for p in processes() if p['pid'] in pids]
                allowed = {r['uuid'] for r in rows.values()}
                assert all(a['uuid'] in allowed for a in apps), 'Unexpected GPU use'
                for gpu, row in rows.items():
                    own = sum(p['used_mib'] for p in apps if p['uuid'] == row['uuid'])
                    peak = peaks[str(gpu)]
                    peak['process_mib'] = max(peak['process_mib'], own)
                    peak['total_used_mib'] = max(peak['total_used_mib'], row['used_mib'])
                    peak['minimum_free_mib'] = min(peak['minimum_free_mib'], row['free_mib'])
                    peak['utilization_peak_pct'] = max(peak['utilization_peak_pct'], row['utilization_pct'])
                    if own > protocol['driver_limit_mib']:
                        reason = 'GPU%d_PROCESS_MEMORY_LIMIT' % gpu
                    if row['free_mib'] < protocol['minimum_free_mib']:
                        reason = 'GPU%d_LOW_FREE_MEMORY' % gpu
                sample = dict(unix_time=time.time(), elapsed_seconds=time.monotonic()-started,
                              gpus=rows, owned_processes=apps)
                samples.write(json.dumps(sample)+'\n'); samples.flush()
                heartbeats = {p.stem: load(p) for p in out.glob('rank_*_heartbeat.json')}
                atomic_json(out/'run_status.json', dict(state='RUNNING', pid=os.getpid(),
                    trainer_pid=child.pid, latest=sample, peaks=peaks,
                    heartbeats=heartbeats, auto_restart=False, TEST=False))
                if reason:
                    kill_training(child, out)
                    break
                time.sleep(2)
        except BaseException as error:
            reason = 'MONITOR_ERROR:'+repr(error)
            kill_training(child, out)
        code = child.wait()
    complete = [load(p) for p in sorted(out.glob('rank_*_complete.json'))]
    passed = code == 0 and reason is None and len(complete) == 2 and all(
        r['completed_epochs'] == 24 for r in complete)
    if passed:
        verify_sources(protocol)
        passed = (out/'train/epoch_24.pth').is_file()
    result = dict(state='COMPLETED' if passed else 'STOPPED_OR_FAILED',
        passed=passed, exit_code=code, reason=reason, peaks=peaks,
        elapsed_seconds=time.monotonic()-started, completed_workers=complete,
        after=gpu_rows(), auto_restart=False, TEST=False)
    write_new(out/'completion.json', result)
    atomic_json(out/'run_status.json', result)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Observe memory around the existing native FAA trainer')
    parser.add_argument('--launch', action='store_true', help='Prepare and detach native training with monitoring')
    parser.add_argument('--monitor', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.launch != args.monitor
    os.chdir(ROOT)
    out = args.out.resolve()
    if args.monitor:
        monitor(out)
        return
    prepare(out)
    command = [sys.executable, '-m', 'crane_project.utils.faa_training_memory_v1',
               '--monitor', '--out', str(out)]
    with (out/'supervisor.log').open('x') as log:
        process = subprocess.Popen(command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(state='LAUNCHED', supervisor_pid=process.pid,
        out=str(out), config=CONFIG, command=command, physical_gpus=[2, 3],
        auto_restart=False, TEST=False)
    write_new(out/'launch.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
