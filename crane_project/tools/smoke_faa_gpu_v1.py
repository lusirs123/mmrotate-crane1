"""Bounded FAA GPU2/3 DDP trial, never a formal training/TEST launcher.

Two microbatches/rank, one disposable AdamW update, then TRAIN-image inference.
No checkpoint is written; the final parameters are discarded. GPU guardrails
are checked by an independent controller while each worker runs native hooks.
"""
import argparse
from datetime import datetime, timezone, timedelta
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
from crane_project.tools.check_faa_adaptation_v1 import CONFIG, check_config

GPUS = (2, 3)
ALLOCATOR_MIB = 6144
DRIVER_MIB = 6656
MIN_FREE_MIB = 1536
MIN_START_FREE_MIB = 7168
TIMEOUT_SECONDS = 600


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_rows():
    value = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in value.splitlines():
        index, uuid, pci, total, used, free, util = [p.strip() for p in line.split(',')]
        rows.append(dict(index=int(index), uuid=uuid, pci=pci,
                         total_mib=int(total), used_mib=int(used),
                         free_mib=int(free), utilization_pct=int(util)))
    return rows


def processes():
    value = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in value.splitlines():
        uuid, pid, memory = [p.strip() for p in line.split(',')]
        rows.append(dict(uuid=uuid, pid=int(pid), used_mib=int(memory)))
    return rows


def worker_pids(out):
    return {json.loads(p.read_text())['pid'] for p in out.glob('rank_*_identity.json')}


def stop_group(child, out):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for pid in worker_pids(out):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)


def controller(out):
    _, static = check_config()
    assert len(gpu_rows()) >= 4
    before = {r['index']: r for r in gpu_rows() if r['index'] in GPUS}
    assert set(before) == set(GPUS)
    assert all(r['free_mib'] >= MIN_START_FREE_MIB for r in before.values()), before
    pci_order = sorted(gpu_rows(), key=lambda r: r['pci'])
    ordinals = [next(i for i, r in enumerate(pci_order) if r['uuid'] == before[g]['uuid'])
                for g in GPUS]
    out.mkdir(parents=True, exist_ok=False)
    save(out/'protocol.json', dict(protocol='faa_gpu23_bounded_ddp_smoke_v1',
        created_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        config=CONFIG, source_sha256=static['source_sha256'],
        references_sha256=static['references_sha256'], script_sha256=sha(__file__),
        gpus=list(GPUS), before=before, ordinals=ordinals,
        allocator_limit_mib=ALLOCATOR_MIB, driver_limit_mib=DRIVER_MIB,
        minimum_free_mib=MIN_FREE_MIB, timeout_seconds=TIMEOUT_SECONDS,
        microbatches_per_rank=2, samples_per_gpu=1, gradient_accumulation=2,
        disposable_optimizer_updates=1, trial_effective_global_batch=4,
        config_default_effective_global_batch=2, config_modified=False,
        trial_note='Two-GPU resource trial; formal config still declares one GPU.',
        formal_training=False, checkpoints=False, VAL=False, TEST=False))
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CUDA_VISIBLE_DEVICES=','.join(map(str, ordinals)),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=str(ROOT))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nnodes=1', '--nproc_per_node=2', str(Path(__file__).resolve()),
               '--worker', '--out', str(out)]
    peaks = {g: dict(process_mib=0, gpu_used_mib=before[g]['used_mib'],
                    minimum_free_mib=before[g]['free_mib'], utilization_max_pct=0,
                    utilization_samples=[]) for g in GPUS}
    start = time.monotonic()
    reason = None
    with (out/'native.log').open('x') as log, (out/'gpu_samples.jsonl').open('x') as samples:
        child = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        save(out/'launcher.json', dict(pid=child.pid, command=command))
        while child.poll() is None:
            try:
                rows = {r['index']: r for r in gpu_rows() if r['index'] in GPUS}
                pids = worker_pids(out)
                apps = [a for a in processes() if a['pid'] in pids]
                assert all(a['uuid'] in {r['uuid'] for r in before.values()} for a in apps)
                record = dict(seconds=time.monotonic()-start, gpus=rows, owned_processes=apps)
                samples.write(json.dumps(record)+'\n'); samples.flush()
                for gpu in GPUS:
                    row = rows[gpu]
                    own = sum(a['used_mib'] for a in apps if a['uuid'] == row['uuid'])
                    p = peaks[gpu]
                    p['process_mib'] = max(p['process_mib'], own)
                    p['gpu_used_mib'] = max(p['gpu_used_mib'], row['used_mib'])
                    p['minimum_free_mib'] = min(p['minimum_free_mib'], row['free_mib'])
                    p['utilization_max_pct'] = max(p['utilization_max_pct'], row['utilization_pct'])
                    p['utilization_samples'].append(row['utilization_pct'])
                    if own > DRIVER_MIB:
                        reason = 'GPU%d_PROCESS_MEMORY_LIMIT' % gpu
                    if row['free_mib'] < MIN_FREE_MIB:
                        reason = 'GPU%d_LOW_FREE_MEMORY' % gpu
                if time.monotonic()-start > TIMEOUT_SECONDS:
                    reason = 'TIMEOUT'
            except Exception as error:
                reason = 'MONITOR_ERROR:'+repr(error)
            if reason:
                stop_group(child, out)
                break
            time.sleep(.5)
        code = child.wait()
    for row in peaks.values():
        values = row.pop('utilization_samples')
        row['utilization_mean_pct_including_startup'] = sum(values)/len(values) if values else None
        row['sample_count'] = len(values)
    reports = [json.loads(p.read_text()) for p in sorted(out.glob('rank_*_result.json'))]
    passed = code == 0 and reason is None and len(reports) == 2 and all(r['passed'] for r in reports)
    assert not list(out.rglob('*.pth')), 'Checkpoint writing is forbidden'
    result = dict(passed=passed, status='PASS' if passed else 'FAILED_OR_GUARDED_STOP',
        child_exit_code=code, stop_reason=reason, elapsed_seconds=time.monotonic()-start,
        gpu_sample_interval_seconds=.5, sampled_peaks=peaks,
        after=gpu_rows(), worker_reports=reports,
        failures=[json.loads(p.read_text()) for p in sorted(out.glob('rank_*_failure.json'))],
        formal_training=False, checkpoint_files=0, VAL=False, TEST=False)
    save(out/'result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if passed else 1


def worker(out):
    rank = int(os.environ['LOCAL_RANK'])
    save(out/('rank_%d_identity.json' % rank), dict(pid=os.getpid(), local_rank=rank))
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset
    from mmcv import Config
    from mmcv.parallel import MMDistributedDataParallel, collate, scatter
    from mmcv.runner import EpochBasedRunner, Hook, build_optimizer, init_dist
    from mmcv.runner.checkpoint import get_torchvision_models
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from mmdet.apis import set_random_seed
    from functools import partial
    import copy
    import logging

    protocol = json.loads((out/'protocol.json').read_text())
    assert sha(__file__) == protocol['script_sha256']
    for name, digest in {**protocol['source_sha256'], **protocol['references_sha256']}.items():
        assert sha(ROOT/name) == digest, name
    assert torch.cuda.device_count() == 2
    torch.cuda.set_device(rank)
    total = torch.cuda.get_device_properties(rank).total_memory
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_MIB*2**20/total, rank)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    probe = torch.empty(1, device='cuda:%d' % rank)
    expected_uuid = protocol['before'][str(GPUS[rank])]['uuid']
    observed = set()
    for _ in range(20):
        observed = {p['uuid'] for p in processes() if p['pid'] == os.getpid()}
        if observed:
            break
        time.sleep(.1)
    assert observed == {expected_uuid}, (observed, expected_uuid)
    del probe
    phase = 'initialize'
    updates = [0]
    try:
        init_dist('pytorch', backend='nccl')
        set_random_seed(0)
        cfg = Config.fromfile(str(ROOT/CONFIG))
        url = get_torchvision_models()['resnet50']
        pretrained = Path(torch.hub.get_dir())/'checkpoints'/url.rsplit('/', 1)[-1]
        assert pretrained.is_file(), 'Existing ImageNet cache required; no downloads in this trial'
        pretrained_sha = sha(pretrained)
        model = build_detector(cfg.model)
        model.init_weights()
        model = model.cuda(rank)
        wrapped = MMDistributedDataParallel(model, device_ids=[rank],
                    broadcast_buffers=False, find_unused_parameters=cfg.find_unused_parameters)
        optimizer = build_optimizer(wrapped, cfg.optimizer)
        original_step = optimizer.step
        gradient_checks = []
        def step(*args, **kwargs):
            assert updates[0] == 0, 'Only one disposable optimizer update allowed'
            for name, module in [('FAA_neck', model.neck), ('FAA_head', model.roi_head.bbox_head),
                                 ('RPN', model.rpn_head)]:
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads), name
                gradient_checks.append(name)
            value = original_step(*args, **kwargs)
            updates[0] += 1
            return value
        optimizer.step = step
        ds = build_dataset(cfg.data.train)
        # Real+Sim on each rank, four distinct TRAIN frames globally.
        indices = [rank, 1810+rank]
        loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=partial(collate, samples_per_gpu=1))
        logger = logging.getLogger('faa_smoke_rank%d' % rank)
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())
        runner = EpochBasedRunner(wrapped, optimizer=optimizer, work_dir=str(out),
                                  logger=logger, max_epochs=1)
        runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                       checkpoint_config=None, log_config=None)
        rows = []
        initial = model.roi_head.bbox_head.gamma.detach().cpu().clone()
        class Observe(Hook):
            def before_train_iter(self, runner):
                nonlocal phase
                assert runner.iter in (0, 1)
                phase = 'microbatch_%d_forward_backward' % runner.iter
                print('FAA_PHASE', rank, phase, flush=True)
            def after_train_iter(self, runner):
                torch.cuda.synchronize(rank)
                assert torch.isfinite(runner.outputs['loss']).all()
                rows.append(dict(microbatch=int(runner.iter),
                    loss=float(runner.outputs['loss'].detach()),
                    loss_terms=runner.outputs['log_vars'],
                    optimizer_updates=updates[0],
                    peak_allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20))
                save(out/('rank_%d_microbatch_%d.json' % (rank, runner.iter)), rows[-1])
        runner.register_hook(Observe(), priority='LOWEST')
        torch.cuda.reset_peak_memory_stats(rank)
        phase = 'native_runner'
        runner.run([loader], [('train', 1)])
        assert updates[0] == 1 and len(rows) == 2
        changed = not torch.equal(initial, model.roi_head.bbox_head.gamma.detach().cpu())
        assert changed, 'FAA learned scale did not change after disposable update'
        train_peak = dict(allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
                          reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20)
        optimizer.zero_grad(set_to_none=True)
        phase = 'TRAIN_native_inference'
        print('FAA_PHASE', rank, phase, flush=True)
        infer_spec = copy.deepcopy(cfg.data.train[rank])
        infer_spec['pipeline'] = cfg.test_pipeline
        infer_spec['test_mode'] = True
        infer_ds = build_dataset(infer_spec)
        item = collate([infer_ds[0]], samples_per_gpu=1)
        gpu_item = scatter(item, [rank])[0]
        model.eval()
        with torch.no_grad():
            predictions = model(return_loss=False, rescale=True, **gpu_item)
        assert len(predictions) == 1 and len(predictions[0]) == 1
        import numpy as np
        boxes = np.asarray(predictions[0][0])
        assert boxes.ndim == 2 and boxes.shape[1] == 6 and np.isfinite(boxes).all()
        torch.cuda.synchronize(rank)
        assert sha(pretrained) == pretrained_sha
        save(out/('rank_%d_result.json' % rank), dict(passed=True, rank=rank,
            physical_gpu=GPUS[rank], gpu_uuid=expected_uuid,
            microbatches=rows, disposable_optimizer_updates=updates[0],
            finite_gradient_modules=gradient_checks, FAA_gamma_changed=changed,
            pretrained_path=str(pretrained), pretrained_sha256=pretrained_sha,
            input_shape=[1, 3, 1024, 1024], train_indices=indices,
            train_peak=train_peak, peak_allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20,
            TRAIN_inference=dict(image=infer_ds.data_infos[0]['filename'],
                                 finite_native_boxes=int(len(boxes))),
            parameters_discarded=True, checkpoints=False,
            formal_training=False, VAL=False, TEST=False))
        dist.barrier()
        return 0
    except Exception as error:
        save(out/('rank_%d_failure.json' % rank), dict(rank=rank, phase=phase,
            exception=type(error).__name__, message=str(error),
            disposable_optimizer_updates=updates[0],
            peak_allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20,
            formal_training=False, checkpoint_files=0))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    return worker(args.out.resolve()) if args.worker else controller(args.out.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
