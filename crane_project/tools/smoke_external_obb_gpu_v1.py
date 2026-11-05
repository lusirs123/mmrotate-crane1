"""Guarded two-GPU DDP trial for official GauCho / YOLO26. No formal training.

Two native TRAIN microbatches per rank, at most two disposable updates, then
TRAIN-image native inference. No checkpoint or VAL/TEST inference is allowed.
Uses existing resource helpers; all model/loss/optimizer math remains native.
"""
import argparse
from datetime import datetime, timezone, timedelta
from functools import partial
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.prepare_external_obb_v1 import (
    GAUCHO_CONFIG, YOLO_CONFIG, check_configs, check_sources, read, sha, write_new)
from crane_project.tools.smoke_faa_gpu_v1 import (
    gpu_rows, processes, worker_pids, stop_group)

GPUS = (2, 3)
ALLOCATOR_MIB = 6144
DRIVER_MIB = 6656
MIN_FREE_MIB = 1536


def source_identity():
    check_sources()
    check_configs()
    names = {str(Path(__file__).resolve().relative_to(ROOT)), GAUCHO_CONFIG, YOLO_CONFIG,
             'crane_project/tools/prepare_external_obb_v1.py',
             'crane_project/tools/smoke_faa_gpu_v1.py'}
    for name in ('gaucho', 'yolo26'):
        path = 'configs/%s/upstream_manifest.json' % name
        manifest = read(ROOT / path)
        names.add(path)
        names.update(manifest['files'])
        names.update(manifest.get('shared_dependencies', {}))
    return {name: sha(ROOT / name) for name in sorted(names)}


def controller(model_name, out, gpus):
    identities = source_identity()
    before = {r['index']: r for r in gpu_rows() if r['index'] in gpus}
    if set(before) != set(gpus) or any(r['free_mib'] < 7168 for r in before.values()):
        raise RuntimeError('Selected GPUs must each have at least 7168 MiB free')
    uuids = {r['uuid'] for r in before.values()}
    if any(p['uuid'] in uuids for p in processes()):
        raise RuntimeError('Selected GPUs have another compute process')
    ordered = sorted(gpu_rows(), key=lambda r: r['pci'])
    ordinals = [next(i for i, r in enumerate(ordered) if r['uuid'] == before[g]['uuid']) for g in gpus]
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'protocol.json', dict(model=model_name, sources=identities,
        before=before, gpus=list(gpus), ordinals=ordinals, microbatches_per_rank=2,
        maximum_disposable_updates_per_rank=2, parameters_discarded=True,
        allocator_mib=ALLOCATOR_MIB, process_mib=DRIVER_MIB,
        minimum_free_mib=MIN_FREE_MIB, formal_training=False,
        checkpoint_files=0, VAL_inference=False, TEST=False, auto_retry=False,
        created_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat()))
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUDA_VISIBLE_DEVICES=','.join(map(str, ordinals)), PYTHONPATH=str(ROOT),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true', WANDB_DISABLED='true',
        YOLO_CONFIG_DIR=str(out / 'settings'), TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=2', str(Path(__file__).resolve()), '--worker',
        '--model', model_name, '--out', str(out)]
    start = time.monotonic()
    peaks = {g: dict(process_mib=0, gpu_used_mib=before[g]['used_mib'],
                     minimum_free_mib=before[g]['free_mib'], utilization_max_pct=0) for g in gpus}
    reason = None
    with (out / 'native.log').open('x') as log, (out / 'gpu_samples.jsonl').open('x') as samples:
        child = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        write_new(out / 'launcher.json', dict(pid=child.pid, command=command))
        while child.poll() is None:
            try:
                rows = {r['index']: r for r in gpu_rows() if r['index'] in gpus}
                pids = worker_pids(out)
                apps = [p for p in processes() if p['pid'] in pids]
                outsiders = [p for p in processes() if p['uuid'] in uuids and p['pid'] not in pids]
                # Allow the short gap between native CUDA initialization and
                # the worker's identity file becoming visible on shared disk.
                if outsiders and time.monotonic() - start > 15:
                    reason = 'OTHER_PROCESS_ON_SELECTED_GPUS'
                if any(p['uuid'] not in uuids for p in apps):
                    reason = 'WRONG_GPU'
                samples.write(json.dumps(dict(seconds=time.monotonic()-start,
                    gpus=rows, owned_processes=apps)) + '\n')
                samples.flush()
                for gpu in gpus:
                    row = rows[gpu]
                    own = sum(p['used_mib'] for p in apps if p['uuid'] == row['uuid'])
                    peaks[gpu]['process_mib'] = max(peaks[gpu]['process_mib'], own)
                    peaks[gpu]['gpu_used_mib'] = max(peaks[gpu]['gpu_used_mib'], row['used_mib'])
                    peaks[gpu]['minimum_free_mib'] = min(peaks[gpu]['minimum_free_mib'], row['free_mib'])
                    peaks[gpu]['utilization_max_pct'] = max(peaks[gpu]['utilization_max_pct'], row['utilization_pct'])
                    if own > DRIVER_MIB or row['free_mib'] < MIN_FREE_MIB:
                        reason = 'GPU%d_MEMORY_GUARD' % gpu
                if time.monotonic() - start > 900:
                    reason = 'TIMEOUT'
            except Exception as error:
                reason = 'MONITOR_ERROR:' + repr(error)
            if reason:
                stop_group(child, out)
                break
            time.sleep(.5)
        code = child.wait()
    reports = [read(p) for p in sorted(out.glob('rank_*_result.json'))]
    unchanged = identities == source_identity()
    passed = code == 0 and reason is None and len(reports) == 2 and all(r['passed'] for r in reports) and unchanged
    checkpoints = list(out.rglob('*.pth')) + list(out.rglob('*.pt'))
    if checkpoints:
        passed = False
        reason = 'UNEXPECTED_CHECKPOINT'
    result = dict(passed=passed, model=model_name, exit_code=code, stop_reason=reason,
        elapsed_seconds=time.monotonic()-start, sampled_peaks=peaks,
        sample_interval_seconds=.5, after=gpu_rows(), workers=reports,
        failures=[read(p) for p in sorted(out.glob('rank_*_failure.json'))],
        sources_unchanged=unchanged, formal_training=False,
        checkpoints=len(checkpoints), VAL_inference=False, TEST=False)
    write_new(out / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if passed else 1


def gaucho_worker(rank, out, torch):
    import copy
    import logging
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from mmcv import Config
    from mmcv.parallel import MMDistributedDataParallel, collate, scatter
    from mmcv.runner import EpochBasedRunner, Hook, build_optimizer, init_dist
    from mmrotate.models import build_detector
    from mmrotate.datasets import build_dataset
    init_dist('pytorch', backend='nccl')
    cfg = Config.fromfile(str(ROOT / GAUCHO_CONFIG))
    pretrained = Path(torch.hub.get_dir()) / 'checkpoints/resnet50-0676ba61.pth'
    if sha(pretrained) != '0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a':
        raise ValueError('Expected the same ImageNet ResNet50 cache as FAA')
    model = build_detector(cfg.model)
    model.init_weights()
    model.cuda(rank)
    wrapped = MMDistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False,
                                       find_unused_parameters=cfg.find_unused_parameters)
    optimizer = build_optimizer(wrapped, cfg.optimizer)
    observations = []
    updates = [0]
    original_step = optimizer.step
    no_gradient = set()
    def step(*args, **kwargs):
        if updates[0] >= 2:
            raise RuntimeError('Trial optimizer update bound exceeded')
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError('Nonfinite or missing GauCho gradients')
        no_gradient.update(n for n, p in model.named_parameters() if p.requires_grad and p.grad is None)
        result = original_step(*args, **kwargs)
        updates[0] += 1
        return result
    optimizer.step = step
    dataset = build_dataset(cfg.data.train)
    indices = [rank, 1810+rank]
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=partial(collate, samples_per_gpu=1))
    logger = logging.getLogger('gaucho_trial_%d' % rank)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    runner = EpochBasedRunner(wrapped, optimizer=optimizer, work_dir=str(out), logger=logger, max_epochs=1)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config, checkpoint_config=None, log_config=None)
    class Observe(Hook):
        def after_train_iter(self, runner):
            if runner.iter >= 2 or not math.isfinite(float(runner.outputs['loss'])):
                raise ValueError('GauCho trial bound/nonfinite loss')
            observations.append(dict(iteration=int(runner.iter), loss=float(runner.outputs['loss']),
                                     terms=runner.outputs['log_vars'], updates=updates[0]))
            write_new(out / ('rank_%d_batch_%d.json' % (rank, runner.iter)), observations[-1])
    runner.register_hook(Observe(), priority='LOWEST')
    initial = model.bbox_head.conv_reg.weight.detach().clone()
    runner.run([loader], [('train', 1)])
    assert updates[0] == 2 and len(observations) == 2
    changed = not torch.equal(initial, model.bbox_head.conv_reg.weight)
    if not changed:
        raise ValueError('GauCho temporary update did not affect its regression head')
    train_peak = dict(allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
                      reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20)
    optimizer.zero_grad(set_to_none=True)
    infer_cfg = copy.deepcopy(cfg.data.train[rank])
    infer_cfg.update(pipeline=cfg.test_pipeline, test_mode=True)
    inference_dataset = build_dataset(infer_cfg)
    item = scatter(collate([inference_dataset[0]], samples_per_gpu=1), [rank])[0]
    model.eval()
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **item)
    boxes = np.asarray(result[0][0])
    if boxes.ndim != 2 or boxes.shape[1] != 6 or not np.isfinite(boxes).all():
        raise ValueError('GauCho native original-image result invalid')
    return dict(passed=True, microbatches=observations, disposable_updates=updates[0],
        regression_weight_changed=changed, train_peak=train_peak,
        no_gradient_parameters=sorted(no_gradient), optimizer='SGD',
        native_TRAIN_boxes=len(boxes), train_indices=indices,
        pretrained_sha256=sha(pretrained), parameters_discarded=True)


class BoundedLoader:
    """Limit iteration only; preserve the native dataset and all its transforms."""
    def __init__(self, loader):
        self.loader = loader
    def __len__(self):
        return 2
    def __iter__(self):
        return itertools.islice(iter(self.loader), 2)
    def __getattr__(self, name):
        return getattr(self.loader, name)


def yolo_worker(rank, out, torch):
    sys.path.insert(0, str(ROOT / 'third_party/ultralytics'))
    from ultralytics.models.yolo.obb.train import OBBTrainer
    from ultralytics.utils import callbacks
    from ultralytics.utils.torch_utils import unwrap_model
    from ultralytics import YOLO
    # No experiment tracking/account communication in this bounded local trial.
    callbacks.add_integration_callbacks = lambda instance: None
    cfg = read(ROOT / YOLO_CONFIG)
    init = ROOT / cfg['initialization']['checkpoint']
    identity = read(init.with_suffix('.initialization.json'))
    if sha(init) != identity['checkpoint_sha256']:
        raise ValueError('YOLO ImageNet initialization changed')
    args = dict(cfg['train_args'], model=str(ROOT / cfg['architecture']),
        data=str(ROOT / cfg['data']), pretrained=str(init), workers=0,
        val=False, save=False, save_period=-1, project=str(out / 'native'),
        name='bounded', plots=False, exist_ok=True)
    observations = []
    updates = [0]
    class TrialTrainer(OBBTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
            # The native trainer constructs a validator even with val=False.
            # Give it TRAIN metadata; validate() is separately forbidden below.
            dataset_path = self.data['train'] if mode != 'train' else dataset_path
            loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
            return BoundedLoader(loader) if mode == 'train' else loader
        def validate(self):
            raise RuntimeError('VAL inference is forbidden in the trial')
        def save_model(self):
            raise RuntimeError('Checkpoint writing is forbidden in the trial')
        def optimizer_step(self):
            if updates[0] >= 2:
                raise RuntimeError('Trial update bound exceeded')
            gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError('Nonfinite YOLO gradients')
            value = super().optimizer_step()
            updates[0] += 1
            return value
    trainer = TrialTrainer(overrides=args)
    def observe(t):
        loss = float(t.loss.detach())
        if not math.isfinite(loss) or len(observations) >= 2:
            raise ValueError('Nonfinite YOLO loss or microbatch bound exceeded')
        items = ({k: float(v.detach().cpu()) for k, v in t.loss_items.items()}
                 if isinstance(t.loss_items, dict) else t.loss_items.detach().cpu().tolist())
        observations.append(dict(iteration=len(observations), loss=loss, updates=updates[0],
                                  loss_items=items))
        write_new(out / ('rank_%d_batch_%d.json' % (rank, len(observations)-1)), observations[-1])
    trainer.add_callback('on_train_batch_end', observe)
    trainer.add_callback('on_fit_epoch_end', lambda t: setattr(t, 'stop', True))
    trainer.train()
    if len(observations) != 2 or not 1 <= updates[0] <= 2:
        raise ValueError('YOLO trial did not execute the intended bounded steps')
    train_peak = dict(allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
                      reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20)
    model = unwrap_model(trainer.model)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite updated YOLO parameters')
    trainer.optimizer.zero_grad(set_to_none=True)
    # Hand the temporary module to the official predictor, without saving it.
    inference = YOLO(str(ROOT / cfg['architecture']), task='obb', verbose=False)
    inference.model = model.eval()
    image = next((ROOT / cfg['converted_data'] / 'train/images').glob('*.jpg'))
    with torch.no_grad():
        result = inference.predict(source=str(image), device=rank, **cfg['prediction'])[0]
    boxes = result.obb.xywhr
    if not torch.isfinite(boxes).all() or not inference.predictor.model.end2end:
        raise ValueError('YOLO native TRAIN inference failed')
    return dict(passed=True, microbatches=observations, disposable_updates=updates[0],
        train_peak=train_peak, optimizer=trainer.optimizer.__class__.__name__,
        optimizer_lr0=trainer.args.lr0, native_TRAIN_boxes=len(boxes),
        initialization_sha256=identity['checkpoint_sha256'], parameters_discarded=True,
        training_dataset_frames=len(trainer.train_loader.dataset),
        original_budget_epochs=cfg['train_args']['epochs'], bounded_epoch_slices=1)


def worker(model_name, out):
    rank = int(os.environ['LOCAL_RANK'])
    write_new(out / ('rank_%d_identity.json' % rank), dict(pid=os.getpid(), rank=rank))
    protocol = read(out / 'protocol.json')
    for name, digest in protocol['sources'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Trial source changed: ' + name)
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    total = torch.cuda.get_device_properties(rank).total_memory
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_MIB*2**20/total, rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats(rank)
    try:
        result = gaucho_worker(rank, out, torch) if model_name == 'gaucho' else yolo_worker(rank, out, torch)
        torch.cuda.synchronize(rank)
        result.update(rank=rank, physical_gpu=protocol.get('gpus', list(GPUS))[rank], torch=torch.__version__,
            cuda=torch.version.cuda, input_shape=[1, 3, 1024, 1024],
            peak_allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20,
            formal_training=False, VAL_inference=False, TEST=False)
        write_new(out / ('rank_%d_result.json' % rank), result)
        if dist.is_initialized():
            dist.barrier()
    except BaseException as error:
        write_new(out / ('rank_%d_failure.json' % rank), dict(rank=rank,
            error=repr(error), peak_allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('gaucho', 'yolo26'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs=2, default=list(GPUS))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        worker(args.model, args.out.resolve())
        return 0
    if tuple(args.gpus) not in ((0, 1), (2, 3)):
        parser.error('--gpus must select physical 0 1 or 2 3')
    return controller(args.model, args.out.resolve(), tuple(args.gpus))


if __name__ == '__main__':
    raise SystemExit(main())
