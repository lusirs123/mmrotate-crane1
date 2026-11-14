"""Guarded native Oriented R-CNN training on physical GPUs 0/1.

Smoke: two disposable TRAIN updates per rank and TRAIN-only inference.
Formal: fresh ImageNet initialization, author 12-epoch optimizer schedule.
No offline checkpoint selection or TEST is performed by this launcher.
"""
import argparse
from datetime import datetime, timezone, timedelta
from functools import partial
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.check_oriented_rcnn_comparison_v1 import (
    CONFIG, EPOCHS, check_config, source_identity)
from crane_project.tools.prepare_external_obb_v1 import (
    DATA, MANIFEST_SHA, dataset_manifest, sha, write_new)
from crane_project.utils import external_training_monitor_io_v1 as io
from crane_project.tools.smoke_faa_gpu_v1 import stop_group

GPUS = (0, 1)
RESNET_SHA = '0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a'
MODEL = 'oriented_rcnn'


def idle():
    all_rows = io.gpu_rows()
    rows = {r['index']: r for r in all_rows if r['index'] in GPUS}
    if set(rows) != set(GPUS) or any(r['free_mib'] < 7168 for r in rows.values()):
        raise RuntimeError('Physical GPU0/1 each require at least 7168 MiB free')
    if any(p['uuid'] in {r['uuid'] for r in rows.values()} for p in io.processes()):
        raise RuntimeError('Selected GPU has another compute process')
    pci_order = sorted(all_rows, key=lambda r: r['pci'])
    if [next(i for i, r in enumerate(pci_order) if r['uuid'] == rows[g]['uuid']) for g in GPUS] != [0, 1]:
        raise RuntimeError('GPU0/1 are not PCI ordinals 0/1')
    return rows


def assert_sources(protocol):
    for path, digest in protocol['sources'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Source drift: '+path)
    for path, digest in protocol['initialization'].items():
        if sha(path) != digest:
            raise ValueError('Initialization drift: '+path)


def runtime_check():
    from mmcv import Config
    cfg, _ = check_config()
    actual = Config.fromfile(str(ROOT/CONFIG), import_custom_modules=False)
    if dict(actual.items()) != cfg:
        raise ValueError('MMCV and static configuration resolution differ')
    return cfg


def prepare(out, smoke=None):
    cfg = runtime_check()
    sources = source_identity()
    before = idle()
    verified = 0
    for row in dataset_manifest():
        if row['split'] == 'test':
            continue
        for kind in ('images', 'annfiles'):
            if sha(DATA/row[kind]['path']) != row[kind]['sha256']:
                raise ValueError('TRAIN/VAL file drift: '+row['id'])
            verified += 1
    if verified != 6890:
        raise ValueError('Incomplete TRAIN/VAL verification')
    resnet = Path.home()/'.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'
    if sha(resnet) != RESNET_SHA:
        raise ValueError('Wrong ImageNet initialization')
    if smoke:
        report = io.read_json(smoke/'result.json')
        trial = io.read_json(smoke/'protocol.json')
        if not report['passed'] or trial['sources'] != sources:
            raise ValueError('Need passing smoke on identical sources')
        if trial['manifest_sha256'] != MANIFEST_SHA or trial['initialization'] != {str(resnet): RESNET_SHA}:
            raise ValueError('Smoke data/initialization differs')
    from crane_project.utils.crane_frozen_pipeline_v1 import read_constants
    selection = read_constants(ROOT/'crane_project/tools/ckpt_sweep.py', ['SELECTION_CONFIG'])['SELECTION_CONFIG']
    protocol = dict(model=MODEL, config=CONFIG, sources=sources,
        created_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        initialization={str(resnet): RESNET_SHA}, manifest_sha256=MANIFEST_SHA,
        physical_gpus=list(GPUS), ordinals=list(GPUS), before=before,
        allocator_mib=6144, driver_mib=6656, minimum_free_mib=1536,
        verified_train_val_files=verified, train_frames=2558, val_frames=887,
        global_batch=2, samples_per_gpu=1, dtype='FP32', seed=0,
        optimizer=cfg['optimizer'], lr_config=cfg['lr_config'], epochs=12,
        own_scale_augmentation=False, formal_training=bool(smoke),
        smoke_directory=str(smoke) if smoke else None,
        checkpoint_selection=dict(split='val', candidates=EPOCHS, rule=selection,
            deferred_until_training_sufficiency_review=True),
        evaluation_plan='Review TRAIN/VAL sufficiency before locking one checkpoint and TEST.',
        auto_resume=False, auto_restart=False, auto_TEST=False)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'protocol.json', protocol)
    return protocol


def guarded_run(folder, protocol, smoke=False):
    assert_sources(protocol)
    idle()
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0,1',
        EXTERNAL_RUN_DIR=str(folder), PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128', NCCL_ASYNC_ERROR_HANDLING='1')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        str(Path(__file__).resolve()), '--stage', 'smoke-worker' if smoke else 'worker', '--out', str(folder)]
    peaks = {str(g): dict(process_mib=0, total_used_mib=0, minimum_free_mib=8192) for g in GPUS}
    reason = None
    start = time.monotonic()
    rows = protocol['before']
    with (folder/'native_train.log').open('x') as log, (folder/'gpu_samples.jsonl').open('x') as samples:
        child = subprocess.Popen(command, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_new(folder/'launcher.json', dict(pid=child.pid, command=command))
        try:
            while child.poll() is None:
                rows = {r['index']: r for r in io.gpu_rows() if r['index'] in GPUS}
                if set(rows) != set(GPUS):
                    raise RuntimeError('Selected GPU disappeared')
                pids = {io.read_json(p)['pid'] for p in folder.glob('rank_*_identity.json')}
                apps = io.processes()
                owned = [p for p in apps if p['pid'] in pids]
                uuids = {r['uuid'] for r in rows.values()}
                if any(p['uuid'] not in uuids for p in owned):
                    reason = 'WRONG_GPU'
                if time.monotonic()-start > 15 and any(p['uuid'] in uuids and p['pid'] not in pids for p in apps):
                    reason = 'OTHER_PROCESS_ON_GPU01'
                for g, row in rows.items():
                    memory = sum(p['used_mib'] for p in owned if p['uuid'] == row['uuid'])
                    peaks[str(g)]['process_mib'] = max(peaks[str(g)]['process_mib'], memory)
                    peaks[str(g)]['total_used_mib'] = max(peaks[str(g)]['total_used_mib'], row['used_mib'])
                    peaks[str(g)]['minimum_free_mib'] = min(peaks[str(g)]['minimum_free_mib'], row['free_mib'])
                    if memory > protocol['driver_mib'] or row['free_mib'] < protocol['minimum_free_mib']:
                        reason = 'GPU%d_MEMORY_GUARD' % g
                if smoke and time.monotonic()-start > 900:
                    reason = 'SMOKE_TIMEOUT'
                item = dict(unix_time=time.time(), elapsed_seconds=time.monotonic()-start,
                            gpus=rows, owned_processes=owned)
                samples.write(json.dumps(item)+'\n'); samples.flush()
                heartbeats = {p.stem: io.read_json(p) for p in folder.glob('rank_*_heartbeat.json')}
                io.atomic_json(folder/'status.json', dict(state='RUNNING', model=MODEL,
                    formal_training=not smoke, peaks=peaks, latest=item, heartbeats=heartbeats, TEST=False))
                if reason:
                    stop_group(child, folder)
                    break
                time.sleep(0.5 if smoke else 2)
        except BaseException as error:
            reason = 'MONITOR_ERROR:'+repr(error)
            traceback.print_exc()
            stop_group(child, folder)
        code = child.wait()
    reports = [io.read_json(p) for p in sorted(folder.glob('rank_*_result.json' if smoke else 'rank_*_complete.json'))]
    passed = code == 0 and reason is None and len(reports) == 2 and all(r['passed'] for r in reports)
    if passed:
        try:
            assert_sources(protocol)
            if smoke:
                if list(folder.rglob('*.pth')):
                    raise ValueError('Unexpected smoke checkpoint')
            elif not all((folder/'train'/('epoch_%d.pth' % e)).is_file() for e in EPOCHS):
                raise ValueError('Missing VAL candidate checkpoint')
        except BaseException as error:
            passed = False; reason = repr(error)
    result = dict(passed=passed, state='COMPLETED' if passed else 'STOPPED_OR_FAILED',
        model=MODEL, exit_code=code, reason=reason, elapsed_seconds=time.monotonic()-start,
        sampled_peaks=peaks, workers=reports, final_observed_gpus=rows,
        failures=[io.read_json(p) for p in sorted(folder.glob('rank_*_failure.json'))],
        formal_training=not smoke, auto_resume=False, auto_restart=False, TEST=False)
    write_new(folder/('result.json' if smoke else 'completion.json'), result)
    io.atomic_json(folder/'status.json', result)
    return passed


def smoke_worker(rank, out, torch):
    import copy
    import logging
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from mmcv import Config
    from mmcv.parallel import MMDistributedDataParallel, collate, scatter
    from mmcv.runner import EpochBasedRunner, Hook, build_optimizer, init_dist
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from mmrotate.datasets import build_dataset
    init_dist('pytorch', backend='nccl')
    cfg = Config.fromfile(str(ROOT/CONFIG))
    import_modules_from_strings(**cfg.custom_imports)
    model = build_detector(cfg.model)
    model.init_weights()
    model.cuda(rank)
    wrapped = MMDistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False,
                                       find_unused_parameters=cfg.find_unused_parameters)
    optimizer = build_optimizer(wrapped, cfg.optimizer)
    updates = [0]
    observations = []
    step = optimizer.step
    def counted(*args, **kwargs):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if updates[0] >= 2 or not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError('Smoke update bound or invalid gradients')
        result = step(*args, **kwargs); updates[0] += 1
        return result
    optimizer.step = counted
    dataset = build_dataset(cfg.data.train)
    if len(dataset) != 2558:
        raise ValueError('Wrong training dataset')
    indices = [rank, 1810+rank]
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=partial(collate, samples_per_gpu=1))
    logger = logging.getLogger('oriented_rcnn_trial_%d' % rank)
    logger.setLevel(logging.INFO); logger.addHandler(logging.StreamHandler())
    runner = EpochBasedRunner(wrapped, optimizer=optimizer, work_dir=str(out), logger=logger, max_epochs=1)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config, checkpoint_config=None, log_config=None)
    class Observe(Hook):
        def after_train_iter(self, runner):
            if runner.iter >= 2 or not math.isfinite(float(runner.outputs['loss'])):
                raise ValueError('Smoke bound/nonfinite loss')
            observations.append(dict(iteration=int(runner.iter), loss=float(runner.outputs['loss']),
                                     terms=runner.outputs['log_vars'], updates=updates[0]))
            write_new(out/('rank_%d_batch_%d.json' % (rank, runner.iter)), observations[-1])
    runner.register_hook(Observe(), priority='LOWEST')
    initial = model.roi_head.bbox_head.fc_reg.weight.detach().clone()
    runner.run([loader], [('train', 1)])
    changed = not torch.equal(initial, model.roi_head.bbox_head.fc_reg.weight)
    if updates[0] != 2 or len(observations) != 2 or not changed:
        raise ValueError('Two native updates and regression change required')
    optimizer.zero_grad(set_to_none=True)
    inference_cfg = copy.deepcopy(cfg.data.train[rank])
    inference_cfg.update(pipeline=cfg.test_pipeline, test_mode=True)
    inference_dataset = build_dataset(inference_cfg)
    item = scatter(collate([inference_dataset[0]], samples_per_gpu=1), [rank])[0]
    if list(item['img'][0].shape) != [1, 3, 1024, 1024]:
        raise ValueError('Unexpected actual input shape')
    model.eval()
    with torch.no_grad():
        boxes = np.asarray(model(return_loss=False, rescale=True, **item)[0][0])
    if boxes.ndim != 2 or boxes.shape[1] != 6 or not np.isfinite(boxes).all():
        raise ValueError('Native TRAIN inference invalid')
    return dict(passed=True, disposable_updates=updates[0], regression_weight_changed=changed,
        microbatches=observations, train_indices=indices, native_TRAIN_boxes=len(boxes),
        input_shape=[1, 3, 1024, 1024], parameters_discarded=True)


def worker(folder, smoke=False):
    protocol = io.read_json(folder/'protocol.json')
    assert_sources(protocol)
    rank = int(os.environ['LOCAL_RANK'])
    write_new(folder/('rank_%d_identity.json' % rank), dict(pid=os.getpid(), rank=rank, physical_gpu=GPUS[rank]))
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(protocol['allocator_mib']*2**20/
        torch.cuda.get_device_properties(rank).total_memory, rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats(rank)
    try:
        if smoke:
            result = smoke_worker(rank, folder, torch)
            torch.cuda.synchronize(rank)
            result.update(rank=rank, physical_gpu=GPUS[rank],
                peak_allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20)
            write_new(folder/('rank_%d_result.json' % rank), result)
            dist.barrier()
        else:
            sys.argv = ['tools/train.py', str(ROOT/CONFIG), '--launcher', 'pytorch',
                        '--seed', '0', '--deterministic', '--work-dir', str(folder/'train')]
            runpy.run_path(str(ROOT/'tools/train.py'), run_name='__main__')
    except BaseException as error:
        io.atomic_json(folder/('rank_%d_failure.json' % rank), dict(error=repr(error),
            traceback=traceback.format_exc(), rank=rank,
            peak_allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'smoke', 'smoke-worker', 'launch', 'queue', 'worker'), required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--smoke', type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.stage == 'check':
        cfg = runtime_check()
        print(json.dumps(dict(passed=True, config=CONFIG, sources=len(source_identity()),
            optimizer=cfg['optimizer'], lr_config=cfg['lr_config'], epochs=12, candidate_epochs=EPOCHS), indent=2))
        return
    if args.out is None:
        parser.error('--out is required')
    out = args.out.resolve()
    if args.stage in ('worker', 'smoke-worker'):
        return worker(out, args.stage == 'smoke-worker')
    if args.stage == 'queue':
        return 0 if guarded_run(out, io.read_json(out/'protocol.json')) else 1
    if args.stage == 'smoke':
        protocol = prepare(out)
        return 0 if guarded_run(out, protocol, smoke=True) else 1
    if args.smoke is None:
        parser.error('Formal launch requires --smoke pointing to a passed trial')
    prepare(out, args.smoke.resolve())
    with (out/'supervisor.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--stage', 'queue', '--out', str(out)],
            cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(model=MODEL, supervisor_pid=child.pid, physical_gpus=list(GPUS),
                   out=str(out), epochs=12, auto_TEST=False)
    write_new(out/'launch.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    sys.exit(main())
