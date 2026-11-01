"""Formal frozen pipeline: check, predict one image, or verify 8 saved VAL rows.

Default check is stdlib-only. No training, TEST sweep, epoch selection, old
trainer patching, or source modifications. Runtime requires an explicit GPU.
"""
import argparse
from datetime import datetime, timedelta, timezone
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import crane_frozen_pipeline_v1 as p


def read_rows(spec):
    path = p.resolve_checkpoint(spec)  # same immutable file/mirror rule
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return {r['image']: r for r in map(json.loads, stream)}


def compare(actual, midpoint_ref, size_ref, atol=1e-4, rtol=1e-6):
    import numpy as np
    maxima = {}
    for key in ('b', 'midpoint', 'edge_residual'):
        a, b = actual[key], size_ref[key]
        if (a is None) != (b is None): raise ValueError('Output presence differs: ' + key)
        if a is not None and (a[5] != b[5] or not np.allclose(a[:5], b[:5], atol=atol, rtol=rtol)):
            raise ValueError('Saved VAL output differs: ' + key)
        maxima[key] = max(abs(x-y) for x, y in zip(a[:5], b[:5])) if a is not None else 0.
    if actual['accepted'] != midpoint_ref['accepted'] or actual['failed_checks'] != midpoint_ref['failed_checks']:
        raise ValueError('Midpoint fallback decision differs')
    if (actual['size_delivery'] != size_ref['size_delivery'] or
            actual['vector_valid'] != size_ref['vector_valid'] or
            actual['size_failed_checks'] != size_ref['failed_checks']):
        raise ValueError('LVSR fallback decision differs')
    return maxima


def memory(gpu):
    text = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
    used, free = map(int, text.strip().split(','))
    return dict(used_mib=used, free_mib=free)


def run(args):
    inspection = p.inspect(args.config)
    if args.stage == 'check': return inspection
    if args.gpu is None or args.out_dir is None:
        raise ValueError('Runtime requires explicit --gpu and new --out-dir')
    if args.out_dir.exists(): raise FileExistsError('Never overwrite a run directory')
    cfg, _ = p.read_config(args.config)
    resource = cfg['memory']
    if memory(args.gpu)['free_mib'] < resource['startup_free_mib']:
        raise RuntimeError('Insufficient free memory; stop without retry')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch
    import numpy as np
    torch.set_num_threads(1)
    random.seed(1703); np.random.seed(1703); torch.manual_seed(1703); torch.cuda.manual_seed_all(1703)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    limit = resource['allocator_mib'] * 2**20 / torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(limit, 0)
    original = torch.optim.Optimizer.__init__
    def forbidden(*a, **kw): raise AssertionError('No optimizer in frozen inference')
    torch.optim.Optimizer.__init__ = forbidden
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = dict(stage=args.stage, config=args.config, identity=inspection['identity'],
        captured_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        training_updates=0, TEST=False, reselection=False, physical_gpu=args.gpu,
        core_sources=cfg['source_sha256'])
    try:
        model = p.FrozenCranePipeline(args.config)
        # The historical R4 constructor sets 6 GiB; reapply the tighter limit.
        torch.cuda.set_per_process_memory_fraction(limit, 0)
        import cv2, mmcv, mmdet, mmrotate
        runtime = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmrotate=mmrotate.__version__,
                       opencv=cv2.__version__, gpu=torch.cuda.get_device_name(0))
        if runtime != cfg['runtime']:
            raise ValueError('Runtime differs from frozen training/inference: ' + str(runtime))
        before = model.states()
        if args.stage == 'verify-val':
            mrefs, lrefs = (read_rows(cfg['validation'][k]) for k in ('midpoint_rows', 'lvsr_rows'))
            names = cfg['validation']['images']
            if len(names) != 8 or len(set(names)) != 8:
                raise ValueError('Exactly the eight fixed VAL samples required')
            manifest = json.loads(p.local_path(cfg['validation']['data_root']).joinpath('manifest.json').read_text())
            records = {r['id']: r for r in manifest['records'] if r['split'] == 'val'}
            jobs = []
            for name in names:
                record = records[name]
                image = p.local_path(cfg['validation']['data_root']) / record['images']['path']
                if p.sha(image) != record['images']['sha256']:
                    raise ValueError('VAL image identity differs')
                jobs.append((name, image))
        else:
            if args.image is None: raise ValueError('predict requires --image')
            jobs = [(args.image.stem, args.image)]
        rows, resources = [], []
        for name, image_path in jobs:
            image, metas = model.load_image(image_path)
            prediction = model.predict(image, metas)
            row = dict(image=name, **prediction)
            if args.stage == 'verify-val':
                row['max_box_difference'] = compare(prediction, mrefs[name], lrefs[name])
            rows.append(row)
            observation = memory(args.gpu)
            if observation['free_mib'] < resource['minimum_free_mib']:
                raise RuntimeError('Free memory limit; stop without retry')
            apps = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
                '--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
            own = sum(int(line.split(',')[1]) for line in apps.splitlines()
                      if line.split(',')[0].strip() == str(os.getpid()))
            if own >= resource['process_stop_mib']:
                raise RuntimeError('Process memory limit; stop without retry')
            observation['own_process_mib'] = own
            resources.append(observation)
        after = model.states()
        if after != before: raise ValueError('Frozen model state changed')
        if model.counts != dict(feature_extractions=len(jobs),native_head_calls=3*len(jobs),
                               midpoint_calls=len(jobs),lvsr_calls=len(jobs)):
            raise ValueError('Pipeline call budget differs')
        if p.inspect(args.config) != inspection:
            raise ValueError('Source or weight changed during inference')
        report.update(passed=True, frames=len(rows), runtime=runtime, counters=model.counts,
            state_before=before,state_after=after, resources=resources,
            allocator_peak_mib=torch.cuda.max_memory_allocated(0)/2**20)
        with (args.out_dir/'predictions.jsonl').open('x') as f:
            for row in rows: f.write(json.dumps(row, allow_nan=False)+'\n')
        with (args.out_dir/'completion.json').open('x') as f:
            json.dump(report, f, indent=2, allow_nan=False)
        return report
    except BaseException as error:
        with (args.out_dir/'failure.json').open('x') as f:
            json.dump(dict(report, passed=False, error=repr(error)), f, indent=2)
        raise
    finally:
        torch.optim.Optimizer.__init__ = original


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=p.DEFAULT_CONFIG)
    ap.add_argument('--stage', choices=('check', 'predict', 'verify-val'), default='check')
    ap.add_argument('--gpu', type=int, choices=(0,1,2,3))
    ap.add_argument('--image', type=Path)
    ap.add_argument('--out-dir', type=Path)
    return ap


if __name__ == '__main__':
    print(json.dumps(run(parser().parse_args()), indent=2))
