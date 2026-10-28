#!/usr/bin/env python3
"""Pair-specific LVSR preparation only: check or four discarded GPU updates.

There is deliberately no formal training/TEST/depth stage in this version.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_symeood_110_lvsr_v1 as base
from crane_project.tools.run_port_size_native_train_v1 import (
    sha, read, write, seed, rng_get, rng_set, equal)

COMMON = ROOT/'work_dirs/symeood_110_retrain_seed0_20261026_v1/downstream/lvsr'
CONTRACT = ROOT/'crane_project/data_contracts/symeood_110_lvsr_pair_v1.json'
OUT = COMMON/'pair_specific'
KEYS = ('roi', 'support', 'boxes_original', 'boxes_model', 'scale_xy',
        'midpoint_original', 'gt_original')


def checked_sources():
    proof = read(CONTRACT)
    design = ROOT/proof['design']['path']
    if sha(design) != proof['design']['sha256']:
        raise ValueError('Original design differs')
    d = read(design)
    if (d['architecture']['total_parameters'] != 2480 or
            d['detector_sha256'] != base.DETECTOR_SHA or
            d['midpoint_sha256'] != base.MIDPOINT_SHA or
            proof['scope']['formal_training_authorized']):
        raise ValueError('Limited scope or fixed identity differs')
    for name, digest in {**d['source_hashes'], **proof['sources'], **proof['input_files']}.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Source/input identity differs: '+name)
    identity = read(COMMON/'identity.json')['inputs']
    if identity['detector_sha256'] != d['detector_sha256'] or identity['midpoint_sha256'] != d['midpoint_sha256']:
        raise ValueError('Saved M/input frontend identity differs')
    return proof


def new_head(torch, device):
    from crane_project.utils.port_size_pair_specific_v1 import PairSpecificSizeHead
    return PairSpecificSizeHead().to(device)


def bounded_inputs(proof, torch):
    """Deserialize sealed shards, retain ONLY 16 TRAIN and four VAL views.

    M21 is taken from existing native rows, never the ep7 learned size output.
    No parent head is loaded. Selection sees domain/scale/presence/ID only.
    """
    saved, selected = {}, {}
    for split in ('train', 'val'):
        with gzip.open(COMMON/('fixed_b/epoch_07_'+split+'.rows.jsonl.gz'), 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        present = sorted([r for r in rows if r['midpoint'] is not None], key=lambda r: (r['scale'], r['image']))
        saved[split] = {(r['image'], r['scale']): r for r in rows}
        selected[split] = [r for domain in ('real', 'sim')
            for scale in ((.5, 1.) if split == 'train' else (1.,))
            for r in [x for x in present if x['domain'] == domain and x['scale'] == scale][:(4 if split == 'train' else 2)]]
    assert len(selected['train']) == 16 and len(selected['val']) == 4
    manifest = read(base.FORMAL/'roi_cache/cache_manifest.json')
    found = {s: {} for s in selected}
    for name in ('train_s1.pt', 'train_s05.pt', 'val_s1.pt'):
        path = base.FORMAL/'roi_cache'/name
        if sha(path) != manifest['files'][name]:
            raise ValueError('Sealed cache shard differs: '+name)
        payload = torch.load(str(path), map_location='cpu')
        if payload['identity'] != manifest['identity'] or payload['detector_state'] != manifest['detector_state']:
            raise ValueError('Cache provenance differs')
        split = 'val' if name.startswith('val') else 'train'
        wanted = {(r['image'], r['scale']) for r in selected[split]}
        for r in payload['records']:
            k = r['image'], r['scale']
            if k not in wanted:
                continue
            v = saved[split][k]
            if any(r[t] != v[t] for t in ('image', 'scale', 'domain', 'sequence', 'frame_id', 'split')):
                raise ValueError('Selected metadata differs')
            if r['boxes_original'][0].tolist() != v['b'] or r['gt_original'][0].tolist() != v['gt']:
                raise ValueError('Selected B/GT differs')
            r = dict(r, midpoint_original=r['boxes_original'].new_tensor([v['midpoint']]), shard=v['shard'])
            if any(r[t].requires_grad or r[t].grad_fn is not None for t in KEYS):
                raise ValueError('Frozen input must be detached')
            found[split][k] = r
        del payload
    result = {}
    for split, rows in selected.items():
        if set(found[split]) != {(r['image'], r['scale']) for r in rows}:
            raise ValueError('Missing bounded view')
        result[split] = [found[split][r['image'], r['scale']] for r in rows]
    return result


def input_digest(rows):
    import hashlib
    h = hashlib.sha256()
    for r in rows:
        for k in KEYS:
            t = r[k].detach().cpu().contiguous()
            h.update(str((r['image'], r['scale'], k, tuple(t.shape), t.dtype)).encode())
            h.update(t.numpy().tobytes())
    return h.hexdigest()


def sample_ids(rows):
    return [{k: r[k] for k in ('image', 'domain', 'scale', 'sequence', 'split', 'shard')} for r in rows]


def checked_update(h, opt, rows, torch):
    begin = time.monotonic()
    stats = base.update(h, opt, base.batch(rows, torch), 'fixed_b', torch)
    stats['branch_gradients'] = []
    for stem, terminal in zip(h.stem.branches, h.terminal.branches):
        norm = lambda module: sum(float(p.grad.detach().double().square().sum()) for p in module.parameters())**.5
        stats['branch_gradients'].append(dict(stem=norm(stem), terminal=norm(terminal)))
    if any(x['terminal'] <= 0 for x in stats['branch_gradients']):
        raise ValueError('A pair branch did not receive terminal gradients')
    stats['optimizer_steps'] = sorted(set(int(s['step']) for s in opt.state.values()))
    stats['branch_gradient_stage'] = 'after_original_clip10'
    stats['clip_triggered'] = stats['grad_norm'] > 10.
    stats['seconds'] = time.monotonic()-begin
    return stats


def smoke(proof, torch):
    from crane_project.tools import train_symeood_110_midpoint_v1 as f
    f.runtime_imports()
    runtime = f.runtime(0)
    previous = read(base.FORMAL/'roi_cache/cache_manifest.json')['runtime']
    if runtime != previous:
        raise ValueError('Runtime differs from formal cache: '+repr(dict(actual=runtime, previous=previous)))
    for name, digest in proof['weight_files'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Fixed frontend weight bytes differ: '+name)
    # This small-head-only trial caps its allocator below the original ceiling.
    torch.cuda.set_per_process_memory_fraction(2048*2**20/torch.cuda.get_device_properties(0).total_memory, 0)
    inputs = bounded_inputs(proof, torch)
    allrows = inputs['train']+inputs['val']
    frozen_digest = input_digest(allrows)
    seed(torch)
    h = new_head(torch, 'cuda:0')
    initial = f.g.state_digest(h)
    opt = torch.optim.Adam(h.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    assert not opt.state
    h.eval()
    with torch.no_grad():
        for rr in (inputs['train'][:8], inputs['train'][8:], inputs['val']):
            x = base.batch(rr, torch)
            out = h(*x[:-1])
            if not torch.equal(out['boxes_original'], x[-2]):
                raise ValueError('Complete neutral output differs from M21')
    logs = [checked_update(h, opt, inputs['train'][:8], torch),
            checked_update(h, opt, inputs['train'][8:], torch)]
    if any(v['stem'] != 0 for v in logs[0]['branch_gradients']):
        raise ValueError('Zero-terminal initial stem gradients differ')
    if any(v['stem'] <= 0 for v in logs[1]['branch_gradients']):
        raise ValueError('A spatial branch did not receive learned gradients')
    saved = dict(head=deepcopy(h.state_dict()), opt=deepcopy(opt.state_dict()), rng=rng_get(torch), updates=2)
    stream = io.BytesIO()
    torch.save(saved, stream)
    stream.seek(0)
    got = torch.load(stream, map_location='cpu')
    if not equal(saved, got, torch):
        raise ValueError('Serialization readback differs')
    clone = new_head(torch, 'cuda:0')
    clone.load_state_dict(got['head'], strict=True)
    other = torch.optim.Adam(clone.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    other.load_state_dict(got['opt'])
    rng_set(saved['rng'], torch)
    a = checked_update(h, opt, inputs['train'][:8], torch)
    after = rng_get(torch)
    rng_set(got['rng'], torch)
    b = checked_update(clone, other, inputs['train'][:8], torch)
    # Wall-clock times are observational, not part of numerical restoration.
    if (not equal({k:v for k,v in a.items() if k!='seconds'}, {k:v for k,v in b.items() if k!='seconds'}, torch)
            or not equal(after, rng_get(torch), torch) or not equal(h.state_dict(), clone.state_dict(), torch)
            or not equal(opt.state_dict(), other.state_dict(), torch)):
        raise ValueError('Resumed next update differs')
    logs += [dict(a, trajectory='original_step3'), dict(b, trajectory='restored_step3')]
    evaluated = base.evaluate(h, inputs['val'], torch)
    if input_digest(allrows) != frozen_digest or any(r[k].grad is not None for r in allrows for k in KEYS):
        raise ValueError('Frozen input changed')
    if any(x['optimizer_steps'] != [i] for x, i in zip(logs, (1, 2, 3, 3))):
        raise ValueError('Optimizer counter differs')
    resource = f.resource_check(0)
    report = dict(passed=True, scope='engineering_only_not_performance',
        runtime=runtime, physical_gpu=0, gpu_uuid=torch.cuda.get_device_properties(0).uuid if hasattr(torch.cuda.get_device_properties(0), 'uuid') else None,
        parameters=sum(p.numel() for p in h.parameters()), baseline_parameters=2800,
        initial_digest=initial, final_temporary_digest=f.g.state_digest(h),
        actual_optimizer_updates=4, main_trajectory_steps=3, restored_trajectory_steps=3,
        unique_train_views=16, unique_val_views=4, train_samples=sample_ids(inputs['train']), val_samples=sample_ids(inputs['val']),
        neutral_output_exact_views=20, model_Adam_rng_next_update_exact=True,
        state_discarded=True, formal_initialization_reusable=False, formal_training_started=False,
        optimizer_updates=logs, val_rows=evaluated, frozen_input_digest=frozen_digest,
        input_tensors_unchanged=True, detector_forwards=0, midpoint_forwards=0,
        detector_updates=0, midpoint_updates=0, feature_extractions=0,
        entire_VAL_replayed=False, TEST=False, depth=False, resource=resource,
        contract_sha256=sha(CONTRACT), design_sha256=proof['design']['sha256'],
        sources=proof['sources'], input_files=proof['input_files'])
    del h, opt, clone, other, saved, got
    torch.cuda.empty_cache()
    return report


def main(args):
    proof = checked_sources()
    if args.stage == 'check':
        print(json.dumps(dict(passed=True, stage='static_check', updates=0, formal_training_started=False)))
        return
    if not args.run or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('WORLD_SIZE', '1') != '1':
        raise ValueError('Smoke needs --run, single CUDA_VISIBLE_DEVICES=0')
    OUT.mkdir(exist_ok=True)
    target = OUT/'smoke.json'
    if target.exists() or (OUT/'smoke_start.json').exists():
        raise ValueError('Existing trial preserved; no automatic retry or overwrite')
    write(OUT/'smoke_start.json', dict(time=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        contract_sha256=sha(CONTRACT), physical_gpu=0, formal_training_started=False))
    try:
        import torch
        torch.set_num_threads(1)
        result = smoke(proof, torch)
        # Recheck the immutable source and selected input-file identities.
        checked_sources()
        result['completed_time'] = datetime.now(timezone.utc).isoformat()
        write(target, result)
        print(json.dumps(dict(passed=True, actual_optimizer_updates=4, formal_training_started=False,
            receipt=str(target), resource=result['resource'])), flush=True)
    except Exception as exc:
        write(OUT/'smoke_failure.json', dict(error=repr(exc), time=datetime.now(timezone.utc).isoformat(), formal_training_started=False))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('check', 'smoke'), default='check')
    p.add_argument('--run', action='store_true')
    return p


if __name__ == '__main__':
    main(parser().parse_args())
