#!/usr/bin/env python3
"""Post-training protocol amendment: complete VAL for all 48 saved heads.

No optimizer, backward, TRAIN tensors, TEST, detector or feature extraction.
Historical runner, artifacts and TRAIN gates remain immutable evidence.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import run_port_geometry_size_distribution_v1 as old

sealed, metrics = old.sealed, old.metrics
VERSION = 'port_geometry_size_distribution_val_v2'


def protocol():
    return dict(protocol=VERSION, amendment_utc='2026-10-16T13:06:19+00:00',
        timing='After both arms completed training; before supplementary fitted VAL inference.',
        evaluation='All 24 saved epochs of each original arm, complete 887 VAL frames each. TRAIN pass is diagnostic only; no filtering.',
        representative='Per arm all 24 epochs: Real long-edge MAE, then short-edge MAE, then earliest epoch. Descriptive selection, no automatic adoption.',
        adoption='Separately report unchanged original VAL-only checks and best passing epoch by same key; TRAIN never enters either selection. No post-result relaxation.',
        comparison='B and M replay from original SHA-indexed baseline rows, matched to immutable VAL cache and selected M references. Same metric_protocol_v2 and cached-GT AP11.',
        limits=dict(training_updates=0,optimizer_restored=False,train_tensor_access=False,
            detector_loaded=False,feature_extractions=0,test_access=False,depth_access=False,
            automatic_promotion=False),
        failure='Engineering identity, counts, coordinates, nonfinite or immutable-input drift stops evaluation; size diagnostic failures never skip VAL.')


def choose(results):
    if len(results) != 24 or sorted(x['epoch'] for x in results) != list(range(1, 25)):
        raise ValueError('Exactly all 24 epochs required, independent of TRAIN diagnostics')
    representative = min(results, key=lambda x: x['selection_key'])
    admissible = [x for x in results if x['legacy_val_only']['val_pass']]
    return dict(representative=representative,
        legacy_val_only_selected=min(admissible, key=lambda x: x['selection_key']) if admissible else None,
        legacy_val_only_passing_epochs=[x['epoch'] for x in admissible], epochs=results)


def check_checkpoint_metadata(payload, metadata, arm, epoch, proof):
    expected = dict(protocol=old.VERSION, arm=arm, epoch=epoch, updates=epoch*639, proof=proof)
    if any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError('Checkpoint identity/arm/epoch/proof differs')
    if payload['head_digest'] != metadata['checkpoint']['head_digest']:
        raise ValueError('Checkpoint digest metadata differs')


def checked_file(run, index, name):
    if name not in index or sealed.sha(run/name) != index[name]:
        raise ValueError('Original artifact SHA differs: '+name)
    return run/name


def compare_numeric(a, b, path=''):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError('Replay keys differ: '+path)
        for k in a:
            compare_numeric(a[k], b[k], path+'/'+str(k))
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError('Replay length differs: '+path)
        for i, (x, y) in enumerate(zip(a, b)):
            compare_numeric(x, y, path+'/'+str(i))
    elif isinstance(a, float):
        if not math.isfinite(a) or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Replay number differs: '+path)
    elif a != b:
        raise ValueError('Replay value differs: '+path)


def load_val(directory, cache, baseline, references, torch, model):
    path = directory/'val_s1.pt'
    if sealed.sha(path) != cache['files'][path.name]:
        raise ValueError('Immutable VAL shard SHA differs')
    payload = sealed.torch_load(torch, path)
    records = payload['records']
    if (payload['identity'] != cache['identity'] or payload['detector_state'] != cache['detector_state'] or
        len(records) != 887 or len({r['image'] for r in records}) != 887 or
        Counter(r['sequence'] for r in records) != Counter(sealed.VAL_COUNTS)):
        raise ValueError('VAL cache identity/count differs')
    base = {r['image']: r for r in baseline}
    if len(base) != 887 or set(base) != {r['image'] for r in records}:
        raise ValueError('Baseline VAL identities differ')
    prepared = []
    for r in records:
        sealed.checked_record(r, 'val_s1', torch, model.residual.original)
        saved = base[r['image']]
        present = bool(len(r['boxes_original']))
        for key in ('image', 'sequence', 'domain', 'frame_id', 'scale', 'split'):
            if r[key] != saved[key]:
                raise ValueError('Cached baseline frame differs: '+key)
        if (r['gt_original'][0].tolist() != saved['gt'] or
            (r['boxes_original'][0].tolist() if present else None) != saved['b'] or
            (saved['midpoint'] is not None) != present):
            raise ValueError('B/GT/output cache replay differs')
        source = dict(saved, diagnosis=dict(baseline_joint10=False, baseline_center_hit=False))
        if present:
            source['diagnosis'] = dict(baseline_joint10=max(abs(x/y-1) for x,y in
                zip(sorted(saved['midpoint'][2:4]), sorted(saved['gt'][2:4]))) <= .1,
                baseline_center_hit=math.hypot(saved['midpoint'][0]-saved['gt'][0], saved['midpoint'][1]-saved['gt'][1]) < 15.)
        sealed.compare_reference(source, references[r['image']])
        m = torch.tensor([saved['midpoint']], dtype=torch.float32) if present else torch.empty((0,6), dtype=torch.float32)
        prepared.append(dict(r, shard='val_s1', midpoint_original=m))
    return sorted(prepared, key=lambda r:r['image'])


def run(args):
    start = time.monotonic()
    source = old.checked_sources()
    run_dir, out = Path(args.run_dir).resolve(), Path(args.out_dir).resolve()
    if out.parent != (ROOT/'work_dirs'/old.VERSION).resolve() or out.exists() or out == run_dir:
        raise ValueError('New sibling output directory required; refuse overwrite')
    index = sealed.read(run_dir/'artifacts.json')
    if index['protocol'] != old.VERSION:
        raise ValueError('Wrong historical run')
    files = index['files']
    proof = sealed.read(checked_file(run_dir, files, 'input_runtime_proof.json'))
    if proof['source_identity'] != source:
        raise ValueError('Historical executed source identity differs')
    original_completion = sealed.read(checked_file(run_dir, files, 'completion.json'))
    if original_completion['head_updates'] != 30672:
        raise ValueError('Expected completed fixed-budget training')
    out.mkdir()
    report = dict(protocol=VERSION, started_utc=datetime.now(timezone.utc).isoformat(), status='FAILED',
        **protocol()['limits'])
    try:
        sealed.write(out/'protocol.json', protocol())
        import torch
        if (not torch.cuda.is_available() or not os.environ.get('CUDA_VISIBLE_DEVICES') or
            len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) != 1 or os.environ.get('WORLD_SIZE','1') != '1'):
            raise ValueError('Explicit single GPU required')
        torch.set_num_threads(1); old.seed_all(torch)
        device = torch.device('cuda:0'); torch.cuda.set_device(device)
        cache_dir, head_dir, b_weight = [Path(p).resolve() for p in (args.cache_dir,args.head_dir,args.b_weight)]
        historical, differences = old.historical_identity()
        cache = sealed.checked_cache(cache_dir, historical)
        selection, references, selected_files = sealed.checked_selection(head_dir, historical)
        if sealed.sha(b_weight) != old.B_SHA:
            raise ValueError('Fixed B weight SHA differs')
        baseline = old.read_rows(checked_file(run_dir, files, 'baseline_val.rows.jsonl.gz'))
        model = old.head_module()
        records = load_val(cache_dir, cache, baseline, references, torch, model)
        packed = old.pack(records, device, torch)
        if len(records) != 887 or len(packed['indices']) != 886:
            raise ValueError('Full VAL denominator/output differs')
        riou_cache = {}
        baseline_summary = metrics.summary(baseline, riou_cache)
        compare_numeric(baseline_summary, sealed.read(checked_file(run_dir, files, 'baseline_val_summary.json')))
        sealed.write(out/'baseline_val_summary.json', baseline_summary)
        runtime = dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),
            gpu=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
        provenance = dict(historical_run=str(run_dir),original_artifacts_sha256=sealed.sha(run_dir/'artifacts.json'),
            original_proof=proof,original_source_identity=source,runtime=runtime,
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),
            evaluator_sha256=sealed.sha(Path(__file__)),cache_manifest_sha256=sealed.CACHE_SHA,
            selected_m_artifacts=selected_files,unused_detector_historical_differences=differences,
            baseline_summary_replay=True,baseline_m_reused=True)
        sealed.write(out/'input_runtime_proof.json', provenance)
        arms = {}
        for arm in old.ARMS:
            dest = out/arm; dest.mkdir(); results=[]
            for epoch in range(1,25):
                relative = arm+'/head_epoch_%02d.pth'%epoch
                path = checked_file(run_dir, files, relative)
                metadata = sealed.read(checked_file(run_dir, files, arm+'/epoch_%02d.json'%epoch))
                if sealed.sha(path) != metadata['checkpoint']['sha256']:
                    raise ValueError('Epoch/checkpoint SHA differs')
                payload = sealed.torch_load(torch,path)
                check_checkpoint_metadata(payload, metadata, arm, epoch, proof)
                head = model.DistributionSizeHead().to(device)
                head.load_state_dict(payload['head_state'], strict=True)
                head.eval()
                for p in head.parameters(): p.requires_grad_(False)
                before = sealed.state_digest(head)
                if before != payload['head_digest']:
                    raise ValueError('Reloaded checkpoint digest differs')
                rows = old.evaluate_rows(head,records,packed,torch)
                if sealed.state_digest(head) != before or any(p.grad is not None for p in head.parameters()):
                    raise ValueError('Evaluation updated head')
                value = metrics.summary(rows,riou_cache)
                old.write_rows(dest/('val_epoch_%02d.rows.jsonl.gz'%epoch),rows)
                sealed.write(dest/('val_epoch_%02d.summary.json'%epoch),value)
                result = dict(arm=arm,epoch=epoch,checkpoint_sha256=files[relative],
                    train_pass_diagnostic_only=metadata['train_pass'],
                    train_failed_checks_diagnostic_only=metadata['train_failed_checks'],
                    legacy_val_only=metrics.val_gate(value),selection_key=list(metrics.selection_key(value,epoch)))
                sealed.write(dest/('epoch_%02d.json'%epoch),result);results.append(result)
                print('VAL_COMPLETE',arm,epoch,'key',result['selection_key'],
                    'legacy_val_only',result['legacy_val_only']['val_pass'],flush=True)
            arms[arm]=choose(results);sealed.write(dest/'selection.json',arms[arm])
        # Recheck every original SHA-indexed artifact, including all TRAIN
        # diagnostics and weights; no source or historical result is rewritten.
        for name in files: checked_file(run_dir,files,name)
        if (old.checked_sources()!=source or sealed.sha(b_weight)!=old.B_SHA or
            sealed.sha(head_dir/sealed.FIXED['path'])!=sealed.FIXED['sha256'] or
            sealed.sha(cache_dir/'val_s1.pt')!=cache['files']['val_s1.pt']):
            raise ValueError('Immutable input drift')
        report.update(status='COMPLETE_ALL_48_VAL_REVIEW_REQUIRED',arms=arms,
            fitted_val_evaluations=48,val_frames_per_epoch=887,original_indexed_files_unchanged=len(files),
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(0))
    except BaseException as exc:
        report['error']=repr(exc)
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        sealed.write(out/'completion.json',report)
        sealed.write(out/'artifacts.json',dict(protocol=VERSION,files={str(p.relative_to(out)):sealed.sha(p)
            for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifacts.json'}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','out-dir','cache-dir','head-dir','b-weight'):
        parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
