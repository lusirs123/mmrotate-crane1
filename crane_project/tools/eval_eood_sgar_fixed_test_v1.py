"""One authorized TEST of the original VAL-selected EOOD+SGAR epoch24.

No checkpoint scan, training, changed threshold, downstream head or retry.
Uses a separate protocol so the completed TRAIN/VAL source lock is unchanged.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import runpy
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils.eood_sgar_protocol_v1 import read, save, sha, verify

CHECKPOINT_SHA = '9890f389a031eb6ed00439b32e5a4efa63b5eee11c27f451b69a9faf238d1b6b'
REFERENCE = 'work_dirs/smallpaper_readiness_20261028_v1/saved_predictions_unified.rows.jsonl'
REFERENCE_SHA = '1abae6afe6ece14c356c7b5608a5ceb692316172605d970444c41a2a12fa8481'
R4_ROWS = 'work_dirs/symeood_110_reg_tower_20261030_v1/fixed_TEST_20261030_v1/analysis/per_frame.jsonl'
ANNOTATIONS_SHA = 'e0dbb1bd8aea7209314d8ed60bc44e965550ed606135cc0016e1075d717de13e'
COUNTS = dict(real_seq03=200, real_seq04=668, sim_seq09=572)
OUT_NAME = 'fixed_TEST_20261101_v1'


def fixed_selection(sweep, selection_config):
    if sweep['selected_checkpoint'] != 'epoch_24':
        raise ValueError('Only original VAL-selected epoch24 is authorized')
    if sweep['selection_config'] != selection_config:
        raise ValueError('Original VAL selection rule differs')
    record = sweep['all_checkpoints']['epoch_24']
    if record['checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('Frozen checkpoint differs')
    return record


def top1(native):
    import numpy as np
    if len(native) != 1:
        raise ValueError('Single-class predictions required')
    boxes = np.asarray(native[0])
    if boxes.ndim != 2 or boxes.shape[1] != 6 or not np.isfinite(boxes).all():
        raise ValueError('Invalid native OBB predictions')
    return boxes[boxes[:,5].argmax()].tolist() if len(boxes) else None


def annotation_sha(directory):
    h = hashlib.sha256()
    for p in sorted(Path(directory).glob('*.txt')):
        if p.name.startswith('._'):
            continue
        h.update(p.name.encode()); h.update(b'\0'); h.update(p.read_bytes())
    return h.hexdigest()


def memory(gpu):
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
    used, free = map(int, row.strip().split(','))
    return dict(used_mib=used, free_mib=free)


def source_identity(base):
    c = verify(base, launch=True)
    assert read(base / 'run_status.json')['state'] == 'COMPLETE'
    assert read(base / 'training_verification.json')['passed']
    selection = base / 'train/ckpt_sweep/sweep_results.json'
    sweep = read(selection)
    record = fixed_selection(sweep, c['checkpoint_selection']['selection_config'])
    assert record['checkpoint_sha256'] == CHECKPOINT_SHA == sha(record['checkpoint'])
    assert read(base / 'selected_identity.json')['sha256'] == CHECKPOINT_SHA
    return c, selection, record


def prepare(base, gpu):
    from crane_project.tools.ckpt_sweep import dataset_paths, get_val_img_ids
    c, selection, record = source_identity(base)
    out = base / OUT_NAME
    assert not out.exists(), 'Never overwrite a fixed TEST attempt'
    ann, images = dataset_paths(str(ROOT / c['config']), 'test')
    assert annotation_sha(ann) == ANNOTATIONS_SHA
    assert sha(ROOT / REFERENCE) == REFERENCE_SHA
    refs = [json.loads(s) for s in (ROOT / REFERENCE).read_text().splitlines()]
    assert len(refs) == 1440 and Counter(r['sequence'] for r in refs) == Counter(COUNTS)
    assert all(r['split'] == 'test' for r in refs)
    assert get_val_img_ids(ann) == [r['image'] for r in refs]
    r4_path = ROOT / R4_ROWS
    old = [json.loads(s) for s in r4_path.read_text().splitlines()]
    r4 = [r for r in old if r['side'] == 'R4']
    assert [r['image'] for r in r4] == [r['image'] for r in refs]
    assert all(a['gt'] == b['gt'] for a, b in zip(r4, refs))
    data = ROOT / 'crane_project/data/crane_grab_port_day2night_v1'
    manifest = read(data / 'manifest.json')
    test = [r for r in manifest['records'] if r['split'] == 'test']
    assert {r['id'] for r in test} == {r['image'] for r in refs}
    inventory = []
    for row in test:
        for key in ('images', 'annfiles'):
            path = data / row[key]['path']
            assert sha(path) == row[key]['sha256'], str(path)
            inventory.append(dict(path=str(path.relative_to(ROOT)), sha256=row[key]['sha256']))
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id',
                                      '--format=csv,noheader'], text=True)
    ordered = sorted([tuple(x.strip() for x in s.split(',')) for s in devices.splitlines()],
                     key=lambda row: row[2])
    ordinal = next(i for i, row in enumerate(ordered) if int(row[0]) == gpu)
    free = memory(gpu); assert free['free_mib'] >= 4096, free
    out.mkdir(exist_ok=False)
    (out / 'preds').mkdir()
    sources = dict(c['sources'])
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    sources['tests/test_eood_sgar_fixed_test_v1.py'] = sha(ROOT / 'tests/test_eood_sgar_fixed_test_v1.py')
    save(out / 'protocol.json', dict(
        protocol='eood_sgar_ep24_fixed_exposed_TEST_v1',
        frozen_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        user_authorized=True, base=str(base), checkpoint=record['checkpoint'],
        checkpoint_sha256=CHECKPOINT_SHA, selected_epoch=24,
        selection_source=str(selection), selection_source_sha256=sha(selection),
        training_protocol_sha256=sha(base / 'protocol.json'),
        config=str(ROOT / c['config']), config_sha256=c['config_sha256'],
        sources=sources, references={REFERENCE: REFERENCE_SHA, R4_ROWS: sha(r4_path),
            'crane_project/data/crane_grab_port_day2night_v1/manifest.json': sha(data / 'manifest.json')},
        annotations=ann, images=images, annotation_sha256=ANNOTATIONS_SHA,
        input_files=inventory, frames=1440, counts=COUNTS,
        physical_gpu=gpu, cuda_ordinal=ordinal, GPU_UUID=ordered[ordinal][1],
        allocator_limit_mib=3072, own_driver_limit_mib=4096,
        minimum_free_during_run_mib=1024, free_before=free,
        seed=0, checkpoint_scan=False, threshold_changes=False,
        training_updates=0, optimizer_construction=False, retry=False,
        exposed_TEST=True, unbiased_holdout_claim=False, automatic_adoption=False))
    print('EOOD_SGAR_FIXED_TEST_PREPARED', flush=True)


def checked(base):
    out = base / OUT_NAME
    p = read(out / 'protocol.json')
    source_identity(base)
    assert sha(p['config']) == p['config_sha256']
    assert sha(p['selection_source']) == p['selection_source_sha256']
    assert sha(base / 'protocol.json') == p['training_protocol_sha256']
    assert annotation_sha(p['annotations']) == p['annotation_sha256']
    for name, digest in {**p['sources'], **p['references']}.items():
        assert sha(ROOT / name) == digest, name
    return out, p


def infer(base):
    out, p = checked(base)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(p['cuda_ordinal'])
    assert not (out / 'preds/results.pkl').exists()
    import torch
    import mmdet.apis
    import random
    import numpy as np
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(0)
    assert torch.cuda.device_count() == 1
    def cap():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(p['allocator_limit_mib'] * 2**20 / total, 0)
    cap()
    torch.cuda.reset_peak_memory_stats(0)
    probe = torch.empty(1, device='cuda:0')
    apps = subprocess.check_output(['nvidia-smi', '-i', str(p['physical_gpu']),
        '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    observed = [s.split(',')[1].strip() for s in apps.splitlines()
                if s.split(',')[0].strip() == str(os.getpid())]
    assert observed == [p['GPU_UUID']]; del probe
    original = mmdet.apis.single_gpu_test
    calls = []
    def digest(model):
        h = hashlib.sha256()
        for name, value in model.state_dict().items():
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def guarded(model, loader, *args, **kwargs):
        cap()
        assert len(loader.dataset) == len(loader) == 1440
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before = digest(model); count = [0]
        def audit(module, inputs):
            assert not torch.is_grad_enabled() and not module.training
            count[0] += 1
        handle = model.register_forward_pre_hook(audit)
        start = time.perf_counter()
        try:
            values = original(model, loader, *args, **kwargs)
        finally:
            handle.remove()
        after = digest(model)
        assert before == after and len(values) == count[0] == 1440
        save(out / 'frozen_model_state.json', dict(passed=True, before=before, after=after,
             parameters_and_buffers_unchanged=True, forward_calls=count[0], frames=1440,
             no_grad=True, training_updates=0, optimizer_construction_forbidden=True,
             peak_allocated_mib=torch.cuda.max_memory_allocated(0)/2**20,
             peak_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20,
             elapsed_seconds=time.perf_counter()-start))
        calls.append(True)
        return values
    def forbidden(*args, **kwargs):
        raise RuntimeError('Optimizer construction forbidden during fixed TEST')
    torch.optim.Optimizer.__init__ = forbidden
    mmdet.apis.single_gpu_test = guarded
    sys.argv = ['tools/test.py', p['config'], p['checkpoint'], '--work-dir', str(out / 'native'),
        '--out', str(out / 'preds/results.pkl'),
        '--result-identity-out', str(out / 'runtime_result_identity.json'),
        '--runtime-audit-out', str(out / 'runtime_audit.json')]
    runpy.run_path(str(ROOT / 'tools/test.py'), run_name='__main__')
    assert calls == [True]
    from crane_project.tools.ckpt_sweep import check_or_record_prediction
    check_or_record_prediction(p['config'], p['checkpoint'], str(out / 'preds/results.pkl'),
                               p['annotations'], 'fixed_test', generated=True)
    assert sha(p['checkpoint']) == CHECKPOINT_SHA


def longest_failure(rows):
    run = maximum = 0; previous = None
    for r in sorted(rows, key=lambda x: (x['sequence'], x['frame_id'])):
        contiguous = previous is not None and previous['sequence'] == r['sequence'] and previous['frame_id']+1 == r['frame_id']
        run = (run+1 if contiguous else 1) if r['riou'] < .5 else 0
        maximum = max(maximum, run); previous = r
    return maximum


def analyze(base):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out, p = checked(base)
    import numpy as np
    from crane_project.tools.eval_port_geometry_ap_v1 import evaluate_images, verify_native, validate_saved_gt
    from crane_project.tools.eval_crane_offline import compute_riou
    from crane_project.tools.analyze_symeood_ablation_first4_v1 import txtbox
    from crane_project.utils.port_geometry_midpoint_edge_residual_metrics_v1 import canonical, row_metrics
    raw_path = out / 'preds/results.pkl'; provenance = read(str(raw_path) + '.provenance.json')
    assert provenance['results_pkl_sha256'] == sha(raw_path)
    assert provenance['checkpoint_sha256'] == CHECKPOINT_SHA
    assert provenance['annotations_sha256'] == ANNOTATIONS_SHA
    frozen = read(out / 'frozen_model_state.json'); assert frozen['passed']
    identity = read(out / 'runtime_result_identity.json')
    assert identity['checkpoint_sha256'] == CHECKPOINT_SHA and identity['results_sha256'] == sha(raw_path)
    refs = [json.loads(s) for s in (ROOT / REFERENCE).read_text().splitlines()]
    assert [r['frame_key'] for r in identity['runtime_dataset_order']] == [r['image'] for r in refs]
    old = [json.loads(s) for s in (ROOT / R4_ROWS).read_text().splitlines()]
    r4 = [r for r in old if r['side'] == 'R4']
    with raw_path.open('rb') as stream:
        predictions = pickle.load(stream)
    assert len(predictions) == len(refs) == 1440
    for r in refs:
        ann = Path(p['annotations']) / (r['image'] + '.txt')
        poly = np.asarray([float(v) for v in ann.read_text().split()[:8]]).reshape(4, 2)
        validate_saved_gt(r['gt'], poly, txtbox(ann))
    metrics, frames, cache, native_checks = [], [], {}, []
    for side in ('EOOD', 'EOOD_plus_SGAR', 'R4'):
        rows = []
        for ref, raw, before in zip(refs, predictions, r4):
            assert before['image'] == ref['image'] and before['gt'] == ref['gt']
            if side == 'EOOD_plus_SGAR':
                pred = top1(raw)
            else:
                pred = ref['eood_aug'] if side == 'EOOD' else before['prediction']
            r = {k: ref[k] for k in ('image', 'sequence', 'domain', 'frame_id', 'gt')}
            riou = compute_riou(np.asarray(canonical(pred)), np.asarray(canonical(ref['gt']))) if pred else 0.
            r.update(side=side, prediction=pred, score=pred[5] if pred else 0.)
            r.update(row_metrics(ref['gt'], pred, riou)); rows.append(r)
        frames.extend(rows); cache[side] = rows
        for domain in ('all', 'real', 'sim', *COUNTS):
            group = [r for r in rows if domain == 'all' or domain in (r['domain'], r['sequence'])]
            valid = [r for r in group if r['output']]; sim = [r for r in valid if r['domain'] == 'sim']
            count = len(valid)
            value = dict(side=side, domain=domain, frames=len(group), outputs=count,
                TP50=sum(r['riou'] >= .5 for r in group), TP75=sum(r['riou'] >= .75 for r in group),
                Recall_pct=100*sum(r['riou'] >= .5 for r in group)/len(group),
                mean_RIoU=sum(r['riou'] for r in group)/len(group),
                center_MAE_px=sum(r['center_error_px'] for r in valid)/count if count else None,
                center_RMSE_px=math.sqrt(sum(r['center_error_px']**2 for r in valid)/count) if count else None,
                Sim_angle_RMSE_deg=math.sqrt(sum(r['angle_error_deg']**2 for r in sim)/len(sim)) if sim else None,
                long_MAE_pct=100*sum(r['long_relative_error'] for r in valid)/count if count else None,
                short_MAE_pct=100*sum(r['short_relative_error'] for r in valid)/count if count else None,
                joint10=sum(r['joint_size10'] for r in group),
                joint10_pct=100*sum(r['joint_size10'] for r in group)/len(group),
                MCML=longest_failure(group))
            dets = [[canonical(r['prediction']) + [r['score']]] if r['output'] else [] for r in group]
            gt = [[canonical(r['gt'])] for r in group]
            for threshold in (.5, .75):
                result = evaluate_images(dets, gt, threshold)
                assert result['tp'] == value['TP50' if threshold == .5 else 'TP75']
                value['AP'+str(round(threshold*100))] = result['ap_percent']
                if side == 'EOOD_plus_SGAR' and domain in ('all', 'real', 'sim'):
                    native_checks.append(dict(domain=domain, threshold=threshold,
                        **verify_native(dets, gt, threshold, result)))
            metrics.append(value)
    paired = []
    for reference in ('EOOD', 'R4'):
        for domain in ('all', 'real', 'sim'):
            pairs = [(a,b) for a,b in zip(cache[reference],cache['EOOD_plus_SGAR'])
                     if domain == 'all' or a['domain'] == domain]
            common = [(a,b) for a,b in pairs if a['output'] and b['output']]
            paired.append(dict(reference=reference, domain=domain, frames=len(pairs), common_outputs=len(common),
                recovered50=sum(a['riou'] < .5 <= b['riou'] for a,b in pairs),
                lost50=sum(b['riou'] < .5 <= a['riou'] for a,b in pairs),
                recovered75=sum(a['riou'] < .75 <= b['riou'] for a,b in pairs),
                lost75=sum(b['riou'] < .75 <= a['riou'] for a,b in pairs),
                recovered_size10=sum(not a['joint_size10'] and b['joint_size10'] for a,b in pairs),
                lost_size10=sum(a['joint_size10'] and not b['joint_size10'] for a,b in pairs),
                common_mean_RIoU_before=sum(a['riou'] for a,b in common)/len(common),
                common_mean_RIoU_after=sum(b['riou'] for a,b in common)/len(common),
                common_center_MAE_before=sum(a['center_error_px'] for a,b in common)/len(common),
                common_center_MAE_after=sum(b['center_error_px'] for a,b in common)/len(common)))
    analysis = out / 'analysis'; analysis.mkdir(exist_ok=False)
    with (analysis / 'metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    with (analysis / 'per_frame.jsonl').open('x') as stream:
        for row in frames:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    save(analysis / 'summary.json', dict(passed=True, TEST=True, exposed_TEST=True,
        selected_epoch=24, checkpoint_sha256=CHECKPOINT_SHA, results_pkl_sha256=sha(raw_path),
        GT_source_sha256=REFERENCE_SHA, frames=1440, training_updates=0,
        AP='Top1 OBB VOC07 eleven-point; not native complete-output AP',
        native_AP_checks=native_checks, metrics=metrics, paired=paired,
        native_boxes=sum(len(r[0]) for r in predictions),
        native_multi_output_frames=sum(len(r[0]) > 1 for r in predictions),
        reselection=False, automatic_adoption=False))
    save(out / 'completion.json', dict(status='FIXED_EOOD_SGAR_EP24_TEST_COMPLETE',
        completed_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        selected_epoch=24, checkpoint_sha256=CHECKPOINT_SHA, frames=1440,
        model_forwards=1440, training_updates=0, frozen_state_unchanged=True,
        runtime_frame_order_verified=True, TEST=True, native_AP_checks=len(native_checks),
        results_pkl_sha256=sha(raw_path), automatic_adoption=False, reselection=False))
    print('FIXED_EOOD_SGAR_EP24_TEST_COMPLETE', flush=True)


def run(base):
    out, p = checked(base)
    save(out / 'run_claim.json', dict(pid=os.getpid(), TEST=True, retry=False))
    def status(state, **values):
        value = dict(state=state, time_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(), **values)
        (out / 'run_status.json').write_text(json.dumps(value, indent=2))
    proc = None
    try:
        for stage in ('infer', 'analyze'):
            checked(base)
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            training = read(base / 'launch_protocol.json')
            for k,v in training['launch_environment'].items():
                if v is None: env.pop(k, None)
                else: env[k] = v
            env['CUDA_VISIBLE_DEVICES'] = str(p['cuda_ordinal']) if stage == 'infer' else ''
            env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            if stage == 'infer':
                assert memory(p['physical_gpu'])['free_mib'] >= 4096
            else:
                env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            with (out / (stage + '.log')).open('x') as log:
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    '--stage', stage, '--base', str(base)], cwd=str(ROOT), env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status('RUNNING', stage=stage, pid=proc.pid, controller_pid=os.getpid())
                while proc.poll() is None:
                    time.sleep(5)
                    if stage == 'analyze': continue
                    resource = memory(p['physical_gpu'])
                    assert resource['free_mib'] >= p['minimum_free_during_run_mib'], resource
                    apps = subprocess.check_output(['nvidia-smi', '-i', str(p['physical_gpu']),
                        '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
                    own = {}
                    for line in apps.splitlines():
                        fields = line.split(',')
                        if len(fields) != 2 or not all(x.strip().isdigit() for x in fields): continue
                        pid, used = map(int, fields)
                        try:
                            if os.getpgid(pid) == proc.pid: own[pid] = used
                        except ProcessLookupError: pass
                    assert all(v <= p['own_driver_limit_mib'] for v in own.values()), own
                    with (out / 'memory_samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(time=time.time(), **resource, own_process_mib=own))+'\n')
                assert proc.returncode == 0, (stage, proc.returncode)
        status('COMPLETE', TEST=True, completion=read(out / 'completion.json'))
    except BaseException as error:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=15)
        status('STOPPED_ON_ERROR', error=repr(error), retry=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare','infer','analyze','run'), required=True)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(); base = args.base.resolve()
    def terminated(signum, frame):
        raise RuntimeError('Fixed TEST controller terminated')
    signal.signal(signal.SIGTERM, terminated)
    if args.stage == 'prepare':
        if args.gpu is None: parser.error('An explicit --gpu is required')
        prepare(base, args.gpu)
    elif args.stage == 'run':
        if not args.execute: parser.error('Running fixed TEST requires --execute')
        run(base)
    else:
        globals()[args.stage](base)


if __name__ == '__main__':
    main()
