"""Fixed TEST efficiency: external models or R4 -> M20 -> L5 stages.

Default check is stdlib-only and performs no inference. Benchmark is explicit,
single-GPU and guarded by a separate process. Saved boxes verify identity only;
no annotations, accuracy evaluator, checkpoint search, or training is used.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import gc
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import crane_frozen_pipeline_v1 as frozen
from crane_project.utils.port_detection_runtime_v1 import timing_summary

DEFAULT_CONFIG = 'configs/crane_final_models_efficiency_v1.json'
ARMS = ('final', 'eood', 'faa')
STAGE_ARMS = ('r4', 'r4_m', 'final')
STAGE_DEPTHS = dict(r4=1, r4_m=2, final=3)
STAGE_CONFIG = 'configs/crane_stage_efficiency_v1.json'
WINDOWS = ('decoded_frame_to_obb', 'model_and_postprocess',
           'file_to_obb', 'preprocessing_h2d')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def local(name):
    path = Path(name)
    return path if path.is_absolute() else ROOT/path


def resolve(spec):
    return frozen.resolve_checkpoint(spec)


def config_closure(path, stack=()):
    """Inspect inheritance paths, including MMCV {{_base_.x}} child syntax."""
    path = Path(path).resolve()
    if not any(parent in path.parents for parent in
               ((ROOT/'configs').resolve(), (ROOT/'crane_project/configs').resolve())):
        raise ValueError('Model definitions must stay in configs: '+str(path))
    if path in stack:
        raise ValueError('Cyclic config inheritance')
    names = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_base_'
                                               for t in node.targets):
            names = ast.literal_eval(node.value)
    if isinstance(names, str): names = [names]
    result = [path]
    for name in names:
        result.extend(config_closure(path.parent/name, stack+(path,)))
    return result


def contract(path=DEFAULT_CONFIG, require_weights=False):
    cfg = read(local(path))
    stages = cfg['schema'] == 'crane_stage_TEST_efficiency_v1'
    if (cfg['schema'] not in ('crane_final_models_TEST_efficiency_v1', 'crane_stage_TEST_efficiency_v1')
            or tuple(cfg['models']) != (STAGE_ARMS if stages else ARMS) or cfg['split'] != 'test'
            or cfg['frames'] != 1440 or cfg['batch_size'] != 1 or cfg['precision'] != 'FP32'
            or cfg['warmup_frames'] != 50 or cfg['repeats'] != 3
            or cfg['module_efficiency'] != stages):
        raise ValueError('Only the fixed three-model TEST protocol is supported')
    for name, digest in cfg['source_sha256'].items():
        if frozen.sha(local(name)) != digest:
            raise ValueError('Source/config SHA differs: '+name)
    pipeline, _ = frozen.read_config(cfg['models']['final']['config'])
    expected = dict(r4=(16,), r4_m=(16,20), final=(16,20,5)) if stages else {
        'final': (16,20,5), 'eood': (24,), 'faa': (24,)}
    weights = {}
    for arm, spec in cfg['models'].items():
        if tuple(w['epoch'] for w in spec['weights']) != expected[arm]:
            raise ValueError('Fixed epoch differs: '+arm)
        if stages or arm == 'final':
            depth = STAGE_DEPTHS[arm]
            if spec['weights'] != [dict(epoch=s['epoch'], sha256=s['sha256'], paths=s['paths'])
                                   for s in pipeline['stages'][:depth]]:
                raise ValueError('Final chain differs from the formal pipeline')
            if spec['config'] != cfg['models']['final']['config']:
                raise ValueError('Stage background/config differs')
        else:
            for parent in config_closure(local(spec['config'])):
                if str(parent.relative_to(ROOT)) not in cfg['source_sha256']:
                    raise ValueError('Unpinned config ancestor: '+str(parent))
        weights[arm] = [str(resolve(w)) if require_weights else None for w in spec['weights']]
    return cfg, weights


def references(cfg):
    """Read saved predictions; never pass GT or previous predictions to a model."""
    output = {}
    for arm, spec in cfg['models'].items():
        path = resolve(spec['reference'])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != cfg['frames'] or len({r['image'] for r in rows}) != cfg['frames']:
            raise ValueError('Incomplete/duplicated saved TEST rows: '+arm)
        # Deliberately discard every GT and metric field before returning.
        output[arm] = [dict(image=r['image'], box=r[spec['reference_key']]) for r in rows]
    ids = [r['image'] for r in output['final']]
    if ids != sorted(ids) or any([r['image'] for r in output[a]] != ids for a in cfg['models']):
        raise ValueError('Saved TEST image order differs')
    if Counter('_'.join(n.split('_')[:2]) for n in ids) != Counter(cfg['sequence_counts']):
        raise ValueError('TEST sequence identity differs')
    return output


def images(cfg, ids, verify_bytes=False):
    data = local(cfg['data_root'])
    if frozen.sha(data/'manifest.json') != cfg['manifest_sha256']:
        raise ValueError('Dataset manifest differs')
    selected = {r['id']: r for r in read(data/'manifest.json')['records'] if r['split'] == 'test'}
    if sorted(selected) != ids:
        raise ValueError('Manifest TEST set differs')
    paths = []
    for name in ids:
        spec = selected[name]['images']; path = data/spec['path']
        if not path.is_file(): raise FileNotFoundError(path)
        if verify_bytes and frozen.sha(path) != spec['sha256']:
            raise ValueError('TEST image SHA differs: '+name)
        paths.append(path)
    return paths


def compare_top1(actual, expected, geometry_atol=1e-4, score_atol=1e-5):
    """Same-frame float tolerances; never choose boxes by GT agreement."""
    if (actual is None) != (expected is None):
        raise ValueError('Saved output presence differs')
    if actual is None: return 0.
    if len(actual) != 6 or len(expected) != 6 or any(not math.isfinite(v) for v in actual):
        raise ValueError('Invalid Top1 box')
    if min(actual[2:4]) <= 0 or not 0 <= actual[5] <= 1:
        raise ValueError('Invalid box size/score')
    if any(abs(a-b) > geometry_atol+1e-6*abs(b) for a, b in zip(actual[:5], expected[:5])):
        raise ValueError('Saved box geometry differs')
    if abs(actual[5]-expected[5]) > score_atol:
        raise ValueError('Saved box score differs')
    return max(abs(a-b) for a, b in zip(actual, expected))


def top1(result):
    if len(result) == 0: return None
    return result[result[:, 5].argmax()].tolist()


class FinalForward:
    """One active prefix of the frozen chain; no diagnostic replays."""
    def __init__(self, pipeline, depth=3):
        if depth not in (1,2,3): raise ValueError('Unknown frozen stage depth')
        self.pipeline = pipeline
        self.depth = depth
        self.calls = dict(features=0, detector_head=0, sampling=0, midpoint=0, lvsr=0)

    def __call__(self, image, metas):
        import numpy as np
        p = self.pipeline
        features = p.detector.extract_feat(image); self.calls['features'] += 1
        native = p.detector.simple_test_from_features(features, metas, rescale=self.depth==1)
        self.calls['detector_head'] += 1
        raw_np = np.asarray(native[0][0])
        if raw_np.ndim != 2 or raw_np.shape[1] != 6 or len(raw_np) > 1:
            raise ValueError('Expected native single-object R4 output')
        if self.depth == 1:
            # Standalone R4 keeps native restoration and performs no ROI work.
            return raw_np
        raw = image.new_tensor(raw_np)
        # Same component-wise restoration as SymEOODHead(rescale=True).
        original = p.torch.cat((p.geometry.map_boxes(raw[:, :5], metas[0], inverse=True), raw[:, 5:]), 1)
        roi, support, _ = p.geometry.sample_local(features[0], raw[:, :5], metas[0], 'aligned')
        self.calls['sampling'] += 1
        xy = original.new_tensor(np.asarray(metas[0]['scale_factor']).reshape(-1)[:2]).reshape(1, 2)
        xy = xy.expand(len(original), -1)
        mid = p.midpoint(roi, support, original, raw[:, :5], xy)
        self.calls['midpoint'] += 1
        if self.depth == 2:
            self.last = (original, mid['boxes_original'], None)
            return mid['boxes_original'].cpu().numpy()
        size = p.lvsr(roi, support, original, raw[:, :5], xy, mid['boxes_original'])
        self.calls['lvsr'] += 1
        self.last = (original, mid['boxes_original'], size['boxes_original'])
        return size['boxes_original'].cpu().numpy()

    def verify_invariants(self):
        """Called outside timers; validation adds no model forward."""
        if self.depth == 1: return
        b, m, l = self.last
        torch = self.pipeline.torch
        if len(b) != len(m) or l is not None and len(l) != len(m):
            raise ValueError('Refinement changed output presence')
        if not torch.equal(b[:, 5], m[:, 5]) or (l is not None and
                not torch.equal(m[:, [0, 1, 4, 5]], l[:, [0, 1, 4, 5]])):
            raise ValueError('Refinement violated frozen score/center/angle contract')


def build(arm, cfg, weights):
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from crane_project.tools.benchmark_port_detection_runtime_v1 import flatten
    spec = cfg['models'][arm]
    if arm in STAGE_DEPTHS:
        depth = STAGE_DEPTHS[arm]
        # The audited constructor checks all chain weights on CPU. Drop inactive
        # heads BEFORE CUDA transfer; bare R4 must not carry M/L GPU memory.
        p = frozen.FrozenCranePipeline(spec['config'], device='cpu')
        if depth < 3: p.lvsr = None
        if depth < 2: p.midpoint = None
        models = tuple(m for m in (p.detector, p.midpoint, p.lvsr) if m is not None)
        for m in models: m.cuda(0).eval().requires_grad_(False)
        p.device = 'cuda:0'
        return FinalForward(p, depth), p.cfg, models, p.detector
    config = Config.fromfile(str(local(spec['config'])))
    import_modules_from_strings(**config.custom_imports)
    model_spec = deepcopy(config.model)
    model_spec.pretrained = None
    model_spec.backbone.init_cfg = None
    detector = build_detector(model_spec)
    payload = load_checkpoint(detector, weights[arm][0], map_location='cpu', strict=True)
    if payload.get('meta', {}).get('epoch') != 24:
        raise ValueError('Wrong checkpoint epoch: '+arm)
    del payload
    detector.cuda(0).eval().requires_grad_(False)
    if arm == 'faa' and (detector.neck.__class__.__name__ != 'FAAFusionFPN'
                        or detector.roi_head.bbox_head.__class__.__name__ != 'FAAHead'):
        raise ValueError('Expected full official FAA')
    if arm == 'eood' and detector.__class__.__name__ != 'Eood':
        raise ValueError('Expected the scale-augmented EOOD baseline')
    def predict(image, metas):
        return flatten(detector.simple_test(image, metas, rescale=True), __import__('numpy'))
    return predict, config, (detector,), detector


def parameter_counts(arm, models, detector):
    parameters = {id(p): p for m in models for p in m.parameters()}
    auxiliary = list(detector.bbox_head.predictors[1:]) if arm == 'eood' else (
        list(detector.aux_heads) if arm in STAGE_DEPTHS and detector.aux_heads is not None else [])
    excluded = {id(p) for m in auxiliary for p in m.parameters()}
    return dict(registered_including_training_aux=sum(p.numel() for p in parameters.values()),
                excluding_declared_training_aux=sum(p.numel() for key, p in parameters.items() if key not in excluded),
                declared_training_aux=sum(p.numel() for key, p in parameters.items() if key in excluded),
                note='Auxiliary modules stay loaded for strict checkpoint replay; no graph pruning or FLOPs claim.')


def verify_single_pass(counts, expected, depth=3):
    target = dict(features=expected, detector_head=expected,
                  sampling=expected if depth>=2 else 0, midpoint=expected if depth>=2 else 0,
                  lvsr=expected if depth==3 else 0)
    if counts != target:
        raise ValueError('Final pipeline repeated/skipped a stage: '+str(counts))


def run_worker(config_path, out):
    cfg, weights = contract(config_path, require_weights=True)
    refs = references(cfg); ids = [r['image'] for r in refs['final']]
    paths = images(cfg, ids, verify_bytes=True)
    import torch
    import numpy as np
    import cv2
    import mmcv, mmdet, mmrotate
    from crane_project.tools.benchmark_port_detection_runtime_v1 import transform, prepare, host_info
    from crane_project.tools.smoke_faa_gpu_v1 import processes
    torch.set_num_threads(1); cv2.setNumThreads(1)
    if torch.cuda.device_count() != 1 or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use exactly one GPU process')
    torch.cuda.set_device(0)
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    def limit():
        torch.cuda.set_per_process_memory_fraction(cfg['memory']['allocator_mib']*2**20/
                                                  torch.cuda.get_device_properties(0).total_memory, 0)
    limit(); probe = torch.empty(1, device='cuda:0')
    expected_gpu = read(out/'launch_protocol.json')['gpu']['uuid']
    if {r['uuid'] for r in processes() if r['pid'] == os.getpid()} != {expected_gpu}:
        raise ValueError('Physical GPU mapping differs')
    del probe
    def forbidden(*a, **k): raise RuntimeError('Optimizer construction forbidden')
    torch.optim.Optimizer.__init__ = forbidden
    runtime = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                   mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmrotate=mmrotate.__version__,
                   opencv=cv2.__version__, gpu=torch.cuda.get_device_name(0))
    frozen_runtime = read(local(cfg['models']['final']['config']))['runtime']
    if runtime != frozen_runtime:
        raise ValueError('Runtime differs from the fixed inference: '+str(runtime))
    report = dict(protocol=cfg['schema'], runtime=runtime, host=host_info(),
                  split='test', frames=len(ids), repeats=cfg['repeats'], warmup=cfg['warmup_frames'],
                  batch_size=1, precision='FP32', training_updates=0, accuracy_evaluated=False,
                  checkpoint_selection=False, modules_benchmarked=cfg['module_efficiency'], arms={})
    common_transform = None
    for arm in cfg['models']:
        active = out/'active_model.tmp'
        active.write_text(json.dumps(dict(model=arm)))
        os.replace(str(active), str(out/'active_model.json'))
        gc.collect(); torch.cuda.empty_cache()
        predict, model_cfg, models, detector = build(arm, cfg, weights)
        limit()  # The legacy R4 constructor also sets its allocator limit.
        spec = frozen.plain(model_cfg.data.test.pipeline)
        if common_transform is not None and spec != common_transform:
            raise ValueError('Three-model TEST preprocessing differs')
        common_transform = spec
        compose = transform(model_cfg)
        before = [frozen.state_digest(m) for m in models]
        if any(m.training or any(p.requires_grad for p in m.parameters()) for m in models):
            raise ValueError('Frozen inference required')
        params = parameter_counts(arm, models, detector)
        report.setdefault('input_pipeline', spec)
        report.setdefault('output_verification_tolerances', dict(geometry_atol=1e-4,
            geometry_rtol=1e-6, historical_score_atol=1e-5, same_runtime_repeat_score_atol=0.,
            reason='One common deterministic runtime; historical EOOD inference used different cuDNN flags.'))
        records = []; maximum = 0.; first_outputs = []; hook_calls = [0]
        handle = detector.backbone.register_forward_pre_hook(lambda *a: hook_calls.__setitem__(0, hook_calls[0]+1))
        # Native complete detection/NMS plus score-only Top1 extraction is timed.
        # M/L sampling, reconstruction and fallback are part of FinalForward.
        with torch.no_grad(), (out/(arm+'.timings.jsonl')).open('x') as log:
            for index in range(cfg['warmup_frames']):
                frame = cv2.imread(str(paths[index]))
                if frame is None: raise ValueError('Image decode failed')
                image, metas = prepare(compose, frame, 0)
                result = predict(image, metas); value = top1(result)
                compare_top1(value, refs[arm][index]['box'])
                if arm in STAGE_DEPTHS: predict.verify_invariants()
            torch.cuda.synchronize(0); torch.cuda.reset_peak_memory_stats(0)
            for repeat in range(cfg['repeats']):
                for index, path in enumerate(paths):
                    torch.cuda.synchronize(0); file_start = time.perf_counter()
                    frame = cv2.imread(str(path))
                    if frame is None: raise ValueError('Image decode failed')
                    app_start = time.perf_counter()
                    image, metas = prepare(compose, frame, 0)
                    torch.cuda.synchronize(0); model_start = time.perf_counter()
                    result = predict(image, metas); value = top1(result)
                    torch.cuda.synchronize(0); finish = time.perf_counter()
                    # No GT, reference parsing, digest or logging inside timers.
                    maximum = max(maximum, compare_top1(value, refs[arm][index]['box']))
                    if arm in STAGE_DEPTHS: predict.verify_invariants()
                    if repeat == 0: first_outputs.append(value)
                    else: compare_top1(value, first_outputs[index], score_atol=0.)
                    record = dict(arm=arm, repeat=repeat+1, image=ids[index],
                                  native_candidates=len(result), decoded_frame_to_obb=finish-app_start,
                                  model_and_postprocess=finish-model_start, file_to_obb=finish-file_start,
                                  preprocessing_h2d=model_start-app_start)
                    records.append(record); log.write(json.dumps(record)+'\n')
                log.flush(); print(arm, 'repeat', repeat+1, 'complete', flush=True)
                save(out/(arm+'.repeat_%d.json'%(repeat+1)), dict(completed_frames=len(paths)))
        handle.remove()
        expected = cfg['warmup_frames']+cfg['repeats']*len(paths)
        if hook_calls[0] != expected:
            raise ValueError('Repeated/skipped backbone execution: '+arm)
        if arm in STAGE_DEPTHS: verify_single_pass(predict.calls, expected, STAGE_DEPTHS[arm])
        after = [frozen.state_digest(m) for m in models]
        if before != after: raise ValueError('Parameters/buffers changed during timing')
        groups = {}
        for domain in ('all', 'real', 'sim'):
            rows = [r for r in records if domain == 'all' or r['image'].startswith(domain+'_')]
            groups[domain] = {key: timing_summary([r[key] for r in rows]) for key in WINDOWS}
        arm_report = dict(groups=groups, parameters=params, frozen_states=before,
                          frozen_states_unchanged=True, forward_frames=hook_calls[0],
                          final_stage_calls=predict.calls if arm in STAGE_DEPTHS else None,
                          reference_max_component_error=maximum, verified_timed_frames=len(records),
                          per_repeat={str(i):timing_summary([r['decoded_frame_to_obb'] for r in records if r['repeat']==i])
                                      for i in range(1, cfg['repeats']+1)},
                          gpu_peak_mib=dict(allocated=torch.cuda.max_memory_allocated(0)/2**20,
                                            reserved=torch.cuda.max_memory_reserved(0)/2**20))
        save(out/(arm+'.completed.json'), arm_report); report['arms'][arm] = arm_report
        del predict, model_cfg, models, detector, compose, result, image, metas
        gc.collect(); torch.cuda.empty_cache()
    # Recheck source, checkpoint, saved prediction and dataset image bytes.
    contract(config_path, require_weights=True); references(cfg); images(cfg, ids, verify_bytes=True)
    if cfg['module_efficiency']:
        report['stage_increments'] = {}
        for left, right in zip(STAGE_ARMS, STAGE_ARMS[1:]):
            a, b = report['arms'][left], report['arms'][right]
            x = a['groups']['all']['decoded_frame_to_obb']['mean_ms']
            y = b['groups']['all']['decoded_frame_to_obb']['mean_ms']
            report['stage_increments'][left+' -> '+right] = dict(
                mean_ms=y-x, relative_pct=100*(y/x-1),
                added_parameters=b['parameters']['excluding_declared_training_aux']-
                                 a['parameters']['excluding_declared_training_aux'],
                note='Difference of complete same-run active prefixes, not isolated kernel time.')
    report['passed'] = True
    save(out/'runtime_compare.json', report)
    with (out/'runtime_compare.csv').open('x') as stream:
        fields = ['model', 'mean_ms', 'p50_ms', 'p95_ms', 'fps', 'allocated_peak_mib',
                  'reserved_peak_mib', 'parameters_excluding_declared_aux', 'parameters_registered']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for arm, value in report['arms'].items():
            t = value['groups']['all']['decoded_frame_to_obb']; p = value['parameters']; m = value['gpu_peak_mib']
            writer.writerow(dict(model=arm, **{k:t[k] for k in ('mean_ms','p50_ms','p95_ms','fps')},
                                 allocated_peak_mib=m['allocated'], reserved_peak_mib=m['reserved'],
                                 parameters_excluding_declared_aux=p['excluding_declared_training_aux'],
                                 parameters_registered=p['registered_including_training_aux']))


def terminate(child):
    try: os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    try: child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try: os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        child.wait(timeout=10)


def benchmark(args):
    cfg, weights = contract(args.config, require_weights=True)
    refs = references(cfg); images(cfg, [r['image'] for r in refs['final']], verify_bytes=True)
    from crane_project.tools.smoke_faa_gpu_v1 import gpu_rows, processes
    if args.gpu not in (2, 3): raise ValueError('Use physical GPU 2 or 3')
    if args.out_dir is None: raise ValueError('A new --out-dir is required')
    all_gpus = gpu_rows(); gpu = next(r for r in all_gpus if r['index'] == args.gpu)
    if gpu['free_mib'] < cfg['memory']['startup_free_mib']:
        raise RuntimeError('Insufficient free GPU memory; no retry')
    if any(a['uuid'] == gpu['uuid'] for a in processes()):
        raise RuntimeError('Selected GPU has another compute process; use an idle card')
    ordinal = next(i for i, r in enumerate(sorted(all_gpus, key=lambda r:r['pci'])) if r['uuid']==gpu['uuid'])
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    # Pin the executable and dependencies for the whole background run.
    runtime_sources = dict(cfg['source_sha256'])
    for n in ('crane_project/tools/benchmark_crane_final_models_v1.py',
              'crane_project/tools/benchmark_port_detection_runtime_v1.py',
              'crane_project/utils/port_detection_runtime_v1.py',
              'crane_project/tools/smoke_faa_gpu_v1.py'):
        runtime_sources[n] = frozen.sha(local(n))
    launch = dict(protocol=cfg, config=str(local(args.config).resolve()), config_sha256=frozen.sha(local(args.config)),
                  source_sha256=runtime_sources, gpu=gpu, pci_cuda_ordinal=ordinal,
                  weights=weights, created_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  automatic_retry=False, training_updates=0)
    save(out/'launch_protocol.json', launch)
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(ordinal),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(ROOT),
               PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    command = [sys.executable, str(Path(__file__).resolve()), '--stage', 'worker',
               '--config', str(local(args.config).resolve()), '--out-dir', str(out)]
    reason = None; peak = 0; minimum_free = gpu['free_mib']; started = time.monotonic()
    per_model_peaks = {arm:0 for arm in cfg['models']}
    with (out/'native.log').open('x') as log, (out/'gpu_samples.jsonl').open('x') as samples:
        child = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        save(out/'launcher.json', dict(pid=child.pid, command=command))
        try:
            while child.poll() is None:
                row = next(r for r in gpu_rows() if r['uuid']==gpu['uuid']); apps = processes()
                own = sum(a['used_mib'] for a in apps if a['pid']==child.pid and a['uuid']==gpu['uuid'])
                peak = max(peak, own); minimum_free = min(minimum_free, row['free_mib'])
                active = read(out/'active_model.json')['model'] if (out/'active_model.json').exists() else None
                if active in per_model_peaks:
                    per_model_peaks[active] = max(per_model_peaks[active], own)
                if any(a['uuid']==gpu['uuid'] and a['pid']!=child.pid for a in apps): reason='GPU_CONTENTION'
                if own > cfg['memory']['process_stop_mib']: reason='PROCESS_MEMORY_LIMIT'
                if row['free_mib'] < cfg['memory']['minimum_free_mib']: reason='LOW_FREE_MEMORY'
                if time.monotonic()-started > cfg['timeout_seconds']: reason='TIMEOUT'
                samples.write(json.dumps(dict(unix_time=time.time(), model=active,
                                               gpu=row, own_process_mib=own))+'\n');samples.flush()
                if reason: terminate(child); break
                time.sleep(2)
        except BaseException:
            terminate(child); raise
        code = child.wait()
    unchanged = frozen.sha(local(args.config)) == launch['config_sha256'] and all(
        frozen.sha(local(n)) == digest for n, digest in runtime_sources.items())
    result = dict(passed=code==0 and reason is None and unchanged and (out/'runtime_compare.json').is_file(),
                  exit_code=code, stop_reason=reason, sources_unchanged=unchanged,
                  process_peak_mib=peak, minimum_free_mib=minimum_free,
                  per_model_sampled_process_peak_mib=per_model_peaks,
                  process_memory_sampling_seconds=2,
                  elapsed_seconds=time.monotonic()-started, training_updates=0, retry=False)
    save(out/'completion.json', result)
    save(out/'artifacts.json', {str(p.relative_to(out)):frozen.sha(p) for p in out.rglob('*')
                               if p.is_file() and p.name!='artifacts.json'})
    print(json.dumps(result, indent=2), flush=True)
    if not result['passed']: raise RuntimeError('Benchmark failed; preserve evidence, no automatic retry')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('check', 'benchmark', 'worker'), default='check')
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--require-weights', action='store_true', help='Also hash the existing local/server checkpoints')
    p.add_argument('--gpu', type=int, choices=(2, 3), help='Physical GPU; benchmark runs sequentially on one idle card')
    p.add_argument('--out-dir', type=Path)
    return p


def main():
    args = parser().parse_args(); os.chdir(ROOT)
    if args.stage == 'worker': return run_worker(args.config, args.out_dir)
    if args.stage == 'benchmark': return benchmark(args)
    cfg, _ = contract(args.config, require_weights=args.require_weights)
    refs = references(cfg); paths = images(cfg, [r['image'] for r in refs['final']])
    print(json.dumps(dict(passed=True, models=list(cfg['models']), TEST_frames=len(paths),
                          TEST_images_read=False, checkpoints_hashed=args.require_weights,
                          model_forwards=0, GPU_access=False, training_updates=0,
                          warmup=cfg['warmup_frames'], repeats=cfg['repeats']), indent=2))


if __name__ == '__main__': main()
