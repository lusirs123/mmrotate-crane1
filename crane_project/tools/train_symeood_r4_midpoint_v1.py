"""Adapt sigma1.5 midpoint to frozen R4. Default: static check, no training.

All generated stages live below one experiment root / midpoint / attempt_NNN.
No TEST or LVSR stage. Historical trainers, caches and heads remain unchanged.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import symeood_r4_midpoint_protocol_v1 as protocol


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def resource_check(physical, torch=None, before=False):
    text = subprocess.check_output(['nvidia-smi', '-i', str(physical),
        '--query-gpu=index,memory.free,memory.used,temperature.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    values = [int(v.strip()) for v in text.strip().split(',')]
    if len(values)!=4 or values[0]!=physical:
        raise ValueError('Selected-GPU telemetry differs')
    text = subprocess.check_output(['nvidia-smi', '-i', str(physical),
        '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=10)
    own = 0
    for line in text.splitlines():
        fields = [v.strip() for v in line.split(',')]
        if len(fields)==2 and fields[0]==str(os.getpid()):
            own += int(fields[1])
    allocated = torch.cuda.max_memory_allocated(0)/2**20 if torch is not None else 0.
    policy = protocol.MEMORY
    if (values[1] < policy['startup_free_mib' if before else 'minimum_free_mib']
            or own >= policy['process_stop_mib'] or allocated > policy['allocator_mib']):
        raise RuntimeError('R4 midpoint resource limit; stop without retry')
    return dict(physical_gpu=physical, free_mib=values[1], total_used_mib=values[2],
                temperature_C=values[3], own_process_mib=own, allocator_peak_mib=allocated)


def configuration():
    from mmcv import Config
    cfg = Config.fromfile(str(ROOT/protocol.R4_CONFIG))
    if (cfg.model.type != 'SymEOOD' or cfg.model.bbox_head.type != 'SymEOODRegTowerHeadV1'
            or cfg.model.bbox_head.loss_cls.use_geometry_weight is not False
            or cfg.model.bbox_head.loss_bbox.type != 'SymKLDLoss'
            or cfg.model.test_cfg.max_per_img != 1):
        raise ValueError('Frozen R4 effective model differs')
    return cfg


def load_frontend(engine, cfg, gpu):
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    checkpoint = ROOT/protocol.R4_CHECKPOINT
    if protocol.sha(checkpoint) != protocol.R4_SHA:
        raise ValueError('Frozen R4 checkpoint SHA differs')
    import_modules_from_strings(**cfg.custom_imports)
    spec = deepcopy(cfg.model)
    spec.pretrained = None; spec.backbone.init_cfg = None; spec.train_cfg = None
    detector = build_detector(spec)
    loaded = load_checkpoint(detector, str(checkpoint), map_location='cpu', strict=True)
    protocol.check_checkpoint_metadata(loaded['meta'], cfg.to_dict())
    if protocol.sha(checkpoint) != protocol.R4_SHA:
        raise ValueError('R4 checkpoint changed during loading')
    # Compatibility attribute read by the unchanged cache routine; identity is explicitly R4.
    detector.new110_load_receipt = dict(identity=protocol.frontend_identity(),
        strict=True, missing_keys=[], unexpected_keys=[], epoch=16, iteration=10240, seed=0)
    del loaded
    engine.g.freeze_detector(detector.cuda(gpu))
    return detector


def make_engine(physical):
    """Private module instance: bind R4 IO while reusing sealed midpoint mathematics."""
    if physical not in (0, 1, 2, 3) or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('One explicitly selected physical GPU0..3 only')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(physical)
    spec = importlib.util.spec_from_file_location('_r4_midpoint_engine',
        ROOT/'crane_project/tools/train_symeood_110_midpoint_v1.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.runtime_imports()
    engine.VERSION = protocol.VERSION
    engine.SOURCES = protocol.CONTRACT
    engine.frontend_identity = protocol.frontend_identity
    engine.checked_sources = protocol.checked_sources
    engine.check_cfg = configuration
    engine.build_detector = lambda cfg, gpu: load_frontend(engine, cfg, gpu)

    def runtime(gpu):
        if gpu != 0:
            raise ValueError('The selected physical GPU is logical CUDA0')
        import cv2
        import mmcv
        import mmdet
        import mmrotate
        torch = engine.torch
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(
            protocol.MEMORY['allocator_mib']*2**20/torch.cuda.get_device_properties(0).total_memory, 0)
        torch.cuda.reset_peak_memory_stats(0)
        return dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmrotate=mmrotate.__version__,
            opencv=cv2.__version__, gpu=torch.cuda.get_device_name(0))

    engine.runtime = runtime
    engine.resource_check = lambda gpu: resource_check(physical, engine.torch)
    return engine


def load_selected_pipeline(selection_path, physical_gpu):
    """Future frozen inference entry point; loads R4 and its bound selected head only."""
    protocol.checked_sources()
    resource_check(physical_gpu, before=True)
    engine = make_engine(physical_gpu)
    engine.runtime(0)
    return engine.load_selected_pipeline(selection_path, 0)


def runtime_prerequisites(paths, stage, identity):
    if stage == 'cache':
        return
    manifest = paths['cache']/'cache_manifest.json'
    if not manifest.exists():
        raise ValueError('A fresh, complete R4 cache is required')
    cache = json.loads(manifest.read_text())
    if (cache.get('protocol') != protocol.VERSION or cache.get('identity') != identity
            or cache.get('test_access') is not False or cache.get('detector_updates') != 0):
        raise ValueError('Old110/other-source cache cannot replace the R4 cache')
    if stage == 'train':
        gate_path = paths['verify']/'completion.json'
        if not gate_path.exists():
            raise ValueError('Zero-update verification must complete before training')
        protocol.verification_binding(json.loads(gate_path.read_text()), identity, protocol.sha(manifest))


def static_review():
    identity = protocol.checked_sources()
    audit = protocol.ablation_audit()
    return dict(status='R4_MIDPOINT_LOCAL_PREPARATION_CHECKED_RUNTIME_PENDING',
        identity=identity, ablation_audit=audit, optimizer_updates=0, training_runs=0,
        detector_updates=0, cache_created=False, server_access=False, test_access=False,
        pending=['R4 strict weight/metadata load', 'fresh R4 TRAIN/VAL cache',
                 'zero-update neutral/backward/save-reload/online verification', 'authorized head training'],
        weights_present_locally=(ROOT/protocol.R4_CHECKPOINT).exists())


def prepare(paths, review):
    dest = paths['preparation']; dest.mkdir(parents=True, exist_ok=False)
    write_new(dest/'review.json', review)
    write_new(dest/'plan.json', dict(protocol=protocol.VERSION, frozen_b=protocol.frontend_identity(),
        settings=review['identity']['settings'], sigma_cells=1.5,
        artifacts={k: str(v.relative_to(ROOT)) for k, v in paths.items()},
        memory=protocol.MEMORY, full_cache_extractions=6003,
        old_cache_allowed=False, old_midpoint_weights_allowed=False,
        initialization='fresh seed1703 zero-output head; sigma1.5 architecture/loss/decoder unchanged',
        detector_training=False, head_training_started=False, TEST=False, LVSR=False,
        checkpoint_selection='existing all-24-epoch full-VAL rule; no TEST selection or automatic adoption',
        ablation='R4 vs R4+M isolates midpoint; E/EA/EB/R4 does not isolate A/B',
        note='Legacy internal cache status token is retained for compatibility; protocol and frontend SHA must both bind R4.'))
    (dest/'README.md').write_text(
        '# R4下游适配\n\n'
        '当前只完成本地代码准备，尚未生成缓存、运行GPU或训练。\n\n'
        '准备资料保存在本目录；缓存、零更新检查和训练均放在'
        '`../midpoint/attempt_NNN/`，重试只新增内部子目录。'
        '不复用旧110缓存或旧midpoint权重。\n\n'
        '默认check仅做静态检查；cache、verify、train必须显式指定。'
        'verify不构造优化器；没有TEST或LVSR运行入口。\n')


def verify(engine, train, val, cache, identity, directory):
    """Forward/backward and neutral save/reload only; no optimizer construction."""
    torch = engine.torch
    original = torch.optim.Optimizer.__init__
    def forbidden(*a, **kw):
        raise AssertionError('Optimizer construction forbidden in verify stage')
    torch.optim.Optimizer.__init__ = forbidden
    try:
        engine.seed_all(); head = engine.new_head().to('cuda:0')
        before = engine.g.state_digest(head)
        chosen = []
        for domain in ('real', 'sim'):
            for scale in (1., .5):
                indices = [i for i, r in enumerate(train) if r['eligible'] and r['domain']==domain and r['scale']==scale]
                if len(indices) < 2:
                    raise ValueError('Need two eligible views in each TRAIN domain/scale')
                chosen += indices[:2]
        records = [train[i] for i in chosen]
        if any(r['b'] != r['midpoint'] for r in engine.evaluate(head, records+val, 'cuda:0')):
            raise ValueError('Neutral midpoint changed R4 outputs')
        roi, support, b, bm, xy, gt = engine.batch(train, chosen, 'cuda:0')
        out = head(roi, support, b, bm, xy)
        loss, _ = engine.m.point_loss(out['points_roi'], engine.m.target_points(gt, b, bm, xy))
        loss.backward()
        if (not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters())
                or float(head.output.weight.grad.norm()) <= 0):
            raise ValueError('Missing/nonfinite midpoint gradient')
        head.zero_grad(set_to_none=True)
        if engine.g.state_digest(head) != before:
            raise ValueError('Zero-update backward mutated midpoint')
        saved = directory/'neutral_head.pth'
        engine.atomic_save(saved, dict(protocol=protocol.VERSION, identity=identity,
            frozen_b=protocol.frontend_identity(), head_state=head.state_dict()))
        reloaded = engine.new_head().to('cuda:0')
        reloaded.load_state_dict(torch.load(str(saved), map_location='cpu')['head_state'], strict=True)
        if engine.g.state_digest(reloaded) != before:
            raise ValueError('Neutral head reload differs')
        online = engine.online_smoke(configuration(), 0, head, val, cache)
        return dict(status='R4_MIDPOINT_ZERO_UPDATE_VERIFIED', identity=identity,
            cache_manifest_sha256=cache['manifest_sha256'], detector_updates=0,
            optimizer_updates=0, test_access=False, neutral_train_views=len(chosen),
            neutral_val_frames=len(val), head_state_unchanged=True, neutral_save_reload_exact=True,
            online=online, resource=engine.resource_check(0))
    finally:
        torch.optim.Optimizer.__init__ = original


def run(args):
    paths = protocol.artifact_paths(args.root, args.attempt)
    review = static_review()
    if args.stage == 'check':
        return review
    if args.stage == 'prepare':
        prepare(paths, review)
        return dict(status='PREPARED_NO_RUNTIME', root=str(paths['root']), optimizer_updates=0)
    if not (paths['preparation']/'review.json').exists():
        raise ValueError('Prepare this experiment root first')
    prepared = json.loads((paths['preparation']/'review.json').read_text())
    if prepared['identity'] != review['identity']:
        raise ValueError('Prepared source identity differs')
    if args.stage not in ('cache', 'verify', 'train'):
        raise ValueError('Unknown stage')
    dest = paths[args.stage]  # cache and verification never dispatch training.
    if dest.exists():
        raise ValueError('Stage directory exists; preserve it and use another attempt child')
    runtime_prerequisites(paths, args.stage, review['identity'])
    resource_check(args.gpu, before=True)
    engine = make_engine(args.gpu)
    identity = review['identity']
    paths['attempt'].mkdir(parents=True, exist_ok=True)
    def progress(value):
        with (paths['attempt']/'progress.jsonl').open('a') as f:
            f.write(json.dumps(engine.g.json_native(value), allow_nan=False)+'\n')
        if value['stage']=='cache' or value['stage']=='epoch_complete' or value.get('step', 0)%200==0:
            engine.resource_check(0)
            print(json.dumps(engine.g.json_native(value)), flush=True)
    try:
        if args.stage == 'cache':
            result = engine.create_cache(dest, configuration(), 0, identity, progress)
        else:
            engine.runtime(0)
            train, val, cache = engine.load_cache(paths['cache'], identity)
            cache['manifest_sha256'] = protocol.sha(paths['cache']/'cache_manifest.json')
            if args.stage == 'verify':
                dest.mkdir()
                result = verify(engine, train, val, cache, identity, dest)
                write_new(dest/'completion.json', result)
            else:
                gate = json.loads((paths['verify']/'completion.json').read_text())
                protocol.verification_binding(gate, identity, cache['manifest_sha256'])
                dest.mkdir()
                result = engine.train_formal(train, val, cache, identity, 'cuda:0', dest, progress)
                write_new(dest/'completion.json', result)
        if protocol.checked_sources() != identity:
            raise ValueError('Source identity changed during stage')
        return dict(stage=args.stage, completed=True, result=result, test_access=False)
    except BaseException as error:
        failure = paths['attempt']/(args.stage+'_failure.json')
        if not failure.exists():
            write_new(failure, dict(stage=args.stage, error=repr(error), retry=False, test_access=False))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('check', 'prepare', 'cache', 'verify', 'train'), default='check')
    p.add_argument('--root', type=Path, default=protocol.DEFAULT_ROOT)
    p.add_argument('--attempt', default='attempt_001')
    p.add_argument('--gpu', type=int, choices=(0, 1, 2, 3), default=0)
    return p


if __name__ == '__main__':
    result = run(parser().parse_args())
    print(json.dumps(result, indent=2, allow_nan=False))
