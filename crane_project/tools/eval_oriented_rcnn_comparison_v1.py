"""Native Oriented R-CNN: fixed VAL -> one locked TEST -> common timing.

Process-local adapters reuse the existing exporter, metrics, VAL selector,
two-card inference guard and single-card timing loop. Original experiments and
the frozen training sources are not edited. Every new receipt names this model.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import check_oriented_rcnn_comparison_v1 as check
from crane_project.tools import eval_gaucho_val_test_v1 as workflow
from crane_project.tools import export_external_obb_v1 as exporter
from crane_project.tools import evaluate_external_obb_saved_v1 as saved
from crane_project.tools import select_external_obb_val_v1 as selector
from crane_project.tools import benchmark_external_obb_v1 as efficiency
from crane_project.tools import benchmark_crane_final_models_v1 as timing
from crane_project.tools import smoke_faa_gpu_v1 as resources
from crane_project.utils import external_training_monitor_io_v1 as monitor
from crane_project.tools.prepare_external_obb_v1 import (
    DATA, MANIFEST_SHA, dataset_manifest, read, sha, write_new)

MODEL = 'oriented_rcnn'
SPEC = 'configs/oriented_rcnn/experiment_comparison_v1.json'
SELF = 'crane_project/tools/eval_oriented_rcnn_comparison_v1.py'
TRAIN = ROOT/'work_dirs/oriented_rcnn_no_port_shrink_v1/attempt_001/oriented_rcnn'
EPOCHS = tuple(check.EPOCHS)
TEST_MARKER = TRAIN/'TEST_execution.json'
EXTRA = [SELF, SPEC, efficiency.TEMPLATE,
    'crane_project/tools/export_external_obb_v1.py',
    'crane_project/tools/evaluate_external_obb_saved_v1.py',
    'crane_project/tools/select_external_obb_val_v1.py',
    'crane_project/tools/eval_gaucho_val_test_v1.py',
    'crane_project/tools/benchmark_external_obb_v1.py',
    'crane_project/tools/benchmark_crane_final_models_v1.py',
    'crane_project/tools/benchmark_port_detection_runtime_v1.py',
    'crane_project/utils/port_detection_runtime_v1.py',
    'crane_project/utils/external_obb_io_v1.py',
    'crane_project/tools/eval_faa_val_test_v1.py',
    'crane_project/tools/eval_crane_offline.py',
    'crane_project/tools/eval_port_geometry_ap_v1.py',
    'crane_project/tools/analyze_symeood_ablation_first4_v1.py',
    'crane_project/utils/port_geometry_midpoint_edge_residual_metrics_v1.py',
    'crane_project/utils/port_geometry_midpoint_edge_feasibility_v1.py']
_installed = False


def configuration(model):
    if model != MODEL:
        raise ValueError('This dispatcher only evaluates Oriented R-CNN')
    return dict(check.source_identity(), **{name: sha(ROOT/name) for name in EXTRA})


def official_sources():
    provenance = check.check_sources()
    return {MODEL: dict(repository=provenance['repository'], tag=provenance['tag'],
        files={name: item['sha256'] for name, item in provenance['files'].items()},
        provenance_sha256=sha(ROOT/check.PROVENANCE))}


def experiment_read(path):
    # The reused two-model selector/lock validator has one legacy hard-coded
    # recipe path. Redirect only that recipe read; all result identities and
    # exported model names remain oriented_rcnn, never gaucho.
    if Path(path).resolve() == ROOT/'configs/gaucho/experiment.json':
        return monitor.read_json(ROOT/SPEC)
    return monitor.read_json(path)


def export_write(path, value):
    if Path(path).name == 'export_receipt.json' and value.get('model') == MODEL:
        value = dict(value, geometry='original pixels, radians, native Oriented R-CNN le90 OBB')
    return write_new(path, value)


def checkpoint(epoch):
    if epoch not in EPOCHS:
        raise ValueError('Epoch is outside the preregistered candidate set')
    return TRAIN/'train'/('epoch_%d.pth' % epoch)


def load_backend(model_name, path, device):
    if model_name != MODEL:
        raise ValueError('Wrong model')
    import torch
    import mmrotate
    from mmcv.runner import load_checkpoint
    from mmdet.apis import init_detector, inference_detector
    if Path(mmrotate.__file__).resolve().parent != ROOT/'mmrotate':
        raise ValueError('Wrong MMRotate package')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    module = init_detector(str(ROOT/check.CONFIG), checkpoint=None, device='cpu')
    payload = load_checkpoint(module, str(path), map_location='cpu', strict=True)
    epoch = int(Path(path).stem.split('_')[-1])
    if epoch not in EPOCHS or payload.get('meta', {}).get('epoch') != epoch:
        raise ValueError('Checkpoint epoch differs from native metadata')
    if payload['meta']['iter'] != epoch*1279:
        raise ValueError('Wrong completed training iteration')
    del payload
    if (module.__class__.__name__ != 'OrientedRCNN'
            or module.neck.__class__.__name__ != 'FPN'
            or module.roi_head.bbox_head.__class__.__name__ != 'RotatedShared2FCBBoxHead'
            or module.roi_head.bbox_head.num_classes != 1):
        raise ValueError('Not the standard single-class Oriented R-CNN')
    module.CLASSES = ('grab',)
    module.to(device).eval().requires_grad_(False)
    first = [True]
    def predict(image):
        before = exporter.digest(module) if first[0] else None
        result = inference_detector(module, str(image))
        if before is not None and exporter.digest(module) != before:
            raise ValueError('First native inference changed model state')
        first[0] = False
        if len(result) != 1:
            raise ValueError('Expected single-class native result')
        return exporter.common_result(result[0])
    identity = dict(package=str(Path(mmrotate.__file__).resolve()),
        detector_source=inspect.getfile(module.__class__),
        head_source=inspect.getfile(module.roi_head.bbox_head.__class__))
    return module, predict, identity, lambda: module


def timing_build(arm, cfg, weights):
    if arm != MODEL or tuple(cfg['models']) != (MODEL,):
        raise ValueError('Timing may only load this locked model')
    import numpy as np
    from crane_project.tools.benchmark_port_detection_runtime_v1 import flatten
    detector, _, _, _ = load_backend(MODEL, Path(weights[arm][0]), 'cuda:0')
    def predict(image, metas):
        return flatten(detector.simple_test(image, metas, rescale=True), np)
    return predict, detector.cfg, (detector,), detector


def route_command(command):
    command = list(command)
    if len(command) > 1 and command[1] == str(ROOT/'crane_project/tools/export_external_obb_v1.py'):
        remaining = command[2:]
        index = remaining.index('--model')+1
        if remaining[index] != 'gaucho':
            raise ValueError('Unexpected shared inference dispatcher request')
        remaining[index] = MODEL
        return [command[0], str(ROOT/SELF), '--stage', 'export']+remaining
    if len(command) > 1 and command[1] == str(ROOT/'crane_project/tools/benchmark_crane_final_models_v1.py'):
        if command[2:4] != ['--stage', 'worker']:
            raise ValueError('Only the fixed timing worker may be redirected')
        return [command[0], str(ROOT/SELF), '--stage', 'timing_worker']+command[4:]
    raise ValueError('Unexpected child command')


def spawn(command, **kwargs):
    return subprocess.Popen(route_command(command), **kwargs)


def child_process_api():
    return SimpleNamespace(Popen=spawn, TimeoutExpired=subprocess.TimeoutExpired,
                           DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT)


def completed_identity(evaluation):
    evaluation = Path(evaluation).resolve()
    done = read(evaluation/'completion.json')
    lock_path = evaluation/'selection/selected_identity.json'
    lock = read(lock_path)
    if (not done['passed'] or done['TEST_runs'] != 1 or done['training_updates'] != 0
            or done['TEST_reselection'] is not False or lock['model'] != MODEL
            or done['selected_epoch'] != lock['selected_completed_epoch']
            or done['checkpoint_sha256'] != lock['checkpoint_sha256']):
        raise ValueError('Completed VAL-locked Oriented R-CNN TEST required')
    exporter.validate_test_lock(lock, MODEL, done['checkpoint_sha256'], configuration(MODEL))
    test = evaluation/'test'/('epoch_%d' % lock['selected_completed_epoch'])
    _, receipt = saved.validate_export(test)
    metrics = read(test/'paper_metrics.json')
    if (receipt['split'] != 'test' or receipt['frames'] != 1440
            or receipt['selection_lock_sha256'] != sha(lock_path)
            or receipt['initialized_state_sha256'] != receipt['final_state_sha256']
            or metrics['split'] != 'test' or metrics['metrics'][0]['frames'] != 1440
            or not all(c['pass_check'] for c in metrics['native_checks'])):
        raise ValueError('Inconsistent frozen TEST outputs')
    return lock, test


def make_profile(evaluation):
    evaluation = Path(evaluation).resolve()
    lock, test = completed_identity(evaluation)
    cfg = read(ROOT/efficiency.TEMPLATE)
    training = read(evaluation/'protocol.json')
    sources = dict(training['training_sources'], **configuration(MODEL))
    sources[cfg['runtime_reference']] = sha(ROOT/cfg['runtime_reference'])
    epoch = lock['selected_completed_epoch']
    cfg.update(evaluation_dir=str(evaluation), accuracy_evidence_sha256={
        str(p): sha(p) for p in (evaluation/'completion.json', evaluation/'selection/selected_identity.json',
            test/'export_receipt.json', test/'paper_metrics.json', test/'per_frame.jsonl')},
        source_sha256=sources, models={MODEL: dict(label='Standard Oriented R-CNN R50/FPN, VAL ep%d' % epoch,
            config=check.CONFIG, weights=[dict(epoch=epoch, sha256=lock['checkpoint_sha256'],
                paths=[lock['checkpoint']])], reference=dict(sha256=sha(test/'per_frame.jsonl'),
                paths=[str(test/'per_frame.jsonl')]), reference_key='prediction')})
    validate_profile(cfg)
    return cfg


def validate_profile(cfg):
    template = read(ROOT/efficiency.TEMPLATE)
    if any(cfg.get(k) != v for k, v in template.items()) or tuple(cfg['models']) != (MODEL,):
        raise ValueError('Common efficiency protocol changed')
    for name, digest in cfg['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Efficiency source changed: '+name)
    for name, digest in cfg['accuracy_evidence_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Accuracy evidence changed: '+name)
    lock, test = completed_identity(cfg['evaluation_dir'])
    expected_weight = dict(epoch=lock['selected_completed_epoch'], sha256=lock['checkpoint_sha256'],
                           paths=[lock['checkpoint']])
    expected_ref = dict(sha256=sha(test/'per_frame.jsonl'), paths=[str(test/'per_frame.jsonl')])
    spec = cfg['models'][MODEL]
    if (spec['config'] != check.CONFIG or spec['weights'] != [expected_weight]
            or spec['reference'] != expected_ref or spec['reference_key'] != 'prediction'):
        raise ValueError('Timing checkpoint/reference differs from locked TEST')
    protocol = read(Path(cfg['evaluation_dir'])/'protocol.json')
    required = dict(protocol['training_sources'], **configuration(MODEL))
    if any(cfg['source_sha256'].get(name) != digest for name, digest in required.items()):
        raise ValueError('Missing frozen dependency in timing profile')


def install():
    global _installed
    if _installed:
        return
    _installed = True
    exporter.config_identity = configuration
    exporter.check_sources = official_sources
    exporter.check_configs = check.check_config
    exporter.load_backend = load_backend
    exporter.read = experiment_read
    exporter.write_new = export_write
    saved.check_sources = official_sources
    selector.read = experiment_read
    workflow.EPOCHS = EPOCHS
    workflow.checkpoint = checkpoint
    workflow.read = monitor.read_json
    workflow.status = monitor.atomic_json
    workflow.gpu_rows = monitor.gpu_rows
    workflow.processes = monitor.processes
    workflow.subprocess = child_process_api()
    efficiency.completed_identity = completed_identity
    efficiency.make_profile = make_profile
    efficiency.validate_profile = validate_profile
    timing.build = timing_build
    timing.read = monitor.read_json
    timing.subprocess = child_process_api()
    resources.gpu_rows = monitor.gpu_rows
    resources.processes = monitor.processes


def static_check():
    spec = read(ROOT/SPEC)
    if (spec['model'] != MODEL or ROOT/spec['training_dir'] != TRAIN
            or spec['configuration'] != check.CONFIG or spec['own_scale_augmentation'] is not False
            or spec['selection']['candidate_completed_epochs'] != list(EPOCHS)
            or spec['selection']['native_checkpoint_names'] != ['epoch_%d.pth' % e for e in EPOCHS]):
        raise ValueError('Experiment differs from preregistered training candidates')
    config = configuration(MODEL)
    return dict(passed=True, model=MODEL, candidates=list(EPOCHS), physical_gpus=[0, 1],
        configuration_files=len(config), VAL_frames=887, TEST_frames=1440,
        training_updates=0, inference_runs=0, TEST_after_VAL_lock=True,
        timing=dict(warmup=50, frames=1440, repeats=3, batch=1, precision='FP32'))


def verify(protocol):
    if configuration(MODEL) != protocol['configuration']:
        raise ValueError('Evaluation source/configuration changed')
    for name, digest in protocol['training_sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Trained source changed: '+name)
    if sha(DATA/'manifest.json') != protocol['manifest_sha256']:
        raise ValueError('Dataset manifest changed')


def prepare(out):
    static_check()
    if TEST_MARKER.exists():
        raise FileExistsError('TEST already reserved; inspect existing evidence instead of repeating')
    done = read(TRAIN/'completion.json')
    if (not done['passed'] or done['exit_code'] != 0 or len(done['workers']) != 2
            or any(w['completed_epochs'] != 12 or w['optimizer_updates'] != 15348 for w in done['workers'])):
        raise ValueError('Incomplete native 12-epoch training')
    train = read(TRAIN/'protocol.json')
    from crane_project.tools.run_oriented_rcnn_comparison_v1 import assert_sources, idle
    assert_sources(train)
    if train['checkpoint_selection']['candidates'] != list(EPOCHS):
        raise ValueError('Candidates differ from pre-training registration')
    rows = idle()
    dataset_manifest()
    from crane_project.tools.ckpt_sweep import SELECTION_CONFIG
    from crane_project.utils.crane_frozen_pipeline_v1 import read_constants
    reference = read_constants(ROOT/'crane_project/tools/eval_faa_val_test_v1.py', ['REF', 'REF_SHA'])
    if sha(ROOT/reference['REF']) != reference['REF_SHA']:
        raise ValueError('Shared frozen TEST reference changed')
    out.mkdir(parents=True, exist_ok=False)
    (out/'val').mkdir()
    write_new(out/'protocol.json', dict(protocol='oriented_rcnn_fixed_VAL_then_one_TEST_v1',
        model=MODEL, created_Taipei=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        training_dir=str(TRAIN), training_completion_sha256=sha(TRAIN/'completion.json'),
        training_protocol_sha256=sha(TRAIN/'protocol.json'), candidates=list(EPOCHS),
        weights={str(e): sha(checkpoint(e)) for e in EPOCHS}, configuration=configuration(MODEL),
        selection_config=SELECTION_CONFIG, training_sources=train['sources'],
        manifest_sha256=MANIFEST_SHA, test_reference=reference, physical_gpus=[0, 1], before=rows,
        allocator_mib=6144, driver_mib=6656, minimum_free_mib=1536,
        training_updates=0, automatic_retry=False, TEST_after_VAL_lock=True,
        TEST_runs=1, TEST_reselection=False, own_scale_augmentation=False,
        training_sufficiency_review='Native 12 epochs complete; finite final losses, checkpoints verified; user authorized VAL/TEST/timing.'))


def controller(out):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    protocol = read(out/'protocol.json')
    try:
        for index in range(0, len(EPOCHS), 2):
            jobs = [('val', epoch, gpu) for gpu, epoch in enumerate(EPOCHS[index:index+2])]
            verify(protocol); workflow.infer(out, jobs, protocol)
        verify(protocol)
        for epoch in EPOCHS:
            monitor.atomic_json(out/'status.json', dict(state='CPU_METRICS', split='val', epoch=epoch))
            saved.evaluate(out/'val'/('epoch_%d' % epoch))
        selector.select(MODEL, out/'val', out/'selection')
        lock = read(out/'selection/selected_identity.json')
        epoch = lock['selected_completed_epoch']
        if sha(lock['checkpoint']) != protocol['weights'][str(epoch)]:
            raise ValueError('Selected weight changed')
        exporter.validate_test_lock(lock, MODEL, protocol['weights'][str(epoch)], protocol['configuration'])
        verify(protocol)
        monitor.atomic_json(out/'status.json', dict(state='VAL_LOCKED', selected_epoch=epoch,
            locked_Taipei=lock['locked_Taipei'], TEST_started=False))
        (out/'test').mkdir()
        workflow.reserve_test(TEST_MARKER, dict(evaluation_dir=str(out), selected_epoch=epoch,
            checkpoint_sha256=protocol['weights'][str(epoch)],
            selection_lock_sha256=sha(out/'selection/selected_identity.json'),
            training_protocol_sha256=protocol['training_protocol_sha256']))
        workflow.infer(out, [('test', epoch, 0)], protocol)
        monitor.atomic_json(out/'status.json', dict(state='CPU_METRICS', split='test', epoch=epoch))
        saved.evaluate(out/'test'/('epoch_%d' % epoch))
        verify(protocol)
        write_new(out/'completion.json', dict(passed=True, selected_epoch=epoch,
            checkpoint_sha256=protocol['weights'][str(epoch)], TEST_runs=1, TEST_reselection=False,
            training_updates=0, automatic_retry=False, accuracy_complete=True, efficiency_required=True,
            metrics=str(out/'test'/('epoch_%d' % epoch)/'paper_metrics.json')))
        monitor.atomic_json(out/'status.json', dict(state='EFFICIENCY', selected_epoch=epoch, training_updates=0))
        efficiency.run(out, out/'efficiency', gpu=0)
        write_new(out/'pipeline_completion.json', dict(passed=True, selected_epoch=epoch,
            accuracy_complete=True, efficiency_complete=True, TEST_runs=1, training_updates=0))
        monitor.atomic_json(out/'status.json', dict(state='COMPLETED', selected_epoch=epoch,
            accuracy_complete=True, efficiency_complete=True, TEST_runs=1, training_updates=0))
    except BaseException as error:
        monitor.atomic_json(out/'status.json', dict(state='FAILED', error=repr(error), automatic_retry=False))
        raise


def main():
    os.chdir(ROOT)
    install()
    stage = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == '--stage' else None
    if stage == 'timing_worker':
        sys.argv = [str(ROOT/SELF), '--stage', 'worker']+sys.argv[3:]
        return timing.main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'launch', 'controller', 'export'), default='check')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--model', choices=(MODEL,), default=MODEL)
    parser.add_argument('--split', choices=('val', 'test'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--physical-gpus', type=int, nargs=2, default=[0, 1])
    parser.add_argument('--selection-lock', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.stage == 'check':
        print(json.dumps(static_check(), indent=2)); return
    if args.out is None:
        parser.error('--out required')
    out = args.out.resolve()
    if args.stage == 'export':
        if not args.execute or args.checkpoint is None or args.split is None:
            parser.error('Export requires explicit --execute, --checkpoint and --split')
        args.out = out
        return exporter.export(args)
    if args.stage == 'controller':
        return controller(out)
    prepare(out)
    with (out/'pipeline.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(ROOT/SELF), '--stage', 'controller', '--out', str(out)],
            cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    receipt = dict(pid=child.pid, model=MODEL, out=str(out), physical_gpus=[0, 1],
        VAL_candidates=list(EPOCHS), TEST_after_VAL_lock=True, efficiency_required=True)
    write_new(out/'launch.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
