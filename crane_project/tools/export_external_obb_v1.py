"""Thin official inference adapters to the existing common OBB evaluator.

Default prints a plan. --execute is explicit inference, never training. TEST
requires a prior VAL-selection lock. This tool does not select checkpoints or
report runtime/FPS. Official decoding/postprocessing remain inside each backend.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools.prepare_external_obb_v1 import (
    DATA, GAUCHO_CONFIG, MANIFEST_SHA, YOLO_CONFIG, check_configs,
    check_sources, dataset_manifest, read, sha, write_new)
from crane_project.utils.external_obb_io_v1 import common_result, yolo_to_common


def gpu_binding(devices, visible, authorized, logical_device):
    """Resolve a PCI-ordered binding without touching CUDA or importing Torch."""
    authorized = tuple(authorized)
    if authorized not in ((0, 1), (2, 3)):
        raise ValueError('Choose an explicitly authorized physical GPU pair')
    ordered = sorted(devices, key=lambda r: r['pci'])
    tokens = visible.split(',')
    if (not tokens or any(not x.isdigit() for x in tokens)
            or len(set(tokens)) != len(tokens)):
        raise ValueError('Expected distinct numeric PCI CUDA ordinals')
    ordinals = [int(x) for x in tokens]
    if len(set(ordinals)) != len(ordinals):
        raise ValueError('Expected distinct numeric PCI CUDA ordinals')
    if any(i >= len(ordered) or ordered[i]['index'] not in authorized for i in ordinals):
        raise ValueError('CUDA binding differs from the authorized physical GPU pair')
    if logical_device not in ('cuda:0', 'cuda:1'):
        raise ValueError('Expected cuda:0 or cuda:1')
    index = int(logical_device.split(':')[-1])
    if index >= len(ordinals):
        raise ValueError('Logical CUDA device is not present in the binding')
    return ordered[ordinals[index]]


def config_identity(model):
    if model == 'gaucho':
        from crane_project.tools.check_faa_adaptation_v1 import resolve
        _, files = resolve(ROOT / GAUCHO_CONFIG)
        files += ['configs/gaucho/experiment.json',
                  'mmrotate/models/dense_heads/gaussian_fcos_head.py',
                  'mmrotate/models/dense_heads/gaucho_rotated_fcos_head.py',
                  'mmrotate/models/losses/gaucho_gaussian_dist_loss_v1.py',
                  'mmrotate/datasets/crane_custom_dota.py',
                  'mmrotate/datasets/pipelines/port_train_augment.py']
    else:
        spec = read(ROOT / YOLO_CONFIG)
        files = [YOLO_CONFIG, spec['architecture'], spec['data'],
                 'configs/yolo26/crane_yolo26m_obb_train_v1.yaml']
    files += ['configs/%s/upstream_manifest.json' % ('yolo26' if model == 'yolo26' else model),
              'crane_project/tools/export_external_obb_v1.py',
              'crane_project/utils/external_obb_io_v1.py',
              'crane_project/tools/select_external_obb_val_v1.py',
              'crane_project/tools/evaluate_external_obb_saved_v1.py',
              'crane_project/tools/ckpt_sweep.py',
              'crane_project/tools/eval_faa_val_test_v1.py',
              'crane_project/tools/eval_crane_offline.py',
              'crane_project/tools/eval_port_geometry_ap_v1.py',
              'crane_project/tools/analyze_symeood_ablation_first4_v1.py',
              'crane_project/utils/port_geometry_midpoint_edge_residual_metrics_v1.py',
              'crane_project/utils/port_geometry_midpoint_edge_feasibility_v1.py']
    return {name: sha(ROOT / name) for name in files}


def validate_test_lock(lock, model, checkpoint_sha256, configuration):
    if lock.get('evidence_role') != 'source_val_checkpoint_selection' or lock.get('TEST_used_for_selection') is not False:
        raise ValueError('TEST requires a prior source-VAL selection lock')
    if lock.get('model') != model or lock.get('checkpoint_sha256') != checkpoint_sha256:
        raise ValueError('TEST weight/model differs from the VAL lock')
    if lock.get('manifest_sha256') != MANIFEST_SHA or lock.get('configuration') != configuration:
        raise ValueError('TEST dataset/configuration differs from the VAL lock')
    selection = Path(lock['selection_results'])
    if not selection.is_absolute():
        selection = ROOT / selection
    if sha(selection) != lock['selection_results_sha256']:
        raise ValueError('VAL selection results changed')
    result = read(selection)
    cfg = read(ROOT / (YOLO_CONFIG if model == 'yolo26' else 'configs/gaucho/experiment.json'))
    names = ['epoch_%d' % e for e in cfg['selection']['candidate_completed_epochs']]
    if set(result['all_checkpoints']) != set(names) or result['selected_checkpoint'] not in names:
        raise ValueError('VAL candidate set is incomplete or expanded')
    chosen = result['all_checkpoints'][result['selected_checkpoint']]
    if (result['model'] != model or result['configuration'] != configuration
            or result['candidate_completed_epochs'] != cfg['selection']['candidate_completed_epochs']
            or chosen['checkpoint_sha256'] != checkpoint_sha256
            or result['TEST_used_for_selection'] is not False):
        raise ValueError('Locked checkpoint differs from complete source-VAL selection')
    epoch = int(result['selected_checkpoint'].split('_')[-1])
    if (lock.get('selected_completed_epoch') != epoch
            or lock.get('checkpoint') != chosen['checkpoint']
            or result.get('selected_path') != chosen['checkpoint']):
        raise ValueError('VAL selected epoch/path is inconsistent')
    from crane_project.utils.crane_frozen_pipeline_v1 import read_constants
    rule_path = ROOT / 'crane_project/tools/ckpt_sweep.py'
    rule = read_constants(rule_path, ['SELECTION_CONFIG'])['SELECTION_CONFIG']
    if (result.get('selection_config') != rule
            or result.get('selection_rule_source_sha256') != sha(rule_path)
            or result.get('center_thresh_px') != 15.0):
        raise ValueError('VAL selection rule differs from the common task rule')


def load_backend(model_name, checkpoint, device):
    import torch
    if model_name == 'yolo26':
        sys.path.insert(0, str(ROOT / 'third_party/ultralytics'))
        from ultralytics import YOLO
        import ultralytics
        if Path(ultralytics.__file__).resolve().parent != ROOT / 'third_party/ultralytics/ultralytics':
            raise ValueError('Wrong Ultralytics package')
        model = YOLO(str(checkpoint), task='obb', verbose=False)
        if model.model.model[-1].__class__.__name__ != 'OBB26' or model.model.model[-1].nc != 1:
            raise ValueError('Expected trained official YOLO26 single-class OBB model')
        options = read(ROOT / YOLO_CONFIG)['prediction']
        def predict(image):
            result = model.predict(source=str(image), device=device, **options)[0]
            if not model.predictor.model.end2end:
                raise ValueError('Official YOLO26 one-to-one inference branch was not selected')
            obb = result.obb
            if obb is None:
                raise ValueError('Native OBB results missing')
            return yolo_to_common(obb.xywhr.detach().cpu().numpy(),
                                  obb.conf.detach().cpu().numpy(),
                                  obb.cls.detach().cpu().numpy())
        module = model.model
        identity = dict(package=str(Path(ultralytics.__file__).resolve()),
                        head_source=inspect.getfile(module.model[-1].__class__))
        # This official release deep-copies the checkpoint module into its
        # predictor. Audit the executed backend, not the untouched source copy.
        live_module = lambda: model.predictor.model.backend.model
    else:
        package = ROOT
        sys.path.insert(0, str(ROOT))
        import mmrotate
        if Path(mmrotate.__file__).resolve().parent != package / 'mmrotate':
            raise ValueError('GauCho must use the current project mmrotate package')
        # MMRotate 0.3.x uses MMDetection's public single-image APIs,
        # as in demo/image_demo.py; mmrotate.apis only adds patch inference.
        from mmdet.apis import inference_detector, init_detector
        from mmcv.runner import load_checkpoint
        module = init_detector(str(ROOT / GAUCHO_CONFIG), checkpoint=None, device='cpu')
        load_checkpoint(module, str(checkpoint), map_location='cpu', strict=True)
        module.CLASSES = ('grab',)
        if module.bbox_head.__class__.__name__ != 'GaussianFCOSHead' or not module.bbox_head.gaucho_encoding:
            raise ValueError('Expected official GauCho head and encoding')
        if (module.bbox_head.__class__.__mro__[1].__module__ != 'mmrotate.models.dense_heads.gaucho_rotated_fcos_head'
                or module.bbox_head.loss_bbox.__class__.__module__ != 'mmrotate.models.losses.gaucho_gaussian_dist_loss_v1'):
            raise ValueError('GauCho official parent/loss registration differs')
        module.to(device)
        def predict(image):
            result = inference_detector(module, str(image))
            if len(result) != 1:
                raise ValueError('Expected native one-class GauCho result')
            return common_result(result[0])
        identity = dict(package=str(Path(mmrotate.__file__).resolve()),
                        head_source=inspect.getfile(module.bbox_head.__class__))
        live_module = lambda: module
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module, predict, identity, live_module


def digest(module):
    value = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def export(args):
    sources = check_sources()
    check_configs()
    configuration = config_identity(args.model)
    checkpoint = args.checkpoint.resolve()
    weight_sha = sha(checkpoint)
    if args.split == 'test':
        if args.selection_lock is None:
            raise ValueError('TEST requires --selection-lock')
        validate_test_lock(read(args.selection_lock), args.model, weight_sha, configuration)
    rows = sorted([r for r in dataset_manifest() if r['split'] == args.split], key=lambda r: r['id'])
    for row in rows:
        for kind in ('images', 'annfiles'):
            if sha(DATA / row[kind]['path']) != row[kind]['sha256']:
                raise ValueError('Dataset bytes differ: ' + row['id'])
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(tempfile.gettempdir()) / 'crane_yolo26_export_settings'))
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif args.device not in ('cuda:0', 'cuda:1') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Bind authorized GPUs explicitly via CUDA_VISIBLE_DEVICES, then select cuda:0/1')
    if args.device != 'cpu':
        from crane_project.tools.smoke_faa_gpu_v1 import gpu_rows
        if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
            raise ValueError('CUDA_DEVICE_ORDER must be PCI_BUS_ID')
        selected = gpu_binding(gpu_rows(), os.environ['CUDA_VISIBLE_DEVICES'],
                               getattr(args, 'physical_gpus', (2, 3)), args.device)
        if selected['free_mib'] < 7168:
            raise ValueError('Selected GPU does not have the required starting free memory')
    import torch
    torch.set_num_threads(1)
    if args.device != 'cpu':
        index = int(args.device.split(':')[-1])
        torch.cuda.set_per_process_memory_fraction(0.75, index)
    if args.device != 'cpu':
        torch.cuda.set_device(index)
        torch.cuda.reset_peak_memory_stats(index)
    module, predict, identity, live_module = load_backend(args.model, checkpoint, args.device)
    started = time.monotonic()
    before_first = digest(module) if args.model == 'gaucho' else None
    def progress(completed):
        value = dict(split=args.split, frames=len(rows), completed=completed,
                     elapsed_seconds=time.monotonic()-started, training_updates=0)
        if args.device != 'cpu':
            value.update(peak_allocated_mib=torch.cuda.max_memory_allocated(index)/2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved(index)/2**20)
        path = args.out / 'progress.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, indent=2)+'\n')
        os.replace(str(tmp), str(path))
    predictions = []
    # Native YOLO may fuse BN and select its inference branch on first predict.
    # Freeze/hash the resulting inference module after this initialization.
    with torch.inference_mode():
        predictions.append(predict(DATA / rows[0]['images']['path']))
        module = live_module()
        state = digest(module)
        if before_first is not None and before_first != state:
            raise ValueError('GauCho inference changed model state on the first frame')
        progress(1)
        for number, row in enumerate(rows[1:], 2):
            predictions.append(predict(DATA / row['images']['path']))
            if number % 25 == 0 or number == len(rows):
                progress(number)
    if digest(module) != state or module.training or sha(checkpoint) != weight_sha:
        raise ValueError('Inference model or checkpoint changed')
    if configuration != config_identity(args.model) or sources != check_sources():
        raise ValueError('Source/configuration changed during export')
    (args.out / 'preds').mkdir()
    with (args.out / 'preds/results.pkl').open('xb') as stream:
        pickle.dump(predictions, stream, protocol=4)
    frame_ids = [r['id'] for r in rows]
    write_new(args.out / 'task.json', dict(model=args.model, split=args.split,
        annotations=str(DATA / args.split / 'annfiles'), images=str(DATA / args.split / 'images'),
        checkpoint=str(checkpoint), checkpoint_sha256=weight_sha, frames=len(rows), frame_ids=frame_ids))
    write_new(args.out / 'export_receipt.json', dict(passed=True, model=args.model, split=args.split,
        checkpoint_sha256=weight_sha, configuration=configuration, sources=sources,
        manifest_sha256=MANIFEST_SHA, runtime_identity=identity,
        frames=len(rows), frame_ids=frame_ids, results_sha256=sha(args.out / 'preds/results.pkl'),
        initialized_state_sha256=state, final_state_sha256=state,
        geometry='original pixels, radians, canonical le90 for YOLO, native GauCho OBB',
        selection_lock_sha256=sha(args.selection_lock) if args.selection_lock else None,
        physical_gpu=selected['index'] if args.device != 'cpu' else None,
        peak_allocated_mib=torch.cuda.max_memory_allocated(index)/2**20 if args.device != 'cpu' else None,
        peak_reserved_mib=torch.cuda.max_memory_reserved(index)/2**20 if args.device != 'cpu' else None,
        training_updates=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('gaucho', 'yolo26'), required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--physical-gpus', type=int, nargs=2, default=[2, 3])
    parser.add_argument('--selection-lock', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(model=args.model, split=args.split, executed=False,
            configuration=config_identity(args.model), training_updates=0), indent=2))
        return
    if args.checkpoint is None or args.out is None:
        parser.error('--execute requires --checkpoint and --out')
    export(args)


if __name__ == '__main__':
    main()
