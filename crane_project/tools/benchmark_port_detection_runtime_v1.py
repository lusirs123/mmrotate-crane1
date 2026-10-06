#!/usr/bin/env python3
"""Four frozen TEST arms: EOOD, SymEOOD, SymEOOD+B, and sigma1.5 midpoint.

CPU --stage check loads only sources/protocol. Actual timing is single-GPU,
batch1 FP32. Frozen outputs are checked outside timers, no GT scoring/selection.
CPU --stage inputs diagnoses existing file metadata/SHA without tensor loading.
"""
import argparse
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_detection_runtime_v1 as c

PROTOCOL = ROOT/'crane_project/tools/port_detection_runtime_v1_protocol.json'
SOURCES = ROOT/'crane_project/tools/port_detection_runtime_v1_sources.json'
EOOD_DIR = ROOT/'work_dirs/crane_eood_k1_port_day2night_v1'
SYM_DIR = ROOT/'work_dirs/crane_symeood_k1_port_day2night_v1'
B_WEIGHT = ROOT/'work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth'
M_WEIGHT = ROOT/'work_dirs/crane_symeood_k1_port_day2night_midpoint_sigma_v1/sigma_1p5/head_epoch_03.pth'
DATA = ROOT/'crane_project/data/crane_grab_port_day2night_v1'
SOURCE_PATHS = (
    'crane_project/tools/benchmark_port_detection_runtime_v1.py',
    'crane_project/tools/port_detection_runtime_v1_protocol.json',
    'crane_project/utils/port_detection_runtime_v1.py',
    'tests/test_port_detection_runtime_v1.py',
    'crane_project/configs/crane_eood_k1.py', c.EOOD_CONFIG,
    'crane_project/configs/crane_symeood_k1.py',
    'crane_project/configs/crane_symeood_k1_port_day2night_v1.py', c.B_CONFIG,
    'configs/_base_/default_runtime.py',
    'configs/_base_/schedules/schedule_1x.py',
    'mmrotate/models/detectors/base.py',
    'mmrotate/models/detectors/single_stage.py',
    'mmrotate/models/detectors/eood.py',
    'mmrotate/models/detectors/sym_eood_detector.py',
    'mmrotate/models/dense_heads/rotated_anchor_head.py',
    'mmrotate/models/dense_heads/rotated_retina_head.py',
    'mmrotate/models/dense_heads/eood_head.py',
    'mmrotate/models/dense_heads/rotated_eood_head.py',
    'mmrotate/models/dense_heads/sym_eood_head.py',
    'mmrotate/models/dense_heads/rotated_atss_head.py',
    'mmrotate/core/anchor/anchor_generator.py',
    'mmrotate/core/bbox/coder/delta_xywha_rbbox_coder.py',
    'mmrotate/core/bbox/transforms.py',
    'mmrotate/core/post_processing/bbox_nms_rotated.py',
    'mmrotate/datasets/pipelines/transforms.py',
    'crane_project/utils/port_geometry_refine_g_v1.py',
    'crane_project/utils/port_structure_reliability_v1.py',
    'crane_project/utils/port_geometry_midpoint_v1.py',
    'crane_project/utils/port_geometry_midpoint_sigma_v1.py',
)


def source_contract():
    source = c.read(SOURCES)
    if (source['protocol'] != c.VERSION or set(source['sources']) != set(SOURCE_PATHS) or
            c.read(PROTOCOL) != c.protocol_document()):
        raise ValueError('Timing protocol/source identity differs')
    actual = {name: c.sha(ROOT/name) for name in source['sources']}
    if actual != source['sources']:
        raise ValueError('Timing source SHA differs; synchronize all listed files')
    return dict(protocol_sha256=c.sha(PROTOCOL), sources_sha256=c.sha(SOURCES), sources=actual)


def inspect_detector_inputs(arm, model_dir, sweep_dir=None, test_dir=None):
    """Bounded existing-source diagnosis, including inconsistent legacy files.

    Reports candidates independently; never picks a different epoch/config,
    executes saved configs or unpickles outputs. Source failures stay failures.
    """
    spec = c.detector_identity(arm)
    key = 'epoch_%d' % spec['selected_epoch']
    weight = model_dir/(key+'.pth')
    config = ROOT/spec['config']
    weight_sha = c.sha(weight) if weight.is_file() else None
    config_sha = c.sha(config)
    result = dict(arm=arm, fixed_identity=spec, model_dir=str(model_dir.resolve()),
        checkpoint=str(weight.resolve()), checkpoint_exists=weight.is_file(),
        checkpoint_sha256=weight_sha, config=str(config.resolve()), config_sha256=config_sha,
        candidates=[], tensor_loads=0, prediction_unpickles=0, gpu_calls=0)
    directories = [Path(sweep_dir)] if sweep_dir else [model_dir/'val_sweep_port_v1', model_dir/'ckpt_sweep']
    seen = set()
    for directory in directories:
        directory = directory.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        path = directory/'sweep_results.json'
        item = dict(sweep=str(directory), selection_exists=path.is_file())
        if not path.is_file():
            result['candidates'].append(item)
            continue
        try:
            selection = c.read(path)
            checks = c.selection_checks(selection, config_sha, weight_sha, spec['selected_epoch'])
            records = selection.get('all_checkpoints', {})
            record = records.get(key, {}) if isinstance(records, dict) else {}
            record = record if isinstance(record, dict) else {}
            item.update(selection_sha256=c.sha(path), selection_checks=checks,
                failed_fields=[name for name, value in checks.items() if not value['passed']],
                recorded_selection=dict(selected_path=selection.get('selected_path'),
                    checkpoint_path=record.get('checkpoint'), selection_config=selection.get('selection_config'),
                    candidate_epochs=selection.get('candidate_epochs'),
                    metric_protocol_version=selection.get('metric_protocol_version')))
            for field, value in (('selected_path', selection.get('selected_path')), ('checkpoint_path', record.get('checkpoint'))):
                target = Path(value).resolve() if isinstance(value, str) and value else None
                item[field+'_matches_fixed_path'] = target == weight.resolve()
                item[field+'_resolved'] = str(target) if target is not None else None
            marker = directory/'selected_checkpoint.txt'
            if marker.is_file():
                item['selected_checkpoint_txt'] = marker.read_text()[:4096].strip()
            final = Path(test_dir).resolve() if test_dir else directory/'final_test'/key
            item['expected_test_dir'] = str(final)
            receipts = {}
            # Legacy reports are identified, not accepted instead of the v2 receipt.
            for name in ('final_test_metrics_v2.json', 'final_test_metrics.json'):
                receipt_path = final/name
                if not receipt_path.is_file():
                    receipts[name] = dict(exists=False)
                    continue
                receipt = c.read(receipt_path)
                fields = ('protocol', 'evidence_role', 'config', 'config_sha256', 'checkpoint',
                    'checkpoint_sha256', 'results_pkl', 'results_pkl_sha256', 'frame_count',
                    'metric_protocol_version')
                receipts[name] = dict(exists=True, sha256=c.sha(receipt_path),
                    fields={field:receipt.get(field) for field in fields})
            item['test_receipts'] = receipts
            predictions = final/'preds/results.pkl'
            item['prediction_file'] = dict(path=str(predictions), exists=predictions.is_file(),
                sha256=c.sha(predictions) if predictions.is_file() else None, unpickled=False)
        except (ValueError, TypeError, OSError, AttributeError) as error:
            item['read_error'] = dict(type=type(error).__name__, error=str(error))
        result['candidates'].append(item)
    return result


def input_diagnosis(args):
    result = dict(eood=inspect_detector_inputs('eood', EOOD_DIR, args.eood_sweep_dir, args.eood_test_dir),
        symeood=inspect_detector_inputs('symeood', SYM_DIR, args.symeood_sweep_dir, args.symeood_test_dir),
        fixed_weights={name:dict(path=str(path.resolve()), exists=path.is_file(),
            expected_sha256=expected, actual_sha256=c.sha(path) if path.is_file() else None)
            for name, path, expected in (('b', B_WEIGHT, c.B_SHA), ('midpoint', M_WEIGHT, c.M_SHA))},
        selection_on_test=False, source_files_modified=False, cuda_calls=0,
        model_loads=0, tensor_loads=0, prediction_unpickles=0, image_reads=0, annotation_reads=0)
    for arm in ('eood', 'symeood'):
        for candidate in result[arm]['candidates']:
            print('INPUT', arm, candidate['sweep'], 'failed selection fields:',
                  candidate.get('failed_fields', 'missing/unreadable selection'), flush=True)
            for name in candidate.get('failed_fields', []):
                print(' ', name, candidate['selection_checks'][name], flush=True)
    return result


def checked_detector_inputs(arm, model_dir, sweep_dir=None, test_dir=None):
    spec = c.detector_identity(arm)
    key = 'epoch_%d' % spec['selected_epoch']
    sweep = c.unique_directory([sweep_dir] if sweep_dir else
        [model_dir/'val_sweep_port_v1', model_dir/'ckpt_sweep'], ['sweep_results.json'])
    weight = model_dir/(key+'.pth')
    weight_sha = c.sha(weight)
    config_sha = c.sha(ROOT/spec['config'])
    selection = c.read(sweep/'sweep_results.json')
    try:
        c.check_selection(selection, config_sha, weight_sha, spec['selected_epoch'])
    except ValueError as error:
        raise ValueError(arm+' '+str(sweep/'sweep_results.json')+': '+str(error)) from error
    for path in (selection['selected_path'], selection['all_checkpoints'][key]['checkpoint']):
        if Path(path).resolve() != weight.resolve():
            raise ValueError(arm+' selection points to a different checkpoint')
    final = c.unique_directory([test_dir] if test_dir else
        [sweep/'final_test'/key], ['final_test_metrics_v2.json', 'preds/results.pkl'])
    receipt = c.read(final/'final_test_metrics_v2.json')
    predictions = final/'preds/results.pkl'
    if (receipt['evidence_role'] != 'fixed_test_after_source_val_selection' or
            receipt['config_sha256'] != config_sha or
            receipt['checkpoint_sha256'] != weight_sha or receipt['frame_count'] != 1440 or
            receipt['results_pkl_sha256'] != c.sha(predictions)):
        raise ValueError(arm+' sealed TEST reference identity differs')
    # Read references only, never recompute their GT metrics or selection.
    with predictions.open('rb') as stream:
        references = pickle.load(stream)
    if len(references) != 1440:
        raise ValueError(arm+' reference frame count differs')
    proof = dict(arm=arm, selected_epoch=spec['selected_epoch'], config_sha256=config_sha,
        checkpoint_sha256=weight_sha, selection_sha256=c.sha(sweep/'sweep_results.json'),
        test_report_sha256=c.sha(final/'final_test_metrics_v2.json'),
        reference_sha256=c.sha(predictions), sweep=str(sweep), test=str(final))
    return references, weight, proof


def inputs(args):
    eood_refs, eood_weight, eood_proof = checked_detector_inputs(
        'eood', EOOD_DIR, args.eood_sweep_dir, args.eood_test_dir)
    sym_refs, sym_weight, sym_proof = checked_detector_inputs(
        'symeood', SYM_DIR, args.symeood_sweep_dir, args.symeood_test_dir)
    roots = [ROOT/'work_dirs', ROOT/'work_dirs/port_results/geometry']
    roots += sorted((ROOT/'work_dirs').glob('port_geometry_midpoint_sigma15_v1_results_*'))
    mdir = c.unique_directory([args.midpoint_test_dir] if args.midpoint_test_dir else
        [p/'port_geometry_midpoint_sigma15_v1_test_eval' for p in roots],
        ['completion.json', 'artifacts.json', 'test_rows.jsonl'])
    index = c.read(mdir/'artifacts.json')['files']
    for name in ('completion.json', 'test_rows.jsonl'):
        if c.sha(mdir/name) != index[name]:
            raise ValueError('Midpoint sealed TEST artifact differs')
    report = c.read(mdir/'completion.json')
    fixed = report['selection']['selected_checkpoint']
    if (report['status'] != 'FROZEN_SIGMA15_TEST_COMPLETE_REVIEW_REQUIRED' or
            report['frames'] != 1440 or fixed['epoch'] != 3 or fixed['sigma_cells'] != 1.5 or
            fixed['updates'] != 2706 or fixed['sha256'] != c.M_SHA or
            report['state_before'] != report['state_after'] or
            c.sha(mdir/'test_rows.jsonl') != c.M_ROWS_SHA or
            c.sha(B_WEIGHT) != c.B_SHA or c.sha(M_WEIGHT) != c.M_SHA):
        raise ValueError('Requires sealed B24 + sigma1.5/epoch03, not a new candidate')
    rows = [json.loads(line) for line in (mdir/'test_rows.jsonl').read_text().splitlines() if line.strip()]
    if (len(rows) != 1440 or len({r['image'] for r in rows}) != 1440 or
            Counter(r['sequence'] for r in rows) != Counter(c.COUNTS) or
            [r['image'] for r in rows] != sorted(r['image'] for r in rows) or
            any(r['scale'] != 1. for r in rows)):
        raise ValueError('Sealed standard-scale TEST frame identity differs')
    frame_hash = hashlib.sha256()
    for row in rows:
        image = DATA/'test/images'/(row['image']+'.jpg')
        frame_hash.update(image.name.encode()); frame_hash.update(b'\0')
        frame_hash.update(bytes.fromhex(c.sha(image)))
    if frame_hash.hexdigest() != c.IMAGE_SHA:
        raise ValueError('Fixed TEST image bytes differ')
    proof = dict(eood=eood_proof, symeood=sym_proof, b_checkpoint_sha256=c.B_SHA,
        midpoint_checkpoint_sha256=c.M_SHA, image_identity_sha256=c.IMAGE_SHA,
        midpoint_reference_sha256=c.M_ROWS_SHA,
        midpoint_test_report_sha256=c.sha(mdir/'completion.json'),
        midpoint_test=str(mdir),
        frames=1440, annotation_reads=0, gt_scoring=False)
    references = dict(eood=eood_refs, symeood=sym_refs)
    weights = dict(eood=eood_weight, symeood=sym_weight, symeood_b=B_WEIGHT,
                   symeood_b_midpoint=B_WEIGHT)
    return rows, references, weights, report, proof


def flatten(value, np):
    leaves = []
    def visit(x):
        if isinstance(x, np.ndarray):
            if x.ndim != 2 or x.shape[1] != 6:
                raise ValueError('Expected native Nx6 single-class boxes')
            leaves.append(x)
        elif isinstance(x, (list, tuple)):
            for item in x:
                visit(item)
        else:
            raise ValueError('Unknown native output type')
    visit(value)
    return np.concatenate(leaves, axis=0) if leaves else np.empty((0, 6), dtype=np.float32)


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        x = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(x.dtype).encode())
        digest.update(str(tuple(x.shape)).encode()); digest.update(x.numpy().tobytes())
    return digest.hexdigest()


def host_info():
    info = dict(os=platform.platform(), python=platform.python_version(),
                processor=platform.processor(), logical_cpus=os.cpu_count())
    cpu = Path('/proc/cpuinfo')
    if cpu.is_file():
        for line in cpu.read_text().splitlines():
            if line.startswith('model name'):
                info['processor'] = line.split(':', 1)[1].strip()
                break
    memory = Path('/proc/meminfo')
    if memory.is_file():
        for line in memory.read_text().splitlines():
            if line.startswith('MemTotal:'):
                info['host_ram_mib'] = int(line.split()[1])/1024
                break
    return info


class Pipeline:
    """One native head and one feature extraction; no reference/GT inputs."""
    def __init__(self, detector, head, torch, np, geometry):
        self.detector, self.head = detector, head
        self.torch, self.np, self.g = torch, np, geometry

    def __call__(self, image, metas):
        if self.head is None and not hasattr(self.detector, 'simple_test_from_features'):
            # Preserve EOOD's native shared towers, predictor0 and NMS.
            return flatten(self.detector.simple_test(image, metas, rescale=True), self.np)
        features = self.detector.extract_feat(image)
        if self.head is None:
            # Standalone B keeps its original native rescale path and performs
            # no ROI extraction or CPU -> CUDA box round trip for midpoint.
            result = flatten(self.detector.simple_test_from_features(features, metas, rescale=True), self.np)
            if len(result) > 1:
                raise ValueError('Frozen SymEOOD must retain max_per_img=1')
            return result
        raw = flatten(self.detector.simple_test_from_features(features, metas, rescale=False), self.np)
        if raw.shape[0] > 1:
            raise ValueError('Frozen SymEOOD must retain max_per_img=1')
        b = image.new_tensor(raw)
        original = self.torch.cat((self.g.map_boxes(b[:, :5], metas[0], inverse=True), b[:, 5:]), 1)
        roi, support, _ = self.g.sample_local(features[0], b[:, :5], metas[0], 'aligned')
        scales = original.new_tensor(self.np.asarray(metas[0]['scale_factor']).reshape(-1)[:2]).reshape(1, 2)
        scales = scales.expand(len(original), -1)
        result = self.head(roi, support, original, b[:, :5], scales)
        return result['boxes_original'].cpu().numpy()


def build(arm, gpu, weights, reference_report, torch, Config):
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from crane_project.utils import port_geometry_refine_g_v1 as geometry
    from crane_project.utils.port_geometry_midpoint_sigma_v1 import SigmaMidpointHead
    spec_identity = c.detector_identity(arm)
    cfg = Config.fromfile(str(ROOT/spec_identity['config']))
    import_modules_from_strings(**cfg.custom_imports)
    spec = deepcopy(cfg.model); spec.pretrained = None
    if arm != 'eood':
        spec.train_cfg = None
    detector = build_detector(spec)
    weight = weights[arm]
    payload = load_checkpoint(detector, str(weight), map_location='cpu', strict=True)
    meta = c.check_checkpoint_meta(payload['meta'], cfg.to_dict(), spec_identity['selected_epoch'])
    del payload
    detector.cuda(gpu).eval().requires_grad_(False)
    head = None
    if arm == 'symeood_b_midpoint':
        payload = torch.load(str(M_WEIGHT), map_location='cpu')
        fixed = reference_report['selection']['selected_checkpoint']
        if (payload['sigma_cells'] != 1.5 or payload['epoch'] != 3 or payload['updates'] != 2706 or
                payload['head_digest'] != fixed['head_digest'] or
                payload['frozen_b']['checkpoint_sha256'] != c.B_SHA):
            raise ValueError('Midpoint checkpoint payload identity differs')
        head = SigmaMidpointHead(1.5).cuda(gpu)
        head.load_state_dict(payload['head_state'], strict=True)
        head.eval().requires_grad_(False)
        if sum(p.numel() for p in head.parameters()) != 17696:
            raise ValueError('Midpoint structure differs')
        del payload
    return detector, head, cfg, geometry, meta


def transform(cfg):
    from mmdet.datasets.pipelines import Compose
    # The standard ndarray inference path: decode once outside the app timer.
    pipeline = deepcopy(cfg.data.test.pipeline)
    if pipeline[0]['type'] != 'LoadImageFromFile':
        raise ValueError('Requires original image TEST pipeline')
    pipeline[0]['type'] = 'LoadImageFromWebcam'
    return Compose(pipeline)


def prepare(compose, frame, gpu):
    from mmcv.parallel import collate, scatter
    data = scatter(collate([compose(dict(img=frame))], samples_per_gpu=1), [gpu])[0]
    image, metas = data['img'][0], data['img_metas'][0]
    if (len(data['img']) != 1 or tuple(image.shape) != (1, 3, 1024, 1024) or
            str(image.dtype) != 'torch.float32' or len(metas) != 1 or metas[0].get('flip', False)):
        raise ValueError('Requires original single-scale 1024, batch1, no flip TEST input')
    return image, metas


def run(args):
    identity = source_contract()
    out = Path(args.out_dir).resolve()
    if out == ROOT/'work_dirs' or ROOT/'work_dirs' not in out.parents:
        raise ValueError('Use a new child directory under project work_dirs')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=c.VERSION, stage=args.stage, identity=identity,
        training_updates=0, selection_on_test=False, gt_scoring=False, test_repeatedly_exposed=True)
    c.write_new(out/'protocol.json', c.protocol_document())
    c.write_new(out/'sources.json', c.read(SOURCES))
    try:
        if args.stage == 'check':
            report['status'] = 'STATIC_RUNTIME_CONTRACT_PASS_NO_MODEL_DATA_GPU'
        elif args.stage == 'inputs':
            report['input_diagnosis'] = input_diagnosis(args)
            report['status'] = 'INPUT_METADATA_DIAGNOSIS_COMPLETE_REVIEW_REQUIRED'
            c.write_new(out/'input_diagnosis.json', report)
        else:
            import cv2
            import numpy as np
            import torch
            import mmcv
            import mmdet
            import mmrotate
            from mmcv import Config
            if int(os.environ.get('WORLD_SIZE', '1')) != 1 or not torch.cuda.is_available():
                raise ValueError('Use one CUDA process, not torchrun/DDP')
            if not 0 <= args.gpu < torch.cuda.device_count():
                raise ValueError('Invalid logical CUDA device')
            torch.cuda.set_device(args.gpu)
            torch.set_num_threads(1); cv2.setNumThreads(1)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            rows, references, weights, sealed, proof = inputs(args)
            report['inputs'] = proof
            props = torch.cuda.get_device_properties(args.gpu)
            report['host'] = host_info()
            report['runtime'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), mmcv=mmcv.__version__, mmdet=mmdet.__version__,
                mmrotate=mmrotate.__version__, opencv=cv2.__version__, numpy=np.__version__,
                gpu=props.name, gpu_total_mib=props.total_memory/2**20, cpu_threads=1,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), logical_gpu=args.gpu,
                warmup_frames=50, repeats=3, batch_size=1, precision='FP32')
            # Keep the original runtime for a fair precision/output identity check.
            for key in ('torch', 'cuda', 'cudnn', 'mmcv', 'mmdet', 'mmrotate', 'opencv'):
                if report['runtime'][key] != sealed['runtime'][key]:
                    raise ValueError('Runtime differs from sealed inference: '+key)
            arms = {}
            with (out/'timings.jsonl').open('x', encoding='utf-8') as log:
                for arm in c.ARMS:
                    torch.cuda.empty_cache()
                    detector, head, cfg, geometry, meta = build(arm, args.gpu, weights, sealed, torch, Config)
                    if arm != 'eood' and cfg.data.test.pipeline != eood_pipeline:
                        raise ValueError('EOOD and SymEOOD TEST transforms differ')
                    if arm == 'eood':
                        eood_pipeline = deepcopy(cfg.data.test.pipeline)
                    if arm == 'symeood':
                        sym_structure = c.plain(cfg.model)
                    elif arm.startswith('symeood_b') and c.plain(cfg.model) != sym_structure:
                        raise ValueError('B must not add or change the deployed SymEOOD structure')
                    pipe = Pipeline(detector, head, torch, np, geometry)
                    compose = transform(cfg)
                    before = dict(detector=state_sha(detector), head=state_sha(head) if head is not None else None)
                    total_params = sum(p.numel() for p in detector.parameters())+(17696 if head is not None else 0)
                    if any(m.training for m in detector.modules()) or any(p.requires_grad for p in detector.parameters()):
                        raise ValueError('Detector is not frozen')
                    worst = 0.
                    with torch.no_grad():
                        for index in range(50):
                            frame = cv2.imread(str(DATA/'test/images'/(rows[index]['image']+'.jpg')))
                            if frame is None:
                                raise ValueError('Cannot decode warmup image')
                            image, metas = prepare(compose, frame, args.gpu)
                            pipe(image, metas)
                        torch.cuda.synchronize(args.gpu)
                        torch.cuda.reset_peak_memory_stats(args.gpu)
                        records = []
                        for repeat in range(3):
                            for index, row in enumerate(rows):
                                file_start = time.perf_counter()
                                frame = cv2.imread(str(DATA/'test/images'/(row['image']+'.jpg')))
                                if frame is None:
                                    raise ValueError('Cannot decode TEST image')
                                torch.cuda.synchronize(args.gpu)
                                app_start = time.perf_counter()
                                image, metas = prepare(compose, frame, args.gpu)
                                torch.cuda.synchronize(args.gpu)
                                model_start = time.perf_counter()
                                result = pipe(image, metas)
                                torch.cuda.synchronize(args.gpu)
                                finish = time.perf_counter()
                                # All source/ref matching is outside every timer.
                                if arm in references:
                                    expected = flatten(references[arm][index], np).tolist()
                                else:
                                    value = row['b' if arm == 'symeood_b' else 'midpoint']
                                    expected = [] if value is None else [value]
                                worst = max(worst, c.check_boxes(result.tolist(), expected))
                                record = dict(arm=arm, repeat=repeat+1, image=row['image'], sequence=row['sequence'],
                                    native_candidates=len(result), decoded_frame_to_obb=finish-app_start,
                                    model_and_postprocess=finish-model_start, file_to_obb=finish-file_start,
                                    preprocessing_h2d=model_start-app_start)
                                records.append(record)
                                log.write(json.dumps(record, allow_nan=False)+'\n')
                            log.flush()
                            print(arm, 'repeat', repeat+1, '/3 complete; sealed outputs match', flush=True)
                    after = dict(detector=state_sha(detector), head=state_sha(head) if head is not None else None)
                    if before != after:
                        raise ValueError('Inference changed frozen parameters/buffers')
                    groups = {}
                    for group in ('overall', 'real', 'sim', *c.COUNTS):
                        selected = [r for r in records if group == 'overall' or r['sequence'] == group or r['sequence'].startswith(group+'_')]
                        groups[group] = {field:c.timing_summary([r[field] for r in selected]) for field in
                            ('decoded_frame_to_obb', 'model_and_postprocess', 'file_to_obb', 'preprocessing_h2d')}
                    per_repeat = {str(k):c.timing_summary([r['decoded_frame_to_obb'] for r in records if r['repeat'] == k]) for k in (1,2,3)}
                    arms[arm] = dict(groups=groups, per_repeat=per_repeat, state_before=before, state_after=after,
                        verified_predictions=4320, max_box_component_difference=worst, detector_checkpoint_meta=meta,
                        registered_parameters_including_training_aux=total_params,
                        midpoint_parameters=17696 if head is not None else 0,
                        gpu_peak_mib=dict(allocated=torch.cuda.max_memory_allocated(args.gpu)/2**20,
                            reserved=torch.cuda.max_memory_reserved(args.gpu)/2**20))
                    print(arm, groups['overall']['decoded_frame_to_obb'], flush=True)
                    del pipe, compose, detector, head, image, result, records
                    gc.collect(); torch.cuda.empty_cache()
            if (c.sha(B_WEIGHT) != c.B_SHA or c.sha(M_WEIGHT) != c.M_SHA or
                    any(c.sha(weights[arm]) != proof[arm]['checkpoint_sha256'] for arm in references)):
                raise ValueError('Weight bytes changed during benchmark')
            if source_contract() != identity:
                raise ValueError('Source bytes changed during benchmark')
            report['arms'] = arms
            b = arms['symeood_b']['groups']['overall']['decoded_frame_to_obb']['mean_ms']
            m = arms['symeood_b_midpoint']['groups']['overall']['decoded_frame_to_obb']['mean_ms']
            report['midpoint_overhead'] = dict(mean_ms=m-b, relative_percent=100*(m/b-1),
                note='Difference of same-condition measured complete pipelines, not isolated CUDA kernel time.')
            report['adjacent_comparisons'] = {}
            for left, right in zip(c.ARMS, c.ARMS[1:]):
                x = arms[left]['groups']['overall']['decoded_frame_to_obb']['mean_ms']
                y = arms[right]['groups']['overall']['decoded_frame_to_obb']['mean_ms']
                report['adjacent_comparisons'][left+' -> '+right] = dict(
                    mean_ms_difference=y-x, relative_percent=100*(y/x-1))
            report['status'] = 'FROZEN_TEST_RUNTIME_COMPLETE_REVIEW_REQUIRED'
            c.write_new(out/'runtime_compare.json', report)
        c.write_new(out/'completion.json', report)
        c.write_new(out/'artifacts.json', dict(protocol=c.VERSION,
            files={p.name:c.sha(p) for p in out.iterdir() if p.is_file() and p.name != 'artifacts.json'}))
        print('Saved', out/'completion.json', 'status', report['status'], flush=True)
    except Exception as error:
        c.write_new(out/'failure.json', dict(protocol=c.VERSION, status='FAILED_NOT_A_VALID_SPEED_RESULT',
            error_type=type(error).__name__, error=str(error)))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('check', 'inputs', 'benchmark'), required=True)
    p.add_argument('--gpu', type=int, default=0, help='Logical CUDA device after CUDA_VISIBLE_DEVICES')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--eood-sweep-dir', help='Existing source-VAL sweep directory; no reselection')
    p.add_argument('--eood-test-dir', help='Existing EOOD epoch24 final_test directory')
    p.add_argument('--symeood-sweep-dir', help='Existing unaugmented SymEOOD source-VAL sweep; no reselection')
    p.add_argument('--symeood-test-dir', help='Existing unaugmented SymEOOD epoch20 final_test directory')
    p.add_argument('--midpoint-test-dir', help='Existing sealed sigma1.5 TEST evaluation directory')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
