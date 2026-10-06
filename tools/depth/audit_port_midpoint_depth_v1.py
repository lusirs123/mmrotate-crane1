#!/usr/bin/env python3
"""Train-03 only: frozen B24 vs sigma1.5/epoch03 through unchanged Raw-opt."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_midpoint_depth_v1 as c

PROTOCOL = ROOT/'tools/depth/port_midpoint_depth_v1_protocol.json'
SOURCES = ROOT/'tools/depth/port_midpoint_depth_v1_sources.json'
PARENT = ROOT/'crane_project/tools/port_geometry_midpoint_sigma15_v1_eval_sources.json'
EXTRA = {
    'crane_project/utils/port_midpoint_depth_v1.py',
    'tools/depth/audit_port_midpoint_depth_v1.py',
    'tools/depth/port_midpoint_depth_v1_protocol.json',
    'tests/test_port_midpoint_depth_v1.py',
    c.CALIBRATION,
    'docs/data/webots_depth_formal_frame_index_v1.manifest.json',
    'tools/depth/evaluate_detector_obb_depth.py',
    'tools/depth/fit_obb_dual_scale.py',
    str(PARENT.relative_to(ROOT)),
}


def checked_sources():
    original = json.loads(PARENT.read_text())
    manifest = json.loads(SOURCES.read_text())
    paths = set(original['sources']) | EXTRA
    if (manifest.get('protocol') != c.VERSION or set(manifest.get('sources', {})) != paths
            or manifest.get('parent_manifest_sha256') != c.sha(PARENT)
            or json.loads(PROTOCOL.read_text()) != c.protocol_document()):
        raise ValueError('Reviewed depth source/protocol closure differs')
    actual = {p:c.sha(ROOT/p) for p in sorted(paths)}
    if actual != manifest['sources'] or any(actual[p] != v for p, v in original['sources'].items()):
        raise ValueError('A reviewed source has changed')
    if c.sha(ROOT/c.CALIBRATION) != c.CAL_SHA:
        raise ValueError('Original Raw-opt calibration bytes differ')
    calibration = json.loads((ROOT/c.CALIBRATION).read_text())
    if (calibration['coordinate_contract']['id'] != 'raw_opt_v1'
            or calibration['coordinate_contract']['target'] != 'z_cg_opt_m'
            or calibration['parameters']['fit_offset'] is not False):
        raise ValueError('Original Raw-opt contract differs')
    return dict(protocol=c.VERSION, source_manifest_sha256=c.sha(SOURCES),
        parent_manifest_sha256=c.sha(PARENT), calibration_sha256=c.CAL_SHA,
        sources=actual), calibration


def checked_dataset(root):
    root = Path(root).resolve()
    sequence = root/c.SEQUENCE
    manifest_path = sequence/'metadata/manifest.json'
    truth_path = sequence/'metadata/frames.jsonl'
    if c.sha(manifest_path) != c.MANIFEST_SHA or c.sha(truth_path) != c.TRUTH_SHA:
        raise ValueError('Train-03 metadata bytes differ; no replacement/refit allowed')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('sequence_id') != c.SEQUENCE or manifest.get('split') != 'calibration_train':
        raise ValueError('This entry admits calibration-train only')
    index = json.loads((ROOT/'docs/data/webots_depth_formal_frame_index_v1.manifest.json').read_text())
    expected = index['constant_contract']
    if (manifest['camera'] != expected['camera']
            or manifest['obb_reference_geometry']['long_edge_mean_m'] != expected['obb_reference_geometry']['long_edge_mean_m']
            or manifest['obb_reference_geometry']['short_edge_mean_m'] != expected['obb_reference_geometry']['short_edge_mean_m']):
        raise ValueError('Original physical camera/top-beam contract differs')
    images = c.image_sources(root)
    return manifest, truth_path, images, dict(root=str(root), sequence_id=c.SEQUENCE,
        manifest_sha256=c.MANIFEST_SHA, frames_jsonl_sha256=c.TRUTH_SHA,
        image_identity_sha256=c.IMAGE_SHA, frame_count=c.COUNT)


def load_truth(path, manifest):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    c.validate_truth(rows, manifest)
    return rows


def load_native(b_path, head_path, gpu):
    """Read SHA-bound weights only in audit. Never change global source paths."""
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from crane_project.tools import eval_port_geometry_midpoint_sigma15_v1 as sealed
    from crane_project.tools.diagnose_port_shape_e_h_train_gradients_v1 import checkpoint_contract
    torch, g = sealed.torch, sealed.g
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Single-GPU batch1 only; no distributed launcher')
    if not torch.cuda.is_available() or not 0 <= gpu < torch.cuda.device_count():
        raise ValueError('A valid logical CUDA device is required')
    torch.cuda.set_device(gpu)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    cfg = g.check_cfg()
    if c.sha(b_path) != c.B_SHA or c.sha(head_path) != c.HEAD_SHA:
        raise ValueError('Fixed weights changed before native loading')
    payload = torch.load(str(head_path), map_location='cpu')
    c.validate_head_header(payload, g.ready.FROZEN_B)
    import_modules_from_strings(**cfg.custom_imports)
    spec = deepcopy(cfg.model)
    spec.pretrained = None
    spec.train_cfg = None
    detector = build_detector(spec)
    checkpoint = load_checkpoint(detector, str(b_path), map_location='cpu', strict=True)
    checkpoint_contract(checkpoint['meta'], cfg, 'b')
    del checkpoint
    g.freeze_detector(detector.cuda(gpu))
    if g.state_digest(detector) != payload['context']['detector_state']:
        raise ValueError('B tensor state is not the sealed midpoint training context')
    head = sealed.t.model.SigmaMidpointHead(1.5).to(next(detector.parameters()).device)
    head.load_state_dict(payload['head_state'], strict=True)
    head.eval().requires_grad_(False)
    if g.state_digest(head) != c.HEAD_DIGEST:
        raise ValueError('Midpoint tensor digest differs')
    runtime = sealed.f.runtime(gpu)
    expected_runtime = payload['context']['runtime']
    fields = ('torch', 'cuda', 'cudnn', 'mmcv', 'mmdet', 'mmrotate', 'opencv')
    if any(runtime.get(k) != expected_runtime.get(k) for k in fields):
        raise ValueError('Native software differs from the sealed head runtime')
    from mmdet.datasets.pipelines import Compose
    if cfg.data.val.pipeline != cfg.data.test.pipeline:
        raise ValueError('Original B single-scale inference pipelines differ')
    pipeline = Compose(deepcopy(cfg.data.val.pipeline))
    return detector, head, pipeline, sealed.old, runtime


def native_view(pipeline, source, gpu, helpers):
    from mmcv.parallel import collate, scatter
    value = pipeline(dict(img_info=dict(filename=source['image_path']), img_prefix=None))
    if value is None:
        raise ValueError('Original B inference pipeline skipped a frame')
    value = scatter(collate([value], samples_per_gpu=1), [gpu])[0]
    if len(value['img']) != 1 or len(value['img_metas']) != 1:
        raise ValueError('Expected one deterministic inference view')
    image, metas = value['img'][0], value['img_metas'][0]
    if (tuple(image.shape) != (1, 3, 1024, 1024) or len(metas) != 1
            or Path(metas[0]['filename']).resolve() != Path(source['image_path']).resolve()
            or tuple(metas[0]['ori_shape'][:2]) != (1024, 1024) or metas[0].get('flip', False)):
        raise ValueError('Webots original coordinates/view identity differs')
    helpers.g.checked_meta(metas[0])
    return image, metas


def native_predictions(out, images, b_path, head_path, gpu):
    detector, head, pipeline, helpers, runtime = load_native(b_path, head_path, gpu)
    g, torch = helpers.g, helpers.torch
    initial_detector, initial_head = g.state_digest(detector), g.state_digest(head)
    counters = dict(feature_extractions=0, native_head_calls=0)
    def audit(event):
        if event['stage'] == 'test_feature_extracted':
            counters['feature_extractions'] += 1
        elif event['stage'] == 'test_native_head_called':
            counters['native_head_calls'] += 1
        else:
            raise ValueError('Undeclared inference event')
    predictions = []
    start = time.monotonic()
    # A sealed stream is finished before metric truth is parsed/evaluated.
    with (out/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for index, source in enumerate(images):
            if c.sha(source['image_path']) != source['sha256']:
                raise ValueError('Image bytes changed after preflight')
            image, metas = native_view(pipeline, source, gpu, helpers)
            result = helpers.capture(detector, head, image, metas, audit)
            result.update(frame_id=source['frame_id'], image_sha256=source['sha256'],
                original_size_hw=list(metas[0]['ori_shape'][:2]),
                scale_xy=[float(v) for v in metas[0]['scale_factor'][:2]])
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            predictions.append(result)
            if index == 0 or (index+1) % 25 == 0 or index+1 == len(images):
                print('inference %d/%d elapsed %.1fs'%(index+1, len(images), time.monotonic()-start), flush=True)
    final_detector, final_head = g.state_digest(detector), g.state_digest(head)
    if initial_detector != final_detector or initial_head != final_head:
        raise ValueError('Frozen detector/midpoint parameters or buffers changed')
    g.assert_detector_frozen(detector)
    if head.training or any(p.requires_grad or p.grad is not None for p in head.parameters()):
        raise ValueError('Midpoint received gradients or left eval mode')
    if counters != dict(feature_extractions=c.COUNT, native_head_calls=3*c.COUNT):
        raise ValueError('Incomplete/extra inference work')
    if c.sha(b_path) != c.B_SHA or c.sha(head_path) != c.HEAD_SHA:
        raise ValueError('Weights changed during inference')
    proof = dict(runtime=runtime, detector_state_before=initial_detector,
        detector_state_after=final_detector, midpoint_state_before=initial_head,
        midpoint_state_after=final_head, counters=counters, detector_updates=0,
        midpoint_updates=0, model_metric_truth_read=False, output_count_and_scores_preserved=True,
        wall_seconds=time.monotonic()-start, wall_time_role='diagnostic duration, not deployment FPS')
    del detector, head, pipeline
    torch.cuda.empty_cache()
    return predictions, proof


def publish_artifacts(out):
    files = {p.name:c.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'artifacts.json'}
    c.write_new(out/'artifacts.json', dict(protocol=c.VERSION, sha256=files))


def run(args):
    identity, calibration = checked_sources()
    historical = c.historical_reference(args.historical_report)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    c.write_new(out/'protocol.json', c.protocol_document())
    c.write_new(out/'source_identity.json', identity)
    c.write_new(out/'historical_reference.json', historical)
    if args.stage == 'check':
        c.write_new(out/'completion.json', dict(status='STATIC_DEPTH_CONTRACT_PASS_NO_MODEL_DATA_GPU',
            protocol=c.VERSION, source_count=len(identity['sources'])))
        publish_artifacts(out)
        print('STATIC_DEPTH_CONTRACT_PASS_NO_MODEL_DATA_GPU', flush=True)
        return
    manifest, truth_path, images, data_identity = checked_dataset(args.data_root)
    c.write_new(out/'dataset_identity.json', data_identity)
    c.write_new(out/'image_sources.json', images)
    if args.stage == 'oracle':
        predictions = [dict(frame_id=s['frame_id'], b=None, midpoint=None) for s in images]
        proof = dict(model_loaded=False, gpu_used=False, parameter_refit_performed=False)
    else:
        b_path = c.checked_weight(ROOT, args.b_checkpoint, 'epoch_24.pth', c.B_SHA)
        head_path = c.checked_weight(ROOT, args.midpoint_checkpoint, 'head_epoch_03.pth', c.HEAD_SHA)
        c.write_new(out/'weight_identity.json', dict(b_checkpoint=str(b_path), b_sha256=c.B_SHA,
            midpoint_checkpoint=str(head_path), midpoint_sha256=c.HEAD_SHA))
        predictions, proof = native_predictions(out, images, b_path, head_path, args.gpu)
        c.write_new(out/'inference_proof.json', proof)
    truths = load_truth(truth_path, manifest)
    rows, summary = c.evaluate(predictions, truths, manifest, calibration)
    if args.stage == 'oracle':
        summary = dict(protocol=c.VERSION, role='existing_GT_OBB_formula_reproduction_only',
                       gt_obb=summary['groups']['gt_obb'], historical_reference=historical)
    else:
        summary['historical_reference'] = historical
        with (out/'evaluation_rows.jsonl').open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    c.write_new(out/'summary.json', summary)
    if c.sha(truth_path) != c.TRUTH_SHA:
        raise ValueError('Truth changed during evaluation')
    # Recheck source/calibration bytes before recording success.
    if checked_sources()[0] != identity:
        raise ValueError('Sources changed during the diagnostic')
    status = ('FROZEN_TRAIN03_GT_ORACLE_REPRODUCED_NO_DETECTOR' if args.stage == 'oracle'
              else 'FROZEN_TRAIN03_DEPTH_DIAGNOSTIC_COMPLETE_REVIEW_REQUIRED')
    c.write_new(out/'completion.json', dict(status=status, protocol=c.VERSION,
        frame_count=len(rows), parameter_refit_performed=False, reliability_used=False,
        fixed_dev_read=False, unknown_sequence_read=False, inference=proof))
    publish_artifacts(out)
    print(status, flush=True)
    groups = {'gt_obb':summary['gt_obb']} if args.stage == 'oracle' else summary['groups']
    for method, group in groups.items():
        m = group['depth_metrics']
        print('%s outputs=%d/%d numeric=%d/%d q-support=%d/%d MAE=%s RMSE=%s AbsRel=%s P95=%s'%(
            method, group['output_frame_count'], group['frame_count'], group['numeric_depth_count'],
            group['frame_count'], group['q_in_fit_support_count'], group['frame_count'],
            m['mae_m'], m['rmse_m'], m['abs_rel'], m['p95_abs_error_m']), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('check', 'oracle', 'audit'), required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--data-root', type=Path, default=ROOT/'data/webots_depth/yolo_obb_depth_dataset')
    p.add_argument('--b-checkpoint', type=Path)
    p.add_argument('--midpoint-checkpoint', type=Path)
    p.add_argument('--historical-report', type=Path,
                   help='Optional old Raw-opt Train-03 report; rejected for legacy/fixed-dev/mismatched oracle')
    p.add_argument('--gpu', type=int, default=0, help='Logical GPU after CUDA_VISIBLE_DEVICES')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
