#!/usr/bin/env python3
"""check/fit reuse reviewed TRAIN/VAL caches on CPU; smoke verifies six VAL views.

infer emits three flags for an image folder using the unchanged frozen B.
No ROI head, detector training, history, DINO, or TEST calibration is present.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.utils import port_simple_component_reliability_v1 as simple

PROTOCOL = ROOT/'crane_project/tools/port_simple_reliability_v1_protocol.json'
MANIFEST = ROOT/'crane_project/tools/port_simple_reliability_v1_sources.json'
B_CONFIG = ROOT/'crane_project/configs/crane_symeood_k1_port_day2night_aug_b_v1.py'
DATA = ROOT/'crane_project/data/crane_grab_port_day2night_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='simple-reliability-', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)  # Exclusive publication; no prior evidence overwritten.
    finally:
        os.unlink(temporary)


def checked_sources():
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads(MANIFEST.read_text())
    current = {p: sha(ROOT/p) for p in manifest['sources']}
    if (manifest.get('protocol') != simple.VERSION or current != manifest['sources']
            or manifest['protocol_sha256'] != sha(PROTOCOL)
            or protocol['protocol'] != simple.VERSION
            or protocol['feature_names'] != list(simple.FEATURES)):
        raise ValueError('Reviewed simple-interface sources/protocol differ')
    return protocol, dict(manifest_sha256=sha(MANIFEST), sources=current, protocol_sha256=sha(PROTOCOL))


def _checked_rows(rows, counts, split):
    if (Counter(r['sequence'] for r in rows) != Counter(counts)
            or len({r['image'] for r in rows}) != len(rows)):
        raise ValueError('Wrong frame roles/counts or duplicate images')
    for r in rows:
        if (r['sequence'] != r['image'].rsplit('_', 1)[0]
                or r['domain'] != r['image'].split('_')[0]
                or r['frame_id'] != int(r['image'].rsplit('_', 1)[1])
                or r['split'] not in (('train', 'train_sim') if split == 'train' else ('val',))):
            raise ValueError('Non-allowlisted TRAIN/VAL role: '+r['image'])
        simple.canonical(r['gt'])
        simple.prediction(r['pred'])
        if r['pred'] is not None:
            simple.descriptor(r['pred'], r['image_size'])
    return rows


def checked_inputs(args, protocol):
    """Use the reviewed numeric server GT; no OpenCV reconstruction or refitting labels."""
    hashes = protocol['reviewed_artifacts']
    paths = dict(train_identity=args.train_cache/'identity.json', train_complete=args.train_cache/'complete.json',
        train_snapshot=args.input_snapshot, train_predictions=args.train_cache/'predictions.jsonl',
        val_report=args.val_dir/'val_compare.json', val_predictions=args.val_dir/'val_qualities.jsonl')
    actual = {k: sha(p) for k, p in paths.items()}
    if actual != hashes:
        raise ValueError('Requires the already reviewed genuine TRAIN and native B VAL files: '+
                         str([k for k in hashes if actual[k] != hashes[k]]))
    identity = json.loads(paths['train_identity'].read_text())
    complete = json.loads(paths['train_complete'].read_text())
    snapshot = json.loads(paths['train_snapshot'].read_text())
    frozen = protocol['frozen_b']
    if (snapshot['identity'] != identity or simple.fingerprint(identity) != complete['identity_sha256']
            or simple.fingerprint(snapshot['inputs']) != identity['input_manifest_sha256']
            or identity['frozen_b'] != frozen or identity['frames'] != 2558
            or complete['frames'] != 2558 or complete['detector_optimizer_steps'] != 0
            or complete['b_state_before'] != complete['b_state_after']
            or complete['checkpoint_sha256'] != frozen['checkpoint_sha256']
            or complete['prediction_file_sha256'] != actual['train_predictions']):
        raise ValueError('Genuine TRAIN cache identity/freeze differs')
    predictions = [json.loads(line) for line in paths['train_predictions'].read_text().splitlines()]
    if len(predictions) != len(snapshot['inputs']):
        raise ValueError('TRAIN row count differs')
    train_rows = []
    for index, (source, saved) in enumerate(zip(snapshot['inputs'], predictions)):
        if (saved['index'] != index or saved['image'] != source['image']
                or saved['input_sha256'] != simple.fingerprint(source)):
            raise ValueError('TRAIN row/source hash or order differs')
        qualification = source['direction_qualification']
        train_rows.append(dict(image=source['image'], sequence=source['sequence'], domain=source['domain'],
            split=source['split'], frame_id=int(source['image'].rsplit('_', 1)[1]),
            image_size=source['image_size'], gt=source['gt_original'], pred=saved['pred_original'],
            train_angle_eligible=qualification['train_angle_eligible'],
            angle_axis_well_defined=qualification['evaluation_angle_eligible']))
    val_report = json.loads(paths['val_report'].read_text())
    b = val_report['cache_identity']['identity']
    if (val_report['protocol'] != 'port_reliability_branches_v1_cached_val_repair'
            or val_report['status'] != 'FIXED_CACHED_B_EPOCH8_VAL_SCORED_REVIEW_REQUIRED'
            or val_report['rows_sha256'] != actual['val_predictions'] or val_report['test_read'] is not False
            or val_report['cached_b_outputs_scores_exactly_preserved'] is not True
            or val_report['detector_optimizer_steps'] != 0 or val_report['branch_optimizer_steps'] != 0
            or b['checkpoint_sha256'] != frozen['checkpoint_sha256']
            or b['config_sha256'] != frozen['config_sha256'] or b['selected_epoch'] != 'epoch_24'):
        raise ValueError('Reviewed native B VAL cache proof differs')
    # Discard former quality scores/P3 descriptors completely.
    keys = ('image', 'sequence', 'domain', 'split', 'frame_id', 'image_size', 'gt', 'pred',
            'angle_axis_well_defined', 'image_sha256')
    val_rows = [{k: r[k] for k in keys} for r in
                (json.loads(line) for line in paths['val_predictions'].read_text().splitlines())]
    for r in val_rows:
        r['train_angle_eligible'] = False
    _checked_rows(train_rows, protocol['counts']['train'], 'train')
    _checked_rows(val_rows, protocol['counts']['val'], 'val')
    proof = dict(artifact_sha256=actual, train_frames=len(train_rows), val_frames=len(val_rows),
        train_angle_eligible=sum(r['train_angle_eligible'] for r in train_rows),
        original_b_frozen_state=complete['b_state_before'], frozen_b=frozen,
        val_native_cache_identity=b, old_quality_scores_used=False, detector_inferences=0,
        current_image_and_annotation_bytes_rehashed=False,
        scope='exact reviewed cached numeric records; no current dataset rebuild or new detector output')
    return train_rows, val_rows, proof


def load_policy(path, sources, protocol):
    policy = json.loads(Path(path).read_text())
    completion = json.loads((Path(path).parent/'completion.json').read_text())
    if ((Path(path).parent/'failure.json').exists()
            or completion.get('status') != 'SIMPLE_POLICY_FIT_SAVE_RELOAD_PASS_REVIEW_REQUIRED'
            or completion['policy_sha256'] != sha(path) or policy['sources'] != sources
            or policy['fixed_protocol'] != protocol
            or policy['frozen_b'] != protocol['frozen_b']
            or policy['input_proof']['artifact_sha256'] != protocol['reviewed_artifacts']):
        raise ValueError('Requires the completed fixed simple policy and unchanged execution contract')
    if (completion['report_sha256'] != sha(Path(path).parent/'fit_report.json')
            or completion['rows_sha256'] != sha(Path(path).parent/'val_decisions.jsonl')
            or completion['save_reload_decisions_exact'] is not True
            or completion['cached_b_inputs_unchanged'] is not True or completion['detector_updates'] != 0):
        raise ValueError('Incomplete or changed simple-policy fitting evidence')
    simple.validate_policy(policy)
    return policy


def fit(args, train_rows, val_rows, proof, protocol, sources):
    before = simple.fingerprint([train_rows, val_rows])
    policy = simple.create_policy(train_rows, val_rows, protocol)
    policy.update(fixed_protocol=protocol, sources=sources, input_proof=proof, frozen_b=protocol['frozen_b'])
    policy_path = args.out_dir/'policy.json'
    write_new(policy_path, policy)
    reloaded = json.loads(policy_path.read_text())
    if reloaded != policy:
        raise ValueError('Simple policy JSON save/reload differs')
    records, val_stats = simple.evaluate(val_rows, policy)
    replay, _ = simple.evaluate(val_rows, reloaded)
    _, train_stats = simple.evaluate(train_rows, policy)
    if records != replay or before != simple.fingerprint([train_rows, val_rows]):
        raise ValueError('Save/reload flags differ or cached B inputs changed')
    rows_path = args.out_dir/'val_decisions.jsonl'
    with rows_path.open('x') as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    report = dict(protocol=simple.VERSION, status='SIMPLE_TRAIN_VAL_FIXED_POLICY_REVIEW_REQUIRED',
        sources=sources, input_proof=proof, fixed_protocol=protocol, models=policy['models'],
        cutoffs=policy['cutoffs'], policy_sha256=sha(policy_path), rows_sha256=sha(rows_path),
        train_in_sample_descriptive=train_stats, val_calibration_descriptive=val_stats,
        detector_updates=0, new_detector_inferences=0, roi_training=False, test_read=False,
        limitations=['VAL determines operating coverage and is not independent performance validation.',
            'TRAIN frames are correlated and clean cached views; no OOF or augmented-view claim.',
            'Center flag retains B output, not a newly learned guarantee of center correctness.',
            'Balanced sigmoid risk is not a calibrated error probability.',
            'Matched score uses offline tie breaking for diagnosis only; deployment uses whole cutoff ties.',
            'No independent depth accuracy or geometry correction is established.'])
    write_new(args.out_dir/'fit_report.json', report)
    for domain in sorted({r['domain'] for r in val_rows}):
        s = val_stats['domain:'+domain]
        print(domain, 'B outputs', s['output_frames'], '/', s['frames'],
              'center/output', s['center_hit_rate_on_outputs'],
              'full-frame center', s['all_frame_center_correct_coverage'], flush=True)
        for c in ('size', 'angle'):
            a = s['components'][c]
            print(' ', c, 'simple accepted/bad', a['simple']['accepted_frames'], a['simple']['incorrect_accepted'],
                  'matched-score bad', a['matched_score_diagnostic']['incorrect_accepted'], flush=True)
    write_new(args.out_dir/'completion.json', dict(
        status='SIMPLE_POLICY_FIT_SAVE_RELOAD_PASS_REVIEW_REQUIRED', policy_sha256=sha(policy_path),
        report_sha256=sha(args.out_dir/'fit_report.json'), rows_sha256=sha(rows_path),
        save_reload_decisions_exact=True, cached_b_inputs_unchanged=True,
        detector_updates=0, test_read=False, sources=sources))


def checked_meta(meta):
    scale = np.asarray(meta['scale_factor'], dtype=float)
    oh, ow = meta['ori_shape'][:2]; ih, iw = meta['img_shape'][:2]; ph, pw = meta['pad_shape'][:2]
    if (scale.shape != (4,) or not np.isfinite(scale).all() or min(scale) <= 0
            or scale[0] != scale[2] or scale[1] != scale[3]
            or min(oh, ow, ih, iw) <= 0 or ih > ph or iw > pw
            or abs(scale[0]-scale[1]) > 1/min(oh, ow)+1e-6 or meta.get('flip', False)):
        raise ValueError('Only original B keep-ratio single-view coordinates allowed')


def state_digest(detector):
    h = hashlib.sha256()
    for name, tensor in sorted(detector.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def build_online(checkpoint, gpu, protocol):
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    from mmdet.datasets.pipelines import Compose
    if sha(checkpoint) != protocol['frozen_b']['checkpoint_sha256']:
        raise ValueError('Only retained B epoch24 checkpoint allowed')
    cfg = Config.fromfile(str(B_CONFIG))
    transforms = cfg.data.val.pipeline
    if (cfg.model.test_cfg.score_thr != .05 or cfg.model.test_cfg.max_per_img != 1
            or transforms[0]['type'] != 'LoadImageFromFile' or len(transforms) != 2
            or transforms[1]['type'] != 'MultiScaleFlipAug' or transforms[1].get('flip', False)
            or tuple(transforms[1]['img_scale']) != (1024, 1024)
            or [t['type'] for t in transforms[1]['transforms']] !=
               ['RResize', 'Normalize', 'Pad', 'DefaultFormatBundle', 'Collect']):
        raise ValueError('B inference configuration/pipeline differs')
    import_modules_from_strings(**cfg.custom_imports)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    detector = build_detector(deepcopy(cfg.model))
    loaded = load_checkpoint(detector, str(checkpoint), map_location='cpu', strict=True)
    if loaded.get('meta', {}).get('epoch') != 24:
        raise ValueError('Wrong B checkpoint epoch')
    del loaded
    detector.cuda(gpu).eval().requires_grad_(False)
    return detector, Compose(deepcopy(transforms))


def infer_one(detector, pipeline, path, gpu):
    import torch
    from mmcv.parallel import collate, scatter
    data = pipeline(dict(img_info=dict(filename=str(path)), img_prefix=None))
    batch = scatter(collate([data], samples_per_gpu=1), [gpu])[0]
    if len(batch['img']) != 1 or len(batch['img_metas']) != 1:
        raise ValueError('B requires one image/one scale')
    meta = batch['img_metas'][0][0]; checked_meta(meta)
    with torch.no_grad():
        output = detector(return_loss=False, rescale=True, **batch)
    if len(output) != 1 or len(output[0]) != 1:
        raise ValueError('Expected one image and one grab class')
    a = np.asarray(output[0][0], dtype=float)
    if a.ndim != 2 or a.shape[1] != 6 or len(a) > 1:
        raise ValueError('B must retain its original top1 decision')
    pred = a[0].tolist() if len(a) else None
    simple.prediction(pred)
    return pred, list(meta['ori_shape'][:2][::-1])


def online(args, policy, protocol, val_rows=None):
    import torch
    runtime = simple.SimpleComponentReliability(policy)
    detector, pipeline = build_online(args.b_checkpoint, args.gpu, protocol)
    before = state_digest(detector); policy_before = simple.fingerprint(policy)
    if args.mode == 'smoke':
        chosen = []
        for sequence in sorted(protocol['counts']['val']):
            group = sorted((r for r in val_rows if r['sequence'] == sequence), key=lambda r: r['frame_id'])
            chosen.extend((group[0], group[-1]))
        paths = [(DATA/'val/images'/(r['image']+'.jpg'), r) for r in chosen]
    else:
        if args.image_dir is None:
            raise ValueError('infer requires --image-dir containing original images')
        paths = [(p, None) for p in sorted(args.image_dir.iterdir())
                 if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png') and not p.name.startswith('._')]
        if not paths:
            raise ValueError('No images found')
    torch.cuda.reset_peak_memory_stats(args.gpu)
    rows = []; start = time.perf_counter()
    with (args.out_dir/'decisions.jsonl').open('x') as stream:
        for path, source in paths:
            if source is not None and sha(path) != source['image_sha256']:
                raise ValueError('Smoke image bytes differ')
            pred, size = infer_one(detector, pipeline, path, args.gpu)
            decision = runtime.decide(pred, size)
            row = dict(image=path.name, **decision)
            if source is not None:
                if size != source['image_size']:
                    raise ValueError('Original image dimensions differ')
                repeated, _ = infer_one(detector, pipeline, path, args.gpu)
                if pred != repeated:
                    raise ValueError('Same-runtime B output changed during side decision')
                cached = source['pred']
                cached_flags = {c: runtime.decide(cached, size)[c] for c in
                                ('center_accepted', 'size_accepted', 'angle_accepted')}
                row['diagnostic'] = dict(raw_b_before_after_exact=True,
                    historical_cache_close=(pred is None and cached is None) or (
                        pred is not None and cached is not None and bool(np.allclose(pred, cached, atol=1e-4, rtol=1e-6))),
                    cached_flags=cached_flags,
                    runtime_vs_cached_flags_equal=all(decision[c] == v for c,v in cached_flags.items()))
            stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
            rows.append(row)
    torch.cuda.synchronize(args.gpu)
    if before != state_digest(detector) or simple.fingerprint(policy) != policy_before:
        raise ValueError('B/policy state changed')
    write_new(args.out_dir/'completion.json', dict(
        status='SIMPLE_B_ONLINE_SMOKE_PASS_REVIEW_REQUIRED' if args.mode == 'smoke' else 'SIMPLE_B_ONLINE_INFERENCE_COMPLETE',
        frames=len(rows), b_state_before=before, b_state_after=state_digest(detector),
        detector_updates=0, policy_updates=0, policy_sha256=sha(args.policy),
        decisions_sha256=sha(args.out_dir/'decisions.jsonl'), elapsed_seconds=time.perf_counter()-start,
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(args.gpu),
        max_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20,
        max_reserved_mib=torch.cuda.max_memory_reserved(args.gpu)/2**20,
        historical_cache_parity_mismatches=sum(not r['diagnostic']['historical_cache_close'] for r in rows) if args.mode == 'smoke' else None,
        historical_cache_flag_differences=sum(not r['diagnostic']['runtime_vs_cached_flags_equal'] for r in rows) if args.mode == 'smoke' else None,
        label='current-frame online flags; no accuracy claim; smoke timing includes repeated B forward'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('check', 'fit', 'smoke', 'infer'), required=True)
    ap.add_argument('--train-cache', type=Path, default=Path('work_dirs/port_reliability_train_support_v1_cache'))
    ap.add_argument('--input-snapshot', type=Path, default=Path('work_dirs/port_reliability_train_support_v1/train_input_snapshot.json'))
    ap.add_argument('--val-dir', type=Path, default=Path('work_dirs/port_reliability_branches_v1_val_cached_v1'))
    ap.add_argument('--policy', type=Path, default=Path('work_dirs/port_simple_reliability_v1_fit/policy.json'))
    ap.add_argument('--b-checkpoint', type=Path, default=Path('work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth'))
    ap.add_argument('--image-dir', type=Path)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args(); os.chdir(ROOT)
    if args.out_dir.exists():
        raise FileExistsError('Use a NEW --out-dir; prior evidence is preserved: '+str(args.out_dir))
    protocol, sources = checked_sources()
    if args.mode != 'infer':
        train_rows, val_rows, proof = checked_inputs(args, protocol)
    if args.mode in ('smoke', 'infer'):
        policy = load_policy(args.policy, sources, protocol)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    try:
        if args.mode == 'check':
            write_new(args.out_dir/'input_check.json', dict(status='SIMPLE_INPUT_CHECK_PASS_NO_FIT_NO_GPU',
                proof=proof, protocol=protocol, sources=sources, test_read=False, parameter_updates=0))
        elif args.mode == 'fit':
            fit(args, train_rows, val_rows, proof, protocol, sources)
        else:
            online(args, policy, protocol, val_rows if args.mode == 'smoke' else None)
        if checked_sources()[1] != sources:
            raise ValueError('Execution sources changed during run')
    except Exception as error:
        write_new(args.out_dir/'failure.json', dict(status='FAILED_PRESERVE_OUTPUTS', error=str(error), mode=args.mode))
        raise
    print('Saved', args.out_dir, 'mode', args.mode, flush=True)


if __name__ == '__main__':
    main()
