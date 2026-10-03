#!/usr/bin/env python3
"""Frozen B: fixed12 TRAIN identities x2 views, actual ROI/P3, zero updates."""
import argparse
from copy import deepcopy
import csv
import html
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import check_port_train_geometry_evidence_v1 as q
from crane_project.utils import port_geometry_spatial_response_v1 as s

g = q.o.g
VERSION = 'port_geometry_spatial_response_v1'
PROTOCOL = ROOT/'crane_project/tools/port_geometry_spatial_response_v1_protocol.json'
MANIFEST = ROOT/'crane_project/tools/port_geometry_spatial_response_v1_sources.json'
PRIOR_MANIFEST_SHA = '4051f9c8cb804a424d0f6fdf8bfa54bd0634c86595ab8635b069054a17e9ebd5'
EVIDENCE_SHAS = dict(
    completion='9224032e9f6ef16665d590e2f30e8e85e443159b76175720ba9490e02f4388cd',
    analysis='419ef9128ea9c8c4db05140e1e79adeb83e116f7bb2ca76d7a184e3a7ff78b77',
    selection='1c6825cb087d230c4b2051844122ee5416c0c9d5d7f5e3f3551285c8d3d27dd8')
IMAGES = ('real_seq13_00179', 'real_seq13_00066', 'real_seq01_00283',
    'sim_seq08_00202', 'sim_seq08_00744', 'sim_seq08_00190',
    'real_seq13_00168', 'real_seq05_00100', 'real_seq05_00420',
    'sim_seq08_00709', 'sim_seq08_00168', 'sim_seq08_00432')


def protocol_document():
    return dict(protocol=VERSION, settings=deepcopy(s.SETTINGS), fixed_images=list(IMAGES),
        evidence_role='repeated_TRAIN_spatial_representation_diagnosis_not_training',
        prior_manifest_sha256=PRIOR_MANIFEST_SHA, evidence_sha256=EVIDENCE_SHAS,
        cache_manifest_sha256=q.o.c.a.CACHE_MANIFEST_SHA,
        shifts={k:list(v) for k,v in s.SHIFTS.items()},
        scope=dict(images=12,views=24,feature_extractions=24,head_inference_calls=48,
            detector_updates=0,head_updates=0,formal_training=False,
            checkpoint_export=False,val_access=False,test_access=False),
        cache='Read reviewed CPU cache once; all128 records validated, measure only24 fixed views. Static checks manifest only, no torch.load, checkpoint or CUDA.',
        replay='Same B ep24 eval/no_grad, TRAIN deterministic1024/isotropic.5. One view at a time, one FPN extraction and two native head calls for raw/original agreement. Re-extracted ordinary AND aligned ROI/support must agree elementwise with cache at atol1e-5/rtol1e-4. Failure stops, no cache replacement.',
        runtime='Full extraction requires reviewed python/torch/cuda/cudnn/numpy/opencv strings and cached mmcv/mmdet/mmrotate/gpu model. No runtime override. Static is CPU compatible.',
        intervention='Ordinary ROI center ONLY +/-1 model input pixel in x/y, unchanged B context/shape/angle. No shifted image, new predictions, GT-driven channels, ranking, fitting, loss or seed sweep. Central difference is feature RMS per input/original pixel, not signed correction evidence.',
        measurements='All256 channels: RMS, spatial demeaning and channel quantiles, neighbor differences; native P3 patch and valid mask separately. Report absolute and relative changes, null ratio at zero denominator. Native patch <=64x64 cells; persist scalar2D maps only, no full FPN or channel tensor.',
        coordinates='Existing raw w/h-angle association and original restoration; GT annotation sizes use sqrt(sx*sy). Offline GT/B/center-only/joint positions mapped to ordinary/aligned9x9 indices and stride8 native grid. Model and original displacement units explicit. GT never enters B forward or shift sampling.',
        display='Native decoded input crop, uniform display affine; discrete cell heatmaps, printed ranges, four shifts share per-view range. No energy maximizer labeled as correct center; no interpolation interpreted as additional resolution.',
        review='All24 views and eight role/domain/scale groups; human review required, no response threshold approves formal training. Engineering identity/finite guards only.',
        limitations=['Fixed12 extremes/ordinary identities selected by previous saved outcomes; not prevalence estimates.',
            'TRAIN probe repeatedly exposed and B trained on all; two scales are related views, not independent VAL.',
            'P3 energy/derivatives do not establish semantic boundaries or a recoverable correction direction.',
            'GT boxes and raw pixels cannot certify subpixel true center; no relabeling.',
            'B remains retained; sparse output/conditional/all-frame fractions do not guarantee video continuity or depth accuracy.',
            'TEST repeatedly exposed; no new VAL/TEST, thresholds, weights or method selection here.'])


def checked_sources():
    if g.ready.sha(q.MANIFEST) != PRIOR_MANIFEST_SHA:
        raise ValueError('Reviewed supervision/pixel source identity differs')
    q.checked_sources()
    manifest = json.loads(MANIFEST.read_text())
    expected = set(json.loads(q.MANIFEST.read_text())['sources']) | {
        'crane_project/tools/check_port_geometry_spatial_response_v1.py',
        'crane_project/utils/port_geometry_spatial_response_v1.py',
        'crane_project/tools/port_geometry_spatial_response_v1_protocol.json'}
    if (manifest.get('protocol') != VERSION or manifest.get('settings') != s.SETTINGS
            or set(manifest['sources']) != expected
            or json.loads(PROTOCOL.read_text()) != protocol_document()):
        raise ValueError('Spatial source/protocol contract differs')
    actual = {p:g.ready.sha(ROOT/p) for p in expected}
    if actual != manifest['sources']:
        raise ValueError('Spatial source SHA differs: '+', '.join(p for p in actual if actual[p] != manifest['sources'][p]))
    return dict(manifest_sha256=g.ready.sha(MANIFEST), protocol_sha256=g.ready.sha(PROTOCOL), sources=actual)


def checked_inputs(args):
    identity = checked_sources()
    samples, _, prior, _, selection = q.checked_inputs(Path(args.baseline_report),
        Path(args.joint_report), Path(args.center_only_report))
    previous = {}
    for name, digest in EVIDENCE_SHAS.items():
        path = Path(args.evidence_dir)/(name+'.json')
        if g.ready.sha(path) != digest:
            raise ValueError('Requires exact reviewed TRAIN pixel evidence: '+name)
        previous[name] = json.loads(path.read_text())
    if (previous['completion']['identity'] != prior or
            previous['completion']['status'] != 'TRAIN_GEOMETRY_EVIDENCE_COMPLETE_HUMAN_REVIEW_REQUIRED'
            or previous['selection'] != selection
            or tuple(v['image'] for v in selection['slots']) != IMAGES):
        raise ValueError('Reviewed12 image selection/identity differs')
    baseline = json.loads(Path(args.baseline_report).read_text())
    cache_path = Path(args.cache_dir)/'cache_manifest.json'
    if g.ready.sha(cache_path) != q.o.c.a.CACHE_MANIFEST_SHA:
        raise ValueError('Requires unchanged complete server ROI cache manifest')
    cache = json.loads(cache_path.read_text())
    if cache != baseline['cache']:
        raise ValueError('Cache manifest differs from reviewed report')
    gi = prior['prior_identity']['prior_identity']['prior_identity']['g_identity']
    g.accepted_cache_identity(cache['identity'], gi)
    rows = {(r['image'], r['scale']):r for r in previous['analysis']['rows']}
    ordered = deepcopy([rows[(image, scale)] for scale in s.SETTINGS['scales'] for image in IMAGES])
    sizes = {v['image']:v['image_size'] for v in samples}
    for row in ordered: row['original_image_wh'] = sizes[row['image']]
    if len(ordered) != 24 or not all(r['eligible'] and r['arms']['b']['pred'] for r in ordered):
        raise ValueError('Expected24 fixed eligible B outputs, no filtering')
    identity.update(prior_identity=prior, evidence_sha256=EVIDENCE_SHAS,
        cache_manifest_sha256=g.ready.sha(cache_path))
    return samples, identity, ordered, baseline, gi, selection


def model_boxes(row, meta, bm):
    result = dict(b=bm[0].cpu().tolist())
    gt = bm.new_tensor(row['gt_original']).reshape(1, 5)
    result['gt'] = g.map_boxes(gt, meta, size_mode='annotation')[0].cpu().tolist()
    for arm in ('center_only', 'joint'):
        value = bm.new_tensor(row['arms'][arm]['pred'][:5]).reshape(1, 5)
        result[arm] = g.map_boxes(value, meta)[0].cpu().tolist()
    return result


def fixed_meta(row):
    geom = row['geometry']; ow, oh = row['original_image_wh']; iw, ih = geom['image_wh']
    pw, ph = geom['pad_wh']
    return dict(ori_shape=(oh, ow, 3), img_shape=(ih, iw, 3), pad_shape=(ph, pw, 3),
        scale_factor=geom['scale_factor'], flip=False)


def cache_audit(records, rows):
    lookup = {(r['image'], r['scale']):r for r in records}; audited = []
    for row in rows:
        r = lookup[(row['image'], row['scale'])]
        if r['boxes_original'].tolist() != [row['arms']['b']['pred']]:
            raise ValueError('Selected cached B output differs')
        meta = fixed_meta(row); bm = r['boxes_model']
        if not torch.allclose(g.map_boxes(bm, meta, inverse=True), r['boxes_original'][:, :5], atol=1e-4, rtol=1e-6):
            raise ValueError('Cached input/original restore differs')
        boxes = model_boxes(row, meta, bm)
        centers = bm.new_tensor([v[:2] for v in boxes.values()])
        item = {k:row[k] for k in ('image', 'role', 'domain', 'sequence', 'scale')}
        item.update(geometry=row['geometry'], boxes_model=boxes,
            original_image_wh=row['original_image_wh'],
            reused_geometry={k:deepcopy(row['arms'][k]['metrics']) for k in ('b', 'center_only', 'joint')},
            cached_roi={}, reference_roi_coordinates={})
        for arm in g.ARMS:
            value = r['local'][arm]
            item['cached_roi'][arm] = s.summarize_feature(value['roi'])
            item['cached_roi'][arm].update(support_min=float(value['support'].min()),
                support_mean=float(value['support'].mean()), support_map=value['support'][0, 0].tolist())
            coordinates = s.roi_coordinates(centers, bm, arm)
            item['reference_roi_coordinates'][arm] = dict(zip(boxes, coordinates))
        audited.append(item)
    return audited


def extract_views(args, samples, rows, payload, audited, baseline, scope, out, progress):
    import mmcv
    import mmdet
    import mmrotate
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    from crane_project.tools.diagnose_port_shape_e_h_train_gradients_v1 import checkpoint_contract
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Single-view diagnostic only; do not use torchrun/DDP')
    runtime = q.o.c.a.checked_runtime(baseline)
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        raise ValueError('Valid logical CUDA device required')
    runtime.update(mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmrotate=mmrotate.__version__,
        gpu=torch.cuda.get_device_name(args.gpu))
    if any(runtime[k] != payload['extraction_runtime'][k] for k in payload['extraction_runtime'] if k != 'cuda_visible_devices'):
        raise ValueError('Original cached B extraction runtime/GPU model differs')
    if g.ready.sha(g.CHECKPOINT) != g.ready.FROZEN_B['checkpoint_sha256']:
        raise ValueError('B checkpoint changed')
    cfg = g.check_cfg(); g.seed_all(); torch.cuda.set_device(args.gpu)
    scope['gpu_devices_used'] = [args.gpu]
    import_modules_from_strings(**cfg.custom_imports)
    model_cfg = deepcopy(cfg.model); model_cfg.pretrained = None; model_cfg.train_cfg = None
    detector = build_detector(model_cfg)
    loaded = load_checkpoint(detector, str(g.CHECKPOINT), map_location='cpu', strict=True)
    checkpoint_meta = checkpoint_contract(loaded['meta'], cfg, 'b'); del loaded
    g.freeze_detector(detector.cuda(args.gpu)); before = g.state_digest(detector)
    if before != payload['detector_state_before']:
        raise ValueError('Frozen B parameter/buffer state differs from original extraction')
    _, protocol, _, _ = g.checked_inputs()
    cached = {(r['image'], r['scale']):r for r in payload['records']}
    fixtures = {v['image']:v for v in samples}; done = []
    (out/'views').mkdir(); (out/'panels').mkdir()
    for scale in s.SETTINGS['scales']:
        dataset = ConcatDataset([build_dataset(spec) for spec in g.fixed_specs(cfg, scale)])
        lookup = {Path(info['filename']).stem:offset+i
            for part, offset in zip(dataset.datasets, (0, len(dataset.datasets[0])))
            for i, info in enumerate(part.data_infos)}
        if len(lookup) != sum(g.ready.TRAIN_COUNTS.values()):
            raise ValueError('TRAIN count/unique identity differs')
        for row, audit in zip(rows, audited):
            if row['scale'] != scale: continue
            sample = fixtures[row['image']]
            image_path = g.ready.DATA/sample['split']/'images'/(row['image']+'.jpg')
            ann_path = g.ready.DATA/sample['split']/'annfiles'/(row['image']+'.txt')
            if g.ready.sha(image_path) != sample['image_sha256'] or g.ready.sha(ann_path) != sample['annotation_sha256']:
                raise ValueError('Selected TRAIN bytes changed before forward')
            batch = scatter(collate([dataset[lookup[row['image']]]], samples_per_gpu=1), [args.gpu])[0]
            image, metas = batch['img'][0], batch['img_metas'][0]; meta = metas[0]
            expected = fixed_meta(row)
            if (len(batch['img']) != 1 or image.shape != (1, 3, 1024, 1024)
                    or Path(meta['filename']).stem != row['image'] or meta.get('flip', False)
                    or any(list(meta[k]) != list(expected[k]) for k in expected if k != 'flip')):
                raise ValueError('Native TRAIN view geometry differs')
            g.assert_detector_frozen(detector)
            def forward():
                with torch.no_grad():
                    scope['feature_extractions'] += 1
                    features = detector.extract_feat(image)
                    scope['head_inference_calls'] += 1
                    raw = g.flatten_prediction(detector.simple_test_from_features(features, metas, rescale=False))
                    scope['head_inference_calls'] += 1
                    original = g.flatten_prediction(detector.simple_test_from_features(features, metas, rescale=True))
                return features, image.new_tensor(raw), image.new_tensor(original)
            (features, raw, original), cost = g.measured(forward, args.gpu)
            if (raw.shape != (1, 6) or original.shape != (1, 6) or features[0].shape != (1, 256, 128, 128)
                    or any(f.requires_grad or f.grad_fn is not None for f in features)):
                raise ValueError('B output/P3 shape or detached contract differs')
            if (not torch.allclose(g.map_boxes(raw[:, :5], meta, inverse=True), original[:, :5], atol=1e-4, rtol=1e-6)
                    or not torch.equal(raw[:, 5], original[:, 5])):
                raise ValueError('Native B restore or score changed')
            previous = g.check_previous(original, sample['previous_b_predictions'][str(scale)], protocol)
            r = cached[(row['image'], scale)]
            result = deepcopy(audit); result.update(previous_b_agreement=previous, roi_replay={})
            result['b_original_output_agreement'] = s.agreement(original, r['boxes_original'])
            with torch.no_grad():
                # Same cached model box isolates FEATURE replay from small B output numerical changes.
                bm = r['boxes_model'].to(image)
                result['b_model_output_agreement'] = s.agreement(raw[:, :5], bm)
                for arm in g.ARMS:
                    roi, mask, _ = g.sample_local(features[0], bm, meta, arm)
                    result['roi_replay'][arm] = dict(roi=s.agreement(roi, r['local'][arm]['roi']),
                        support=s.agreement(mask, r['local'][arm]['support']))
                result['responses'], points = s.shift_responses(features[0], bm, meta)
                centers = bm.new_tensor([v[:2] for v in result['boxes_model'].values()])
                result['p3_patch'] = s.native_patch(features[0], points, centers, meta)
            cost['peak_allocated_including_sampling_mib'] = torch.cuda.max_memory_allocated(args.gpu)/2**20
            cost['peak_reserved_including_sampling_mib'] = torch.cuda.max_memory_reserved(args.gpu)/2**20
            result['gpu_cost'] = cost
            g.assert_detector_frozen(detector)
            del features, roi, mask, points, centers, bm, raw, original, image, batch, metas
            # Only JSON scalar maps survive; no full FPN or image tensor accumulation.
            native, geometry = q.e.native_view(q.e.native_rgb(image_path), scale)
            if geometry != row['geometry']: raise ValueError('Rendered native input geometry differs')
            name = row['image']+'_scale'+str(scale)
            result['panel'] = s.render_panel(result, native, out/'panels'/(name+'.png'))
            q.write_json(out/'views'/(name+'.json'), result)
            done.append(result); scope['completed_views'] = len(done)
            progress(dict(stage='view', image=row['image'], scale=scale, completed=len(done), gpu=cost))
            print('TRAIN frozen spatial', len(done), '/24', row['image'], scale, flush=True)
            del native, result
        del dataset, lookup
    g.assert_detector_frozen(detector); after = g.state_digest(detector)
    if before != after or g.ready.sha(g.CHECKPOINT) != g.ready.FROZEN_B['checkpoint_sha256']:
        raise ValueError('Frozen detector/checkpoint mutation')
    del detector; torch.cuda.empty_cache()
    return done, dict(runtime=runtime, detector_state_before=before, detector_state_after=after,
        detector_state_unchanged=True, checkpoint_meta=checkpoint_meta)


def publish_review(out, rows):
    groups = {}
    for role in ('fit', 'probe'):
        for domain in ('real', 'sim'):
            for scale in s.SETTINGS['scales']:
                subset = [r for r in rows if r['role'] == role and r['domain'] == domain and r['scale'] == scale]
                key = role+'/'+domain+'/'+str(scale)
                metrics = [r['reused_geometry']['b'] for r in subset]
                groups[key] = dict(views=len(subset), images=[r['image'] for r in subset],
                    b_output_coverage=dict(numerator=sum(m['output'] for m in metrics), denominator=len(subset)),
                    b_conditional_center_correct=dict(numerator=sum(m['center_error_px'] < 15 for m in metrics if m['output']), denominator=sum(m['output'] for m in metrics)),
                    b_all_frame_center_correct=dict(numerator=sum(m['output'] and m['center_error_px'] < 15 for m in metrics), denominator=len(subset)),
                    ordinary_spatial_rms=g.describe([r['cached_roi']['ordinary']['spatial_demeaned_rms'] for r in subset]),
                    perturbation_rms={k:g.describe([r['responses']['shifts'][k]['difference_rms'] for r in subset]) for k in s.SHIFTS})
    q.write_json(out/'groups.json', dict(groups=groups, note='Three related TRAIN examples per group; reused B geometry, no performance improvement or independent samples.'))
    fields = ['image', 'role', 'domain', 'scale', 'boundary_or_surrounding_structure',
              'spatial_response_interpretation', 'correction_direction_supported', 'reviewer', 'notes']
    with (out/'human_review_template.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader()
        for row in rows:
            writer.writerow(dict({k:row[k] for k in fields[:4]},
                **{k:'NOT_REVIEWED' for k in fields[4:7]}, reviewer='', notes=''))
    body = ['<!doctype html><meta charset="utf-8"><title>Frozen TRAIN spatial evidence</title>',
        '<style>body{font:16px sans-serif;margin:24px}img{max-width:1600px;width:100%}</style>',
        '<h1>冻结B：真实ROI/P3空间证据</h1><p>TRAIN诊断，零更新。能量和响应变化不等于正确中心；',
        '不按热图选候选/通道/权重。按原图边界与周围结构审查全部24视图。',
        '9×9插值没有增加原生分辨率；四个位移图只在同视图共享色阶。',
        '保留B；人工模板另存填写，不能据完成状态放行训练。</p>']
    for row in rows:
        body.append('<h2>'+html.escape(row['image']+' scale'+str(row['scale']))+'</h2><img src="panels/'+html.escape(row['panel']['file'], quote=True)+'">')
    (out/'index.html').write_text('\n'.join(body), encoding='utf-8')


def run(args):
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    scope = dict(feature_extractions=0, head_inference_calls=0, detector_updates=0,
        head_updates=0, feature_cache_loads=0, completed_views=0, gpu_devices_used=[],
        val_access=False, test_access=False, formal_training=False,
        checkpoint_exported=False, annotation_changes=False)
    report = dict(protocol=VERSION, status='RUNNING', scope=scope, check_only=args.check_only,
        formal_training_approved=False, human_review='NOT_REVIEWED')
    def progress(value):
        with (out/'progress.jsonl').open('a') as f: f.write(json.dumps(value, allow_nan=False)+'\n')
    try:
        progress(dict(stage='begin', check_only=args.check_only))
        samples, identity, rows, baseline, gi, selection = checked_inputs(args)
        report.update(identity=identity, fixed_images=list(IMAGES), expected_views=24,
            limitations=protocol_document()['limitations'])
        q.write_json(out/'protocol.json', protocol_document()); q.write_json(out/'selection.json', selection)
        q.write_json(out/'cache_manifest.json', baseline['cache'])
        progress(dict(stage='verified', selected_images=12, selected_views=24))
        if not args.check_only:
            scope['feature_cache_loads'] += 1
            payload, cache = g.checked_cache(Path(args.cache_dir), gi, samples)
            g.validate_records(payload['records'], samples)
            q.o.c.a.check_cache_reference(payload['records'], cache, baseline)
            audited = cache_audit(payload['records'], rows)
            q.write_json(out/'cached_roi.json', dict(rows=audited, cpu_roi_bytes=g.cpu_roi_bytes(payload['records'])))
            progress(dict(stage='cache_measured', views=len(audited)))
            done, extraction = extract_views(args, samples, rows, payload, audited, baseline, scope, out, progress)
            report.update(extraction=extraction)
            if scope['feature_extractions'] != 24 or scope['head_inference_calls'] != 48 or len(done) != 24:
                raise ValueError('Fixed extraction/inference budget differs')
            if (g.ready.sha(Path(args.cache_dir)/'cache_manifest.json') != q.o.c.a.CACHE_MANIFEST_SHA
                    or any(g.ready.sha(Path(args.cache_dir)/p) != digest for p,digest in cache['files'].items())):
                raise ValueError('Reviewed cache changed during run')
            checked_sources(); publish_review(out, done)
        report['status'] = ('STATIC_SPATIAL_CONTRACT_COMPLETE_NO_CACHE_LOAD_NO_GPU_NO_UPDATES' if args.check_only
            else 'TRAIN_SPATIAL_RESPONSE_COMPLETE_HUMAN_REVIEW_REQUIRED')
        progress(dict(stage='complete', status=report['status']))
    except Exception as error:
        report.update(status='FAILED', error=str(error), error_type=type(error).__name__)
        progress(dict(stage='failed', error=str(error)))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-start
        q.write_json(out/'completion.json', report)
        q.write_json(out/'artifacts.json', dict(protocol=VERSION, status=report['status'],
            files={str(p.relative_to(out)):g.ready.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'artifacts.json'},
            excluded=['weights', 'full_FPN', 'channel_tensors', 'ROI_cache_payload', 'source_bundle']))
    print('Saved', out/'completion.json', 'status', report['status'], flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--baseline-report', default=str(q.BASELINE))
    p.add_argument('--joint-report', default=str(q.JOINT))
    p.add_argument('--center-only-report', default=str(q.CENTER))
    p.add_argument('--evidence-dir', default='work_dirs/port_train_geometry_evidence_v1_train')
    p.add_argument('--cache-dir', default='work_dirs/port_geometry_g_v1_roi_cache')
    p.add_argument('--out-dir', required=True)
    return p


if __name__ == '__main__':
    os.chdir(ROOT)
    print('Frozen B spatial check: 12 TRAIN images/24 views, <=24 extractions/48 native head calls, updates=0. Not formal training.', flush=True)
    run(parser().parse_args())
