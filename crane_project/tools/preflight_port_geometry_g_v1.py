#!/usr/bin/env python3
"""G v1: 64 fixed TRAIN images x two views, frozen B, 200 head steps/arm.

No detector updates, formal training, checkpoint export, VAL/TEST access or
step selection. Ordinary/aligned arms share initialization and minibatches.
CPU caches hold only detached local ROIs; complete caches can be reused.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import check_port_reliability_readiness_v1 as ready
from crane_project.tools.preflight_port_structure_reliability_v1 import (
    check_cfg, flatten_prediction, measured, state_digest, write_new)
from crane_project.tools.audit_port_train_val_geometry_v1 import decompose, describe, summarize
from crane_project.utils.port_geometry_refine_g_v1 import (
    ARMS, SETTINGS, LocalGeometryRefiner, assert_detector_frozen, canonical_boxes,
    checked_meta, freeze_detector, map_boxes, regression_loss, sample_local, wrap_pi)
from crane_project.utils.port_structure_reliability_v1 import map_points

MANIFEST = ROOT/'crane_project/tools/port_geometry_g_v1_sources.json'
PROTOCOL = ROOT/'crane_project/tools/port_geometry_g_v1_protocol.json'
FIXTURE = ROOT/'crane_project/tools/port_geometry_g_v1_train_samples.json'
CHECKPOINT = ROOT/'work_dirs/crane_symeood_k1_port_day2night_aug_b_v1/epoch_24.pth'


def checked_inputs():
    manifest = json.loads(MANIFEST.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    fixture = json.loads(FIXTURE.read_text())
    if any(x.get('protocol') != SETTINGS['protocol'] for x in (manifest, protocol, fixture)):
        raise ValueError('G protocol identity differs')
    if manifest['settings'] != SETTINGS or protocol['settings'] != SETTINGS:
        raise ValueError('Predeclared settings differ')
    actual = {p: ready.sha(ROOT/p) for p in manifest['sources']}
    if actual != manifest['sources']:
        raise ValueError('Reviewed source SHA differs: '+', '.join(p for p in actual if actual[p] != manifest['sources'][p]))
    if protocol['fixture_sha256'] != ready.sha(FIXTURE):
        raise ValueError('Fixed TRAIN fixture differs')
    frozen = {k: ready.FROZEN_B[k] for k in protocol['frozen_b']}
    if frozen != protocol['frozen_b'] or protocol['train_sequence_counts'] != ready.TRAIN_COUNTS:
        raise ValueError('B/TRAIN identity differs')
    if protocol['arms'] != list(ARMS):
        raise ValueError('Mechanism comparison differs')
    current = {}
    for split in ('train', 'train_sim'):
        paths = ready.files(ready.DATA/split/'annfiles', '.txt')
        expected = {k: v for k, v in ready.TRAIN_COUNTS.items() if ready.TRAIN_SPLITS[k] == split}
        if Counter(p.stem.rsplit('_', 1)[0] for p in paths) != Counter(expected):
            raise ValueError('TRAIN sequence/count differs')
        current[split+'_annotation_sha256'] = ready.set_sha(paths)
    if current != fixture['train_annotation_identities']:
        raise ValueError('TRAIN annotation bytes differ')
    samples = fixture['samples']
    if len(samples) != 64 or len({r['image'] for r in samples}) != 64:
        raise ValueError('Expected 64 unique fixed TRAIN images')
    counts = Counter((r['role'], r['domain']) for r in samples)
    if counts != Counter({(role, domain): n for role, domains in protocol['role_counts'].items() for domain, n in domains.items()}):
        raise ValueError('Fixed fit/probe role differs')
    seen = Counter()
    for r in samples:
        seq = r['image'].rsplit('_', 1)[0]
        role = 'probe' if seen[r['domain']] % 4 == 0 else 'fit'
        seen[r['domain']] += 1
        if (seq != r['sequence'] or seq not in ready.TRAIN_SPLITS or
                r['split'] != ready.TRAIN_SPLITS[seq] or r['domain'] != seq.split('_')[0] or r['role'] != role):
            raise ValueError('TRAIN image/role substitution')
        image = ready.DATA/r['split']/'images'/(r['image']+'.jpg')
        ann = ready.DATA/r['split']/'annfiles'/(r['image']+'.txt')
        with Image.open(image) as im:
            size = list(im.size)
        if ready.sha(image) != r['image_sha256'] or ready.sha(ann) != r['annotation_sha256'] or size != r['image_size']:
            raise ValueError('TRAIN sample bytes differ: '+r['image'])
        canonical_boxes(torch.tensor(r['gt']).reshape(1, 5))
    identity = dict(manifest_sha256=ready.sha(MANIFEST), protocol_sha256=ready.sha(PROTOCOL),
                    fixture_sha256=ready.sha(FIXTURE), sources=actual, frozen_b=frozen,
                    train_annotation_identities=current)
    return check_cfg(), protocol, samples, identity


def fixed_specs(cfg, scale):
    """Definitions reused from old audit; never instantiate VAL/TEST datasets."""
    multi = cfg.data.val.pipeline[1]
    if (multi['type'] != 'MultiScaleFlipAug' or tuple(multi['img_scale']) != (1024, 1024)
            or multi.get('flip', False) or [t['type'] for t in multi['transforms']] !=
            ['RResize', 'Normalize', 'Pad', 'DefaultFormatBundle', 'Collect']):
        raise ValueError('Historical deterministic view definition differs')
    specs = deepcopy(cfg.data.train)
    if len(specs) != 2:
        raise ValueError('Expected real/sim TRAIN only')
    for spec, split in zip(specs, ('train', 'train_sim')):
        if ((ROOT/spec.get('data_root', '')/spec['ann_file']).resolve() != (ready.DATA/split/'annfiles').resolve()
                or (ROOT/spec.get('data_root', '')/spec['img_prefix']).resolve() != (ready.DATA/split/'images').resolve()):
            raise ValueError('Non-TRAIN dataset path')
        spec['test_mode'] = True
        spec['pipeline'] = deepcopy(cfg.data.val.pipeline)
        if scale == .5:
            spec['pipeline'][1]['transforms'].insert(1, dict(type='PortIsotropicShrink', prob=1., scale_range=(.5, .5)))
        elif scale != 1.:
            raise ValueError('Undeclared view scale')
    return specs


def seed_all():
    random.seed(SETTINGS['seed']); np.random.seed(SETTINGS['seed'])
    torch.manual_seed(SETTINGS['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SETTINGS['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def schedule(records):
    """Same domain-balanced fit indices for both arms; probe never optimized."""
    pools = {d: [i for i, r in enumerate(records) if r['role'] == 'fit' and r['eligible'] and r['domain'] == d]
             for d in ('real', 'sim')}
    if any(not x for x in pools.values()):
        raise ValueError('No fit support in one domain')
    rng = np.random.RandomState(SETTINGS['seed'])
    return [np.concatenate([rng.choice(pools[d], SETTINGS['batch_size']//2, replace=True) for d in ('real', 'sim')]).tolist()
            for _ in range(SETTINGS['steps_per_arm'])]


def cpu_roi_bytes(records):
    return sum(t.numel()*t.element_size() for r in records for a in ARMS for t in r['local'][a].values())


def preview(path, sample, original, points, meta):
    """Diagnostic GT overlay only; never passed to the online head."""
    image = ready.DATA/sample['split']/'images'/(sample['image']+'.jpg')
    with Image.open(image) as im:
        im = im.convert('RGB')
    # Crop the observed B/GT neighbourhood to make spatial support readable.
    b = sample['gt'] if not len(original) else original[0, :5].cpu().tolist()
    radius = max(b[2:4])*1.5+32
    x0, y0 = max(0, int(b[0]-radius)), max(0, int(b[1]-radius))
    x1, y1 = min(im.width, int(b[0]+radius)), min(im.height, int(b[1]+radius))
    im = im.crop((x0, y0, x1, y1)); draw = ImageDraw.Draw(im)
    xy = lambda p: (float(p[0])-x0, float(p[1])-y0)
    draw.line([xy(p) for p in ready.box_polygon(sample['gt'])]+[xy(ready.box_polygon(sample['gt'])[0])], fill='lime', width=2)
    if len(original):
        poly = ready.box_polygon(original[0, :5].cpu().numpy())
        draw.line([xy(p) for p in poly]+[xy(poly[0])], fill='yellow', width=2)
        for arm, color in (('ordinary', 'cyan'), ('aligned', 'magenta')):
            p = map_points(points[arm], meta, inverse=True).cpu().numpy().reshape(-1, 2)
            for q in p:
                x, y = xy(q); draw.ellipse((x-1.5, y-1.5, x+1.5, y+1.5), fill=color)
    draw.text((3, 3), 'GT green; B yellow; ordinary cyan; aligned magenta', fill='white')
    im.save(path)


def check_previous(pred, old, protocol):
    if (old is None) != (len(pred) == 0):
        raise ValueError('Historical B output presence differs')
    if not len(pred):
        return None
    b = canonical_boxes(pred[:, :5]).cpu().numpy()[0]
    g = ready.canonical(old[:5])
    delta = [float(np.max(np.abs(b[:4]-g[:4]))), ready.angle_error(b[4], g[4])*math.pi/180, abs(float(pred[0, 5])-old[5])]
    tol = protocol['baseline_tolerances']
    if any(x > t for x, t in zip(delta, (tol['center_and_edges_px'], tol['angle_rad'], tol['score']))):
        raise ValueError('Frozen B differs from previous fixed TRAIN prediction')
    return dict(max_center_edge_delta_px=delta[0], angle_delta_rad=delta[1], score_delta=delta[2])


def collect(cfg, protocol, samples, identity, cache_dir, gpu, progress):
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
    if ready.sha(CHECKPOINT) != identity['frozen_b']['checkpoint_sha256']:
        raise ValueError('Requires unchanged B ep24 checkpoint')
    import_modules_from_strings(**cfg.custom_imports)
    model_cfg = deepcopy(cfg.model); model_cfg.pretrained = None; model_cfg.train_cfg = None
    detector = build_detector(model_cfg)  # No init_weights/pretrained download.
    loaded = load_checkpoint(detector, str(CHECKPOINT), map_location='cpu', strict=True)
    checkpoint_meta = checkpoint_contract(loaded['meta'], cfg, 'b'); del loaded
    detector.cuda(gpu); freeze_detector(detector); before = state_digest(detector)
    records, costs, previews = [], [], []
    for scale in SETTINGS['scales']:
        dataset = ConcatDataset([build_dataset(s) for s in fixed_specs(cfg, scale)])
        lookup = {Path(info['filename']).stem: (part, i, offset+i)
                  for part, offset in zip(dataset.datasets, (0, len(dataset.datasets[0])))
                  for i, info in enumerate(part.data_infos)}
        if len(lookup) != sum(ready.TRAIN_COUNTS.values()):
            raise ValueError('TRAIN dataset count/unique image mismatch')
        previewed = set()
        for sample in samples:
            part, local_index, index = lookup[sample['image']]
            gt = torch.tensor(part.get_ann_info(local_index)['bboxes'])
            expected = canonical_boxes(torch.tensor(sample['gt']).reshape(1, 5))
            parsed_gt = canonical_boxes(gt)
            if gt.shape != (1, 5) or not torch.allclose(parsed_gt, expected, atol=1e-3, rtol=1e-5):
                raise ValueError('Current TRAIN GT differs from existing evidence')
            batch = scatter(collate([dataset[index]], samples_per_gpu=1), [gpu])[0]
            image, metas = batch['img'][0], batch['img_metas'][0]; meta = metas[0]
            if (len(batch['img']) != 1 or image.shape != (1, 3, 1024, 1024) or
                    Path(meta['filename']).stem != sample['image'] or meta.get('flip', False)):
                raise ValueError('Fixed TRAIN view substitution')
            checked_meta(meta); assert_detector_frozen(detector)

            def extract():
                with torch.no_grad():
                    features = detector.extract_feat(image)
                    raw = flatten_prediction(detector.simple_test_from_features(features, metas, rescale=False))
                    original = flatten_prediction(detector.simple_test_from_features(features, metas, rescale=True))
                return features, image.new_tensor(raw), image.new_tensor(original)
            (features, raw, original), detector_cost = measured(extract, gpu)
            if any(f.requires_grad or f.grad_fn is not None for f in features) or features[0].shape != (1, 256, 128, 128):
                raise ValueError('Frozen P3 graph/shape differs')
            restored = map_boxes(raw[:, :5], meta, inverse=True)
            if raw.shape != original.shape or not torch.allclose(restored, original[:, :5], atol=1e-4, rtol=1e-6):
                raise ValueError('Native B restoration differs')
            if not torch.equal(raw[:, 5], original[:, 5]):
                raise ValueError('Native rescale changed B scores')
            previous = check_previous(original, sample['previous_b_predictions'][str(scale)], protocol)
            def pool():
                local, points = {}, {}
                for arm in ARMS:
                    roi, support, points[arm] = sample_local(features[0], raw[:, :5], meta, arm)
                    local[arm] = dict(roi=roi.cpu(), support=support.cpu())
                return local, points
            (local, points), sampling_cost = measured(pool, gpu)
            cost = dict(detector_forward=detector_cost, local_sampling=sampling_cost)
            g = expected.to(original.device)
            center = float((original[0, :2]-g[0, :2]).norm()) if len(original) else None
            eligible = center is not None and center < SETTINGS['train_center_match_px']
            target = None
            if len(original):
                b = canonical_boxes(original[:, :5])
                target = torch.cat((torch.log(g[:, 2:4]/b[:, 2:4]), wrap_pi(g[:, 4]-b[:, 4])[:, None]), dim=1).cpu().tolist()[0]
            model_gt = map_boxes(g, meta, size_mode='annotation')
            record = dict(image=sample['image'], domain=sample['domain'], sequence=sample['sequence'],
                frame_id=int(sample['image'].rsplit('_', 1)[1]), role=sample['role'], scale=scale,
                eligible=eligible, center_error_px=center, boxes_original=original.cpu(),
                boxes_model=raw[:, :5].cpu(), gt_original=expected, local=local,
                parsed_gt_original=parsed_gt[0].tolist(),
                reference_gt_absolute_delta=(parsed_gt[0]-expected[0]).abs().tolist(),
                gt_input_short_cells=float(model_gt[:, 2:4].min()/SETTINGS['stride']),
                target_residual=target, previous_b_agreement=previous)
            records.append(record); costs.append(cost)
            if cpu_roi_bytes(records) > SETTINGS['max_cpu_roi_bytes']:
                raise ValueError('CPU local ROI cache exceeds declared 32MiB limit')
            if sample['role'] == 'fit' and sample['sequence'] not in previewed:
                path = cache_dir/'previews'/(sample['image']+'_scale'+str(scale)+'.png')
                preview(path, sample, original, points, meta); previews.append(str(path.relative_to(cache_dir)))
                previewed.add(sample['sequence'])
            progress(dict(stage='extract', image=sample['image'], scale=scale, eligible=eligible,
                          gt_short_cells=record['gt_input_short_cells'], cpu_roi_mib=cpu_roi_bytes(records)/2**20,
                          support_means={a: float(local[a]['support'].mean()) if len(original) else None for a in ARMS}, gpu=cost))
            # CPU record is the only retained view. No full FPN/graph accumulation.
            del features, raw, original, points, restored, image, batch, g, model_gt, metas
        del dataset, lookup
    assert_detector_frozen(detector); after = state_digest(detector)
    if before != after:
        raise ValueError('Frozen detector parameter/buffer mutation')
    del detector; torch.cuda.empty_cache()
    payload = dict(identity=identity, records=records, detector_state_before=before,
        detector_state_after=after, checkpoint_meta=checkpoint_meta, detector_costs=costs,
        extraction_runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmrotate=mmrotate.__version__, opencv=cv2.__version__,
            gpu=torch.cuda.get_device_name(gpu), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    path = cache_dir/'local_roi.pt'; torch.save(payload, str(path))
    cache = dict(status='COMPLETE', identity=identity, record_count=len(records),
        cpu_roi_bytes=cpu_roi_bytes(records), files={p: ready.sha(cache_dir/p) for p in ['local_roi.pt']+previews})
    write_new(cache_dir/'cache_manifest.json', cache)
    return payload, cache


def checked_cache(cache_dir, identity, samples):
    cache = json.loads((cache_dir/'cache_manifest.json').read_text())
    if cache['status'] != 'COMPLETE' or cache['identity'] != identity or cache['record_count'] != 128:
        raise ValueError('Incomplete/different local ROI cache')
    for p, digest in cache['files'].items():
        if Path(p).is_absolute() or '..' in Path(p).parts or ready.sha(cache_dir/p) != digest:
            raise ValueError('ROI cache artifact identity differs')
    payload = torch.load(str(cache_dir/'local_roi.pt'), map_location='cpu')
    if payload['identity'] != identity or payload['detector_state_before'] != payload['detector_state_after']:
        raise ValueError('ROI cache detector identity differs')
    keys = [(r['image'], r['scale'], r['role']) for r in payload['records']]
    expected = [(s['image'], scale, s['role']) for scale in SETTINGS['scales'] for s in samples]
    if keys != expected or cpu_roi_bytes(payload['records']) != cache['cpu_roi_bytes'] or cache['cpu_roi_bytes'] > SETTINGS['max_cpu_roi_bytes']:
        raise ValueError('ROI cache sample order/size differs')
    return payload, cache


def validate_records(records, samples):
    expected = [(s, scale) for scale in SETTINGS['scales'] for s in samples]
    if len(records) != len(expected):
        raise ValueError('Local cache view count differs')
    for r, (sample, scale) in zip(records, expected):
        if any(r[k] != sample[k] for k in ('image', 'domain', 'sequence', 'role')) or r['scale'] != scale:
            raise ValueError('Local cache role/order differs')
        b, bm, gt = (r[k] for k in ('boxes_original', 'boxes_model', 'gt_original'))
        n = len(b)
        tensors = [b, bm, gt]+[t for arm in ARMS for t in r['local'][arm].values()]
        if (n not in (0, 1) or b.shape != (n, 6) or bm.shape != (n, 5) or gt.shape != (1, 5)
                or any(t.device.type != 'cpu' or t.dtype != torch.float32 or t.requires_grad or t.grad_fn is not None
                       or not bool(torch.isfinite(t).all()) for t in tensors)):
            raise ValueError('Local cache must hold finite detached CPU float32 tensors')
        expected_gt = canonical_boxes(torch.tensor(sample['gt']).reshape(1, 5))
        if not torch.allclose(gt, expected_gt, atol=1e-6, rtol=0.):
            raise ValueError('Local cache GT differs')
        for arm in ARMS:
            roi, support = r['local'][arm]['roi'], r['local'][arm]['support']
            if roi.shape != (n, 256, 9, 9) or support.shape != (n, 1, 9, 9) or bool(((support < -1e-6) | (support > 1+1e-6)).any()):
                raise ValueError('Local cache feature/support shape differs')
        canonical_boxes(b[:, :5]); canonical_boxes(bm)
        center = float((b[0, :2]-gt[0, :2]).norm()) if n else None
        if r['eligible'] != (center is not None and center < SETTINGS['train_center_match_px']):
            raise ValueError('Local cache TRAIN eligibility differs')


def batch_tensors(records, indices, arm, device):
    selected = [records[i] for i in indices]
    values = [torch.cat([r['local'][arm][k] for r in selected]).to(device) for k in ('roi', 'support')]
    values += [torch.cat([r[k] for r in selected]).to(device) for k in ('boxes_original', 'boxes_model', 'gt_original')]
    return values


def evaluate(head, records, arm, device):
    rows, loss_groups, projections, saturations = [], {}, 0, [0, 0, 0]
    head.eval()
    with torch.no_grad():
        for i, r in enumerate(records):
            roi, support, b, bm, gt = batch_tensors(records, [i], arm, device)
            output = head(roi, support, b, bm)
            pred = output['boxes_original'].cpu().numpy()
            if len(pred) != len(b) or not torch.equal(output['boxes_original'][:, [0, 1, 5]], b[:, [0, 1, 5]]):
                raise ValueError('Refinement changed B outputs/centers/scores')
            row = {k: r[k] for k in ('image', 'domain', 'sequence', 'frame_id', 'role', 'scale', 'eligible')}
            row.update(pred=pred[0].tolist() if len(pred) else None,
                       gt_original=gt[0].cpu().tolist(),
                       parsed_gt_original=r.get('parsed_gt_original'),
                       reference_gt_absolute_delta=r.get('reference_gt_absolute_delta'),
                       metrics=decompose(gt[0].cpu().numpy(), pred[0, :5] if len(pred) else None))
            rows.append(row)
            if r['eligible']:
                loss, _ = regression_loss(output['boxes_original'], gt)
                loss_groups.setdefault(r['role']+'/'+r['domain']+'/'+str(r['scale']), []).append(float(loss))
            projections += int(output['edge_projection'].sum())
            caps = b.new_tensor([SETTINGS['max_log_edge_residual']]*2+[SETTINGS['max_angle_residual_rad']])
            saturations = [v+int(x) for v, x in zip(saturations, (output['residual'].abs() >= .95*caps).sum(dim=0))]
    groups = {}
    for role in ('fit', 'probe'):
        for domain in ('real', 'sim'):
            for scale in SETTINGS['scales']:
                key = role+'/'+domain+'/'+str(scale)
                groups[key] = summarize([r for r in rows if r['role'] == role and r['domain'] == domain and r['scale'] == scale], continuous=False)
    return dict(rows=rows, groups=groups, eligible_loss={k: float(np.mean(v)) for k, v in loss_groups.items()},
                edge_projection_views=projections, residual_saturation_counts=saturations)


def fit_arm(records, batches, initial, arm, device, progress):
    head = LocalGeometryRefiner().to(device); head.load_state_dict(initial, strict=True)
    before = state_digest(head); start = evaluate(head, records, arm, device)
    if any(not np.array_equal(np.asarray(row['pred']), r['boxes_original'][0].numpy())
           for row, r in zip(start['rows'], records) if row['pred'] is not None):
        raise ValueError('Zero-initialized G is not exactly B')
    optimizer = torch.optim.Adam(head.parameters(), lr=SETTINGS['lr'], weight_decay=0.)
    logs = []
    head.train()
    for step, indices in enumerate(batches, 1):
        roi, support, b, bm, gt = batch_tensors(records, indices, arm, device)
        optimizer.zero_grad(); output = head(roi, support, b, bm)
        loss, parts = regression_loss(output['boxes_original'], gt)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite G loss')
        tasks = None
        if step == 1:
            tasks = [float(torch.autograd.grad(p, head.output.weight, retain_graph=True)[0].double().norm()) for p in parts]
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in head.parameters()):
            raise ValueError('Missing/nonfinite head gradient')
        stem_norm = sum(float(p.grad.double().square().sum()) for p in head.stem.parameters())**.5
        norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), SETTINGS['clip_norm']))
        if not math.isfinite(norm):
            raise ValueError('Nonfinite G clipping norm')
        optimizer.step()
        progress(dict(stage='update', arm=arm, step=step))
        if any(not bool(torch.isfinite(p).all()) for p in head.parameters()):
            raise ValueError('Nonfinite updated head')
        if step in (1, 2) or step % 25 == 0:
            log = dict(stage='fit', arm=arm, step=step, loss=float(loss.detach()),
                       parts=parts.detach().cpu().tolist(), grad_norm_before_clip=norm,
                       clip_multiplier=min(1., SETTINGS['clip_norm']/(norm+1e-6)), stem_grad_norm=stem_norm,
                       final_layer_task_grad_norms=tasks)
            logs.append(log); progress(log)
    final = evaluate(head, records, arm, device)
    result = dict(initial_state=before, final_state=state_digest(head), initial=start, final=final, logs=logs,
        finite_gradients=True, initial_all_task_gradients_effective=all(x > 0 for x in logs[0]['final_layer_task_grad_norms']),
        stem_gradient_after_zero_output_step_effective=logs[1]['stem_grad_norm'] > 0,
        fit_loss_decreased_by_group={k: final['eligible_loss'][k] < v-1e-6 for k, v in start['eligible_loss'].items() if k.startswith('fit/')})
    del head, optimizer; torch.cuda.empty_cache()
    return result


def support_report(records, protocol):
    counts = Counter((r['role'], r['domain']) for r in records if r['eligible'])
    minimum = protocol['minimum_eligible_views']
    if any(counts[(role, domain)] < n for role, domains in minimum.items() for domain, n in domains.items()):
        raise ValueError('Insufficient genuine B center-matched TRAIN support')
    report = {}
    for role in ('fit', 'probe'):
        for domain in ('real', 'sim'):
            for scale in SETTINGS['scales']:
                rows = [r for r in records if r['role'] == role and r['domain'] == domain and r['scale'] == scale]
                out = [r for r in rows if len(r['boxes_original'])]
                caps = np.asarray([SETTINGS['max_log_edge_residual']]*2+[SETTINGS['max_angle_residual_rad']])
                report[role+'/'+domain+'/'+str(scale)] = dict(frames=len(rows), outputs=len(out), eligible_views=sum(r['eligible'] for r in rows),
                    gt_short_cells=describe([r['gt_input_short_cells'] for r in rows]),
                    target_outside_bounds_by_component=[sum(abs(r['target_residual'][j]) > caps[j] for r in out) for j in range(3)],
                    support_means={a: describe([float(r['local'][a]['support'].mean()) for r in out]) for a in ARMS})
    return report


def geometry_deltas(reference, current):
    """Signed final-minus-reference changes; no score/step selection."""
    result = {}
    for key, old in reference['groups'].items():
        new = current['groups'][key]
        fixed = ('frames', 'output_frames', 'output_coverage_pct', 'output_center_hit_pct', 'all_frame_center_hit_pct')
        if any(old[k] != new[k] for k in fixed):
            raise ValueError('Center coverage changed after refinement')
        delta = dict(all_frame_mean_riou=new['all_frame_mean_riou']-old['all_frame_mean_riou'])
        for field, stat in (('center_error_px', 'mean'), ('center_error_px', 'rmse'),
                            ('long_edge_relative_error', 'mean'), ('short_edge_relative_error', 'mean'),
                            ('angle_error_deg', 'rmse'), ('angle_error_deg', 'p90'), ('protocol_angle', 'rmse')):
            a, b = old[field][stat], new[field][stat]
            delta[field+'/'+stat] = b-a if a is not None and b is not None else None
        result[key] = delta
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--cache-dir', default='work_dirs/port_geometry_g_v1_roi_cache')
    ap.add_argument('--reuse-cache', action='store_true')
    args = ap.parse_args(); os.chdir(ROOT)
    out = Path(args.out_json).resolve(); progress_path = out.with_suffix('.progress.jsonl')
    artifacts = out.with_suffix('.artifacts.json'); cache_dir = Path(args.cache_dir).resolve()
    for p in (out, progress_path, artifacts):
        if p.exists():
            raise FileExistsError('Refusing to overwrite '+str(p))
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=SETTINGS['protocol'], evidence_role='TRAIN_only_finite_check', settings=SETTINGS,
                  status='STARTED', detector_updates=0, head_updates_total=0, checkpoint_exported=False)
    with progress_path.open('x') as stream:
        extracted = [0]
        def progress(value):
            if value['stage'] == 'update':
                report['head_updates_total'] += 1
                return
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()
            if value['stage'] == 'fit':
                print('TRAIN', value.get('arm'), 'step', value.get('step'), 'loss', value.get('loss'), flush=True)
            elif value['stage'] == 'extract':
                extracted[0] += 1
                if extracted[0] % 8 == 0:
                    print('TRAIN cached', extracted[0], '/128 views; CPU ROI MiB', value['cpu_roi_mib'], flush=True)
        try:
            cfg, protocol, samples, identity = checked_inputs()
            for scale in SETTINGS['scales']:
                fixed_specs(cfg, scale)
            report['identity'] = identity
            report['fixed_sample_counts'] = {role: dict(Counter(s['domain'] for s in samples if s['role'] == role)) for role in ('fit', 'probe')}
            report['runtime'] = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                                    cudnn=torch.backends.cudnn.version(), numpy=np.__version__, opencv=cv2.__version__,
                                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
            if args.check_only:
                if args.reuse_cache:
                    raise ValueError('--check-only does not load a feature cache')
                report['status'] = 'STATIC_CHECK_COMPLETE_NO_GPU_NO_UPDATES'
            else:
                if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
                    raise ValueError('A valid logical CUDA device is required')
                torch.cuda.set_device(args.gpu)
                seed_all(); device = torch.device('cuda', args.gpu)
                if args.reuse_cache:
                    payload, cache = checked_cache(cache_dir, identity, samples)
                else:
                    cache_dir.mkdir(parents=True, exist_ok=False); (cache_dir/'previews').mkdir()
                    payload, cache = collect(cfg, protocol, samples, identity, cache_dir, args.gpu, progress)
                records = payload['records']; report['cache'] = cache
                validate_records(records, samples)
                report['cache_reused'] = args.reuse_cache
                report['extraction_runtime'] = payload['extraction_runtime']
                report['detector_state_unchanged'] = payload['detector_state_before'] == payload['detector_state_after']
                report['checkpoint_meta'] = payload['checkpoint_meta']; report['detector_costs'] = payload['detector_costs']
                report['train_support'] = support_report(records, protocol)
                seed_all(); initial = LocalGeometryRefiner().state_dict()
                batches = schedule(records); report['minibatches'] = batches
                report['arms'], report['refiner_costs'] = {}, {}
                for arm in ARMS:
                    result, cost = measured(lambda: fit_arm(records, batches, initial, arm, device, progress), args.gpu)
                    report['arms'][arm] = result; report['refiner_costs'][arm] = cost
                if report['arms']['ordinary']['initial_state'] != report['arms']['aligned']['initial_state']:
                    raise ValueError('Arm initialization differs')
                report['final_geometry_deltas'] = {arm: geometry_deltas(report['arms'][arm]['initial'], report['arms'][arm]['final']) for arm in ARMS}
                report['aligned_minus_ordinary_final'] = geometry_deltas(report['arms']['ordinary']['final'], report['arms']['aligned']['final'])
                report['status'] = 'TRAIN_SHORT_FIT_COMPLETE_REVIEW_REQUIRED'
                report['limitations'] = ['TRAIN probe shares training sequences and B has trained on every image; not VAL/generalization.',
                    'No contiguous video evaluation, center improvement or independent depth verification.',
                    'Fixed final step only; manual spatial-support previews and joint geometry review required before formal training.',
                    'TEST repeatedly exposed in prior work; no TEST access or tuning in this check.']
        except Exception as error:
            report['status'] = 'FAILED_REVIEW_REQUIRED'; report['error'] = type(error).__name__+': '+str(error)
            write_new(out, report)
            stream.flush()
            write_new(artifacts, dict(protocol=SETTINGS['protocol'], status=report['status'],
                files={str(p): ready.sha(p) for p in (out, progress_path)},
                cache_manifest_sha256=ready.sha(cache_dir/'cache_manifest.json') if (cache_dir/'cache_manifest.json').exists() else None))
            raise
    write_new(out, report)
    artifact = dict(protocol=SETTINGS['protocol'], status=report['status'], files={str(p): ready.sha(p) for p in (out, progress_path)},
                    cache_manifest_sha256=ready.sha(cache_dir/'cache_manifest.json') if not args.check_only else None)
    write_new(artifacts, artifact)
    print('Saved', out, 'status', report['status'], flush=True)


if __name__ == '__main__':
    main()
