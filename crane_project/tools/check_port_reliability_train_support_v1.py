#!/usr/bin/env python3
"""Fix TRAIN direction policy and count genuine frozen-B TRAIN error support.

check: CPU qualification + replay of already returned preflight scores.
collect: one clean view per TRAIN image, batch1, no optimization; reuse a complete
verified cache automatically. reuse: CPU report from that same cache only.
No VAL/TEST datasets, artificial probes in support counts, or fitted heads.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import check_port_reliability_readiness_v1 as ready
from crane_project.tools import preflight_port_structure_reliability_v1 as prior
from crane_project.utils.port_structure_reliability_v1 import (
    assert_detector_frozen, canonical_boxes, checked_meta, component_probes_original,
    freeze_detector, quality_loss)
from crane_project.utils.port_reliability_train_policy_v1 import (
    POLICY, direction_qualification, train_quality_targets_original)

VERSION = 'port_reliability_train_support_v1'
MANIFEST = ROOT/'crane_project/tools/port_reliability_train_support_v1_sources.json'
PREFLIGHT_SHA = '65cdca07965058b554c5f253301ac316bc8264045a750ac34c753841af56fa04'
VIEW = dict(name='clean_train_original_equivalent', scale=1., horizontal_flip=False,
            pipeline='unchanged_B_single_scale_inference_pipeline_on_TRAIN_paths',
            batch_size=1, rescale=True, score_thr=.05, max_per_img=1)
LIMITS = dict(center_px=15., size_max_relative=.10, angle_deg=3.)
REPLAY_CORNER_TOL_PX = 1e-3  # numeric identity only, not an accuracy threshold


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def checked_sources():
    previous = prior.checked_sources()
    m = json.loads(MANIFEST.read_text())
    if m.get('protocol') != VERSION or m.get('direction_policy') != POLICY or m.get('view') != VIEW:
        raise ValueError('Fixed TRAIN support policy/view differs')
    current = {p: ready.sha(ROOT/p) for p in m['sources']}
    if current != m['sources']:
        raise ValueError('Reviewed TRAIN support source SHA differs')
    return dict(manifest_sha256=ready.sha(MANIFEST), sources=current,
                prior_44_source_contract=previous)


def train_inputs():
    """Bind current images and offline policy to reviewed TRAIN bytes only.

    The caller first verifies the complete structure report through prerequisites.
    This reads numeric annotations/axis identity, without repeating visual audit.
    """
    legacy = json.loads((ready.DATA/'provenance/axis_legacy_train_v1/manifest.json').read_text())
    rows = []
    for split in ('train', 'train_sim'):
        for ann in ready.files(ready.DATA/split/'annfiles', '.txt'):
            name = ann.stem; sequence = name.rsplit('_', 1)[0]
            if ready.TRAIN_SPLITS.get(sequence) != split:
                raise ValueError('Non-TRAIN image role: '+name)
            image = ready.DATA/split/'images'/(name+'.jpg')
            with Image.open(image) as im:
                size = im.size
            _, box = ready.polygon_box(ann)
            gt = canonical_boxes(torch.tensor(box, dtype=torch.float32).reshape(1, 5))[0].tolist()
            row = dict(image=name, sequence=sequence, domain=name.split('_', 1)[0],
                       split=split, image_size=list(size), gt_original=gt,
                       image_sha256=ready.sha(image), annotation_sha256=ready.sha(ann),
                       dota_difficulty=int(ann.read_text().split()[9]))
            axis = None
            k = None
            if row['domain'] == 'real':
                folder = 'axis_legacy_train_v1' if sequence in ready.LEGACY_TRAIN else 'axis_k2p1'
                path = ready.DATA/'provenance'/folder/sequence/'axis_json'/(name+'.json')
                axis, _ = ready.read_axis(path, name, size, legacy=sequence in ready.LEGACY_TRAIN)
                row['axis_json_sha256'] = ready.sha(path)
                k = ready.NATIVE_K[sequence]
                if sequence in ready.LEGACY_TRAIN:
                    old = legacy['rows'][name]
                    if any(old[key] != row[key] for key in (
                            'image_sha256', 'annotation_sha256', 'axis_json_sha256')):
                        raise ValueError('Legacy native-axis source binding differs: '+name)
            # Sim explicitly has no native-axis path or real conversion k.
            row['direction_qualification'] = direction_qualification(gt, row['domain'], axis, k)
            rows.append(row)
    if Counter(r['sequence'] for r in rows) != Counter(ready.TRAIN_COUNTS):
        raise ValueError('Expected exact 2558 TRAIN frames and six sequences')
    return rows


def qualification_counts(inputs):
    groups = dict(all=inputs)
    groups.update({'domain:'+d: [r for r in inputs if r['domain'] == d] for d in ('real', 'sim')})
    groups.update({'sequence:'+s: [r for r in inputs if r['sequence'] == s] for s in ready.TRAIN_COUNTS})
    return {name: dict(frames=len(rows),
        train_direction_eligible=sum(r['direction_qualification']['train_angle_eligible'] for r in rows),
        evaluation_direction_eligible=sum(r['direction_qualification']['evaluation_angle_eligible'] for r in rows),
        added_train_direction_frames=sum(r['direction_qualification']['train_angle_eligible'] and
            not r['direction_qualification']['evaluation_angle_eligible'] for r in rows),
        direction_gt_sources=dict(Counter(r['direction_qualification']['direction_gt_source'] for r in rows)))
            for name, rows in groups.items()}


def replay_preflight(path, inputs):
    """CPU loss-output derivatives at cached q; no new network/GPU experiment."""
    if ready.sha(path) != PREFLIGHT_SHA:
        raise ValueError('Requires the already reviewed returned 14-view preflight')
    saved = json.loads(path.read_text())
    if saved['status'] != 'TRAIN_PREFLIGHT_COMPLETE_REVIEW_REQUIRED' or len(saved['rows']) != 14:
        raise ValueError('Unexpected returned preflight')
    indexed = {r['image']: r for r in inputs}
    results = []
    for row in saved['rows']:
        source = indexed[row['image']]
        gt = torch.tensor(row['original_gt'], dtype=torch.float32)
        corner_delta = ready.corner_difference(ready.box_polygon(gt.numpy()[0]),
                                              ready.box_polygon(source['gt_original']))
        if corner_delta > REPLAY_CORNER_TOL_PX:
            raise ValueError('Returned preflight original GT differs')
        for key in ('image_sha256', 'annotation_sha256'):
            if row[key] != source[key]:
                raise ValueError('Returned preflight TRAIN identity differs')
        probes, _ = component_probes_original(gt)
        genuine = torch.tensor(row['original_b_output_original'], dtype=torch.float32).reshape(-1, 5)
        boxes = torch.cat((genuine, probes))
        # OpenCV versions can differ by subpixel rounding on the same polygon.
        # Replay against the RETURNED original GT to preserve its cached labels;
        # the new TRAIN cache separately binds current loader GT/numeric runtime.
        qualification = direction_qualification(gt.numpy()[0], source['domain'],
            np.asarray(row['axis_original']) if source['domain'] == 'real' else None,
            ready.NATIVE_K[source['sequence']] if source['domain'] == 'real' else None)
        if qualification['train_angle_eligible'] != source['direction_qualification']['train_angle_eligible']:
            raise ValueError('Returned/current TRAIN direction eligibility differs')
        target, mask, _ = train_quality_targets_original(boxes, gt, qualification)
        old_target = torch.tensor(row['offline_quality_targets'])
        if not torch.allclose(target, old_target, atol=2e-5, rtol=1e-5):
            raise ValueError('Continuous quality values changed in replay')
        q = torch.tensor(row['qualities_before_update'], requires_grad=True)
        loss, _ = quality_loss(q, target, mask, row['genuine_count'])
        loss.backward()
        results.append(dict(image=row['image'], view=row['view'],
            old_angle_mask=float(row['offline_quality_mask'][0][2]),
            train_angle_mask=float(mask[0, 2]),
            quality_loss_new_mask=float(loss.detach()),
            angle_dloss_dq_norm=float(q.grad[:, 2].norm()),
            continuous_target_max_numeric_delta=float((target-old_target).abs().max()),
            current_vs_returned_gt_corner_max_px=corner_delta,
            continuous_target_formula_unchanged=True))
    return dict(preflight_sha256=PREFLIGHT_SHA, rows=results,
                newly_supervised_views=sum(r['old_angle_mask'] == 0 and r['train_angle_mask'] == 1 for r in results),
                scope='CPU loss derivatives wrt cached q only; no new network gradient or fitted benefit claim')


def inference_specs(cfg):
    specs = deepcopy(cfg.data.train)
    if len(specs) != 2:
        raise ValueError('Expected two TRAIN dataset specs')
    pipeline = deepcopy(cfg.data.val.pipeline)  # config only; never open VAL data
    multi = pipeline[1]
    if (len(pipeline) != 2 or pipeline[0]['type'] != 'LoadImageFromFile'
            or multi['type'] != 'MultiScaleFlipAug' or tuple(multi['img_scale']) != (1024, 1024)
            or multi.get('flip', False)
            or [t['type'] for t in multi['transforms']] !=
               ['RResize', 'Normalize', 'Pad', 'DefaultFormatBundle', 'Collect']):
        raise ValueError('Unexpected fixed B inference pipeline')
    for spec, split in zip(specs, ('train', 'train_sim')):
        root = Path(spec.get('data_root', ''))
        if ((root/spec['ann_file']).resolve() != (ready.DATA/split/'annfiles').resolve()
                or (root/spec['img_prefix']).resolve() != (ready.DATA/split/'images').resolve()):
            raise ValueError('Only allowlisted TRAIN paths permitted')
        spec.update(pipeline=deepcopy(pipeline), test_mode=True)
    return specs, pipeline


def checked_prediction(pred):
    if pred is None:
        return None
    a = np.asarray(pred, dtype=float)
    if (a.shape != (6,) or not np.isfinite(a).all() or min(a[2:4]) <= 0
            or not .05 < a[5] <= 1.):
        raise ValueError('Expected genuine B original-coordinate top1 or no output')
    return a.tolist()


def support_rows(inputs, predictions):
    if len(inputs) != len(predictions):
        raise ValueError('Prediction frame count differs')
    rows = []
    for source, pred in zip(inputs, predictions):
        pred = checked_prediction(pred)
        rows.append(dict(source, pred_original=pred, prediction_source='genuine_frozen_b',
                         errors=None if pred is None else ready.geometry_errors(source['gt_original'], pred[:5])))
    return rows


def summarize(rows):
    n = len(rows); out = [r for r in rows if r['pred_original'] is not None]
    hits = sum(r['errors']['center_px'] < 15 for r in out)
    result = dict(frames=n, output_frames=len(out), no_output_frames=n-len(out),
        output_coverage=len(out)/n if n else None, center_hits=hits,
        center_hit_rate_on_outputs=hits/len(out) if out else None,
        all_frame_center_correct_coverage=hits/n if n else None,
        center_bad_outputs=len(out)-hits,
        error_distributions={k: ready.describe([r['errors'][k] for r in out]) for k in (
            'center_px', 'long_relative', 'short_relative', 'size_max_relative',
            'long_signed_log', 'short_signed_log')},
        size_bad_outputs=sum(r['errors']['size_max_relative'] > .10 for r in out))
    for role, key in [('train', 'train_angle_eligible'), ('evaluation', 'evaluation_angle_eligible')]:
        eligible = [r for r in rows if r['direction_qualification'][key]]
        assessed = [r for r in out if r['direction_qualification'][key]]
        result[role+'_angle'] = dict(eligible_frames=len(eligible), assessed_output_frames=len(assessed),
            eligible_no_output_frames=sum(r['pred_original'] is None for r in eligible),
            unassessed_output_frames=len(out)-len(assessed),
            bad_outputs=sum(r['errors']['angle_deg'] > 3 for r in assessed),
            errors_deg=ready.describe([r['errors']['angle_deg'] for r in assessed]),
            definition='unsigned pi-periodic pure error; no missing/center 90-degree penalty')
        # Joint failures only where all three components are assessable.
        patterns = Counter('center=%d,size=%d,angle=%d' % (
            r['errors']['center_px'] >= 15, r['errors']['size_max_relative'] > .10,
            r['errors']['angle_deg'] > 3) for r in assessed)
        result[role+'_joint_error_patterns_on_assessed_outputs'] = dict(patterns)
    result['support_note'] = 'Descriptive genuine TRAIN frames, temporally correlated; no fitted reliability or support-sufficiency verdict'
    return result


def support_report(rows):
    groups = dict(all=rows)
    groups.update({'domain:'+d: [r for r in rows if r['domain'] == d] for d in ('real', 'sim')})
    groups.update({'sequence:'+s: [r for r in rows if r['sequence'] == s] for s in ready.TRAIN_COUNTS})
    return {key: summarize(values) for key, values in groups.items()}


def read_cache(cache, identity, inputs):
    stored = json.loads((cache/'identity.json').read_text())
    complete = json.loads((cache/'complete.json').read_text())
    if stored != identity or complete['identity_sha256'] != fingerprint(identity):
        raise ValueError('TRAIN cache provenance differs; use a new cache directory')
    raw = cache/'predictions.jsonl'
    if complete['prediction_file_sha256'] != ready.sha(raw):
        raise ValueError('TRAIN prediction cache content SHA differs')
    if (complete['frames'] != len(inputs)
            or complete['checkpoint_sha256'] != identity['frozen_b']['checkpoint_sha256']
            or complete['b_state_before'] != complete['b_state_after']
            or complete['detector_optimizer_steps'] != 0):
        raise ValueError('Incomplete/unfrozen TRAIN cache')
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    if len(records) != len(inputs):
        raise ValueError('TRAIN cache row count differs')
    predictions = []
    for index, (record, source) in enumerate(zip(records, inputs)):
        if (record['index'] != index or record['image'] != source['image']
                or record['input_sha256'] != fingerprint(source)):
            raise ValueError('TRAIN cache row identity/order differs')
        predictions.append(checked_prediction(record['pred_original']))
    return predictions, complete


def collect(cache, identity, inputs, cfg, specs, checkpoint, gpu):
    """Sequential batch1 eval; no features/graphs accumulated across frames."""
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    import mmcv
    import cv2
    if ready.sha(checkpoint) != ready.FROZEN_B['checkpoint_sha256']:
        raise ValueError('Requires retained B epoch24 checksum')
    import_modules_from_strings(**cfg.custom_imports)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prior.seed_all()
    detector = build_detector(deepcopy(cfg.model))
    loaded = load_checkpoint(detector, str(checkpoint), map_location='cpu', strict=True)
    if loaded.get('meta', {}).get('epoch') != 24:
        raise ValueError('Checkpoint epoch differs')
    del loaded
    freeze_detector(detector.cuda(gpu))
    before = prior.state_digest(detector)
    datasets = [build_dataset(spec) for spec in specs]
    indexed = [(part, index) for part in datasets for index in range(len(part))]
    if len(indexed) != len(inputs):
        raise ValueError('Actual TRAIN dataset count differs')
    for (part, index), source in zip(indexed, inputs):
        if Path(part.data_infos[index]['filename']).stem != source['image']:
            raise ValueError('Actual TRAIN dataset index differs')
        actual_gt = part.get_ann_info(index)['bboxes']
        if actual_gt.shape != (1, 5):
            raise ValueError('Expected one genuine TRAIN GT')
        g = canonical_boxes(torch.tensor(actual_gt))[0].numpy()
        if not np.allclose(g, source['gt_original'], atol=1e-5, rtol=1e-6):
            raise ValueError('Actual loader GT differs from offline policy GT')
    cache.mkdir(parents=True, exist_ok=False)
    prior.write_new(cache/'identity.json', identity)
    torch.cuda.reset_peak_memory_stats(gpu)
    with (cache/'predictions.jsonl').open('x') as stream:
        for i, ((part, index), source) in enumerate(zip(indexed, inputs)):
            batch = scatter(collate([part[index]], samples_per_gpu=1), [gpu])[0]
            meta = batch['img_metas'][0][0]
            checked_meta(meta)  # keep-ratio / padding / scale contract
            if (len(batch['img']) != 1 or meta.get('flip', False)
                    or Path(meta['filename']).stem != source['image']
                    or list(meta['ori_shape'][:2][::-1]) != source['image_size']):
                raise ValueError('TRAIN view/coordinate identity differs')
            with torch.no_grad():
                result = detector(return_loss=False, rescale=True, **batch)
            array = prior.flatten_prediction(result)
            pred = checked_prediction(array[0].tolist() if len(array) else None)
            record = dict(index=i, image=source['image'], input_sha256=fingerprint(source), pred_original=pred)
            stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
            if (i+1) % 100 == 0 or i+1 == len(inputs):
                print('genuine TRAIN B', i+1, '/', len(inputs), flush=True)
            del batch, meta, result, array
    assert_detector_frozen(detector)
    after = prior.state_digest(detector)
    if before != after:
        raise ValueError('Frozen B parameters or buffers changed')
    complete = dict(identity_sha256=fingerprint(identity), frames=len(inputs),
        prediction_file_sha256=ready.sha(cache/'predictions.jsonl'),
        b_state_before=before, b_state_after=after, detector_optimizer_steps=0,
        checkpoint_sha256=ready.sha(checkpoint), torch_version=torch.__version__,
        mmcv_version=mmcv.__version__, opencv_version=cv2.__version__,
        cuda_device=torch.cuda.get_device_name(gpu),
        max_allocated_mib=torch.cuda.max_memory_allocated(gpu)/2**20,
        max_reserved_mib=torch.cuda.max_memory_reserved(gpu)/2**20)
    prior.write_new(cache/'complete.json', complete)
    del detector
    return read_cache(cache, identity, inputs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('check', 'collect', 'reuse'), default='check')
    ap.add_argument('--structure-report', type=Path,
                    default=Path('work_dirs/port_reliability_readiness_v1_structure_complete/train_structure_check.json'))
    ap.add_argument('--preflight-json', type=Path,
                    default=Path('work_dirs/port_structure_reliability_v1_train_preflight.json'))
    ap.add_argument('--checkpoint', type=Path, default=prior.CHECKPOINT)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--cache-dir', type=Path, default=Path('work_dirs/port_reliability_train_support_v1_cache'))
    ap.add_argument('--out-dir', type=Path, default=Path('work_dirs/port_reliability_train_support_v1'))
    args = ap.parse_args()
    import os
    os.chdir(ROOT)
    output = args.out_dir/('train_contract_check.json' if args.mode == 'check' else 'b_train_error_support.json')
    if output.exists():
        raise FileExistsError('Preserve existing report; choose a new --out-dir: '+str(output))
    sources = checked_sources()
    prerequisite = prior.prerequisites(args.structure_report)
    cfg = prior.check_cfg()
    specs, pipeline = inference_specs(cfg)
    inputs = train_inputs()
    replay = replay_preflight(args.preflight_json, inputs)
    identity = dict(protocol=VERSION, source_contract=sources, direction_policy=POLICY,
        frozen_b=dict(epoch=24, config_sha256=ready.FROZEN_B['config_sha256'],
                      checkpoint_sha256=ready.FROZEN_B['checkpoint_sha256']),
        reviewed_structure_sha256=prior.STRUCTURE_SHA, reviewed_preflight_sha256=PREFLIGHT_SHA,
        train_sources=prerequisite['source_identities'],
        input_manifest_sha256=fingerprint(inputs), frames=len(inputs),
        view=VIEW, inference_pipeline_sha256=fingerprint(pipeline))
    report = dict(protocol=VERSION, mode=args.mode, identity=identity, qualification=qualification_counts(inputs),
        cpu_cached_q_replay=replay, optimizer_steps=0, val_or_test_read=False,
        test_repeatedly_exposed=True, direction_policy=POLICY, error_limits=LIMITS,
        limitations=['TRAIN clean view only, not the distribution of all random augmented TRAIN views.',
                     'Errors are relative to existing OBB GT; real short edges remain axis-derived L/k.',
                     'No native sim axis or visibility label; Webots OBB is GT, derived line is a structural proxy.',
                     'No independent depth GT, reliability fitting, or evidence of generalization gain.'])
    if args.mode == 'check':
        report['status'] = 'TRAIN_POLICY_CHECK_PASS_NO_INFERENCE'
    else:
        if args.cache_dir.exists():
            predictions, runtime = read_cache(args.cache_dir, identity, inputs)
            report['cache_action'] = 'reused_complete_verified_cache_no_gpu'
        elif args.mode == 'reuse':
            raise FileNotFoundError('Complete TRAIN cache is required for CPU reuse')
        else:
            predictions, runtime = collect(args.cache_dir, identity, inputs, cfg, specs, args.checkpoint, args.gpu)
            report['cache_action'] = 'collected_genuine_train_once'
        report['cache'] = dict(directory=str(args.cache_dir.resolve()), completion=runtime)
        report['strata'] = support_report(support_rows(inputs, predictions))
        report['status'] = 'GENUINE_TRAIN_SUPPORT_COMPLETE_REVIEW_REQUIRED'
    if checked_sources() != sources or fingerprint(train_inputs()) != identity['input_manifest_sha256']:
        raise ValueError('TRAIN sources/images changed during the run')
    prior.write_new(output, report)
    print('qualification', report['qualification']['all'], flush=True)
    if 'strata' in report:
        for key in ('domain:real', 'domain:sim'):
            s = report['strata'][key]
            print(key, 'frames', s['frames'], 'outputs', s['output_frames'],
                  'center_bad', s['center_bad_outputs'], 'size_bad', s['size_bad_outputs'],
                  'angle_bad_train/eval', s['train_angle']['bad_outputs'],
                  s['evaluation_angle']['bad_outputs'], flush=True)
    print('Saved', output, 'status', report['status'], flush=True)


if __name__ == '__main__':
    main()
