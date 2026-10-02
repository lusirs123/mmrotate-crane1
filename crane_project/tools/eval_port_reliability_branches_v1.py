#!/usr/bin/env python3
"""Fixed epoch8 source-VAL quality comparison. No training, probes, or TEST.

Preserves the previously selected B boxes and independently reports raw center
coverage and each component's matched-count diagnostic acceptance. Exports no
deployment threshold and never removes centers when size/angle is rejected.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import train_port_reliability_branches_v1 as train
from crane_project.tools import check_port_reliability_readiness_v1 as ready
from crane_project.tools import preflight_port_structure_reliability_v1 as prior
from crane_project.utils import port_reliability_branches_v1 as branch
from crane_project.utils.port_structure_reliability_v1 import assert_detector_frozen, checked_meta, map_boxes
from crane_project.utils.port_reliability_train_policy_v1 import POLICY


def val_cache(path, require_native):
    if branch.sha(path) != train.VAL_REPORT_SHA:
        raise ValueError('Requires unchanged reviewed B/E-H source-VAL report')
    saved = json.loads(path.read_text())
    if (saved['protocol'] != 'port_shape_e_h_v1_val_compare'
            or saved['evidence_role'] != 'source_val_only_fixed_experiment_comparison'
            or saved['metric_consistency_review_required'] is not False):
        raise ValueError('Wrong cached evidence role')
    identity = saved['identities']['b']
    if any(identity.get(k) != v for k, v in ready.FROZEN_B.items()):
        raise ValueError('Cached B identity differs')
    annotations = ready.files(ready.DATA/'val/annfiles', '.txt')
    if ready.set_sha(annotations) != ready.FROZEN_B['annotation_sha256']:
        raise ValueError('VAL annotation bytes differ')
    original = saved['geometry']['rows']['b']
    rows = ready.validate_cached_rows(original, annotations, ready.DATA/'val/images')
    native = ready.native_cache_checks(identity, original, require_native=require_native)
    original_index = {r['image']: r for r in original}
    for r in rows:
        old = original_index[r['image']]
        r.update(frame_id=old['frame_id'], image_sha256=old['image_sha256'],
                 split='val', image_size=None)
    return rows, dict(report_sha256=branch.sha(path), native=native, identity=identity)


def fixed_bundle(path, sources, protocol):
    payload = branch.read_checkpoint(path, for_val=True)
    c = payload['contract']
    if (c['sources'] != sources or c['protocol'] != protocol or c['direction_policy'] != POLICY
            or c['frozen_b'] != ready.FROZEN_B or c['numeric_input_snapshot_sha256'] != train.SNAPSHOT_SHA
            or c['train_input_manifest_sha256'] != '2094bb4cde19dd97909bcd02b36db19c09c6d5c348c0b246e9b159a68c1b4022'):
        raise ValueError('Formal bundle uses another reviewed source/data/policy')
    completion = json.loads((path.parent/'completion.json').read_text())
    if (completion['status'] != 'PAIRED_FORMAL_TRAIN_COMPLETE_FIXED_EPOCH8'
            or completion['sources'] != sources or completion['protocol'] != protocol
            or completion['checkpoint_sha256'] != branch.sha(path)
            or completion['optimizer_steps_per_arm'] != 20464
            or not completion['save_reload_quality_exact'] or not completion['b_raw_unchanged']
            or completion['detector_optimizer_steps'] != 0
            or completion['b_state'] != c['frozen_b_state']
            or completion['train_log_sha256'] != branch.sha(path.parent/'train_steps.jsonl')):
        raise ValueError('Incomplete/unverified formal training completion')
    return payload


def evaluate(args, rows, cache_identity, sources, protocol, cfg):
    from mmcv.parallel import collate, scatter
    from mmrotate.datasets import build_dataset
    payload = fixed_bundle(args.branch_checkpoint, sources, protocol)
    detector, runtime = train.build_runtime(cfg, args.b_checkpoint, args.gpu)
    frozen = prior.state_digest(detector)
    if frozen != payload['frozen_b_state']:
        raise ValueError('B state differs from formal TRAIN')
    arms = branch.make_arms('cuda:'+str(args.gpu))
    if branch.architecture(arms) != payload['contract']['architecture']:
        raise ValueError('Online architecture differs from training')
    branch.load_bundle(payload, arms)
    del payload
    arm_before = {k: prior.state_digest(v) for k, v in arms.items()}
    spec = deepcopy(cfg.data.val)
    if ((Path(spec.get('data_root', ''))/spec.ann_file).resolve() != (ready.DATA/'val/annfiles').resolve()
            or (Path(spec.get('data_root', ''))/spec.img_prefix).resolve() != (ready.DATA/'val/images').resolve()):
        raise ValueError('Only allowlisted VAL paths permitted')
    spec.test_mode = True
    dataset = build_dataset(spec)
    if len(dataset) != 887 or len(rows) != 887:
        raise ValueError('Expected 887 VAL frames')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats(args.gpu)
    records = []
    with (args.out_dir/'val_qualities.jsonl').open('x') as stream:
        for index, source in enumerate(rows):
            if Path(dataset.data_infos[index]['filename']).stem != source['image']:
                raise ValueError('Actual VAL dataset index/order differs')
            batch = scatter(collate([dataset[index]], samples_per_gpu=1), [args.gpu])[0]
            if len(batch['img']) != 1 or len(batch['img_metas']) != 1:
                raise ValueError('Only fixed single-scale B inference permitted')
            image, metas = batch['img'][0], batch['img_metas'][0]
            meta = metas[0]
            checked_meta(meta)
            if meta.get('flip', False) or Path(meta['filename']).stem != source['image']:
                raise ValueError('Unexpected VAL reflection/image role')
            with torch.no_grad():
                features = detector.extract_feat(image)
                raw = prior.flatten_prediction(detector.simple_test_from_features(features, metas, rescale=False))
                native = prior.flatten_prediction(detector.simple_test_from_features(features, metas, rescale=True))
            if any(f.requires_grad or f.grad_fn is not None for f in features):
                raise ValueError('Frozen B retained autograd graph')
            boxes, scores = image.new_tensor(raw[:, :5]), image.new_tensor(raw[:, 5])
            restored = map_boxes(boxes, meta, inverse=True).cpu().numpy()
            expected = np.asarray([] if source['pred'] is None else [source['pred']], dtype=float).reshape(-1, 6)
            # Numerical parity is an identity check, not an accuracy threshold.
            if (native.shape != expected.shape or not np.allclose(native, expected, atol=1e-4, rtol=1e-6)
                    or not np.allclose(restored, native[:, :5], atol=1e-4, rtol=1e-6)
                    or not np.array_equal(raw[:, 5], native[:, 5])):
                raise ValueError('Runtime B differs from frozen native VAL cache: '+source['image'])
            qualities = branch.online_qualities(arms, features[0], boxes, scores, meta)
            for name, q in qualities.items():
                if np.asarray(q).reshape(-1, 3).shape != (len(raw), 3) or not np.isfinite(q).all():
                    raise ValueError('Invalid online quality output: '+name)
            record = dict(source, qualities={k: v[0] if v else None for k, v in qualities.items()},
                runtime_b_original=native.tolist(), runtime_cache_max_abs_delta=float(np.abs(native-expected).max()) if len(raw) else 0.,
                image_size=list(meta['ori_shape'][:2][::-1]), online_inputs='P3_current_B_boxes_scores_metadata_only')
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            stream.flush()
            if (index+1) % 100 == 0 or index+1 == len(rows):
                print('fixed epoch8 VAL', index+1, '/', len(rows), flush=True)
            del batch, image, features, boxes, scores
    assert_detector_frozen(detector)
    if prior.state_digest(detector) != frozen or arm_before != {k: prior.state_digest(v) for k, v in arms.items()}:
        raise ValueError('Evaluation changed B/head parameters or buffers')
    # Rebind current bytes at completion; never silently compare changing data.
    after_rows, after_cache = val_cache(args.b_report, require_native=True)
    if after_rows != rows or after_cache != cache_identity or train.checked_sources() != (sources, protocol):
        raise ValueError('VAL inputs/cache/sources changed during scoring')
    report = dict(protocol=branch.VERSION, evidence_role=protocol['evidence_role'],
        status='FIXED_EPOCH8_SOURCE_VAL_SCORED_REVIEW_REQUIRED', fixed_epoch=8,
        branch_checkpoint_sha256=branch.sha(args.branch_checkpoint), sources=sources, protocol_contract=protocol,
        cache_identity=cache_identity, frozen_b_state=frozen, runtime=runtime,
        detector_optimizer_steps=0, branch_optimizer_steps=0, b_outputs_preserved=True,
        native_cache_numeric_atol=1e-4, native_cache_numeric_rtol=1e-6,
        rows_sha256=branch.sha(args.out_dir/'val_qualities.jsonl'),
        strata=branch.compare_component_rankings(records),
        max_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20,
        max_reserved_mib=torch.cuda.max_memory_reserved(args.gpu)/2**20,
        test_read=False, test_repeatedly_exposed=True,
        limitations=['Source VAL is exploratory and already used for detector selection/development.',
            'GT direction eligibility defines offline evaluation only; no online GT filter.',
            'Diagnostic top-k counts are matched per stratum/component, not deployment thresholds.',
            'Size/angle acceptance leaves every raw B center available; no depth guarantee.',
            'One seed and correlated video frames cannot establish stable/significant gain.',
            'Structure contrast includes response input AND auxiliary supervision; not isolated capacity/loss causality.',
            'Quality sigmoid is continuous error regression, not a calibrated probability.'])
    branch.write_new(args.out_dir/'val_compare.json', report)
    for name in ('domain:real', 'domain:sim'):
        s = report['strata'][name]
        print(name, 'outputs', s['output_frames'], '/', s['frames'],
              'center/output', s['center_hit_rate_on_outputs'],
              'full-frame center', s['all_frame_center_correct_coverage'], flush=True)
    print('Saved', args.out_dir/'val_compare.json', report['status'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--branch-checkpoint', type=Path)
    parser.add_argument('--b-checkpoint', type=Path, default=prior.CHECKPOINT)
    parser.add_argument('--b-report', type=Path, default=Path('work_dirs/port_shape_e_h_v1_val_compare.json'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    import os
    os.chdir(ROOT)
    if args.out_dir.exists():
        raise FileExistsError('Use a new --out-dir; existing evidence is preserved')
    sources, protocol = train.checked_sources()
    cfg = prior.check_cfg()
    rows, identity = val_cache(args.b_report, require_native=not args.check_only)
    if args.check_only:
        if args.branch_checkpoint:
            fixed_bundle(args.branch_checkpoint, sources, protocol)
        branch.write_new(args.out_dir/'val_contract_check.json', dict(status='VAL_CACHE_CONTRACT_PASS_CPU_ONLY',
            frames=len(rows), source_sha256=identity['report_sha256'], sources=sources,
            native=identity['native'], fitting=False, inference=False, test_read=False))
        print('Saved', args.out_dir/'val_contract_check.json', 'CPU only')
    else:
        if not args.branch_checkpoint:
            parser.error('--branch-checkpoint is required for fixed epoch8 scoring')
        evaluate(args, rows, identity, sources, protocol, cfg)


if __name__ == '__main__':
    main()
