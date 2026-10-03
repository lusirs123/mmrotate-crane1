#!/usr/bin/env python3
"""Diagnose one VAL parity failure and score the locked native B predictions.

Additive evaluation repair: original TRAIN sources/checkpoints stay unchanged.
Scores are conditional on the verified saved B outputs, not a claim that a new
online detector run reproduced them. No tolerance tuning, frame skipping or TEST.
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
from crane_project.tools import eval_port_reliability_branches_v1 as base
from crane_project.utils.port_structure_reliability_v1 import assert_detector_frozen, checked_meta, map_boxes

branch, train, ready, prior = base.branch, base.train, base.ready, base.prior
VERSION = 'port_reliability_branches_v1_cached_val_repair'
MANIFEST = ROOT/'crane_project/tools/port_reliability_branches_v1_cached_val_sources.json'
TRAIN_MANIFEST_SHA = '422161facfa1251c54ff5f95b7b31ecb9ae7e830134a3b03f569987507296a03'
DIAGNOSE_IMAGE = 'sim_seq10_00220'
ATOL, RTOL = 1e-4, 1e-6  # unchanged identity check; never an accuracy threshold


def checked_sources():
    training_sources, protocol = train.checked_sources()
    manifest = json.loads(MANIFEST.read_text())
    current = {p: branch.sha(ROOT/p) for p in manifest['sources']}
    if (manifest['protocol'] != VERSION or manifest['sources'] != current
            or training_sources['manifest_sha256'] != TRAIN_MANIFEST_SHA
            or manifest['parent_training_manifest_sha256'] != TRAIN_MANIFEST_SHA
            or manifest['diagnose_image'] != DIAGNOSE_IMAGE
            or manifest['numeric_atol'] != ATOL or manifest['numeric_rtol'] != RTOL):
        raise ValueError('Reviewed evaluation repair or unchanged TRAIN source contract differs')
    return training_sources, protocol, dict(manifest_sha256=branch.sha(MANIFEST), sources=current)


def prediction_array(value):
    a = np.asarray([] if value is None else value, dtype=float).reshape(-1, 6)
    if (len(a) > 1 or not np.isfinite(a).all()
            or (len(a) and (min(a[0, 2:4]) <= 0 or not .05 < a[0, 5] <= 1.))):
        raise ValueError('Expected a finite genuine B top1 or no output')
    return a


def prediction_difference(actual, expected):
    a, e = prediction_array(actual), prediction_array(expected)
    same_shape = a.shape == e.shape
    fields = ('cx_px', 'cy_px', 'width_px', 'height_px', 'angle_rad', 'score')
    delta = np.abs(a-e).max(axis=0) if same_shape and len(a) else np.zeros(6) if same_shape else None
    geometric = ready.geometry_errors(e[0, :5], a[0, :5]) if same_shape and len(a) else None
    return dict(actual=a.tolist(), expected=e.tolist(), actual_outputs=len(a), expected_outputs=len(e),
        same_shape=same_shape, all_six_close=bool(same_shape and np.allclose(a, e, atol=ATOL, rtol=RTOL)),
        coordinates_close=bool(same_shape and np.allclose(a[:, :5], e[:, :5], atol=ATOL, rtol=RTOL)),
        score_close=bool(same_shape and np.allclose(a[:, 5], e[:, 5], atol=ATOL, rtol=RTOL)),
        score_exact=bool(same_shape and np.array_equal(a[:, 5], e[:, 5])),
        abs_delta_by_field=None if delta is None else {k: float(v) for k, v in zip(fields, delta)},
        geometry_difference_relative_to_cached_prediction=geometric)


def cached_model_inputs(pred_original, image, meta):
    """Only the genuine cached prediction and observed transform, never GT."""
    expected = prediction_array(pred_original)
    original = image.new_tensor(expected[:, :5])
    boxes = map_boxes(original, meta, size_mode='detector')
    scores = image.new_tensor(expected[:, 5])
    restored = map_boxes(boxes, meta, inverse=True, size_mode='detector')
    if not torch.allclose(original, restored, atol=ATOL, rtol=RTOL):
        raise ValueError('Cached B model/original coordinate roundtrip failed')
    return boxes, scores, float((original-restored).abs().max()) if len(original) else 0.


def val_dataset(cfg, rows):
    from mmrotate.datasets import build_dataset
    spec = deepcopy(cfg.data.val)
    if ((Path(spec.get('data_root', ''))/spec.ann_file).resolve() != (ready.DATA/'val/annfiles').resolve()
            or (Path(spec.get('data_root', ''))/spec.img_prefix).resolve() != (ready.DATA/'val/images').resolve()):
        raise ValueError('Only allowlisted VAL paths permitted')
    spec.test_mode = True
    dataset = build_dataset(spec)
    if len(dataset) != 887 or len(rows) != 887:
        raise ValueError('Expected all 887 VAL frames')
    if [Path(r['filename']).stem for r in dataset.data_infos] != [r['image'] for r in rows]:
        raise ValueError('VAL loader order differs from native cache')
    return dataset


def val_view(dataset, index, source, gpu):
    from mmcv.parallel import collate, scatter
    batch = scatter(collate([dataset[index]], samples_per_gpu=1), [gpu])[0]
    if len(batch['img']) != 1 or len(batch['img_metas']) != 1:
        raise ValueError('Only unchanged single-scale B VAL preprocessing permitted')
    image, metas = batch['img'][0], batch['img_metas'][0]
    meta = metas[0]
    checked_meta(meta)
    if meta.get('flip', False) or Path(meta['filename']).stem != source['image']:
        raise ValueError('Unexpected VAL transform/image identity')
    return image, metas


def diagnostic(detector, image, metas, source):
    """Separate the three original predicates and repeated head reproducibility."""
    with torch.no_grad():
        features = detector.extract_feat(image)
        raw = prior.flatten_prediction(detector.simple_test_from_features(features, metas, rescale=False))
        native = prior.flatten_prediction(detector.simple_test_from_features(features, metas, rescale=True))
        raw_repeat = prior.flatten_prediction(detector.simple_test_from_features(features, metas, rescale=False))
        native_repeat = prior.flatten_prediction(detector.simple_test_from_features(features, metas, rescale=True))
    if any(f.requires_grad or f.grad_fn is not None for f in features):
        raise ValueError('Frozen B retained a computation graph')
    restored = map_boxes(image.new_tensor(raw[:, :5]), metas[0], inverse=True).cpu().numpy()
    restored_full = np.concatenate((restored, raw[:, 5:6]), axis=1)
    expected = prediction_array(source['pred'])
    cache_delta = prediction_difference(native, expected)
    wrapper_delta = prediction_difference(restored_full, native)
    score_exact = raw.shape == native.shape and np.array_equal(raw[:, 5], native[:, 5])
    failed = []
    if not cache_delta['all_six_close']:
        failed.append('runtime_native_vs_locked_cache')
    if not wrapper_delta['coordinates_close']:
        failed.append('wrapper_original_coordinates_vs_runtime_native')
    if not score_exact:
        failed.append('raw_vs_native_score_exactness')
    return dict(image=source['image'], image_sha256=source['image_sha256'],
        status='ORIGINAL_THREE_PARITY_CHECKS_PASS' if not failed else 'PARITY_MISMATCH_DIAGNOSED_REVIEW_REQUIRED',
        failed_original_predicates=failed, native_vs_cache=cache_delta, wrapper_vs_native=wrapper_delta,
        raw_model=raw.tolist(), raw_vs_native_score_exact=bool(score_exact),
        repeat_raw_exact=bool(np.array_equal(raw, raw_repeat)),
        repeat_native_exact=bool(np.array_equal(native, native_repeat)),
        repeated_native_vs_first=prediction_difference(native_repeat, native),
        image_tensor_sha256=prior.tensor_sha(image), p3_sha256=prior.tensor_sha(features[0]),
        scale_factor=np.asarray(metas[0]['scale_factor']).tolist(),
        img_shape=list(metas[0]['img_shape']), ori_shape=list(metas[0]['ori_shape']),
        pad_shape=list(metas[0]['pad_shape']), numeric_atol=ATOL, numeric_rtol=RTOL,
        note='Differences are relative to the locked prediction, not annotation accuracy. No tolerance changed.')


def run(args, rows, cache_identity, training_sources, protocol, repair_sources, cfg):
    payload = base.fixed_bundle(args.branch_checkpoint, training_sources, protocol)
    detector, runtime = train.build_runtime(cfg, args.b_checkpoint, args.gpu)
    frozen = prior.state_digest(detector)
    if frozen != payload['frozen_b_state']:
        raise ValueError('B state differs from completed formal TRAIN')
    dataset = val_dataset(cfg, rows)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats(args.gpu)
    index = next(i for i, r in enumerate(rows) if r['image'] == DIAGNOSE_IMAGE)
    image, metas = val_view(dataset, index, rows[index], args.gpu)
    diagnosis = diagnostic(detector, image, metas, rows[index])
    diagnostic_record = dict(protocol=VERSION, mode='single_frame_parity_diagnosis', runtime=runtime,
        training_sources=training_sources, repair_sources=repair_sources,
        branch_checkpoint_sha256=branch.sha(args.branch_checkpoint), cache_identity=cache_identity,
        frozen_b_state=frozen, diagnosis=diagnosis, detector_optimizer_steps=0,
        branch_optimizer_steps=0, test_read=False, test_repeatedly_exposed=True)
    assert_detector_frozen(detector)
    if prior.state_digest(detector) != frozen:
        raise ValueError('Single-frame diagnostic changed B parameters/buffers')
    branch.write_new(args.out_dir/'parity_diagnosis.json', diagnostic_record)
    print(DIAGNOSE_IMAGE, diagnosis['status'], diagnosis['failed_original_predicates'], flush=True)
    del image, metas
    if args.mode == 'diagnose':
        after_rows, after_cache = base.val_cache(args.b_report, require_native=True)
        if (after_rows != rows or after_cache != cache_identity
                or checked_sources() != (training_sources, protocol, repair_sources)):
            raise ValueError('Inputs/cache/sources changed during diagnostic')
        print('Saved', args.out_dir/'parity_diagnosis.json', 'one VAL frame; no training', flush=True)
        return

    arms = branch.make_arms('cuda:'+str(args.gpu))
    if branch.architecture(arms) != payload['contract']['architecture']:
        raise ValueError('Quality architecture differs from completed training')
    branch.load_bundle(payload, arms)
    del payload
    arm_before = {k: prior.state_digest(v) for k, v in arms.items()}
    records = []
    with (args.out_dir/'val_qualities.jsonl').open('x') as stream:
        for index, source in enumerate(rows):
            image, metas = val_view(dataset, index, source, args.gpu)
            with torch.no_grad():
                features = detector.extract_feat(image)
            if any(f.requires_grad or f.grad_fn is not None for f in features):
                raise ValueError('Frozen P3 retained detector graph')
            if features[0].shape != (1, 256, 128, 128):
                raise ValueError('Unchanged B P3 shape required')
            boxes, scores, roundtrip = cached_model_inputs(source['pred'], image, metas[0])
            qualities = branch.online_qualities(arms, features[0], boxes, scores, metas[0])
            for q in qualities.values():
                if np.asarray(q).reshape(-1, 3).shape != (len(boxes), 3) or not np.isfinite(q).all():
                    raise ValueError('Invalid genuine cached-output quality array')
            record = dict(source, qualities={k: v[0] if v else None for k, v in qualities.items()},
                genuine_online_qualities=qualities, cached_b_model=boxes.cpu().tolist(),
                cached_box_roundtrip_max_abs=roundtrip, image_size=list(metas[0]['ori_shape'][:2][::-1]),
                prediction_source='sha_verified_native_B_VAL_cache',
                online_inputs='detached_P3_verified_cached_B_boxes_scores_metadata_only')
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            stream.flush()
            if (index+1) % 100 == 0 or index+1 == len(rows):
                print('fixed cached B epoch8 VAL', index+1, '/', len(rows), flush=True)
            del image, metas, features, boxes, scores
    assert_detector_frozen(detector)
    if prior.state_digest(detector) != frozen or arm_before != {k: prior.state_digest(v) for k, v in arms.items()}:
        raise ValueError('Scoring changed B/quality parameters or buffers')
    after_rows, after_cache = base.val_cache(args.b_report, require_native=True)
    if after_rows != rows or after_cache != cache_identity or checked_sources() != (training_sources, protocol, repair_sources):
        raise ValueError('Inputs/cache/sources changed during cached scoring')
    report = dict(protocol=VERSION, status='FIXED_CACHED_B_EPOCH8_VAL_SCORED_REVIEW_REQUIRED',
        evidence_role='component_quality_conditional_on_locked_native_B_outputs_source_VAL_only',
        fixed_epoch=8, training_protocol_unchanged=protocol, training_sources=training_sources,
        repair_sources=repair_sources, branch_checkpoint_sha256=branch.sha(args.branch_checkpoint),
        cache_identity=cache_identity, frozen_b_state=frozen, runtime=runtime,
        detector_optimizer_steps=0, branch_optimizer_steps=0, cached_b_outputs_scores_exactly_preserved=True,
        fresh_runtime_B_reproduces_all_887_cached_predictions=None,
        single_frame_runtime_diagnosis=diagnosis, parity_diagnosis_sha256=branch.sha(args.out_dir/'parity_diagnosis.json'),
        rows_sha256=branch.sha(args.out_dir/'val_qualities.jsonl'), strata=branch.compare_component_rankings(records),
        max_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20,
        max_reserved_mib=torch.cuda.max_memory_reserved(args.gpu)/2**20,
        test_read=False, test_repeatedly_exposed=True,
        limitations=['Scores condition on the SHA-verified saved detector boxes/scores; fresh online B outputs may differ.',
            'Only the reported failing frame is decoded for runtime parity diagnosis; no claim of all-frame runtime equivalence.',
            'Training weights/budget/direction qualification and all metric thresholds remain unchanged.',
            'Source VAL is exposed/development data; one seed and correlated frames do not establish stable or significant gains.',
            'No GT enters online heads; no probes, optimization, frame exclusion, or TEST.',
            'Matched-count component acceptance is diagnostic; size/angle rejection does not remove centers or define complete OBBs.',
            'No calibrated probability, depth accuracy or deployment performance claim.'])
    branch.write_new(args.out_dir/'val_compare.json', report)
    for name in ('domain:real', 'domain:sim'):
        s = report['strata'][name]
        print(name, 'outputs', s['output_frames'], '/', s['frames'], 'center/output',
            s['center_hit_rate_on_outputs'], 'full-frame center', s['all_frame_center_correct_coverage'], flush=True)
    print('Saved', args.out_dir/'val_compare.json', report['status'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('diagnose', 'score-cache'), default='diagnose')
    parser.add_argument('--branch-checkpoint', type=Path, required=True)
    parser.add_argument('--b-checkpoint', type=Path, default=prior.CHECKPOINT)
    parser.add_argument('--b-report', type=Path, default=Path('work_dirs/port_shape_e_h_v1_val_compare.json'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    import os
    os.chdir(ROOT)
    if args.out_dir.exists():
        raise FileExistsError('Preserve previous evidence; use a NEW --out-dir')
    training_sources, protocol, repair_sources = checked_sources()
    cfg = prior.check_cfg()
    rows, identity = base.val_cache(args.b_report, require_native=True)
    run(args, rows, identity, training_sources, protocol, repair_sources, cfg)


if __name__ == '__main__':
    main()
