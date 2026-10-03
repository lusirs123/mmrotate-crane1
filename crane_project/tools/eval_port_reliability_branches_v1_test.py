#!/usr/bin/env python3
"""Frozen B ep24 + original quality ep08 on the exposed 1440-frame TEST.

Default cache source reuses SHA/provenance-verified native B boxes and scores.
Explicit online source measures a fresh run of the same frozen B; it never
overwrites or merges historical predictions. Both keep every missing frame.
No fitting, probes, threshold search, frame selection or checkpoint selection.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crane_project.tools import eval_port_reliability_branches_v1_cached_val as cached
from crane_project.utils.port_structure_reliability_v1 import assert_detector_frozen, map_boxes

base, branch, train, ready, prior = cached.base, cached.branch, cached.train, cached.ready, cached.prior
VERSION = 'port_reliability_branches_v1_fixed_test'
PROTOCOL = ROOT/'crane_project/tools/port_reliability_branches_v1_test_protocol.json'
MANIFEST = ROOT/'crane_project/tools/port_reliability_branches_v1_test_sources.json'
PARENT_SHA = 'c77b4cf89fe4cdc50b39869166ae3939452706ae0ac51346ce3ec906291a6b10'
SWEEP = prior.CHECKPOINT.parent/'val_sweep_port_v1'
NATIVE_PKL = SWEEP/'final_test/epoch_24/preds/results.pkl'
NATIVE_REPORT = SWEEP/'final_test/epoch_24/final_test_metrics_v2.json'
SELECTION = SWEEP/'sweep_results.json'
BRANCH_CHECKPOINT = ROOT/'work_dirs/port_reliability_branches_v1_train/epoch_08.pth'


def checked_sources():
    training, training_protocol, repair = cached.checked_sources()
    contract = json.loads(PROTOCOL.read_text())
    manifest = json.loads(MANIFEST.read_text())
    current = {name: branch.sha(ROOT/name) for name in manifest['sources']}
    if (contract['protocol'] != VERSION or manifest['protocol'] != VERSION
            or current != manifest['sources'] or repair['manifest_sha256'] != manifest['parent_cached_val_manifest_sha256']
            or repair['manifest_sha256'] != PARENT_SHA
            or contract['coverages'] != training_protocol['coverages']
            or contract['methods'] != ['score']+training_protocol['arms']
            or any(contract[k] != training_protocol[k] for k in
                   ('center_px', 'size_max_relative', 'angle_deg', 'evaluation_angle_gt_aspect_min'))
            or contract['numeric_atol'] != cached.ATOL or contract['numeric_rtol'] != cached.RTOL
            or contract['detector_epoch'] != 24 or contract['branch_epoch'] != 8
            or contract['selection_on_test'] or contract['new_fitting'] or contract['probes']
            or not contract['test_repeatedly_exposed']):
        raise ValueError('Fixed TEST evaluation or unchanged parent source contract differs')
    return training, training_protocol, contract, dict(
        manifest_sha256=branch.sha(MANIFEST), sources=current, parent_cached_val=repair)


def write_json(path, value):
    """Serialize first, publish exclusively; no half COMPLETE on bad JSON."""
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='test-report-', suffix='.json', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded)
        os.link(temporary, str(path))
    finally:
        os.unlink(temporary)


def test_inputs(contract):
    """Fixed TEST bytes/roles only; no prediction metric or model selection."""
    data_manifest = ready.DATA/'manifest.json'
    data = json.loads(data_manifest.read_text())
    records = sorted((r for r in data['records'] if r['split'] == 'test'), key=lambda r: r['id'])
    expected = {r['id']: r for r in records}
    ann = ready.files(ready.DATA/'test/annfiles', '.txt')
    images = ready.files(ready.DATA/'test/images', '.jpg')
    count = sum(contract['test_counts'].values())
    if (len(expected) != count or len(records) != count
            or {p.stem for p in ann} != set(expected) or {p.stem for p in images} != set(expected)
            or Counter(r['sequence'] for r in records) != Counter(contract['test_counts'])
            or ready.set_sha(ann) != contract['test_annotations_sha256']):
        raise ValueError('Fixed TEST annotations/counts/names differ; no resplitting')
    image_identity = hashlib.sha256()
    sources = []
    for path in ann:
        name = path.stem
        row = expected[name]
        sequence, number = name.rsplit('_', 1)
        if (row['sequence'] != sequence or row['images']['path'] != 'test/images/'+name+'.jpg'
                or row['annfiles']['path'] != 'test/annfiles/'+name+'.txt'
                or branch.sha(path) != row['annfiles']['sha256']):
            raise ValueError('TEST annotation/frame provenance mismatch: '+name)
        image = ready.DATA/'test/images'/(name+'.jpg')
        digest = branch.sha(image)
        if digest != row['images']['sha256']:
            raise ValueError('TEST image bytes differ from fixed records: '+name)
        image_identity.update(image.name.encode()); image_identity.update(b'\0')
        image_identity.update(bytes.fromhex(digest))
        _, gt = ready.polygon_box(path)
        with Image.open(image) as im:
            size = list(im.size)
        sources.append(dict(image=name, sequence=sequence, frame_id=int(number),
            domain=sequence.split('_')[0], split='test', gt=gt.tolist(),
            image_sha256=digest, annotation_sha256=row['annfiles']['sha256'],
            image_size=size, angle_axis_well_defined=bool(gt[2]/gt[3] >= 1.2)))
    if image_identity.hexdigest() != contract['test_image_identity_sha256']:
        raise ValueError('Fixed TEST image set differs')
    return sources, dict(frames=len(sources), manifest_sha256=branch.sha(data_manifest),
        annotations_sha256=contract['test_annotations_sha256'],
        image_identity_sha256=image_identity.hexdigest(), records_sha256=branch.fingerprint(sources),
        sequence_counts=dict(Counter(r['sequence'] for r in sources)))


def checked_test_cfg():
    cfg = prior.check_cfg()
    spec = cfg.data.test
    if ((Path(spec.get('data_root', ''))/spec.ann_file).resolve() != (ready.DATA/'test/annfiles').resolve()
            or (Path(spec.get('data_root', ''))/spec.img_prefix).resolve() != (ready.DATA/'test/images').resolve()
            or spec.pipeline != cfg.data.val.pipeline):
        raise ValueError('Only fixed TEST paths and unchanged single-scale B preprocessing permitted')
    return cfg


def same_path(value, expected):
    return isinstance(value, str) and Path(value).resolve() == Path(expected).resolve()


def native_cache(args, sources, data_identity, require):
    """Require generation proof before deserializing native Nx6 predictions."""
    sidecar = Path(str(args.b_test_pkl)+'.provenance.json')
    assets = dict(checkpoint=args.b_checkpoint, selection=args.b_selection,
                  pkl=args.b_test_pkl, report=args.b_test_report, sidecar=sidecar)
    available = {k: Path(p).is_file() for k, p in assets.items()}
    complete = all(available[k] for k in ('checkpoint', 'selection', 'pkl')) and (
        available['report'] or available['sidecar'])
    if not complete:
        if require:
            raise FileNotFoundError('Native B TEST cache/proof unavailable: '+str(available)+
                '. Use --prediction-source online in a NEW output directory for the same frozen B; do not invent cache provenance.')
        return None, dict(status='NATIVE_CACHE_UNAVAILABLE_NOT_READY', availability=available,
                          paths={k: str(p) for k, p in assets.items()})
    if branch.sha(args.b_checkpoint) != ready.FROZEN_B['checkpoint_sha256']:
        raise ValueError('Native TEST generating checkpoint is not fixed B ep24')
    if branch.sha(args.b_selection) != ready.FROZEN_B['selection_sha256']:
        raise ValueError('Original source-VAL B selection identity differs')
    selection = json.loads(args.b_selection.read_text())
    if (selection.get('evidence_role') != 'source_val_checkpoint_selection'
            or selection.get('selected_checkpoint') != 'epoch_24'
            or selection.get('config_sha256') != ready.FROZEN_B['config_sha256']
            or not same_path(selection.get('selected_path'), args.b_checkpoint)):
        raise ValueError('Native TEST must retain original VAL-selected B ep24')
    digest = branch.sha(args.b_test_pkl)
    common = dict(config_sha256=ready.FROZEN_B['config_sha256'],
                  checkpoint_sha256=ready.FROZEN_B['checkpoint_sha256'], results_pkl_sha256=digest)
    proof_hashes = {}
    if available['report']:
        report = json.loads(args.b_test_report.read_text())
        if (report.get('protocol') != 'crane_ckpt_sweep_final_test_v2'
                or report.get('metric_protocol_version') != 2
                or report.get('evidence_role') != 'fixed_test_after_source_val_selection'
                or report.get('frame_count') != len(sources) or report.get('center_thresh_px') != 15.
                or report.get('gt_annotations_sha256') != data_identity['annotations_sha256']
                or not same_path(report.get('config'), ready.B_CONFIG)
                or not same_path(report.get('checkpoint'), args.b_checkpoint)
                or not same_path(report.get('results_pkl'), args.b_test_pkl)
                or not same_path(report.get('gt_dir'), ready.DATA/'test/annfiles')
                or any(report.get(k) != v for k, v in common.items())):
            raise ValueError('Native TEST report does not bind fixed B/checkpoint/PKL/annotations')
        proof_hashes['report'] = branch.sha(args.b_test_report)
    if available['sidecar']:
        proof = json.loads(sidecar.read_text())
        if (proof.get('protocol') != 'crane_prediction_provenance_v1' or proof.get('split') != 'fixed_test'
                or proof.get('annotations_sha256') != data_identity['annotations_sha256']
                or not same_path(proof.get('config'), ready.B_CONFIG)
                or not same_path(proof.get('checkpoint'), args.b_checkpoint)
                or not same_path(proof.get('results_pkl'), args.b_test_pkl)
                or any(proof.get(k) != v for k, v in common.items())):
            raise ValueError('Native TEST PKL generation sidecar differs')
        proof_hashes['sidecar'] = branch.sha(sidecar)
    with args.b_test_pkl.open('rb') as stream:
        predictions = pickle.load(stream)  # project-generated file after generation/hash validation
    if not isinstance(predictions, (list, tuple)) or len(predictions) != len(sources):
        raise ValueError('Native TEST PKL must contain all fixed frames')
    rows = []
    for source, prediction in zip(sources, predictions):
        if not isinstance(prediction, (list, tuple)) or len(prediction) != 1:
            raise ValueError('Native TEST PKL must be single-class grab')
        a = np.asarray(prediction[0])
        if a.ndim != 2 or a.shape[1] != 6:
            raise ValueError('Native TEST predictions must be Nx6; never DOTA score=1')
        a = cached.prediction_array(a)
        rows.append(dict(source, pred=a[0].tolist() if len(a) else None))
    return rows, dict(status='GENERATION_BOUND_NATIVE_TEST_CACHE_VERIFIED',
        availability=available, paths={k: str(p) for k, p in assets.items()},
        checkpoint_sha256=common['checkpoint_sha256'], selection_sha256=branch.sha(args.b_selection),
        pkl_sha256=digest, proof_sha256=proof_hashes, frame_count=len(rows),
        row_order='fixed sorted TEST annotation names / original dataset loader order')


def fixed_bundle(path, sources, protocol, contract):
    if branch.sha(path) != contract['branch_checkpoint_sha256']:
        raise ValueError('Requires original completed quality epoch_08 SHA, not diagnostic/smoke/new fit weights')
    return base.fixed_bundle(path, sources, protocol)


def test_dataset(cfg, rows):
    from mmrotate.datasets import build_dataset
    spec = deepcopy(cfg.data.test)
    spec.test_mode = True
    dataset = build_dataset(spec)
    if len(dataset) != len(rows) or [Path(r['filename']).stem for r in dataset.data_infos] != [r['image'] for r in rows]:
        raise ValueError('TEST dataset loader order/count differs from fixed frame identity')
    return dataset


def test_view(dataset, index, source, gpu):
    image, metas = cached.val_view(dataset, index, source, gpu)
    meta = metas[0]
    if list(meta['ori_shape'][:2][::-1]) != source['image_size']:
        raise ValueError('Original TEST image shape differs')
    return image, metas


def capture(detector, arms, image, metas, source, prediction_source):
    """Online scoring input deliberately excludes source GT and domain/sequence."""
    with torch.no_grad():
        features = detector.extract_feat(image)
        raw = prior.flatten_prediction(detector.simple_test_from_features(features, metas, rescale=False))
        native = prior.flatten_prediction(detector.simple_test_from_features(features, metas, rescale=True))
    if features[0].shape != (1, 256, 128, 128) or any(f.requires_grad or f.grad_fn is not None for f in features):
        raise ValueError('Frozen detached B P3 shape/graph differs')
    raw = cached.prediction_array(raw)
    native = cached.prediction_array(native)
    restored = map_boxes(image.new_tensor(raw[:, :5]), metas[0], inverse=True, size_mode='detector').cpu().numpy()
    wrapper = np.concatenate((restored, raw[:, 5:6]), axis=1)
    delta = cached.prediction_difference(wrapper, native)
    if not delta['all_six_close'] or not np.array_equal(raw[:, 5], native[:, 5]):
        raise ValueError('B wrapper/original coordinates or raw/native scores differ: '+source['image'])
    pred = source['pred'] if prediction_source == 'cache' else (native[0].tolist() if len(native) else None)
    boxes, scores, roundtrip = cached.cached_model_inputs(pred, image, metas[0])
    qualities = branch.online_qualities(arms, features[0], boxes, scores, metas[0])
    for name, q in qualities.items():
        q = np.asarray(q).reshape(-1, 3)
        if q.shape != (len(boxes), 3) or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
            raise ValueError('Invalid genuine TEST qualities: '+name)
    with torch.no_grad():
        after = prior.flatten_prediction(detector.simple_test_from_features(features, metas, rescale=False))
    if not np.array_equal(raw, after):
        raise ValueError('Side scoring changed original B raw output: '+source['image'])
    parity = cached.prediction_difference(native, source['pred']) if prediction_source == 'cache' else None
    # Labels/errors enter only this offline record, after scoring all arms.
    result = dict(source, pred=pred,
        qualities={name: value[0] if value else None for name, value in qualities.items()},
        genuine_online_qualities=qualities, model_boxes=boxes.cpu().tolist(),
        original_coordinate_roundtrip_max_abs=roundtrip, runtime_b_original=native.tolist(),
        b_raw_exact_before_after=True, runtime_vs_native_cache=parity,
        prediction_source='verified_native_B_TEST_cache' if prediction_source == 'cache' else 'fresh_frozen_B_TEST_runtime',
        online_inputs='detached_P3_B_boxes_scores_observed_transform_metadata_only')
    if pred is not None:
        result['errors'] = ready.geometry_errors(source['gt'], pred[:5])
    return result


def parity_summary(records):
    pairs = [r for r in records if r['runtime_vs_native_cache'] is not None]
    fields = ('cx_px', 'cy_px', 'width_px', 'height_px', 'angle_rad', 'score')
    return dict(compared_frames=len(pairs),
        all_six_close_frames=sum(r['runtime_vs_native_cache']['all_six_close'] for r in pairs),
        shape_mismatch_images=[r['image'] for r in pairs if not r['runtime_vs_native_cache']['same_shape']],
        numeric_mismatch_images=[r['image'] for r in pairs if not r['runtime_vs_native_cache']['all_six_close']],
        max_abs_delta_by_field={k: max((r['runtime_vs_native_cache']['abs_delta_by_field'][k]
            for r in pairs if r['runtime_vs_native_cache']['abs_delta_by_field'] is not None), default=0.) for k in fields},
        claim='full TEST frame comparisons retained; differing frames never skipped or substituted')


def run(args, rows, data_identity, cache_identity, training, training_protocol, contract, evaluation, cfg):
    payload = fixed_bundle(args.branch_checkpoint, training, training_protocol, contract)
    detector, runtime = train.build_runtime(cfg, args.b_checkpoint, args.gpu)
    frozen = prior.state_digest(detector)
    if frozen != payload['frozen_b_state']:
        raise ValueError('B state differs from original completed TRAIN')
    arms = branch.make_arms('cuda:'+str(args.gpu))
    if branch.architecture(arms) != payload['contract']['architecture']:
        raise ValueError('Original quality architecture differs')
    branch.load_bundle(payload, arms)
    del payload
    head_before = {name: prior.state_digest(arm) for name, arm in arms.items()}
    bundle_sha = branch.sha(args.branch_checkpoint)
    dataset = test_dataset(cfg, rows)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir/'input_check.json', dict(status='FIXED_TEST_INPUTS_CHECKED_BEFORE_SCORING',
        prediction_source=args.prediction_source, data_identity=data_identity, native_cache=cache_identity,
        training_sources=training, test_contract=contract, evaluation_sources=evaluation,
        branch_checkpoint_sha256=bundle_sha, detector_updates=0, quality_updates=0))
    torch.cuda.reset_peak_memory_stats(args.gpu)
    started = time.monotonic()
    records = []
    with (args.out_dir/'test_qualities.jsonl').open('x') as stream:
        for index, source in enumerate(rows):
            image, metas = test_view(dataset, index, source, args.gpu)
            record = capture(detector, arms, image, metas, source, args.prediction_source)
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
            if (index+1) % 100 == 0 or index+1 == len(rows):
                print('fixed B24 / quality8 TEST', args.prediction_source, index+1, '/', len(rows), flush=True)
            del image, metas
    seconds = time.monotonic()-started
    assert_detector_frozen(detector)
    if prior.state_digest(detector) != frozen or head_before != {name: prior.state_digest(arm) for name, arm in arms.items()}:
        raise ValueError('TEST scoring changed B/quality parameters or buffers')
    if any(p.grad is not None or arm.training for arm in arms.values() for p in arm.parameters()):
        raise ValueError('Quality heads retained gradients or training mode')
    after_rows, after_data = test_inputs(contract)
    if args.prediction_source == 'cache':
        after_rows, after_cache = native_cache(args, after_rows, after_data, require=True)
        if after_cache != cache_identity:
            raise ValueError('Native TEST cache/proof changed during scoring')
    if (after_rows != rows or after_data != data_identity or branch.sha(args.branch_checkpoint) != bundle_sha
            or branch.sha(args.b_checkpoint) != ready.FROZEN_B['checkpoint_sha256']
            or checked_sources() != (training, training_protocol, contract, evaluation)):
        raise ValueError('TEST data/weights/sources changed during scoring')
    parity = parity_summary(records)
    metrics = branch.compare_component_rankings(records, coverages=contract['coverages'])
    for group, values in metrics.items():
        if group == 'all':
            members = records
        else:
            field, value = group.split(':', 1)
            members = [r for r in records if r[field] == value]
        values['center_correct_output_frames'] = sum(
            r['pred'] is not None and r['errors']['center_px'] < 15 for r in members)
        values['missing_output_frames'] = values['frames']-values['output_frames']
    report = dict(protocol=VERSION, status='FIXED_B24_QUALITY8_TEST_SCORED_REVIEW_REQUIRED',
        evidence_role=contract['evidence_role'], prediction_source=args.prediction_source,
        detector_epoch=24, branch_epoch=8, branch_checkpoint_sha256=bundle_sha,
        training_protocol_unchanged=training_protocol, training_sources=training,
        test_contract=contract, evaluation_sources=evaluation, data_identity=data_identity,
        native_cache=cache_identity, frozen_b_state=frozen, original_head_states=head_before, runtime=runtime,
        detector_optimizer_steps=0, branch_optimizer_steps=0, original_B_side_output_exact_all_frames=True,
        primary_quality_arm='roi', strata=metrics, runtime_cache_parity=parity,
        cached_B_predictions_scores_exactly_preserved=(True if args.prediction_source == 'cache' else None),
        fresh_runtime_B_reproduces_all_cached_predictions=(parity['all_six_close_frames']==len(rows)
            if args.prediction_source == 'cache' else None),
        rows_sha256=branch.sha(args.out_dir/'test_qualities.jsonl'),
        wall_seconds_including_full_parity_checks=seconds,
        max_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20,
        max_reserved_mib=torch.cuda.max_memory_reserved(args.gpu)/2**20,
        test_read=True, test_repeatedly_exposed=True, test_used_for_selection=False,
        deployment_threshold_selected=False,
        limitations=['Repeatedly exposed TEST reports the frozen exploratory scheme, not untouched confirmation.',
            'All four methods and all six coverage settings are retained; no best TEST arm/epoch/threshold selection.',
            'GT defines offline errors/angle eligibility only and is absent from the online quality-head API.',
            'Matched-count component curves are descriptive ranking evaluation, not deployment score cutoffs.',
            'Original B boxes/scores/centers remain available; component acceptance does not define a complete OBB.',
            'Cache scores condition on saved predictions; full runtime differences are retained separately.',
            'Fresh online results use the same fixed weights but may numerically differ from historical predictions.',
            'Continuous quality regression is not a calibrated correctness probability.',
            'One seed and correlated video frames do not establish significant/stable gains.',
            'Timing includes integrity/parity work and all heads; no isolated ROI latency or depth/control claim.'])
    write_json(args.out_dir/'test_compare.json', report)
    write_json(args.out_dir/'completion.json', dict(status='FIXED_TEST_SCORING_COMPLETE_REVIEW_REQUIRED',
        report_sha256=branch.sha(args.out_dir/'test_compare.json'), rows_sha256=report['rows_sha256'],
        frames=len(records), prediction_source=args.prediction_source, detector_updates=0, quality_updates=0,
        test_used_for_selection=False))
    for name in ('domain:real', 'domain:sim'):
        s = metrics[name]
        print(name, 'outputs', s['output_frames'], '/', s['frames'], 'center/output',
              s['center_hit_rate_on_outputs'], 'full-frame center', s['all_frame_center_correct_coverage'], flush=True)
    print('Runtime-cache mismatches', len(parity['numeric_mismatch_images']), '/', parity['compared_frames'], flush=True)
    print('Saved', args.out_dir/'test_compare.json', report['status'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'run'), default='check')
    parser.add_argument('--prediction-source', choices=('cache', 'online'), default='cache')
    parser.add_argument('--branch-checkpoint', type=Path, default=BRANCH_CHECKPOINT)
    parser.add_argument('--b-checkpoint', type=Path, default=prior.CHECKPOINT)
    parser.add_argument('--b-test-pkl', type=Path, default=NATIVE_PKL)
    parser.add_argument('--b-test-report', type=Path, default=NATIVE_REPORT)
    parser.add_argument('--b-selection', type=Path, default=SELECTION)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.out_dir.exists():
        raise FileExistsError('Preserve previous evidence; use a NEW --out-dir')
    training, training_protocol, contract, evaluation = checked_sources()
    cfg = checked_test_cfg()
    rows, data_identity = test_inputs(contract)
    cache_rows = None
    if args.prediction_source == 'cache':
        cache_rows, cache_identity = native_cache(args, rows, data_identity, require=args.mode == 'run')
        if cache_rows is not None:
            rows = cache_rows
    else:
        cache_identity = dict(status='EXPLICIT_FRESH_ONLINE_SOURCE_NO_HISTORICAL_CACHE_SUBSTITUTION')
    if args.mode == 'check':
        available = args.branch_checkpoint.is_file() and args.b_checkpoint.is_file()
        if args.branch_checkpoint.is_file():
            fixed_bundle(args.branch_checkpoint, training, training_protocol, contract)
        if args.b_checkpoint.is_file():
            if branch.sha(args.b_checkpoint) != ready.FROZEN_B['checkpoint_sha256']:
                raise ValueError('Fixed B checkpoint SHA differs')
        status = 'FIXED_TEST_CHECK_READY_NO_INFERENCE' if available and (
            args.prediction_source == 'online' or cache_rows is not None) else 'STATIC_TEST_CHECK_COMPLETE_NATIVE_ASSETS_REQUIRED'
        write_json(args.out_dir/'input_check.json', dict(status=status, data_identity=data_identity,
            native_cache=cache_identity, branch_and_B_weights_available=available,
            prediction_source=args.prediction_source, test_contract=contract, training_sources=training,
            evaluation_sources=evaluation, fitting=False, inference=False, prediction_metrics_computed=False))
        print('Saved', args.out_dir/'input_check.json', status, 'CPU only; no fitting/inference', flush=True)
        if status != 'FIXED_TEST_CHECK_READY_NO_INFERENCE':
            print('Check native cache/proof and original weights; explicit online source is permitted in a new directory.', flush=True)
    else:
        run(args, rows, data_identity, cache_identity, training, training_protocol, contract, evaluation, cfg)


if __name__ == '__main__':
    main()
