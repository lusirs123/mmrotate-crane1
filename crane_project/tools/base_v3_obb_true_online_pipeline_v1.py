#!/usr/bin/env python3
"""True online image-to-OBB observation integration for the focused paper.

Pipeline:
  RGB image -> frozen native-S14 DINO proposal
            -> Base V3 epoch-9 (internal frozen SymEOOD K1 + causal refiner)
            -> frozen score-only and V5.1 component observation policies.

This entrypoint performs model inference.  It never reads annotations or GT,
and it does not turn the image-plane OBB into depth, physical sway, or control
state.  Sequence/frame identity is used only for chronological reset.
"""

import argparse
import collections
import copy
import json
import os
import re
from pathlib import Path

import numpy as np

from crane_project.tools.base_v3_obb_paper_runtime_v1 import (
    METHODS, PaperObservationManagerV1)
from crane_project.tools.base_v3_obb_reliability_baseline import (
    _identity, _write_exact)


PROTOCOL = 'base_v3_obb_true_online_pipeline_v1'
CONTRACT_PROTOCOL = 'base_v3_obb_true_online_pipeline_contract_v1'
_FRAME_RE = re.compile(
    r'^(?P<domain>real|sim)_(?P<sequence>seq\d+)_(?P<frame>\d+)$')
_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_frame_identity(path):
    stem = Path(path).stem
    match = _FRAME_RE.match(stem)
    if match is None:
        raise ValueError(
            'Image name must be real|sim_seqNN_frame: ' + stem)
    return dict(
        frame_key=stem, domain=match.group('domain'),
        sequence=match.group('sequence'), frame=int(match.group('frame')))


def discover_images(image_dir, requested_sequences=None, max_frames=None):
    root = Path(image_dir).resolve()
    if not root.is_dir():
        raise RuntimeError('Image directory does not exist: ' + os.fspath(root))
    requested = set(requested_sequences or [])
    rows = []
    seen = set()
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        identity = parse_frame_identity(path)
        if requested and identity['sequence'] not in requested:
            continue
        key = (identity['domain'], identity['sequence'], identity['frame'])
        if key in seen:
            raise RuntimeError('Duplicate frame identity: {!r}'.format(key))
        seen.add(key)
        rows.append((identity, os.fspath(path.resolve())))
    rows.sort(key=lambda item: (
        item[0]['domain'], item[0]['sequence'], item[0]['frame'],
        item[0]['frame_key']))
    if max_frames is not None and int(max_frames) > 0:
        rows = rows[:int(max_frames)]
    if not rows:
        raise RuntimeError('No named sequence images were discovered')
    return rows


class OnlineDinoHistoryBuffer:
    """Keep exactly K strictly previous raw images and DINO proposals."""

    def __init__(self, horizon=4):
        if int(horizon) <= 0:
            raise ValueError('History horizon must be positive')
        self.horizon = int(horizon)
        self.reset()

    def reset(self):
        self._identity = None
        self._last_frame = None
        self._records = collections.deque(maxlen=self.horizon)

    def _boundary(self, identity):
        current = (identity['domain'], identity['sequence'])
        return (current != self._identity or self._last_frame is None or
                int(identity['frame']) != int(self._last_frame) + 1)

    def prepare(self, identity, current_image):
        if self._boundary(identity):
            self.reset()
        images, proposals, valid, keys = [], [], [], []
        records = list(reversed(self._records))
        for age in range(1, self.horizon + 1):
            record = records[age - 1] if age <= len(records) else None
            usable = bool(
                record is not None
                and int(record['frame']) == int(identity['frame']) - age
                and record['dino_box'] is not None)
            if usable:
                images.append(record['image'].copy())
                proposals.append(
                    np.asarray(record['dino_box'], dtype=np.float32)
                    .reshape(1, 5).copy())
                keys.append(record['frame_key'])
            else:
                images.append(np.zeros_like(current_image))
                proposals.append(np.asarray(
                    [[0.0, 0.0, 1.0, 1.0, 0.0]], dtype=np.float32))
                keys.append(None)
            valid.append(usable)
        return dict(
            causal_history_images_raw=images,
            causal_history_proposals_raw=proposals,
            causal_history_valid_mask=np.asarray(valid, dtype=np.bool_),
            causal_history_ages=np.arange(
                1, self.horizon + 1, dtype=np.int64),
            causal_history_frame_keys=keys)

    def commit(self, identity, image, dino_box):
        current = (identity['domain'], identity['sequence'])
        self._records.append(dict(
            frame_key=identity['frame_key'], frame=int(identity['frame']),
            image=image.copy(),
            dino_box=(None if dino_box is None else
                      [float(value) for value in dino_box[:5]])))
        self._identity = current
        self._last_frame = int(identity['frame'])


def top1_detection(result):
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise RuntimeError('Expected one class in detector result')
    detections = np.asarray(result[0], dtype=np.float64)
    if detections.size == 0:
        return None
    detections = detections.reshape((-1, 6))
    if not np.isfinite(detections).all():
        raise RuntimeError('Detector produced non-finite values')
    best = detections[int(np.argmax(detections[:, 5]))]
    if np.any(best[2:4] <= 0.0) or not 0.0 <= float(best[5]) <= 1.0:
        raise RuntimeError('Detector produced an invalid Top-1 OBB')
    return [float(value) for value in best]


def assemble_detector_row(identity, image_shape, dino_detection, base_audit):
    if base_audit.get('coordinate_space') != 'original_image':
        raise RuntimeError('Base V3 audit must use original-image coordinates')
    dino_box = (None if dino_detection is None else dino_detection[:5])
    audited_dino = base_audit.get('dino_box')
    if dino_box is None and audited_dino is not None:
        raise RuntimeError('Base V3 received an unexpected DINO proposal')
    if dino_box is not None:
        if audited_dino is None:
            raise RuntimeError('Base V3 lost the online DINO proposal')
        if float(np.max(np.abs(
                np.asarray(dino_box)-np.asarray(audited_dino)))) > 1e-3:
            raise RuntimeError('DINO proposal coordinate mapping mismatch')
    source = base_audit.get('anchor_source')
    if source == 'k1':
        anchor_score = base_audit.get('k1_score')
    elif source == 'dino_fallback':
        anchor_score = (None if dino_detection is None else
                        float(dino_detection[5]))
    elif source == 'none':
        anchor_score = None
    else:
        raise RuntimeError('Unknown Base V3 anchor source: {!r}'.format(source))
    return dict(
        frame_key=identity['frame_key'], domain=identity['domain'],
        sequence=identity['sequence'], frame=int(identity['frame']),
        timestamp_seconds=None,
        image_size=[int(image_shape[1]), int(image_shape[0])],
        anchor_source=source, anchor_score=anchor_score,
        base_v3_box=copy.deepcopy(base_audit.get('base_v3_box')),
        k1_box=copy.deepcopy(base_audit.get('k1_box')),
        dino_box=copy.deepcopy(dino_box))


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected true-online pipeline contract')
    required = {
        'execution_mode': 'true_online_model_inference',
        'cached_dino_predictions_used': False,
        'cached_base_v3_predictions_used': False,
        'annotations_or_gt_read': False,
        'future_frames_used': False,
        'domain_identity_used_in_model_decisions': False,
        'history_horizon': 4,
        'base_v3_epoch': 9,
        'v51_parameters_frozen': True,
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise ValueError(
                'Online pipeline contract mismatch for {}: {!r}'.format(
                    key, contract.get(key)))
    if contract.get('pipeline') != [
            'rgb_image', 'frozen_dino_native_s14',
            'symeood_k1_epoch24', 'base_v3_epoch9_causal_refiner',
            'v51_component_observation']:
        raise ValueError('Online pipeline stage order is not frozen')
    return contract


def verify_artifacts(contract, paths):
    identities = {}
    for role, path in paths.items():
        identities[role] = _identity(path)
        expected = contract['artifacts'][role]['sha256']
        if identities[role]['sha256'] != expected:
            raise RuntimeError(
                '{} identity mismatch: expected {}, got {}'.format(
                    role, expected, identities[role]['sha256']))
    return identities


def build_constructor_loaded_detector(config_path, device, overrides=None):
    """Build wrapper detectors without MMDetection's backbone assumption."""
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector

    config = Config.fromfile(config_path)
    imports = config.get('custom_imports')
    if imports:
        import_modules_from_strings(**imports)
    model_config = copy.deepcopy(config.model)
    for key, value in dict(overrides or {}).items():
        model_config[key] = value
    model = build_detector(
        model_config, train_cfg=None, test_cfg=config.get('test_cfg'))
    model.cfg = config
    model.to(device)
    model.eval()
    return model, config


class BaseV3OnlineAdapter:
    """Assemble the exact fixed Base V3 transforms from in-memory DINO/history."""

    def __init__(self, model, config):
        import torch
        from mmcv.parallel import collate, scatter
        from mmdet.datasets.pipelines import Compose

        self._torch = torch
        self._collate = collate
        self._scatter = scatter
        self.model = model
        self.pipeline = Compose(config.online_test_pipeline)
        self.device = next(model.parameters()).device
        if self.device.type != 'cuda':
            raise RuntimeError('Formal Base V3 online inference requires CUDA')

    def infer(self, image_path, dino_box, history):
        proposal = (np.zeros((0, 5), dtype=np.float32)
                    if dino_box is None else
                    np.asarray(dino_box, dtype=np.float32).reshape(1, 5))
        sample = dict(
            img_info=dict(filename=os.path.abspath(image_path)),
            img_prefix=None, bbox_fields=['dino_proposals'],
            dino_proposals=proposal)
        sample.update(history)
        data = self.pipeline(sample)
        data = self._collate([data], samples_per_gpu=1)
        data = self._scatter(data, [self.device])[0]
        with self._torch.no_grad():
            results = self.model(
                return_loss=False, rescale=True, **data)
        if not isinstance(results, list) or len(results) != 1:
            raise RuntimeError('Base V3 must return one result per frame')
        records = self.model.last_inference_records()
        if len(records) != 1:
            raise RuntimeError('Base V3 runtime evidence is unavailable')
        result_top = top1_detection(results[0])
        audited = records[0]
        if ((result_top is None) != (audited.get('base_v3_box') is None)):
            raise RuntimeError('Base V3 output/audit availability mismatch')
        if result_top is not None and float(np.max(np.abs(
                np.asarray(result_top[:5]) -
                np.asarray(audited['base_v3_box'])))) > 1e-4:
            raise RuntimeError('Base V3 output/audit geometry mismatch')
        return results[0], audited


def _numeric_difference(first, second):
    if first is None or second is None:
        return 0.0 if first is None and second is None else None
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape:
        return None
    return float(np.max(np.abs(a-b))) if a.size else 0.0


def compare_paper_report(records, report, tolerance=1e-3):
    """Audit online reproduction without using the report in decisions."""
    if report.get('protocol') != 'base_v3_obb_paper_final_report_v1':
        raise RuntimeError('Unexpected paper final report protocol')
    reference = {row['frame_key']: row for row in report.get('records', [])}
    if len(reference) != len(report.get('records', [])):
        raise RuntimeError('Paper report contains duplicate frame keys')
    geometry_max = 0.0
    value_max = 0.0
    state_mismatches = 0
    source_mismatches = 0
    geometry_value_mismatches = 0
    component_value_mismatches = 0
    missing = []
    for row in records:
        expected = reference.get(row['frame_key'])
        if expected is None:
            missing.append(row['frame_key'])
            continue
        geometry_difference = _numeric_difference(
            row['detector']['raw_obb'],
            expected['detector']['raw_obb'])
        if geometry_difference is None:
            geometry_value_mismatches += 1
        else:
            geometry_max = max(geometry_max, geometry_difference)
        for method in METHODS:
            for component in ('center', 'scale', 'angle'):
                current = row['observations'][method][component]
                target = expected['observations'][method][component]
                state_mismatches += int(
                    current['state'] != target['state']
                    or bool(current['valid']) != bool(target['valid']))
                source_mismatches += int(
                    current.get('source') != target.get('source'))
                difference = _numeric_difference(
                    current.get('value'), target.get('value'))
                if difference is None:
                    component_value_mismatches += 1
                else:
                    value_max = max(value_max, difference)
    extra = sorted(set(reference)-{row['frame_key'] for row in records})
    complete = not missing and not extra and len(records) == len(reference)
    passed = bool(
        complete and state_mismatches == 0 and source_mismatches == 0
        and geometry_value_mismatches == 0
        and component_value_mismatches == 0
        and geometry_max <= float(tolerance)
        and value_max <= float(tolerance))
    return dict(
        reference_frame_count=len(reference), online_frame_count=len(records),
        complete_frame_set=complete, missing_reference_keys=missing,
        unprocessed_reference_keys=extra,
        maximum_base_v3_geometry_difference=geometry_max,
        maximum_component_value_difference=value_max,
        component_state_mismatch_count=state_mismatches,
        component_source_mismatch_count=source_mismatches,
        geometry_availability_or_shape_mismatch_count=(
            geometry_value_mismatches),
        component_value_availability_or_shape_mismatch_count=(
            component_value_mismatches),
        tolerance=float(tolerance), passed=passed,
        reference_used_in_online_decisions=False,
        parameter_tuning_authorized=False)


def summarize(records):
    sources = collections.Counter()
    present = collections.Counter()
    states = {
        method: {component: collections.Counter()
                 for component in ('center', 'scale', 'angle')}
        for method in METHODS}
    for record in records:
        detector = record['detector_components']
        sources[detector['anchor_source']] += 1
        for name in ('base_v3_box', 'k1_box', 'dino_box'):
            present[name] += int(detector.get(name) is not None)
        for method in METHODS:
            for component in ('center', 'scale', 'angle'):
                state = record['observations'][method][component]['state']
                states[method][component][state] += 1
    return dict(
        frame_count=len(records), anchor_source_counts=dict(sources),
        detector_present_counts=dict(present),
        component_state_counts={
            method: {component: dict(counts)
                     for component, counts in components.items()}
            for method, components in states.items()})


def run(args, contract, identities):
    import cv2
    from mmdet.apis import inference_detector

    image_rows = discover_images(
        args.image_dir, args.sequence, args.max_frames)
    if (args.expected_frame_count is not None
            and len(image_rows) != int(args.expected_frame_count)):
        raise RuntimeError(
            'Image count mismatch: expected {}, got {}'.format(
                args.expected_frame_count, len(image_rows)))
    dino_model, _dino_cfg = build_constructor_loaded_detector(
        args.dino_config, args.device,
        overrides={'dino_head_checkpoint': args.dino_checkpoint})
    base_model, base_cfg = build_constructor_loaded_detector(
        args.base_v3_config, args.device)
    if not hasattr(base_model, 'last_inference_records'):
        raise RuntimeError('Base V3 runtime evidence interface is missing')
    base_adapter = BaseV3OnlineAdapter(base_model, base_cfg)
    calibration = json.loads(Path(
        args.runtime_calibration_v51).read_text(encoding='utf-8'))
    observation = PaperObservationManagerV1(calibration)
    history = OnlineDinoHistoryBuffer(contract['history_horizon'])
    records = []
    for identity, image_path in image_rows:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Cannot read image: ' + image_path)
        dino_result = inference_detector(dino_model, image_path)
        dino_detection = top1_detection(dino_result)
        history_inputs = history.prepare(identity, image)
        _base_result, base_audit = base_adapter.infer(
            image_path,
            None if dino_detection is None else dino_detection[:5],
            history_inputs)
        detector_row = assemble_detector_row(
            identity, image.shape, dino_detection, base_audit)
        online = observation.update(detector_row)
        online['detector_components'] = dict(
            base_v3_box=copy.deepcopy(detector_row['base_v3_box']),
            k1_box=copy.deepcopy(detector_row['k1_box']),
            dino_box=copy.deepcopy(detector_row['dino_box']),
            anchor_source=detector_row['anchor_source'],
            anchor_score=detector_row['anchor_score'],
            base_v3_output_score=base_audit.get('base_v3_output_score'),
            base_v3_output_score_semantics=base_audit.get(
                'base_v3_output_score_semantics'))
        online['history'] = dict(
            frame_keys=history_inputs['causal_history_frame_keys'],
            valid_mask=[
                bool(value) for value in
                history_inputs['causal_history_valid_mask']])
        records.append(online)
        history.commit(
            identity, image,
            None if dino_detection is None else dino_detection[:5])
    base_contract = base_model.runtime_inference_contract()
    if base_contract.get('inference_component_mode') != 'full':
        raise RuntimeError('Base V3 must run the full causal component mode')
    payload = dict(
        protocol=PROTOCOL,
        execution_mode='true_online_model_inference',
        pipeline=list(contract['pipeline']),
        input=dict(image_dir=os.fspath(Path(args.image_dir).resolve())),
        artifacts=identities,
        detector_runtime_contracts=dict(
            base_v3=base_contract,
            base_v3_forward_counts=base_model.runtime_forward_counts(),
            dino_class=dino_model.__class__.__name__),
        history_contract=dict(
            horizon=contract['history_horizon'],
            strictly_previous=True, reset_on_sequence_or_frame_gap=True,
            stored_proposal='native_dino_top1'),
        observation_contract=dict(
            protocol='base_v3_obb_paper_runtime_v1',
            methods=list(METHODS),
            states=['measurement', 'prediction', 'unavailable'],
            v51_parameters_frozen=True),
        summary=summarize(records), records=records,
        paper_report_reproduction_audit=(
            None if args.paper_final_report is None else
            compare_paper_report(
                records, json.loads(Path(args.paper_final_report).read_text(
                    encoding='utf-8')))),
        leakage_controls=dict(
            annotations_or_gt_read=False, future_frames_used=False,
            domain_identity_used_in_model_decisions=False,
            sequence_frame_identity_used_for_reset_only=True),
        claim_status='TRUE_ONLINE_PIPELINE_INTEGRATION_ONLY',
        claim_limit=contract['claim_limit'])
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--dino-config', required=True)
    parser.add_argument('--dino-checkpoint', required=True)
    parser.add_argument('--base-v3-config', required=True)
    parser.add_argument('--runtime-calibration-v51', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sequence', action='append')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--expected-frame-count', type=int)
    parser.add_argument('--paper-final-report')
    parser.add_argument('--identity-only', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text(encoding='utf-8'))
    validate_contract(contract)
    paths = dict(
        dino_config=args.dino_config,
        dino_checkpoint=args.dino_checkpoint,
        dino_backbone=contract['artifacts']['dino_backbone']['path'],
        base_v3_config=args.base_v3_config,
        base_v3_fixed_config=contract['artifacts'][
            'base_v3_fixed_config']['path'],
        base_v3_source_v3_config=contract['artifacts'][
            'base_v3_source_v3_config']['path'],
        base_v3_source_v2_config=contract['artifacts'][
            'base_v3_source_v2_config']['path'],
        base_v3_source_v1_config=contract['artifacts'][
            'base_v3_source_v1_config']['path'],
        k1_config=contract['artifacts']['k1_config']['path'],
        k1_checkpoint=contract['artifacts']['k1_checkpoint']['path'],
        base_v3_refiner=contract['artifacts']['base_v3_refiner']['path'],
        base_v3_promotion_report=contract['artifacts'][
            'base_v3_promotion_report']['path'],
        runtime_calibration_v51=args.runtime_calibration_v51)
    identities = verify_artifacts(contract, paths)
    if args.paper_final_report is not None:
        reference = _identity(args.paper_final_report)
        expected = contract['reference_artifacts'][
            'paper_final_report']['sha256']
        if reference['sha256'] != expected:
            raise RuntimeError(
                'paper_final_report identity mismatch: expected {}, got {}'
                .format(expected, reference['sha256']))
        identities['paper_final_report'] = reference
    image_rows = discover_images(
        args.image_dir, args.sequence, args.max_frames)
    if (args.expected_frame_count is not None
            and len(image_rows) != int(args.expected_frame_count)):
        raise RuntimeError(
            'Image count mismatch: expected {}, got {}'.format(
                args.expected_frame_count, len(image_rows)))
    if args.identity_only:
        print(json.dumps(dict(
            protocol=PROTOCOL, identity_only=True,
            image_count=len(image_rows), artifacts=identities), indent=2))
        return
    payload = run(args, contract, identities)
    output = _write_exact(args.out_json, payload)
    print(json.dumps(dict(
        output=output, execution_mode=payload['execution_mode'],
        pipeline=payload['pipeline'], summary=payload['summary'],
        leakage_controls=payload['leakage_controls'],
        claim_status=payload['claim_status']), indent=2))


if __name__ == '__main__':
    main()
