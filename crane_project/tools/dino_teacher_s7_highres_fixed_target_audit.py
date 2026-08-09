#!/usr/bin/env python3
"""One-shot fixed target-dev audit for the source-gated high-res S7 ranker.

The checkpoint architecture is validated with its training-time promotion
margin (0.25).  The only inference policy evaluated here is the source-selected
runtime margin (0.225).  Target labels are used for reporting and the fixed
pass/fail decision only; they cannot select a checkpoint, margin, or model.
"""

import argparse
import json
import math
import os
from typing import Dict, Sequence

import numpy as np
import torch

from crane_project.tools import dino_teacher_common as common
from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import (
    dino_teacher_token_scale_rpn_coverage_audit as coverage)


AUDIT_NAME = 'DINO S7 High-Resolution Fixed Three-Slice Target-Dev Audit V1'
PROTOCOL_VERSION = 25
SOURCE_PROTOCOL_VERSION = 24
EXPECTED_CHECKPOINT_EPOCH = 3
CHECKPOINT_ARCHITECTURE_MARGIN = 0.25
LOCKED_RUNTIME_MARGIN = 0.225
EXPECTED_MARGINS = (0.20, 0.225, 0.25)
EXPECTED_TARGET_SLICES = {
    row['name']: row for row in (
        coverage.parse_target_slice(value)
        for value in coverage.DEFAULT_TARGET_SLICES)}
TARGET_GATES = dict(
    seq02_far=dict(
        frame_count=40, baseline_top1=38, baseline_mcml=1,
        min_candidate_top1=39, max_candidate_mcml=1),
    seq02_dark=dict(
        frame_count=33, baseline_top1=29, baseline_mcml=1,
        min_candidate_top1=29, max_candidate_mcml=1),
    seq03_small=dict(
        frame_count=64, baseline_top1=50, baseline_mcml=6,
        baseline_recall_at_100=55, min_candidate_top1=51,
        max_candidate_mcml=6, min_candidate_recall_at_100=64,
        require_strict_top1_gain=True))


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument(
        '--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-result-json', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--candidate-checkpoint', required=True)
    parser.add_argument('--dinov2-repo', required=True)
    parser.add_argument('--dinov2-checkpoint', required=True)
    parser.add_argument('--dinov2-model', default='dinov2_vitl14')
    parser.add_argument('--dino-gpus', type=int, nargs='+', required=True)
    parser.add_argument('--head-gpu', type=int, default=0)
    parser.add_argument('--legacy-sdpa-query-chunk', type=int, default=512)
    parser.add_argument('--feature-cache-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def _close(left, right, tolerance=1e-12):
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def strict_source_margin_gate(result: Dict) -> Dict:
    """Revalidate the complete source-only authorization for target-dev."""
    protocol = result.get('protocol') or {}
    isolation = result.get('isolation') or {}
    audit = result.get('source_highres_margin_audit') or {}
    rows = audit.get('results') or []
    selected_rows = [
        row for row in rows
        if _close(row.get('promotion_margin'), LOCKED_RUNTIME_MARGIN)]
    selected = selected_rows[0] if len(selected_rows) == 1 else {}
    selected_full = selected.get('full_summary') or {}
    selected_small = selected.get('small_summary') or {}
    retention = selected.get('source_exact_retention') or {}
    selected_gate = selected.get('source_gate') or {}
    reference_rows = [
        row for row in rows
        if _close(row.get('promotion_margin'),
                  CHECKPOINT_ARCHITECTURE_MARGIN)]
    reference = reference_rows[0] if len(reference_rows) == 1 else {}
    checks = dict(
        protocol_version=(
            int(result.get('protocol_version', -1))
            == SOURCE_PROTOCOL_VERSION),
        passed_decision=(result.get('decision') ==
                         'SOURCE_ONLY_HIGHRES_MARGIN_AUDIT_GATE_PASSED_'
                         'TARGET_NOT_READ'),
        selected_margin=(
            _close(result.get('selected_promotion_margin'),
                   LOCKED_RUNTIME_MARGIN)
            and _close(audit.get('selected_margin'),
                       LOCKED_RUNTIME_MARGIN)),
        fixed_margin_grid=(
            len(audit.get('margins') or []) == len(EXPECTED_MARGINS)
            and all(_close(left, right) for left, right in zip(
                audit.get('margins') or [], EXPECTED_MARGINS))
            and len(rows) == len(EXPECTED_MARGINS)),
        shared_forward_readonly=(
            protocol.get('shared_model_forward') is True
            and audit.get('shared_model_forward') is True
            and int(audit.get('shared_model_forward_count', -1)) == 738
            and int(audit.get('margin_decision_count', -1)) == 2214
            and protocol.get('parameter_update') is False),
        source_only_target_unread=(
            protocol.get('source_only') is True
            and protocol.get('target_read') is False
            and audit.get('target_read') is False
            and result.get('target_dev') is None
            and isolation.get('target_used_for_training') is False
            and isolation.get('target_used_for_checkpoint_selection') is False
            and isolation.get('target_labels_used_for_evaluation_only') is False),
        frozen_components_preserved=(
            isolation.get('dino_parameters_unchanged') is True
            and isolation.get('detector_parameters_unchanged') is True
            and isolation.get('read_only_evaluation') is True
            and isolation.get('parameter_updates_performed') is False
            and int(isolation.get('trainable_parameter_count', -1)) == 0),
        checkpoint_epoch=(
            int(audit.get('checkpoint_epoch', -1))
            == EXPECTED_CHECKPOINT_EPOCH),
        checkpoint_selected=(
            bool(result.get('source_selected_checkpoint'))
            and result.get('source_selected_checkpoint')
            == audit.get('checkpoint')),
        selected_row_unique=(len(selected_rows) == 1),
        selected_formal_gate=(
            audit.get('formal_gate_passed') is True
            and selected.get('gate_passed') is True
            and selected_gate.get('passed') is True),
        selected_exact_retention=(
            int(retention.get('baseline_correct_count', -1)) == 677
            and int(retention.get('retained_correct_count', -1)) == 677
            and int(retention.get('lost_correct_count', -1)) == 0
            and int(retention.get('gained_correct_count', -1)) == 11),
        selected_absolute_metrics=(
            int(selected_full.get('frame_count', -1)) == 738
            and int(selected_full.get('top1_hits', -1)) == 688
            and int(selected_full.get('top1_mcml', -1)) <= 3
            and int(selected_small.get('frame_count', -1)) == 350
            and int(selected_small.get('top1_hits', -1)) == 311
            and int(selected_small.get('top1_mcml', -1)) <= 3),
        selected_temporal_metric_gate=(
            (selected_gate.get('checks') or {}).get(
                'source_dfr_nonregression') is True
            and (selected_gate.get('checks') or {}).get(
                'source_aci_nonregression') is True),
        training_margin_reproduced=(
            len(reference_rows) == 1
            and reference.get('epoch3_reference_reproduced') is True))
    return dict(
        checks=checks, passed=all(checks.values()),
        checkpoint=result.get('source_selected_checkpoint'),
        checkpoint_epoch=audit.get('checkpoint_epoch'),
        checkpoint_architecture_margin=CHECKPOINT_ARCHITECTURE_MARGIN,
        runtime_margin=LOCKED_RUNTIME_MARGIN,
        selected_source_full=selected_full,
        selected_source_small=selected_small,
        selected_source_retention=retention)


def configure_locked_model(args):
    """Install checkpoint architecture plus the separately locked runtime."""
    args.dino_height = 600
    args.dino_max_long_side = 1333
    args.patch_size = 14
    args.rpn_feat_channels = 256
    args.roi_fc_channels = 1024
    args.roi_samples = 256
    args.proposal_count = 2000
    args.max_detections = 2000
    args.feature_strides = None
    args.roi_nms_iou_thr = 0.5
    args.riou_thr = 0.5
    args.deployment_score_thr = 0.05
    args.valid_content_tolerance = 1e-3
    args.border_margin_ratio = 0.02

    args.s7_residual = True
    args.s7_channels = 128
    args.s7_rpn_feat_channels = 128
    args.s7_proposal_count = 500
    args.s7_nms_pre = 2000
    args.s7_anchor_sizes = [16.0, 32.0, 64.0, 128.0, 256.0]
    args.s7_protected_merge = True
    args.s7_merge_init_bias = -2.0
    args.s7_lane_arbitration = False
    args.s7_quality_suppression = False
    args.s7_temporal_association = False
    args.s7_temporal_quality_head = False
    args.s7_temporal_student = False
    args.s7_static_domain_ranker = False
    args.s7_selective_promotion = False
    args.s7_selective_two_frame = False

    args.train_components = 's7_highres_roi_ranker'
    args.s7_highres_roi_ranker = True
    args.s7_highres_channels = 32
    args.s7_highres_hidden = 32
    args.s7_highres_max_candidates = 32
    args.s7_highres_score_weight = 1.0
    # This is architecture provenance and must match the saved checkpoint.
    args.s7_highres_promotion_margin = CHECKPOINT_ARCHITECTURE_MARGIN
    # This is the source-selected read-only inference policy.
    args.runtime_highres_promotion_margin = LOCKED_RUNTIME_MARGIN


def validate_args(args):
    if int(args.seed) != 0:
        raise ValueError('The fixed target-dev protocol requires --seed 0')
    if args.dinov2_model != 'dinov2_vitl14':
        raise ValueError('The fixed target-dev protocol requires dinov2_vitl14')
    for name in ('source_result_json', 'baseline_checkpoint',
                 'candidate_checkpoint', 'dinov2_checkpoint'):
        path = getattr(args, name)
        if not os.path.isfile(path):
            raise ValueError('{} does not exist: {}'.format(name, path))
    if not os.path.isdir(args.dinov2_repo):
        raise ValueError('dinov2_repo does not exist: {}'.format(
            args.dinov2_repo))
    if os.path.realpath(args.baseline_checkpoint) == os.path.realpath(
            args.candidate_checkpoint):
        raise ValueError('Baseline and candidate checkpoints must differ')
    if os.path.exists(args.out_json):
        raise ValueError('Refusing to overwrite completed diagnosis: {}'.format(
            args.out_json))
    if not args.dino_gpus or len(args.dino_gpus) != len(set(args.dino_gpus)):
        raise ValueError('DINO GPU ids must be non-empty and unique')
    if args.head_gpu in args.dino_gpus:
        raise ValueError('Head GPU must be separate from DINO GPUs')
    if int(args.legacy_sdpa_query_chunk) <= 0:
        raise ValueError('--legacy-sdpa-query-chunk must be positive')

    with open(args.source_result_json, 'r') as handle:
        args.source_result = json.load(handle)
    args.strict_source_gate = strict_source_margin_gate(args.source_result)
    if not args.strict_source_gate['passed']:
        failed = sorted(
            name for name, passed in args.strict_source_gate['checks'].items()
            if not passed)
        raise ValueError('Strict source margin gate failed: {}'.format(
            ', '.join(failed)))
    selected = args.strict_source_gate.get('checkpoint')
    if not selected or os.path.realpath(selected) != os.path.realpath(
            args.candidate_checkpoint):
        raise ValueError(
            'Candidate checkpoint must equal the source-audit selection')
    configure_locked_model(args)
    args.parsed_target_slices = [
        dict(EXPECTED_TARGET_SLICES[name])
        for name in ('seq02_far', 'seq02_dark', 'seq03_small')]


def candidate_checkpoint_gate(payload: Dict, source_gate: Dict) -> Dict:
    architecture = payload.get('s7_architecture') or {}
    training = payload.get('training_protocol') or {}
    highres = training.get('s7_highres_roi_ranker') or {}
    checks = dict(
        source_audit_authorized=(source_gate.get('passed') is True),
        source_only_frozen=(
            payload.get('source_only') is True
            and payload.get('frozen_dinov2') is True),
        checkpoint_epoch=(
            int(payload.get('epoch', -1)) == EXPECTED_CHECKPOINT_EPOCH
            == int(source_gate.get('checkpoint_epoch', -2))),
        training_mode=(
            training.get('train_components') == 's7_highres_roi_ranker'),
        s7_inference_enabled=(payload.get('s7_inference_enabled') is True),
        highres_architecture=(
            architecture.get('enabled') is True
            and architecture.get('protected_merge') is True
            and architecture.get('highres_roi_ranker') is True
            and int(architecture.get('highres_channels', -1)) == 32
            and int(architecture.get('highres_hidden', -1)) == 32
            and int(architecture.get('highres_max_candidates', -1)) == 32
            and _close(architecture.get('highres_score_weight'), 1.0)
            and _close(architecture.get('highres_promotion_margin'),
                       CHECKPOINT_ARCHITECTURE_MARGIN)),
        highres_training_protocol=(
            highres.get('frozen_detector') is True
            and highres.get('source_only') is True
            and highres.get('target_read') is False
            and highres.get('inference_slice_routing') is False
            and highres.get('sequence_identity_feature') is False
            and highres.get('additional_dino_forward') is False
            and highres.get('dense_feature_history') is False
            and int(highres.get('highres_channels', -1)) == 32
            and int(highres.get('hidden', -1)) == 32
            and int(highres.get('max_candidates', -1)) == 32
            and _close(highres.get('score_weight'), 1.0)
            and _close(highres.get('promotion_margin'),
                       CHECKPOINT_ARCHITECTURE_MARGIN)))
    return dict(checks=checks, passed=all(checks.values()))


def compact_rows(rows: Sequence[Dict]) -> Sequence[Dict]:
    compact = []
    for row in rows:
        detections = row.get('detections') or []
        merge = row.get('candidate_merge')
        ranker = None if merge is None else (
            merge.get('s7_highres_roi_ranker') or {})
        compact.append(dict(
            role=row.get('role'), split=row.get('split'), seq=row.get('seq'),
            frame=int(row.get('frame', -1)),
            feature_cache_hit=bool(row.get('feature_cache_hit', False)),
            metrics=row.get('metrics'),
            selected_source=(None if merge is None else
                             merge.get('raw_top1_source')),
            highres_selection=ranker,
            top1_detection=(None if not detections else detections[0])))
    return compact


def native_lane_reproduction(
        baseline_rows: Sequence[Dict], native_rows: Sequence[Dict]) -> Dict:
    baseline = {labeller.source_frame_key(row): row for row in baseline_rows}
    native = {labeller.source_frame_key(row): row for row in native_rows}
    same_keys = set(baseline) == set(native)
    mismatched = []
    for key in sorted(set(baseline) & set(native)):
        left = (baseline[key].get('detections') or [])[:1]
        right = (native[key].get('detections') or [])[:1]
        if len(left) != len(right) or (
                left and not np.allclose(
                    np.asarray(left, dtype=np.float32),
                    np.asarray(right, dtype=np.float32),
                    rtol=0.0, atol=1e-5)):
            mismatched.append(key)
    return dict(
        same_frame_set=bool(same_keys),
        exact_native_top1=bool(same_keys and not mismatched),
        mismatched_frame_count=len(mismatched),
        mismatched_frame_keys=mismatched)


def fixed_slice_result(name: str, baseline_rows: Sequence[Dict],
                       candidate_rows: Sequence[Dict],
                       native_reproduction: Dict) -> Dict:
    policy = TARGET_GATES[name]
    baseline = labeller.summarize_rows(baseline_rows)
    candidate = labeller.summarize_rows(candidate_rows)
    baseline_hits = {
        labeller.source_frame_key(row): bool(row['metrics']['top1_hit'])
        for row in baseline_rows}
    candidate_hits = {
        labeller.source_frame_key(row): bool(row['metrics']['top1_hit'])
        for row in candidate_rows}
    gained = sorted(
        key for key in baseline_hits
        if not baseline_hits[key] and candidate_hits.get(key, False))
    lost = sorted(
        key for key in baseline_hits
        if baseline_hits[key] and not candidate_hits.get(key, False))
    checks = dict(
        native_lane_reproduced=(
            native_reproduction.get('exact_native_top1') is True),
        baseline_frame_count=(
            int(baseline.get('frame_count', -1)) == policy['frame_count']),
        candidate_frame_count=(
            int(candidate.get('frame_count', -1)) == policy['frame_count']),
        baseline_top1_reference=(
            int(baseline.get('top1_hits', -1)) == policy['baseline_top1']),
        baseline_mcml_reference=(
            int(baseline.get('top1_mcml', -1)) == policy['baseline_mcml']),
        candidate_top1_floor=(
            int(candidate.get('top1_hits', -1))
            >= policy['min_candidate_top1']),
        candidate_mcml_bound=(
            int(candidate.get('top1_mcml', 10 ** 9))
            <= policy['max_candidate_mcml']),
        top1_nonregression=(
            int(candidate.get('top1_hits', -1))
            >= int(baseline.get('top1_hits', 10 ** 9))),
        mcml_nonregression=(
            int(candidate.get('top1_mcml', 10 ** 9))
            <= int(baseline.get('top1_mcml', -1))))
    if 'baseline_recall_at_100' in policy:
        checks['baseline_recall_at_100_reference'] = (
            int(baseline.get('recall_at_100', -1))
            == policy['baseline_recall_at_100'])
    if policy.get('require_strict_top1_gain', False):
        checks['strict_top1_gain'] = (
            int(candidate.get('top1_hits', -1))
            > int(baseline.get('top1_hits', 10 ** 9)))
    if 'min_candidate_recall_at_100' in policy:
        checks['candidate_recall_at_100_floor'] = (
            int(candidate.get('recall_at_100', -1))
            >= policy['min_candidate_recall_at_100'])
    return dict(
        policy=policy, checks=checks, passed=all(checks.values()),
        delta_top1=int(candidate['top1_hits']) - int(baseline['top1_hits']),
        delta_mcml=int(candidate['top1_mcml']) - int(baseline['top1_mcml']),
        delta_recall_at_20=(int(candidate['recall_at_20'])
                            - int(baseline['recall_at_20'])),
        delta_recall_at_100=(int(candidate['recall_at_100'])
                             - int(baseline['recall_at_100'])),
        delta_mean_top1_riou=(float(candidate['mean_top1_riou'])
                              - float(baseline['mean_top1_riou'])),
        delta_dfr_percent_per_frame=(
            float(candidate['top1_dfr_percent_per_frame'])
            - float(baseline['top1_dfr_percent_per_frame'])),
        delta_aci=float(candidate['top1_aci']) - float(baseline['top1_aci']),
        gained_frame_keys=gained, lost_frame_keys=lost,
        native_lane_reproduction=native_reproduction,
        baseline=baseline, candidate=candidate,
        baseline_rows=compact_rows(baseline_rows),
        candidate_rows=compact_rows(candidate_rows))


def main():
    args = parse_args()
    validate_args(args)
    labeller.set_seed(args.seed)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    head_device = torch.device('cuda:{}'.format(args.head_gpu))
    dino_devices = [torch.device('cuda:{}'.format(gpu))
                    for gpu in args.dino_gpus]
    dino_device = dino_devices[0]
    dino, loaded_patch_size = common.load_frozen_dinov2(
        args.dinov2_repo, args.dinov2_checkpoint, args.dinov2_model,
        dino_devices, args.legacy_sdpa_query_chunk)
    if int(loaded_patch_size) != int(args.patch_size):
        raise RuntimeError('Unexpected DINO patch size')
    dino_versions = common.module_parameter_versions(dino)
    in_channels = int(getattr(dino, 'embed_dim', 0))
    if in_channels <= 0:
        raise RuntimeError('Frozen DINO backbone has no embed_dim')

    baseline_payload = torch.load(args.baseline_checkpoint, map_location='cpu')
    candidate_payload = torch.load(args.candidate_checkpoint, map_location='cpu')
    labeller.validate_checkpoint(
        baseline_payload, in_channels, args,
        allow_training_mode_mismatch=True,
        allow_s7_base_initialization=True,
        allow_highres_roi_initialization=True)
    labeller.validate_checkpoint(candidate_payload, in_channels, args)
    checkpoint_gate = candidate_checkpoint_gate(
        candidate_payload, args.strict_source_gate)
    if not checkpoint_gate['passed']:
        failed = sorted(
            name for name, passed in checkpoint_gate['checks'].items()
            if not passed)
        raise RuntimeError('Candidate checkpoint gate failed: {}'.format(
            ', '.join(failed)))

    baseline_heads = labeller.FrozenDinoRotatedHeads(
        in_channels, args).to(head_device)
    candidate_heads = labeller.FrozenDinoRotatedHeads(
        in_channels, args).to(head_device)
    labeller.load_heads_checkpoint_state(
        baseline_heads, baseline_payload,
        allow_s7_base_initialization=True,
        allow_highres_roi_initialization=True)
    labeller.load_heads_checkpoint_state(candidate_heads, candidate_payload)
    if baseline_heads.s7_inference_enabled():
        raise RuntimeError('Native baseline unexpectedly enabled S7 inference')
    if not candidate_heads.s7_inference_enabled():
        raise RuntimeError('Candidate checkpoint did not enable S7 inference')
    baseline_versions = common.module_parameter_versions(baseline_heads)
    candidate_versions = common.module_parameter_versions(candidate_heads)

    slices = {}
    total_candidate_forwards = 0
    for spec in args.parsed_target_slices:
        records = coverage.discover_target_records(args.data_root, spec)
        baseline_rows = labeller.evaluate_records(
            dino, baseline_heads, records, args, dino_device, head_device,
            role='target_dev_diagnosis_only')
        evaluated = labeller.evaluate_highres_margin_grid_records(
            dino, candidate_heads, records, args, dino_device, head_device,
            [args.runtime_highres_promotion_margin],
            role='target_dev_diagnosis_only')
        candidate_rows = evaluated['rows_by_margin'][
            args.runtime_highres_promotion_margin]
        native_reproduction = native_lane_reproduction(
            baseline_rows, evaluated['baseline_rows'])
        slices[spec['name']] = fixed_slice_result(
            spec['name'], baseline_rows, candidate_rows,
            native_reproduction)
        total_candidate_forwards += int(
            evaluated['shared_model_forward_count'])

    dino_unchanged = dino_versions == common.module_parameter_versions(dino)
    baseline_unchanged = (
        baseline_versions == common.module_parameter_versions(baseline_heads))
    candidate_unchanged = (
        candidate_versions == common.module_parameter_versions(candidate_heads))
    if not (dino_unchanged and baseline_unchanged and candidate_unchanged):
        raise RuntimeError('Read-only parameter invariant failed')

    passed = all(result['passed'] for result in slices.values())
    decision = (
        'S7_HIGHRES_FIXED_TARGET_DEV_PASS_FULL_TEST_AUTHORIZED'
        if passed else
        'S7_HIGHRES_FIXED_TARGET_DEV_FAIL_KEEP_NATIVE_BASELINE')
    payload = dict(
        audit=AUDIT_NAME, protocol_version=PROTOCOL_VERSION,
        decision=decision,
        eligible_for_deployment=False,
        eligible_for_full_test=bool(passed),
        eligible_for_unseen_generalization_claim=False,
        source_gate=args.strict_source_gate,
        checkpoint_gate=checkpoint_gate,
        checkpoints=dict(
            baseline=os.path.abspath(args.baseline_checkpoint),
            candidate=os.path.abspath(args.candidate_checkpoint),
            candidate_epoch=EXPECTED_CHECKPOINT_EPOCH,
            checkpoint_architecture_margin=CHECKPOINT_ARCHITECTURE_MARGIN,
            runtime_promotion_margin=LOCKED_RUNTIME_MARGIN),
        protocol=dict(
            fixed_one_shot=True,
            all_frames_unified_policy=True,
            target_role='target_dev_diagnosis_only',
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_used_for_parameter_tuning=False,
            target_labels_used_for_evaluation_only=True,
            checkpoint_and_margin_selected_on_source=True,
            runtime_margin_grid_search_on_target=False,
            candidate_mode='s7_highres_roi_ranker_native_protected',
            target_slices=args.parsed_target_slices,
            target_gates=TARGET_GATES,
            dfr_aci_reporting_only=True,
            slice_identity_used_as_model_input=False,
            sequence_identity_used_as_model_input=False,
            per_slice_routing=False,
            additional_dino_forward=False,
            dense_feature_history=False,
            pass_rule=(
                'far_39_dark_nonregression_small_51_r100_64_'
                'with_mcml_bounds_and_native_lane_reproduction')),
        isolation=dict(
            optimizer_steps=0, parameter_updates_performed=False,
            dino_parameters_unchanged=dino_unchanged,
            baseline_parameters_unchanged=baseline_unchanged,
            candidate_parameters_unchanged=candidate_unchanged,
            candidate_model_forward_count=total_candidate_forwards),
        target_dev=dict(slices=slices, all_gates_passed=bool(passed)))
    replacements = common.write_json_atomic(args.out_json, payload)
    print('[highres-fixed-target] {}'.format(decision))
    for name in ('seq02_far', 'seq02_dark', 'seq03_small'):
        row = slices[name]
        print('[highres-fixed-target] {} top1={}->{} mcml={}->{} '
              'r100={}->{} passed={}'.format(
                  name, row['baseline']['top1_hits'],
                  row['candidate']['top1_hits'],
                  row['baseline']['top1_mcml'],
                  row['candidate']['top1_mcml'],
                  row['baseline']['recall_at_100'],
                  row['candidate']['recall_at_100'], row['passed']))
    print('[json] nonfinite_replacements={}'.format(replacements))


if __name__ == '__main__':
    main()
