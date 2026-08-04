#!/usr/bin/env python3
"""One-shot fixed target-dev audit for the source-selected temporal S7 model.

This entry is deliberately narrower than the historical relaxed-gate audit.
It accepts only the source-selected epoch-4 relative-quality checkpoint, first
revalidates its exact-retention source evidence, and then compares it with the
native source-safe checkpoint on three immutable target-dev slices.  Target
labels are evaluation-only and cannot select a checkpoint or tune a parameter.
"""

import argparse
import json
import os
from typing import Dict, Sequence

import torch

from crane_project.tools import dino_teacher_common as common
from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import (
    dino_teacher_token_scale_rpn_coverage_audit as coverage)


AUDIT_NAME = 'DINO S7 Temporal Fixed Three-Slice Target-Dev Audit V1'
PROTOCOL_VERSION = 1
EXPECTED_SOURCE_EPOCH = 4
SOURCE_GATE = dict(
    min_full_top1=688,
    min_small_top1=311,
    max_full_mcml=3,
    max_small_mcml=3,
    max_lost_correct=0)
EXPECTED_TARGET_SLICES = {
    row['name']: row for row in (
        coverage.parse_target_slice(value)
        for value in coverage.DEFAULT_TARGET_SLICES)}
TARGET_GATES = dict(
    seq02_far=dict(
        frame_count=40, baseline_top1=38, baseline_mcml=1,
        min_candidate_top1=38, max_candidate_mcml=1),
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


def _selected_source_history(result: Dict) -> Dict:
    best_epoch = int(result.get('source', {}).get('best_epoch', -1))
    history = result.get('source', {}).get('history', [])
    matches = [row for row in history
               if int(row.get('epoch', -1)) == best_epoch]
    if len(matches) != 1:
        raise ValueError(
            'Expected one source history row for best epoch {}, found {}'
            .format(best_epoch, len(matches)))
    return matches[0]


def strict_source_gate(result: Dict) -> Dict:
    """Revalidate the exact source evidence that authorizes target-dev."""
    source = result.get('source', {})
    best_epoch = int(source.get('best_epoch', -1))
    row = _selected_source_history(result)
    full = row.get('source_val') or {}
    small = row.get('source_small_val') or {}
    retention = row.get('source_exact_retention') or {}
    stored_gate = row.get('source_selection_gate') or {}
    stored_s7_gate = row.get('s7_source_gate') or {}
    isolation = result.get('isolation', {})
    temporal_protocol = result.get('protocol', {}).get(
        's7_temporal_association') or {}
    baseline_correct = int(retention.get('baseline_correct_count', -1))
    retained_correct = int(retention.get('retained_correct_count', -2))
    checks = dict(
        protocol_version_current=int(result.get('protocol_version', -1)) >= 20,
        fixed_best_epoch=best_epoch == EXPECTED_SOURCE_EPOCH,
        selected_history_row=bool(row.get('selected_as_best', False)),
        checkpoint_saved=bool(row.get('checkpoint_saved', False)),
        stored_source_gate_passed=(
            row.get('source_selection_gate_passed') is True
            and stored_gate.get('passed') is True
            and stored_s7_gate.get('passed') is True),
        stored_retention_passed=(row.get('source_retention_passed') is True),
        exact_old_correct_retention=(
            int(retention.get('lost_correct_count', -1)) == 0
            and baseline_correct >= 0
            and retained_correct == baseline_correct),
        full_top1_absolute=(
            int(full.get('top1_hits', -1))
            >= SOURCE_GATE['min_full_top1']),
        small_top1_absolute=(
            int(small.get('top1_hits', -1))
            >= SOURCE_GATE['min_small_top1']),
        full_mcml_absolute=(
            int(full.get('top1_mcml', 10 ** 9))
            <= SOURCE_GATE['max_full_mcml']),
        small_mcml_absolute=(
            int(small.get('top1_mcml', 10 ** 9))
            <= SOURCE_GATE['max_small_mcml']),
        temporal_metric_gate=(
            (stored_gate.get('checks') or {}).get(
                'source_dfr_nonregression') is True
            and (stored_gate.get('checks') or {}).get(
                'source_aci_nonregression') is True),
        temporal_relative_quality_protocol=(
            isolation.get('train_components') == 's7_temporal_association'
            and temporal_protocol.get('candidate_quality_head') is True
            and temporal_protocol.get('relative_quality') is True
            and int(temporal_protocol.get('min_confirmations', -1)) == 1),
        source_result_target_not_read=(
            result.get('target_dev') is None
            and temporal_protocol.get('target_read') is False
            and isolation.get('target_used_for_training') is False
            and isolation.get('target_used_for_checkpoint_selection') is False
            and isolation.get('target_labels_used_for_evaluation_only') is False),
        frozen_components_preserved=(
            isolation.get('dino_parameters_unchanged') is True
            and isolation.get('frozen_head_parameters_unchanged') is True))
    return dict(
        best_epoch=best_epoch,
        selected_checkpoint=result.get('source_selected_checkpoint'),
        checks=checks,
        passed=all(checks.values()),
        full=dict(
            top1_hits=full.get('top1_hits'),
            frame_count=full.get('frame_count'),
            top1_mcml=full.get('top1_mcml'),
            recall_at_100=full.get('recall_at_100'),
            top1_dfr_fraction_per_frame=full.get(
                'top1_dfr_fraction_per_frame'),
            top1_aci=full.get('top1_aci')),
        small=dict(
            top1_hits=small.get('top1_hits'),
            frame_count=small.get('frame_count'),
            top1_mcml=small.get('top1_mcml'),
            recall_at_100=small.get('recall_at_100')),
        retention=retention)


def configure_locked_model(args):
    """Install the exact source-selected architecture; no target knobs."""
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
    args.s7_lane_hidden = 32
    args.s7_lane_max_adjustment = 2.0
    args.s7_quality_suppression = False

    args.train_components = 's7_temporal_association'
    args.s7_temporal_association = True
    args.s7_temporal_quality_head = True
    args.s7_temporal_quality_hidden = 128
    args.s7_temporal_relative_quality = True
    args.s7_temporal_max_candidates = 100
    args.s7_temporal_min_confirmations = 1
    args.s7_temporal_override_margin = 0.25
    args.s7_temporal_max_center_distance = 3.0
    args.s7_temporal_min_riou = 0.05
    args.s7_temporal_min_appearance = 0.20
    args.source_temporal_attribution_audit = False
    args.source_temporal_immediate_override_audit = False


def validate_args(args):
    if int(args.seed) != 0:
        raise ValueError('The fixed target-dev protocol requires --seed 0')
    if args.dinov2_model != 'dinov2_vitl14':
        raise ValueError('The fixed protocol requires dinov2_vitl14')
    for name in ('source_result_json', 'baseline_checkpoint',
                 'candidate_checkpoint', 'dinov2_checkpoint'):
        path = getattr(args, name)
        if not os.path.isfile(path):
            raise ValueError('{} does not exist: {}'.format(name, path))
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
    args.strict_source_gate = strict_source_gate(args.source_result)
    if not args.strict_source_gate['passed']:
        failed = sorted(
            name for name, passed in args.strict_source_gate['checks'].items()
            if not passed)
        raise ValueError('Strict source gate failed: {}'.format(
            ', '.join(failed)))
    selected = args.strict_source_gate.get('selected_checkpoint')
    if not selected or os.path.realpath(selected) != os.path.realpath(
            args.candidate_checkpoint):
        raise ValueError(
            'Candidate checkpoint must equal source_selected_checkpoint')
    configure_locked_model(args)
    args.parsed_target_slices = [
        dict(EXPECTED_TARGET_SLICES[name])
        for name in ('seq02_far', 'seq02_dark', 'seq03_small')]


def candidate_checkpoint_gate(payload: Dict, source_gate: Dict) -> Dict:
    stored = labeller.source_selected_checkpoint_gate(
        payload,
        min_full_top1=SOURCE_GATE['min_full_top1'],
        min_small_top1=SOURCE_GATE['min_small_top1'],
        max_mcml=SOURCE_GATE['max_full_mcml'])
    architecture = payload.get('s7_architecture') or {}
    temporal_protocol = payload.get('training_protocol', {}).get(
        's7_temporal_association') or {}
    checks = dict(stored.get('checks') or {})
    checks.update(
        checkpoint_epoch=(
            int(payload.get('epoch', -1)) == EXPECTED_SOURCE_EPOCH
            == int(source_gate.get('best_epoch', -2))),
        checkpoint_best_epoch=(
            int(payload.get('best_epoch', -1)) == EXPECTED_SOURCE_EPOCH),
        s7_inference_enabled=(payload.get('s7_inference_enabled') is True),
        temporal_quality_architecture=(
            architecture.get('temporal_association') is True
            and architecture.get('temporal_quality_head') is True
            and int(architecture.get('temporal_min_confirmations', -1)) == 1),
        relative_quality_checkpoint=(
            temporal_protocol.get('relative_quality') is True
            and temporal_protocol.get('candidate_quality_head') is True
            and temporal_protocol.get('target_read') is False))
    return dict(checks=checks, passed=all(checks.values()))


def compact_rows(rows: Sequence[Dict]) -> Sequence[Dict]:
    compact = []
    for row in rows:
        detections = row.get('detections') or []
        candidate_merge = row.get('candidate_merge')
        if candidate_merge is not None:
            candidate_merge = dict(candidate_merge)
            candidate_merge.pop('temporal_selection', None)
        compact.append(dict(
            role=row.get('role'), split=row.get('split'), seq=row.get('seq'),
            frame=int(row.get('frame', -1)),
            feature_cache_hit=bool(row.get('feature_cache_hit', False)),
            metrics=row.get('metrics'), candidate_merge=candidate_merge,
            temporal_selection=row.get('temporal_selection'),
            top1_detection=(None if not detections else detections[0])))
    return compact


def fixed_slice_result(name: str, baseline_rows: Sequence[Dict],
                       candidate_rows: Sequence[Dict]) -> Dict:
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
        delta_top1=(int(candidate['top1_hits'])
                    - int(baseline['top1_hits'])),
        delta_mcml=(int(candidate['top1_mcml'])
                    - int(baseline['top1_mcml'])),
        delta_recall_at_100=(int(candidate['recall_at_100'])
                             - int(baseline['recall_at_100'])),
        gained_frame_keys=gained, lost_frame_keys=lost,
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
        allow_temporal_association_initialization=True)
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
        allow_temporal_association_initialization=True)
    labeller.load_heads_checkpoint_state(candidate_heads, candidate_payload)
    if baseline_heads.s7_inference_enabled():
        raise RuntimeError('Native baseline unexpectedly enabled S7 inference')
    if not candidate_heads.s7_inference_enabled():
        raise RuntimeError('Candidate checkpoint did not enable S7 inference')
    baseline_versions = common.module_parameter_versions(baseline_heads)
    candidate_versions = common.module_parameter_versions(candidate_heads)

    slices = {}
    for spec in args.parsed_target_slices:
        records = coverage.discover_target_records(args.data_root, spec)
        baseline_rows = labeller.evaluate_records(
            dino, baseline_heads, records, args, dino_device, head_device,
            role='target_dev_diagnosis_only')
        candidate_rows = labeller.evaluate_records(
            dino, candidate_heads, records, args, dino_device, head_device,
            role='target_dev_diagnosis_only')
        slices[spec['name']] = fixed_slice_result(
            spec['name'], baseline_rows, candidate_rows)

    dino_unchanged = dino_versions == common.module_parameter_versions(dino)
    baseline_unchanged = (
        baseline_versions == common.module_parameter_versions(baseline_heads))
    candidate_unchanged = (
        candidate_versions == common.module_parameter_versions(candidate_heads))
    if not (dino_unchanged and baseline_unchanged and candidate_unchanged):
        raise RuntimeError('Read-only parameter invariant failed')

    passed = all(result['passed'] for result in slices.values())
    decision = (
        'S7_TEMPORAL_FIXED_TARGET_DEV_PASS_STAGE3_AUTHORIZED'
        if passed else
        'S7_TEMPORAL_FIXED_TARGET_DEV_FAIL_KEEP_CURRENT_SOURCE_MODEL')
    payload = dict(
        audit=AUDIT_NAME, protocol_version=PROTOCOL_VERSION,
        decision=decision,
        eligible_for_deployment=False,
        eligible_for_final_test=False,
        eligible_for_stage3_student_training=bool(passed),
        source_gate=dict(policy=SOURCE_GATE, result=args.strict_source_gate),
        checkpoint_gate=checkpoint_gate,
        checkpoints=dict(
            baseline=os.path.abspath(args.baseline_checkpoint),
            candidate=os.path.abspath(args.candidate_checkpoint),
            candidate_epoch=EXPECTED_SOURCE_EPOCH),
        protocol=dict(
            fixed_one_shot=True,
            all_frames_unified_policy=True,
            candidate_mode='s7_temporal_association_relative_quality',
            target_role='target_dev_diagnosis_only',
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_used_for_parameter_tuning=False,
            target_labels_used_for_evaluation_only=True,
            target_slices=args.parsed_target_slices,
            target_gates=TARGET_GATES,
            dfr_aci_reporting_only=True,
            pass_rule=(
                'far_dark_absolute_and_baseline_nonregression_plus_'
                'seq03_small_strict_top1_gain_r100_64_mcml_nonregression')),
        isolation=dict(
            optimizer_steps=0,
            parameter_updates_performed=False,
            dino_parameters_unchanged=dino_unchanged,
            baseline_parameters_unchanged=baseline_unchanged,
            candidate_parameters_unchanged=candidate_unchanged),
        slices=slices)
    replacements = common.write_json_atomic(args.out_json, payload)
    for name in ('seq02_far', 'seq02_dark', 'seq03_small'):
        result = slices[name]
        print(
            '[fixed-target] {} baseline={}/{} candidate={}/{} '
            'mcml={}->{} r100={}->{} gained={} lost={} passed={}'.format(
                name,
                result['baseline']['top1_hits'],
                result['baseline']['frame_count'],
                result['candidate']['top1_hits'],
                result['candidate']['frame_count'],
                result['baseline']['top1_mcml'],
                result['candidate']['top1_mcml'],
                result['baseline']['recall_at_100'],
                result['candidate']['recall_at_100'],
                len(result['gained_frame_keys']),
                len(result['lost_frame_keys']),
                result['passed']))
    print('[dino-labeller] {}'.format(decision))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
