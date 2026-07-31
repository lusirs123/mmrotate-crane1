#!/usr/bin/env python3
"""One-shot target-dev comparison for the relaxed-gate S7 merge candidate.

The exact-retention selector remains unchanged.  This read-only audit admits
one already-fixed S7 checkpoint through a documented relaxed source gate, then
compares it with the native source-safe checkpoint on three predeclared target
slices.  Target labels never select a checkpoint or tune a parameter.
"""

import argparse
import json
import os
from typing import Dict, Sequence

import numpy as np
import torch

from crane_project.tools import dino_teacher_common as common
from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import (
    dino_teacher_token_scale_rpn_coverage_audit as coverage)


AUDIT_NAME = 'DINO S7 Relaxed-Gate Target-Dev Audit V1'
PROTOCOL_VERSION = 1
RELAXED_SOURCE_GATE = dict(
    max_lost_correct=1,
    min_retained_correct=676,
    min_full_top1=685,
    min_small_top1=308,
    max_full_mcml=3,
    max_small_mcml=3,
    min_gain_loss_ratio=5.0)
EXPECTED_TARGET_SLICES = {
    row['name']: row for row in (
        coverage.parse_target_slice(value)
        for value in coverage.DEFAULT_TARGET_SLICES)}


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument(
        '--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-result-json', required=True)
    parser.add_argument('--source-epoch', type=int, default=1)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--candidate-checkpoint', required=True)
    parser.add_argument(
        '--target-slice', action='append', dest='target_slices',
        help='NAME:SPLIT:SEQ:START:END; defaults to the fixed three slices.')
    parser.add_argument('--dinov2-repo', required=True)
    parser.add_argument('--dinov2-checkpoint', required=True)
    parser.add_argument('--dinov2-model', default='dinov2_vitl14')
    parser.add_argument('--dino-gpus', type=int, nargs='+', required=True)
    parser.add_argument('--head-gpu', type=int, default=0)
    parser.add_argument('--legacy-sdpa-query-chunk', type=int, default=512)
    parser.add_argument('--dino-height', type=int, default=600)
    parser.add_argument('--dino-max-long-side', type=int, default=1333)
    parser.add_argument('--patch-size', type=int, default=14)
    parser.add_argument('--rpn-feat-channels', type=int, default=256)
    parser.add_argument('--roi-fc-channels', type=int, default=1024)
    parser.add_argument('--roi-samples', type=int, default=256)
    parser.add_argument('--proposal-count', type=int, default=2000)
    parser.add_argument('--max-detections', type=int, default=2000)
    parser.add_argument('--roi-nms-iou-thr', type=float, default=0.5)
    parser.add_argument('--s7-channels', type=int, default=128)
    parser.add_argument('--s7-rpn-feat-channels', type=int, default=128)
    parser.add_argument('--s7-proposal-count', type=int, default=500)
    parser.add_argument('--s7-nms-pre', type=int, default=2000)
    parser.add_argument(
        '--s7-anchor-sizes', type=float, nargs='+',
        default=[16.0, 32.0, 64.0, 128.0, 256.0])
    parser.add_argument('--s7-merge-init-bias', type=float, default=-2.0)
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--deployment-score-thr', type=float, default=0.05)
    parser.add_argument('--valid-content-tolerance', type=float, default=1e-3)
    parser.add_argument('--border-margin-ratio', type=float, default=0.02)
    parser.add_argument('--feature-cache-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def relaxed_source_gate(result: Dict, epoch: int) -> Dict:
    history = result.get('source', {}).get('history', [])
    matches = [row for row in history if int(row.get('epoch', -1)) == int(epoch)]
    if len(matches) != 1:
        raise ValueError(
            'Expected one source history row for epoch {}, found {}'.format(
                epoch, len(matches)))
    row = matches[0]
    full = row.get('source_val') or {}
    small = row.get('source_small_val') or {}
    retention = row.get('source_exact_retention') or {}
    lost = int(retention.get('lost_correct_count', -1))
    gained = int(retention.get('gained_correct_count', -1))
    gain_loss_ratio = (
        None if lost == 0 else float(gained) / float(lost))
    isolation = result.get('isolation', {})
    checks = dict(
        source_result_target_not_read=(
            result.get('target_dev') is None
            and isolation.get('target_used_for_training') is False
            and isolation.get('target_used_for_checkpoint_selection') is False
            and isolation.get('target_labels_used_for_evaluation_only') is False),
        lost_correct_within_bound=(
            0 <= lost <= RELAXED_SOURCE_GATE['max_lost_correct']),
        retained_correct_floor=(
            int(retention.get('retained_correct_count', -1))
            >= RELAXED_SOURCE_GATE['min_retained_correct']),
        full_top1_floor=(
            int(full.get('top1_hits', -1))
            >= RELAXED_SOURCE_GATE['min_full_top1']),
        small_top1_floor=(
            int(small.get('top1_hits', -1))
            >= RELAXED_SOURCE_GATE['min_small_top1']),
        full_mcml_bound=(
            int(full.get('top1_mcml', 10 ** 9))
            <= RELAXED_SOURCE_GATE['max_full_mcml']),
        small_mcml_bound=(
            int(small.get('top1_mcml', 10 ** 9))
            <= RELAXED_SOURCE_GATE['max_small_mcml']),
        gain_loss_ratio=(
            lost == 0 or gain_loss_ratio
            >= RELAXED_SOURCE_GATE['min_gain_loss_ratio']))
    return dict(
        epoch=int(epoch), checks=checks, passed=all(checks.values()),
        full=full, small=small, retention=retention,
        gain_loss_ratio=gain_loss_ratio,
        protocol_amendment=(
            'Post-source-val relaxed gate authorizes target-dev diagnosis only; '
            'it does not replace the exact-retention deployment selector.'))


def validate_args(args):
    if args.seed != 0:
        raise ValueError('The audit requires --seed 0')
    for name in ('source_result_json', 'baseline_checkpoint',
                 'candidate_checkpoint', 'dinov2_checkpoint'):
        path = getattr(args, name)
        if not os.path.isfile(path):
            raise ValueError('{} does not exist: {}'.format(name, path))
    if os.path.realpath(args.baseline_checkpoint) == os.path.realpath(
            args.candidate_checkpoint):
        raise ValueError('Baseline and candidate checkpoints must differ')
    if os.path.exists(args.out_json):
        raise ValueError('Refusing to overwrite output: {}'.format(
            args.out_json))
    if not args.dino_gpus or len(args.dino_gpus) != len(set(args.dino_gpus)):
        raise ValueError('DINO GPU ids must be non-empty and unique')
    if args.head_gpu in args.dino_gpus:
        raise ValueError('Head GPU must be separate from DINO GPUs')
    if int(args.source_epoch) <= 0:
        raise ValueError('--source-epoch must be positive')
    positive = (
        args.dino_height, args.dino_max_long_side, args.patch_size,
        args.rpn_feat_channels, args.roi_fc_channels,
        args.roi_samples, args.proposal_count, args.max_detections,
        args.s7_channels, args.s7_rpn_feat_channels,
        args.s7_proposal_count, args.s7_nms_pre)
    if any(int(value) <= 0 for value in positive):
        raise ValueError('Head and proposal settings must be positive')
    if not 0.0 < float(args.roi_nms_iou_thr) <= 1.0:
        raise ValueError('--roi-nms-iou-thr must be in (0, 1]')
    args.s7_anchor_sizes = sorted(set(float(value)
                                       for value in args.s7_anchor_sizes))
    if not args.s7_anchor_sizes or any(
            value <= 0.0 for value in args.s7_anchor_sizes):
        raise ValueError('--s7-anchor-sizes must be positive')
    target_values = (coverage.DEFAULT_TARGET_SLICES
                     if args.target_slices is None else args.target_slices)
    args.parsed_target_slices = [coverage.parse_target_slice(value)
                                 for value in target_values]
    names = [row['name'] for row in args.parsed_target_slices]
    if len(args.parsed_target_slices) != 3 or len(set(names)) != 3:
        raise ValueError('Relaxed S7 audit requires three unique target slices')
    parsed = {row['name']: row for row in args.parsed_target_slices}
    if parsed != EXPECTED_TARGET_SLICES:
        raise ValueError(
            'Relaxed S7 audit requires exactly seq02_far, seq02_dark, and '
            'seq03_small with the predeclared frame ranges')
    with open(args.source_result_json, 'r') as handle:
        args.source_result = json.load(handle)
    args.relaxed_source_gate = relaxed_source_gate(
        args.source_result, args.source_epoch)
    if not args.relaxed_source_gate['passed']:
        failed = sorted(name for name, passed in
                        args.relaxed_source_gate['checks'].items() if not passed)
        raise ValueError('Relaxed source gate failed: {}'.format(
            ', '.join(failed)))
    args.feature_strides = None
    args.s7_residual = True
    args.s7_protected_merge = True
    args.train_components = 's7_merge'
    args.s7_component_checkpoint = None


def enrich_candidate_merge(candidate_merge: Dict, original: np.ndarray,
                           args) -> Dict:
    if candidate_merge is None:
        return None
    result = dict(candidate_merge)
    source_metrics = {}
    for source, detection in result.get('source_top1_detections', {}).items():
        values = np.asarray(
            [] if detection is None else [detection],
            dtype=np.float32).reshape((-1, 6))
        metrics = labeller.ranked_detection_metrics(
            values, original, args.riou_thr, args.deployment_score_thr)
        source_metrics[source] = dict(
            top1_hit=bool(metrics['top1_hit']),
            top1_riou=float(metrics['top1_riou']),
            top1_score=metrics['top1_score'])
    result['source_top1_metrics'] = source_metrics
    return result


def evaluate_one(heads, feature: torch.Tensor, img_meta: Dict,
                 original: np.ndarray, args) -> Dict:
    raw_detections = heads.simple_test(feature, img_meta)
    candidate_merge = enrich_candidate_merge(
        heads._last_candidate_merge, original, args)
    raw_metrics = labeller.ranked_detection_metrics(
        raw_detections, original, args.riou_thr, args.deployment_score_thr)
    detections, filter_stats = labeller.filter_valid_rotated_detections(
        raw_detections, img_meta, args.valid_content_tolerance)
    metrics = labeller.ranked_detection_metrics(
        detections, original, args.riou_thr, args.deployment_score_thr)
    metrics.update(filter_stats)
    metrics['raw_unfiltered'] = raw_metrics
    metrics['filter_effect'] = dict(
        removed_usable_geometry=bool(
            raw_metrics['best_usable_rank'] is not None
            and metrics['best_usable_rank'] is None),
        promoted_to_top1=bool(
            not raw_metrics['top1_hit'] and metrics['top1_hit']),
        demoted_from_top1=bool(
            raw_metrics['top1_hit'] and not metrics['top1_hit']),
        raw_best_usable_rank=raw_metrics['best_usable_rank'],
        filtered_best_usable_rank=metrics['best_usable_rank'])
    metrics.update(labeller.gt_border_metrics(
        original, img_meta, args.border_margin_ratio,
        args.valid_content_tolerance))
    return dict(
        candidate_merge=candidate_merge, metrics=metrics,
        top1_detection=(None if detections.shape[0] == 0 else
                        [float(value) for value in detections[0].tolist()]))


def evaluate_pair(dino, baseline_heads, candidate_heads,
                  records: Sequence[Dict], args, dino_device,
                  head_device, slice_name: str) -> Dict:
    baseline_heads.eval()
    candidate_heads.eval()
    baseline_rows = []
    candidate_rows = []
    with torch.no_grad():
        for index, record in enumerate(records):
            feature, img_meta, gt_boxes, gt_labels, original, cached = (
                labeller.prepare_record(
                    dino, record, args, dino_device, head_device))
            shared = dict(
                role='target_dev_diagnosis_only', split=record['split'],
                seq=record['seq'], frame=int(record['frame']),
                feature_cache_hit=bool(cached))
            baseline_rows.append(dict(
                shared, **evaluate_one(
                    baseline_heads, feature, img_meta, original, args)))
            candidate_rows.append(dict(
                shared, **evaluate_one(
                    candidate_heads, feature, img_meta, original, args)))
            if (index + 1) % 10 == 0 or index + 1 == len(records):
                print('[s7-target] slice={} {}/{} baseline={} candidate={}'
                      .format(
                          slice_name, index + 1, len(records),
                          sum(row['metrics']['top1_hit']
                              for row in baseline_rows),
                          sum(row['metrics']['top1_hit']
                              for row in candidate_rows)))
            del feature, gt_boxes, gt_labels
    return dict(baseline_rows=baseline_rows, candidate_rows=candidate_rows)


def compare_slice(name: str, baseline_rows: Sequence[Dict],
                  candidate_rows: Sequence[Dict]) -> Dict:
    baseline_summary = labeller.summarize_rows(baseline_rows)
    candidate_summary = labeller.summarize_rows(candidate_rows)
    baseline_hits = {
        labeller.source_frame_key(row): bool(row['metrics']['top1_hit'])
        for row in baseline_rows}
    candidate_hits = {
        labeller.source_frame_key(row): bool(row['metrics']['top1_hit'])
        for row in candidate_rows}
    gained = sorted(key for key in baseline_hits
                    if not baseline_hits[key] and candidate_hits[key])
    lost = sorted(key for key in baseline_hits
                  if baseline_hits[key] and not candidate_hits[key])
    checks = dict(
        top1=(
            int(candidate_summary['top1_hits'])
            > int(baseline_summary['top1_hits'])
            if name == 'seq03_small' else
            int(candidate_summary['top1_hits'])
            >= int(baseline_summary['top1_hits'])),
        mcml=(int(candidate_summary['top1_mcml'])
              <= int(baseline_summary['top1_mcml'])))
    return dict(
        baseline=baseline_summary,
        candidate=candidate_summary,
        delta_top1=(int(candidate_summary['top1_hits'])
                    - int(baseline_summary['top1_hits'])),
        delta_mcml=(int(candidate_summary['top1_mcml'])
                    - int(baseline_summary['top1_mcml'])),
        gained_frame_keys=gained, lost_frame_keys=lost,
        checks=checks, passed=all(checks.values()),
        baseline_rows=list(baseline_rows),
        candidate_rows=list(candidate_rows))


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
        allow_s7_base_initialization=True)
    labeller.validate_checkpoint(candidate_payload, in_channels, args)
    if int(candidate_payload.get('epoch', -1)) != int(args.source_epoch):
        raise RuntimeError(
            'Candidate checkpoint epoch {} does not match source evidence {}'
            .format(candidate_payload.get('epoch'), args.source_epoch))
    if candidate_payload.get('s7_inference_enabled') is not True:
        raise RuntimeError('Candidate checkpoint has S7 inference disabled')

    baseline_heads = labeller.FrozenDinoRotatedHeads(
        in_channels, args).to(head_device)
    candidate_heads = labeller.FrozenDinoRotatedHeads(
        in_channels, args).to(head_device)
    labeller.load_heads_checkpoint_state(
        baseline_heads, baseline_payload, allow_s7_base_initialization=True)
    labeller.load_heads_checkpoint_state(candidate_heads, candidate_payload)
    baseline_versions = common.module_parameter_versions(baseline_heads)
    candidate_versions = common.module_parameter_versions(candidate_heads)

    slices = {}
    for spec in args.parsed_target_slices:
        records = coverage.discover_target_records(args.data_root, spec)
        paired = evaluate_pair(
            dino, baseline_heads, candidate_heads, records, args,
            dino_device, head_device, spec['name'])
        slices[spec['name']] = compare_slice(
            spec['name'], paired['baseline_rows'], paired['candidate_rows'])

    dino_unchanged = dino_versions == common.module_parameter_versions(dino)
    baseline_unchanged = (
        baseline_versions == common.module_parameter_versions(baseline_heads))
    candidate_unchanged = (
        candidate_versions == common.module_parameter_versions(candidate_heads))
    if not (dino_unchanged and baseline_unchanged and candidate_unchanged):
        raise RuntimeError('Read-only parameter invariant failed')
    passed = all(result['passed'] for result in slices.values())
    decision = ('S7_RELAXED_TARGET_DEV_DIAGNOSTIC_PASS'
                if passed else 'S7_RELAXED_TARGET_DEV_DIAGNOSTIC_FAIL')
    payload = dict(
        audit=AUDIT_NAME, protocol_version=PROTOCOL_VERSION,
        decision=decision,
        eligible_for_deployment=False,
        eligible_for_final_test=False,
        eligible_for_next_stage=bool(passed),
        source_gate=dict(
            policy=RELAXED_SOURCE_GATE,
            result_json=os.path.abspath(args.source_result_json),
            result=args.relaxed_source_gate),
        checkpoints=dict(
            baseline=os.path.abspath(args.baseline_checkpoint),
            candidate=os.path.abspath(args.candidate_checkpoint),
            candidate_epoch=int(args.source_epoch)),
        protocol=dict(
            target_role='target_dev_diagnosis_only',
            target_used_for_checkpoint_selection=False,
            target_used_for_parameter_tuning=False,
            target_slices=args.parsed_target_slices,
            pass_rule=(
                'far_and_dark_top1_mcml_nonregression_and_'
                'seq03_small_strict_top1_improvement_mcml_nonregression')),
        isolation=dict(
            optimizer_steps=0, dino_parameters_unchanged=dino_unchanged,
            baseline_parameters_unchanged=baseline_unchanged,
            candidate_parameters_unchanged=candidate_unchanged),
        slices=slices)
    replacements = common.write_json_atomic(args.out_json, payload)
    for name in ('seq02_far', 'seq02_dark', 'seq03_small'):
        result = slices[name]
        print('[s7-result] {} baseline={}/{} candidate={}/{} mcml={}->{} '
              'gained={} lost={} passed={}'.format(
                  name, result['baseline']['top1_hits'],
                  result['baseline']['frame_count'],
                  result['candidate']['top1_hits'],
                  result['candidate']['frame_count'],
                  result['baseline']['top1_mcml'],
                  result['candidate']['top1_mcml'],
                  len(result['gained_frame_keys']),
                  len(result['lost_frame_keys']), result['passed']))
    print('[dino-labeller] {}'.format(decision))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
