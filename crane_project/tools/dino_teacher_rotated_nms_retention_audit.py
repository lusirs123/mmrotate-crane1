#!/usr/bin/env python3
"""Source-only rotated-NMS retention selection and target diagnosis.

This is a read-only companion to the RPN-to-ROI attrition audit.  It evaluates
predeclared hard-NMS IoU thresholds on official source validation, selects a
threshold using source metrics only, and then diagnoses the three fixed target
slices with that one selected threshold.  No optimizer step or checkpoint
write is allowed.
"""

import argparse
import json
import os
from typing import Dict, Sequence

import torch

from crane_project.tools import dino_teacher_common as common
from crane_project.tools import (
    dino_teacher_rpn_roi_attrition_latency_audit as audit)
from crane_project.tools import (
    dino_teacher_token_scale_rpn_coverage_audit as coverage)


AUDIT_NAME = 'DINO Rotated-NMS Retention Audit V1'
DEFAULT_NMS_IOU_CANDIDATES = (0.1, 0.2, 0.3, 0.5)


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument(
        '--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument(
        '--source-rpn-datasets', nargs='+', default=['val:val'])
    parser.add_argument(
        '--target-slice', action='append', dest='target_slices')
    parser.add_argument('--coverage-audit-json', required=True)
    parser.add_argument('--baseline-attrition-json', required=True)
    parser.add_argument('--labeller-checkpoint', required=True)
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
    parser.add_argument(
        '--recall-ks', type=int, nargs='+',
        default=[20, 100, 1000, 2000])
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--nms-iou-candidates', type=float, nargs='+',
                        default=list(DEFAULT_NMS_IOU_CANDIDATES))
    parser.add_argument('--feature-cache-dir', required=True)
    parser.add_argument('--target-feature-mode',
                        choices=['fresh_fp32', 'cache_fp16'],
                        default='fresh_fp32')
    parser.add_argument('--reconstruction-check-count', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def validate_args(args):
    args.source_selection_json = None
    args.expected_source_retention_rate = 0.985
    args.source_rpn_limit = 0
    args.source_fresh_latency_samples = 0
    args.latency_warmup = 1
    audit.protocol_args(args)
    audit.validate_args(args)
    if not os.path.isfile(args.baseline_attrition_json):
        raise ValueError('Baseline attrition JSON does not exist')
    candidates = sorted(set(
        float(value) for value in args.nms_iou_candidates))
    if any(value <= 0.0 or value > 1.0 for value in candidates):
        raise ValueError('NMS IoU candidates must be in (0, 1]')
    args.nms_iou_candidates = candidates


def frame_hits(rows: Sequence[Dict]) -> Dict[str, bool]:
    return {
        '{}|{}|{}'.format(row['split'], row['seq'], int(row['frame'])):
        bool(any(obj['post_valid_content']['top1_hit']
                 for obj in row['objects']))
        for row in rows}


def retention_against_baseline(baseline_rows: Sequence[Dict],
                               candidate_rows: Sequence[Dict]) -> Dict:
    baseline = frame_hits(baseline_rows)
    candidate = frame_hits(candidate_rows)
    keys = sorted(set(baseline) | set(candidate))
    old_correct = [key for key in keys if baseline.get(key, False)]
    retained = [key for key in old_correct if candidate.get(key, False)]
    return dict(
        baseline_correct_count=len(old_correct),
        retained_correct_count=len(retained),
        lost_correct_count=len(old_correct) - len(retained),
        exact_retention_rate=(
            1.0 if not old_correct else len(retained) / len(old_correct)))


def mean_detection_count(rows: Sequence[Dict]) -> float:
    counts = []
    for row in rows:
        if row.get('counts', {}).get('post_nms') is not None:
            counts.append(int(row['counts']['post_nms']))
        elif row['objects']:
            counts.append(int(row['objects'][0]['post_nms'][
                'detection_count']))
        else:
            counts.append(0)
    return 0.0 if not counts else float(sum(counts) / len(counts))


def hit_count(rows: Sequence[Dict]) -> int:
    return int(sum(frame_hits(rows).values()))


def candidate_summary(rows: Sequence[Dict], baseline_rows: Sequence[Dict],
                      threshold: float) -> Dict:
    summary = audit.summarize_attrition(rows)
    small = summary['source_token_bins']['source_small']
    retention = retention_against_baseline(baseline_rows, rows)
    return dict(
        nms_iou_thr=float(threshold),
        summary=summary,
        source_small=dict(
            top1_hits=small['final_top1_hits'],
            frame_count=small['frame_count'],
            post_nms_recall_at_20=small['post_nms_recall_at_20'],
            post_nms_recall_at_100=small['post_nms_recall_at_100']),
        source_retention=retention,
        mean_post_nms_detection_count=mean_detection_count(rows))


def gate_candidate(candidate: Dict, baseline: Dict) -> Dict:
    retention = candidate['source_retention']
    source = candidate['summary']
    base_source = baseline['summary']
    small = candidate['source_small']
    base_small = baseline['source_small']
    checks = dict(
        exact_source_retention=(
            retention['exact_retention_rate'] >= 1.0),
        full_top1_nonregression=(
            source['final_top1_hits'] >= base_source['final_top1_hits']),
        full_post_nms_r20_nonregression=(
            source['post_nms_recall_at_20'] >=
            base_source['post_nms_recall_at_20']),
        small_post_nms_r20_nonregression=(
            small['post_nms_recall_at_20'] >=
            base_small['post_nms_recall_at_20']))
    return dict(checks=checks, passed=bool(all(checks.values())))


def select_candidate(candidates: Sequence[Dict], baseline: Dict) -> Dict:
    evaluated = []
    for candidate in candidates:
        row = dict(candidate)
        row['gate'] = gate_candidate(row, baseline)
        evaluated.append(row)
    eligible = [row for row in evaluated if row['gate']['passed']]
    if not eligible:
        selected = baseline
        decision = 'SOURCE_ONLY_KEEP_FORMAL_NMS'
    else:
        selected = max(
            eligible,
            key=lambda row: (
                row['source_small']['post_nms_recall_at_20'],
                row['source_small']['post_nms_recall_at_100'],
                row['summary']['final_top1_hits'],
                -row['mean_post_nms_detection_count'],
                -row['nms_iou_thr']))
        decision = 'SOURCE_ONLY_NMS_POLICY_SELECTED'
    return dict(decision=decision, selected=selected,
                candidates=evaluated,
                eligible_thresholds=[row['nms_iou_thr'] for row in eligible])


def set_nms_iou_threshold(heads, value: float):
    nms = heads.roi_head.test_cfg.nms
    if hasattr(nms, 'iou_thr'):
        nms.iou_thr = float(value)
    elif hasattr(nms, 'iou_threshold'):
        nms.iou_threshold = float(value)
    else:
        raise RuntimeError('ROI NMS config lacks an IoU threshold')


def load_baseline(path: str, args) -> Dict:
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if payload.get('audit') != audit.AUDIT_NAME:
        raise RuntimeError('Baseline is not the RPN-to-ROI attrition audit')
    if payload.get('labeller_checkpoint_sha256') != common.file_sha256(
            args.labeller_checkpoint):
        raise RuntimeError('Baseline labeller checkpoint does not match')
    if payload.get('dinov2_checkpoint_sha256') != common.file_sha256(
            args.dinov2_checkpoint):
        raise RuntimeError('Baseline DINO checkpoint does not match')
    isolation = payload.get('isolation', {})
    target_readonly = isolation.get(
        'target_labels_used_for_evaluation_only')
    if (int(isolation.get('optimizer_steps', -1)) != 0
            or target_readonly is not True):
        raise RuntimeError('Baseline isolation contract is invalid')
    required = {'seq02_far', 'seq02_dark', 'seq03_small'}
    if set(payload.get('target_diagnoses', {})) != required:
        raise RuntimeError('Baseline target slices are not the fixed protocol')
    rows = payload['source_roi_control']['rows']
    return dict(path=os.path.abspath(path), sha256=common.file_sha256(path),
                payload=payload, rows=rows)


def main():
    args = parse_args()
    validate_args(args)
    baseline = load_baseline(args.baseline_attrition_json, args)
    coverage_payload, boundaries = audit.load_coverage_contract(
        args.coverage_audit_json, args)
    source_records = coverage.discover_dataset_records(
        args.data_root, args.source_rpn_datasets)
    target_groups = {
        spec['name']: coverage.discover_target_records(args.data_root, spec)
        for spec in args.parsed_target_slices}
    coverage.assert_disjoint(source_records, target_groups)
    baseline_keys = set(frame_hits(baseline['rows']))
    source_keys = set(frame_hits([
        dict(split=record['split'], seq=record['seq'],
             frame=record['frame'], objects=[])
        for record in source_records]))
    if baseline_keys != source_keys:
        raise RuntimeError(
            'Baseline source rows do not match current source val')

    dino_devices = [torch.device('cuda:{}'.format(gpu))
                    for gpu in args.dino_gpus]
    head_device = torch.device('cuda:{}'.format(args.head_gpu))
    dino, heads = audit.far_audit.load_frozen_labeller(
        args, dino_devices, head_device)
    dino_versions = common.module_parameter_versions(dino)
    head_versions = common.module_parameter_versions(heads)
    base_threshold = float(heads.roi_head.test_cfg.nms.iou_thr)
    if not any(abs(value - base_threshold) <= 1e-8
               for value in args.nms_iou_candidates):
        raise ValueError('NMS candidates must include the formal threshold')
    audit.warmup(
        dino, heads, source_records[0], args, dino_devices, head_device)

    baseline_source_summary = audit.summarize_attrition(baseline['rows'])
    baseline_source_hits = hit_count(baseline['rows'])
    baseline_small = baseline_source_summary['source_token_bins'][
        'source_small']
    baseline_source = dict(
        nms_iou_thr=base_threshold,
        summary=baseline_source_summary,
        source_small=dict(
            top1_hits=baseline_small['final_top1_hits'],
            frame_count=baseline_small['frame_count'],
            post_nms_recall_at_20=baseline_small[
                'post_nms_recall_at_20'],
            post_nms_recall_at_100=baseline_small[
                'post_nms_recall_at_100']),
        source_retention=dict(
            baseline_correct_count=baseline_source_hits,
            retained_correct_count=baseline_source_hits,
            lost_correct_count=0, exact_retention_rate=1.0),
        mean_post_nms_detection_count=mean_detection_count(
            baseline['rows']))

    candidates = []
    source_rows_by_threshold = {}
    for threshold in args.nms_iou_candidates:
        set_nms_iou_threshold(heads, threshold)
        rows, _memory = audit.evaluate_records(
            dino, heads, source_records, args, dino_devices, head_device,
            'source_validation_nms_control', boundaries,
            force_fresh=False, fresh_latency_samples=0)
        source_rows_by_threshold[threshold] = rows
        candidate = candidate_summary(rows, baseline['rows'], threshold)
        candidates.append(candidate)
        print('[source-nms] iou={:.3f} top1={}/{} small_r20={:.4f}'.format(
            threshold, candidate['summary']['final_top1_hits'],
            candidate['summary']['frame_count'],
            candidate['source_small']['post_nms_recall_at_20']))
    selection = select_candidate(candidates, baseline_source)
    selected_threshold = float(selection['selected']['nms_iou_thr'])
    set_nms_iou_threshold(heads, selected_threshold)

    targets = {}
    for name, records in target_groups.items():
        rows, memory = audit.evaluate_records(
            dino, heads, records, args, dino_devices, head_device,
            'target_diagnosis_only_nms_selected', boundaries,
            force_fresh=args.target_feature_mode == 'fresh_fp32',
            fresh_latency_samples=0)
        summary = audit.summarize_attrition(rows)
        old_summary = baseline['payload']['target_diagnoses'][name][
            'summary']
        old_rows = baseline['payload']['target_diagnoses'][name]['rows']
        targets[name] = dict(
            diagnosis=audit.diagnose(summary),
            specification=next(
                row for row in args.parsed_target_slices
                if row['name'] == name),
            baseline=dict(
                top1_hits=hit_count(old_rows),
                frame_count=old_summary['frame_count'],
                final_top1_recall=old_summary['final_top1_recall'],
                post_nms_recall=old_summary['post_nms_recall']),
            selected=dict(
                top1_hits=summary['final_top1_hits'],
                frame_count=summary['frame_count'],
                final_top1_recall=summary['final_top1_recall'],
                final_top1_mcml=summary['final_top1_mcml'],
                post_nms_recall=summary['post_nms_recall'],
                post_nms_recall_at_20=summary['post_nms_recall_at_20'],
                nms_suppression_statuses=summary[
                    'nms_suppression_statuses']),
            summary=summary, rows=rows, peak_memory_mib=memory)
        print(
            '[target-nms] {} thr={:.3f} top1={}/{} mcml={} '
            'nms_r20={:.4f}'.format(
                name, selected_threshold, summary['final_top1_hits'],
                summary['frame_count'], summary['final_top1_mcml'],
                summary['post_nms_recall_at_20']))

    if (dino_versions != common.module_parameter_versions(dino)
            or head_versions != common.module_parameter_versions(heads)):
        raise RuntimeError('Read-only parameter invariant failed')
    payload = dict(
        audit=AUDIT_NAME, protocol_version=1,
        baseline_attrition_json=baseline['path'],
        baseline_attrition_sha256=baseline['sha256'],
        coverage_audit_json=os.path.abspath(args.coverage_audit_json),
        coverage_audit_sha256=common.file_sha256(
            args.coverage_audit_json),
        labeller_checkpoint=os.path.abspath(args.labeller_checkpoint),
        labeller_checkpoint_sha256=common.file_sha256(
            args.labeller_checkpoint),
        dinov2_checkpoint=os.path.abspath(args.dinov2_checkpoint),
        dinov2_checkpoint_sha256=common.file_sha256(args.dinov2_checkpoint),
        protocol=dict(
            source_role='source_validation_only',
            target_role='diagnosis_only',
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            source_defined_token_bins=boundaries,
            nms_iou_candidates=args.nms_iou_candidates,
            selected_nms_iou=selected_threshold,
            score_thr=float(heads.roi_head.test_cfg.score_thr),
            riou_thr=float(args.riou_thr)),
        isolation=dict(
            optimizer_steps=0, checkpoint_writes=0,
            dino_frozen=True, dino_parameters_unchanged=True,
            labeller_heads_frozen=True,
            labeller_parameters_unchanged=True,
            target_labels_used_for_evaluation_only=True),
        selection=selection, target_diagnoses=targets,
        source_rows=(
            source_rows_by_threshold.get(selected_threshold)
            if selected_threshold in source_rows_by_threshold else None))
    replacements = common.write_json_atomic(args.out_json, payload)
    print('[source-nms] decision={} selected_iou={:.3f}'.format(
        selection['decision'], selected_threshold))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
