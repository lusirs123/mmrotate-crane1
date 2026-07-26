#!/usr/bin/env python3
"""Evaluate the frozen-DINO scoped composition on the complete test set.

BrightAug owns every frame outside an externally declared scope.  Inside the
scope, the frozen DINO labeller is the primary one-target expert and falls
back to BrightAug only when it produces no detection.  This tool is read-only:
it never trains, selects checkpoints, fuses scores, or reads target labels for
scope decisions.  Labels are used only by the offline report.

The target-dev-derived scope used during method development is accepted only
with ``--confirm-diagnosis-scope`` and is labelled diagnosis-only in the JSON.
An unbiased final-test scope must declare ``target_label_derived=false`` and
``eligible_for_final_test=true`` in its manifest.
"""

import argparse
import hashlib
import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

from crane_project.tools import (
    dino_teacher_baseline_first_rescue_audit as rescue,
    dino_teacher_rotated_labeller as labeller,
    frozen_p3_feature_alignment_audit as alignment,
    frozen_p3_objectness_transfer_probe as transfer,
)
from crane_project.tools.ckpt_sweep import run_offline_eval
from crane_project.tools.dino_teacher_source_roi_head_probe import (
    write_json_atomic,
)


AUDIT_NAME = 'Frozen DINO Scoped Complete Test Evaluation V1'
PROTOCOL_VERSION = 1
EXPECTED_SEQUENCE_COUNTS = {
    'real_seq02': 220,
    'real_seq03': 200,
    'sim_seq09': 572,
}


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--baseline-gpu', type=int, default=0)
    parser.add_argument(
        '--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--test-split', default='test')
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
    parser.add_argument('--feature-cache-dir', required=True)
    parser.add_argument('--scope-manifest', required=True)
    parser.add_argument(
        '--confirm-diagnosis-scope', action='store_true',
        help='Allow a target-derived scope, but mark output diagnosis-only.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_args(args):
    if args.seed != 0:
        raise ValueError('The complete-test protocol requires --seed 0')
    if args.test_split != 'test':
        raise ValueError('Complete evaluation is fixed to the test split')
    if not args.dino_gpus or len(set(args.dino_gpus)) != len(args.dino_gpus):
        raise ValueError('DINO GPU ids must be non-empty and unique')
    if args.head_gpu in args.dino_gpus:
        raise ValueError('Head GPU must be separate from DINO GPUs')
    positive = (
        args.patch_size, args.rpn_feat_channels, args.roi_fc_channels,
        args.roi_samples, args.proposal_count, args.max_detections,
        args.dino_height, args.dino_max_long_side)
    if any(int(value) <= 0 for value in positive):
        raise ValueError('Architecture and image sizes must be positive')
    required = (
        args.baseline_config, args.baseline_checkpoint,
        args.labeller_checkpoint, args.dinov2_checkpoint,
        args.scope_manifest)
    for path in required:
        if not os.path.isfile(path):
            raise ValueError('Required file does not exist: {}'.format(path))
    if os.path.exists(args.out_json):
        raise ValueError('Refusing to overwrite {}'.format(args.out_json))
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir):
        raise ValueError(
            'Refusing to reuse non-empty out-dir: {}'.format(args.out_dir))


def all_test_records(args) -> List[Dict]:
    records = transfer.discover_labeled_records(
        args.data_root, args.test_split, 0)
    records.sort(key=lambda row: (row['seq'], int(row['frame'])))
    if not records:
        raise RuntimeError('No complete-test records were found')
    counts = Counter(row['seq'] for row in records)
    if dict(counts) != EXPECTED_SEQUENCE_COUNTS:
        raise RuntimeError(
            'Unexpected test sequence counts: expected={} got={}'.format(
                EXPECTED_SEQUENCE_COUNTS, dict(counts)))
    keys = [(row['seq'], int(row['frame'])) for row in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError('Duplicate test sequence/frame records')
    return records


def load_final_scope(args, records: Sequence[Dict]) -> Dict:
    scope = rescue.load_scope_manifest(args.scope_manifest, records)
    if scope is None:
        raise RuntimeError('Complete test requires an explicit scope manifest')
    diagnosis_only = bool(scope['target_label_derived']
                          or not scope['eligible_for_final_test'])
    if diagnosis_only and not args.confirm_diagnosis_scope:
        raise ValueError(
            'Scope is target-derived/ineligible; pass '
            '--confirm-diagnosis-scope for a diagnosis-only run')
    if (scope['target_label_derived']
            and scope['eligible_for_final_test']):
        raise ValueError(
            'A target-derived scope cannot be marked final-test eligible')
    scope['diagnosis_only'] = diagnosis_only
    return scope


def empty_dino_row(record: Dict) -> Dict:
    return dict(
        role='scope_disabled_no_dino_inference', split=record['split'],
        seq=record['seq'], frame=int(record['frame']), detections=[])


def by_key(records: Sequence[Dict]) -> Dict[Tuple[str, int], Dict]:
    return {(row['seq'], int(row['frame'])): row for row in records}


def selected_dino_rows(dino, heads, enabled_records, all_records, args,
                       dino_device, head_device) -> List[Dict]:
    evaluated = {}
    if enabled_records:
        rows = labeller.evaluate_records(
            dino, heads, enabled_records, args, dino_device, head_device,
            role='target_full_test_readonly')
        evaluated = by_key(rows)
    return [evaluated.get(
        (record['seq'], int(record['frame'])), empty_dino_row(record))
        for record in all_records]


def box_to_dota_line(box: Sequence[float]) -> str:
    if len(box) < 6:
        raise ValueError('Expected [cx,cy,w,h,theta,score]')
    cx, cy, width, height, theta, score = [float(value) for value in box[:6]]
    points = cv2.boxPoints(((cx, cy), (width, height),
                            float(np.degrees(theta))))
    coords = ' '.join(
        '{:.2f}'.format(float(value)) for value in points.ravel())
    return '{} {:.6f}'.format(coords, score)


def write_dota_predictions(rows: Sequence[Dict], policy: str,
                           records: Sequence[Dict], output_dir: str) -> str:
    task_dir = os.path.join(output_dir, 'Task1_grab')
    if os.path.isdir(task_dir) and os.listdir(task_dir):
        raise ValueError('Prediction directory is not empty: {}'.format(
            task_dir))
    os.makedirs(task_dir, exist_ok=True)
    count = 0
    for row, record in zip(rows, records):
        detections = row['policies'][policy]['detections']
        name = os.path.splitext(os.path.basename(record['image']))[0]
        path = os.path.join(task_dir, name + '.txt')
        with open(path, 'w', encoding='utf-8') as handle:
            for detection in detections:
                handle.write(box_to_dota_line(detection) + '\n')
                count += 1
    if len(os.listdir(task_dir)) != len(records):
        raise RuntimeError('DOTA prediction count does not match test count')
    return task_dir


def policy_results(rows: Sequence[Dict],
                   policy: str) -> List[List[np.ndarray]]:
    results = []
    for row in rows:
        detections = np.asarray(
            row['policies'][policy]['detections'], dtype=np.float32)
        if detections.size == 0:
            detections = np.zeros((0, 6), dtype=np.float32)
        results.append([detections.reshape((-1, 6))])
    return results


def write_results_pickle(rows: Sequence[Dict], policy: str,
                         output_path: str):
    results = policy_results(rows, policy)
    with open(output_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_standard_metrics(config_path: str, records: Sequence[Dict],
                              results: Sequence) -> Dict:
    """Run the same CraneDataset evaluation used by tools/test.py."""
    from mmcv import Config
    from mmrotate.datasets import build_dataset

    cfg = Config.fromfile(config_path)
    dataset = build_dataset(cfg.data.test)
    dataset_ids = [str(info.get('img_id', ''))
                   for info in dataset.data_infos]
    record_ids = [os.path.splitext(os.path.basename(row['image']))[0]
                  for row in records]
    if dataset_ids != record_ids:
        raise RuntimeError('Dataset/result ordering mismatch')
    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ('interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'dynamic_intervals', 'metric'):
        eval_kwargs.pop(key, None)
    return dataset.evaluate(results, metric='mAP', **eval_kwargs)


def group_summary(rows: Sequence[Dict], policy: str) -> Dict[str, Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['seq']].append(row)
    return {seq: rescue.summarize_policy(items, policy)
            for seq, items in sorted(grouped.items())}


def metric_delta(baseline: Dict, combined: Dict) -> Dict:
    delta = {}
    numeric = (int, float, np.integer, np.floating)
    for key in sorted(set(baseline) & set(combined)):
        if isinstance(baseline[key], numeric) and isinstance(
                combined[key], numeric):
            delta[key] = float(combined[key]) - float(baseline[key])
    return delta


def paper_log_lines(baseline_metrics: Dict, combined_metrics: Dict,
                    baseline_standard: Dict, combined_standard: Dict,
                    overall: Dict, sequence_summaries: Dict,
                    scope: Dict, records: Sequence[Dict]) -> List[str]:
    enabled = sum(bool(scope['values'].get(
        (row['split'], row['seq'], int(row['frame'])), False))
                   for row in records)
    lines = ['[full-test] frames={} sequences={} scope_enabled={} '
             'scope_disabled={} diagnosis_only={}'.format(
                 len(records), len(set(row['seq'] for row in records)),
                 enabled, len(records) - enabled,
                 scope['diagnosis_only'])]
    for method, metrics in (('BrightAug', baseline_metrics),
                            ('ScopedDINO', combined_metrics)):
        lines.append('[paper-result] method={} {}'.format(
            method, ' '.join('{}={}'.format(key, metrics[key])
                             for key in sorted(metrics))))
    for method, metrics in (('BrightAug', baseline_standard),
                            ('ScopedDINO', combined_standard)):
        lines.append('[paper-standard] method={} {}'.format(
            method, ' '.join('{}={}'.format(key, metrics[key])
                             for key in sorted(metrics))))
    primary = overall['scoped_dino_primary']
    routing = overall['routing_diagnostics']
    lines.append(
        '[paper-routing] top1={}/{} MCML={} deployment_top1={}/{} '
        'dino_selected={} baseline_fallback={} scope_disabled={}'.format(
            primary['top1_hits'], primary['frame_count'],
            primary['top1_mcml'], primary['deployment_top1_hits'],
            primary['frame_count'],
            routing['scoped_primary_dino_selected_count'],
            routing['scoped_primary_baseline_fallback_count'],
            primary['scope_disabled_count']))
    for seq, summary in sorted(sequence_summaries.items()):
        lines.append(
            '[paper-sequence] seq={} top1={}/{} MCML={} '
            'deployment_top1={}/{}'.format(
                seq, summary['top1_hits'], summary['frame_count'],
                summary['top1_mcml'], summary['deployment_top1_hits'],
                summary['frame_count']))
    lines.append(
        '[paper-scope] source={} target_label_derived={} '
        'eligible_for_final_test={}'.format(
            scope['source'], scope['target_label_derived'],
            scope['eligible_for_final_test']))
    return lines


def main():
    args = rescue.protocol_args(parse_args())
    validate_args(args)
    labeller.set_seed(args.seed)
    records = all_test_records(args)
    scope = load_final_scope(args, records)
    scope_by_key = scope['values']
    enabled_records = [
        row for row in records
        if bool(scope_by_key.get((row['split'], row['seq'], int(row['frame'])),
                                 False))]
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    baseline_device = torch.device('cuda:{}'.format(args.baseline_gpu))
    baseline, baseline_cfg, baseline_policy = rescue.load_baseline(
        args.baseline_config, args.baseline_checkpoint, baseline_device)
    baseline_versions = alignment.module_parameter_versions(baseline)
    baseline_rows = rescue.evaluate_baseline(
        baseline, baseline_cfg, records, baseline_device,
        role='complete_test_baseline')
    baseline_unchanged = (
        baseline_versions == alignment.module_parameter_versions(baseline))
    if not baseline_unchanged:
        raise RuntimeError('Baseline parameter invariant failed')
    del baseline
    torch.cuda.empty_cache()

    dino_unchanged = True
    heads_unchanged = True
    dino_sha = file_sha256(args.dinov2_checkpoint)
    dino_rows = [empty_dino_row(record) for record in records]
    if enabled_records:
        dino_devices = [torch.device('cuda:{}'.format(gpu))
                        for gpu in args.dino_gpus]
        head_device = torch.device('cuda:{}'.format(args.head_gpu))
        dino, heads = rescue.load_frozen_labeller(
            args, dino_devices, head_device)
        dino_versions = alignment.module_parameter_versions(dino)
        head_versions = alignment.module_parameter_versions(heads)
        dino_rows = selected_dino_rows(
            dino, heads, enabled_records, records, args, dino_devices[0],
            head_device)
        dino_unchanged = (
            dino_versions == alignment.module_parameter_versions(dino))
        heads_unchanged = (
            head_versions == alignment.module_parameter_versions(heads))
        if not dino_unchanged or not heads_unchanged:
            raise RuntimeError('DINO/head parameter invariant failed')

    combined_rows = rescue.combine_rows(
        baseline_rows, dino_rows, records, scope_by_key)
    overall = rescue.summarize_combination(combined_rows)
    disabled_changed = overall['routing_diagnostics'][
        'scoped_primary_disabled_scope_changed_count']
    if disabled_changed:
        raise RuntimeError(
            'Scope-disabled BrightAug fallback changed on {} frames'.format(
                disabled_changed))

    baseline_pred_dir = write_dota_predictions(
        combined_rows, 'baseline', records,
        os.path.join(args.out_dir, 'baseline'))
    combined_pred_dir = write_dota_predictions(
        combined_rows, 'scoped_dino_primary', records,
        os.path.join(args.out_dir, 'scoped_dino'))
    baseline_pkl = os.path.join(args.out_dir, 'baseline_results.pkl')
    combined_pkl = os.path.join(args.out_dir, 'scoped_dino_results.pkl')
    write_results_pickle(combined_rows, 'baseline', baseline_pkl)
    write_results_pickle(combined_rows, 'scoped_dino_primary', combined_pkl)

    gt_dir = os.path.join(args.data_root, args.test_split, 'annfiles')
    baseline_metrics = run_offline_eval(
        baseline_pred_dir, gt_dir, mode='test', center_thresh=15.0)
    combined_metrics = run_offline_eval(
        combined_pred_dir, gt_dir, mode='test', center_thresh=15.0)
    baseline_standard = evaluate_standard_metrics(
        args.baseline_config, records,
        policy_results(combined_rows, 'baseline'))
    combined_standard = evaluate_standard_metrics(
        args.baseline_config, records,
        policy_results(combined_rows, 'scoped_dino_primary'))

    sequence_summaries = group_summary(
        combined_rows, 'scoped_dino_primary')
    summary_lines = paper_log_lines(
        baseline_metrics, combined_metrics, baseline_standard,
        combined_standard, overall, sequence_summaries, scope, records)
    paper_summary_path = os.path.join(args.out_dir, 'paper_summary.txt')
    with open(paper_summary_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(summary_lines) + '\n')

    payload = dict(
        audit=AUDIT_NAME, protocol_version=PROTOCOL_VERSION,
        baseline_config=os.path.abspath(args.baseline_config),
        baseline_checkpoint=os.path.abspath(args.baseline_checkpoint),
        baseline_checkpoint_sha256=file_sha256(args.baseline_checkpoint),
        labeller_checkpoint=os.path.abspath(args.labeller_checkpoint),
        labeller_checkpoint_sha256=file_sha256(args.labeller_checkpoint),
        dinov2_checkpoint=os.path.abspath(args.dinov2_checkpoint),
        dinov2_checkpoint_sha256=dino_sha,
        protocol=dict(
            test_split=args.test_split,
            routing='scope_enabled_dino_primary_else_exact_brightaug',
            score_fusion=False, checkpoint_selection=False,
            target_labels_used_for_scope=scope['target_label_derived'],
            target_labels_used_for_metrics_only=(
                not scope['target_label_derived']),
            diagnosis_only=scope['diagnosis_only'],
            scope_eligible_for_final_test=scope['eligible_for_final_test']),
        dataset=dict(
            frame_count=len(records),
            sequence_counts=dict(Counter(row['seq'] for row in records)),
            enabled_scope_count=len(enabled_records),
            disabled_scope_count=len(records) - len(enabled_records)),
        scope_gate=dict(
            source=scope['source'],
            target_label_derived=scope['target_label_derived'],
            eligible_for_final_test=scope['eligible_for_final_test'],
            diagnosis_only=scope['diagnosis_only'],
            manifest=os.path.abspath(args.scope_manifest),
            manifest_sha256=file_sha256(args.scope_manifest)),
        isolation=dict(
            optimizer_steps=0, checkpoint_writes=0,
            baseline_frozen=True,
            baseline_parameters_unchanged=baseline_unchanged,
            dino_frozen=True,
            dino_parameters_unchanged=dino_unchanged,
            labeller_heads_frozen=True,
            labeller_parameters_unchanged=heads_unchanged,
            dino_inference_count=len(enabled_records),
            brightaug_only_count=len(records) - len(enabled_records)),
        routing=dict(
            overall=overall,
            by_sequence=sequence_summaries),
        offline_metrics=dict(
            baseline=baseline_metrics,
            scoped_dino=combined_metrics,
            delta=metric_delta(baseline_metrics, combined_metrics)),
        standard_metrics=dict(
            baseline=baseline_standard,
            scoped_dino=combined_standard,
            delta=metric_delta(baseline_standard, combined_standard)),
        artifacts=dict(
            baseline_dota_dir=os.path.abspath(baseline_pred_dir),
            scoped_dino_dota_dir=os.path.abspath(combined_pred_dir),
            baseline_results_pkl=os.path.abspath(baseline_pkl),
            scoped_dino_results_pkl=os.path.abspath(combined_pkl),
            paper_summary=os.path.abspath(paper_summary_path)),
        decision=('COMPLETE_TEST_DIAGNOSIS_ONLY_TARGET_AWARE_SCOPE'
                  if scope['diagnosis_only']
                  else 'COMPLETE_TEST_SCOPED_DINO_EVALUATED'))
    replacements = write_json_atomic(args.out_json, payload)
    for line in summary_lines:
        print(line)
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
