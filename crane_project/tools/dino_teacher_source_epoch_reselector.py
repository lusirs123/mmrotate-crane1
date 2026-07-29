#!/usr/bin/env python3
"""Reselect a trained DINO ROI-classifier epoch using source validation only.

This tool does not run inference and never discovers target data. It consumes
the source-only training report, applies a fixed exact-frame retention gate,
verifies that only ``fc_cls`` changed from the initialized checkpoint, and
writes a new checkpoint with corrected source-selection metadata.
"""

import argparse
import copy
import json
import os
from typing import Dict, Sequence, Tuple

import torch

from crane_project.tools import dino_teacher_common as common
from crane_project.tools import dino_teacher_rotated_labeller as labeller


SELECTOR_NAME = 'Frozen DINO Source-Retained Epoch Reselector V1'
PROTOCOL_VERSION = 1
FC_CLS_KEYS = (
    'roi_head.bbox_head.fc_cls.weight',
    'roi_head.bbox_head.fc_cls.bias',
)


def parse_args():
    parser = argparse.ArgumentParser(description=SELECTOR_NAME)
    parser.add_argument('--train-result-json', required=True)
    parser.add_argument('--initial-checkpoint', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--min-retention-rate', type=float, default=0.985)
    parser.add_argument('--out-checkpoint', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def validate_args(args):
    if not 0.0 < float(args.min_retention_rate) <= 1.0:
        raise ValueError('--min-retention-rate must be in (0, 1]')
    for path in (args.train_result_json, args.initial_checkpoint):
        if not os.path.isfile(path):
            raise ValueError('Required input does not exist: {}'.format(path))
    if not os.path.isdir(args.checkpoint_dir):
        raise ValueError(
            'Checkpoint directory does not exist: {}'.format(
                args.checkpoint_dir))
    for path in (args.out_checkpoint, args.out_json):
        if os.path.exists(path):
            raise ValueError('Refusing to overwrite: {}'.format(path))


def load_source_only_report(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as handle:
        report = json.load(handle)
    isolation = report.get('isolation', {})
    if (report.get('decision') !=
            'SOURCE_ONLY_TRAINING_COMPLETE_TARGET_NOT_READ'):
        raise RuntimeError('Training report is not source-only completion')
    if report.get('target_dev') is not None:
        raise RuntimeError('Training report contains target-dev results')
    required_false = (
        'target_used_for_training', 'target_used_for_checkpoint_selection',
        'target_labels_used_for_evaluation_only')
    if any(isolation.get(key) is not False for key in required_false):
        raise RuntimeError('Training report does not prove target isolation')
    if (isolation.get('dino_frozen') is not True
            or isolation.get('dino_parameters_unchanged') is not True
            or isolation.get('frozen_head_parameters_unchanged') is not True):
        raise RuntimeError('Frozen-model invariants failed in training report')
    if isolation.get('train_components') != 'roi_cls_pairwise':
        raise RuntimeError('Report is not from roi_cls_pairwise training')
    trainable = tuple(isolation.get('trainable_parameter_names', ()))
    if set(trainable) != set(FC_CLS_KEYS):
        raise RuntimeError(
            'Training report authorized parameters beyond fc_cls')
    return report


def retention_rate(retention: Dict) -> float:
    baseline = int(retention.get('baseline_correct_count', 0))
    retained = int(retention.get('retained_correct_count', -1))
    if baseline <= 0 or retained < 0 or retained > baseline:
        raise RuntimeError('Invalid exact source-retention counts')
    return float(retained / baseline)


def candidate_gate(history_row: Dict, baseline_full: Dict,
                   baseline_small: Dict,
                   min_retention_rate: float) -> Dict:
    full = history_row.get('source_val')
    small = history_row.get('source_small_val')
    retention = history_row.get('source_exact_retention')
    if not all(isinstance(value, dict)
               for value in (full, small, retention)):
        raise RuntimeError('History row lacks source validation evidence')
    rate = retention_rate(retention)
    checks = dict(
        selection_epoch=bool(history_row.get('selection_eligible')),
        checkpoint_saved=bool(history_row.get('checkpoint_saved')),
        exact_retention_rate=bool(rate >= float(min_retention_rate)),
        full_top1_strictly_higher=bool(
            int(full['top1_hits']) > int(baseline_full['top1_hits'])),
        small_top1_strictly_higher=bool(
            int(small['top1_hits']) > int(baseline_small['top1_hits'])),
        full_mcml_not_higher=bool(
            int(full['top1_mcml']) <= int(baseline_full['top1_mcml'])),
        small_mcml_not_higher=bool(
            int(small['top1_mcml']) <= int(baseline_small['top1_mcml'])))
    return dict(
        passed=bool(all(checks.values())), checks=checks,
        exact_retention_rate=rate,
        lost_correct_count=int(retention['lost_correct_count']),
        gained_correct_count=int(retention['gained_correct_count']))


def candidate_key(history_row: Dict) -> Tuple:
    small = history_row['source_small_val']
    full = history_row['source_val']
    return (
        int(small['top1_hits']), -int(small['top1_mcml']),
        int(full['top1_hits']), -int(full['top1_mcml']),
        -int(history_row['epoch']))


def select_source_epoch(
        report: Dict, min_retention_rate: float
        ) -> Tuple[Dict, Sequence[Dict]]:
    source = report.get('source', {})
    baseline_full = source.get('baseline_validation_summary')
    baseline_small = source.get('baseline_small_validation_summary')
    history = source.get('history')
    if not isinstance(baseline_full, dict) or not isinstance(
            baseline_small, dict) or not isinstance(history, list):
        raise RuntimeError('Training report lacks source selection evidence')
    candidates = []
    for row in history:
        gate = candidate_gate(
            row, baseline_full, baseline_small, min_retention_rate)
        candidates.append(dict(
            epoch=int(row['epoch']), gate=gate,
            source_full_summary=row['source_val'],
            source_small_summary=row['source_small_val'],
            source_exact_retention=row['source_exact_retention'],
            selected=False))
    eligible = [row for row, candidate in zip(history, candidates)
                if candidate['gate']['passed']]
    if not eligible:
        raise RuntimeError('No epoch passes the fixed source-retention gate')
    selected_history = max(eligible, key=candidate_key)
    selected_epoch = int(selected_history['epoch'])
    for candidate in candidates:
        candidate['selected'] = bool(candidate['epoch'] == selected_epoch)
    return selected_history, candidates


def validate_classifier_only_change(initial_payload: Dict,
                                    candidate_payload: Dict) -> Dict:
    for name, payload in (
            ('initial', initial_payload), ('candidate', candidate_payload)):
        if payload.get('source_only') is not True:
            raise RuntimeError(
                '{} checkpoint is not source-only'.format(name))
        if payload.get('frozen_dinov2') is not True:
            raise RuntimeError(
                '{} checkpoint did not freeze DINO'.format(name))
        if not isinstance(payload.get('heads_state_dict'), dict):
            raise RuntimeError('{} checkpoint lacks head state'.format(name))
    initial = initial_payload['heads_state_dict']
    candidate = candidate_payload['heads_state_dict']
    if set(initial) != set(candidate):
        raise RuntimeError('Initial and candidate head keys differ')
    changed = []
    for key in initial:
        if (initial[key].shape != candidate[key].shape
                or initial[key].dtype != candidate[key].dtype):
            raise RuntimeError('Head tensor contract changed: {}'.format(key))
        if not torch.equal(initial[key], candidate[key]):
            changed.append(key)
    outside = [key for key in changed if key not in FC_CLS_KEYS]
    if outside:
        raise RuntimeError(
            'Frozen head tensor changed: {}'.format(outside[0]))
    if not changed:
        raise RuntimeError(
            'Candidate classifier is identical to initialization')
    return dict(
        changed_parameter_names=sorted(changed),
        changed_parameter_count=int(sum(
            candidate[key].numel() for key in changed)),
        frozen_tensor_count=int(len(candidate) - len(changed)),
        frozen_tensors_bit_identical=True)


def selected_checkpoint_payload(candidate_payload: Dict,
                                selected_history: Dict,
                                report: Dict, args,
                                candidate_checkpoint: str) -> Dict:
    payload = copy.deepcopy(candidate_payload)
    epoch = int(selected_history['epoch'])
    payload['best_epoch'] = epoch
    payload['best_source_val_summary'] = selected_history['source_val']
    payload['best_source_small_val_summary'] = (
        selected_history['source_small_val'])
    payload['optimizer_state_dict'] = None
    payload['scheduler_state_dict'] = None
    payload['source_only_epoch_reselection'] = dict(
        selector=SELECTOR_NAME, protocol_version=PROTOCOL_VERSION,
        target_data_read=False,
        min_exact_retention_rate=float(args.min_retention_rate),
        selected_epoch=epoch,
        selected_candidate_checkpoint=os.path.abspath(candidate_checkpoint),
        selected_candidate_checkpoint_sha256=common.file_sha256(
            candidate_checkpoint),
        training_report=os.path.abspath(args.train_result_json),
        training_report_sha256=common.file_sha256(args.train_result_json),
        initialization_checkpoint=os.path.abspath(args.initial_checkpoint),
        initialization_checkpoint_sha256=common.file_sha256(
            args.initial_checkpoint),
        source_full_summary=selected_history['source_val'],
        source_small_summary=selected_history['source_small_val'],
        source_exact_retention=selected_history['source_exact_retention'],
        original_training_best_epoch=int(report['source']['best_epoch']))
    return payload


def main():
    args = parse_args()
    validate_args(args)
    report = load_source_only_report(args.train_result_json)
    recorded_initial = os.path.abspath(
        report['isolation']['initialization_checkpoint'])
    if recorded_initial != os.path.abspath(args.initial_checkpoint):
        raise RuntimeError(
            'Initial checkpoint does not match the training report')
    selected, candidates = select_source_epoch(
        report, args.min_retention_rate)
    epoch = int(selected['epoch'])
    candidate_checkpoint = os.path.join(
        args.checkpoint_dir,
        'labeller_epoch_{:02d}_source_only.pth'.format(epoch))
    if not os.path.isfile(candidate_checkpoint):
        raise RuntimeError(
            'Selected epoch checkpoint does not exist: {}'.format(
                candidate_checkpoint))
    initial_payload = torch.load(args.initial_checkpoint, map_location='cpu')
    candidate_payload = torch.load(candidate_checkpoint, map_location='cpu')
    if int(candidate_payload.get('epoch', -1)) != epoch:
        raise RuntimeError('Candidate checkpoint epoch metadata mismatch')
    invariants = validate_classifier_only_change(
        initial_payload, candidate_payload)
    selected_payload = selected_checkpoint_payload(
        candidate_payload, selected, report, args, candidate_checkpoint)
    labeller.atomic_torch_save(selected_payload, args.out_checkpoint)
    output = dict(
        selector=SELECTOR_NAME, protocol_version=PROTOCOL_VERSION,
        decision='SOURCE_ONLY_EPOCH_SELECTED_TARGET_NOT_READ',
        source_only=True, target_data_read=False,
        training_report=os.path.abspath(args.train_result_json),
        min_exact_retention_rate=float(args.min_retention_rate),
        baseline=dict(
            full=report['source']['baseline_validation_summary'],
            small=report['source']['baseline_small_validation_summary']),
        candidates=candidates,
        selected=dict(
            epoch=epoch,
            source_full_summary=selected['source_val'],
            source_small_summary=selected['source_small_val'],
            source_exact_retention=selected['source_exact_retention'],
            input_checkpoint=os.path.abspath(candidate_checkpoint),
            input_checkpoint_sha256=common.file_sha256(candidate_checkpoint),
            output_checkpoint=os.path.abspath(args.out_checkpoint),
            output_checkpoint_sha256=common.file_sha256(args.out_checkpoint)),
        parameter_invariants=invariants)
    replacements = common.write_json_atomic(args.out_json, output)
    print('[source-reselect] epoch={} retention={:.6f} full={}/{} '
          'small={}/{}'.format(
              epoch,
              retention_rate(selected['source_exact_retention']),
              selected['source_val']['top1_hits'],
              selected['source_val']['frame_count'],
              selected['source_small_val']['top1_hits'],
              selected['source_small_val']['frame_count']))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] checkpoint={} json={}'.format(
        args.out_checkpoint, args.out_json))


if __name__ == '__main__':
    main()
