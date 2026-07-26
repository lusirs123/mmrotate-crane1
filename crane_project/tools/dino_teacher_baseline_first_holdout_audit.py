#!/usr/bin/env python3
"""Fixed holdout regression audit for baseline-first DINO rescue routing.

The detector checkpoints, score threshold, routing policies, and source gate
are fixed for this run.  The named sequences were inspected by earlier DINO
experiments, so this is a regression audit rather than a pristine final-test
estimate.  It evaluates the unchanged BrightAug detector and the frozen
source-trained DINO labeller separately, then reports their fixed combinations.

It never trains, updates parameters, writes checkpoints, or selects a model
from holdout performance.  Holdout labels are used only for final evaluation.
"""

import argparse
import os
from typing import Dict, Sequence

import torch

from crane_project.tools import dino_teacher_baseline_first_rescue_audit as rescue
from crane_project.tools import dino_teacher_frozen_holdout_audit as holdout
from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import dino_teacher_source_roi_head_probe as roi_probe
from crane_project.tools import frozen_p3_feature_alignment_audit as alignment


AUDIT_NAME = 'Frozen DINO Baseline-First Fixed Holdout Regression Audit V1'
PROTOCOL_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--baseline-gpu', type=int, default=0)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-split', default='val')
    parser.add_argument('--source-seq', default='real_seq07')
    parser.add_argument('--source-val-modulus', type=int, default=5)
    parser.add_argument('--holdout-split', default='test')
    parser.add_argument('--holdout-seqs', nargs='+', required=True)
    parser.add_argument('--confirm-fixed-holdout', action='store_true')
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def validate_args(args):
    if not args.confirm_fixed_holdout:
        raise ValueError('Holdout audit requires --confirm-fixed-holdout')
    if args.seed != 0:
        raise ValueError('The frozen holdout protocol requires --seed 0')
    if args.source_split != 'val' or args.source_seq != 'real_seq07':
        raise ValueError('Source control is fixed to val/real_seq07')
    if args.source_val_modulus != 5:
        raise ValueError('Source control modulus is fixed to 5')
    if args.holdout_split != 'test':
        raise ValueError('Holdout split is fixed to test')
    if sorted(args.holdout_seqs) != sorted(holdout.HOLDOUT_SPECS):
        raise ValueError(
            'One-shot audit requires real_seq03 and sim_seq09 together')
    if len(args.holdout_seqs) != len(set(args.holdout_seqs)):
        raise ValueError('Holdout sequence names must be unique')
    if not args.dino_gpus or len(args.dino_gpus) != len(set(args.dino_gpus)):
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
        args.labeller_checkpoint, args.dinov2_checkpoint)
    for path in required:
        if not os.path.isfile(path):
            raise ValueError('Required file does not exist: {}'.format(path))
    if os.path.exists(args.out_json):
        raise ValueError(
            'Refusing to overwrite a completed one-shot holdout result')


def protocol_args(args):
    """Expose only the fixed fields required by shared inference helpers."""
    return rescue.protocol_args(args)


def source_and_holdout_records(args):
    source = holdout.select_records(
        args.data_root, args.source_split, args.source_seq)
    _source_train, source_val = labeller.split_source_records(
        source, args.source_val_modulus)
    holdouts = {
        seq: holdout.select_records(
            args.data_root, args.holdout_split, seq)
        for seq in args.holdout_seqs}
    for seq, records in holdouts.items():
        holdout.validate_complete_holdout(seq, records)
        labeller.assert_training_target_isolation(source, records)
    return source_val, holdouts


def fixed_policy_non_regression_holds(summary: Dict) -> bool:
    """Require every reported policy to preserve or improve the baseline."""
    if not rescue.baseline_preserving_non_regression_holds(summary):
        return False
    return rescue.confident_override_non_regression_holds(summary)


def make_decision(source_summary: Dict,
                  holdout_summaries: Dict[str, Dict]) -> str:
    if not fixed_policy_non_regression_holds(source_summary):
        return 'INVALID_SOURCE_ROUTING_REGRESSION'
    failed = [
        seq for seq, summary in holdout_summaries.items()
        if not fixed_policy_non_regression_holds(summary)]
    if failed:
        return 'CONFIDENT_OVERRIDE_FAILS_FIXED_HOLDOUTS:' + ','.join(
            sorted(failed))
    return 'CONFIDENT_OVERRIDE_PASSES_FIXED_HOLDOUTS'


def evaluate_dino(dino, heads, records_by_role, args,
                  dino_device, head_device):
    return {
        role: labeller.evaluate_records(
            dino, heads, records, args, dino_device, head_device,
            role='source_validation' if role == 'source'
            else 'target_holdout_readonly')
        for role, records in records_by_role.items()}


def main():
    args = protocol_args(parse_args())
    validate_args(args)
    labeller.set_seed(args.seed)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    source_records, holdout_records = source_and_holdout_records(args)
    all_records = dict(source=source_records, **holdout_records)

    baseline_device = torch.device('cuda:{}'.format(args.baseline_gpu))
    baseline, baseline_cfg, baseline_policy = rescue.load_baseline(
        args.baseline_config, args.baseline_checkpoint, baseline_device)
    baseline_versions = alignment.module_parameter_versions(baseline)
    baseline_rows = {
        role: rescue.evaluate_baseline(
            baseline, baseline_cfg, records, baseline_device,
            role='source_validation' if role == 'source'
            else 'target_holdout_readonly')
        for role, records in all_records.items()}
    baseline_unchanged = (
        baseline_versions == alignment.module_parameter_versions(baseline))
    if not baseline_unchanged:
        raise RuntimeError('Frozen baseline parameter invariant failed')
    del baseline
    torch.cuda.empty_cache()

    dino_devices = [torch.device('cuda:{}'.format(gpu))
                    for gpu in args.dino_gpus]
    dino_device = dino_devices[0]
    head_device = torch.device('cuda:{}'.format(args.head_gpu))
    dino, heads = rescue.load_frozen_labeller(
        args, dino_devices, head_device)
    dino_versions = alignment.module_parameter_versions(dino)
    head_versions = alignment.module_parameter_versions(heads)
    dino_rows = evaluate_dino(
        dino, heads, all_records, args, dino_device, head_device)
    dino_unchanged = (
        dino_versions == alignment.module_parameter_versions(dino))
    heads_unchanged = (
        head_versions == alignment.module_parameter_versions(heads))
    if not dino_unchanged or not heads_unchanged:
        raise RuntimeError('Frozen DINO/labeller parameter invariant failed')

    combined = {
        role: rescue.combine_rows(
            baseline_rows[role], dino_rows[role], records)
        for role, records in all_records.items()}
    summaries = {
        role: rescue.summarize_combination(rows)
        for role, rows in combined.items()}
    source_summary = summaries.pop('source')
    decision = make_decision(source_summary, summaries)

    holdout_payload = {
        seq: dict(summary=summaries[seq], rows=combined[seq])
        for seq in args.holdout_seqs}
    payload = dict(
        audit=AUDIT_NAME, protocol_version=PROTOCOL_VERSION,
        baseline_config=os.path.abspath(args.baseline_config),
        baseline_checkpoint=os.path.abspath(args.baseline_checkpoint),
        baseline_checkpoint_sha256=rescue.file_sha256(
            args.baseline_checkpoint),
        labeller_checkpoint=os.path.abspath(args.labeller_checkpoint),
        labeller_checkpoint_sha256=rescue.file_sha256(
            args.labeller_checkpoint),
        dinov2_checkpoint=os.path.abspath(args.dinov2_checkpoint),
        dinov2_checkpoint_sha256=rescue.file_sha256(
            args.dinov2_checkpoint),
        protocol=dict(
            fixed_holdout_regression_audit=True,
            pristine_unseen_final_test=False,
            prior_holdout_observation_acknowledged=True,
            interpretation=(
                'regression_check_only_not_an_unbiased_generalization_claim'),
            target_dev_informed_prior_method_development=True,
            holdouts=list(args.holdout_seqs),
            threshold=float(rescue.DINO_DEPLOYMENT_SCORE_THR),
            threshold_fixed_before_holdout=True,
            checkpoint_selection='source_validation_only',
            baseline_policy=baseline_policy,
            policies=['baseline', 'strict', 'ranked',
                      'confident_override'],
            confident_override_policy=dict(
                baseline_silent='ranked_dino_top1',
                baseline_active=(
                    'dino_top1_only_if_dino_score_at_least_0.05_else_baseline'),
                cross_model_score_comparison=False),
            ranked_policy_requires_target_present=True,
            no_object_false_positive_gate_available=False,
            holdout_eligible_for_training=False,
            holdout_eligible_for_threshold_tuning=False,
            holdout_eligible_for_checkpoint_selection=False),
        isolation=dict(
            optimizer_steps=0, checkpoint_writes=0,
            baseline_frozen=True,
            baseline_parameters_unchanged=baseline_unchanged,
            dino_frozen=True, dino_parameters_unchanged=dino_unchanged,
            labeller_heads_frozen=True,
            labeller_parameters_unchanged=heads_unchanged,
            baseline_and_dino_loaded_sequentially=True,
            feature_cache_key_includes_image_and_checkpoint_identity=True,
            feature_cache_namespaced_by_dataset_split=True,
            holdout_labels_used_for_evaluation_only=True),
        source_control=dict(summary=source_summary,
                            rows=combined['source']),
        holdouts=holdout_payload,
        decision=decision)
    replacements = roi_probe.write_json_atomic(args.out_json, payload)
    print('[holdout-routing] {}'.format(decision))
    for seq in args.holdout_seqs:
        summary = summaries[seq]
        print('[holdout-routing] {} baseline={}/{} override={}/{} '
              'baseline_mcml={} override_mcml={} harmful_overrides={}'
              .format(
                  seq, summary['baseline']['top1_hits'],
                  summary['baseline']['frame_count'],
                  summary['confident_override']['top1_hits'],
                  summary['confident_override']['frame_count'],
                  summary['baseline']['top1_mcml'],
                  summary['confident_override']['top1_mcml'],
                  summary['routing_diagnostics'][
                      'baseline_correct_overridden_to_incorrect_count']))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
