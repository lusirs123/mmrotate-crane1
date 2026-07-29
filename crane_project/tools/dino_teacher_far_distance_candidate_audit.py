#!/usr/bin/env python3
"""Read-only DINO candidate audit for real_seq02 far-distance frames 2..41.

The frozen source-selected DINO rotated labeller is evaluated without training.
The audit asks only whether any valid DINO detection reaches RIoU >= 0.5.  If
geometry is absent, retraining a classifier/ranker is not authorized; the next
work must change candidate generation or spatial resolution instead.
"""

import argparse
import os
from typing import Dict, Sequence, Tuple

import torch

from crane_project.tools import dino_teacher_common as common
from crane_project.tools import dino_teacher_rotated_labeller as labeller


AUDIT_NAME = 'Frozen DINO Far-Distance Candidate Coverage Audit V1'
PROTOCOL_VERSION = 1
TARGET_SPLIT = 'test'
TARGET_SEQ = 'real_seq02'
TARGET_START = 2
TARGET_END = 41
TARGET_COUNT = TARGET_END - TARGET_START + 1
MIN_GEOMETRY_COUNT = 32
SOURCE_MIN_TOP1_RATE = 0.8


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-split', default='val')
    parser.add_argument('--source-seq', default='real_seq07')
    parser.add_argument('--source-val-modulus', type=int, default=5)
    parser.add_argument('--target-split', default=TARGET_SPLIT)
    parser.add_argument('--target-seq', default=TARGET_SEQ)
    parser.add_argument('--target-start', type=int, default=TARGET_START)
    parser.add_argument('--target-end', type=int, default=TARGET_END)
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
    parser.add_argument('--feature-strides', type=int, nargs='+', default=None)
    parser.add_argument('--feature-cache-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def validate_args(args):
    if args.seed != 0:
        raise ValueError('The frozen far-distance protocol requires --seed 0')
    if args.source_split != 'val' or args.source_seq != 'real_seq07':
        raise ValueError('Source control is fixed to val/real_seq07')
    if args.source_val_modulus != 5:
        raise ValueError('Source control modulus is fixed to 5')
    if (args.target_split != TARGET_SPLIT or args.target_seq != TARGET_SEQ
            or args.target_start != TARGET_START
            or args.target_end != TARGET_END):
        raise ValueError(
            'Far-distance diagnosis is fixed to test/real_seq02 frames 2..41')
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
    feature_strides = getattr(args, 'feature_strides', None)
    if feature_strides is not None:
        args.feature_strides = sorted(set(int(value)
                                         for value in feature_strides))
        if (not args.feature_strides
                or any(value <= 0 for value in args.feature_strides)
                or args.patch_size not in args.feature_strides):
            raise ValueError(
                '--feature-strides must be positive and include patch size')
    for path in (args.labeller_checkpoint, args.dinov2_checkpoint):
        if not os.path.isfile(path):
            raise ValueError('Required checkpoint does not exist: {}'.format(
                path))
    if os.path.exists(args.out_json):
        raise ValueError('Refusing to overwrite a completed audit result')


def protocol_args(args):
    args.valid_content_tolerance = 1e-3
    args.deployment_score_thr = 0.05
    args.border_margin_ratio = 0.02
    args.riou_thr = 0.5
    args.source_min_top1_rate = SOURCE_MIN_TOP1_RATE
    args.epochs = 1
    args.lr = 1.0
    args.max_grad_norm = 1.0
    args.resume_checkpoint = None
    args.eval_only_checkpoint = args.labeller_checkpoint
    args.target_min_wins = MIN_GEOMETRY_COUNT
    args.max_mcml = TARGET_COUNT
    return args


def source_and_target_records(args) -> Tuple[Sequence[Dict], Sequence[Dict]]:
    source = [
        row for row in common.discover_labeled_records(
            args.data_root, args.source_split, 0)
        if row['seq'] == args.source_seq]
    _source_train, source_val = labeller.split_source_records(
        source, args.source_val_modulus)
    targets = labeller.target_records(args)
    labeller.assert_training_target_isolation(source, targets)
    if len(targets) != TARGET_COUNT:
        raise RuntimeError('Incomplete far-distance target slice')
    return source_val, targets


def load_frozen_labeller(args, dino_devices, head_device):
    dino, loaded_patch_size = common.load_frozen_dinov2(
        args.dinov2_repo, args.dinov2_checkpoint,
        args.dinov2_model, dino_devices,
        args.legacy_sdpa_query_chunk)
    if int(loaded_patch_size) != int(args.patch_size):
        raise RuntimeError('Unexpected DINO patch size')
    dino.eval()
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    in_channels = int(getattr(dino, 'embed_dim', 0))
    if in_channels <= 0:
        raise RuntimeError('DINO model does not expose embed_dim')
    heads = labeller.FrozenDinoRotatedHeads(in_channels, args).to(head_device)
    checkpoint = torch.load(args.labeller_checkpoint, map_location='cpu')
    labeller.validate_checkpoint(checkpoint, in_channels, args)
    heads.load_state_dict(checkpoint['heads_state_dict'], strict=True)
    heads.eval()
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    return dino, heads


def make_decision(source_summary: Dict, target_summary: Dict) -> str:
    source_count = int(source_summary.get('frame_count', 0))
    source_hits = int(source_summary.get('top1_hits', 0))
    source_rate = float(source_hits) / source_count if source_count else 0.0
    if source_rate < SOURCE_MIN_TOP1_RATE:
        return 'AUDIT_INVALID_SOURCE_CONTROL'
    if int(target_summary.get('frame_count', 0)) != TARGET_COUNT:
        return 'AUDIT_INVALID_TARGET_FRAME_COUNT'
    valid_geometry = int(target_summary.get('geometry_eligible_count', 0))
    raw_geometry = int(target_summary.get(
        'raw_unfiltered_geometry_eligible_count', 0))
    if valid_geometry >= MIN_GEOMETRY_COUNT:
        return 'AUTHORIZE_SOURCE_ONLY_FAR_SCALE_RANKING_TRAINING'
    if raw_geometry >= MIN_GEOMETRY_COUNT:
        return 'FAR_GEOMETRY_BORDER_FILTER_CONFLICT'
    return 'DINO_FAR_DISTANCE_CANDIDATE_GENERATION_INSUFFICIENT'


def main():
    args = protocol_args(parse_args())
    validate_args(args)
    labeller.set_seed(args.seed)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    source_records, target_records = source_and_target_records(args)

    dino_devices = [torch.device('cuda:{}'.format(gpu))
                    for gpu in args.dino_gpus]
    dino_device = dino_devices[0]
    head_device = torch.device('cuda:{}'.format(args.head_gpu))
    dino, heads = load_frozen_labeller(
        args, dino_devices, head_device)
    dino_versions = common.module_parameter_versions(dino)
    head_versions = common.module_parameter_versions(heads)

    source_rows = labeller.evaluate_records(
        dino, heads, source_records, args, dino_device, head_device,
        role='source_validation')
    target_rows = labeller.evaluate_records(
        dino, heads, target_records, args, dino_device, head_device,
        role='target_dev_diagnosis_only')
    source_summary = labeller.summarize_rows(source_rows)
    target_summary = labeller.summarize_rows(target_rows)

    dino_unchanged = (
        dino_versions == common.module_parameter_versions(dino))
    heads_unchanged = (
        head_versions == common.module_parameter_versions(heads))
    if not dino_unchanged or not heads_unchanged:
        raise RuntimeError('Frozen DINO/labeller parameter invariant failed')
    decision = make_decision(source_summary, target_summary)

    payload = dict(
        audit=AUDIT_NAME, protocol_version=PROTOCOL_VERSION,
        labeller_checkpoint=os.path.abspath(args.labeller_checkpoint),
        labeller_checkpoint_sha256=common.file_sha256(
            args.labeller_checkpoint),
        dinov2_checkpoint=os.path.abspath(args.dinov2_checkpoint),
        dinov2_checkpoint_sha256=common.file_sha256(
            args.dinov2_checkpoint),
        protocol=dict(
            target_slice='test/real_seq02[2..41]',
            target_role='diagnosis_only',
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            geometry_definition='any_valid_detection_with_RIoU_at_least_0.5',
            candidate_limit=int(args.max_detections),
            minimum_geometry_count=MIN_GEOMETRY_COUNT,
            minimum_geometry_rate=float(MIN_GEOMETRY_COUNT / TARGET_COUNT),
            interpretation=dict(
                pass_result=(
                    'geometry_exists_so_source_only_scale_balanced_ranking_'
                    'training_may_be_tested'),
                fail_result=(
                    'classification_retraining_not_authorized_change_'
                    'resolution_or_candidate_generation'))),
        isolation=dict(
            optimizer_steps=0, checkpoint_writes=0,
            dino_frozen=True, dino_parameters_unchanged=dino_unchanged,
            labeller_heads_frozen=True,
            labeller_parameters_unchanged=heads_unchanged,
            target_labels_used_for_evaluation_only=True),
        source_control=dict(summary=source_summary, rows=source_rows),
        target_far_distance=dict(summary=target_summary, rows=target_rows),
        decision=decision)
    replacements = common.write_json_atomic(args.out_json, payload)
    print('[far-distance] {} geometry={}/{} r20={} r100={} top1={}'
          .format(
              decision, target_summary['geometry_eligible_count'],
              TARGET_COUNT, target_summary['recall_at_20'],
              target_summary['recall_at_100'],
              target_summary['top1_hits']))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
