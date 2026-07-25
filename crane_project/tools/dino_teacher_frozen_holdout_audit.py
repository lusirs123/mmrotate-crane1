#!/usr/bin/env python3
"""Post-hoc frozen raw/filter audit for the DINO rotated labeller.

The checkpoint and all inference rules are fixed before either holdout is
read.  Raw ROI outputs are compared with the unchanged valid-content result.
The audit is ineligible for checkpoint selection, tuning, or training.
"""

import argparse
import hashlib
import os
from typing import Dict, Sequence

import torch

from crane_project.tools import dino_teacher_rotated_labeller as labeller
from crane_project.tools import dino_teacher_source_roi_head_probe as roi_probe
from crane_project.tools import frozen_p3_feature_alignment_audit as alignment
from crane_project.tools import frozen_p3_objectness_transfer_probe as transfer


AUDIT_NAME = 'Frozen DINO Rotated Labeller Raw/Filter Holdout Audit V2'
PROTOCOL_VERSION = 2
HOLDOUT_SPECS = {
    'real_seq03': dict(first=1, last=200, count=200),
    'sim_seq09': dict(first=0, last=571, count=572),
}
VALID_CONTENT_TOLERANCE = 1e-3
DEPLOYMENT_SCORE_THR = 0.05
BORDER_MARGIN_RATIO = 0.02
RIOU_THR = 0.5
MIN_TOP1_RATE = 0.8
MAX_MCML = 5
SOURCE_MIN_TOP1_RATE = 0.8


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-split', default='val')
    parser.add_argument('--source-seq', default='real_seq07')
    parser.add_argument('--source-val-modulus', type=int, default=5)
    parser.add_argument('--holdout-split', default='test')
    parser.add_argument('--holdout-seqs', nargs='+', required=True)
    parser.add_argument('--confirm-frozen-holdout', action='store_true')
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
    if not args.confirm_frozen_holdout:
        raise ValueError('Holdout audit requires --confirm-frozen-holdout')
    if args.seed != 0:
        raise ValueError('The frozen holdout protocol requires --seed 0')
    if args.source_split != 'val' or args.source_seq != 'real_seq07':
        raise ValueError('Source control is fixed to val/real_seq07')
    if args.source_val_modulus != 5:
        raise ValueError('Source control modulus is fixed to 5')
    if args.holdout_split != 'test':
        raise ValueError('Holdout split is fixed to test')
    if sorted(args.holdout_seqs) != sorted(HOLDOUT_SPECS):
        raise ValueError(
            'One-shot audit requires real_seq03 and sim_seq09 together')
    if len(set(args.holdout_seqs)) != len(args.holdout_seqs):
        raise ValueError('Holdout sequence names must be unique')
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
    for path in (args.labeller_checkpoint, args.dinov2_checkpoint):
        if not os.path.isfile(path):
            raise ValueError('Required checkpoint does not exist: {}'.format(
                path))
    if os.path.exists(args.out_json):
        raise ValueError(
            'Refusing to overwrite a completed one-shot holdout result')


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_args(args):
    """Expose fixed evaluation values expected by shared labeller helpers."""
    args.valid_content_tolerance = VALID_CONTENT_TOLERANCE
    args.deployment_score_thr = DEPLOYMENT_SCORE_THR
    args.border_margin_ratio = BORDER_MARGIN_RATIO
    args.riou_thr = RIOU_THR
    args.source_min_top1_rate = SOURCE_MIN_TOP1_RATE
    args.epochs = 1
    args.lr = 1.0
    args.max_grad_norm = 1.0
    args.resume_checkpoint = None
    args.eval_only_checkpoint = args.labeller_checkpoint
    return args


def select_records(data_root: str, split: str, seq: str):
    records = [
        row for row in transfer.discover_labeled_records(data_root, split, 0)
        if row['seq'] == seq]
    records.sort(key=lambda row: int(row['frame']))
    return records


def validate_complete_holdout(seq: str, records: Sequence[Dict]):
    spec = HOLDOUT_SPECS[seq]
    frames = [int(row['frame']) for row in records]
    expected = list(range(spec['first'], spec['last'] + 1))
    if len(records) != spec['count'] or frames != expected:
        raise RuntimeError(
            '{} holdout is incomplete: expected {} frames {}..{}, got {}'
            .format(seq, spec['count'], spec['first'], spec['last'],
                    len(records)))


def source_control_passes(summary: Dict) -> bool:
    count = int(summary['frame_count'])
    return (count > 0
            and float(summary['top1_hits']) / count >= SOURCE_MIN_TOP1_RATE)


def holdout_passes(summary: Dict) -> bool:
    count = int(summary['frame_count'])
    return (count > 0
            and float(summary['top1_hits']) / count >= MIN_TOP1_RATE
            and int(summary['top1_mcml']) <= MAX_MCML)


def main():
    args = protocol_args(parse_args())
    validate_args(args)
    labeller.set_seed(args.seed)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)

    source_records = select_records(
        args.data_root, args.source_split, args.source_seq)
    _source_train, source_val = labeller.split_source_records(
        source_records, args.source_val_modulus)
    holdout_records = {
        seq: select_records(args.data_root, args.holdout_split, seq)
        for seq in args.holdout_seqs}
    for seq, records in holdout_records.items():
        validate_complete_holdout(seq, records)
        labeller.assert_training_target_isolation(source_records, records)

    dino_devices = [torch.device('cuda:{}'.format(gpu))
                    for gpu in args.dino_gpus]
    dino_device = dino_devices[0]
    head_device = torch.device('cuda:{}'.format(args.head_gpu))
    dino, loaded_patch_size = labeller.audit.load_frozen_dinov2(
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
    dino_versions = alignment.module_parameter_versions(dino)
    head_versions = alignment.module_parameter_versions(heads)

    source_rows = labeller.evaluate_records(
        dino, heads, source_val, args, dino_device, head_device,
        role='source_validation')
    source_summary = labeller.summarize_rows(source_rows)

    holdout_payload = {}
    combined_rows = []
    for seq in args.holdout_seqs:
        rows = labeller.evaluate_records(
            dino, heads, holdout_records[seq], args,
            dino_device, head_device, role='target_holdout_readonly')
        summary = labeller.summarize_rows(rows)
        holdout_payload[seq] = dict(
            summary=summary, passed_fixed_gate=holdout_passes(summary),
            rows=rows)
        combined_rows.extend(rows)
        print('[holdout] seq={} filtered_top1={}/{} raw_geometry={} '
              'removed_geometry={} mcml={} deployment_top1={}'
              .format(seq, summary['top1_hits'], summary['frame_count'],
                      summary['raw_unfiltered_geometry_eligible_count'],
                      summary['filter_removed_usable_geometry_count'],
                      summary['top1_mcml'],
                      summary['deployment_top1_hits']))

    dino_unchanged = (
        dino_versions == alignment.module_parameter_versions(dino))
    heads_unchanged = (
        head_versions == alignment.module_parameter_versions(heads))
    if not dino_unchanged or not heads_unchanged:
        raise RuntimeError('Frozen parameter invariant failed')
    source_ok = source_control_passes(source_summary)
    holdouts_ok = all(
        row['passed_fixed_gate'] for row in holdout_payload.values())
    decision = ('FROZEN_HOLDOUT_PASS' if source_ok and holdouts_ok
                else 'FROZEN_HOLDOUT_FAIL_NO_RETUNING')

    payload = dict(
        audit=AUDIT_NAME, protocol_version=PROTOCOL_VERSION,
        source_selected_checkpoint=os.path.abspath(args.labeller_checkpoint),
        checkpoint_sha256=file_sha256(args.labeller_checkpoint),
        dinov2_checkpoint=os.path.abspath(args.dinov2_checkpoint),
        dinov2_sha256=file_sha256(args.dinov2_checkpoint),
        protocol=dict(
            role='target_holdout_posthoc_diagnosis', one_shot=False,
            posthoc_raw_filter_comparison=True,
            complete_sequences=list(args.holdout_seqs),
            eligible_for_model_selection=False,
            eligible_for_threshold_tuning=False,
            eligible_for_training=False,
            valid_content_tolerance=VALID_CONTENT_TOLERANCE,
            deployment_score_thr=DEPLOYMENT_SCORE_THR,
            border_margin_ratio=BORDER_MARGIN_RATIO,
            riou_thr=RIOU_THR, min_top1_rate=MIN_TOP1_RATE,
            max_mcml=MAX_MCML,
            source_min_top1_rate=SOURCE_MIN_TOP1_RATE),
        isolation=dict(
            dino_frozen=True, dino_parameters_unchanged=dino_unchanged,
            labeller_heads_frozen=True,
            labeller_parameters_unchanged=heads_unchanged,
            optimizer_steps=0, checkpoint_writes=0,
            holdout_used_for_model_selection=False,
            holdout_labels_used_for_evaluation_only=True),
        source_control=dict(
            summary=source_summary, passed_fixed_gate=source_ok),
        holdouts=holdout_payload,
        combined_summary=labeller.summarize_rows(combined_rows),
        decision=decision)
    replacements = roi_probe.write_json_atomic(args.out_json, payload)
    print('[holdout-audit] {}'.format(decision))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
