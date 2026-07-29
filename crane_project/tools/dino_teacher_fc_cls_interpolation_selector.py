#!/usr/bin/env python3
"""Select a source-only interpolation of old and updated ROI classifiers.

The frozen DINOv2 backbone, RPN, shared ROI representation, and OBB regressor
must be bit-identical between the two input checkpoints. Only the final fc_cls
weight and bias are interpolated. Official source validation selects alpha;
target data are never discovered or read.
"""

import argparse
import copy
import os
from typing import Dict, List, Sequence, Tuple

import torch

from crane_project.tools import dino_teacher_common as common
from crane_project.tools import dino_teacher_rotated_labeller as labeller


SELECTOR_NAME = 'Frozen DINO ROI Classifier Source Interpolation Selector V1'
PROTOCOL_VERSION = 1
FC_CLS_KEYS = (
    'roi_head.bbox_head.fc_cls.weight',
    'roi_head.bbox_head.fc_cls.bias',
)


def parse_args():
    parser = argparse.ArgumentParser(description=SELECTOR_NAME)
    parser.add_argument(
        '--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument(
        '--source-val-datasets', nargs='+', default=['val:val'])
    parser.add_argument('--old-checkpoint', required=True)
    parser.add_argument('--updated-checkpoint', required=True)
    parser.add_argument(
        '--alphas', type=float, nargs='+',
        default=[0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
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
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--feature-cache-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-checkpoint', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def protocol_args(args):
    args.valid_content_tolerance = 1e-3
    args.deployment_score_thr = 0.05
    args.border_margin_ratio = 0.02
    return args


def validate_args(args):
    if args.seed != 0:
        raise ValueError('The source-only protocol requires --seed 0')
    labeller.parse_dataset_specs(args.source_val_datasets)
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
    if not 0.0 < float(args.riou_thr) <= 1.0:
        raise ValueError('--riou-thr must be in (0, 1]')
    alphas = sorted(set(float(value) for value in args.alphas))
    if any(value < 0.0 or value > 1.0 for value in alphas):
        raise ValueError('--alphas must be within [0, 1]')
    if 0.0 not in alphas or 1.0 not in alphas:
        raise ValueError('--alphas must include 0 and 1')
    args.alphas = alphas
    for path in (args.old_checkpoint, args.updated_checkpoint,
                 args.dinov2_checkpoint):
        if not os.path.isfile(path):
            raise ValueError('Required input does not exist: {}'.format(path))
    for path in (args.out_checkpoint, args.out_json):
        if os.path.exists(path):
            raise ValueError('Refusing to overwrite: {}'.format(path))


def discover_source_val(data_root: str, specs: Sequence[str]) -> List[Dict]:
    records = []
    for annotation_split, image_split in labeller.parse_dataset_specs(specs):
        records.extend(labeller.discover_labeled_records_with_image_split(
            data_root, annotation_split, image_split))
    records = sorted(
        records, key=lambda row: (row['split'], row['seq'], row['frame']))
    paths = [os.path.realpath(row['image']) for row in records]
    if len(paths) != len(set(paths)):
        raise RuntimeError('Source validation contains duplicate images')
    return records


def validate_checkpoint_pair(old_payload: Dict, updated_payload: Dict,
                             in_channels: int, args) -> Dict:
    labeller.validate_checkpoint(
        old_payload, in_channels, args, allow_training_mode_mismatch=True)
    labeller.validate_checkpoint(
        updated_payload, in_channels, args, allow_training_mode_mismatch=True)
    old_state = old_payload['heads_state_dict']
    updated_state = updated_payload['heads_state_dict']
    if set(old_state) != set(updated_state):
        raise RuntimeError('Checkpoint head state keys do not match')
    for key in FC_CLS_KEYS:
        if key not in old_state:
            raise RuntimeError('Checkpoint lacks {}'.format(key))
        if old_state[key].shape != updated_state[key].shape:
            raise RuntimeError('Classifier tensor shape mismatch: {}'.format(
                key))
        if old_state[key].dtype != updated_state[key].dtype:
            raise RuntimeError('Classifier tensor dtype mismatch: {}'.format(
                key))
    changed_outside_classifier = [
        key for key in old_state
        if key not in FC_CLS_KEYS
        and (old_state[key].shape != updated_state[key].shape
             or old_state[key].dtype != updated_state[key].dtype
             or not torch.equal(old_state[key], updated_state[key]))]
    if changed_outside_classifier:
        raise RuntimeError(
            'Updated checkpoint changed frozen head tensor: {}'.format(
                changed_outside_classifier[0]))
    changed_classifier = [
        key for key in FC_CLS_KEYS
        if not torch.equal(old_state[key], updated_state[key])]
    if not changed_classifier:
        raise RuntimeError('Classifier checkpoints are identical')
    sampling = updated_payload.get('source_sampling')
    if (not isinstance(sampling, dict)
            or sampling.get('definition') !=
            'source_train_short_token_lower_tertile'
            or 'short_token_threshold' not in sampling):
        raise RuntimeError(
            'Updated checkpoint lacks its source-train small-token contract')
    return sampling


def interpolated_state(old_state: Dict, updated_state: Dict,
                       alpha: float) -> Dict:
    state = {}
    for key, old_value in old_state.items():
        if key in FC_CLS_KEYS:
            updated_value = updated_state[key].to(
                device=old_value.device, dtype=old_value.dtype)
            state[key] = (old_value * (1.0 - float(alpha))
                          + updated_value * float(alpha))
        else:
            state[key] = old_value
    return state


def row_key(row: Dict) -> Tuple[str, int]:
    return str(row['seq']), int(row['frame'])


def subset_rows(rows: Sequence[Dict], records: Sequence[Dict]) -> List[Dict]:
    required = {(str(row['seq']), int(row['frame'])) for row in records}
    selected = [row for row in rows if row_key(row) in required]
    if len(selected) != len(required):
        raise RuntimeError('Source-small validation row mapping is incomplete')
    return selected


def frame_outcomes(rows: Sequence[Dict]) -> List[Dict]:
    return [dict(
        seq=str(row['seq']), frame=int(row['frame']),
        top1_hit=bool(row['metrics']['top1_hit']),
        best_usable_rank=row['metrics']['best_usable_rank'],
        top1_riou=float(row['metrics']['top1_riou']),
        top1_score=row['metrics']['top1_score']) for row in rows]


def retention_counts(baseline_rows: Sequence[Dict],
                     candidate_rows: Sequence[Dict]) -> Dict:
    baseline = {row_key(row): bool(row['metrics']['top1_hit'])
                for row in baseline_rows}
    candidate = {row_key(row): bool(row['metrics']['top1_hit'])
                 for row in candidate_rows}
    if set(baseline) != set(candidate):
        raise RuntimeError('Candidate source-validation frames do not align')
    old_correct = {key for key, hit in baseline.items() if hit}
    retained = sum(bool(candidate[key]) for key in old_correct)
    gains = sum(not baseline[key] and candidate[key] for key in baseline)
    losses = sum(baseline[key] and not candidate[key] for key in baseline)
    return dict(
        baseline_correct_count=int(len(old_correct)),
        retained_correct_count=int(retained),
        old_correct_retention_rate=(
            float(retained / len(old_correct)) if old_correct else 1.0),
        newly_correct_count=int(gains), newly_incorrect_count=int(losses))


def source_gate(baseline_full: Dict, baseline_small: Dict,
                candidate_full: Dict, candidate_small: Dict,
                retention: Dict) -> Dict:
    checks = dict(
        old_correct_frames_fully_retained=bool(
            retention['newly_incorrect_count'] == 0),
        full_top1_not_lower=bool(
            int(candidate_full['top1_hits']) >=
            int(baseline_full['top1_hits'])),
        small_top1_strictly_higher=bool(
            int(candidate_small['top1_hits']) >
            int(baseline_small['top1_hits'])),
        small_mcml_not_higher=bool(
            int(candidate_small['top1_mcml']) <=
            int(baseline_small['top1_mcml'])))
    return dict(passed=bool(all(checks.values())), checks=checks)


def candidate_key(row: Dict) -> Tuple:
    small = row['source_small_summary']
    full = row['source_full_summary']
    return (
        int(small['top1_hits']), -int(small['top1_mcml']),
        int(full['top1_hits']), -int(full['top1_mcml']),
        -float(row['alpha']))


def selected_checkpoint_payload(old_payload: Dict, selected_state: Dict,
                                selected: Dict, baseline: Dict, args) -> Dict:
    payload = copy.deepcopy(old_payload)
    payload['heads_state_dict'] = {
        key: value.detach().cpu().clone()
        for key, value in selected_state.items()}
    payload['optimizer_state_dict'] = None
    payload['scheduler_state_dict'] = None
    payload['best_source_val_summary'] = selected['source_full_summary']
    payload['best_source_small_val_summary'] = (
        selected['source_small_summary'])
    payload['source_baseline_val_summary'] = (
        baseline['source_full_summary'])
    payload['source_baseline_small_val_summary'] = (
        baseline['source_small_summary'])
    payload['source_only_fc_cls_interpolation'] = dict(
        selector=SELECTOR_NAME, protocol_version=PROTOCOL_VERSION,
        alpha=float(selected['alpha']),
        old_checkpoint=os.path.abspath(args.old_checkpoint),
        old_checkpoint_sha256=common.file_sha256(args.old_checkpoint),
        updated_checkpoint=os.path.abspath(args.updated_checkpoint),
        updated_checkpoint_sha256=common.file_sha256(
            args.updated_checkpoint),
        source_gate=selected['source_gate'], target_data_read=False)
    return payload


def main():
    args = protocol_args(parse_args())
    validate_args(args)
    labeller.set_seed(args.seed)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    source_val = discover_source_val(args.data_root, args.source_val_datasets)

    dino_devices = [torch.device('cuda:{}'.format(gpu))
                    for gpu in args.dino_gpus]
    dino_device = dino_devices[0]
    head_device = torch.device('cuda:{}'.format(args.head_gpu))
    dino, loaded_patch_size = common.load_frozen_dinov2(
        args.dinov2_repo, args.dinov2_checkpoint, args.dinov2_model,
        dino_devices, args.legacy_sdpa_query_chunk)
    if int(loaded_patch_size) != int(args.patch_size):
        raise RuntimeError('Unexpected DINO patch size')
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    in_channels = int(getattr(dino, 'embed_dim', 0))
    if in_channels <= 0:
        raise RuntimeError('DINO model does not expose embed_dim')

    old_payload = torch.load(args.old_checkpoint, map_location='cpu')
    updated_payload = torch.load(args.updated_checkpoint, map_location='cpu')
    sampling = validate_checkpoint_pair(
        old_payload, updated_payload, in_channels, args)
    source_small = labeller.source_small_records(
        source_val, args, float(sampling['short_token_threshold']))
    if not source_small:
        raise RuntimeError('Source validation has no source-small records')

    heads = labeller.FrozenDinoRotatedHeads(in_channels, args).to(head_device)
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    dino_versions = common.module_parameter_versions(dino)
    old_state = old_payload['heads_state_dict']
    updated_state = updated_payload['heads_state_dict']
    candidates = []
    baseline_rows = None
    baseline_result = None
    for alpha in args.alphas:
        state = interpolated_state(old_state, updated_state, alpha)
        heads.load_state_dict(state, strict=True)
        heads.eval()
        rows = labeller.evaluate_records(
            dino, heads, source_val, args, dino_device, head_device,
            role='source_interpolation_validation')
        small_rows = subset_rows(rows, source_small)
        full_summary = labeller.summarize_rows(rows)
        small_summary = labeller.summarize_rows(small_rows)
        if baseline_rows is None:
            baseline_rows = rows
            retention = retention_counts(rows, rows)
            baseline_result = dict(
                alpha=float(alpha), source_full_summary=full_summary,
                source_small_summary=small_summary)
            gate = dict(
                passed=False, checks=dict(
                    baseline_reference=True,
                    not_a_replacement_candidate=True))
        else:
            retention = retention_counts(baseline_rows, rows)
            gate = source_gate(
                baseline_result['source_full_summary'],
                baseline_result['source_small_summary'],
                full_summary, small_summary, retention)
        candidate = dict(
            alpha=float(alpha), source_full_summary=full_summary,
            source_small_summary=small_summary, retention=retention,
            source_gate=gate, frame_outcomes=frame_outcomes(rows))
        candidates.append(candidate)
        print('[alpha] value={} full_top1={}/{} small_top1={}/{} '
              'losses={} gate={}'.format(
                  alpha, full_summary['top1_hits'],
                  full_summary['frame_count'], small_summary['top1_hits'],
                  small_summary['frame_count'],
                  retention['newly_incorrect_count'], gate['passed']))

    passing = [row for row in candidates if row['source_gate']['passed']]
    selected = max(passing, key=candidate_key) if passing else None
    checkpoint_written = False
    selected_checkpoint_sha256 = None
    if selected is not None:
        selected_state = interpolated_state(
            old_state, updated_state, float(selected['alpha']))
        payload = selected_checkpoint_payload(
            old_payload, selected_state, selected, baseline_result, args)
        labeller.atomic_torch_save(payload, args.out_checkpoint)
        checkpoint_written = True
        selected_checkpoint_sha256 = common.file_sha256(args.out_checkpoint)
        decision = 'SOURCE_ONLY_FC_CLS_INTERPOLATION_SELECTED'
    else:
        decision = 'STOP_NO_SOURCE_SAFE_INTERPOLATION'

    dino_unchanged = (
        dino_versions == common.module_parameter_versions(dino))
    if not dino_unchanged:
        raise RuntimeError('Frozen DINO parameter invariant failed')
    output = dict(
        selector=SELECTOR_NAME, protocol_version=PROTOCOL_VERSION,
        old_checkpoint=os.path.abspath(args.old_checkpoint),
        old_checkpoint_sha256=common.file_sha256(args.old_checkpoint),
        updated_checkpoint=os.path.abspath(args.updated_checkpoint),
        updated_checkpoint_sha256=common.file_sha256(
            args.updated_checkpoint),
        dinov2_checkpoint=os.path.abspath(args.dinov2_checkpoint),
        dinov2_checkpoint_sha256=common.file_sha256(
            args.dinov2_checkpoint),
        protocol=dict(
            alpha_candidates=list(args.alphas),
            interpolation_parameters=list(FC_CLS_KEYS),
            source_val_datasets=list(args.source_val_datasets),
            source_small_definition=sampling,
            selection_rule=(
                '100pct_old_correct_retention_and_full_nonregression_and_'
                'strict_small_top1_gain_and_small_mcml_nonregression'),
            tie_break=(
                'small_top1_small_mcml_full_top1_full_mcml_then_lower_alpha'),
            target_data_discovered=False, target_data_read=False,
            target_used_for_selection=False),
        isolation=dict(
            optimizer_steps=0, dino_frozen=True,
            dino_parameters_unchanged=dino_unchanged,
            non_classifier_head_tensors_identical=True,
            selected_checkpoint_written=checkpoint_written),
        source=dict(
            val_count=len(source_val), small_val_count=len(source_small),
            baseline=baseline_result, candidates=candidates),
        selected_alpha=(None if selected is None
                        else float(selected['alpha'])),
        selected_checkpoint=(None if selected is None
                             else os.path.abspath(args.out_checkpoint)),
        selected_checkpoint_sha256=selected_checkpoint_sha256,
        decision=decision)
    replacements = common.write_json_atomic(args.out_json, output)
    print('[selector] {} alpha={}'.format(
        decision, output['selected_alpha']))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
