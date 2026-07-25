#!/usr/bin/env python3
"""Read-only baseline-first audit for the frozen DINO rotated labeller.

The BrightAug detector keeps ownership of every frame for which its unchanged
test configuration returns a detection.  The frozen DINO labeller is consulted
only when BrightAug is silent.  Two fixed policies are reported:

* strict: accept DINO top-1 only when its score is at least 0.05;
* ranked: accept DINO top-1 regardless of score (diagnostic upper bound).

This script does not train, write checkpoints, fuse scores, or modify either
model's normal inference path.
"""

import argparse
import hashlib
import os
import sys
from typing import Dict, Sequence, Tuple

import numpy as np
import torch


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import dino_teacher_frozen_region_audit as dino_audit  # noqa: E402
from crane_project.tools import dino_teacher_rotated_labeller as labeller  # noqa: E402
from crane_project.tools import dino_teacher_source_roi_head_probe as roi_probe  # noqa: E402
from crane_project.tools import frozen_p3_feature_alignment_audit as alignment  # noqa: E402
from crane_project.tools import frozen_p3_objectness_transfer_probe as transfer  # noqa: E402


AUDIT_NAME = 'Frozen DINO Baseline-First Silence Rescue Audit V1'
PROTOCOL_VERSION = 1
DINO_DEPLOYMENT_SCORE_THR = 0.05
VALID_CONTENT_TOLERANCE = 1e-3
RIOU_THR = 0.5
TARGET_MIN_HITS = 26
TARGET_MAX_MCML = 5


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--baseline-gpu', type=int, default=0)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-split', default='val')
    parser.add_argument('--source-seq', default='real_seq07')
    parser.add_argument('--source-val-modulus', type=int, default=5)
    parser.add_argument('--target-split', default='test')
    parser.add_argument('--target-seq', default='real_seq02')
    parser.add_argument('--target-start', type=int, default=137)
    parser.add_argument('--target-end', type=int, default=169)
    parser.add_argument('--labeller-checkpoint', required=True)
    parser.add_argument('--dinov2-repo', required=True)
    parser.add_argument('--dinov2-checkpoint', required=True)
    parser.add_argument('--dinov2-model', default=dino_audit.CANONICAL_MODEL)
    parser.add_argument('--dino-gpus', type=int, nargs='+', required=True)
    parser.add_argument('--head-gpu', type=int, default=0)
    parser.add_argument('--legacy-sdpa-query-chunk', type=int, default=512)
    parser.add_argument('--dino-height', type=int,
                        default=dino_audit.CANONICAL_DINO_HEIGHT)
    parser.add_argument('--dino-max-long-side', type=int,
                        default=dino_audit.CANONICAL_DINO_MAX_LONG_SIDE)
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
    if args.seed != 0:
        raise ValueError('The frozen rescue protocol requires --seed 0')
    if args.source_split != 'val' or args.source_seq != 'real_seq07':
        raise ValueError('Source control is fixed to val/real_seq07')
    if args.source_val_modulus != 5:
        raise ValueError('Source control modulus is fixed to 5')
    if (args.target_split != 'test' or args.target_seq != 'real_seq02'
            or args.target_start != 137 or args.target_end != 169):
        raise ValueError(
            'Target diagnosis is fixed to test/real_seq02 frames 137..169')
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


def protocol_args(args):
    """Supply fixed labeller helper fields without exposing tuning knobs."""
    args.valid_content_tolerance = VALID_CONTENT_TOLERANCE
    args.deployment_score_thr = DINO_DEPLOYMENT_SCORE_THR
    args.border_margin_ratio = 0.02
    args.riou_thr = RIOU_THR
    args.source_min_top1_rate = 0.8
    args.epochs = 1
    args.lr = 1.0
    args.max_grad_norm = 1.0
    args.resume_checkpoint = None
    args.eval_only_checkpoint = args.labeller_checkpoint
    return args


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_and_target_records(args) -> Tuple[Sequence[Dict], Sequence[Dict]]:
    source = [
        row for row in transfer.discover_labeled_records(
            args.data_root, args.source_split, 0)
        if row['seq'] == args.source_seq]
    _unused, source_val = labeller.split_source_records(
        source, args.source_val_modulus)
    target = labeller.target_records(args)
    labeller.assert_training_target_isolation(source, target)
    return source_val, target


def normalize_baseline_result(result) -> np.ndarray:
    """Normalize one-image, one-class MMRotate output without reordering it."""
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise RuntimeError('Expected one-image baseline result')
    per_image = result[0]
    if not isinstance(per_image, (list, tuple)) or len(per_image) != 1:
        raise RuntimeError('Expected one-class baseline result')
    array = np.asarray(per_image[0], dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 6:
        raise RuntimeError('Baseline detections must have shape [N,6]')
    if not np.isfinite(array).all():
        raise RuntimeError('Baseline produced non-finite detections')
    return array.copy()


def load_baseline(config_path: str, checkpoint_path: str,
                  device: torch.device):
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector

    cfg = Config.fromfile(config_path)
    custom_imports = cfg.get('custom_imports')
    if custom_imports:
        import_modules_from_strings(**custom_imports)
    model = build_detector(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    test_cfg = cfg.model.get('test_cfg')
    if test_cfg is None:
        raise RuntimeError('Baseline config has no model.test_cfg')
    policy = dict(
        score_thr=float(test_cfg.get('score_thr', 0.0)),
        max_per_img=int(test_cfg.get('max_per_img', 0)))
    if policy['max_per_img'] != 1:
        raise RuntimeError(
            'Baseline-first protocol requires max_per_img=1, got {}'.format(
                policy['max_per_img']))
    return model, cfg, policy


def evaluate_baseline(model, cfg, records: Sequence[Dict],
                      device: torch.device, role: str):
    # Keep module import lazy so --help and policy unit tests do not require
    # the CUDA/MMCV runtime used only by actual detector inference.
    diag = transfer.entry_probe.get_diag()
    transform, img_scale, flip = diag.build_test_transforms(cfg)
    rows = []
    with torch.no_grad():
        for index, record in enumerate(records):
            image, meta, stats = diag.preprocess_image(
                record['image'], transform, img_scale, flip)
            if image is None:
                raise RuntimeError('Cannot preprocess {}'.format(record['image']))
            detections = normalize_baseline_result(
                model.simple_test(image.to(device), [meta], rescale=True))
            gt = labeller.parse_original_gt(record['annotation'])
            metrics = labeller.ranked_detection_metrics(
                detections, gt, RIOU_THR, DINO_DEPLOYMENT_SCORE_THR)
            rows.append(dict(
                role=role, split=record['split'], seq=record['seq'],
                frame=int(record['frame']), metrics=metrics,
                brightness=float(stats['raw_brightness']),
                detections=detections.tolist()))
            if ((index + 1) % 25 == 0 or index + 1 == len(records)):
                print('[baseline] role={} {}/{} active={} top1_hits={}'.format(
                    role, index + 1, len(records),
                    sum(bool(row['detections']) for row in rows),
                    sum(row['metrics']['top1_hit'] for row in rows)))
            del image
    return rows


def choose_baseline_first(baseline: np.ndarray, dino: np.ndarray,
                          dino_score_thr=None):
    """Keep baseline byte-for-byte; consult DINO only for baseline silence."""
    if baseline.shape[0] > 0:
        return baseline.copy(), 'baseline'
    if dino.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32), 'silence'
    if (dino_score_thr is not None
            and float(dino[0, 5]) < float(dino_score_thr)):
        return np.zeros((0, 6), dtype=np.float32), 'silence'
    return dino[:1].copy(), 'dino_rescue'


def combine_rows(baseline_rows: Sequence[Dict], dino_rows: Sequence[Dict],
                 records: Sequence[Dict]):
    if not (len(baseline_rows) == len(dino_rows) == len(records)):
        raise RuntimeError('Baseline/DINO/record count mismatch')
    combined = []
    for baseline_row, dino_row, record in zip(
            baseline_rows, dino_rows, records):
        baseline_key = (baseline_row['seq'], int(baseline_row['frame']))
        dino_key = (dino_row['seq'], int(dino_row['frame']))
        record_key = (record['seq'], int(record['frame']))
        if baseline_key != record_key or dino_key != record_key:
            raise RuntimeError('Baseline/DINO frame alignment mismatch')
        baseline = np.asarray(baseline_row['detections'], dtype=np.float32)
        dino = np.asarray(dino_row['detections'], dtype=np.float32)
        baseline = baseline.reshape((-1, 6))
        dino = dino.reshape((-1, 6))
        strict, strict_source = choose_baseline_first(
            baseline, dino, DINO_DEPLOYMENT_SCORE_THR)
        ranked, ranked_source = choose_baseline_first(baseline, dino, None)
        dino_top1 = dino[:1].copy()
        baseline_active = baseline.shape[0] > 0
        strict_preserved = (
            not baseline_active or np.array_equal(strict, baseline))
        ranked_preserved = (
            not baseline_active or np.array_equal(ranked, baseline))
        if not strict_preserved or not ranked_preserved:
            raise RuntimeError('Baseline preservation invariant failed')
        gt = labeller.parse_original_gt(record['annotation'])
        combined.append(dict(
            split=record['split'], seq=record['seq'],
            frame=int(record['frame']),
            baseline_active=bool(baseline_active),
            dino_top1_score=(float(dino[0, 5]) if dino.shape[0] else None),
            dino_top1_metrics=labeller.ranked_detection_metrics(
                dino_top1, gt, RIOU_THR, DINO_DEPLOYMENT_SCORE_THR),
            policies=dict(
                baseline=dict(
                    source='baseline' if baseline_active else 'silence',
                    preserved=True,
                    metrics=labeller.ranked_detection_metrics(
                        baseline, gt, RIOU_THR, DINO_DEPLOYMENT_SCORE_THR),
                    detections=baseline.tolist()),
                strict=dict(
                    source=strict_source, preserved=strict_preserved,
                    metrics=labeller.ranked_detection_metrics(
                        strict, gt, RIOU_THR, DINO_DEPLOYMENT_SCORE_THR),
                    detections=strict.tolist()),
                ranked=dict(
                    source=ranked_source, preserved=ranked_preserved,
                    metrics=labeller.ranked_detection_metrics(
                        ranked, gt, RIOU_THR, DINO_DEPLOYMENT_SCORE_THR),
                    detections=ranked.tolist()))))
    return combined


def summarize_policy(rows: Sequence[Dict], policy: str) -> Dict:
    flat = []
    scores = []
    for row in rows:
        item = row['policies'][policy]
        metrics = item['metrics']
        flat.append(dict(
            seq=row['seq'], frame=int(row['frame']),
            hit=bool(metrics['top1_hit']),
            deployment_hit=bool(metrics['deployment_top1_hit'])))
        if metrics['top1_score'] is not None:
            scores.append(float(metrics['top1_score']))
    return dict(
        frame_count=len(rows),
        output_frame_count=int(sum(
            row['policies'][policy]['metrics']['detection_count'] > 0
            for row in rows)),
        silence_count=int(sum(
            row['policies'][policy]['metrics']['detection_count'] == 0
            for row in rows)),
        top1_hits=int(sum(item['hit'] for item in flat)),
        top1_mcml=labeller.longest_miss(flat, 'hit'),
        deployment_top1_hits=int(sum(item['deployment_hit'] for item in flat)),
        deployment_top1_mcml=labeller.longest_miss(flat, 'deployment_hit'),
        baseline_selected_count=int(sum(
            row['policies'][policy]['source'] == 'baseline' for row in rows)),
        dino_rescue_selected_count=int(sum(
            row['policies'][policy]['source'] == 'dino_rescue'
            for row in rows)),
        dino_rescue_correct_count=int(sum(
            row['policies'][policy]['source'] == 'dino_rescue'
            and row['policies'][policy]['metrics']['top1_hit']
            for row in rows)),
        dino_rescue_incorrect_count=int(sum(
            row['policies'][policy]['source'] == 'dino_rescue'
            and not row['policies'][policy]['metrics']['top1_hit']
            for row in rows)),
        baseline_preservation_failures=int(sum(
            not row['policies'][policy]['preserved'] for row in rows)),
        median_output_score=(float(np.median(scores)) if scores else None))


def summarize_combination(rows: Sequence[Dict]) -> Dict:
    summary = {
        name: summarize_policy(rows, name)
        for name in ('baseline', 'strict', 'ranked')}
    summary['routing_diagnostics'] = dict(
        baseline_active_count=int(sum(row['baseline_active'] for row in rows)),
        baseline_silent_count=int(sum(
            not row['baseline_active'] for row in rows)),
        baseline_silent_dino_available_count=int(sum(
            not row['baseline_active']
            and row['dino_top1_metrics']['detection_count'] > 0
            for row in rows)),
        baseline_silent_dino_above_threshold_count=int(sum(
            not row['baseline_active']
            and row['dino_top1_metrics']['detection_count'] > 0
            and row['dino_top1_metrics']['top1_score']
            >= DINO_DEPLOYMENT_SCORE_THR
            for row in rows)),
        correct_dino_blocked_by_active_baseline_count=int(sum(
            row['baseline_active']
            and not row['policies']['baseline']['metrics']['top1_hit']
            and row['dino_top1_metrics']['top1_hit']
            for row in rows)))
    return summary


def non_regression_holds(summary: Dict) -> bool:
    baseline = summary['baseline']
    return all(
        item['baseline_preservation_failures'] == 0
        and item['top1_hits'] >= baseline['top1_hits']
        and item['top1_mcml'] <= baseline['top1_mcml']
        for item in (summary['strict'], summary['ranked']))


def make_decision(source_summary: Dict, target_summary: Dict) -> str:
    if not non_regression_holds(source_summary):
        return 'INVALID_SOURCE_NON_REGRESSION'
    if not non_regression_holds(target_summary):
        return 'INVALID_TARGET_BASELINE_PRESERVATION'
    strict = target_summary['strict']
    ranked = target_summary['ranked']
    if (strict['top1_hits'] >= TARGET_MIN_HITS
            and strict['top1_mcml'] <= TARGET_MAX_MCML):
        return 'STRICT_BASELINE_FIRST_DINO_RESCUE_PASSES'
    if (ranked['top1_hits'] >= TARGET_MIN_HITS
            and ranked['top1_mcml'] <= TARGET_MAX_MCML):
        return 'RANKED_RESCUE_UPPER_BOUND_ONLY'
    return 'BASELINE_FIRST_DINO_RESCUE_INSUFFICIENT'


def load_frozen_labeller(args, dino_devices, head_device):
    dino, loaded_patch_size = dino_audit.load_frozen_dinov2(
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


def main():
    args = protocol_args(parse_args())
    validate_args(args)
    labeller.set_seed(args.seed)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    source_records, target_records = source_and_target_records(args)

    # The two detectors share the 8GB head card sequentially.  Releasing the
    # baseline before loading DINO heads avoids changing either computation.
    baseline_device = torch.device('cuda:{}'.format(args.baseline_gpu))
    baseline, baseline_cfg, baseline_policy = load_baseline(
        args.baseline_config, args.baseline_checkpoint, baseline_device)
    baseline_versions = alignment.module_parameter_versions(baseline)
    source_baseline = evaluate_baseline(
        baseline, baseline_cfg, source_records, baseline_device,
        role='source_validation')
    target_baseline = evaluate_baseline(
        baseline, baseline_cfg, target_records, baseline_device,
        role='target_dev_diagnosis_only')
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
    dino, heads = load_frozen_labeller(args, dino_devices, head_device)
    dino_versions = alignment.module_parameter_versions(dino)
    head_versions = alignment.module_parameter_versions(heads)
    source_dino = labeller.evaluate_records(
        dino, heads, source_records, args, dino_device, head_device,
        role='source_validation')
    target_dino = labeller.evaluate_records(
        dino, heads, target_records, args, dino_device, head_device,
        role='target_dev_diagnosis_only')
    dino_unchanged = (
        dino_versions == alignment.module_parameter_versions(dino))
    heads_unchanged = (
        head_versions == alignment.module_parameter_versions(heads))
    if not dino_unchanged or not heads_unchanged:
        raise RuntimeError('Frozen DINO/labeller parameter invariant failed')

    source_rows = combine_rows(
        source_baseline, source_dino, source_records)
    target_rows = combine_rows(
        target_baseline, target_dino, target_records)
    source_summary = summarize_combination(source_rows)
    target_summary = summarize_combination(target_rows)
    decision = make_decision(source_summary, target_summary)

    payload = dict(
        audit=AUDIT_NAME, protocol_version=PROTOCOL_VERSION,
        baseline_config=os.path.abspath(args.baseline_config),
        baseline_checkpoint=os.path.abspath(args.baseline_checkpoint),
        baseline_checkpoint_sha256=file_sha256(args.baseline_checkpoint),
        labeller_checkpoint=os.path.abspath(args.labeller_checkpoint),
        labeller_checkpoint_sha256=file_sha256(args.labeller_checkpoint),
        dinov2_checkpoint=os.path.abspath(args.dinov2_checkpoint),
        dinov2_checkpoint_sha256=file_sha256(args.dinov2_checkpoint),
        protocol=dict(
            routing='baseline_output_if_nonempty_else_dino_top1',
            baseline_inference_config_unchanged=True,
            baseline_policy=baseline_policy,
            strict_dino_score_thr=DINO_DEPLOYMENT_SCORE_THR,
            ranked_policy=(
                'threshold_free_top1_upper_bound_for_always_present_single_'
                'target_protocol_only'),
            score_fusion=False, brightness_gate=False,
            target_dev_role='diagnosis_only',
            target_eligible_for_threshold_tuning=False,
            target_used_for_training=False,
            checkpoint_selection=False),
        isolation=dict(
            optimizer_steps=0, checkpoint_writes=0,
            baseline_frozen=True,
            baseline_parameters_unchanged=baseline_unchanged,
            dino_frozen=True, dino_parameters_unchanged=dino_unchanged,
            labeller_heads_frozen=True,
            labeller_parameters_unchanged=heads_unchanged,
            baseline_and_dino_loaded_sequentially=True),
        source_control=dict(
            summary=source_summary, rows=source_rows),
        target_dev=dict(
            summary=target_summary, rows=target_rows),
        decision=decision)
    replacements = roi_probe.write_json_atomic(args.out_json, payload)
    print('[rescue] {} baseline={}/{} strict={}/{} ranked={}/{} '
          'strict_mcml={} ranked_mcml={}'.format(
              decision,
              target_summary['baseline']['top1_hits'], len(target_rows),
              target_summary['strict']['top1_hits'], len(target_rows),
              target_summary['ranked']['top1_hits'], len(target_rows),
              target_summary['strict']['top1_mcml'],
              target_summary['ranked']['top1_mcml']))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
