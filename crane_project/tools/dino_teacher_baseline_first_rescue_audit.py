#!/usr/bin/env python3
"""Read-only baseline-first audit for the frozen DINO rotated labeller.

The BrightAug detector keeps ownership of every frame for which its unchanged
test configuration returns a detection.  The frozen DINO labeller is consulted
only when BrightAug is silent.  Three fixed rescue policies are reported:

* strict: accept DINO top-1 only when its score is at least 0.05;
* ranked: accept DINO top-1 regardless of score (diagnostic upper bound);
* confident override: use ranked rescue for baseline silence, and allow DINO
  to replace an active baseline only when DINO score is at least 0.05.

This script does not train, write checkpoints, fuse scores, or modify either
model's normal inference path.
"""

import argparse
import hashlib
import json
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


AUDIT_NAME = 'Frozen DINO Baseline-First Confident Override Audit V2'
PROTOCOL_VERSION = 2
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
    parser.add_argument(
        '--scope-manifest',
        help=('Optional external low-light mode manifest. Without it, the '
              'legacy all-enabled diagnostic policy is used.'))
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
    if args.scope_manifest and not os.path.isfile(args.scope_manifest):
        raise ValueError('Scope manifest does not exist: {}'.format(
            args.scope_manifest))


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


def load_scope_manifest(path: str, records: Sequence[Dict]):
    """Load an external mode manifest and cover every evaluated record once."""
    if path is None:
        return None
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    source = str(payload.get('scope_source', '')).strip().lower()
    target_label_derived = bool(payload.get('target_label_derived', False))
    eligible_for_final_test = bool(payload.get(
        'eligible_for_final_test', not target_label_derived))
    if source in ('target_labels', 'target_dev_labels',
                  'manual_target_tuning') and not target_label_derived:
        raise ValueError(
            'Target-derived scope must declare target_label_derived=true')
    if not source:
        raise ValueError('Scope manifest requires a non-empty scope_source')
    entries = payload.get('entries')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Scope manifest requires a non-empty entries list')
    scope = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Scope manifest entries must be objects')
        required = ('split', 'seq', 'start', 'end', 'dino_enabled')
        if any(key not in entry for key in required):
            raise ValueError('Scope manifest entry is missing a required field')
        start, end = int(entry['start']), int(entry['end'])
        if end < start:
            raise ValueError('Scope manifest entry has end before start')
        for frame in range(start, end + 1):
            key = (str(entry['split']), str(entry['seq']), int(frame))
            if key in scope:
                raise ValueError('Scope manifest has overlapping entries')
            scope[key] = bool(entry['dino_enabled'])
    expected = {(row['split'], row['seq'], int(row['frame']))
                for row in records}
    missing = sorted(expected - set(scope))
    if missing:
        raise ValueError(
            'Scope manifest must cover every evaluated record; missing={}'
            .format(missing[:1]))
    evaluated_scope = {key: bool(scope[key]) for key in expected}
    return dict(source=source, values=evaluated_scope,
                target_label_derived=target_label_derived,
                eligible_for_final_test=eligible_for_final_test,
                path=os.path.abspath(path))


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


def choose_ranked_confident_override(
        baseline: np.ndarray, dino: np.ndarray,
        dino_score_thr: float = DINO_DEPLOYMENT_SCORE_THR):
    """Ranked rescue on silence plus fixed-threshold DINO override.

    Scores from the two detectors are never compared.  An active baseline is
    replaced only when DINO independently clears its pre-declared threshold.
    When baseline is silent, the one-target protocol keeps the ranked DINO
    top-1 even if its absolute score is poorly calibrated.
    """
    if dino.shape[0] > 0 and float(dino[0, 5]) >= float(dino_score_thr):
        source = 'dino_override' if baseline.shape[0] > 0 else 'dino_rescue'
        return dino[:1].copy(), source
    if baseline.shape[0] > 0:
        return baseline.copy(), 'baseline'
    if dino.shape[0] > 0:
        return dino[:1].copy(), 'dino_rescue'
    return np.zeros((0, 6), dtype=np.float32), 'silence'


def choose_scoped_confident_override(
        baseline: np.ndarray, dino: np.ndarray,
        dino_enabled: bool,
        dino_score_thr: float = DINO_DEPLOYMENT_SCORE_THR):
    """Apply DINO only when an external low-light scope enables it.

    The scope signal must come from deployment metadata or a separately
    validated mode controller.  It must not be inferred from target labels or
    from a target-derived sequence allowlist.  Disabled scope is an exact
    BrightAug-preserving fallback.
    """
    if not bool(dino_enabled):
        if baseline.shape[0] > 0:
            return baseline.copy(), 'baseline_scope_disabled'
        return np.zeros((0, 6), dtype=np.float32), 'silence_scope_disabled'
    return choose_ranked_confident_override(
        baseline, dino, dino_score_thr=dino_score_thr)


def choose_scoped_dino_primary(
        baseline: np.ndarray, dino: np.ndarray, dino_enabled: bool):
    """Use DINO as the low-light expert and BrightAug as exact fallback.

    This policy deliberately avoids cross-model score comparison and absolute
    DINO thresholding.  It is valid only when an external, target-independent
    operating-mode signal enables the low-light expert.
    """
    if not bool(dino_enabled):
        if baseline.shape[0] > 0:
            return baseline.copy(), 'baseline_scope_disabled'
        return np.zeros((0, 6), dtype=np.float32), 'silence_scope_disabled'
    if dino.shape[0] > 0:
        return dino[:1].copy(), 'dino_primary'
    if baseline.shape[0] > 0:
        return baseline.copy(), 'baseline_fallback'
    return np.zeros((0, 6), dtype=np.float32), 'silence'


def combine_rows(baseline_rows: Sequence[Dict], dino_rows: Sequence[Dict],
                 records: Sequence[Dict], scope_by_key=None):
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
        confident_override, override_source = (
            choose_ranked_confident_override(baseline, dino))
        scope_key = (record['split'], record['seq'], int(record['frame']))
        scoped_enabled = (True if scope_by_key is None else
                           bool(scope_by_key.get(scope_key, False)))
        scoped_override, scoped_source = choose_scoped_confident_override(
            baseline, dino, scoped_enabled)
        scoped_primary, scoped_primary_source = choose_scoped_dino_primary(
            baseline, dino, scoped_enabled)
        dino_top1 = dino[:1].copy()
        baseline_active = baseline.shape[0] > 0
        strict_preserved = (
            not baseline_active or np.array_equal(strict, baseline))
        ranked_preserved = (
            not baseline_active or np.array_equal(ranked, baseline))
        override_preserved = (
            not baseline_active
            or np.array_equal(confident_override, baseline))
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
                    detections=ranked.tolist()),
                confident_override=dict(
                    source=override_source, preserved=override_preserved,
                    metrics=labeller.ranked_detection_metrics(
                        confident_override, gt, RIOU_THR,
                        DINO_DEPLOYMENT_SCORE_THR),
                    detections=confident_override.tolist()),
                scoped_override=dict(
                    source=scoped_source,
                    scope_enabled=bool(scoped_enabled),
                    preserved=(not baseline_active
                               or np.array_equal(scoped_override, baseline)),
                    metrics=labeller.ranked_detection_metrics(
                        scoped_override, gt, RIOU_THR,
                        DINO_DEPLOYMENT_SCORE_THR),
                    detections=scoped_override.tolist()),
                scoped_dino_primary=dict(
                    source=scoped_primary_source,
                    scope_enabled=bool(scoped_enabled),
                    preserved=(not baseline_active
                               or np.array_equal(scoped_primary, baseline)),
                    metrics=labeller.ranked_detection_metrics(
                        scoped_primary, gt, RIOU_THR,
                        DINO_DEPLOYMENT_SCORE_THR),
                    detections=scoped_primary.tolist()))))
    return combined


def summarize_policy(rows: Sequence[Dict], policy: str) -> Dict:
    flat = []
    scores = []
    top1_rious = []
    for row in rows:
        item = row['policies'][policy]
        metrics = item['metrics']
        flat.append(dict(
            seq=row['seq'], frame=int(row['frame']),
            hit=bool(metrics['top1_hit']),
            deployment_hit=bool(metrics['deployment_top1_hit'])))
        if metrics['top1_score'] is not None:
            scores.append(float(metrics['top1_score']))
        top1_rious.append(float(metrics['top1_riou']))
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
            row['policies'][policy]['source']
            in ('baseline', 'baseline_scope_disabled') for row in rows)),
        dino_selected_count=int(sum(
            row['policies'][policy]['source']
            in ('dino_rescue', 'dino_override') for row in rows)),
        dino_rescue_selected_count=int(sum(
            row['policies'][policy]['source'] == 'dino_rescue'
            for row in rows)),
        dino_override_selected_count=int(sum(
            row['policies'][policy]['source'] == 'dino_override'
            for row in rows)),
        dino_selected_correct_count=int(sum(
            row['policies'][policy]['source']
            in ('dino_rescue', 'dino_override')
            and row['policies'][policy]['metrics']['top1_hit']
            for row in rows)),
        dino_selected_incorrect_count=int(sum(
            row['policies'][policy]['source']
            in ('dino_rescue', 'dino_override')
            and not row['policies'][policy]['metrics']['top1_hit']
            for row in rows)),
        baseline_preservation_failures=int(sum(
            not row['policies'][policy]['preserved'] for row in rows)),
        baseline_overridden_count=int(sum(
            row['policies'][policy]['source'] == 'dino_override'
            for row in rows)),
        scope_enabled_count=int(sum(
            row['policies'][policy].get('scope_enabled') is True
            for row in rows)),
        scope_disabled_count=int(sum(
            row['policies'][policy].get('scope_enabled') is False
            for row in rows)),
        median_output_score=(float(np.median(scores)) if scores else None),
        mean_top1_riou=(float(np.mean(top1_rious))
                        if top1_rious else 0.0))


def summarize_combination(rows: Sequence[Dict]) -> Dict:
    summary = {
        name: summarize_policy(rows, name)
        for name in ('baseline', 'strict', 'ranked', 'confident_override',
                     'scoped_override', 'scoped_dino_primary')}
    dino_flat = [dict(
        seq=row['seq'], frame=int(row['frame']),
        hit=bool(row['dino_top1_metrics']['top1_hit'])) for row in rows]
    summary['dino_top1'] = dict(
        frame_count=len(rows),
        top1_hits=int(sum(item['hit'] for item in dino_flat)),
        top1_mcml=labeller.longest_miss(dino_flat, 'hit'),
        mean_top1_riou=(float(np.mean([
            row['dino_top1_metrics']['top1_riou'] for row in rows]))
                        if rows else 0.0))
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
            for row in rows)),
        confident_dino_override_count=int(sum(
            row['policies']['confident_override']['source'] == 'dino_override'
            for row in rows)),
        baseline_correct_overridden_to_incorrect_count=int(sum(
            row['policies']['baseline']['metrics']['top1_hit']
            and row['policies']['confident_override']['source']
            == 'dino_override'
            and not row['policies']['confident_override']['metrics']['top1_hit']
            for row in rows)),
        baseline_incorrect_overridden_to_correct_count=int(sum(
            row['baseline_active']
            and not row['policies']['baseline']['metrics']['top1_hit']
            and row['policies']['confident_override']['source']
            == 'dino_override'
            and row['policies']['confident_override']['metrics']['top1_hit']
            for row in rows)),
        scoped_enabled_count=int(sum(
            row['policies']['scoped_override']['scope_enabled']
            for row in rows)),
        scoped_dino_override_count=int(sum(
            row['policies']['scoped_override']['source'] == 'dino_override'
            for row in rows)),
        scoped_baseline_correct_overridden_to_incorrect_count=int(sum(
            row['policies']['baseline']['metrics']['top1_hit']
            and row['policies']['scoped_override']['source'] == 'dino_override'
            and not row['policies']['scoped_override']['metrics']['top1_hit']
            for row in rows)),
        scoped_baseline_incorrect_overridden_to_correct_count=int(sum(
            row['baseline_active']
            and not row['policies']['baseline']['metrics']['top1_hit']
            and row['policies']['scoped_override']['source'] == 'dino_override'
            and row['policies']['scoped_override']['metrics']['top1_hit']
            for row in rows)),
        scoped_primary_dino_selected_count=int(sum(
            row['policies']['scoped_dino_primary']['source'] == 'dino_primary'
            for row in rows)),
        scoped_primary_baseline_fallback_count=int(sum(
            row['policies']['scoped_dino_primary']['source']
            == 'baseline_fallback' for row in rows)),
        scoped_primary_baseline_correct_to_incorrect_count=int(sum(
            row['policies']['baseline']['metrics']['top1_hit']
            and row['policies']['scoped_dino_primary']['source']
            == 'dino_primary'
            and not row['policies']['scoped_dino_primary']['metrics'][
                'top1_hit']
            for row in rows)),
        scoped_primary_baseline_incorrect_to_correct_count=int(sum(
            row['baseline_active']
            and not row['policies']['baseline']['metrics']['top1_hit']
            and row['policies']['scoped_dino_primary']['source']
            == 'dino_primary'
            and row['policies']['scoped_dino_primary']['metrics']['top1_hit']
            for row in rows)),
        dino_primary_baseline_correct_to_incorrect_count=int(sum(
            row['policies']['baseline']['metrics']['top1_hit']
            and not row['dino_top1_metrics']['top1_hit']
            for row in rows)))
    return summary


def baseline_preserving_non_regression_holds(summary: Dict) -> bool:
    baseline = summary['baseline']
    return all(
        item['baseline_preservation_failures'] == 0
        and item['top1_hits'] >= baseline['top1_hits']
        and item['top1_mcml'] <= baseline['top1_mcml']
        for item in (summary['strict'], summary['ranked']))


def confident_override_non_regression_holds(summary: Dict) -> bool:
    baseline = summary['baseline']
    override = summary['confident_override']
    return (override['top1_hits'] >= baseline['top1_hits']
            and override['top1_mcml'] <= baseline['top1_mcml']
            and override['mean_top1_riou'] >= baseline['mean_top1_riou']
            and summary['routing_diagnostics'][
                'baseline_correct_overridden_to_incorrect_count'] == 0)


def scoped_primary_non_regression_holds(summary: Dict) -> bool:
    baseline = summary['baseline']
    primary = summary['dino_top1']
    return (primary['top1_hits'] >= baseline['top1_hits']
            and primary['top1_mcml'] <= baseline['top1_mcml']
            and primary['mean_top1_riou'] >= baseline['mean_top1_riou']
            and summary['routing_diagnostics'][
                'dino_primary_baseline_correct_to_incorrect_count'] == 0)


def make_scoped_primary_decision(source_summary: Dict,
                                 target_summary: Dict,
                                 scope_manifest_applied: bool,
                                 scope_eligible_for_final_test: bool = True
                                 ) -> str:
    if not scope_manifest_applied:
        return 'SCOPE_SIGNAL_NOT_SUPPLIED'
    if not scoped_primary_non_regression_holds(source_summary):
        return 'INVALID_SOURCE_SCOPED_DINO_PRIMARY_REGRESSION'
    primary = target_summary['scoped_dino_primary']
    if (primary['top1_hits'] >= TARGET_MIN_HITS
            and primary['top1_mcml'] <= TARGET_MAX_MCML):
        if not scope_eligible_for_final_test:
            return 'TARGET_AWARE_SCOPE_DIAGNOSIS_ONLY'
        return 'SCOPED_DINO_PRIMARY_LOW_LIGHT_EXPERT_PASSES'
    return 'SCOPED_DINO_PRIMARY_LOW_LIGHT_EXPERT_INSUFFICIENT'


def make_decision(source_summary: Dict, target_summary: Dict) -> str:
    if not baseline_preserving_non_regression_holds(source_summary):
        return 'INVALID_SOURCE_NON_REGRESSION'
    if not baseline_preserving_non_regression_holds(target_summary):
        return 'INVALID_TARGET_BASELINE_PRESERVATION'
    if not confident_override_non_regression_holds(source_summary):
        return 'INVALID_SOURCE_CONFIDENT_OVERRIDE_REGRESSION'
    strict = target_summary['strict']
    ranked = target_summary['ranked']
    override = target_summary['confident_override']
    if (strict['top1_hits'] >= TARGET_MIN_HITS
            and strict['top1_mcml'] <= TARGET_MAX_MCML):
        return 'STRICT_BASELINE_FIRST_DINO_RESCUE_PASSES'
    if (override['top1_hits'] >= TARGET_MIN_HITS
            and override['top1_mcml'] <= TARGET_MAX_MCML):
        return 'CONFIDENT_DINO_OVERRIDE_CANDIDATE_PASSES'
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
    scope_manifest = load_scope_manifest(
        args.scope_manifest, list(source_records) + list(target_records))
    scope_by_key = (None if scope_manifest is None
                    else scope_manifest['values'])

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
        source_baseline, source_dino, source_records, scope_by_key)
    target_rows = combine_rows(
        target_baseline, target_dino, target_records, scope_by_key)
    source_summary = summarize_combination(source_rows)
    target_summary = summarize_combination(target_rows)
    decision = make_decision(source_summary, target_summary)
    scoped_primary_decision = make_scoped_primary_decision(
        source_summary, target_summary, scope_manifest is not None,
        scope_eligible_for_final_test=(
            False if scope_manifest is None else
            scope_manifest['eligible_for_final_test']))

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
            routing='fixed_policies_plus_external_scope_policies',
            baseline_inference_config_unchanged=True,
            baseline_policy=baseline_policy,
            strict_dino_score_thr=DINO_DEPLOYMENT_SCORE_THR,
            ranked_policy=(
                'threshold_free_top1_upper_bound_for_always_present_single_'
                'target_protocol_only'),
            confident_override_policy=dict(
                baseline_silent='ranked_dino_top1',
                baseline_active=(
                    'dino_top1_only_if_dino_score_at_least_0.05_else_baseline'),
                cross_model_score_comparison=False,
                threshold_source='predeclared_labeller_deployment_threshold',
                requires_source_non_regression=True),
            scoped_dino_primary_policy=(
                'external_scope_enabled_dino_top1_else_baseline_fallback'),
            scoped_override_policy=(
                'external_scope_enabled_confident_override'),
            score_fusion=False, brightness_gate=False,
            target_dev_role='diagnosis_only',
            target_dev_informed_method_development=True,
            target_dev_eligible_for_unbiased_final_test=False,
            target_eligible_for_threshold_tuning=False,
            target_used_for_training=False,
            checkpoint_selection=False),
        scope_gate=(dict(
            enabled=True,
            source=(scope_manifest['source'] if scope_manifest else
                    'legacy_all_records_enabled'),
            manifest=(None if scope_manifest is None else dict(
                path=scope_manifest['path'],
                sha256=file_sha256(args.scope_manifest))),
            target_label_derived=(False if scope_manifest is None else
                                  scope_manifest['target_label_derived']),
            eligible_for_final_test=(False if scope_manifest is None else
                                     scope_manifest[
                                         'eligible_for_final_test']))),
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
        decision=decision,
        scoped_primary_decision=scoped_primary_decision)
    replacements = roi_probe.write_json_atomic(args.out_json, payload)
    print('[rescue] {} baseline={}/{} strict={}/{} ranked={}/{} '
          'override={}/{} scoped_primary={}/{} strict_mcml={} '
          'ranked_mcml={} override_mcml={} scoped_primary_mcml={}'
          .format(
              decision,
              target_summary['baseline']['top1_hits'], len(target_rows),
              target_summary['strict']['top1_hits'], len(target_rows),
              target_summary['ranked']['top1_hits'], len(target_rows),
              target_summary['confident_override']['top1_hits'],
              len(target_rows),
              target_summary['scoped_dino_primary']['top1_hits'],
              len(target_rows),
              target_summary['strict']['top1_mcml'],
              target_summary['ranked']['top1_mcml'],
              target_summary['confident_override']['top1_mcml'],
              target_summary['scoped_dino_primary']['top1_mcml']))
    print('[json] nonfinite_replacements={}'.format(replacements))
    print('[scope] {}'.format(scoped_primary_decision))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
