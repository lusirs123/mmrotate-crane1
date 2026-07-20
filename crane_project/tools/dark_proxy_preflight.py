#!/usr/bin/env python3
"""Audit target-dev failure signatures and dark proxy suitability.

The local dataset roles are explicit because the project uses video sequences
and its local ``test`` split is in practice a target-domain development set.
``real_seq02`` may be used as labelled ``target_dev``; ``real_seq03`` and
``sim_seq09`` remain ``target_holdout`` and are never eligible for model
selection.  The tool measures whether a source-validation proxy reproduces:

* a non-trivial, temporally continuous main-branch silence rate;
* deep classification rank for otherwise usable candidates;
* high pre-threshold top-10000 geometric oracle recall.

Passing this audit only authorizes a frozen P1-A probe.  Outputs record whether
target-domain labels influenced the decision and therefore whether a run can
still be called zero-shot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import candidate_pool_oracle_probe as pool_probe  # noqa: E402
from crane_project.tools import ctx_entry_probe as entry_probe  # noqa: E402
from crane_project.utils.dark_degradation import (  # noqa: E402
    SUPPORTED_DARK_FAMILIES,
    SUPPORTED_TEMPORAL_PROFILES,
    apply_dark_degradation,
)


DATA_ROLE_RULES = {
    'source_val': dict(split='val', sequences=('real_seq07', 'sim_seq10')),
    'target_dev': dict(split='test', sequences=('real_seq02',)),
    'target_holdout': dict(
        split='test', sequences=('real_seq03', 'sim_seq09')),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit source-val proxies against an explicit target-dev '
                    'video signature while protecting target holdout videos.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--role', required=True,
                        choices=list(DATA_ROLE_RULES))
    parser.add_argument('--split', default='val',
                        choices=['val', 'test', 'train', 'train_sim'])
    parser.add_argument('--seq', required=True)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--focus-start', type=int, default=None,
                        help='Target-dev difficulty window start. All sequence '
                             'frames are still processed.')
    parser.add_argument('--focus-end', type=int, default=None)
    parser.add_argument('--reference-only', action='store_true',
                        help='Measure the clean sequence only. Required for '
                             'target_dev and target_holdout roles.')
    parser.add_argument('--reference-json', default=None,
                        help='A full target_dev reference-only JSON. Source '
                             'proxy gates are matched against its focus '
                             'signature and become target-aware.')
    parser.add_argument('--confirm-frozen-holdout', action='store_true',
                        help='Required for target_holdout. The output remains '
                             'ineligible for model selection.')
    parser.add_argument('--allow-train-proxy', action='store_true',
                        help='Allow train/train_sim for a smoke test. Such a '
                             'run is never protocol-ready.')
    parser.add_argument('--families', nargs='+',
                        default=list(SUPPORTED_DARK_FAMILIES),
                        choices=list(SUPPORTED_DARK_FAMILIES))
    parser.add_argument('--severities', type=float, nargs='+',
                        default=[0.55, 0.75, 0.95])
    parser.add_argument('--temporal-profile', default='ramp-plateau',
                        choices=list(SUPPORTED_TEMPORAL_PROFILES))
    parser.add_argument('--candidate-source', default='main', choices=['main'])
    parser.add_argument('--topks', type=int, nargs='+',
                        default=[1, 100, 1000, 10000])
    parser.add_argument('--pool-size', type=int, default=10000)
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--score-thr', type=float, default=0.05,
                        help='Original main score threshold used to define '
                             'main-branch silence.')
    parser.add_argument('--min-silence-rate', type=float, default=0.20)
    parser.add_argument('--max-silence-rate', type=float, default=0.60)
    parser.add_argument('--min-silent-run', type=int, default=5)
    parser.add_argument('--min-pool-oracle-recall', type=float, default=0.80)
    parser.add_argument('--min-oracle-retention', type=float, default=0.80)
    parser.add_argument('--min-dense-riou-retention', type=float, default=0.80)
    parser.add_argument('--min-rank-ratio', type=float, default=10.0)
    parser.add_argument('--max-target-silence-gap', type=float, default=0.15)
    parser.add_argument('--min-target-silent-run-ratio', type=float,
                        default=0.25)
    parser.add_argument('--min-target-rank-ratio', type=float, default=0.10)
    parser.add_argument('--max-target-rank-ratio', type=float, default=10.0)
    parser.add_argument('--max-target-pool-recall-gap', type=float,
                        default=0.15)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--preview-dir', default=None)
    parser.add_argument('--preview-count', type=int, default=3)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def normalize_topks(values: Sequence[int], pool_size: int) -> List[int]:
    topks = {int(value) for value in values if int(value) > 0}
    topks.add(int(pool_size))
    result = sorted(topks)
    if not result or pool_size <= 0:
        raise ValueError('topks and pool_size must be positive')
    return result


def validate_data_role(data_root: str,
                       role: str,
                       split: str,
                       seq: str,
                       allow_train_proxy: bool = False,
                       reference_only: bool = False,
                       confirm_frozen_holdout: bool = False) -> Dict:
    """Validate the video-level role and return its reporting policy."""
    if role not in DATA_ROLE_RULES:
        raise ValueError(f'Unknown data role: {role}')
    split_lower = str(split).lower()
    seq_lower = str(seq).lower()
    root_parts = {
        part.lower() for part in os.path.realpath(data_root).split(os.sep)
        if part
    }
    rule = DATA_ROLE_RULES[role]
    protocol_source = True

    target_sequences = set(
        DATA_ROLE_RULES['target_dev']['sequences']
        + DATA_ROLE_RULES['target_holdout']['sequences'])
    if role == 'source_val' and seq_lower in target_sequences:
        raise ValueError(
            f'source_val cannot use target sequence {seq}')

    if role == 'source_val' and split_lower in ('train', 'train_sim'):
        if not allow_train_proxy:
            raise ValueError(
                'source_val protocol requires --split val. Use '
                '--allow-train-proxy only for a non-authorizing smoke test.')
        protocol_source = False
    else:
        if split_lower != rule['split']:
            raise ValueError(
                f'role={role} requires --split {rule["split"]}, got {split}')
        if seq_lower not in rule['sequences']:
            raise ValueError(
                f'role={role} permits only {rule["sequences"]}, got {seq}')

    if role == 'source_val' and ('test' in root_parts or split_lower == 'test'):
        raise ValueError('source_val must not resolve to a TEST directory')
    if role in ('target_dev', 'target_holdout') and not reference_only:
        raise ValueError(f'role={role} requires --reference-only')
    if role == 'target_holdout' and not confirm_frozen_holdout:
        raise ValueError(
            'target_holdout requires --confirm-frozen-holdout and remains '
            'ineligible for model selection')

    return dict(
        role=role,
        protocol_source=bool(protocol_source),
        uses_target_domain=role != 'source_val',
        uses_target_labels=role != 'source_val',
        eligible_for_model_selection=role in ('source_val', 'target_dev'),
        zero_shot_compliant=role == 'source_val',
    )


def validate_args(args) -> Tuple[List[int], Dict]:
    if args.seed != 0:
        raise ValueError('The unified experiment protocol requires --seed 0')
    if (args.start is None) ^ (args.end is None):
        raise ValueError('--start and --end must be provided together')
    if args.start is not None and args.end < args.start:
        raise ValueError('--end must be greater than or equal to --start')
    if (args.focus_start is None) ^ (args.focus_end is None):
        raise ValueError(
            '--focus-start and --focus-end must be provided together')
    if (args.focus_start is not None
            and args.focus_end < args.focus_start):
        raise ValueError(
            '--focus-end must be greater than or equal to --focus-start')
    if not 0.0 <= args.riou_thr <= 1.0:
        raise ValueError('--riou-thr must be in [0, 1]')
    if args.score_thr < 0.0:
        raise ValueError('--score-thr must be non-negative')
    if not 0.0 <= args.min_silence_rate <= args.max_silence_rate <= 1.0:
        raise ValueError('invalid silence-rate gate')
    if args.min_silent_run < 0:
        raise ValueError('--min-silent-run must be non-negative')
    if not args.reference_only:
        for severity in args.severities:
            if not 0.0 < severity <= 1.0:
                raise ValueError('--severities must be in (0, 1]')
        if len(set(args.families)) < 2:
            raise ValueError(
                'Source proxy preflight requires at least two independent '
                'families')
    if args.role == 'target_dev' and args.focus_start is None:
        raise ValueError(
            'target_dev requires an explicit full-video difficulty focus '
            'window, e.g. --focus-start 137 --focus-end 169')
    if args.role != 'target_dev' and args.focus_start is not None:
        raise ValueError('focus windows are defined only for target_dev')
    if args.role != 'source_val' and args.reference_json:
        raise ValueError('--reference-json is accepted only for source_val')
    topks = normalize_topks(args.topks, args.pool_size)
    role_policy = validate_data_role(
        args.data_root, args.role, args.split, args.seq,
        args.allow_train_proxy, args.reference_only,
        args.confirm_frozen_holdout)
    return topks, role_policy


def discover_frame_ids(args) -> Tuple[List[int], bool]:
    diag = entry_probe.get_diag()
    all_frame_ids = diag.discover_frames(
        args.data_root, args.split, args.seq, max_count=None)
    frame_ids = list(all_frame_ids)
    if args.start is not None:
        frame_ids = [
            frame for frame in frame_ids
            if args.start <= frame <= args.end
        ]
    if not frame_ids:
        raise RuntimeError(
            f'No proxy frames found for {args.split}/{args.seq}')
    return frame_ids, bool(frame_ids == all_frame_ids)


def build_manifest(data_root: str, split: str, seq: str,
                   frame_ids: Sequence[int], role: str) -> Dict:
    diag = entry_probe.get_diag()
    entries = []
    for frame in frame_ids:
        img_path, ann_path = diag.find_files(data_root, split, seq, frame)
        if img_path is None or ann_path is None:
            raise RuntimeError(
                f'Missing proxy image/annotation for {seq}_{frame:05d}')
        real_img = os.path.realpath(img_path)
        real_ann = os.path.realpath(ann_path)
        image_is_test = any(
            part.lower() == 'test' for part in real_img.split(os.sep))
        ann_is_test = any(
            part.lower() == 'test' for part in real_ann.split(os.sep))
        if role == 'source_val' and (image_is_test or ann_is_test):
            raise ValueError(
                f'source_val resolved to forbidden TEST data: {real_img}')
        if role in ('target_dev', 'target_holdout') \
                and not (image_is_test and ann_is_test):
            raise ValueError(
                f'role={role} must resolve both image and annotation from '
                f'the local TEST directory: image={real_img}, ann={real_ann}')
        entries.append(dict(
            frame=int(frame),
            image=os.path.relpath(real_img, os.path.realpath(data_root)),
            annotation=os.path.relpath(real_ann, os.path.realpath(data_root)),
            image_size=int(os.path.getsize(real_img)),
            annotation_size=int(os.path.getsize(real_ann)),
        ))
    serialized = json.dumps(entries, sort_keys=True).encode('utf-8')
    return dict(
        split=split,
        seq=seq,
        role=role,
        frames=len(entries),
        first_frame=int(frame_ids[0]),
        last_frame=int(frame_ids[-1]),
        sha256=hashlib.sha256(serialized).hexdigest(),
        entries=entries,
    )


def preprocess_bgr_array(image_bgr, filename, transform_compose,
                         img_scale, flip):
    """Run the exact configured test transforms on an in-memory BGR image."""
    diag = entry_probe.get_diag()
    results = dict(
        img=image_bgr.copy(),
        filename=filename,
        ori_filename=os.path.basename(filename),
        img_shape=image_bgr.shape,
        ori_shape=image_bgr.shape,
        scale=tuple(img_scale),
        flip=flip,
        flip_direction='horizontal' if flip else None,
        img_fields=['img'],
    )
    results = transform_compose(results)
    image_tensor = diag._unwrap_pipeline_value(results['img'])
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.from_numpy(image_tensor)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    meta = diag._unwrap_pipeline_value(results.get('img_metas'))
    if not isinstance(meta, dict):
        raise RuntimeError('test pipeline did not return dict img_metas')
    meta = dict(meta)
    meta.setdefault('filename', filename)
    meta.setdefault('ori_filename', os.path.basename(filename))
    diag._validate_preprocess_metadata(image_tensor, meta, img_scale)
    return image_tensor, meta


def _longest_run(rows: Sequence[Dict], key: str, expected: bool = True) -> int:
    longest = 0
    current = 0
    previous = None
    for row in sorted(rows, key=lambda item: int(item['frame'])):
        frame = int(row['frame'])
        if previous is None or frame != previous + 1:
            current = 0
        if bool(row[key]) == bool(expected):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
        previous = frame
    return int(longest)


def analyze_variant(model, image_bgr, meta_source: Dict, gt_ori: Dict,
                    transform_compose, img_scale, flip, args,
                    topks: Sequence[int]) -> Dict:
    from mmcv.ops import box_iou_rotated
    import cv2

    image_tensor, meta = preprocess_bgr_array(
        image_bgr, meta_source['img_path'], transform_compose,
        img_scale, flip)
    image_tensor = image_tensor.cuda(f'cuda:{args.gpu}')

    with torch.no_grad():
        features = model.extract_feat(image_tensor)
        candidate_head, cls_scores, bbox_preds = (
            entry_probe.forward_candidate_head(
                model, features, args.candidate_source))
        boxes, scores, levels, _, alignment = (
            entry_probe.flatten_decode_candidates(
                candidate_head, cls_scores, bbox_preds, meta['img_shape']))
        gt = pool_probe.scale_gt_to_img(gt_ori, meta)
        gt_box = entry_probe.gt_to_tensor(gt, boxes.device)
        ious = box_iou_rotated(
            boxes.float(), gt_box.float()).reshape(-1)

        candidate_count = int(scores.numel())
        max_k = min(max(topks), candidate_count)
        top_scores, top_indices = torch.topk(
            scores, k=max_k, largest=True, sorted=True)
        top_ious = ious[top_indices]
        dense_best_riou = float(ious.max().item())
        usable = ious >= float(args.riou_thr)
        if bool(usable.any()):
            usable_scores = scores[usable]
            usable_best_score = float(usable_scores.max().item())
            usable_best_rank = int(
                (scores > usable_scores.max()).sum().item()) + 1
        else:
            usable_best_score = None
            usable_best_rank = None

        per_k = {}
        for topk in topks:
            actual_k = min(int(topk), candidate_count)
            best_riou = float(top_ious[:actual_k].max().item())
            per_k[str(topk)] = dict(
                actual_k=int(actual_k),
                oracle_hit=bool(best_riou >= args.riou_thr),
                best_riou=best_riou,
            )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    scores_over_thr = int((scores > float(args.score_thr)).sum().item())
    return dict(
        candidate_count=candidate_count,
        candidates_over_score_thr=scores_over_thr,
        main_silent=bool(scores_over_thr == 0),
        global_max=float(scores.max().item()),
        top1_score=float(top_scores[0].item()),
        top1_riou=float(top_ious[0].item()),
        dense_best_riou=dense_best_riou,
        dense_oracle_hit=bool(dense_best_riou >= args.riou_thr),
        usable_best_score=usable_best_score,
        usable_best_rank=usable_best_rank,
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        per_k=per_k,
        preprocess=dict(
            ori_shape=list(meta['ori_shape']),
            img_shape=list(meta['img_shape']),
            pad_shape=list(meta['pad_shape']),
            scale_factor=np.asarray(
                meta['scale_factor'], dtype=np.float64).reshape(-1).tolist(),
        ),
        decode_alignment=alignment,
    )


def _numeric_stats(values: Sequence[float]) -> Dict:
    values = [float(value) for value in values if value is not None]
    if not values:
        return dict(count=0, mean=None, median=None, p90=None, min=None,
                    max=None)
    array = np.asarray(values, dtype=np.float64)
    return dict(
        count=int(array.size),
        mean=float(array.mean()),
        median=float(np.median(array)),
        p90=float(np.percentile(array, 90)),
        min=float(array.min()),
        max=float(array.max()),
    )


def summarize_variant(rows: Sequence[Dict], topks: Sequence[int],
                      riou_thr: float) -> Dict:
    total = len(rows)
    silent = sum(bool(row['main_silent']) for row in rows)
    top1_rows = [
        dict(frame=row['frame'], top1_hit=float(row['top1_riou']) >= riou_thr)
        for row in rows
    ]
    top1_hits = sum(bool(row['top1_hit']) for row in top1_rows)
    summary = dict(
        frames=total,
        riou_thr=float(riou_thr),
        silent_frames=int(silent),
        silence_rate=float(silent / total) if total else 0.0,
        longest_silent_run=_longest_run(rows, 'main_silent') if rows else 0,
        top1_hits=int(top1_hits),
        top1_recall=float(top1_hits / total) if total else 0.0,
        top1_mcml=_longest_run(
            [dict(frame=row['frame'], miss=not row['top1_hit'])
             for row in top1_rows], 'miss') if rows else 0,
        dense_oracle_hits=sum(bool(row['dense_oracle_hit']) for row in rows),
        dense_oracle_recall=(
            sum(bool(row['dense_oracle_hit']) for row in rows) / total
            if total else 0.0),
        dense_best_riou=_numeric_stats(
            [row['dense_best_riou'] for row in rows]),
        usable_rank=_numeric_stats(
            [row['usable_best_rank'] for row in rows]),
        global_max=_numeric_stats([row['global_max'] for row in rows]),
        brightness=_numeric_stats([row['brightness'] for row in rows]),
        per_k={},
    )
    for topk in topks:
        key = str(topk)
        hits = sum(bool(row['per_k'][key]['oracle_hit']) for row in rows)
        summary['per_k'][key] = dict(
            hits=int(hits),
            misses=int(total - hits),
            recall=float(hits / total) if total else 0.0,
            oracle_mcml=_longest_run(
                [dict(frame=row['frame'], miss=not row['per_k'][key]['oracle_hit'])
                 for row in rows], 'miss') if rows else 0,
            best_riou=_numeric_stats(
                [row['per_k'][key]['best_riou'] for row in rows]),
        )
    return summary


def load_target_reference(path: str, pool_size: int,
                          config: str, checkpoint: str) -> Tuple[Dict, Dict]:
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get('data_role') != 'target_dev':
        raise ValueError('--reference-json must have data_role=target_dev')
    if not payload.get('reference_only'):
        raise ValueError('--reference-json must be a reference-only run')
    if not payload.get('full_sequence'):
        raise ValueError('--reference-json must cover the full target-dev video')
    if not payload.get('reference_signature_ready'):
        raise ValueError('target-dev reference signature is not ready')
    reference = payload.get('reference_signature')
    if not isinstance(reference, dict):
        raise ValueError('target-dev reference signature is missing')
    if str(pool_size) not in reference.get('per_k', {}):
        raise ValueError(
            f'target-dev reference does not contain pool_size={pool_size}')
    if os.path.basename(payload.get('config', '')) != os.path.basename(config):
        raise ValueError('target-dev reference config does not match this run')
    if os.path.basename(payload.get('checkpoint', '')) \
            != os.path.basename(checkpoint):
        raise ValueError(
            'target-dev reference checkpoint does not match this run')
    return reference, payload


def _plateau_rows(rows: Sequence[Dict]) -> List[Dict]:
    selected = []
    for row in rows:
        degradation = row.get('degradation', {})
        severity = float(degradation.get('severity', 0.0))
        strength = float(degradation.get('strength', 0.0))
        if severity > 0.0 and strength >= 0.95 * severity:
            selected.append(row)
    return selected


def _rows_in_window(rows: Sequence[Dict], start: int,
                    end: int) -> List[Dict]:
    return [
        row for row in rows if start <= int(row['frame']) <= end
    ]


def _rows_for_frames(rows: Sequence[Dict], frames: Sequence[int]) -> List[Dict]:
    frame_set = {int(frame) for frame in frames}
    return [row for row in rows if int(row['frame']) in frame_set]


def evaluate_gate(summary: Dict, clean: Dict, pool_size: int, args,
                  target_reference: Optional[Dict] = None) -> Dict:
    key = str(pool_size)
    pool_recall = float(summary['per_k'][key]['recall'])
    clean_pool_recall = float(clean['per_k'][key]['recall'])
    oracle_retention = (
        pool_recall / clean_pool_recall if clean_pool_recall > 0.0 else 0.0)
    dense_mean = summary['dense_best_riou']['mean'] or 0.0
    clean_dense_mean = clean['dense_best_riou']['mean'] or 0.0
    dense_riou_retention = (
        dense_mean / clean_dense_mean if clean_dense_mean > 0.0 else 0.0)
    rank_median = summary['usable_rank']['median']
    clean_rank_median = clean['usable_rank']['median']
    rank_ratio = (
        float(rank_median) / max(float(clean_rank_median), 1.0)
        if rank_median is not None and clean_rank_median is not None else 0.0)

    checks = dict(
        pool_oracle_recall=pool_recall >= args.min_pool_oracle_recall,
        oracle_retention=oracle_retention >= args.min_oracle_retention,
        dense_riou_retention=(
            dense_riou_retention >= args.min_dense_riou_retention),
        rank_shift=rank_ratio >= args.min_rank_ratio,
    )
    target_match = None
    if target_reference is None:
        checks.update(
            silence_rate=(
                args.min_silence_rate <= summary['silence_rate']
                <= args.max_silence_rate),
            silent_run=summary['longest_silent_run'] >= args.min_silent_run,
        )
    else:
        target_pool_recall = float(
            target_reference['per_k'][key]['recall'])
        target_silence = float(target_reference['silence_rate'])
        target_run = int(target_reference['longest_silent_run'])
        target_rank = target_reference['usable_rank']['median']
        silence_gap = abs(float(summary['silence_rate']) - target_silence)
        pool_recall_gap = abs(pool_recall - target_pool_recall)
        silent_run_ratio = (
            float(summary['longest_silent_run']) / float(target_run)
            if target_run > 0 else 1.0)
        target_rank_ratio = (
            float(rank_median) / max(float(target_rank), 1.0)
            if rank_median is not None and target_rank is not None else 0.0)
        checks.update(
            target_silence=(
                silence_gap <= args.max_target_silence_gap),
            target_silent_run=(
                silent_run_ratio >= args.min_target_silent_run_ratio),
            target_pool_recall=(
                pool_recall_gap <= args.max_target_pool_recall_gap),
            target_rank=(
                args.min_target_rank_ratio <= target_rank_ratio
                <= args.max_target_rank_ratio),
        )
        target_match = dict(
            target_silence_rate=target_silence,
            silence_gap=float(silence_gap),
            target_longest_silent_run=target_run,
            silent_run_ratio=float(silent_run_ratio),
            target_pool_oracle_recall=target_pool_recall,
            pool_recall_gap=float(pool_recall_gap),
            target_usable_rank_median=target_rank,
            target_rank_ratio=float(target_rank_ratio),
        )
    return dict(
        passed=bool(all(checks.values())),
        checks=checks,
        pool_size=int(pool_size),
        pool_oracle_recall=pool_recall,
        clean_pool_oracle_recall=clean_pool_recall,
        oracle_retention=float(oracle_retention),
        dense_riou_retention=float(dense_riou_retention),
        rank_ratio=float(rank_ratio),
        target_match=target_match,
    )


def _variant_name(family: str, severity: Optional[float] = None) -> str:
    if family == 'clean':
        return 'clean'
    return f'{family}_s{severity:.2f}'.replace('.', 'p')


def print_variant(name: str, summary: Dict, gate: Optional[Dict],
                  gate_summary: Optional[Dict] = None):
    display = (
        gate_summary['variant'] if gate_summary is not None else summary)
    scope = 'plateau' if gate_summary is not None else 'all'
    rank = display['usable_rank']
    pool_key = max(display['per_k'], key=lambda value: int(value))
    pool = display['per_k'][pool_key]
    status = '-' if gate is None else ('PASS' if gate['passed'] else 'FAIL')
    print(
        f'{name:<24} scope={scope:<7} '
        f'silence={display["silence_rate"]:6.1%} '
        f'run={display["longest_silent_run"]:3d} '
        f'top1={display["top1_recall"]:6.1%} '
        f'top1_MCML={display["top1_mcml"]:3d} '
        f'R@{pool_key}={pool["recall"]:6.1%} '
        f'dense={display["dense_oracle_recall"]:6.1%} '
        f'rank_med={str(rank["median"]):>8} {status}')


def main():
    args = parse_args()
    topks, role_policy = validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_reference = None
    target_reference_payload = None
    if args.reference_json:
        target_reference, target_reference_payload = load_target_reference(
            args.reference_json, args.pool_size,
            args.config, args.checkpoint)
        role_policy['uses_target_domain'] = True
        role_policy['uses_target_labels'] = True
        role_policy['zero_shot_compliant'] = False

    import cv2

    model, cfg = entry_probe.load_model(
        args.config, args.checkpoint, args.gpu)
    diag = entry_probe.get_diag()
    transform_compose, img_scale, flip = diag.build_test_transforms(cfg)
    frame_ids, full_sequence = discover_frame_ids(args)
    manifest = build_manifest(
        args.data_root, args.split, args.seq, frame_ids, args.role)
    trajectory_start = int(frame_ids[0])
    trajectory_end = int(frame_ids[-1])

    variants = [('clean', None)]
    if not args.reference_only:
        variants.extend([
            (family, float(severity))
            for family in args.families
            for severity in args.severities
        ])
    rows_by_variant = {
        _variant_name(family, severity): []
        for family, severity in variants
    }

    if args.preview_dir:
        os.makedirs(args.preview_dir, exist_ok=True)

    print('=' * 100)
    print('DARK PROXY / TARGET-DEV VIDEO ROLE AUDIT')
    print('=' * 100)
    print(f'role:       {args.role}')
    print(f'data:       {args.split}/{args.seq}')
    print(f'frames:     {trajectory_start}..{trajectory_end} ({len(frame_ids)})')
    print(f'full_seq:   {full_sequence}')
    print(f'manifest:   {manifest["sha256"]}')
    print(f'ref_only:   {args.reference_only}')
    print(f'reference:  {args.reference_json or "-"}')
    print(f'families:   {args.families if not args.reference_only else "-"}')
    print(f'severities: {args.severities if not args.reference_only else "-"}')
    print(f'profile:    {args.temporal_profile}')
    print(f'topks:      {topks}')

    for index, frame in enumerate(frame_ids):
        img_path, ann_path = diag.find_files(
            args.data_root, args.split, args.seq, frame)
        raw = cv2.imread(img_path)
        if raw is None:
            raise RuntimeError(f'Failed to read {img_path}')
        gts = diag.parse_dota_ann(ann_path)
        if not gts:
            raise RuntimeError(f'No GT in proxy annotation {ann_path}')
        if len(gts) > 1:
            print(f'[warn] {args.seq}_{frame:05d}: using first of {len(gts)} GTs')

        for family, severity in variants:
            name = _variant_name(family, severity)
            if family == 'clean':
                image = raw
                degradation = dict(
                    family='clean', severity=0.0, strength=0.0,
                    geometry_preserving=True)
            else:
                image, degradation = apply_dark_degradation(
                    raw, family=family, sequence=args.seq, frame=frame,
                    start=trajectory_start, end=trajectory_end,
                    severity=severity, seed=args.seed,
                    profile=args.temporal_profile)
            row = analyze_variant(
                model, image, dict(img_path=img_path), gts[0],
                transform_compose, img_scale, flip, args, topks)
            row.update(
                frame=int(frame),
                fname=f'{args.seq}_{frame:05d}',
                split=args.split,
                seq=args.seq,
                variant=name,
                degradation=degradation,
            )
            rows_by_variant[name].append(row)

            if args.preview_dir and index < args.preview_count:
                variant_dir = os.path.join(args.preview_dir, name)
                os.makedirs(variant_dir, exist_ok=True)
                cv2.imwrite(os.path.join(
                    variant_dir, f'{args.seq}_{frame:05d}.jpg'), image)

        if index == 0 or (index + 1) % 20 == 0 or index + 1 == len(frame_ids):
            print(f'[progress] {index + 1}/{len(frame_ids)} frames')

    summaries = {
        name: summarize_variant(rows, topks, args.riou_thr)
        for name, rows in rows_by_variant.items()
    }
    clean = summaries['clean']
    reference_signature = None
    reference_signature_ready = False
    focus_frames = []
    if args.role == 'target_dev':
        focus_rows = _rows_in_window(
            rows_by_variant['clean'], args.focus_start, args.focus_end)
        focus_frames = [int(row['frame']) for row in focus_rows]
        expected_focus_frames = args.focus_end - args.focus_start + 1
        reference_signature_ready = bool(
            full_sequence and len(focus_rows) == expected_focus_frames)
        reference_signature = summarize_variant(
            focus_rows, topks, args.riou_thr)

    gates = {}
    gate_summaries = {}
    passing_by_family = {
        family: [] for family in args.families
    } if not args.reference_only else {}
    if not args.reference_only:
        for family, severity in variants:
            name = _variant_name(family, severity)
            if family == 'clean':
                continue
            variant_gate_rows = _plateau_rows(rows_by_variant[name])
            gate_frames = [int(row['frame']) for row in variant_gate_rows]
            clean_gate_rows = _rows_for_frames(
                rows_by_variant['clean'], gate_frames)
            variant_gate_summary = summarize_variant(
                variant_gate_rows, topks, args.riou_thr)
            clean_gate_summary = summarize_variant(
                clean_gate_rows, topks, args.riou_thr)
            gate_summaries[name] = dict(
                scope='dark_plateau',
                frames=gate_frames,
                variant=variant_gate_summary,
                matched_clean=clean_gate_summary,
            )
            gate = evaluate_gate(
                variant_gate_summary, clean_gate_summary,
                args.pool_size, args, target_reference=target_reference)
            gates[name] = gate
            if gate['passed']:
                passing_by_family[family].append(float(severity))

    protocol_ready = bool(
        args.role == 'source_val'
        and role_policy['protocol_source']
        and full_sequence
        and target_reference is not None
        and len(passing_by_family) >= 2
        and all(passing_by_family[family] for family in args.families))

    print('\n' + '-' * 100)
    print('VARIANT SUMMARY')
    print('-' * 100)
    for family, severity in variants:
        name = _variant_name(family, severity)
        print_variant(
            name, summaries[name], gates.get(name), gate_summaries.get(name))
    if reference_signature is not None:
        print_variant('target_focus', reference_signature, None)
    print('-' * 100)
    if args.role == 'target_dev':
        print(
            f'target-dev focus: {args.focus_start}..{args.focus_end} '
            f'({len(focus_frames)} frames)')
        print(f'REFERENCE_SIGNATURE_READY={reference_signature_ready}')
    elif args.role == 'target_holdout':
        print('TARGET_HOLDOUT: ineligible for model/checkpoint selection')
    else:
        print(f'passing severities by family: {passing_by_family}')
        print(f'PROTOCOL_READY_FOR_P1_A={protocol_ready}')

    output_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = dict(
        probe='dark_proxy_preflight',
        deployable=False,
        data_role=args.role,
        uses_test_data=args.split == 'test',
        uses_target_domain=role_policy['uses_target_domain'],
        uses_target_labels=role_policy['uses_target_labels'],
        zero_shot_compliant=role_policy['zero_shot_compliant'],
        eligible_for_model_selection=(
            role_policy['eligible_for_model_selection']),
        authorizes_only=(
            'frozen_P1-A_probe' if args.role == 'source_val' else None),
        protocol_source=role_policy['protocol_source'],
        reference_only=args.reference_only,
        full_sequence=full_sequence,
        focus_frames=focus_frames,
        reference_signature_ready=reference_signature_ready,
        reference_signature=reference_signature,
        target_reference_json=args.reference_json,
        target_reference_manifest=(
            None if target_reference_payload is None
            else target_reference_payload.get('manifest', {}).get('sha256')),
        protocol_ready_for_p1_a=protocol_ready,
        config=args.config,
        checkpoint=args.checkpoint,
        args=vars(args),
        manifest=manifest,
        passing_severities_by_family=passing_by_family,
        summaries=summaries,
        gate_summaries=gate_summaries,
        gates=gates,
        rows=rows_by_variant,
    )
    with open(output_path, 'w') as handle:
        json.dump(payload, handle, indent=2)
    print(f'[out] wrote {output_path}')


if __name__ == '__main__':
    main()
