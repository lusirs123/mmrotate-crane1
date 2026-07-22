#!/usr/bin/env python3
"""Source-only guided background optimization for a 16-frame discriminator.

This is a diagnosis probe, not a training-data generator. It optimizes one
shared low-frequency RGB texture across 16 contiguous source-validation
frames. The texture is applied only outside target boxes and their margins.

The objective raises a source-background candidate relative to the best usable
target candidate while softly keeping the background score below the original
0.05 inference threshold. A passing run authorizes only a full source-val
confirmation; it never authorizes residual-adapter training directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import candidate_pool_oracle_probe as pool_probe  # noqa: E402
from crane_project.tools import ctx_entry_probe as entry_probe  # noqa: E402
from crane_project.tools import dark_proxy_preflight as preflight  # noqa: E402
from crane_project.utils.dark_degradation import (  # noqa: E402
    SUPPORTED_DARK_FAMILIES,
    apply_dark_degradation,
)
from crane_project.utils.structured_dark_proxy import (  # noqa: E402
    target_exclusion_mask,
)


ALLOWED_SOURCE_SEQUENCES = ('real_seq07', 'sim_seq10')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Optimize one shared source-only background texture over '
                    'a contiguous 16-frame validation window.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default='val', choices=['val'])
    parser.add_argument('--seq', required=True,
                        choices=list(ALLOWED_SOURCE_SEQUENCES))
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--candidate-source', default='main', choices=['main'])
    parser.add_argument('--dark-family', default='photometric',
                        choices=list(SUPPORTED_DARK_FAMILIES))
    parser.add_argument('--dark-severity', type=float, default=0.45)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--learning-rate', type=float, default=0.10)
    parser.add_argument('--latent-size', type=int, default=32)
    parser.add_argument('--pixel-epsilon', type=float, default=24.0,
                        help='Maximum RGB change in 0..255 pixel units.')
    parser.add_argument('--region-size-ratio', type=float, default=0.28)
    parser.add_argument('--region-aspect', type=float, default=1.5)
    parser.add_argument('--drift-ratio', type=float, default=0.025)
    parser.add_argument('--target-margin-ratio', type=float, default=0.35)
    parser.add_argument('--background-topm', type=int, default=32)
    parser.add_argument('--smoothmax-temperature', type=float, default=0.5)
    parser.add_argument('--rank-margin', type=float, default=3.0,
                        help='Required background-minus-usable logit margin.')
    parser.add_argument('--background-score-ceiling', type=float,
                        default=0.02)
    parser.add_argument('--ceiling-weight', type=float, default=1.0)
    parser.add_argument('--tv-weight', type=float, default=0.05)
    parser.add_argument('--l2-weight', type=float, default=0.01)
    parser.add_argument('--false-iou-thr', type=float, default=0.10)
    parser.add_argument('--riou-thr', type=float, default=0.50)
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--topks', type=int, nargs='+',
                        default=[1, 100, 1000, 10000])
    parser.add_argument('--pool-size', type=int, default=10000)
    parser.add_argument('--min-silence-rate', type=float, default=0.79)
    parser.add_argument('--max-silence-rate', type=float, default=1.0)
    parser.add_argument('--max-top1-recall', type=float, default=0.20)
    parser.add_argument('--min-top1-error-run', type=int, default=16)
    parser.add_argument('--min-rank-median', type=float, default=500.0)
    parser.add_argument('--max-rank-median', type=float, default=8000.0)
    parser.add_argument('--min-pool-oracle-recall', type=float, default=0.80)
    parser.add_argument('--min-oracle-retention', type=float, default=0.80)
    parser.add_argument('--min-dense-riou-retention', type=float, default=0.80)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--preview-dir', default=None)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def validate_args(args) -> Tuple[List[int], List[int]]:
    if args.seed != 0:
        raise ValueError('The unified protocol requires --seed 0')
    if args.split != 'val' or args.seq not in ALLOWED_SOURCE_SEQUENCES:
        raise ValueError('Guided optimization accepts source validation only')
    if args.end < args.start or args.end - args.start + 1 != 16:
        raise ValueError('The discriminator requires exactly 16 contiguous frames')
    if not 0.0 <= args.dark_severity <= 1.0:
        raise ValueError('--dark-severity must be in [0, 1]')
    if args.steps <= 0 or args.learning_rate <= 0.0:
        raise ValueError('steps and learning-rate must be positive')
    if args.latent_size < 4 or args.pixel_epsilon <= 0.0:
        raise ValueError('invalid texture parameterization')
    if not 0.05 <= args.region_size_ratio <= 0.50:
        raise ValueError('--region-size-ratio must be in [0.05, 0.50]')
    if not 0.5 <= args.region_aspect <= 2.0:
        raise ValueError('--region-aspect must be in [0.5, 2.0]')
    if not 0.0 <= args.drift_ratio <= 0.10:
        raise ValueError('--drift-ratio must be in [0, 0.10]')
    if not 0.0 <= args.target_margin_ratio <= 1.0:
        raise ValueError('--target-margin-ratio must be in [0, 1]')
    if args.background_topm <= 0 or args.smoothmax_temperature <= 0.0:
        raise ValueError('invalid smooth-max settings')
    if args.rank_margin <= 0.0:
        raise ValueError('--rank-margin must be positive')
    if not 0.0 < args.background_score_ceiling < args.score_thr:
        raise ValueError('background ceiling must be below score-thr')
    if min(args.ceiling_weight, args.tv_weight, args.l2_weight) < 0.0:
        raise ValueError('loss weights must be non-negative')
    if not 0.0 <= args.false_iou_thr < args.riou_thr <= 1.0:
        raise ValueError('Require false-iou-thr < riou-thr')
    if args.min_top1_error_run != 16:
        raise ValueError('The 16-frame discriminator requires error run 16')
    topks = preflight.normalize_topks(args.topks, args.pool_size)
    preflight.validate_data_role(
        args.data_root, 'source_val', args.split, args.seq)
    return list(range(args.start, args.end + 1)), topks


def _stable_seed(*parts) -> int:
    payload = ':'.join(str(part) for part in parts).encode('utf-8')
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8], 'little', signed=False)


def smooth_region_centers(base_xy: Tuple[float, float], frame_count: int,
                          drift_ratio: float, seed: int,
                          sequence: str) -> List[Tuple[float, float]]:
    """Return a deterministic slow trajectory around a source-chosen center."""
    rng = np.random.default_rng(
        _stable_seed(seed, sequence, 'guided_region_trajectory'))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    speed = float(rng.uniform(0.18, 0.35))
    centers = []
    for index in range(frame_count):
        progress = index / max(float(frame_count - 1), 1.0)
        angle = phase + 2.0 * np.pi * speed * progress
        centers.append((
            float(np.clip(
                base_xy[0] + drift_ratio * math.cos(angle), 0.02, 0.98)),
            float(np.clip(
                base_xy[1] + drift_ratio * math.sin(angle), 0.02, 0.98)),
        ))
    return centers


def region_from_center(center_xy: Tuple[float, float], valid_shape,
                       size_ratio: float, aspect: float) -> Tuple[int, int, int, int]:
    height, width = int(valid_shape[0]), int(valid_shape[1])
    area_side = float(min(height, width)) * float(size_ratio)
    region_w = int(np.clip(
        round(area_side * math.sqrt(aspect)), 16, width))
    region_h = int(np.clip(
        round(area_side / math.sqrt(aspect)), 16, height))
    center_x = float(center_xy[0]) * width
    center_y = float(center_xy[1]) * height
    x = int(np.clip(
        round(center_x - region_w / 2.0), 0, max(width - region_w, 0)))
    y = int(np.clip(
        round(center_y - region_h / 2.0), 0, max(height - region_h, 0)))
    return x, y, region_w, region_h


def build_region_alpha(pad_shape, valid_shape, region,
                       target_exclusion: np.ndarray) -> torch.Tensor:
    """Build a feathered model-space alpha with target pixels forced to zero."""
    pad_h, pad_w = int(pad_shape[0]), int(pad_shape[1])
    valid_h, valid_w = int(valid_shape[0]), int(valid_shape[1])
    x, y, region_w, region_h = [int(value) for value in region]
    yy = np.linspace(-1.0, 1.0, region_h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, region_w, dtype=np.float32)[None, :]
    radius = np.sqrt(xx * xx + yy * yy)
    feather = np.clip((1.0 - radius) / 0.24, 0.0, 1.0)
    feather = cv2.GaussianBlur(
        feather, (0, 0), sigmaX=max(1.0, min(region_h, region_w) * 0.02))
    alpha = np.zeros((pad_h, pad_w), dtype=np.float32)
    alpha[y:y + region_h, x:x + region_w] = feather
    allowed = np.ones((pad_h, pad_w), dtype=np.float32)
    allowed[:valid_h, :valid_w] = (target_exclusion == 0).astype(np.float32)
    allowed[valid_h:, :] = 0.0
    allowed[:, valid_w:] = 0.0
    alpha *= allowed
    return torch.from_numpy(alpha).unsqueeze(0).unsqueeze(0)


def anchor_region_mask(anchor_centers: torch.Tensor,
                       alpha: torch.Tensor,
                       threshold: float = 0.20) -> torch.Tensor:
    height, width = alpha.shape[-2:]
    x = torch.floor(anchor_centers[:, 0]).long()
    y = torch.floor(anchor_centers[:, 1]).long()
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    result = torch.zeros_like(valid)
    if bool(valid.any()):
        result[valid] = alpha[0, 0, y[valid], x[valid]] >= float(threshold)
    return result


def score_to_logit(scores: torch.Tensor) -> torch.Tensor:
    return torch.logit(scores.clamp(1e-6, 1.0 - 1e-6))


def guided_rank_loss(scores: torch.Tensor,
                     background_mask: torch.Tensor,
                     usable_mask: torch.Tensor,
                     topm: int,
                     temperature: float,
                     rank_margin: float,
                     score_ceiling: float) -> Tuple[torch.Tensor, Dict]:
    """Compute a stable relative-rank loss plus normalized soft ceiling."""
    background = scores[background_mask]
    usable = scores[usable_mask]
    if background.numel() == 0:
        raise RuntimeError('No background candidates inside guided region')
    if usable.numel() == 0:
        raise RuntimeError('No usable target candidate during optimization')
    actual_topm = min(int(topm), int(background.numel()))
    top_background = torch.topk(
        background, k=actual_topm, largest=True, sorted=False).values
    background_logits = score_to_logit(top_background)
    smooth_background_logit = float(temperature) * (
        torch.logsumexp(background_logits / float(temperature), dim=0)
        - math.log(actual_topm))
    background_max = top_background.max()
    usable_max = usable.max()
    usable_logit = score_to_logit(usable_max)
    logit_gap = smooth_background_logit - usable_logit
    margin_loss = F.relu(float(rank_margin) - logit_gap).square()
    ceiling_violation = F.relu(
        background_max - float(score_ceiling)) / float(score_ceiling)
    ceiling_loss = ceiling_violation.square()
    return margin_loss, dict(
        ceiling_loss=ceiling_loss,
        background_max=background_max,
        usable_max=usable_max,
        logit_gap=logit_gap,
        background_candidates=int(background.numel()),
        usable_candidates=int(usable.numel()),
    )


def select_fixed_objective_masks(scores: torch.Tensor,
                                 ious: torch.Tensor,
                                 region_mask: torch.Tensor,
                                 false_iou_thr: float,
                                 riou_thr: float) -> Tuple[torch.Tensor,
                                                           torch.Tensor,
                                                           Dict]:
    """Freeze background candidates and one usable carrier at baseline."""
    background_mask = (
        region_mask & (ious < float(false_iou_thr)))
    usable_indices = torch.nonzero(
        ious >= float(riou_thr), as_tuple=False).reshape(-1)
    if not bool(background_mask.any()):
        raise RuntimeError('No baseline background candidates in guided region')
    if usable_indices.numel() == 0:
        raise RuntimeError('No baseline usable target carrier')
    carrier_index = int(
        usable_indices[torch.argmax(scores[usable_indices])].item())
    carrier_mask = torch.zeros_like(background_mask)
    carrier_mask[carrier_index] = True
    return background_mask, carrier_mask, dict(
        candidate_count=int(scores.numel()),
        background_candidates=int(background_mask.sum().item()),
        carrier_index=carrier_index,
        carrier_score=float(scores[carrier_index].item()),
        carrier_riou=float(ious[carrier_index].item()),
        carrier_rank=int(
            (scores > scores[carrier_index]).sum().item()) + 1)


def total_variation(tensor: torch.Tensor) -> torch.Tensor:
    horizontal = (tensor[..., :, 1:] - tensor[..., :, :-1]).abs().mean()
    vertical = (tensor[..., 1:, :] - tensor[..., :-1, :]).abs().mean()
    return horizontal + vertical


def extract_normalization(cfg) -> Tuple[List[float], List[float]]:
    for transform in cfg.test_pipeline:
        if transform.get('type') != 'MultiScaleFlipAug':
            continue
        for inner in transform.get('transforms', []):
            if inner.get('type') == 'Normalize':
                if not bool(inner.get('to_rgb', True)):
                    raise ValueError('Guided probe requires Normalize(to_rgb=True)')
                return list(inner['mean']), list(inner['std'])
    raise ValueError('Could not find Normalize in test_pipeline')


def make_guided_input(base_tensor: torch.Tensor,
                      latent: torch.Tensor,
                      alpha: torch.Tensor,
                      region: Tuple[int, int, int, int],
                      mean: torch.Tensor,
                      std: torch.Tensor,
                      pixel_epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Insert a bounded low-frequency RGB delta into an exact model input."""
    x, y, region_w, region_h = [int(value) for value in region]
    pad_h, pad_w = base_tensor.shape[-2:]
    unit_patch = F.interpolate(
        torch.tanh(latent), size=(region_h, region_w),
        mode='bicubic', align_corners=False).clamp(-1.0, 1.0)
    full_delta = F.pad(
        unit_patch,
        (x, pad_w - x - region_w, y, pad_h - y - region_h))
    pixel_delta = float(pixel_epsilon) * full_delta * alpha
    pixels = base_tensor * std + mean
    modified_pixels = (pixels + pixel_delta).clamp(0.0, 255.0)
    return (modified_pixels - mean) / std, unit_patch


def render_guided_bgr(darkened_bgr: np.ndarray,
                      latent: torch.Tensor,
                      alpha: torch.Tensor,
                      region: Tuple[int, int, int, int],
                      valid_shape,
                      pixel_epsilon: float,
                      gts: Sequence[Dict],
                      target_margin_ratio: float) -> Tuple[np.ndarray, Dict]:
    """Replay the optimized model-space texture into the original BGR image."""
    x, y, region_w, region_h = [int(value) for value in region]
    valid_h, valid_w = int(valid_shape[0]), int(valid_shape[1])
    with torch.no_grad():
        patch = F.interpolate(
            torch.tanh(latent.detach().cpu()),
            size=(region_h, region_w), mode='bicubic',
            align_corners=False).clamp(-1.0, 1.0)[0].permute(
                1, 2, 0).numpy()
    delta_rgb = np.zeros((valid_h, valid_w, 3), dtype=np.float32)
    delta_rgb[y:y + region_h, x:x + region_w] = (
        float(pixel_epsilon) * patch)
    alpha_valid = alpha[0, 0, :valid_h, :valid_w].cpu().numpy()[..., None]
    delta_rgb *= alpha_valid
    image_h, image_w = darkened_bgr.shape[:2]
    delta_rgb = cv2.resize(
        delta_rgb, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    delta_bgr = delta_rgb[..., ::-1]
    output = np.clip(
        darkened_bgr.astype(np.float32) + delta_bgr,
        0.0, 255.0).astype(np.uint8)
    exclusion = target_exclusion_mask(
        darkened_bgr.shape, gts, margin_ratio=target_margin_ratio)
    output[exclusion > 0] = darkened_bgr[exclusion > 0]
    actual_delta = output.astype(np.float32) - darkened_bgr.astype(np.float32)
    changed = np.any(actual_delta != 0.0, axis=2)
    return output, dict(
        changed_pixels=int(changed.sum()),
        target_pixels_changed=int(np.any(
            actual_delta[exclusion > 0] != 0.0, axis=1).sum()),
        delta_abs_max=float(np.abs(actual_delta).max()),
        delta_rms=float(np.sqrt(np.mean(actual_delta * actual_delta))),
    )


def forward_candidates(model, image_tensor, meta, candidate_source):
    features = model.extract_feat(image_tensor)
    candidate_head, cls_scores, bbox_preds = (
        entry_probe.forward_candidate_head(
            model, features, candidate_source))
    boxes, scores, levels, anchors, alignment = (
        entry_probe.flatten_decode_candidates(
            candidate_head, cls_scores, bbox_preds, meta['img_shape']))
    return boxes, scores, levels, anchors, alignment


def select_source_background_center(model, frame: Dict, args) -> Tuple[float, float]:
    """Choose the strongest source-only false anchor as trajectory origin."""
    from mmcv.ops import box_iou_rotated

    device = frame['base_tensor'].device
    with torch.no_grad():
        boxes, scores, _, anchors, _ = forward_candidates(
            model, frame['base_tensor'], frame['meta'], args.candidate_source)
        gt_box = entry_probe.gt_to_tensor(frame['scaled_gt'], device)
        ious = box_iou_rotated(
            boxes.float(), gt_box.float()).reshape(-1)
        valid_h, valid_w = frame['meta']['img_shape'][:2]
        valid_anchor = (
            (anchors[:, 0] >= 0) & (anchors[:, 0] < valid_w)
            & (anchors[:, 1] >= 0) & (anchors[:, 1] < valid_h))
        false = (ious < float(args.false_iou_thr)) & valid_anchor
        if not bool(false.any()):
            raise RuntimeError('No source background anchor for initialization')
        false_indices = torch.nonzero(false, as_tuple=False).reshape(-1)
        best = int(false_indices[torch.argmax(scores[false_indices])].item())
        return (
            float(anchors[best, 0].item() / max(float(valid_w), 1.0)),
            float(anchors[best, 1].item() / max(float(valid_h), 1.0)),
        )


def prepare_frames(transform_compose, img_scale, flip,
                   frame_ids: Sequence[int], args) -> List[Dict]:
    diag = entry_probe.get_diag()
    device = torch.device(f'cuda:{args.gpu}')
    frames = []
    for frame_id in frame_ids:
        img_path, ann_path = diag.find_files(
            args.data_root, args.split, args.seq, frame_id)
        raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if raw is None:
            raise RuntimeError(f'Failed to read {img_path}')
        gts = diag.parse_dota_ann(ann_path)
        if not gts:
            raise RuntimeError(f'Missing source GT at {ann_path}')
        if len(gts) != 1:
            raise RuntimeError(
                'The guided discriminator requires exactly one source GT per '
                f'frame, got {len(gts)} at {ann_path}')
        darkened, dark_meta = apply_dark_degradation(
            raw, family=args.dark_family, sequence=args.seq,
            frame=frame_id, start=frame_ids[0], end=frame_ids[-1],
            severity=args.dark_severity, seed=args.seed, profile='constant')
        base_tensor, meta = preflight.preprocess_bgr_array(
            darkened, img_path, transform_compose, img_scale, flip)
        scaled_gts = [pool_probe.scale_gt_to_img(gt, meta) for gt in gts]
        frames.append(dict(
            frame=int(frame_id), img_path=img_path, ann_path=ann_path,
            raw=raw, darkened=darkened, dark_meta=dark_meta,
            gts=gts, scaled_gts=scaled_gts, scaled_gt=scaled_gts[0],
            meta=meta, base_tensor=base_tensor.to(device)))
    return frames


def attach_regions(frames: List[Dict], base_center, args):
    centers = smooth_region_centers(
        base_center, len(frames), args.drift_ratio, args.seed, args.seq)
    for frame, center in zip(frames, centers):
        valid_shape = frame['meta']['img_shape']
        pad_shape = frame['meta']['pad_shape']
        region = region_from_center(
            center, valid_shape, args.region_size_ratio, args.region_aspect)
        exclusion = target_exclusion_mask(
            valid_shape, frame['scaled_gts'],
            margin_ratio=args.target_margin_ratio)
        alpha = build_region_alpha(
            pad_shape, valid_shape, region, exclusion)
        if int((alpha > 0.20).sum().item()) == 0:
            raise RuntimeError(
                f'Guided region is fully excluded at frame {frame["frame"]}')
        frame['region_center'] = [float(center[0]), float(center[1])]
        frame['region'] = [int(value) for value in region]
        frame['alpha'] = alpha.to(frame['base_tensor'].device)


def freeze_objective_candidates(model, frames: List[Dict], args) -> List[Dict]:
    """Freeze per-frame candidate masks before any pixel optimization."""
    from mmcv.ops import box_iou_rotated

    records = []
    for frame in frames:
        device = frame['base_tensor'].device
        with torch.no_grad():
            boxes, scores, _, anchors, _ = forward_candidates(
                model, frame['base_tensor'], frame['meta'],
                args.candidate_source)
            gt_box = entry_probe.gt_to_tensor(frame['scaled_gt'], device)
            ious = box_iou_rotated(
                boxes.float(), gt_box.float()).reshape(-1)
            region_mask = anchor_region_mask(anchors, frame['alpha'])
            background_mask, carrier_mask, record = (
                select_fixed_objective_masks(
                    scores, ious, region_mask, args.false_iou_thr,
                    args.riou_thr))
        frame['objective_background_mask'] = background_mask
        frame['objective_carrier_mask'] = carrier_mask
        frame['objective_candidate_count'] = int(scores.numel())
        records.append(dict(frame=frame['frame'], **record))
    return records


def optimize_shared_texture(model, frames: List[Dict], mean, std, args):
    device = frames[0]['base_tensor'].device
    latent = torch.zeros(
        (1, 3, args.latent_size, args.latent_size),
        dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([latent], lr=args.learning_rate)
    mean_tensor = torch.tensor(
        mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(
        std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    trace = []

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        frame_stats = []
        for frame in frames:
            guided_input, _ = make_guided_input(
                frame['base_tensor'], latent, frame['alpha'],
                tuple(frame['region']), mean_tensor, std_tensor,
                args.pixel_epsilon)
            _, scores, _, _, _ = forward_candidates(
                model, guided_input, frame['meta'], args.candidate_source)
            if scores.numel() != frame['objective_candidate_count']:
                raise RuntimeError(
                    'Candidate ordering changed during guided optimization')
            margin_loss, components = guided_rank_loss(
                scores, frame['objective_background_mask'],
                frame['objective_carrier_mask'],
                args.background_topm, args.smoothmax_temperature,
                args.rank_margin, args.background_score_ceiling)
            data_loss = (
                margin_loss
                + float(args.ceiling_weight) * components['ceiling_loss'])
            (data_loss / len(frames)).backward()
            frame_stats.append(dict(
                margin_loss=float(margin_loss.detach().item()),
                ceiling_loss=float(
                    components['ceiling_loss'].detach().item()),
                background_max=float(
                    components['background_max'].detach().item()),
                usable_max=float(components['usable_max'].detach().item()),
                logit_gap=float(components['logit_gap'].detach().item()),
                background_candidates=components['background_candidates'],
                usable_candidates=components['usable_candidates']))

        unit_texture = torch.tanh(latent)
        tv_loss = total_variation(unit_texture)
        l2_loss = unit_texture.square().mean()
        regularization = (
            float(args.tv_weight) * tv_loss
            + float(args.l2_weight) * l2_loss)
        regularization.backward()
        optimizer.step()

        record = dict(
            step=int(step + 1),
            loss_margin_mean=float(np.mean([
                item['margin_loss'] for item in frame_stats])),
            loss_ceiling_mean=float(np.mean([
                item['ceiling_loss'] for item in frame_stats])),
            background_max_mean=float(np.mean([
                item['background_max'] for item in frame_stats])),
            background_max_max=float(np.max([
                item['background_max'] for item in frame_stats])),
            usable_max_mean=float(np.mean([
                item['usable_max'] for item in frame_stats])),
            logit_gap_mean=float(np.mean([
                item['logit_gap'] for item in frame_stats])),
            tv=float(tv_loss.detach().item()),
            l2=float(l2_loss.detach().item()))
        trace.append(record)
        if (step == 0 or (step + 1) % args.log_interval == 0
                or step + 1 == args.steps):
            print(
                '[opt {}/{}] gap={:.3f} bg_mean/max={:.4f}/{:.4f} '
                'usable={:.4f} margin={:.3f} ceil={:.3f} '
                'tv={:.4f} l2={:.4f}'.format(
                    step + 1, args.steps, record['logit_gap_mean'],
                    record['background_max_mean'],
                    record['background_max_max'],
                    record['usable_max_mean'],
                    record['loss_margin_mean'],
                    record['loss_ceiling_mean'], record['tv'], record['l2']))
    return latent.detach(), trace


def evaluate_replayed_frames(model, frames, latent, transform_compose,
                             img_scale, flip, topks, args):
    rows = dict(clean=[], dark_only=[], guided=[])
    replay_stats = []
    if args.preview_dir:
        os.makedirs(args.preview_dir, exist_ok=True)
    for frame in frames:
        guided, render_stats = render_guided_bgr(
            frame['darkened'], latent, frame['alpha'], tuple(frame['region']),
            frame['meta']['img_shape'], args.pixel_epsilon, frame['gts'],
            args.target_margin_ratio)
        replay_stats.append(dict(frame=frame['frame'], **render_stats))
        variants = (
            ('clean', frame['raw']),
            ('dark_only', frame['darkened']),
            ('guided', guided),
        )
        for name, image in variants:
            row = preflight.analyze_variant(
                model, image, dict(img_path=frame['img_path']),
                frame['gts'][0], transform_compose, img_scale, flip,
                args, topks)
            row.update(
                frame=frame['frame'], fname=f'{args.seq}_{frame["frame"]:05d}',
                split=args.split, seq=args.seq, variant=name,
                degradation=dict(
                    family=name, dark_severity=args.dark_severity,
                    shared_texture=name == 'guided',
                    region=frame['region'],
                    region_center=frame['region_center'],
                    target_geometry_modified=False))
            rows[name].append(row)
            if args.preview_dir:
                directory = os.path.join(args.preview_dir, name)
                overlay_dir = os.path.join(args.preview_dir, name + '_overlay')
                os.makedirs(directory, exist_ok=True)
                os.makedirs(overlay_dir, exist_ok=True)
                filename = f'{args.seq}_{frame["frame"]:05d}.jpg'
                cv2.imwrite(os.path.join(directory, filename), image)
                overlay = preflight._draw_proxy_preview(
                    image, frame['gts'][0], row)
                cv2.imwrite(os.path.join(overlay_dir, filename), overlay)
    return rows, replay_stats


def save_texture_preview(latent, pixel_epsilon, preview_dir):
    if preview_dir is None:
        return None
    with torch.no_grad():
        unit = torch.tanh(latent.cpu())[0].permute(1, 2, 0).numpy()
    rgb = np.clip(127.5 + float(pixel_epsilon) * unit, 0.0, 255.0)
    bgr = rgb[..., ::-1].astype(np.uint8)
    bgr = cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_NEAREST)
    path = os.path.abspath(os.path.join(preview_dir, 'shared_texture.png'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bgr)
    return path


def main():
    args = parse_args()
    frame_ids, topks = validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, cfg = entry_probe.load_model(
        args.config, args.checkpoint, args.gpu)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    diag = entry_probe.get_diag()
    transform_compose, img_scale, flip = diag.build_test_transforms(cfg)
    mean, std = extract_normalization(cfg)
    frames = prepare_frames(
        transform_compose, img_scale, flip, frame_ids, args)
    base_center = select_source_background_center(model, frames[0], args)
    attach_regions(frames, base_center, args)
    objective_candidates = freeze_objective_candidates(model, frames, args)
    print('GUIDED BACKGROUND RANK PROBE')
    print(f'data:       {args.split}/{args.seq}[{args.start}..{args.end}]')
    print(f'base center:{base_center}')
    print(f'dark:       {args.dark_family} severity={args.dark_severity}')
    print(f'ceiling:    {args.background_score_ceiling}')
    print(f'steps:      {args.steps}')

    latent, trace = optimize_shared_texture(
        model, frames, mean, std, args)
    rows, replay_stats = evaluate_replayed_frames(
        model, frames, latent, transform_compose,
        img_scale, flip, topks, args)
    summaries = {
        name: preflight.summarize_variant(values, topks, args.riou_thr)
        for name, values in rows.items()
    }
    gate = preflight.evaluate_gate(
        summaries['guided'], summaries['clean'], args.pool_size, args)
    discriminator_passed = bool(gate['passed'])
    texture_preview = save_texture_preview(
        latent, args.pixel_epsilon, args.preview_dir)
    manifest = preflight.build_manifest(
        args.data_root, args.split, args.seq, frame_ids, 'source_val')
    naturalness = dict(
        target_pixels_changed=sum(
            item['target_pixels_changed'] for item in replay_stats),
        delta_abs_max=max(item['delta_abs_max'] for item in replay_stats),
        delta_rms_mean=float(np.mean([
            item['delta_rms'] for item in replay_stats])),
        changed_pixels_mean=float(np.mean([
            item['changed_pixels'] for item in replay_stats])))
    naturalness_passed = bool(
        naturalness['target_pixels_changed'] == 0
        and naturalness['delta_abs_max'] <= args.pixel_epsilon + 1.0)
    discriminator_passed = bool(discriminator_passed and naturalness_passed)

    payload = dict(
        probe='guided_background_rank_probe',
        protocol_version=1,
        diagnosis_only=True,
        deployable=False,
        data_role='source_val',
        uses_test_data=False,
        uses_target_domain=False,
        uses_target_labels=False,
        uses_source_labels=True,
        target_informed_thresholds=True,
        gate_provenance='real_seq02_target_dev_diagnostic',
        zero_shot_compliant=False,
        eligible_for_model_selection=True,
        shared_texture_across_frames=True,
        target_pixels_optimized=False,
        target_margin_ratio=float(args.target_margin_ratio),
        frame_count=len(frame_ids),
        discriminator_passed=discriminator_passed,
        naturalness_passed=naturalness_passed,
        requires_visual_naturalness_review=True,
        protocol_ready_for_p1_a=False,
        authorizes_only=(
            'visual_review_then_full_source_val_guided_proxy_confirmation'
            if discriminator_passed else None),
        next_decision=(
            'inspect_texture_then_run_full_source_val_confirmation'
            if discriminator_passed
            else 'close_proxy_residual_adapter_route'),
        config=args.config,
        checkpoint=args.checkpoint,
        args=vars(args),
        manifest=manifest,
        initialization=dict(
            strategy='highest_scoring_source_only_false_anchor',
            base_center=[float(base_center[0]), float(base_center[1])]),
        objective_candidates=objective_candidates,
        loss=dict(
            background_definition=(
                'fixed_baseline_non_target_anchors_inside_source_region'),
            usable_definition='fixed_highest_score_baseline_usable_carrier',
            relative_rank='background_logit_minus_fixed_carrier_logit',
            soft_score_ceiling=float(args.background_score_ceiling),
            low_frequency_latent=True,
            tv_weight=float(args.tv_weight),
            l2_weight=float(args.l2_weight)),
        naturalness=naturalness,
        texture_preview=texture_preview,
        optimization_trace=trace,
        replay_stats=replay_stats,
        summaries=summaries,
        gate=gate,
        rows=rows)
    output_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as handle:
        json.dump(payload, handle, indent=2)

    print('\nGUIDED REPLAY SUMMARY')
    for name in ('clean', 'dark_only', 'guided'):
        preflight.print_variant(name, summaries[name], None)
    print('gate checks: {}'.format(gate['checks']))
    print(f'NATURALNESS_PASSED={naturalness_passed}')
    print(f'GUIDED_16_FRAME_DISCRIMINATOR_PASSED={discriminator_passed}')
    print(f'[out] wrote {output_path}')
    print('[policy] SOURCE-VAL DIAGNOSIS ONLY; DO NOT TRAIN AN ADAPTER FROM '
          'THIS 16-FRAME RESULT')


if __name__ == '__main__':
    main()
