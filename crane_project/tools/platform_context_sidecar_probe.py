#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe 5: BrightAug + independent RGB platform-context sidecar.

This is deliberately a two-stage experiment:

1. A selected ``crane_symeood_k1_brightaug`` detector is loaded, put in eval
   mode and frozen permanently.
2. A separate RGB encoder learns an internal platform heatmap from train-only
   ``K(beam_gt)`` masks.  Checkpoint selection uses manual polygons from a
   held-out *training* sequence.
3. At test time the sidecar only reranks pre-threshold beam candidates.  The
   selected OBB is an unchanged main-head candidate; no platform box is output
   and no platform-to-beam inverse transform exists.

The test manual polygons and beam GT are evaluation-only.  Candidate mapping K
defaults to the median K fitted on training sequences, never the test polygons.
"""

import argparse
import ast
import glob
import json
import os
import random
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import candidate_pool_oracle_probe as pool_probe  # noqa: E402
from crane_project.tools import ctx_entry_probe as entry_probe  # noqa: E402
from crane_project.utils.platform_context_sidecar import (  # noqa: E402
    PlatformContextSidecar,
    candidate_platform_boxes,
    platform_heatmap_loss,
    sample_candidate_context,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Independent RGB platform heatmap and BrightAug rerank.')
    parser.add_argument(
        '--config',
        default='crane_project/configs/crane_symeood_k1_brightaug.py')
    parser.add_argument('--checkpoint', required=True,
                        help='Exact selected BrightAug checkpoint.')
    parser.add_argument(
        '--train-k-config',
        default=('crane_project/configs/archived_ablation/'
                 'crane_symeood_k1_platform_ctx.py'))
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--train-splits', nargs='+',
                        default=['train', 'train_sim'])
    parser.add_argument('--holdout-seq', default='real_seq01')
    parser.add_argument('--max-train-frames', type=int, default=1200)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--base-channels', type=int, default=24)
    parser.add_argument('--decoder-channels', type=int, default=64)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--pos-alpha', type=float, default=0.75)
    parser.add_argument('--dice-weight', type=float, default=1.0)

    # BrightAug-compatible photometric augmentation plus target-coverage
    # geometric augmentation.  Image and pseudo mask share the same transform.
    parser.add_argument('--aug-prob', type=float, default=0.8)
    parser.add_argument('--gamma-min', type=float, default=0.4)
    parser.add_argument('--gamma-max', type=float, default=1.0)
    parser.add_argument('--blur-prob', type=float, default=0.25)
    parser.add_argument('--noise-std-max', type=float, default=12.0)
    parser.add_argument('--translate-x-min', type=float, default=-0.15)
    parser.add_argument('--translate-x-max', type=float, default=0.35)
    parser.add_argument('--translate-y-min', type=float, default=-0.15)
    parser.add_argument('--translate-y-max', type=float, default=0.15)
    parser.add_argument('--scale-min', type=float, default=0.75)
    parser.add_argument('--scale-max', type=float, default=1.50)

    parser.add_argument(
        '--manual-train-json',
        default=('work_dirs/crane_symeood_k1/'
                 'manual_platform_polygons_train_v2.json'))
    parser.add_argument(
        '--manual-test-json',
        default=('work_dirs/crane_symeood_k1/'
                 'manual_platform_polygons_real_seq02.json'))
    parser.add_argument(
        '--sidecar-out',
        default=('work_dirs/crane_symeood_k1_brightaug_platform_sidecar/'
                 'sidecar_best.pth'))
    parser.add_argument('--sidecar-in', default='')
    parser.add_argument('--eval-only', action='store_true')

    parser.add_argument('--eval-split', default='test')
    parser.add_argument('--eval-seq', default='real_seq02')
    parser.add_argument('--eval-start', type=int, default=133)
    parser.add_argument('--eval-end', type=int, default=171)
    parser.add_argument('--topks', type=int, nargs='+',
                        default=[200, 500, 1000])
    parser.add_argument('--log-lambdas', type=float, nargs='+',
                        default=[0.0, 0.25, 0.5, 1.0, 2.0],
                        help='lambda=0 is the mandatory BrightAug identity control.')
    parser.add_argument('--candidate-source', default='main',
                        choices=['main', 'aux1'])
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--sample-grid-size', type=int, default=3)
    parser.add_argument('--gate-peak', type=float, default=0.50)
    parser.add_argument('--gate-contrast', type=float, default=0.20)
    parser.add_argument('--manual-min-hits', type=int, default=5)
    parser.add_argument('--manual-min-frames', type=int, default=6)
    parser.add_argument('--manual-min-median-iou', type=float, default=0.50)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--out-json',
        default=('work_dirs/crane_symeood_k1_brightaug_platform_sidecar/'
                 'probe5_result.json'))
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def freeze_detector(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def assert_isolated(model, sidecar):
    trainable = [name for name, param in model.named_parameters()
                 if param.requires_grad]
    gradients = [name for name, param in model.named_parameters()
                 if param.grad is not None]
    if trainable or gradients or model.training:
        raise RuntimeError(
            'BrightAug isolation failed: '
            f'trainable={trainable[:5]} grads={gradients[:5]} '
            f'training={model.training}')
    if not any(param.requires_grad for param in sidecar.parameters()):
        raise RuntimeError('Sidecar has no trainable parameters')


def _safe_config_value(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        return {_safe_config_value(key): _safe_config_value(value)
                for key, value in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_safe_config_value(item) for item in node.elts]
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_config_value(node.operand)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'dict' and not node.args):
        return {kw.arg: _safe_config_value(kw.value)
                for kw in node.keywords if kw.arg is not None}
    raise ValueError(f'Unsupported config expression: {ast.dump(node)}')


def load_train_seq_k(config_path: str) -> Dict[str, Dict]:
    with open(config_path, 'r', encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=config_path)
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(isinstance(target, ast.Name)
               and target.id == 'seq_platform_k'
               for target in statement.targets):
            value = _safe_config_value(statement.value)
            return {str(key): dict(item) for key, item in value.items()}
    raise RuntimeError(f'No seq_platform_k found in {config_path}')


def median_k(seq_k: Dict[str, Dict]) -> Dict:
    result = {}
    for key in ('width_k', 'height_k', 'offset_long_k',
                'offset_short_k', 'dtheta'):
        result[key] = float(np.median([
            float(item.get(key, 0.0)) for item in seq_k.values()]))
    result['source'] = 'train_seq_k_median'
    result['sequences'] = sorted(seq_k)
    return result


def parse_seq_frame(path: str) -> Optional[Tuple[str, int]]:
    match = re.search(
        r'((?:real|sim)_seq\d+)_(\d+)$',
        os.path.splitext(os.path.basename(path))[0])
    return (match.group(1), int(match.group(2))) if match else None


def enumerate_records(data_root: str, splits: Sequence[str],
                      allowed_seqs: Sequence[str]) -> List[Tuple[str, str, int]]:
    allowed = set(allowed_seqs)
    records = []
    for split in splits:
        pattern = os.path.join(data_root, split, 'annfiles', '*.txt')
        for ann_path in sorted(glob.glob(pattern)):
            parsed = parse_seq_frame(ann_path)
            if parsed is not None and parsed[0] in allowed:
                records.append((split, parsed[0], parsed[1]))
    return records


def sample_records(records, maximum: int, seed: int):
    records = list(records)
    random.Random(seed).shuffle(records)
    return records[:maximum] if maximum > 0 else records


def load_manual_items(path: str, split: Optional[str] = None,
                      seq: Optional[str] = None,
                      start: Optional[int] = None,
                      end: Optional[int] = None) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    items = []
    for key, value in payload.get('frames', {}).items():
        item = dict(value)
        item['key'] = key
        frame = int(item['frame'])
        if split is not None and item.get('split') != split:
            continue
        if seq is not None and item.get('seq') != seq:
            continue
        if start is not None and frame < start:
            continue
        if end is not None and frame > end:
            continue
        items.append(item)
    return sorted(items, key=lambda item: (
        str(item.get('split')), str(item.get('seq')), int(item['frame'])))


def load_frame(record, args, transform_compose, img_scale, flip):
    split, seq, frame = record
    diag = entry_probe.get_diag()
    img_path, ann_path = diag.find_files(args.data_root, split, seq, frame)
    gts = diag.parse_dota_ann(ann_path) if ann_path else []
    if img_path is None or not gts:
        return None
    image, meta, stats = diag.preprocess_image(
        img_path, transform_compose, img_scale, flip)
    if image is None:
        return None
    return image.cuda(f'cuda:{args.gpu}'), meta, stats, gts[0], img_path


def normalized_to_rgb01(image: torch.Tensor, cfg) -> torch.Tensor:
    norm = getattr(cfg, 'img_norm_cfg', {})
    mean = image.new_tensor(norm.get(
        'mean', [123.675, 116.28, 103.53])).view(1, 3, 1, 1)
    std = image.new_tensor(norm.get(
        'std', [58.395, 57.12, 57.375])).view(1, 3, 1, 1)
    return ((image * std + mean) / 255.0).clamp(0.0, 1.0)


def augment_rgb_and_box(rgb: torch.Tensor, box: torch.Tensor, args):
    """Apply synchronized scale/translation plus BrightAug-like photometry."""
    if random.random() >= args.aug_prob:
        return rgb, box
    _, _, height, width = rgb.shape
    scale = random.uniform(args.scale_min, args.scale_max)
    tx_frac = random.uniform(args.translate_x_min, args.translate_x_max)
    ty_frac = random.uniform(args.translate_y_min, args.translate_y_max)
    tx, ty = tx_frac * width, ty_frac * height

    # affine_grid maps output coordinates back to input coordinates.  The
    # inverse below corresponds to a forward image transform around center:
    # x_out = scale * (x_in - center) + center + translation.
    theta = rgb.new_tensor([[
        [1.0 / scale, 0.0, -2.0 * tx / (scale * width)],
        [0.0, 1.0 / scale, -2.0 * ty / (scale * height)],
    ]])
    grid = F.affine_grid(theta, rgb.size(), align_corners=False)
    rgb = F.grid_sample(
        rgb, grid, mode='bilinear', padding_mode='border',
        align_corners=False)

    transformed = box.clone()
    transformed[:, 0] = (
        scale * (box[:, 0] - width * 0.5) + width * 0.5 + tx)
    transformed[:, 1] = (
        scale * (box[:, 1] - height * 0.5) + height * 0.5 + ty)
    transformed[:, 2:4] = box[:, 2:4] * scale

    # Match RandomBrightnessContrast: gamma<1 means exponent=1/gamma and
    # therefore darkens the RGB image.
    gamma = random.uniform(args.gamma_min, args.gamma_max)
    rgb = rgb.clamp(min=1e-6).pow(1.0 / max(gamma, 1e-6))
    if random.random() < args.blur_prob:
        rgb = F.avg_pool2d(rgb, kernel_size=3, stride=1, padding=1)
    if args.noise_std_max > 0:
        std = random.uniform(0.0, args.noise_std_max) / 255.0
        rgb = rgb + torch.randn_like(rgb) * std
    return rgb.clamp(0.0, 1.0), transformed


def target_from_obbs(boxes: torch.Tensor, output_shape: Tuple[int, int],
                     image_shape: Tuple[int, int]) -> torch.Tensor:
    """Rasterize OBBs directly at heatmap resolution."""
    out_h, out_w = int(output_shape[0]), int(output_shape[1])
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    ys = (torch.arange(out_h, device=boxes.device, dtype=boxes.dtype) + 0.5)
    xs = (torch.arange(out_w, device=boxes.device, dtype=boxes.dtype) + 0.5)
    ys = ys * (float(image_h) / out_h)
    xs = xs * (float(image_w) / out_w)
    try:
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    except TypeError:  # PyTorch < 1.10
        yy, xx = torch.meshgrid(ys, xs)
    mask = torch.zeros((out_h, out_w), dtype=torch.bool, device=boxes.device)
    for box in boxes:
        cx, cy, width, height, theta = box
        dx, dy = xx - cx, yy - cy
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        local_x = dx * cos_t + dy * sin_t
        local_y = -dx * sin_t + dy * cos_t
        mask |= ((local_x.abs() <= width * 0.5)
                 & (local_y.abs() <= height * 0.5))
    return mask.to(boxes.dtype).view(1, 1, out_h, out_w)


def manual_target(item: Dict, meta: Dict, output_shape: Tuple[int, int],
                  image_shape: Tuple[int, int]) -> np.ndarray:
    poly = np.asarray(item['platform_corners'], dtype=np.float32)
    scale_factor = meta.get('scale_factor', 1.0)
    if isinstance(scale_factor, torch.Tensor):
        scale_factor = scale_factor.detach().cpu().numpy()
    flat = np.asarray(scale_factor, dtype=np.float32).reshape(-1)
    sx = float(flat[0]) if flat.size else 1.0
    sy = float(flat[1]) if flat.size >= 2 else sx
    poly[:, 0] *= sx
    poly[:, 1] *= sy
    out_h, out_w = int(output_shape[0]), int(output_shape[1])
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    poly[:, 0] *= out_w / float(image_w)
    poly[:, 1] *= out_h / float(image_h)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    return mask.astype(bool)


def train_one_epoch(model, sidecar, optimizer, records, seq_k, cfg,
                    transform_compose, img_scale, flip, args, epoch):
    # Only the sidecar enters train mode.  The host remains frozen/eval.
    sidecar.train()
    model.eval()
    records = list(records)
    random.Random(args.seed + epoch).shuffle(records)
    running = dict(loss=0.0, focal=0.0, dice=0.0,
                   grad_norm=0.0, max_loss=0.0, frames=0)
    for index, record in enumerate(records, 1):
        loaded = load_frame(
            record, args, transform_compose, img_scale, flip)
        if loaded is None:
            continue
        image, meta, _, gt, _ = loaded
        beam_gt = entry_probe.gt_to_tensor(
            pool_probe.scale_gt_to_img(gt, meta), image.device)
        platform_gt = candidate_platform_boxes(beam_gt, seq_k[record[1]])
        rgb = normalized_to_rgb01(image, cfg)
        rgb, platform_gt = augment_rgb_and_box(rgb, platform_gt, args)
        logits = sidecar(rgb)
        target = target_from_obbs(
            platform_gt, logits.shape[-2:], image.shape[-2:])
        if target.sum() <= 0:
            continue
        loss, parts = platform_heatmap_loss(
            logits, target, focal_gamma=args.focal_gamma,
            pos_alpha=args.pos_alpha, dice_weight=args.dice_weight)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f'Non-finite sidecar loss at record={record}: '
                f'loss={float(loss.detach())}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            sidecar.parameters(), args.grad_clip if args.grad_clip > 0
            else float('inf'))
        optimizer.step()
        assert_isolated(model, sidecar)

        running['loss'] += float(loss.detach())
        running['focal'] += float(parts['focal'])
        running['dice'] += float(parts['dice'])
        running['grad_norm'] += float(grad_norm)
        running['max_loss'] = max(running['max_loss'], float(loss.detach()))
        running['frames'] += 1
        if index % 100 == 0:
            denom = max(running['frames'], 1)
            print(f'[train] epoch={epoch:02d} {index}/{len(records)} '
                  f'loss={running["loss"] / denom:.5f}')
    denom = max(running['frames'], 1)
    return {
        key: (int(value) if key == 'frames' else
              float(value) if key == 'max_loss' else float(value) / denom)
        for key, value in running.items()
    }


def evaluate_manual(sidecar, items, cfg, transform_compose, img_scale,
                    flip, args, label: str):
    diag = entry_probe.get_diag()
    sidecar.eval()
    rows = []
    for item in items:
        frame, split, seq = (
            int(item['frame']), str(item['split']), str(item['seq']))
        img_path, _ = diag.find_files(args.data_root, split, seq, frame)
        if img_path is None:
            continue
        image, meta, _ = diag.preprocess_image(
            img_path, transform_compose, img_scale, flip)
        if image is None:
            continue
        image = image.cuda(f'cuda:{args.gpu}')
        with torch.no_grad():
            prob = torch.sigmoid(sidecar(normalized_to_rgb01(image, cfg)))[
                0, 0].detach().cpu().numpy()
        target = manual_target(
            item, meta, prob.shape, image.shape[-2:])
        pos_count = int(target.sum())
        if pos_count <= 0:
            continue
        peak_index = np.unravel_index(int(np.argmax(prob)), prob.shape)
        peak_hit = bool(target[peak_index])
        inside = float(prob[target].mean())
        outside = float(prob[~target].mean()) if (~target).any() else 0.0
        contrast = inside - outside

        # Calibration-free localization IoU: compare the manual mask with the
        # same-area set of highest heatmap pixels.
        top_indices = np.argpartition(prob.reshape(-1), -pos_count)[-pos_count:]
        pred_mask = np.zeros(prob.size, dtype=bool)
        pred_mask[top_indices] = True
        pred_mask = pred_mask.reshape(prob.shape)
        intersection = int(np.logical_and(pred_mask, target).sum())
        union = int(np.logical_or(pred_mask, target).sum())
        area_iou = intersection / union if union else 0.0
        rows.append(dict(
            split=split, seq=seq, frame=frame, peak_hit=peak_hit,
            area_matched_iou=float(area_iou), inside_mean=inside,
            outside_mean=outside, contrast=contrast,
            peak=float(prob[peak_index]), peak_y=int(peak_index[0]),
            peak_x=int(peak_index[1])))
        print(f'[{label}] {seq}_{frame:05d} peak_hit={int(peak_hit)} '
              f'area_IoU={area_iou:.3f} contrast={contrast:.3f}')
    ious = [row['area_matched_iou'] for row in rows]
    contrasts = [row['contrast'] for row in rows]
    hits = sum(bool(row['peak_hit']) for row in rows)
    summary = dict(
        frames=len(rows), peak_hits=hits,
        peak_recall=hits / len(rows) if rows else 0.0,
        area_iou_mean=float(np.mean(ious)) if ious else 0.0,
        area_iou_median=float(np.median(ious)) if ious else 0.0,
        contrast_mean=float(np.mean(contrasts)) if contrasts else 0.0,
        contrast_median=float(np.median(contrasts)) if contrasts else 0.0)
    return rows, summary


def heatmap_reliability(prob: torch.Tensor) -> Tuple[float, float]:
    flat = prob.reshape(-1)
    peak = float(flat.max().item())
    median = float(flat.median().item())
    return peak, peak - median


def context_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(np.percentile(values, 10))
    high = float(np.percentile(values, 99))
    denom = max(high - low, 1e-6)
    return np.clip((values - low) / denom, 1e-4, 1.0)


def select_metric(mode: str, cls_scores: np.ndarray,
                  context_scores: np.ndarray, beam_ious: np.ndarray,
                  reliable: bool) -> np.ndarray:
    if mode == 'beam_oracle':
        return beam_ious
    if mode == 'cls':
        return cls_scores
    if mode == 'context_only':
        return context_scores
    if mode.startswith('log_lambda_'):
        if not reliable:
            return cls_scores
        lam = float(mode[len('log_lambda_'):])
        context = context_normalize(context_scores)
        return (np.log(np.clip(cls_scores, 1e-12, None))
                + lam * np.log(context))
    raise ValueError(mode)


def mode_names(lambdas):
    return ['cls', 'beam_oracle', 'context_only'] + [
        f'log_lambda_{value:g}' for value in lambdas]


def evaluate_rerank(model, sidecar, eval_k, cfg, transform_compose,
                    img_scale, flip, args):
    from mmcv.ops import box_iou_rotated

    diag = entry_probe.get_diag()
    topks = pool_probe.normalize_topks(args.topks)
    modes = mode_names(args.log_lambdas)
    rows = []
    sidecar.eval()
    for frame in range(args.eval_start, args.eval_end + 1):
        img_path, ann_path = diag.find_files(
            args.data_root, args.eval_split, args.eval_seq, frame)
        gts = diag.parse_dota_ann(ann_path) if ann_path else []
        if img_path is None or not gts:
            continue
        image, meta, stats = diag.preprocess_image(
            img_path, transform_compose, img_scale, flip)
        if image is None:
            continue
        image = image.cuda(f'cuda:{args.gpu}')
        with torch.no_grad():
            features = model.extract_feat(image)
            candidate_head, cls_scores, bbox_preds = (
                entry_probe.forward_candidate_head(
                    model, features, args.candidate_source))
            boxes, scores, levels, _, _ = (
                entry_probe.flatten_decode_candidates(
                    candidate_head, cls_scores, bbox_preds,
                    meta['img_shape']))
            platform_prob = torch.sigmoid(
                sidecar(normalized_to_rgb01(image, cfg)))

        max_k = min(max(topks), int(scores.numel()))
        top_scores_t, indices = torch.topk(scores, k=max_k, sorted=True)
        top_boxes_t = boxes[indices]
        top_levels_t = levels[indices]
        gt = pool_probe.scale_gt_to_img(gts[0], meta)
        gt_box = entry_probe.gt_to_tensor(gt, boxes.device)
        top_ious_t = box_iou_rotated(
            top_boxes_t.float(), gt_box.float()).reshape(-1)
        mapped_platforms = candidate_platform_boxes(top_boxes_t, eval_k)
        context_t = sample_candidate_context(
            platform_prob, mapped_platforms, image.shape[-2:],
            grid_size=args.sample_grid_size)
        peak, contrast = heatmap_reliability(platform_prob)
        reliable = bool(
            peak >= args.gate_peak and contrast >= args.gate_contrast)

        top_scores = top_scores_t.detach().cpu().numpy().astype(np.float64)
        top_boxes = top_boxes_t.detach().cpu().numpy().astype(np.float32)
        top_levels = top_levels_t.detach().cpu().numpy().astype(np.int64)
        top_ious = top_ious_t.detach().cpu().numpy().astype(np.float64)
        context = context_t.detach().cpu().numpy().astype(np.float64)
        per_k = {}
        for topk in topks:
            actual_k = min(topk, max_k)
            per_mode = {}
            for mode in modes:
                metric = select_metric(
                    mode, top_scores[:actual_k], context[:actual_k],
                    top_ious[:actual_k], reliable)
                selected = int(np.argmax(metric))
                riou = float(top_ious[selected])
                per_mode[mode] = dict(
                    selected_rank=selected + 1,
                    selected_beam_riou=riou,
                    selected_cls_score=float(top_scores[selected]),
                    selected_context_score=float(context[selected]),
                    selected_level=int(top_levels[selected]),
                    selected_beam_box=top_boxes[selected].astype(float).tolist(),
                    hit=bool(riou >= args.riou_thr))
            per_k[str(topk)] = dict(actual_k=actual_k, modes=per_mode)
        row = dict(
            frame=frame,
            fname=os.path.splitext(os.path.basename(img_path))[0],
            brightness=float(stats['raw_brightness']),
            global_max=float(top_scores[0]),
            platform_peak=peak,
            platform_contrast=contrast,
            sidecar_reliable=reliable,
            per_k=per_k)
        rows.append(row)
        focus_k = 500 if 500 in topks else topks[-1]
        focus = per_k[str(focus_k)]['modes']
        positive_lambdas = [value for value in args.log_lambdas if value > 0]
        display_lambda = positive_lambdas[0] if positive_lambdas else 0.0
        best_fusion = f'log_lambda_{display_lambda:g}'
        print(f'[{row["fname"]}] reliable={int(reliable)} '
              f'peak={peak:.3f} cls={focus["cls"]["selected_beam_riou"]:.3f} '
              f'ctx={focus[best_fusion]["selected_beam_riou"]:.3f} '
              f'ceiling={focus["beam_oracle"]["selected_beam_riou"]:.3f}')
    assert_isolated(model, sidecar)
    return rows, summarize_rerank(rows, topks, modes)


def summarize_rerank(rows, topks, modes):
    summary = dict(
        frames=len(rows),
        reliable_frames=sum(bool(row['sidecar_reliable']) for row in rows),
        per_k={})
    for topk in topks:
        per_mode = {}
        for mode in modes:
            hit_key = f'hit_{topk}_{mode}'
            proxies, rious = [], []
            for row in rows:
                item = row['per_k'][str(topk)]['modes'][mode]
                proxies.append(dict(frame=row['frame'], **{hit_key: item['hit']}))
                rious.append(float(item['selected_beam_riou']))
            hits = sum(bool(item[hit_key]) for item in proxies)
            per_mode[mode] = dict(
                hits=hits,
                recall=hits / len(rows) if rows else 0.0,
                mcml=pool_probe.longest_consecutive_miss(
                    proxies, hit_key) if rows else 0,
                mean_riou=float(np.mean(rious)) if rious else 0.0,
                min_riou=float(np.min(rious)) if rious else 0.0,
                max_riou=float(np.max(rious)) if rious else 0.0)
        summary['per_k'][str(topk)] = per_mode
    return summary


def print_rerank_summary(summary, topks, modes):
    print('\n' + '=' * 108)
    print('PROBE 5 SUMMARY: BRIGHTAUG + INDEPENDENT RGB PLATFORM SIDECAR')
    print('=' * 108)
    print(f"frames={summary['frames']}  "
          f"reliable_sidecar_frames={summary['reliable_frames']}")
    for topk in topks:
        print(f'K={topk}')
        print(f"  {'mode':<24} {'hits':>10} {'recall':>10} "
              f"{'MCML':>8} {'mean_RIoU':>12} {'min':>8} {'max':>8}")
        for mode in modes:
            item = summary['per_k'][str(topk)][mode]
            print(f"  {mode:<24} {item['hits']:>4d}/{summary['frames']:<5d} "
                  f"{item['recall']:>10.3f} {item['mcml']:>8d} "
                  f"{item['mean_riou']:>12.3f} "
                  f"{item['min_riou']:>8.3f} {item['max_riou']:>8.3f}")


def cpu_state_dict(module):
    return {key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()}


def load_sidecar(path: str, sidecar):
    payload = torch.load(path, map_location='cpu')
    state_dict = payload.get('state_dict', payload)
    sidecar.load_state_dict(state_dict, strict=True)
    return payload


def main():
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Probe 5 requires CUDA for MMRotate rotated ops')
    if args.eval_end < args.eval_start:
        raise ValueError('--eval-end must be >= --eval-start')

    model, cfg = entry_probe.load_model(
        args.config, args.checkpoint, args.gpu)
    freeze_detector(model)
    transform_compose, img_scale, flip = (
        entry_probe.get_diag().build_transform_from_cfg(cfg))
    train_seq_k = load_train_seq_k(args.train_k_config)
    eval_k = median_k(train_seq_k)
    sidecar = PlatformContextSidecar(
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels).cuda(f'cuda:{args.gpu}')
    assert_isolated(model, sidecar)

    manual_val_items = load_manual_items(
        args.manual_train_json, seq=args.holdout_seq)
    manual_test_items = load_manual_items(
        args.manual_test_json, split=args.eval_split, seq=args.eval_seq,
        start=args.eval_start, end=args.eval_end)
    if not manual_val_items:
        raise RuntimeError(
            f'No manual validation polygons for holdout {args.holdout_seq}')

    history, val_history = [], []
    checkpoint_meta = None
    if args.eval_only:
        sidecar_path = args.sidecar_in or args.sidecar_out
        checkpoint_meta = load_sidecar(sidecar_path, sidecar)
        print(f'[load] sidecar: {sidecar_path}')
    else:
        train_seqs = sorted(set(train_seq_k) - {args.holdout_seq})
        train_records = enumerate_records(
            args.data_root, args.train_splits, train_seqs)
        train_records = sample_records(
            train_records, args.max_train_frames, args.seed)
        if not train_records:
            raise RuntimeError('No sidecar training records found')
        print(f'[data] train_seqs={train_seqs} frames={len(train_records)} '
              f'holdout={args.holdout_seq} manual_val={len(manual_val_items)}')
        optimizer = torch.optim.AdamW(
            sidecar.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)
        best_score, best_state, best_epoch = -float('inf'), None, None
        for epoch in range(1, args.epochs + 1):
            train_row = train_one_epoch(
                model, sidecar, optimizer, train_records, train_seq_k, cfg,
                transform_compose, img_scale, flip, args, epoch)
            val_rows, val_row = evaluate_manual(
                sidecar, manual_val_items, cfg, transform_compose,
                img_scale, flip, args, label='manual-val')
            train_row['epoch'] = epoch
            val_row['epoch'] = epoch
            val_row['rows'] = val_rows
            history.append(train_row)
            val_history.append(val_row)
            selection_score = (
                10.0 * val_row['peak_recall']
                + val_row['area_iou_median']
                + val_row['contrast_median'])
            print(f'[epoch] {epoch:02d} loss={train_row["loss"]:.5f} '
                  f'focal={train_row["focal"]:.5f} '
                  f'dice={train_row["dice"]:.5f} '
                  f'grad={train_row["grad_norm"]:.3f} '
                  f'val_peak={val_row["peak_recall"]:.3f} '
                  f'val_IoU={val_row["area_iou_median"]:.3f} '
                  f'val_contrast={val_row["contrast_median"]:.3f}')
            if selection_score > best_score:
                best_score = selection_score
                best_state = cpu_state_dict(sidecar)
                best_epoch = epoch
        if best_state is None:
            raise RuntimeError('No sidecar checkpoint was produced')
        sidecar.load_state_dict(best_state, strict=True)
        checkpoint_meta = dict(
            state_dict=best_state,
            architecture=dict(
                base_channels=args.base_channels,
                decoder_channels=args.decoder_channels,
                output='stride-4 internal platform heatmap'),
            host=dict(config=args.config, checkpoint=args.checkpoint,
                      frozen=True, eval_mode=True),
            train_seq_k=train_seq_k,
            holdout_seq=args.holdout_seq,
            best_epoch=best_epoch,
            history=history,
            val_history=val_history,
            isolation=dict(
                independent_rgb_encoder=True,
                host_forward_no_grad=True,
                optimizer_scope='PlatformContextSidecar only',
                shared_backbone_fpn=False))
        ensure_parent(args.sidecar_out)
        torch.save(checkpoint_meta, args.sidecar_out)
        print(f'[out] sidecar: {args.sidecar_out}')

    manual_rows, manual_summary = evaluate_manual(
        sidecar, manual_test_items, cfg, transform_compose,
        img_scale, flip, args, label='manual-test')
    manual_gate_pass = bool(
        manual_summary['frames'] >= args.manual_min_frames
        and manual_summary['peak_hits'] >= args.manual_min_hits
        and manual_summary['area_iou_median']
        >= args.manual_min_median_iou)
    print('[manual-test-summary] '
          + json.dumps(manual_summary, ensure_ascii=False))
    print('[manual-test-gate] ' + ('PASS' if manual_gate_pass else 'FAIL'))

    rows, rerank_summary = evaluate_rerank(
        model, sidecar, eval_k, cfg, transform_compose,
        img_scale, flip, args)
    topks = pool_probe.normalize_topks(args.topks)
    modes = mode_names(args.log_lambdas)
    print_rerank_summary(rerank_summary, topks, modes)

    result = dict(
        probe='brightaug_independent_rgb_platform_context_sidecar',
        args=vars(args),
        isolation=dict(
            host='selected BrightAug checkpoint',
            detector_frozen=True,
            detector_eval_mode=True,
            independent_rgb_encoder=True,
            shared_backbone_fpn=False,
            optimizer_scope='PlatformContextSidecar only',
            inference_fusion_only=True,
            output_contract='unchanged main-head beam OBB only',
            platform_box_output=False,
            inverse_k=False),
        eval_k=eval_k,
        checkpoint=dict(
            best_epoch=checkpoint_meta.get('best_epoch')
            if isinstance(checkpoint_meta, dict) else None,
            path=args.sidecar_in or args.sidecar_out),
        history=history,
        val_history=val_history,
        manual_test=dict(
            rows=manual_rows,
            summary=manual_summary,
            gate_pass=manual_gate_pass,
            gate_rule=(
                f'>={args.manual_min_hits}/{args.manual_min_frames} peak hits '
                f'and median area-matched IoU '
                f'>={args.manual_min_median_iou:g}')),
        rerank=dict(rows=rows, summary=rerank_summary))
    ensure_parent(args.out_json)
    with open(args.out_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f'[out] result: {args.out_json}')


if __name__ == '__main__':
    main()
