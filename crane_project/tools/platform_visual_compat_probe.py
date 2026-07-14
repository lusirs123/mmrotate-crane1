#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe 3: candidate-conditioned visual platform compatibility.

This is a deliberately isolated feasibility probe, not a new detector training
recipe.  A frozen SymEOOD checkpoint first produces pre-threshold beam
candidates.  Each candidate is mapped only in the forward direction
``beam -> platform ROI`` and a small CNN judges whether the image crop looks
like a real grab platform.

Safety/contract:
  * the backbone, FPN and beam head are frozen and always run under no_grad;
  * candidate boxes/scores are detached before platform processing;
  * the optimizer owns only PlatformCropCompatHead parameters;
  * test-time reranking is part of this probe (the branch is not train-only);
  * the selected/output OBB is always an unchanged main-head beam candidate;
  * platform geometry is never inverted to synthesize a beam box;
  * test annotations are used only to report RIoU/MCML, never for training.

The optional ``manual`` eval K source is an oracle-calibration upper bound.  It
fits K from the supplied test platform polygons, so it must not be presented as
a deployment-clean result.  Use ``train_median`` for the leakage-free control.
"""

import argparse
import ast
import glob
import json
import math
import os
import random
import re
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

pool_probe = None
entry_probe = None
platform_probe = None


def load_project_helpers():
    """Delay MMRotate/MMCV imports so ``--help`` works off-server."""
    global pool_probe, entry_probe, platform_probe
    from crane_project.tools import candidate_pool_oracle_probe
    from crane_project.tools import ctx_entry_probe
    from crane_project.tools import platform_context_probe
    pool_probe = candidate_pool_oracle_probe
    entry_probe = ctx_entry_probe
    platform_probe = platform_context_probe


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train/evaluate an isolated visual platform reranker.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-k-config', required=True,
                        help='Config containing train-only seq_platform_k.')
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--train-splits', nargs='+', default=['train', 'train_sim'])
    parser.add_argument('--max-train-frames', type=int, default=800)
    parser.add_argument('--train-topk', type=int, default=300)
    parser.add_argument('--positive-riou', type=float, default=0.5)
    parser.add_argument('--negative-riou', type=float, default=0.3)
    parser.add_argument('--negatives-per-frame', type=int, default=4)
    parser.add_argument('--crop-size', type=int, default=64)
    parser.add_argument('--crop-scale', type=float, default=1.25)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--pair-loss-weight', type=float, default=1.0)
    parser.add_argument('--bce-loss-weight', type=float, default=0.25)
    parser.add_argument('--dark-aug-prob', type=float, default=0.7)
    parser.add_argument('--eval-split', default='test')
    parser.add_argument('--eval-seq', default='real_seq02')
    parser.add_argument('--eval-start', type=int, default=133)
    parser.add_argument('--eval-end', type=int, default=171)
    parser.add_argument('--eval-topks', type=int, nargs='+',
                        default=[200, 500, 1000])
    parser.add_argument('--log-lambdas', type=float, nargs='+',
                        default=[0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument('--eval-k-source', choices=['train_median', 'manual'],
                        default='train_median')
    parser.add_argument('--eval-manual-platform-json', default='')
    parser.add_argument('--candidate-source', default='main',
                        choices=['main', 'aux1'])
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--head-in', default='',
                        help='Existing Probe-3 head checkpoint for eval-only.')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip pair extraction/training and load --head-in.')
    parser.add_argument('--head-out', required=True)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


class PlatformCropCompatHead(nn.Module):
    """Small crop classifier; it has no reference to the detector graph."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 1)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, crops):
        crops = (crops - 0.5) / 0.25
        return self.classifier(self.features(crops).flatten(1)).flatten()


class CropPairDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[np.ndarray, np.ndarray]]):
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pos, neg = self.pairs[index]
        pos_t = torch.from_numpy(pos.copy()).permute(2, 0, 1).float() / 255.0
        neg_t = torch.from_numpy(neg.copy()).permute(2, 0, 1).float() / 255.0
        return pos_t, neg_t


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_detector(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
        param.grad = None


def assert_detector_isolated(model):
    bad_requires_grad = [name for name, p in model.named_parameters()
                         if p.requires_grad]
    bad_grads = [name for name, p in model.named_parameters()
                 if p.grad is not None]
    if bad_requires_grad or bad_grads:
        raise RuntimeError(
            'Detector isolation violated: '
            f'requires_grad={bad_requires_grad[:5]} grads={bad_grads[:5]}')


def parse_seq_frame(path: str) -> Optional[Tuple[str, int]]:
    name = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r'((?:real|sim)_seq\d+)_(\d+)$', name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _safe_config_value(node):
    """Evaluate the literal/dict(...) subset used by MMConfig files."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        return {
            _safe_config_value(key): _safe_config_value(value)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_safe_config_value(item) for item in node.elts]
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_config_value(node.operand)
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'dict'
            and not node.args):
        return {
            keyword.arg: _safe_config_value(keyword.value)
            for keyword in node.keywords
            if keyword.arg is not None
        }
    raise ValueError(
        f'Unsupported expression in seq_platform_k: {ast.dump(node)}')


def load_train_seq_k(config_path: str) -> Dict[str, Dict]:
    """Read only seq_platform_k without resolving a config's _base_ chain.

    Archived configs may have stale relative ``_base_`` paths after they are
    moved.  Probe 3 needs only this standalone calibration dictionary, so
    parsing the assignment directly is both narrower and more robust than
    ``mmcv.Config.fromfile``.
    """
    with open(config_path, 'r', encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=config_path)
    seq_k = None
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(isinstance(target, ast.Name)
               and target.id == 'seq_platform_k'
               for target in statement.targets):
            seq_k = _safe_config_value(statement.value)
            break
    if not seq_k:
        raise RuntimeError(f'No seq_platform_k found in {config_path}')
    return {str(key): dict(value) for key, value in seq_k.items()}


def median_k(seq_k: Dict[str, Dict]) -> Dict:
    keys = ('width_k', 'height_k', 'offset_long_k',
            'offset_short_k', 'dtheta')
    result = {}
    for key in keys:
        values = [float(item.get(key, 0.0)) for item in seq_k.values()]
        result[key] = float(np.median(values))
    result['source'] = 'train_seq_k_median'
    result['sequences'] = sorted(seq_k)
    return result


def fit_manual_eval_k(path: str, args) -> Dict:
    if not path:
        raise ValueError('--eval-manual-platform-json is required for manual K')
    manual = platform_probe.load_manual_platforms(
        path, args.eval_split, args.eval_seq)
    fit_args = SimpleNamespace(
        data_root=args.data_root,
        split=args.eval_split,
        seq=args.eval_seq,
        seq_platform_angle_mode='beam')
    seq_k = platform_probe.fit_seq_platform_k(manual, fit_args)
    if seq_k is None:
        raise RuntimeError('Could not fit eval K from manual platform polygons')
    seq_k['source'] = 'manual_test_polygon_oracle_calibration'
    return seq_k


def enumerate_training_records(data_root: str, splits: Sequence[str],
                               seq_k: Dict[str, Dict], seed: int,
                               max_frames: int) -> List[Tuple[str, str, int]]:
    records = []
    for split in splits:
        ann_dir = os.path.join(data_root, split, 'annfiles')
        for ann_path in sorted(glob.glob(os.path.join(ann_dir, '*.txt'))):
            parsed = parse_seq_frame(ann_path)
            if parsed is None:
                continue
            seq, frame = parsed
            if seq in seq_k:
                records.append((split, seq, frame))
    rng = random.Random(seed)
    rng.shuffle(records)
    if max_frames > 0:
        records = records[:max_frames]
    return records


def tensor_to_rgb_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    image = img_tensor[0].detach().float().cpu().permute(1, 2, 0).numpy()
    mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
    return np.clip(image * std + mean, 0, 255).astype(np.uint8)


def obbs_to_polygons(boxes: np.ndarray) -> np.ndarray:
    from mmrotate.core import obb2poly_np

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    with_score = np.concatenate(
        [boxes, np.zeros((len(boxes), 1), dtype=np.float32)], axis=1)
    polys = obb2poly_np(with_score, version='le90')[:, :8]
    return np.asarray(polys, dtype=np.float32).reshape(-1, 4, 2)


def expand_polygon(poly: np.ndarray, scale: float) -> np.ndarray:
    center = np.asarray(poly, dtype=np.float32).mean(axis=0, keepdims=True)
    return center + (np.asarray(poly, dtype=np.float32) - center) * float(scale)


def crop_polygon(image_rgb: np.ndarray, poly: np.ndarray,
                 crop_size: int, crop_scale: float) -> np.ndarray:
    src = expand_polygon(poly, crop_scale).astype(np.float32)
    side = int(crop_size)
    dst = np.asarray([
        [0.0, 0.0], [side - 1.0, 0.0],
        [side - 1.0, side - 1.0], [0.0, side - 1.0],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image_rgb, matrix, (side, side), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))


def candidate_platform_polygons(boxes: np.ndarray, seq_k: Dict) -> np.ndarray:
    beam_polys = obbs_to_polygons(boxes)
    return np.asarray([
        platform_probe.platform_poly_from_seq_k(poly, seq_k)
        for poly in beam_polys
    ], dtype=np.float32)


def polygon_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    a = platform_probe.order_corners(poly_a).astype(np.float32)
    b = platform_probe.order_corners(poly_b).astype(np.float32)
    area_a = abs(float(cv2.contourArea(a)))
    area_b = abs(float(cv2.contourArea(b)))
    if area_a <= 1e-6 or area_b <= 1e-6:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a, b)
    return float(inter) / max(area_a + area_b - float(inter), 1e-6)


def frozen_candidates(model, img_tensor, meta, candidate_source: str):
    with torch.no_grad():
        feats = model.extract_feat(img_tensor)
        candidate_head, cls_scores, bbox_preds = (
            entry_probe.forward_candidate_head(
                model, feats, candidate_source))
        boxes, scores, levels, _, _ = entry_probe.flatten_decode_candidates(
            candidate_head, cls_scores, bbox_preds, meta['img_shape'])
    return boxes.detach(), scores.detach(), levels.detach()


def choose_negative_indices(scores: np.ndarray, beam_ious: np.ndarray,
                            platform_ious: np.ndarray, negative_thr: float,
                            count: int) -> List[int]:
    valid = np.flatnonzero(beam_ious < float(negative_thr))
    if valid.size == 0:
        return []
    cls_order = valid[np.argsort(-scores[valid])]
    plat_order = valid[np.argsort(-platform_ious[valid])]
    selected = []
    for source in (cls_order, plat_order):
        for index in source.tolist():
            if index not in selected:
                selected.append(index)
            if len(selected) >= count:
                return selected
    return selected[:count]


def extract_training_pairs(model, transform_compose, img_scale, flip,
                           records, seq_k, args):
    from mmcv.ops import box_iou_rotated

    diag = entry_probe.get_diag()
    pairs = []
    stats = dict(requested=len(records), processed=0, no_positive=0,
                 no_negative=0, pairs=0)
    for ordinal, (split, seq, frame) in enumerate(records, 1):
        img_path, ann_path = diag.find_files(args.data_root, split, seq, frame)
        gts = diag.parse_dota_ann(ann_path) if ann_path else []
        if img_path is None or not gts:
            continue
        img_tensor, meta, _ = diag.preprocess_image(
            img_path, transform_compose, img_scale, flip)
        if img_tensor is None:
            continue
        img_tensor = img_tensor.cuda(f'cuda:{args.gpu}')
        boxes, scores, _ = frozen_candidates(
            model, img_tensor, meta, args.candidate_source)
        k = min(int(args.train_topk), int(scores.numel()))
        top_scores_t, top_indices = torch.topk(scores, k=k, sorted=True)
        top_boxes_t = boxes[top_indices]
        gt = pool_probe.scale_gt_to_img(gts[0], meta)
        gt_box = entry_probe.gt_to_tensor(gt, boxes.device)
        beam_ious_t = box_iou_rotated(
            top_boxes_t.float(), gt_box.float()).reshape(-1)

        top_scores = top_scores_t.cpu().numpy().astype(np.float64)
        top_boxes = top_boxes_t.cpu().numpy().astype(np.float32)
        beam_ious = beam_ious_t.cpu().numpy().astype(np.float64)
        positives = np.flatnonzero(beam_ious >= float(args.positive_riou))
        if positives.size == 0:
            stats['no_positive'] += 1
            continue
        positive = int(positives[np.argmax(top_scores[positives])])

        platform_polys = candidate_platform_polygons(top_boxes, seq_k[seq])
        gt_poly = obbs_to_polygons(np.asarray([[
            gt['cx'], gt['cy'], gt['w'], gt['h'],
            math.radians(gt['angle'])]], dtype=np.float32))[0]
        expected_platform = platform_probe.platform_poly_from_seq_k(
            gt_poly, seq_k[seq])
        platform_ious = np.asarray([
            polygon_iou(poly, expected_platform) for poly in platform_polys
        ], dtype=np.float64)
        negatives = choose_negative_indices(
            top_scores, beam_ious, platform_ious, args.negative_riou,
            args.negatives_per_frame)
        if not negatives:
            stats['no_negative'] += 1
            continue

        image_rgb = tensor_to_rgb_uint8(img_tensor)
        pos_crop = crop_polygon(
            image_rgb, platform_polys[positive],
            args.crop_size, args.crop_scale)
        for negative in negatives:
            neg_crop = crop_polygon(
                image_rgb, platform_polys[negative],
                args.crop_size, args.crop_scale)
            pairs.append((pos_crop, neg_crop))
        stats['processed'] += 1
        stats['pairs'] = len(pairs)
        if ordinal % 50 == 0:
            print(f'[train-cache] {ordinal}/{len(records)} '
                  f'usable_frames={stats["processed"]} pairs={len(pairs)}')

    assert_detector_isolated(model)
    return pairs, stats


def augment_crops(crops: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return crops
    batch = crops.shape[0]
    device = crops.device
    active = (torch.rand(batch, 1, 1, 1, device=device) < probability).float()
    gain = 0.20 + 0.80 * torch.rand(batch, 1, 1, 1, device=device)
    gamma = 0.70 + 1.80 * torch.rand(batch, 1, 1, 1, device=device)
    color = 0.85 + 0.30 * torch.rand(batch, 3, 1, 1, device=device)
    degraded = torch.clamp(crops * color, 0, 1).pow(gamma) * gain
    noise = torch.randn_like(degraded) * (
        0.03 * torch.rand(batch, 1, 1, 1, device=device))
    degraded = torch.clamp(degraded + noise, 0, 1)
    if random.random() < probability * 0.5:
        degraded = F.avg_pool2d(degraded, kernel_size=3, stride=1, padding=1)
    return crops * (1.0 - active) + degraded * active


def train_head(head, pairs, model, args):
    dataset = CropPairDataset(pairs)
    if not dataset:
        raise RuntimeError('No training pairs were extracted')
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    head.train()
    for epoch in range(1, args.epochs + 1):
        sum_loss = sum_pair = sum_bce = sum_correct = count = 0.0
        for pos, neg in loader:
            pos = pos.cuda(f'cuda:{args.gpu}', non_blocking=True)
            neg = neg.cuda(f'cuda:{args.gpu}', non_blocking=True)
            both = augment_crops(
                torch.cat([pos, neg], dim=0), args.dark_aug_prob)
            pos_aug, neg_aug = both.chunk(2, dim=0)
            pos_logit = head(pos_aug)
            neg_logit = head(neg_aug)
            pair_loss = F.softplus(neg_logit - pos_logit).mean()
            bce_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    pos_logit, torch.ones_like(pos_logit))
                + F.binary_cross_entropy_with_logits(
                    neg_logit, torch.zeros_like(neg_logit)))
            loss = (args.pair_loss_weight * pair_loss
                    + args.bce_loss_weight * bce_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            assert_detector_isolated(model)

            n = int(pos.shape[0])
            sum_loss += float(loss.item()) * n
            sum_pair += float(pair_loss.item()) * n
            sum_bce += float(bce_loss.item()) * n
            sum_correct += float((pos_logit > neg_logit).sum().item())
            count += n
        row = dict(
            epoch=epoch,
            loss=sum_loss / max(count, 1),
            pair_loss=sum_pair / max(count, 1),
            bce_loss=sum_bce / max(count, 1),
            pair_accuracy=sum_correct / max(count, 1))
        history.append(row)
        print(f'[train] epoch={epoch:02d} loss={row["loss"]:.5f} '
              f'pair_acc={row["pair_accuracy"]:.3f}')
    return history


def score_crops(head, image_rgb: np.ndarray, platform_polys: np.ndarray,
                args) -> np.ndarray:
    crops = np.stack([
        crop_polygon(image_rgb, poly, args.crop_size, args.crop_scale)
        for poly in platform_polys
    ], axis=0)
    tensor = torch.from_numpy(crops).permute(0, 3, 1, 2).float() / 255.0
    outputs = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(tensor), args.batch_size):
            batch = tensor[start:start + args.batch_size].cuda(
                f'cuda:{args.gpu}', non_blocking=True)
            outputs.append(torch.sigmoid(head(batch)).cpu())
    return torch.cat(outputs).numpy().astype(np.float64)


def mode_names(lambdas: Sequence[float]) -> List[str]:
    names = ['cls', 'beam_oracle', 'compat_only', 'cls_x_compat']
    names.extend(f'log_lambda_{value:g}' for value in lambdas)
    return names


def select_index(mode: str, cls_scores: np.ndarray,
                 compat_scores: np.ndarray, beam_ious: np.ndarray) -> int:
    eps = 1e-12
    if mode == 'cls':
        metric = cls_scores
    elif mode == 'beam_oracle':
        metric = beam_ious
    elif mode == 'compat_only':
        metric = compat_scores
    elif mode == 'cls_x_compat':
        metric = cls_scores * compat_scores
    elif mode.startswith('log_lambda_'):
        lam = float(mode[len('log_lambda_'):])
        metric = (np.log(np.clip(cls_scores, eps, None))
                  + lam * np.log(np.clip(compat_scores, eps, None)))
    else:
        raise ValueError(mode)
    return int(np.argmax(metric))


def evaluate_head(head, model, transform_compose, img_scale, flip,
                  eval_k: Dict, args):
    from mmcv.ops import box_iou_rotated

    diag = entry_probe.get_diag()
    topks = pool_probe.normalize_topks(args.eval_topks)
    modes = mode_names(args.log_lambdas)
    rows = []
    for frame in range(args.eval_start, args.eval_end + 1):
        img_path, ann_path = diag.find_files(
            args.data_root, args.eval_split, args.eval_seq, frame)
        gts = diag.parse_dota_ann(ann_path) if ann_path else []
        if img_path is None or not gts:
            continue
        img_tensor, meta, stats = diag.preprocess_image(
            img_path, transform_compose, img_scale, flip)
        if img_tensor is None:
            continue
        img_tensor = img_tensor.cuda(f'cuda:{args.gpu}')
        boxes, scores, levels = frozen_candidates(
            model, img_tensor, meta, args.candidate_source)
        max_k = min(max(topks), int(scores.numel()))
        top_scores_t, top_indices = torch.topk(
            scores, k=max_k, sorted=True)
        top_boxes_t = boxes[top_indices]
        top_levels_t = levels[top_indices]
        gt = pool_probe.scale_gt_to_img(gts[0], meta)
        gt_box = entry_probe.gt_to_tensor(gt, boxes.device)
        top_ious_t = box_iou_rotated(
            top_boxes_t.float(), gt_box.float()).reshape(-1)

        top_scores = top_scores_t.cpu().numpy().astype(np.float64)
        top_boxes = top_boxes_t.cpu().numpy().astype(np.float32)
        top_levels = top_levels_t.cpu().numpy().astype(np.int64)
        top_ious = top_ious_t.cpu().numpy().astype(np.float64)
        platform_polys = candidate_platform_polygons(top_boxes, eval_k)
        compat_scores = score_crops(
            head, tensor_to_rgb_uint8(img_tensor), platform_polys, args)

        per_k = {}
        for topk in topks:
            actual_k = min(int(topk), max_k)
            per_mode = {}
            for mode in modes:
                selected = select_index(
                    mode, top_scores[:actual_k],
                    compat_scores[:actual_k], top_ious[:actual_k])
                riou = float(top_ious[selected])
                per_mode[mode] = dict(
                    selected_rank=selected + 1,
                    selected_cls_score=float(top_scores[selected]),
                    selected_compat_score=float(compat_scores[selected]),
                    selected_beam_riou=riou,
                    selected_level=int(top_levels[selected]),
                    selected_beam_box=top_boxes[selected].astype(float).tolist(),
                    hit=bool(riou >= args.riou_thr))
            per_k[str(topk)] = dict(actual_k=actual_k, modes=per_mode)
        row = dict(
            frame=frame,
            fname=os.path.splitext(os.path.basename(img_path))[0],
            brightness=float(stats['raw_brightness']),
            global_max=float(top_scores[0]),
            per_k=per_k)
        rows.append(row)
        focus_k = 500 if 500 in topks else topks[-1]
        focus = per_k[str(focus_k)]['modes']
        print(f'[{row["fname"]}] K={focus_k} '
              f'cls={focus["cls"]["selected_beam_riou"]:.3f} '
              f'compat={focus["compat_only"]["selected_beam_riou"]:.3f} '
              f'ceiling={focus["beam_oracle"]["selected_beam_riou"]:.3f}')

    assert_detector_isolated(model)
    return rows, build_eval_summary(rows, topks, modes)


def build_eval_summary(rows, topks, modes):
    summary = dict(frames=len(rows), per_k={})
    for topk in topks:
        per_mode = {}
        for mode in modes:
            hit_key = f'hit_{topk}_{mode}'
            proxy = []
            rious = []
            for row in rows:
                item = row['per_k'][str(topk)]['modes'][mode]
                proxy.append(dict(
                    frame=row['frame'], **{hit_key: bool(item['hit'])}))
                rious.append(float(item['selected_beam_riou']))
            hits = sum(bool(item[hit_key]) for item in proxy)
            per_mode[mode] = dict(
                hits=hits,
                recall=hits / len(rows) if rows else 0.0,
                mcml=pool_probe.longest_consecutive_miss(
                    proxy, hit_key) if rows else 0,
                mean_riou=float(np.mean(rious)) if rious else 0.0,
                min_riou=float(np.min(rious)) if rious else 0.0,
                max_riou=float(np.max(rious)) if rious else 0.0)
        summary['per_k'][str(topk)] = per_mode
    return summary


def print_eval_summary(summary, topks, modes):
    print('\n' + '=' * 106)
    print('PROBE 3 SUMMARY: VISUAL PLATFORM COMPATIBILITY RERANK')
    print('=' * 106)
    for topk in topks:
        print(f'K={topk}')
        print(f"  {'mode':<24} {'hits':>10} {'recall':>10} "
              f"{'MCML':>8} {'mean_RIoU':>12} {'min':>8} {'max':>8}")
        for mode in modes:
            item = summary['per_k'][str(topk)][mode]
            print(f"  {mode:<24} {item['hits']:>4d}/{summary['frames']:<5d} "
                  f"{item['recall']:>10.3f} {item['mcml']:>8d} "
                  f"{item['mean_riou']:>12.3f} {item['min_riou']:>8.3f} "
                  f"{item['max_riou']:>8.3f}")
        print('-' * 106)


def ensure_parent(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    args = parse_args()
    load_project_helpers()
    if args.eval_end < args.eval_start:
        raise ValueError('--eval-end must be >= --eval-start')
    if args.negative_riou >= args.positive_riou:
        raise ValueError('--negative-riou must be below --positive-riou')
    set_seed(args.seed)

    model, cfg = entry_probe.load_model(
        args.config, args.checkpoint, args.gpu)
    freeze_detector(model)
    assert_detector_isolated(model)
    diag = entry_probe.get_diag()
    transform_compose, img_scale, flip = diag.build_test_transforms(cfg)

    train_seq_k = load_train_seq_k(args.train_k_config)
    head = PlatformCropCompatHead().cuda(f'cuda:{args.gpu}')
    if args.eval_only:
        if not args.head_in:
            raise ValueError('--eval-only requires --head-in')
        saved = torch.load(args.head_in, map_location='cpu')
        head.load_state_dict(saved['state_dict'], strict=True)
        extraction = saved.get('extraction', {})
        history = saved.get('history', [])
        saved_crop_size = int(saved.get('crop_size', args.crop_size))
        saved_crop_scale = float(saved.get('crop_scale', args.crop_scale))
        if (saved_crop_size != args.crop_size
                or abs(saved_crop_scale - args.crop_scale) > 1e-9):
            raise ValueError(
                'Crop settings differ from the saved head: '
                f'saved=({saved_crop_size}, {saved_crop_scale}) '
                f'args=({args.crop_size}, {args.crop_scale})')
        print(f'[load] eval-only head: {args.head_in}')
    else:
        records = enumerate_training_records(
            args.data_root, args.train_splits, train_seq_k,
            args.seed, args.max_train_frames)
        print(f'[data] train records={len(records)} '
              f'sequences={sorted(train_seq_k)}')
        pairs, extraction = extract_training_pairs(
            model, transform_compose, img_scale, flip,
            records, train_seq_k, args)
        print(f'[data] training pairs={len(pairs)} stats={extraction}')
        history = train_head(head, pairs, model, args)
        ensure_parent(args.head_out)
        torch.save(dict(
            state_dict=head.state_dict(),
            crop_size=args.crop_size,
            crop_scale=args.crop_scale,
            train_seq_k=train_seq_k,
            extraction=extraction,
            history=history), args.head_out)
        print(f'[out] head checkpoint: {args.head_out}')

    if args.eval_k_source == 'manual':
        print('[warning] eval K uses TEST manual platform polygons: '
              'oracle-calibration upper bound, not deployment-clean.')
        eval_k = fit_manual_eval_k(args.eval_manual_platform_json, args)
    else:
        eval_k = median_k(train_seq_k)
    print('[eval-k] ' + json.dumps(eval_k, ensure_ascii=False))

    rows, summary = evaluate_head(
        head, model, transform_compose, img_scale, flip, eval_k, args)
    topks = pool_probe.normalize_topks(args.eval_topks)
    modes = mode_names(args.log_lambdas)
    print_eval_summary(summary, topks, modes)

    result = dict(
        probe='candidate_conditioned_visual_platform_compatibility',
        isolation=dict(
            detector_frozen=True,
            detector_forward_no_grad=True,
            candidates_detached=True,
            optimizer_scope='PlatformCropCompatHead only',
            test_time_rerank=True,
            output_contract='unchanged main-head beam OBB only',
            no_inverse_k=True),
        args=vars(args),
        train_seq_k=train_seq_k,
        eval_k=eval_k,
        extraction=extraction,
        history=history,
        summary=summary,
        rows=rows)
    ensure_parent(args.out_json)
    with open(args.out_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f'[out] result json: {args.out_json}')


if __name__ == '__main__':
    main()
