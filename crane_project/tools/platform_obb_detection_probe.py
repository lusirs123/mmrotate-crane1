#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe 4: detached platform OBB detection and beam-candidate reranking.

The frozen K1 detector supplies FPN features and pre-threshold beam candidates.
An independent one-class PlatformOBBATSSHead is trained from train-only
``platform_gt = K(beam_gt)`` pseudo labels.  At test time it predicts a visual
platform OBB ``P_hat``.  Beam candidates are only mapped forward to platform
space and ranked by ``RIoU(K(B_i), P_hat)``.

Hard contracts:
  * detector parameters and BN statistics stay frozen;
  * platform-head inputs are detached FPN tensors;
  * optimizer owns only platform-head parameters;
  * test-time platform detection and reranking are actually executed;
  * final boxes are unchanged main-head beam candidates;
  * no platform-to-beam inverse transform exists in this script;
  * test beam/platform annotations are evaluation-only.
"""

import argparse
import ast
import copy
import glob
import json
import math
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

pool_probe = None
entry_probe = None
platform_probe = None


def load_project_helpers():
    global pool_probe, entry_probe, platform_probe
    from crane_project.tools import candidate_pool_oracle_probe
    from crane_project.tools import ctx_entry_probe
    from crane_project.tools import platform_context_probe
    # Import registers PlatformOBBATSSHead with MMRotate's registry.
    from mmrotate.models.dense_heads import platform_obb_atss_head  # noqa: F401
    pool_probe = candidate_pool_oracle_probe
    entry_probe = ctx_entry_probe
    platform_probe = platform_context_probe


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detached platform OBB detector and reranking probe.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-k-config', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--train-splits', nargs='+', default=['train', 'train_sim'])
    parser.add_argument('--holdout-seq', default='real_seq01')
    parser.add_argument('--max-train-frames', type=int, default=1200)
    parser.add_argument('--max-val-frames', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--dark-aug-prob', type=float, default=0.5)
    parser.add_argument('--dark-gain-min', type=float, default=0.2)
    parser.add_argument('--dark-gain-max', type=float, default=1.0)
    parser.add_argument('--noise-std-max', type=float, default=20.0)
    parser.add_argument('--platform-head-in', default='')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--platform-head-out', required=True)
    parser.add_argument('--manual-platform-json', required=True)
    parser.add_argument('--eval-k-source', choices=['train_median', 'manual'],
                        default='manual')
    parser.add_argument('--eval-split', default='test')
    parser.add_argument('--eval-seq', default='real_seq02')
    parser.add_argument('--eval-start', type=int, default=133)
    parser.add_argument('--eval-end', type=int, default=171)
    parser.add_argument('--eval-topks', type=int, nargs='+',
                        default=[200, 500, 1000])
    parser.add_argument('--log-lambdas', type=float, nargs='+',
                        default=[0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument('--platform-topm', type=int, default=1)
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--candidate-source', default='main',
                        choices=['main', 'aux1'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-json', required=True)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_detector(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def assert_isolated(model, platform_head):
    detector_trainable = [name for name, param in model.named_parameters()
                          if param.requires_grad]
    detector_grads = [name for name, param in model.named_parameters()
                      if param.grad is not None]
    if detector_trainable or detector_grads:
        raise RuntimeError(
            'Detector gradient isolation failed: '
            f'trainable={detector_trainable[:5]} grads={detector_grads[:5]}')
    if not any(param.requires_grad for param in platform_head.parameters()):
        raise RuntimeError('Platform head has no trainable parameters')


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


def build_platform_head():
    from mmcv import ConfigDict
    from mmrotate.models.builder import build_head

    return build_head(ConfigDict(dict(
        type='PlatformOBBATSSHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        stacked_convs=4,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=1,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range='le90', norm_factor=None,
            edge_swap=True, proj_xy=True,
            target_means=(0., 0., 0., 0., 0.),
            target_stds=(1., 1., 1., 1., 1.)),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True,
            gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                type='ATSSObbAssigner', topk=9, angle_version='le90',
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1, pos_weight=-1, debug=False),
        test_cfg=dict(
            nms_pre=2000, min_bbox_size=0, score_thr=0.0,
            nms=dict(type='nms_rotated', iou_thr=0.1),
            max_per_img=10))))


def platform_boxes_from_beam_gt(beam_boxes: torch.Tensor,
                                seq_k: Dict) -> torch.Tensor:
    if beam_boxes.numel() == 0:
        return beam_boxes.new_zeros((0, 5))
    center = beam_boxes[:, :2]
    width = beam_boxes[:, 2].clamp(min=1e-6)
    height = beam_boxes[:, 3].clamp(min=1e-6)
    theta = beam_boxes[:, 4]
    width_is_long = width >= height
    long_len = torch.where(width_is_long, width, height)
    short_len = torch.where(width_is_long, height, width)
    long_theta = torch.where(width_is_long, theta, theta + math.pi / 2)
    ux = torch.stack([torch.cos(long_theta), torch.sin(long_theta)], dim=1)
    flip = ((ux[:, 0] < 0)
            | ((ux[:, 0].abs() < 1e-6) & (ux[:, 1] < 0)))
    ux = torch.where(flip[:, None], -ux, ux)
    uy = torch.stack([-ux[:, 1], ux[:, 0]], dim=1)
    platform_center = (
        center
        + ux * (float(seq_k.get('offset_long_k', 0.0)) * long_len)[:, None]
        + uy * (float(seq_k.get('offset_short_k', 0.0)) * short_len)[:, None])
    platform_w = float(seq_k['width_k']) * long_len
    platform_h = float(seq_k['height_k']) * short_len
    platform_theta = long_theta + float(seq_k.get('dtheta', 0.0))
    # le90 is pi-periodic; keep values in [-pi/2, pi/2).
    platform_theta = torch.remainder(
        platform_theta + math.pi / 2, math.pi) - math.pi / 2
    return torch.stack([
        platform_center[:, 0], platform_center[:, 1],
        platform_w, platform_h, platform_theta], dim=1)


def build_dark_view(image: torch.Tensor, args) -> torch.Tensor:
    if random.random() >= args.dark_aug_prob:
        return image
    mean = image.new_tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    std = image.new_tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
    raw = (image * std + mean).clamp(0.0, 255.0) / 255.0
    gain = random.uniform(args.dark_gain_min, args.dark_gain_max)
    gamma = random.uniform(1.0, 2.5)
    raw = raw.clamp(min=1e-6).pow(gamma) * gain
    if random.random() < 0.5:
        raw = F.avg_pool2d(raw, kernel_size=3, stride=1, padding=1)
    if args.noise_std_max > 0:
        noise_std = random.uniform(0.0, args.noise_std_max) / 255.0
        raw = raw + torch.randn_like(raw) * noise_std
    raw = raw.clamp(0.0, 1.0) * 255.0
    return (raw - mean) / std


def sum_loss_dict(losses: Dict[str, object]) -> Tuple[torch.Tensor, Dict]:
    total = None
    scalars = {}
    for name, value in losses.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        component = sum(values)
        total = component if total is None else total + component
        scalars[name] = float(component.detach().item())
    if total is None:
        raise RuntimeError('Platform head returned no losses')
    return total, scalars


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


def frozen_features(model, image):
    with torch.no_grad():
        return tuple(feature.detach() for feature in model.extract_feat(image))


def platform_detections(platform_head, features, meta, topm: int):
    platform_head.eval()
    with torch.no_grad():
        cls_scores, bbox_preds = platform_head(features)
        det_bboxes, _ = platform_head.get_bboxes(
            cls_scores, bbox_preds, [meta], rescale=False,
            with_nms=True)[0]
    if det_bboxes.numel() == 0:
        return det_bboxes.new_zeros((0, 5)), det_bboxes.new_zeros((0,))
    topm = min(max(int(topm), 1), int(det_bboxes.shape[0]))
    return det_bboxes[:topm, :5].detach(), det_bboxes[:topm, 5].detach()


def train_one_epoch(model, platform_head, optimizer, records, seq_k,
                    transform_compose, img_scale, flip, args, epoch):
    platform_head.train()
    random.Random(args.seed + epoch).shuffle(records)
    running = dict(loss=0.0, loss_cls=0.0, loss_bbox=0.0, frames=0)
    for index, record in enumerate(records, 1):
        loaded = load_frame(
            record, args, transform_compose, img_scale, flip)
        if loaded is None:
            continue
        image, meta, _, gt, _ = loaded
        image = build_dark_view(image, args)
        features = frozen_features(model, image)
        beam_gt = entry_probe.gt_to_tensor(
            pool_probe.scale_gt_to_img(gt, meta), image.device)
        platform_gt = platform_boxes_from_beam_gt(beam_gt, seq_k[record[1]])
        labels = [torch.zeros(
            (platform_gt.shape[0],), dtype=torch.long, device=image.device)]
        cls_scores, bbox_preds = platform_head(features)
        losses = platform_head.loss(
            cls_scores, bbox_preds, [platform_gt], labels, [meta])
        if losses is None:
            continue
        loss, scalars = sum_loss_dict(losses)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                platform_head.parameters(), args.grad_clip)
        optimizer.step()
        assert_isolated(model, platform_head)
        running['loss'] += float(loss.detach().item())
        running['loss_cls'] += scalars.get('loss_cls', 0.0)
        running['loss_bbox'] += scalars.get('loss_bbox', 0.0)
        running['frames'] += 1
        if index % 100 == 0:
            denom = max(running['frames'], 1)
            print(f'[train] epoch={epoch:02d} {index}/{len(records)} '
                  f'loss={running["loss"] / denom:.5f}')
    denom = max(running['frames'], 1)
    return {key: (value / denom if key != 'frames' else int(value))
            for key, value in running.items()}


def evaluate_pseudo_platform(model, platform_head, records, seq_k,
                             transform_compose, img_scale, flip, args):
    from mmcv.ops import box_iou_rotated

    ious, scores = [], []
    hits = missing = 0
    for record in records:
        loaded = load_frame(
            record, args, transform_compose, img_scale, flip)
        if loaded is None:
            continue
        image, meta, _, gt, _ = loaded
        features = frozen_features(model, image)
        boxes, det_scores = platform_detections(
            platform_head, features, meta, 1)
        beam_gt = entry_probe.gt_to_tensor(
            pool_probe.scale_gt_to_img(gt, meta), image.device)
        platform_gt = platform_boxes_from_beam_gt(beam_gt, seq_k[record[1]])
        if boxes.numel() == 0:
            iou = score = 0.0
            missing += 1
        else:
            iou = float(box_iou_rotated(
                boxes[:1].float(), platform_gt.float()).reshape(-1)[0].item())
            score = float(det_scores[0].item())
        ious.append(iou)
        scores.append(score)
        hits += int(iou >= args.riou_thr)
    return metric_summary(ious, scores, hits, missing)


def metric_summary(ious, scores, hits, missing):
    return dict(
        frames=len(ious), hits=int(hits), missing=int(missing),
        recall=hits / len(ious) if ious else 0.0,
        riou_mean=float(np.mean(ious)) if ious else 0.0,
        riou_median=float(np.median(ious)) if ious else 0.0,
        riou_min=float(np.min(ious)) if ious else 0.0,
        riou_max=float(np.max(ious)) if ious else 0.0,
        score_mean=float(np.mean(scores)) if scores else 0.0,
        score_median=float(np.median(scores)) if scores else 0.0)


def load_filtered_manual_platforms(path: str, args) -> Dict[int, Dict]:
    manual = platform_probe.load_manual_platforms(
        path, args.eval_split, args.eval_seq)
    return {int(frame): item for frame, item in manual.items()
            if args.eval_start <= int(frame) <= args.eval_end}


def fit_manual_eval_k(manual: Dict[int, Dict], args) -> Dict:
    samples = []
    diag = entry_probe.get_diag()
    for frame, item in sorted(manual.items()):
        platform_poly = platform_probe.manual_polygon(item)
        _, ann_path = diag.find_files(
            args.data_root, args.eval_split, args.eval_seq, frame)
        beam_poly = platform_probe.ann_to_poly(ann_path) if ann_path else None
        if platform_poly is not None and beam_poly is not None:
            samples.append(platform_probe.frame_platform_k(
                beam_poly, platform_poly, frame))
    if not samples:
        raise RuntimeError('No in-window manual platform polygons for eval K')
    result = dict(
        source='manual_test_oracle_calibration',
        sample_count=len(samples),
        sample_frames=[int(item['frame']) for item in samples])
    for key in ('width_k', 'height_k', 'offset_long_k',
                'offset_short_k', 'dtheta'):
        result[key] = float(np.median([float(item[key]) for item in samples]))
    result['dtheta'] = 0.0
    return result


def scale_polygon(poly: np.ndarray, meta: Dict) -> np.ndarray:
    scale = meta.get('scale_factor', 1.0)
    if isinstance(scale, torch.Tensor):
        scale = scale.detach().cpu().numpy()
    flat = np.asarray(scale, dtype=np.float64).reshape(-1)
    sx = float(flat[0]) if flat.size else 1.0
    sy = float(flat[1]) if flat.size >= 2 else sx
    result = np.asarray(poly, dtype=np.float32).copy()
    result[:, 0] *= sx
    result[:, 1] *= sy
    return result


def polygon_to_obb(poly: np.ndarray, device) -> torch.Tensor:
    from mmrotate.core import poly2obb_np

    obb = poly2obb_np(
        np.asarray(poly, dtype=np.float32).reshape(-1), version='le90')
    return torch.as_tensor(obb, dtype=torch.float32, device=device).reshape(1, 5)


def evaluate_manual_platform(model, platform_head, manual,
                             transform_compose, img_scale, flip, args):
    from mmcv.ops import box_iou_rotated

    diag = entry_probe.get_diag()
    rows, ious, scores = [], [], []
    hits = missing = 0
    for frame, item in sorted(manual.items()):
        img_path, _ = diag.find_files(
            args.data_root, args.eval_split, args.eval_seq, frame)
        platform_poly = platform_probe.manual_polygon(item)
        if img_path is None or platform_poly is None:
            continue
        image, meta, _ = diag.preprocess_image(
            img_path, transform_compose, img_scale, flip)
        if image is None:
            continue
        image = image.cuda(f'cuda:{args.gpu}')
        features = frozen_features(model, image)
        boxes, det_scores = platform_detections(
            platform_head, features, meta, 1)
        manual_obb = polygon_to_obb(scale_polygon(platform_poly, meta), image.device)
        if boxes.numel() == 0:
            iou = score = 0.0
            missing += 1
            predicted_box = None
        else:
            iou = float(box_iou_rotated(
                boxes[:1].float(), manual_obb.float()).reshape(-1)[0].item())
            score = float(det_scores[0].item())
            predicted_box = boxes[0].detach().cpu().numpy().astype(float).tolist()
        hit = iou >= args.riou_thr
        hits += int(hit)
        ious.append(iou)
        scores.append(score)
        rows.append(dict(frame=frame, platform_riou=iou,
                         platform_score=score, hit=bool(hit),
                         predicted_platform_box=predicted_box,
                         manual_platform_box=manual_obb[0].detach().cpu()
                         .numpy().astype(float).tolist()))
        print(f'[manual-platform] frame={frame} RIoU={iou:.3f} '
              f'score={score:.4f} hit={int(hit)}')
    return rows, metric_summary(ious, scores, hits, missing)


def obbs_to_polygons(boxes: np.ndarray) -> np.ndarray:
    from mmrotate.core import obb2poly_np

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    with_score = np.concatenate(
        [boxes, np.zeros((len(boxes), 1), dtype=np.float32)], axis=1)
    return obb2poly_np(with_score, version='le90')[:, :8].reshape(-1, 4, 2)


def candidate_platform_obbs(boxes: np.ndarray, eval_k: Dict,
                            device) -> torch.Tensor:
    from mmrotate.core import poly2obb_np

    result = []
    for beam_poly in obbs_to_polygons(boxes):
        platform_poly = platform_probe.platform_poly_from_seq_k(
            beam_poly, eval_k)
        result.append(poly2obb_np(
            platform_poly.astype(np.float32).reshape(-1), version='le90'))
    return torch.as_tensor(
        np.asarray(result, dtype=np.float32), device=device)


def compatibility_scores(candidate_platforms, detected_boxes,
                         detected_scores):
    from mmcv.ops import box_iou_rotated

    if detected_boxes.numel() == 0:
        zeros = candidate_platforms.new_zeros((candidate_platforms.shape[0],))
        return zeros, zeros
    ious = box_iou_rotated(
        candidate_platforms.float(), detected_boxes.float())
    best_iou, _ = ious.max(dim=1)
    confidence_weighted = (ious * detected_scores[None, :]).max(dim=1)[0]
    return best_iou, confidence_weighted


def mode_names(lambdas):
    names = ['cls', 'beam_oracle', 'platform_only',
             'platform_conf', 'cls_x_platform']
    names.extend(f'log_lambda_{value:g}' for value in lambdas)
    return names


def select_index(mode, cls_scores, platform_scores,
                 platform_conf_scores, beam_ious):
    eps = 1e-12
    if mode == 'cls':
        metric = cls_scores
    elif mode == 'beam_oracle':
        metric = beam_ious
    elif mode == 'platform_only':
        metric = platform_scores
    elif mode == 'platform_conf':
        metric = platform_conf_scores
    elif mode == 'cls_x_platform':
        metric = cls_scores * platform_scores
    elif mode.startswith('log_lambda_'):
        lam = float(mode[len('log_lambda_'):])
        metric = (np.log(np.clip(cls_scores, eps, None))
                  + lam * np.log(np.clip(platform_scores, eps, None)))
    else:
        raise ValueError(mode)
    return int(np.argmax(metric))


def evaluate_rerank(model, platform_head, eval_k,
                    transform_compose, img_scale, flip, args):
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
        image, meta, stats = diag.preprocess_image(
            img_path, transform_compose, img_scale, flip)
        if image is None:
            continue
        image = image.cuda(f'cuda:{args.gpu}')
        features = frozen_features(model, image)
        with torch.no_grad():
            candidate_head, cls_scores, bbox_preds = (
                entry_probe.forward_candidate_head(
                    model, features, args.candidate_source))
            boxes, scores, levels, _, _ = entry_probe.flatten_decode_candidates(
                candidate_head, cls_scores, bbox_preds, meta['img_shape'])
        platform_boxes, platform_det_scores = platform_detections(
            platform_head, features, meta, args.platform_topm)

        max_k = min(max(topks), int(scores.numel()))
        top_scores_t, indices = torch.topk(scores, k=max_k, sorted=True)
        top_boxes_t = boxes[indices]
        top_levels_t = levels[indices]
        gt = pool_probe.scale_gt_to_img(gts[0], meta)
        gt_box = entry_probe.gt_to_tensor(gt, boxes.device)
        top_ious_t = box_iou_rotated(
            top_boxes_t.float(), gt_box.float()).reshape(-1)
        mapped_platforms = candidate_platform_obbs(
            top_boxes_t.detach().cpu().numpy(), eval_k, boxes.device)
        platform_iou_t, platform_conf_t = compatibility_scores(
            mapped_platforms, platform_boxes, platform_det_scores)

        top_scores = top_scores_t.detach().cpu().numpy().astype(np.float64)
        top_boxes = top_boxes_t.detach().cpu().numpy().astype(np.float32)
        top_levels = top_levels_t.detach().cpu().numpy().astype(np.int64)
        top_ious = top_ious_t.detach().cpu().numpy().astype(np.float64)
        platform_iou = platform_iou_t.cpu().numpy().astype(np.float64)
        platform_conf = platform_conf_t.cpu().numpy().astype(np.float64)
        per_k = {}
        for topk in topks:
            actual_k = min(topk, max_k)
            per_mode = {}
            for mode in modes:
                selected = select_index(
                    mode, top_scores[:actual_k], platform_iou[:actual_k],
                    platform_conf[:actual_k], top_ious[:actual_k])
                riou = float(top_ious[selected])
                per_mode[mode] = dict(
                    selected_rank=selected + 1,
                    selected_beam_riou=riou,
                    selected_cls_score=float(top_scores[selected]),
                    selected_platform_iou=float(platform_iou[selected]),
                    selected_platform_conf=float(platform_conf[selected]),
                    selected_level=int(top_levels[selected]),
                    selected_beam_box=top_boxes[selected].astype(float).tolist(),
                    hit=bool(riou >= args.riou_thr))
            per_k[str(topk)] = dict(actual_k=actual_k, modes=per_mode)
        top_platform_score = (
            float(platform_det_scores[0].item())
            if platform_det_scores.numel() else 0.0)
        row = dict(
            frame=frame,
            fname=os.path.splitext(os.path.basename(img_path))[0],
            brightness=float(stats['raw_brightness']),
            global_max=float(top_scores[0]),
            platform_detection_count=int(platform_boxes.shape[0]),
            top_platform_score=top_platform_score,
            per_k=per_k)
        rows.append(row)
        focus = per_k[str(500 if 500 in topks else topks[-1])]['modes']
        print(f'[{row["fname"]}] plat_score={top_platform_score:.4f} '
              f'platform={focus["platform_only"]["selected_beam_riou"]:.3f} '
              f'ceiling={focus["beam_oracle"]["selected_beam_riou"]:.3f}')
    assert_isolated(model, platform_head)
    return rows, summarize_rerank(rows, topks, modes)


def summarize_rerank(rows, topks, modes):
    summary = dict(frames=len(rows), per_k={})
    for topk in topks:
        per_mode = {}
        for mode in modes:
            key = f'hit_{topk}_{mode}'
            proxies, rious = [], []
            for row in rows:
                item = row['per_k'][str(topk)]['modes'][mode]
                proxies.append(dict(frame=row['frame'], **{key: item['hit']}))
                rious.append(float(item['selected_beam_riou']))
            hits = sum(bool(item[key]) for item in proxies)
            per_mode[mode] = dict(
                hits=hits, recall=hits / len(rows) if rows else 0.0,
                mcml=pool_probe.longest_consecutive_miss(
                    proxies, key) if rows else 0,
                mean_riou=float(np.mean(rious)) if rious else 0.0,
                min_riou=float(np.min(rious)) if rious else 0.0,
                max_riou=float(np.max(rious)) if rious else 0.0)
        summary['per_k'][str(topk)] = per_mode
    return summary


def print_rerank_summary(summary, topks, modes):
    print('\n' + '=' * 106)
    print('PROBE 4 SUMMARY: DETECTED PLATFORM OBB RERANK')
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


def cpu_state_dict(module):
    return {key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()}


def ensure_parent(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def main():
    args = parse_args()
    load_project_helpers()
    if args.eval_end < args.eval_start:
        raise ValueError('--eval-end must be >= --eval-start')
    set_seed(args.seed)

    model, cfg = entry_probe.load_model(
        args.config, args.checkpoint, args.gpu)
    freeze_detector(model)
    platform_head = build_platform_head().cuda(f'cuda:{args.gpu}')
    assert_isolated(model, platform_head)
    diag = entry_probe.get_diag()
    transform_compose, img_scale, flip = diag.build_test_transforms(cfg)
    train_seq_k = load_train_seq_k(args.train_k_config)

    history = []
    val_history = []
    if args.eval_only:
        if not args.platform_head_in:
            raise ValueError('--eval-only requires --platform-head-in')
        saved = torch.load(args.platform_head_in, map_location='cpu')
        platform_head.load_state_dict(saved['state_dict'], strict=True)
        history = saved.get('history', [])
        val_history = saved.get('val_history', [])
        print(f'[load] platform head: {args.platform_head_in}')
    else:
        if args.holdout_seq not in train_seq_k:
            raise ValueError(
                f'holdout sequence {args.holdout_seq} has no train K')
        all_records = enumerate_records(
            args.data_root, args.train_splits, sorted(train_seq_k))
        train_records = sample_records(
            [item for item in all_records if item[1] != args.holdout_seq],
            args.max_train_frames, args.seed)
        val_records = sample_records(
            [item for item in all_records if item[1] == args.holdout_seq],
            args.max_val_frames, args.seed + 1)
        print(f'[data] train={len(train_records)} val={len(val_records)} '
              f'holdout={args.holdout_seq}')
        optimizer = torch.optim.AdamW(
            platform_head.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)
        best_score = -1.0
        best_state = None
        for epoch in range(1, args.epochs + 1):
            train_row = train_one_epoch(
                model, platform_head, optimizer, train_records, train_seq_k,
                transform_compose, img_scale, flip, args, epoch)
            val_row = evaluate_pseudo_platform(
                model, platform_head, val_records, train_seq_k,
                transform_compose, img_scale, flip, args)
            train_row['epoch'] = epoch
            val_row['epoch'] = epoch
            history.append(train_row)
            val_history.append(val_row)
            selection_score = val_row['recall'] * 10.0 + val_row['riou_mean']
            print(f'[epoch] {epoch:02d} train_loss={train_row["loss"]:.5f} '
                  f'val_recall={val_row["recall"]:.3f} '
                  f'val_RIoU={val_row["riou_mean"]:.3f}')
            if selection_score > best_score:
                best_score = selection_score
                best_state = cpu_state_dict(platform_head)
        if best_state is None:
            raise RuntimeError('No platform-head checkpoint was produced')
        platform_head.load_state_dict(best_state, strict=True)
        ensure_parent(args.platform_head_out)
        torch.save(dict(
            state_dict=best_state,
            detector_config=args.config,
            detector_checkpoint=args.checkpoint,
            train_seq_k=train_seq_k,
            holdout_seq=args.holdout_seq,
            history=history,
            val_history=val_history,
            isolation=dict(
                detector_frozen=True, detached_fpn=True,
                optimizer_scope='PlatformOBBATSSHead only')),
            args.platform_head_out)
        print(f'[out] platform head: {args.platform_head_out}')

    manual = load_filtered_manual_platforms(
        args.manual_platform_json, args)
    manual_rows, manual_summary = evaluate_manual_platform(
        model, platform_head, manual,
        transform_compose, img_scale, flip, args)
    manual_gate_pass = bool(
        manual_summary['frames'] >= 6
        and manual_summary['hits'] >= 5
        and manual_summary['riou_median'] >= args.riou_thr)
    print('[manual-platform-summary] '
          + json.dumps(manual_summary, ensure_ascii=False))
    print('[manual-platform-gate] '
          + ('PASS' if manual_gate_pass else 'FAIL')
          + ' (requires >=5/6 hits and median RIoU>=0.5)')

    if args.eval_k_source == 'manual':
        print('[warning] candidate mapping K uses in-window TEST manual '
              'polygons: oracle-calibration upper bound.')
        eval_k = fit_manual_eval_k(manual, args)
    else:
        eval_k = median_k(train_seq_k)
    print('[eval-k] ' + json.dumps(eval_k, ensure_ascii=False))
    rows, rerank_summary = evaluate_rerank(
        model, platform_head, eval_k,
        transform_compose, img_scale, flip, args)
    topks = pool_probe.normalize_topks(args.eval_topks)
    modes = mode_names(args.log_lambdas)
    print_rerank_summary(rerank_summary, topks, modes)

    result = dict(
        probe='detached_platform_obb_detection_and_rerank',
        args=vars(args),
        isolation=dict(
            detector_frozen=True, detector_forward_no_grad=True,
            detached_fpn=True,
            optimizer_scope='PlatformOBBATSSHead only',
            test_time_platform_head=True,
            output_contract='unchanged main-head beam OBB only',
            no_inverse_k=True),
        train_seq_k=train_seq_k, eval_k=eval_k,
        history=history, val_history=val_history,
        manual_platform=dict(
            rows=manual_rows, summary=manual_summary,
            gate_pass=manual_gate_pass,
            gate_rule='at least 5/6 hits and median RIoU >= 0.5'),
        rerank=dict(rows=rows, summary=rerank_summary))
    ensure_parent(args.out_json)
    with open(args.out_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f'[out] result: {args.out_json}')


if __name__ == '__main__':
    main()
