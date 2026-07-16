#!/usr/bin/env python3
"""
mcml_diag.py — MCML 漏检帧诊断工具 (v3)

v3 新增:
  --split train        跑训练集 real 数据
  --split train_sim    跑训练集 sim 数据 (自动找 train_sim/annfiles/)
  --split all          同时跑 test + train real + train sim, 一键对比
  --sample N           每个数据源随机抽 N 帧 (默认 10)

preprocessing 策略: 直接调用 config 中 test_pipeline 内部各 transform,
跳过 MultiScaleFlipAug wrapper (避免 DataContainer 嵌套问题),
保证与训练/推理 preprocessing 100% 一致.

Run:
    cd ~/workspace/symEOOD

    # 一键全对比
    PYTHONPATH=. python3 crane_project/tools/mcml_diag.py \\
        --config crane_project/configs/crane_eood_k1.py \\
        --checkpoint work_dirs/crane_eood_k1/epoch_24.pth \\
        --split all --gpu 0 --sample 10

    # 单数据源
    PYTHONPATH=. python3 crane_project/tools/mcml_diag.py \\
        --config crane_project/configs/crane_eood_k1.py \\
        --checkpoint work_dirs/crane_eood_k1/epoch_24.pth \\
        --split train --seq real_seq01 --gpu 0
"""
import argparse
import os
import glob
import re
import numpy as np
import torch
import random
import cv2
from mmrotate.core import poly2obb_np


def parse_args():
    parser = argparse.ArgumentParser(description='MCML miss-frame diagnosis v3')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seq', default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--split', default='test',
                        choices=['test', 'train', 'train_sim', 'all'])
    parser.add_argument('--sample', type=int, default=10,
                        help='Frames per source when --split all, or no --start/--end')
    parser.add_argument('--threshold', type=float, default=0.02)
    parser.add_argument('--topk', type=int, default=5,
                        help='Number of decoded boxes to compare with GT')
    parser.add_argument('--giant-size-thr', type=float, default=4096.0,
                        help='Absolute w/h threshold for giant decoded boxes')
    parser.add_argument('--giant-ratio-thr', type=float, default=20.0,
                        help='Relative w/h ratio threshold for giant decoded boxes')
    parser.add_argument('--topk-all', action='store_true',
                        help='对所有帧输出 top-k decoded box, 不仅是 DEAD-local/EDGE')
    parser.add_argument('--vis-dir', default=None,
                        help='可视化输出目录, 绘制 GT + top-k decoded box')
    parser.add_argument(
        '--preproc', default='none',
        choices=[
            'none', 'linear-brighten', 'clahe', 'gray-world',
            'linear-clahe', 'linear-clahe-gray',
        ],
        help='Test-time image normalization probe before the config test pipeline')
    parser.add_argument('--linear-gain', type=float, default=2.0,
                        help='Linear-light gain for --preproc linear-*')
    parser.add_argument('--clahe-clip-limit', type=float, default=2.0)
    parser.add_argument('--clahe-tile-grid', type=int, default=8)
    parser.add_argument('--save-preproc-dir', default=None,
                        help='Optional directory to save preprocessed images for QA')
    parser.add_argument('--adabn', action='store_true',
                        help='Adapt BatchNorm running stats on the target frames before diagnosis')
    parser.add_argument('--adabn-frames', type=int, default=32,
                        help='Max frames used for AdaBN per source')
    parser.add_argument('--adabn-momentum', type=float, default=0.1)
    return parser.parse_args()


# =====================================================================
# 数据发现
# =====================================================================

def find_files(data_root, split, seq, frame_id):
    """返回 (img_path, ann_path) 或 (None, None)。"""
    img_dir = os.path.join(data_root, split, 'images')
    if not os.path.isdir(img_dir) and split == 'train_sim':
        img_dir = os.path.join(data_root, 'train', 'images')

    ann_dir = os.path.join(data_root, split, 'annfiles')
    if not os.path.isdir(ann_dir) and split == 'train_sim':
        ann_dir = os.path.join(data_root, 'train_sim', 'annfiles')

    base = seq if 'seq' in seq else f'real_{seq}'
    fname = f'{base}_{frame_id:05d}'

    img_path = None
    for ext in ['.jpg', '.png', '.bmp', '.tif']:
        p = os.path.join(img_dir, fname + ext)
        if os.path.exists(p):
            img_path = p
            break
    if img_path is None:
        pattern = os.path.join(img_dir, f'*{seq}*{frame_id:05d}*')
        files = glob.glob(pattern)
        img_path = files[0] if files else None

    ann_path = os.path.join(ann_dir, fname + '.txt')
    if not os.path.exists(ann_path):
        pattern = os.path.join(ann_dir, f'*{seq}*{frame_id:05d}*.txt')
        files = glob.glob(pattern)
        ann_path = files[0] if files else None

    return img_path, ann_path


def discover_frames(data_root, split, seq, max_count=None):
    ann_dir = os.path.join(data_root, split, 'annfiles')
    if not os.path.isdir(ann_dir) and split == 'train_sim':
        ann_dir = os.path.join(data_root, 'train_sim', 'annfiles')
    if not os.path.isdir(ann_dir):
        return []

    pattern = os.path.join(ann_dir, f'*{seq}*.txt')
    files = glob.glob(pattern)
    frame_ids = []
    for f in files:
        m = re.search(r'_(\d{5})\.txt$', f)
        if m:
            frame_ids.append(int(m.group(1)))
    frame_ids.sort()

    if max_count and len(frame_ids) > max_count:
        frame_ids = sorted(random.sample(frame_ids, max_count))
    return frame_ids


def discover_sequences(data_root, split):
    img_dir = os.path.join(data_root, split, 'images')
    if not os.path.isdir(img_dir) and split == 'train_sim':
        img_dir = os.path.join(data_root, 'train', 'images')
    if not os.path.isdir(img_dir):
        return []

    files = os.listdir(img_dir)
    seqs = set()
    for f in files:
        m = re.match(r'(.+_seq\d+)_\d{5}\.\w+', f)
        if m:
            seqs.add(m.group(1))
    return sorted(seqs)


def parse_dota_ann(ann_path):
    """解析 DOTA 标注, 使用训练数据集同口径的 le90 poly2obb。"""
    if ann_path is None or not os.path.exists(ann_path):
        return []
    gts = []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            coords = np.array([float(x) for x in parts[:8]], dtype=np.float32)
            cls = parts[8]
            obb = poly2obb_np(coords, version='le90')
            if obb is None:
                continue
            cx, cy, w, h, angle_rad = obb
            gts.append(dict(
                cx=float(cx), cy=float(cy), w=float(w), h=float(h),
                angle=float(np.degrees(angle_rad)), cls=cls))
    return gts


# =====================================================================
# Preprocessing: 从 config 解析 test_pipeline 并逐个调用
# =====================================================================

def srgb_to_linear(img):
    img = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.where(img <= 0.04045, img / 12.92,
                    ((img + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(img):
    img = np.clip(img, 0.0, 1.0)
    srgb = np.where(img <= 0.0031308, img * 12.92,
                    1.055 * np.power(img, 1.0 / 2.4) - 0.055)
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


def linear_brighten_bgr(img, gain):
    linear = srgb_to_linear(img)
    return linear_to_srgb(linear * gain)


def clahe_bgr(img, clip_limit=2.0, tile_grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_chan = clahe.apply(l_chan)
    lab = cv2.merge([l_chan, a_chan, b_chan])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gray_world_bgr(img):
    work = img.astype(np.float32)
    means = work.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    scale = gray / np.maximum(means, 1e-6)
    return np.clip(work * scale.reshape(1, 1, 3), 0, 255).astype(np.uint8)


def image_stats_bgr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h = gray.shape[0]
    top = float(gray[:h // 2].mean()) if h >= 2 else float(gray.mean())
    bottom = float(gray[h // 2:].mean()) if h >= 2 else float(gray.mean())
    return dict(
        brightness=float(img.mean()),
        contrast=float(gray.std()),
        ud_delta=top - bottom,
    )


def apply_test_time_preproc(raw, preproc_cfg):
    mode = preproc_cfg.get('mode', 'none')
    img = raw.copy()

    if mode == 'none':
        return img
    if mode == 'linear-brighten':
        return linear_brighten_bgr(img, preproc_cfg.get('linear_gain', 2.0))
    if mode == 'clahe':
        return clahe_bgr(
            img, preproc_cfg.get('clahe_clip_limit', 2.0),
            preproc_cfg.get('clahe_tile_grid', 8))
    if mode == 'gray-world':
        return gray_world_bgr(img)
    if mode == 'linear-clahe':
        img = linear_brighten_bgr(img, preproc_cfg.get('linear_gain', 2.0))
        return clahe_bgr(
            img, preproc_cfg.get('clahe_clip_limit', 2.0),
            preproc_cfg.get('clahe_tile_grid', 8))
    if mode == 'linear-clahe-gray':
        img = gray_world_bgr(img)
        img = linear_brighten_bgr(img, preproc_cfg.get('linear_gain', 2.0))
        return clahe_bgr(
            img, preproc_cfg.get('clahe_clip_limit', 2.0),
            preproc_cfg.get('clahe_tile_grid', 8))

    raise ValueError(f'Unsupported preproc mode: {mode}')

def build_test_transforms(cfg):
    """从 cfg.test_pipeline 解析内部 transform 实例列表。
    跳过 LoadImageFromFile 和 MultiScaleFlipAug wrapper,
    返回内部的 transform 序列 + MultiScaleFlipAug 参数。
    """
    from mmdet.datasets.pipelines import Compose

    pipeline_cfg = cfg.test_pipeline
    inner_transforms = []
    img_scale = (1024, 1024)
    flip = False

    for t in pipeline_cfg:
        ttype = t.get('type', '')
        if ttype == 'LoadImageFromFile':
            continue  # 我们自己加载
        if ttype == 'MultiScaleFlipAug':
            img_scale = t.get('img_scale', (1024, 1024))
            flip = t.get('flip', False)
            inner_transforms = t.get('transforms', [])
            break

    if not inner_transforms:
        raise RuntimeError(
            'test_pipeline must contain MultiScaleFlipAug with inner transforms')

    if isinstance(img_scale, list):
        if len(img_scale) == 2 and all(
                isinstance(item, (int, float)) for item in img_scale):
            img_scale = tuple(img_scale)
        elif len(img_scale) == 1:
            img_scale = tuple(img_scale[0])
        else:
            raise RuntimeError(
                'Probe preprocessing supports exactly one test img_scale, got '
                f'{img_scale!r}')
    if not isinstance(img_scale, tuple) or len(img_scale) != 2:
        raise RuntimeError(f'Invalid test img_scale: {img_scale!r}')
    if flip:
        raise RuntimeError(
            'Probe preprocessing supports deterministic flip=False only; '
            'MultiScaleFlipAug flip=True produces multiple test views')

    compose = Compose(inner_transforms)
    return compose, img_scale, flip


def _unwrap_pipeline_value(value):
    """Unwrap an MMCV DataContainer and a possible singleton wrapper."""
    if hasattr(value, 'data'):
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
        if hasattr(value, 'data'):
            value = value.data
    return value


def _shape_tuple(value, name):
    if isinstance(value, torch.Size):
        value = tuple(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise RuntimeError(f'Invalid {name} from test pipeline: {value!r}')
    return tuple(int(item) for item in value)


def _validate_preprocess_metadata(img_tensor, img_metas, img_scale):
    """Fail fast when the manual probe path diverges from test-time geometry."""
    ori_shape = _shape_tuple(img_metas.get('ori_shape'), 'ori_shape')
    img_shape = _shape_tuple(img_metas.get('img_shape'), 'img_shape')
    pad_shape = _shape_tuple(img_metas.get('pad_shape'), 'pad_shape')

    tensor_h, tensor_w = map(int, img_tensor.shape[-2:])
    if (tensor_h, tensor_w) != tuple(pad_shape[:2]):
        raise RuntimeError(
            'Test pipeline metadata/tensor mismatch: '
            f'tensor={(tensor_h, tensor_w)}, pad_shape={pad_shape[:2]}')
    if img_shape[0] > pad_shape[0] or img_shape[1] > pad_shape[1]:
        raise RuntimeError(
            f'img_shape={img_shape[:2]} exceeds pad_shape={pad_shape[:2]}')

    scale_factor = img_metas.get('scale_factor')
    if isinstance(scale_factor, torch.Tensor):
        scale_factor = scale_factor.detach().cpu().numpy()
    scale = np.asarray(scale_factor, dtype=np.float64).reshape(-1)
    if scale.size == 0 or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise RuntimeError(
            f'Invalid scale_factor from test pipeline: {scale_factor!r}')
    sx = float(scale[0])
    sy = float(scale[1]) if scale.size >= 2 else sx
    expected_sx = float(img_shape[1]) / float(ori_shape[1])
    expected_sy = float(img_shape[0]) / float(ori_shape[0])
    tol_x = max(1e-3, 2.0 / float(ori_shape[1]))
    tol_y = max(1e-3, 2.0 / float(ori_shape[0]))
    if abs(sx - expected_sx) > tol_x or abs(sy - expected_sy) > tol_y:
        raise RuntimeError(
            'Test pipeline scale metadata is inconsistent: '
            f'ori={ori_shape[:2]}, img={img_shape[:2]}, '
            f'scale_factor={scale.tolist()}')

    target_w, target_h = map(int, img_scale)
    resize_ratio = min(
        float(target_w) / float(ori_shape[1]),
        float(target_h) / float(ori_shape[0]))
    requested_h = int(float(ori_shape[0]) * resize_ratio + 0.5)
    requested_w = int(float(ori_shape[1]) * resize_ratio + 0.5)
    if (abs(img_shape[0] - requested_h) > 1
            or abs(img_shape[1] - requested_w) > 1):
        raise RuntimeError(
            'Test pipeline ignored MultiScaleFlipAug img_scale: '
            f'ori={ori_shape[:2]}, requested_scale={tuple(img_scale)}, '
            f'expected_img_shape={(requested_h, requested_w)}, '
            f'actual_img_shape={img_shape[:2]}. Ensure RResize receives '
            'results["scale"], not a fabricated scale_factor=1.0.')


def scale_obb_to_img(gt, img_metas):
    """Map one original-image OBB to the resized test-image coordinates."""
    scale_factor = img_metas.get('scale_factor')
    if isinstance(scale_factor, torch.Tensor):
        scale_factor = scale_factor.detach().cpu().numpy()
    scale = np.asarray(scale_factor, dtype=np.float64).reshape(-1)
    if scale.size == 0:
        raise RuntimeError('Missing scale_factor for GT coordinate mapping')
    sx = float(scale[0])
    sy = float(scale[1]) if scale.size >= 2 else sx
    size_scale = float(np.sqrt(sx * sy))
    mapped = dict(gt)
    mapped['cx'] = float(gt['cx']) * sx
    mapped['cy'] = float(gt['cy']) * sy
    mapped['w'] = float(gt['w']) * size_scale
    mapped['h'] = float(gt['h']) * size_scale
    return mapped


def detections_to_ori(detections, img_metas):
    """Map decoded probe boxes back for drawing on the original image."""
    scale_factor = img_metas.get('scale_factor')
    if isinstance(scale_factor, torch.Tensor):
        scale_factor = scale_factor.detach().cpu().numpy()
    scale = np.asarray(scale_factor, dtype=np.float64).reshape(-1)
    sx = float(scale[0])
    sy = float(scale[1]) if scale.size >= 2 else sx
    size_scale = float(np.sqrt(sx * sy))
    mapped = []
    for detection in detections:
        item = dict(detection)
        item['det_cx'] = float(item['det_cx']) / sx
        item['det_cy'] = float(item['det_cy']) / sy
        item['det_w'] = float(item['det_w']) / size_scale
        item['det_h'] = float(item['det_h']) / size_scale
        mapped.append(item)
    return mapped


def preprocess_image(img_path, transform_compose, img_scale, flip,
                     preproc_cfg=None, save_preproc_dir=None):
    """用 mmrotate test pipeline 的内部 transforms 处理图片。
    手动加载图片 + 填充必要的 meta 字段, 然后走 pipeline.
    """
    raw = cv2.imread(img_path)
    if raw is None:
        return None, None, None
    preproc_cfg = preproc_cfg or dict(mode='none')
    proc = apply_test_time_preproc(raw, preproc_cfg)
    raw_stats = image_stats_bgr(raw)
    proc_stats = image_stats_bgr(proc)
    if save_preproc_dir and preproc_cfg.get('mode', 'none') != 'none':
        os.makedirs(save_preproc_dir, exist_ok=True)
        out_path = os.path.join(save_preproc_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, proc)

    # 构造 results dict, 模拟 LoadImageFromFile 的输出
    # 与正式 LoadImageFromFile 保持一致：OpenCV 读入结果维持 BGR。
    # 后续 Normalize(to_rgb=True) 会且只会执行一次 BGR -> RGB。
    # 若在这里提前转 RGB，Normalize 会再次交换通道，导致所有复用本函数
    # 的 probe 输入与正式 test pipeline 不一致。
    img = proc.copy()
    results = dict(
        img=img,
        filename=img_path,
        ori_filename=os.path.basename(img_path),
        img_shape=img.shape,
        ori_shape=img.shape,
        # MultiScaleFlipAug normally injects scale before its inner transforms.
        # RResize must see this key; scale_factor=1.0 selects a no-resize path.
        scale=tuple(img_scale),
        flip=flip,
        flip_direction='horizontal' if flip else None,
        img_fields=['img'],
    )

    # 逐个调用内部 transforms
    results = transform_compose(results)

    # 提取 tensor (DefaultFormatBundle 已包成 DataContainer)
    img_tensor = _unwrap_pipeline_value(results['img'])
    if not isinstance(img_tensor, torch.Tensor):
        img_tensor = torch.from_numpy(img_tensor)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # add batch dim

    # Collect moves authoritative metadata into img_metas.  The outer results
    # no longer contains ori_shape/img_shape/scale_factor after Collect.
    img_metas = _unwrap_pipeline_value(results.get('img_metas'))
    if not isinstance(img_metas, dict):
        raise RuntimeError(
            'Test pipeline must Collect img_metas; got '
            f'{type(img_metas).__name__}')
    img_metas = dict(img_metas)
    img_metas.setdefault('filename', img_path)
    img_metas.setdefault('ori_filename', os.path.basename(img_path))
    _validate_preprocess_metadata(img_tensor, img_metas, img_scale)

    stats = dict(
        raw_brightness=raw_stats['brightness'],
        raw_contrast=raw_stats['contrast'],
        raw_ud_delta=raw_stats['ud_delta'],
        proc_brightness=proc_stats['brightness'],
        proc_contrast=proc_stats['contrast'],
        proc_ud_delta=proc_stats['ud_delta'],
        preprocess=dict(
            ori_shape=list(_shape_tuple(img_metas['ori_shape'], 'ori_shape')),
            img_shape=list(_shape_tuple(img_metas['img_shape'], 'img_shape')),
            pad_shape=list(_shape_tuple(img_metas['pad_shape'], 'pad_shape')),
            scale_factor=np.asarray(
                img_metas['scale_factor'], dtype=np.float64).reshape(-1).tolist(),
            tensor_shape=list(img_tensor.shape),
            requested_scale=list(img_scale),
            flip=bool(img_metas.get('flip', False)),
        ),
    )
    return img_tensor, img_metas, stats


# =====================================================================
# Forward hook
# =====================================================================

class HeadHook:
    def __init__(self):
        self.cls_scores = None
        self.bbox_preds = None
        self._active = True

    def hook_fn(self, module, input, output):
        if not self._active:
            return
        cls_list, bbox_list = output
        self.cls_scores = [s[0].detach().cpu() for s in cls_list]
        self.bbox_preds = [b[0].detach().cpu() for b in bbox_list]


# =====================================================================
# AdaBN probe
# =====================================================================

def batchnorm_modules(model):
    from torch.nn.modules.batchnorm import _BatchNorm
    return [m for m in model.modules() if isinstance(m, _BatchNorm)]


def snapshot_bn_state(model):
    state = []
    for m in batchnorm_modules(model):
        state.append(dict(
            module=m,
            running_mean=None if m.running_mean is None else m.running_mean.detach().clone(),
            running_var=None if m.running_var is None else m.running_var.detach().clone(),
            num_batches_tracked=(
                None if m.num_batches_tracked is None
                else m.num_batches_tracked.detach().clone()),
            momentum=m.momentum,
        ))
    return state


def restore_bn_state(state):
    for item in state:
        m = item['module']
        if item['running_mean'] is not None:
            m.running_mean.data.copy_(item['running_mean'])
        if item['running_var'] is not None:
            m.running_var.data.copy_(item['running_var'])
        if item['num_batches_tracked'] is not None:
            m.num_batches_tracked.data.copy_(item['num_batches_tracked'])
        m.momentum = item['momentum']


def adapt_bn_on_frames(model, hook_mgr, transform_compose, img_scale, flip,
                       data_root, split, seq, frame_ids, gpu, preproc_cfg,
                       save_preproc_dir=None, max_frames=32, momentum=0.1):
    bn_layers = batchnorm_modules(model)
    if not bn_layers:
        print('  [AdaBN] no BatchNorm layers found, skip')
        return 0

    adapt_ids = frame_ids[:max_frames] if max_frames > 0 else frame_ids
    if not adapt_ids:
        print('  [AdaBN] no frames for adaptation, skip')
        return 0

    model.eval()
    for m in bn_layers:
        m.train()
        m.momentum = momentum

    hook_mgr._active = False
    used = 0
    with torch.no_grad():
        for fid in adapt_ids:
            img_path, _ = find_files(data_root, split, seq, fid)
            if img_path is None:
                continue
            img_tensor, _, _ = preprocess_image(
                img_path, transform_compose, img_scale, flip,
                preproc_cfg=preproc_cfg, save_preproc_dir=save_preproc_dir)
            if img_tensor is None:
                continue
            img_tensor = img_tensor.cuda(f'cuda:{gpu}')
            feat = model.backbone(img_tensor)
            feat = model.neck(feat)
            model.bbox_head(feat)
            used += 1

    hook_mgr._active = True
    model.eval()
    print(f'  [AdaBN] adapted BN on {used}/{len(adapt_ids)} frames '
          f'(momentum={momentum})')
    return used


# =====================================================================
# GT 位置 score 分析
# =====================================================================

def gt_center_score(cls_scores, gt_cx, gt_cy, img_shape,
                    feat_strides, num_anchors_per_level):
    H_img, W_img = img_shape[:2]
    results = {
        'gt_max_score': 0.0,
        'gt_max_level': -1,
        'gt_scores_per_level': [],
        'global_max_score': 0.0,
        'global_max_level': -1,
    }

    for lvl, (cls_feat, stride) in enumerate(zip(cls_scores, feat_strides)):
        C, H_feat, W_feat = cls_feat.shape

        feat_cx = gt_cx / stride
        feat_cy = gt_cy / stride
        feat_col = max(0, min(int(round(feat_cx)), W_feat - 1))
        feat_row = max(0, min(int(round(feat_cy)), H_feat - 1))

        gt_anchor_scores = cls_feat[:, feat_row, feat_col].sigmoid()
        gt_max = gt_anchor_scores.max().item()

        roi_scores = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r = max(0, min(feat_row + dr, H_feat - 1))
                c = max(0, min(feat_col + dc, W_feat - 1))
                roi_scores.append(cls_feat[:, r, c].sigmoid())
        roi_max = torch.cat(roi_scores).max().item()

        global_max = cls_feat.sigmoid().max().item()

        results['gt_scores_per_level'].append({
            'level': lvl, 'stride': stride,
            'gt_max_score': gt_max,
            'roi_3x3_max_score': roi_max,
            'global_max_score': global_max,
        })

        if gt_max > results['gt_max_score']:
            results['gt_max_score'] = gt_max
            results['gt_max_level'] = lvl
        if global_max > results['global_max_score']:
            results['global_max_score'] = global_max
            results['global_max_level'] = lvl

    return results


# =====================================================================
# Top-k decoded box 分析
# =====================================================================

def generate_anchors(anchor_generator, cls_scores, device='cpu'):
    """用模型自己的 anchor_generator 生成 anchors, 顺序与推理完全一致。"""
    featmap_sizes = [score.shape[-2:] for score in cls_scores]
    return anchor_generator.grid_priors(featmap_sizes, device=device)


def get_inference_head(model):
    """Return the head that supplies anchors/coder for inference outputs."""
    head = model.bbox_head
    predictors = getattr(head, 'predictors', None)
    if predictors is not None and len(predictors) > 0:
        return predictors[0]
    return head


def repeat_scores_for_anchors(scores, bbox_flat, anchors, lvl):
    """Align per-location scores with per-anchor bbox/anchor tensors."""
    if anchors.shape[0] == scores.shape[0] == bbox_flat.shape[0]:
        return scores

    if anchors.shape[0] != bbox_flat.shape[0]:
        raise RuntimeError(
            f'Anchor/bbox mismatch at level {lvl}: '
            f'anchors={anchors.shape}, bbox={bbox_flat.shape}')

    if anchors.shape[0] % scores.shape[0] != 0:
        raise RuntimeError(
            f'Anchor/order mismatch at level {lvl}: '
            f'anchors={anchors.shape}, scores={scores.shape}, '
            f'bbox={bbox_flat.shape}')

    repeat_factor = anchors.shape[0] // scores.shape[0]
    return scores[:, None].expand(-1, repeat_factor).reshape(-1)


def topk_decoded_boxes(cls_scores, bbox_preds, anchors_per_level, bbox_coder,
                       img_shape=None, topk=10, score_thr=0.01):
    """Decode 所有 anchor 的 bbox_pred, 返回 top-k 高分 decoded box。

    展平顺序严格复制 SymEOODHead._get_bboxes_single:
    (A, H, W) -> permute(1, 2, 0) -> reshape(-1, C/5)。

    Returns:
        list of dict(score, cx, cy, w, h, angle, level)
    """
    all_boxes = []
    all_scores = []
    all_levels = []

    for lvl, (cls_feat, bbox_feat, anchors) in enumerate(
            zip(cls_scores, bbox_preds, anchors_per_level)):
        cls_flat = cls_feat.permute(1, 2, 0).reshape(-1, 1)
        scores = cls_flat.sigmoid().reshape(-1)
        bbox_flat = bbox_feat.permute(1, 2, 0).reshape(-1, 5)
        scores = repeat_scores_for_anchors(scores, bbox_flat, anchors, lvl)

        if anchors.shape[0] != scores.shape[0] or anchors.shape[0] != bbox_flat.shape[0]:
            raise RuntimeError(
                f'Anchor/order mismatch at level {lvl}: '
                f'anchors={anchors.shape}, scores={scores.shape}, '
                f'bbox={bbox_flat.shape}')

        # Decode
        decoded = bbox_coder.decode(anchors, bbox_flat, max_shape=img_shape)

        all_boxes.append(decoded)
        all_scores.append(scores)
        all_levels.append(torch.full((scores.shape[0],), lvl, dtype=torch.long))

    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_levels = torch.cat(all_levels, dim=0)

    # 过滤
    mask = all_scores > score_thr
    all_boxes = all_boxes[mask]
    all_scores = all_scores[mask]
    all_levels = all_levels[mask]

    # Top-k
    if all_scores.shape[0] == 0:
        return []
    k = min(topk, all_scores.shape[0])
    topk_scores, topk_idx = all_scores.topk(k)

    results = []
    for i in range(k):
        idx = topk_idx[i]
        box = all_boxes[idx].cpu().numpy()
        results.append(dict(
            score=topk_scores[i].item(),
            cx=box[0], cy=box[1], w=box[2], h=box[3], angle=box[4],
            level=all_levels[idx].item(),
        ))
    return results


def rbbox_iou(box_a, box_b):
    """计算两个 rotated bbox 的 IoU。
    box: [cx, cy, w, h, angle(rad)] — angle 已经是弧度。
    """
    def box_to_corners(cx, cy, w, h, angle_rad):
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        dx = w / 2.0
        dy = h / 2.0
        corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = corners @ rot.T
        corners[:, 0] += cx
        corners[:, 1] += cy
        return corners.astype(np.float32)

    # cv2.rotatedRectangleIntersection 需要角度制
    angle_a_deg = np.degrees(box_a[4])
    angle_b_deg = np.degrees(box_b[4])
    rect_a = ((box_a[0], box_a[1]), (box_a[2], box_a[3]), angle_a_deg)
    rect_b = ((box_b[0], box_b[1]), (box_b[2], box_b[3]), angle_b_deg)

    ret, inter_pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if ret == cv2.INTERSECT_NONE:
        return 0.0
    if ret == cv2.INTERSECT_FULL:
        area_a = box_a[2] * box_a[3]
        area_b = box_b[2] * box_b[3]
        inter_area = min(area_a, area_b)
    else:
        # 计算交集多边形面积
        inter_pts = cv2.convexHull(inter_pts)
        inter_area = cv2.contourArea(inter_pts)

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter_area
    return inter_area / max(union, 1e-6)


def analyze_topk_vs_gt(cls_scores, bbox_preds, anchors_per_level, bbox_coder,
                       gt, img_shape, topk=5):
    """对比 top-k decoded box 与 GT, 输出诊断信息。

    Args:
        cls_scores, bbox_preds: hook 拦截的 head 原始输出
        anchors_per_level: 各层 anchor tensor
        bbox_coder: DeltaXYWHAOBBoxCoder 实例
        gt: dict(cx, cy, w, h, angle, cls)
        img_shape: (H, W, C)

    Returns:
        list of dict: 每个 top-k box 的诊断信息
    """
    topk_boxes = topk_decoded_boxes(
        cls_scores, bbox_preds, anchors_per_level, bbox_coder,
        img_shape=img_shape, topk=topk, score_thr=0.01)

    if not topk_boxes:
        return []

    gt_box = np.array([gt['cx'], gt['cy'], gt['w'], gt['h'],
                       np.radians(gt['angle'])])  # GT angle 度→弧度

    results = []
    for i, det in enumerate(topk_boxes):
        det_box = np.array([det['cx'], det['cy'], det['w'], det['h'],
                            det['angle']])  # decoded angle 已经是弧度
        center_dist = np.sqrt((det['cx'] - gt['cx'])**2 +
                              (det['cy'] - gt['cy'])**2)
        rious = rbbox_iou(det_box, gt_box)

        # 角度差 (都是弧度, 转角度输出)
        angle_diff_rad = abs(det['angle'] - np.radians(gt['angle']))
        angle_diff_rad = min(angle_diff_rad, np.pi - angle_diff_rad)
        angle_diff = np.degrees(angle_diff_rad)

        # 尺度比
        w_ratio = det['w'] / max(gt['w'], 1e-6)
        h_ratio = det['h'] / max(gt['h'], 1e-6)

        # 判断问题类型
        if rious > 0.5:
            diagnosis = '✓ match (IoU>0.5)'
        elif center_dist < gt['w'] * 0.5:
            diagnosis = '△ near-center but low IoU (scale/angle issue)'
        elif center_dist < img_shape[0] * 0.1:
            diagnosis = '△ close (<10% img)'
        else:
            diagnosis = '✗ far from GT'

        results.append(dict(
            rank=i + 1,
            score=det['score'],
            level=det['level'],
            center_dist=center_dist,
            rious=rious,
            angle_diff=angle_diff,
            w_ratio=w_ratio,
            h_ratio=h_ratio,
            det_cx=det['cx'], det_cy=det['cy'],
            det_w=det['w'], det_h=det['h'],
            det_angle=np.degrees(det['angle']),  # 弧度→度 用于显示
            gt_angle=gt['angle'],  # 已经是度
            diagnosis=diagnosis,
        ))
    return results


def draw_rotated_box(img, cx, cy, w, h, angle_rad, color, thickness=2, label=None):
    """在 img 上画一个旋转矩形。angle 是弧度。"""
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    dx, dy = w / 2.0, h / 2.0
    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    corners = corners @ rot.T
    corners[:, 0] += cx
    corners[:, 1] += cy
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    if label:
        tx, ty = int(corners[0][0]), int(corners[0][1]) - 5
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def visualize_frame(img_path, gt, topk_analysis, vis_path):
    """在原图上绘制 GT (绿) + top-k decoded box (红/蓝)。"""
    img = cv2.imread(img_path)
    if img is None:
        return

    # GT box: 绿色, 粗线
    gt_angle_rad = np.radians(gt['angle'])
    draw_rotated_box(img, gt['cx'], gt['cy'], gt['w'], gt['h'],
                     gt_angle_rad, color=(0, 255, 0), thickness=3, label='GT')

    # Top-k decoded boxes
    for det in topk_analysis:
        # 红色 = 最高分 (#1), 蓝色 = 其他
        if det['rank'] == 1:
            color = (0, 0, 255)  # 红
            thick = 2
        else:
            color = (255, 128, 0)  # 蓝
            thick = 1

        # det_angle 是度, 需要转弧度给 draw_rotated_box
        det_angle_rad = np.radians(det['det_angle'])
        draw_rotated_box(img, det['det_cx'], det['det_cy'],
                         det['det_w'], det['det_h'], det_angle_rad,
                         color=color, thickness=thick,
                         label=f'#{det["rank"]} s={det["score"]:.3f} RIoU={det["rious"]:.2f}')

    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    cv2.imwrite(vis_path, img)


def diagnose_source(model, hook_mgr, transform_compose, img_scale, flip,
                    data_root, split, seq, frame_ids,
                    feat_strides, num_anchors_per_level, threshold, gpu,
                    source_label, bbox_coder=None, anchor_generator=None,
                    topk=5, topk_all=False, vis_dir=None,
                    giant_size_thr=4096.0, giant_ratio_thr=20.0,
                    preproc_cfg=None, save_preproc_dir=None):
    print()
    print('=' * 80)
    print(f'  Source: {source_label}')
    print(f'  Split: {split}  |  Seq: {seq}  |  Frames: {len(frame_ids)}')
    print('=' * 80)

    results = []

    for fid in frame_ids:
        base = seq if 'seq' in seq else f'real_{seq}'
        fname = f'{base}_{fid:05d}'

        img_path, ann_path = find_files(data_root, split, seq, fid)
        if img_path is None:
            continue

        gts = parse_dota_ann(ann_path)

        # Preprocess 用 config 中的 test transforms
        img_tensor, meta, img_stats = preprocess_image(
            img_path, transform_compose, img_scale, flip,
            preproc_cfg=preproc_cfg, save_preproc_dir=save_preproc_dir)
        if img_tensor is None:
            continue
        brightness = img_stats['raw_brightness']

        img_tensor = img_tensor.cuda(f'cuda:{gpu}')

        hook_mgr.cls_scores = None
        hook_mgr._active = True

        with torch.no_grad():
            feat = model.backbone(img_tensor)
            feat = model.neck(feat)
            cls_scores, bbox_preds = model.bbox_head(feat)

        if hook_mgr.cls_scores is None:
            continue

        global_max = max(s.sigmoid().max().item() for s in hook_mgr.cls_scores)
        global_max_level = int(np.argmax(
            [s.sigmoid().max().item() for s in hook_mgr.cls_scores]))

        gt_score = 0.0
        roi_score = 0.0
        gt_analysis = None
        if gts:
            gt = scale_obb_to_img(gts[0], meta)
            gt_analysis = gt_center_score(
                hook_mgr.cls_scores, gt['cx'], gt['cy'],
                meta['img_shape'],
                feat_strides, num_anchors_per_level)
            gt_score = gt_analysis['gt_max_score']
            roi_score = max(l['roi_3x3_max_score']
                           for l in gt_analysis['gt_scores_per_level'])

        if gt_score < threshold:
            status = 'DEAD-global' if global_max < threshold else 'DEAD-local'
        elif gt_score < 0.05:
            status = 'EDGE'
        else:
            status = 'OK'

        results.append(dict(
            frame=fid, fname=fname, status=status,
            brightness=brightness,
            raw_contrast=img_stats['raw_contrast'],
            raw_ud_delta=img_stats['raw_ud_delta'],
            proc_brightness=img_stats['proc_brightness'],
            proc_contrast=img_stats['proc_contrast'],
            proc_ud_delta=img_stats['proc_ud_delta'],
            gt_score=gt_score, roi_score=roi_score,
            global_max=global_max, global_max_level=global_max_level,
            gt_analysis=gt_analysis, split=split, seq=seq,
            n_gt=len(gts),
            preprocess=img_stats['preprocess'],
        ))

        marker = {'DEAD-global': '✗✗', 'DEAD-local': '✗ ',
                  'EDGE': '△ ', 'OK': '✓ '}[status]
        print(f'  [{fname}] {marker} brightness={brightness or 0:6.1f}  '
              f'gt={gt_score:.4f}  roi={roi_score:.4f}  '
              f'global={global_max:.4f}@P{global_max_level}  gt#={len(gts)}  '
              f'proc_brightness={img_stats["proc_brightness"]:.1f}  '
              f'contrast={img_stats["raw_contrast"]:.1f}'
              f'->{img_stats["proc_contrast"]:.1f}  '
              f'ud={img_stats["raw_ud_delta"]:.1f}'
              f'->{img_stats["proc_ud_delta"]:.1f}')

        # 对 DEAD-local / EDGE 帧做 top-k decoded box vs GT 分析
        # --topk-all 时对所有帧都做
        need_topk = (status in ('DEAD-local', 'EDGE') or topk_all)
        if need_topk and gts and bbox_coder is not None and anchor_generator is not None:
            anchors_per_level = generate_anchors(
                anchor_generator, hook_mgr.cls_scores, device='cpu')
            gt = scale_obb_to_img(gts[0], meta)
            topk_analysis = analyze_topk_vs_gt(
                hook_mgr.cls_scores, hook_mgr.bbox_preds,
                anchors_per_level, bbox_coder,
                gt, meta['img_shape'], topk=topk)
            for det in topk_analysis:
                det['giant_box'] = (
                    max(det['det_w'], det['det_h']) > giant_size_thr
                    or max(det['w_ratio'], det['h_ratio']) > giant_ratio_thr)
                print(f'    #{det["rank"]} score={det["score"]:.4f} '
                      f'P{det["level"]}  '
                      f'c_dist={det["center_dist"]:.1f}px  '
                      f'RIoU={det["rious"]:.3f}  '
                      f'Δangle={det["angle_diff"]:.1f}° '
                      f'(det={det["det_angle"]:.1f} gt={det["gt_angle"]:.1f})  '
                      f'w×h={det["det_w"]:.0f}×{det["det_h"]:.0f} '
                      f'(ratio={det["w_ratio"]:.2f}×{det["h_ratio"]:.2f})  '
                      f'{"GIANT " if det["giant_box"] else ""}'
                      f'{det["diagnosis"]}')
            # 记录到 results
            results[-1]['topk_analysis'] = topk_analysis
            if topk_analysis:
                top1 = topk_analysis[0]
                results[-1]['top1_match'] = top1['rious'] > 0.5
                results[-1]['top1_far'] = top1['diagnosis'] == '✗ far from GT'
                results[-1]['top1_giant'] = top1['giant_box']
                results[-1]['any_topk_match'] = any(
                    det['rious'] > 0.5 for det in topk_analysis)

            # 可视化
            if vis_dir and topk_analysis:
                vis_path = os.path.join(vis_dir, f'{fname}_det.jpg')
                visualize_frame(
                    img_path, gts[0],
                    detections_to_ori(topk_analysis, meta), vis_path)

    return results


# =====================================================================
# 汇总报告
# =====================================================================

def print_summary(all_results, threshold):
    print('\n' + '=' * 80)
    print('  SUMMARY')
    print('=' * 80)

    sources = {}
    for r in all_results:
        key = f"{r['split']}/{r['seq']}"
        if key not in sources:
            sources[key] = []
        sources[key].append(r)

    for source, results in sources.items():
        dead_g = sum(1 for r in results if r['status'] == 'DEAD-global')
        dead_l = sum(1 for r in results if r['status'] == 'DEAD-local')
        edge = sum(1 for r in results if r['status'] == 'EDGE')
        ok = sum(1 for r in results if r['status'] == 'OK')
        total = len(results)

        gt_scores = [r['gt_score'] for r in results]
        global_scores = [r['global_max'] for r in results]
        brightnesses = [r['brightness'] for r in results if r['brightness']]

        print(f'\n  Source: {source}')
        print(f'  Total: {total} frames')
        print(f'  ✗✗ DEAD-global: {dead_g}  |  ✗ DEAD-local: {dead_l}  '
              f'|  △ EDGE: {edge}  |  ✓ OK: {ok}')
        print(f'  gt_score:    mean={np.mean(gt_scores):.4f}  '
              f'max={np.max(gt_scores):.4f}  min={np.min(gt_scores):.4f}')
        print(f'  global_max:  mean={np.mean(global_scores):.4f}  '
              f'max={np.max(global_scores):.4f}  min={np.min(global_scores):.4f}')
        if brightnesses:
            print(f'  brightness:  mean={np.mean(brightnesses):.1f}  '
                  f'range=[{min(brightnesses):.1f}, {max(brightnesses):.1f}]')

        analyzed = [r for r in results if 'topk_analysis' in r]
        nonempty = [r for r in analyzed if r.get('topk_analysis')]
        if analyzed:
            top1_match = sum(1 for r in nonempty if r.get('top1_match'))
            any_match = sum(1 for r in nonempty if r.get('any_topk_match'))
            top1_far = sum(1 for r in nonempty if r.get('top1_far'))
            top1_giant = sum(1 for r in nonempty if r.get('top1_giant'))
            top1_rious = [r['topk_analysis'][0]['rious'] for r in nonempty]
            top1_scores = [r['topk_analysis'][0]['score'] for r in nonempty]
            print(f'  top-k decoded: analyzed={len(analyzed)}  '
                  f'nonempty={len(nonempty)}')
            if nonempty:
                print(f'    top1_match(IoU>0.5): {top1_match}/{len(nonempty)}  '
                      f'any_topk_match: {any_match}/{len(nonempty)}  '
                      f'top1_far: {top1_far}/{len(nonempty)}  '
                      f'top1_giant: {top1_giant}/{len(nonempty)}')
                print(f'    top1_RIoU: mean={np.mean(top1_rious):.3f}  '
                      f'min={np.min(top1_rious):.3f}  '
                      f'top1_score_mean={np.mean(top1_scores):.4f}')

    # Cross-source per-source summary
    print('\n' + '=' * 80)
    print('  CROSS-SOURCE DIAGNOSIS')
    print('=' * 80)

    src_global = {}
    for source, results in sources.items():
        if not results:
            continue
        dead_g = sum(1 for r in results if r['status'] == 'DEAD-global')
        dead_l = sum(1 for r in results if r['status'] == 'DEAD-local')
        total = len(results)
        gt_mean = np.mean([r['gt_score'] for r in results])
        global_mean = np.mean([r['global_max'] for r in results])
        src_global[source] = global_mean

        nonempty_topk = [
            r for r in results if r.get('topk_analysis')]
        top1_match_count = sum(
            1 for r in nonempty_topk if r.get('top1_match'))
        top1_giant_count = sum(
            1 for r in nonempty_topk if r.get('top1_giant'))

        if dead_g > total * 0.5:
            print(f'  {source}: ✗✗ {dead_g}/{total} DEAD-global  '
                  f'(gt_mean={gt_mean:.4f}, global_mean={global_mean:.4f})')
        elif top1_giant_count > max(0, len(nonempty_topk) * 0.3):
            print(f'  {source}: [WARN] {top1_giant_count}/{len(nonempty_topk)} '
                  f'top1 giant/far-risk  '
                  f'(gt_mean={gt_mean:.4f}, global_mean={global_mean:.4f})')
        elif nonempty_topk and top1_match_count >= len(nonempty_topk) * 0.7:
            print(f'  {source}: ✓ top1 decoded mostly matches '
                  f'({top1_match_count}/{len(nonempty_topk)})  '
                  f'(gt_mean={gt_mean:.4f}, global_mean={global_mean:.4f})')
        elif dead_l > total * 0.3:
            print(f'  {source}: △ {dead_l}/{total} low-gt/high-global  '
                  f'(gt_mean={gt_mean:.4f}, global_mean={global_mean:.4f})')
        else:
            print(f'  {source}: ✓  '
                  f'(gt_mean={gt_mean:.4f}, global_mean={global_mean:.4f})')

    # Decision tree
    print()
    if len(src_global) >= 2:
        train_keys = [k for k in src_global if 'train' in k]
        test_keys = [k for k in src_global if 'test' in k]

        if train_keys and test_keys:
            train_global = max(src_global[k] for k in train_keys)
            test_global = max(src_global[k] for k in test_keys)

            if train_global > 0.5 and test_global < 0.05:
                print('  → 结论: 训练域 cls head 响应健康; '
                      '测试域 global_max 坍缩 → OOD 全局失活')
                print('    DEAD-global 帧: score_thr 无法挽救, '
                      '需轨迹外推 hold-last/EKF 或 cls 外观鲁棒性预训练')
            elif train_global > 0.5 and test_global < 0.3:
                print('  → 结论: 训练域响应健康; 测试域存在显著外观 OOD')
                print('    DEAD-global 帧是 MCML 主因; '
                      'low gt / high roi 帧需 decoded box 对比确认')
            elif train_global > 0.5 and test_global > 0.3:
                print('  → 结论: 两个域都有响应, 测试域较弱')
            else:
                print('  → 结论: 训练域 global_mean 也低 → cls head 训练不充分')

    # Note about DEAD-local interpretation
    has_dead_local = any(
        sum(1 for r in results if r['status'] == 'DEAD-local') > 0
        for results in sources.values() if results)
    if has_dead_local:
        print('\n  NOTE: 部分帧出现 low gt_score + high roi/global,')
        print('    当前 gt_score 只采样四舍五入后的单个 feature cell,')
        print('    不能直接判定为位置漂移.')
        print('    需 decoded box vs GT 对比 (center dist / RIoU) 才能确认.')
        print('    → 下一步: top-k decoded boxes vs GT 分析')


# =====================================================================
# 主流程
# =====================================================================

def main():
    args = parse_args()
    random.seed(42)

    from mmcv import Config
    from mmrotate.models import build_detector

    cfg = Config.fromfile(args.config)
    cfg.model.test_cfg.score_thr = 0.0
    cfg.model.test_cfg.max_per_img = 100

    model = build_detector(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.cuda(f'cuda:{args.gpu}')
    model.eval()

    hook_mgr = HeadHook()
    handle = model.bbox_head.register_forward_hook(hook_mgr.hook_fn)

    # 从 config 解析 test transforms
    transform_compose, img_scale, flip = build_test_transforms(cfg)

    # Anchor info
    infer_head = get_inference_head(model)
    anchor_gen = infer_head.anchor_generator
    feat_strides = anchor_gen.strides
    feat_strides = [s[0] if isinstance(s, (tuple, list)) else s for s in feat_strides]
    num_base_anchors = anchor_gen.num_base_anchors
    if isinstance(num_base_anchors, list):
        num_anchors_per_level = [n for n in num_base_anchors]
    else:
        num_anchors_per_level = [num_base_anchors] * len(feat_strides)

    # Bbox coder & model anchor generator (for top-k decoded box analysis)
    bbox_coder = infer_head.bbox_coder
    topk = args.topk

    data_root = args.data_root
    all_results = []
    preproc_cfg = dict(
        mode=args.preproc,
        linear_gain=args.linear_gain,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=args.clahe_tile_grid,
    )
    bn_initial_state = snapshot_bn_state(model) if args.adabn else None
    if args.preproc != 'none' or args.adabn:
        print(f'  [probe] preproc={args.preproc}  adabn={args.adabn}')

    if args.split == 'all':
        sources = []

        for seq in discover_sequences(data_root, 'test'):
            if 'sim' in seq:
                continue
            fids = discover_frames(data_root, 'test', seq, args.sample)
            sources.append(('test', seq, fids, f'test/{seq}'))

        for seq in discover_sequences(data_root, 'train'):
            fids = discover_frames(data_root, 'train', seq, args.sample)
            sources.append(('train', seq, fids, f'train/{seq}'))

        for seq in discover_sequences(data_root, 'train_sim'):
            fids = discover_frames(data_root, 'train_sim', seq, args.sample)
            sources.append(('train_sim', seq, fids, f'train_sim/{seq}'))

        for split, seq, fids, label in sources:
            if not fids:
                print(f'  [skip] {label}: no frames found')
                continue
            if args.adabn:
                restore_bn_state(bn_initial_state)
                adapt_bn_on_frames(
                    model, hook_mgr, transform_compose, img_scale, flip,
                    data_root, split, seq, fids, args.gpu, preproc_cfg,
                    save_preproc_dir=args.save_preproc_dir,
                    max_frames=args.adabn_frames,
                    momentum=args.adabn_momentum)
            results = diagnose_source(
                model, hook_mgr, transform_compose, img_scale, flip,
                data_root, split, seq, fids,
                feat_strides, num_anchors_per_level, args.threshold,
                args.gpu, label, bbox_coder, anchor_gen, topk,
                args.topk_all, args.vis_dir,
                args.giant_size_thr, args.giant_ratio_thr,
                preproc_cfg=preproc_cfg,
                save_preproc_dir=args.save_preproc_dir)
            all_results.extend(results)

    else:
        seq = args.seq
        if seq is None:
            seqs = discover_sequences(data_root, args.split)
            if not seqs:
                print(f'No sequences found in {data_root}/{args.split}/')
                return
            seq = seqs[0]
            print(f'  Auto-selected seq: {seq}')

        if args.start is not None and args.end is not None:
            frame_ids = list(range(args.start, args.end + 1))
        else:
            frame_ids = discover_frames(data_root, args.split, seq, args.sample)

        if args.adabn:
            restore_bn_state(bn_initial_state)
            adapt_bn_on_frames(
                model, hook_mgr, transform_compose, img_scale, flip,
                data_root, args.split, seq, frame_ids, args.gpu, preproc_cfg,
                save_preproc_dir=args.save_preproc_dir,
                max_frames=args.adabn_frames,
                momentum=args.adabn_momentum)
        results = diagnose_source(
            model, hook_mgr, transform_compose, img_scale, flip,
            data_root, args.split, seq, frame_ids,
            feat_strides, num_anchors_per_level, args.threshold,
            args.gpu, f'{args.split}/{seq}', bbox_coder, anchor_gen, topk,
            args.topk_all, args.vis_dir,
            args.giant_size_thr, args.giant_ratio_thr,
            preproc_cfg=preproc_cfg,
            save_preproc_dir=args.save_preproc_dir)
        all_results.extend(results)

    handle.remove()
    print_summary(all_results, args.threshold)


if __name__ == '__main__':
    main()
