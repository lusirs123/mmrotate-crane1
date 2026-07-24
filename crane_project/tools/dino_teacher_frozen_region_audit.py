#!/usr/bin/env python3
"""Read-only DINO Teacher-inspired frozen DINOv2 region audit.

The CVPR 2025 DINO Teacher labeller freezes a DINOv2 backbone and trains a
detector head on labelled source images. Before authorizing that training in
SymEOOD, this audit checks the paper's central premise: whether frozen DINOv2
features already separate source objects/background and transfer to the
correct target candidate geometry.

The BrightAug detector is used only to recover candidate boxes. DINOv2-L/14 is
frozen and supplies its final patch-token map. A 7x7 orientation-aware region
pool adapts the paper's single-scale Faster R-CNN ROI pooling to rotated boxes.
No optimizer, pseudo-label, feature alignment, parameter update, or checkpoint
write is performed.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from crane_project.tools import frozen_p3_feature_alignment_audit as alignment  # noqa: E402
from crane_project.tools import frozen_p3_objectness_transfer_probe as transfer  # noqa: E402
from crane_project.tools import p3_p4_multimodal_knn_audit as multimodal  # noqa: E402
from crane_project.tools import p3_p4_neighborhood_rescue_audit as neighborhood  # noqa: E402


AUDIT_NAME = 'DINO Teacher Frozen DINOv2 Region Semantic Audit'
PROTOCOL_VERSION = 1
PAPER_URL = 'https://arxiv.org/abs/2503.23220'
PAPER_CODE_URL = 'https://github.com/TRAILab/DINO_Teacher'
CANONICAL_MODEL = 'dinov2_vitl14'
CANONICAL_PATCH_SIZE = 14
CANONICAL_DINO_HEIGHT = 600
CANONICAL_DINO_MAX_LONG_SIDE = 1333
CANONICAL_POOL_RESOLUTION = 7
CANONICAL_SOURCE_FOLDS = 5
CANONICAL_NEIGHBORS = 5
CANONICAL_POSITIVE_QUANTILE = 0.1
CANONICAL_NEGATIVE_QUANTILE = 0.9
CANONICAL_MIN_FOLD_VOTES = 4
CANONICAL_LEGACY_SDPA_QUERY_CHUNK = 512
_LEGACY_SDPA_QUERY_CHUNK = CANONICAL_LEGACY_SDPA_QUERY_CHUNK


def parse_args():
    parser = argparse.ArgumentParser(description=AUDIT_NAME)
    parser.add_argument('--config', required=True)
    parser.add_argument('--detector-checkpoint', required=True)
    parser.add_argument('--data-root', default='crane_project/data/crane_grab/')
    parser.add_argument('--source-seq', default=neighborhood.SOURCE_SEQ)
    parser.add_argument('--dinov2-repo', required=True,
                        help='Local clone of facebookresearch/dinov2')
    parser.add_argument('--dinov2-checkpoint', required=True,
                        help='Official dinov2_vitl14_pretrain.pth')
    parser.add_argument('--dinov2-model', default=CANONICAL_MODEL)
    parser.add_argument(
        '--dino-gpus', type=int, nargs='+', default=None,
        help='GPU ids for contiguous DINO transformer block sharding; '
             'defaults to --gpu')
    parser.add_argument(
        '--legacy-sdpa-query-chunk', type=int,
        default=CANONICAL_LEGACY_SDPA_QUERY_CHUNK,
        help='Query chunk size for the PyTorch<2.0 attention fallback')
    parser.add_argument('--dino-height', type=int,
                        default=CANONICAL_DINO_HEIGHT,
                        help='DINO input short side; canonical labeller value is 600')
    parser.add_argument('--dino-max-long-side', type=int,
                        default=CANONICAL_DINO_MAX_LONG_SIDE,
                        help='Maximum DINO input long side')
    parser.add_argument('--patch-size', type=int,
                        default=CANONICAL_PATCH_SIZE)
    parser.add_argument('--pool-resolution', type=int,
                        default=CANONICAL_POOL_RESOLUTION)
    parser.add_argument('--source-folds', type=int,
                        default=CANONICAL_SOURCE_FOLDS)
    parser.add_argument('--neighbors', type=int,
                        default=CANONICAL_NEIGHBORS)
    parser.add_argument('--positive-quantile', type=float,
                        default=CANONICAL_POSITIVE_QUANTILE)
    parser.add_argument('--negative-quantile', type=float,
                        default=CANONICAL_NEGATIVE_QUANTILE)
    parser.add_argument('--min-fold-votes', type=int,
                        default=CANONICAL_MIN_FOLD_VOTES)
    parser.add_argument('--max-source-samples', type=int, default=0)
    parser.add_argument('--riou-thr', type=float, default=0.5)
    parser.add_argument('--false-iou-thr', type=float, default=0.1)
    parser.add_argument('--source-min-accuracy', type=float, default=0.8)
    parser.add_argument('--target-min-wins', type=int, default=26)
    parser.add_argument('--target-start', type=int,
                        default=neighborhood.TARGET_START)
    parser.add_argument('--target-end', type=int,
                        default=neighborhood.TARGET_END)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--allow-noncanonical', action='store_true')
    return parser.parse_args()


def validate_args(args) -> bool:
    if args.seed != 0:
        raise ValueError('The unified protocol requires --seed 0')
    if (args.dino_height <= 0 or args.dino_max_long_side <= 0
            or args.patch_size <= 0):
        raise ValueError('DINO dimensions must be positive')
    if args.dino_max_long_side < args.dino_height:
        raise ValueError('--dino-max-long-side must be >= --dino-height')
    if args.pool_resolution <= 0:
        raise ValueError('--pool-resolution must be positive')
    if args.source_folds < 2:
        raise ValueError('--source-folds must be at least 2')
    if args.neighbors <= 0:
        raise ValueError('--neighbors must be positive')
    if not 1 <= args.min_fold_votes <= args.source_folds:
        raise ValueError('--min-fold-votes must be in [1, source-folds]')
    if not 0.0 < args.positive_quantile < 0.5:
        raise ValueError('--positive-quantile must be in (0, 0.5)')
    if not 0.5 < args.negative_quantile < 1.0:
        raise ValueError('--negative-quantile must be in (0.5, 1)')
    if args.max_source_samples < 0:
        raise ValueError('--max-source-samples must be non-negative')
    if not 0.0 <= args.false_iou_thr < args.riou_thr <= 1.0:
        raise ValueError('Require 0 <= false-iou-thr < riou-thr <= 1')
    if not 0.0 < args.source_min_accuracy <= 1.0:
        raise ValueError('--source-min-accuracy must be in (0, 1]')
    if args.target_min_wins <= 0:
        raise ValueError('--target-min-wins must be positive')
    if args.legacy_sdpa_query_chunk <= 0:
        raise ValueError('--legacy-sdpa-query-chunk must be positive')
    if args.dino_gpus is not None:
        if not args.dino_gpus or any(value < 0 for value in args.dino_gpus):
            raise ValueError('--dino-gpus must contain non-negative ids')
        if len(set(args.dino_gpus)) != len(args.dino_gpus):
            raise ValueError('--dino-gpus must not contain duplicates')

    checks = dict(
        config=(os.path.basename(args.config)
                == neighborhood.CANONICAL_CONFIG),
        detector_checkpoint=(os.path.basename(args.detector_checkpoint)
                             == neighborhood.CANONICAL_CHECKPOINT),
        source_seq=args.source_seq == neighborhood.SOURCE_SEQ,
        dino_model=args.dinov2_model == CANONICAL_MODEL,
        dino_height=int(args.dino_height) == CANONICAL_DINO_HEIGHT,
        dino_max_long_side=(int(args.dino_max_long_side)
                            == CANONICAL_DINO_MAX_LONG_SIDE),
        patch_size=int(args.patch_size) == CANONICAL_PATCH_SIZE,
        pool_resolution=(int(args.pool_resolution)
                         == CANONICAL_POOL_RESOLUTION),
        source_folds=int(args.source_folds) == CANONICAL_SOURCE_FOLDS,
        neighbors=int(args.neighbors) == CANONICAL_NEIGHBORS,
        positive_quantile=math.isclose(
            float(args.positive_quantile), CANONICAL_POSITIVE_QUANTILE),
        negative_quantile=math.isclose(
            float(args.negative_quantile), CANONICAL_NEGATIVE_QUANTILE),
        fold_votes=int(args.min_fold_votes) == CANONICAL_MIN_FOLD_VOTES,
        full_source=args.max_source_samples == 0,
        thresholds=(math.isclose(args.riou_thr, 0.5)
                    and math.isclose(args.false_iou_thr, 0.1)),
        source_gate=math.isclose(args.source_min_accuracy, 0.8),
        target_gate=int(args.target_min_wins) == 26,
        target_slice=(int(args.target_start) == neighborhood.TARGET_START
                      and int(args.target_end) == neighborhood.TARGET_END))
    canonical = all(checks.values())
    if not canonical and not args.allow_noncanonical:
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(
            'Canonical DINO audit mismatch: {}. '
            'Use --allow-noncanonical only for smoke tests.'.format(
                ', '.join(failed)))
    return canonical


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _number(value) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def _torch_load(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def legacy_scaled_dot_product_attention(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        attn_mask=None, dropout_p: float = 0.0,
        is_causal: bool = False) -> torch.Tensor:
    """PyTorch <2.0 fallback matching eval-time SDPA semantics."""
    scale = float(query.shape[-1]) ** -0.5
    query_length = int(query.shape[-2])
    key_length = int(key.shape[-2])
    key_transposed = key.transpose(-2, -1)
    chunk_size = min(int(_LEGACY_SDPA_QUERY_CHUNK), query_length)
    outputs = []
    for start in range(0, query_length, chunk_size):
        end = min(start + chunk_size, query_length)
        scores = torch.matmul(
            query[..., start:end, :], key_transposed) * scale
        if is_causal:
            query_positions = torch.arange(
                start, end, device=query.device).view(-1, 1)
            key_positions = torch.arange(
                key_length, device=query.device).view(1, -1)
            scores = scores.masked_fill(
                key_positions > query_positions, float('-inf'))
        if attn_mask is not None:
            mask = attn_mask
            if mask.ndim >= 2 and int(mask.shape[-2]) == query_length:
                mask = mask[..., start:end, :]
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float('-inf'))
            else:
                scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        if float(dropout_p) > 0.0:
            weights = F.dropout(
                weights, p=float(dropout_p), training=True)
        outputs.append(torch.matmul(weights, value))
    return torch.cat(outputs, dim=-2)


def configure_legacy_sdpa_query_chunk(chunk_size: int):
    global _LEGACY_SDPA_QUERY_CHUNK
    if int(chunk_size) <= 0:
        raise ValueError('Legacy SDPA query chunk must be positive')
    _LEGACY_SDPA_QUERY_CHUNK = int(chunk_size)


def install_torch_sdpa_compatibility(functional_module=F) -> bool:
    if hasattr(functional_module, 'scaled_dot_product_attention'):
        return False
    setattr(functional_module, 'scaled_dot_product_attention',
            legacy_scaled_dot_product_attention)
    return True


def _unwrap_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    state = checkpoint
    while isinstance(state, dict):
        next_state = None
        for key in ('teacher', 'model', 'state_dict', 'student', 'backbone'):
            value = state.get(key)
            if isinstance(value, dict) and value:
                next_state = value
                break
        if next_state is None:
            break
        state = next_state
    if not isinstance(state, dict) or not state:
        raise RuntimeError('DINO checkpoint does not contain a state dict')
    prefixes = ('module.', 'backbone.', 'encoder.')
    cleaned = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
                    changed = True
        cleaned[clean] = value
    if not cleaned:
        raise RuntimeError('DINO checkpoint has no tensor parameters')
    return cleaned


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device)
                for key, item in value.items()}
    return value


def contiguous_device_indices(block_count: int,
                              device_count: int) -> List[int]:
    if block_count < 0 or device_count <= 0:
        raise ValueError('Invalid block/device count')
    if block_count and device_count > block_count:
        raise ValueError('DINO device count exceeds transformer block count')
    return [min((index * device_count) // max(block_count, 1),
                device_count - 1)
            for index in range(block_count)]


def shard_frozen_dinov2(model, devices: Sequence[torch.device]) -> Dict:
    devices = [torch.device(device) for device in devices]
    if not devices:
        raise ValueError('At least one DINO device is required')
    model.to(devices[0])
    blocks = getattr(model, 'blocks', None)
    if len(devices) > 1 and (
            blocks is None or not isinstance(blocks, torch.nn.ModuleList)
            or len(blocks) < len(devices)):
        raise RuntimeError(
            'Multi-GPU DINO sharding requires a flat ModuleList of blocks')
    block_devices = []
    hooks = []
    if blocks is not None:
        block_count = len(blocks)
        assignments = contiguous_device_indices(block_count, len(devices))
        for block, device_index in zip(blocks, assignments):
            target = devices[device_index]
            block.to(target)
            block_devices.append(str(target))

            def move_inputs(_module, inputs, target_device=target):
                return _move_to_device(inputs, target_device)

            hooks.append(block.register_forward_pre_hook(move_inputs))
    final_device = devices[-1] if block_devices else devices[0]
    norm = getattr(model, 'norm', None)
    if isinstance(norm, torch.nn.Module):
        norm.to(final_device)
    model._sym_dino_device_hooks = hooks
    metadata = dict(
        input_device=str(devices[0]), final_device=str(final_device),
        requested_devices=[str(device) for device in devices],
        block_count=len(block_devices), block_devices=block_devices,
        blocks_per_device={
            str(device): int(sum(item == str(device)
                                 for item in block_devices))
            for device in devices})
    model._sym_dino_device_map = metadata
    return metadata


def load_frozen_dinov2(repo: str, checkpoint: str, model_name: str,
                       devices, legacy_sdpa_query_chunk: int =
                       CANONICAL_LEGACY_SDPA_QUERY_CHUNK):
    hubconf = os.path.join(repo, 'hubconf.py')
    if not os.path.isfile(hubconf):
        raise RuntimeError(
            '--dinov2-repo must be a local DINOv2 clone with hubconf.py')
    if not os.path.isfile(checkpoint):
        raise RuntimeError('--dinov2-checkpoint does not exist')
    configure_legacy_sdpa_query_chunk(legacy_sdpa_query_chunk)
    sdpa_compatibility = install_torch_sdpa_compatibility()
    print('[torch-compat] legacy_sdpa={} query_chunk={}'.format(
        bool(sdpa_compatibility), int(legacy_sdpa_query_chunk)))
    try:
        model = torch.hub.load(
            repo, model_name, source='local', pretrained=False)
    except TypeError as error:
        if sys.version_info < (3, 10) and 'unsupported operand type' in str(error):
            raise RuntimeError(
                'DINOv2 uses Python 3.10 union annotations. Run '
                '`PYTHONPATH=. python3 crane_project/tools/'
                'patch_dinov2_py38.py --repo {}` once, then retry.'.format(
                    repo)) from error
        raise
    state = _unwrap_state_dict(_torch_load(checkpoint))
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            'DINO checkpoint/model mismatch: missing={} unexpected={}'.format(
                missing, unexpected))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if isinstance(devices, (str, torch.device)):
        devices = [torch.device(devices)]
    device_map = shard_frozen_dinov2(model, devices)
    print('[dino-shard] blocks_per_device={} input={} final={}'.format(
        device_map['blocks_per_device'], device_map['input_device'],
        device_map['final_device']))
    model._sym_legacy_sdpa_installed = bool(sdpa_compatibility)
    patch_size = getattr(model, 'patch_size', None)
    if isinstance(patch_size, (tuple, list, torch.Size)):
        patch_size = patch_size[0]
    if patch_size is None and hasattr(model, 'patch_embed'):
        patch_size = getattr(model.patch_embed, 'patch_size', None)
        if isinstance(patch_size, (tuple, list, torch.Size)):
            patch_size = patch_size[0]
    if patch_size is None:
        raise RuntimeError('Loaded DINOv2 model does not expose patch size')
    return model, int(patch_size)


def resize_and_normalize_bgr(
        image_bgr: np.ndarray, target_height: int, patch_size: int,
        max_long_side: int = CANONICAL_DINO_MAX_LONG_SIDE
        ) -> Tuple[torch.Tensor, Dict]:
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError('Expected a BGR image [H,W,3]')
    ori_h, ori_w = [int(value) for value in image_bgr.shape[:2]]
    if ori_h <= 0 or ori_w <= 0:
        raise ValueError('DINO input image must be non-empty')
    if target_height <= 0 or max_long_side <= 0 or patch_size <= 0:
        raise ValueError('DINO resize dimensions must be positive')
    scale = min(
        float(target_height) / float(min(ori_h, ori_w)),
        float(max_long_side) / float(max(ori_h, ori_w)))
    resized_h = max(1, int(round(float(ori_h) * scale)))
    resized_w = max(1, int(round(float(ori_w) * scale)))
    resized = cv2.resize(
        image_bgr, (resized_w, resized_h),
        interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pad_h = int(math.ceil(resized_h / float(patch_size)) * patch_size)
    pad_w = int(math.ceil(resized_w / float(patch_size)) * patch_size)
    padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    padded[:resized_h, :resized_w] = rgb.astype(np.float32)
    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(
        [123.675, 116.280, 103.530], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(
        [58.395, 57.120, 57.375], dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor, dict(
        ori_shape=[ori_h, ori_w],
        resized_shape=[resized_h, resized_w],
        padded_shape=[pad_h, pad_w],
        scale=_number(scale), patch_size=int(patch_size),
        target_short_side=int(target_height),
        max_long_side=int(max_long_side))


def extract_patch_grid(model, tensor: torch.Tensor,
                       patch_size: int) -> torch.Tensor:
    height, width = [int(value) for value in tensor.shape[-2:]]
    if height % patch_size or width % patch_size:
        raise ValueError('DINO tensor dimensions must be patch-divisible')
    with torch.no_grad():
        outputs = model.get_intermediate_layers(tensor, n=1)
    tokens = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    if isinstance(tokens, (list, tuple)):
        tokens = tokens[0]
    if tokens.ndim == 4:
        feature = tokens
    elif tokens.ndim == 3:
        grid_h = height // patch_size
        grid_w = width // patch_size
        if int(tokens.shape[1]) != grid_h * grid_w:
            raise RuntimeError(
                'DINO patch-token count does not match the input grid')
        feature = tokens.transpose(1, 2).contiguous().reshape(
            int(tokens.shape[0]), int(tokens.shape[2]), grid_h, grid_w)
    else:
        raise RuntimeError('Unsupported DINO intermediate feature shape')
    return feature.detach()


def detector_box_to_dino(box: Sequence[float], detector_meta: Dict,
                         dino_meta: Dict) -> torch.Tensor:
    scale_factor = np.asarray(
        detector_meta['scale_factor'], dtype=np.float64).reshape(-1)
    detector_sx = float(scale_factor[0])
    detector_sy = float(scale_factor[1] if scale_factor.size >= 2
                        else scale_factor[0])
    if not math.isclose(detector_sx, detector_sy, rel_tol=1e-3,
                        abs_tol=1e-3):
        raise RuntimeError('Rotated OBB mapping requires isotropic resize')
    if detector_sx <= 0.0 or detector_sy <= 0.0:
        raise RuntimeError('Detector scale factor must be positive')
    dino_scale = float(dino_meta['scale'])
    values = torch.as_tensor(box, dtype=torch.float32).reshape(-1)[:5].clone()
    values[0] = values[0] / detector_sx * dino_scale
    values[1] = values[1] / detector_sy * dino_scale
    values[2] = values[2] / detector_sx * dino_scale
    values[3] = values[3] / detector_sy * dino_scale
    return values


def oriented_roi_vector(feature: torch.Tensor, box: torch.Tensor,
                        patch_size: int, output_size: int) -> Tuple[torch.Tensor, Dict]:
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError('Expected one DINO feature map [1,C,H,W]')
    if box.numel() < 5:
        raise ValueError('Expected a rotated box [cx,cy,w,h,angle]')
    if patch_size <= 0 or output_size <= 0:
        raise ValueError('ROI patch size and output size must be positive')
    _, _, height, width = feature.shape
    cx, cy, box_w, box_h, angle = box[:5].to(
        device=feature.device, dtype=feature.dtype)
    bins = (torch.arange(
        output_size, device=feature.device, dtype=feature.dtype) + 0.5
            ) / float(output_size) - 0.5
    local_y, local_x = torch.meshgrid(bins, bins, indexing='ij')
    local_x = local_x * box_w.abs().clamp_min(1.0)
    local_y = local_y * box_h.abs().clamp_min(1.0)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    pixel_x = cx + cos_a * local_x - sin_a * local_y
    pixel_y = cy + sin_a * local_x + cos_a * local_y
    feature_x = pixel_x / float(patch_size) - 0.5
    feature_y = pixel_y / float(patch_size) - 0.5
    if width > 1:
        norm_x = 2.0 * feature_x / float(width - 1) - 1.0
    else:
        norm_x = torch.zeros_like(feature_x)
    if height > 1:
        norm_y = 2.0 * feature_y / float(height - 1) - 1.0
    else:
        norm_y = torch.zeros_like(feature_y)
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        feature, grid, mode='bilinear', padding_mode='zeros',
        align_corners=True)
    vector = sampled.mean(dim=(0, 2, 3)).detach().cpu()
    inside = ((pixel_x >= 0.0)
              & (pixel_x < float(width * patch_size))
              & (pixel_y >= 0.0)
              & (pixel_y < float(height * patch_size)))
    return vector, dict(
        pool_resolution=int(output_size),
        sampled_points=int(output_size * output_size),
        in_bounds_fraction=_number(inside.float().mean().item()),
        vector_norm=_number(vector.norm().item()))


def _candidate_record(index: int, boxes: torch.Tensor,
                      scores: torch.Tensor, ious: torch.Tensor,
                      layout: Sequence[Dict]) -> Dict:
    location = layout[int(index)]
    return dict(
        candidate_index=int(index), level=int(location['level']),
        row=int(location['row']), col=int(location['col']),
        anchor_id=int(location['anchor_id']),
        main_cls_score=_number(scores[index].item()),
        riou=_number(ious[index].item()),
        decoded_obb=[_number(value) for value in boxes[index, :5].tolist()])


def _prepare_image_features(model, image_path: str, target_height: int,
                            patch_size: int, max_long_side: int,
                            device: torch.device):
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError('Failed to read image {}'.format(image_path))
    tensor, meta = resize_and_normalize_bgr(
        image, target_height, patch_size, max_long_side)
    tensor = tensor.to(device=device)
    feature = extract_patch_grid(model, tensor, patch_size)
    return feature, meta


def collect_source(detector, dino, records: Sequence[Dict], transforms,
                   img_scale, flip, args, device: torch.device):
    from mmcv.ops import box_iou_rotated

    diag = transfer.entry_probe.get_diag()
    samples = []
    rows = []
    for record_index, record in enumerate(records):
        img_tensor, detector_meta, image_stats = diag.preprocess_image(
            record['image'], transforms, img_scale, flip)
        if img_tensor is None:
            continue
        img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))
        with torch.no_grad():
            features = detector.extract_feat(img_tensor)
            _head, boxes, scores, layout, decode_alignment = (
                transfer.forward_main_candidates(
                    detector, features, detector_meta['img_shape']))
            gt_boxes = transfer.scaled_gt_tensors(
                record, detector_meta, boxes.device)
            if gt_boxes.numel() == 0:
                continue
            ious = box_iou_rotated(
                boxes.float(), gt_boxes.float()).max(dim=1).values
            false_index = transfer.select_level_candidate(
                scores, ious, layout, 0, max_iou=args.false_iou_thr)
            if false_index is None:
                continue
            boxes_cpu = boxes.detach().cpu()
            scores_cpu = scores.detach().cpu()
            ious_cpu = ious.detach().cpu()
            gt_boxes_cpu = gt_boxes.detach().cpu()
        del img_tensor, features, boxes, scores, ious, gt_boxes
        dino_feature, dino_meta = _prepare_image_features(
            dino, record['image'], args.dino_height,
            args.patch_size, args.dino_max_long_side, device)
        false_box = detector_box_to_dino(
            boxes_cpu[int(false_index), :5], detector_meta, dino_meta)
        negative_vector, negative_pool = oriented_roi_vector(
            dino_feature, false_box.to(device), args.patch_size,
            args.pool_resolution)
        false_record = _candidate_record(
            false_index, boxes_cpu, scores_cpu, ious_cpu, layout)
        false_record['dino_box'] = [
            _number(value) for value in false_box.tolist()]
        false_record['dino_pool'] = negative_pool
        for gt_index, gt_box in enumerate(gt_boxes_cpu):
            positive_box = detector_box_to_dino(
                gt_box[:5], detector_meta, dino_meta)
            positive_vector, positive_pool = oriented_roi_vector(
                dino_feature, positive_box.to(device), args.patch_size,
                args.pool_resolution)
            row = dict(
                role='source_real_validation_control',
                split=neighborhood.SOURCE_SPLIT, seq=record['seq'],
                frame=int(record['frame']), gt_index=int(gt_index),
                image_stats=image_stats,
                dino_preprocess=dino_meta,
                positive=dict(
                    detector_obb=[_number(value)
                                  for value in gt_box[:5].tolist()],
                    dino_obb=[_number(value)
                              for value in positive_box.tolist()],
                    dino_pool=positive_pool),
                hard_negative=copy.deepcopy(false_record),
                decode_alignment=decode_alignment)
            rows.append(row)
            samples.append(dict(
                order=len(samples), row=row,
                positive_vector=positive_vector,
                negative_vector=negative_vector.clone()))
        if (record_index + 1) % 25 == 0:
            print('[source-dino] {}/{} images'.format(
                record_index + 1, len(records)))
    if not samples:
        raise RuntimeError('No source DINO controls were collected')
    return samples, rows


def build_source_ensemble(samples: Sequence[Dict], folds: int,
                          neighbors: int, positive_quantile: float,
                          negative_quantile: float):
    fold_ids = neighborhood.contiguous_fold_ids(len(samples), int(folds))
    fold_models = []
    control_records = []
    for fold_id in range(int(folds)):
        reference = [sample for index, sample in enumerate(samples)
                     if fold_ids[index] != fold_id]
        controls = [sample for index, sample in enumerate(samples)
                    if fold_ids[index] == fold_id]
        if len(reference) < int(neighbors) or not controls:
            raise RuntimeError('DINO source fold {} is too small'.format(
                fold_id))
        bank = multimodal.build_knn_bank(
            torch.stack([sample['positive_vector'] for sample in reference]),
            torch.stack([sample['negative_vector'] for sample in reference]),
            neighbors)
        pending = []
        for sample in controls:
            positive_scores = multimodal.knn_scores(
                sample['positive_vector'], bank)
            negative_scores = multimodal.knn_scores(
                sample['negative_vector'], bank)
            pending.append((sample, positive_scores, negative_scores))
        raw_calibration = multimodal.calibrated_threshold(
            [item[1]['cosine_preference_positive'] for item in pending],
            [item[2]['cosine_preference_positive'] for item in pending],
            positive_quantile, negative_quantile)
        white_calibration = multimodal.calibrated_threshold(
            [item[1]['whitened_preference_positive'] for item in pending],
            [item[2]['whitened_preference_positive'] for item in pending],
            positive_quantile, negative_quantile)
        fold_model = dict(
            fold_id=int(fold_id), bank=bank,
            reference_count=len(reference), control_count=len(controls),
            cosine_preference_threshold=raw_calibration['threshold'],
            whitened_preference_threshold=white_calibration['threshold'],
            raw_calibration=raw_calibration,
            whitened_calibration=white_calibration)
        fold_models.append(fold_model)
        for sample, positive_scores, negative_scores in pending:
            control = dict(
                fold_id=int(fold_id),
                cosine_preference_threshold=raw_calibration['threshold'],
                whitened_preference_threshold=white_calibration['threshold'],
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                positive_cosine_pass=bool(
                    positive_scores['cosine_preference_positive']
                    >= raw_calibration['threshold']),
                negative_cosine_pass=bool(
                    negative_scores['cosine_preference_positive']
                    < raw_calibration['threshold']),
                positive_whitened_pass=bool(
                    positive_scores['whitened_preference_positive']
                    >= white_calibration['threshold']),
                negative_whitened_pass=bool(
                    negative_scores['whitened_preference_positive']
                    < white_calibration['threshold']),
                zero_margin_positive_cosine_pass=bool(
                    positive_scores['cosine_preference_positive'] > 0.0),
                zero_margin_negative_cosine_pass=bool(
                    negative_scores['cosine_preference_positive'] < 0.0),
                zero_margin_positive_whitened_pass=bool(
                    positive_scores['whitened_preference_positive'] > 0.0),
                zero_margin_negative_whitened_pass=bool(
                    negative_scores['whitened_preference_positive'] < 0.0))
            sample['row']['source_knn_control'] = control
            control_records.append(control)
    source_summary = multimodal.zero_margin_control_summary(control_records)
    calibrated_summary = neighborhood.source_level_control_summary(
        control_records)
    metadata = dict(
        folds=int(folds), sample_count=len(samples),
        dimension=int(fold_models[0]['bank']['raw_positive'].shape[1]),
        neighbors=int(neighbors),
        positive_quantile=_number(positive_quantile),
        negative_quantile=_number(negative_quantile),
        fold_sizes=[dict(
            fold_id=model['fold_id'],
            reference_count=model['reference_count'],
            control_count=model['control_count'],
            cosine_preference_threshold=model[
                'cosine_preference_threshold'],
            whitened_preference_threshold=model[
                'whitened_preference_threshold'],
            raw_calibration=model['raw_calibration'],
            whitened_calibration=model['whitened_calibration'])
            for model in fold_models])
    return fold_models, source_summary, calibrated_summary, metadata


def score_region(vector: torch.Tensor, false_vector: torch.Tensor,
                 fold_models: Sequence[Dict], min_fold_votes: int) -> Dict:
    folds = []
    votes = 0
    decision_margins = []
    raw_margins = []
    white_margins = []
    for model in fold_models:
        usable = multimodal.knn_scores(vector, model['bank'])
        false = multimodal.knn_scores(false_vector, model['bank'])
        raw_margin = (
            usable['cosine_preference_positive']
            - model['cosine_preference_threshold'])
        white_margin = (
            usable['whitened_preference_positive']
            - model['whitened_preference_threshold'])
        passes = bool(
            raw_margin >= 0.0 and white_margin >= 0.0
            and usable['cosine_preference_positive']
            > false['cosine_preference_positive']
            and usable['whitened_preference_positive']
            > false['whitened_preference_positive'])
        votes += int(passes)
        raw_margins.append(raw_margin)
        white_margins.append(white_margin)
        decision_margins.append(min(raw_margin, white_margin))
        folds.append(dict(
            fold_id=int(model['fold_id']),
            cosine_preference_threshold=model[
                'cosine_preference_threshold'],
            whitened_preference_threshold=model[
                'whitened_preference_threshold'],
            usable=usable, matched_false=false,
            calibrated_cosine_margin=_number(raw_margin),
            calibrated_whitened_margin=_number(white_margin),
            passes=passes))
    return dict(
        fold_count=len(folds), fold_votes=int(votes),
        required_fold_votes=int(min_fold_votes),
        rescued=bool(votes >= int(min_fold_votes)),
        mean_decision_margin=_number(np.mean(decision_margins)),
        mean_calibrated_cosine_margin=_number(np.mean(raw_margins)),
        mean_calibrated_whitened_margin=_number(np.mean(white_margins)),
        folds=folds)


def collect_target(detector, dino, transforms, img_scale, flip, args,
                   device: torch.device, fold_models: Sequence[Dict]):
    from mmcv.ops import box_iou_rotated

    diag = transfer.entry_probe.get_diag()
    rows = []
    for frame_id in range(args.target_start, args.target_end + 1):
        img_path, ann_path = diag.find_files(
            args.data_root, neighborhood.TARGET_SPLIT,
            neighborhood.TARGET_SEQ, frame_id)
        if img_path is None or ann_path is None:
            raise RuntimeError('Missing target-dev frame {}'.format(frame_id))
        record = dict(
            split=neighborhood.TARGET_SPLIT,
            seq=neighborhood.TARGET_SEQ, frame=frame_id,
            image=img_path, annotation=ann_path, domain='real')
        img_tensor, detector_meta, image_stats = diag.preprocess_image(
            img_path, transforms, img_scale, flip)
        if img_tensor is None:
            raise RuntimeError('Target preprocessing failed')
        img_tensor = img_tensor.cuda('cuda:{}'.format(args.gpu))
        with torch.no_grad():
            features = detector.extract_feat(img_tensor)
            _head, boxes, scores, layout, decode_alignment = (
                transfer.forward_main_candidates(
                    detector, features, detector_meta['img_shape']))
            gt_boxes = transfer.scaled_gt_tensors(
                record, detector_meta, boxes.device)
            if gt_boxes.numel() == 0:
                raise RuntimeError('Missing target GT')
            ious = box_iou_rotated(
                boxes.float(), gt_boxes.float()).max(dim=1).values
            usable_index = transfer.select_level_candidate(
                scores, ious, layout, 0, min_iou=args.riou_thr)
            false_index = transfer.select_level_candidate(
                scores, ious, layout, 0, max_iou=args.false_iou_thr)
            if false_index is None:
                raise RuntimeError('No target P3 matched false candidate')
            boxes_cpu = boxes.detach().cpu()
            scores_cpu = scores.detach().cpu()
            ious_cpu = ious.detach().cpu()
            dense_best_riou = _number(ious_cpu.max().item())
        del img_tensor, features, boxes, scores, ious, gt_boxes
        dino_feature, dino_meta = _prepare_image_features(
            dino, img_path, args.dino_height, args.patch_size,
            args.dino_max_long_side, device)
        false_box = detector_box_to_dino(
            boxes_cpu[int(false_index), :5], detector_meta, dino_meta)
        false_vector, false_pool = oriented_roi_vector(
            dino_feature, false_box.to(device), args.patch_size,
            args.pool_resolution)
        false_record = _candidate_record(
            false_index, boxes_cpu, scores_cpu, ious_cpu, layout)
        false_record['dino_obb'] = [
            _number(value) for value in false_box.tolist()]
        false_record['dino_pool'] = false_pool
        usable_record = None
        region_score = None
        if usable_index is not None:
            usable_box = detector_box_to_dino(
                boxes_cpu[int(usable_index), :5], detector_meta, dino_meta)
            usable_vector, usable_pool = oriented_roi_vector(
                dino_feature, usable_box.to(device), args.patch_size,
                args.pool_resolution)
            usable_record = _candidate_record(
                usable_index, boxes_cpu, scores_cpu, ious_cpu, layout)
            usable_record['dino_obb'] = [
                _number(value) for value in usable_box.tolist()]
            usable_record['dino_pool'] = usable_pool
            region_score = score_region(
                usable_vector, false_vector, fold_models,
                args.min_fold_votes)
        rows.append(dict(
            role='target_dev_diagnosis_only',
            split=neighborhood.TARGET_SPLIT,
            seq=neighborhood.TARGET_SEQ, frame=int(frame_id),
            image_stats=image_stats, dino_preprocess=dino_meta,
            eligible=usable_record is not None,
            geometry_miss=usable_record is None,
            dense_best_riou=dense_best_riou,
            usable_candidate=usable_record,
            matched_false=false_record,
            dino_region_score=region_score,
            decode_alignment=decode_alignment))
        print('[target-dino] frame {} eligible={}'.format(
            frame_id, usable_record is not None))
    return rows


def _margin_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return dict(minimum=None, median=None, maximum=None)
    return dict(
        minimum=_number(min(values)),
        median=_number(np.median(values)),
        maximum=_number(max(values)))


def summarize_target(rows: Sequence[Dict], folds: int) -> Dict:
    eligible = [row for row in rows if row['eligible']]
    scores = [row['dino_region_score'] for row in eligible]
    wins = [score['rescued'] for score in scores]
    vote_histogram = {
        str(votes): int(sum(score['fold_votes'] == votes for score in scores))
        for votes in range(int(folds) + 1)}
    decision_margins = [score['mean_decision_margin'] for score in scores]
    leave_one_out_accuracy = []
    leave_one_out_median = []
    if len(scores) >= 2:
        win_array = np.asarray(wins, dtype=np.float64)
        margin_array = np.asarray(decision_margins, dtype=np.float64)
        for index in range(len(scores)):
            leave_one_out_accuracy.append(_number(
                np.delete(win_array, index).mean()))
            leave_one_out_median.append(_number(
                np.median(np.delete(margin_array, index))))
    return dict(
        eligible_count=len(eligible), win_count=int(sum(wins)),
        accuracy=_number(alignment.accuracy(wins)),
        vote_histogram=vote_histogram,
        decision_margin=_margin_summary(decision_margins),
        cosine_margin=_margin_summary([
            score['mean_calibrated_cosine_margin'] for score in scores]),
        whitened_margin=_margin_summary([
            score['mean_calibrated_whitened_margin'] for score in scores]),
        leave_one_out_min_accuracy=(
            None if not leave_one_out_accuracy
            else _number(min(leave_one_out_accuracy))),
        leave_one_out_min_median_margin=(
            None if not leave_one_out_median
            else _number(min(leave_one_out_median))))


def make_gate(source_summary: Dict, calibrated_source_summary: Dict,
              target_summary: Dict, geometry_misses: Sequence[int],
              args) -> Dict:
    source_zero_margin_valid = neighborhood.source_level_valid(
        source_summary, float(args.source_min_accuracy))
    source_calibrated_valid = neighborhood.source_level_valid(
        calibrated_source_summary, float(args.source_min_accuracy))
    source_valid = bool(
        source_zero_margin_valid and source_calibrated_valid)
    common = dict(
        eligible_count=(target_summary['eligible_count']
                        == neighborhood.EXPECTED_ELIGIBLE),
        expected_geometry_misses=(list(geometry_misses)
                                  == neighborhood.EXPECTED_GEOMETRY_MISSES))
    target_checks = dict(
        wins=(target_summary['win_count'] >= int(args.target_min_wins)),
        median_margin=(target_summary['decision_margin']['median'] is not None
                       and target_summary['decision_margin']['median'] > 0.0),
        single_frame_robust=(
            target_summary['leave_one_out_min_accuracy'] is not None
            and target_summary['leave_one_out_min_accuracy']
            >= float(args.source_min_accuracy)
            and target_summary['leave_one_out_min_median_margin'] is not None
            and target_summary['leave_one_out_min_median_margin'] > 0.0))
    if not all(common.values()):
        decision = 'AUDIT_INVALID'
        interpretation = (
            'Target geometry invariants failed. Do not select an architecture.')
    elif not source_valid:
        decision = 'DINO_SOURCE_CONTROL_INCONCLUSIVE'
        interpretation = (
            'Frozen DINOv2 regions do not reliably separate source object '
            'and background controls. Do not train a DINO labeller.')
    elif all(target_checks.values()):
        decision = 'AUTHORIZE_DINO_TEACHER_SOURCE_LABELLER'
        interpretation = (
            'Frozen DINOv2-L/14 region semantics transfer to the target hard '
            'slice. Authorize one source-only frozen-backbone labeller or '
            'objectness-head experiment; target adaptation remains closed.')
    else:
        decision = 'DINO_TEACHER_FEATURES_INSUFFICIENT'
        interpretation = (
            'The paper-style frozen DINOv2 representation does not recover '
            'source-calibrated target semantics. Do not train a DINO labeller '
            'from this source split; next reconsider data/domain support.')
    return dict(
        decision=decision, common_checks=common,
        source_zero_margin_valid=bool(source_zero_margin_valid),
        source_calibrated_valid=bool(source_calibrated_valid),
        source_valid=bool(source_valid), target_checks=target_checks,
        target_min_wins=int(args.target_min_wins),
        interpretation=interpretation)


def main():
    args = parse_args()
    canonical = validate_args(args)
    set_seed(args.seed)
    dino_gpu_ids = (list(args.dino_gpus) if args.dino_gpus is not None
                    else [int(args.gpu)])
    if torch.cuda.is_available():
        invalid = [gpu for gpu in dino_gpu_ids
                   if gpu >= torch.cuda.device_count()]
        if invalid:
            raise RuntimeError(
                'DINO GPU ids {} exceed visible CUDA device count {}'.format(
                    invalid, torch.cuda.device_count()))
    dino_devices = [torch.device('cuda:{}'.format(gpu))
                    for gpu in dino_gpu_ids]
    device = dino_devices[0]

    detector, cfg = transfer.entry_probe.load_model(
        args.config, args.detector_checkpoint, args.gpu)
    transfer.freeze_detector(detector)
    detector_versions = alignment.module_parameter_versions(detector)
    dino, loaded_patch_size = load_frozen_dinov2(
        args.dinov2_repo, args.dinov2_checkpoint,
        args.dinov2_model, dino_devices,
        args.legacy_sdpa_query_chunk)
    if loaded_patch_size != int(args.patch_size):
        raise RuntimeError(
            'Loaded DINO patch size {} != protocol {}'.format(
                loaded_patch_size, args.patch_size))
    dino_versions = alignment.module_parameter_versions(dino)

    diag = transfer.entry_probe.get_diag()
    transforms, img_scale, flip = diag.build_test_transforms(cfg)
    source_records = [
        record for record in transfer.discover_labeled_records(
            args.data_root, neighborhood.SOURCE_SPLIT, 0)
        if record['seq'] == args.source_seq]
    if args.max_source_samples > 0:
        source_records = source_records[:args.max_source_samples]
    if not source_records:
        raise RuntimeError('No source-real validation records found')
    source_samples, source_rows = collect_source(
        detector, dino, source_records, transforms,
        img_scale, flip, args, device)
    (fold_models, source_summary,
     calibrated_source_summary, bank_metadata) = build_source_ensemble(
         source_samples, args.source_folds, args.neighbors,
         args.positive_quantile, args.negative_quantile)

    # Target is first accessed only after all source-only banks and thresholds.
    target_rows = collect_target(
        detector, dino, transforms, img_scale, flip,
        args, device, fold_models)
    target_summary = summarize_target(target_rows, args.source_folds)
    geometry_misses = [
        int(row['frame']) for row in target_rows if row['geometry_miss']]
    gate = make_gate(
        source_summary, calibrated_source_summary,
        target_summary, geometry_misses, args)

    detector_unchanged = (
        detector_versions == alignment.module_parameter_versions(detector))
    dino_unchanged = (
        dino_versions == alignment.module_parameter_versions(dino))
    if not detector_unchanged or not dino_unchanged:
        raise RuntimeError('Read-only parameter invariant failed')
    payload = dict(
        audit=AUDIT_NAME, protocol_version=PROTOCOL_VERSION,
        canonical_protocol=bool(canonical),
        paper=dict(
            title='Large Self-Supervised Models Bridge the Gap in '
                  'Domain Adaptive Object Detection',
            venue='CVPR 2025', url=PAPER_URL, code=PAPER_CODE_URL),
        paper_mapping=dict(
            retained=[
                'frozen DINOv2 ViT-L/14 source-labeller backbone',
                'ImageNet RGB normalization',
                '600-pixel short side with 1333-pixel long-side cap',
                'input padded to patch size 14',
                'final intermediate patch-token feature map',
                'single-scale region representation',
                'source controls and calibration frozen before target use'],
            rotated_task_adaptation=(
                '7x7 orientation-aware grid_sample pooling replaces the '
                'paper labeller single-scale axis-aligned ROI pool'),
            intentionally_omitted=[
                'source-trained Faster R-CNN labeller head',
                'target pseudo-label generation',
                'student-target pseudo-label training',
                'two-layer student projection MLP',
                'ViT-B/14 source/target online cosine alignment loss']),
        data_role='source_reference_target_dev_diagnosis_only',
        config=os.path.abspath(args.config),
        detector_checkpoint=os.path.abspath(args.detector_checkpoint),
        dinov2=dict(
            repo=os.path.abspath(args.dinov2_repo),
            checkpoint=os.path.abspath(args.dinov2_checkpoint),
            checkpoint_size=int(os.path.getsize(args.dinov2_checkpoint)),
            checkpoint_sha256=file_sha256(args.dinov2_checkpoint),
            model=args.dinov2_model, patch_size=int(args.patch_size),
            input_short_side=int(args.dino_height),
            input_max_long_side=int(args.dino_max_long_side),
            pool_resolution=int(args.pool_resolution),
            legacy_sdpa_compatibility=bool(getattr(
                dino, '_sym_legacy_sdpa_installed', False)),
            legacy_sdpa_query_chunk=int(args.legacy_sdpa_query_chunk),
            device_map=getattr(dino, '_sym_dino_device_map', None)),
        protocol=dict(
            source_seq=args.source_seq,
            source_folds=int(args.source_folds),
            neighbors=int(args.neighbors),
            positive_quantile=_number(args.positive_quantile),
            negative_quantile=_number(args.negative_quantile),
            min_fold_votes=int(args.min_fold_votes),
            source_gate_definition=(
                'cross_fold_zero_margin_and_calibrated_accuracy'),
            target_threshold_calibration=(
                'source_fold_controls_positive_p10_negative_p90'),
            target_candidate_route='detector_level0_decoded_obb',
            target_slice='real_seq02[137..169]',
            riou_threshold=_number(args.riou_thr),
            false_iou_threshold=_number(args.false_iou_thr),
            source_min_accuracy=_number(args.source_min_accuracy),
            target_min_wins=int(args.target_min_wins)),
        isolation=dict(
            creates_optimizer=False, performs_optimizer_step=False,
            performs_backward=False, writes_checkpoint=False,
            writes_pseudo_labels=False, detector_frozen=True,
            detector_parameters_unchanged=detector_unchanged,
            dinov2_frozen=True, dinov2_parameters_unchanged=dino_unchanged,
            source_banks_frozen_before_target=True,
            target_used_for_source_banks=False,
            target_used_for_training=False,
            target_used_for_checkpoint_selection=False,
            target_labels_used_for_diagnosis_only=True,
            raw_dino_features_serialized=False),
        bank_metadata=bank_metadata,
        source=dict(
            summary=source_summary,
            calibrated_summary=calibrated_source_summary,
            rows=source_rows),
        target_dev=dict(
            geometry_misses=geometry_misses,
            summary=target_summary, rows=target_rows),
        gate=gate)
    if not canonical:
        payload['gate']['decision'] = 'NONCANONICAL_NO_AUTHORIZATION'
        payload['gate']['interpretation'] = (
            'Smoke test only; run the canonical full audit.')
    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, 'w') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False,
                  allow_nan=False)
    print('[dino-audit] {} wins={}/{}'.format(
        payload['gate']['decision'], target_summary['win_count'],
        target_summary['eligible_count']))
    print('[out] {}'.format(args.out_json))


if __name__ == '__main__':
    main()
