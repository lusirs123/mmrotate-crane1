#!/usr/bin/env python3
"""Shared runtime helpers for the formal frozen-DINOv2 branch."""

import glob
import hashlib
import importlib
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from crane_project.tools import ctx_entry_probe as entry_probe
from crane_project.tools import patch_dinov2_py38 as py38_patcher


SOURCE_SPLIT = 'val'
SOURCE_SEQ = 'real_seq07'
TARGET_SPLIT = 'test'
TARGET_SEQ = 'real_seq02'
TARGET_START = 137
TARGET_END = 169
EXPECTED_GEOMETRY_MISSES = [164, 167]
EXPECTED_ELIGIBLE = 31

CANONICAL_MODEL = 'dinov2_vitl14'
CANONICAL_PATCH_SIZE = 14
CANONICAL_DINO_HEIGHT = 600
CANONICAL_DINO_MAX_LONG_SIDE = 1333
CANONICAL_LEGACY_SDPA_QUERY_CHUNK = 512

_LEGACY_SDPA_QUERY_CHUNK = CANONICAL_LEGACY_SDPA_QUERY_CHUNK


def module_parameter_versions(module) -> Dict[str, int]:
    return {name: int(parameter._version)
            for name, parameter in module.named_parameters()}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def discover_labeled_records(data_root: str, split: str,
                             limit: int = 0) -> List[Dict]:
    ann_dir = os.path.join(data_root, split, 'annfiles')
    img_dir = os.path.join(data_root, split, 'images')
    records = []
    for ann_path in sorted(glob.glob(os.path.join(ann_dir, '*.txt'))):
        base = os.path.splitext(os.path.basename(ann_path))[0]
        match = re.match(r'(.+_seq\d+)_(\d{5})$', base)
        if match is None:
            continue
        img_path = None
        for extension in ('.jpg', '.png', '.bmp', '.tif'):
            candidate = os.path.join(img_dir, base + extension)
            if os.path.isfile(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue
        records.append(dict(
            split=split, seq=match.group(1), frame=int(match.group(2)),
            image=img_path, annotation=ann_path,
            domain=base.split('_', 1)[0]))
    if limit > 0:
        records = records[:limit]
    return records


def json_safe(value):
    """Replace non-finite numeric leaves and report how many were replaced."""
    if isinstance(value, dict):
        output = {}
        replacements = 0
        for key, item in value.items():
            safe_item, item_replacements = json_safe(item)
            output[key] = safe_item
            replacements += item_replacements
        return output, replacements
    if isinstance(value, (list, tuple)):
        output = []
        replacements = 0
        for item in value:
            safe_item, item_replacements = json_safe(item)
            output.append(safe_item)
            replacements += item_replacements
        return output, replacements
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None, 1
        return number, 0
    if isinstance(value, np.integer):
        return int(value), 0
    return value, 0


def write_json_atomic(path: str, payload: Dict) -> int:
    safe_payload, replacements = json_safe(payload)
    safe_payload['serialization'] = dict(
        nonfinite_values_replaced=int(replacements),
        replacement_value=None, atomic_write=True)
    output_path = os.path.abspath(path)
    temporary_path = output_path + '.tmp'
    with open(temporary_path, 'w') as handle:
        json.dump(safe_payload, handle, indent=2, ensure_ascii=False,
                  allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, output_path)
    return int(replacements)


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
            weights = F.dropout(weights, p=float(dropout_p), training=True)
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


def _legacy_annotation_error(error: TypeError) -> bool:
    return bool(sys.version_info < (3, 10)
                and 'unsupported operand type' in str(error))


def _clear_repo_module_cache(repo: str):
    repo = os.path.realpath(os.path.abspath(repo))
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if not path:
            continue
        path = os.path.realpath(os.path.abspath(path))
        try:
            inside_repo = os.path.commonpath([repo, path]) == repo
        except ValueError:
            inside_repo = False
        if inside_repo:
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _load_local_dinov2(repo: str, model_name: str):
    return torch.hub.load(repo, model_name, source='local', pretrained=False)


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
    annotation_compatibility = dict(
        attempted=False, changed_count=0, scanned_files=0,
        operation='not_needed')
    try:
        model = _load_local_dinov2(repo, model_name)
    except TypeError as error:
        if _legacy_annotation_error(error):
            try:
                annotation_compatibility = py38_patcher.patch_repo(repo)
            except (OSError, RuntimeError) as patch_error:
                raise RuntimeError(
                    'DINOv2 Python 3.8 annotation compatibility patch failed: '
                    '{}'.format(patch_error)) from patch_error
            annotation_compatibility['attempted'] = True
            print('[dinov2-py38-auto] changed={}/{} operation={}'.format(
                annotation_compatibility['changed_count'],
                annotation_compatibility['scanned_files'],
                annotation_compatibility['operation']))
            _clear_repo_module_cache(repo)
            try:
                model = _load_local_dinov2(repo, model_name)
            except TypeError as retry_error:
                if _legacy_annotation_error(retry_error):
                    raise RuntimeError(
                        'DINOv2 still uses evaluated Python 3.10 annotations '
                        'after the automatic compatibility patch') from retry_error
                raise
        else:
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
    model._sym_annotation_compatibility = annotation_compatibility
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
    scale = min(float(target_height) / float(min(ori_h, ori_w)),
                float(max_long_side) / float(max(ori_h, ori_w)))
    resized_h = max(1, int(round(float(ori_h) * scale)))
    resized_w = max(1, int(round(float(ori_w) * scale)))
    resized = cv2.resize(
        image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
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
        ori_shape=[ori_h, ori_w], resized_shape=[resized_h, resized_w],
        padded_shape=[pad_h, pad_w], scale=_number(scale),
        patch_size=int(patch_size), target_short_side=int(target_height),
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
